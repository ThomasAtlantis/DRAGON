import pickle
import torch
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from .config import DragonConfig
from .generator.generator import Generator
from .retriever.retriever import Retriever
from .utils.mlogging import Logger
from .utils.meter import TimeMeter


logger = Logger.build(__name__, level="INFO")
time_meter = TimeMeter()


def group_docs(docs: List[str], scores: List[float], s_aggregate: int) -> Tuple[List[str], List[float]]:
    """
    Group documents into s_aggregate number of documents.
    """
    chunk_size = len(docs) // s_aggregate
    if chunk_size == 0:
        logger.info(f"{len(scores)=} {s_aggregate=}")
    chunk_scores, chunk_docs = [], []
    for i in range(s_aggregate):
        beg, end = i * chunk_size, (i + 1) * chunk_size
        chunk_docs.append('\n'.join(docs[beg: end]))
        chunk_scores.append(sum(scores[beg: end]) / chunk_size)
    return chunk_docs, chunk_scores
    
    
class RagModule:

    def __init__(self, config: DragonConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.context_size: int = config.retriever.s_context  # need to be same across device and cloud
        self.aggregate_size: int = config.retriever.s_aggregate
        self.do_retrieve = config.retriever.n_docs > 0
        self.do_rerank = config.reranker.do_rerank

        self.generator = Generator(config)

        self.retriever = None
        if self.do_retrieve:
            self.retriever = Retriever(config)
            self.retriever.prepare_retrieval(config)
            
        self.reranker = None
        if self.do_rerank:
            self.rerank_model = CrossEncoder(
                model_name=config.reranker.model,
                max_length=512, device=config.device)
            self.rerank_momentum = config.reranker.momentum

    def _prepare_inputs_for_generation(
        self, query_ids: List[int],
        input_ids: List[int],
        template: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.generator.tokenizer.decode(query_ids)
        input_text = self.generator.tokenizer.decode(input_ids)
        with time_meter.timer("Retrieval"):
            docs, scores = self.retriever.retrieve_passages([query])[0]
        logger.debug(f"Retrieval complete in {time_meter.timer('Retrieval').duration:.4f} seconds.")

        with time_meter.timer("Tokenization"):
            doc_texts = [doc["text"] for doc in docs]
            doc_texts, scores = group_docs(doc_texts, scores, self.aggregate_size)
            
            doc_texts = [
                self.generator.tokenizer.batch_decode(self.generator.tokenizer.encode(doc_text)[: self.context_size])
                for doc_text in doc_texts
            ]
            input_text_list = [
                template.format(doc_text=doc_text, query=query, input_text=input_text)
                for doc_text in doc_texts
            ]

            inputs_encoding = self.generator.tokenizer.batch_encode_plus(
                input_text_list, padding='longest', return_tensors='pt')
            context_input_ids = inputs_encoding['input_ids']
            context_input_ids = context_input_ids.to(self.device)
            attention_mask = inputs_encoding['attention_mask'].to(self.device)

            scores = scores[: self.aggregate_size]
            scores = torch.as_tensor(scores, dtype=torch.float32)
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
            scores = scores.to(self.device)
        logger.debug(f"Tokenization complete in {time_meter.timer('Tokenization').duration:.4f} seconds.")
        return context_input_ids, attention_mask, scores, doc_texts
    
    def _generate(
            self, input_ids: torch.LongTensor, 
            attention_mask: torch.LongTensor,
            scores: torch.Tensor, n_logits: int, 
            past_key_values=None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Prefill concatenations of context_ids and input_ids, generate the next token, and return the logprobs
        during the prefilling process. 
        
        Output Aggregation: 
        probs = softmax(w)^T softmax(z) -> log(probs) = logsumexp(logsoftmax(w)+logsoftmax(z))
        """
        with time_meter.timer("Decoding"):
            output = self.generator(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                past_key_values=past_key_values
            )
            past_key_values = output.past_key_values
            logits = output.logits[:, -n_logits: ]
            logprobs = torch.nn.functional.log_softmax(     # (s_aggregate, s_sequence, s_vocab)
                logits / self.generator.sampler.temperature, dim=-1)
            logprobs = logprobs.permute(1, 0, 2)            # (s_sequence, s_aggregate, s_vocab)
            logprobs = logprobs + scores.unsqueeze(dim=-1)  # (s_sequence, s_aggregate, s_vocab) + (s_aggregate, 1)
            logprobs = torch.logsumexp(logprobs, dim=1)
            next_token = self.generator.sampler(torch.exp(logprobs[-1]).unsqueeze(0))[0]
        logger.debug(f"Decoding complete in {time_meter.timer('Decoding').duration:.4f} seconds.")
        return next_token, logprobs, past_key_values


    def _rerank_documents(
            self, query_ids: List[int], 
            past_generated: List[int], 
            doc_texts: List[str],
            scores: torch.Tensor
        ) -> torch.Tensor:
        new_query_ids = query_ids + past_generated
        new_query = self.generator.tokenizer.decode(new_query_ids)
        
        with time_meter.timer("Re-ranking"):
            pairs = [[new_query, doc] for doc in doc_texts]
            new_scores = self.rerank_model.predict(
                pairs, activation_fct=torch.sigmoid,
                batch_size=len(pairs), convert_to_numpy=False
            )
        logger.debug(f"Re-ranking complete in {time_meter.timer('Re-ranking').duration:.4f} seconds.")
        
        new_scores = torch.nn.functional.log_softmax(torch.as_tensor(new_scores), dim=-1).to(self.device)
        new_scores = scores * self.rerank_momentum + new_scores * (1 - self.rerank_momentum)
        return new_scores
    
    def pre_generate(self, 
            query_ids: List[int],
            template: str
        ) -> Tuple[List[int], List[torch.Tensor]]:
        context_input_ids, attention_mask, scores, doc_texts = self._prepare_inputs_for_generation(query_ids, [], template)

        # Pre-fill the context
        next_token, logprobs, past_key_values = self._generate(context_input_ids, attention_mask, scores, n_logits=1)
        logprobs, output_ids = [logprobs[0]], [next_token]
        return output_ids

    def generate(
            self, 
            token: int,
            query_ids: List[int],
            do_rerank: bool,
            scores: torch.Tensor,
        ) -> Tuple[List[int], List[torch.Tensor]]:
        # Dynamic re-ranking every `self.config.reranker.period` steps
        if do_rerank:
            scores = self._rerank_documents()

        input_ids = torch.as_tensor([token], dtype=torch.long).to(self.device)
        input_ids = input_ids.repeat(self.aggregate_size, 1)
        attention_mask = torch.ones_like(input_ids)
        next_token, logprobs, past_key_values = self._generate(
            input_ids, attention_mask, scores, n_logits=1, past_key_values=past_key_values)
        return next_token, past_key_values

class Messager


class Server:

    def __init__(self, config):
        self.rag = RagModule(config)
        self.messager = Messager(config)
    
    def run(self):
        while rec

