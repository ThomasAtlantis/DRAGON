import torch
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from ..config import DragonConfig
from ..generator import Generator
# from .retriever.retriever import Retriever
from ..retriever.retriever import DPRRetrieverClient as Retriever
from ..utils.mlogging import Logger
from ..utils.meter import TimeMeter


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

class RagForGeneration:

    def __init__(self, config: DragonConfig):
        self.config = config
        self.generator = Generator(config)
        self.retriever = None 
        self.do_retrieval = config.retriever.n_docs > 0
        self.device = torch.device(config.device)
        self.context_size: int = config.retriever.s_context
        self.aggregate_size: int = config.retriever.s_aggregate
        
        if self.do_retrieval:
            self.retriever = Retriever(config, logger)
            self.retriever.prepare_retrieval(config)

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
                self.generator.tokenizer.decode(self.generator.tokenizer.encode(doc_text)[: self.context_size])
                for doc_text in doc_texts
            ]
            input_text_list = [
                template.format(context=doc_text, query=query, input_text=input_text)
                for doc_text in doc_texts
            ]
            # print(input_text_list)

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
    
    def __call__(self, query_ids: List[int], input_ids: List[int], template) -> torch.Tensor:
        """
        Given query_ids and input_ids, return the next token logprobs of each input_id.
        """
        if not self.do_retrieval or len(query_ids) == 0:
            input_ids = torch.as_tensor(query_ids + input_ids, dtype=torch.long).to(self.device).unsqueeze(0)
            output = self.generator(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            logprobs = torch.log_softmax(output.logits[0, len(query_ids):], dim=-1)
            return logprobs
        context_input_ids, attention_mask, scores, doc_texts = self._prepare_inputs_for_generation(query_ids, input_ids, template)
        _, logprobs, _ = self._generate(context_input_ids, attention_mask, scores, n_logits=len(input_ids))
        return logprobs
    
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
            # output = self.generator(
            #     input_ids=input_ids, 
            #     attention_mask=attention_mask, 
            #     past_key_values=past_key_values
            # )
            half_size = input_ids.shape[0] // 2
            # past_key_values_1 = [[], []] if past_key_values is not None else None
            # past_key_values_2 = [[], []] if past_key_values is not None else None
            # if past_key_values is not None:
            #     for layer_idx in range(len(past_key_values)):
            #         past_key_values_1[0].append(past_key_values[layer_idx][0][: half_size])                    
            #         past_key_values_1[1].append(past_key_values[layer_idx][1][: half_size])
            #         past_key_values_2[0].append(past_key_values[layer_idx][0][half_size: ])
            #         past_key_values_2[1].append(past_key_values[layer_idx][1][half_size: ])
            past_key_values_1 = past_key_values_2 = None
            if past_key_values is not None:
                past_key_values_1, past_key_values_2 = past_key_values

            output_1 = self.generator(
                input_ids=input_ids[: half_size], 
                attention_mask=attention_mask[: half_size], 
                past_key_values=past_key_values_1
            )
            output_2 = self.generator(
                input_ids=input_ids[half_size: ], 
                attention_mask=attention_mask[half_size: ], 
                past_key_values=past_key_values_2
            )
            # past_key_values = [[], []]
            # for layer_idx in range(len(output_1.past_key_values)):
            #     past_key_values[0].append(torch.vstack([output_1.past_key_values[layer_idx][0], output_2.past_key_values[layer_idx][0]]))
            #     past_key_values[1].append(torch.vstack([output_1.past_key_values[layer_idx][1], output_2.past_key_values[layer_idx][1]]))
            past_key_values = (output_1.past_key_values, output_2.past_key_values)
            logits = torch.vstack([output_1.logits[:, -n_logits: ], output_2.logits[:, -n_logits: ]])
            # print(logits.squeeze(-1).cpu().max(dim=-1))
            logprobs = torch.nn.functional.log_softmax(     # (s_aggregate, s_sequence, s_vocab)
                logits / self.generator.sampler.temperature, dim=-1)
            logprobs = logprobs.permute(1, 0, 2)            # (s_sequence, s_aggregate, s_vocab)
            logprobs = logprobs + scores.unsqueeze(dim=-1)  # (s_sequence, s_aggregate, s_vocab) + (s_aggregate, 1)
            logprobs = torch.logsumexp(logprobs, dim=1)
            next_token = self.generator.sampler(torch.exp(logprobs[-1]).unsqueeze(0))[0]
        logger.debug(f"Decoding complete in {time_meter.timer('Decoding').duration:.4f} seconds.")
        return next_token, logprobs, past_key_values
    
class RagTokenForGeneration(RagForGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.do_rerank = config.reranker.do_rerank
        if self.do_rerank:
            self.rerank_model = CrossEncoder(
                model_name=config.reranker.model,
                max_length=512, device=config.device)
            self.rerank_momentum = config.reranker.momentum

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

    def generate(
            self, 
            query_ids: List[int],
            max_new_tokens: int,
            template: str
        ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Prefill the context that consists of retrieved passages and the query_ids, and generate max_new_tokens 
        tokens autoregressively. Return the newly generated tokens and their corresponding logprobs.

        Aggregate tokens from multiple generation processes at each step to obtain the next token. Repeat this
        autoregressive generation process for max_new_tokens times.
        """
        scores_list = []  # for debugging
        pbar = tqdm(total=max_new_tokens, desc="Generating", leave=False, initial=0)
        context_input_ids, attention_mask, scores, doc_texts = self._prepare_inputs_for_generation(query_ids, [], template)
        # with open("logs/docs.pkl", "wb") as f:
        #     pickle.dump(doc_texts, f)
        scores_list.append(scores)

        # Pre-fill the context
        next_token, logprobs, past_key_values = self._generate(context_input_ids, attention_mask, scores, n_logits=1)
        logprobs, output_ids = [logprobs[0]], [next_token]
        pbar.update(1)

        for t in range(max_new_tokens - 1):  # Auto-regressive generation
            # Dynamic re-ranking every `self.config.reranker.period` steps
            if self.do_rerank and (self.config.reranker.period == 0 or (t + 1) % self.config.reranker.period == 0):
                scores = self._rerank_documents(query_ids, output_ids, doc_texts, scores)
            scores_list.append(scores)
            input_ids = torch.as_tensor([next_token], dtype=torch.long).to(self.device)
            input_ids = input_ids.repeat(self.aggregate_size, 1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=1)
            next_token, logprobs_token_wise, past_key_values = self._generate(
                input_ids, attention_mask, scores, n_logits=1, past_key_values=past_key_values)
            logprobs.append(logprobs_token_wise)
            output_ids.append(next_token)
            pbar.update(1)
        pbar.close()
        logprobs = torch.vstack(logprobs)    # (max_new_tokens, s_vocab)
        scores_list = torch.vstack(scores_list).exp()  # (max_new_tokens, s_aggregate)
        # with open("logs/scores.pkl", "wb") as f:
        #     pickle.dump(scores_list.cpu().tolist(), f)
        return output_ids, logprobs
    
class RagSequenceForGeneration(RagForGeneration):

    def generate(
            self, 
            query_ids: List[int],
            max_new_tokens: int,
            template: str
        ) -> Tuple[List[int], List[torch.Tensor]]:
        
        """
        Prefill the context that consists of retrieved passages and the query_ids, and generate max_new_tokens
        tokens autoregressively. Return the newly generated tokens and their corresponding logprobs.

        After generating multiple sequence of max_new_tokens tokens, aggregate tokens at each step to obtain
        the final sequence. This is exactly what the REPLUG model does. Refer to its official implementation: 
        https://github.com/swj0419/REPLUG/blob/master/downstream_eval/qa_final.py
        """

        context_input_ids, attention_mask, scores, doc_texts = self._prepare_inputs_for_generation(query_ids, [], template)
        output = self.generator.generate(
            context_input_ids, attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens, output_logits=True
        )
        logits = torch.stack(output.logits)          # (s_sequence, s_aggregate, s_vocab)
        logprobs = torch.nn.functional.log_softmax(
            logits / self.generator.sampler.temperature, dim=-1)
        logprobs = logprobs + scores.unsqueeze(dim=-1)
        logprobs = torch.logsumexp(logprobs, dim=1)  # (s_sequence, s_vocab)
        output_ids = self.generator.sampler(torch.exp(logprobs))
        return output_ids, logprobs