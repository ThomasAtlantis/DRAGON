from dataclasses import dataclass
import time
import torch
from typing import List, Tuple
from sentence_transformers import CrossEncoder

from .transceiver import Message
from .config import DragonConfig
from .generator.generator import Generator
from .retriever.retriever import Retriever
from .transceiver import Transceiver
from .utils.mlogging import Logger
from .utils.meter import TimeMeter


time_meter = TimeMeter()

@dataclass
class CausalOutput:
    next_token: int
    logprobs: torch.Tensor
    past_key_values: List[torch.Tensor] = None
    
class RagModule:

    def __init__(self, config: DragonConfig, logger: Logger):
        self.logger = logger
        self.config = config
        self.device = torch.device(config.device)
        self.context_size: int = config.retriever.s_context  # need to be same across device and cloud
        self.aggregate_size: int = config.retriever.s_aggregate
        self.do_retrieve = config.retriever.n_docs > 0
        self.do_rerank = config.reranker.do_rerank

        self.generator = Generator(config)

        self.retriever = None
        if self.do_retrieve:
            self.retriever = Retriever(config, logger)
            self.retriever.prepare_retrieval(config)
            
        self.reranker = None
        if self.do_rerank:
            self.rerank_model = CrossEncoder(
                model_name=config.reranker.model,
                max_length=512, device=config.device)
            self.rerank_momentum = config.reranker.momentum

    def _crop_context(self, context: str) -> str:
        return self.generator.tokenizer.decode(
            self.generator.tokenizer.encode(context)[: self.context_size]
        )

    def _group_passages(self, passages: List[str], scores: List[float]) -> Tuple[List[str], List[float]]:
        """
        Partition the documents into groups of size self.aggregate_size, 
        and calculate the average score for each group.
        """
        group_size = len(passages) // self.aggregate_size
        if group_size == 0:
            self.logger.info(f"{len(scores)=} {self.aggregate_size=}")
        group_scores, group_passages = [], []
        for i in range(self.aggregate_size):
            beg, end = i * group_size, (i + 1) * group_size
            group_passages.append('\n'.join(passages[beg: end]))
            group_scores.append(sum(scores[beg: end]) / group_size)
        return group_passages, group_scores

    def _prepare_inputs_for_generation(
        self, query: str, prompt_template: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        
        with time_meter.timer("Retrieval"):
            passages, scores = self.retriever.retrieve_passages([query])[0]
        self.logger.debug(f"Retrieval complete in {time_meter.timer('Retrieval').duration:.4f} seconds.")

        with time_meter.timer("Tokenization"):
            # assemble input from passages, query and prompt_template
            passages = [p["text"] for p in passages]
            passages, scores = self._group_passages(passages, scores)
            input_text_list = [
                prompt_template.format(
                    doc_text=self._crop_context(doc_text), 
                    query=query
                ) for doc_text in passages
            ]

            # encode input into input_ids and attention_mask    
            inputs_encoding = self.generator.tokenizer.batch_encode_plus(
                input_text_list, padding='longest', return_tensors='pt')
            input_ids = inputs_encoding['input_ids'].to(self.device)
            attention_mask = inputs_encoding['attention_mask'].to(self.device)

        self.logger.debug(f"Tokenization complete in {time_meter.timer('Tokenization').duration:.4f} seconds.")
        
        # pre-compute the logprobs of scores for reusing in the generation process
        scores = torch.nn.functional.log_softmax(
            torch.as_tensor(scores, dtype=torch.float32), dim=-1).to(self.device)

        return input_ids, attention_mask, scores, passages

    def _rerank_passages(
            self, query: str, generated: str, passages: List[str], scores: torch.Tensor
        ) -> torch.Tensor:
        new_query = f"{query} {generated}"
        
        with time_meter.timer("Re-ranking"):
            pairs = [[new_query, doc] for doc in passages]
            new_scores = self.rerank_model.predict(
                pairs, activation_fct=torch.sigmoid,
                batch_size=len(pairs), convert_to_numpy=False
            )
        self.logger.debug(f"Re-ranking complete in {time_meter.timer('Re-ranking').duration:.4f} seconds.")
        
        new_scores = torch.nn.functional.log_softmax(torch.as_tensor(new_scores), dim=-1).to(self.device)
        new_scores = scores * self.rerank_momentum + new_scores * (1 - self.rerank_momentum)
        return new_scores
    
    def rerank_passages(self, query: str, generated: str, passages: List[str], scores: torch.Tensor, step: int) -> torch.Tensor:
        # Dynamic re-ranking every `self.config.reranker.period` steps
        if self.do_rerank and (self.config.reranker.period == 0 or (step + 1) % self.config.reranker.period == 0):
            scores = self._rerank_passages(query, generated, passages, scores)    
        return scores
    
    def _generate(
            self, input_ids: torch.LongTensor, 
            attention_mask: torch.LongTensor,
            scores: torch.Tensor, 
            past_key_values=None) -> CausalOutput:
        """
        Prefill concatenations of context_ids and input_ids, generate the next token, and return the logprobs
        during the prefilling process. 
        
        Output Aggregation: 
        probs = softmax(w)^T softmax(z) -> log(probs) = logsumexp(logsoftmax(w)+logsoftmax(z))
        """
        with time_meter.timer("Decoding"):
            output = self.generator(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
            logprobs = torch.nn.functional.log_softmax(    # (s_aggregate, s_vocab)
                output.logits[:, -1] / self.generator.sampler.temperature, dim=-1)
            logprobs = logprobs.permute(1, 0)              # (s_aggregate, s_vocab)
            logprobs = logprobs + scores.unsqueeze(dim=-1) # (s_aggregate, s_vocab) + (s_aggregate, 1)
            logprobs = torch.logsumexp(logprobs, dim=0)    # (s_vocab)
            next_token = self.generator.sampler(torch.exp(logprobs).unsqueeze(0))[0]
        self.logger.debug(f"Decoding complete in {time_meter.timer('Decoding').duration:.4f} seconds.")

        return CausalOutput(
            next_token=next_token, logprobs=logprobs, 
            past_key_values=output.past_key_values
        )

    def generate(self, last_token: int, scores: torch.Tensor, past_key_values: List[torch.Tensor]) -> CausalOutput:
        input_ids = torch.as_tensor([last_token], dtype=torch.long, device=self.device)
        input_ids = input_ids.repeat(self.aggregate_size, 1)
        attention_mask = torch.ones_like(input_ids)
        return self._generate(input_ids, attention_mask, scores, past_key_values=past_key_values)


class Dragon(Transceiver):

    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.rag = RagModule(config, self.logger)
        self.ready_for_generation = False
        self.register_observers(self.collect_observers())
        self.logger.info("Dragon listening for connections.")
        self.send_with_retry(Message.READY_FOR_CONNECTION, b'')

    def generate(self, query: str, prompt_template: str, max_new_tokens: int):
        self.send(Message.GENERATE, query.encode('utf-8'))
        generated_ids, generated_text = [], ""

        input_ids, attention_mask, scores, passages = self.rag._prepare_inputs_for_generation(query, prompt_template)
        output = self.rag._generate(input_ids, attention_mask, scores)
        for step in range(max_new_tokens):
            generated_ids.append(output.next_token)
            generated_text = self.rag.generator.tokenizer.decode(generated_ids, skip_special_tokens=True)
            scores = self.rag.rerank_passages(query, generated_text, passages, scores, step)
            output = self.rag.generate(
                output.next_token, scores, past_key_values=output.past_key_values)
        return generated_text

    def collect_observers(self):
        return [getattr(self, attr) for attr in self.__dict__ if attr.startswith("_rx_")]

    def _rx_ready_for_generation(self, mtype, mbody):
        if mtype == Message.READY_FOR_GENERATION:
            self.ready_for_generation = True
            self.logger.info("Remote is ready for generation.")
    
    def _rx_generate(self, mtype, mbody):
        if mtype == Message.GENERATE:
            query = mbody.decode('utf-8')
            self.logger.info(f"Generating response for query: {query}")

