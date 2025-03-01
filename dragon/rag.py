import ipdb
import torch
from typing import List, Tuple

from .config import DragonConfig
from .generator.generator import Generator
from .retriever.retriever import Retriever
from .utils.mlogging import Logger


logger = Logger.build(__name__, level="INFO")

class RagForGeneration:

    def __init__(self, config: DragonConfig):
        self.config = config
        self.generator = Generator(config)
        self.retriever = Retriever(config)
        self.retriever.prepare_retrieval()
        self.device = torch.device(config.device)

        self.enable_aggregate: bool = config.retriever.s_aggregate > 0
        self.context_size: int = config.retriever.s_context
        self.aggregate_size: int = config.retriever.s_aggregate

    def _prepare_inputs_for_generation(self, query_ids: List[int]) -> Tuple[List[List[int]], List[float]]:
        if len(query_ids) == 0: return None, None
        query = self.generator.tokenizer.decode(query_ids)
        docs, scores = self.retriever.retrieve_passages([query])[0]
        doc_texts = [doc["text"] for doc in docs]
        if not self.enable_aggregate:
            scores, doc_texts = [1], ["\n".join(doc_texts)]
        doc_texts = doc_texts[: self.aggregate_size]
        scores = scores[: self.aggregate_size]
        scores = torch.as_tensor(scores, dtype=torch.float32).to(self.device)
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        context_ids_list = [
            self.generator.tokenizer.encode(
                doc, padding='max_length', truncation=True, max_length=self.context_size
            ) + query_ids for doc in doc_texts]
        return scores, context_ids_list
    
    def __call__(self, query_ids: List[int], input_ids: List[int]) -> torch.Tensor:
        """
        Given query_ids and input_ids, return the next token logprobs of each input_id.
        """
        scores, context_ids_list = self._prepare_inputs_for_generation(query_ids) 
        _, logprobs = self._generate(input_ids, [None] * self.aggregate_size, scores, context_ids_list)
        return logprobs
    
    def _generate(self, input_ids, past_key_values, scores, context_ids_list=None) -> Tuple[int, torch.Tensor]:
        """
        Prefill concatenations of context_ids and input_ids, generate the next token, and return the logprobs
        during the prefilling process. 
        
        Output Aggregation: 
        probs = softmax(w)^T softmax(z) -> log(probs) = logsumexp(logsoftmax(w)+logsoftmax(z))
        """
        logprobs = []
        for i in range(self.aggregate_size):
            context_ids = context_ids_list[i] if context_ids_list is not None else []
            context_input_ids = torch.LongTensor(context_ids + input_ids).to(self.device).unsqueeze(dim=0)
            output = self.generator(context_input_ids, past_key_values=past_key_values[i])
            past_key_values[i] = output.past_key_values
            logits = output.logits.squeeze(0)[-len(input_ids): ]
            logprobs_doc_wise = torch.nn.functional.log_softmax(
                logits / self.generator.sampler.temperature, dim=-1)
            logprobs.append(logprobs_doc_wise)
        logprobs = torch.stack(logprobs, dim=1)      # (s_sequence, n_docs, s_vocab)
        if scores is not None:
            logprobs = logprobs + scores.unsqueeze(dim=-1)
        logprobs = torch.logsumexp(logprobs, dim=1)  # (s_sequence, s_vocab)
        next_token = self.generator.sampler(torch.exp(logprobs[-1]).unsqueeze(0))[0]
        return next_token, logprobs
    
class RagTokenForGeneration(RagForGeneration):

    def generate(
            self, 
            query_ids: List[int], 
            input_ids: List[int], 
            max_new_tokens: int
        ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Prefill the context that consists of documents retrieved based on query_ids, query_ids and input_ids,
        and generate max_new_tokens tokens autoregressively. Return the newly generated tokens and their 
        corresponding logprobs.

        Aggregate tokens from multiple generation processes at each step to obtain the next token. Repeat this
        autoregressive generation process for max_new_tokens times.
        """
        scores, context_ids_list = self._prepare_inputs_for_generation(query_ids)
        past_key_values = [None] * self.aggregate_size

        # Pre-fill the context
        next_token, logprobs = self._generate(input_ids, past_key_values, scores, context_ids_list)
        logprobs, output_ids = [logprobs[-1]], [next_token]

        for _ in range(max_new_tokens - 1):  # Auto-regressive generation
            next_token, logprobs_token_wise = self._generate([next_token], past_key_values, scores, None)
            logprobs.append(logprobs_token_wise)
            output_ids.append(next_token)
        logprobs = torch.vstack(logprobs)    # (max_new_tokens, s_vocab)
        return output_ids, logprobs
    
class RagSequenceForGeneration(RagForGeneration):

    def generate(
            self, 
            query_ids: List[int], 
            input_ids: List[int], 
            max_new_tokens: int
        ) -> Tuple[List[int], List[torch.Tensor]]:
        
        """
        Prefill the context that consists of documents retrieved based on query_ids, query_ids and input
        ids, and generate max_new_tokens tokens autoregressively. Return the newly generated tokens and
        their corresponding logprobs.

        After generating multiple sequence of max_new_tokens tokens, aggregate tokens at each step to obtain
        the final sequence. This is exactly what the REPLUG model does. Refer to its official implementation: 
        https://github.com/swj0419/REPLUG/blob/master/downstream_eval/qa_final.py
        """

        scores, context_ids_list = self._prepare_inputs_for_generation(query_ids)
        
        logprobs = []
        for i, context_ids in enumerate(context_ids_list):
            context_input_ids = torch.LongTensor(context_ids + input_ids).to(self.device).unsqueeze(dim=0)
            output = self.generator.generate(context_input_ids, max_new_tokens=max_new_tokens, output_logits=True)
            logits = torch.vstack(output.logits)
            logprobs_doc_wise = torch.nn.functional.log_softmax(
                logits / self.generator.sampler.temperature, dim=-1)
            logprobs.append(logprobs_doc_wise)

        logprobs = torch.stack(logprobs, dim=1)      # (s_sequence, n_docs, s_vocab)
        if scores is not None:
            logprobs = logprobs + scores.unsqueeze(dim=-1)
        logprobs = torch.logsumexp(logprobs, dim=1)  # (s_sequence, s_vocab)
        output_ids = self.generator.sampler(torch.exp(logprobs))
        return output_ids, logprobs
        
    