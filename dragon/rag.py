import torch
from typing import List, Tuple

from .config import DragonConfig
from .generator.generator import Generator
from .retriever.retriever import Retriever
from .utils.mlogging import Logger


logger = Logger.build(__name__, level="INFO")

class RagSequenceForGeneration:

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
        if len(query_ids) == 0: return [[]], [1]
        query = self.generator.tokenizer.decode(query_ids)
        docs, scores = self.retriever.retrieve_passages([query])[0]
        doc_texts = [doc["text"] for doc in docs]
        if not self.enable_aggregate:
            doc_texts, scores = ["\n".join(doc_texts)], [1]
        doc_texts = doc_texts[: self.aggregate_size]
        scores = scores[: self.aggregate_size]
        context_ids_list = [
            self.generator.tokenizer.encode(doc)[: self.context_size] + query_ids for doc in doc_texts]
        return context_ids_list, scores
    
    def __call__(self, query_ids: List[int], input_ids: List[int]) -> Tuple[torch.Tensor, Tuple]:
        context_ids_list, scores = self._prepare_inputs_for_generation(query_ids)
        scores = torch.as_tensor(scores, dtype=torch.float32).to(self.device)
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        past_key_values = [None] * self.aggregate_size
        logprobs_list = []
        for i, context_ids in enumerate(context_ids_list):
            context_input_ids = torch.LongTensor(context_ids + input_ids).to(self.device).unsqueeze(dim=0)
            output = self.generator(context_input_ids, past_key_values=past_key_values[i])
            past_key_values[i] = output.past_key_values
            logits = output.logits.squeeze()[-len(input_ids): ]
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            logprobs_list.append(logprobs)
        logprobs_list = torch.stack(logprobs_list, dim=1)         # (s_sequence, n_docs, s_vocab)
        logprobs_list = logprobs_list + scores.unsqueeze(dim=-1)  # (s_sequence, n_docs, s_vocab)
        logprobs = torch.logsumexp(logprobs_list, dim=1)          # (s_sequence, s_vocab)
        return logprobs, past_key_values
    
    def generate(self, input_ids):
        pass
    