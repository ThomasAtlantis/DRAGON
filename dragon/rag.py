import torch
from typing import List, Tuple

from .config import DragonConfig
from .generator.generator import Generator
from .retriever.retriever import Retriever
from .utils.mlogging import Logger
from .utils.meter import TimeMeter


logger = Logger.build(__name__, level="INFO")
time_meter = TimeMeter()

class RagForGeneration:

    def __init__(self, config: DragonConfig):
        self.config = config
        self.generator = Generator(config)
        self.retriever = None 
        self.do_retrieval = config.retriever.n_docs > 0
        self.device = torch.device(config.device)
        self.enable_aggregate: bool = config.retriever.s_aggregate > 0
        self.context_size: int = config.retriever.s_context
        self.aggregate_size: int = config.retriever.s_aggregate
        
        if self.do_retrieval:
            self.retriever = Retriever(config)
            self.retriever.prepare_retrieval(config)

    def _prepare_inputs_for_generation(
            self, query_ids: List[int],
            input_ids: List[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.generator.tokenizer.decode(query_ids)
        with time_meter.timer("Retrieval"):
            docs, scores = self.retriever.retrieve_passages([query])[0]
        logger.debug(f"Retrieval complete in {time_meter.timer('Retrieval').duration:.4f} seconds.")

        with time_meter.timer("Tokenization"):
            doc_texts = [doc["text"] for doc in docs]
            if not self.enable_aggregate:
                scores, doc_texts = [1], ["\n".join(doc_texts)]
            doc_texts = doc_texts[: self.aggregate_size]
            doc_ids = self.generator.tokenizer.batch_encode_plus(
                doc_texts, max_length=self.context_size,
                padding='max_length', truncation=True,
                return_tensors='pt', return_attention_mask=False
            )['input_ids']
            input_ids = torch.as_tensor(
                input_ids, dtype=torch.long).repeat(self.aggregate_size, 1)
            query_ids = torch.as_tensor(
                query_ids, dtype=torch.long).repeat(self.aggregate_size, 1)
            context_input_ids = torch.cat([doc_ids, query_ids, input_ids], dim=1)

            scores = scores[: self.aggregate_size]
            scores = torch.as_tensor(scores, dtype=torch.float32)
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
            context_input_ids = context_input_ids.to(self.device)
            scores = scores.to(self.device)
        logger.debug(f"Tokenization complete in {time_meter.timer('Tokenization').duration:.4f} seconds.")
        return context_input_ids, scores
    
    def __call__(self, query_ids: List[int], input_ids: List[int]) -> torch.Tensor:
        """
        Given query_ids and input_ids, return the next token logprobs of each input_id.
        """
        if query_ids is None or len(query_ids) == 0:
            output = self.generator(torch.as_tensor(input_ids, dtype=torch.long).to(self.device))
            logprobs = torch.log_softmax(output.logits[0], dim=-1)
            return logprobs
        if not self.do_retrieval:
            output = self.generator(torch.cat([
                torch.as_tensor(query_ids, dtype=torch.long),
                torch.as_tensor(input_ids, dtype=torch.long)
            ]).to(self.device))
            logprobs = torch.log_softmax(output.logits[0, -len(input_ids):], dim=-1)
            return logprobs
        context_input_ids, scores = self._prepare_inputs_for_generation(query_ids, input_ids)
        _, logprobs, _ = self._generate(context_input_ids, scores, n_logits=len(input_ids))
        return logprobs
    
    def _generate(
            self, input_ids: torch.LongTensor, 
            scores: torch.Tensor, n_logits: int, 
            past_key_values=None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Prefill concatenations of context_ids and input_ids, generate the next token, and return the logprobs
        during the prefilling process. 
        
        Output Aggregation: 
        probs = softmax(w)^T softmax(z) -> log(probs) = logsumexp(logsoftmax(w)+logsoftmax(z))
        """
        with time_meter.timer("Decoding"):
            output = self.generator(input_ids=input_ids, past_key_values=past_key_values)
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
        context_input_ids, scores = self._prepare_inputs_for_generation(query_ids, input_ids)

        # Pre-fill the context
        next_token, logprobs, past_key_values = self._generate(context_input_ids, scores, n_logits=1)
        logprobs, output_ids = [logprobs[0]], [next_token]

        for _ in range(max_new_tokens - 1):  # Auto-regressive generation
            input_ids = torch.as_tensor([next_token], dtype=torch.long).to(self.device)
            input_ids = input_ids.repeat(self.aggregate_size, 1)
            next_token, logprobs_token_wise, past_key_values = self._generate(
                input_ids, scores, n_logits=1, past_key_values=past_key_values)
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

        context_input_ids, scores = self._prepare_inputs_for_generation(query_ids, input_ids)
        output = self.generator.generate(
            context_input_ids, max_new_tokens=max_new_tokens, output_logits=True)
        logits = torch.stack(output.logits)          # (s_sequence, s_aggregate, s_vocab)
        logprobs = torch.nn.functional.log_softmax(
            logits / self.generator.sampler.temperature, dim=-1)
        logprobs = logprobs + scores.unsqueeze(dim=-1)
        logprobs = torch.logsumexp(logprobs, dim=1)  # (s_sequence, s_vocab)
        output_ids = self.generator.sampler(torch.exp(logprobs))
        return output_ids, logprobs