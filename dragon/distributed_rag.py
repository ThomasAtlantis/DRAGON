from dataclasses import dataclass
import math
import queue
import threading
import torch
from typing import List, Tuple
from sentence_transformers import CrossEncoder

from .transceiver import Message
from .config import DragonConfig
from .generator.generator import Generator, Sampler
from .retriever.retriever import Retriever
from .transceiver import Transceiver
from .utils.mlogging import Logger
from logging import Logger as PyLogger
from .utils.meter import TimeMeter


time_meter = TimeMeter()

@dataclass
class CausalOutput:
    next_token: int
    logprobs: torch.Tensor
    past_key_values: List[torch.Tensor] = None

class Preempted(Exception):
    pass

class PreemptableGenerator(threading.Thread, Generator):

    
    def __init__(self, config: DragonConfig, input_queue: queue.Queue, output_queue: queue.Queue):
        Generator.__init__(self, config)
        threading.Thread.__init__(self)
        self.logger = Logger.build(
            __class__.__name__, level="INFO")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = threading.Event()

        total_modules = sum(1 for _ in self.model.modules())
        def forward_hook(module_name, depth):
            def hook(module, input, output):
                if self.stop_event.is_set():
                    raise Preempted(f"Preempted after Module `{module_name}` at {depth / total_modules:.2%}")
                    # return 0
                return output
            return hook

        for idx, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, torch.nn.Module):
                module.register_forward_hook(forward_hook(name, idx))

    def run(self):
        while True:
            try:
                input_ids, attention_mask, kwargs = self.input_queue.get()
            except Exception as e:
                self.logger.debug("Generator stopped")
                break
            try:
                self.output_queue.put(
                    super().__call__(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        **kwargs
                    )
                )
            except Preempted as e:
                self.logger.warning(e)
                self.output_queue.put(None)
                pass
    
    def close(self):
        self.stop_event.set()
        self.input_queue.put(None)
        self.join()

class RagPipeline:

    def __init__(self, config: DragonConfig, logger: PyLogger, rank_idx: int):
        self.rank_idx = rank_idx
        self.logger = logger
        self.config = config
        self.device = torch.device(config.device)
        self.context_size: int = config.retriever.s_context  # need to be same across device and cloud
        self.aggregate_size: int = config.retriever.s_aggregate
        self.do_retrieve = config.retriever.n_docs > 0
        self.do_rerank = config.reranker.do_rerank
        self.input_queue = queue.Queue(0)
        self.output_queue = queue.Queue(0)
        self.generator = PreemptableGenerator(
            config, self.input_queue, self.output_queue)
        self.generator.start()

        self.retriever = None
        if self.do_retrieve:
            self.retriever = Retriever(config, logger)
            self.retriever.prepare_retrieval(config)
            
        self.reranker = None
        if self.do_rerank:
            self.rerank_model = CrossEncoder(
                model_name=config.reranker.model,
                max_length=512).to(config.device)
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
        # ipdb.set_trace()
        with time_meter.timer("Tokenization"):
            # assemble input from passages, query and prompt_template
            passages = [p["text"] for p in passages]
            passages, scores = self._group_passages(passages, scores)
            input_text_list = [
                prompt_template.format(
                    context=self._crop_context(passage), 
                    query=query
                ) for passage in passages
            ]

            # encode input into input_ids and attention_mask    
            inputs_encoding = self.generator.tokenizer.batch_encode_plus(
                input_text_list, padding='longest', return_tensors='pt')
            input_ids = inputs_encoding['input_ids'].to(self.device)
            attention_mask = inputs_encoding['attention_mask'].to(self.device)

        self.logger.debug(f"Tokenization complete in {time_meter.timer('Tokenization').duration:.4f} seconds.")
        
        # pre-compute the logprobs of scores for reusing in the generation process
        # scores = torch.nn.functional.log_softmax(
        #     torch.as_tensor(scores, dtype=torch.float32), dim=-1).to(self.device)
        scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)

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
        
        new_scores = scores * self.rerank_momentum + new_scores * (1 - self.rerank_momentum)
        new_scores = torch.nn.functional.log_softmax(new_scores, dim=-1)
        # new_scores = torch.logsumexp(
        #     torch.stack([
        #         scores + torch.log(self.rerank_momentum), 
        #         new_scores + torch.log(1 - self.rerank_momentum)
        #     ]), dim=0
        # )
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
            past_key_values=None, preemptable=False) -> CausalOutput:
        """
        Prefill concatenations of context_ids and input_ids, generate the next token, and return the logprobs
        during the prefilling process. 
        
        Output Aggregation: 
        probs = softmax(w)^T softmax(z) -> log(probs) = logsumexp(logsoftmax(w)+logsoftmax(z))
        """
        # ipdb.set_trace()
        with time_meter.timer("Decoding"):
            if preemptable:
                self.input_queue.put((input_ids, attention_mask, {"past_key_values": past_key_values}))
                output = self.output_queue.get()
                if output is None: return None
            else:
                output = self.generator(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    past_key_values=past_key_values
                )
            logscores = torch.nn.functional.log_softmax(scores, dim=0)
            logprobs = torch.nn.functional.log_softmax(    # (s_aggregate, s_vocab)
                output.logits[:, -1] / self.generator.sampler.temperature, dim=-1)
            logprobs = logprobs.permute(1, 0)              # (s_vocab, s_aggregate)
            logprobs = logprobs + logscores                # (s_vocab, s_aggregate) + (s_aggregate,)
            logprobs = torch.logsumexp(logprobs, dim=1)    # (s_vocab,)
            next_token = self.generator.sampler(torch.exp(logprobs).unsqueeze(0))[0]
        self.logger.debug(f"Decoding complete in {time_meter.timer('Decoding').duration:.4f} seconds.")

        return CausalOutput(
            next_token=next_token, logprobs=logprobs, 
            past_key_values=output.past_key_values
        )

    def generate(self, last_token: int, scores: torch.Tensor, attention_mask: torch.Tensor, past_key_values: List[torch.Tensor]):
        input_ids = torch.as_tensor([last_token], dtype=torch.long, device=self.device)
        input_ids = input_ids.repeat(self.aggregate_size, 1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=1)
        return self._generate(input_ids, attention_mask, scores, past_key_values=past_key_values, preemptable=True), attention_mask


class Aggregator(threading.Thread):
    def __init__(
            self, 
            draft_tokens_loc: queue.Queue, 
            draft_tokens_rem: queue.Queue, 
            target_tokens: queue.Queue,
            sampler: Sampler,
            transceiver: Transceiver,
            max_new_tokens: int
        ):
        super().__init__()
        self.draft_tokens_loc = draft_tokens_loc
        self.draft_tokens_rem = draft_tokens_rem
        self.target_tokens = target_tokens
        self.sampler = sampler
        self.transceiver = transceiver
        self.logger = Logger.build(__class__.__name__, level="INFO")
        self.logger.info("Aggregator initialized.")
        self.max_new_tokens = max_new_tokens

    def run(self):
        while self.target_tokens.qsize() < self.max_new_tokens:
            draft_token_l, logprobs_l, score_l, step = self.draft_tokens_loc.get()
            while step != self.target_tokens.qsize():
                draft_token_l, logprobs_l, score_l, step = self.draft_tokens_loc.get()
            draft_token_r, logprobs_r, score_r, step = self.draft_tokens_rem.get()
            while step != self.target_tokens.qsize():
                draft_token_r, logprobs_r, score_r, step = self.draft_tokens_rem.get()
            next_token = self.aggregate(
                draft_token_l, draft_token_r, logprobs_l, logprobs_r, score_l, score_r)
            self.target_tokens.put(next_token)
            self.transceiver.check_recompute(False)
            self.transceiver._send_target_token(next_token, False)

    def aggregate(self, draft_token_l: int, draft_token_r: int, logprobs_l: torch.Tensor, logprobs_r: torch.Tensor, score_l: float, score_r: float):
        scores = torch.as_tensor([score_l, score_r], dtype=torch.float32, device=logprobs_l.device)
        scores = scores - torch.logsumexp(scores, dim=0)
        logprobs = torch.stack([logprobs_l, logprobs_r], dim=1)  # (s_vocab, 2)
        logprobs = logprobs + scores                             # (s_vocab, 2) + (2,)
        logprobs = torch.logsumexp(logprobs, dim=1)              # (s_vocab,)
        next_token = self.sampler(torch.exp(logprobs).unsqueeze(0))[0]
        
        real_weight_l = math.exp(score_l) / (math.exp(score_l) + math.exp(score_r))
        real_weight_r = 1 - real_weight_l
        self.logger.debug(
            f"Local(draft={draft_token_l}, weight={real_weight_l:>.2f}), Remote(draft={draft_token_r}, weight={real_weight_r:>.2f}) => Target({next_token})"
        )
        return next_token

class Dragon(Transceiver):

    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.rag = RagPipeline(config, self.logger, self.rank)
        self.ready_for_generation = False
        self.register_observers(self.collect_observers())
        self.send(Message.READY_FOR_GENERATION, None)

        # Define draft queues
        self.draft_tokens_rem = queue.Queue(0)
        self.draft_tokens_loc = queue.Queue(0)

        # Define target queue
        self.target_tokens = queue.Queue(0)
    
        # Aggregator
        self.aggregator = None
    
    def shutdown(self):
        self.send(Message.SHUTDOWN, None)
        self._shutdown()

    def _shutdown(self):
        self.logger.info("Shutting down.")
        if self.aggregator and self.aggregator.is_alive():
            self.aggregator.join()
        if self.decoding_thread.is_alive():
            self.decoding_thread.join()
        if self.rag.generator.is_alive():
            self.rag.generator.close()
        self.terminate()

    def init_aggregator(self, max_new_tokens: int):
        if self.rank == 1:  # 1 for device-side, 0 for cloud-side
            self.aggregator = Aggregator(
                self.draft_tokens_loc, 
                self.draft_tokens_rem, 
                self.target_tokens, 
                self.rag.generator.sampler,
                self,
                max_new_tokens
            )
            self.aggregator.start()

    def query(self, query: str, prompt_template: str, max_new_tokens: int):
        self.output_ids = []
        self.generate_rem(query, prompt_template, max_new_tokens)
        self.init_aggregator(max_new_tokens)
        self.decoding_thread = threading.Thread(target=self.generate_loc, args=(query, prompt_template, max_new_tokens))
        self.decoding_thread.start()
        # decoded_output = self.generate_loc(query, prompt_template, max_new_tokens)
        self.decoding_thread.join()
        self.aggregator.join()
        self.logger.debug(f"Got output_ids {self.output_ids} of length {len(self.output_ids)}.")
        decoded_output = self.rag.generator.tokenizer.decode(self.output_ids, skip_special_tokens=True)
        return decoded_output
    
    def generate_rem(self, query: str, prompt_template: str, max_new_tokens: int):
        self.send(Message.BEGIN_GENERATE, (query, prompt_template, max_new_tokens))

    def generate_loc(self, query: str, prompt_template: str, max_new_tokens: int):
        # start a new thread to do generation

        step = 0
        input_ids, attention_mask, scores, passages = self.rag._prepare_inputs_for_generation(query, prompt_template)
        context_length = input_ids.shape[1]
        output = self.rag._generate(input_ids, attention_mask, scores)
        score = torch.logsumexp(scores, dim=0).item()
        self._send_draft_token(output.next_token, output.logprobs, score, step)
        while True:
            self.logger.debug(f"n_targets={self.target_tokens.qsize()}, n_locals={self.draft_tokens_loc.qsize()}, n_remotes={self.draft_tokens_rem.qsize()}")
            if self.target_tokens.qsize() >= max_new_tokens:
                break
            # if self.draft_tokens_loc.qsize() >= max_new_tokens:
            #     continue
            # generated_ids.append(next_token)
            # generated_text = self.rag.generator.tokenizer.decode(generated_ids)
            # scores = self.rag.rerank_passages(query, generated_text, passages, scores, step)
            temp_output = self.rag.generate(
                output.next_token, scores, attention_mask, past_key_values=output.past_key_values)
            if temp_output[0] is None:
                # re-computing the last token
                self.logger.debug("Recomputing the last token.")
                self.rag.generator.stop_event.clear()
                step = self.target_tokens.qsize() - 1
                target_last_token = self.target_tokens.queue[-1]
                # TODO: scroll back the scores
                attention_mask = attention_mask[:, : context_length + step + 1]
                # self.logger.info("Scrolling back the past_key_values.")
                # self.logger.info(f"Before: {output.past_key_values[0][0].shape=}")
                output.past_key_values = list(output.past_key_values)
                for i, _ in enumerate(output.past_key_values):
                    output.past_key_values[i] = list(output.past_key_values[i])
                    output.past_key_values[i][0] = output.past_key_values[i][0][..., : context_length + step + 1, :]
                    output.past_key_values[i][1] = output.past_key_values[i][1][..., : context_length + step + 1, :]
                # self.logger.info(f"After: {output.past_key_values[0][0].shape=}")
                temp_output = self.rag.generate(
                    target_last_token, scores, attention_mask, past_key_values=output.past_key_values)
                continue
            step += 1
            output, attention_mask = temp_output
            self._send_draft_token(output.next_token, output.logprobs, score, step)
        self.logger.debug("Generation complete.")
        self.output_ids = [self.target_tokens.get() for _ in range(max_new_tokens)]
        # clean up
        self.receive_queue.queue.clear()
        self.rag.generator.input_queue.queue.clear()
        self.draft_tokens_loc.queue.clear()
        self.draft_tokens_rem.queue.clear()
        self.target_tokens.queue.clear()
        self.logger.debug("Cleaned up.")

    def check_recompute(self, accept: bool):
        if not accept:
            self.rag.generator.input_queue.queue.clear()
            self.rag.generator.stop_event.set()
            self.logger.debug("Preempting the current generation process.")

            self.draft_tokens_loc.queue.clear()
            self.draft_tokens_rem.queue.clear()
            

    def collect_observers(self):
        return [
            self._rx_ready_for_generation,
            self._rx_begin_generate,
            self._rx_draft_token,
            self._rx_target_token,
            self._rx_shutdown
        ]

    def _rx_ready_for_generation(self, mtype: int, mbody: object):
        if mtype != Message.READY_FOR_GENERATION: return False
        self.ready_for_generation = True
        self.logger.debug("Remote is ready for generation.")
        return True
    
    def _rx_begin_generate(self, mtype: int, mbody: object):
        if mtype != Message.BEGIN_GENERATE: return False
        query, prompt_template, max_new_tokens = mbody
        self.logger.debug(f"Generating response for query: {query}")
        self.decoding_thread = threading.Thread(target=self.generate_loc, args=(query, prompt_template, max_new_tokens))
        self.decoding_thread.start()
        # decoded_output = self.generate_loc(query, prompt_template, max_new_tokens)
        return True
    
    def _rx_draft_token(self, mtype: int, mbody: object):
        if mtype != Message.DRAFT_TOKEN: return False
        next_token, logprobs, score, step = mbody
        logprobs = torch.as_tensor(logprobs, dtype=torch.float32, device=self.rag.device)
        self.draft_tokens_rem.put((next_token, logprobs, score, step))
        return True
    
    def _rx_target_token(self, mtype: int, mbody: object):
        if mtype != Message.TARGET_TOKEN: return False
        target_token, accept = mbody
        self.target_tokens.put(target_token)
        self.check_recompute(accept)
        # self.logger.info("Successfully received target token.")
        return True

    def _rx_shutdown(self, mtype: int, mbody: object):
        if mtype != Message.SHUTDOWN: return False
        self.logger.info("Received shutdown signal.")
        self._shutdown()
        return True

    def _send_draft_token(self, token: int, logprobs: torch.Tensor, score: float, step: int):
        self.draft_tokens_loc.put((token, logprobs, score, step))
        self.send(Message.DRAFT_TOKEN, (token, logprobs.tolist(), score, step))
    
    def _send_target_token(self, token: int, accept: bool):
        self.send(Message.TARGET_TOKEN, (token, accept))
