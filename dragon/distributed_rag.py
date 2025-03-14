from dataclasses import dataclass
import math
from queue import Queue
import threading
import torch
from typing import List, Tuple
from sentence_transformers import CrossEncoder


from .transceiver import Message
from .config import DragonConfig
from .generator.generator import Generator, Sampler
# from .retriever.retriever import CustomRetriever as Retriever
# from .retriever.retriever import DPRRetriever as Retriever
from .retriever.retriever import DPRRetrieverClient as Retriever
from .utils.stable import terminate_thread
from .transceiver import Transceiver
from .utils.mlogging import Logger
from logging import Logger as PyLogger
from .utils.meter import TimeMeter


logging_level = "INFO"
time_meter = TimeMeter()

@dataclass
class CausalOutput:
    next_token: int
    logprobs: torch.Tensor
    past_key_values: List[torch.Tensor] = None
    fingerprint: list = None

class Preempted(Exception):
    pass

class PreemptableGenerator(threading.Thread, Generator):

    
    def __init__(self, config: DragonConfig, input_queue: Queue, output_queue: Queue):
        Generator.__init__(self, config)
        threading.Thread.__init__(self, name=__class__.__name__)
        self.logger = Logger.build(
            __class__.__name__, level=logging_level)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.preempt_event = threading.Event()
        self.stop_event = threading.Event()

        total_modules = sum(1 for _ in self.model.modules())
        def forward_hook(module_name, depth):
            def hook(module, input, output):
                if self.preempt_event.is_set():
                    raise Preempted(f"Preempted at {depth / total_modules:.2%}")
                    # return 0
                return output
            return hook

        for idx, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, torch.nn.Module):
                module.register_forward_hook(forward_hook(name, idx))

    def close(self):
        self.stop_event.set()
        self.preempt_event.set()
        self.input_queue.put(None)

    def run(self):
        while not self.stop_event.is_set():
            try:
                input_ids, attention_mask, kwargs = self.input_queue.get()
            except Exception as e:
                self.logger.debug("Generator stopped")
                break
            try:
                self.output_queue.put(
                    self(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        **kwargs
                    )
                )
            except Preempted as e:
                self.logger.warning(e)
                self.preempt_event.clear()
                self.input_queue.queue.clear()
                self.output_queue.put(None)
            except RuntimeError as e:
                self.logger.error(e)


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
        self.input_queue = Queue(0)
        self.output_queue = Queue(0)
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
            self.retriever.n_docs = self.config.retriever.n_docs * 2
            passages, scores = self.retriever.retrieve_passages([query])[0]
            # passages, scores = passages[self.rank_idx::2], scores[self.rank_idx::2]
            if self.rank_idx == 0:
                passages, scores = passages[: self.config.retriever.n_docs], scores[: self.config.retriever.n_docs]
            else:
                passages, scores = passages[self.config.retriever.n_docs:], scores[self.config.retriever.n_docs:]
        self.logger.debug(f"Retrieval complete in {time_meter.timer('Retrieval').duration:.4f} seconds.")
        with time_meter.timer("Tokenization"):
            # assemble input from passages, query and prompt_template
            passages = [p["text"] for p in passages]
            passages, scores = self._group_passages(passages, scores)
            passages = [self._crop_context(passage) for passage in passages]
            input_text_list = [
                prompt_template.format(
                    context=passage, 
                    query=query
                ) for passage in passages
            ]

            # encode input into input_ids and attention_mask    
            inputs_encoding = self.generator.tokenizer.batch_encode_plus(
                input_text_list, padding='longest', return_tensors='pt')
            input_ids = inputs_encoding['input_ids'].to(self.device)
            attention_mask = inputs_encoding['attention_mask'].to(self.device)

        self.logger.debug(f"Tokenization complete in {time_meter.timer('Tokenization').duration:.4f} seconds.")
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
            past_key_values=output.past_key_values,
            fingerprint=output.logits[:, -1].cpu().max(dim=-1)
        )

    def generate(self, last_token: int, scores: torch.Tensor, attention_mask: torch.Tensor, past_key_values: List[torch.Tensor]):
        input_ids = torch.as_tensor([last_token], dtype=torch.long, device=self.device)
        input_ids = input_ids.repeat(self.aggregate_size, 1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=1)
        return self._generate(input_ids, attention_mask, scores, past_key_values=past_key_values, preemptable=True), attention_mask

@dataclass
class DraftItem:

    token: int
    logprobs: torch.Tensor
    weight: float
    step: int
    fingerprint: torch.Tensor = None

    @staticmethod
    def from_tuple(args):
        item = DraftItem(*args)
        item.logprobs = torch.as_tensor(item.logprobs, dtype=torch.float32)
        return item

    def as_tuple(self):
        return (
            self.token,
            self.logprobs.cpu().tolist(),
            self.weight,
            self.step,
            self.fingerprint
        )

class DraftQueue:

    def __init__(self):
        self.queue = Queue(0)

    def put(self, item: DraftItem):
        self.queue.put(item)

    def get(self) -> DraftItem:
        return self.queue.get()

    def clear(self):
        self.queue.queue.clear()

    def qsize(self) -> int:
        return self.queue.qsize()
    
class Aggregator(threading.Thread):

    def __init__(
            self, 
            draft_queue_loc: DraftQueue, 
            draft_queue_rem: DraftQueue, 
            target_tokens: Queue,
            sampler: Sampler,
            transceiver: Transceiver,
            max_new_tokens: int
        ):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.draft_queue_loc = draft_queue_loc
        self.draft_queue_rem = draft_queue_rem
        self.target_tokens = target_tokens
        self.sampler = sampler
        self.transceiver = transceiver
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.logger.info("Aggregator initialized.")
        self.max_new_tokens = max_new_tokens
    
    def _get_draft_item(self, queue: DraftQueue) -> DraftItem:
        draft_item = queue.get()
        while draft_item.step != self.target_tokens.qsize():
            draft_item = queue.get()
        return draft_item
    
    def run(self):
        while self.target_tokens.qsize() < self.max_new_tokens:
            draft_loc = self._get_draft_item(self.draft_queue_loc)
            draft_rem = self._get_draft_item(self.draft_queue_rem)
            next_token = self.aggregate(draft_loc, draft_rem)
            self.target_tokens.put(next_token)
            self.transceiver.check_recompute(False)
            self.transceiver._send_target_token(next_token, False)

    def aggregate(self, draft_loc: DraftItem, draft_rem: DraftItem):
        device = draft_loc.logprobs.device
        draft_rem.logprobs = draft_rem.logprobs.to(device)
        scores = torch.as_tensor([draft_loc.weight, draft_rem.weight], dtype=torch.float32, device=device)
        scores = scores - torch.logsumexp(scores, dim=0)
        logprobs = torch.stack([draft_loc.logprobs, draft_rem.logprobs], dim=1)  # (s_vocab, 2)
        logprobs = logprobs + scores                             # (s_vocab, 2) + (2,)
        logprobs = torch.logsumexp(logprobs, dim=1)              # (s_vocab,)
        next_token = self.sampler(torch.exp(logprobs).unsqueeze(0))[0]
        
        real_weight_l = math.exp(draft_loc.weight) / (math.exp(draft_loc.weight) + math.exp(draft_rem.weight))
        real_weight_r = 1 - real_weight_l
        self.logger.debug(
            f"Local(draft={draft_loc.token}, weight={real_weight_l:>.2f}), Remote(draft={draft_rem.token}, weight={real_weight_r:>.2f}) => Target({next_token})"
        )
        return next_token
    
#         self.rag.generator.preempt_event.clear()
#         output = self.rag._generate(self.input_ids, self.attention_mask, self.scores)
#         self.transceiver._send_draft_token(
#             DraftItem(
#                 token=output.next_token, 
#                 logprobs=output.logprobs, 
#                 weight=self.weight, step=self.step, 
#                 fingerprint=output.fingerprint
#             )
#         )
#         while True:
#             if self.target_tokens.qsize() >= self.max_new_tokens:
#                 break
#             temp_output = self.rag.generate(
#                 output.next_token, self.scores, self.attention_mask, past_key_values=output.past_key_values)
#             if temp_output[0] is None:
#                 # re-computing the last token
#                 self.logger.debug("Recomputing the last token.")
#                 self.step = self.target_tokens.qsize() - 1
#                 output.next_token = self.target_tokens.queue[-1]
#                 # TODO: scroll back the scores
#                 self.attention_mask = self.attention_mask[:, : self.context_length + self.step]
#                 output.past_key_values = list(output.past_key_values)
#                 for i, _ in enumerate(output.past_key_values):
#                     output.past_key_values[i] = list(output.past_key_values[i])
#                     output.past_key_values[i][0] = output.past_key_values[i][0][..., : self.context_length + self.step, :]
#                     output.past_key_values[i][1] = output.past_key_values[i][1][..., : self.context_length + self.step, :]
#                     output.past_key_values[i] = tuple(output.past_key_values[i])
#                 output.past_key_values = tuple(output.past_key_values)
#                 continue
#             self.step += 1
#             output, self.attention_mask = temp_output
#             self.transceiver._send_draft_token(
#                 DraftItem(
#                     token=output.next_token, 
#                     logprobs=output.logprobs, 
#                     weight=self.weight, step=self.step, 
#                     fingerprint=output.fingerprint
#                 )
#             )
        
#         self.logger.debug("Generation complete.")

class Decoder(threading.Thread):
    
    def __init__(self, rag: RagPipeline, transceiver: Transceiver, target_tokens: Queue, query: str, prompt_template: str, max_new_tokens: int, aggregator: Aggregator):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.rag = rag
        self.transceiver = transceiver
        self.query = query
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.logger = Logger.build(__class__.__name__, logging_level)
        self.target_tokens = target_tokens
        self.aggregator = aggregator
        self.logger.info("Decoder initialized.")

        self.input_ids, self.attention_mask, self.scores, self.passages = self.rag._prepare_inputs_for_generation(self.query, self.prompt_template)
        self.context_length = self.input_ids.shape[1]
        self.target_tokens.queue.clear()
        self.rag.generator.preempt_event.clear()

        self.output_ids = []

    def prefilling(self) -> CausalOutput:
        return self.rag._generate(self.input_ids, self.attention_mask, self.scores)

    def run(self):
        step = 0
        
        output = self.prefilling()
        weight = torch.logsumexp(self.scores, dim=0).item()
        self.transceiver._send_draft_token(
            DraftItem(
                token=output.next_token, 
                logprobs=output.logprobs, 
                weight=weight, step=step, 
                fingerprint=output.fingerprint
            )
        )
        while True:
            if self.target_tokens.qsize() >= self.max_new_tokens:
                break
            temp_output = self.rag.generate(
                output.next_token, self.scores, self.attention_mask, past_key_values=output.past_key_values)
            if temp_output[0] is None:
                # re-computing the last token
                self.logger.debug("Recomputing the last token.")
                step = self.target_tokens.qsize() - 1
                output.next_token = self.target_tokens.queue[-1]
                # TODO: scroll back the scores
                self.attention_mask = self.attention_mask[:, : self.context_length + step]
                output.past_key_values = list(output.past_key_values)
                for i, _ in enumerate(output.past_key_values):
                    output.past_key_values[i] = list(output.past_key_values[i])
                    output.past_key_values[i][0] = output.past_key_values[i][0][..., : self.context_length + step, :]
                    output.past_key_values[i][1] = output.past_key_values[i][1][..., : self.context_length + step, :]
                    output.past_key_values[i] = tuple(output.past_key_values[i])
                output.past_key_values = tuple(output.past_key_values)
                continue
            step += 1
            output, self.attention_mask = temp_output
            # self._send_draft_token(output.next_token, output.logprobs, score, step, fingerprint=output.fingerprint)
            self.transceiver._send_draft_token(
                DraftItem(
                    token=output.next_token, 
                    logprobs=output.logprobs, 
                    weight=weight, step=step, 
                    fingerprint=output.fingerprint
                )
            )
        self.rag.generator.preempt_event.set()
        self.output_ids = [self.target_tokens.get() for _ in range(self.max_new_tokens)]
        self.logger.debug("Generation complete.")


class Dragon(Transceiver):

    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.rag = RagPipeline(config, self.logger, self.rank)
        self.ready_for_generation = False
        self.register_observers(self.collect_observers())
        self.send(Message.READY_FOR_GENERATION, None)

        # Define draft queues
        self.draft_queue_rem = DraftQueue()
        self.draft_queue_loc = DraftQueue()

        # Define target queue
        self.target_tokens = Queue(0)
    
        self.aggregator = None
        self.decoder = None
    
    def shutdown(self):
        self.send(Message.SHUTDOWN, None)
        self._shutdown()

    def _shutdown(self):
        self.logger.info("Shutting down.")
        terminate_thread(self.aggregator)
        terminate_thread(self.decoder)
        terminate_thread(self.rag.generator)
        self.logger.info("Dragon threads shutdown.")
        self.terminate()

    def _build_aggregator(self, max_new_tokens: int):
        thread = Aggregator(
            self.draft_queue_loc, 
            self.draft_queue_rem, 
            self.target_tokens, 
            self.rag.generator.sampler,
            self,
            max_new_tokens
        )
        thread.start()
        return thread
    
    def _build_decoder(self, query, prompt_template, max_new_tokens):
        thread = Decoder(
            self.rag, self, self.target_tokens, 
            query, prompt_template, max_new_tokens, 
            self.aggregator)
        thread.start()
        return thread
    
    def _clean_up(self):
        self.receive_queue.queue.clear()
        self.rag.generator.input_queue.queue.clear()
        self.rag.generator.output_queue.queue.clear()
        self.draft_queue_loc.clear()
        self.draft_queue_rem.clear()
        self.target_tokens.queue.clear()
        self.logger.debug("Cleaned up.")

    def query(self, query: str, prompt_template: str, max_new_tokens: int):
        # Inform remote decoding
        self._send_begin_generate(query, prompt_template, max_new_tokens)
        
        # Local decoding and aggregating
        self.decoder = self._build_decoder(query, prompt_template, max_new_tokens)
        self.aggregator = self._build_aggregator(max_new_tokens)
        self.decoder.join()
        self.aggregator.join()
        self._clean_up()
        
        # Get output text
        output_ids = self.decoder.output_ids
        output_txt = self.rag.generator.tokenizer.decode(
            output_ids, skip_special_tokens=True)
        return output_txt

    def check_recompute(self, accept: bool):
        if not accept:
            self.logger.debug("Preempting the current generation process.")
            self.rag.generator.preempt_event.set()
            self.draft_queue_loc.clear()
            self.draft_queue_rem.clear()
            

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
        self.decoder = self._build_decoder(query, prompt_template, max_new_tokens)
        return True
    
    def _rx_draft_token(self, mtype: int, mbody: object):
        if mtype != Message.DRAFT_TOKEN: return False
        self.draft_queue_rem.put(DraftItem.from_tuple(mbody))
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
    
    def _send_begin_generate(self, query: str, prompt_template: str, max_new_tokens: int):
        self.send(Message.BEGIN_GENERATE, (query, prompt_template, max_new_tokens))

    def _send_draft_token(self, draft_item: DraftItem):
        self.draft_queue_loc.put(draft_item)
        self.send(Message.DRAFT_TOKEN, draft_item.as_tuple())
    
    def _send_target_token(self, token: int, accept: bool):
        self.send(Message.TARGET_TOKEN, (token, accept))
