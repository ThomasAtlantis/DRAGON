import math
from queue import Queue
import threading
import torch

from .rag import Rag
from .transceiver import Message
from .config import DragonConfig
from .generator import Sampler
from .decoder import Decoder
from .queues import DraftQueue, DraftItem
from .transceiver import BaseTransceiver
from .utils.stable import terminate_thread
from .utils.mlogging import Logger
from .utils.meter import TimeMeter


logging_level = "INFO"
time_meter = TimeMeter()

    
class Aggregator(threading.Thread):

    def __init__(
            self, 
            draft_queue_loc: DraftQueue, 
            draft_queue_rem: DraftQueue, 
            target_tokens: Queue,
            sampler: Sampler,
            transceiver: BaseTransceiver,
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
    

class Dragon(BaseTransceiver):

    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.rag = Rag(config)
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
            query, prompt_template, max_new_tokens)
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
