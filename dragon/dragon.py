import json
from queue import Queue
import threading
from typing import List

from pathlib import Path
from .rag import Rag
from .transceiver import Message
from .config import DragonConfig
from .aggregator import Aggregator
from .decoder import Decoder
from .queues import DraftQueue, DraftItem
from .transceiver import Transceiver
from .profiler import Profiler
from .utils.stable import terminate_thread
from .utils.mlogging import Logger
from .utils.meter import Statistics, TimeMeter


logging_level = "INFO"
time_meter = TimeMeter()

class Dragon:

    def __init__(self, config: DragonConfig):
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.ready_for_generation = False
        self.config = config
        
        self.transceiver = Transceiver(config)
        self.transceiver.register_observers(self._collect_observers())
        self.rag = Rag(config)

        self.draft_queue_rem = DraftQueue()
        self.draft_queue_loc = DraftQueue()

        self.target_tokens = Queue(0)
        self.output_tokens = Queue(0)
    
        self.aggregator = None
        self.decoder = None

        self.is_client = config.trans.rank != 0
        self.aggregaton_mode = config.aggregator.mode
        self.transceiver.send(Message.READY_FOR_GENERATION, None)

        self.stats = Statistics()
        self.stats_output_dir = Path(f"outputs/stats/{config.aggregator.mode}/")
        self.platform = f"{'device' if self.is_client else 'cloud'}"
    
    def shutdown(self):
        self.transceiver.send(Message.SHUTDOWN, None)
        self._shutdown()

    def _shutdown(self):
        from .aggregator import stats as aggregator_stats
        (self.stats | aggregator_stats).dump(self.stats_output_dir / f"{self.platform}.json")
        self.logger.info("Shutting down.")
        terminate_thread(self.aggregator)
        terminate_thread(self.decoder)
        terminate_thread(self.rag.generator)
        self.logger.info("Dragon threads shutdown.")
        self.transceiver.terminate()

    def _build_aggregator(self, max_new_tokens: int):
        thread = Aggregator(
            self.draft_queue_loc, 
            self.draft_queue_rem, 
            self.target_tokens, 
            self.rag.generator.sampler,
            max_new_tokens,
            self.aggregaton_mode
        )
        thread.start()
        return thread
    
    def _build_decoder(self, query, prompt_template, max_new_tokens):
        thread = Decoder(
            self.rag, self._send_draft_token, self.output_tokens, 
            query, prompt_template, max_new_tokens)
        thread.start()
        return thread
    
    def _build_profiler(self, query, prompt_template, max_new_tokens):
        thread = Profiler(
            self.config, self.rag, query, prompt_template, max_new_tokens
        )
        thread.start()
        return thread
    
    def _clean_up(self):
        self.transceiver.receive_queue.queue.clear()
        self.rag.generator.input_queue.queue.clear()
        self.rag.generator.output_queue.queue.clear()
        self.draft_queue_loc.clear()
        self.draft_queue_rem.clear()
        self.target_tokens.queue.clear()
        self.logger.debug("Cleaned up.")

    def _start_up(self, query, prompt_template, max_new_tokens):
        self.stats.new_record()
        self.recompute_checker = threading.Thread(
            target=self.check_recompute, args=(max_new_tokens,))
        self.recompute_checker.start()
        self.decoder = self._build_decoder(query, prompt_template, max_new_tokens)
        if self.is_client:
            self.aggregator = self._build_aggregator(max_new_tokens)
            self.decoder.join()
            self.aggregator.join()
            self.recompute_checker.join()
            self._clean_up()

    def query(self, query: str, prompt_template: str, max_new_tokens: int):
        self.profiler = self._build_profiler(query, prompt_template, max_new_tokens)
        self.profiler.join()
        
        # Inform remote decoding
        self._send_begin_generate(query, prompt_template, max_new_tokens)
        self._start_up(query, prompt_template, max_new_tokens)

        # Get output text
        output_ids = self.decoder.output_ids
        output_txt = self.rag.generator.tokenizer.decode(
            output_ids, skip_special_tokens=True)
        return output_txt

    def check_recompute(self, n_steps: int):
        step = 0
        while step < n_steps:
            with time_meter.timer("LatencyPerToken"):
                target_token, accept_loc, accept_rem = self.target_tokens.get()
            self.stats.update(time_meter.timer('LatencyPerToken'))
            self.stats.update(name='AcceptanceLoc', stat=accept_loc)
            self.stats.update(name='AcceptanceRem', stat=accept_rem)
            if self.is_client:
                self._send_target_token(target_token, accept_loc, accept_rem)
            self.output_tokens.put(target_token)
            if not accept_loc:
                self.logger.debug("Preempting the current generation process.")
                self.rag.generator.preempt_event.set()
                self.draft_queue_loc.clear()
            if not accept_rem:
                self.draft_queue_rem.clear()
            step += 1

    def _collect_observers(self):
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
        self._start_up(query, prompt_template, max_new_tokens)
        return True
    
    def _rx_draft_token(self, mtype: int, mbody: object):
        if mtype != Message.DRAFT_TOKEN: return False
        self.draft_queue_rem.put(DraftItem.from_tuple(mbody))
        return True
    
    def _rx_target_token(self, mtype: int, mbody: object):
        if mtype != Message.TARGET_TOKEN: return False
        self.target_tokens.put(mbody)
        return True

    def _rx_shutdown(self, mtype: int, mbody: object):
        if mtype != Message.SHUTDOWN: return False
        self.logger.info("Received shutdown signal.")
        self._shutdown()
        return True
    
    def _send_begin_generate(self, query: str, prompt_template: str, max_new_tokens: int):
        self.transceiver.send(Message.BEGIN_GENERATE, (query, prompt_template, max_new_tokens))

    def _send_draft_token(self, draft_item: DraftItem):
        self.draft_queue_loc.put(draft_item)
        self.transceiver.send(Message.DRAFT_TOKEN, draft_item.as_tuple())
    
    def _send_target_token(self, token: int, accept_loc: bool, accept_rem: bool):
        # remote and local are relative concepts
        self.transceiver.send(Message.TARGET_TOKEN, (token, accept_rem, accept_loc))
