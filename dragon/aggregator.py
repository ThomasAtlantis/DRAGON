 
import math
from queue import Queue
import threading

import torch

from dragon.generator import Sampler
from dragon.queues import DraftItem, DraftQueue
from dragon.utils.meter import TimeMeter, Statistics
from dragon.utils.mlogging import Logger


logging_level = "INFO"
time_meter = TimeMeter()
stats = Statistics()


class Aggregator(threading.Thread):

    def __init__(
            self, 
            draft_queue_loc: DraftQueue, 
            draft_queue_rem: DraftQueue, 
            target_tokens: Queue,
            sampler: Sampler,
            n_steps: int,
            mode: str
        ):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.draft_queue_loc = draft_queue_loc
        self.draft_queue_rem = draft_queue_rem
        self.target_tokens = target_tokens
        self.sampler = sampler
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.logger.info("Aggregator initialized.")
        self.n_steps = n_steps
        self.step = 0
        self.mode = mode
        if mode == "synchronized":
            self.aggregate = self.aggregate_synchronized
        elif mode == "speculative":
            self.aggregate = self.aggregate_speculative
    
    def _get_draft_item(self, queue: DraftQueue) -> DraftItem:
        draft_item = queue.get()
        while draft_item.step != self.step:
            draft_item = queue.get()
        return draft_item
    
    def run(self):
        self.target_tokens.queue.clear()
        stats.new_record()
        while self.step < self.n_steps:
            draft_loc = self._get_draft_item(self.draft_queue_loc)
            draft_rem = self._get_draft_item(self.draft_queue_rem)
            with time_meter.timer("AggregateLatency"):
                next_token, accept_loc, accept_rem = self.aggregate(draft_loc, draft_rem)
            stats.update(time_meter.timer("AggregateLatency"))
            self.target_tokens.put((next_token, accept_loc, accept_rem))
            self.step += 1
        self.logger.info("Aggregation complete.")

    def aggregate_synchronized(self, draft_loc: DraftItem, draft_rem: DraftItem):
        device = draft_loc.logprobs.device
        draft_rem.logprobs = draft_rem.logprobs.to(device)
        scores = torch.as_tensor([draft_loc.weight, draft_rem.weight], dtype=torch.float32, device=device)
        scores = torch.log_softmax(scores, dim=0)
        logprobs = torch.stack([draft_loc.logprobs, draft_rem.logprobs], dim=1)  # (s_vocab, 2)
        logprobs = logprobs + scores                             # (s_vocab, 2) + (2,)
        logprobs = torch.logsumexp(logprobs, dim=1)              # (s_vocab,)
        next_token = self.sampler(torch.exp(logprobs))
        
        real_weight_l = math.exp(draft_loc.weight) / (math.exp(draft_loc.weight) + math.exp(draft_rem.weight))
        real_weight_r = 1 - real_weight_l
        self.logger.debug(
            f"Local(draft={draft_loc.token}, weight={real_weight_l:>.2f}), Remote(draft={draft_rem.token}, weight={real_weight_r:>.2f}) => Target({next_token})"
        )
        return next_token, False, False
    
    def _speculative_sampling(self, draft_token: int, draft_probs: torch.Tensor, target_probs: torch.Tensor, residual_probs: torch.Tensor):
        if draft_probs[draft_token] <= target_probs[draft_token] \
            or torch.rand(1).to(draft_probs.device) < target_probs[draft_token] / draft_probs[draft_token]:
            return draft_token
        else:
            token = torch.multinomial(residual_probs, 1).cpu().item()
            return token

    def aggregate_speculative(self, draft_loc: DraftItem, draft_rem: DraftItem):
        device = draft_loc.logprobs.device
        draft_rem.logprobs = draft_rem.logprobs.to(device)
        scores = torch.as_tensor([draft_loc.weight, draft_rem.weight], dtype=torch.float32, device=device)
        scores = torch.log_softmax(scores, dim=0)
        logprobs = torch.stack([draft_loc.logprobs, draft_rem.logprobs], dim=1)  # (s_vocab, 2)
        logprobs = logprobs + scores                             # (s_vocab, 2) + (2,)
        logprobs = torch.logsumexp(logprobs, dim=1)              # (s_vocab,)

        probs_loc = torch.exp(draft_loc.logprobs)
        probs_rem = torch.exp(draft_rem.logprobs)
        probs_agg = self.sampler.transform(logprobs.exp())
        residual_probs_loc = torch.maximum(torch.zeros_like(probs_agg), probs_agg - probs_loc)
        residual_probs_loc = residual_probs_loc / residual_probs_loc.sum()
        residual_probs_rem = torch.maximum(torch.zeros_like(probs_agg), probs_agg - probs_rem)
        residual_probs_rem = residual_probs_rem / residual_probs_rem.sum()

        next_token_loc = self._speculative_sampling(draft_loc.token, probs_loc, probs_agg, residual_probs_loc)
        next_token_rem = self._speculative_sampling(draft_rem.token, probs_rem, probs_agg, residual_probs_rem)
        next_token = next_token_loc if torch.rand(1) < 0.5 else next_token_rem
        
        accept_loc = next_token == draft_loc.token
        accept_rem = next_token == draft_rem.token
        return next_token, accept_loc, accept_rem
        