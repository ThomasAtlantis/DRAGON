 
import math
from queue import Queue
import threading

import torch

from dragon.generator import Sampler
from dragon.queues import DraftItem, DraftQueue
from dragon.utils.mlogging import Logger


logging_level = "INFO"


class Aggregator(threading.Thread):

    def __init__(
            self, 
            draft_queue_loc: DraftQueue, 
            draft_queue_rem: DraftQueue, 
            target_tokens: Queue,
            sampler: Sampler,
            n_steps: int
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
    
    def _get_draft_item(self, queue: DraftQueue) -> DraftItem:
        draft_item = queue.get()
        while draft_item.step != self.step:
            draft_item = queue.get()
        return draft_item
    
    def run(self):
        self.target_tokens.queue.clear()
        while self.step < self.n_steps:
            draft_loc = self._get_draft_item(self.draft_queue_loc)
            draft_rem = self._get_draft_item(self.draft_queue_rem)
            next_token, accept_loc, accept_rem = self.aggregate(draft_loc, draft_rem)
            self.target_tokens.put((next_token, accept_loc, accept_rem))
            self.step += 1
        self.logger.info("Aggregation complete.")

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
        return next_token, False, False
    