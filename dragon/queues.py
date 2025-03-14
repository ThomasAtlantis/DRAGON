from dataclasses import dataclass
from queue import Queue
import torch


@dataclass
class DraftItem:

    token: int
    logprobs: torch.Tensor
    weight: float
    step: int

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
            self.step
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