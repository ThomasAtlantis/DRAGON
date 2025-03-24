from dataclasses import dataclass
from queue import Queue
from typing import Tuple
import lz4.frame
import numpy as np
import torch


def topp_pack(logprobs: torch.Tensor, probs_topp: float, probs_type: np.dtype) -> Tuple[bytes, bytes]:
    probs = torch.exp(logprobs)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    reserved_indices = sorted_indices[cumulative_probs <= probs_topp]
    if reserved_indices.numel() == 0:
        reserved_indices = torch.LongTensor([sorted_indices[0]])
    reserved_logprobs = logprobs[reserved_indices]
    reserved_logprobs -= torch.logsumexp(reserved_logprobs, dim=0)
    reserved_indices = reserved_indices.cpu().numpy()
    reserved_indices = (reserved_indices - 30000).astype(np.int16)
    reserved_indices = lz4.frame.compress(reserved_indices.tobytes(), compression_level=9)
    reserved_logprobs = reserved_logprobs.cpu().to(torch.float16).numpy().astype(probs_type)
    reserved_logprobs = lz4.frame.compress(reserved_logprobs.tobytes(), compression_level=9)
    return reserved_indices, reserved_logprobs

def topp_unpack(compressed_data: Tuple[bytes, bytes], vocab_size: int, probs_type: np.dtype) -> torch.Tensor:
    reserved_indices, reserved_logprobs = compressed_data
    reserved_indices = np.frombuffer(
        lz4.frame.decompress(reserved_indices), dtype=np.int16)
    reserved_indices = torch.LongTensor(reserved_indices.copy()) + 30000
    reserved_logprobs = np.frombuffer(
        lz4.frame.decompress(reserved_logprobs), dtype=probs_type)
    reserved_logprobs = torch.FloatTensor(reserved_logprobs.copy())
    logprobs = torch.full((vocab_size, ), fill_value=-torch.inf).float()
    logprobs[reserved_indices] = reserved_logprobs
    return logprobs
    

@dataclass
class DraftItem:

    token: int
    logprobs: torch.Tensor
    weight: float
    step: int

    _probs_type: np.dtype = np.float16
    _vocab_size: int = 50257
    _probs_topp: float = 0.8

    @staticmethod
    def from_tuple(args):
        item = DraftItem(*args)
        if DraftItem._probs_topp < 1.0:
            item.logprobs = topp_unpack(item.logprobs, DraftItem._vocab_size, DraftItem._probs_type)
        else:
            logprobs_bytes = item.logprobs
            logprobs_decompressed = np.array(lz4.frame.decompress(logprobs_bytes))
            item.logprobs = torch.from_numpy(np.frombuffer(logprobs_decompressed, dtype=DraftItem._probs_type)).float()
        return item

    def as_tuple(self):
        if DraftItem._probs_topp < 1.0:
            logprobs_compressed = topp_pack(self.logprobs, DraftItem._probs_topp, DraftItem._probs_type)
        else:
            logprobs_bytes = self.logprobs.cpu().to(torch.float16).numpy().astype(DraftItem._probs_type).tobytes()
            logprobs_compressed = lz4.frame.compress(logprobs_bytes, compression_level=9)
        return (
            self.token,
            logprobs_compressed,
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