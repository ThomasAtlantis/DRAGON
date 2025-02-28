import torch
import numpy as np
from abc import ABC, abstractmethod


class Metric(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self):
        pass
    
    @abstractmethod
    def compute(self):        
        pass

class CrossEntropy(Metric):

    def __init__(self, device):
        super().__init__()
        self.data = []
        self.device = torch.device(device)

    def update(self, logprobs=None, labels=None):
        """
        @param logprobs: in the shape of (n_samples, s_vocab)
        @param labels  : in the shape of (n_samples, )
        """
        if logprobs is not None and labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.LongTensor(labels).to(self.device)
            labels = labels.unsqueeze(-1)
            cross_entropy = -torch.gather(logprobs, dim=1, index=labels).squeeze(-1)
            cross_entropy = cross_entropy.cpu().numpy()
            self.data.append(cross_entropy)
    
    def reset(self):
        self.data = []

    def compute(self):
        return np.mean(np.concatenate(self.data)) if self.data else None