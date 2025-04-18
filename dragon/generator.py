from dataclasses import dataclass
from queue import Queue
import threading
from typing import List
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from .config import DragonConfig
from .utils.mlogging import Logger
from .utils.stable import seed_everything
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch

logging_level = "INFO"

def _transform_greedy(probs: torch.Tensor) -> torch.Tensor:
    if probs.dim() == 1:
        probs = probs.unsqueeze_(0)
    index = torch.argmax(probs, dim=-1).unsqueeze_(-1)
    src = torch.ones_like(index, dtype=probs.dtype)
    probs.zero_().scatter_(-1, index, src)
    if probs.shape[0] == 1:
        probs = probs.squeeze_(0)
    return probs

def _transform_top_k(probs: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    values, indices = torch.topk(probs, top_k, dim=-1)
    probs = torch.zeros_like(probs).scatter_(-1, indices, values)
    return probs

def _transform_top_p(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_sorted_indices_mask = cumulative_probs > top_p
    remove_sorted_indices_mask[:, 0] = False  # at least one token should be kept
    i_indices, j_indices = torch.where(remove_sorted_indices_mask)
    vocab_indices = sorted_indices[i_indices, j_indices]
    probs[i_indices, vocab_indices] = 0
    return probs

class Sampler:
    def __init__(self, config: DragonConfig.sampler):
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temperature = config.temperature
        self.greedy = not config.do_sample

    def transform(self, probs: torch.Tensor) -> torch.Tensor:
        if self.greedy:
            return _transform_greedy(probs)
        if self.top_k > 0:
            probs = _transform_top_k(probs, torch.as_tensor(self.top_k, device=probs.device))
        if self.top_p < 1.0:
            probs = _transform_top_p(probs, torch.as_tensor(self.top_p, device=probs.device))
        return probs
    
    def __call__(self, probs: torch.Tensor) -> np.ndarray:
        """
        @param probs: (s_sequence, s_vocab) tensor
        """
        probs = self.transform(probs)
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token_id.cpu().tolist()

@dataclass
class CausalOutput:
    next_token: int
    logprobs: torch.Tensor
    weight: float
    past_key_values: List[torch.Tensor] = None

class Generator:

    def __init__(self, config: DragonConfig):
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.model_name = config.generator.model
        self.device = torch.device(config.device)
        self.sampler = Sampler(config.sampler)
        
        dtype = torch.bfloat16 if config.generator.use_fp16 else torch.float32
        if self.model_name == "BART":
            from transformers import RagTokenForGeneration, RagTokenizer
            self.model = RagTokenForGeneration.from_pretrained(
                "/data/lsy/workspace/DragonLab/weights/rag-token-nq",
                device_map="cpu", torch_dtype=torch.bfloat16
            ).generator.eval()
            self.tokenizer = RagTokenizer.from_pretrained(
                "/data/lsy/workspace/DragonLab/weights/rag-token-nq").question_encoder
            # self.tokenizer.encode_plus = self.tokenizer.__call__
            # self.tokenizer.batch_encode_plus = self.tokenizer.__call__
            # import types
            # self.tokenizer.encode = types.MethodType(
            #     lambda s, text: s(text).input_ids, self.tokenizer)
        else:
            if "Qwen" in self.model_name and self.device.type == "cuda":
                self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="cpu", torch_dtype=dtype, attn_implementation='flash_attention_2'
                ).eval()
            else:
                self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="cpu", torch_dtype=dtype
                ).eval()
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                self.model_name, add_bos_token=False, add_eos_token=False)  # removing bos/eos tokens is crucial
        self.tokenizer.padding_side = "left"
        self.model.to(self.device)
        from .queues import DraftItem
        DraftItem._vocab_size = self.model.config.vocab_size
        # The token <|endoftext|> serves as a content separator between distinct 'texts' 
        # within the training data for GPT-2 (and likely GPT-3 as well). By using this token, 
        # we enforce a shift in context both before and after <|endoftext|>.
        # self.context_switching_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]
        self.context_switching_id = None
        self.max_seq_len = min(config.generator.s_sequence, self.model.config.max_position_embeddings)
    
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> CausalLMOutputWithPast:  # prefilling
        seed_everything(42)
        # if not self.model_name in ["BART", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        #     kwargs.update(
        #         pad_token_id=self.model.config.eos_token_id
        #     )
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                return_dict=True, 
                use_cache=True, 
                **kwargs
            )
        return output
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> CausalLMOutputWithPast:  # generation
        # if not self.model_name in ["BART", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        #     kwargs.update(
        #         pad_token_id=self.model.config.eos_token_id
        #     )
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict_in_generate=True, 
                use_cache=True, 
                **kwargs
            )
        return output    

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
                # self.logger.warning(e)
                self.preempt_event.clear()
                self.input_queue.queue.clear()
                self.output_queue.put(None)
            except RuntimeError as e:
                self.logger.error(e)