import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..config import DragonConfig
from ..utils.mlogging import Logger
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch

logger = Logger.build(__name__, level="INFO")

class Sampler:
    def __init__(self, config: DragonConfig.sampler):
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temperature = config.temperature
        self.greedy = not config.do_sample

    def __call__(self, probs: torch.Tensor) -> np.ndarray:
        """
        @param probs: (s_sequence, s_vocab) tensor
        """
        if self.greedy:
            return torch.argmax(probs, dim=-1).cpu().tolist()
        if self.top_k > 0:
            values, indices = torch.topk(probs, self.top_k, dim=-1)
            probs = torch.zeros_like(probs).scatter_(-1, indices, values)

        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_sorted_indices_mask = cumulative_probs > self.top_p
            remove_sorted_indices_mask[:, 0] = False  # at least one token should be kept
            i_indices, j_indices = torch.where(remove_sorted_indices_mask)
            vocab_indices = sorted_indices[i_indices, j_indices]
            probs[i_indices, vocab_indices] = 0
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token_id.cpu().tolist()

class Generator:

    def __init__(self, config: DragonConfig):
        model_name = config.generator.model
        self.device = torch.device(config.device)
        self.sampler = Sampler(config.sampler)
        
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float16, 
            # attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, add_bos_token=False, add_eos_token=False)  # removing bos/eos tokens is crucial
        self.tokenizer.padding_side = "left"
        self.model.to(self.device)
        # The token <|endoftext|> serves as a content separator between distinct 'texts' 
        # within the training data for GPT-2 (and likely GPT-3 as well). By using this token, 
        # we enforce a shift in context both before and after <|endoftext|>.
        self.context_switching_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]
        self.max_seq_len = min(config.generator.s_sequence, self.model.config.max_position_embeddings)
    
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> CausalLMOutputWithPast:  # prefilling
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pad_token_id=self.model.config.eos_token_id,
                return_dict=True, 
                use_cache=True, 
                **kwargs
            )
        return output
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> CausalLMOutputWithPast:  # generation
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                pad_token_id=self.model.config.eos_token_id, 
                return_dict_in_generate=True, 
                use_cache=True, 
                **kwargs
            )
        return output    
