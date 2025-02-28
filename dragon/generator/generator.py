from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..config import DragonConfig
from typing import List, Tuple
import torch


class Generator:

    def __init__(self, config: DragonConfig):
        model_name = config.generator.model
        self.device = torch.device(config.device)

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float16).eval()
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, add_bos_token=False, add_eos_token=False)  # removing bos/eos tokens is crucial
        self.model.to(self.device)
        # The token <|endoftext|> serves as a content separator between distinct 'texts' 
        # within the training data for GPT-2 (and likely GPT-3 as well). By using this token, 
        # we enforce a shift in context both before and after <|endoftext|>.
        self.context_switching_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]
        self.max_seq_len = min(config.generator.s_sequence, self.model.config.max_position_embeddings)
    
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            output = self.model.generate(
                input_ids, return_dict_in_generate=True, 
                max_new_tokens=1, output_logits=True, **kwargs)
            next_token = output.sequences[0][-1]
            logits = output.logits[0][0]
        return next_token, logits
    
    def __call__(self, input_ids: torch.Tensor, **kwargs):  # prefilling
        with torch.inference_mode():
            output = self.model(input_ids, return_dict=True, **kwargs)
        return output
