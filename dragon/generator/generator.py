from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..config import DragonConfig
import torch


class Generator:

    def __init__(self, config: DragonConfig):
        model_name = config.model_config_path
        self.context_len = config.context_len
        self.max_seq_len = self.context_len + config.pred_len

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float16).eval()
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(torch.device(config.device))
        # The token <|endoftext|> serves as a content separator between distinct 'texts' 
        # within the training data for GPT-2 (and likely GPT-3 as well). By using this token, 
        # we enforce a shift in context both before and after <|endoftext|>.
        self.context_switching_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]
