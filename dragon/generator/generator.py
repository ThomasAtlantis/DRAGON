from transformers import AutoModelForCausalLM, AutoTokenizer
from ..retriever.retriever import Retriever
import torch


class Generator:

    def __init__(self, model_name, max_seq_len=1024, context_len=512):
        self.model_name = model_name
        self.context_len = context_len
        self.max_seq_len = max_seq_len

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float16).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # The token <|endoftext|> serves as a content separator between distinct 'texts' 
        # within the training data for GPT-2 (and likely GPT-3 as well). By using this token, 
        # we enforce a shift in context both before and after <|endoftext|>.
        self.context_switching_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]
