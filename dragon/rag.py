from tqdm import tqdm
import numpy as np
import torch

from .config import DragonConfig
from .generator.generator import Generator
from .retriever.retriever import Retriever
from .utils.data_process.data_utils import ContextualLMSampleLoader
from .utils.mlogging import Logger


logger = Logger.build(__name__, level="INFO")


class RagSequenceForGeneration:

    def __init__(self, config: DragonConfig):
        self.config = config
        self.generator = Generator(config)
        self.retriever = Retriever(config)
        self.retriever.prepare_retrieval()
        self.device = torch.device(config.device)

    def generate(self, inputs):
        raise NotImplementedError

    
    def get_perplexity_data(self, text):
        all_logprobs = []

        input_token_list = self.generator.tokenizer.encode_plus(text)["input_ids"]
        loader, total = ContextualLMSampleLoader(
            token_list=input_token_list,
            bos_token=self.generator.context_switching_id,
            max_seq_len=self.generator.max_seq_len,
            prefix_len=self.config.evaluator.s_prefix)
        for query_tokens, input_tokens, label_tokens in tqdm(loader(), total=total):
            if query_tokens != [] and self.config.retriever.n_docs > 0:
                query = self.generator.tokenizer.decode(query_tokens)
                docs, scores = self.retriever.retrieve_passages([query])[0]
                plain_docs = [doc["text"] for doc in docs]
                if self.config.retriever.s_aggregate == 0:
                    doc_str = "\n".join(plain_docs)
                    logger.debug(f"Query: {[query]}\nRetrieved Documents: {[doc_str]}")
                    doc_encodings = self.generator.tokenizer.encode(doc_str)[:self.config.retriever.s_context]
                    input_tokens = torch.concat((
                        torch.LongTensor(doc_encodings), 
                        torch.LongTensor(query_tokens),
                        torch.LongTensor(input_tokens)
                    ), dim=-1)
                    logprobs = self.get_token_logprobs(input_tokens=input_tokens, label_tokens=label_tokens)
                else:
                    logprobs_list = []
                    logprobs = None
                    assert self.config.retriever.s_aggregate <= len(plain_docs), "Not enough documents for aggregation"
                    for i in range(self.config.retriever.s_aggregate):
                        doc_str = plain_docs[i]
                        doc_encodings = self.generator.tokenizer.encode(doc_str)[:self.config.retriever.s_context]
                        input_tokens_tmp = torch.concat((
                            torch.LongTensor(doc_encodings), 
                            torch.LongTensor(query_tokens), 
                            torch.LongTensor(input_tokens)
                        ), dim=-1)
                        logprobs = self.get_token_logprobs(input_tokens=input_tokens_tmp, label_tokens=label_tokens)
                        logprobs_list.append(logprobs.tolist())
                    logprobs_list = torch.as_tensor(logprobs_list, dtype=torch.float32)
                    scores = np.array(scores)[: self.config.retriever.s_aggregate]
                    scores = np.exp(scores) / np.sum(np.exp(scores), axis=0)
                    log_scores = torch.log(torch.FloatTensor(scores)).reshape(-1, 1)
                    log_scores = log_scores.repeat(1, len(logprobs_list[0]))
                    logprobs = torch.logsumexp(logprobs_list + log_scores, dim=0).numpy()
            else:
                logprobs = self.get_token_logprobs(input_tokens=input_tokens, label_tokens=label_tokens)
            all_logprobs.append(logprobs)
        
        all_logprobs = np.concatenate(all_logprobs)
        assert len(all_logprobs) == len(input_token_list), f"Length mismatch: {len(all_logprobs)} vs {len(input_token_list)}"
        return all_logprobs
    
    def get_token_logprobs(self, input_tokens, label_tokens) -> np.ndarray:
        input_tokens = torch.LongTensor(input_tokens).to(self.device).unsqueeze(dim=0)
        label_tokens = torch.LongTensor(label_tokens).to(self.device).unsqueeze(dim=0)
        with torch.inference_mode():
            output = self.generator.model(input_tokens, return_dict=True)
        label_tokens = label_tokens.squeeze()
        pred_tokens = output.logits.squeeze()
        pred_tokens = pred_tokens[-len(label_tokens):]
        neg_logprobs = -torch.nn.functional.cross_entropy(
            pred_tokens, label_tokens, reduction="none").detach().cpu().numpy()

        return neg_logprobs
