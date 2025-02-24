from tqdm import tqdm
from typing import Optional

from .config import DragonConfig
from .generator.generator import Generator
from .retriever.retriever import Retriever
from .utils.data_process.data_utils import ContextualLMSampleLoader
from .utils.mlogging import Logger

logger = Logger.build(__name__, level="INFO")

class Decoder:

    def __init__(self, generator: Generator, retriever: Retriever, config: DragonConfig):
        self.generator = generator
        self.retriever = retriever
        self.config = config
    
    def get_perplexity_data(self, text) -> Optional[dict]:

        all_logprobs = []
        all_positions = []

        input_ids = self.generator.tokenizer.encode_plus(text)["input_ids"]
        loader, total = ContextualLMSampleLoader(
            token_list=input_ids,
            bos_token=self.generator.context_switching_id,
            max_seq_len=self.generator.max_seq_len,
            context_len=self.generator.context_len)
        for prompt, inputs, labels in tqdm(loader(), total=total):
            print(prompt, inputs, labels)
            break
        #     if prompt != []:
        #         prompt = self.generator.tokenizer.decode(prompt)  # TODO: Encoding-Decoding inconsistency may affect performance
        #         docs, scores = self.retriever.retrieve_passages([prompt])[0]
        #         plain_docs = [doc["text"] for doc in docs]
        #         if self.config.ensemble == 0:
        #             doc_str = "\n".join(plain_docs)
        #             print(f"query: {[prompt]}\nretrieved doc: {[doc_str]}")
        #             doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
        #             input_tokens = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
        #             print("retrieve + context: ", len(input_tokens)-len(labels))
        #         else:
        #             '''
        #             a + b + c = log(e^log(a) + e^log(b) + e^log(c))
        #             '''
        #             logprobs_list = []
        #             block_output = None
        #             assert self.args.ensemble <= len(plain_docs)
                    
        #             for i in range(self.args.ensemble):
        #                 doc_str = plain_docs[i]
        #                 doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
        #                 input_tokens_tmp = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
        #                 block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=labels,)
        #                 logprobs_list.append(block_output["logprobs"])
        #                 # sum(np.isinf(block_output["logprobs"]))
        #             # block_output["logprobs"] = np.log(np.mean(np.exp(logprobs_list), axis=0))
        #             # len(logprobs_list) = number of ensemble
        #             # block_output["logprobs"] = torch.logsumexp(torch.FloatTensor(logprobs_list), dim=0) - np.log(len(logprobs_list))
        #             # apply softmax to scores 
        #             scores = np.exp(scores) / np.sum(np.exp(scores), axis=0)
        #             scores = torch.log(torch.FloatTensor(scores)).reshape(-1, 1)
        #             scores = scores.repeat(1, len(logprobs_list[0]))
        #             block_output["logprobs"] = torch.logsumexp(torch.FloatTensor(logprobs_list)+scores, dim=0) 
        #             block_output["logprobs"] = block_output["logprobs"].numpy()
        #     else:                # bp()
        #         block_output = self.get_token_logprobs(input_tokens=input_tokens, pred_tokens=labels,)
        #     all_logprobs.append(block_output["logprobs"])
        #     all_positions.append(block_output["positions"])
        # if not all_logprobs:
        #     return None

        # # Gather
        # all_logprobs = np.concatenate(all_logprobs)
        # all_positions = np.concatenate(all_positions)
        # assert len(all_logprobs) == len(input_ids)
        # return {
        #     "logprobs": all_logprobs,
        #     "positions": all_positions,
        #     "length": len(all_logprobs),
        #     "utf8_length": len(text.encode('utf-8')),
        # }
