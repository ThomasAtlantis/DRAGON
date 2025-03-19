from abc import ABC, abstractmethod
from functools import partial
import importlib
import glob
import copy
from pathlib import Path
import socket
import struct
import torch

import json
from typing import List, Dict, Protocol, Tuple
from tqdm import trange
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import RagRetriever
from transformers import DPRQuestionEncoder
from .utils.data_process.data_utils import load_passages
from .indexer import Indexer
from .config import DragonConfig
from .utils.mlogging import Logger
from .utils.meter import TimeMeter
from .utils.cache import Cache
from .utils.data_process import text_utils

time_meter = TimeMeter()
logging_level = "INFO"

# Or, if you want to support Python 3.7 and below, install the typing_extensions
# module via pip and do the below:
from typing_extensions import Protocol

class TextProcessor(Protocol):
    def __call__(self) -> List[str]: ...


class BaseRetriever(ABC):

    def __init__(self, config: DragonConfig):
        self.logger = Logger.build(__class__.__name__, logging_level)
        self.config = config

    @abstractmethod
    def prepare_retrieval(self, config: DragonConfig):
        raise NotImplementedError

    @abstractmethod
    def retrieve_passages(self, queries: List[str]) -> List[Tuple[List[Dict], List[float]]]:
        raise NotImplementedError


class CustomRetriever(BaseRetriever):
    """
    Retriever class for retrieving data from the database.
    """

    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizer
        self.n_docs = config.retriever.n_docs
        self.bs_encode = config.retriever.bs_encode
        assert self.n_docs >= config.retriever.s_aggregate, "Not enough documents for aggregation."

        # Model and Tokenizer initialization
        retriever_module = importlib.import_module(
            f"dragon.models.{config.retriever.model}")
        self.model, self.tokenizer = retriever_module.load_retriever()
        self.model = self.model.to(torch.device(config.device))
        if config.fp16: self.model = self.model.half()

    def prepare_retrieval(self, config: DragonConfig):
        # Indexer initialization
        self.indexer = Indexer(config.indexer.s_embedding, config.indexer.n_subquantizers, config.indexer.n_bits)
        embedding_files = sorted(glob.glob(config.retriever.passages_embeddings))
        self.indexer.index_embeddings(embedding_files, config.indexer.bs_indexing, config.cache.load_index, config.cache.dump_index)
        
        # Inverted index for passages
        passages = load_passages(config.retriever.passages, config.retriever.s_passage, config.cache.directory)
        self.id2passage: Dict[str, Dict] = {x['id']: x for x in passages}

        # Retrieval cache
        _, dataset_id = config.retriever.passages.split(",")
        self.query2docs = Cache(
            "query2docs", config.cache.load_query2docs, 
            config.cache.dump_query2docs, Path(config.cache.directory) / dataset_id)
        
    def embed_texts(self, texts: List[str], batch_size: int, post_processor: TextProcessor = None, progress=False):
        embeddings = []
        with torch.inference_mode():
            range_function = partial(trange, desc="Encoding texts", leave=False) if progress else range
            for i in range_function(0, len(texts), batch_size):
                batch_text = texts[i:i + batch_size]
                if post_processor is not None: 
                    batch_text = post_processor(batch_text)
                encoded_batch = self.tokenizer.batch_encode_plus(
                    batch_text, return_tensors="pt", padding='longest')
                embeddings.append(self.model(**{k: v.cuda() for k, v in encoded_batch.items()}))
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        return embeddings
    
    def process_docs(self, doc, doc_id: int):
        self.logger.debug("Before:", doc["text"])
        if self.config.text.remove_broken_sents:
            doc["text"] = text_utils.remove_broken_sentences(doc["text"])
        # TODO: The sentence rounding approach assumes passages are in consecutive order
        elif self.config.text.round_broken_sents and doc_id > 0:
            pre_doc = self.id2passage[doc_id - 1]
            if text_utils.ends_mid_sentence(pre_doc["text"]):
                first_half = text_utils.get_incomplete_sentence(pre_doc["text"], from_end=True)
                doc["text"] = first_half + " " + doc["text"].lstrip()
            if text_utils.ends_mid_sentence(doc["text"]):
                if doc_id < len(self.id2passage) - 1:
                    next_doc = self.id2passage[doc_id + 1]
                    second_half = text_utils.get_incomplete_sentence(next_doc["text"], from_end=False)
                    doc["text"] = doc["text"].rstrip() + " " + second_half
        self.logger.debug("After:", doc["text"])
        return doc
    
    def retrieve_passages(self, queries: List[str]):
        if len(queries) == 1 and queries[0] in self.query2docs:
            docs, scores = self.query2docs.get(queries[0])
            docs, scores = docs[:self.n_docs], scores[:self.n_docs]
            return [(docs, scores)]
        else:
            def query_processor(query_batch):
                if self.config.text.normalize:
                    for j in range(len(query_batch)):
                        query_batch[j] = text_utils.normalize(query_batch[j])
                return query_batch

            query_embeddings = self.embed_texts(
                queries, batch_size=self.bs_encode,
                post_processor=query_processor)
            
            with time_meter.timer("retrieval"):
                doc_ids, scores = self.indexer.search_knn(query_embeddings, self.n_docs)
            assert(len(doc_ids) == len(queries) and len(scores) == len(queries))
            self.logger.debug(f"Retrieval finished in {time_meter.timer('retrieval').duration * 1e3:.1f} ms.")
            
            docs_scores = []
            for query, query_doc_ids, query_scores in zip(queries, doc_ids, scores):
                docs = []
                for doc_id in query_doc_ids:
                    doc = copy.deepcopy(self.id2passage[doc_id])
                    doc = self.process_docs(doc, doc_id)
                    docs.append(doc)
                docs_scores.append((docs, query_scores))
                self.query2docs.set(query, (docs, query_scores))
            return docs_scores


class DPRRetriever(BaseRetriever):

    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.n_docs = config.retriever.n_docs
    
    def prepare_retrieval(self, config: DragonConfig):
        model_path = "/data/lsy/workspace/DragonLab/weights/rag-token-nq"
        index_path = "/data/lsy/workspace/DragonLab/data/wiki_dpr/nq_compressed.faiss"
        query_encoder_path = "models/dpr-nq/query_encoder"
        self.retriever = RagRetriever.from_pretrained(model_path, n_docs=8, index_path=index_path)
        self.retriever.init_retrieval()  # This is crucial
        self.query_encoder = DPRQuestionEncoder.from_pretrained(query_encoder_path)
    
    def retrieve_passages(self, queries: List[str]):
        docs_scores = []
        with torch.inference_mode():
            query_encodings = self.retriever.question_encoder_tokenizer(
                queries, return_tensors="pt", padding='longest')
            query_ids, attention_mask = query_encodings["input_ids"], query_encodings["attention_mask"]
            question_hidden_states = self.query_encoder(query_ids, attention_mask=attention_mask)[0]
            output = self.retriever(query_ids, question_hidden_states.numpy(), n_docs=self.n_docs, return_tensors="pt")
            scores_list = torch.bmm(
                question_hidden_states.unsqueeze(1), output["retrieved_doc_embeds"].float().transpose(1, 2)
            ).squeeze(1).tolist()
            passage_ids_list = output["context_input_ids"].reshape(len(queries), self.n_docs, output["context_input_ids"].shape[-1])
        for passage_ids, scores in zip(passage_ids_list, scores_list):
            passage_text = self.retriever.generator_tokenizer.batch_decode(passage_ids.tolist(), skip_special_tokens=True)
            passage_text = [{"text": text} for text in passage_text]
            docs_scores.append((passage_text, scores))
        return docs_scores
    
class DPRRetrieverClient(BaseRetriever):

    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.n_docs = config.retriever.n_docs
        self.protocol = struct.Struct("I")
        self.header_size = struct.calcsize("I")
        self.address = (config.retriever.host, config.retriever.port)
    
    def prepare_retrieval(self, config: DragonConfig):
        pass

    def _make_message(self, queries, n_docs):
        mbody = json.dumps({
            "queries": queries, 
            "n_docs": n_docs
        }).encode()
        return self.protocol.pack(len(mbody)) + mbody
    
    def retrieve_passages(self, queries):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(self.address)
        client.send(self._make_message(queries, self.n_docs))
        header = b''
        while len(header) < self.header_size:
            chunk = client.recv(self.header_size - len(header))
            if not chunk:
                client.close()
                break
            header += chunk
        body_len = self.protocol.unpack(header)[0]
        mbody = b''
        while len(mbody) < body_len:
            chunk = client.recv(body_len - len(mbody))
            if not chunk:
                client.close()
                break
            mbody += chunk
        docs_scores = json.loads(mbody.decode())
        for ds in docs_scores:
            for doc in ds[0]:  # RagRetriever has concatenated passages with the query
                doc["text"] = doc["text"].split("//")[0]
        return docs_scores

