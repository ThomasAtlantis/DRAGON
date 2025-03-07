from functools import partial
import importlib
import glob
import copy
from pathlib import Path
import torch

from typing import List, Dict, Protocol
from tqdm import trange
from transformers import PreTrainedTokenizer, PreTrainedModel

from ..utils.data_process.data_utils import load_passages
from .indexer import Indexer
from ..config import DragonConfig
from ..utils.mlogging import Logger
from ..utils.meter import TimeMeter
from ..utils.cache import Cache
from ..utils.data_process import text_utils

time_meter = TimeMeter()

# Or, if you want to support Python 3.7 and below, install the typing_extensions
# module via pip and do the below:
from typing_extensions import Protocol

class TextProcessor(Protocol):
    def __call__(self) -> List[str]: ...


class Retriever():
    """
    Retriever class for retrieving data from the database.
    """

    def __init__(self, config: DragonConfig, logger: Logger):
        self.logger = logger
        self.config = config
        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizer
        self.n_docs = config.retriever.n_docs
        self.bs_encode = config.retriever.bs_encode
        assert self.n_docs >= config.retriever.s_aggregate, "Not enough documents for aggregation."

        # Model and Tokenizer initialization
        retriever_module = importlib.import_module(
            f"dragon.retriever.models.{config.retriever.model}")
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
