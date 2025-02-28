import importlib
import glob
import copy
import torch

from typing import List, Dict, Protocol
from transformers import PreTrainedTokenizer, PreTrainedModel

from ..utils.data_process.data_utils import load_passages
from .indexer import Indexer
from ..config import DragonConfig
from ..utils.mlogging import Logger
from ..utils.meter import TimeMeter
from ..utils.cache import Cache
from ..utils.data_process import text_utils


logger = Logger.build(__name__, level="INFO")
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

    def __init__(self, config: DragonConfig):
        self.config = config
        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizer

        # Model and Tokenizer initialization
        retriever_module = importlib.import_module(
            f"dragon.retriever.models.{config.retriever.model}")
        self.model, self.tokenizer = retriever_module.load_retriever()
        self.model = self.model.to(torch.device(config.device))
        if config.fp16: self.model = self.model.half()

    def prepare_retrieval(self):
        config = self.config
        
        # Indexer initialization
        self.indexer = Indexer(config.indexer.s_embedding, config.indexer.n_subquantizers, config.indexer.n_bits)
        embedding_files = sorted(glob.glob(config.retriever.passages_embeddings))
        self.indexer.index_embeddings(embedding_files, config.indexer.bs_indexing, config.cache.load_index, config.cache.dump_index)
        
        # Inverted index for passages
        passages = load_passages(config.retriever.passages, config.retriever.s_passage_chunk, config.cache.directory)
        self.id2passage: Dict[str, Dict] = {x['id']: x for x in passages}

        # Retrieval cache
        self.query2docs = Cache(
            "query2docs", config.cache.load_query2docs, 
            config.cache.dump_query2docs, config.cache.directory)
        
    def embed_texts(self, texts: List[str], batch_size: int, text_size: int, post_processor: TextProcessor = None):
        embeddings = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch_text = texts[i:i + batch_size]
                if post_processor is not None: 
                    batch_text = post_processor(batch_text)
                encoded_batch = self.tokenizer.batch_encode_plus(
                    batch_text, return_tensors="pt", max_length=text_size, padding=True, truncation=True)
                embeddings.append(self.model(**{k: v.cuda() for k, v in encoded_batch.items()}))
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        return embeddings
    
    def process_docs(self, doc, doc_id: int):
        logger.debug("Before:", doc["text"])
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
        logger.debug("After:", doc["text"])
        return doc
    
    def retrieve_passages(self, queries: List[str]):
        if len(queries) == 1 and queries[0] in self.query2docs:
            return [self.query2docs.get(queries[0])]
        else:
            def query_processor(query_batch):
                if self.config.text.normalize:
                    for j in range(len(query_batch)):
                        query_batch[j] = text_utils.normalize(query_batch[j])
                return query_batch

            query_embeddings = self.embed_texts(
                queries, batch_size=self.config.retriever.bs_encode,
                text_size=self.config.retriever.s_query, 
                post_processor=query_processor)
            
            with time_meter.timer("retrieval"):
                doc_ids, scores = self.indexer.search_knn(query_embeddings, self.config.retriever.n_docs)
            assert(len(doc_ids) == len(queries) and len(scores) == len(queries))
            logger.debug(f"Retrieval finished in {time_meter.timer('retrieval').duration * 1e3:.1f} ms.")
            
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
