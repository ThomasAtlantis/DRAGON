import importlib
import glob
import faiss
import pickle
import numpy as np
import copy
import torch

from typing import List, Dict
from transformers import PreTrainedTokenizer, PreTrainedModel

from pathlib import Path
from ..utils.data_process.data_utils import load_passages
from .indexer import Indexer
from ..config import DragonConfig
from ..utils.mlogging import Logger
from ..utils.meter import TimeMeter
from ..utils.cache import Cache
from ..utils.data_process import text_utils


logger = Logger.build(__name__, level="INFO")
time_meter = TimeMeter()


class Retriever():
    """
    Retriever class for retrieving data from the database.
    """

    def __init__(self, config: DragonConfig):
        self.config = config
        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizer

        retriever_module = importlib.import_module(
            f"dragon.retriever.models.{config.retriever}")
        self.model, self.tokenizer = retriever_module.load_retriever(config.re_model_name_or_path)
        self.model = self.model.to(torch.device(config.device))
        self.indexer = Indexer(config.embedding_dim, config.n_subquantizers, config.n_bits)

        embedding_files = sorted(glob.glob(config.passages_embeddings))
        embeddings_dir = Path(embedding_files[0]).parent
        index_file, meta_file = Indexer.disk_file(embeddings_dir)  # TODO: move this line into indexer, and implement ID2DBID
        if config.load_index and Path(index_file).exists():
            self.indexer.load(embeddings_dir)
        else:
            self.index_encoded_data(embedding_files, config.indexing_batch_size)
            if config.dump_index: self.indexer.dump(embeddings_dir)

        num_gpus = faiss.get_num_gpus()
        if config.use_faiss_gpu and num_gpus > 0:
            with time_meter.timer("conversion"):
                logger.info(f"Converting index to GPU index")
                cloner_options = faiss.GpuMultipleClonerOptions()
                cloner_options.shard = True
                cloner_options.useFloat16 = True
                self.indexer.index = faiss.index_cpu_to_all_gpus(
                    self.indexer.index, co=cloner_options, ngpu=num_gpus)
            logger.debug(f"Conversion time: {time_meter.timer('conversion').duration:.6f} s.")

        self.query2docs = Cache(
            "query2docs", config.cache.load_query2docs, 
            config.cache.dump_query2docs, config.cache.directory)
        passages = load_passages(config.passages, config.chunk_size, config.cache.directory)
        self.passage_id_map: Dict[str, Dict] = {x['id']: x for x in passages}

    def embed_queries(self, queries: List[str]):
        def query_loader(queries, batch_size):
            for i in range(0, len(queries), batch_size):
                query_batch = queries[i:i + batch_size]
                if self.config.normalize_text:
                    for j in range(len(query_batch)):
                        query_batch[j] = text_utils.normalize(query_batch[j])
                yield [], query_batch
        _, embeddings = text_utils.embed_texts(
            self.model, self.tokenizer, queries, 
            batch_size=self.config.per_gpu_batch_size, 
            text_size=self.config.question_maxlength, 
            text_loader=query_loader)
        return embeddings
    
    def process_docs(self, doc, doc_id: int):
        logger.debug("Before:", doc["text"])
        if self.config.remove_broken_sents:
            doc["text"] = text_utils.remove_broken_sentences(doc["text"])
        # TODO: The sentence rounding approach assumes passages are in consecutive order
        elif self.config.round_broken_sents and doc_id > 0:
            pre_doc = self.passage_id_map[doc_id - 1]
            if text_utils.ends_mid_sentence(pre_doc["text"]):
                first_half = text_utils.get_incomplete_sentence(pre_doc["text"], from_end=True)
                doc["text"] = first_half + " " + doc["text"].lstrip()
            if text_utils.ends_mid_sentence(doc["text"]):
                if doc_id < len(self.passage_id_map) - 1:
                    next_doc = self.passage_id_map[doc_id + 1]
                    second_half = text_utils.get_incomplete_sentence(next_doc["text"], from_end=False)
                    doc["text"] = doc["text"].rstrip() + " " + second_half
        logger.debug("After:", doc["text"])
        return doc
    
    def retrieve_passages(self, queries: List[str]):
        if len(queries) == 1 and queries[0] in self.query2docs:
            return [self.query2docs.get(queries[0])]
        else:
            query_embeddings = self.embed_queries(queries)
            
            with time_meter.timer("retrieval"):
                doc_ids, scores = self.indexer.search_knn(query_embeddings, self.config.n_docs)
            assert(len(doc_ids) == len(queries) and len(scores) == len(queries))
            logger.debug(f"Retrieval finished in {time_meter.timer('retrieval').duration * 1e3:.1f} ms.")
            
            docs_scores = []
            for query, query_doc_ids, query_scores in zip(queries, doc_ids, scores):
                docs = []
                for doc_id in query_doc_ids:
                    doc = copy.deepcopy(self.passage_id_map[doc_id])
                    doc = self.process_docs(doc, doc_id)
                    docs.append(doc)
                docs_scores.append((docs, query_scores))
                self.query2docs.set(query, (docs, query_scores))
            return docs_scores
    

    def add_embeddings(self, embeddings: np.ndarray, ids: List[int], batch_size: int):
        end_idx = min(batch_size, embeddings.shape[0])
        self.indexer.index_data(ids[:end_idx], embeddings[:end_idx])
        return embeddings[end_idx:], ids[end_idx:]
    
    def add_embeddings_remaining(self, embeddings: np.ndarray, ids: List[int], threshold: int, batch_size: int):
        # Iteratively add embeddings until the remaining embeddings equal or less than threshold
        while embeddings.shape[0] > threshold:
            embeddings, ids = self.add_embeddings(embeddings, ids, batch_size)
        return embeddings, ids
    

    def index_encoded_data(self, embedding_files, indexing_batch_size):
        # TODO: Use deque to improve this logic
        logger.info(f'Indexing passages from files {embedding_files} ...')
        with time_meter.timer("indexing"):
            allids, allembeddings = [], np.array([])
            for file in embedding_files:
                logger.info(f'Loading file `{file}` ...')
                with open(file, 'rb') as ifstream:
                    ids, embeddings = pickle.load(ifstream)

                allembeddings = np.vstack(
                    (allembeddings, embeddings)) if allembeddings.size else embeddings
                allids.extend(ids)
                
                allembeddings, allids = self.add_embeddings_remaining(
                    allembeddings, allids, threshold=indexing_batch_size, batch_size=indexing_batch_size)
    
            allembeddings, allids = self.add_embeddings_remaining(
                allembeddings, allids, threshold=0, batch_size=indexing_batch_size)

        logger.debug(f"Indexing finished after {time_meter.timer('indexing').duration:.1f} s.")
