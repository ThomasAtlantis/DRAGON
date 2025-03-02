import pickle
from typing import List, Tuple
import faiss
import numpy as np
from pathlib import Path

from ..utils.meter import TimeMeter
from ..utils.mlogging import Logger


logger = Logger.build(__name__, level="INFO")
time_meter = TimeMeter()

class Indexer(object):

    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8):
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        self.indexId2DBId = []

    def search_knn(
            self, query_embeddings: np.ndarray, 
            top_docs: int, batch_size: int = 2048
        ) -> Tuple[List[List[int]], List[List[float]]]:
        query_embeddings = query_embeddings.astype('float32')
        db_ids, scores = [], []
        for k in range(0, len(query_embeddings), batch_size):
            query_embeddings_batch = query_embeddings[k:k + batch_size]
            scores_batch, indexes = self.index.search(query_embeddings_batch, top_docs)
            db_ids_batch = [[self.indexId2DBId[i] for i in ids_each_query] for ids_each_query in indexes]
            db_ids.extend(db_ids_batch)
            scores.extend(scores_batch)
        return db_ids, scores
    
    @staticmethod
    def disk_file(directory):
        directory = Path(directory)
        return directory / 'index.faiss', directory / 'index_meta.faiss'

    def dump(self, directory):
        index_file, meta_file = self.disk_file(directory)
        logger.info(f'Dumping index into `{index_file}`, meta data into `{meta_file}` ...')

        faiss.write_index(self.index, str(index_file))
        logger.info(f'Dumped index of type {type(self.index)} and size {self.index.ntotal}')

        with open(meta_file, mode='wb') as f: pickle.dump(self.indexId2DBId, f)
        logger.info(f'Dumped meta data of size {len(self.indexId2DBId)}')

    def load(self, directory):
        index_file, meta_file = self.disk_file(directory)
        logger.info(f'Loading index from `{index_file}`, meta data from `{meta_file}` ...')

        self.index = faiss.read_index(str(index_file))
        logger.info(f'Loaded index of type {self.index.__class__.__name__} and size {self.index.ntotal}')

        with open(meta_file, "rb") as reader: self.indexId2DBId = pickle.load(reader)
        logger.info(f'Loaded meta data of size {len(self.indexId2DBId)}')
        assert len(self.indexId2DBId) == self.index.ntotal, 'Loaded indexId2DBId and faiss index size do not match'

    def _update_id_mapping(self, db_ids: List):
        self.indexId2DBId.extend(db_ids)

    def _index_data(self, ids: List[int], embeddings: np.ndarray):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        logger.info(f'Indexed {len(self.indexId2DBId)} embeddings.')

    def _add_embeddings(self, embeddings: np.ndarray, ids: List[int], batch_size: int):
        end_idx = min(batch_size, embeddings.shape[0])
        logger.info(f'Indexing {end_idx}/{embeddings.shape[0]} embeddings ...')
        self._index_data(ids[:end_idx], embeddings[:end_idx])
        return embeddings[end_idx:], ids[end_idx:]
    
    def _add_embeddings_remaining(self, embeddings: np.ndarray, ids: List[int], threshold: int, batch_size: int):
        # Iteratively add embeddings until the remaining embeddings equal or less than threshold
        while embeddings.shape[0] > threshold:
            embeddings, ids = self._add_embeddings(embeddings, ids, batch_size)
        return embeddings, ids
    

    def index_embeddings(self, embedding_files, indexing_batch_size, load_index=False, dump_index=False):
        embeddings_dir = Path(embedding_files[0]).parent
        index_file, _ = Indexer.disk_file(embeddings_dir)
        if load_index and Path(index_file).exists():
            self.load(embeddings_dir)
        else:
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
                    
                    allembeddings, allids = self._add_embeddings_remaining(
                        allembeddings, allids, threshold=indexing_batch_size, batch_size=indexing_batch_size)
        
                allembeddings, allids = self._add_embeddings_remaining(
                    allembeddings, allids, threshold=0, batch_size=indexing_batch_size)

            logger.debug(f"Indexing finished after {time_meter.timer('indexing').duration:.1f} s.")
            if dump_index: self.dump(embeddings_dir)