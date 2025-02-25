import pickle
from typing import List, Tuple
import faiss
import ipdb
import numpy as np
from pathlib import Path
from ..utils.mlogging import Logger


logger = Logger.build(__name__, level="INFO")

class Indexer(object):

    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8):
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        self.indexId2DBId = []

    def index_data(self, ids: List[int], embeddings: np.ndarray):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        logger.info(f'Indexed {len(self.indexId2DBId)} embeddings.')

    def search_knn(
            self, query_embeddings: np.ndarray, 
            top_docs: int, batch_size: int = 2048
        ) -> Tuple[List[List[int]], List[List[float]]]:
        query_embeddings = query_embeddings.astype('float32')
        db_ids, scores = [], []
        for k in range(0, len(query_embeddings), batch_size):
            query_embeddings_batch = query_embeddings[k:k + batch_size]
            # ipdb.set_trace()
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
        logger.info(f'Loaded index of type {type(self.index)} and size {self.index.ntotal}')

        with open(meta_file, "rb") as reader: self.indexId2DBId = pickle.load(reader)
        logger.info(f'Loaded meta data of size {len(self.indexId2DBId)}')
        assert len(self.indexId2DBId) == self.index.ntotal, 'Loaded indexId2DBId and faiss index size do not match'

    def _update_id_mapping(self, db_ids: List):
        self.indexId2DBId.extend(db_ids)