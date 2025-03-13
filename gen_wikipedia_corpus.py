from transformers import RagRetriever


def run(argv):
    passages_path = "./data/wiki_dpr/nq"
    index_path = "./data/wiki_dpr/nq_compressed.faiss"
    retriever = RagRetriever.from_pretrained(
        "weights/rag-token-nq", 
        index_name="custom", use_dummy_dataset=False,
        passages_path=passages_path, index_path=index_path)
    dataset = retriever.index.dataset
    dataset = dataset.remove_columns(['embeddings', 'text'])
    dataset.save_to_disk("data/wiki_dpr_title")
    # dataset columns: ['id', 'title']


from dragon.utils.profiling import TimeMeter
from datasets import Dataset
from elasticsearch import Elasticsearch
from dragon.config import Config
from elasticsearch.helpers import bulk
from tqdm import tqdm
import os


index_name = "wiki_dpr_title"
batch_size = 10000

def run(argv):
    with TimeMeter("Creating Index"):
        es_client = Elasticsearch(
        Config.elastic_search.host,
        ca_certs=os.environ["ES_CA_CERTS"],
        basic_auth=("elastic", os.environ['ES_PASSWORD']))

        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)

        es_client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "title": {"type": "text"}
                }
            },
        )

    with TimeMeter("Loading Dataset"):
        dataset = Dataset.load_from_disk('data/wiki_dpr_title')

    with TimeMeter("Update indexes"):
        for batch in tqdm(dataset.iter(batch_size=batch_size), total=len(dataset) // batch_size):
            actions = [{
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": _id,
                    "title": title
            } for _id, title in zip(batch['id'], batch['title'])]
            bulk(es_client, actions)
            es_client.indices.refresh(index=index_name)
            es_client.indices.flush(index=index_name)
    print("Total:", es_client.count(index=index_name)['count'])
    es_client.close()