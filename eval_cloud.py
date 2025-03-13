import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dragon.utils.stable import seed_everything
from dragon.config import DragonConfig
from dragon.distributed_rag import Dragon

seed_everything(42)

if __name__ == "__main__":

    config = DragonConfig()
    
    config.retriever.passages = "Salesforce/wikitext,wikitext-103-raw-v1"
    config.retriever.passages_embeddings = "data/wikitext103/*.pkl"
    config.generator.model = "facebook/opt-1.3b"
    config.generator.s_sequence = 896
    config.retriever.s_context = 128
    config.retriever.n_docs = 4
    config.retriever.s_aggregate = 4
    config.sampler.do_sample = False
    config.sampler.top_k = 1
    config.cache.load_index = True
    
    config.trans.rank = 0
    config.trans.tx_port = 5000
    config.trans.rx_port = 6000
    cloud = Dragon(config)
