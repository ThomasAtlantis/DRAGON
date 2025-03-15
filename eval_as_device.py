import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
from dragon.utils.stable import seed_everything
from dragon.config import DragonConfig
from dragon.dragon import Dragon
from dragon import generator
from dragon import transceiver
from dragon import decoder
from dragon import dragon
from dragon import aggregator
generator.logging_level = "DEBUG"
transceiver.logging_level = "DEBUG"
decoder.logging_level = "DEBUG"
dragon.logging_level = "DEBUG"
aggregator.logging_level = "DEBUG"
seed_everything(42)

if __name__ == "__main__":

    config = DragonConfig()
    
    config.retriever.passages = "wikipedia[remote]"
    config.generator.model = "facebook/opt-1.3b"
    config.generator.s_sequence = 896
    config.retriever.s_context = 128
    config.retriever.n_docs = 4
    config.retriever.s_aggregate = 4
    config.sampler.do_sample = False

    config.trans.rank = 1
    config.trans.tx_port = 6000
    config.trans.rx_port = 5000
    device = Dragon(config)
    while not device.ready_for_generation:
        time.sleep(0.1)
    queries = [
        "who came up with the theory of relativity",
        "in greek mythology who was the goddess of spring growth",
    ]
    max_new_tokens = 10
    template = "context: {context} given the context, answer the question: {query}? " 
    for query in queries:
        response = device.query(query, template, max_new_tokens)
        print(response)
    device.shutdown()
