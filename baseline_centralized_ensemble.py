import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dragon.utils.stable import seed_everything
from dragon.config import DragonConfig
from dragon.rag import RagTokenForGeneration
from dragon.utils.mlogging import Logger
    
seed_everything(42)

logger = Logger.build("CentralizedEnsemble", "INFO")
config = DragonConfig()
config.retriever.passages = "Salesforce/wikitext,wikitext-2-raw-v1"
config.retriever.passages_embeddings = "data/wikitext2/*.pkl"
config.generator.model = "facebook/opt-1.3b"
config.generator.s_sequence = 896
config.retriever.s_context = 128
config.retriever.n_docs = 8
config.retriever.s_aggregate = 8
config.sampler.do_sample = False
config.sampler.top_k = 1

rag_tok = RagTokenForGeneration(config)
tokenizer = rag_tok.generator.tokenizer
queries = [
    "who came up with the theory of relativity",
    "in greek mythology who was the goddess of spring growth",
]
max_new_tokens = 10
template = "context: {context} given the context, answer the question: {query}? " 

results = []
for query in queries:
    query_ids = tokenizer.encode(query)    
    output_ids_tok, _ = rag_tok.generate(query_ids, max_new_tokens=max_new_tokens, template=template)
    output_tok = tokenizer.decode(output_ids_tok, skip_special_tokens=True)

    logger.info(f"Query: {query}")
    logger.info(f"Response: {output_tok}")