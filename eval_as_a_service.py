"""
HF_ENDPOINT=https://hf-mirror.com \
python -u eval_as_a_service.py \
    --retriever.passages Salesforce/wikitext,wikitext-103-raw-v1 \
    --retriever.passages_embeddings data/wikitext103/*.pkl \
    --retriever.s_context 1024 \
    --retriever.n_docs 16 \
    --retriever.s_aggregate 2 \
    --generator.model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --generator.s_sequence 896 \
    --cache.load_index \
    --reranker.do_rerank \
    --sampler.do_sample \
    --sampler.top_k 10
"""
import json
import sys
from flask import Flask, request
from dragon.config import DragonConfig
from dragon.rag import RagSequenceForGeneration, RagTokenForGeneration
from dragon.utils.mlogging import Logger
from dragon.utils.stable import seed_everything
seed_everything(42)


class EvalServer:
    
    def __init__(self, config: DragonConfig):
        self.rag_seq = RagSequenceForGeneration(config)
        self.rag_tok = RagTokenForGeneration(config)
        self.tokenizer = self.rag_seq.generator.tokenizer
        self.logger = Logger.build(__class__.__name__, level="INFO")

    def generate(self, query, max_new_tokens, mode):
        query_ids = self.tokenizer.encode(query)
        self.logger.info(f"Query: {query}")

        if mode == "seq":
            output_ids_seq, _ = self.rag_seq.generate(query_ids, max_new_tokens=max_new_tokens)
            output_seq = self.tokenizer.decode(output_ids_seq, skip_special_tokens=True)
            self.logger.info(f"Output (Seq): {output_seq}")
            
            return {
                "query": query,
                "output": output_seq
            }
        if mode == "tok":
            output_ids_tok, _ = self.rag_tok.generate(query_ids, max_new_tokens=max_new_tokens)
            output_tok = self.tokenizer.decode(output_ids_tok, skip_special_tokens=True)
            self.logger.info(f"Output (Tok): {output_tok}")

            return {
                "query": query,
                "output": output_tok
            }


app = Flask(__name__)
config = DragonConfig()
config.parse_sys_args()
evaluator = EvalServer(config)

@app.route("/quit", methods=["POST"])
def quit():
    sys.exit(0)

@app.route("/test", methods=["POST"])
def test():
    query = request.json.get("query")
    max_new_tokens = request.json.get("max_new_tokens")
    return {"query": query, "max_new_tokens": max_new_tokens}

@app.route("/eval", methods=["POST"])
def evaluate_endpoint():
    query = request.json.get("query")
    max_new_tokens = request.json.get("max_new_tokens")
    mode = request.json.get("mode", "tok")
    result = evaluator.generate(query, max_new_tokens, mode)
    return result

app.run(host='localhost', port=8000)
