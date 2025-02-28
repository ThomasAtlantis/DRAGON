import sys; sys.path.append(".")
from dragon.config import DragonConfig
from experiments.LanguageModeling.eval import LanguageModelingEvaluator
"""
HF_ENDPOINT=https://hf-mirror.com \
python -u run.py \
    --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
    --retriever.passages_embeddings "data/wikitext2/*.pkl" \
    --retriever.s_context 128 \
    --retriever.n_docs 8 \
    --retriever.s_aggregate 8 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 896 \
    --evaluator.output_dir "outputs/" \
    --evaluator.dataset "Salesforce/wikitext,wikitext-2-raw-v1" \
    --evaluator.s_prefix 128 \
    --cache.load_index
"""
if __name__ == "__main__":
    config = DragonConfig()
    config.parse_sys_args()
    evaluator = LanguageModelingEvaluator(config)
    evaluator.evaluate()