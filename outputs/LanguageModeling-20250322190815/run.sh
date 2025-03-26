python -u experiments/LanguageModeling/eval.py \
    --retriever.passages Salesforce/wikitext,wikitext-2-raw-v1 \
    --retriever.passages_embeddings data/wikitext2/*.pkl \
    --retriever.s_context 256 \
    --generator.model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --generator.s_sequence 640 \
    --evaluator.s_prefix 256 \
    --evaluator.dataset Salesforce/wikitext,wikitext-2-raw-v1 \
    --evaluator.output_dir outputs/ \
    --cache.load_index \
    --cache.load_query2docs \
    --generator.use_fp16 \
    --retriever.n_docs 0 \
    --retriever.s_aggregate 0 \
    --retriever.downsample_type 0