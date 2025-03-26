python -u experiments/LanguageModeling/eval.py \
    --retriever.passages Salesforce/wikitext,wikitext-2-raw-v1 \
    --retriever.passages_embeddings data/wikitext2/*.pkl \
    --retriever.s_context 256 \
    --generator.model Qwen/Qwen2.5-1.5B \
    --generator.s_sequence 512 \
    --evaluator.s_prefix 64 \
    --evaluator.dataset Salesforce/wikitext,wikitext-2-raw-v1 \
    --evaluator.output_dir outputs/ \
    --cache.load_index \
    --cache.load_query2docs \
    --generator.use_fp16 \
    --retriever.n_docs 5 \
    --retriever.s_aggregate 1 \
    --retriever.downsample_type 2