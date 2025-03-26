python -u experiments/LanguageModeling/eval.py \
    --retriever.passages Salesforce/wikitext,wikitext-103-raw-v1 \
    --retriever.passages_embeddings data/wikitext103/*.pkl \
    --retriever.s_context 256 \
    --generator.model facebook/opt-1.3b \
    --generator.s_sequence 1024 \
    --evaluator.s_prefix 128 \
    --evaluator.dataset Salesforce/wikitext,wikitext-103-raw-v1 \
    --evaluator.output_dir outputs/ \
    --cache.load_index \
    --cache.load_query2docs \
    --generator.use_fp16 \
    --retriever.n_docs 4 \
    --retriever.s_aggregate 1 \
    --retriever.downsample_type 1