python -u run.py \
    --retriever.passages Salesforce/wikitext,wikitext-2-raw-v1 \
    --retriever.passages_embeddings data/wikitext2/*.pkl \
    --retriever.s_context 128 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 2 \
    --generator.model facebook/opt-1.3b \
    --generator.s_sequence 896 \
    --cache.load_index \
    --evaluator.max_new_tokens 15