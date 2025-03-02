# wikitext-103
# Generate passage embeddings:
# Don't forget to clear cache before running this command
HF_ENDPOINT=https://hf-mirror.com \
python -m dragon.toolbox.embed_passages \
    --retriever.model "contriever" \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --output_dir "./data/wikitext103" \
    --retriever.bs_encode 1024 \
    --retriever.s_passage 64 \
    --text.with_title
# Evaluate cross entropy for language modeling:
python -u run.py \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --retriever.passages_embeddings "data/wikitext103/*.pkl" \
    --retriever.s_context 128 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 896 \
    --evaluator.output_dir "outputs/" \
    --evaluator.dataset "Salesforce/wikitext,wikitext-103-raw-v1" \
    --evaluator.s_prefix 128 \
    --cache.dump_index \
    --cache.dump_query2docs \
    --retriever.n_docs 16 \
    --retriever.s_aggregate 16

# wikitext-2
# Generate passage embeddings:
# Don't forget to clear cache before running this command
HF_ENDPOINT=https://hf-mirror.com \
python -m dragon.toolbox.embed_passages \
    --retriever.model "contriever" \
    --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
    --output_dir "./data/wikitext2" \
    --retriever.bs_encode 512 \
    --retriever.s_passage 64 \
    --text.with_title
# Evaluate cross entropy for language modeling:
HF_ENDPOINT=https://hf-mirror.com \
python -u run.py \
    --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
    --retriever.passages_embeddings "data/wikitext2/*.pkl" \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 896 \
    --evaluator.output_dir "outputs/" \
    --evaluator.dataset "Salesforce/wikitext,wikitext-2-raw-v1" \
    --evaluator.s_prefix 128 \
    --cache.load_index \
    --cache.dump_query2docs \
    --retriever.s_context 128 \
    --retriever.n_docs 16 \
    --retriever.s_aggregate 16