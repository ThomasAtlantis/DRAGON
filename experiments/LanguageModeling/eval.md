# OPT1.3B on WikiText103
## Tips
1. Remember to clear the cache before running: 
   - remove `.cache/wikitext-104-raw-v1`
   - remove `data/wikitext103`
2. Enable `cache.dump_index` at the first run and `cache.load_index`.
3. Running experiments in descending order of `retriever.n_docs`, enabling `cache.dump_query2docs` at the first run and `cache.load_query2docs` in the following can help reduce overall experiment time.
4. Try `generator.use_fp16` when you don't have sufficient CUDA memory.

## Generate Passage Embeddings
```shell
HF_ENDPOINT=https://hf-mirror.com \
python -m dragon.toolbox.embed_passages \
    --retriever.model "contriever" \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --output_dir "./data/wikitext103" \
    --retriever.bs_encode 512 \
    --retriever.s_passage 64 \
    --text.with_title
```
## Evaluate Language Modeling
### DRAGON
```shell
python -u experiments/LanguageModeling/eval.py \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --retriever.passages_embeddings "data/wikitext103/*.pkl" \
    --retriever.s_context 256 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 1024 \
    --evaluator.s_prefix 128 \
    --evaluator.dataset "Salesforce/wikitext,wikitext-103-raw-v1" \
    --evaluator.output_dir "outputs/" \
    --cache.dump_index \
    --cache.dump_query2docs \
    --generator.use_fp16 \
    --retriever.n_docs 16 \
    --retriever.s_aggregate 16 \
    --retriever.downsample_type 0
```
### W/O Retrieval
```shell
python -u experiments/LanguageModeling/eval.py \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --retriever.passages_embeddings "data/wikitext103/*.pkl" \
    --retriever.s_context 256 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 1024 \
    --evaluator.s_prefix 128 \
    --evaluator.dataset "Salesforce/wikitext,wikitext-103-raw-v1" \
    --evaluator.output_dir "outputs/" \
    --cache.load_index \
    --cache.load_query2docs \
    --generator.use_fp16 \
    --retriever.n_docs 0 \
    --retriever.s_aggregate 0 \
    --retriever.downsample_type 0
```
### DRCG
```shell
python -u experiments/LanguageModeling/eval.py \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --retriever.passages_embeddings "data/wikitext103/*.pkl" \
    --retriever.s_context 256 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 1024 \
    --evaluator.s_prefix 128 \
    --evaluator.dataset "Salesforce/wikitext,wikitext-103-raw-v1" \
    --evaluator.output_dir "outputs/" \
    --cache.load_index \
    --cache.load_query2docs \
    --generator.use_fp16 \
    --retriever.n_docs 16 \
    --retriever.s_aggregate 1 \
    --retriever.downsample_type 0
```
### CRCG
#### Cloud Side
```shell
python -u experiments/LanguageModeling/eval.py \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --retriever.passages_embeddings "data/wikitext103/*.pkl" \
    --retriever.s_context 256 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 1024 \
    --evaluator.s_prefix 128 \
    --evaluator.dataset "Salesforce/wikitext,wikitext-103-raw-v1" \
    --evaluator.output_dir "outputs/" \
    --cache.load_index \
    --cache.load_query2docs \
    --generator.use_fp16 \
    --retriever.n_docs 16 \
    --retriever.s_aggregate 1 \
    --retriever.downsample_type 1
```
#### Device Side
```shell
python -u experiments/LanguageModeling/eval.py \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --retriever.passages_embeddings "data/wikitext103/*.pkl" \
    --retriever.s_context 256 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 1024 \
    --evaluator.s_prefix 128 \
    --evaluator.dataset "Salesforce/wikitext,wikitext-103-raw-v1" \
    --evaluator.output_dir "outputs/" \
    --cache.load_index \
    --cache.load_query2docs \
    --generator.use_fp16 \
    --retriever.n_docs 16 \
    --retriever.s_aggregate 1 \
    --retriever.downsample_type 2
```
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
    --retriever.s_context 256 \
    --retriever.n_docs 16 \
    --retriever.s_aggregate 16