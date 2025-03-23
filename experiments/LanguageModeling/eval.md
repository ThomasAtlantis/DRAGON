## Tips
1. Remember to clear the cache before running: 
   - remove `.cache/wikitext-103-raw-v1`
   - remove `data/wikitext103`
2. Enable `cache.dump_index` at the first run and `cache.load_index`.
3. Running experiments in descending order of `retriever.n_docs`, enabling `cache.dump_query2docs` at the first run and `cache.load_query2docs` in the following can help reduce overall experiment time.
4. Try `generator.use_fp16` when you don't have sufficient CUDA memory.

## Scripts
- `lm_opt_wikitext103.sh`
- `lm_qwen_wikitext2.sh`