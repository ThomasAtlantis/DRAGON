# DRAGON
DRAGON is a device-cloud distributed RAG framework, which enables a simultaneous integration of personalized information and generic knowledge

## Terminology
|Term        |Description                                                         |
|------------|--------------------------------------------------------------------|
|passage     |the paragraph/chunk of text to be retrieved from assistant documents|
|context     |the passages inserted before the query                              |
|aggregate   |fusion/ensemble of multiple output distributions                    |
|embed       |transform the query/passages into vectors using the retriever model |
|encode      |transform raw text into tokens                                      |
|decode      |transform tokens into raw text                                      |
|generate    |predict the output sequence given the query                         |
|rag         |predict the output sequence using generator, given the query concatenated with context retrieved by the retriever|
|s_*(single) |size/length of *                                                    |
|n_*(plural) |number of *                                                         |
|bs_*(single)|batch size of *                                                     |
## Language Modeling
### Loss Calculation
$$\begin{aligned}CrossEntropy(\hat y,y)&=-\log \sum_{k} w^k\cdot \frac{\exp(z^k_y)}{\sum_j \exp(z^k_j)}\\
&=-\log \sum_k \exp(\log ( w^k\cdot\frac{\exp(z^k_y)}{\sum_j \exp(z^k_j)}) )\\
&=-\log \sum_k \exp(\log w^k + CrossEntropy(z^k, y))\end{aligned}$$
### wikitext-103
Generate passage embeddings:
```shell
HF_ENDPOINT=https://hf-mirror.com \
python -m dragon.toolbox.embed_passages \
    --retriever.model "contriever" \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --output_dir "./data/wikitext103" \
    --retriever.bs_encode 512 \
    --retriever.s_passage 128 \
    --retriever.s_passage_chunk 64
```
Evaluate bpb score for language modeling:
```shell
python -u eval_LM.py \
    --retriever.passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --retriever.passages_embeddings "data/wikitext103/*.pkl" \
    --retriever.s_context 128 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 2 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 896 \
    --evaluator.output_dir "outputs/" \
    --evaluator.dataset "Salesforce/wikitext,wikitext-103-raw-v1" \
    --evaluator.s_prefix 128 \
    --cache.dump_index true
```

### wikitext-2
Generate passage embeddings:
```shell
HF_ENDPOINT=https://hf-mirror.com \
python -m dragon.toolbox.embed_passages \
    --retriever.model "contriever" \
    --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
    --output_dir "./data/wikitext2" \
    --retriever.bs_encode 512 \
    --retriever.s_passage 128 \
    --retriever.s_passage_chunk 64
```
Evaluate bpb score for language modeling:
```shell
python -u eval_LM.py \
    --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
    --retriever.passages_embeddings "data/wikitext2/*.pkl" \
    --retriever.s_context 128 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 2 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 896 \
    --evaluator.output_dir "outputs/" \
    --evaluator.dataset "Salesforce/wikitext,wikitext-2-raw-v1" \
    --evaluator.s_prefix 128 \
    --cache.load_index true
```
### Results
|Generator  |Retriever  |Dataset   |Ensemble|BPB    |
|-----------|-----------|----------|--------|-------|
|opt-1.3b   |Contriever |wikitext-2|0       |2.8867 |
|           |           |          |2       |2.7041 |
|           |           |          |4       |2.6968 |
|           |           |          |4_concat|2.7070 |
|           |           |          |10      |2.6899 |
|           |           |          |16      |       |

## TODO List
### Logic
- [ ] Revise Replug to support token-wise aggregation
- [ ] Dynamic document weight
- [ ] Distributed RAG
- [ ] Speculative DRAG
- [ ] Scheduling algorithm

### Optimization
- [ ] Rename decoder to RAG
- [ ] Decouple the evaluator from decoder
- [ ] Remove position information
- [ ] Batch the output ensemble
- [ ] Decoupling decoding and evaluation