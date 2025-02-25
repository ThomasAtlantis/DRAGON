# DRAGON
DRAGON is a device-cloud distributed RAG framework, which enables a simultaneous integration of personalized information and generic knowledge

## Loss Calculation
$$\begin{aligned}CrossEntropy(\hat y,y)&=-\log \sum_{k} w^k\cdot \frac{\exp(z^k_y)}{\sum_j \exp(z^k_j)}\\
&=-\log \sum_k \exp(\log ( w^k\cdot\frac{\exp(z^k_y)}{\sum_j \exp(z^k_j)}) )\\
&=-\log \sum_k \exp(\log w^k + CrossEntropy(z^k, y))\end{aligned}$$

## wikitext-103
Generate passage embeddings:
```shell
HF_ENDPOINT=https://hf-mirror.com python -m dragon.toolbox.generate_passage_embeddings \
    --retriever "facebook/contriever" \
    --passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --output_dir data/wikitext103 \
    --batch_size 512 \
    --passage_size 128 \
    --chunk_size 64
```
Evaluate bpb score for language modeling:
```shell
python -u eval_LM.py \
    --repo_id "Salesforce/wikitext" \
    --passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --dataset_id "wikitext-103-raw-v1" \
    --model_config_path facebook/opt-1.3b  \
    --passages_embeddings "./data/embeddings/*.pkl" \
    --re_model_name_or_path "facebook/contriever" \
    --retrieved_max_length 128 \
    --context_len 128 \
    --pred_len 768 \
    --output_path outputs/ppl.data \
    --n_docs 10 \
    --ensemble 10 \
    --load_index true \
    --cache.load_query2docs true
```

## wikitext-2
Generate passage embeddings:
```shell
HF_ENDPOINT=https://hf-mirror.com python -m dragon.toolbox.generate_passage_embeddings \
    --retriever "facebook/contriever" \
    --passages "Salesforce/wikitext,wikitext-2-raw-v1" \
    --output_dir "./data/wikitext2" \
    --batch_size 512 \
    --passage_size 128 \
    --chunk_size 64
```
Evaluate bpb score for language modeling:
```shell
python -u eval_LM.py \
    --repo_id "Salesforce/wikitext" \
    --dataset_id "wikitext-2-raw-v1" \
    --passages "Salesforce/wikitext,wikitext-2-raw-v1" \
    --model_config_path facebook/opt-1.3b  \
    --passages_embeddings "./data/wikitext2/*.pkl" \
    --re_model_name_or_path "facebook/contriever" \
    --retrieved_max_length 128 \
    --context_len 128 \
    --pred_len 768 \
    --output_path "outputs/wikitext2.data" \
    --n_docs 2 \
    --ensemble 2 \
    --dump_index true
```
Results:
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
- [x] Find why the results are different from that of the official implementation
- [x] rename passages.txt to passages.jsonl
- [ ] Dynamic document weight
- [ ] Distributed RAG
- [ ] Speculative DRAG
- [ ] Scheduling algorithm

### Optimization
- [ ] Remove position information
- [ ] Batch the output ensemble
- [ ] Decoupling decoding and evaluation