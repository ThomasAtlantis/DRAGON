## Measure the Prefix Proportion
```shell
python -u experiments/TTFT/eval_prefix_proportion.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "Qwen/Qwen2.5-1.5B" \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

## Evaluate Retrieval Latency
### Device
```shell
python -u experiments/TTFT/eval_retrieval_latency.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "Qwen/Qwen2.5-1.5B" \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

### Cloud
```shell
python -u experiments/TTFT/eval_retrieval_latency.py \
    --trans.rank 0 \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "Qwen/Qwen2.5-1.5B" \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

## Evaluate TTFT
### Device
```shell
python -u experiments/TTFT/eval.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "Qwen/Qwen2.5-1.5B" \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

### Cloud
```shell
python -u experiments/TTFT/eval.py \
    --trans.rank 0 \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "Qwen/Qwen2.5-1.5B" \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

## Evaluate Latency of KV Downloading
### Device
```shell
python -u experiments/TTFT/eval_download_kv.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "Qwen/Qwen2.5-1.5B" \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

### Cloud
```shell
python -u experiments/TTFT/eval_download_kv.py \
    --trans.rank 0 \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "Qwen/Qwen2.5-1.5B" \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```