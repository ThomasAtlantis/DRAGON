## Profiling
### OPT
#### Cloud
```shell
python experiments/Scheduling/eval_profiling.py \
    --trans.rank 0 \
    --aggregator.mode synchronized \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --generator.use_fp16 \
    --evaluator.n_prompts 1 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

#### Device
```shell
python experiments/Scheduling/eval_profiling.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 20 \
    --generator.use_fp16 \
    --aggregator.mode synchronized \
    --evaluator.n_prompts 1 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

### Qwen
#### Cloud
```shell
python experiments/Scheduling/profiling.py \
    --trans.rank 0 \
    --aggregator.mode synchronized \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "Qwen/Qwen2.5-1.5B" \
    --generator.use_fp16 \
    --evaluator.n_prompts 1 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

#### Device
```shell
python experiments/Scheduling/profiling.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "Qwen/Qwen2.5-1.5B" \
    --evaluator.max_new_tokens 20 \
    --generator.use_fp16 \
    --aggregator.mode synchronized \
    --evaluator.n_prompts 1 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```
