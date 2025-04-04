## Profiling
### OPT
#### Cloud
```shell
python experiments/Scheduling/profiling.py \
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
python experiments/Scheduling/profiling.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 100 \
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
    --evaluator.max_new_tokens 100 \
    --generator.use_fp16 \
    --aggregator.mode synchronized \
    --evaluator.n_prompts 1 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

#### Simulation
```shell
python -u experiments/Scheduling/eval.py \
    --max_draft_tokens 10 \
    --max_new_tokens 99 \
    --stats "outputs/Latency-20250330144956/stats.json" \
    --offline_profile "outputs/Latency-20250330144956/profile.json" \
    --online_profile "outputs/Latency-20250330144956/decoder_stats.json"
```