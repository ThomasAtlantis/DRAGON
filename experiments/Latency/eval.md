## DRDG/TW
### Cloud
```shell
python experiments/Latency/eval.py \
    --trans.rank 0 \
    --aggregator.mode synchronized \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --generator.use_fp16
```

### Device
```shell
python experiments/Latency/eval.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 10 \
    --generator.use_fp16 \
    --aggregator.mode synchronized
```

## DRAGON
### Cloud
```shell
python experiments/Latency/eval.py \
    --trans.rank 0 \
    --aggregator.mode speculative \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --generator.use_fp16
```

### Device
```shell
python experiments/Latency/eval.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 10 \
    --generator.use_fp16 \
    --aggregator.mode speculative
```

## DRDG/SW
### Cloud
```shell
python experiments/Latency/eval_DRDG_SW.py \
    --trans.rank 0 \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --generator.use_fp16
```

### Device
```shell
python experiments/Latency/eval_DRDG_SW.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 10 \
    --generator.use_fp16

```