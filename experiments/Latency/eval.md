## DRDG/TW
### Cloud
```shell
python experiments/Latency/eval.py \
    --trans.rank 0 \
    --aggregator.mode synchronized \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

### Device
```shell
python experiments/Latency/eval.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 20 \
    --generator.use_fp16 \
    --aggregator.mode synchronized \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
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
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

### Device
```shell
python experiments/Latency/eval.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 20 \
    --generator.use_fp16 \
    --aggregator.mode speculative \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1
```

## DRDG/SW
### Cloud
```shell
python experiments/Latency/eval_DRDG_SW.py \
    --trans.rank 0 \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --generator.use_fp16 \
    --evaluator.n_prompts 5
```

### Device
```shell
python experiments/Latency/eval_DRDG_SW.py \
    --trans.rank 1 \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 20 \
    --generator.use_fp16 \
    --evaluator.n_prompts 5
```

## CRCG/Cloud
```shell
python experiments/Latency/eval_CRCG.py \
    --device cuda:0 \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 20 \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1 \
    --retriever.downsample_type 2
```

## CRCG/Device
```shell
python experiments/Latency/eval_CRCG.py \
    --device cpu \
    --retriever.passages "wikipedia[remote]" \
    --generator.model "facebook/opt-1.3b" \
    --evaluator.max_new_tokens 20 \
    --generator.use_fp16 \
    --evaluator.n_prompts 5 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 1 \
    --retriever.downsample_type 1
```