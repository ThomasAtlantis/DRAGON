# DRAGON
DRAGON is a device-cloud distributed RAG framework, which enables a simultaneous integration of personalized information and generic knowledge

```bash
HF_ENDPOINT=https://hf-mirror.com python -m dragon.toolbox.generate_passage_embeddings \
    --retriever "facebook/contriever" \
    --passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --output_dir data/embeddings \
    --batch_size 512 \
    --passage_size 128 \
    --chunk_size 64
```

```bash
python test.py \
    --repo_id "Salesforce/wikitext" \
    --passages "Salesforce/wikitext,wikitext-103-raw-v1" \
    --dataset_id "wikitext-103-raw-v1" \
    --model_config_path facebook/opt-125m  \
    --passages_embeddings "./data/embeddings/" \
    --re_model_name_or_path "facebook/contriever" \
    --retrieved_max_length 128 \
    --context_len 128 \
    --pred_len 768 \
    --output_path outputs/ppl.data \
    --ensemble 10 \
    --n_docs 10
```

TODO: Revise the huggingface cache position
TODO: Check whether we can use wikitext-103