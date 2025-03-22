#!/bin/bash
set -e

DRAGON() {
    # $1: n_docs
    # $2: start or following
    if [ "$2" == "start" ]; then
        index_flag="dump_index"
        query2docs_flag="dump_query2docs"
    else
        index_flag="load_index"
        query2docs_flag="load_query2docs"
    fi
    echo "Evaluating DRAGON with n_docs=$1"
    python -u experiments/LanguageModeling/eval.py \
        --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
        --retriever.passages_embeddings "data/wikitext2/*.pkl" \
        --retriever.s_context 256 \
        --generator.model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  \
        --generator.s_sequence 640 \
        --evaluator.s_prefix 64 \
        --evaluator.dataset "Salesforce/wikitext,wikitext-2-raw-v1" \
        --evaluator.output_dir "outputs/" \
        --cache.$index_flag \
        --cache.$query2docs_flag \
        --generator.use_fp16 \
        --retriever.n_docs $1 \
        --retriever.s_aggregate $1 \
        --retriever.downsample_type 0
}

NoRetrieval() {
    echo "Evaluating NoRetrieval"
    python -u experiments/LanguageModeling/eval.py \
        --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
        --retriever.passages_embeddings "data/wikitext2/*.pkl" \
        --retriever.s_context 256 \
        --generator.model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  \
        --generator.s_sequence 640 \
        --evaluator.s_prefix 64 \
        --evaluator.dataset "Salesforce/wikitext,wikitext-2-raw-v1" \
        --evaluator.output_dir "outputs/" \
        --cache.load_index \
        --cache.load_query2docs \
        --generator.use_fp16 \
        --retriever.n_docs 0 \
        --retriever.s_aggregate 0 \
        --retriever.downsample_type 0
}

DRCG() {
    # $1: n_docs
    echo "Evaluating DRCG with n_docs=$1"
    python -u experiments/LanguageModeling/eval.py \
        --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
        --retriever.passages_embeddings "data/wikitext2/*.pkl" \
        --retriever.s_context 256 \
        --generator.model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  \
        --generator.s_sequence 640 \
        --evaluator.s_prefix 64 \
        --evaluator.dataset "Salesforce/wikitext,wikitext-2-raw-v1" \
        --evaluator.output_dir "outputs/" \
        --cache.load_index \
        --cache.load_query2docs \
        --generator.use_fp16 \
        --retriever.n_docs $1 \
        --retriever.s_aggregate 1 \
        --retriever.downsample_type 0
}

CRCG() {
    # $1: n_docs
    # $2: device or cloud
    echo "Evaluating CRCG/$2 with n_docs=$1"
    n_docs_half=$(( $1 / 2 ))
    if [ "$2" == "cloud" ]; then
        downsample_type=1
    else
        downsample_type=2
    fi
    python -u experiments/LanguageModeling/eval.py \
        --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
        --retriever.passages_embeddings "data/wikitext2/*.pkl" \
        --retriever.s_context 256 \
        --generator.model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  \
        --generator.s_sequence 640 \
        --evaluator.s_prefix 64 \
        --evaluator.dataset "Salesforce/wikitext,wikitext-2-raw-v1" \
        --evaluator.output_dir "outputs/" \
        --cache.load_index \
        --cache.load_query2docs \
        --generator.use_fp16 \
        --retriever.n_docs $n_docs_half \
        --retriever.s_aggregate 1 \
        --retriever.downsample_type $downsample_type
}

# echo "Embedding passages for WikiText2 ..."
# HF_ENDPOINT=https://hf-mirror.com \
# python -m dragon.toolbox.embed_passages \
#     --retriever.model "contriever" \
#     --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
#     --output_dir "./data/wikitext2" \
#     --retriever.bs_encode 640 \
#     --retriever.s_passage 64 \
#     --text.with_title

echo "Evaluating Qwen1.5B on WikiText2 ..."
NoRetrieval
DRAGON 16 start
for n_docs in {2,4,6,8,10,12,14}; do
    DRAGON $n_docs following
    DRCG $n_docs
    CRCG $n_docs cloud
    CRCG $n_docs device
done

echo "All done!"