#!/usr/bin/env bash
# Stage 2 — Hierarchical Log-Sifting (Substage A + B: retrieval).
#
# Substage A: character-level inverted-index filtering with Jaccard similarity
#             on Skills/Knowledge plus a difficulty matching weight.
# Substage B: BGE-M3 / API embedding fine-ranking by cosine similarity.
#
# Output: for each user query, the Top-k most relevant historical logs with
#         their decomposition + ids (consumed by the Log Evaluator in Substage C).
#
# Paths are relative to this directory (stage2_hierarchical_sifting/retrieval).

set -e

DATA=../../data
CORPUS="${DATA}/corpus/corpus_decomposition_for_retrieval.jsonl"

# ----------------------------------------------------------------------------
# (Optional) Step 0: pre-build an embedding + FAISS index over the corpus.
# Requires an embedding API key (export DEEPINFRA_API_KEY=...) when using
# --use_api_embedding, or omit it to use a local sentence-transformers model.
# ----------------------------------------------------------------------------
# python run_retrieval.py --build_index \
#     --corpus "${CORPUS}" \
#     --use_api_embedding \
#     --use_faiss \
#     --faiss_path "${DATA}/faiss_index/corpus_index" \
#     --api_batch_size 50

# ----------------------------------------------------------------------------
# Step 1: retrieve Top-k logs for each user query (hybrid coarse->fine mode).
# Replace --input with your decomposed user queries (Stage 1 output).
# ----------------------------------------------------------------------------
python run_retrieval.py \
    --input  "${DATA}/stage1_deconstruction/merged_query_decomposition_test.jsonl" \
    --output "${DATA}/stage3_test/retrieval_results_test.jsonl" \
    --corpus "${CORPUS}" \
    --top_k 3 \
    --mode hybrid \
    --coarse_threshold 0.5 \
    --coarse_top_k 100 \
    --fine_top_k 10 \
    --alpha 0.3 \
    --beta 0.7
