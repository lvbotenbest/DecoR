#!/usr/bin/env bash
# Stage 2 — Substage C: Log-Alignment Evaluation (Log Evaluator inference).
#
# Serve the trained Log Evaluator with an OpenAI-compatible endpoint first, e.g.
#   vllm serve /path/to/log_evaluator/merge_model --port 8080 --served-model-name log_evaluator
#
# Then run this script. It reads the retrieval results (Top-k logs per query),
# asks the Log Evaluator which logs can represent the user query, and writes the
# selected "valid_representatives" set per query.

set -e

DATA=../../../data

export VLLM_OPENAI_BASE_URL="${VLLM_OPENAI_BASE_URL:-http://127.0.0.1:8080/v1}"
export VLLM_OPENAI_API_KEY="${VLLM_OPENAI_API_KEY:-EMPTY}"
export LOG_EVALUATOR_MODEL="${LOG_EVALUATOR_MODEL:-/path/to/log_evaluator/merge_model}"

python predict_log_evaluator.py \
    --input  "${DATA}/stage3_test/Test_Data_Top3.jsonl" \
    --output "${DATA}/stage3_test/Test_Data_Top3.result.jsonl" \
    --base-url "${VLLM_OPENAI_BASE_URL}" \
    --api-key  "${VLLM_OPENAI_API_KEY}" \
    --model    "${LOG_EVALUATOR_MODEL}" \
    --batch-size 64 \
    --temperature 0.0 \
    --max-tokens 256
