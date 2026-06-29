#!/usr/bin/env bash
# Stage 3 — Empirical Decision Stage.
# Reads the Log Evaluator output (valid representatives per query), aggregates
# the historical performance/cost of each candidate model over the selected
# logs, and picks the model with the highest utility U = alpha*V + beta*C.
#
# All paths are relative to this directory (decor_code/stage3_empirical_decision).

set -e

DATA=../data

python score.py \
    --judge_file        "${DATA}/stage3_test/Test_Data_Top3.result.jsonl" \
    --history_corpus    "${DATA}/corpus/corpus_with_perf_cost.jsonl" \
    --user_query_full   "${DATA}/stage3_test/Label_Test_Data.jsonl" \
    --alpha 1.0 \
    --beta 1.0 \
    --default_model "Qwen_Qwen3-235B-A22B-Instruct-2507" \
    --filter_file   "${DATA}/stage3_test/filter_test_v1_rl.jsonl" \
    --filter_exempt_eval_names IFEVAL \
    --enable_coverage_fallback \
    --coverage_min_models 5 \
    --excel_output "${DATA}/stage3_test/result.xlsx" \
    --output       "${DATA}/stage3_test/Test_testset_results.json"
