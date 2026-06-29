# Stage 3 — Empirical Decision

The final, **training-free** routing step. Given the representative set `V`
produced by the Log Evaluator, it aggregates each candidate model's historical
performance and cost over `V`, normalizes both to `[0,1]`, and selects the model
with the highest utility:

```
U_j = λ · V_j^norm + (1 − λ) · C_j^norm        (paper λ = 0.5)
route = argmax_j U_j
```

`V = ∅` (OOD) ⇒ fall back to a high-performance default model.

| File | Role |
|------|------|
| `score.py` | Full evaluator + CLI. Aggregates perf/cost over `V`, applies the utility/argmax decision, OOD + coverage fallbacks, optional test-id filtering, and writes per-query JSON + an xlsx summary. Holds `MODEL_MAPPING` (8-model pool A–H). |
| `model_selection_by_similarity.py` | Minimal reference implementation of the core logic: `balance_function`, `aggregate_model_scores`, `select_best_model`, `process_judge_results`. Clean starting point for reuse. |
| `report_writer.py` | Pivot-table xlsx report writer used by `score.py`. |
| `run_decision.sh` | End-to-end example over the bundled Stage 3 test data. |

## Key CLI arguments (`score.py`)

| Arg | Meaning |
|-----|---------|
| `--judge_file` / `-j` | Log Evaluator output (valid representatives per query), JSONL. |
| `--history_corpus` / `-hc` | Corpus with per-model performance & cost. |
| `--user_query_full` / `-uq` | Ground-truth test queries (with perf/cost) for evaluation. |
| `--alpha` / `-a`, `--beta` / `-b` | Performance vs. cost-penalty weights in the utility function. |
| `--default_model` / `-dm` | Fallback model when `V` is empty (OOD). |
| `--enable_coverage_fallback`, `--coverage_min_models` | Fall back to the default model when too many models under-cover `V`. |
| `--filter_file`, `--filter_exempt_eval_names` | Restrict evaluation to a test-id subset (with eval-name exemptions, e.g. `IFEVAL`). |
| `--excel_output`, `--output` | xlsx summary path / per-query JSON results path. |

## Run

```bash
bash run_decision.sh
```

**Data:** `../data/stage3_test/`
- `Test_Data_Top3.jsonl` / `Test_Data_Top3.result.jsonl` — retrieval Top-3 inputs / Log-Evaluator results.
- `Label_Test_Data.jsonl` — ground-truth queries with per-model performance & cost.
- `filter_test_v1_rl.jsonl` — test-id filter subset.
