# End-to-End Demo

Runs the **entire DecoR routing pipeline** on an input query dataset and, if
ground-truth labels are provided, evaluates the routing quality — all in one
command.

```
input queries
    │
    ▼  Stage 1   Query Deconstruction      q → capability profile C(q)={S,K,D}
    ▼  Stage 2AB Hierarchical Log-Sifting   retrieve Top-k analogous historical logs
    ▼  Stage 2C  Log Evaluator              keep representative set V ⊆ Top-k
    ▼  Stage 3   Empirical Decision          aggregate perf/cost over V → argmax utility
    ▼
routed model per query (+ evaluation vs. ground truth)
```

## Quick start (fully offline)

No API key or GPU required — uses the shipped corpus + sample queries:

```bash
bash demo/run_demo.sh
```

Expected tail of the output:

```
PIPELINE SUMMARY
  q9      V=['A']           -> A (similarity_based) perf=1
  q4      V=['A', 'B', 'C'] -> A (similarity_based) perf=1
  ...
  q9560   V=[]              -> H (default_ood)      perf=0
Routed accuracy (avg performance): 0.80 over 10 queries
OOD / default-fallback: 1/10
```

## Files

| File | Role |
|------|------|
| `run_pipeline.py` | The orchestrator. Chains Stage 1 → 2AB → 2C → 3, each with a real-service path and an offline fallback. Writes per-stage artifacts. |
| `run_demo.sh` | One-command offline run with sensible defaults. |
| `sample_queries.jsonl` | 10 example queries (`{id, prompt, decomposition, eval_name}`) drawn from the test set. |
| `output/` | Per-stage artifacts produced by a run (see below). |

## How each stage runs (real service vs. offline)

| Stage | Real service | Offline fallback (default) |
|-------|--------------|----------------------------|
| 1 Deconstruction | deconstructor API (`QIANFAN_API_KEY`) | `--use-precomputed-decomp` reuses the `decomposition` field in the input |
| 2AB Retrieval | semantic embeddings (`DEEPINFRA_API_KEY`) via `--use-api-embedding` + `--retrieval-mode fine/hybrid` | `--retrieval-mode coarse` — structured S/K/D matching only |
| 2C Log Evaluator | served model via `--le-mode vllm` (`VLLM_OPENAI_BASE_URL`) | `--le-mode rule` — transparent difficulty/skill/knowledge coverage rule |
| 3 Decision | — (always rule-based) | rule-based utility `U = α·perf − β·cost`, argmax |

## Output artifacts (`demo/output/`)

| File | Contents |
|------|----------|
| `stage1_decomposition.jsonl` | queries with parsed capability profile |
| `stage2_retrieval.jsonl` | Top-k retrieved logs per query (with scores) |
| `stage2c_log_evaluator.jsonl` | representative set `V` (labels) per query |
| `stage3_decisions.jsonl` | selected model + method + (if labeled) perf/cost per query |
| `metrics.json` | aggregate routed accuracy, cost, OOD count |

## Running on your own data

Provide an input JSONL of `{ "id", "prompt" }` (add `"decomposition"` to skip
Stage 1, or drop `--use-precomputed-decomp` to decompose via the API):

```bash
python3 demo/run_pipeline.py \
    --input   my_queries.jsonl \
    --corpus  data/corpus/corpus_with_perf_cost.jsonl \
    --labels  ""                       `# omit to skip evaluation` \
    --retrieval-mode hybrid --use-api-embedding \
    --le-mode vllm \
    --top-k 3 --alpha 0.5 --beta 0.5
```

Full-strength (paper) setup uses `--retrieval-mode hybrid` with an embedding
API and `--le-mode vllm` with the GRPO-trained Log Evaluator; the offline
defaults trade some accuracy for zero external dependencies.
