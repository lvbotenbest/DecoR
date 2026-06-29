# Stage 2 — Hierarchical Log-Sifting

Given a deconstructed query, progressively narrows the historical log corpus
down to a small, high-quality **representative set `V`** of analogous logs.
Three substages, coarse → fine → semantic:

```
query+C(q)  ──A──►  candidate logs  ──B──►  Top-k logs  ──C──►  representative set V
            (char)                  (embed)             (LLM)        │
                                                                     └─ V = ∅ ⇒ OOD
```

## Substage A + B — Retrieval (`retrieval/`)

| File | Role |
|------|------|
| `query_retrieval.py` | Core engine: `SynonymMapper`, `StructuredMatcher` (Substage A — inverted index + Jaccard on Skills/Knowledge + difficulty weight), `APIEmbedding`/`FAISSIndex` (Substage B — BGE-M3 cosine fine-rank), and the `QueryRetriever` orchestrator (coarse / fine / hybrid modes). |
| `run_retrieval.py` | CLI entry point: build index or retrieve Top-k logs per query. |
| `run_retrieval.sh` | Example hybrid run (τ=0.5 coarse threshold, Top-k=3, α=0.3, β=0.7). |
| `synonym_dict.json` | Skill/Knowledge synonym lexicon used by `SynonymMapper`. |
| `update_synonym.py` | Expand the synonym lexicon via an LLM. Token via `QIANFAN_API_KEY`. |
| `similarity_judge.py` | LLM-as-judge to label whether two logs are genuinely similar (builds the Log-Evaluator training labels). Key via `OPENAI_API_KEY`. |
| `build_similarity_dataset.py` / `build_similarity_dataset_for_test_dataset.py` | Assemble (query, Top-k logs) pairs into the Log-Evaluator train / test sets. |
| `calculate_model_stats.py` | Aggregate per-model performance/cost statistics over the corpus. |
| `filter_zero_performance_cases.py` | Drop logs where every model scores zero (uninformative). |
| `read_cache.py` | Inspect cached embeddings / intermediate retrieval artifacts. |

Substage A returns OOD when no candidate clears τ; Substage B re-ranks the
survivors by cosine similarity on `q ⊕ String(p)` and keeps the Top-k (=3).

```bash
cd retrieval
export DEEPINFRA_API_KEY=...     # only if using --use_api_embedding
bash run_retrieval.sh
```

## Substage C — Log Evaluator (`log_evaluator/`)

A small LLM (paper: Qwen3-0.6B, GRPO, lr 1e-6) that reads the Top-k logs and
outputs the subset `V` that is genuinely representative of the query. `V = ∅`
⇒ the query is treated as OOD.

### Training (`log_evaluator/train/`)

| File | Role |
|------|------|
| `process_data_for_parquet.py` | Convert similarity datasets into VERL parquet; embeds the `SIMILARITY_JUDGE_PROMPT`. |
| `reward_router.py` | `compute_score_llmroute` — the GRPO reward `R(V,G)` (paper Eq. 13): `6` if `V=G`; `-2·|V|` if `G=∅ ∧ V≠∅`; `-6` if no hit; else `6/|G|·(hits − false_pos)`. |
| `train_grpo.sh` | VERL GRPO launch script. Parameterized via `VERL_MAIN`, `DATA_DIR`, `REWARD_FN`, `BASE_MODEL` env vars. |

```bash
cd log_evaluator/train
python process_data_for_parquet.py        # builds train.parquet / test.parquet
export VERL_MAIN=/path/to/verl/trainer/main_ppo.py
export BASE_MODEL=/path/to/Qwen3-0.6B
bash train_grpo.sh
```

### Prediction (`log_evaluator/predict/`)

| File | Role |
|------|------|
| `predict_log_evaluator.py` | Run the trained Log Evaluator over retrieval results to produce `V`. I/O paths relative; `BASE_URL`/`API_KEY`/`MODEL` via env vars. |
| `vllm_test.py` | Minimal vLLM smoke test against the merged Log-Evaluator model. |
| `run_predict.sh` | Example prediction run. |

```bash
cd log_evaluator/predict
# serve the merged model with vLLM first, then:
export BASE_URL=http://localhost:8000/v1
bash run_predict.sh
```

**Data:** `../data/stage2_log_evaluator/{train,test}.parquet` (VERL GRPO format).
