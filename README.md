<div align="center">

# 🧭 DecoR

### Beyond Query Memorization: LLM Routing with Query Decomposition and Historical Matching

<p>
  <img src="https://img.shields.io/badge/Paper-ACL%202026-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper">
  <img src="https://img.shields.io/badge/License-MIT-3da639?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
</p>

<p>
  <b>🔍 Query Decomposition</b> &nbsp;•&nbsp;
  <b>📚 Hierarchical Log-Sifting</b> &nbsp;•&nbsp;
  <b>⚖️ Empirical Decision</b>
</p>

<p>
  <a href="https://aclanthology.org/2026.acl-long.1852/"><b>📄 Paper</b></a> &nbsp;|&nbsp;
  <a href="#-pipeline-overview"><b>🛠 Pipeline</b></a> &nbsp;|&nbsp;
  <a href="#-end-to-end-usage"><b>🚀 Usage</b></a> &nbsp;|&nbsp;
  <a href="#-citation"><b>📌 Citation</b></a>
</p>

</div>

---

Reference implementation for **DecoR (Decomposition-based Routing)** — a model
routing framework that recasts routing as a *matching* problem: instead of
learning a direct query→model mapping (which over-fits surface semantics and
collapses on OOD data), DecoR decomposes each query into a **capability
profile**, sifts similar historical query–response logs, and picks the model
that best balances performance and cost on those matched logs.

> Paper: *Beyond Query Memorization: Large Language Model Routing with Query
> Decomposition and Historical Matching.* ACL 2026 (Long Papers).
> [aclanthology.org/2026.acl-long.1852](https://aclanthology.org/2026.acl-long.1852/)

The whole pipeline is **training-light** and **model-pool agnostic**: updating
the candidate LLM pool only requires refreshing the historical log corpus, not
retraining the router.

---

## 🛠 Pipeline overview

```
            ┌─────────────────────────────────────────────────────────────┐
 user query │  Stage 1: Query Deconstruction                              │
   q  ──────▶  fdec(q) → capability profile p = {S, K, D} (+ reasons)     │  (SFT'd Qwen3)
            └─────────────────────────────────────────────────────────────┘
                                    │ p
                                    ▼
            ┌─────────────────────────────────────────────────────────────┐
            │  Stage 2: Hierarchical Log-Sifting                          │
            │  A. Character-level sifting  (inverted index, Jaccard on    │
            │     S/K + difficulty weight w; threshold τ → OOD if empty)  │
            │  B. Fine-ranking            (BGE-M3 cosine on q ⊕ p, Top-k) │
            │  C. Log-Alignment Evaluation (Log Evaluator LLM → set V of  │  (GRPO'd Qwen3)
            │     representative logs; V = ∅ ⇒ treat as OOD)              │
            └─────────────────────────────────────────────────────────────┘
                                    │ V (representative logs)
                                    ▼
            ┌─────────────────────────────────────────────────────────────┐
            │  Stage 3: Empirical Decision                                │
            │  Aggregate avg perf V̄ⱼ and cost C̄ⱼ over V for each model,  │  (rule-based)
            │  normalize, U = λ·Vⱼⁿᵒʳᵐ + (1-λ)·Cⱼⁿᵒʳᵐ, pick argmaxⱼ Uⱼ.  │
            │  OOD fallback ⇒ a high-performance default model.           │
            └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          selected model m*
```

---

## 📁 Repository layout

```
decor_code/
├── README.md
├── requirements.txt
│
├── stage1_query_deconstruction/         # Stage 1: q → {S,K,D} capability profile
│   ├── gen_sft_data_gpt.py              #   generate SFT data via a frontier model (GPT-5)
│   ├── qianfan_client.py                #   OpenAI/Qianfan client + decomposition prompt
│   ├── run_deconstruction_inference.py  #   batch-decompose queries with the trained deconstructor
│   ├── build_capability_dataset.py      #   build capability-stratified SFT dataset
│   └── eval_accuracy_stats.py           #   per-capability accuracy statistics
│
├── stage2_hierarchical_sifting/         # Stage 2: A/B retrieval + C log evaluator
│   ├── retrieval/                       #   Substage A (coarse) + B (fine-rank)
│   │   ├── query_retrieval.py           #     core retriever (inverted index, Jaccard,
│   │   │                                #     BGE-M3/API embedding, FAISS; coarse/fine/hybrid)
│   │   ├── run_retrieval.py             #     CLI entry (build index / single / batch retrieval)
│   │   ├── run_retrieval.sh             #     example end-to-end retrieval command
│   │   ├── similarity_judge.py          #     LLM "similarity judge" → Substage C training labels
│   │   ├── build_similarity_dataset*.py #     turn retrieval results into judge/LE datasets
│   │   ├── calculate_model_stats.py     #     per-model perf/cost stats over the corpus
│   │   ├── filter_zero_performance_cases.py
│   │   ├── update_synonym.py            #     auto-maintain the S/K synonym table
│   │   ├── synonym_dict.json
│   │   └── read_cache.py
│   └── log_evaluator/                   #   Substage C: the Log Evaluator (LE)
│       ├── train/                       #     GRPO training (VERL)
│       │   ├── reward_router.py         #       reward R(V,G) — Eq. (13) in the paper
│       │   ├── process_data_for_parquet.py  #   build VERL train/test parquet from judge data
│       │   └── train_grpo.sh            #       GRPO launch script
│       └── predict/                     #     LE inference (Substage C at test time)
│           ├── predict_log_evaluator.py #       query a served LE, output valid_representatives
│           ├── vllm_test.py             #       OpenAI-compatible client wrapper
│           └── run_predict.sh
│
├── stage3_empirical_decision/           # Stage 3: rule-based model selection
│   ├── score.py                         #   full evaluator: utility, OOD fallback, filtering, xlsx
│   ├── model_selection_by_similarity.py #   minimal reference implementation of the rule
│   ├── report_writer.py                 #   pivot-table xlsx writer
│   └── run_decision.sh                  #   example end-to-end decision command
│
└── data/
    ├── corpus/                          # Historical Response Log library H
    │   ├── corpus_with_perf_cost.jsonl  #   q + decomposition + per-model perf & cost  (Stage 3)
    │   ├── corpus_decomposition_for_retrieval.jsonl  # q + decomposition only          (Stage 2 index)
    │   └── model_statistics.json
    ├── stage1_deconstruction/
    │   ├── SFT_decomposition_train.jsonl          # instruction-tuning data for the deconstructor
    │   └── merged_query_decomposition_test.jsonl  # decomposed test queries
    ├── stage2_log_evaluator/
    │   ├── train.parquet                # GRPO train split (VERL format)
    │   └── test.parquet                 # GRPO val split
    └── stage3_test/
        ├── Test_Data_Top3.jsonl         # per-query Top-3 retrieved logs (LE input)
        ├── Test_Data_Top3.result.jsonl  # LE output: valid_representatives per query
        ├── Label_Test_Data.jsonl        # ground-truth per-model perf & cost on the test queries
        └── filter_test_v1_rl.jsonl      # evaluation id filter
```

---

## 🧩 Capability profile

The deconstructor maps a query to `C(q) = {S, K, D}` (plus optional format
constraints `F`):

- **S — Skill Set**: atomic functional operations (e.g. `mathematics`, `coding`,
  `summarization`) + `S_reason`.
- **K — Knowledge Domain**: domain expertise required (e.g. `medicine`, `law`)
  + `K_reason`.
- **D — Difficulty**: one of `D0` (trivial) … `D3` (hard) + `D_reason`.

Categories are **not** predefined — the deconstructor generates them per query.

---

## 🚀 End-to-end usage

### 0. Install

```bash
pip install -r requirements.txt
# For Log Evaluator training, install VERL separately: https://github.com/volcengine/verl
# For local serving, install vLLM: pip install vllm
```

Set API credentials via environment variables (no keys are hard-coded):

```bash
export OPENAI_API_KEY=...        # used by gen_sft_data_gpt.py / similarity_judge.py
export OPENAI_BASE_URL=...        # optional, OpenAI-compatible endpoint
export QIANFAN_API_KEY=...        # only if you serve the deconstructor on Qianfan
export DEEPINFRA_API_KEY=...      # only if using --use_api_embedding for retrieval
```

### Stage 1 — Query Deconstruction

Train a small base model (paper: **Qwen3-0.6B**, SFT, lr `2e-5`) to produce the
capability profile. Data is synthesized by a frontier model and expert-reviewed.

```bash
cd stage1_query_deconstruction

# (a) (optional) regenerate SFT data with a frontier model
python gen_sft_data_gpt.py

# (b) SFT: train the deconstructor on data/stage1_deconstruction/SFT_decomposition_train.jsonl
#     with your preferred SFT framework (e.g. LLaMA-Factory / TRL). The dataset is in
#     {"instruction", "input", "output"} format ready for instruction tuning.

# (c) batch-decompose queries with the trained deconstructor (served via OpenAI/Qianfan API)
python run_deconstruction_inference.py \
    --input  ../data/corpus/corpus_decomposition_for_retrieval.jsonl \
    --output ../data/stage1_deconstruction/decomposition_result/out.jsonl \
    --model  <your_served_deconstructor> --workers 8
```

### Stage 2 — Hierarchical Log-Sifting

**Substages A + B (retrieval):**

```bash
cd stage2_hierarchical_sifting/retrieval
bash run_retrieval.sh        # builds candidate Top-k logs per query
# Modes: --mode coarse|fine|hybrid ; tune --coarse_threshold (τ), --top_k (k),
# --alpha/--beta (structured vs. semantic weight). See README in run_retrieval.py docstring.
```

**Substage C (Log Evaluator):**

```bash
# 1) Build GRPO training data from similarity-judge labels:
cd retrieval
python similarity_judge.py            # LLM judges which retrieved logs represent the query
python build_similarity_dataset.py    # format into judge/LE dataset

cd ../log_evaluator/train
python process_data_for_parquet.py \
    --train_data_source <train_rl.jsonl> \
    --test_data_source  <test_rl.jsonl> \
    --output_dir ../../../data/stage2_log_evaluator

# 2) GRPO-train the Log Evaluator (paper: Qwen3-0.6B, VERL, lr 1e-6).
#    Reward R(V,G) is implemented in reward_router.py (Eq. 13).
bash train_grpo.sh

# 3) Serve the trained LE and run Substage C inference:
#    vllm serve <merge_model> --port 8080 --served-model-name log_evaluator
cd ../predict
bash run_predict.sh                    # → Test_Data_Top3.result.jsonl (valid_representatives)
```

### Stage 3 — Empirical Decision

Pure rule-based selection — **no training**. Aggregates historical performance &
cost of each candidate model over the selected logs `V`, normalizes both to
`[0,1]`, and picks `argmaxⱼ Uⱼ` with `Uⱼ = λ·Vⱼⁿᵒʳᵐ + (1-λ)·Cⱼⁿᵒʳᵐ`. Queries
flagged OOD (empty `V`) fall back to a high-performance default model.

```bash
cd stage3_empirical_decision
bash run_decision.sh
# Key args (see score.py):
#   --alpha / --beta        performance vs. cost weighting (λ)
#   --default_model         OOD fallback model
#   --enable_coverage_fallback / --coverage_min_models   robustness fallback
```

---

## 🤖 Candidate LLM pool

The shipped corpus / test data score these 8 models (labels `A`–`H` used in code):

| Label | Model |
|-------|-------|
| A | deepseek-ai/DeepSeek-V3.1-Terminus |
| B | deepseek-ai/DeepSeek-V3.2-Exp |
| C | google/gemma-3-12b-it |
| D | google/gemma-3-27b-it |
| E | mistralai/Mistral-Small-3.2-24B-Instruct-2506 |
| F | moonshotai/Kimi-K2-Instruct-0905 |
| G | openai/gpt-oss-120b |
| H | Qwen/Qwen3-235B-A22B-Instruct-2507 |

To swap the pool, regenerate the corpus with new models' perf/cost — **no router
retraining needed**.

---

## 📝 Notes

- All hard-coded API keys, tokens and cluster-absolute paths from the original
  research scripts have been removed; configure them via env vars / CLI args.
- The `data/` directory ships representative datasets for in-distribution (ID)
  evaluation. Large intermediate artifacts (full base-model outputs, OOD splits,
  pickled caches) are omitted to keep the repo lightweight; the scripts above
  regenerate them.
- Default hyper-parameters used in the paper: `τ = 0.5`, `k = 3`, `λ = 0.5`.

---

## 📌 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lv-etal-2026-beyond,
    title = "Beyond Query Memorization: Large Language Model Routing with Query Decomposition and Historical Matching",
    author = "Lv, Bo  and
      Sun, Jingbo  and
      Lv, Jianwei  and
      Tang, Chen  and
      Zhang, Shaojie  and
      Liu, Nayu  and
      Yu, Guoxin  and
      Li, Zihao  and
      Zhang, Qichao  and
      Zhao, Dongbin  and
      Luo, Ping  and
      Yu, Yue",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1852/",
    pages = "39876--39892",
    ISBN = "979-8-89176-390-6",
}
```
