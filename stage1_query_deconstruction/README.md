# Stage 1 — Query Deconstruction

Transforms a raw query `q` into a structured **capability profile**
`C(q) = {S, K, D}` (+ reasons), decoupling linguistic surface form from
task-intrinsic requirements. Implemented as a small SFT'd base model
(paper: Qwen3-0.6B, lr 2e-5).

| File | Role |
|------|------|
| `gen_sft_data_gpt.py` | Generate SFT data by prompting a frontier model with the Capability Decomposition prompt. Reads merged base data, writes `{S,K,D,F}` decompositions. |
| `qianfan_client.py` | OpenAI/Qianfan client + the decomposition `prompt_template`. Token via `QIANFAN_API_KEY`. |
| `run_deconstruction_inference.py` | Multi-threaded batch decomposition with the trained deconstructor (served via API). |
| `build_capability_dataset.py` | Build a capability-stratified SFT dataset from merged decompositions. |
| `eval_accuracy_stats.py` | Per-capability (S/K/D) accuracy statistics. |

**Data:** `../data/stage1_deconstruction/SFT_decomposition_train.jsonl`
(`{instruction, input, output}` instruction-tuning format) and
`merged_query_decomposition_test.jsonl` (decomposed test queries).

The actual SFT training uses an external framework (e.g. LLaMA-Factory / TRL) —
the data is ready to plug in.
