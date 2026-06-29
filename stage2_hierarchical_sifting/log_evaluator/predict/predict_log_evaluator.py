import argparse
import json
import os
import re
import concurrent.futures
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm
from vllm_test import chat_once


INPUT_PATH = "../../../data/stage3_test/Test_Data_Top3.jsonl"
OUTPUT_PATH = "../../../data/stage3_test/Test_Data_Top3.result.jsonl"
BASE_URL = os.getenv("VLLM_OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
API_KEY = os.getenv("VLLM_OPENAI_API_KEY", "EMPTY")
MODEL = os.getenv("LOG_EVALUATOR_MODEL", "/path/to/log_evaluator/merge_model")

BATCH_SIZE = 64

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 256



SIMILARITY_JUDGE_PROMPT = """You are a Query Similarity Judge. Your task is to determine which historical queries can represent the user's query in terms of capability requirements.

**Core Principle**: If a model can correctly answer a historical query, it should also be able to correctly answer the user's query. This means the historical query's capability requirements (Skills, Knowledge, Difficulty) should be similar to or greater than the user's query.

**Judgment Criteria**:
1. **Skills (S)**: The historical query's required skills should be SIMILAR to or cover the skills needed by the user's query. Note: Exact wording match is NOT required. Different terms describing similar skills (e.g., "arithmetic" vs "mathematics", "reasoning" vs "logical thinking") should be considered equivalent. Use your judgment to assess semantic similarity.
2. **Knowledge (K)**: The historical query's required knowledge domains should be SIMILAR to or cover the domains needed by the user's query. Again, exact match is NOT required - focus on whether the knowledge areas are conceptually similar or overlapping.
3. **Difficulty (D)**: The historical query's difficulty should be EQUAL TO or HIGHER than the user's query (D0 < D1 < D2 < D3).

**Important**: You should make your OWN judgment about whether the queries are truly similar in capability requirements. The provided decompositions may have some errors or use different terminology - use your understanding of both queries to make a fair assessment.

---

**User Query**:
{user_query}

**User Query Decomposition**:
- Skills (S): {user_skills}
- Skills Reason: {user_s_reason}
- Knowledge (K): {user_knowledge}
- Knowledge Reason: {user_k_reason}
- Difficulty (D): {user_difficulty}
- Difficulty Reason: {user_d_reason}

---

**Historical Queries**:

**[A]** {query_a}
- Skills (S): {skills_a}
- Knowledge (K): {knowledge_a}
- Difficulty (D): {difficulty_a}

**[B]** {query_b}
- Skills (S): {skills_b}
- Knowledge (K): {knowledge_b}
- Difficulty (D): {difficulty_b}

**[C]** {query_c}
- Skills (S): {skills_c}
- Knowledge (K): {knowledge_c}
- Difficulty (D): {difficulty_c}

---

**Instructions**:
Please think step by step before making your final judgment:

1. First, analyze the User Query's capability requirements (S, K, D).
2. Then, for EACH historical query (A, B, C), compare its capabilities with the User Query:
   - Does it cover all required Skills?
   - Does it cover all required Knowledge domains?
   - Is its Difficulty >= User Query's Difficulty?
3. Finally, based on your analysis, determine which historical queries can represent the User Query.

**Output Format (STRICT)**:
1. Output a short justification BETWEEN <think> and </think> (max 2–5 sentences, ≤150 words). No step-by-step reasoning.
2. Then output the final answer on a NEW line starting with "#### " followed by a JSON list of labels.

Examples:
- If ALL three historical queries can represent the user's query, output: #### ["A", "B", "C"]
- If only some can, output those letters in order, e.g.: #### ["A", "C"]
- If NONE can represent the user's query, output: #### []

IMPORTANT:
- Do NOT provide detailed step-by-step reasoning.
- The <think> content MUST be a concise justification only.
- The <think> content MUST NOT exceed 150 words.

No extra text outside the required format."""


def format_list(items):
    if isinstance(items, list):
        return ", ".join(str(item) for item in items)
    return str(items)


def build_prompt(entry: dict) -> str:
    user_decomp = entry.get("user_decomposition", {})
    retrieved = list(entry.get("retrieved", []) or [])

    while len(retrieved) < 3:
        retrieved.append({
            "query": "[No query available]",
            "decomposition": {"S": [], "K": [], "D": "N/A"}
        })

    def get_decomp(r, key, default=""):
        decomp = r.get("decomposition", {})
        return decomp.get(key, default)

    prompt = SIMILARITY_JUDGE_PROMPT.format(
        user_query=entry.get("user_query", ""),
        user_skills=format_list(user_decomp.get("S", [])),
        user_s_reason=user_decomp.get("S_reason", ""),
        user_knowledge=format_list(user_decomp.get("K", [])),
        user_k_reason=user_decomp.get("K_reason", ""),
        user_difficulty=user_decomp.get("D", ""),
        user_d_reason=user_decomp.get("D_reason", ""),
        query_a=retrieved[0].get("query", ""),
        skills_a=format_list(get_decomp(retrieved[0], "S", [])),
        knowledge_a=format_list(get_decomp(retrieved[0], "K", [])),
        difficulty_a=get_decomp(retrieved[0], "D", ""),
        query_b=retrieved[1].get("query", ""),
        skills_b=format_list(get_decomp(retrieved[1], "S", [])),
        knowledge_b=format_list(get_decomp(retrieved[1], "K", [])),
        difficulty_b=get_decomp(retrieved[1], "D", ""),
        query_c=retrieved[2].get("query", ""),
        skills_c=format_list(get_decomp(retrieved[2], "S", [])),
        knowledge_c=format_list(get_decomp(retrieved[2], "K", [])),
        difficulty_c=get_decomp(retrieved[2], "D", ""),
    )
    return prompt


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _extract_thinking(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        # Fallback: if there is no <think> tag, use everything before the last ####
        # as a compact justification.
        idx = text.rfind("####")
        if idx == -1:
            return ""
        return text[:idx].strip()
    return m.group(1).strip()


def _extract_valid_representatives(text: str) -> List[str]:
    if not text:
        return []
    idx = text.rfind("####")
    if idx == -1:
        return []
    tail = text[idx + 4 :].strip()
    m = re.search(r"\[[\s\S]*?\]", tail)
    if not m:
        return []
    list_str = m.group(0)
    try:
        parsed = json.loads(list_str)
    except Exception:
        try:
            parsed = json.loads(list_str.replace("'", '"'))
        except Exception:
            return []
    if not isinstance(parsed, list):
        return []
    out: List[str] = []
    for x in parsed:
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
    return out


def _example_id(example: Dict[str, Any], fallback_idx: int) -> Any:
    if "id" in example:
        return example.get("id")
    if "test_id" in example:
        return example.get("test_id")
    return fallback_idx


def _min_retrieved(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    retrieved = list(example.get("retrieved", []) or [])
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(retrieved):
        if not isinstance(r, dict):
            continue
        label = chr(ord("A") + i) if i < 26 else str(i)
        rid = None
        for k in ("id", "test_id", "sample_id", "query_id"):
            if k in r:
                rid = r.get(k)
                break
        if rid is None:
            rid = i
        out.append({"label": label, "id": rid, "query": r.get("query", "")})
    return out


def run_generation(
    *,
    input_path: str,
    output_path: str,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    limit: int = 0,
) -> None:
    examples = list(_read_jsonl(input_path))
    if limit and limit > 0:
        examples = examples[:limit]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    extra_create_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    with open(output_path, "w", encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(examples), batch_size)):
            batch = examples[start : start + batch_size]
            prompts = [build_prompt(ex) for ex in batch]

            def _call_one(prompt: str) -> Dict[str, Any]:
                return chat_once(
                    prompt,
                    model,
                    base_url=base_url,
                    api_key=api_key,
                    extra_create_kwargs=extra_create_kwargs,
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as ex:
                responses = list(ex.map(_call_one, prompts))

            for i, (exm, resp) in enumerate(zip(batch, responses)):
                text = resp.get("content") or ""
                result = {
                    "test_id": _example_id(exm, start + i),
                    "user_query": exm.get("user_query", ""),
                    "retrieved": _min_retrieved(exm),
                    "judgment": {
                        "thinking": _extract_thinking(text),
                        "valid_representatives": _extract_valid_representatives(text),
                    },
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=INPUT_PATH)
    p.add_argument("--output", default=OUTPUT_PATH)
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--api-key", default=API_KEY)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--top-p", type=float, default=TOP_P)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    run_generation(
        input_path=args.input,
        output_path=args.output,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )



