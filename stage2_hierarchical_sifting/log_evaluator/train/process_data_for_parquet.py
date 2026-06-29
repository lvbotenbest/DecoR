import argparse
import json
import os
import re

import datasets

# from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    if isinstance(solution_str, dict):
        reps = solution_str.get("valid_representatives", [])
        if reps is None:
            return []
        if isinstance(reps, list):
            return reps
        return []

    if solution_str is None:
        return []

    if isinstance(solution_str, list):
        return solution_str

    if not isinstance(solution_str, str):
        try:
            solution_str = json.dumps(solution_str, ensure_ascii=False)
        except Exception:
            return []

    s = solution_str.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("valid_representatives"), list):
            return obj.get("valid_representatives")
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    m = re.search(r"\"valid_representatives\"\s*:\s*(\[[^\]]*\])", s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return []

    m = re.search(r"(\[[\s\"A-C\,]*\])\s*$", s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return []

    return []


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


def gen_verl_rows(path: str, split: str):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except Exception:
                continue

            question_raw = build_prompt(example)
            question = question_raw

            answer_raw = example.get("judgment", {})
            solution = extract_solution(answer_raw)

            try:
                answer_str = json.dumps(answer_raw, ensure_ascii=False)
            except Exception:
                answer_str = str(answer_raw)

            yield {
                "data_source": "similarity_judge",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "test_id": example.get("test_id"),
                    "answer": answer_str,
                    "question": question_raw,
                },
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_source",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "train", "train_rl.jsonl"),
    )
    parser.add_argument(
        "--test_data_source",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "test", "similarity_judge_results_v1_rl.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "rltrain", "v1"),
    )
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    train_dataset = datasets.Dataset.from_generator(
        gen_verl_rows,
        gen_kwargs={"path": args.train_data_source, "split": "train"},
    )
    test_dataset = datasets.Dataset.from_generator(
        gen_verl_rows,
        gen_kwargs={"path": args.test_data_source, "split": "test"},
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.output_dir, "test.parquet"))
