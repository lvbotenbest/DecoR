"""
相似性判断脚本

调用大模型判断历史query是否能代表用户query：
- 能答对历史query，是否一定能答对用户query
- 输出符合条件的历史query编号（A/B/C）或 None
"""
import os
import json
import time
import logging
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# --- 相似性判断 Prompt ---
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
Return a JSON object with the following structure:

{{
  "thinking": "Brief justification (max 2–5 sentences, ≤150 words). No step-by-step reasoning.",
  "valid_representatives": ["A", "B", "C"]
}}

IMPORTANT:
- Do NOT provide detailed step-by-step reasoning.
- The "analysis" field MUST be a concise justification only.
- The "analysis" field MUST NOT exceed 150 words.

- If ALL three historical queries can represent the user's query, output: {{"valid_representatives": ["A", "B", "C"], ...}}
- If only some can, output those letters in order, e.g.: {{"valid_representatives": ["A", "C"], ...}}
- If NONE can represent the user's query, output: {{"valid_representatives": [], ...}}

No extra text outside the JSON. Only valid JSON."""


# --- Default Configuration ---
DEFAULT_INPUT_FILE = os.path.join(os.path.dirname(__file__), "Dataset/output/TestData/V1/retrieval_results_test_with_it_self.filtered_top3.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Dataset/output/similarity_dataset/llm_judge_v2/test")
DEFAULT_OUTPUT_FILE_NAME = "similarity_judge_results_test_v1.jsonl"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_CONCURRENCY = 100

# 全局模型名称（由 main 函数设置）
MODEL_NAME = DEFAULT_MODEL

MODEL_PRICING = {
    "gpt-4.1-mini": {"in": 0.400, "out": 1.6},
}

# --- OpenAI Client Setup ---
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="相似性判断脚本 - 调用大模型判断历史query是否能代表用户query")
    
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_FILE,
                        help=f'输入文件路径 (默认: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output_dir', '-o', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL,
                        help=f'模型名称 (默认: {DEFAULT_MODEL})')
    parser.add_argument('--concurrency', '-c', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'并发数 (默认: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='限制处理数量 (默认: 无限制)')
    
    return parser.parse_args()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def format_list(items):
    """格式化列表为字符串"""
    if isinstance(items, list):
        return ", ".join(str(item) for item in items)
    return str(items)


def build_prompt(entry: dict) -> str:
    """构建判断 prompt"""
    user_decomp = entry.get('user_decomposition', {})
    retrieved = entry.get('retrieved', [])
    
    # 确保有3个历史query
    while len(retrieved) < 3:
        retrieved.append({
            'query': '[No query available]',
            'decomposition': {'S': [], 'K': [], 'D': 'N/A'}
        })
    
    # 提取历史query信息
    def get_decomp(r, key, default=''):
        decomp = r.get('decomposition', {})
        return decomp.get(key, default)
    
    prompt = SIMILARITY_JUDGE_PROMPT.format(
        user_query=entry.get('user_query', ''),
        user_skills=format_list(user_decomp.get('S', [])),
        user_s_reason=user_decomp.get('S_reason', ''),
        user_knowledge=format_list(user_decomp.get('K', [])),
        user_k_reason=user_decomp.get('K_reason', ''),
        user_difficulty=user_decomp.get('D', ''),
        user_d_reason=user_decomp.get('D_reason', ''),
        # Query A
        query_a=retrieved[0].get('query', ''),
        skills_a=format_list(get_decomp(retrieved[0], 'S', [])),
        knowledge_a=format_list(get_decomp(retrieved[0], 'K', [])),
        difficulty_a=get_decomp(retrieved[0], 'D', ''),
        # Query B
        query_b=retrieved[1].get('query', ''),
        skills_b=format_list(get_decomp(retrieved[1], 'S', [])),
        knowledge_b=format_list(get_decomp(retrieved[1], 'K', [])),
        difficulty_b=get_decomp(retrieved[1], 'D', ''),
        # Query C
        query_c=retrieved[2].get('query', ''),
        skills_c=format_list(get_decomp(retrieved[2], 'S', [])),
        knowledge_c=format_list(get_decomp(retrieved[2], 'K', [])),
        difficulty_c=get_decomp(retrieved[2], 'D', ''),
    )
    
    # print(prompt)
    return prompt


def call_llm(prompt: str):
    """调用大模型"""
    prices = MODEL_PRICING.get(MODEL_NAME, {"in": 0, "out": 0})
    
    try:
        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        content = chat_completion.choices[0].message.content
        p_tokens = chat_completion.usage.prompt_tokens
        c_tokens = chat_completion.usage.completion_tokens
        
        total_cost = (p_tokens * prices["in"] / 1000000) + (c_tokens * prices["out"] / 1000000)
        return content, total_cost
    except Exception as e:
        logger.error(f"API Error: {str(e)[:100]}")
        raise


def safe_call_llm(prompt: str):
    """带重试的 API 调用"""
    # 第一次尝试
    try:
        return call_llm(prompt)
    except Exception as e:
        logger.warning(f"[Retry 2s]: {str(e)[:100]}")
        time.sleep(2)
    
    # 第二次重试
    try:
        return call_llm(prompt)
    except Exception as e:
        logger.warning(f"[Retry 10s]: {str(e)[:100]}")
        time.sleep(10)
    
    # 第三次重试
    try:
        return call_llm(prompt)
    except Exception as e:
        logger.error(f"[FAILED]: {str(e)[:100]}")
        return None, 0.0


def parse_result(content: str) -> dict:
    """解析模型输出"""
    try:
        # 清理可能的 markdown 代码块
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        result = json.loads(content)
        return result
    except json.JSONDecodeError:
        return {"valid_representatives": [], "reason": "Parse error", "raw": content}


def process_item(entry: dict):
    """处理单条数据"""
    test_id = entry.get('test_id')
    
    try:
        prompt = build_prompt(entry)
        response, cost = safe_call_llm(prompt)
        
        if response is None:
            return None
        
        parsed = parse_result(response)
        
        return {
            "test_id": test_id,
            "user_query": entry.get('user_query'),
            "user_decomposition": entry.get('user_decomposition'),
            "retrieved": [
                {
                    "label": chr(65 + i),  # A, B, C
                    "id": r.get('id'),
                    "query": r.get('query'),
                    "decomposition": r.get('decomposition')
                }
                for i, r in enumerate(entry.get('retrieved', [])[:3])
            ],
            "judgment": parsed,
            "cost": cost
        }
    except Exception as e:
        logger.error(f"Error processing test_id {test_id}: {str(e)[:100]}")
        return None


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置全局模型名称
    global MODEL_NAME
    MODEL_NAME = args.model
    
    input_file = args.input
    output_dir = args.output_dir
    output_file = os.path.join(output_dir, DEFAULT_OUTPUT_FILE_NAME)
    concurrency = args.concurrency
    limit = args.limit
    
    logger.info(f"Configuration:")
    logger.info(f"  Input: {input_file}")
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Concurrency: {concurrency}")
    if limit:
        logger.info(f"  Limit: {limit}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查已处理的 ID
    processed_ids = set()
    if os.path.exists(output_file):
        logger.info(f"Checking existing output: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "test_id" in record:
                        processed_ids.add(record["test_id"])
                except json.JSONDecodeError:
                    continue
        logger.info(f"Found {len(processed_ids)} processed records.")
    
    # 读取输入数据
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
    
    logger.info(f"Reading input file: {input_file}")
    data_to_process = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    if entry.get('test_id') not in processed_ids:
                        data_to_process.append(entry)
                except json.JSONDecodeError:
                    continue
    
    # 应用限制
    if limit and len(data_to_process) > limit:
        data_to_process = data_to_process[:limit]
    
    total_items = len(data_to_process)
    logger.info(f"Total items to process: {total_items}")
    
    if total_items == 0:
        logger.info("No new items to process.")
        return
    
    # 并行处理
    batch_size = concurrency
    total_cost = 0.0
    processed_count = 0
    
    with open(output_file, 'a', encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for i in range(0, total_items, batch_size):
                batch = data_to_process[i:i + batch_size]
                futures = {executor.submit(process_item, item): item for item in batch}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                        total_cost += result.get('cost', 0)
                        processed_count += 1
                
                logger.info(f"Batch {i // batch_size + 1}/{(total_items + batch_size - 1) // batch_size} | "
                           f"Processed: {processed_count} | Cost: ${total_cost:.4f}")
    
    logger.info(f"Processing complete. Total cost: ${total_cost:.4f}")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
