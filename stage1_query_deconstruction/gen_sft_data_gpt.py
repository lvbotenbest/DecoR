
import os
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# --- CAPABILITY DECOMPOSITION PROMPT ---
# capability_decomposition_prompt.py

CAPABILITY_DECOMPOSITION_PROMPT = """You are a Capability Decomposition Engine. Your task is to decompose the user query into its capability-space representation C(q) = {S, K, D, F}. Follow all rules strictly and output JSON only.

---

1. Skill Set (S):
Identify the skills required to answer the query.
- You may freely generate categories.
- Examples: reasoning, logical inference, mathematics, coding, translation, writing, information extraction, multi-step planning, role-playing, style imitation, summarization.
- Output as a list.
Also output "S_reason": one sentence explaining why these skills are required.

2. Knowledge Domain (K):
Identify the knowledge domains needed for the query.
- Freely generated categories; no fixed list.
- Examples: general knowledge, medicine, law, finance, computer science, physics, ACG, history, philosophy.
- If no specific knowledge is required, output "none".
- Output as a list.
Also output "K_reason": one sentence explaining why these domains are needed.

3. Difficulty / Instruction Complexity (D):
Choose exactly one:
- D0 (Trivial): almost no reasoning; direct/simple request.
- D1 (Simple): mild understanding; single goal; light reasoning.
- D2 (Moderate): multiple requirements or multi-step tasks; needs organization/judgment.
- D3 (Hard): complex tasks requiring deep reasoning, planning, abstraction, or structured logic.
Also output "D_reason": one sentence explaining why this difficulty level matches the query.

4. Format Constraints (F):
List all explicit formatting or stylistic constraints from the query.
Examples: JSON, Markdown, tables, no-yapping, strict step-by-step, code formatting, templates.
If none exist, output "none".
- Output as a list.

---

IMPORTANT RULES:
- Output MUST be valid pure JSON.
- Do NOT include markdown code fences such as ```json or ``` anywhere.
- Do NOT infer formatting constraints from the demonstration examples.
- Only include format constraints (F) if the *user query explicitly requests* a format. Otherwise use ["none"].

Output Format (STRICT):

Return ONLY the following JSON structure:

{
  "S": [...],
  "S_reason": "...",
  "K": [...],
  "K_reason": "...",
  "D": "D0 or D1 or D2 or D3",
  "D_reason": "...",
  "F": [...],
}

No explanations. No extra text. Only valid JSON."""

# --- Configuration ---
INPUT_FILE = "ResultData/All_Merged_Data_with_ids.jsonl"
OUTPUT_DIR = "SFTData"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "query_decomposition_reason.jsonl")
MODEL_NAME = "gpt-4.1-mini"
CONCURRENCY = 200  # Adjust as needed

MODEL_PRICING = {
    "gpt-4.1-mini": {"in": 0.400, "out": 1.6},
    # 在这里添加更多模型...
}


# --- OpenAI Client Setup ---
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def get_decomposition(prompt):
    """
    Sends the prompt to the model with the capability decomposition system prompt.
    Returns the partial record content (decomposition) or None if failed.
    """
    prices = MODEL_PRICING.get(MODEL_NAME, {"in": 0, "out": 0})

    full_prompt = f"{CAPABILITY_DECOMPOSITION_PROMPT}\n\nUser Query:\n{prompt}"

    try:
        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
        )
        content = chat_completion.choices[0].message.content
        p_tokens = chat_completion.usage.prompt_tokens
        c_tokens = chat_completion.usage.completion_tokens
        
        # 计算价格 (假设价格单位是 Per Million)
        total_cost = (p_tokens * prices["in"] / 1000000) + (c_tokens * prices["out"] / 1000000)
        return content, total_cost

    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)[:100]}")
        return None

def safe_run_single(prompt):
    """重试逻辑：2s -> 30s -> Fail"""
    # 第一次重试
    try:
        return get_decomposition(prompt)
    except Exception as e:
        logger.warning(f"[Retry 2s] : {str(e)[:100]}")
        time.sleep(2)
        
    # 第二次重试
    try:
        return get_decomposition(prompt)
    except Exception as e:
        logger.warning(f"[Retry 10s]: {str(e)[:100]}")
        time.sleep(10)

    # 第三次重试
    try:
        return get_decomposition(prompt)
    except Exception as e:
        logger.error(f"[FAILED] : {str(e)[:100]}")
        return None, 0.0


def process_item(item):
    """
    Process a single item: extract prompt, call API, return result record.
    """
    item_id = item.get("id")
    prompt = item.get("prompt")
    
    if prompt is None:
        logger.warning(f"Item {item_id} missing 'prompt' field.")
        return None

    decomposition_result,cost = safe_run_single(prompt)
    
    if decomposition_result:
        return {
            "id": item_id,
            "prompt": prompt,
            "decomposition": decomposition_result,
            "cost": cost
        }
    return None

def generate_sft_data():
    """
    Main function to read input, process in parallel, and write output with checkpointing.
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for processed IDs to resume
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        logger.info(f"Checking existing output file: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "id" in record:
                        processed_ids.add(record["id"])
                except json.JSONDecodeError:
                    continue
        logger.info(f"Found {len(processed_ids)} processed records.")

    # Read input data
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    logger.info(f"Reading input file: {INPUT_FILE}")
    data_to_process = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("id") not in processed_ids:
                    data_to_process.append(item)
            except json.JSONDecodeError:
                continue
    
    total_items = len(data_to_process)
    logger.info(f"Total items to process: {total_items}")

    if total_items == 0:
        logger.info("No new items to process.")
        return

    # Process in batches
    batch_size = CONCURRENCY
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            for i in range(0, total_items, batch_size):
                batch = data_to_process[i : i + batch_size]
                futures = {executor.submit(process_item, item): item for item in batch}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        # Flush periodically or after each write to ensure safety
                        out_f.flush()
                
                logger.info(f"Processed batch {i // batch_size + 1}/{(total_items + batch_size - 1) // batch_size}")

    logger.info("Processing complete.")



if __name__ == "__main__":
    generate_sft_data()
