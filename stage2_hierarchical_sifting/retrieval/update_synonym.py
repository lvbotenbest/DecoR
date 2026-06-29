"""
同义词表自动更新脚本

功能：
1. 从知识库中提取所有 S (skills) 和 K (knowledge) 词语
2. 去重并统计词频
3. 通过大模型判断词之间是否为同义词
4. 自动更新同义词表 JSON 文件

使用方式：
python update_synonym.py --corpus SFTData/out_common_file/merged_query_decomposition_train.jsonl
"""

import json
import os
import sys
import argparse
from collections import Counter
from typing import List, Dict, Set, Tuple, Callable, Optional
from datetime import datetime
from difflib import SequenceMatcher
import re
from abc import ABC, abstractmethod

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 同义词表路径
SYNONYM_JSON_PATH = os.path.join(os.path.dirname(__file__), "synonym_dict.json")


# ============================================================================
# LLM Interface - 大模型调用接口
# ============================================================================

class LLMInterface(ABC):
    """
    大模型调用接口基类
    
    如需使用其他模型，继承此类并实现 call 方法
    """
    
    @abstractmethod
    def call(self, prompt: str) -> str:
        """
        调用大模型
        
        Args:
            prompt: 输入提示词
        
        Returns:
            模型返回的文本
        """
        pass


class QianfanLLM(LLMInterface):
    """
    千帆平台大模型
    
    支持两种调用方式：
    - "sample": 使用 req_qianfan_sample (推荐，更简洁)
    - "model": 使用 req_model (原始方式)
    """
    
    def __init__(self, model_name: str = "grt8bohz_test", call_method: str = "sample"):
        """
        Args:
            model_name: 模型名称
            call_method: 调用方式，"sample" 或 "model"
        """
        self.model_name = model_name
        self.call_method = call_method
        self._client = None

        self.headers = {
            'Content-Type': 'application/json',
            # Set your Qianfan API token via the QIANFAN_API_KEY environment variable.
            'Authorization': 'Bearer ' + os.getenv('QIANFAN_API_KEY', 'YOUR_QIANFAN_TOKEN')
        }
        self.url = "https://qianfan.baidubce.com/v2/chat/completions"
        # self.url = "https://qianfan.baidubce.com/v2/router/chat/completions"

    def req_model(self, model_name, content, is_system=False, is_thinking=False):
        """
        支持千帆平台模型，demo：
            ## ernie-x1-32k、ernie-4.5-8k-preview、ernie-4.0-8k
            ## deepseek-r1、deepseek-v3、deepseek-r1-250528
            ## qwq-32b、qwen3-30b-a3b（"enable_thinking": False）、qwen3-235b-a22b（"enable_thinking": False）
        """

        messages = [{"role": "user", "content": content}]
        if is_system:
            messages = content

        payload = json.dumps({
            "model": model_name,
            "stream": False,
            "messages": messages,
            "temperature": 0.01,
            "top_p": 0.8,
            "max_output_tokens": 1 * 1024,
            "penalty_score": 1.0
        }, ensure_ascii=False)

        if model_name in ("qwen3-30b-a3b", "qwen3-235b-a22b"):
            payload = json.dumps({
                "model": model_name,
                "stream": False,
                "messages": messages,
                "temperature": 0.8,
                "top_p": 0.8,
                "max_output_tokens": 1 * 1024,
                "penalty_score": 1.0,
                "enable_thinking": is_thinking
            }, ensure_ascii=False)

        response = requests.request("POST", self.url, headers=self.headers, data=payload.encode("utf-8"))
       
        return response.text
    
    def req_qianfan_sample(self, prompt, model_name):
        """
            单线程请求大模型（简化版，直接返回content）
        """
        try:
            answer = self.req_model(model_name=model_name, content=prompt)
            choices = json.loads(answer).get("choices", [{}])
            content = choices[0].get("message", {}).get("content", "ERR") if choices else "ERR"
            return content
        except Exception as e:  # 捕获异常并绑定到变量e
            traceback.print_exc()
            return "ERR"
    
    def call(self, prompt: str) -> str:
       
        if self.call_method == "sample":
            # 使用 req_qianfan_sample (推荐)
            content = self.req_qianfan_sample(prompt=prompt, model_name=self.model_name)
            return content.strip() if content else ""
        else:
            # 使用 req_model (原始方式)
            result = self.req_model(model_name=self.model_name, content=prompt)
            response = json.loads(result)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip()


class OpenAILLM(LLMInterface):
    """OpenAI 兼容接口（可用于本地部署的模型）"""
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        base_url: str = None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    
    def call(self, prompt: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return ""


class CustomLLM(LLMInterface):
    """
    自定义LLM接口
    
    使用方式：
    ```python
    def my_llm_func(prompt: str) -> str:
        # 你的模型调用逻辑
        return response
    
    llm = CustomLLM(my_llm_func)
    ```
    """
    
    def __init__(self, call_func: Callable[[str], str]):
        self.call_func = call_func
    
    def call(self, prompt: str) -> str:
        return self.call_func(prompt)


# 全局 LLM 实例（可在运行时替换）
_LLM_INSTANCE: Optional[LLMInterface] = None


def set_llm(llm: LLMInterface):
    """设置全局 LLM 实例"""
    global _LLM_INSTANCE
    _LLM_INSTANCE = llm


def get_llm(model_name: str = "grt8bohz_test", call_method: str = "sample") -> LLMInterface:
    """
    获取 LLM 实例
    
    Args:
        model_name: 模型名称
        call_method: 调用方式，"sample" 或 "model"
    """
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = QianfanLLM(model_name=model_name, call_method=call_method)
    return _LLM_INSTANCE


def load_synonym_dict(path: str = SYNONYM_JSON_PATH) -> Dict:
    """加载同义词表"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"synonym_groups": {}}


def save_synonym_dict(data: Dict, path: str = SYNONYM_JSON_PATH):
    """保存同义词表"""
    data["_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Synonym dictionary saved to: {path}")


def extract_terms_from_corpus(corpus_path: str) -> Tuple[Counter, Counter]:
    """
    从知识库中提取所有 S 和 K 词语
    
    Returns:
        (skill_counter, knowledge_counter)
    """
    skill_counter = Counter()
    knowledge_counter = Counter()
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                decomposition = data.get('decomposition', {})
                
                # 如果是字符串，解析
                if isinstance(decomposition, str):
                    try:
                        decomposition = json.loads(decomposition)
                    except json.JSONDecodeError:
                        continue
                
                # 提取 skills
                skills = decomposition.get('S', [])
                for skill in skills:
                    skill_counter[skill.lower().strip()] += 1
                
                # 提取 knowledge
                knowledge = decomposition.get('K', [])
                for k in knowledge:
                    knowledge_counter[k.lower().strip()] += 1
                    
            except json.JSONDecodeError:
                continue
    
    return skill_counter, knowledge_counter


def string_similarity(s1: str, s2: str) -> float:
    """计算字符串相似度（用于粗筛）"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def normalize_term(term: str) -> str:
    """标准化词语"""
    # 转小写，去除多余空格
    term = term.lower().strip()
    # 去除标点
    term = re.sub(r'[^\w\s]', ' ', term)
    # 合并多个空格
    term = re.sub(r'\s+', ' ', term).strip()
    return term


def find_potential_synonyms(terms: List[str], similarity_threshold: float = 0.6) -> List[Tuple[str, str, float]]:
    """
    找出可能是同义词的词对（基于字符串相似度粗筛）
    
    Args:
        terms: 词语列表
        similarity_threshold: 相似度阈值
    
    Returns:
        List of (term1, term2, similarity)
    """
    potential_pairs = []
    terms = list(set(terms))  # 去重
    n = len(terms)
    
    print(f"Checking {n} terms for potential synonyms...")
    
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = terms[i], terms[j]
            
            # 跳过完全相同的
            if normalize_term(t1) == normalize_term(t2):
                continue
            
            # 计算相似度
            sim = string_similarity(t1, t2)
            
            # 额外检查：一个是另一个的子串
            if t1 in t2 or t2 in t1:
                sim = max(sim, 0.7)
            
            # 检查词根相似（如 mathematics/mathematical）
            if t1.rstrip('s') == t2.rstrip('s'):
                sim = max(sim, 0.9)
            if t1.rstrip('al') == t2 or t2.rstrip('al') == t1:
                sim = max(sim, 0.85)
            if t1.rstrip('ing') == t2 or t2.rstrip('ing') == t1:
                sim = max(sim, 0.85)
            
            if sim >= similarity_threshold:
                potential_pairs.append((t1, t2, sim))
    
    # 按相似度排序
    potential_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Found {len(potential_pairs)} potential synonym pairs")
    return potential_pairs


# Prompt template for synonym checking (English)
SYNONYM_CHECK_PROMPT = """Determine whether the following two terms are synonyms or near-synonyms in the context of "describing AI/ML task capabilities and skills".

Term 1: {term1}
Term 2: {term2}

Rules:
- Only consider meanings when describing AI task capabilities, skills, or knowledge domains.
- If the two terms can be used interchangeably with essentially the same meaning, answer YES.
- If the two terms have clearly different meanings, answer NO.

Answer only YES or NO, no explanation needed."""


def call_llm_for_synonym_check(
    term1: str, 
    term2: str, 
    model_name: str = "ernie-4.0-8k",
    llm: LLMInterface = None
) -> bool:
    """
    调用大模型判断两个词是否为同义词
    
    Args:
        term1: 词语1
        term2: 词语2
        model_name: 模型名称（当 llm 为 None 时使用）
        llm: LLM 接口实例（可选，传入则忽略 model_name）
    
    Returns:
        是否为同义词
    """
    try:
        # 使用传入的 LLM 或获取全局实例
        if llm is None:
            llm = get_llm(model_name)
        
        # 构建 prompt
        prompt = SYNONYM_CHECK_PROMPT.format(term1=term1, term2=term2)
        
        # 调用模型
        response = llm.call(prompt)
        
        return "YES" in response.upper()
        
    except Exception as e:
        print(f"LLM call failed: {e}")
        return False


def batch_check_synonyms_with_llm(
    pairs: List[Tuple[str, str, float]], 
    model_name: str = "ernie-4.0-8k",
    max_pairs: int = 100,
    llm: LLMInterface = None
) -> List[Tuple[str, str]]:
    """
    批量调用大模型检查同义词
    
    Args:
        pairs: 潜在同义词对列表
        model_name: 模型名称（当 llm 为 None 时使用）
        max_pairs: 最大检查对数
        llm: LLM 接口实例（可选）
    
    Returns:
        确认的同义词对列表
    """
    confirmed_pairs = []
    pairs_to_check = pairs[:max_pairs]
    
    # 获取 LLM 实例
    if llm is None:
        llm = get_llm(model_name)
    
    print(f"\nStarting LLM synonym check for {len(pairs_to_check)} pairs...")
    
    for i, (t1, t2, sim) in enumerate(pairs_to_check):
        print(f"[{i+1}/{len(pairs_to_check)}] Checking: {t1} ↔ {t2} (similarity: {sim:.2f})")
        
        is_synonym = call_llm_for_synonym_check(t1, t2, llm=llm)
        
        if is_synonym:
            print(f"  → Confirmed synonym ✓")
            confirmed_pairs.append((t1, t2))
        else:
            print(f"  → Not synonym ✗")
    
    print(f"\nConfirmed {len(confirmed_pairs)} synonym pairs")
    return confirmed_pairs


def merge_synonyms_to_dict(
    existing_dict: Dict,
    new_pairs: List[Tuple[str, str]],
    corpus_path: str
) -> Dict:
    """
    将新的同义词对合并到现有词典
    
    策略：
    1. 如果两个词都不在现有词典中，创建新组，词频高的作为标准词
    2. 如果一个在、一个不在，将不在的加入到在的组
    3. 如果都在但在不同组，合并两组
    """
    # 加载词频
    skill_counter, knowledge_counter = extract_terms_from_corpus(corpus_path)
    all_counter = skill_counter + knowledge_counter

    print(f"all_counter: {all_counter}")
    
    synonym_groups = existing_dict.get("synonym_groups", {})
    
    # 构建反向索引
    term_to_standard = {}
    for standard, synonyms in synonym_groups.items():
        term_to_standard[standard.lower()] = standard.lower()
        for syn in synonyms:
            term_to_standard[syn.lower()] = standard.lower()
    
    # 处理新同义词对
    for t1, t2 in new_pairs:
        t1_lower, t2_lower = t1.lower(), t2.lower()
        
        std1 = term_to_standard.get(t1_lower)
        std2 = term_to_standard.get(t2_lower)
        
        if std1 is None and std2 is None:
            # 都不在词典中，创建新组
            # 词频高的作为标准词
            if all_counter[t1_lower] >= all_counter[t2_lower]:
                standard, synonym = t1_lower, t2_lower
            else:
                standard, synonym = t2_lower, t1_lower
            
            synonym_groups[standard] = [synonym]
            term_to_standard[standard] = standard
            term_to_standard[synonym] = standard
            print(f"Created new synonym group: {standard} → [{synonym}]")
            
        elif std1 is not None and std2 is None:
            # t1在词典，t2不在
            if t2_lower not in synonym_groups.get(std1, []):
                if std1 not in synonym_groups:
                    synonym_groups[std1] = []
                synonym_groups[std1].append(t2_lower)
                term_to_standard[t2_lower] = std1
                print(f"Added to existing group: {std1} ← {t2_lower}")
                
        elif std1 is None and std2 is not None:
            # t2在词典，t1不在
            if t1_lower not in synonym_groups.get(std2, []):
                if std2 not in synonym_groups:
                    synonym_groups[std2] = []
                synonym_groups[std2].append(t1_lower)
                term_to_standard[t1_lower] = std2
                print(f"Added to existing group: {std2} ← {t1_lower}")
                
        elif std1 != std2:
            # 都在词典但在不同组，合并
            # 词频高的组保留为标准
            if all_counter[std1] >= all_counter[std2]:
                primary, secondary = std1, std2
            else:
                primary, secondary = std2, std1
            
            # 将 secondary 组合并到 primary
            if primary not in synonym_groups:
                synonym_groups[primary] = []
            
            # 添加 secondary 作为同义词
            if secondary not in synonym_groups[primary]:
                synonym_groups[primary].append(secondary)
            
            # 合并 secondary 的同义词
            for syn in synonym_groups.get(secondary, []):
                if syn not in synonym_groups[primary] and syn != primary:
                    synonym_groups[primary].append(syn)
                term_to_standard[syn] = primary
            
            term_to_standard[secondary] = primary
            
            # 删除 secondary 组
            if secondary in synonym_groups:
                del synonym_groups[secondary]
            
            print(f"Merged synonym groups: {primary} ← {secondary}")
    
    existing_dict["synonym_groups"] = synonym_groups
    existing_dict["_source_corpus"] = corpus_path
    
    return existing_dict


def manual_review_mode(pairs: List[Tuple[str, str, float]]) -> List[Tuple[str, str]]:
    """
    Manual review mode: display potential synonym pairs one by one for user confirmation
    """
    confirmed_pairs = []
    
    print("\n" + "=" * 60)
    print("Manual Review Mode")
    print("Enter: y (confirm), n (skip), q (quit)")
    print("=" * 60)
    
    for i, (t1, t2, sim) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {t1} ↔ {t2}")
        print(f"Similarity: {sim:.2f}")
        
        while True:
            choice = input("Is this a synonym pair? (y/n/q): ").strip().lower()
            if choice == 'y':
                confirmed_pairs.append((t1, t2))
                print("  → Confirmed ✓")
                break
            elif choice == 'n':
                print("  → Skipped")
                break
            elif choice == 'q':
                print("Exiting review")
                return confirmed_pairs
            else:
                print("Invalid input, please enter y/n/q")
    
    return confirmed_pairs


def parse_args():
    parser = argparse.ArgumentParser(description='Synonym Dictionary Auto-Update Tool')
    
    parser.add_argument('--corpus', '-c', type=str, required=True,
                        help='Corpus JSONL file path')
    parser.add_argument('--output', '-o', type=str, default="synonym_dict.json",
                        help='Output synonym dictionary path')
    parser.add_argument('--similarity', '-s', type=float, default=0.6,
                        help='String similarity threshold (default: 0.6)')
    parser.add_argument('--max_pairs', '-m', type=int, default=1000,
                        help='Maximum synonym pairs to check (default: 100)')
    parser.add_argument('--model', type=str, default='grt8bohz_test',
                        help='LLM model name (default: grt8bohz_test)')
    parser.add_argument('--call_method', type=str, default='sample',
                        choices=['sample', 'model'],
                        help='Qianfan API call method: "sample" (req_qianfan_sample) or "model" (req_model) (default: sample)')
    parser.add_argument('--manual', action='store_true',
                        help='Manual review mode (no LLM calls)')
    parser.add_argument('--stats_only', action='store_true',
                        help='Only show term statistics, do not update dictionary')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check corpus file
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file not found: {args.corpus}")
        sys.exit(1)
    
    print("=" * 60)
    print("Synonym Dictionary Update Tool")
    print("=" * 60)
    print(f"Corpus: {args.corpus}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    
    # Step 1: Extract terms
    print("\n[Step 1] Extracting terms from corpus...")
    skill_counter, knowledge_counter = extract_terms_from_corpus(args.corpus)
    
    print(f"\nSkills Statistics ({len(skill_counter)} unique terms):")
    for term, count in skill_counter.most_common(20):
        print(f"  {term}: {count}")
    
    print(f"\nKnowledge Statistics ({len(knowledge_counter)} unique terms):")
    for term, count in knowledge_counter.most_common(20):
        print(f"  {term}: {count}")
    
    if args.stats_only:
        print("\nStats-only mode, exiting")
        return
    
    # Step 2: Find potential synonyms
    print("\n[Step 2] Finding potential synonyms...")
    all_terms = list(skill_counter.keys()) + list(knowledge_counter.keys())
    all_terms = list(set(all_terms))  # deduplicate
    
    potential_pairs = find_potential_synonyms(all_terms, args.similarity)
    
    if not potential_pairs:
        print("No potential synonym pairs found")
        return
    
    # Display top 20 potential synonym pairs
    print("\nPotential synonym pairs (top 20):")
    for t1, t2, sim in potential_pairs[:20]:
        print(f"  {t1} ↔ {t2} (similarity: {sim:.2f})")
    
    # Step 3: Confirm synonyms
    print("\n[Step 3] Confirming synonyms...")
    
    # Initialize LLM with specified model and call method
    print(f"Using call method: {args.call_method}")
    set_llm(QianfanLLM(model_name=args.model, call_method=args.call_method))
    
    if args.manual:
        # Manual review mode
        confirmed_pairs = manual_review_mode(potential_pairs[:args.max_pairs])
    else:
        # LLM-assisted mode
        confirmed_pairs = batch_check_synonyms_with_llm(
            potential_pairs, args.model, args.max_pairs
        )
    
    if not confirmed_pairs:
        print("No synonym pairs confirmed")
        return
    
    # Step 4: Update synonym dictionary
    print("\n[Step 4] Updating synonym dictionary...")
    existing_dict = load_synonym_dict(args.output)
    updated_dict = merge_synonyms_to_dict(existing_dict, confirmed_pairs, args.corpus)
    save_synonym_dict(updated_dict, args.output)
    
    print("\nUpdate complete!")
    print(f"Total synonym groups: {len(updated_dict.get('synonym_groups', {}))}")


if __name__ == '__main__':
    main()

