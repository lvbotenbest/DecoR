"""
构造相似性判断数据集

功能：
1. 正样本：从检索结果中挑选 top 3，过滤掉与 user_query 重复的
2. 负样本：随机选择 user_query，随机匹配 3 个不相关样本
"""
import json
import os
import random
from typing import List, Dict

# 输入输出路径
INPUT_PATH = os.path.join(os.path.dirname(__file__), "Dataset/output/IID/retrieval_results_train.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Dataset/output/similarity_dataset/sim_v2")


def load_data(input_path: str) -> List[Dict]:
    """加载检索结果数据"""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def filter_and_select_top3(entry: Dict) -> List[Dict]:
    """
    过滤重复并选择 top 3
    
    - 过滤掉 retrieved 中 id 与 test_id 相同的
    - 选择 top 3
    """
    test_id = entry.get('test_id')
    retrieved = entry.get('retrieved', [])
    
    # 过滤掉与 test_id 重复的
    filtered = [r for r in retrieved if r.get('id') != test_id]
    
    # 取 top 3（已经按分数排序）
    return filtered[:3]


def format_item_with_score(item: Dict) -> Dict:
    """格式化单条 retrieved，保留 score"""
    return {
        'id': item.get('id'),
        'query': item.get('query'),
        'decomposition': item.get('decomposition'),
        'scores': item.get('scores', item.get('score'))  # 兼容新旧格式
    }


def format_item_without_score(item: Dict) -> Dict:
    """格式化单条 retrieved，不保留 score"""
    return {
        'id': item.get('id'),
        'query': item.get('query'),
        'decomposition': item.get('decomposition')
    }


def build_positive_samples(data: List[Dict]) -> tuple:
    """
    构建正样本数据集
    
    Returns:
        (带分数的数据, 不带分数的数据)
    """
    samples_with_score = []
    samples_without_score = []
    
    for entry in data:
        # 过滤并选择 top 3
        top3 = filter_and_select_top3(entry)
        
        if not top3:  # 过滤后为空则跳过
            continue
        
        # 带分数版本
        sample_with_score = {
            'test_id': entry.get('test_id'),
            'user_query': entry.get('user_query'),
            'user_decomposition': entry.get('user_decomposition'),
            'label': 'positive',
            'retrieved': [format_item_with_score(r) for r in top3]
        }
        samples_with_score.append(sample_with_score)
        
        # 不带分数版本
        sample_without_score = {
            'test_id': entry.get('test_id'),
            'user_query': entry.get('user_query'),
            'user_decomposition': entry.get('user_decomposition'),
            'label': 'positive',
            'retrieved': [format_item_without_score(r) for r in top3]
        }
        samples_without_score.append(sample_without_score)
    
    return samples_with_score, samples_without_score


def build_negative_samples(data: List[Dict], num_samples: int = 500) -> tuple:
    """
    构建负样本数据集
    
    随机选择 user_query，随机匹配 3 个不相关样本
    
    Args:
        data: 原始数据
        num_samples: 负样本数量
    
    Returns:
        (带分数的数据, 不带分数的数据)
    """
    # 收集所有 retrieved 项作为候选池
    all_retrieved_items = []
    for entry in data:
        for item in entry.get('retrieved', []):
            all_retrieved_items.append(item)
    
    if len(all_retrieved_items) < 3:
        print("候选池太小，无法构建负样本")
        return [], []
    
    # 随机选择 user_query
    selected_entries = random.sample(data, min(num_samples, len(data)))
    
    samples_with_score = []
    samples_without_score = []
    
    for entry in selected_entries:
        test_id = entry.get('test_id')
        
        # 随机选择 3 个不相关样本（排除自己和原 retrieved 中的）
        original_ids = {r.get('id') for r in entry.get('retrieved', [])}
        original_ids.add(test_id)
        
        # 从候选池中筛选出不相关的
        candidates = [r for r in all_retrieved_items if r.get('id') not in original_ids]
        
        if len(candidates) < 3:
            continue
        
        # 随机选择 3 个
        random_retrieved = random.sample(candidates, 3)
        
        # 带分数版本（负样本分数设为0或保留原分数标记为负样本）
        sample_with_score = {
            'test_id': test_id,
            'user_query': entry.get('user_query'),
            'user_decomposition': entry.get('user_decomposition'),
            'label': 'negative',
            'retrieved': [format_item_with_score(r) for r in random_retrieved]
        }
        samples_with_score.append(sample_with_score)
        
        # 不带分数版本
        sample_without_score = {
            'test_id': test_id,
            'user_query': entry.get('user_query'),
            'user_decomposition': entry.get('user_decomposition'),
            'label': 'negative',
            'retrieved': [format_item_without_score(r) for r in random_retrieved]
        }
        samples_without_score.append(sample_without_score)
    
    return samples_with_score, samples_without_score


def save_jsonl(data: List[Dict], path: str):
    """保存为 JSONL 格式"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存 {len(data)} 条到: {path}")


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    data = load_data(INPUT_PATH)
    print(f"共加载 {len(data)} 条数据")
    
    # 构建正样本
    print("\n构建正样本...")
    pos_with_score, pos_without_score = build_positive_samples(data)
    print(f"正样本数量: {len(pos_with_score)}")
    
    # 构建负样本
    print("\n构建负样本（随机500个）...")
    neg_with_score, neg_without_score = build_negative_samples(data, num_samples=500)
    print(f"负样本数量: {len(neg_with_score)}")
    
    # 保存正样本
    save_jsonl(pos_with_score, os.path.join(OUTPUT_DIR, "positive_with_score.jsonl"))
    save_jsonl(pos_without_score, os.path.join(OUTPUT_DIR, "positive_without_score.jsonl"))
    
    # 保存负样本
    save_jsonl(neg_with_score, os.path.join(OUTPUT_DIR, "negative_with_score.jsonl"))
    save_jsonl(neg_without_score, os.path.join(OUTPUT_DIR, "negative_without_score.jsonl"))
    
    # 合并数据集（正负样本混合）
    print("\n合并正负样本...")
    all_with_score = pos_with_score + neg_with_score
    all_without_score = pos_without_score + neg_without_score
    
    # 打乱顺序
    random.shuffle(all_with_score)
    random.shuffle(all_without_score)
    
    save_jsonl(all_with_score, os.path.join(OUTPUT_DIR, "similarity_dataset_with_score.jsonl"))
    save_jsonl(all_without_score, os.path.join(OUTPUT_DIR, "similarity_dataset_without_score.jsonl"))
    
    print("\n完成！输出文件：")
    print(f"  - {OUTPUT_DIR}/positive_with_score.jsonl")
    print(f"  - {OUTPUT_DIR}/positive_without_score.jsonl")
    print(f"  - {OUTPUT_DIR}/negative_with_score.jsonl")
    print(f"  - {OUTPUT_DIR}/negative_without_score.jsonl")
    print(f"  - {OUTPUT_DIR}/similarity_dataset_with_score.jsonl")
    print(f"  - {OUTPUT_DIR}/similarity_dataset_without_score.jsonl")


if __name__ == "__main__":
    random.seed(42)  # 可复现
    main()
