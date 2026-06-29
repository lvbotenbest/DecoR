"""构造相似性判断数据集（测试集检索结果筛选版）"""
import json
import os
import random
import argparse
from typing import List, Dict

# 输入输出路径
DEFAULT_INPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    ".\OOD_Dataset\stage2_out\sim_result_ood_test_corpus_v1.jsonl",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    ".\OOD_Dataset\stage2_out\Test_ood_corpus_v1.jsonl",
)


def load_data(input_path: str) -> List[Dict]:
    """加载检索结果数据"""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def filter_and_select_topk(entry: Dict, top_k: int) -> List[Dict]:
    """
    过滤重复并选择 top 3
    
    - 过滤掉 retrieved 中 id 与 test_id 相同的
    - 选择 top k
    """
    test_id = entry.get('test_id')
    retrieved = entry.get('retrieved', [])
    
    # 过滤掉与 test_id 重复的
    filtered = [r for r in retrieved if r.get('id') != test_id]
    
    # 取 top k（已经按分数排序）
    return filtered[:top_k]


def format_item(item: Dict, keep_scores: bool) -> Dict:
    out = {
        'id': item.get('id'),
        'query': item.get('query'),
        'decomposition': item.get('decomposition')
    }
    if keep_scores:
        out['scores'] = item.get('scores', item.get('score'))
    return out




def save_jsonl(data: List[Dict], path: str):
    """保存为 JSONL 格式"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存 {len(data)} 条到: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description='构造相似性判断数据集（测试集）')
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_PATH,
                        help='输入检索结果 JSONL（默认: TestData/V1/retrieval_results_test_with_it_self.jsonl）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出 JSONL 路径（默认: 在 output_dir 下生成 filtered 文件）')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='输出目录（仅当未指定 --output 时生效）')
    parser.add_argument('--top_k', '-k', type=int, default=3,
                        help='从 retrieved 中选择的 top k（过滤掉与 test_id 重复的后再截断）')
    parser.add_argument('--keep_scores', action='store_true',
                        help='在输出中保留检索分数（scores/score 字段）')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(
            args.output_dir,
            f"{base_name}.filtered_top{args.top_k}.jsonl",
        )
    
    # 加载数据
    print("加载数据...")
    data = load_data(args.input)
    print(f"共加载 {len(data)} 条数据")
    
    # 筛选并写出
    print("\n筛选并写出...")
    output_data = []
    for entry in data:
        topk = filter_and_select_topk(entry, top_k=args.top_k)

        output_data.append({
            'test_id': entry.get('test_id'),
            'user_query': entry.get('user_query'),
            'user_decomposition': entry.get('user_decomposition'),
            'eval_name': entry.get('eval_name', ''),
            'retrieved': [format_item(r, keep_scores=args.keep_scores) for r in topk]
        })

    save_jsonl(output_data, args.output)
    print("\n完成！输出文件：")
    print(f"  - {args.output}")


if __name__ == "__main__":

    main()
