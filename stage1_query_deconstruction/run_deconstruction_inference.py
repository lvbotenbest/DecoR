"""多线程批量处理JSONL文件中的prompt，调用千帆模型API"""

import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from qianfan_client import QianfanRequest, prompt_template

# 全局锁，用于线程安全的文件写入
write_lock = Lock()

# 需要排除的字段（不从原始数据中保留）
EXCLUDE_FIELDS = {'decomposition', 'cost'}

def load_existing_results(output_file):
    """
    加载已存在的结果文件，返回已处理的 (id, prompt) 集合
    
    Args:
        output_file: 输出文件路径
    
    Returns:
        set: 已处理的 (id, prompt) 元组集合
    """
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        key = (data.get('id'), data.get('prompt'))
                        processed.add(key)
                    except json.JSONDecodeError:
                        continue
    return processed

def process_single_query(original_data, model_name, output_file):
    """
    处理单个query
    
    Args:
        original_data: 原始JSONL中的一行数据（dict）
        model_name: 模型名称
        output_file: 输出文件路径
    """
    idx = original_data.get('id', 0)
    prompt = original_data.get('prompt', '')
    
    # 构建完整的content（prompt_template + prompt）
    content = prompt_template + "\n" + prompt
    
    # 调用API
    qianfan = QianfanRequest()
    api_result = qianfan.req_decomposition(index=idx, prompt=content, model_name=model_name, query=prompt)
    
    # 构建最终结果：保留原始数据中的字段（排除decomposition和cost），合并API返回的字段
    result = {}
    for key, value in original_data.items():
        if key not in EXCLUDE_FIELDS:
            result[key] = value
    
    # 合并API返回的字段（content, input_tokens, output_tokens等）
    result['decomposition'] = api_result.get('content', 'ERR')
    result['decomposition_input_tokens'] = api_result.get('input_tokens', 0)
    result['decomposition_output_tokens'] = api_result.get('output_tokens', 0)
    if 'error' in api_result:
        result['error'] = api_result['error']
    
    # 线程安全地写入文件
    with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    return result

def process_jsonl_multithread(input_file, output_file, model_name, max_workers=5):
    """
    多线程处理JSONL文件中的prompt
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSONL文件路径
        model_name: 模型名称
        max_workers: 最大线程数
    """
    # 读取所有数据
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get('prompt'):
                    all_data.append(data)
    
    print(f"总共读取到 {len(all_data)} 条数据")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    
    # 加载已处理的数据，跳过重复的
    processed_keys = load_existing_results(output_file)
    if processed_keys:
        print(f"检测到已存在 {len(processed_keys)} 条已处理数据")
    
    # 过滤出需要处理的数据
    data_list = []
    for data in all_data:
        key = (data.get('id'), data.get('prompt'))
        if key not in processed_keys:
            data_list.append(data)
    
    skipped = len(all_data) - len(data_list)
    if skipped > 0:
        print(f"跳过 {skipped} 条已处理数据，还需处理 {len(data_list)} 条")
    
    if not data_list:
        print("所有数据已处理完毕，无需重新处理")
        return
    
    # 使用线程池处理
    completed = 0
    total = len(data_list)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_data = {
            executor.submit(process_single_query, data, model_name, output_file): data
            for data in data_list
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_data):
            data = future_to_data[future]
            idx = data.get('id', 'unknown')
            try:
                result = future.result()
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"进度: {completed}/{total} ({completed*100//total}%)")
            except Exception as e:
                print(f"处理 id={idx} 时出错: {str(e)}")
                completed += 1
    
    print(f"处理完成！结果已保存到: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='多线程批量处理JSONL文件，调用千帆模型API进行query decomposition')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入的JSONL文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出的JSONL文件路径（默认在SFTData/out_common_file/decomposition_result/下）')
    parser.add_argument('--model', '-m', type=str, default='g4jdefcg_test-3',
                        help='模型名称（默认: x26xxgbo_test_2')
    parser.add_argument('--workers', '-w', type=int, default=5,
                        help='线程数（默认: 5）')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    input_file = args.input
    model_name = args.model
    max_workers = args.workers
    
    # 设置输出路径
    if args.output:
        output_file = args.output
    else:
        # Default output directory
        output_dir = "../data/stage1_deconstruction/decomposition_result"
        os.makedirs(output_dir, exist_ok=True)
        input_basename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"result_{input_basename}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        exit(1)
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"模型名称: {model_name}")
    print(f"线程数: {max_workers}")
    print("-" * 50)
    
    # 开始处理
    process_jsonl_multithread(input_file, output_file, model_name, max_workers)

