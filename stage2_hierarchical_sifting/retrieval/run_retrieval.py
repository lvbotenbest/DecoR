"""
检索系统命令行接口

使用方式：
# 预构建 embedding 索引（离线运行，之后检索可直接加载）
python run_retrieval.py --build_index --corpus Dataset/corpus/IID/merged_query_decomposition_train.jsonl --faiss_path Dataset/faiss_index/corpus_index

# 单query检索
python run_retrieval.py --query "Your query text here" --top_k 10

# 批量检索（新格式：输出 top5 的 corpus query 带 decomposition 和 id）
python run_retrieval.py --input Dataset/TestData/Test_query_decomposition_QWen.jsonl --output Dataset/output/IID/retrieval_results.jsonl --top_k 5

# 使用 API embedding 和 FAISS
python run_retrieval.py --input test.jsonl --output output.jsonl --use_api_embedding --use_faiss --faiss_path Dataset/faiss_index/corpus_index
"""

import json
import argparse
import os
import sys
from typing import Dict, List

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_retrieval import QueryRetriever, QueryItem, RetrievalMode


def parse_args():
    parser = argparse.ArgumentParser(description='Query检索系统')
    
    # ============ 运行模式 ============
    parser.add_argument('--build_index', action='store_true',
                        help='预构建 embedding 索引模式（离线运行）')
    
    # 输入方式（二选一）
    parser.add_argument('--query', '-q', type=str, default=None,
                        help='单个query文本（交互式检索）')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='输入JSONL文件路径（批量检索）')
    
    # 输出
    parser.add_argument('--output', '-o', type=str, default='Dataset/output/IID/retrieval_results.jsonl',
                        help='输出文件路径')
    
    # 语料库
    parser.add_argument('--corpus', '-c', type=str,
                        default='Dataset/corpus/IID/merged_query_decomposition_train.jsonl',
                        help='历史query库路径')
    
    # ============ Embedding 配置 ============
    parser.add_argument('--use_api_embedding', action='store_true',
                        help='使用 API 调用 embedding（DeepInfra/OpenAI 兼容接口）')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API Key（也可通过 DEEPINFRA_API_KEY 环境变量设置）')
    parser.add_argument('--api_base_url', type=str, default='https://api.deepinfra.com/v1/openai',
                        help='API Base URL')
    parser.add_argument('--embedding_model', type=str, default='Qwen/Qwen3-Embedding-8B',
                        help='Embedding 模型名称')
    parser.add_argument('--api_batch_size', type=int, default=100,
                        help='API 每次请求的样本数')
    
    # ============ FAISS 配置 ============
    parser.add_argument('--use_faiss', action='store_true',
                        help='使用 FAISS 索引加速检索')
    parser.add_argument('--faiss_path', type=str, default='Dataset/faiss_index/corpus_index',
                        help='FAISS 索引路径（不含后缀）')
    parser.add_argument('--rebuild_faiss', action='store_true',
                        help='强制重建 FAISS 索引')
    
    # ============ 检索参数 ============
    parser.add_argument('--top_k', '-k', type=int, default=5,
                        help='最终返回结果数量（默认: 5）')
    parser.add_argument('--mode', type=str, default='hybrid',
                        choices=['coarse', 'fine', 'hybrid'],
                        help='检索模式: coarse(仅粗排), fine(仅精排), hybrid(粗精混合，默认)')
    parser.add_argument('--coarse_threshold', type=float, default=0.6,
                        help='粗排阈值（0.0-1.0，默认: 0.5）')
    parser.add_argument('--coarse_top_k', type=int, default=100,
                        help='粗排阶段保留的候选数量')
    parser.add_argument('--fine_top_k', type=int, default=None,
                        help='精排阶段返回数量（默认等于top_k）')
    parser.add_argument('--no_semantic', action='store_true',
                        help='禁用语义匹配（更快但效果稍差）')
    parser.add_argument('--no_synonym', action='store_true',
                        help='禁用同义词标准化')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='结构化分数权重（默认: 0.3）')
    parser.add_argument('--beta', type=float, default=0.6,
                        help='语义分数权重（默认: 0.7）')
    
    # ============ 缓存 ============
    parser.add_argument('--cache', type=str, default='CacheFile/index_cache.pkl',
                        help='索引缓存路径')
    parser.add_argument('--rebuild_cache', action='store_true',
                        help='强制重建索引缓存')

    parser.add_argument('--disable_relax_when_few_candidates', action='store_true',
                        help='禁用 coarse_filter 中“候选太少则放宽条件”的逻辑（用于测试集检索）')
    
    return parser.parse_args()


def build_embedding_index(args):
    """预构建 embedding 索引（仅支持 API Embedding）"""
    print("=" * 60)
    print("预构建 Embedding 索引模式")
    print("=" * 60)
    
    # 确保目录存在
    faiss_dir = os.path.dirname(args.faiss_path)
    if faiss_dir and not os.path.exists(faiss_dir):
        os.makedirs(faiss_dir)
        print(f"创建目录: {faiss_dir}")
    
    # 初始化检索器（始终使用 API Embedding）
    api_config = {
        'api_key': args.api_key,
        'base_url': args.api_base_url,
        'model': args.embedding_model,
        'batch_size': args.api_batch_size
    }
    
    retriever = QueryRetriever(
        corpus_path=args.corpus,
        use_semantic=not args.no_semantic,
        use_synonym=not args.no_synonym,
        use_api_embedding=True,  # 始终使用 API Embedding
        api_embedding_config=api_config,
        use_faiss=True,
        faiss_index_path=args.faiss_path
    )
    
    # 预构建索引
    retriever.build_embedding_index(
        output_path=args.faiss_path,
        batch_size=args.api_batch_size,
        force_rebuild=args.rebuild_faiss
    )
    
    print("\n预构建完成！")


def get_retriever(args) -> QueryRetriever:
    """初始化检索器（仅支持 API Embedding）"""
    api_config = {
        'api_key': args.api_key,
        'base_url': args.api_base_url,
        'model': args.embedding_model,
        'batch_size': args.api_batch_size
    }
    
    retriever = QueryRetriever(
        corpus_path=args.corpus,
        use_semantic=not args.no_semantic,
        use_synonym=not args.no_synonym,
        use_api_embedding=True,  # 始终使用 API Embedding
        api_embedding_config=api_config,
        use_faiss=args.use_faiss,
        faiss_index_path=args.faiss_path if args.use_faiss else None,
        relax_when_few_candidates=not args.disable_relax_when_few_candidates
    )
    
    return retriever


def format_retrieved_item(result: Dict, corpus_data: Dict = None) -> Dict:
    """
    格式化检索结果项
    
    输出格式：
    {
        "id": corpus_id,
        "query": corpus_prompt,
        "decomposition": {...},
        "scores": {
            "final_score": 综合最终分数,
            "semantic_score": 语义相似度分数,
            "structured_score": 结构化分数（S/K/D的平均值）,
            "skill_similarity": 技能相似度,
            "knowledge_similarity": 知识领域相似度,
            "difficulty_similarity": 难度匹配分数
        }
    }
    """
    # 获取详细分数
    score_details = result.get('score_details', {})
    
    return {
        'id': result['id'],
        'query': result['prompt'],
        'decomposition': {
            'S': result['skills'],
            'K': result['knowledge'],
            'D': result['difficulty'],
            'S_reason': result.get('s_reason', ''),
            'K_reason': result.get('k_reason', ''),
            'D_reason': result.get('d_reason', '')
        },
        'scores': {
            'final_score': result.get('score', 0.0),
            'semantic_score': score_details.get('semantic_score', 0.0),
            'structured_score': score_details.get('structured_score', 0.0),
            'skill_similarity': score_details.get('skill_similarity', 0.0),
            'knowledge_similarity': score_details.get('knowledge_similarity', 0.0),
            'difficulty_similarity': score_details.get('difficulty_similarity', 0.0)
        }
    }


def batch_search_new_format(retriever, input_path, output_path, top_k, alpha, beta, mode, 
                            coarse_threshold, coarse_top_k, fine_top_k):
    """
    批量检索（新输出格式）
    
    输出格式：
    {
        "test_id": xxx,
        "user_query": "...",
        "user_decomposition": {...},
        "retrieved": [
            {
                "id": corpus_id, 
                "query": corpus_prompt, 
                "decomposition": {...},
                "scores": {
                    "final_score": 0.9,           # 综合最终分数
                    "semantic_score": 0.85,       # 语义相似度分数
                    "structured_score": 0.95,     # 结构化分数（S/K/D平均）
                    "skill_similarity": 0.9,      # 技能相似度
                    "knowledge_similarity": 1.0,  # 知识领域相似度
                    "difficulty_similarity": 0.95 # 难度匹配分数
                }
            },
            ...
        ]
    }
    
    注意：
    - 空的 retrieved 结果不会被存储
    - 已检索过的 test_id 会被跳过（重复检测）
    """
    print(f"\n批量检索模式")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"检索模式: {mode} | TopK: {top_k}")
    print(f"粗排阈值: {coarse_threshold} | 粗排TopK: {coarse_top_k}")
    print("-" * 60)
    
    # 加载查询
    queries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    print(f"共 {len(queries)} 条query待检索")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 重复检测：读取已有输出文件中的 test_id
    processed_ids = set()
    if os.path.exists(output_path):
        print(f"检测到已有输出文件，读取已处理的 test_id...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        existing = json.loads(line)
                        test_id = existing.get('test_id')
                        if test_id is not None:
                            processed_ids.add(test_id)
                    except json.JSONDecodeError:
                        continue
        print(f"已处理 {len(processed_ids)} 条，将跳过这些记录")
    
    # 过滤掉已处理的 queries
    queries_to_process = [q for q in queries if q.get('id') not in processed_ids]
    print(f"本次需要检索 {len(queries_to_process)} 条query")
    
    if not queries_to_process:
        print("所有query已处理完毕，无需重复检索")
        return
    
    # 批量检索（追加模式）
    write_mode = 'a' if processed_ids else 'w'
    saved_count = 0
    skipped_empty = 0
    
    with open(output_path, write_mode, encoding='utf-8') as f:
        for i, query_data in enumerate(queries_to_process):
            results = retriever.retrieve(
                query_data,
                top_k=top_k,
                retrieval_mode=mode,
                coarse_threshold=coarse_threshold,
                coarse_top_k=coarse_top_k,
                fine_top_k=fine_top_k or top_k,
                fine_rank_config={'alpha': alpha, 'beta': beta}
            )
            
            if results is None:
                results = []
            
            if not results:
                skipped_empty += 1
            
            # 获取原始 decomposition
            decomposition = query_data.get('decomposition', {})
            if isinstance(decomposition, str):
                try:
                    decomposition = json.loads(decomposition)
                except json.JSONDecodeError:
                    decomposition = {}
            
            # 新的输出格式
            output = {
                'test_id': query_data.get('id'),
                'user_query': query_data.get('user_query') or query_data.get('prompt', ''),
                'user_decomposition': decomposition,
                'eval_name': query_data.get('eval_name', ''),
                'retrieved': [format_retrieved_item(r) for r in results]
            }
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            saved_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"进度: {i + 1}/{len(queries_to_process)} ({(i+1)*100//len(queries_to_process)}%)")
    
    print(f"\n检索完成！")
    print(f"  - 保存: {saved_count} 条")
    print(f"  - 跳过空结果: {skipped_empty} 条")
    print(f"  - 跳过已处理: {len(processed_ids)} 条")
    print(f"结果已保存到: {output_path}")


def interactive_search(retriever, top_k, alpha, beta, mode, coarse_threshold, coarse_top_k, fine_top_k):
    """交互式检索"""
    print("\n" + "=" * 60)
    print("交互式检索模式（输入 'quit' 退出）")
    print(f"当前模式: {mode} | 粗排阈值: {coarse_threshold} | 粗排TopK: {coarse_top_k}")
    print("=" * 60)
    
    while True:
        print("\n请输入query（或 'quit' 退出）:")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("退出检索系统")
            break
        
        if not user_input:
            continue
        
        query_text = user_input
        
        # 构造query数据（没有decomposition时只用语义匹配）
        query_data = {
            'id': 0,
            'prompt': query_text,
            'decomposition': {}  # 空的decomposition
        }
        
        # 检索
        results = retriever.retrieve(
            query_data,
            top_k=top_k,
            retrieval_mode=mode,
            coarse_threshold=coarse_threshold,
            coarse_top_k=coarse_top_k,
            fine_top_k=fine_top_k or top_k,
            fine_rank_config={'alpha': alpha, 'beta': beta}
        )
        
        # 显示结果
        print(f"\n找到 {len(results)} 条相关结果 (模式: {mode}):")
        print("-" * 60)
        
        for i, result in enumerate(results):
            print(f"\n#{i+1} [Score: {result['score']:.4f}]")
            print(f"   ID: {result['id']} | Eval: {result['eval_name']}")
            prompt_preview = result['prompt'][:150] + '...' if len(result['prompt']) > 150 else result['prompt']
            print(f"   Prompt: {prompt_preview}")
            print(f"   S: {result['skills']}")
            print(f"   K: {result['knowledge']}")
            print(f"   D: {result['difficulty']}")


def single_query_search(retriever, query_text, top_k, alpha, beta, mode,
                        coarse_threshold, coarse_top_k, fine_top_k):
    """单query检索"""
    query_data = {
        'id': 0,
        'prompt': query_text,
        'decomposition': {}
    }
    
    results = retriever.retrieve(
        query_data,
        top_k=top_k,
        retrieval_mode=mode,
        coarse_threshold=coarse_threshold,
        coarse_top_k=coarse_top_k,
        fine_top_k=fine_top_k or top_k,
        fine_rank_config={'alpha': alpha, 'beta': beta}
    )
    
    print(f"\nQuery: {query_text[:100]}...")
    print(f"检索模式: {mode} | 粗排阈值: {coarse_threshold}")
    print(f"\n找到 {len(results)} 条相关结果:")
    print("-" * 60)
    
    for i, result in enumerate(results):
        print(f"\n#{i+1} [Score: {result['score']:.4f}]")
        print(f"   ID: {result['id']} | Eval: {result['eval_name']}")
        prompt_preview = result['prompt'][:150] + '...' if len(result['prompt']) > 150 else result['prompt']
        print(f"   Prompt: {prompt_preview}")
        print(f"   S: {result['skills']}")
        print(f"   K: {result['knowledge']}")
        print(f"   D: {result['difficulty']}")
    
    return results


def main():
    args = parse_args()
    
    # 切换到 retrievel 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 预构建索引模式
    if args.build_index:
        if not os.path.exists(args.corpus):
            print(f"错误: 语料库文件不存在: {args.corpus}")
            sys.exit(1)
        build_embedding_index(args)
        return
    
    # 检查语料库是否存在
    if not os.path.exists(args.corpus):
        print(f"错误: 语料库文件不存在: {args.corpus}")
        sys.exit(1)
    
    # 初始化检索器
    print("初始化检索器...")
    retriever = get_retriever(args)
    
    # 构建索引
    cache_path = None if args.rebuild_cache else args.cache
    if args.rebuild_cache and os.path.exists(args.cache):
        os.remove(args.cache)
        print(f"已删除旧缓存: {args.cache}")
    
    print("构建/加载索引...")
    retriever.build_index(cache_path=args.cache)
    
    # 打印配置信息
    print(f"\n配置: 模式={args.mode}, 粗排阈值={args.coarse_threshold}, "
          f"粗排TopK={args.coarse_top_k}, 精排TopK={args.fine_top_k or args.top_k}")
    print(f"同义词标准化: {'启用' if not args.no_synonym else '禁用'}")
    print(f"API Embedding: {'启用' if args.use_api_embedding else '禁用'}")
    print(f"FAISS: {'启用' if args.use_faiss else '禁用'}")
    
    # 根据输入方式选择检索模式
    if args.input:
        # 批量检索模式
        if not os.path.exists(args.input):
            print(f"错误: 输入文件不存在: {args.input}")
            sys.exit(1)
        batch_search_new_format(
            retriever, args.input, args.output, args.top_k, 
            args.alpha, args.beta, args.mode, 
            args.coarse_threshold, args.coarse_top_k, args.fine_top_k
        )
    elif args.query:
        # 单query检索
        single_query_search(
            retriever, args.query, args.top_k, args.alpha, args.beta, 
            args.mode, args.coarse_threshold, args.coarse_top_k, args.fine_top_k
        )
    else:
        # 交互式检索
        interactive_search(
            retriever, args.top_k, args.alpha, args.beta, args.mode,
            args.coarse_threshold, args.coarse_top_k, args.fine_top_k
        )


if __name__ == '__main__':
    main()
