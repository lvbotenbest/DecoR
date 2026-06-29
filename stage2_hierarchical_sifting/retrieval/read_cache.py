import argparse
import os
import pickle
from typing import List, Dict, Any
import json

from query_retrieval import QueryRetriever, QueryItem


class CacheUnpickler(pickle.Unpickler):
    """Custom unpickler to handle module name changes"""
    def find_class(self, module, name):
        if module == "__main__":
            module = "query_retrieval"
        return super().find_class(module, name)


def load_pkl_file(path: str) -> Dict[str, Any]:
    """Load any pkl file with custom unpickler"""
    with open(path, "rb") as f:
        return CacheUnpickler(f).load()


def describe_inverted_index(inv_index) -> None:
    """Print basic stats about the inverted indices."""
    def summarize(index_name: str, index_map) -> None:
        total_terms = len(index_map)
        total_refs = sum(len(v) for v in index_map.values())
        print(f"  - {index_name}: {total_terms} terms, {total_refs} total references")

    print("倒排索引情况：")
    summarize("skill_index", inv_index.skill_index)
    summarize("knowledge_index", inv_index.knowledge_index)
    summarize("difficulty_index", inv_index.difficulty_index)
    summarize("eval_index", inv_index.eval_index)


def print_samples(corpus: List[QueryItem], sample_count: int) -> None:
    """Print a few sample entries from the corpus."""
    print(f"\n展示前 {sample_count} 条语料（如果有）：")
    for item in corpus[:sample_count]:
        print("-" * 60)
        print(f"id={item.id} | difficulty={item.difficulty}")
        prompt_preview = item.prompt if len(item.prompt) <= 150 else item.prompt[:150] + "..."
        print(f"prompt: {prompt_preview}")
        print(f"S: {sorted(item.skills)}")
        print(f"K: {sorted(item.knowledge)}")
        print(f"S_reason: {item.s_reason or 'N/A'}")
        print(f"K_reason: {item.k_reason or 'N/A'}")
        if item.d_reason:
            print(f"D_reason: {item.d_reason}")


def detect_cache_type(data: Dict[str, Any]) -> str:
    """Detect cache file type by inspecting keys"""
    if 'corpus' in data and 'id_to_idx' in data and 'inverted_index' in data:
        return 'index_cache'
    elif 'id_map' in data and 'embedding_dim' in data:
        return 'faiss_meta'
    else:
        return 'unknown'


def read_index_cache(cache_path: str, sample_count: int) -> None:
    """Read and display index_cache.pkl"""
    print(f"\n{'='*60}")
    print(f"文件类型: Index Cache (索引缓存)")
    print(f"{'='*60}")
    
    cache_data = load_pkl_file(cache_path)
    
    retriever = QueryRetriever(
        corpus_path="",
        use_semantic=False,
        use_synonym=False,
    )
    
    retriever.corpus = cache_data["corpus"]
    retriever.id_to_idx = cache_data["id_to_idx"]
    retriever.inverted_index = cache_data["inverted_index"]

    print(f"\n📊 统计信息：")
    print(f"  - 语料总数：{len(retriever.corpus)}")
    print(f"  - id_to_idx 映射：{len(retriever.id_to_idx)} 条")
    print()
    describe_inverted_index(retriever.inverted_index)

    if retriever.corpus:
        print_samples(retriever.corpus, sample_count)


def read_faiss_meta(cache_path: str) -> None:
    """Read and display FAISS meta.pkl"""
    print(f"\n{'='*60}")
    print(f"文件类型: FAISS Meta (FAISS 索引元数据)")
    print(f"{'='*60}")
    
    meta_data = load_pkl_file(cache_path)
    
    print(f"\n📊 FAISS 索引信息：")
    print(f"  - Embedding 维度: {meta_data.get('embedding_dim', 'N/A')}")
    print(f"  - ID 映射数量: {len(meta_data.get('id_map', {}))}")
    
    id_map = meta_data.get('id_map', {})
    if id_map:
        print(f"\n🔍 ID 映射示例（前 10 个）：")
        print(f"  faiss_idx -> original_id")
        for faiss_idx, orig_id in list(id_map.items())[:10]:
            print(f"    {faiss_idx:6d} -> {orig_id}")
        
        if len(id_map) > 10:
            print(f"    ... (共 {len(id_map)} 条映射)")
    
    # 如果有对应的 .ids.json 文件，也读取显示
    base_path = cache_path.replace('.meta.pkl', '')
    ids_path = base_path + '.ids.json'
    if os.path.exists(ids_path):
        print(f"\n📄 对应的 IDs 文件: {ids_path}")
        with open(ids_path, 'r') as f:
            ids = json.load(f)
        print(f"  - IDs 数量: {len(ids)}")
        print(f"  - 前 10 个 IDs: {ids[:10]}")
    
    # 检查是否有 .embeddings.npy 文件
    embeddings_path = base_path + '.embeddings.npy'
    if os.path.exists(embeddings_path):
        print(f"\n💾 对应的 Embeddings 文件: {embeddings_path}")
        try:
            import numpy as np
            embeddings = np.load(embeddings_path)
            print(f"  - Shape: {embeddings.shape}")
            print(f"  - Dtype: {embeddings.dtype}")
            print(f"  - Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"  - 读取失败: {e}")
    
    # 检查是否有 .faiss 文件
    faiss_path = base_path + '.faiss'
    if os.path.exists(faiss_path):
        print(f"\n🗂️  对应的 FAISS 索引文件: {faiss_path}")
        file_size = os.path.getsize(faiss_path) / 1024 / 1024
        print(f"  - 文件大小: {file_size:.2f} MB")


def read_generic_pkl(cache_path: str) -> None:
    """Read and display generic pkl file"""
    print(f"\n{'='*60}")
    print(f"文件类型: Unknown PKL (通用 pickle 文件)")
    print(f"{'='*60}")
    
    data = load_pkl_file(cache_path)
    
    print(f"\n📦 文件内容类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"  - Keys: {list(data.keys())}")
        print(f"\n详细内容：")
        for key, value in data.items():
            value_type = type(value).__name__
            if isinstance(value, (list, dict, set)):
                size_info = f" (length: {len(value)})"
            else:
                size_info = ""
            print(f"  - {key}: {value_type}{size_info}")
            
            # 如果是简单类型，直接显示
            if isinstance(value, (int, float, str, bool)):
                print(f"      值: {value}")
    else:
        print(f"\n内容预览：")
        print(str(data)[:500])


def main():
    parser = argparse.ArgumentParser(
        description="查看 PKL 文件内容（支持 index_cache.pkl、FAISS meta.pkl 等）"
    )
    parser.add_argument(
        "--cache",
        "-c",
        type=str,
        default="CacheFile/index_cache.pkl",
        help="PKL 文件路径",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=3,
        help="对于 index_cache，打印几个 sample",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=['auto', 'index_cache', 'faiss_meta', 'generic'],
        default='auto',
        help="强制指定文件类型（默认自动检测）",
    )
    args = parser.parse_args()

    cache_path = os.path.abspath(args.cache)
    if not os.path.exists(cache_path):
        print(f"❌ 文件不存在: {cache_path}")
        return

    print(f"📂 读取文件: {cache_path}")
    print(f"   文件大小: {os.path.getsize(cache_path) / 1024:.2f} KB")
    
    # 检测文件类型
    if args.type == 'auto':
        try:
            data = load_pkl_file(cache_path)
            cache_type = detect_cache_type(data)
        except Exception as e:
            print(f"\n❌ 读取文件失败: {e}")
            return
    else:
        cache_type = args.type
    
    # 根据类型读取
    try:
        if cache_type == 'index_cache':
            read_index_cache(cache_path, args.samples)
        elif cache_type == 'faiss_meta':
            read_faiss_meta(cache_path)
        else:
            read_generic_pkl(cache_path)
    except Exception as e:
        print(f"\n❌ 处理文件时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

