"""
基于Query Decomposition的多阶段检索系统

检索方案设计：
================

## 核心思想
充分利用query拆解后的结构化信息（S技能、K知识领域、D难度）进行多阶段检索：
1. 粗筛阶段：基于结构化字段快速过滤候选集
2. 精排阶段：基于语义相似度进行精细排序

## 检索流程
┌─────────────────────────────────────────────────────────────┐
│  新Query  →  拆解(S,K,D)  →  粗筛  →  精排  →  Top-K结果   │
└─────────────────────────────────────────────────────────────┘

### Stage 1: 粗筛 (Coarse Filtering)
- 难度过滤：D级别相同或相邻(±1)的候选
- 技能匹配：S集合Jaccard相似度 > 阈值
- 知识匹配：K集合Jaccard相似度 > 阈值

### Stage 2: 精排 (Fine Ranking)
- Prompt语义相似度（Embedding cosine similarity）
- 综合评分：α*结构化分数 + β*语义分数

## 使用方式
```python
from query_retrieval import QueryRetriever

# 初始化检索器
retriever = QueryRetriever(
    corpus_path="SFTData/out_common_file/merged_query_decomposition_train.jsonl"
)

# 构建索引
retriever.build_index()

# 检索
query_data = {
    "prompt": "...",
    "decomposition": {"S": [...], "K": [...], "D": "D1", ...}
}
results = retriever.retrieve(query_data, top_k=10)
```
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field
import pickle


# 检索模式枚举
class RetrievalMode:
    COARSE = "coarse"      # 仅粗排（基于结构化字段）
    FINE = "fine"          # 仅精排（全量语义匹配）
    HYBRID = "hybrid"      # 粗精混合（默认）


# 同义词表 JSON 文件路径
SYNONYM_JSON_PATH = os.path.join(os.path.dirname(__file__), "synonym_dict.json")


class SynonymMapper:
    """
    同义词映射器
    解决 "mathematics" vs "math", "arithmetic" vs "calculation" 等同义词问题
    将不同表述的词标准化到统一的词
    
    同义词表存储在 synonym_dict.json 文件中，可通过 update_synonym.py 自动更新
    """
    
    # 默认同义词表（当JSON文件不存在时使用）
    DEFAULT_SYNONYM_GROUPS = {
        "mathematics": ["math", "maths", "mathematical"],
        "arithmetic": ["calculation", "calculations", "computing", "computation"],
        "reasoning": ["reason", "deduction", "inference", "logical inference"],
        "coding": ["programming", "code", "software development"],
        "problem solving": ["problem-solving", "solving problems"],
    }
    
    def __init__(self, json_path: str = None):
        """
        初始化同义词映射器
        
        Args:
            json_path: 同义词表JSON文件路径，为None时使用默认路径
        """
        self.json_path = json_path or SYNONYM_JSON_PATH
        self.synonym_groups: Dict[str, List[str]] = {}
        self.synonym_to_standard: Dict[str, str] = {}
        
        # 加载同义词表
        self.load()
    
    def load(self, json_path: str = None) -> bool:
        """
        从JSON文件加载同义词表
        
        Args:
            json_path: 同义词表路径，为None时使用初始化时的路径
        
        Returns:
            是否加载成功
        """
        path = json_path or self.json_path
        
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.synonym_groups = data.get("synonym_groups", {})
                print(f"从 {path} 加载了 {len(self.synonym_groups)} 个同义词组")
            else:
                print(f"同义词表文件不存在: {path}，使用默认同义词表")
                self.synonym_groups = self.DEFAULT_SYNONYM_GROUPS.copy()
        except Exception as e:
            print(f"加载同义词表失败: {e}，使用默认同义词表")
            self.synonym_groups = self.DEFAULT_SYNONYM_GROUPS.copy()
        
        # 构建反向映射
        self._build_reverse_mapping()
        return True
    
    def reload(self) -> bool:
        """重新加载同义词表（用于更新后刷新）"""
        return self.load()
    
    def _build_reverse_mapping(self):
        """构建反向映射：同义词 -> 标准词"""
        self.synonym_to_standard = {}
        for standard, synonyms in self.synonym_groups.items():
            # 标准词映射到自己
            self.synonym_to_standard[standard.lower()] = standard.lower()
            # 同义词映射到标准词
            for syn in synonyms:
                self.synonym_to_standard[syn.lower()] = standard.lower()
    
    def normalize(self, term: str) -> str:
        """将词标准化（转小写 + 同义词映射）"""
        term_lower = term.lower().strip()
        return self.synonym_to_standard.get(term_lower, term_lower)
    
    def normalize_set(self, terms: Set[str]) -> Set[str]:
        """标准化词集合"""
        return {self.normalize(t) for t in terms}
    
    def add_synonym(self, standard: str, synonym: str):
        """动态添加同义词（仅内存，不保存到文件）"""
        self.synonym_to_standard[synonym.lower()] = standard.lower()
        self.synonym_to_standard[standard.lower()] = standard.lower()
        
        # 更新 synonym_groups
        if standard.lower() not in self.synonym_groups:
            self.synonym_groups[standard.lower()] = []
        if synonym.lower() not in self.synonym_groups[standard.lower()]:
            self.synonym_groups[standard.lower()].append(synonym.lower())
    
    def add_synonym_group(self, standard: str, synonyms: List[str]):
        """动态添加同义词组（仅内存，不保存到文件）"""
        for syn in synonyms:
            self.add_synonym(standard, syn)
    
    def save(self, json_path: str = None):
        """
        保存同义词表到JSON文件
        
        Args:
            json_path: 保存路径，为None时使用初始化时的路径
        """
        path = json_path or self.json_path
        
        data = {
            "_description": "同义词映射表，key为标准词，value为同义词列表",
            "_updated_at": "",
            "synonym_groups": self.synonym_groups
        }
        
        # 添加更新时间
        from datetime import datetime
        data["_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"同义词表已保存到: {path}")
    
    def get_stats(self) -> Dict:
        """获取同义词表统计信息"""
        total_synonyms = sum(len(syns) for syns in self.synonym_groups.values())
        return {
            "num_groups": len(self.synonym_groups),
            "total_synonyms": total_synonyms,
            "total_terms": len(self.synonym_to_standard)
        }
    
    def find_standard(self, term: str) -> Optional[str]:
        """查找词的标准形式"""
        return self.synonym_to_standard.get(term.lower().strip())
    
    def get_synonyms(self, term: str) -> List[str]:
        """获取词的所有同义词"""
        standard = self.find_standard(term)
        if standard and standard in self.synonym_groups:
            return [standard] + self.synonym_groups[standard]
        return [term.lower().strip()]


# 全局同义词映射器实例（延迟初始化）
_SYNONYM_MAPPER = None

def get_synonym_mapper() -> SynonymMapper:
    """获取全局同义词映射器实例"""
    global _SYNONYM_MAPPER
    if _SYNONYM_MAPPER is None:
        _SYNONYM_MAPPER = SynonymMapper()
    return _SYNONYM_MAPPER

# 兼容旧代码的全局变量（直接调用获取实例）
SYNONYM_MAPPER = None  # 将在首次使用时初始化

def _get_global_synonym_mapper():
    """内部函数：获取全局同义词映射器"""
    global SYNONYM_MAPPER
    if SYNONYM_MAPPER is None:
        SYNONYM_MAPPER = get_synonym_mapper()
    return SYNONYM_MAPPER


def reload_synonym_mapper():
    """重新加载同义词表"""
    global _SYNONYM_MAPPER
    if _SYNONYM_MAPPER is not None:
        _SYNONYM_MAPPER.reload()
    else:
        _SYNONYM_MAPPER = SynonymMapper()


@dataclass
class QueryItem:
    """存储单个query的拆解信息"""
    id: int
    prompt: str
    skills: Set[str] = field(default_factory=set)  # S
    knowledge: Set[str] = field(default_factory=set)  # K
    difficulty: str = "D1"  # D
    s_reason: str = ""  # S_reason
    k_reason: str = ""  # K_reason
    d_reason: str = ""  # D_reason
    eval_name: str = ""
    raw_data: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None  # prompt + reasons 的复合embedding
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QueryItem':
        """从字典创建QueryItem"""
        decomposition = data.get('decomposition', {})
        
        # 如果decomposition是字符串，尝试解析
        if isinstance(decomposition, str):
            try:
                decomposition = json.loads(decomposition)
            except json.JSONDecodeError:
                decomposition = {}
        
        skills = set(decomposition.get('S', []))
        knowledge = set(decomposition.get('K', []))
        difficulty = decomposition.get('D', 'D1')
        s_reason = decomposition.get('S_reason', '')
        k_reason = decomposition.get('K_reason', '')
        d_reason = decomposition.get('D_reason', '')
        
        # 标准化难度格式
        if difficulty and not difficulty.startswith('D'):
            difficulty = f"D{difficulty}"
        
        return cls(
            id=data.get('id', 0),
            prompt=data.get('prompt', ''),
            skills=skills,
            knowledge=knowledge,
            difficulty=difficulty,
            s_reason=s_reason,
            k_reason=k_reason,
            d_reason=d_reason,
            eval_name=data.get('eval_name', ''),
            raw_data=data
        )
    
    def get_semantic_text(self) -> str:
        """
        获取用于语义匹配的复合文本
        将 prompt 和 reasons 拼接，充分利用拆解信息
        """
        parts = [self.prompt]
        
        # 拼接reason信息（如果存在）
        reasons = []
        if self.s_reason:
            reasons.append(f"Skills: {self.s_reason}")
        if self.k_reason:
            reasons.append(f"Knowledge: {self.k_reason}")
        if self.d_reason:
            reasons.append(f"Difficulty: {self.d_reason}")
        
        if reasons:
            parts.append(" | ".join(reasons))
        
        return " || ".join(parts)


class StructuredMatcher:
    """基于结构化字段的匹配器（支持同义词标准化）"""
    
    def __init__(self, use_synonym: bool = True):
        """
        Args:
            use_synonym: 是否使用同义词标准化
        """
        self.use_synonym = use_synonym
        self.synonym_mapper = _get_global_synonym_mapper() if use_synonym else None
    
    def _normalize_set(self, terms: Set[str]) -> Set[str]:
        """标准化词集合"""
        if self.use_synonym and self.synonym_mapper:
            return self.synonym_mapper.normalize_set(terms)
        return {t.lower() for t in terms}
    
    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """计算Jaccard相似度（先标准化同义词）"""
        if not set1 or not set2:
            return 0.0
        
        # 标准化同义词
        norm_set1 = self._normalize_set(set1)
        norm_set2 = self._normalize_set(set2)
        
        intersection = len(norm_set1 & norm_set2)
        union = len(norm_set1 | norm_set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def difficulty_distance(d1: str, d2: str) -> int:
        """计算难度等级距离"""
        level_map = {'D0': 0, 'D1': 1, 'D2': 2, 'D3': 3}
        l1 = level_map.get(d1, 1)
        l2 = level_map.get(d2, 1)
        return abs(l1 - l2)
    
    @staticmethod
    def difficulty_score(query_difficulty: str, corpus_difficulty: str) -> float:
        """
        难度匹配分数
        
        逻辑：
        - 如果 corpus 难度 >= query 难度，说明检索到的样本更难或同等难度，
          模型能回答正确检索到的样本，也应该能回答正确新的 query，给满分
        - 如果 corpus 难度 < query 难度，说明检索到的样本更简单，
          模型能回答正确检索到的样本，不一定能回答正确更难的新 query，降分
        
        Args:
            query_difficulty: 新输入 query 的难度（如 "D1"）
            corpus_difficulty: 检索到的 corpus 样本难度（如 "D2"）
        
        Returns:
            难度匹配分数 (0.0 - 1.0)
        """
        level_map = {'D0': 0, 'D1': 1, 'D2': 2, 'D3': 3}
        query_level = level_map.get(query_difficulty, 1)
        corpus_level = level_map.get(corpus_difficulty, 1)
        
        if corpus_level >= query_level:
            # corpus 难度 >= query 难度，满分
            return 1.0
        else:
            # corpus 难度 < query 难度，根据差距降分
            diff = query_level - corpus_level  # diff > 0
            return 1.0 - diff * 0.25  # 每级扣0.25分，最多扣0.75分
    
    def compute_structured_score(self, query: QueryItem, candidate: QueryItem) -> Dict[str, float]:
        """计算结构化匹配分数"""
        skill_sim = self.jaccard_similarity(query.skills, candidate.skills)
        knowledge_sim = self.jaccard_similarity(query.knowledge, candidate.knowledge)
        difficulty_sim = self.difficulty_score(query.difficulty, candidate.difficulty)
        
        return {
            'skill_similarity': skill_sim,
            'knowledge_similarity': knowledge_sim,
            'difficulty_similarity': difficulty_sim,
          #  'structured_score': (skill_sim + knowledge_sim + difficulty_sim) / 3
            'structured_score': difficulty_sim*(skill_sim + knowledge_sim ) / 2  #难度越高，权重越大
        }


class APIEmbedding:
    """
    通过 API 调用 Embedding 模型（支持 DeepInfra/OpenAI 兼容接口）
    
    使用方式：
    ```python
    embedder = APIEmbedding(
        api_key="your-api-key",
        base_url="https://api.deepinfra.com/v1/openai",
        model="Qwen/Qwen3-Embedding-8B"
    )
    embeddings = embedder.encode(["text1", "text2", "text3"], batch_size=10)
    ```
    """
    
    def __init__(
        self, 
        api_key: str = None,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        model: str = "Qwen/Qwen3-Embedding-8B",
        batch_size: int = 50  # API 一次请求的最大 sample 数
    ):
        """
        Args:
            api_key: API Key（可通过环境变量 DEEPINFRA_API_KEY 设置）
            base_url: API Base URL
            model: Embedding 模型名称
            batch_size: 每次 API 请求的最大样本数
        """
        import os
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY", "HLwaq5OfA0dhif7NLl7afm7kUFuP0toS")
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size
        self._client = None
        self._embedding_dim = None
    
    def _get_client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def encode(self, texts: List[str], batch_size: int = None, show_progress: bool = True) -> np.ndarray:
        """
        批量编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 每批次处理的数量（API一次请求的数量）
            show_progress: 是否显示进度
        
        Returns:
            np.ndarray: shape (len(texts), embedding_dim)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        client = self._get_client()
        all_embeddings = []
        
        # 分批处理
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_idx = i // batch_size + 1
            
            if show_progress:
                print(f"  Encoding batch {batch_idx}/{total_batches} ({len(batch_texts)} samples)...")
            
            try:
                response = client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                    encoding_format="float"
                )
                
                # 提取 embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # 记录 embedding 维度
                if self._embedding_dim is None and batch_embeddings:
                    self._embedding_dim = len(batch_embeddings[0])
                    
            except Exception as e:
                print(f"  API call failed: {e}")
                # 返回零向量作为 fallback
                fallback_dim = self._embedding_dim or 4096
                all_embeddings.extend([[0.0] * fallback_dim for _ in batch_texts])
        
        return np.array(all_embeddings, dtype=np.float32)
    
    @property
    def embedding_dim(self) -> int:
        """获取 embedding 维度"""
        return self._embedding_dim or 4096
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class FAISSIndex:
    """
    FAISS 向量索引，用于快速最近邻检索
    
    支持：
    - 预构建索引并保存到文件
    - 从文件加载预构建的索引
    - 快速 Top-K 检索
    """
    
    def __init__(self, embedding_dim: int = 4096, use_gpu: bool = False):
        """
        Args:
            embedding_dim: Embedding 向量维度
            use_gpu: 是否使用 GPU 加速
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        self.id_map: Dict[int, int] = {}  # faiss_idx -> original_id
        self._faiss = None
    
    def _get_faiss(self):
        """延迟导入 faiss"""
        if self._faiss is None:
            try:
                import faiss
                self._faiss = faiss
            except ImportError:
                raise ImportError("请安装 faiss: pip install faiss-cpu 或 faiss-gpu")
        return self._faiss
    
    def build(self, embeddings: np.ndarray, ids: List[int] = None):
        """
        构建 FAISS 索引
        
        Args:
            embeddings: shape (n_samples, embedding_dim)
            ids: 每个样本的 ID（可选，默认使用序号）
        """
        faiss = self._get_faiss()
        
        # 确保数据类型正确
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        n_samples = embeddings.shape[0]
        
        # 创建索引（使用 Inner Product，需要对向量归一化后等价于 cosine similarity）
        # 先归一化
        faiss.normalize_L2(embeddings)
        
        # 使用 IndexFlatIP（内积索引，归一化后等价于余弦相似度）
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # GPU 加速
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"GPU 不可用，使用 CPU: {e}")
        
        # 添加向量
        self.index.add(embeddings)
        
        # 构建 ID 映射
        if ids is None:
            ids = list(range(n_samples))
        self.id_map = {i: id_ for i, id_ in enumerate(ids)}
        
        print(f"FAISS 索引构建完成: {n_samples} 条向量, 维度 {self.embedding_dim}")
    
    def search(self, query_embeddings: np.ndarray, top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        检索最相似的向量
        
        Args:
            query_embeddings: shape (n_queries, embedding_dim)
            top_k: 返回 Top-K 结果
        
        Returns:
            List of List[(original_id, similarity_score)]
        """
        if self.index is None:
            raise RuntimeError("索引未构建，请先调用 build()")
        
        faiss = self._get_faiss()
        
        # 确保数据类型和格式正确
        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # 归一化查询向量
        faiss.normalize_L2(query_embeddings)
        
        # 搜索
        distances, indices = self.index.search(query_embeddings, top_k)
        
        # 转换结果
        results = []
        for i in range(len(query_embeddings)):
            query_results = []
            for j in range(top_k):
                idx = int(indices[i][j])
                if idx >= 0:  # -1 表示没有找到
                    original_id = self.id_map.get(idx, idx)
                    score = float(distances[i][j])
                    query_results.append((original_id, score))
            results.append(query_results)
        
        return results
    
    def save(self, path: str):
        """保存索引到文件"""
        if self.index is None:
            raise RuntimeError("索引未构建")
        
        faiss = self._get_faiss()
        
        # 保存 FAISS 索引
        index_path = path + ".faiss"
        if self.use_gpu:
            # GPU 索引需要先转回 CPU
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # 保存 ID 映射
        meta_path = path + ".meta.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"FAISS 索引已保存: {index_path}")
    
    def load(self, path: str):
        """从文件加载索引"""
        faiss = self._get_faiss()
        
        # 加载 FAISS 索引
        index_path = path + ".faiss"
        self.index = faiss.read_index(index_path)
        
        # GPU 加速
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"GPU 不可用，使用 CPU: {e}")
        
        # 加载 ID 映射
        meta_path = path + ".meta.pkl"
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.id_map = meta['id_map']
        self.embedding_dim = meta['embedding_dim']
        
        print(f"FAISS 索引已加载: {self.index.ntotal} 条向量")


class InvertedIndex:
    """倒排索引，用于快速过滤（支持同义词标准化）"""
    
    def __init__(self, use_synonym: bool = True):
        """
        Args:
            use_synonym: 是否使用同义词标准化
        """
        self.skill_index: Dict[str, Set[int]] = defaultdict(set)  # skill -> query_ids
        self.knowledge_index: Dict[str, Set[int]] = defaultdict(set)  # knowledge -> query_ids
        self.difficulty_index: Dict[str, Set[int]] = defaultdict(set)  # difficulty -> query_ids
        self.eval_index: Dict[str, Set[int]] = defaultdict(set)  # eval_name -> query_ids
        self.use_synonym = use_synonym
        self.synonym_mapper = _get_global_synonym_mapper() if use_synonym else None
    
    def _normalize(self, term: str) -> str:
        """标准化词"""
        if self.use_synonym and self.synonym_mapper:
            return self.synonym_mapper.normalize(term)
        return term.lower()
    
    def add(self, query_id: int, item: QueryItem):
        """添加索引（使用标准化后的词）"""
        for skill in item.skills:
            normalized_skill = self._normalize(skill)
            self.skill_index[normalized_skill].add(query_id)
        for knowledge in item.knowledge:
            normalized_knowledge = self._normalize(knowledge)
            self.knowledge_index[normalized_knowledge].add(query_id)
        self.difficulty_index[item.difficulty].add(query_id)
        if item.eval_name:
            self.eval_index[item.eval_name].add(query_id)
    
    def get_candidates_by_difficulty(self, difficulty: str, allow_adjacent: bool = True) -> Set[int]:
        """根据难度获取候选集"""
        candidates = self.difficulty_index.get(difficulty, set()).copy()
        if allow_adjacent:
            level_map = {'D0': 0, 'D1': 1, 'D2': 2, 'D3': 3}
            current_level = level_map.get(difficulty, 1)
            for d, level in level_map.items():
                if abs(level - current_level) == 1:
                    candidates |= self.difficulty_index.get(d, set())
        return candidates
    
    def get_candidates_by_skills(self, skills: Set[str], min_overlap: int = 1) -> Set[int]:
        """根据技能获取候选集（使用标准化后的词匹配）"""
        if not skills:
            return set()
        
        candidate_counts = defaultdict(int)
        for skill in skills:
            normalized_skill = self._normalize(skill)
            for qid in self.skill_index.get(normalized_skill, set()):
                candidate_counts[qid] += 1
        
        return {qid for qid, count in candidate_counts.items() if count >= min_overlap}
    
    def get_candidates_by_knowledge(self, knowledge: Set[str], min_overlap: int = 1) -> Set[int]:
        """根据知识领域获取候选集（使用标准化后的词匹配）"""
        if not knowledge:
            return set()
        
        candidate_counts = defaultdict(int)
        for k in knowledge:
            normalized_k = self._normalize(k)
            for qid in self.knowledge_index.get(normalized_k, set()):
                candidate_counts[qid] += 1
        
        return {qid for qid, count in candidate_counts.items() if count >= min_overlap}


class QueryRetriever:
    """Query检索器主类（仅支持 API Embedding）"""
    
    def __init__(
        self,
        corpus_path: str,
        use_semantic: bool = True,
        use_synonym: bool = False,
        use_api_embedding: bool = True,
        api_embedding_config: Dict = None,
        use_faiss: bool = False,
        faiss_index_path: str = None,
        relax_when_few_candidates: bool = True,
        **kwargs  # 兼容旧参数
    ):
        """
        初始化检索器
        
        Args:
            corpus_path: 历史query库路径
            use_semantic: 是否使用语义匹配（通过 API Embedding）
            use_synonym: 是否使用同义词标准化
            use_api_embedding: 是否使用 API 调用 embedding（DeepInfra/OpenAI 兼容接口）
            api_embedding_config: API embedding 配置，包含：
                - api_key: API Key
                - base_url: API Base URL (默认 DeepInfra)
                - model: Embedding 模型名称 (默认 Qwen/Qwen3-Embedding-8B)
                - batch_size: 每次 API 请求的样本数 (默认 50)
            use_faiss: 是否使用 FAISS 索引加速检索
            faiss_index_path: FAISS 索引保存/加载路径
        """
        self.corpus_path = corpus_path
        self.use_semantic = use_semantic
        self.use_synonym = use_synonym
        self.use_api_embedding = use_api_embedding
        self.use_faiss = use_faiss
        self.faiss_index_path = faiss_index_path
        self.relax_when_few_candidates = relax_when_few_candidates
        
        # 存储
        self.corpus: List[QueryItem] = []
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}  # 反向映射：idx -> original_id
        
        # 匹配器（支持同义词）
        self.structured_matcher = StructuredMatcher(use_synonym=use_synonym)
        
        # 语义匹配器（仅支持 API Embedding）
        if use_api_embedding or use_semantic:
            config = api_embedding_config or {}
            self.embedder = APIEmbedding(
                api_key=config.get('api_key'),
                base_url=config.get('base_url', 'https://api.deepinfra.com/v1/openai'),
                model=config.get('model', 'Qwen/Qwen3-Embedding-8B'),
                batch_size=config.get('batch_size', 50)
            )
        else:
            self.embedder = None
        
        # 索引（支持同义词）
        self.inverted_index = InvertedIndex(use_synonym=use_synonym)
        
        # FAISS 索引
        self.faiss_index: Optional[FAISSIndex] = None
        
        # 是否已构建索引
        self.index_built = False
    
    def load_corpus(self) -> int:
        """加载语料库"""
        self.corpus = []
        self.id_to_idx = {}
        self.idx_to_id = {}
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        item = QueryItem.from_dict(data)
                        self.corpus.append(item)
                        self.id_to_idx[item.id] = idx
                        self.idx_to_id[idx] = item.id
                    except json.JSONDecodeError:
                        continue
        
        print(f"已加载 {len(self.corpus)} 条历史query")
        return len(self.corpus)
    
    def build_index(self, cache_path: Optional[str] = None):
        """
        构建索引
        
        Args:
            cache_path: 索引缓存路径（用于加速重复加载）
        """
        # 尝试从缓存加载
        if cache_path and os.path.exists(cache_path):
            print(f"从缓存加载索引: {cache_path}")
            self._load_cache(cache_path)
            self.index_built = True
            
            # 如果使用 FAISS 且索引文件存在，加载 FAISS 索引
            if self.use_faiss and self.faiss_index_path:
                faiss_file = self.faiss_index_path + ".faiss"
                if os.path.exists(faiss_file):
                    print(f"从文件加载 FAISS 索引...")
                    self.faiss_index = FAISSIndex()
                    self.faiss_index.load(self.faiss_index_path)
                    
                    # 同时加载预构建的 embeddings 到 corpus
                    self._load_prebuilt_embeddings()
            return
        
        # 加载语料
        self.load_corpus()
        
        # 构建倒排索引
        print("构建倒排索引...")
        for idx, item in enumerate(self.corpus):
            self.inverted_index.add(idx, item)
        
        # 检查是否有预构建的 FAISS 索引和 embeddings
        embeddings_loaded = False
        if self.use_faiss and self.faiss_index_path:
            faiss_file = self.faiss_index_path + ".faiss"
            embeddings_file = self.faiss_index_path + ".embeddings.npy"
            
            if os.path.exists(faiss_file) and os.path.exists(embeddings_file):
                print(f"从文件加载预构建的 FAISS 索引和 embeddings...")
                # 加载 FAISS 索引
                self.faiss_index = FAISSIndex()
                self.faiss_index.load(self.faiss_index_path)
                
                # 加载预构建的 embeddings
                embeddings_loaded = self._load_prebuilt_embeddings()
        
        # 如果没有预构建的 embeddings，则重新计算
        if not embeddings_loaded:
            embeddings = None
            if self.embedder and (self.use_api_embedding or self.use_semantic):
                # 使用 API 获取 embedding
                print("使用 API 构建语义索引...")
                semantic_texts = [item.get_semantic_text() for item in self.corpus]
                embeddings = self.embedder.encode(semantic_texts)
                for idx, emb in enumerate(embeddings):
                    self.corpus[idx].embedding = emb
            
            # 构建 FAISS 索引
            if self.use_faiss and embeddings is not None:
                print("构建 FAISS 索引...")
                embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])
                self.faiss_index = FAISSIndex(embedding_dim=embedding_dim)
                ids = [item.id for item in self.corpus]
                self.faiss_index.build(embeddings, ids)
                
                # 保存 FAISS 索引
                if self.faiss_index_path:
                    self.faiss_index.save(self.faiss_index_path)
        
        self.index_built = True
        
        # 保存缓存
        if cache_path:
            self._save_cache(cache_path)
            print(f"索引已缓存到: {cache_path}")
    
    def _load_prebuilt_embeddings(self) -> bool:
        """
        从预构建的 .embeddings.npy 文件加载 embeddings 到 corpus
        
        Returns:
            是否成功加载
        """
        if not self.faiss_index_path:
            return False
        
        embeddings_file = self.faiss_index_path + ".embeddings.npy"
        ids_file = self.faiss_index_path + ".ids.json"
        
        if not os.path.exists(embeddings_file):
            return False
        
        try:
            print(f"加载预构建的 embeddings: {embeddings_file}")
            embeddings = np.load(embeddings_file)
            
            # 加载 ID 映射
            if os.path.exists(ids_file):
                with open(ids_file, 'r') as f:
                    ids = json.load(f)
                # 构建 id -> embedding 映射
                id_to_emb = {id_: emb for id_, emb in zip(ids, embeddings)}
                
                # 将 embedding 赋值给 corpus 中的对应项
                for item in self.corpus:
                    if item.id in id_to_emb:
                        item.embedding = id_to_emb[item.id]
            else:
                # 按顺序赋值
                for idx, emb in enumerate(embeddings):
                    if idx < len(self.corpus):
                        self.corpus[idx].embedding = emb
            
            print(f"已加载 {len(embeddings)} 个预构建的 embeddings")
            return True
            
        except Exception as e:
            print(f"加载预构建 embeddings 失败: {e}")
            return False
    
    def build_embedding_index(
        self,
        output_path: str,
        batch_size: int = 50,
        force_rebuild: bool = False
    ):
        """
        预构建 Embedding 索引并保存到文件
        
        用于离线构建索引，之后检索时可以直接加载，无需重新计算 embedding
        
        Args:
            output_path: 输出路径（不含后缀，会生成 .faiss 和 .meta.pkl 文件）
            batch_size: API 批量请求大小
            force_rebuild: 是否强制重建（即使文件已存在）
        """
        # 检查是否已存在
        if not force_rebuild and os.path.exists(output_path + ".faiss"):
            print(f"FAISS 索引已存在: {output_path}.faiss")
            print("如需重建，请使用 force_rebuild=True")
            return
        
        # 加载语料
        if not self.corpus:
            self.load_corpus()
        
        # 获取所有文本
        print(f"准备为 {len(self.corpus)} 条数据构建 Embedding 索引...")
        semantic_texts = [item.get_semantic_text() for item in self.corpus]
        ids = [item.id for item in self.corpus]
        
        # 计算 embeddings
        if self.embedder:
            print(f"使用 API 计算 embeddings (batch_size={batch_size})...")
            self.embedder.batch_size = batch_size
            embeddings = self.embedder.encode(semantic_texts, show_progress=True)
        else:
            raise RuntimeError("未配置 embedding 方式，请设置 use_api_embedding=True 或 use_semantic=True")
        
        # 保存 embedding 到 corpus 对象
        for idx, emb in enumerate(embeddings):
            self.corpus[idx].embedding = emb
        
        # 构建并保存 FAISS 索引
        embedding_dim = embeddings.shape[1]
        print(f"构建 FAISS 索引 (dim={embedding_dim})...")
        
        faiss_index = FAISSIndex(embedding_dim=embedding_dim)
        faiss_index.build(embeddings, ids)
        faiss_index.save(output_path)
        
        # 同时保存 embeddings 到 numpy 文件（备份）
        embeddings_path = output_path + ".embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"Embeddings 已保存: {embeddings_path}")
        
        # 保存 id 映射
        ids_path = output_path + ".ids.json"
        with open(ids_path, 'w') as f:
            json.dump(ids, f)
        print(f"IDs 已保存: {ids_path}")
        
        print(f"\n预构建完成！文件列表：")
        print(f"  - {output_path}.faiss (FAISS 索引)")
        print(f"  - {output_path}.meta.pkl (索引元数据)")
        print(f"  - {output_path}.embeddings.npy (原始向量)")
        print(f"  - {output_path}.ids.json (ID 列表)")
    
    def faiss_search(
        self,
        query_texts: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[int, float]]]:
        """
        使用 FAISS 进行快速向量检索
        
        Args:
            query_texts: 查询文本列表
            top_k: 返回 Top-K 结果
        
        Returns:
            List of List[(corpus_id, similarity_score)]
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS 索引未构建，请先调用 build_index() 或 load_faiss_index()")
        
        # 计算查询向量
        if self.embedder:
            query_embeddings = self.embedder.encode(query_texts, show_progress=False)
        else:
            raise RuntimeError("未配置 embedding 方式")
        
        # FAISS 检索
        return self.faiss_index.search(query_embeddings, top_k)
    
    def load_faiss_index(self, path: str):
        """
        加载预构建的 FAISS 索引
        
        Args:
            path: FAISS 索引路径（不含后缀）
        """
        self.faiss_index = FAISSIndex()
        self.faiss_index.load(path)
        self.use_faiss = True
    
    def _save_cache(self, cache_path: str):
        """保存索引缓存"""
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        cache_data = {
            'corpus': self.corpus,
            'id_to_idx': self.id_to_idx,
            'inverted_index': self.inverted_index
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_cache(self, cache_path: str):
        """加载索引缓存"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        self.corpus = cache_data['corpus']
        self.id_to_idx = cache_data['id_to_idx']
        self.inverted_index = cache_data['inverted_index']
        print(f"已从缓存加载 {len(self.corpus)} 条历史query")
    
    def coarse_filter(
        self,
        query: QueryItem,
        difficulty_filter: bool = True,
        skill_filter: bool = True,
        knowledge_filter: bool = True,
        max_candidates: int = 500
    ) -> List[int]:
        """
        粗筛阶段：基于结构化字段快速过滤
        
        Args:
            query: 查询query
            difficulty_filter: 是否使用难度过滤
            skill_filter: 是否使用技能过滤
            knowledge_filter: 是否使用知识领域过滤
            max_candidates: 最大候选数量
        
        Returns:
            候选query的索引列表
        """
        candidates = set(range(len(self.corpus)))
        
        # 难度过滤
        if difficulty_filter and query.difficulty:
            difficulty_candidates = self.inverted_index.get_candidates_by_difficulty(
                query.difficulty, allow_adjacent=True
            )
            if difficulty_candidates:
                candidates &= difficulty_candidates
        
        # 技能过滤（至少有1个技能重叠）
        if skill_filter and query.skills:
            skill_candidates = self.inverted_index.get_candidates_by_skills(
                query.skills, min_overlap=1
            )
            if skill_candidates:
                candidates &= skill_candidates
        
        # 知识领域过滤
        if knowledge_filter and query.knowledge:
            knowledge_candidates = self.inverted_index.get_candidates_by_knowledge(
                query.knowledge, min_overlap=1
            )
            if knowledge_candidates:
                candidates &= knowledge_candidates
        
        # 如果过滤后候选太少，放宽条件， 在写测试得时候这块给干掉*****************
        # if self.relax_when_few_candidates and len(candidates) < 10:
        #     # 只用难度过滤
        #     candidates = self.inverted_index.get_candidates_by_difficulty(
        #         query.difficulty, allow_adjacent=True
        #     )
        #     if not candidates:
        #         candidates = set(range(len(self.corpus)))
        
        # 如果候选数量超过 max_candidates，按结构化分数排序后截断
        candidate_list = list(candidates)
        if len(candidate_list) > max_candidates:
            # 计算结构化分数并排序
            scored_candidates = []
            for idx in candidate_list:
                candidate = self.corpus[idx]
                struct_scores = self.structured_matcher.compute_structured_score(query, candidate)
                scored_candidates.append((idx, struct_scores['structured_score']))
            # 按分数降序排序
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            candidate_list = [idx for idx, _ in scored_candidates[:max_candidates]]
        
        return candidate_list
    
    def fine_rank(
        self,
        query: QueryItem,
        candidate_indices: List[int],
        alpha: float = 0.4,  # 结构化分数权重
        beta: float = 0.6,   # 语义分数权重
        top_k: int = 10
    ) -> List[Tuple[int, float, Dict]]:
        """
        精排阶段：综合结构化和语义分数排序
        
        Args:
            query: 查询query
            candidate_indices: 候选索引列表
            alpha: 结构化分数权重
            beta: 语义分数权重
            top_k: 返回数量
        
        Returns:
            List of (index, score, details)
        """
        # 计算query的embedding（如果需要）
        # 使用复合文本：prompt + reasons
        if query.embedding is None and self.embedder:
            semantic_text = query.get_semantic_text()
            query.embedding = self.embedder.encode([semantic_text], show_progress=False)[0]
        
        results = []
        for idx in candidate_indices:
            candidate = self.corpus[idx]
            
            # 结构化分数
            struct_scores = self.structured_matcher.compute_structured_score(query, candidate)
            struct_score = struct_scores['structured_score']
            
            # 语义分数（基于 prompt + reasons 的复合embedding）
            semantic_score = 0.0
            if self.embedder and query.embedding is not None and candidate.embedding is not None:
                semantic_score = self.embedder.cosine_similarity(query.embedding, candidate.embedding)
            
            # 综合分数
            if self.embedder:
                final_score = alpha * struct_score + beta * semantic_score
            else:
                final_score = struct_score
            
            details = {
                **struct_scores,
                'semantic_score': semantic_score,
                'final_score': final_score
            }
            
            results.append((idx, final_score, details))
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def retrieve(
        self,
        query_data: Dict,
        top_k: int = 10,
        return_details: bool = True,
        retrieval_mode: str = RetrievalMode.HYBRID,
        coarse_threshold: float = 0.0,
        coarse_top_k: int = 500,
        fine_top_k: Optional[int] = None,
        coarse_filter_config: Optional[Dict] = None,
        fine_rank_config: Optional[Dict] = None
    ) -> List[Dict]:
        """
        检索接口
        
        Args:
            query_data: 查询数据，包含prompt和decomposition
            top_k: 最终返回数量
            return_details: 是否返回详细分数
            retrieval_mode: 检索模式
                - "coarse": 仅粗排（基于结构化字段S/K/D）
                - "fine": 仅精排（全量语义匹配，不做粗筛）
                - "hybrid": 粗精混合（默认，先粗筛后精排）
            coarse_threshold: 粗排阈值，结构化分数低于此值的直接过滤（0.0-1.0）
            coarse_top_k: 粗排阶段保留的候选数量
            fine_top_k: 精排阶段返回数量（默认等于top_k）
            coarse_filter_config: 粗筛配置
            fine_rank_config: 精排配置
        
        Returns:
            检索结果列表
        """
        if not self.index_built:
            raise RuntimeError("请先调用 build_index() 构建索引")
        
        # 设置精排top_k
        if fine_top_k is None:
            fine_top_k = top_k
        
        # 解析query
        query = QueryItem.from_dict(query_data)
        
        # 根据检索模式执行不同逻辑
        if retrieval_mode == RetrievalMode.COARSE:
            # 仅粗排：只用结构化字段过滤和排序
            ranked_results = self._coarse_only_retrieve(
                query, top_k, coarse_threshold, coarse_top_k, coarse_filter_config
            )
        elif retrieval_mode == RetrievalMode.FINE:
            # 仅精排：全量数据进行语义匹配
            ranked_results = self._fine_only_retrieve(query, fine_top_k, fine_rank_config)
        else:  # HYBRID
            # 粗精混合：先粗筛再精排
            ranked_results = self._hybrid_retrieve(
                query, top_k, coarse_threshold, coarse_top_k, fine_top_k,
                coarse_filter_config, fine_rank_config
            )
        
        # 格式化结果
        results = []
        for idx, score, details in ranked_results:
            item = self.corpus[idx]
            result = {
                'id': item.id,
                'prompt': item.prompt,
                'eval_name': item.eval_name,
                'skills': list(item.skills),
                'knowledge': list(item.knowledge),
                'difficulty': item.difficulty,
                's_reason': item.s_reason,
                'k_reason': item.k_reason,
                'd_reason': item.d_reason,
                'score': score
            }
            if return_details:
                result['score_details'] = details
            results.append(result)
        
        return results
    
    def _coarse_only_retrieve(
        self,
        query: QueryItem,
        top_k: int,
        coarse_threshold: float = 0.0,
        coarse_top_k: int = 500,
        coarse_filter_config: Optional[Dict] = None
    ) -> List[Tuple[int, float, Dict]]:
        """
        仅粗排检索：基于结构化字段进行过滤和排序
        
        Args:
            query: 查询
            top_k: 最终返回数量
            coarse_threshold: 粗排分数阈值，低于此值的过滤掉
            coarse_top_k: 粗排阶段保留的候选数量
            coarse_filter_config: 粗筛配置
        """
        coarse_config = coarse_filter_config or {}
        coarse_config['max_candidates'] = max(coarse_config.get('max_candidates', coarse_top_k), top_k * 10)
        candidate_indices = self.coarse_filter(query, **coarse_config)
        
        # 使用结构化分数排序
        results = []
        for idx in candidate_indices:
            candidate = self.corpus[idx]
            struct_scores = self.structured_matcher.compute_structured_score(query, candidate)
            score = struct_scores['structured_score']
            
            # 应用阈值过滤
            if score < coarse_threshold:
                continue
            
            details = {
                **struct_scores,
                'semantic_score': 0.0,
                'final_score': score,
                'mode': 'coarse'
            }
            results.append((idx, score, details))
        
        # 按结构化分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _fine_only_retrieve(
        self,
        query: QueryItem,
        top_k: int,
        fine_rank_config: Optional[Dict] = None
    ) -> List[Tuple[int, float, Dict]]:
        """
        仅精排检索：全量数据进行语义匹配（不做粗筛）
        """
        # 全量候选
        all_indices = list(range(len(self.corpus)))
        
        # 精排
        fine_config = fine_rank_config or {}
        fine_config['top_k'] = top_k
        # 精排模式下提高语义权重
        if 'alpha' not in fine_config:
            fine_config['alpha'] = 0.2
        if 'beta' not in fine_config:
            fine_config['beta'] = 0.8
        
        results = self.fine_rank(query, all_indices, **fine_config)
        
        # 标记模式
        for idx, score, details in results:
            details['mode'] = 'fine'
        
        return results
    
    def _hybrid_retrieve(
        self,
        query: QueryItem,
        top_k: int,
        coarse_threshold: float = 0.0,
        coarse_top_k: int = 500,
        fine_top_k: Optional[int] = None,
        coarse_filter_config: Optional[Dict] = None,
        fine_rank_config: Optional[Dict] = None
    ) -> List[Tuple[int, float, Dict]]:
        """
        粗精混合检索：先粗筛后精排（默认模式）
        
        Args:
            query: 查询
            top_k: 最终返回数量
            coarse_threshold: 粗排分数阈值
            coarse_top_k: 粗排阶段保留的候选数量
            fine_top_k: 精排阶段返回数量
            coarse_filter_config: 粗筛配置
            fine_rank_config: 精排配置
        """
        if fine_top_k is None:
            fine_top_k = top_k
        
        # 粗筛
        coarse_config = coarse_filter_config or {}
        coarse_config['max_candidates'] = coarse_top_k
        candidate_indices = self.coarse_filter(query, **coarse_config)
        
        # 粗排阶段应用阈值过滤
        if coarse_threshold > 0:
            filtered_indices = []
            for idx in candidate_indices:
                candidate = self.corpus[idx]
                struct_scores = self.structured_matcher.compute_structured_score(query, candidate)
                if struct_scores['structured_score'] >= coarse_threshold:
                    filtered_indices.append(idx)
            candidate_indices = filtered_indices
        
        # 精排
        fine_config = fine_rank_config or {}
        fine_config['top_k'] = fine_top_k
        
        results = self.fine_rank(query, candidate_indices, **fine_config)
        
        # 标记模式
        for idx, score, details in results:
            details['mode'] = 'hybrid'
        
        return results[:top_k]
    
    def batch_retrieve(
        self,
        queries_path: str,
        output_path: str,
        top_k: int = 10,
        retrieval_mode: str = RetrievalMode.HYBRID,
        **kwargs
    ):
        """
        批量检索
        
        Args:
            queries_path: 查询文件路径
            output_path: 输出文件路径
            top_k: 每个query返回的数量
            retrieval_mode: 检索模式 ("coarse", "fine", "hybrid")
        """
        # 加载查询
        queries = []
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
        
        print(f"开始批量检索 {len(queries)} 条query...")
        print(f"检索模式: {retrieval_mode}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 检索
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, query_data in enumerate(queries):
                results = self.retrieve(
                    query_data, 
                    top_k=top_k, 
                    retrieval_mode=retrieval_mode,
                    **kwargs
                )
                
                output = {
                    'query_id': query_data.get('id'),
                    'query_prompt': query_data.get('prompt'),
                    'retrieval_mode': retrieval_mode,
                    'retrieved': results
                }
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
                
                if (i + 1) % 100 == 0:
                    print(f"进度: {i + 1}/{len(queries)}")
        
        print(f"批量检索完成！结果已保存到: {output_path}")


def demo():
    """演示用法"""
    # 示例数据（包含reason字段）
    query_data = {
        "id": 2740,
        "prompt": "Zoe wants to go on the field trip to Washington DC with her middle school this spring and the cost is $485. Her grandma gave her $250 toward her fees and she must earn the rest by selling candy bars. She makes $1.25 for every candy bar she sells. How many candy bars does Zoe need to sell to earn the money for the trip?",
        "decomposition": {
            "S": ["mathematics", "arithmetic", "problem solving"],
            "S_reason": "The query requires calculating the number of candy bars needed to cover a cost difference using basic arithmetic.",
            "K": ["basic finance", "elementary mathematics"],
            "K_reason": "Understanding of cost, money, and simple multiplication/division is needed to solve the problem.",
            "D": "D1",
            "D_reason": "The problem involves a single-step calculation with mild reasoning to find the number of candy bars."
        }
    }
    
    print("=" * 60)
    print("Query Retrieval System Demo")
    print("=" * 60)
    
    # 路径配置
    corpus_path = "./Dataset/corpus/IID/merged_query_decomposition_train.jsonl"
    cache_path = "./CacheFile/index_cache.pkl"
    
    # 检查文件是否存在
    if not os.path.exists(corpus_path):
        print(f"语料库文件不存在: {corpus_path}")
        print("请确保路径正确后重试")
        return
    
    # 初始化检索器
    print("\n[1] 初始化检索器...")
    retriever = QueryRetriever(
        corpus_path=corpus_path,
        use_semantic=False  # 设为False可禁用语义匹配
    )
    
    # 构建索引
    print("\n[2] 构建索引...")
    retriever.build_index(cache_path=cache_path)
    
    # 演示不同检索模式
    modes = [
        (RetrievalMode.COARSE, "仅粗排 (结构化字段)"),
        # (RetrievalMode.FINE, "仅精排 (全量语义)"),
        # (RetrievalMode.HYBRID, "粗精混合 (默认)")
    ]
    
    for mode, mode_name in modes:
        print(f"\n{'=' * 60}")
        print(f"[检索模式] {mode_name}")
        print("=" * 60)
        
        results = retriever.retrieve(query_data, top_k=3, retrieval_mode=mode)
        
        for i, result in enumerate(results):
            print(f"\n#{i+1} [Score: {result['score']:.4f}]")
            print(f"   ID: {result['id']} | Eval: {result['eval_name']}")
            print(f"   Prompt: {result['prompt'][:80]}...")
            print(f"   S: {result['skills']}")
            print(f"   K: {result['knowledge']}")
            print(f"   D: {result['difficulty']}")
            if 'score_details' in result:
                d = result['score_details']
                print(f"   Scores: struct={d['structured_score']:.3f}, "
                      f"semantic={d['semantic_score']:.3f}, "
                      f"final={d['final_score']:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)


if __name__ == '__main__':
    demo()

