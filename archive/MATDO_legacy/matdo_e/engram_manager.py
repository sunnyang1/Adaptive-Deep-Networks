"""
Engram Manager - DRAM端静态知识管理器

实现功能：
1. Faiss HNSW索引管理 (Wikipedia 2023 Dump)
2. 异步检索 (ThreadPoolExecutor)
3. 预取缓存 (隐藏PCIe延迟)
4. Engram到GPU的传输

论文参考: §4 Heterogeneous Resource Arbitrage
"""

import numpy as np
import torch
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.matdo.common.config import config


@dataclass
class EngramBuffer:
    """Engram检索结果缓冲区"""
    request_id: str
    keys: torch.Tensor    # Shape: [E, L_embedding]
    values: torch.Tensor  # Shape: [E, L_embedding]  
    retrieved_at: float
    
    def is_valid(self, timeout_ms: float = 1000) -> bool:
        """检查缓冲区是否在有效期内"""
        return (time.time() - self.retrieved_at) * 1000 < timeout_ms


class MockFaissIndex:
    """
    模拟Faiss HNSW索引
    
    在实际部署中，这应该替换为真实的faiss.IndexHNSWFlat
    """
    
    def __init__(self, dim: int = 384, n_entries: int = 128000):
        self.dim = dim
        self.n_entries = n_entries
        # 模拟随机embeddings
        np.random.seed(42)
        self.embeddings = np.random.randn(n_entries, dim).astype(np.float32)
        # L2归一化
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟检索：返回k个最近邻
        
        Returns:
            distances: [batch, k]
            indices: [batch, k]
        """
        # 计算L2距离
        distances = np.linalg.norm(self.embeddings - query, axis=1)
        # 获取top-k
        top_k_idx = np.argsort(distances)[:k]
        top_k_dist = distances[top_k_idx]
        return top_k_dist.reshape(1, -1), top_k_idx.reshape(1, -1)


class EngramManager:
    """
    Engram管理器
    
    负责DRAM端Engram的异步检索和缓存管理。
    核心目标：通过预取隐藏检索延迟 (Proposition 4.1: tau_ret < tau_pre)
    """
    
    def __init__(self, 
                 index_path: Optional[str] = None,
                 embedding_dim: int = 384,
                 n_entries: int = 128000,
                 max_workers: int = 4,
                 device: str = "cuda"):
        """
        Args:
            index_path: Faiss索引路径，None则使用模拟索引
            embedding_dim: Embedding维度
            n_entries: Engram条目数
            max_workers: 异步检索线程数
            device: GPU设备
        """
        self.embedding_dim = embedding_dim
        self.n_entries = n_entries
        self.device = device
        
        # 加载索引
        if index_path and Path(index_path).exists():
            # 实际部署中加载真实Faiss索引
            try:
                import faiss
                self.index = faiss.read_index(index_path)
                print(f"Loaded Faiss index from {index_path}")
            except ImportError:
                print("Faiss not available, using mock index")
                self.index = MockFaissIndex(embedding_dim, n_entries)
        else:
            self.index = MockFaissIndex(embedding_dim, n_entries)
            
        # 异步检索线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 预取缓存: request_id -> Future[EngramBuffer]
        self._prefetch_cache: Dict[str, Future] = {}
        self._ready_cache: Dict[str, EngramBuffer] = {}
        self._cache_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'async_success': 0,
            'avg_retrieval_time_ms': 0,
        }
        
    def prefetch(self, request_id: str, E: int, query_embedding: Optional[np.ndarray] = None):
        """
        预触发Engram异步检索
        
        在vLLM调度器中调用，确保在GPU forward前数据已就绪。
        
        Args:
            request_id: 请求ID
            E: 需要检索的Engram数量
            query_embedding: 查询embedding，None则使用随机查询
        """
        if E <= 0:
            return
            
        with self._cache_lock:
            # 检查是否已在缓存中
            if request_id in self._ready_cache:
                return
            if request_id in self._prefetch_cache:
                return
                
            # 提交异步任务
            future = self.executor.submit(self._retrieve_engram, request_id, E, query_embedding)
            self._prefetch_cache[request_id] = future
            
    def _retrieve_engram(self, request_id: str, E: int, 
                         query_embedding: Optional[np.ndarray] = None) -> EngramBuffer:
        """
        实际执行Engram检索 (在后台线程中运行)
        
        Returns:
            EngramBuffer: 检索结果
        """
        start_time = time.time()
        
        # 模拟或执行真实检索
        if query_embedding is None:
            query_embedding = np.random.randn(1, self.embedding_dim).astype(np.float32)
            query_embedding /= np.linalg.norm(query_embedding)
            
        # 检索top-E个Engram
        distances, indices = self.index.search(query_embedding, min(E, self.n_entries))
        
        # 获取对应的embeddings作为keys
        if isinstance(self.index, MockFaissIndex):
            keys_np = self.index.embeddings[indices[0]]
        else:
            # 真实Faiss索引需要另外存储向量
            keys_np = np.random.randn(len(indices[0]), self.embedding_dim).astype(np.float32)
            
        # 模拟value embeddings (可以不同，这里简化相同)
        values_np = keys_np.copy()
        
        # 转换为torch tensor并移到GPU
        keys = torch.from_numpy(keys_np).to(self.device)
        values = torch.from_numpy(values_np).to(self.device)
        
        retrieval_time = (time.time() - start_time) * 1000  # ms
        
        # 更新统计
        self.stats['total_requests'] += 1
        self.stats['avg_retrieval_time_ms'] = (
            0.95 * self.stats['avg_retrieval_time_ms'] + 0.05 * retrieval_time
        )
        
        return EngramBuffer(
            request_id=request_id,
            keys=keys,
            values=values,
            retrieved_at=time.time()
        )
    
    def get_buffer(self, request_id: str, timeout_ms: float = 50) -> Optional[EngramBuffer]:
        """
        获取预取好的Engram缓冲区
        
        在vLLM attention层调用，应该在GPU forward时立即返回。
        
        Args:
            request_id: 请求ID
            timeout_ms: 等待超时时间
            
        Returns:
            EngramBuffer或None
        """
        # 检查就绪缓存
        with self._cache_lock:
            if request_id in self._ready_cache:
                buffer = self._ready_cache[request_id]
                if buffer.is_valid():
                    self.stats['cache_hits'] += 1
                    return buffer
                else:
                    del self._ready_cache[request_id]
                    
            # 检查预取缓存
            if request_id in self._prefetch_cache:
                future = self._prefetch_cache[request_id]
            else:
                return None
                
        # 等待异步任务完成
        try:
            buffer = future.result(timeout=timeout_ms/1000)
            with self._cache_lock:
                self._ready_cache[request_id] = buffer
                del self._prefetch_cache[request_id]
            self.stats['async_success'] += 1
            return buffer
        except Exception as e:
            print(f"Engram retrieval failed for {request_id}: {e}")
            return None
            
    def release_buffer(self, request_id: str):
        """释放请求对应的Engram缓冲区"""
        with self._cache_lock:
            self._ready_cache.pop(request_id, None)
            self._prefetch_cache.pop(request_id, None)
            
    def clear_cache(self):
        """清空所有缓存"""
        with self._cache_lock:
            self._ready_cache.clear()
            self._prefetch_cache.clear()
            
    def get_stats(self) -> Dict:
        """获取统计信息"""
        cache_hit_rate = 0
        if self.stats['total_requests'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['total_requests']
            
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'prefetch_pending': len(self._prefetch_cache),
            'ready_buffers': len(self._ready_cache),
        }


class DummyEngramManager:
    """虚拟Engram管理器 (用于非套利模式)"""
    
    def prefetch(self, *args, **kwargs):
        pass
        
    def get_buffer(self, *args, **kwargs):
        return None
        
    def release_buffer(self, *args, **kwargs):
        pass
        
    def get_stats(self):
        return {}


if __name__ == "__main__":
    # 测试EngramManager
    print("=" * 60)
    print("Engram Manager Test")
    print("=" * 60)
    
    manager = EngramManager()
    
    # 测试异步预取
    print("\nTesting async prefetch...")
    request_ids = [f"req_{i}" for i in range(5)]
    
    for rid in request_ids:
        manager.prefetch(rid, E=128)
        print(f"  Prefetch triggered for {rid}")
        
    # 模拟GPU forward延迟
    time.sleep(0.02)
    
    # 获取缓冲区
    print("\nGetting buffers...")
    for rid in request_ids:
        buffer = manager.get_buffer(rid)
        if buffer:
            print(f"  {rid}: keys shape = {buffer.keys.shape}, "
                  f"retrieval took {(time.time()-buffer.retrieved_at)*1000:.2f}ms ago")
        else:
            print(f"  {rid}: buffer not ready")
            
    # 统计
    print(f"\nStats: {manager.get_stats()}")
