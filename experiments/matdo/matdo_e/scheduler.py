"""
MATDO-E vLLM调度器集成

模拟在vLLM内部的关键修改点：
1. vllm/core/scheduler.py: _schedule() - 动态配置求解
2. vllm/core/block_manager.py: can_allocate() - 套利标记
3. vllm/engine/llm_engine.py: 调度循环 - 预取触发

核心逻辑:
  rho = get_gpu_cache_usage()
  if rho > 0.95:
      config = solver.solve(rho)  # 四维优化
      request.is_arbitrage = True
      engram_manager.prefetch(request.id, config.E)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.matdo.common.config import config
from experiments.matdo.matdo_e.solver import MATDOESolver, OptimalConfig
from experiments.matdo.matdo_e.engram_manager import EngramManager, DummyEngramManager


@dataclass
class MATDORequest:
    """模拟vLLM中的序列组请求"""
    request_id: str
    prompt_len: int
    max_new_tokens: int = 100
    
    # MATDO-E动态配置
    optimal_config: Optional[OptimalConfig] = None
    is_arbitrage: bool = False
    
    # 统计
    scheduled_at: float = field(default_factory=time.time)
    first_token_at: Optional[float] = None
    completed_at: Optional[float] = None


class BlockManagerMock:
    """
    模拟vLLM BlockManager
    
    管理GPU KV Cache Blocks的分配。
    """
    
    def __init__(self, 
                 num_gpu_blocks: int = 1024,
                 block_size: int = 16):  # 1 Block = 16 tokens (论文建议)
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.block_size_bytes = block_size * config.d_model * 2  # FP16
        
        # 已分配blocks
        self.allocated_blocks: Dict[str, int] = {}
        self.total_allocated = 0
        
    def can_allocate(self, request: MATDORequest) -> bool:
        """
        检查是否可以分配blocks给请求
        
        MATDO-E修改: 即使物理blocks不足，只要不满足"彻底崩溃"条件，
        标记为is_arbitrage=True继续调度。
        """
        required_blocks = (request.prompt_len + request.max_new_tokens) // self.block_size + 1
        available = self.num_gpu_blocks - self.total_allocated
        
        return required_blocks <= available
    
    def allocate(self, request: MATDORequest) -> int:
        """分配blocks给请求"""
        required_blocks = (request.prompt_len + request.max_new_tokens) // self.block_size + 1
        
        self.allocated_blocks[request.request_id] = required_blocks
        self.total_allocated += required_blocks
        
        return required_blocks
    
    def free(self, request_id: str):
        """释放blocks"""
        if request_id in self.allocated_blocks:
            self.total_allocated -= self.allocated_blocks[request_id]
            del self.allocated_blocks[request_id]
    
    def get_gpu_cache_usage(self) -> float:
        """获取当前GPU缓存使用率 rho"""
        return self.total_allocated / self.num_gpu_blocks
    
    def get_available_blocks(self) -> int:
        """获取可用blocks数"""
        return self.num_gpu_blocks - self.total_allocated


class MATDOEScheduler:
    """
    MATDO-E调度器
    
    集成求解器和Engram管理器，实现论文中的动态配置策略。
    """
    
    def __init__(self,
                 num_gpu_blocks: int = 1024,
                 block_size: int = 16,
                 enable_arbitrage: bool = True):
        """
        Args:
            num_gpu_blocks: GPU block总数 (对应80GB HBM)
            block_size: 每个block的token数
            enable_arbitrage: 是否启用异构套利
        """
        self.block_manager = BlockManagerMock(num_gpu_blocks, block_size)
        self.solver = MATDOESolver()
        
        # Engram管理器
        if enable_arbitrage and self.solver.arbitrage_feasible:
            self.engram_manager = EngramManager()
        else:
            self.engram_manager = DummyEngramManager()
            
        self.enable_arbitrage = enable_arbitrage
        
        # 请求队列
        self.waiting: List[MATDORequest] = []
        self.running: Dict[str, MATDORequest] = {}
        self.completed: List[MATDORequest] = []
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'arbitrage_requests': 0,
            'rejected_requests': 0,
            'avg_rho': 0,
        }
        
    def add_request(self, request: MATDORequest):
        """添加请求到等待队列"""
        self.waiting.append(request)
        self.stats['total_requests'] += 1
        
    def _schedule(self) -> List[MATDORequest]:
        """
        核心调度逻辑 (对应vLLM scheduler._schedule)
        
        1. 监控显存压力 rho
        2. 对高压力请求求解最优配置
        3. 预触发Engram异步检索
        """
        scheduled = []
        rho = self.block_manager.get_gpu_cache_usage()
        
        # 更新平均rho
        self.stats['avg_rho'] = 0.95 * self.stats['avg_rho'] + 0.05 * rho
        
        # 按顺序调度等待队列
        i = 0
        while i < len(self.waiting):
            request = self.waiting[i]
            
            # 检查是否能分配
            if not self.block_manager.can_allocate(request):
                # MATDO-E: 检查是否可以通过套利继续
                if self.enable_arbitrage and rho < 0.995:  # 不到彻底崩溃
                    # 标记为套利模式，强制调度
                    request.is_arbitrage = True
                else:
                    # 彻底崩溃，拒绝请求
                    self.stats['rejected_requests'] += 1
                    i += 1
                    continue
            
            # 高压力时求解最优配置
            if rho > 0.90 or request.is_arbitrage:
                opt_config = self.solver.solve(rho)
                request.optimal_config = opt_config
                request.is_arbitrage = opt_config.is_arbitrage
                
                if request.is_arbitrage:
                    self.stats['arbitrage_requests'] += 1
                    # 限制context长度以匹配M
                    max_tokens = opt_config.M * self.block_manager.block_size
                    request.max_new_tokens = min(request.max_new_tokens, max_tokens)
                    
                    # 预触发Engram检索
                    if opt_config.E > 0:
                        self.engram_manager.prefetch(request.request_id, opt_config.E)
            
            # 分配资源
            self.block_manager.allocate(request)
            
            # 移动到运行队列
            scheduled.append(request)
            self.running[request.request_id] = request
            self.waiting.pop(i)
            
            # 更新rho
            rho = self.block_manager.get_gpu_cache_usage()
            
        return scheduled
    
    def step(self) -> Dict:
        """
        执行一步调度 (对应vLLM LLMEngine.step)
        
        Returns:
            调度结果统计
        """
        # 1. 调度新请求
        scheduled = self._schedule()
        
        # 2. 模拟运行中的请求完成
        completed_this_step = []
        for req_id in list(self.running.keys()):
            request = self.running[req_id]
            
            # 模拟生成延迟 (简化模型)
            if request.optimal_config:
                # TTA步数增加延迟
                delay = 0.001 * request.optimal_config.T
            else:
                delay = 0.001
                
            # 模拟完成
            if np.random.random() < 0.3:  # 30%概率完成
                request.completed_at = time.time()
                self.completed.append(request)
                completed_this_step.append(request)
                self.block_manager.free(req_id)
                del self.running[req_id]
                
                # 释放Engram
                self.engram_manager.release_buffer(req_id)
        
        return {
            'scheduled': len(scheduled),
            'completed': len(completed_this_step),
            'running': len(self.running),
            'waiting': len(self.waiting),
            'rho': self.block_manager.get_gpu_cache_usage(),
        }
    
    def run_simulation(self, 
                       requests: List[MATDORequest],
                       num_steps: int = 100) -> Dict:
        """
        运行完整模拟
        
        Args:
            requests: 请求列表
            num_steps: 最大步数
            
        Returns:
            模拟统计
        """
        # 添加所有请求
        for req in requests:
            self.add_request(req)
            
        # 运行调度循环
        history = []
        for step in range(num_steps):
            if not self.waiting and not self.running:
                break
                
            result = self.step()
            history.append(result)
            
        # 计算指标
        total = len(self.completed)
        arbitrage = sum(1 for r in self.completed if r.is_arbitrage)
        
        latencies = []
        for r in self.completed:
            if r.completed_at and r.scheduled_at:
                latencies.append(r.completed_at - r.scheduled_at)
        
        return {
            'total_requests': self.stats['total_requests'],
            'completed': total,
            'rejected': self.stats['rejected_requests'],
            'arbitrage_used': arbitrage,
            'arbitrage_ratio': arbitrage / total if total > 0 else 0,
            'avg_latency_ms': np.mean(latencies) * 1000 if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000 if latencies else 0,
            'peak_rho': max(h['rho'] for h in history) if history else 0,
            'avg_rho': self.stats['avg_rho'],
            'history': history,
        }
    
    def get_stats(self) -> Dict:
        """获取调度器统计"""
        return {
            **self.stats,
            'current_rho': self.block_manager.get_gpu_cache_usage(),
            'running': len(self.running),
            'waiting': len(self.waiting),
            'engram_stats': self.engram_manager.get_stats(),
        }


if __name__ == "__main__":
    # 测试调度器
    print("=" * 70)
    print("MATDO-E Scheduler Test")
    print("=" * 70)
    
    # 创建调度器
    scheduler = MATDOEScheduler(num_gpu_blocks=512, enable_arbitrage=True)
    
    # 生成测试请求 (模拟高负载)
    np.random.seed(42)
    test_requests = []
    for i in range(50):
        req = MATDORequest(
            request_id=f"req_{i}",
            prompt_len=np.random.randint(512, 4096),
            max_new_tokens=np.random.randint(50, 200)
        )
        test_requests.append(req)
    
    print(f"\nGenerated {len(test_requests)} test requests")
    print(f"Block manager: {scheduler.block_manager.num_gpu_blocks} blocks")
    print(f"Arbitrage feasible: {scheduler.solver.arbitrage_feasible}")
    
    # 运行模拟
    print("\nRunning simulation...")
    results = scheduler.run_simulation(test_requests, num_steps=200)
    
    # 输出结果
    print(f"\n{'='*70}")
    print("Simulation Results")
    print(f"{'='*70}")
    print(f"Total requests: {results['total_requests']}")
    print(f"Completed: {results['completed']}")
    print(f"Rejected: {results['rejected']}")
    print(f"Arbitrage used: {results['arbitrage_used']} ({results['arbitrage_ratio']*100:.1f}%)")
    print(f"Avg latency: {results['avg_latency_ms']:.2f} ms")
    print(f"P99 latency: {results['p99_latency_ms']:.2f} ms")
    print(f"Peak rho: {results['peak_rho']:.4f}")
    print(f"Avg rho: {results['avg_rho']:.4f}")
    
    print("\n✓ Scheduler test completed!")
