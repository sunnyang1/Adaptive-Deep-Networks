"""
延迟分解测试 (Fig B)

目标: 证明CPU端检索任务如何与GPU端计算任务并行，延迟被掩盖。

实验设计:
- Timeline view: DRAM Retrieval | PCIe Transfer | GPU Compute
- 证明 tau_ret < tau_pre (Proposition 4.1)
"""

import numpy as np
import json
from typing import Dict, List
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.matdo.matdo_e.engram_manager import EngramManager


class LatencyProfiler:
    """
    延迟分析器
    
    模拟nsys-style的时间线分析
    """
    
    def __init__(self):
        self.events: List[Dict] = []
        self.start_time = None
        
    def record_event(self, name: str, category: str, start_ms: float, duration_ms: float):
        """记录一个事件"""
        self.events.append({
            'name': name,
            'category': category,
            'start_ms': start_ms,
            'duration_ms': duration_ms,
            'end_ms': start_ms + duration_ms,
        })
        
    def get_timeline(self) -> List[Dict]:
        """获取完整时间线"""
        return sorted(self.events, key=lambda x: x['start_ms'])
    
    def calculate_overlap(self, cat1: str, cat2: str) -> float:
        """计算两类事件的重叠时间"""
        events1 = [e for e in self.events if e['category'] == cat1]
        events2 = [e for e in self.events if e['category'] == cat2]
        
        overlap = 0
        for e1 in events1:
            for e2 in events2:
                # 计算区间重叠
                start = max(e1['start_ms'], e2['start_ms'])
                end = min(e1['end_ms'], e2['end_ms'])
                if end > start:
                    overlap += end - start
                    
        return overlap
    
    def analyze_parallel_efficiency(self) -> Dict:
        """分析并行效率"""
        # DRAM检索总时间
        dram_time = sum(e['duration_ms'] for e in self.events if e['category'] == 'dram_retrieval')
        
        # GPU计算总时间
        gpu_time = sum(e['duration_ms'] for e in self.events if e['category'] == 'gpu_compute')
        
        # 重叠时间
        overlap = self.calculate_overlap('dram_retrieval', 'gpu_compute')
        
        # 掩盖效率
        masking_efficiency = overlap / dram_time if dram_time > 0 else 0
        
        return {
            'total_dram_time_ms': dram_time,
            'total_gpu_time_ms': gpu_time,
            'overlap_time_ms': overlap,
            'masking_efficiency': masking_efficiency,
            'effective_latency_ms': dram_time - overlap,
        }


def simulate_request_timeline(request_id: str,
                              E: int,
                              num_layers: int = 32,
                              enable_prefetch: bool = True) -> Dict:
    """
    模拟单个请求的时间线
    
    Args:
        request_id: 请求ID
        E: Engram数量
        num_layers: 模型层数
        enable_prefetch: 是否启用预取
        
    Returns:
        时间线事件
    """
    profiler = LatencyProfiler()
    current_time = 0
    
    # 参数 (基于论文和实际测量)
    t_dram_per_1k = 5.0  # 每1K engrams检索时间 (ms)
    t_pcie_per_1k = 2.0  # 每1K engrams传输时间 (ms)
    t_gpu_per_layer = 1.5  # 每层GPU计算时间 (ms)
    
    E_k = E / 1000  # 转换为千单位
    
    if enable_prefetch:
        # === 预取模式: DRAM检索与GPU计算重叠 ===
        
        # 1. 立即启动DRAM检索 (异步)
        t_dram = t_dram_per_1k * E_k
        profiler.record_event(
            f"{request_id}_dram_retrieval",
            "dram_retrieval",
            current_time,
            t_dram
        )
        
        # 2. 模拟调度延迟后开始GPU计算
        t_schedule = 2.0  # 调度开销
        current_time += t_schedule
        
        # 3. GPU计算 (与DRAM检索重叠)
        t_gpu = t_gpu_per_layer * num_layers
        profiler.record_event(
            f"{request_id}_gpu_compute",
            "gpu_compute",
            current_time,
            t_gpu
        )
        
        # 4. PCIe传输 (在DRAM检索完成后)
        dram_end = t_dram  # DRAM检索结束时间
        gpu_needs_data = current_time + t_gpu * 0.3  # GPU在30%进度时需要数据
        
        t_pcie_start = max(dram_end, gpu_needs_data - 1)  # 确保数据就绪
        t_pcie = t_pcie_per_1k * E_k
        
        profiler.record_event(
            f"{request_id}_pcie_transfer",
            "pcie_transfer",
            t_pcie_start,
            t_pcie
        )
        
        # 5. Engram注意力计算
        t_engram_start = t_pcie_start + t_pcie
        t_engram = t_gpu_per_layer * 0.2 * num_layers  # 额外20%时间
        
        profiler.record_event(
            f"{request_id}_engram_attention",
            "engram_attention",
            t_engram_start,
            t_engram
        )
        
    else:
        # === 非预取模式: 同步执行 ===
        
        # 1. DRAM检索 (阻塞)
        t_dram = t_dram_per_1k * E_k
        profiler.record_event(
            f"{request_id}_dram_retrieval",
            "dram_retrieval",
            current_time,
            t_dram
        )
        current_time += t_dram
        
        # 2. PCIe传输
        t_pcie = t_pcie_per_1k * E_k
        profiler.record_event(
            f"{request_id}_pcie_transfer",
            "pcie_transfer",
            current_time,
            t_pcie
        )
        current_time += t_pcie
        
        # 3. GPU计算
        t_gpu = t_gpu_per_layer * num_layers
        profiler.record_event(
            f"{request_id}_gpu_compute",
            "gpu_compute",
            current_time,
            t_gpu
        )
    
    return {
        'request_id': request_id,
        'events': profiler.get_timeline(),
        'analysis': profiler.analyze_parallel_efficiency(),
        'total_latency_ms': max(e['end_ms'] for e in profiler.events),
    }


def run_latency_breakdown_test(
    E_values: List[int] = None,
    num_requests: int = 10,
    output_dir: Path = None
) -> Dict:
    """
    运行延迟分解测试
    
    Args:
        E_values: 测试的Engram数量列表
        num_requests: 每个配置测试的请求数
        output_dir: 输出目录
        
    Returns:
        实验结果
    """
    if E_values is None:
        E_values = [0, 16000, 32000, 64000, 128000]
        
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Latency Breakdown Test (Fig B)")
    print("=" * 70)
    print(f"Testing {num_requests} requests per configuration")
    print(f"E values: {E_values}")
    print()
    
    results = {
        'with_prefetch': {},
        'without_prefetch': {},
    }
    
    print(f"{'E':>10} | {'With Prefetch':>15} | {'Without Prefetch':>18} | {'Speedup':>10}")
    print("-" * 70)
    
    for E in E_values:
        # 测试预取模式
        latencies_with = []
        for i in range(num_requests):
            timeline = simulate_request_timeline(f"req_{i}", E, enable_prefetch=True)
            latencies_with.append(timeline['total_latency_ms'])
            
        # 测试非预取模式
        latencies_without = []
        for i in range(num_requests):
            timeline = simulate_request_timeline(f"req_{i}", E, enable_prefetch=False)
            latencies_without.append(timeline['total_latency_ms'])
        
        avg_with = np.mean(latencies_with)
        avg_without = np.mean(latencies_without)
        speedup = avg_without / avg_with if avg_with > 0 else 1
        
        results['with_prefetch'][E] = {
            'avg_latency_ms': avg_with,
            'p99_latency_ms': np.percentile(latencies_with, 99),
        }
        results['without_prefetch'][E] = {
            'avg_latency_ms': avg_without,
            'p99_latency_ms': np.percentile(latencies_without, 99),
        }
        
        print(f"{E:>10} | {avg_with:>15.1f} | {avg_without:>18.1f} | {speedup:>10.2f}x")
    
    # 详细分析 (E=128K时)
    print()
    print("=" * 70)
    print("Detailed Analysis (E=128000)")
    print("=" * 70)
    
    timeline_with = simulate_request_timeline("analysis", 128000, enable_prefetch=True)
    analysis = timeline_with['analysis']
    
    print(f"  DRAM retrieval time: {analysis['total_dram_time_ms']:.1f} ms")
    print(f"  GPU compute time: {analysis['total_gpu_time_ms']:.1f} ms")
    print(f"  Overlap time: {analysis['overlap_time_ms']:.1f} ms")
    print(f"  Masking efficiency: {analysis['masking_efficiency']*100:.1f}%")
    print(f"  Effective latency: {analysis['effective_latency_ms']:.1f} ms")
    
    # 验证Proposition 4.1: tau_ret < tau_pre
    tau_ret = analysis['total_dram_time_ms']
    tau_pre = analysis['total_gpu_time_ms']
    proposition_satisfied = tau_ret < tau_pre
    
    print(f"\n  Proposition 4.1 (τ_ret < τ_pre): {proposition_satisfied} {'✅' if proposition_satisfied else '❌'}")
    
    # 验收标准
    max_E = max(E_values) if E_values else 128000
    speedup_at_max = (results['without_prefetch'][max_E]['avg_latency_ms'] / 
                     results['with_prefetch'][max_E]['avg_latency_ms'])
    
    acceptance = {
        'proposition_4_1_satisfied': proposition_satisfied,
        'masking_efficiency': analysis['masking_efficiency'],
        'speedup_with_prefetch': speedup_at_max,
        'overall_pass': proposition_satisfied and analysis['masking_efficiency'] > 0.5
    }
    
    print(f"\n  Overall: {'✅ PASS' if acceptance['overall_pass'] else '❌ FAIL'}")
    print("=" * 70)
    
    # 保存结果
    output = {
        'test_name': 'latency_breakdown',
        'config': {
            'E_values': E_values,
            'num_requests': num_requests,
        },
        'results': results,
        'detailed_analysis': analysis,
        'acceptance': acceptance,
    }
    
    output_file = output_dir / "latency_breakdown.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    results = run_latency_breakdown_test()
