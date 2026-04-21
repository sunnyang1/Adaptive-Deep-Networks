"""
吞吐量可持续性测试 (Fig A)

目标: 证明当rho=0.99时，MATDO-E仍能保持吞吐量，而原生vLLM会在90%时阻塞。

实验设计:
- X轴: 并发请求数 (Batch Size)
- Y轴: 吞吐量 (tokens/sec)
- 对比: Native vLLM vs MATDO-E
"""

import numpy as np
import json
from typing import Dict, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.matdo.matdo_e.scheduler import MATDOEScheduler, MATDORequest
from experiments.matdo.common.config import config


def generate_request_load(num_requests: int, 
                          avg_prompt_len: int = 2048,
                          avg_output_len: int = 100) -> List[MATDORequest]:
    """生成请求负载"""
    np.random.seed(42)
    requests = []
    for i in range(num_requests):
        prompt_len = int(np.random.normal(avg_prompt_len, avg_prompt_len * 0.2))
        output_len = int(np.random.normal(avg_output_len, avg_output_len * 0.3))
        requests.append(MATDORequest(
            request_id=f"req_{i}",
            prompt_len=max(256, prompt_len),
            max_new_tokens=max(20, output_len)
        ))
    return requests


def simulate_native_vllm(requests: List[MATDORequest],
                         num_gpu_blocks: int = 512) -> Dict:
    """
    模拟原生vLLM行为
    
    特点:
    - 在rho > 0.90时开始拒绝请求
    - 无套利机制
    """
    scheduler = MATDOEScheduler(
        num_gpu_blocks=num_gpu_blocks,
        enable_arbitrage=False  # 原生vLLM无套利
    )
    
    # 设置更严格的拒绝阈值 (模拟原生vLLM)
    scheduler.block_manager.num_gpu_blocks = int(num_gpu_blocks * 0.90)
    
    results = scheduler.run_simulation(requests, num_steps=300)
    
    # 计算吞吐量
    total_tokens = sum(r.max_new_tokens for r in scheduler.completed)
    total_time_sec = sum(r.completed_at - r.scheduled_at 
                        for r in scheduler.completed if r.completed_at) if scheduler.completed else 1
    
    throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0
    
    return {
        'method': 'Native vLLM',
        'throughput': throughput,
        'completed': results['completed'],
        'rejected': results['rejected'],
        'avg_latency_ms': results['avg_latency_ms'],
        'peak_rho': results['peak_rho'],
    }


def simulate_matdo_e(requests: List[MATDORequest],
                     num_gpu_blocks: int = 512,
                     target_rho: float = 0.99) -> Dict:
    """
    模拟MATDO-E行为
    
    特点:
    - 在rho > 0.95时启用套利
    - 通过Engram补偿保持吞吐量直到0.99
    """
    scheduler = MATDOEScheduler(
        num_gpu_blocks=num_gpu_blocks,
        enable_arbitrage=True
    )
    
    results = scheduler.run_simulation(requests, num_steps=300)
    
    # 计算吞吐量
    total_tokens = sum(r.max_new_tokens for r in scheduler.completed)
    total_time_sec = sum(r.completed_at - r.scheduled_at 
                        for r in scheduler.completed if r.completed_at) if scheduler.completed else 1
    
    throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0
    
    return {
        'method': 'MATDO-E',
        'throughput': throughput,
        'completed': results['completed'],
        'rejected': results['rejected'],
        'arbitrage_ratio': results['arbitrage_ratio'],
        'avg_latency_ms': results['avg_latency_ms'],
        'peak_rho': results['peak_rho'],
    }


def run_throughput_sustainability_test(
    concurrency_levels: List[int] = None,
    num_gpu_blocks: int = 512,
    output_dir: Path = None
) -> Dict:
    """
    运行吞吐量可持续性测试
    
    Args:
        concurrency_levels: 测试的并发级别列表
        num_gpu_blocks: GPU block总数
        output_dir: 输出目录
        
    Returns:
        实验结果
    """
    if concurrency_levels is None:
        concurrency_levels = [10, 20, 30, 40, 50, 60, 70, 80]
        
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Throughput Sustainability Test (Fig A)")
    print("=" * 70)
    print(f"GPU blocks: {num_gpu_blocks}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Target: MATDO-E maintains throughput at ρ=0.99")
    print()
    
    results = {
        'native_vllm': [],
        'matdo_e': [],
    }
    
    print(f"{'Concurrency':>12} | {'Native TP':>12} | {'MATDO-E TP':>12} | {'Improvement':>12}")
    print("-" * 70)
    
    for concurrency in concurrency_levels:
        # 生成请求
        requests = generate_request_load(concurrency)
        
        # 测试原生vLLM
        native_result = simulate_native_vllm(requests, num_gpu_blocks)
        results['native_vllm'].append({
            'concurrency': concurrency,
            **native_result
        })
        
        # 测试MATDO-E
        matdo_result = simulate_matdo_e(requests, num_gpu_blocks)
        results['matdo_e'].append({
            'concurrency': concurrency,
            **matdo_result
        })
        
        improvement = (matdo_result['throughput'] / native_result['throughput'] - 1) * 100
        
        print(f"{concurrency:>12} | {native_result['throughput']:>12.1f} | "
              f"{matdo_result['throughput']:>12.1f} | {improvement:>11.1f}%")
    
    # 分析临界点
    native_peak_idx = np.argmax([r['throughput'] for r in results['native_vllm']])
    matdo_peak_idx = np.argmax([r['throughput'] for r in results['matdo_e']])
    
    print()
    print("=" * 70)
    print("Analysis:")
    print(f"  Native vLLM peak throughput: {results['native_vllm'][native_peak_idx]['throughput']:.1f} "
          f"at concurrency={results['native_vllm'][native_peak_idx]['concurrency']}")
    print(f"  MATDO-E peak throughput: {results['matdo_e'][matdo_peak_idx]['throughput']:.1f} "
          f"at concurrency={results['matdo_e'][matdo_peak_idx]['concurrency']}")
    
    # 验收标准
    final_improvement = (results['matdo_e'][-1]['throughput'] / 
                        results['native_vllm'][-1]['throughput'] - 1) * 100
    
    acceptance = {
        'matdo_sustains_at_high_concurrency': results['matdo_e'][-1]['throughput'] > 
                                              results['native_vllm'][-1]['throughput'],
        'improvement_at_max_concurrency_pct': final_improvement,
        'overall_pass': final_improvement > 20  # 至少20%提升
    }
    
    print(f"\n  Improvement at max concurrency: {final_improvement:.1f}%")
    print(f"  Acceptance: {'✅ PASS' if acceptance['overall_pass'] else '❌ FAIL'}")
    print("=" * 70)
    
    # 保存结果
    output = {
        'test_name': 'throughput_sustainability',
        'config': {
            'num_gpu_blocks': num_gpu_blocks,
            'concurrency_levels': concurrency_levels,
        },
        'results': results,
        'acceptance': acceptance,
    }
    
    output_file = output_dir / "throughput_sustainability.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    results = run_throughput_sustainability_test()
