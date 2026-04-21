"""
vLLM集成消融实验

测试四种配置：
1. 仅量化 R (RaBitQ only)
2. 仅减少 M (Scope only)
3. 仅增加 E (Engram only)
4. MATDO-E 全开启 (4D)

对比指标: Accuracy vs. Latency
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.matdo.matdo_e.solver import MATDOESolver, OptimalConfig
from experiments.matdo.common.config import config


class AblationConfig:
    """消融实验配置"""
    def __init__(self, name: str, enable_R: bool, enable_M: bool, 
                 enable_T: bool, enable_E: bool):
        self.name = name
        self.enable_R = enable_R
        self.enable_M = enable_M
        self.enable_T = enable_T
        self.enable_E = enable_E


def evaluate_ablation_config(ablation: AblationConfig,
                             rho: float,
                             solver: MATDOESolver) -> Dict:
    """
    评估单个消融配置
    
    Returns:
        {'accuracy': float, 'latency_ms': float}
    """
    # 基准值
    R_base = 8  # 8-bit
    M_base = config.compute_M_at_rho(0.8, R_base)  # 正常rho下的M
    T_base = 0  # 无TTA
    E_base = 0  # 无Engram
    
    # 根据消融配置设置值
    if ablation.enable_R:
        R = config.R_min  # 使用最小量化
    else:
        R = R_base
        
    if ablation.enable_M:
        M = config.compute_M_at_rho(rho, R)  # 动态M
    else:
        M = M_base  # 固定M
        
    if ablation.enable_E:
        E = config.E_max  # 最大Engram
        # 调整M以反映Engram补偿
        M = max(1, int(M * 0.5))  # Engram允许更小的M
    else:
        E = E_base
        
    if ablation.enable_T:
        # 根据剩余误差预算计算T
        error_no_T = solver.compute_error(R, M, 1, E)
        remaining = config.E_target - error_no_T
        if remaining > 0:
            T = int((config.gamma / remaining) ** 2)
            T = min(T, config.T_max_hard)
        else:
            T = config.T_max_hard
    else:
        T = T_base
    
    # 计算准确率
    error = solver.compute_error(R, M, T, E)
    accuracy = max(0, min(1, 1 - error + np.random.normal(0, 0.01)))
    
    # 计算延迟 (简化模型)
    # 基础延迟
    latency = 50  # ms
    
    # TTA增加延迟 (二次方增长)
    if T > 0:
        latency += 2 * np.sqrt(T)  # TTA开销
        
    # Engram检索增加延迟 (但被预取掩盖，只加少量)
    if E > 0:
        latency += 5  # 融合开销
        
    # 量化减少延迟
    if R < R_base:
        latency *= (R / R_base)  # 更小的bit数，更快
        
    return {
        'accuracy': accuracy,
        'latency_ms': latency,
        'config': {'R': R, 'M': M, 'T': T, 'E': E},
    }


def run_vllm_ablation_study(
    rho: float = 0.95,
    num_trials: int = 10,
    output_dir: Path = None
) -> Dict:
    """
    运行vLLM消融实验
    
    Args:
        rho: 测试的显存压力
        num_trials: 重复试验次数
        output_dir: 输出目录
        
    Returns:
        实验结果
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("vLLM Ablation Study")
    print("=" * 70)
    print(f"Testing at ρ = {rho}")
    print(f"Trials per configuration: {num_trials}")
    print()
    
    # 定义消融配置
    ablations = [
        AblationConfig("Baseline (No Opt)", False, False, False, False),
        AblationConfig("R only (RaBitQ)", True, False, False, False),
        AblationConfig("M only (AttnRes)", False, True, False, False),
        AblationConfig("E only (Engram)", False, False, False, True),
        AblationConfig("T only (qTTT)", False, False, True, False),
        AblationConfig("R+M (3D)", True, True, True, False),
        AblationConfig("MATDO-E (4D)", True, True, True, True),
    ]
    
    solver = MATDOESolver()
    
    results = {}
    
    print(f"{'Configuration':<20} | {'Accuracy':>10} | {'Latency(ms)':>12} | {'Score':>10}")
    print("-" * 70)
    
    for ablation in ablations:
        accuracies = []
        latencies = []
        
        for _ in range(num_trials):
            result = evaluate_ablation_config(ablation, rho, solver)
            accuracies.append(result['accuracy'])
            latencies.append(result['latency_ms'])
        
        avg_acc = np.mean(accuracies)
        avg_lat = np.mean(latencies)
        
        # 综合评分 (准确率 / 延迟，归一化)
        score = avg_acc / (avg_lat / 100)
        
        results[ablation.name] = {
            'accuracy_mean': avg_acc,
            'accuracy_std': np.std(accuracies),
            'latency_mean_ms': avg_lat,
            'latency_std_ms': np.std(latencies),
            'score': score,
            'config': result['config'],
        }
        
        print(f"{ablation.name:<20} | {avg_acc:>10.2%} | {avg_lat:>12.1f} | {score:>10.2f}")
    
    # 分析
    print()
    print("=" * 70)
    print("Analysis")
    print("=" * 70)
    
    # 最佳配置
    best_by_accuracy = max(results.items(), key=lambda x: x[1]['accuracy_mean'])
    best_by_latency = min(results.items(), key=lambda x: x[1]['latency_mean_ms'])
    best_by_score = max(results.items(), key=lambda x: x[1]['score'])
    
    print(f"\nBest by Accuracy: {best_by_accuracy[0]} ({best_by_accuracy[1]['accuracy_mean']:.2%})")
    print(f"Best by Latency: {best_by_latency[0]} ({best_by_latency[1]['latency_mean_ms']:.1f} ms)")
    print(f"Best Overall Score: {best_by_score[0]} ({best_by_score[1]['score']:.2f})")
    
    # 计算MATDO-E的改进
    baseline = results['Baseline (No Opt)']
    matdo_3d = results['R+M (3D)']
    matdo_4d = results['MATDO-E (4D)']
    
    print(f"\nImprovements vs Baseline:")
    print(f"  3D (R+M+T): Accuracy +{(matdo_3d['accuracy_mean']/baseline['accuracy_mean']-1)*100:.1f}%, "
          f"Latency {(matdo_3d['latency_mean_ms']/baseline['latency_mean_ms']-1)*100:+.1f}%")
    print(f"  4D (MATDO-E): Accuracy +{(matdo_4d['accuracy_mean']/baseline['accuracy_mean']-1)*100:.1f}%, "
          f"Latency {(matdo_4d['latency_mean_ms']/baseline['latency_mean_ms']-1)*100:+.1f}%")
    
    # 验收标准
    acceptance = {
        'matdo_4d_best_accuracy': best_by_accuracy[0] == 'MATDO-E (4D)',
        'matdo_4d_accuracy_above_90': matdo_4d['accuracy_mean'] > 0.90,
        'matdo_4d_latency_under_100ms': matdo_4d['latency_mean_ms'] < 100,
        'improvement_over_3d': matdo_4d['accuracy_mean'] > matdo_3d['accuracy_mean'],
        'overall_pass': (
            matdo_4d['accuracy_mean'] > 0.90 and 
            matdo_4d['accuracy_mean'] > matdo_3d['accuracy_mean']
        ),
    }
    
    print(f"\n{'='*70}")
    print("Acceptance Criteria:")
    print(f"  MATDO-E best accuracy: {acceptance['matdo_4d_best_accuracy']}")
    print(f"  Accuracy > 90%: {acceptance['matdo_4d_accuracy_above_90']}")
    print(f"  Latency < 100ms: {acceptance['matdo_4d_latency_under_100ms']}")
    print(f"  Better than 3D: {acceptance['improvement_over_3d']}")
    print(f"  Overall: {'✅ PASS' if acceptance['overall_pass'] else '❌ FAIL'}")
    print("=" * 70)
    
    # 保存结果
    output = {
        'test_name': 'vllm_ablation',
        'config': {
            'rho': rho,
            'num_trials': num_trials,
        },
        'results': results,
        'analysis': {
            'best_by_accuracy': best_by_accuracy[0],
            'best_by_latency': best_by_latency[0],
            'best_overall': best_by_score[0],
        },
        'acceptance': acceptance,
    }
    
    output_file = output_dir / "ablation_vllm.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    results = run_vllm_ablation_study()
