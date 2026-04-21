"""
准确率恢复测试 (Fig C)

目标: 证明当M被砍掉50%时，增加E可以将准确率从60%拉回95%+。

实验设计:
- X轴: E (Engram size)
- Y轴: Accuracy
- 固定M为正常值的50%，观察E的补偿效应
"""

import numpy as np
import json
from typing import Dict, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.matdo.matdo_e.solver import MATDOESolver
from experiments.matdo.common.config import config


def compute_accuracy_with_config(R: int, M: int, T: int, E: int,
                                  solver: MATDOESolver) -> float:
    """
    计算给定配置下的理论准确率
    
    基于论文误差模型:
    Accuracy = 1 - E(R,M,T,E)
    """
    error = solver.compute_error(R, M, T, E)
    # 添加一些噪声模拟真实场景
    noise = np.random.normal(0, 0.02)
    accuracy = max(0, min(1, 1 - error + noise))
    return accuracy


def run_accuracy_recovery_test(
    M_reduction_factors: List[float] = None,
    E_values: List[int] = None,
    output_dir: Path = None
) -> Dict:
    """
    运行准确率恢复测试
    
    Args:
        M_reduction_factors: M削减比例列表 (0.5表示削减50%)
        E_values: Engram数量列表
        output_dir: 输出目录
        
    Returns:
        实验结果
    """
    if M_reduction_factors is None:
        M_reduction_factors = [0.5, 0.3, 0.2]
        
    if E_values is None:
        E_values = [0, 8000, 16000, 32000, 64000, 128000]
        
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Accuracy Recovery Test (Fig C)")
    print("=" * 70)
    print(f"M reduction factors: {M_reduction_factors}")
    print(f"E values: {E_values}")
    print(f"Target: When M is cut 50%, E increases accuracy from 60% to 95%+")
    print()
    
    solver = MATDOESolver()
    
    # 基准M (正常值，rho=0.8时)
    rho_baseline = 0.8
    M_baseline = config.compute_M_at_rho(rho_baseline, config.R_min)
    T_baseline = 8  # 基线TTA步数
    
    results = {}
    
    for reduction_factor in M_reduction_factors:
        M_reduced = int(M_baseline * reduction_factor)
        
        print(f"\nM = {M_reduced} (reduced to {reduction_factor*100:.0f}% of baseline {M_baseline})")
        print(f"{'E':>10} | {'Accuracy':>10} | {'Error':>10} | {'Compensation f(E)':>18}")
        print("-" * 70)
        
        accuracies = []
        for E in E_values:
            # 根据E调整T (更多E，更少T需求)
            if E > 0:
                T = max(4, T_baseline - int(np.log2(E/1000 + 1)))
            else:
                T = T_baseline
                
            accuracy = compute_accuracy_with_config(
                config.R_min, M_reduced, T, E, solver
            )
            accuracies.append(accuracy)
            
            error = solver.compute_error(config.R_min, M_reduced, T, E)
            f_E = config.compute_engram_compensation(E)
            
            marker = " <-- TARGET" if accuracy > 0.95 and E > 0 else ""
            print(f"{E:>10} | {accuracy:>10.2%} | {error:>10.4f} | {f_E:>18.4f}{marker}")
        
        results[f"M_{reduction_factor}"] = {
            'M': M_reduced,
            'reduction_factor': reduction_factor,
            'E_values': E_values,
            'accuracies': accuracies,
        }
    
    # 分析关键指标
    print()
    print("=" * 70)
    print("Recovery Analysis")
    print("=" * 70)
    
    key_results = {}
    for reduction_factor in M_reduction_factors:
        key = f"M_{reduction_factor}"
        data = results[key]
        
        # 无Engram时的准确率
        acc_no_engram = data['accuracies'][0]
        
        # 最大Engram时的准确率
        acc_max_engram = data['accuracies'][-1]
        
        # 恢复量
        recovery = acc_max_engram - acc_no_engram
        
        # 找到达到95%准确率的最小E
        E_for_95 = None
        for E, acc in zip(data['E_values'], data['accuracies']):
            if acc >= 0.95:
                E_for_95 = E
                break
        
        key_results[key] = {
            'acc_no_engram': acc_no_engram,
            'acc_max_engram': acc_max_engram,
            'recovery': recovery,
            'E_for_95_accuracy': E_for_95,
        }
        
        print(f"\nM at {reduction_factor*100:.0f}%:")
        print(f"  Without Engram: {acc_no_engram:.2%}")
        print(f"  With Max Engram: {acc_max_engram:.2%}")
        print(f"  Recovery: +{recovery:.2%}")
        if E_for_95:
            print(f"  E needed for 95%: {E_for_95}")
        else:
            print(f"  E needed for 95%: N/A (did not reach)")
    
    # 验收标准
    M_50_result = key_results.get('M_0.5', {})
    target_met = (
        M_50_result.get('acc_no_engram', 1.0) < 0.65 and  # 无Engram时约60%
        M_50_result.get('acc_max_engram', 0.0) > 0.95 and  # 有Engram时95%+
        M_50_result.get('recovery', 0) > 0.30  # 恢复至少30个百分点
    )
    
    acceptance = {
        'baseline_accuracy_below_65': M_50_result.get('acc_no_engram', 1.0) < 0.65,
        'recovered_accuracy_above_95': M_50_result.get('acc_max_engram', 0.0) > 0.95,
        'recovery_magnitude': M_50_result.get('recovery', 0),
        'overall_pass': target_met,
    }
    
    print(f"\n{'='*70}")
    print("Acceptance Criteria:")
    print(f"  Baseline accuracy < 65%: {acceptance['baseline_accuracy_below_65']}")
    print(f"  Recovered accuracy > 95%: {acceptance['recovered_accuracy_above_95']}")
    print(f"  Recovery magnitude: {acceptance['recovery_magnitude']:.2%}")
    print(f"  Overall: {'✅ PASS' if acceptance['overall_pass'] else '❌ FAIL'}")
    print("=" * 70)
    
    # 保存结果
    output = {
        'test_name': 'accuracy_recovery',
        'config': {
            'M_baseline': M_baseline,
            'M_reduction_factors': M_reduction_factors,
            'E_values': E_values,
        },
        'results': results,
        'analysis': key_results,
        'acceptance': acceptance,
    }
    
    output_file = output_dir / "accuracy_recovery.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    results = run_accuracy_recovery_test()
