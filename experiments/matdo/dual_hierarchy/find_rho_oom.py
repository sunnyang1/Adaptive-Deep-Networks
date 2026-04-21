"""
US2: 双奇点层次验证

证明 ρ_OOM < ρ_collapse，即硬件墙先于信息墙
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.matdo.common.config import config
from experiments.matdo.singularity.measure_t_opt import (
    measure_optimal_t_at_rho,
    singularity_model,
    fit_singularity_law
)


def compute_T_max() -> float:
    """计算硬件限制的最大T"""
    return config.compute_T_max()


def find_rho_oom(
    rho_collapse: float,
    T_max: float,
    tolerance: float = 0.001
) -> float:
    """
    找到ρ_OOM（T* = T_max时的ρ值）
    
    使用二分搜索在[0.90, ρ_collapse)区间内寻找
    
    Args:
        rho_collapse: 已拟合的坍缩点
        T_max: 硬件最大T
        tolerance: 搜索容差
    
    Returns:
        rho_oom: 硬件OOM点
    """
    print(f"搜索ρ_OOM (T_max={T_max:.1f})...")
    
    # 首先粗略扫描确定范围
    test_rhos = np.linspace(0.90, rho_collapse - 0.01, 20)
    
    rho_low, rho_high = None, None
    
    for rho in test_rhos:
        result = measure_optimal_t_at_rho(rho, rho_collapse)
        print(f"  ρ={rho:.4f}: T*={result.T_star}, meets_sla={result.meets_sla}")
        
        if result.T_star < T_max * 0.9:
            rho_low = rho
        elif result.T_star > T_max:
            rho_high = rho
            break
    
    if rho_low is None:
        print("  WARNING: 即使在ρ=0.90，T*也已接近T_max")
        rho_low = 0.85
    
    if rho_high is None:
        print("  WARNING: 即使接近ρ_collapse，T*仍未超过T_max")
        rho_high = rho_collapse - 0.005
    
    print(f"  二分搜索范围: [{rho_low:.4f}, {rho_high:.4f}]")
    
    # 二分搜索精确定位
    while rho_high - rho_low > tolerance:
        rho_mid = (rho_low + rho_high) / 2
        result = measure_optimal_t_at_rho(rho_mid, rho_collapse)
        
        if result.T_star < T_max:
            rho_low = rho_mid
        else:
            rho_high = rho_mid
    
    rho_oom = (rho_low + rho_high) / 2
    return rho_oom


def verify_dual_hierarchy(
    singularity_results_file: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """
    验证双奇点层次结构
    
    Args:
        singularity_results_file: US1的结果文件路径
        output_dir: 输出目录
    
    Returns:
        results: 包含ρ_OOM和ρ_collapse的验证结果
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("US2: 双奇点层次验证")
    print("=" * 70)
    
    # 获取US1的结果（ρ_collapse）
    if singularity_results_file is None:
        singularity_results_file = (
            Path(__file__).parent.parent / "singularity" / "results" / 
            "singularity_results.json"
        )
    
    if not singularity_results_file.exists():
        print(f"ERROR: 未找到US1结果文件: {singularity_results_file}")
        print("请先运行US1实验")
        return {}
    
    with open(singularity_results_file) as f:
        us1_results = json.load(f)
    
    rho_collapse = us1_results['fit']['rho_collapse']
    rho_collapse_err = us1_results['fit']['rho_collapse_err']
    
    print(f"从US1获得: ρ_collapse = {rho_collapse:.4f} ± {rho_collapse_err:.4f}")
    print()
    
    # 计算T_max
    T_max = compute_T_max()
    print(f"硬件限制: T_max = {T_max:.1f}")
    print(f"          (B_max = {config.B_max:.2e} FLOPs)")
    print()
    
    # 找到ρ_OOM
    rho_oom = find_rho_oom(rho_collapse, T_max)
    
    # 最终验证测量
    result_at_oom = measure_optimal_t_at_rho(rho_oom, rho_collapse)
    print(f"\n在ρ_OOM = {rho_oom:.4f}:")
    print(f"  T* = {result_at_oom.T_star:.1f}")
    print(f"  T_max = {T_max:.1f}")
    print(f"  ratio = {result_at_oom.T_star / T_max:.3f}")
    print()
    
    # 计算差距
    gap = rho_collapse - rho_oom
    gap_relative = gap / rho_collapse
    
    print("验证结果:")
    print(f"  ρ_collapse = {rho_collapse:.4f}")
    print(f"  ρ_OOM      = {rho_oom:.4f}")
    print(f"  gap        = {gap:.4f} ({gap_relative*100:.1f}%)")
    print()
    
    # 验收标准
    hierarchy_verified = rho_oom < rho_collapse
    gap_sufficient = gap > 0.02  # 至少2%的差距
    
    print("验收标准:")
    print(f"  ρ_OOM < ρ_collapse: {hierarchy_verified} {'✅' if hierarchy_verified else '❌'}")
    print(f"  gap > 0.02: {gap:.4f} {'✅' if gap_sufficient else '❌'}")
    
    # 保存结果
    results = {
        'rho_collapse': float(rho_collapse),
        'rho_collapse_err': float(rho_collapse_err),
        'rho_oom': float(rho_oom),
        'T_max': float(T_max),
        'gap': float(gap),
        'gap_relative': float(gap_relative),
        'acceptance': {
            'hierarchy_verified': bool(hierarchy_verified),
            'gap_sufficient': bool(gap_sufficient),
            'overall_pass': bool(hierarchy_verified and gap_sufficient)
        }
    }
    
    output_file = output_dir / "dual_hierarchy_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存至: {output_file}")
    
    # 最终结论
    print()
    print("=" * 70)
    if results['acceptance']['overall_pass']:
        print("✅ US2 PASSED: 双奇点层次验证成功")
        print(f"   ρ_OOM = {rho_oom:.4f} < ρ_collapse = {rho_collapse:.4f}")
    else:
        print("❌ US2 FAILED: 未通过验收标准")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    results = verify_dual_hierarchy()
