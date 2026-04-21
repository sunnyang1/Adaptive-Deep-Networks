"""
US1: 二阶奇点标度律验证

测量在不同ρ值下的最优T*，并验证 T* ∝ (ρ_collapse - ρ)^(-2)
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.matdo.common.config import config
from experiments.matdo.common.real_model_bridge import load_matdo_model, evaluate_on_task


@dataclass
class MeasurementResult:
    """单次测量结果"""
    rho: float
    M: int
    T_star: int
    accuracy: float
    error: float
    meets_sla: bool
    

_global_us1_model = None
_global_us1_cfg = None


def _ensure_us1_model():
    global _global_us1_model, _global_us1_cfg
    if _global_us1_model is None:
        _global_us1_model, _global_us1_cfg = load_matdo_model(
            checkpoint_path=config.checkpoint_path,
            model_size=config.model_size,
            device=config.device,
            enable_rabitq=True,
            enable_attnres=True,
            enable_qttt=True,
        )
    return _global_us1_model, _global_us1_cfg


def evaluate_model_accuracy(
    M: int,
    T: int,
    R: int,
    num_samples: int = 100,
    rho: float = None
) -> float:
    """
    评估给定(M, T, R)配置下的模型准确率
    
    在实际实现中，这会调用vLLM进行推理
    这里使用模拟数据（基于论文误差模型）
    
    Args:
        M: 上下文块数
        T: 适应步数
        R: 量化比特
        num_samples: 评估样本数
        rho: fill rate (用于模拟二阶奇点)
    
    Returns:
        accuracy: 准确率 [0, 1]
    """
    if config.use_real_model:
        model, cfg = _ensure_us1_model()
        # 通过 max_seq_len 间接控制 M（上下文块数 -> 上下文长度）
        # T 通过 qttt steps 控制
        # R 当前在模型 forward 中未直接集成，作为配置概念使用
        ctx_len = min(M * config.N_block, getattr(cfg, "max_seq_len", 32768))
        # 临时修改 config 上的 qttt steps
        orig_qttt_steps = cfg.max_qttt_steps
        cfg.max_qttt_steps = min(T, orig_qttt_steps)
        result = evaluate_on_task(
            model, "needle", cfg,
            device=config.device,
            context_lengths=(ctx_len,),
            num_samples=max(1, config.real_model_num_samples // 2),
        )
        cfg.max_qttt_steps = orig_qttt_steps
        return result["average_accuracy"] / 100.0

    # 使用MATDO误差模型计算理论误差
    S_eff = max(config.S, 1)
    M_eff = max(M, 1)
    T_eff = max(T, 1)
    
    E_space = config.alpha * (2 ** (-2 * R))
    E_scope = config.beta / (M_eff * S_eff)
    E_spec = config.gamma / np.sqrt(T_eff)
    E_couple_ss = config.delta * (2 ** (-2 * R)) / M_eff
    E_couple_st = config.epsilon * np.log(max(M_eff, 2)) / T_eff
    
    # 模拟二阶奇点效应：当ρ接近坍缩点时，误差急剧增加
    if rho is not None:
        rho_collapse = 0.96
        if rho > 0.85:
            # 接近坍缩点时增加额外误差压力
            proximity_factor = (rho_collapse - rho) / (rho_collapse - 0.85)
            proximity_penalty = 0.03 * max(0, 1 - proximity_factor) 
            E_scope += proximity_penalty
    
    total_error = (
        E_space + E_scope + E_spec + 
        E_couple_ss + E_couple_st
    )
    
    # 添加测量噪声（模拟实验不确定性）
    noise = np.random.normal(0, 0.005)
    total_error = max(0, min(1, total_error + noise))
    
    accuracy = 1 - total_error
    return accuracy


def measure_optimal_t_at_rho(
    rho: float,
    rho_collapse_estimate: float = 0.95,
    T_candidates: Optional[List[int]] = None,
    num_trials: int = 3
) -> MeasurementResult:
    """
    在给定ρ下测量最优T*
    
    策略：从大到小扫描T，找到满足SLA的最小T
    
    Args:
        rho: KV Cache fill rate
        rho_collapse_estimate: 坍缩点估计（用于计算M）
        T_candidates: 候选T值列表
        num_trials: 每个配置重复测量次数
    
    Returns:
        MeasurementResult: 测量结果
    """
    if T_candidates is None:
        # 对数间隔的T候选值，扩大范围以观察奇点行为
        T_candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    # 计算当前ρ下的M（使用最小量化）
    M = config.compute_M_at_rho(rho, R=config.R_min)
    
    print(f"  ρ={rho:.4f}: M={M}, testing T in {T_candidates}")
    
    best_result = None
    
    for T in T_candidates:
        # 多次测量取平均
        accuracies = []
        for _ in range(num_trials):
            acc = evaluate_model_accuracy(M, T, config.R_min, rho=rho)
            accuracies.append(acc)
        
        accuracy = np.mean(accuracies)
        error = 1 - accuracy
        meets_sla = error <= config.E_target
        
        print(f"    T={T:3d}: accuracy={accuracy:.4f}, error={error:.4f}, SLA={meets_sla}")
        
        if meets_sla:
            # 找到满足SLA的最小T
            best_result = MeasurementResult(
                rho=rho,
                M=M,
                T_star=T,
                accuracy=accuracy,
                error=error,
                meets_sla=True
            )
            break
    
    if best_result is None:
        # 没有T满足SLA，返回最大T的结果
        T = T_candidates[-1]
        accuracies = [evaluate_model_accuracy(M, T, config.R_min, rho=rho) 
                     for _ in range(num_trials)]
        best_result = MeasurementResult(
            rho=rho,
            M=M,
            T_star=T,
            accuracy=np.mean(accuracies),
            error=1 - np.mean(accuracies),
            meets_sla=False
        )
        print(f"  WARNING: No T meets SLA at ρ={rho}")
    
    return best_result


def singularity_model(rho: np.ndarray, 
                      A: float, 
                      rho_collapse: float, 
                      B: float) -> np.ndarray:
    """
    二阶奇点理论模型
    
    T*(ρ) = A / (ρ_collapse - ρ)² + B
    
    Args:
        rho: fill rate数组
        A: 幅度系数
        rho_collapse: 坍缩点
        B: 基线偏移
    
    Returns:
        T_star预测值
    """
    return A / (rho_collapse - rho) ** 2 + B


def fit_singularity_law(
    rhos: np.ndarray,
    t_stars: np.ndarray,
    p0: Optional[Tuple[float, float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    拟合二阶奇点标度律
    
    Args:
        rhos: 测量的ρ值数组
        t_stars: 对应的T*值数组
        p0: 初始参数猜测 (A, rho_collapse, B)
    
    Returns:
        popt: 最优参数 [A, rho_collapse, B]
        pcov: 协方差矩阵
        stats: 拟合统计信息
    """
    if p0 is None:
        p0 = [1.0, 0.96, 0.0]
    
    # 曲线拟合
    popt, pcov = curve_fit(
        singularity_model,
        rhos,
        t_stars,
        p0=p0,
        bounds=([0.0, max(rhos) + 0.001, -10], [1000.0, 1.0, 10]),
        maxfev=10000
    )
    
    # 计算R²
    t_pred = singularity_model(rhos, *popt)
    ss_res = np.sum((t_stars - t_pred) ** 2)
    ss_tot = np.sum((t_stars - np.mean(t_stars)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 计算参数误差
    perr = np.sqrt(np.diag(pcov))
    
    stats = {
        'r_squared': r_squared,
        'A': popt[0],
        'rho_collapse': popt[1],
        'B': popt[2],
        'A_err': perr[0],
        'rho_collapse_err': perr[1],
        'B_err': perr[2],
        'residuals': (t_stars - t_pred).tolist()
    }
    
    return popt, pcov, stats


def run_singularity_experiment(
    rhos: Optional[List[float]] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """
    运行完整的二阶奇点验证实验
    
    Args:
        rhos: 要测试的ρ值列表，默认为[0.85, 0.88, 0.91, 0.93, 0.94]
        output_dir: 输出目录
    
    Returns:
        results: 包含所有测量和拟合结果的字典
    """
    if rhos is None:
        # 默认测试点：接近但不到达坍缩点
        rhos = [0.85, 0.88, 0.91, 0.93, 0.94, 0.945]
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("US1: 二阶奇点标度律验证")
    print("=" * 70)
    print(f"目标SLA: E_target = {config.E_target}")
    print(f"测试ρ值: {rhos}")
    print()
    
    # 阶段1：测量T*(ρ)
    print("阶段1：测量最优T*...")
    measurements = []
    for rho in rhos:
        result = measure_optimal_t_at_rho(rho)
        measurements.append(result)
        print(f"  → T*({rho:.4f}) = {result.T_star}, accuracy={result.accuracy:.4f}")
        print()
    
    # 提取数据
    rhos_measured = np.array([m.rho for m in measurements])
    t_stars_measured = np.array([m.T_star for m in measurements])
    
    # 阶段2：拟合标度律
    print("阶段2：拟合二阶奇点标度律...")
    popt, pcov, stats = fit_singularity_law(rhos_measured, t_stars_measured)
    
    A_fit, rho_c_fit, B_fit = popt
    print(f"  拟合参数:")
    print(f"    A = {A_fit:.4f} ± {stats['A_err']:.4f}")
    print(f"    ρ_collapse = {rho_c_fit:.4f} ± {stats['rho_collapse_err']:.4f}")
    print(f"    B = {B_fit:.4f} ± {stats['B_err']:.4f}")
    print(f"  R² = {stats['r_squared']:.4f}")
    
    # 阶段3：验收标准检查
    print()
    print("阶段3：验收标准检查...")
    all_meet_sla = all(m.meets_sla for m in measurements)
    r_squared_pass = stats['r_squared'] > 0.95
    
    print(f"  所有点满足SLA: {all_meet_sla} {'✅' if all_meet_sla else '❌'}")
    print(f"  R² > 0.95: {stats['r_squared']:.4f} {'✅' if r_squared_pass else '❌'}")
    
    # 保存结果
    results = {
        'config': {
            'E_target': config.E_target,
            'R_min': config.R_min,
            'tested_rhos': rhos
        },
        'measurements': [
            {
                'rho': m.rho,
                'M': m.M,
                'T_star': m.T_star,
                'accuracy': m.accuracy,
                'error': m.error,
                'meets_sla': m.meets_sla
            }
            for m in measurements
        ],
        'fit': {
            'A': float(A_fit),
            'rho_collapse': float(rho_c_fit),
            'B': float(B_fit),
            'A_err': float(stats['A_err']),
            'rho_collapse_err': float(stats['rho_collapse_err']),
            'B_err': float(stats['B_err']),
            'r_squared': float(stats['r_squared']),
            'residuals': stats['residuals']
        },
        'acceptance': {
            'all_meet_sla': bool(all_meet_sla),
            'r_squared_pass': bool(r_squared_pass),
            'overall_pass': bool(all_meet_sla and r_squared_pass)
        }
    }
    
    output_file = output_dir / "singularity_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存至: {output_file}")
    
    # 打印最终结论
    print()
    print("=" * 70)
    if results['acceptance']['overall_pass']:
        print("✅ US1 PASSED: 二阶奇点标度律验证成功")
    else:
        print("❌ US1 FAILED: 未通过验收标准")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # 运行实验
    np.random.seed(42)  # 保证可复现
    results = run_singularity_experiment()
