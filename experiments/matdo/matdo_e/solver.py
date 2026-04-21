"""
MATDO-E 四维优化求解器

实现论文中的核心优化问题：
  min_{R,M,T,E}  B_total = c_R*R*d + c_M*M*S*d + c_T*T*d^2 + c_E*E*L
  s.t.           E(R,M,T,E) <= E_target
                  M*N_block*R*C_unit <= C_HBM*(1-rho)
                  E*L <= C_DRAM*(1-rho_DRAM)
                  T <= T_max

关键公式:
- 误差模型: E = alpha*2^(-2R) + beta*f(E)/(M*S) + gamma/sqrt(T) + eta/E + couplings
- 补偿函数: f(E) = 1 - zeta*(1 - exp(-E/E0))
- 套利不等式: zeta > eta/(E_max * E_target)
- 二次爆发: T* ~ (rho_ctx - rho)^(-2)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.matdo.common.config import config


@dataclass
class OptimalConfig:
    """最优四维配置"""
    R: int          # 量化比特数 (2, 4, 8)
    M: int          # 上下文块数 (Scope)
    T: int          # TTA步数 (Specificity)
    E: int          # Engram条目数
    
    # 计算出的指标
    estimated_error: float
    estimated_cost: float
    rho_ctx_effective: float  # 有效context wall
    
    # 标记
    is_arbitrage: bool  # 是否启用了异构套利
    within_compute_wall: bool  # 是否在计算墙内


class MATDOESolver:
    """
    MATDO-E 优化求解器
    
    根据当前显存压力rho，求解最优(R, M, T, E)配置。
    """
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        
        # 预计算套利不等式是否成立
        self.arbitrage_feasible = self._check_arbitrage_feasibility()
        
        # rho阈值 (从论文§5)
        self.rho_arbitrage_zone = 0.93  # 进入套利区
        self.rho_critical = 0.98        # 临界区域
        
    def _check_arbitrage_feasibility(self) -> bool:
        """检查异构套利不等式"""
        return self.cfg.check_arbitrage_inequality()
    
    def compute_error(self, R: int, M: int, T: int, E: int) -> float:
        """
        计算给定配置下的估计误差
        
        E(R,M,T,E) = alpha*2^(-2R) + beta*f(E)/(M*S) + gamma/sqrt(T) 
                     + delta*2^(-2R)/M + epsilon*ln(M)/T + eta/E
        """
        cfg = self.cfg
        
        # 各独立误差项
        E_space = cfg.alpha * (2 ** (-2 * R))
        
        # Engram补偿效应: f(E) = 1 - zeta*(1 - exp(-E/E0))
        f_E = cfg.compute_engram_compensation(E)
        E_scope = (cfg.beta * f_E) / (M * cfg.S)
        
        E_spec = cfg.gamma / np.sqrt(max(T, 1))
        E_retrieval = cfg.eta / max(E, 1) if E > 0 else 0
        
        # 耦合项
        E_couple_ss = cfg.delta * (2 ** (-2 * R)) / M
        E_couple_st = cfg.epsilon * np.log(max(M, 2)) / max(T, 1)
        
        total = E_space + E_scope + E_spec + E_retrieval + E_couple_ss + E_couple_st
        return total
    
    def compute_cost(self, R: int, M: int, T: int, E: int) -> float:
        """
        计算计算成本 (FLOPs)
        
        B = c_R*R*d + c_M*M*S*d + c_T*T*d^2 + c_E*E*L
        """
        cfg = self.cfg
        cost = (cfg.c_R * R * cfg.d_model + 
                cfg.c_M * M * cfg.S * cfg.d_model +
                cfg.c_T * T * cfg.d_model ** 2 +
                cfg.c_E * E * cfg.L_embedding)
        return cost
    
    def solve(self, rho: float, target_error: Optional[float] = None) -> OptimalConfig:
        """
        求解给定rho下的最优配置
        
        这是核心算法，实现了论文中的"异构套利"策略。
        
        Args:
            rho: 当前HBM使用率 (0-1)
            target_error: 目标误差 (默认使用cfg.E_target)
            
        Returns:
            OptimalConfig: 最优四维配置
        """
        if target_error is None:
            target_error = self.cfg.E_target
            
        cfg = self.cfg
        
        # 计算当前rho下的最大可行M (HBM约束)
        M_max = cfg.compute_M_at_rho(rho, R=cfg.R_min)
        
        # 计算不同E值下的M_min (考虑Engram补偿)
        rho_ctx_3d = cfg.compute_rho_collapse(E=0)
        rho_ctx_4d = cfg.compute_rho_collapse(E=cfg.E_max) if self.arbitrage_feasible else rho_ctx_3d
        
        # 判断是否进入套利区
        is_arbitrage_zone = rho > self.rho_arbitrage_zone and self.arbitrage_feasible
        
        if is_arbitrage_zone:
            # === 套利模式 ===
            # 论文§4: 当rho接近rho_ctx时，将E>0放入DRAM以降低对HBM的需求
            
            # 使用最大Engram
            E_star = cfg.E_max
            
            # 激进的KV Cache压缩：使用最低量化
            R_star = cfg.R_min  # 通常是2-bit
            
            # 计算补偿后的最小M需求
            M_min_eff = cfg.compute_M_min(E=E_star)
            M_star = max(int(M_min_eff * 1.1), 1)  # 10% margin
            
            # 确保不超过HBM容量
            M_hbm_limit = cfg.compute_M_at_rho(rho, R=R_star)
            M_star = min(M_star, M_hbm_limit)
            
            # 根据Quadratic Blow-up Law计算T
            # T* ~ (rho_ctx - rho)^(-2)
            delta_rho = rho_ctx_4d - rho
            if delta_rho > 0.001:
                T_theoretical = cfg.compute_optimal_T_quadratic(rho, rho_ctx_4d)
                T_star = int(min(T_theoretical, cfg.T_max_hard, cfg.compute_T_max()))
            else:
                T_star = cfg.T_max_hard
                
        else:
            # === 普通模式 ===
            # 标准3D优化
            E_star = 0
            R_star = cfg.R_min
            
            # 使用全部可用HBM
            M_star = M_max
            
            # 计算满足SLA的T
            E_space = cfg.alpha * (2 ** (-2 * R_star))
            E_scope = cfg.beta / (M_star * cfg.S)
            remaining_budget = target_error - E_space - E_scope
            
            if remaining_budget > 0:
                T_star = int((cfg.gamma / remaining_budget) ** 2)
                T_star = min(T_star, cfg.T_max_hard, int(cfg.compute_T_max()))
            else:
                T_star = cfg.T_max_hard
        
        # 确保T不超过硬限制
        T_star = max(1, min(T_star, cfg.T_max_hard))
        
        # 计算估计指标
        est_error = self.compute_error(R_star, M_star, T_star, E_star)
        est_cost = self.compute_cost(R_star, M_star, T_star, E_star)
        
        # 计算有效的context wall
        effective_rho_ctx = rho_ctx_4d if E_star > 0 else rho_ctx_3d
        
        # 检查是否在计算墙内
        T_max_feasible = cfg.compute_T_max()
        within_compute_wall = T_star <= T_max_feasible
        
        return OptimalConfig(
            R=R_star,
            M=M_star,
            T=T_star,
            E=E_star,
            estimated_error=est_error,
            estimated_cost=est_cost,
            rho_ctx_effective=effective_rho_ctx,
            is_arbitrage=is_arbitrage_zone,
            within_compute_wall=within_compute_wall
        )
    
    def solve_batch(self, rhos: np.ndarray) -> list:
        """批量求解多个rho值的配置"""
        return [self.solve(rho) for rho in rhos]
    
    def find_rho_critical_points(self) -> Tuple[float, float]:
        """
        找到两个临界点：compute wall 和 context wall
        
        Returns:
            (rho_comp, rho_ctx): 计算墙和上下文墙的rho值
        """
        # Context wall: 满足 M_min = M(rho) 的rho
        rho_ctx_3d = self.cfg.compute_rho_collapse(E=0)
        rho_ctx_4d = self.cfg.compute_rho_collapse(E=self.cfg.E_max) if self.arbitrage_feasible else rho_ctx_3d
        
        # Compute wall: 满足 T*(rho) = T_max 的rho
        # 通过二分查找找到rho_comp
        rho_low, rho_high = 0.5, rho_ctx_3d
        T_max = self.cfg.compute_T_max()
        
        for _ in range(20):  # 二分查找
            rho_mid = (rho_low + rho_high) / 2
            opt = self.solve(rho_mid)
            if opt.T >= T_max * 0.99:
                rho_high = rho_mid
            else:
                rho_low = rho_mid
        
        rho_comp = (rho_low + rho_high) / 2
        
        return rho_comp, rho_ctx_4d


if __name__ == "__main__":
    # 测试求解器
    solver = MATDOESolver()
    
    print("=" * 60)
    print("MATDO-E Solver Test")
    print("=" * 60)
    
    # 检查套利不等式
    print(f"\nArbitrage Inequality: {solver.arbitrage_feasible}")
    print(f"  zeta = {config.zeta}")
    print(f"  eta/(E_max*E_target) = {config.eta/(config.E_max*config.E_target):.6f}")
    
    # 测试不同rho下的解
    test_rhos = [0.80, 0.90, 0.93, 0.95, 0.97, 0.99]
    
    print(f"\n{'rho':>6} | {'R':>3} | {'M':>6} | {'T':>6} | {'E':>8} | {'Arb':>4} | {'Error':>8} | {'rho_ctx^E':>10}")
    print("-" * 80)
    
    for rho in test_rhos:
        opt = solver.solve(rho)
        print(f"{rho:>6.2f} | {opt.R:>3} | {opt.M:>6} | {opt.T:>6} | {opt.E:>8} | "
              f"{opt.is_arbitrage:>4} | {opt.estimated_error:>8.4f} | {opt.rho_ctx_effective:>10.4f}")
    
    # 找到临界点
    rho_comp, rho_ctx = solver.find_rho_critical_points()
    print(f"\nCritical Points:")
    print(f"  rho_comp (compute wall) = {rho_comp:.4f}")
    print(f"  rho_ctx (context wall with E) = {rho_ctx:.4f}")
    print(f"  Wall postponement = {rho_ctx - config.compute_rho_collapse(E=0):.4f}")
