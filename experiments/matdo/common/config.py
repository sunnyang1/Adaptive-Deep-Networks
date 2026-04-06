"""MATDO实验配置 (MATDO-E Extended)

包含所有实验共享的常数和参数
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class MATDOConfig:
    """MATDO实验配置 - 支持3D (R,M,T) 和 4D (R,M,T,E) 优化"""
    
    # 模型配置
    model_name: str = "meta-llama/Llama-2-7b-hf"
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    
    # MATDO维度配置 (3D: R, M, T)
    R_options: Tuple[int, ...] = (2, 4, 8)  # 量化比特
    R_min: int = 2  # 最小量化
    S: int = 4  # 每层块数
    N_block: int = 1024  # 每块token数
    
    # MATDO-E 新增: 第四维度 Engram (E)
    E_max: int = 128000  # 最大Engram条目数 (128K clusters)
    E_0: float = 10000.0  # Engram补偿函数的特征尺度
    
    # MATDO-E 误差模型扩展系数
    # 4D误差: E = alpha*2^(-2R) + beta*f(E)/(M*S) + gamma/sqrt(T) + eta/E + couplings
    zeta: float = 0.35   # Engram补偿强度 (paper §5.5)
    eta: float = 0.5     # 检索误差系数 (paper §5.5)
    
    # 误差系数（从Phase 1系统辨识获得）
    alpha: float = 0.015   # Space误差系数 (量化误差)
    beta: float = 2.0      # Scope误差系数 (上下文截断)
    gamma: float = 0.10    # Specificity误差系数 (适应步数)
    delta: float = 0.005   # Space-Scope耦合
    epsilon: float = 0.002  # Scope-Spec耦合
    
    # 硬件成本系数（FLOPs）
    c_R: float = 1.2e3   # HBM访问成本
    c_M: float = 2.5e3   # SRAM访问成本
    c_T: float = 8.0e4   # 计算成本
    c_E: float = 5.0e2   # Engram检索成本 (per entry)
    
    # KV Cache配置
    C_KV: int = 80 * 1024**3  # 80GB (A100)
    C_unit: float = None  # 将在__post_init__计算
    
    # MATDO-E: DRAM配置 (用于Engram存储)
    C_DRAM: int = 512 * 1024**3  # 512GB CPU DRAM
    L_embedding: int = 384  # all-MiniLM-L6-v2 embedding维度
    rho_DRAM: float = 0.5   # DRAM使用率上限
    
    # SLA配置
    E_target: float = 0.05  # 目标误差5%
    
    # 计算预算
    B_max: float = 5e13  # 最大FLOPs预算
    
    # MATDO-E: TTA步数上限 (Compute Wall防护)
    T_max_hard: int = 4096
    
    # 真实模型配置
    use_real_model: bool = False
    checkpoint_path: Optional[str] = None
    model_size: str = "small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    real_model_eval_tasks: Tuple[str, ...] = ("needle",)
    real_model_num_samples: int = 5
    real_model_context_lengths: Tuple[int, ...] = (4096, 16384, 65536)
    
    # Engram索引配置
    engram_index_path: Optional[str] = None
    engram_async_workers: int = 4
    
    def __post_init__(self):
        # 计算每token-bit字节数: 2 * d / 8 (Key+Value, FP16转字节)
        self.C_unit = 2 * self.d_model / 8
        
    def compute_T_max(self) -> float:
        """计算最大可行T（受计算预算限制）"""
        return (self.B_max - self.c_R * self.R_min * self.d_model) / (self.c_T * self.d_model ** 2)
    
    def compute_M_at_rho(self, rho: float, R: int = None) -> int:
        """在给定fill rate下计算可行M"""
        if R is None:
            R = self.R_min
        M_float = self.C_KV * (1 - rho) / (self.N_block * R * self.C_unit)
        return max(1, int(M_float))
    
    def compute_M_min(self, E: int = 0) -> float:
        """
        计算满足SLA的最小M
        
        MATDO-E扩展: 考虑Engram补偿效应
        """
        if E == 0:
            # 原始3D公式
            return self.beta / (self.S * self.E_target)
        
        # 4D公式: 考虑Engram补偿和检索开销
        f_E = self.compute_engram_compensation(E)
        retrieval_overhead = self.eta / max(E, 1)
        effective_E_target = self.E_target - retrieval_overhead
        
        if effective_E_target <= 0:
            return float('inf')
        
        M_min_E = (self.beta * f_E) / (self.S * effective_E_target)
        return M_min_E
    
    def compute_rho_collapse(self, E: int = 0) -> float:
        """
        计算信息坍缩点 (Context Wall)
        
        MATDO-E: 当E>0时，rho_ctx^E > rho_ctx (Wall Postponement)
        """
        M_min = self.compute_M_min(E)
        return 1 - (M_min * self.N_block * self.R_min * self.C_unit) / self.C_KV
    
    def compute_engram_compensation(self, E: int) -> float:
        """
        计算Engram补偿函数 f(E) = 1 - zeta * (1 - exp(-E/E0))
        
        Returns:
            f(E) in [1-zeta, 1]
        """
        if E <= 0:
            return 1.0
        exp_term = 1.0 - np.exp(-E / self.E_0)
        return 1.0 - self.zeta * exp_term
    
    def check_arbitrage_inequality(self) -> bool:
        """
        检查异构套利不等式 (Heterogeneous Arbitrage Inequality)
        
        Theorem 4.1: zeta > eta / (E_max * E_target)
        """
        rhs = self.eta / (self.E_max * self.E_target)
        return self.zeta > rhs
    
    def compute_optimal_T_quadratic(self, rho: float, rho_ctx: float) -> float:
        """
        基于二次爆发定律计算理论最优T
        
        T* ~ (rho_ctx - rho)^(-2)
        """
        delta_rho = rho_ctx - rho
        if delta_rho <= 0:
            return float('inf')
        
        M_min = self.compute_M_min()
        M_at_rho = self.compute_M_at_rho(rho)
        delta_M = M_at_rho - M_min
        
        if delta_M <= 0:
            return float('inf')
        
        T_star = (self.gamma * M_min**2 * self.S / (self.beta * delta_M)) ** 2
        return T_star


# 全局配置实例
config = MATDOConfig()
