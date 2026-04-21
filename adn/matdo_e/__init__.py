"""
ADN MATDO-E - 统一资源模型

MATDO-E是ADN的分析框架，包含：
- 资源模型（R, M, T, E四个控制变量）
- 策略决策
- 误差模型
- 在线估计
- 资源理论
"""

from adn.matdo_e.config import MATDOConfig
from adn.matdo_e.policy import (
    RuntimeObservation,
    PolicyDecision,
    solve_policy,
)
from adn.matdo_e.error_model import estimate_error, required_adaptation_steps, ErrorBreakdown
from adn.matdo_e.online_estimation import OnlineEstimate, OnlineRLSEstimator
from adn.matdo_e.resource_theory import (
    dram_max_engram_entries,
    hbm_max_m_blocks,
    m_min_closed_form,
    rho_context_wall,
    rho_compute_wall,
)

__all__ = [
    "MATDOConfig",
    "RuntimeObservation",
    "PolicyDecision",
    "solve_policy",
    "estimate_error",
    "required_adaptation_steps",
    "ErrorBreakdown",
    "OnlineEstimate",
    "OnlineRLSEstimator",
    "dram_max_engram_entries",
    "hbm_max_m_blocks",
    "m_min_closed_form",
    "rho_context_wall",
    "rho_compute_wall",
]
