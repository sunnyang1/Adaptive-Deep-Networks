"""
Adaptive Deep Networks 实验套件

根据 experiment_design.md 实现的完整实验框架
"""

__version__ = "1.0.0"

from utils.measurement import (
    measure_representation_burial,
    measure_attention_margin,
    analyze_margin_distribution,
    measure_gradient_statistics,
    measure_actual_flops,
    compute_flop_equivalent_config,
    compute_synergy_score
)

__all__ = [
    'measure_representation_burial',
    'measure_attention_margin',
    'analyze_margin_distribution',
    'measure_gradient_statistics',
    'measure_actual_flops',
    'compute_flop_equivalent_config',
    'compute_synergy_score',
]
