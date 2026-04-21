"""
实验工具函数
"""

from .measurement import (
    measure_representation_burial,
    measure_attention_margin,
    analyze_margin_distribution,
    measure_gradient_statistics,
    measure_actual_flops,
    compute_flop_equivalent_config,
    compute_synergy_score,
    LayerContribution,
    MarginStats
)

__all__ = [
    'measure_representation_burial',
    'measure_attention_margin',
    'analyze_margin_distribution',
    'measure_gradient_statistics',
    'measure_actual_flops',
    'compute_flop_equivalent_config',
    'compute_synergy_score',
    'LayerContribution',
    'MarginStats'
]
