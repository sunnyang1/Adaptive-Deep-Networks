"""Query-only Test-Time Training (qTTT) implementation."""

from .adaptation import QueryOnlyTTT, qttt_adapt
from .margin_loss import MarginMaximizationLoss, compute_margin_loss

__all__ = [
    'QueryOnlyTTT',
    'qttt_adapt',
    'MarginMaximizationLoss',
    'compute_margin_loss'
]
