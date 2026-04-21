"""
Query-only Test-Time Training (qTTT) Module

Provides:
- QueryOnlyTTT: Standard qTTT with frozen KV cache
- PolarQTTT: Polar-coordinate qTTT with 50% parameter reduction
- DepthPriorityController: Computation allocation policy
"""

from .adaptation import qTTTConfig, KVCache, qttt_adapt, QueryOnlyTTT, AdaptiveInference
from .margin_loss import MarginMaximizationLoss
from .polar_adaptation import (
    PolarQTTTConfig,
    SphericalSGD,
    QueryAdaptationPolarAdapter,
    PolarQTTT,
    DepthPriorityController,
)

__all__ = [
    "qTTTConfig",
    "KVCache",
    "qttt_adapt",
    "QueryOnlyTTT",
    "AdaptiveInference",
    "MarginMaximizationLoss",
    "PolarQTTTConfig",
    "SphericalSGD",
    "QueryAdaptationPolarAdapter",
    "PolarQTTT",
    "DepthPriorityController",
]

# Backward compatibility aliases
PolarQueryAdapter = QueryAdaptationPolarAdapter
