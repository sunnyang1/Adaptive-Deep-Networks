"""
Dynamic Computation Gating Module

Provides:
- DynamicThreshold: Base class for threshold calibration
- EMAThreshold: Exponential moving average calibration
- TargetRateThreshold: Maintain target adaptation rate
- GatingController: High-level gating controller
- DepthPriorityGatingController: RaBitQ-aware depth-priority controller
"""

from .reconstruction import ReconstructionLoss
from .threshold import (
    DynamicThreshold,
    EMAThreshold,
    TargetRateThreshold,
    HybridThreshold,
    GatingController
)
from .depth_priority import (
    DepthPriorityGatingController,
    AdaptiveThresholdWithDepthPriority,
    create_depth_priority_controller
)

__all__ = [
    'ReconstructionLoss',
    'DynamicThreshold',
    'EMAThreshold',
    'TargetRateThreshold',
    'HybridThreshold',
    'GatingController',
    'DepthPriorityGatingController',
    'AdaptiveThresholdWithDepthPriority',
    'create_depth_priority_controller',
]
