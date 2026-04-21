"""
Adaptive Deep Networks (ADN) - 统一包

包含三个主要论文模块：
- ADN核心：AttnRes, qTTT, RaBitQ, Engram, Gating
- QASP：质量感知Stiefel投影
- MATDO-E：统一资源模型
"""

__version__ = "0.2.0"

from adn.core.config import ModelConfig, ADNConfig
from adn.core.base import RMSNorm

__all__ = ["__version__", "ModelConfig", "ADNConfig", "RMSNorm"]
