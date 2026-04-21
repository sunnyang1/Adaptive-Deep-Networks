"""
Model Configurations for Adaptive Deep Networks

Compatibility shim - redirects to adn.core.config.
Kept for backward compatibility with existing imports.
"""

from adn.core.config import (
    ModelConfig,
    ADNConfig,
    AttnResSmallConfig,
    AttnResT4Config,
    AttnResMediumConfig,
    AttnResLargeConfig,
    TrainingConfig,
    ValidationConfig,
    CONFIGS,
    get_config,
    get_model_size_params,
    print_config,
)

__all__ = [
    "ModelConfig",
    "ADNConfig",
    "AttnResSmallConfig",
    "AttnResT4Config",
    "AttnResMediumConfig",
    "AttnResLargeConfig",
    "TrainingConfig",
    "ValidationConfig",
    "CONFIGS",
    "get_config",
    "get_model_size_params",
    "print_config",
]
