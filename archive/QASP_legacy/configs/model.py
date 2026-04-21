"""Model-level configuration for QASP experiments."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Basic model metadata needed by adaptation utilities."""

    hidden_size: int
    rank: int
    dtype: str = "float32"

