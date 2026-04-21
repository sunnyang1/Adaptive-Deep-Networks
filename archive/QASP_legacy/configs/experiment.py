"""Experiment-facing QASP configuration composition."""

from dataclasses import dataclass, field

from QASP.configs.model import ModelConfig
from QASP.configs.qasp import QASPConfig


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration for lightweight QASP runs."""

    model: ModelConfig
    qasp: QASPConfig = field(default_factory=QASPConfig)
    seed: int = 42
    device: str = "cpu"

