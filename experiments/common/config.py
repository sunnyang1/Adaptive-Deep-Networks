"""
Configuration Management for Experiments

Centralized configuration using Pydantic for validation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
import json


@dataclass(frozen=True)
class ModelSizeConfig:
    """Standard model size configurations from paper Table A1."""
    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    mlp_ratio: int = 4
    num_blocks: int = 8
    max_qttt_steps: int = 32
    qttt_span_length: int = 128
    
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads
    
    @property
    def param_count(self) -> float:
        """Estimate parameter count in billions."""
        # Embedding + Attention + FFN + Layer Norm
        embed = self.vocab_size * self.hidden_dim
        attn = self.num_layers * (4 * self.hidden_dim * self.head_dim * self.num_heads)
        ffn = self.num_layers * (2 * self.hidden_dim * self.hidden_dim * self.mlp_ratio)
        norm = self.num_layers * 2 * self.hidden_dim
        return (embed + attn + ffn + norm) / 1e9


# Predefined model sizes
# Optimized for AttnRes (Paper §5.4.1): d_model/L_b ≈ 45, H/L_b ≈ 0.3
# Small: 32L/1408H/8Hd = 1.1B, d/L=44.0, H/L=0.25
# Medium: 56L/2496H/16Hd = 5.7B, d/L=44.6, H/L=0.29
# Large: 88L/4032H/18Hd = 23.0B, d/L=45.8, H/L=0.21
MODEL_SIZES = {
    'small': ModelSizeConfig(
        hidden_dim=1408,
        num_layers=32,
        num_heads=8,
        num_blocks=8,
        max_qttt_steps=16,
    ),
    'medium': ModelSizeConfig(
        hidden_dim=2496,
        num_layers=56,
        num_heads=16,
        num_blocks=8,
        max_qttt_steps=32,
    ),
    'large': ModelSizeConfig(
        hidden_dim=4032,
        num_layers=88,
        num_heads=18,
        num_blocks=11,
        max_qttt_steps=32,
        qttt_span_length=256,
    ),
}


@dataclass
class ExperimentConfig:
    """Base configuration for all experiments."""
    
    # Experiment identification
    name: str = "unnamed_experiment"
    category: str = "core"  # core, validation, real_model
    description: str = ""
    
    # Model configuration
    model_size: str = "small"  # small, medium, large
    model_config: Optional[ModelSizeConfig] = None
    
    # Paths
    output_dir: Path = field(default_factory=lambda: Path("results"))
    checkpoint_path: Optional[Path] = None
    
    # Execution
    device: str = "auto"  # auto, cuda, cpu, mps
    seed: int = 42
    deterministic: bool = False
    
    # Compute resources
    num_workers: int = 4
    batch_size: int = 1
    max_memory_gb: Optional[float] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # Experiment-specific settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Resolve model config from size string
        if self.model_config is None and self.model_size in MODEL_SIZES:
            object.__setattr__(
                self, 
                'model_config', 
                MODEL_SIZES[self.model_size]
            )
        
        # Convert output_dir to Path if string
        if isinstance(self.output_dir, str):
            object.__setattr__(self, 'output_dir', Path(self.output_dir))
        
        # Convert checkpoint_path to Path if string
        if isinstance(self.checkpoint_path, str):
            object.__setattr__(self, 'checkpoint_path', Path(self.checkpoint_path))
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'model_size': self.model_size,
            'output_dir': str(self.output_dir),
            'device': self.device,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'log_level': self.log_level,
            'custom_settings': self.custom_settings,
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        data = {
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'model_size': self.model_size,
            'output_dir': str(self.output_dir),
            'checkpoint_path': str(self.checkpoint_path) if self.checkpoint_path else None,
            'device': self.device,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'log_level': self.log_level,
            'custom_settings': self.custom_settings,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Paper validation targets (for reference)
VALIDATION_TARGETS = {
    'representation_burial': {
        'prenorm_attenuation': 13.5,
        'attnres_attenuation': 1.06,
        'effective_depth_threshold': 0.5,
    },
    'gradient_flow': {
        'prenorm_cv': 0.84,
        'attnres_cv': 0.11,
        'tolerance': 0.05,
    },
    'needle_haystack': {
        4096: 98.5,
        16384: 91.3,
        65536: 78.2,
        131072: 68.2,
        'average': 86.9,
    },
    'turboquant': {
        'compression_ratio': 6.0,
        'kv_cache_reduction': 5.7,
        'throughput_gain': 8.0,
    },
}
