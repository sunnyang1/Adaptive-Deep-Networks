"""
Model Configurations for Adaptive Deep Networks

Based on: Table A1 from Adaptive Deep Networks paper (Appendix A.2)
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Base model configuration."""
    
    # Architecture
    num_layers: int = 32
    hidden_dim: int = 4096
    num_heads: int = 32
    mlp_ratio: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 32768
    
    # AttnRes
    num_blocks: int = 8
    
    # qTTT
    max_qttt_steps: int = 32
    qttt_span_length: int = 128
    qttt_learning_rate: float = 0.005
    
    # Gating
    gating_target_rate: float = 0.3
    
    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0
        self.head_dim = self.hidden_dim // self.num_heads


@dataclass  
class AttnResSmallConfig(ModelConfig):
    """1.5B parameter model (AttnRes-S)."""
    
    num_layers: int = 32
    hidden_dim: int = 2048
    num_heads: int = 32
    num_blocks: int = 8
    max_qttt_steps: int = 16
    qttt_span_length: int = 128


@dataclass
class AttnResMediumConfig(ModelConfig):
    """7B parameter model (AttnRes-M)."""
    
    num_layers: int = 32
    hidden_dim: int = 4096
    num_heads: int = 32
    num_blocks: int = 8
    max_qttt_steps: int = 32
    qttt_span_length: int = 128


@dataclass
class AttnResLargeConfig(ModelConfig):
    """50B parameter model (AttnRes-L)."""
    
    num_layers: int = 64
    hidden_dim: int = 5120
    num_heads: int = 40
    num_blocks: int = 16
    max_qttt_steps: int = 32
    qttt_span_length: int = 256


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # From Table A1
    batch_size_tokens: int = 4_000_000  # 4M tokens
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine"
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    gradient_clipping: float = 1.0
    
    # AttnRes specific
    pseudo_query_lr_multiplier: float = 1.0  # Same LR as other params
    
    # Optimization
    use_gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000


@dataclass
class ValidationConfig:
    """Configuration for validation experiments."""
    
    # Needle-in-Haystack
    nih_context_lengths: List[int] = field(default_factory=lambda: [
        1024, 4096, 16384, 32768, 65536, 131072, 262144
    ])
    nih_depths_per_length: int = 10
    nih_num_trials: int = 5
    
    # MATH
    math_difficulty_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    math_max_samples: Optional[int] = None  # None = all
    
    # Efficiency
    measure_flops: bool = True
    measure_latency: bool = True
    measure_memory: bool = True
    
    # Output
    output_dir: str = "./validation_results"
    save_attention_maps: bool = False
    save_detailed_logs: bool = True


# Registry of configurations
CONFIGS = {
    "small": AttnResSmallConfig,
    "medium": AttnResMediumConfig,
    "large": AttnResLargeConfig,
}


def get_config(name: str) -> ModelConfig:
    """Get configuration by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()


def get_model_size_params(config: ModelConfig) -> int:
    """Calculate approximate parameter count."""
    # Embedding
    embedding_params = config.vocab_size * config.hidden_dim
    
    # Per layer
    # Attention: 4 * hidden_dim^2 (Q, K, V, O projections)
    attn_params = 4 * config.hidden_dim * config.hidden_dim
    # MLP: 2 * hidden_dim * mlp_dim (up and down projections)
    mlp_dim = config.hidden_dim * config.mlp_ratio
    mlp_params = 3 * config.hidden_dim * mlp_dim
    # AttnRes pseudo-queries: 2 * hidden_dim (attn + mlp)
    attnres_params = 2 * config.hidden_dim
    
    layer_params = attn_params + mlp_params + attnres_params
    
    # Total
    total = embedding_params + config.num_layers * layer_params
    
    return total


def print_config(config: ModelConfig):
    """Pretty print configuration."""
    params = get_model_size_params(config)
    param_str = f"{params / 1e9:.1f}B" if params > 1e9 else f"{params / 1e6:.1f}M"
    
    print("=" * 50)
    print(f"Model Configuration")
    print("=" * 50)
    print(f"Parameters: {param_str}")
    print(f"Layers: {config.num_layers}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Num heads: {config.num_heads}")
    print(f"MLP ratio: {config.mlp_ratio}")
    print(f"Num blocks (AttnRes): {config.num_blocks}")
    print(f"Max qTTT steps: {config.max_qttt_steps}")
    print(f"qTTT span: {config.qttt_span_length}")
    print("=" * 50)
