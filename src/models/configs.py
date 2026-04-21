"""
Model Configurations for Adaptive Deep Networks

Based on: Table A1 from Adaptive Deep Networks paper (Appendix A.2)
"""

from dataclasses import dataclass, field
from typing import Optional, List

from src.engram.config import EngramConfig


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

    # Engram
    use_engram: bool = False
    engram_config: Optional[EngramConfig] = None

    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0
        self.head_dim = self.hidden_dim // self.num_heads


@dataclass
class AttnResSmallConfig(ModelConfig):
    """1.1B parameter model (AttnRes-S).

    Architecture optimized based on Attention Residuals paper (§5.4.1):
    - d_model/L_b ≈ 44 (close to optimal ~45 for AttnRes)
    - H/L_b = 0.25 (close to optimal ~0.3)
    - N=8 blocks (paper-recommended sweet spot, §5.3)
    - Fixed at 32 layers for experimentation

    32L/1408H/8Hd = ~1.1B params, d_model/L_b=44.0, H/L_b=0.25
    """

    num_layers: int = 32  # Fixed at 32 layers
    hidden_dim: int = 1408  # 1408/32 = 44.0 (paper optimal ~45)
    num_heads: int = 8  # 8/32 = 0.25 (close to paper optimal 0.3)
    num_blocks: int = 8  # Paper: N=8 recovers most benefit
    max_qttt_steps: int = 16
    qttt_span_length: int = 128


@dataclass
class AttnResT4Config(ModelConfig):
    """T4-friendly configuration (~125M params with GPT-2 vocab).

    This config is intended for smoke/integration training on 15GB GPUs.
    Aligned to §5.4.1 style ratios as closely as possible under T4 constraints:
    - d_model/L_b = 640/14 = 45.7 (near optimal ~45)
    - H/L_b = 4/14 = 0.286 (near optimal ~0.3)
    """

    num_layers: int = 14
    hidden_dim: int = 640
    num_heads: int = 4
    num_blocks: int = 7
    max_qttt_steps: int = 8
    qttt_span_length: int = 64


@dataclass
class AttnResMediumConfig(ModelConfig):
    """5.7B parameter model (AttnRes-M).

    Architecture optimized based on Attention Residuals paper (§5.4.1):
    - d_model/L_b ≈ 44.6 (close to optimal ~45 for AttnRes)
    - H/L_b = 0.29 (close to paper optimal ~0.3)
    - N=8 blocks (paper-recommended sweet spot, §5.3)

    56L/2496H/16Hd = ~5.7B params, d_model/L_b=44.6, H/L_b=0.29
    """

    num_layers: int = 56  # 56 layers
    hidden_dim: int = 2496  # 2496/56 = 44.6 (paper optimal ~45)
    num_heads: int = 16  # 16/56 = 0.29 (close to paper optimal 0.3)
    num_blocks: int = 8  # Paper: N=8 recovers most benefit
    max_qttt_steps: int = 32
    qttt_span_length: int = 128


@dataclass
class AttnResLargeConfig(ModelConfig):
    """23B parameter model (AttnRes-L).

    Architecture optimized based on Attention Residuals paper (§5.4.1):
    - d_model/L_b ≈ 45.8 (close to optimal ~45 for AttnRes)
    - H/L_b = 0.20 (lower than optimal ~0.3 but allows larger head dim)
    - N=11 blocks (for 88 layers)

    88L/4032H/18Hd = ~23B params, d_model/L_b=45.8, H/L_b=0.20
    """

    num_layers: int = 88  # 88 layers
    hidden_dim: int = 4032  # 4032/88 = 45.8 (paper optimal ~45)
    num_heads: int = 18  # 18/88 = 0.20
    num_blocks: int = 11  # N=11 blocks, S=8 layers per block
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
    nih_context_lengths: List[int] = field(
        default_factory=lambda: [1024, 4096, 16384, 32768, 65536, 131072, 262144]
    )
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
    "t4": AttnResT4Config,
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
