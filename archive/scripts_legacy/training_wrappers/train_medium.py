#!/usr/bin/env python3
"""
Train Medium Model (AttnRes-M)

Configuration: 56L/2496H/16Hd = ~5.7B params
Architecture: d_model/L_b = 44.6, H/L_b = 0.29

Usage:
    # Single GPU training (requires 16GB+ VRAM)
    python scripts/training/train_medium.py --output-dir results/medium_model
    
    # Multi-GPU distributed training
    torchrun --nproc_per_node=4 scripts/training/train_medium.py \
        --output-dir results/medium_model \
        --batch-size 2
    
    # With DeepSpeed ZeRO-3 for memory efficiency
    deepspeed --num_gpus=4 scripts/training/train_medium.py \
        --output-dir results/medium_model \
        --deepspeed configs/ds_config_h20.json

Hardware Requirements:
    - Minimum: 1x A100 40GB or 2x RTX 3090
    - Recommended: 4x A100 80GB or 8x H20 96GB
    - Training time: ~12-24 hours on 4x A100 for 3 epochs
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.base_trainer import BaseTrainer, get_common_parser
from src.models.configs import AttnResMediumConfig, ModelConfig


class MediumModelTrainer(BaseTrainer):
    """Trainer for Medium model (5.7B params)."""

    def _get_model_config(self) -> ModelConfig:
        """Return Medium model configuration."""
        return AttnResMediumConfig()

    def get_model_size_name(self) -> str:
        """Return model size name."""
        return "medium"

    def print_model_info(self):
        """Print Medium model specific information."""
        config = self.config
        print(f"\n{'='*70}")
        print(f"MEDIUM MODEL (AttnRes-M) - ~5.7B Parameters")
        print(f"{'='*70}")
        print(f"Architecture:")
        print(f"  Layers: {config.num_layers}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Attention heads: {config.num_heads}")
        print(f"  Head dim: {config.hidden_dim // config.num_heads}")
        print(f"  AttnRes blocks: {config.num_blocks}")
        print(f"  Layers per block: {config.num_layers // config.num_blocks}")
        print(f"\nOptimal Ratios (Paper §5.4.1):")
        print(f"  d_model/L_b = {config.hidden_dim / config.num_layers:.1f} (optimal: ~45)")
        print(f"  H/L_b = {config.num_heads / config.num_layers:.3f} (optimal: ~0.3)")
        print(f"\nHardware Requirements:")
        print(f"  GPU Memory: ~24GB (BF16) or ~48GB (FP32)")
        print(f"  Recommended: 4x A100 80GB or 8x H20 96GB")
        print(f"{'='*70}\n")


def main():
    parser = get_common_parser()
    parser.description = "Train Medium Model (AttnRes-M, ~5.7B params)"

    # Medium model specific defaults
    parser.set_defaults(
        epochs=3,
        batch_size=2,  # Smaller batch for larger model
        lr=2e-4,  # Slightly lower LR for larger model
        seq_len=512,
        train_samples=50000,
        val_samples=5000,
    )

    # Add distributed training option
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file path")

    args = parser.parse_args()

    # Create trainer
    trainer = MediumModelTrainer(args)
    trainer.print_model_info()

    # Setup and train
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
