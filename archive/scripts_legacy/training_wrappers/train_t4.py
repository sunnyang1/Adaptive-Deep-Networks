#!/usr/bin/env python3
"""
Train T4-Friendly Model

Configuration: 12L/768H/12Hd, designed to fit 15GB-class GPUs.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.base_trainer import BaseTrainer, get_common_parser
from src.models.configs import AttnResT4Config, ModelConfig


class T4ModelTrainer(BaseTrainer):
    """Trainer for T4-friendly model."""

    def _get_model_config(self) -> ModelConfig:
        return AttnResT4Config()

    def get_model_size_name(self) -> str:
        return "t4"

    def print_model_info(self):
        config = self.config
        print(f"\n{'='*70}")
        print("T4 MODEL (T4-Friendly) - ~125M Parameters")
        print(f"{'='*70}")
        print("Architecture:")
        print(f"  Layers: {config.num_layers}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Attention heads: {config.num_heads}")
        print(f"  Head dim: {config.hidden_dim // config.num_heads}")
        print(f"  AttnRes blocks: {config.num_blocks}")
        print(f"  Layers per block: {config.num_layers // config.num_blocks}")
        print("\nPaper-ratio alignment (§5.4.1):")
        print(f"  d_model/L_b = {config.hidden_dim / config.num_layers:.1f} (target: ~45)")
        print(f"  H/L_b = {config.num_heads / config.num_layers:.3f} (target: ~0.3)")
        print(f"{'='*70}\n")


def main():
    parser = get_common_parser()
    parser.description = "Train T4-Friendly Model (~150M params)"

    parser.set_defaults(
        epochs=1,
        batch_size=1,
        lr=3e-4,
        seq_len=128,
        train_samples=1000,
        val_samples=128,
    )

    args = parser.parse_args()

    trainer = T4ModelTrainer(args)
    trainer.print_model_info()
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
