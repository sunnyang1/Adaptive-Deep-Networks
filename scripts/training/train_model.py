#!/usr/bin/env python3
"""
Canonical training entrypoint for Adaptive Deep Networks.

This script replaces the previous legacy implementation and now dispatches all
training through the maintained `BaseTrainer` pipeline used by
`train_small.py`, `train_medium.py`, `train_large.py`, and `train_t4.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.base_trainer import BaseTrainer, get_common_parser
from src.models.configs import (
    AttnResLargeConfig,
    AttnResMediumConfig,
    AttnResSmallConfig,
    AttnResT4Config,
    ModelConfig,
)


SIZE_DEFAULTS = {
    "small": {
        "epochs": 3,
        "batch_size": 4,
        "lr": 3e-4,
        "seq_len": 512,
        "train_samples": 10000,
        "val_samples": 1000,
    },
    "medium": {
        "epochs": 3,
        "batch_size": 2,
        "lr": 2e-4,
        "seq_len": 512,
        "train_samples": 50000,
        "val_samples": 5000,
    },
    "large": {
        "epochs": 3,
        "batch_size": 1,
        "lr": 1e-4,
        "seq_len": 512,
        "train_samples": 100000,
        "val_samples": 10000,
        "grad_accum": 4,
    },
    "t4": {
        "epochs": 1,
        "batch_size": 1,
        "lr": 3e-4,
        "seq_len": 128,
        "train_samples": 1000,
        "val_samples": 128,
    },
}


class UnifiedModelTrainer(BaseTrainer):
    """Trainer that selects model config by --model-size."""

    def _get_model_config(self) -> ModelConfig:
        if self.args.model_size == "small":
            return AttnResSmallConfig()
        if self.args.model_size == "medium":
            return AttnResMediumConfig()
        if self.args.model_size == "large":
            return AttnResLargeConfig()
        if self.args.model_size == "t4":
            return AttnResT4Config()
        raise ValueError(f"Unsupported model size: {self.args.model_size}")

    def get_model_size_name(self) -> str:
        return self.args.model_size

    def print_model_info(self) -> None:
        config = self.config
        print(f"\n{'=' * 70}")
        print(f"{self.args.model_size.upper()} MODEL TRAINING")
        print(f"{'=' * 70}")
        print("Architecture:")
        print(f"  Layers: {config.num_layers}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Attention heads: {config.num_heads}")
        print(f"  AttnRes blocks: {config.num_blocks}")
        print(f"  d_model/L_b = {config.hidden_dim / config.num_layers:.1f} (paper target: ~45)")
        print(f"  H/L_b = {config.num_heads / config.num_layers:.3f} (paper target: ~0.3)")
        print(f"{'=' * 70}\n")


def main() -> None:
    parser = get_common_parser()
    parser.description = "Train Adaptive Deep Networks (unified entrypoint)"
    parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["small", "medium", "large", "t4"],
        help="Model size preset",
    )
    args = parser.parse_args()

    parser.set_defaults(**SIZE_DEFAULTS[args.model_size])
    args = parser.parse_args()

    trainer = UnifiedModelTrainer(args)
    trainer.print_model_info()
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
