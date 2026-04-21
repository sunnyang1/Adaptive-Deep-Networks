#!/usr/bin/env python3
"""
Train Small Model (AttnRes-S)

Configuration: 32L/1408H/8Hd = ~1.1B params
Architecture: d_model/L_b = 44.0, H/L_b = 0.25

Usage:
    # Single GPU training (CPU-friendly with small batch)
    python scripts/training/train_small.py --output-dir results/small_model
    
    # With custom settings
    python scripts/training/train_small.py \
        --output-dir results/small_model \
        --epochs 5 \
        --batch-size 8 \
        --lr 3e-4 \
        --seq-len 512
    
    # Quick test run
    python scripts/training/train_small.py \
        --output-dir results/small_test \
        --epochs 1 \
        --train-samples 1000 \
        --val-samples 100
    
    # With Llama-2 tokenizer (requires HuggingFace auth)
    python scripts/training/train_small.py \
        --output-dir results/small_model \
        --epochs 5 \
        --batch-size 1 \
        --grad-accum 8 \
        --lr 1e-4 \
        --warmup-steps 500 \
        --seq-len 32768 \
        --dataset-name HuggingFaceFW/fineweb-edu \
        --dataset-config sample-10BT \
        --tokenizer-name meta-llama/Llama-2-7b-hf \
        --hf-token $HF_TOKEN \
        --streaming
    
    # Alternative high-quality datasets (Llama-2 style training data)
    # RedPajama (1.2T tokens, diverse web crawl + academic)
    python scripts/training/train_small.py ... --dataset-name togethercomputer/RedPajama-Data-1T
    
    # SlimPajama (627B deduplicated tokens, higher quality)
    python scripts/training/train_small.py ... --dataset-name cerebras/SlimPajama-627B
    
    # FineWeb-Edu (1.3T educational content tokens)
    python scripts/training/train_small.py ... --dataset-name HuggingFaceFW/fineweb-edu

Hardware Requirements:
    - Minimum: 8GB RAM, CPU
    - Recommended: 16GB RAM, 1x GPU with 8GB+ VRAM
    - For 32K context on A100: batch-size 1 + grad-accum 8
    - Training time: ~2-4 hours on GPU for 3 epochs (seq_len 512)
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.base_trainer import BaseTrainer, get_common_parser
from src.models.configs import AttnResSmallConfig, ModelConfig


class SmallModelTrainer(BaseTrainer):
    """Trainer for Small model (1.1B params)."""

    def _get_model_config(self) -> ModelConfig:
        """Return Small model configuration."""
        return AttnResSmallConfig()

    def get_model_size_name(self) -> str:
        """Return model size name."""
        return "small"

    def print_model_info(self):
        """Print Small model specific information."""
        config = self.config
        print(f"\n{'='*70}")
        print(f"SMALL MODEL (AttnRes-S) - ~1.1B Parameters")
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
        print(f"{'='*70}\n")


def main():
    parser = get_common_parser()
    parser.description = "Train Small Model (AttnRes-S, ~1.1B params)"

    # Small model specific defaults
    parser.set_defaults(
        epochs=3,
        batch_size=4,
        lr=3e-4,
        seq_len=512,
        train_samples=10000,
        val_samples=1000,
    )

    args = parser.parse_args()

    # Create trainer
    trainer = SmallModelTrainer(args)
    trainer.print_model_info()

    # Setup and train
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
