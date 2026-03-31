#!/usr/bin/env python3
"""
Train Large Model (AttnRes-L)

Configuration: 88L/4032H/18Hd = ~23B params
Architecture: d_model/L_b = 45.8, H/L_b = 0.21

Usage:
    # Multi-GPU distributed training (required)
    torchrun --nproc_per_node=8 scripts/training/train_large.py \
        --output-dir results/large_model \
        --batch-size 1
    
    # With DeepSpeed ZeRO-3 (recommended)
    deepspeed --num_gpus=8 scripts/training/train_large.py \
        --output-dir results/large_model \
        --deepspeed configs/ds_config_h20.json \
        --batch-size 1
    
    # With CPU offloading for optimizer states
    deepspeed --num_gpus=8 scripts/training/train_large.py \
        --output-dir results/large_model \
        --deepspeed configs/ds_config_zero3_offload.json

Hardware Requirements:
    - Minimum: 8x A100 80GB with DeepSpeed ZeRO-3
    - Recommended: 8x H20 96GB or 16x A100 80GB
    - Alternative: 4x A100 80GB with CPU offloading (slower)
    - Training time: ~3-7 days on 8x A100 for 3 epochs
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.base_trainer import BaseTrainer, get_common_parser
from src.models.configs import AttnResLargeConfig, ModelConfig


class LargeModelTrainer(BaseTrainer):
    """Trainer for Large model (23B params)."""
    
    def _get_model_config(self) -> ModelConfig:
        """Return Large model configuration."""
        return AttnResLargeConfig()
    
    def get_model_size_name(self) -> str:
        """Return model size name."""
        return "large"
    
    def print_model_info(self):
        """Print Large model specific information."""
        config = self.config
        print(f"\n{'='*70}")
        print(f"LARGE MODEL (AttnRes-L) - ~23B Parameters")
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
        print(f"  GPU Memory: ~92GB per GPU (BF16) with ZeRO-3")
        print(f"  Recommended: 8x H20 96GB or 16x A100 80GB")
        print(f"  Training time: ~3-7 days on 8x A100 for 3 epochs")
        print(f"{'='*70}\n")
    
    def validate_hardware(self):
        """Validate that hardware is sufficient for large model."""
        import torch
        
        if not torch.cuda.is_available():
            print("WARNING: Large model requires GPU. CPU training is not practical.")
            print("Consider using the Small or Medium model instead.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 4:
            print(f"WARNING: Large model training requires at least 4 GPUs (found {num_gpus}).")
            print("Consider using DeepSpeed ZeRO-3 with CPU offloading.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)


def main():
    parser = get_common_parser()
    parser.description = 'Train Large Model (AttnRes-L, ~23B params)'
    
    # Large model specific defaults
    parser.set_defaults(
        epochs=3,
        batch_size=1,  # Must be 1 for large model
        lr=1e-4,  # Lower LR for large model
        seq_len=512,
        train_samples=100000,
        val_samples=10000,
        grad_accum=4,  # Gradient accumulation for effective larger batch
    )
    
    # Add distributed training options (required for large model)
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training (required for large model)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--deepspeed', type=str, default=None,
                       help='DeepSpeed config file path (highly recommended)')
    parser.add_argument('--skip-hardware-check', action='store_true',
                       help='Skip hardware validation')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = LargeModelTrainer(args)
    trainer.print_model_info()
    
    # Validate hardware
    if not args.skip_hardware_check:
        trainer.validate_hardware()
    
    # Setup and train
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
