"""
Training Scripts for Adaptive Deep Networks

This module provides training scripts for all model sizes:
- Small (AttnRes-S): 1.1B params, 32L/1408H/8Hd
- Medium (AttnRes-M): 5.7B params, 56L/2496H/16Hd  
- Large (AttnRes-L): 23B params, 88L/4032H/18Hd

All configurations optimized for paper §5.4.1:
- d_model/L_b ≈ 45 (AttnRes optimal)
- H/L_b ≈ 0.3

Usage:
    # Small model (CPU-friendly, 1 GPU)
    python -m scripts.training.train_small --output-dir results/small
    
    # Medium model (1-4 GPUs)
    python -m scripts.training.train_medium --output-dir results/medium
    
    # Large model (8+ GPUs, requires distributed training)
    torchrun --nproc_per_node=8 -m scripts.training.train_large --output-dir results/large
"""

from .base_trainer import BaseTrainer, get_common_parser

__all__ = ['BaseTrainer', 'get_common_parser']
