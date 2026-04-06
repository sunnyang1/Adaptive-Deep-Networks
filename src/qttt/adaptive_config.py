"""
Adaptive Configuration for qTTT

Dynamically adjusts qTTT parameters based on sequence length and gradient magnitude.
Based on: Section 3.3 of Adaptive Deep Networks paper

This allows qTTT to:
1. Use more steps for longer sequences (more complex optimization needed)
2. Adjust learning rate based on gradient magnitude (larger gradients → smaller LR)
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union


@dataclass
class AdaptiveQTTTConfig:
    """
    Adaptive configuration for qTTT.
    
    Automatically adjusts num_steps and learning_rate based on:
    - Sequence length (longer sequences need more optimization steps)
    - Gradient magnitude (larger gradients need smaller learning rates)
    
    Args:
        base_steps: Base number of qTTT steps (default: 4)
        max_steps: Maximum number of qTTT steps (default: 16)
        base_lr: Base learning rate (default: 0.01)
        min_lr: Minimum learning rate (default: 0.001)
        seq_len_thresholds: Sequence length thresholds for scaling [128, 512, 1024]
        scaling_mode: 'linear' or 'log' for sequence length scaling
    
    Example:
        >>> cfg = AdaptiveQTTTConfig(base_steps=4, max_steps=16)
        >>> steps = cfg.get_steps_for_seq_len(512)  # Returns scaled steps
        >>> lr = cfg.get_lr_for_gradient(0.5)       # Returns adjusted LR
        >>> qttt_config = cfg.to_dict(seq_len=512)  # Returns config dict
    """
    
    base_steps: int = 4
    max_steps: int = 16
    base_lr: float = 0.01
    min_lr: float = 0.001
    # Paper-scale contexts: below 4K vs 4K–32K vs 32K+ (not tokenizer window 128–1K)
    seq_len_thresholds: List[int] = field(default_factory=lambda: [4096, 32768])
    scaling_mode: str = 'linear'
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.base_steps <= self.max_steps, "base_steps must be <= max_steps"
        assert self.min_lr <= self.base_lr, "min_lr must be <= base_lr"
        assert self.scaling_mode in ['linear', 'log'], "scaling_mode must be 'linear' or 'log'"
    
    def get_steps_for_seq_len(self, seq_len: int) -> int:
        """
        Compute adaptive number of steps based on sequence length.
        
        Longer sequences typically need more optimization steps for effective
        query adaptation.
        
        Args:
            seq_len: Current sequence length
        
        Returns:
            Number of qTTT steps (integer)
        """
        return compute_adaptive_steps(
            seq_len=seq_len,
            base_steps=self.base_steps,
            max_steps=self.max_steps,
            thresholds=self.seq_len_thresholds,
            mode=self.scaling_mode
        )
    
    def get_lr_for_gradient(self, grad_norm: float) -> float:
        """
        Compute adaptive learning rate based on gradient magnitude.
        
        Larger gradients indicate potentially unstable optimization,
        so we reduce the learning rate.
        
        Args:
            grad_norm: L2 norm of the gradient
        
        Returns:
            Adjusted learning rate
        """
        return compute_adaptive_lr(
            grad_norm=grad_norm,
            base_lr=self.base_lr,
            min_lr=self.min_lr
        )
    
    def to_dict(self, seq_len: int = 128, grad_norm: Optional[float] = None) -> Dict[str, float]:
        """
        Convert to plain dictionary for qTTT.
        
        Args:
            seq_len: Current sequence length
            grad_norm: Optional gradient norm for LR adjustment
        
        Returns:
            Dictionary with 'num_steps' and 'learning_rate'
        """
        steps = self.get_steps_for_seq_len(seq_len)
        
        if grad_norm is not None:
            lr = self.get_lr_for_gradient(grad_norm)
        else:
            lr = self.base_lr
        
        return {
            'num_steps': steps,
            'learning_rate': lr,
            'span_length': 128,  # Default span
            'margin_temperature': 1.0,  # Default temperature
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'AdaptiveQTTTConfig':
        """
        Create from plain dictionary (backward compatibility).
        
        Args:
            d: Dictionary with 'num_steps' and 'learning_rate'
        
        Returns:
            AdaptiveQTTTConfig instance
        """
        return cls(
            base_steps=int(d.get('num_steps', 4)),
            base_lr=float(d.get('learning_rate', 0.01)),
        )


def compute_adaptive_steps(
    seq_len: int,
    base_steps: int = 4,
    max_steps: int = 16,
    thresholds: List[int] = None,
    mode: str = 'linear'
) -> int:
    """
    Compute adaptive number of qTTT steps based on sequence length.
    
    Args:
        seq_len: Current sequence length
        base_steps: Minimum number of steps
        max_steps: Maximum number of steps
        thresholds: Length thresholds for scaling levels
        mode: 'linear' or 'log' scaling
    
    Returns:
        Integer number of steps
    
    Example:
        >>> compute_adaptive_steps(256, base_steps=4, max_steps=16)
        8
    """
    if thresholds is None:
        thresholds = [128, 512, 1024]
    
    # Below first threshold: use base steps
    if seq_len <= thresholds[0]:
        return base_steps
    
    # Above last threshold: use max steps
    if seq_len >= thresholds[-1]:
        return max_steps
    
    # Find position between thresholds
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= seq_len < thresholds[i + 1]:
            # Interpolate between levels
            t = (seq_len - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
            
            if mode == 'linear':
                # Linear interpolation in step space
                step_range = max_steps - base_steps
                steps = base_steps + (step_range * (i + t) / len(thresholds))
            else:  # log
                # Logarithmic scaling
                log_len = math.log(seq_len)
                log_min = math.log(thresholds[0])
                log_max = math.log(thresholds[-1])
                ratio = (log_len - log_min) / (log_max - log_min)
                steps = base_steps + (max_steps - base_steps) * ratio
            
            return int(round(steps))
    
    return max_steps


def compute_adaptive_lr(
    grad_norm: float,
    base_lr: float = 0.01,
    min_lr: float = 0.001,
    target_grad: float = 0.1
) -> float:
    """
    Compute adaptive learning rate based on gradient magnitude.
    
    Uses inverse relationship: larger gradients → smaller learning rate
    to maintain stable optimization.
    
    Args:
        grad_norm: L2 norm of gradient
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        target_grad: Target gradient magnitude for base_lr
    
    Returns:
        Adjusted learning rate
    
    Example:
        >>> compute_adaptive_lr(1.0, base_lr=0.01)  # Large gradient
        0.001
        >>> compute_adaptive_lr(0.01, base_lr=0.01)  # Small gradient
        0.01
    """
    if grad_norm <= 0:
        return base_lr
    
    # Inverse scaling: lr = base_lr * target_grad / grad_norm
    # Clipped to [min_lr, base_lr]
    adjusted_lr = base_lr * (target_grad / grad_norm)
    
    # Clip to valid range
    return max(min_lr, min(base_lr, adjusted_lr))


def create_adaptive_config(
    mode: str = 'balanced'
) -> AdaptiveQTTTConfig:
    """
    Factory function to create preset adaptive configs.
    
    Args:
        mode: 'fast', 'balanced', or 'quality'
            - fast: Fewer steps, faster but less accurate
            - balanced: Moderate steps and LR
            - quality: More steps, better quality but slower
    
    Returns:
        Configured AdaptiveQTTTConfig
    """
    presets = {
        'fast': {
            'base_steps': 2,
            'max_steps': 8,
            'base_lr': 0.02,
            'min_lr': 0.005,
        },
        'balanced': {
            'base_steps': 4,
            'max_steps': 16,
            'base_lr': 0.01,
            'min_lr': 0.001,
        },
        'quality': {
            'base_steps': 8,
            'max_steps': 32,
            'base_lr': 0.005,
            'min_lr': 0.0005,
        },
    }
    
    if mode not in presets:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(presets.keys())}")
    
    return AdaptiveQTTTConfig(**presets[mode])
