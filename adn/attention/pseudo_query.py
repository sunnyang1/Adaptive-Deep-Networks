"""
Pseudo-Query Vector Management

Manages learned pseudo-query vectors for Block AttnRes.
Key features:
- Zero initialization for training stability
- Attention weight distribution monitoring
- Specialization pattern tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None


class PseudoQuery(nn.Module):
    """
    Base pseudo-query interface.

    Provides a common interface for all pseudo-query implementations.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self) -> torch.Tensor:
        """Return the pseudo-query vector."""
        raise NotImplementedError


class LearnablePseudoQuery(PseudoQuery):
    """
    Simple learnable pseudo-query vector.

    A single learnable vector used as pseudo-query for AttnRes.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps)
        self.query = nn.Parameter(torch.zeros(dim))

    def forward(self) -> torch.Tensor:
        """Return the pseudo-query vector."""
        return self.query

    def reset_parameters(self):
        """Reset to zero initialization."""
        nn.init.zeros_(self.query)


class PseudoQueryManager(nn.Module):
    """
    Manages pseudo-query vectors for all layers.

    Each layer maintains two pseudo-queries:
    - One for attention layer
    - One for MLP layer
    """

    def __init__(self, num_layers: int, dim: int, num_blocks: int = 8, eps: float = 1e-6):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.num_blocks = num_blocks
        self.eps = eps

        # Create pseudo-queries for each layer (zero initialized)
        # Shape: [num_layers, 2] where 2 = (attn, mlp)
        self.pseudo_queries = nn.Parameter(torch.zeros(num_layers, 2, dim))

        # Track statistics
        self.register_buffer("attention_history", torch.zeros(100, num_layers, num_blocks + 1))
        self.history_ptr = 0
        self.register_buffer("entropy_history", torch.zeros(100, num_layers))

    def get_pseudo_query(self, layer_idx: int, is_mlp: bool = False) -> torch.Tensor:
        """Get pseudo-query for specific layer."""
        q_type = 1 if is_mlp else 0
        return self.pseudo_queries[layer_idx, q_type]

    def get_all_queries(self) -> torch.Tensor:
        """Get all pseudo-queries."""
        return self.pseudo_queries

    def reset_parameters(self):
        """Reset all pseudo-queries to zero."""
        nn.init.zeros_(self.pseudo_queries)

    def compute_attention_weights(
        self, layer_idx: int, block_representations: List[torch.Tensor], is_mlp: bool = False
    ) -> torch.Tensor:
        """
        Compute attention weights for visualization/analysis.

        Args:
            layer_idx: Layer index
            block_representations: List of block representations
            is_mlp: Whether this is for MLP layer

        Returns:
            Attention weights [num_blocks] averaged over batch and seq dims
        """
        pseudo_query = self.get_pseudo_query(layer_idx, is_mlp)

        if len(block_representations) == 0:
            return torch.ones(1) / 1  # Uniform for first block

        # Stack blocks: [N, B, T, D]
        V = torch.stack(block_representations, dim=0)

        # RMSNorm: x / sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(V**2, dim=-1, keepdim=True) + self.eps)
        K = V / rms

        # Compute attention scores: [N, B, T]
        # einsum: 'n b t d, d -> n b t'
        logits = torch.einsum("n b t d, d -> n b t", K, pseudo_query)
        logits = logits / (self.dim**0.5)

        # Average over batch and sequence dimensions
        logits = logits.mean(dim=(1, 2))  # [N]
        weights = F.softmax(logits, dim=0)

        return weights

    def compute_entropy(self, weights: torch.Tensor) -> float:
        """Compute entropy of attention weights (measure of specialization)."""
        # Entropy = -sum(p * log(p))
        entropy = -(weights * torch.log(weights + self.eps)).sum()
        return entropy.item()

    def log_attention_stats(self, layer_idx: int, weights: torch.Tensor):
        """Log attention statistics for analysis."""
        idx = self.history_ptr % 100

        # Pad weights to fixed size
        padded = torch.zeros(self.num_blocks + 1, device=weights.device)
        padded[: len(weights)] = weights
        self.attention_history[idx, layer_idx] = padded.cpu()

        # Compute and log entropy
        entropy = self.compute_entropy(weights)
        self.entropy_history[idx, layer_idx] = entropy

        self.history_ptr += 1

    def get_specialization_report(self) -> Dict[str, any]:
        """
        Generate report on pseudo-query specialization.

        Returns:
            Dictionary with:
            - mean_entropy: Average entropy across layers
            - entropy_by_layer: Per-layer entropy
            - uniform_entropy: Expected entropy for uniform distribution
            - specialization_ratio: How much more focused than uniform
        """
        if self.history_ptr == 0:
            return {"error": "No history recorded yet"}

        num_entries = min(self.history_ptr, 100)

        # Compute mean entropy per layer
        mean_entropy = self.entropy_history[:num_entries].mean(dim=0)  # [num_layers]

        # Expected entropy for uniform distribution
        uniform_probs = torch.ones(self.num_blocks + 1) / (self.num_blocks + 1)
        uniform_entropy = self.compute_entropy(uniform_probs)

        # Specialization ratio (lower is more specialized)
        spec_ratio = mean_entropy / uniform_entropy

        return {
            "mean_entropy": mean_entropy.mean().item(),
            "entropy_by_layer": mean_entropy.tolist(),
            "uniform_entropy": uniform_entropy,
            "specialization_ratio": spec_ratio.mean().item(),
            "is_specialized": spec_ratio.mean().item() < 0.8,  # Threshold
        }

    def visualize_attention_patterns(self):
        """
        Generate heatmap of attention patterns across layers and blocks.

        Returns:
            Numpy array [num_layers, num_blocks + 1] of average attention weights
            or torch.Tensor if numpy is not available
        """
        if self.history_ptr == 0:
            if np is not None:
                return np.zeros((self.num_layers, self.num_blocks + 1))
            return torch.zeros((self.num_layers, self.num_blocks + 1))

        num_entries = min(self.history_ptr, 100)
        avg_weights = self.attention_history[:num_entries].mean(
            dim=0
        )  # [num_layers, num_blocks + 1]

        if np is not None:
            return avg_weights.numpy()
        return avg_weights


class PseudoQueryInitializer:
    """
    Handles initialization strategies for pseudo-queries.

    Critical design choice: Zero initialization ensures uniform attention
    at training start, recovering standard residual behavior.
    """

    @staticmethod
    def zero_init(module: nn.Module):
        """Zero initialization (default, recommended)."""
        for name, param in module.named_parameters():
            if "pseudo_query" in name or "pseudo_queries" in name:
                nn.init.zeros_(param)
                print(f"Zero-initialized: {name}")

    @staticmethod
    def uniform_init(module: nn.Module, std: float = 0.02):
        """Random uniform initialization (for ablation)."""
        for name, param in module.named_parameters():
            if "pseudo_query" in name or "pseudo_queries" in name:
                nn.init.normal_(param, mean=0, std=std)
                print(f"Random-initialized: {name}")

    @staticmethod
    def identity_like_init(module: nn.Module):
        """
        Initialize to favor recent blocks (approximates standard residual).

        This creates an inductive bias towards recent information,
        which may help in some cases but removes the flexibility
        of learning from scratch.
        """
        for name, param in module.named_parameters():
            if "pseudo_query" in name or "pseudo_queries" in name:
                # Initialize with small positive bias for recent blocks
                # This is an experimental initialization
                with torch.no_grad():
                    param.normal_(mean=0, std=0.01)
                    if len(param.shape) >= 2:
                        # Favor later positions slightly
                        param[:, -1] += 0.1
                print(f"Identity-like-initialized: {name}")


class AttentionWeightMonitor:
    """
    Hooks to monitor attention weight distributions during forward pass.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture attention weights."""

        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(module, "last_attention_weights"):
                    self.attention_weights[name] = module.last_attention_weights

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, PseudoQueryManager):
                h = module.register_forward_hook(hook_fn(name))
                self.hooks.append(h)

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get collected attention statistics."""
        return self.attention_weights.copy()
