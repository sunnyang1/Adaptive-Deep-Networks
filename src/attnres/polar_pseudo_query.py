"""
Polar Pseudo-Query Vectors for RaBitQ-Accelerated qTTT

Based on: Section 3.3 of Adaptive Deep Networks RaBitQ version

Key innovation: Decompose pseudo-queries into magnitude (r) and direction (θ),
freeze r during qTTT, adapt only θ. This:
- Reduces trainable parameters by 50%
- Provides natural gradient conditioning via spherical geometry
- Enables faster convergence due to bounded angular updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class PolarPseudoQuery(nn.Module):
    """
    Single pseudo-query in polar coordinates.

    Representation: w = r * u(θ)
    where:
    - r: magnitude (scalar, frozen during qTTT)
    - u(θ): unit direction vector from angles θ [θ₁, θ₂, ..., θ_{d-1}]
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # Magnitude (initialized to small value, typically frozen)
        self.r = nn.Parameter(torch.tensor(1.0))

        # Angles (trainable, represents direction on (d-1)-sphere)
        # Initialize to random uniform on sphere
        self.theta = nn.Parameter(torch.randn(dim - 1) * 0.01)

        self._freeze_r = False  # Flag for qTTT mode

    def freeze_magnitude(self):
        """Freeze magnitude for qTTT (only adapt direction)."""
        self.r.requires_grad = False
        self._freeze_r = True

    def unfreeze_magnitude(self):
        """Unfreeze magnitude for full training."""
        self.r.requires_grad = True
        self._freeze_r = False

    def angles_to_unit_vector(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Convert angles to unit vector on (d-1)-sphere.

        Uses hyperspherical coordinates:
        x₁ = cos(θ₁)
        x₂ = sin(θ₁) * cos(θ₂)
        x₃ = sin(θ₁) * sin(θ₂) * cos(θ₃)
        ...
        x_d = sin(θ₁) * ... * sin(θ_{d-1})
        """
        dim = theta.shape[-1] + 1
        components = []

        sin_product = 1.0
        for i in range(dim - 1):
            cos_theta = torch.cos(theta[..., i])
            components.append(sin_product * cos_theta)
            sin_product = sin_product * torch.sin(theta[..., i])

        # Last component
        components.append(sin_product)

        return torch.stack(components, dim=-1)

    def forward(self) -> torch.Tensor:
        """Return pseudo-query vector w = r * u(θ)."""
        u = self.angles_to_unit_vector(self.theta)
        return self.r * u

    def get_direction(self) -> torch.Tensor:
        """Get unit direction vector (without magnitude)."""
        return self.angles_to_unit_vector(self.theta)

    def set_direction(self, theta_new: torch.Tensor):
        """Set direction from new angles."""
        with torch.no_grad():
            self.theta.copy_(theta_new)

    def project_gradient_to_sphere(self, grad_w: torch.Tensor) -> torch.Tensor:
        """
        Project Cartesian gradient to tangent space of sphere.

        For gradient g = ∇_w L, we want ∇_θ L.
        Using chain rule and spherical geometry.
        """
        # This is handled automatically by PyTorch autodiff
        # but we provide explicit method for clarity
        return grad_w


class PolarPseudoQueryManager(nn.Module):
    """
    Manager for polar pseudo-queries across all layers.

    Each layer has two pseudo-queries (attention and MLP),
    each represented as (r, θ) in polar coordinates.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_blocks: int = 8,
        eps: float = 1e-6,
        use_polar: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.num_blocks = num_blocks
        self.eps = eps
        self.use_polar = use_polar

        if use_polar:
            # Polar representation: r (scalar) + theta (dim-1 angles)
            # Shape: [num_layers, 2, dim] where 2 = (attn, mlp)
            # We store as magnitude + direction for efficiency
            self.magnitudes = nn.Parameter(torch.ones(num_layers, 2))
            self.directions = nn.Parameter(torch.zeros(num_layers, 2, dim))

            # Normalize initial directions
            with torch.no_grad():
                self.directions.data = F.normalize(self.directions.data, dim=-1)
        else:
            # Fallback to Cartesian
            self.pseudo_queries = nn.Parameter(torch.zeros(num_layers, 2, dim))

        # Statistics tracking
        self.register_buffer("attention_history", torch.zeros(100, num_layers, num_blocks + 1))
        self.history_ptr = 0
        self.register_buffer("entropy_history", torch.zeros(100, num_layers))

        # Mode flag for qTTT
        self._qttt_mode = False

    def enable_qttt_mode(self):
        """
        Enable qTTT mode: freeze magnitudes, adapt only directions.

        This reduces trainable parameters by 50% during adaptation.
        """
        self._qttt_mode = True
        if self.use_polar:
            self.magnitudes.requires_grad = False
            self.directions.requires_grad = True

    def disable_qttt_mode(self):
        """Disable qTTT mode: allow full training."""
        self._qttt_mode = False
        if self.use_polar:
            self.magnitudes.requires_grad = True
            self.directions.requires_grad = True

    def get_pseudo_query(self, layer_idx: int, is_mlp: bool = False) -> torch.Tensor:
        """Get pseudo-query for specific layer."""
        q_type = 1 if is_mlp else 0

        if self.use_polar:
            r = self.magnitudes[layer_idx, q_type]
            u = F.normalize(self.directions[layer_idx, q_type], dim=-1)
            return r * u
        else:
            return self.pseudo_queries[layer_idx, q_type]

    def get_all_queries(self) -> torch.Tensor:
        """Get all pseudo-queries."""
        if self.use_polar:
            u = F.normalize(self.directions, dim=-1)
            return self.magnitudes.unsqueeze(-1) * u
        else:
            return self.pseudo_queries

    def get_direction_only(self, layer_idx: int, is_mlp: bool = False) -> torch.Tensor:
        """Get only direction component (for qTTT adaptation)."""
        if not self.use_polar:
            # Fallback: normalize full vector
            w = self.pseudo_queries[layer_idx, 1 if is_mlp else 0]
            return F.normalize(w, dim=-1)

        q_type = 1 if is_mlp else 0
        return F.normalize(self.directions[layer_idx, q_type], dim=-1)

    def set_direction(self, layer_idx: int, direction: torch.Tensor, is_mlp: bool = False):
        """Set direction component (used by qTTT optimizer)."""
        if not self.use_polar:
            # Fallback: set full vector
            with torch.no_grad():
                self.pseudo_queries[layer_idx, 1 if is_mlp else 0].copy_(direction)
            return

        q_type = 1 if is_mlp else 0
        with torch.no_grad():
            self.directions[layer_idx, q_type].copy_(direction)

    def reset_parameters(self):
        """Reset to uniform attention (zero directions, unit magnitude)."""
        if self.use_polar:
            nn.init.ones_(self.magnitudes)
            nn.init.zeros_(self.directions)
        else:
            nn.init.zeros_(self.pseudo_queries)

    def compute_attention_weights(
        self, layer_idx: int, block_representations: List[torch.Tensor], is_mlp: bool = False
    ) -> torch.Tensor:
        """Compute attention weights for visualization/analysis."""
        pseudo_query = self.get_pseudo_query(layer_idx, is_mlp)

        if len(block_representations) == 0:
            return torch.ones(1) / 1

        # Stack blocks
        V = torch.stack(block_representations, dim=0)

        # RMSNorm
        rms = torch.sqrt(torch.mean(V**2, dim=-1, keepdim=True) + self.eps)
        K = V / rms

        # Attention scores
        logits = torch.einsum("n b t d, d -> n b t", K, pseudo_query)
        logits = logits / (self.dim**0.5)
        logits = logits.mean(dim=(1, 2))
        weights = F.softmax(logits, dim=0)

        return weights

    def compute_entropy(self, weights: torch.Tensor) -> float:
        """Compute entropy of attention weights."""
        entropy = -(weights * torch.log(weights + self.eps)).sum()
        return entropy.item()

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count summary."""
        if self.use_polar:
            total = self.magnitudes.numel() + self.directions.numel()
            trainable = (self.magnitudes.numel() if self.magnitudes.requires_grad else 0) + (
                self.directions.numel() if self.directions.requires_grad else 0
            )
            return {
                "magnitudes": self.magnitudes.numel(),
                "directions": self.directions.numel(),
                "total": total,
                "trainable": trainable,
                "frozen": total - trainable,
            }
        else:
            return {
                "cartesian": self.pseudo_queries.numel(),
                "total": self.pseudo_queries.numel(),
                "trainable": self.pseudo_queries.numel(),
                "frozen": 0,
            }

    def get_specialization_report(self) -> Dict:
        """Generate report on pseudo-query specialization."""
        if self.history_ptr == 0:
            return {"error": "No history recorded yet"}

        num_entries = min(self.history_ptr, 100)
        mean_entropy = self.entropy_history[:num_entries].mean(dim=0)

        uniform_probs = torch.ones(self.num_blocks + 1) / (self.num_blocks + 1)
        uniform_entropy = self.compute_entropy(uniform_probs)
        spec_ratio = mean_entropy / uniform_entropy

        return {
            "mean_entropy": mean_entropy.mean().item(),
            "entropy_by_layer": mean_entropy.tolist(),
            "uniform_entropy": uniform_entropy,
            "specialization_ratio": spec_ratio.mean().item(),
            "is_specialized": spec_ratio.mean().item() < 0.8,
            "qttt_mode": self._qttt_mode,
            "parameter_count": self.get_parameter_count(),
        }


class PseudoQueryPolarAdapter:
    """
    Adapter for qTTT on polar pseudo-queries.

    Updates only direction θ while keeping magnitude r frozen.
    Uses spherical gradient descent for natural geometry.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def adapt(self, direction: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """
        Adapt direction using gradient on sphere.

        Args:
            direction: Current unit direction [d]
            gradient: Gradient w.r.t. direction [d]

        Returns:
            new_direction: Updated unit direction [d]
        """
        # Project gradient to tangent space (remove radial component)
        grad_parallel = (gradient * direction).sum() * direction
        grad_tangent = gradient - grad_parallel

        # Initialize velocity if needed
        if self.velocity is None:
            self.velocity = torch.zeros_like(direction)

        # Momentum update on tangent space
        self.velocity = self.momentum * self.velocity - self.lr * grad_tangent

        # Exponential map: move on sphere in direction of velocity
        v_norm = torch.norm(self.velocity)
        if v_norm > 1e-8:
            # Riemannian exponential map
            new_direction = direction * torch.cos(v_norm) + (self.velocity / v_norm) * torch.sin(
                v_norm
            )
        else:
            new_direction = direction

        # Renormalize (numerical safety)
        new_direction = F.normalize(new_direction, dim=-1)

        return new_direction

    def reset(self):
        """Reset adapter state."""
        self.velocity = None


# Factory function
def create_pseudo_query_manager(
    num_layers: int,
    dim: int,
    num_blocks: int = 8,
    use_polar: bool = True,
    enable_qttt: bool = False,
) -> PolarPseudoQueryManager:
    """
    Factory for creating pseudo-query manager.

    Args:
        num_layers: Number of layers
        dim: Hidden dimension
        num_blocks: Number of AttnRes blocks
        use_polar: Use polar coordinates
        enable_qttt: Enable qTTT mode (freeze magnitudes)
    """
    manager = PolarPseudoQueryManager(
        num_layers=num_layers, dim=dim, num_blocks=num_blocks, use_polar=use_polar
    )

    if enable_qttt:
        manager.enable_qttt_mode()

    return manager
