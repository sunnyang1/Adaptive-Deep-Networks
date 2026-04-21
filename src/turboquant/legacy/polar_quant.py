"""
PolarQuant: (b-1)-bit Quantization via Polar Coordinates

Stage 1 of TurboQuant pipeline:
1. Random Hadamard Transform (RHT) - spreads energy uniformly
2. Cartesian-to-Polar conversion
3. Lloyd-Max optimal quantization on angles

Eliminates per-block normalization overhead through geometric insight.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Union, List


class HadamardTransform:
    """Random Hadamard Transform (RHT) for energy spreading."""

    def __init__(self, dim: int, device: Union[str, torch.device] = "cpu"):
        """
        Args:
            dim: Dimension (must be power of 2)
            device: torch device
        """
        assert (dim & (dim - 1)) == 0, "Dimension must be power of 2"
        self.dim = dim
        self.device = device

        # Random diagonal sign matrix D
        self.D = torch.randn(dim, device=device).sign()

        # Hadamard matrix H (recursive construction)
        self.H = self._build_hadamard(dim)

    def _build_hadamard(self, n: int) -> torch.Tensor:
        """Build Hadamard matrix of size n×n."""
        if n == 1:
            return torch.ones(1, 1, device=self.device)

        H_half = self._build_hadamard(n // 2)
        H = torch.zeros(n, n, device=self.device)
        H[: n // 2, : n // 2] = H_half
        H[: n // 2, n // 2 :] = H_half
        H[n // 2 :, : n // 2] = H_half
        H[n // 2 :, n // 2 :] = -H_half

        # Normalize
        return H / math.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RHT: x' = (H @ D @ x^T)^T = x @ D @ H^T

        Args:
            x: Input tensor [..., dim]

        Returns:
            Transformed tensor [..., dim]
        """
        # Apply D (element-wise)
        x = x * self.D

        # Apply H (matrix multiply)
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        x_transformed = x_flat @ self.H.T

        return x_transformed.reshape(original_shape)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse RHT: x = (D^-1 @ H^-1 @ x'^T)^T = x' @ H @ D

        For Hadamard: H^-1 = H^T, so H = (H^T)^-1
        For D: D^-1 = D (since D is diagonal with ±1)
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)

        # Apply H^T inverse = H (since H is orthogonal: H @ H^T = I)
        x_inv = x_flat @ self.H

        # Apply D^-1 = D
        x_inv = x_inv * self.D

        return x_inv.reshape(original_shape)


class CartesianToPolar:
    """
    Convert Cartesian coordinates to polar coordinates.

    For d-dimensional vector:
    - r: magnitude (radius)
    - θ₁, θ₂, ..., θ_{d-1}: angles
    """

    @staticmethod
    def forward(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cartesian to polar conversion.

        Args:
            x: Input tensor [..., d]
            eps: Small constant for numerical stability

        Returns:
            r: Magnitude [..., 1]
            theta: Angles [..., d-1] (each in [0, π] or [0, 2π])
        """
        # Compute radius
        r = torch.norm(x, dim=-1, keepdim=True) + eps

        # Compute angles sequentially
        d = x.shape[-1]
        theta_list = []

        for i in range(d - 1):
            # Project onto remaining dimensions
            remaining_norm = torch.norm(x[..., i:], dim=-1, keepdim=True) + eps

            # Angle from i-th axis
            cos_theta = x[..., i : i + 1] / remaining_norm
            cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
            theta_i = torch.acos(cos_theta)

            theta_list.append(theta_i)

        theta = torch.cat(theta_list, dim=-1)  # [..., d-1]

        return r, theta

    @staticmethod
    def inverse(r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Polar to Cartesian conversion.

        Args:
            r: Magnitude [..., 1]
            theta: Angles [..., d-1]

        Returns:
            x: Cartesian coordinates [..., d]
        """
        d = theta.shape[-1] + 1
        x_list = []

        remaining_r = r.squeeze(-1)  # [...]

        for i in range(d - 1):
            # x_i = r_remaining * cos(θ_i)
            x_i = remaining_r * torch.cos(theta[..., i])
            x_list.append(x_i.unsqueeze(-1))

            # Update remaining radius
            remaining_r = remaining_r * torch.sin(theta[..., i])

        # Last coordinate
        x_list.append(remaining_r.unsqueeze(-1))

        return torch.cat(x_list, dim=-1)


class LloydMaxQuantizer:
    """
        Lloyd-Max optimal quantizer for angle quantization.

        Pre-computes optimal quantization buckets based on known post-rotation
    distribution (Beta distribution concentrated near π/2).
    """

    def __init__(self, num_bits: int, num_samples: int = 10000):
        """
        Args:
            num_bits: Number of bits for quantization
            num_samples: Samples for computing centroids
        """
        self.num_bits = num_bits
        self.num_levels = 2**num_bits

        # Compute optimal centroids for angles in [0, π]
        # Post-RHT angles follow concentrated Beta distribution
        self.centroids = self._compute_centroids(num_samples)
        self.boundaries = self._compute_boundaries()

    def _compute_centroids(self, num_samples: int) -> torch.Tensor:
        """Compute Lloyd-Max centroids for angle distribution."""
        # Approximate Beta distribution concentrated near π/2
        # Use samples from Beta(2, 2) scaled to [0, π]
        beta_dist = torch.distributions.Beta(torch.tensor(2.0), torch.tensor(2.0))
        samples = beta_dist.sample((num_samples,))
        angles = samples * math.pi

        # Lloyd-Max iterations
        centroids = torch.linspace(0, math.pi, self.num_levels + 2)[1:-1]

        for _ in range(20):  # Convergence iterations
            # Assign samples to nearest centroid
            distances = torch.abs(angles.unsqueeze(1) - centroids.unsqueeze(0))
            assignments = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for i in range(self.num_levels):
                mask = assignments == i
                if mask.any():
                    new_centroids[i] = angles[mask].mean()
                else:
                    new_centroids[i] = centroids[i]  # Keep old if empty

            centroids = new_centroids

        return centroids

    def _compute_boundaries(self) -> torch.Tensor:
        """Compute decision boundaries (midpoints between centroids)."""
        boundaries = torch.zeros(self.num_levels + 1)
        boundaries[0] = 0
        boundaries[-1] = math.pi

        for i in range(1, self.num_levels):
            boundaries[i] = (self.centroids[i - 1] + self.centroids[i]) / 2

        return boundaries

    def encode(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Quantize angles to discrete levels.

        Args:
            theta: Angles [..., d-1]

        Returns:
            indices: Quantized indices [..., d-1] (int64)
        """
        # Find which bucket each angle falls into
        theta_clamped = torch.clamp(theta, 0, math.pi - 1e-6)

        # Broadcasting comparison
        boundaries = self.boundaries.to(theta.device)
        indices = torch.searchsorted(boundaries, theta_clamped) - 1
        indices = torch.clamp(indices, 0, self.num_levels - 1)

        return indices.long()

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Dequantize indices back to angles.

        Args:
            indices: Quantized indices [..., d-1]

        Returns:
            theta: Reconstructed angles [..., d-1]
        """
        centroids = self.centroids.to(indices.device)
        return centroids[indices]


class PolarQuant(nn.Module):
    """
    Complete PolarQuant module: RHT + Polar conversion + Lloyd-Max quantization.

    Achieves (b-1)-bit compression by storing:
    - r: Full precision (shared magnitude)
    - θ: (b-1)-bit quantized angles
    """

    def __init__(
        self,
        dim: int,
        angle_bits: int = 3,  # (b-1) bits for angles
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.angle_bits = angle_bits

        # Components
        self.rht = HadamardTransform(dim, device)
        self.cart2pol = CartesianToPolar()
        self.lloyd_max = LloydMaxQuantizer(angle_bits)

    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress tensor using PolarQuant.

        Args:
            x: Input tensor [..., dim]

        Returns:
            r: Magnitude [..., 1] (full precision)
            theta_indices: Quantized angle indices [..., dim-1] (int64)
            x_rht: RHT-transformed tensor (for residual computation)
        """
        # Step 1: Random Hadamard Transform
        x_rht = self.rht.forward(x)

        # Step 2: Cartesian to Polar
        r, theta = self.cart2pol.forward(x_rht)

        # Step 3: Lloyd-Max quantization on angles
        theta_indices = self.lloyd_max.encode(theta)

        return r, theta_indices, x_rht

    def decompress(self, r: torch.Tensor, theta_indices: torch.Tensor) -> torch.Tensor:
        """
        Decompress tensor from PolarQuant representation.

        Args:
            r: Magnitude [..., 1]
            theta_indices: Quantized angle indices [..., dim-1]

        Returns:
            x_reconstructed: Reconstructed tensor [..., dim]
        """
        # Step 1: Dequantize angles
        theta = self.lloyd_max.decode(theta_indices)

        # Step 2: Polar to Cartesian
        x_rht_reconstructed = self.cart2pol.inverse(r, theta)

        # Step 3: Inverse RHT
        x_reconstructed = self.rht.inverse(x_rht_reconstructed)

        return x_reconstructed

    def get_compression_ratio(self) -> float:
        """Compute compression ratio."""
        original_bits = self.dim * 16  # FP16
        compressed_bits = 16 + (self.dim - 1) * self.angle_bits  # r + angles
        return original_bits / compressed_bits


# Convenience functions
def polar_quantize(
    x: torch.Tensor, angle_bits: int = 3, dim: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stateless PolarQuant compression.

    Args:
        x: Input tensor
        angle_bits: Bits for angle quantization
        dim: Dimension to quantize along

    Returns:
        r: Magnitude
        theta_indices: Quantized angle indices
    """
    if dim != -1:
        x = x.transpose(dim, -1)

    d = x.shape[-1]
    pq = PolarQuant(d, angle_bits, x.device)
    r, theta_indices, _ = pq.compress(x)

    if dim != -1:
        r = r.transpose(dim, -1)
        theta_indices = theta_indices.transpose(dim, -1)

    return r, theta_indices


def polar_dequantize(
    r: torch.Tensor, theta_indices: torch.Tensor, angle_bits: int = 3, dim: int = -1
) -> torch.Tensor:
    """Stateless PolarQuant decompression."""
    if dim != -1:
        r = r.transpose(dim, -1)
        theta_indices = theta_indices.transpose(dim, -1)

    d = theta_indices.shape[-1] + 1
    pq = PolarQuant(d, angle_bits, r.device)
    x = pq.decompress(r, theta_indices)

    if dim != -1:
        x = x.transpose(dim, -1)

    return x
