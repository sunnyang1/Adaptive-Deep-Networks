"""
Random Rotation using Fast Walsh-Hadamard Transform (FWHT).

The FWHT provides O(n log n) random rotation vs O(n²) for matrix multiplication.
After rotation, coordinates follow a bell-curve distribution enabling optimal
scalar quantization.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform.

    O(n log n) implementation using butterfly pattern.
    H @ x where H is the Hadamard matrix (implicitly defined).

    Args:
        x: Input tensor [..., n], n must be power of 2

    Returns:
        Transformed tensor [..., n]
    """
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"Input dimension must be power of 2, got {n}"

    h = 2
    while h <= n:
        x = x.reshape(*x.shape[:-1], n // h, h)
        x = torch.stack([x[..., ::2] + x[..., 1::2], x[..., ::2] - x[..., 1::2]], dim=-1)
        x = x.reshape(*x.shape[:-3], n)
        h *= 2

    return x


def fwht_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse Fast Walsh-Hadamard Transform.

    For Hadamard matrix H: H @ H = n * I
    So inverse is H / n, or fwht(x) / n

    Args:
        x: Input tensor [..., n], n must be power of 2

    Returns:
        Inverse transformed tensor [..., n]
    """
    n = x.shape[-1]
    return fwht(x) / n


class RandomRotation:
    """
    Random orthogonal rotation using FWHT.

    Combines FWHT with random diagonal scaling to approximate
    random orthogonal rotation in O(n log n) time.

    Usage:
        >>> rotation = RandomRotation(dim=128, seed=42)
        >>> x_rotated = rotation.rotate(x)  # [..., 128]
        >>> x_recovered = rotation.inverse(x_rotated)
    """

    def __init__(self, dim: int, seed: int = 42, device: str = "cpu"):
        """
        Initialize random rotation.

        Args:
            dim: Dimension (will be padded to next power of 2)
            seed: Random seed for reproducibility
            device: 'cpu' or 'cuda' or 'mps'
        """
        self.dim = _next_power_of_2(dim)
        self.original_dim = dim

        # Generate random diagonal scales (Rademacher variables: +1 or -1)
        generator = torch.Generator(device=device).manual_seed(seed)
        self.scales = (
            torch.randint(0, 2, (self.dim,), generator=generator, device=device) * 2 - 1
        ).float()

        self.device = device

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random rotation.

        Args:
            x: Input tensor [..., d] where d <= self.dim

        Returns:
            Rotated tensor [..., self.dim]
        """
        # Pad if needed
        if x.shape[-1] < self.dim:
            padding = self.dim - x.shape[-1]
            x = F.pad(x, (0, padding))

        # Apply diagonal scaling
        x = x * self.scales

        # Apply FWHT
        x = fwht(x)

        # Normalize for orthogonality
        return x / math.sqrt(self.dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse rotation.

        Args:
            x: Rotated tensor [..., self.dim]

        Returns:
            Original tensor [..., original_dim]
        """
        # Normalize
        x = x * math.sqrt(self.dim)

        # Apply inverse FWHT
        x = fwht_inverse(x)

        # Apply inverse diagonal scaling (same as forward since ±1)
        x = x * self.scales

        # Remove padding
        if self.original_dim < self.dim:
            x = x[..., : self.original_dim]

        return x

    def to(self, device: str) -> "RandomRotation":
        """Move rotation to device."""
        self.device = device
        self.scales = self.scales.to(device)
        return self


class IdentityRotation:
    """No-op rotation for testing/baseline."""

    def __init__(self, dim: int, seed: int = 42, device: str = "cpu"):
        self.dim = dim
        self.original_dim = dim
        self.device = device

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, device: str) -> "IdentityRotation":
        self.device = device
        return self
