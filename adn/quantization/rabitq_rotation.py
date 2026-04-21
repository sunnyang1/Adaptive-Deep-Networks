"""
Random Orthogonal Rotations for RaBitQ.

Supports two rotation strategies:
1. MatrixRotator: QR-based random orthogonal matrix (O(n^2) apply, exact)
2. FhtKacRotator: Fast Hadamard Transform + Kac Walk (O(n log n), efficient)

Original RaBitQ requires dimensions to be padded to multiples of 64 for efficient
popcount/SIMD operations.
"""

import math
import torch
import torch.nn.functional as F
from typing import Protocol


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform. O(n log n).
    Args:
        x: [..., n], n must be power of 2
    Returns:
        H @ x
    """
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"FWHT requires power-of-2 dim, got {n}"
    h = 2
    while h <= n:
        x = x.reshape(*x.shape[:-1], n // h, h)
        x = torch.stack([x[..., ::2] + x[..., 1::2], x[..., ::2] - x[..., 1::2]], dim=-1)
        x = x.reshape(*x.shape[:-3], n)
        h *= 2
    return x


def fwht_inverse(x: torch.Tensor) -> torch.Tensor:
    """Inverse FWHT: H @ H = n * I."""
    n = x.shape[-1]
    return fwht(x) / n


class Rotator(Protocol):
    def rotate(self, x: torch.Tensor) -> torch.Tensor: ...
    def inverse_rotate(self, x: torch.Tensor) -> torch.Tensor: ...
    def padded_dim(self) -> int: ...
    def original_dim(self) -> int: ...


class MatrixRotator:
    """
    QR-based random orthogonal matrix.
    Generates a random Gaussian matrix and applies Gram-Schmidt (via QR).
    """

    def __init__(self, dim: int, seed: int = 42, device: str = "cpu"):
        self._dim = dim
        self._padded_dim = dim  # No padding required for matrix, but we may pad
        # QR is not supported on all devices (e.g., MPS), do it on CPU
        generator = torch.Generator(device="cpu").manual_seed(seed)
        A = torch.randn(dim, dim, generator=generator, device="cpu")
        Q, _ = torch.linalg.qr(A)
        self._matrix = Q.to(device)  # [dim, dim]
        self._device = device

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self._dim
        return x @ self._matrix.T

    def inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self._dim
        return x @ self._matrix

    def padded_dim(self) -> int:
        return self._padded_dim

    def original_dim(self) -> int:
        return self._dim

    def to(self, device: str) -> "MatrixRotator":
        self._device = device
        self._matrix = self._matrix.to(device)
        return self


class FhtKacRotator:
    """
    Fast Hadamard Transform + random diagonal (Kac-style) rotator.
    Approximates random orthogonal rotation in O(n log n).
    Pads dimension to the next multiple of 64 (for RaBitQ SIMD alignment).
    """

    def __init__(self, dim: int, seed: int = 42, device: str = "cpu"):
        self._orig_dim = dim
        # Pad to power of 2 for FWHT, and at least to multiple of 64
        self._padded_dim = _round_up_to_multiple(_next_power_of_2(dim), 64)
        generator = torch.Generator(device=device).manual_seed(seed)
        self._scales = (
            torch.randint(0, 2, (self._padded_dim,), generator=generator, device=device) * 2 - 1
        ).float()
        self._device = device

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self._padded_dim:
            x = F.pad(x, (0, self._padded_dim - x.shape[-1]))
        x = x * self._scales
        x = fwht(x)
        return x / math.sqrt(self._padded_dim)

    def inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self._padded_dim)
        x = fwht_inverse(x)
        x = x * self._scales
        if self._orig_dim < self._padded_dim:
            x = x[..., : self._orig_dim]
        return x

    def padded_dim(self) -> int:
        return self._padded_dim

    def original_dim(self) -> int:
        return self._orig_dim

    def to(self, device: str) -> "FhtKacRotator":
        self._device = device
        self._scales = self._scales.to(device)
        return self


class IdentityRotator:
    """No-op rotation for debugging/baseline."""

    def __init__(self, dim: int, device: str = "cpu"):
        self._dim = dim
        self._padded_dim = dim
        self._device = device

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def padded_dim(self) -> int:
        return self._padded_dim

    def original_dim(self) -> int:
        return self._dim

    def to(self, device: str) -> "IdentityRotator":
        self._device = device
        return self
