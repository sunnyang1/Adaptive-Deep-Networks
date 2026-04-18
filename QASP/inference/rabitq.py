"""RaBitQ 1-bit codec for KV-cache quantization (ADN space dimension, ``sec:adn-rabitq``).

Encodes a vector ``x ∈ R^d`` as a triple ``(signs ∈ {-1, +1}^d, norm)`` after a
shared orthonormal rotation ``Q``:

    y     = x @ Q                       # rotate into the codec's basis
    signs = sign(y)                     # 1-bit per channel
    norm  = ||y||_2 = ||x||_2           # (Q orthonormal ⇒ norm preserved)

Decoding reconstructs an approximation that preserves the original L2 norm:

    y_hat = signs * (norm / sqrt(d))
    x_hat = y_hat @ Q^T

This is a minimal reference implementation sufficient for validating the
"1-bit KV cache works end-to-end" claim in the QASP pipeline. Production
systems would replace this with a fused kernel; the mathematical contract is
the same.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RaBitQCodec(nn.Module):
    """Shared 1-bit sign codec with a fixed random orthonormal rotation."""

    rotation: Tensor

    def __init__(self, dim: int, *, seed: int = 0) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("`dim` must be >= 1.")
        self.dim = int(dim)

        generator = torch.Generator().manual_seed(int(seed))
        gaussian = torch.randn(self.dim, self.dim, generator=generator)
        q, _ = torch.linalg.qr(gaussian)
        self.register_buffer("rotation", q, persistent=True)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode ``x`` (shape ``[..., d]``) to ``(signs_int8, norms)``."""

        if x.shape[-1] != self.dim:
            raise ValueError("last dim of `x` must equal codec.dim")

        rotated = x @ self.rotation
        norms = rotated.norm(dim=-1)
        signs = torch.where(rotated >= 0, torch.ones_like(rotated), -torch.ones_like(rotated))
        return signs.to(torch.int8), norms

    def decode(self, signs: Tensor, norms: Tensor) -> Tensor:
        """Reconstruct an approximation of ``x`` from ``(signs, norms)``."""

        if signs.shape[-1] != self.dim:
            raise ValueError("last dim of `signs` must equal codec.dim")
        if signs.shape[:-1] != norms.shape:
            raise ValueError("`norms` must match all-but-last dims of `signs`.")

        scale = (norms / math.sqrt(float(self.dim))).unsqueeze(-1)
        rotated_approx = signs.to(scale.dtype) * scale
        return rotated_approx @ self.rotation.transpose(-2, -1)

    def quantize(self, x: Tensor) -> Tensor:
        """Convenience: ``decode(encode(x))`` — returns same-shape approximation."""

        signs, norms = self.encode(x)
        return self.decode(signs, norms)
