"""RaBitQ 1-bit codec for KV-cache quantization (ADN space dimension, ``sec:adn-rabitq``).

Encodes a vector ``x ∈ R^d`` as a triple ``(signs ∈ {-1, +1}^d, norm)`` after a
shared orthonormal rotation ``Q``:

    y     = x @ Q                       # rotate into the codec's basis
    signs = sign(y)                     # 1-bit per channel
    norm  = ||y||_2 = ||x||_2           # (Q orthonormal ⇒ norm preserved)

Decoding reconstructs an approximation that preserves the original L2 norm:

    y_hat = signs * (norm / sqrt(d))
    x_hat = y_hat @ Q^T

Sign bits may be stored **packed**: ``ceil(d/8)`` bytes per vector (LSB-first
within each byte for consecutive channel indices). Unpacked ``int8`` ``±1``
vectors remain supported for debugging.

This is a minimal reference implementation sufficient for validating the
"1-bit KV cache works end-to-end" claim in the QASP pipeline. Production
systems would replace this with a fused kernel; the mathematical contract is
the same.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def packed_sign_dim(feature_dim: int) -> int:
    """Number of bytes needed to store ``feature_dim`` sign bits (8 bits per byte)."""

    if feature_dim < 1:
        raise ValueError("`feature_dim` must be >= 1.")
    return (int(feature_dim) + 7) // 8


def pack_sign_bits_pm1(signs_pm1: Tensor) -> Tensor:
    """Pack a ``±1`` sign tensor along its last dimension into ``uint8`` bytes.

    Channel ``i`` maps to byte ``i // 8``, bit ``i % 8`` (LSB = smallest ``i`` in
    that byte). The final byte is zero-padded when ``d`` is not a multiple of 8.
    """

    if signs_pm1.shape[-1] < 1:
        raise ValueError("last dimension must be >= 1.")
    d = signs_pm1.shape[-1]
    pack_len = packed_sign_dim(d)
    pos = (signs_pm1 >= 0).to(torch.uint8)
    remainder = d % 8
    if remainder != 0:
        pos = F.pad(pos, (0, 8 - remainder), value=0)
    blocks = pos.reshape(*pos.shape[:-1], pack_len, 8)
    shifts = torch.arange(8, device=signs_pm1.device, dtype=torch.uint8)
    packed = (blocks << shifts).sum(dim=-1).to(torch.uint8)
    return packed


def unpack_sign_bits_pm1(packed: Tensor, feature_dim: int) -> Tensor:
    """Unpack ``uint8`` sign bytes to ``int8`` ``±1`` with ``feature_dim`` channels."""

    d = int(feature_dim)
    if d < 1:
        raise ValueError("`feature_dim` must be >= 1.")
    if packed.dtype != torch.uint8:
        raise ValueError("`packed` must have dtype torch.uint8.")
    expected = packed_sign_dim(d)
    if packed.shape[-1] != expected:
        raise ValueError(
            f"last dim of `packed` must be {expected} for feature_dim={d}, got {packed.shape[-1]}."
        )
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32).view(1, 1, 8)
    bits = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 1
    flat = bits.reshape(*packed.shape[:-1], -1)[..., :d]
    return torch.where(flat > 0, torch.ones_like(flat, dtype=torch.int8), -torch.ones_like(flat, dtype=torch.int8))


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

    @property
    def packed_last_dim(self) -> int:
        """Length of the packed sign axis (bytes) for this codec."""

        return packed_sign_dim(self.dim)

    def encode(self, x: Tensor, *, packed: bool = True) -> tuple[Tensor, Tensor]:
        """Encode ``x`` (shape ``[..., d]``) to signs and per-vector L2 norms of ``y = xQ``.

        Args:
            packed: If ``True`` (default), return signs as ``uint8`` with last dim
                ``ceil(d/8)``. If ``False``, return ``int8`` ``±1`` per channel
                (legacy layout).
        """

        if x.shape[-1] != self.dim:
            raise ValueError("last dim of `x` must equal codec.dim")

        rotated = x @ self.rotation
        norms = rotated.norm(dim=-1)
        signs = torch.where(rotated >= 0, torch.ones_like(rotated), -torch.ones_like(rotated))
        signs_i8 = signs.to(torch.int8)
        if not packed:
            return signs_i8, norms
        return pack_sign_bits_pm1(signs_i8), norms

    def decode(self, signs: Tensor, norms: Tensor) -> Tensor:
        """Reconstruct an approximation of ``x`` from ``(signs, norms)``.

        ``signs`` may be ``uint8`` packed bytes (last dim ``ceil(d/8)``) or ``int8``
        ``±1`` values (last dim ``d``). :meth:`encode` uses packed ``uint8`` by
        default; distinguish layouts by dtype.
        """

        if signs.dtype == torch.uint8:
            if signs.shape[-1] != self.packed_last_dim:
                raise ValueError(
                    "last dim of packed `signs` must equal codec.packed_last_dim "
                    f"({self.packed_last_dim}), got {signs.shape[-1]}."
                )
            signs_i8 = unpack_sign_bits_pm1(signs, self.dim)
        elif signs.dtype == torch.int8:
            if signs.shape[-1] != self.dim:
                raise ValueError("last dim of `signs` must equal codec.dim")
            signs_i8 = signs
        else:
            raise ValueError("`signs` must be torch.uint8 (packed) or torch.int8 (unpacked).")

        if signs_i8.shape[:-1] != norms.shape:
            raise ValueError("`norms` must match all-but-last dims of `signs`.")

        scale = (norms / math.sqrt(float(self.dim))).unsqueeze(-1)
        rotated_approx = signs_i8.to(scale.dtype) * scale
        return rotated_approx @ self.rotation.transpose(-2, -1)

    def quantize(self, x: Tensor) -> Tensor:
        """Convenience: ``decode(encode(x))`` — returns same-shape approximation."""

        signs, norms = self.encode(x)
        return self.decode(signs, norms)
