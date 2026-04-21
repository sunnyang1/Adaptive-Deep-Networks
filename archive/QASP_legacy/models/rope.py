"""Rotary Position Embedding (RoPE) for QASP.

Implements LLaMA-style RoPE with configurable theta base.
Reference: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape [B, H, T, d_h]
        k: Key tensor of shape [B, H_kv, T, d_h]
        cos: Cosine precomputations of shape [1, 1, T, d_h]
        sin: Sine precomputations of shape [1, 1, T, d_h]

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    """

    # Split each head dim into pairs
    def rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(torch.nn.Module):
    """Precompute and cache RoPE frequency tensors."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cached_seq_len: int = 0
        self._cached_cos: Tensor | None = None
        self._cached_sin: Tensor | None = None

    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        if seq_len > self._cached_seq_len or self._cached_cos is None or self._cached_cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
            emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
            self._cached_cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
            self._cached_sin = emb.sin()[None, None, :, :]  # [1, 1, seq_len, dim]
            self._cached_seq_len = seq_len
        return self._cached_cos, self._cached_sin

    def forward(self, x: Tensor, seq_len: int | None = None) -> tuple[Tensor, Tensor]:
        """Return cos and sin tensors for the given sequence length.

        Args:
            x: Reference tensor to infer device/dtype from. Shape irrelevant.
            seq_len: Sequence length. If None, inferred from x's time dim.

        Returns:
            (cos, sin) each of shape [1, 1, seq_len, dim]
        """

        if seq_len is None:
            seq_len = x.shape[-2]
        cos, sin = self._compute_cos_sin(seq_len, x.device)
        return cos.to(x.dtype), sin.to(x.dtype)
