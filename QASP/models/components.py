"""Core components shared by QASP model modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from QASP.adaptation.quality_score import compute_quality_score


class _KVCodec(Protocol):
    """Structural type for KV-cache codecs (see :class:`QASP.inference.rabitq.RaBitQCodec`)."""

    def quantize(self, x: Tensor) -> Tensor: ...


@dataclass
class QASPTransformerConfig:
    """Configuration for the lightweight QASP transformer stack."""

    vocab_size: int = 32000
    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_ratio: float = 4.0
    max_position_embeddings: int = 2048
    attnres_blocks: int = 4
    use_attnres: bool = True
    use_engram: bool = True
    engram_table_size: int = 4096
    engram_n_gram: int = 3
    adapt_rank: int = 32
    stiefel_overlay_scale: float = 0.0
    quantize_kv: bool = False
    kv_codec_seed: int = 0


class RMSNorm(nn.Module):
    """Simple RMSNorm used for stable transformer blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class CausalSelfAttention(nn.Module):
    """Minimal multi-head causal self-attention."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.overlay_scale = float(config.stiefel_overlay_scale)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def _apply_stiefel_overlay(
        self,
        hidden_states: Tensor,
        stiefel_query: Tensor | None,
    ) -> Tensor:
        """Apply ``h + scale · h W W^T`` where ``W ∈ St(k, d)`` (Stiefel query path; see ``sec:qasp-matrix`` in ``QASP_paper.tex``)."""

        if stiefel_query is None or self.overlay_scale == 0.0:
            return hidden_states
        if stiefel_query.ndim != 2 or stiefel_query.shape[0] != self.hidden_size:
            raise ValueError(
                "stiefel_query must have shape [hidden_size, k] to form a Stiefel overlay."
            )
        projected = hidden_states @ stiefel_query
        overlay = projected @ stiefel_query.transpose(-2, -1)
        return hidden_states + self.overlay_scale * overlay

    def _shape(self, tensor: Tensor) -> Tensor:
        batch, seq, _ = tensor.shape
        tensor = tensor.view(batch, seq, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        stiefel_query: Tensor | None = None,
    ) -> Tensor:
        q_input = self._apply_stiefel_overlay(hidden_states, stiefel_query)
        q = self._shape(self.q_proj(q_input))
        k = self._shape(self.k_proj(hidden_states))
        v = self._shape(self.v_proj(hidden_states))

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(hidden_states.size(0), hidden_states.size(1), self.hidden_size)
        return cast(Tensor, self.o_proj(attn_output))

    def forward_with_cache(
        self,
        hidden_states: Tensor,
        codec: _KVCodec | None = None,
        stiefel_query: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Same as :meth:`forward` but also returns full K/V for the KV cache.

        When ``codec`` is provided, ``K`` and ``V`` are passed through
        ``codec.quantize`` before attention so the attention computation
        matches what subsequent :meth:`step` calls will observe. When
        ``stiefel_query`` is provided, the Stiefel overlay (``sec:qasp-matrix``) is applied to
        the query side before ``q_proj``.
        """

        q_input = self._apply_stiefel_overlay(hidden_states, stiefel_query)
        q = self._shape(self.q_proj(q_input))
        k = self._shape(self.k_proj(hidden_states))
        v = self._shape(self.v_proj(hidden_states))

        if codec is not None:
            k = codec.quantize(k)
            v = codec.quantize(v)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(hidden_states.size(0), hidden_states.size(1), self.hidden_size)
        return cast(Tensor, self.o_proj(attn_output)), k, v

    def step(
        self,
        hidden_new: Tensor,
        cached_k: Tensor | None,
        cached_v: Tensor | None,
        codec: _KVCodec | None = None,
        stiefel_query: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run a single-token incremental attention update with KV caching.

        ``hidden_new`` has shape ``[B, 1, D]`` (the embedding of the newly
        generated token). ``cached_k`` and ``cached_v`` are the previously
        accumulated keys / values, each of shape ``[B, H, T_prev, d_h]`` or
        ``None`` for the very first step.

        Returns ``(output, new_k, new_v)``, where ``new_k`` / ``new_v`` are the
        updated per-head caches of shape ``[B, H, T_prev + 1, d_h]``.
        """

        if hidden_new.ndim != 3 or hidden_new.size(1) != 1:
            raise ValueError("`hidden_new` must have shape [B, 1, D].")

        q_input = self._apply_stiefel_overlay(hidden_new, stiefel_query)
        q = self._shape(self.q_proj(q_input))
        k_new = self._shape(self.k_proj(hidden_new))
        v_new = self._shape(self.v_proj(hidden_new))

        if codec is not None:
            k_new = codec.quantize(k_new)
            v_new = codec.quantize(v_new)

        if cached_k is None or cached_v is None:
            if cached_k is not None or cached_v is not None:
                raise ValueError("`cached_k` and `cached_v` must both be None or both be Tensors.")
            full_k = k_new
            full_v = v_new
        else:
            full_k = torch.cat([cached_k, k_new], dim=2)
            full_v = torch.cat([cached_v, v_new], dim=2)

        attn_output = F.scaled_dot_product_attention(
            q,
            full_k,
            full_v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(hidden_new.size(0), 1, self.hidden_size)
        return cast(Tensor, self.o_proj(attn_output)), full_k, full_v


class FeedForward(nn.Module):
    """SwiGLU-style MLP block."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        inner_dim = int(config.hidden_size * config.mlp_ratio)
        self.gate_proj = nn.Linear(config.hidden_size, inner_dim)
        self.up_proj = nn.Linear(config.hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(hidden_states))
        values = self.up_proj(hidden_states)
        return cast(Tensor, self.down_proj(gated * values))


def compute_block_representations(
    hidden_states: Tensor,
    num_blocks: int,
    low_pass_ratio: float = 0.25,
) -> tuple[Tensor, Tensor]:
    """Pool hidden states into block summaries and block-level quality.

    Implements label ``eq:block-quality`` in ``QASP_paper.tex`` (block-level mean
    quality ``ρ̄_m = (1/|B_m|) Σ_{t∈B_m} ρ(t)``), with ``ρ(t)`` from the spectral
    quality score (``eq:quality-score``, Sec.~3.2).

    **Canonical use.**  Pass the **entire** sequence tensor ``[B, T, D]`` from
    one forward pass so that ``B_m`` and ``ρ̄_m`` match the paper's
    full-context definition.  For prefix-only histories (e.g. incremental
    ``step``), statistics differ from that definition; the manuscript does not
    claim bit-identical equivalence between the two.
    """

    chunks = torch.chunk(hidden_states, chunks=max(1, num_blocks), dim=1)
    block_vectors = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)

    per_token_quality = compute_quality_score(hidden_states, low_pass_ratio=low_pass_ratio)
    quality_chunks = torch.chunk(per_token_quality, chunks=max(1, num_blocks), dim=1)
    block_quality = torch.stack([chunk.mean(dim=1) for chunk in quality_chunks], dim=1)
    return block_vectors, block_quality

