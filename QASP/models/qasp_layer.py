"""QASP transformer layer with optional adaptation hooks."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from QASP.adaptation.stiefel import project_to_stiefel
from QASP.models.components import (
    CausalSelfAttention,
    FeedForward,
    QASPTransformerConfig,
    RMSNorm,
    _KVCodec,
)
from QASP.models.value_weighted_attnres import ValueWeightedAttnRes
from QASP.models.value_weighted_engram import ValueWeightedEngram


class QASPLayer(nn.Module):
    """Single QASP layer with self-attention, MLP, and optional hooks."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        self.use_attnres = config.use_attnres
        self.use_engram = config.use_engram

        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = FeedForward(config)

        self.attnres = ValueWeightedAttnRes(config.hidden_size) if self.use_attnres else None
        self.engram = ValueWeightedEngram(config.hidden_size) if self.use_engram else None

        if config.use_stiefel_query:
            rank = config.hidden_size
            init_iters = 15  # square matrices need more Newton-Schulz iterations
        else:
            rank = max(1, min(config.adapt_rank, config.hidden_size))
            init_iters = 5
        with torch.no_grad():
            init_matrix = project_to_stiefel(torch.randn(config.hidden_size, rank), num_iters=init_iters)
        self.stiefel_query = nn.Parameter(init_matrix, requires_grad=False)

    def forward(
        self,
        hidden_states: Tensor,
        block_representations: Tensor | None = None,
        block_quality: Tensor | None = None,
        memory_vector: Tensor | None = None,
        memory_quality: Tensor | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states),
            stiefel_query=self.stiefel_query,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        if self.attnres is not None and block_representations is not None and block_quality is not None:
            hidden_states = hidden_states + self.attnres(
                hidden_states,
                block_representations,
                block_quality,
            )

        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))

        if self.engram is not None and memory_vector is not None and memory_quality is not None:
            hidden_states = self.engram(hidden_states, memory_vector, memory_quality)

        return hidden_states

    def forward_with_cache(
        self,
        hidden_states: Tensor,
        block_representations: Tensor | None = None,
        block_quality: Tensor | None = None,
        memory_vector: Tensor | None = None,
        memory_quality: Tensor | None = None,
        kv_codec: _KVCodec | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full-sequence forward that also emits per-layer K/V for cache prefill."""

        attn_out, k, v = self.attn.forward_with_cache(
            self.attn_norm(hidden_states),
            codec=kv_codec,
            stiefel_query=self.stiefel_query,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        hidden_states = hidden_states + attn_out

        if self.attnres is not None and block_representations is not None and block_quality is not None:
            hidden_states = hidden_states + self.attnres(
                hidden_states,
                block_representations,
                block_quality,
            )

        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))

        if self.engram is not None and memory_vector is not None and memory_quality is not None:
            hidden_states = self.engram(hidden_states, memory_vector, memory_quality)

        return hidden_states, k, v

    def step(
        self,
        hidden_new: Tensor,
        cached_k: Tensor | None,
        cached_v: Tensor | None,
        block_representations: Tensor | None = None,
        block_quality: Tensor | None = None,
        memory_vector: Tensor | None = None,
        memory_quality: Tensor | None = None,
        kv_codec: _KVCodec | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Incremental single-token forward that reuses cached K/V.

        ``hidden_new`` has shape ``[B, 1, D]``. ``block_representations`` /
        ``block_quality`` should already reflect the full layer-input history
        (including the new token); the AttnRes residual is broadcast to length
        1 to match ``hidden_new``. ``memory_vector`` / ``memory_quality`` are
        expected to be the lookup results for the new token only.
        """

        attn_out, new_k, new_v = self.attn.step(
            self.attn_norm(hidden_new),
            cached_k=cached_k,
            cached_v=cached_v,
            codec=kv_codec,
            stiefel_query=self.stiefel_query,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        hidden_new = hidden_new + attn_out

        if self.attnres is not None and block_representations is not None and block_quality is not None:
            hidden_new = hidden_new + self.attnres(
                hidden_new,
                block_representations,
                block_quality,
            )

        hidden_new = hidden_new + self.mlp(self.mlp_norm(hidden_new))

        if self.engram is not None and memory_vector is not None and memory_quality is not None:
            hidden_new = self.engram(hidden_new, memory_vector, memory_quality)

        return hidden_new, new_k, new_v

