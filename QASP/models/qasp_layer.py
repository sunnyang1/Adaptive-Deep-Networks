"""QASP transformer layer with optional adaptation hooks."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from QASP.models.components import CausalSelfAttention, FeedForward, QASPTransformerConfig, RMSNorm
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

    def forward(
        self,
        hidden_states: Tensor,
        block_representations: Tensor | None = None,
        block_quality: Tensor | None = None,
        memory_vector: Tensor | None = None,
        memory_quality: Tensor | None = None,
    ) -> Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))

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

