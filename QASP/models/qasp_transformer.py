"""Top-level QASP transformer model and factory helper."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from QASP.models.components import QASPTransformerConfig, RMSNorm, compute_block_representations
from QASP.models.qasp_layer import QASPLayer


class QASPTransformer(nn.Module):
    """Minimal runnable transformer with optional QASP adaptation hooks."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([QASPLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError("input sequence length exceeds max_position_embeddings")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        for layer in self.layers:
            block_repr = block_quality = None
            memory_vector = memory_quality = None

            if self.config.use_attnres:
                block_repr, block_quality = compute_block_representations(
                    hidden_states,
                    num_blocks=self.config.attnres_blocks,
                )

            if self.config.use_engram:
                memory_vector = hidden_states.mean(dim=1)
                memory_quality = memory_vector.norm(dim=-1)

            hidden_states = layer(
                hidden_states,
                block_representations=block_repr,
                block_quality=block_quality,
                memory_vector=memory_vector,
                memory_quality=memory_quality,
            )

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


def create_qasp_transformer(**kwargs: int | float | bool) -> QASPTransformer:
    """Factory helper for quick model creation in tests and experiments."""

    config = QASPTransformerConfig(**kwargs)
    return QASPTransformer(config)

