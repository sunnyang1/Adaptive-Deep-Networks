"""Minimal incremental prefill/step API for QASP models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from QASP.inference.kv_cache import KVCache


@dataclass
class IncrementalState:
    """Mutable inference state carried across decoding steps."""

    cache: KVCache
    next_logits: Tensor

    @property
    def seq_len(self) -> int:
        return self.cache.seq_len


class IncrementalInference:
    """Lightweight incremental wrapper around a logits-producing QASP model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def prefill(self, input_ids: Tensor) -> IncrementalState:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        logits = self.model(input_ids)
        return IncrementalState(
            cache=KVCache.from_input_ids(input_ids),
            next_logits=logits[:, -1, :],
        )

    @torch.no_grad()
    def step(self, state: IncrementalState) -> Tensor:
        """Generate one token from cached state and update cache in-place."""
        if state.next_logits.ndim != 2:
            raise ValueError("state.next_logits must have shape [B, vocab_size]")

        next_token = torch.argmax(state.next_logits, dim=-1, keepdim=True)
        state.cache.append(next_token)
        logits = self.model(state.cache.input_ids)
        state.next_logits = logits[:, -1, :]
        return next_token
