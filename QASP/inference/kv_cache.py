"""Per-layer KV cache for O(L) incremental QASP decoding.

Each layer stores its self-attention ``K`` and ``V`` tensors in the shape
``[B, H, T, d_h]`` alongside the hidden-state input it saw (``[B, T, D]``).
The hidden-state history is needed so AttnRes can recompute block
representations without replaying the entire prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor


@dataclass
class KVCache:
    """Mutable cache of generated tokens and per-layer attention state."""

    input_ids: Tensor
    layer_keys: List[Optional[Tensor]] = field(default_factory=list)
    layer_values: List[Optional[Tensor]] = field(default_factory=list)
    layer_inputs: List[Optional[Tensor]] = field(default_factory=list)
    per_token_quality: Optional[Tensor] = None
    last_logits: Optional[Tensor] = None

    @classmethod
    def from_input_ids(cls, input_ids: Tensor, num_layers: int = 0) -> "KVCache":
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        return cls(
            input_ids=input_ids.clone(),
            layer_keys=[None] * num_layers,
            layer_values=[None] * num_layers,
            layer_inputs=[None] * num_layers,
            per_token_quality=None,
            last_logits=None,
        )

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_layers(self) -> int:
        return len(self.layer_keys)

    def append(self, token: Tensor) -> None:
        """Append one generated token per batch element to ``input_ids``."""

        if token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("token must have shape [B, 1]")
        if token.shape[0] != self.input_ids.shape[0]:
            raise ValueError("token batch size must match cache batch size")
        self.input_ids = torch.cat([self.input_ids, token], dim=1)
