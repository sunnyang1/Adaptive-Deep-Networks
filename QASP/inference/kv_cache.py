"""Lightweight cache objects for autoregressive QASP inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KVCache:
    """Minimal cache holding generated token history for decoding."""

    input_ids: Tensor

    @classmethod
    def from_input_ids(cls, input_ids: Tensor) -> "KVCache":
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        return cls(input_ids=input_ids.clone())

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    def append(self, token: Tensor) -> None:
        """Append one generated token per batch element."""
        if token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("token must have shape [B, 1]")
        if token.shape[0] != self.input_ids.shape[0]:
            raise ValueError("token batch size must match cache batch size")
        self.input_ids = torch.cat([self.input_ids, token], dim=1)
