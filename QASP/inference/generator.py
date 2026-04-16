"""Autoregressive generation utilities for QASP models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class QASPGenerator:
    """Simple greedy autoregressive decoder."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        output_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.model(output_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token], dim=1)
        return output_ids
