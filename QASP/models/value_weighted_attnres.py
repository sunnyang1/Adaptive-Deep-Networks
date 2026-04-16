"""Value-weighted attention residual module."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ValueWeightedAttnRes(nn.Module):
    """Aggregate block vectors using quality-weighted softmax scores."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        block_representations: Tensor,
        quality_scores: Tensor,
    ) -> Tensor:
        if block_representations.ndim != 3:
            raise ValueError("block_representations must have shape [B, N, D]")
        if quality_scores.ndim != 2:
            raise ValueError("quality_scores must have shape [B, N]")

        weights = torch.softmax(quality_scores, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", weights, block_representations)
        pooled = pooled.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        return self.output_proj(pooled)

