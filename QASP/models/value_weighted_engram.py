"""Value-weighted engram fusion module."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ValueWeightedEngram(nn.Module):
    """Fuse hidden states with memory vectors and quality-aware gating."""

    def __init__(self, hidden_size: int, init_gate: float = 1.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.default_gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(
        self,
        hidden_states: Tensor,
        memory_vector: Tensor,
        memory_quality: Tensor,
        gate: Tensor | float | None = None,
    ) -> Tensor:
        if memory_vector.ndim == 2:
            memory_vector = memory_vector.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        if memory_vector.shape != hidden_states.shape:
            raise ValueError("memory_vector must broadcast to hidden_states shape [B, T, D]")

        if memory_quality.ndim == 1:
            memory_quality = memory_quality.unsqueeze(1).expand(-1, hidden_states.size(1))
        quality = torch.sigmoid(memory_quality).unsqueeze(-1)

        gate_term = self.default_gate if gate is None else gate
        if not torch.is_tensor(gate_term):
            gate_term = torch.tensor(float(gate_term), device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            gate_term = gate_term.to(device=hidden_states.device, dtype=hidden_states.dtype)

        return hidden_states + gate_term * quality * memory_vector

