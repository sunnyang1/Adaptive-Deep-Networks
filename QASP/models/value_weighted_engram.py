"""Value-weighted Engram fusion (QASP Sec.~5.3, ``eq:value-weighted-engram``).

Implements

    h' = h + α · σ(ρ_mem) · m,   α = σ(w_g · h),

where ``w_g`` is a learned gate (cf.~base Engram fusion in Sec.~2.3 of the paper)
and ``σ(ρ_mem)`` modulates the contribution by the average information
quality stored alongside the memory entry.

For test/debug use, ``forward`` accepts an explicit ``gate`` override that
replaces ``α``; the default path computes ``α`` from the hidden state itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ValueWeightedEngram(nn.Module):
    """Fuse hidden states with retrieved memory under a quality-aware gate."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_proj = nn.Linear(hidden_size, 1, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        memory_vector: Tensor,
        memory_quality: Tensor,
        gate: Tensor | float | None = None,
    ) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, D]")
        batch_size, seq_len, _ = hidden_states.shape

        if memory_vector.ndim == 2:
            memory_vector = memory_vector.unsqueeze(1).expand(-1, seq_len, -1)
        if memory_vector.shape != hidden_states.shape:
            raise ValueError("memory_vector must broadcast to hidden_states shape [B, T, D]")

        if memory_quality.ndim == 1:
            memory_quality = memory_quality.unsqueeze(1).expand(-1, seq_len)
        if memory_quality.shape != (batch_size, seq_len):
            raise ValueError("memory_quality must broadcast to [B, T]")
        quality_gate = torch.sigmoid(memory_quality).unsqueeze(-1)

        if gate is None:
            content_gate = torch.sigmoid(self.gate_proj(hidden_states))
        else:
            content_gate = self._coerce_gate(gate, hidden_states, batch_size, seq_len)

        return hidden_states + content_gate * quality_gate * memory_vector

    def _coerce_gate(
        self,
        gate: Tensor | float,
        hidden_states: Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tensor:
        if not torch.is_tensor(gate):
            return torch.tensor(
                float(gate),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        gate_term = gate.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if gate_term.ndim == 0:
            return gate_term
        if gate_term.ndim == 1:
            if gate_term.shape[0] != batch_size:
                raise ValueError("1D gate tensor must have shape [B]")
            return gate_term.view(batch_size, 1, 1)
        if gate_term.ndim == 2:
            if gate_term.shape != (batch_size, seq_len):
                raise ValueError("2D gate tensor must have shape [B, T]")
            return gate_term.unsqueeze(-1)
        raise ValueError("gate must be scalar, [B], or [B, T]")
