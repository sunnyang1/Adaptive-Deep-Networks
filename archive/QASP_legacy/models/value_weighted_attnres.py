"""Value-weighted AttnRes block (QASP Sec.~5.2; ``eq:block-quality`` / ``eq:value-weighted-attnres``).

Implements the value-weighted attention over preceding block representations
using a learned per-layer pseudo-query ``w_ℓ ∈ R^d``:

    score_{m→ℓ}    = (w_ℓ · B_m) · ρ̄_m / sqrt(d)
    α_{m→ℓ}^{(ρ)} = softmax_m( score_{m→ℓ} )
    h_ℓ           = sum_m α_{m→ℓ}^{(ρ)} · B_m

``ρ̄_m`` is the mean information-quality score over the tokens in block ``m``.
When ``ρ̄_m`` is small, block ``m`` contributes less to the
softmax, biasing depth aggregation towards content-rich blocks.

**Canonical semantics.**  The paper defines ``B_m`` and ``ρ̄_m`` from a *single*
forward pass over the full sequence (fixed context).  Callers should pass
``block_representations`` and ``block_quality`` computed from that full pass
(e.g. via :func:`QASP.models.components.compute_block_representations`).  The
broadcast residual is added at every token position; this matches the
manuscript's evaluation definition and differs in general from prefix-only
incremental decoding (see :meth:`QASP.models.qasp_transformer.QASPTransformer.step`).
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor


class ValueWeightedAttnRes(nn.Module):
    """Per-layer value-weighted aggregation of block representations.

    Inputs must reflect the **full-sequence** block summaries and qualities used
    in the paper's reference forward pass; see module docstring.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_size))
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        block_representations: Tensor,
        block_quality: Tensor,
    ) -> Tensor:
        """Compute the value-weighted residual ``h_ℓ`` and broadcast over T.

        Args:
            hidden_states: ``[B, T, D]`` (only the shape is consumed for the
                output broadcast; the residual itself is layer-level).
            block_representations: ``[B, N, D]`` — block summaries ``B_m``.
            block_quality: ``[B, N]`` — block-level mean quality ``ρ̄_m`` in
                ``[0, 1]``.

        Returns:
            ``[B, T, D]`` residual that the caller can add to ``hidden_states``.
        """

        if block_representations.ndim != 3:
            raise ValueError("block_representations must have shape [B, N, D]")
        if block_representations.shape[-1] != self.hidden_size:
            raise ValueError(
                "block_representations last dim must match hidden_size."
            )
        if block_quality.ndim != 2:
            raise ValueError("block_quality must have shape [B, N]")
        if block_quality.shape != block_representations.shape[:2]:
            raise ValueError("block_quality must align with block_representations.")

        scale = 1.0 / math.sqrt(float(self.hidden_size))
        affinity = torch.einsum("d,bnd->bn", self.pseudo_query, block_representations)
        scores = affinity * block_quality * scale
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", weights, block_representations)

        residual = self.output_proj(pooled)
        expanded = residual.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        return cast(Tensor, expanded)
