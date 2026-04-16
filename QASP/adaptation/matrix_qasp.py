"""Matrix-space QASP adaptation loop with Stiefel projection."""

from __future__ import annotations

from typing import Optional

import torch

from QASP.adaptation.ponder_gate import PonderGate
from QASP.adaptation.stiefel import project_to_stiefel


def matrix_qasp_update(
    matrix: torch.Tensor,
    gradient: torch.Tensor,
    quality_scores: Optional[torch.Tensor] = None,
    *,
    step_size: float = 1e-2,
    num_adapt_steps: int = 1,
    ns_iters: int = 8,
    eps: float = 1e-6,
    gate: Optional[PonderGate] = None,
    logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply QASP adaptation with optional gate and Stiefel projection."""

    if matrix.shape != gradient.shape:
        raise ValueError("`matrix` and `gradient` must have matching shapes.")
    if matrix.ndim != 2:
        raise ValueError("`matrix` must be 2D.")
    if num_adapt_steps < 1:
        raise ValueError("`num_adapt_steps` must be >= 1.")

    if gate is not None:
        if logits is None:
            raise ValueError("`logits` are required when `gate` is provided.")
        if not gate.should_adapt(logits):
            return matrix

    if quality_scores is None:
        quality_weight = matrix.new_tensor(1.0)
    else:
        if quality_scores.numel() == 0:
            raise ValueError("`quality_scores` must be non-empty when provided.")
        mean_quality = quality_scores.mean()
        if not torch.isfinite(mean_quality):
            raise ValueError("`quality_scores` mean must be finite.")
        quality_weight = mean_quality.to(matrix.dtype).clamp(0.0, 1.0)

    adapted = matrix
    for _ in range(num_adapt_steps):
        adapted = adapted - step_size * quality_weight * gradient
        adapted = project_to_stiefel(adapted, num_iters=ns_iters, eps=eps)
    return adapted

