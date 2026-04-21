"""Unit tests for QASP ponder gating and matrix short-circuit behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QASP.adaptation.matrix_qasp import matrix_qasp_update
from QASP.adaptation.ponder_gate import PonderGate


def test_ponder_gate_triggers_on_high_entropy_low_confidence() -> None:
    """Uniform logits should trigger adaptation."""

    gate = PonderGate(entropy_threshold=0.7, confidence_threshold=0.4)
    logits = torch.zeros(2, 32)
    assert gate.should_adapt(logits) is True


def test_ponder_gate_stays_off_for_confident_predictions() -> None:
    """Peaky logits should not trigger adaptation."""

    gate = PonderGate(entropy_threshold=0.7, confidence_threshold=0.4)
    logits = torch.zeros(2, 16)
    logits[:, 0] = 8.0
    assert gate.should_adapt(logits) is False


def test_matrix_qasp_returns_original_when_gate_blocks_adaptation() -> None:
    """Matrix update must short-circuit to original matrix when gate is closed."""

    matrix = torch.eye(6)
    gradient = torch.ones_like(matrix)
    gate = PonderGate(entropy_threshold=0.8, confidence_threshold=0.2)

    confident_logits = torch.zeros(1, 20)
    confident_logits[0, 0] = 10.0

    updated = matrix_qasp_update(
        matrix,
        gradient,
        quality_scores=torch.ones(4),
        step_size=0.1,
        num_adapt_steps=3,
        gate=gate,
        logits=confident_logits,
    )

    assert torch.allclose(updated, matrix)


def test_matrix_qasp_raises_for_empty_quality_scores() -> None:
    """Empty quality scores should fail fast with a clear error."""

    matrix = torch.eye(4)
    gradient = torch.ones_like(matrix)

    with pytest.raises(ValueError, match="non-empty"):
        matrix_qasp_update(matrix, gradient, quality_scores=torch.tensor([]))


def test_matrix_qasp_requires_loss_fn_or_gradient() -> None:
    """At least one of loss_fn or gradient must be provided."""

    matrix = torch.eye(4)
    with pytest.raises(ValueError, match="loss_fn"):
        matrix_qasp_update(matrix)


def test_matrix_qasp_rejects_both_loss_fn_and_gradient() -> None:
    """Both loss_fn and gradient cannot be provided simultaneously."""

    matrix = torch.eye(4)
    with pytest.raises(ValueError, match="exactly one"):
        matrix_qasp_update(
            matrix,
            gradient=torch.ones_like(matrix),
            loss_fn=lambda w: (w * w).sum(),
        )


def test_matrix_qasp_with_loss_fn_decreases_loss_and_stays_on_stiefel() -> None:
    """loss_fn path follows paper Alg 2 Step 1: gradient is recomputed each step."""

    from QASP.adaptation.stiefel import project_to_stiefel

    torch.manual_seed(0)
    target = project_to_stiefel(torch.randn(8, 4, dtype=torch.float64), num_iters=8)
    start = project_to_stiefel(torch.randn(8, 4, dtype=torch.float64), num_iters=8)

    def loss_fn(weights: torch.Tensor) -> torch.Tensor:
        return ((weights - target) ** 2).sum()

    initial_loss = loss_fn(start).item()
    updated = matrix_qasp_update(
        matrix=start,
        loss_fn=loss_fn,
        step_size=0.5,
        num_adapt_steps=10,
        ns_iters=5,
    )
    final_loss = loss_fn(updated).item()

    assert final_loss < initial_loss
    gram = updated.transpose(0, 1) @ updated
    identity = torch.eye(updated.shape[1], dtype=updated.dtype)
    assert torch.allclose(gram, identity, atol=1e-3, rtol=1e-3)


def test_matrix_qasp_loss_fn_must_return_scalar() -> None:
    """loss_fn returning a non-scalar tensor should fail fast."""

    matrix = torch.eye(4)
    with pytest.raises(ValueError, match="scalar"):
        matrix_qasp_update(matrix, loss_fn=lambda w: w * 2.0)
