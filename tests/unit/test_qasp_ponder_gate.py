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

