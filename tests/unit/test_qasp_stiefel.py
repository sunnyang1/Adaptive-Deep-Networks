"""Unit tests for QASP Stiefel projection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QASP.adaptation.stiefel import project_to_stiefel


def test_project_to_stiefel_produces_near_orthonormal_columns() -> None:
    """Projected matrices should have approximately orthonormal columns."""

    torch.manual_seed(0)
    matrix = torch.randn(32, 8, dtype=torch.float64)

    projected = project_to_stiefel(matrix, num_iters=12, eps=1e-8)

    gram = projected.transpose(0, 1) @ projected
    identity = torch.eye(projected.shape[1], dtype=projected.dtype)
    assert torch.allclose(gram, identity, atol=1e-3, rtol=1e-3)


def test_project_to_stiefel_raises_when_rows_less_than_cols() -> None:
    """Column-orthonormal projection requires rows >= cols."""

    invalid = torch.randn(4, 6)
    with pytest.raises(ValueError, match="rows >= cols"):
        project_to_stiefel(invalid)

