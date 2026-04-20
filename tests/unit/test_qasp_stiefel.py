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


def test_project_to_stiefel_five_iters_error_below_1e_minus_4() -> None:
    """With 5 iterations the orthogonality error should be < 1e-4 (paper claim).

    The paper assumes E_0_F ≈ 0.5, which holds when d >> k (e.g. d=2048, k=16).
    For smaller matrices the initial error is larger and more iterations are needed.
    """

    torch.manual_seed(42)
    # Use paper-like dimensions where d >> k to match the paper's convergence claim
    matrix = torch.randn(2048, 16, dtype=torch.float64)

    projected, diag = project_to_stiefel(matrix, num_iters=5, eps=1e-8, return_diagnostics=True)
    assert isinstance(diag, dict)
    assert "orthogonality_error" in diag
    assert diag["orthogonality_error"] < 1e-4, (
        f"Expected orthogonality_error < 1e-4, got {diag['orthogonality_error']}"
    )

    # Backward-compat path still returns a tensor
    tensor_only = project_to_stiefel(matrix, num_iters=5, eps=1e-8)
    assert isinstance(tensor_only, torch.Tensor)
    assert torch.allclose(tensor_only, projected)


def test_project_to_stiefel_diagnostics_improves_with_more_iters() -> None:
    """More iterations should yield smaller orthogonality error."""

    torch.manual_seed(7)
    matrix = torch.randn(32, 8, dtype=torch.float64)

    _, diag1 = project_to_stiefel(matrix, num_iters=1, return_diagnostics=True)
    _, diag3 = project_to_stiefel(matrix, num_iters=3, return_diagnostics=True)
    _, diag5 = project_to_stiefel(matrix, num_iters=5, return_diagnostics=True)

    assert diag1["orthogonality_error"] > diag3["orthogonality_error"] > diag5["orthogonality_error"]
