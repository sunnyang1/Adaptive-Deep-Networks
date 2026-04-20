"""Projection utilities for Stiefel-constrained adaptation matrices.

Implements the matrix sign function ``msign(W) = W (W^T W)^{-1/2}`` via the
Newton-Schulz iteration of QASP paper Algorithm 1 (Section 4.3):

    Y_0 = W / ||W||_2                           # spectral normalisation
    Y_{t+1} = 0.5 * Y_t (3 I_k - Y_t^T Y_t)     # degree-3 polynomial step

The iteration acts directly on ``W`` (rather than on ``W^T W``) and converges
superlinearly to the nearest Stiefel matrix whenever ``||I - Y_0^T Y_0||_F < 1``
(QASP paper, Lemma 1). Spectral normalisation ensures ``||Y_0||_2 = 1``, which
satisfies this assumption for full column rank ``W``. Each call is stateless
(no warm-start cache across adaptation steps; see Implementation Details in
``QASP_paper.tex``).
"""

from __future__ import annotations

from typing import cast

import torch


def project_to_stiefel(
    matrix: torch.Tensor,
    num_iters: int = 5,
    eps: float = 1e-6,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Project ``matrix`` onto the Stiefel manifold via Newton-Schulz.

    Args:
        matrix: ``[d, k]`` tensor with ``d >= k`` and full column rank.
        num_iters: Number of Newton-Schulz polynomial steps ``T``. The QASP
            paper (Table 2) uses ``T = 5``.
        eps: Small positive constant used to guard the spectral-norm
            normalisation against zero matrices.

    Returns:
        A ``[d, k]`` tensor whose columns are approximately orthonormal.
        If ``return_diagnostics`` is True, returns a tuple of
        ``(projected_tensor, diagnostics_dict)`` where the dict contains
        ``orthogonality_error`` = ``||I - Y^T Y||_F`` after the final iteration.
    """

    if matrix.ndim != 2:
        raise ValueError("`matrix` must be 2D.")
    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError(
            "`matrix` must satisfy rows >= cols for column-orthonormal projection."
        )
    if num_iters < 1:
        raise ValueError("`num_iters` must be >= 1.")
    if eps <= 0:
        raise ValueError("`eps` must be positive.")

    spectral_norm = torch.linalg.matrix_norm(matrix, ord=2).clamp_min(eps)
    y = matrix / spectral_norm

    cols = matrix.shape[1]
    eye = torch.eye(cols, device=matrix.device, dtype=matrix.dtype)

    for _ in range(num_iters):
        gram = y.transpose(-2, -1) @ y
        y = 0.5 * y @ (3.0 * eye - gram)

    projected = cast(torch.Tensor, y)
    if not return_diagnostics:
        return projected

    final_gram = projected.transpose(-2, -1) @ projected
    ortho_error = torch.linalg.matrix_norm(final_gram - eye, ord="fro").item()
    return projected, {"orthogonality_error": ortho_error}
