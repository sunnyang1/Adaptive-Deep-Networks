"""Projection utilities for Stiefel-constrained adaptation matrices."""

from __future__ import annotations

import torch


def project_to_stiefel(
    matrix: torch.Tensor,
    num_iters: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project a matrix onto the Stiefel manifold using Newton-Schulz iterations.

    This projection enforces near-orthonormal columns by computing:
    `W_proj = W (W^T W + eps I)^(-1/2)`.
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

    cols = matrix.shape[1]
    eye = torch.eye(cols, device=matrix.device, dtype=matrix.dtype)
    gram = matrix.transpose(0, 1) @ matrix + eps * eye

    # Normalize for Newton-Schulz stability.
    norm_gram = gram.norm(p="fro").clamp_min(eps)
    y = gram / norm_gram
    z = eye.clone()

    for _ in range(num_iters):
        t = 0.5 * (3.0 * eye - z @ y)
        y = y @ t
        z = t @ z

    inv_sqrt_gram = z / torch.sqrt(norm_gram)
    return matrix @ inv_sqrt_gram

