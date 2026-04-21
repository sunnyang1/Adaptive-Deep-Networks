"""Matrix-space QASP adaptation loop with Stiefel projection.

Implements the inner update of QASP paper Algorithm~2 (``alg:qasp``, Sec.~5.4):

    G^(n)        = ∇_W L(W^(n); {h_t})       # Step 1: per-iteration gradient
    G̃^(n)        = ρ̄ · G^(n)                  # Step 2: batch-level quality modulation
    W'           = W^(n) - η · G̃^(n)          # Step 3: Euclidean update
    W^(n+1)      = msign(W')                  # Step 4: Stiefel projection

The function accepts either a ``loss_fn`` callable (recommended; matches the
paper, recomputing the gradient at every iteration) or a static ``gradient``
tensor for legacy callers and minimal smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from adn.qasp.stiefel import project_to_stiefel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QASPConfig:
    """Hyperparameters controlling matrix-space adaptation.

    Defaults align with ``tab:qasp-params`` in ``QASP_paper.tex`` for Stiefel /
    ponder-gate hyperparameters. Optional spectral amortization (``sec:sliding-window``)
    uses ``quality_window_size`` (e.g. 512) in :func:`adn.qasp.quality_score.compute_quality_score`;
    ``None`` keeps a single fused FFT over the sequence.

    Defaults for adaptation match ``tab:qasp-params`` in the QASP paper:
        * Newton-Schulz iterations: 5
        * Test-time learning rate ``eta``: 0.01
        * Max adaptation steps ``N_iter``: 5
        * Gaussian low-pass cut-off ``f_c = d / 4`` (``low_pass_ratio = 0.25``)
        * Optional ``quality_window_size`` (e.g. 512) for chunked ``rho`` along the sequence
        * Ponder-gate entropy threshold ``tau_H``: 0.8
        * Ponder-gate confidence threshold ``tau_c``: 0.6
    """

    step_size: float = 1e-2
    num_adapt_steps: int = 5
    ns_iters: int = 5
    epsilon: float = 1e-6
    low_pass_ratio: float = 0.25
    quality_window_size: int | None = None
    entropy_threshold: float = 0.8
    confidence_threshold: float = 0.6


# ---------------------------------------------------------------------------
# Ponder gate
# ---------------------------------------------------------------------------

@dataclass
class PonderGate:
    """Entropy/confidence ponder gate aligned with QASP Algorithm 2."""

    entropy_threshold: float = 0.8
    confidence_threshold: float = 0.6
    eps: float = 1e-8

    def should_adapt(self, logits: torch.Tensor) -> bool:
        """Return True when any batch element is uncertain or low-confidence.

        Args:
            logits: ``[B, V]`` for a single decoding step, or ``[B, T, V]`` in
                which case the last-token distribution is used (paper Alg. 2).
        """

        if logits.ndim < 2:
            raise ValueError("`logits` must be at least 2D [..., vocab].")
        if logits.ndim > 3:
            raise ValueError("`logits` must be 2D [B, V] or 3D [B, T, V].")

        if logits.ndim == 3:
            logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(self.eps))).sum(dim=-1)
        confidence = probs.max(dim=-1).values

        per_row_adapt = (entropy > self.entropy_threshold) | (
            confidence < self.confidence_threshold
        )
        return bool(per_row_adapt.any())


# ---------------------------------------------------------------------------
# Matrix QASP update
# ---------------------------------------------------------------------------

def matrix_qasp_update(
    matrix: torch.Tensor,
    gradient: Optional[torch.Tensor] = None,
    quality_scores: Optional[torch.Tensor] = None,
    *,
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    step_size: float = 1e-2,
    num_adapt_steps: int = 1,
    ns_iters: int = 5,
    eps: float = 1e-6,
    gate: Optional[PonderGate] = None,
    logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run ``num_adapt_steps`` QASP updates on ``matrix``.

    Args:
        matrix: ``[d, k]`` query matrix to adapt.
        gradient: Optional static gradient tensor with the same shape as
            ``matrix``. Mutually exclusive with ``loss_fn``.
        quality_scores: Optional per-token quality scores ``ρ(t)``. The mean
            ``ρ̄ = quality_scores.mean()`` provides the batch-level modulation
            applied at Step 2 of the paper algorithm.
        loss_fn: Optional callable ``W -> scalar loss`` evaluated under
            ``torch.enable_grad`` at each iteration to produce ``∇_W L``. This
            is the paper-faithful path (Algorithm 2 Step 1).
        step_size: Test-time learning rate ``η`` (paper default 0.01).
        num_adapt_steps: Number ``N_iter`` of inner iterations.
        ns_iters: Newton-Schulz iterations for the Stiefel projection.
        eps: Numerical guard forwarded to ``project_to_stiefel``.
        gate: Optional ponder gate; if it returns False, ``matrix`` is returned
            untouched.
        logits: Logits forwarded to ``gate.should_adapt``.

    Returns:
        Updated matrix on (or close to) the Stiefel manifold.
    """

    if matrix.ndim != 2:
        raise ValueError("`matrix` must be 2D.")
    if num_adapt_steps < 1:
        raise ValueError("`num_adapt_steps` must be >= 1.")
    if loss_fn is None and gradient is None:
        raise ValueError("Either `loss_fn` or `gradient` must be provided.")
    if loss_fn is not None and gradient is not None:
        raise ValueError("Provide exactly one of `loss_fn` or `gradient`.")
    if gradient is not None and gradient.shape != matrix.shape:
        raise ValueError("`matrix` and `gradient` must have matching shapes.")

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

    adapted = matrix.detach().clone()
    for _ in range(num_adapt_steps):
        step_grad = _resolve_gradient(adapted, gradient, loss_fn)
        adapted = adapted - step_size * quality_weight * step_grad
        adapted = project_to_stiefel(adapted, num_iters=ns_iters, eps=eps)
    return adapted


def _resolve_gradient(
    current: torch.Tensor,
    static_gradient: Optional[torch.Tensor],
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """Return the gradient to use this step (recomputed via autograd if possible)."""

    if loss_fn is None:
        assert static_gradient is not None
        return static_gradient

    var = current.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        loss_value = loss_fn(var)
        if loss_value.ndim != 0:
            raise ValueError("`loss_fn` must return a scalar tensor.")
        (grad_tensor,) = torch.autograd.grad(loss_value, var, create_graph=False)
    return grad_tensor.detach()


# ---------------------------------------------------------------------------
# Object-oriented wrapper (for pipeline / config-driven use)
# ---------------------------------------------------------------------------

class MatrixQASP:
    """Config-driven wrapper around :func:`matrix_qasp_update`.

    Example::

        cfg = QASPConfig(step_size=0.01, num_adapt_steps=5)
        adapter = MatrixQASP(cfg)
        W_adapted = adapter.adapt(W, loss_fn=my_loss)
    """

    def __init__(self, config: QASPConfig | None = None) -> None:
        self.config = config or QASPConfig()

    def adapt(
        self,
        matrix: torch.Tensor,
        gradient: Optional[torch.Tensor] = None,
        quality_scores: Optional[torch.Tensor] = None,
        *,
        loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one QASP adaptation pass on ``matrix`` using the stored config."""
        cfg = self.config
        gate = None
        if logits is not None:
            gate = PonderGate(
                entropy_threshold=cfg.entropy_threshold,
                confidence_threshold=cfg.confidence_threshold,
            )
        return matrix_qasp_update(
            matrix=matrix,
            gradient=gradient,
            quality_scores=quality_scores,
            loss_fn=loss_fn,
            step_size=cfg.step_size,
            num_adapt_steps=cfg.num_adapt_steps,
            ns_iters=cfg.ns_iters,
            eps=cfg.epsilon,
            gate=gate,
            logits=logits,
        )
