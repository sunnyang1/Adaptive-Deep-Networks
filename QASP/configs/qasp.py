"""Configuration objects for QASP adaptation behaviour.

Defaults align with ``tab:qasp-params`` in ``QASP_paper.tex`` for Stiefel /
ponder-gate hyperparameters. The table's sliding-window row describes optional
spectral amortization (``sec:sliding-window``); the reference
``compute_quality_score`` path is per-token FFT and does not use ``W``.

Defaults for adaptation match ``tab:qasp-params`` in the QASP paper:
    * Newton-Schulz iterations: 5
    * Test-time learning rate ``eta``: 0.01
    * Max adaptation steps ``N_iter``: 5
    * Gaussian low-pass cut-off ``f_c = d / 4`` (``low_pass_ratio = 0.25``)
    * Ponder-gate entropy threshold ``tau_H``: 0.8
    * Ponder-gate confidence threshold ``tau_c``: 0.6
"""

from dataclasses import dataclass


@dataclass
class QASPConfig:
    """Hyperparameters controlling matrix-space adaptation."""

    step_size: float = 1e-2
    num_adapt_steps: int = 5
    ns_iters: int = 5
    epsilon: float = 1e-6
    low_pass_ratio: float = 0.25
    entropy_threshold: float = 0.8
    confidence_threshold: float = 0.6
