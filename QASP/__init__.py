"""Top-level public exports for QASP package.

**Paper vs. reference scope.**  Algorithmic hyperparameters in ``QASPConfig`` match
the manuscript where implemented. Spectral ``rho(t)`` follows ``eq:quality-score``
(per-token FFT); sliding-window amortization in the paper is optional and not
shipped here. Training-table details (BF16, compile flags) describe the reported
1.5B pipeline, not hard requirements of these modules.

**Canonical evaluation semantics (Path A).**  In the QASP manuscript,
value-weighted AttnRes (labels ``eq:block-quality`` / ``eq:value-weighted-attnres``
in ``QASP_paper.tex``) and block statistics ``ρ̄_m`` are defined on a
*single full-sequence forward pass* over a fixed context.  :meth:`QASPTransformer.forward`
and :meth:`QASPTransformer.prefill` implement that definition.  Autoregressive
:meth:`QASPTransformer.step` uses a prefix-growing hidden history for block
pooling when AttnRes is enabled; logits therefore need not match a hypothetical
full forward over the extended sequence.  See integration tests under
``tests/integration/test_qasp_prefill_step_numeric_parity.py``.
"""

from QASP.adaptation import PonderGate, compute_quality_score, matrix_qasp_update, project_to_stiefel
from QASP.configs import ExperimentConfig, ModelConfig, QASPConfig
from QASP.models import (
    QASPLayer,
    QASPTransformer,
    QASPTransformerConfig,
    ValueWeightedAttnRes,
    ValueWeightedEngram,
    create_qasp_transformer,
)

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "QASPConfig",
    "compute_quality_score",
    "matrix_qasp_update",
    "PonderGate",
    "project_to_stiefel",
    "QASPTransformerConfig",
    "QASPLayer",
    "QASPTransformer",
    "create_qasp_transformer",
    "ValueWeightedAttnRes",
    "ValueWeightedEngram",
]

