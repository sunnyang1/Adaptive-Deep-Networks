"""Ponder gate for deciding whether test-time adaptation should fire.

Implements the QASP paper rule (Sections 2.4 and 5.4, Algorithm 2):

    adapt = 1[ H(p) > tau_H  OR  max_i p_i < tau_c ],

where ``p = softmax(logits_last)`` is evaluated on the final-token
distribution. The defaults ``tau_H = 0.8`` and ``tau_c = 0.6`` match the
paper's Table 2 hyper-parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
