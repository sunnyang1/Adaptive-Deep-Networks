from __future__ import annotations

import math
from dataclasses import dataclass

from adn.matdo_e.config import MATDOConfig
from adn.matdo_e.constraints import positive_int


@dataclass(frozen=True)
class ErrorBreakdown:
    quantization: float
    scope: float
    specificity: float
    space_scope_coupling: float
    scope_specificity_coupling: float
    retrieval: float

    @property
    def total(self) -> float:
        return (
            self.quantization
            + self.scope
            + self.specificity
            + self.space_scope_coupling
            + self.scope_specificity_coupling
            + self.retrieval
        )


def estimate_error(
    *,
    r_bits: int,
    m_blocks: int,
    t_steps: int,
    engram_entries: int,
    config: MATDOConfig,
) -> ErrorBreakdown:
    """Evaluate the lightweight analytic error model from the paper."""
    if m_blocks <= 0:
        return ErrorBreakdown(
            quantization=0.0,
            scope=math.inf,
            specificity=0.0,
            space_scope_coupling=0.0,
            scope_specificity_coupling=0.0,
            retrieval=0.0,
        )

    m_blocks = positive_int(m_blocks)
    t_steps = positive_int(t_steps)
    engram_entries = max(0, engram_entries)

    quantization = config.alpha * (2.0 ** (-2 * r_bits))
    scope = (
        config.beta
        * config.engram_compensation(engram_entries)
        / (m_blocks * config.scope_span)
    )
    specificity = config.gamma / math.sqrt(t_steps)
    space_scope_coupling = config.delta * (2.0 ** (-2 * r_bits)) / m_blocks
    scope_specificity_coupling = config.epsilon * math.log(m_blocks) / t_steps
    retrieval = config.eta / engram_entries if engram_entries > 0 else 0.0

    return ErrorBreakdown(
        quantization=quantization,
        scope=scope,
        specificity=specificity,
        space_scope_coupling=space_scope_coupling,
        scope_specificity_coupling=scope_specificity_coupling,
        retrieval=retrieval,
    )


def required_adaptation_steps(
    *,
    r_bits: int,
    m_blocks: int,
    engram_entries: int,
    target_error: float,
    config: MATDOConfig,
) -> int:
    """Closed-form adaptation steps T to meet the residual budget (same as policy layer)."""
    if m_blocks <= 0:
        return 0

    quantization = config.alpha * (2.0 ** (-2 * r_bits))
    scope = (
        config.beta
        * config.engram_compensation(engram_entries)
        / (m_blocks * config.scope_span)
    )
    retrieval = config.eta / engram_entries if engram_entries > 0 else 0.0
    space_scope_coupling = config.delta * (2.0 ** (-2 * r_bits)) / m_blocks
    residual_budget = target_error - quantization - scope - retrieval - space_scope_coupling

    if residual_budget <= 0.0:
        return config.max_t_steps

    coupling = config.epsilon * math.log(m_blocks)
    if coupling <= 0.0:
        required = math.ceil((config.gamma / residual_budget) ** 2)
        return min(config.max_t_steps, positive_int(required))

    discriminant = config.gamma**2 + 4.0 * coupling * residual_budget
    x_root = (-config.gamma + math.sqrt(discriminant)) / (2.0 * coupling)
    if x_root <= 0.0:
        return config.max_t_steps

    required = math.ceil(1.0 / (x_root**2))
    return min(config.max_t_steps, positive_int(required))
