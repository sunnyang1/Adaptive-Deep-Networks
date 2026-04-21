from __future__ import annotations

import math
from dataclasses import dataclass

from adn.matdo_e.config import MATDOConfig
from adn.matdo_e.constraints import clamp_ratio
from adn.matdo_e.error_model import estimate_error, required_adaptation_steps
from adn.matdo_e.online_estimation import OnlineEstimate
from adn.matdo_e.resource_theory import (
    dram_max_engram_entries,
    hbm_max_m_blocks,
    m_min_closed_form,
    rho_context_wall,
)


@dataclass(frozen=True)
class RuntimeObservation:
    """Online runtime signals used by the MATDO policy."""

    rho_hbm: float
    rho_dram: float = 0.0
    available_hbm_blocks: int | None = None
    target_error: float | None = None


@dataclass(frozen=True)
class PolicyDecision:
    """Resolved MATDO control setting for the current runtime regime."""

    quantization_bits: int
    m_blocks: int
    t_steps: int
    engram_entries: int
    use_engram: bool
    is_arbitrage: bool
    estimated_error: float
    target_error: float
    reason: str
    paper_m_min: float | None = None
    rho_ctx: float | None = None


def _required_t_steps(
    *,
    r_bits: int,
    m_blocks: int,
    engram_entries: int,
    target_error: float,
    config: MATDOConfig,
) -> int:
    return required_adaptation_steps(
        r_bits=r_bits,
        m_blocks=m_blocks,
        engram_entries=engram_entries,
        target_error=target_error,
        config=config,
    )


def _effective_engram_entries(*, config: MATDOConfig, rho_dram: float, use_engram: bool) -> int:
    if not use_engram:
        return 0
    cap = dram_max_engram_entries(rho_dram, config)
    return min(config.e_max, cap)


def _evaluate_candidate(
    *,
    r_bits: int,
    m_blocks: int,
    engram_entries: int,
    target_error: float,
    config: MATDOConfig,
    use_engram: bool,
    reason: str,
    paper_m_min: float,
    rho_ctx_val: float,
) -> PolicyDecision:
    t_steps = _required_t_steps(
        r_bits=r_bits,
        m_blocks=m_blocks,
        engram_entries=engram_entries,
        target_error=target_error,
        config=config,
    )
    estimated = estimate_error(
        r_bits=r_bits,
        m_blocks=m_blocks,
        t_steps=t_steps,
        engram_entries=engram_entries,
        config=config,
    )
    return PolicyDecision(
        quantization_bits=r_bits,
        m_blocks=m_blocks,
        t_steps=t_steps,
        engram_entries=engram_entries,
        use_engram=use_engram,
        is_arbitrage=use_engram,
        estimated_error=estimated.total,
        target_error=target_error,
        reason=reason,
        paper_m_min=paper_m_min,
        rho_ctx=rho_ctx_val,
    )


def _solve_regime(
    *,
    observation: RuntimeObservation,
    config: MATDOConfig,
    target_error: float,
    use_engram: bool,
    reason: str,
) -> PolicyDecision:
    rho_hbm = clamp_ratio(observation.rho_hbm)
    rho_dram = clamp_ratio(observation.rho_dram)
    engram_entries = _effective_engram_entries(
        config=config, rho_dram=rho_dram, use_engram=use_engram
    )
    if use_engram and engram_entries <= 0:
        return _solve_regime(
            observation=observation,
            config=config,
            target_error=target_error,
            use_engram=False,
            reason="dram-wall-disables-engram",
        )

    explicit = observation.available_hbm_blocks

    best: PolicyDecision | None = None
    best_key: tuple[float, int, int] | None = None

    for r_bits in sorted(config.quantization_bits):
        m_cap = hbm_max_m_blocks(rho_hbm, r_bits, config)
        if explicit is not None:
            m_cap = min(m_cap, max(0, int(explicit)))

        m_min_raw = m_min_closed_form(
            r_bits=r_bits,
            target_error=target_error,
            config=config,
            engram_entries=engram_entries,
        )
        if not math.isfinite(m_min_raw):
            continue

        m_floor = max(config.min_scope_blocks, int(math.ceil(m_min_raw)))
        if m_cap < m_floor:
            continue

        m_blocks = m_cap
        rho_ctx_val = rho_context_wall(r_bits=r_bits, m_min=m_min_raw, config=config)
        candidate = _evaluate_candidate(
            r_bits=r_bits,
            m_blocks=m_blocks,
            engram_entries=engram_entries,
            target_error=target_error,
            config=config,
            use_engram=use_engram,
            reason=reason,
            paper_m_min=m_min_raw,
            rho_ctx_val=rho_ctx_val,
        )
        key = (candidate.estimated_error, candidate.t_steps, r_bits)
        if best_key is None or key < best_key:
            best = candidate
            best_key = key

    if best is not None:
        return best

    r_fallback = min(config.quantization_bits)
    return PolicyDecision(
        quantization_bits=r_fallback,
        m_blocks=0,
        t_steps=0,
        engram_entries=engram_entries,
        use_engram=use_engram and engram_entries > 0,
        is_arbitrage=use_engram and engram_entries > 0,
        estimated_error=math.inf,
        target_error=target_error,
        reason=reason,
        paper_m_min=None,
        rho_ctx=None,
    )


def solve_policy(
    observation: RuntimeObservation,
    config: MATDOConfig | None = None,
    online_estimate: OnlineEstimate | None = None,
) -> PolicyDecision:
    """Solve the lightweight MATDO policy for the current runtime observation."""
    resolved_config = config or MATDOConfig()
    if online_estimate is not None:
        resolved_config = online_estimate.apply(resolved_config)

    rho_hbm = clamp_ratio(observation.rho_hbm)
    rho_dram = clamp_ratio(observation.rho_dram)
    target_error = (
        observation.target_error
        if observation.target_error is not None
        else resolved_config.target_error
    )
    in_arbitrage_zone = (
        rho_hbm >= resolved_config.arbitrage_zone_rho
        and rho_dram < resolved_config.dram_utilization_limit
        and resolved_config.arbitrage_inequality_holds(target_error=target_error)
        and dram_max_engram_entries(rho_dram, resolved_config) > 0
    )

    baseline = _solve_regime(
        observation=observation,
        config=resolved_config,
        target_error=target_error,
        use_engram=False,
        reason="baseline-mode",
    )
    if not in_arbitrage_zone:
        return baseline

    engram = _solve_regime(
        observation=observation,
        config=resolved_config,
        target_error=target_error,
        use_engram=True,
        reason="arbitrage-zone-prefers-engram",
    )
    baseline_meets_target = baseline.estimated_error <= target_error
    engram_meets_target = engram.estimated_error <= target_error
    if engram_meets_target and (not baseline_meets_target or engram.estimated_error <= baseline.estimated_error):
        return engram
    return baseline
