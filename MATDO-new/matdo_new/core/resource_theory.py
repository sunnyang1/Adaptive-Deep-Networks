from __future__ import annotations

import math

from matdo_new.core.config import MATDOConfig
from matdo_new.core.constraints import clamp_ratio


def m_min_closed_form(
    *,
    r_bits: int,
    target_error: float,
    config: MATDOConfig,
    engram_entries: int,
) -> float:
    """Definition 3.1 / Eq. (2), and Engram extension Eq. (4) from MATDO-E.

    Returns +inf when the denominator is non-positive (infeasible target).
    """
    r = float(r_bits)
    q_term = config.alpha * (2.0 ** (-2.0 * r))
    dq = config.delta * (2.0 ** (-2.0 * r))
    if engram_entries <= 0:
        numer = config.beta + dq
        denom = config.scope_span * (target_error - q_term)
    else:
        f_e = config.engram_compensation(engram_entries)
        numer = config.beta * f_e + dq
        eta_e = config.eta / float(max(engram_entries, 1))
        denom = config.scope_span * (target_error - q_term - eta_e)
    if denom <= 0.0 or not math.isfinite(denom):
        return math.inf
    value = numer / denom
    return value if math.isfinite(value) else math.inf


def rho_context_wall(
    *,
    r_bits: int,
    m_min: float,
    config: MATDOConfig,
) -> float:
    """Definition 3.2 / Eq. (3): utilization at which HBM fits exactly M_min blocks."""
    if not math.isfinite(m_min) or m_min <= 0.0:
        return 1.0
    cap = config.hbm_kv_capacity()
    if cap <= 0.0:
        return 1.0
    ratio = (m_min * config.n_block * float(r_bits) * config.c_unit_kv) / cap
    return clamp_ratio(1.0 - ratio)


def hbm_max_m_blocks(rho_hbm: float, r_bits: int, config: MATDOConfig) -> int:
    """Upper bound on M from HBM: M * N_block * R * C_unit <= C_HBM * (1 - rho)."""
    rho_hbm = clamp_ratio(rho_hbm)
    denom = config.n_block * float(r_bits) * config.c_unit_kv
    cap = config.hbm_kv_capacity()
    if denom <= 0.0 or cap <= 0.0:
        return 0
    raw = (cap * (1.0 - rho_hbm)) / denom
    if not math.isfinite(raw):
        return 0
    return max(0, int(math.floor(raw)))


def dram_max_engram_entries(rho_dram: float, config: MATDOConfig) -> int:
    """DRAM wall: E * L <= C_DRAM * (1 - rho_dram) in normalized entry units (Appendix A)."""
    rho_dram = clamp_ratio(rho_dram)
    avail = config.c_dram_entries * (1.0 - rho_dram)
    if not math.isfinite(avail) or avail <= 0.0:
        return 0
    return max(0, int(math.floor(avail)))


def t_max_from_compute_budget(*, m_blocks: int, config: MATDOConfig) -> float:
    """Definition 3.3: T_max = (B_max - c_R R_min d - c_M M S d) / (c_T d^2).

    Returns ``+inf`` when ``compute_budget_flops`` is unset (no compute wall).
    """
    if config.compute_budget_flops is None:
        return math.inf
    d = float(config.model_dim_d)
    if d <= 0.0:
        return 0.0
    r_min = float(config.min_quantization_bits)
    s = float(config.scope_span)
    m = float(max(0, m_blocks))
    b = float(config.compute_budget_flops)
    num = b - config.c_r_flops * r_min * d - config.c_m_flops * m * s * d
    den = config.c_t_flops * d * d
    if den <= 0.0 or num <= 0.0:
        return 0.0
    return num / den


def rho_compute_wall(
    *,
    rho_dram: float = 0.0,
    config: MATDOConfig,
    target_error: float | None = None,
    grid_steps: int = 400,
    online_estimate: object | None = None,
) -> float | None:
    """Definition 3.3: largest rho with T*(rho) <= T_max(B_max, M, ...).

    Uses :func:`solve_policy` at each grid point. Returns ``None`` when no FLOPs
    budget is configured (``compute_budget_flops is None``).
    """
    if config.compute_budget_flops is None:
        return None

    from matdo_new.core.policy import RuntimeObservation, solve_policy

    best = 0.0
    rho_dram_eff = clamp_ratio(rho_dram)
    n = max(1, int(grid_steps))
    for i in range(n + 1):
        rho = (i / n) * 0.999
        obs = RuntimeObservation(
            rho_hbm=rho,
            rho_dram=rho_dram_eff,
            target_error=target_error,
        )
        decision = solve_policy(obs, config, online_estimate)
        t_req = float(decision.t_steps)
        t_cap = t_max_from_compute_budget(m_blocks=decision.m_blocks, config=config)
        if t_cap == math.inf:
            continue
        if t_req <= t_cap + 1.0e-6:
            best = rho
    return best
