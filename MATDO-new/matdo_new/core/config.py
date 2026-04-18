from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MATDOConfig:
    """Small paper-aligned configuration for the MATDO runtime policy."""

    quantization_bits: tuple[int, ...] = (2, 4, 8)
    min_quantization_bits: int = 2

    # Scope / capacity knobs
    scope_span: int = 4
    total_hbm_blocks: int = 256
    min_scope_blocks: int = 1
    max_t_steps: int = 4096

    # Appendix A: HBM budget in units of (M * N_block * R * c_unit_kv). At R = min_quantization_bits
    # and rho = 0 this matches total_hbm_blocks blocks of resident scope.
    n_block: int = 8
    c_unit_kv: float = 1.0

    # Appendix A: DRAM budget in Engram-entry units (aligned with e_max by default).
    c_dram_entries: int = 128_000

    # SLA / regime thresholds
    target_error: float = 0.05
    arbitrage_zone_rho: float = 0.93
    critical_zone_rho: float = 0.98
    dram_utilization_limit: float = 0.90

    # Engram parameters
    e_max: int = 128_000
    e0: float = 10_000.0
    zeta: float = 0.35
    eta: float = 0.5

    # Error model coefficients from the MATDO-E paper
    alpha: float = 0.015
    beta: float = 2.0
    gamma: float = 0.10
    delta: float = 0.005
    epsilon: float = 0.002

    # Definition 3.3 (optional): FLOPs budget for T_max and rho_comp
    compute_budget_flops: float | None = None
    c_r_flops: float = 1.2e3
    c_m_flops: float = 2.5e3
    c_t_flops: float = 8.0e4
    model_dim_d: int = 4096

    def engram_compensation(self, engram_entries: int) -> float:
        """Return the paper's compensation term f(E)."""
        if engram_entries <= 0:
            return 1.0
        return 1.0 - self.zeta * (1.0 - math.exp(-engram_entries / self.e0))

    def hbm_kv_capacity(self) -> float:
        """Normalized C_HBM for KV (product M * N_block * R * c_unit uses this budget at rho = 0)."""
        return float(self.total_hbm_blocks * self.n_block * self.min_quantization_bits * self.c_unit_kv)

    def arbitrage_inequality_holds(self, target_error: float | None = None) -> bool:
        """Check the heterogeneous arbitrage inequality."""
        effective_target = self.target_error if target_error is None else target_error
        return self.zeta > self.eta / (self.e_max * effective_target)
