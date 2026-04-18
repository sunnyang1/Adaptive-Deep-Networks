from __future__ import annotations

import math

from matdo_new.core.config import MATDOConfig
from matdo_new.core.policy import RuntimeObservation, solve_policy


def test_policy_prefers_engram_in_arbitrage_zone() -> None:
    config = MATDOConfig()

    decision = solve_policy(
        RuntimeObservation(rho_hbm=0.96, rho_dram=0.25),
        config=config,
    )

    assert decision.use_engram is True
    assert decision.is_arbitrage is True
    assert decision.engram_entries > 0
    assert decision.m_blocks > 0
    assert decision.t_steps > 0


def test_policy_uses_observation_target_error_for_arbitrage_choice() -> None:
    config = MATDOConfig()

    relaxed = solve_policy(
        RuntimeObservation(rho_hbm=0.96, rho_dram=0.25),
        config=config,
    )
    strict = solve_policy(
        RuntimeObservation(rho_hbm=0.96, rho_dram=0.25, target_error=0.01),
        config=config,
    )

    assert relaxed.use_engram is True
    assert strict.use_engram is False
    assert strict.target_error == 0.01


def test_policy_uses_observation_target_error_for_arbitrage_inequality() -> None:
    config = MATDOConfig(zeta=0.35, eta=0.5, e_max=128_000, target_error=0.05)

    assert config.arbitrage_inequality_holds() is True
    assert config.arbitrage_inequality_holds(target_error=1e-5) is False

    relaxed = solve_policy(
        RuntimeObservation(rho_hbm=0.96, rho_dram=0.25),
        config=config,
    )
    override = solve_policy(
        RuntimeObservation(rho_hbm=0.96, rho_dram=0.25, target_error=1e-5),
        config=config,
    )

    assert relaxed.use_engram is True
    assert override.use_engram is False
    assert override.is_arbitrage is False
    assert override.reason == "baseline-mode"


def test_policy_preserves_explicit_zero_hbm_capacity() -> None:
    config = MATDOConfig()

    decision = solve_policy(
        RuntimeObservation(rho_hbm=0.40, available_hbm_blocks=0),
        config=config,
    )

    assert decision.use_engram is False
    assert decision.engram_entries == 0
    assert decision.m_blocks == 0
    assert decision.t_steps == 0
    assert math.isinf(decision.estimated_error)
