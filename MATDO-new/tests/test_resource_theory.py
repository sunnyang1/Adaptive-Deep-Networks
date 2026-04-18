from __future__ import annotations

import math

from matdo_new.core.config import MATDOConfig
from matdo_new.core.resource_theory import (
    dram_max_engram_entries,
    hbm_max_m_blocks,
    m_min_closed_form,
    rho_context_wall,
)


def test_m_min_matches_paper_structure_without_engram() -> None:
    config = MATDOConfig(scope_span=4, alpha=0.015, beta=2.0, delta=0.005)
    r = 2
    target = 0.05
    q = config.alpha * (2.0 ** (-2 * r))
    dq = config.delta * (2.0 ** (-2 * r))
    expected = (config.beta + dq) / (config.scope_span * (target - q))
    got = m_min_closed_form(
        r_bits=r, target_error=target, config=config, engram_entries=0
    )
    assert math.isclose(got, expected)


def test_hbm_max_scales_with_quantization_bits() -> None:
    config = MATDOConfig(total_hbm_blocks=256, n_block=8, min_quantization_bits=2)
    rho = 0.96
    m2 = hbm_max_m_blocks(rho, 2, config)
    m4 = hbm_max_m_blocks(rho, 4, config)
    m8 = hbm_max_m_blocks(rho, 8, config)
    assert m2 > m4 > m8 > 0
    assert m2 >= 2 * m4 - 1
    assert m4 >= 2 * m8 - 1


def test_rho_context_wall_increases_with_m_min() -> None:
    config = MATDOConfig()
    low = rho_context_wall(r_bits=2, m_min=1.0, config=config)
    high = rho_context_wall(r_bits=2, m_min=100.0, config=config)
    assert low > high


def test_dram_wall_caps_entries() -> None:
    config = MATDOConfig(c_dram_entries=1000)
    assert dram_max_engram_entries(0.0, config) == 1000
    assert dram_max_engram_entries(0.5, config) == 500
    assert dram_max_engram_entries(1.0, config) == 0
