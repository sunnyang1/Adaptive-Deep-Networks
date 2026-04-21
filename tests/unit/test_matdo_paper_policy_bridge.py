"""Tests for experiments/matdo/paper_policy_bridge.py (MATDO-new optional bridge)."""

from __future__ import annotations

import json

from experiments.matdo.common.config import MATDOConfig
from experiments.matdo.paper_policy_bridge import (
    legacy_to_paper_config,
    policy_payload_for_experiments,
)


def test_legacy_to_paper_config_maps_coefficients() -> None:
    legacy = MATDOConfig()
    paper = legacy_to_paper_config(legacy)
    assert paper.alpha == legacy.alpha
    assert paper.beta == legacy.beta
    assert paper.target_error == legacy.E_target
    assert tuple(paper.quantization_bits) == tuple(sorted(legacy.R_options))


def test_policy_payload_is_json_serializable() -> None:
    payload = policy_payload_for_experiments(
        MATDOConfig(),
        rho_hbm=0.91,
        rho_dram=0.2,
    )
    json.dumps(payload)
    assert "policy_decision" in payload
    assert "materialized_policy" in payload
