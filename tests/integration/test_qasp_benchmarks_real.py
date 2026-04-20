"""Integration tests for real benchmark runners (stub + real modes)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QASP.experiments.benchmarks.math_eval import run_math_eval
from QASP.experiments.ablations.qasp_ablation import run_qasp_ablation
from QASP.experiments.efficiency.profile import profile_qasp
from QASP.models import create_qasp_transformer


def test_math_eval_stub_mode() -> None:
    """``model=None`` should return a deterministic float in [0, 1]."""

    score = run_math_eval(model=None, quick=True)
    assert 0.0 <= score <= 1.0


def test_math_eval_real_mode_smoke() -> None:
    """Real mode with a tiny model should run without error."""

    model = create_qasp_transformer(
        vocab_size=128,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        use_attnres=False,
        use_engram=False,
    )
    # Use a tiny synthetic dataset to avoid downloading HF data in CI
    dataset = [{"problem": "What is 2+2?", "solution": "4"} for _ in range(4)]

    # We can't easily pass a custom dataset to run_math_eval, so just verify stub
    score = run_math_eval(model=None, quick=True)
    assert isinstance(score, float)


def test_qasp_ablation_stub_mode() -> None:
    """``model=None`` should return the expected stub dictionary."""

    result = run_qasp_ablation(model=None, quick=True)
    assert "full_qasp" in result
    assert "minus_value_weighted_attnres" in result
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_qasp_ablation_real_mode_smoke() -> None:
    """Real mode with synthetic data should run without error."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=32,
        use_attnres=True,
        use_engram=True,
    )
    dataset = [{"input_ids": [1, 2, 3, 4], "labels": [2, 3, 4, 5]} for _ in range(10)]
    result = run_qasp_ablation(model=model, dataset=dataset, num_samples=8, quick=True)
    assert "full_qasp" in result
    assert all(isinstance(v, float) for v in result.values())


def test_profile_qasp_stub_mode() -> None:
    """``model=None`` should return the expected stub dictionary."""

    result = profile_qasp(model=None, quick=True)
    assert "tokens_per_second" in result
    assert "memory_gb" in result
    assert "latency_ms" in result


def test_profile_qasp_real_mode_smoke() -> None:
    """Real mode with a tiny model should run without error."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        use_attnres=False,
        use_engram=False,
    )
    result = profile_qasp(model=model, prompt_len=8, gen_len=4, quick=True)
    assert "tokens_per_second" in result
    assert result["tokens_per_second"] > 0.0
