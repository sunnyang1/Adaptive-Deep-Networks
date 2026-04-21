"""Smoke tests for the real needle-in-a-haystack benchmark runner."""

from __future__ import annotations

import pytest
import torch

from QASP.experiments.benchmarks.needle import run_needle_benchmark
from QASP.models import create_qasp_transformer


def test_needle_benchmark_runs_with_default_model() -> None:
    """The benchmark should execute end-to-end without a user-supplied model."""

    acc = run_needle_benchmark(quick=True)
    assert 0.0 <= acc <= 1.0


def test_needle_benchmark_runs_with_custom_model() -> None:
    """The benchmark should accept an explicit model instance."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=128,
        use_attnres=False,
        use_engram=False,
    )
    acc = run_needle_benchmark(
        model=model,
        vocab_size=64,
        context_length=32,
        needle_len=2,
        num_trials=3,
        quick=False,
    )
    assert 0.0 <= acc <= 1.0


def test_needle_benchmark_position_distributions() -> None:
    """All supported position distributions should run without error."""

    for dist in ("uniform", "front", "end", "middle"):
        acc = run_needle_benchmark(
            quick=True,
            position_distribution=dist,  # type: ignore[arg-type]
        )
        assert 0.0 <= acc <= 1.0


def test_needle_benchmark_deterministic_with_seed() -> None:
    """Fixing the torch manual seed should yield deterministic accuracy."""

    torch.manual_seed(123)
    acc1 = run_needle_benchmark(quick=True, num_trials=4)

    torch.manual_seed(123)
    acc2 = run_needle_benchmark(quick=True, num_trials=4)

    assert acc1 == acc2
