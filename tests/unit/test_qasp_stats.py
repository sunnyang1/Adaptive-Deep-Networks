"""Unit tests for QASP statistical testing utilities."""

from __future__ import annotations

import math

import pytest
import torch

from QASP.experiments.stats import bootstrap_ci, cohens_d, paired_t_test


def test_paired_t_test_identical_samples() -> None:
    """Identical samples should yield t=0 and p=1."""

    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.0, 2.0, 3.0, 4.0]
    t, p = paired_t_test(a, b)
    assert t == pytest.approx(0.0, abs=1e-6)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_paired_t_test_clear_difference() -> None:
    """A clear systematic difference should yield a small p-value."""

    a = [10.0, 11.0, 12.0, 13.0, 14.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    t, p = paired_t_test(a, b)
    assert t > 0.0
    assert p < 0.01


def test_paired_t_test_rejects_mismatched_lengths() -> None:
    """Mismatched lengths must raise ValueError."""

    with pytest.raises(ValueError, match="same length"):
        paired_t_test([1.0, 2.0], [1.0])


def test_bootstrap_ci_mean_contains_true_mean() -> None:
    """The 95% CI for a normal sample should contain the true mean."""

    torch.manual_seed(0)
    data = torch.randn(100).tolist()
    point, lower, upper = bootstrap_ci(data, n_resamples=5_000, confidence=0.95)
    assert lower < point < upper


def test_bootstrap_ci_median() -> None:
    """Bootstrap CI should work for the median statistic."""

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    point, lower, upper = bootstrap_ci(data, n_resamples=2_000, statistic="median")
    assert lower < point < upper
    assert point == pytest.approx(5.5, abs=0.5)


def test_bootstrap_ci_rejects_invalid_confidence() -> None:
    """Confidence outside (0, 1) must raise ValueError."""

    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0, 2.0], confidence=0.0)


def test_cohens_d_zero_effect() -> None:
    """Identical samples should yield d = 0."""

    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    d = cohens_d(a, b)
    assert d == pytest.approx(0.0, abs=1e-6)


def test_cohens_d_large_effect() -> None:
    """Independent N(1,1) vs N(0,1) should yield paired d ≈ 1/√2 ≈ 0.71."""

    torch.manual_seed(1)
    a = (torch.randn(200) + 1.0).tolist()
    b = torch.randn(200).tolist()
    d = cohens_d(a, b)
    # For independent samples with unit variance, SD(diff) = sqrt(2)
    assert d == pytest.approx(1.0 / math.sqrt(2.0), abs=0.15)


def test_cohens_d_rejects_mismatched_lengths() -> None:
    """Mismatched lengths must raise ValueError."""

    with pytest.raises(ValueError, match="same length"):
        cohens_d([1.0, 2.0], [1.0])
