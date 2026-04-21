"""Statistical testing utilities for QASP benchmark comparison.

Implements paired t-test, bootstrap confidence intervals, and Cohen's d
as described in QASP paper Section 6.6.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Paired two-tailed t-test.

    Args:
        a: First set of matched observations.
        b: Second set of matched observations (same length as ``a``).

    Returns:
        ``(t_statistic, p_value)`` where ``p_value`` is two-tailed.
    """

    if len(a) != len(b):
        raise ValueError("`a` and `b` must have the same length.")
    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 observations for a t-test.")

    diffs = torch.tensor([float(x) - float(y) for x, y in zip(a, b)], dtype=torch.float64)
    mean_diff = diffs.mean()
    std_diff = diffs.std(unbiased=True)

    if std_diff.item() == 0.0:
        return float("inf" if mean_diff != 0 else 0.0), 0.0 if mean_diff != 0 else 1.0

    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Two-tailed p-value using the regularized incomplete beta function
    # scipy is not guaranteed to be installed, so we use torch.special
    from torch.special import gammaln

    df = n - 1
    x = df / (df + t_stat.item() ** 2)

    # Approximate p-value via the survival function of the t-distribution
    # For simplicity we use the normal approximation when df is large,
    # otherwise a direct integration would be needed.  Here we fall back to
    # torch.distributions.StudentT when available (PyTorch >= 1.12).
    try:
        from torch.distributions import StudentT

        dist = StudentT(df=df)
        p = 2.0 * (1.0 - dist.cdf(abs(t_stat.item())))
        return float(t_stat.item()), float(p)
    except Exception:
        # Very rough fallback: normal approximation
        from torch.distributions import Normal

        p = 2.0 * (1.0 - Normal(0, 1).cdf(abs(t_stat.item())))
        return float(t_stat.item()), float(p)


def bootstrap_ci(
    data: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    statistic: str = "mean",
) -> tuple[float, float, float]:
    """Bootstrap percentile confidence interval.

    Args:
        data: Observations.
        n_resamples: Number of bootstrap replicates.
        confidence: Desired confidence level (e.g. 0.95).
        statistic: ``"mean"`` or ``"median"``.

    Returns:
        ``(point_estimate, lower_bound, upper_bound)``.
    """

    if len(data) < 1:
        raise ValueError("`data` must be non-empty.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("`confidence` must be in (0, 1).")

    tensor = torch.tensor(data, dtype=torch.float64)
    n = len(tensor)

    # Point estimate
    if statistic == "mean":
        point = tensor.mean().item()
    elif statistic == "median":
        point = tensor.median().item()
    else:
        raise ValueError("`statistic` must be 'mean' or 'median'.")

    # Bootstrap replicates
    indices = torch.randint(0, n, (n_resamples, n))
    samples = tensor[indices]
    if statistic == "mean":
        replicates = samples.mean(dim=1)
    else:
        replicates = samples.median(dim=1).values

    alpha = 1.0 - confidence
    lower = torch.quantile(replicates, alpha / 2.0).item()
    upper = torch.quantile(replicates, 1.0 - alpha / 2.0).item()

    return point, lower, upper


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for paired samples (mean difference / pooled SD).

    Args:
        a: First set of matched observations.
        b: Second set of matched observations.

    Returns:
        Effect size (float).  Positive values mean ``a > b``.
    """

    if len(a) != len(b):
        raise ValueError("`a` and `b` must have the same length.")
    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    diffs = torch.tensor([float(x) - float(y) for x, y in zip(a, b)], dtype=torch.float64)
    mean_diff = diffs.mean().item()
    std_diff = diffs.std(unbiased=True).item()

    if std_diff == 0.0:
        return float("inf" if mean_diff != 0 else 0.0)

    return mean_diff / std_diff
