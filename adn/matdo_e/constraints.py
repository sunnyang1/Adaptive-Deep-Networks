"""Constraint helpers for MATDO-E."""
from __future__ import annotations

import math


def clamp_ratio(ratio: float) -> float:
    """Clamp ratio to [0, 1]."""
    if not math.isfinite(ratio):
        return 0.0
    return max(0.0, min(1.0, ratio))


def positive_int(x: float) -> int:
    """Convert to positive int, minimum 1."""
    if not math.isfinite(x):
        return 1
    return max(1, int(x))
