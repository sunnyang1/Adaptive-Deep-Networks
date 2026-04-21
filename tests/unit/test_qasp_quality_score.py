"""Unit tests for QASP quality score computation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QASP.adaptation.quality_score import compute_quality_score
from QASP.models.components import compute_block_representations


def test_quality_scores_are_bounded_in_unit_interval() -> None:
    """Quality scores must always stay in [0, 1]."""

    torch.manual_seed(0)
    signal = torch.randn(6, 128)

    quality = compute_quality_score(signal, low_pass_ratio=0.25)

    assert quality.shape == (6,)
    assert torch.all(quality >= 0.0)
    assert torch.all(quality <= 1.0)


def test_quality_score_has_per_token_shape_for_sequence_input() -> None:
    """Score is computed along the channel dim and preserves [B, T]."""

    torch.manual_seed(0)
    signal = torch.randn(2, 7, 64)

    quality = compute_quality_score(signal, low_pass_ratio=0.25)

    assert quality.shape == (2, 7)
    assert torch.all(quality >= 0.0)
    assert torch.all(quality <= 1.0)


def test_high_frequency_signal_has_higher_quality_than_smooth_signal() -> None:
    """Per QASP Eq. (5), rho(t) = 1 - s(t): info-rich content scores higher."""

    t = torch.linspace(0.0, 2.0 * torch.pi, 256)
    smooth = torch.sin(t).repeat(4, 1)
    torch.manual_seed(0)
    noisy = torch.randn(4, 256)

    smooth_quality = compute_quality_score(smooth, low_pass_ratio=0.25).mean()
    noisy_quality = compute_quality_score(noisy, low_pass_ratio=0.25).mean()

    assert noisy_quality > smooth_quality


def test_sliding_window_matches_full_pass_float32() -> None:
    """Chunked FFT along ``d`` must match one batched call (within float noise)."""

    torch.manual_seed(1)
    signal = torch.randn(4, 99, 128, dtype=torch.float32)
    ref = compute_quality_score(signal, low_pass_ratio=0.25)
    sliding = compute_quality_score(signal, low_pass_ratio=0.25, window_size=32)
    assert ref.shape == sliding.shape
    assert torch.allclose(ref, sliding, atol=1e-6, rtol=1e-6)


def test_sliding_window_matches_full_pass_float64() -> None:
    torch.manual_seed(2)
    signal = torch.randn(2, 50, 64, dtype=torch.float64)
    ref = compute_quality_score(signal, low_pass_ratio=0.2)
    sliding = compute_quality_score(signal, low_pass_ratio=0.2, window_size=17)
    assert torch.allclose(ref, sliding, atol=1e-12, rtol=1e-12)


def test_window_size_one_treated_as_full_pass() -> None:
    """``window_size=1`` is a no-op (same as ``None``)."""

    torch.manual_seed(3)
    x = torch.randn(2, 11, 32)
    ref = compute_quality_score(x, window_size=None)
    one = compute_quality_score(x, window_size=1)
    assert torch.equal(ref, one)


def test_window_larger_than_sequence_is_single_chunk() -> None:
    torch.manual_seed(4)
    x = torch.randn(3, 8, 64)
    ref = compute_quality_score(x)
    big = compute_quality_score(x, window_size=512)
    assert torch.equal(ref, big)


def test_invalid_window_size_raises() -> None:
    x = torch.randn(1, 5, 16)
    with pytest.raises(ValueError, match="window_size"):
        compute_quality_score(x, window_size=0)


def test_compute_block_representations_respects_quality_window_size() -> None:
    torch.manual_seed(5)
    hidden = torch.randn(2, 24, 48)
    a0, b0 = compute_block_representations(hidden, num_blocks=3, quality_window_size=None)
    a1, b1 = compute_block_representations(hidden, num_blocks=3, quality_window_size=7)
    assert torch.allclose(a0, a1)
    assert torch.allclose(b0, b1, atol=1e-6, rtol=1e-6)
