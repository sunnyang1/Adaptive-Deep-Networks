"""
Simplified unit tests for Gating module.
"""

import pytest
import torch
import torch.nn as nn

from src.gating.threshold import DynamicThreshold, EMAThreshold


class TestDynamicThreshold:
    """Tests for DynamicThreshold base class."""

    def test_dynamic_threshold_init(self):
        """Test DynamicThreshold initialization."""
        dt = DynamicThreshold(initial_threshold=2.0)

        assert dt.threshold.item() == 2.0

    def test_should_adapt(self):
        """Test should_adapt logic."""
        dt = DynamicThreshold(initial_threshold=1.5)

        # Loss below threshold - should not adapt
        assert not dt.should_adapt(1.0)

        # Loss above threshold - should adapt
        assert dt.should_adapt(2.0)

    def test_get_stats_empty(self):
        """Test get_stats with empty history."""
        dt = DynamicThreshold(initial_threshold=1.0)

        stats = dt.get_stats()

        assert stats["threshold"] == 1.0
        assert stats["history_size"] == 0


class TestEMAThreshold:
    """Tests for EMAThreshold."""

    def test_ema_threshold_init(self):
        """Test EMAThreshold initialization."""
        ema = EMAThreshold(initial_threshold=2.0, beta=0.99, percentile=70.0)

        assert ema.threshold.item() == 2.0
        assert ema.beta == 0.99
        assert ema.percentile == 70.0

    def test_ema_update(self):
        """Test EMA update mechanism."""
        ema = EMAThreshold(initial_threshold=1.0, beta=0.9)

        initial = ema.threshold.item()

        # Update with high loss
        ema.update(5.0)

        # Threshold should have moved up
        assert ema.threshold.item() > initial
