"""
Unit tests for Adaptive qTTT Configuration.

TDD: Red → Green → Refactor
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.qttt.adaptive_config import AdaptiveQTTTConfig, compute_adaptive_steps, compute_adaptive_lr


class TestAdaptiveQTTTConfig:
    """Test adaptive qTTT configuration."""

    def test_default_initialization(self):
        """Test default config values."""
        cfg = AdaptiveQTTTConfig()
        assert cfg.base_steps == 4
        assert cfg.max_steps == 16
        assert cfg.base_lr == 0.01
        assert cfg.min_lr == 0.001
        assert cfg.seq_len_thresholds == [4096, 32768]

    def test_custom_initialization(self):
        """Test custom config values."""
        cfg = AdaptiveQTTTConfig(base_steps=8, max_steps=32, base_lr=0.05, min_lr=0.005)
        assert cfg.base_steps == 8
        assert cfg.max_steps == 32
        assert cfg.base_lr == 0.05
        assert cfg.min_lr == 0.005

    def test_get_steps_for_sequence_length(self):
        """Test dynamic steps based on sequence length."""
        cfg = AdaptiveQTTTConfig(base_steps=4, max_steps=16, seq_len_thresholds=[4096, 32768])

        # Short sequence: base steps
        assert cfg.get_steps_for_seq_len(2000) == 4

        # Medium sequence: increased steps
        assert cfg.get_steps_for_seq_len(8000) > 4
        assert cfg.get_steps_for_seq_len(8000) <= 16

        # Long sequence: max steps
        assert cfg.get_steps_for_seq_len(100000) == 16

    def test_get_lr_for_gradient_magnitude(self):
        """Test dynamic learning rate based on gradient."""
        cfg = AdaptiveQTTTConfig(base_lr=0.01, min_lr=0.001)

        # Large gradient: reduce LR
        lr_large_grad = cfg.get_lr_for_gradient(1.0)
        assert lr_large_grad < 0.01

        # Small gradient: increase LR
        lr_small_grad = cfg.get_lr_for_gradient(0.001)
        assert lr_small_grad >= 0.001
        assert lr_small_grad <= 0.01

    def test_backward_compatibility_with_dict(self):
        """Should work with plain dict config."""
        cfg = AdaptiveQTTTConfig.from_dict({"num_steps": 8, "learning_rate": 0.02})
        assert cfg.base_steps == 8
        assert cfg.base_lr == 0.02

    def test_to_dict_conversion(self):
        """Convert to plain dict for qTTT."""
        cfg = AdaptiveQTTTConfig(base_steps=8, base_lr=0.02)
        d = cfg.to_dict(seq_len=256)

        assert "num_steps" in d
        assert "learning_rate" in d
        assert d["num_steps"] >= 8  # Should be scaled up for longer seq


class TestAdaptiveStepComputation:
    """Test step computation functions."""

    def test_compute_adaptive_steps_linear(self):
        """Test linear scaling of steps."""
        steps = compute_adaptive_steps(
            seq_len=256, base_steps=4, max_steps=16, thresholds=[128, 512, 1024], mode="linear"
        )
        assert 4 <= steps <= 16
        assert isinstance(steps, int)

    def test_compute_adaptive_steps_log(self):
        """Test logarithmic scaling."""
        steps_log = compute_adaptive_steps(
            seq_len=1024, base_steps=4, max_steps=16, thresholds=[128, 512, 1024], mode="log"
        )
        assert 4 <= steps_log <= 16

    def test_compute_adaptive_steps_short_sequence(self):
        """Short sequences use base steps."""
        steps = compute_adaptive_steps(
            seq_len=64, base_steps=4, max_steps=16, thresholds=[128, 512, 1024]
        )
        assert steps == 4

    def test_compute_adaptive_steps_long_sequence(self):
        """Long sequences use max steps."""
        steps = compute_adaptive_steps(
            seq_len=2048, base_steps=4, max_steps=16, thresholds=[128, 512, 1024]
        )
        assert steps == 16


class TestAdaptiveLRComputation:
    """Test learning rate computation."""

    def test_compute_adaptive_lr_large_gradient(self):
        """Large gradients should reduce LR."""
        lr = compute_adaptive_lr(grad_norm=1.0, base_lr=0.01, min_lr=0.001)
        assert lr < 0.01
        assert lr >= 0.001

    def test_compute_adaptive_lr_small_gradient(self):
        """Small gradients can use higher LR."""
        lr = compute_adaptive_lr(grad_norm=0.0001, base_lr=0.01, min_lr=0.001)
        assert lr <= 0.01
        assert lr >= 0.001

    def test_compute_adaptive_lr_clipping(self):
        """LR should be clipped to valid range."""
        # Very large gradient
        lr_large = compute_adaptive_lr(grad_norm=100.0, base_lr=0.01, min_lr=0.001)
        assert lr_large >= 0.001

        # Very small gradient
        lr_small = compute_adaptive_lr(grad_norm=0.000001, base_lr=0.01, min_lr=0.001)
        assert lr_small <= 0.01


class TestIntegration:
    """Integration tests."""

    def test_config_with_model_integration(self):
        """Test config works with actual model parameters."""
        from src.models.configs import get_config

        model_config = get_config("small")
        qttt_cfg = AdaptiveQTTTConfig()

        # Should adapt based on model's typical sequence length
        steps = qttt_cfg.get_steps_for_seq_len(512)
        assert steps >= qttt_cfg.base_steps

    def test_monotonic_scaling(self):
        """Steps should increase monotonically with seq_len."""
        cfg = AdaptiveQTTTConfig()

        lengths = [64, 128, 256, 512, 1024, 2048]
        steps_list = [cfg.get_steps_for_seq_len(l) for l in lengths]

        # Should be non-decreasing
        for i in range(len(steps_list) - 1):
            assert steps_list[i] <= steps_list[i + 1]
