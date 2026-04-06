"""
Unit tests for Gating module.

Tests:
- EMAThreshold / TargetRateThreshold calibration (implemented APIs)
- Reconstruction loss computation
"""

import pytest
import torch

from src.gating.threshold import EMAThreshold, TargetRateThreshold
from src.gating.reconstruction import compute_reconstruction_loss


class TestEMAThreshold:
    """Tests for EMAThreshold."""

    def test_ema_threshold_init(self):
        """EMAThreshold holds initial threshold."""
        dt = EMAThreshold(initial_threshold=0.5, beta=0.9, percentile=70.0)

        assert dt.threshold.item() == pytest.approx(0.5)

    def test_should_adapt_initial(self):
        """should_adapt compares loss to current threshold."""
        dt = EMAThreshold(initial_threshold=0.5)

        assert dt.should_adapt(0.6) is True
        assert dt.should_adapt(0.1) is False

    def test_ema_update(self):
        """After updates with enough samples, threshold moves."""
        dt = EMAThreshold(initial_threshold=0.5, beta=0.9, percentile=70.0)

        initial = dt.threshold.item()

        for loss in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            dt.update(loss)

        assert dt.threshold.item() != pytest.approx(initial)


class TestTargetRateThreshold:
    """Tests for TargetRateThreshold."""

    def test_target_rate_update(self):
        """High adaptation pressure should push threshold up."""
        dt = TargetRateThreshold(
            initial_threshold=0.5,
            target_rate=0.3,
            learning_rate=0.05,
            window_size=50,
        )

        for _ in range(100):
            dt.update(0.9)

        assert dt.threshold.item() > 0.5

    def test_get_stats(self):
        """get_stats includes target-rate fields."""
        dt = TargetRateThreshold(initial_threshold=0.5, target_rate=0.3)

        for _ in range(10):
            dt.update(0.4)

        stats = dt.get_stats()
        assert "threshold" in stats
        assert "target_rate" in stats
        assert "current_rate" in stats
        assert stats["total_samples"] == 10

    def test_reset_not_on_base_class(self):
        """TargetRateThreshold tracks counts; clearing is manual if needed."""
        dt = TargetRateThreshold(initial_threshold=0.5)
        dt.update(0.8)
        assert dt.total_count >= 1


class TestReconstructionLoss:
    """Tests for reconstruction loss computation."""

    def test_compute_reconstruction_loss_shape(self):
        """Scalar loss from logits and targets."""
        batch_size = 2
        seq_len = 10
        vocab = 64

        logits = torch.randn(batch_size, seq_len, vocab)
        targets = torch.randint(0, vocab, (batch_size, seq_len))

        loss = compute_reconstruction_loss(logits, targets, span_length=seq_len)

        assert loss.shape == ()

    def test_compute_reconstruction_loss_zero(self):
        """Loss is ~0 when logits match one-hot targets."""
        batch_size = 2
        seq_len = 5
        vocab = 32

        targets = torch.randint(0, vocab, (batch_size, seq_len))
        logits = torch.zeros(batch_size, seq_len, vocab)
        logits.scatter_(-1, targets.unsqueeze(-1), 10.0)

        loss = compute_reconstruction_loss(logits, targets, span_length=seq_len)

        assert torch.allclose(loss, torch.tensor(0.0), atol=5e-3)

    def test_compute_reconstruction_loss_positive(self):
        """Random logits give positive CE."""
        batch_size = 2
        seq_len = 5
        vocab = 32

        logits = torch.randn(batch_size, seq_len, vocab)
        targets = torch.randint(0, vocab, (batch_size, seq_len))

        loss = compute_reconstruction_loss(logits, targets, span_length=seq_len)

        assert loss.item() > 0

    def test_compute_reconstruction_loss_gradient(self):
        """Gradients flow into logits."""
        batch_size = 1
        seq_len = 3
        vocab = 16

        logits = torch.randn(batch_size, seq_len, vocab, requires_grad=True)
        targets = torch.randint(0, vocab, (batch_size, seq_len))

        loss = compute_reconstruction_loss(logits, targets, span_length=seq_len)
        loss.backward()

        assert logits.grad is not None
        assert not torch.allclose(logits.grad, torch.zeros_like(logits))
