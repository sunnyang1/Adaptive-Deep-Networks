"""
Unit tests for Gating module.

Tests:
- EMA / target-rate threshold calibration
- Reconstruction loss computation
"""

import pytest
import torch

from src.gating.threshold import EMAThreshold, TargetRateThreshold
from src.gating.reconstruction import compute_reconstruction_loss


class TestEMAThreshold:
    """Tests for EMAThreshold."""

    def test_ema_init(self):
        dt = EMAThreshold(initial_threshold=0.5, beta=0.99, percentile=70.0)
        assert dt.threshold.item() == 0.5

    def test_ema_update_moves_threshold(self):
        dt = EMAThreshold(initial_threshold=1.0, beta=0.5, percentile=50.0)
        initial = dt.threshold.item()
        for _ in range(20):
            dt.update(2.0)
        assert dt.threshold.item() != initial


class TestTargetRateThreshold:
    """Tests for TargetRateThreshold."""

    def test_target_rate_update(self):
        dt = TargetRateThreshold(initial_threshold=0.5, target_rate=0.3, learning_rate=0.05)
        for _ in range(50):
            dt.update(0.9)  # high loss → adapt often
        assert dt.threshold.item() != 0.5


class TestReconstructionLoss:
    """Tests for reconstruction loss (logits vs token targets)."""

    def test_compute_reconstruction_loss_scalar(self):
        batch_size, seq_len, vocab = 2, 10, 64
        logits = torch.randn(batch_size, seq_len, vocab)
        targets = torch.randint(0, vocab, (batch_size, seq_len))
        loss = compute_reconstruction_loss(logits, targets)
        assert loss.shape == ()
        assert loss.item() == loss.item()  # finite

    def test_compute_reconstruction_loss_zero_for_matching_targets(self):
        """Loss is ~0 when each row is a one-hot at the target token."""
        batch_size, seq_len, vocab = 2, 5, 32
        targets = torch.randint(0, vocab, (batch_size, seq_len))
        logits = torch.full((batch_size, seq_len, vocab), -1e9)
        logits.scatter_(-1, targets.unsqueeze(-1), 0.0)
        loss = compute_reconstruction_loss(logits, targets, span_length=seq_len)
        assert torch.allclose(loss, torch.zeros(()), atol=1e-4)

    def test_compute_reconstruction_loss_positive(self):
        batch_size, seq_len, vocab = 2, 5, 32
        logits = torch.randn(batch_size, seq_len, vocab)
        targets = torch.randint(0, vocab, (batch_size, seq_len))
        loss = compute_reconstruction_loss(logits, targets)
        assert loss.item() >= 0

    def test_compute_reconstruction_loss_gradient(self):
        batch_size, seq_len, vocab = 1, 3, 16
        logits = torch.randn(batch_size, seq_len, vocab, requires_grad=True)
        targets = torch.randint(0, vocab, (batch_size, seq_len))
        loss = compute_reconstruction_loss(logits, targets)
        loss.backward()
        assert logits.grad is not None
