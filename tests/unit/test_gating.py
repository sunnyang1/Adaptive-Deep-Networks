"""
Unit tests for Gating module.

Tests:
- DynamicThreshold calibration
- Reconstruction loss computation
- Gating decision logic
"""

import pytest
import torch
import torch.nn as nn

from src.gating.threshold import DynamicThreshold
from src.gating.reconstruction import compute_reconstruction_loss


class TestDynamicThreshold:
    """Tests for DynamicThreshold."""
    
    def test_dynamic_threshold_init(self):
        """Test DynamicThreshold initialization."""
        dt = DynamicThreshold(initial_threshold=0.5)
        
        assert dt.threshold == config.initial_threshold
        assert dt.ema_loss == 0.0
        assert dt.decision_count == 0
        assert dt.adaptation_count == 0
    
    def test_should_adapt_initial(self):
        """Test should_adapt at initialization."""
        dt = DynamicThreshold(initial_threshold=0.5)
        
        # First few decisions should use initial threshold
        should_adapt = dt.should_adapt(0.6)
        
        # 0.6 > 0.5, so should adapt
        assert should_adapt == True
    
    def test_should_adapt_low_loss(self):
        """Test should_adapt with low reconstruction loss."""
        dt = DynamicThreshold(initial_threshold=0.5)
        
        # Low loss - should not adapt
        should_adapt = dt.should_adapt(0.1)
        
        assert should_adapt == False
    
    def test_should_adapt_high_loss(self):
        """Test should_adapt with high reconstruction loss."""
        dt = DynamicThreshold(initial_threshold=0.5)
        
        # High loss - should adapt
        should_adapt = dt.should_adapt(0.8)
        
        assert should_adapt == True
    
    def test_ema_update(self):
        """Test EMA threshold update."""
        dt = DynamicThreshold(
            initial_threshold=0.5,
            ema_decay=0.9
        )
        
        initial_threshold = dt.threshold
        
        # Update with several losses
        for loss in [0.3, 0.4, 0.5, 0.6]:
            dt.should_adapt(loss)
            dt.update_threshold(loss)
        
        # Threshold should have moved from initial value
        assert dt.threshold != initial_threshold
    
    def test_target_rate_update(self):
        """Test target rate threshold update."""
        dt = DynamicThreshold(
            initial_threshold=0.5,
            target_rate=0.3
        )
        
        # Simulate many decisions with high adaptation rate
        for _ in range(100):
            should_adapt = dt.should_adapt(0.8)  # Always high loss
            dt.record_decision(should_adapt)
        
        # Should adjust threshold to reduce adaptation rate
        # Threshold should increase
        assert dt.threshold > 0.5
    
    def test_get_stats(self):
        """Test get_stats returns expected metrics."""
        dt = DynamicThreshold(initial_threshold=0.5)
        
        # Make some decisions
        for i in range(10):
            should_adapt = dt.should_adapt(0.3 + i * 0.05)
            dt.record_decision(should_adapt)
        
        stats = dt.get_stats()
        
        assert 'threshold' in stats
        assert 'ema_loss' in stats
        assert 'total_decisions' in stats
        assert 'adaptation_count' in stats
        assert 'adaptation_rate' in stats
        assert stats['total_decisions'] == 10
    
    def test_reset(self):
        """Test reset functionality."""
        dt = DynamicThreshold(initial_threshold=0.5)
        
        # Make some decisions
        for _ in range(5):
            dt.should_adapt(0.6)
            dt.record_decision(True)
        
        # Reset
        dt.reset()
        
        assert dt.threshold == 0.5
        assert dt.ema_loss == 0.0
        assert dt.decision_count == 0
        assert dt.adaptation_count == 0


class TestReconstructionLoss:
    """Tests for reconstruction loss computation."""
    
    def test_compute_reconstruction_loss_shape(self):
        """Test reconstruction loss output shape."""
        batch_size = 2
        seq_len = 10
        dim = 64
        
        original = torch.randn(batch_size, seq_len, dim)
        reconstructed = torch.randn(batch_size, seq_len, dim)
        
        loss = compute_reconstruction_loss(original, reconstructed)
        
        # Should return a scalar
        assert loss.shape == ()
    
    def test_compute_reconstruction_loss_zero(self):
        """Test reconstruction loss is zero for identical tensors."""
        batch_size = 2
        seq_len = 5
        dim = 32
        
        original = torch.randn(batch_size, seq_len, dim)
        
        loss = compute_reconstruction_loss(original, original)
        
        # Loss should be approximately zero
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
    
    def test_compute_reconstruction_loss_positive(self):
        """Test reconstruction loss is positive for different tensors."""
        batch_size = 2
        seq_len = 5
        dim = 32
        
        original = torch.randn(batch_size, seq_len, dim)
        reconstructed = torch.randn(batch_size, seq_len, dim)
        
        loss = compute_reconstruction_loss(original, reconstructed)
        
        # Loss should be positive
        assert loss.item() > 0
    
    def test_compute_reconstruction_loss_gradient(self):
        """Test gradients flow through reconstruction loss."""
        batch_size = 1
        seq_len = 3
        dim = 16
        
        original = torch.randn(batch_size, seq_len, dim)
        reconstructed = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        
        loss = compute_reconstruction_loss(original, reconstructed)
        loss.backward()
        
        assert reconstructed.grad is not None
        assert not torch.allclose(reconstructed.grad, torch.zeros_like(reconstructed))
