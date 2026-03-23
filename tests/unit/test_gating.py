"""
Unit tests for Dynamic Computation Gating.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gating.threshold import (
    EMAThreshold,
    TargetRateThreshold,
    HybridThreshold,
    GatingController
)
from gating.reconstruction import ReconstructionLoss, compute_reconstruction_loss


class TestEMAThreshold:
    """Tests for EMA threshold calibration."""
    
    def test_initialization(self):
        calibrator = EMAThreshold(initial_threshold=2.0)
        assert calibrator.threshold.item() == 2.0
        assert calibrator.beta == 0.99
    
    def test_update_decreases_threshold_for_low_loss(self):
        """If losses are consistently low, threshold should decrease."""
        calibrator = EMAThreshold(initial_threshold=2.0, beta=0.9)
        
        # Feed low losses
        for _ in range(50):
            calibrator.update(0.5)
        
        # Threshold should have decreased
        assert calibrator.threshold.item() < 2.0
    
    def test_update_increases_threshold_for_high_loss(self):
        """If losses are consistently high, threshold should increase."""
        calibrator = EMATrThreshold(initial_threshold=2.0, beta=0.9)
        
        # Feed high losses
        for _ in range(50):
            calibrator.update(5.0)
        
        # Threshold should have increased
        assert calibrator.threshold.item() > 2.0
    
    def test_should_adapt(self):
        calibrator = EMAThreshold(initial_threshold=2.0)
        
        # Loss above threshold should trigger adaptation
        assert calibrator.should_adapt(3.0) == True
        
        # Loss below threshold should not trigger
        assert calibrator.should_adapt(1.0) == False
    
    def test_get_stats(self):
        calibrator = EMAThreshold(initial_threshold=2.0)
        
        # Add some history
        for i in range(10):
            calibrator.update(float(i))
        
        stats = calibrator.get_stats()
        
        assert 'threshold' in stats
        assert 'history_size' in stats
        assert stats['history_size'] == 10


class TestTargetRateThreshold:
    """Tests for target rate threshold calibration."""
    
    def test_initialization(self):
        calibrator = TargetRateThreshold(
            initial_threshold=2.0,
            target_rate=0.3
        )
        assert calibrator.threshold.item() == 2.0
        assert calibrator.target_rate == 0.3
    
    def test_maintains_target_rate(self):
        """Should adjust threshold to maintain target rate."""
        calibrator = TargetRateThreshold(
            initial_threshold=1.0,
            target_rate=0.5,
            learning_rate=0.1
        )
        
        # Simulate 50% adaptation rate
        for i in range(100):
            loss = 2.0 if i % 2 == 0 else 0.5
            calibrator.update(loss)
        
        stats = calibrator.get_stats()
        # Should be close to target rate
        assert abs(stats['current_rate'] - 0.5) < 0.15
    
    def test_threshold_stays_positive(self):
        """Threshold should never go below small positive value."""
        calibrator = TargetRateThreshold(initial_threshold=0.1)
        
        # Feed very low losses
        for _ in range(100):
            calibrator.update(0.01)
        
        assert calibrator.threshold.item() >= 0.01


class TestGatingController:
    """Tests for GatingController."""
    
    def test_decide_returns_tuple(self):
        calibrator = EMAThreshold(initial_threshold=2.0)
        controller = GatingController(calibrator)
        
        should_adapt, num_steps, threshold = controller.decide(3.0)
        
        assert isinstance(should_adapt, bool)
        assert isinstance(num_steps, int)
        assert isinstance(threshold, float)
    
    def test_high_loss_triggers_adaptation(self):
        calibrator = EMAThreshold(initial_threshold=2.0)
        controller = GatingController(calibrator)
        
        should_adapt, num_steps, _ = controller.decide(5.0)
        
        assert should_adapt == True
        assert num_steps > 0
    
    def test_low_loss_no_adaptation(self):
        calibrator = EMAThreshold(initial_threshold=2.0)
        controller = GatingController(calibrator)
        
        should_adapt, num_steps, _ = controller.decide(0.5)
        
        assert should_adapt == False
        assert num_steps == 0
    
    def test_num_steps_scales_with_loss(self):
        """Higher excess loss should result in more steps."""
        calibrator = EMAThreshold(initial_threshold=2.0)
        controller = GatingController(calibrator, max_adaptation_steps=32)
        
        _, steps_low, _ = controller.decide(3.0)  # 1.5x threshold
        _, steps_high, _ = controller.decide(6.0)  # 3x threshold
        
        assert steps_high >= steps_low


class TestReconstructionLoss:
    """Tests for reconstruction loss computation."""
    
    def test_initialization(self):
        loss_fn = ReconstructionLoss(vocab_size=1000, hidden_dim=128)
        assert loss_fn.vocab_size == 1000
        assert loss_fn.hidden_dim == 128
    
    def test_forward_shape(self):
        loss_fn = ReconstructionLoss(vocab_size=1000, hidden_dim=128, span_length=10)
        
        hidden_states = torch.randn(2, 50, 128)  # [B, T, D]
        target_tokens = torch.randint(0, 1000, (2, 50))  # [B, T]
        
        loss = loss_fn(hidden_states, target_tokens)
        
        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0  # Loss should be positive
    
    def test_reconstruction_loss_computation(self):
        """Test standalone reconstruction loss function."""
        logits = torch.randn(2, 10, 100)  # [B, T, V]
        targets = torch.randint(0, 100, (2, 10))  # [B, T]
        
        loss = compute_reconstruction_loss(logits, targets)
        
        assert loss.shape == ()
        assert loss.item() >= 0
    
    def test_loss_decreases_with_better_prediction(self):
        """Loss should be lower for better predictions."""
        loss_fn = ReconstructionLoss(vocab_size=100, hidden_dim=64)
        
        hidden = torch.randn(2, 10, 64)
        targets = torch.randint(0, 100, (2, 10))
        
        # Make prediction perfect by setting logits
        with torch.no_grad():
            logits = loss_fn.reconstruction_head(hidden)
            # Set target positions to high logit
            for b in range(2):
                for t in range(10):
                    logits[b, t, targets[b, t]] = 10.0
        
        # Recompute loss manually
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 100),
            targets.reshape(-1)
        )
        
        assert loss.item() < 1.0  # Should be low for perfect prediction


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
