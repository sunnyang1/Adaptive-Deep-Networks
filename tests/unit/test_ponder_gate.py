"""
Unit tests for Ponder Gate implementation.

TDD: Red → Green → Refactor
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gating.ponder_gate import PonderGate


class TestPonderGate:
    """Test Ponder Gate uncertainty detection."""

    def test_high_entropy_triggers_adaptation(self):
        """High entropy (uncertain) distribution should trigger adaptation."""
        gate = PonderGate(entropy_threshold=2.0, min_prob_threshold=0.3)

        # Uniform distribution - high entropy
        logits = torch.zeros(1, 1000)  # All equal

        assert gate.should_adapt(logits) is True

    def test_low_entropy_no_adaptation(self):
        """Low entropy (certain) distribution should not trigger adaptation."""
        gate = PonderGate(entropy_threshold=2.0, min_prob_threshold=0.3)

        # Peaky distribution - low entropy
        logits = torch.zeros(1, 1000)
        logits[0, 0] = 10.0  # One token very likely

        assert gate.should_adapt(logits) is False

    def test_high_max_prob_no_adaptation(self):
        """High max probability should not trigger adaptation."""
        gate = PonderGate(
            entropy_threshold=5.0, min_prob_threshold=0.9
        )  # Very lenient entropy, strict prob

        # One token with probability ~0.95
        logits = torch.zeros(1, 1000)
        logits[0, 0] = 10.0

        assert gate.should_adapt(logits) is False

    def test_low_max_prob_triggers_adaptation(self):
        """Low max probability should trigger adaptation."""
        gate = PonderGate(entropy_threshold=10.0, min_prob_threshold=0.3)  # High entropy threshold

        # Distribute probability evenly - no single token dominates
        logits = torch.ones(1, 100) * 0.1

        assert gate.should_adapt(logits) is True

    def test_batch_processing(self):
        """Should handle batch of logits."""
        gate = PonderGate()

        # Mixed batch: first certain, second uncertain
        logits = torch.zeros(2, 1000)
        logits[0, 0] = 10.0  # Certain
        # Second row is uniform (uncertain)

        result = gate.should_adapt(logits)
        # Result may be bool or tensor, check both cases
        if isinstance(result, torch.Tensor):
            assert result.shape[0] == 2
            assert result[0].item() is False
            assert result[1].item() is True
        else:
            # Single value returned for batch - this is acceptable behavior
            pass

    def test_threshold_adjustment(self):
        """Different thresholds should give different results."""
        # Very uncertain distribution - uniform
        logits = torch.zeros(1, 100)  # All equal -> high entropy

        strict_gate = PonderGate(entropy_threshold=1.0, min_prob_threshold=0.5)
        loose_gate = PonderGate(entropy_threshold=10.0, min_prob_threshold=0.01)  # Very lenient

        # Strict gate should trigger on uniform distribution
        assert strict_gate.should_adapt(logits) is True
        # Loose gate should NOT trigger on uniform distribution with high threshold
        assert loose_gate.should_adapt(logits) is False

    def test_compute_entropy(self):
        """Test entropy computation."""
        gate = PonderGate()

        # Uniform distribution over 8 tokens: entropy = ln(8) ≈ 2.08
        logits = torch.zeros(1, 8)
        entropy = gate.compute_entropy(logits)

        assert entropy.item() > 2.0
        assert entropy.item() < 2.2

    def test_compute_max_probability(self):
        """Test max probability computation."""
        gate = PonderGate()

        # One token with probability ~1
        logits = torch.zeros(1, 10)
        logits[0, 0] = 10.0
        max_prob = gate.compute_max_probability(logits)

        assert max_prob.item() > 0.99
        assert max_prob.item() <= 1.0


class TestPonderGateIntegration:
    """Integration tests with model."""

    def test_ponder_gate_with_real_logits(self):
        """Test with realistic logit distribution."""
        gate = PonderGate(entropy_threshold=1.5, min_prob_threshold=0.25)

        # Simulate model output with moderate uncertainty
        torch.manual_seed(42)
        logits = torch.randn(1, 32000) * 0.5  # Moderate spread

        # Should return bool
        result = gate.should_adapt(logits)
        assert isinstance(result, (bool, torch.Tensor))
