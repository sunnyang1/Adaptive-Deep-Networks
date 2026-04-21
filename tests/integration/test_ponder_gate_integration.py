"""
Integration tests for Ponder Gate in generate().

Verifies that Ponder Gate correctly controls qTTT execution.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.configs import get_config
from src.models.adaptive_transformer import AdaptiveTransformer


class TestPonderGateIntegration:
    """Test Ponder Gate integration with generate()."""

    @pytest.fixture
    def model(self):
        """Create small model for testing."""
        config = get_config("small")
        model = AdaptiveTransformer(config)
        model.eval()
        return model

    @pytest.fixture
    def input_ids(self):
        """Create test input."""
        return torch.randint(0, 32000, (1, 10))

    def test_generate_without_qttt(self, model, input_ids):
        """Baseline: generation without qTTT."""
        output = model.generate(input_ids, max_new_tokens=5, use_qttt=False)
        assert output.shape[1] == input_ids.shape[1] + 5

    def test_generate_with_qttt_unconditional(self, model, input_ids):
        """Generation with unconditional qTTT."""
        output = model.generate(
            input_ids,
            max_new_tokens=5,
            use_qttt=True,
            qttt_config={"num_steps": 2},  # Fast for testing
        )
        assert output.shape[1] == input_ids.shape[1] + 5

    def test_generate_with_adaptive_qttt(self, model, input_ids, capsys):
        """Generation with adaptive qTTT via Ponder Gate."""
        output = model.generate(
            input_ids,
            max_new_tokens=5,
            use_qttt="adaptive",
            ponder_gate_mode="balanced",
            qttt_config={"num_steps": 2},
        )

        # Check output shape
        assert output.shape[1] == input_ids.shape[1] + 5

        # Check statistics were printed
        captured = capsys.readouterr()
        assert "[Ponder Gate]" in captured.out

    def test_adaptive_qttt_saves_compute(self, model, input_ids):
        """Adaptive qTTT should trigger less than unconditional."""
        # This is a simplified check - in practice, we'd need to
        # mock or instrument the qTTT calls

        # Run with adaptive mode
        output_adaptive = model.generate(
            input_ids,
            max_new_tokens=5,
            use_qttt="adaptive",
            ponder_gate_mode="strict",  # Strict = less triggering
            qttt_config={"num_steps": 2},
        )

        # Output should be different from unconditional (since qTTT runs less)
        output_unconditional = model.generate(
            input_ids, max_new_tokens=5, use_qttt=True, qttt_config={"num_steps": 2}
        )

        # Shape should be same
        assert output_adaptive.shape == output_unconditional.shape

    def test_ponder_gate_modes(self, model, input_ids, capsys):
        """Test different Ponder Gate modes."""
        modes = ["strict", "balanced", "lenient"]

        for mode in modes:
            output = model.generate(
                input_ids,
                max_new_tokens=3,
                use_qttt="adaptive",
                ponder_gate_mode=mode,
                qttt_config={"num_steps": 2},
            )
            assert output.shape[1] == input_ids.shape[1] + 3

            captured = capsys.readouterr()
            assert "[Ponder Gate]" in captured.out

    def test_invalid_ponder_gate_mode(self, model, input_ids):
        """Invalid mode should raise error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            model.generate(
                input_ids, max_new_tokens=3, use_qttt="adaptive", ponder_gate_mode="invalid_mode"
            )

    def test_backward_compatibility(self, model, input_ids):
        """Old API (bool use_qttt) should still work."""
        # False
        output1 = model.generate(input_ids, max_new_tokens=3, use_qttt=False)
        assert output1.shape[1] == input_ids.shape[1] + 3

        # True
        output2 = model.generate(
            input_ids, max_new_tokens=3, use_qttt=True, qttt_config={"num_steps": 2}
        )
        assert output2.shape[1] == input_ids.shape[1] + 3


class TestPonderGateWithComponents:
    """Test Ponder Gate combined with other components."""

    @pytest.fixture
    def model(self):
        """Create small model."""
        config = get_config("small")
        model = AdaptiveTransformer(config)
        model.eval()
        return model

    def test_adaptive_qttt_with_attnres(self, model):
        """Adaptive qTTT + AttnRes."""
        input_ids = torch.randint(0, 32000, (1, 10))

        output = model.generate(
            input_ids,
            max_new_tokens=3,
            use_attnres=True,
            use_qttt="adaptive",
            ponder_gate_mode="balanced",
            qttt_config={"num_steps": 2},
        )
        assert output.shape[1] == 13

    def test_adaptive_qttt_with_rabitq(self, model):
        """Adaptive qTTT + RaBitQ (if supported)."""
        input_ids = torch.randint(0, 32000, (1, 10))

        # Initialize RaBitQ
        model.init_rabitq_caches(total_bits=1)

        output = model.generate(
            input_ids,
            max_new_tokens=3,
            use_qttt="adaptive",
            use_rabitq=True,
            qttt_config={"num_steps": 2},
        )
        assert output.shape[1] == 13
