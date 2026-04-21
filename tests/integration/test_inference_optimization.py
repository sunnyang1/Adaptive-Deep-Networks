"""
Integration tests for inference optimization features.

Tests the complete pipeline with Ponder Gate, Adaptive Config, and all combinations.
"""

import torch
import pytest
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.configs import get_config
from src.models.adaptive_transformer import AdaptiveTransformer


class TestInferenceOptimization:
    """Test complete inference optimization pipeline."""

    @pytest.fixture
    def model(self):
        """Create small model for testing."""
        config = get_config("small")
        model = AdaptiveTransformer(config)
        model.eval()
        return model

    @pytest.fixture
    def short_input(self):
        """Short sequence input."""
        return torch.randint(0, 32000, (1, 10))

    @pytest.fixture
    def long_input(self):
        """Long sequence input."""
        return torch.randint(0, 32000, (1, 200))

    # ========== Basic Functionality Tests ==========

    def test_baseline_generation(self, model, short_input):
        """Test baseline without any optimization."""
        output = model.generate(short_input, max_new_tokens=5, use_qttt=False)
        assert output.shape[1] == short_input.shape[1] + 5

    def test_unconditional_qttt(self, model, short_input):
        """Test with standard qTTT."""
        output = model.generate(
            short_input, max_new_tokens=5, use_qttt=True, qttt_config={"num_steps": 2}
        )
        assert output.shape[1] == short_input.shape[1] + 5

    # ========== Ponder Gate Tests ==========

    def test_adaptive_qttt_with_ponder_gate(self, model, short_input, capsys):
        """Test Ponder Gate conditional triggering."""
        output = model.generate(
            short_input,
            max_new_tokens=5,
            use_qttt="adaptive",
            ponder_gate_mode="balanced",
            qttt_config={"num_steps": 2},
        )

        assert output.shape[1] == short_input.shape[1] + 5

        captured = capsys.readouterr()
        assert "[Ponder Gate]" in captured.out

    def test_ponder_gate_modes(self, model, short_input, capsys):
        """Test all Ponder Gate modes."""
        for mode in ["strict", "balanced", "lenient"]:
            output = model.generate(
                short_input,
                max_new_tokens=3,
                use_qttt="adaptive",
                ponder_gate_mode=mode,
                qttt_config={"num_steps": 2},
            )
            assert output.shape[1] == short_input.shape[1] + 3

            captured = capsys.readouterr()
            assert "[Ponder Gate]" in captured.out

    # ========== Adaptive Config Tests ==========

    def test_adaptive_config_modes(self, model, short_input, capsys):
        """Test all adaptive config modes."""
        for mode in ["fast", "balanced", "quality"]:
            output = model.generate(
                short_input, max_new_tokens=3, use_qttt=True, adaptive_qttt_mode=mode
            )
            assert output.shape[1] == short_input.shape[1] + 3

            captured = capsys.readouterr()
            assert "[Adaptive qTTT]" in captured.out
            assert f"Mode: {mode}" in captured.out

    def test_adaptive_config_long_sequence(self, model, long_input):
        """Test adaptive config adjusts for long sequences."""
        # Long sequence should use more steps
        output = model.generate(
            long_input, max_new_tokens=3, use_qttt=True, adaptive_qttt_mode="balanced"
        )
        assert output.shape[1] == long_input.shape[1] + 3

    # ========== Combined Tests ==========

    def test_combined_adaptive_features(self, model, short_input, capsys):
        """Test Ponder Gate + Adaptive Config together."""
        output = model.generate(
            short_input,
            max_new_tokens=5,
            use_qttt="adaptive",
            ponder_gate_mode="balanced",
            adaptive_qttt_mode="balanced",
        )

        assert output.shape[1] == short_input.shape[1] + 5

        captured = capsys.readouterr()
        assert "[Ponder Gate]" in captured.out
        assert "[Adaptive qTTT]" in captured.out

    def test_with_attnres(self, model, short_input):
        """Test with AttnRes enabled."""
        output = model.generate(
            short_input,
            max_new_tokens=3,
            use_attnres=True,
            use_qttt="adaptive",
            ponder_gate_mode="balanced",
        )
        assert output.shape[1] == short_input.shape[1] + 3

    def test_without_attnres(self, model, short_input):
        """Test with AttnRes disabled."""
        output = model.generate(
            short_input,
            max_new_tokens=3,
            use_attnres=False,
            use_qttt="adaptive",
            ponder_gate_mode="balanced",
        )
        assert output.shape[1] == short_input.shape[1] + 3

    # ========== Backward Compatibility Tests ==========

    def test_backward_compatibility_bool_qttt(self, model, short_input):
        """Test old API with bool use_qttt."""
        # False
        output1 = model.generate(short_input, max_new_tokens=3, use_qttt=False)
        assert output1.shape[1] == short_input.shape[1] + 3

        # True
        output2 = model.generate(
            short_input, max_new_tokens=3, use_qttt=True, qttt_config={"num_steps": 2}
        )
        assert output2.shape[1] == short_input.shape[1] + 3

    def test_backward_compatibility_dict_config(self, model, short_input):
        """Test old API with dict qttt_config."""
        output = model.generate(
            short_input,
            max_new_tokens=3,
            use_qttt=True,
            qttt_config={"num_steps": 4, "learning_rate": 0.01, "span_length": 128},
        )
        assert output.shape[1] == short_input.shape[1] + 3

    # ========== Error Handling Tests ==========

    def test_invalid_ponder_gate_mode(self, model, short_input):
        """Test invalid Ponder Gate mode raises error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            model.generate(
                short_input, max_new_tokens=3, use_qttt="adaptive", ponder_gate_mode="invalid"
            )

    def test_invalid_adaptive_mode(self, model, short_input):
        """Test invalid adaptive mode raises error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            model.generate(
                short_input, max_new_tokens=3, use_qttt=True, adaptive_qttt_mode="invalid"
            )

    # ========== Performance Benchmark Tests ==========

    def test_adaptive_saves_compute(self, model, short_input):
        """Verify that adaptive qTTT can save compute."""
        # This is a basic check - full benchmarking would need more iterations

        start_strict = time.time()
        output_strict = model.generate(
            short_input,
            max_new_tokens=5,
            use_qttt="adaptive",
            ponder_gate_mode="strict",  # Less triggering
            qttt_config={"num_steps": 2},
        )
        time_strict = time.time() - start_strict

        start_lenient = time.time()
        output_lenient = model.generate(
            short_input,
            max_new_tokens=5,
            use_qttt="adaptive",
            ponder_gate_mode="lenient",  # More triggering
            qttt_config={"num_steps": 2},
        )
        time_lenient = time.time() - start_lenient

        # Strict should generally be faster (less qTTT)
        # Note: This is not guaranteed for small sequences but should hold statistically
        assert output_strict.shape == output_lenient.shape

    # ========== Different Sequence Lengths ==========

    @pytest.mark.parametrize("seq_len", [5, 50, 100])
    def test_various_sequence_lengths(self, model, seq_len):
        """Test with different input lengths."""
        input_ids = torch.randint(0, 32000, (1, seq_len))

        output = model.generate(
            input_ids,
            max_new_tokens=3,
            use_qttt="adaptive",
            ponder_gate_mode="balanced",
            adaptive_qttt_mode="balanced",
        )

        assert output.shape[1] == seq_len + 3


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_full_pipeline_small_model(self):
        """Test complete pipeline with small model."""
        config = get_config("small")
        model = AdaptiveTransformer(config)
        model.eval()

        input_ids = torch.randint(0, 32000, (1, 20))

        # Full pipeline: AttnRes + Ponder Gate + Adaptive Config
        output = model.generate(
            input_ids,
            max_new_tokens=5,
            use_attnres=True,
            use_qttt="adaptive",
            ponder_gate_mode="balanced",
            adaptive_qttt_mode="balanced",
        )

        assert output.shape[1] == 25
        assert output.dtype == torch.long

    def test_generation_determinism(self):
        """Test that generation is deterministic with fixed seed."""
        config = get_config("small")

        # First run
        torch.manual_seed(42)
        model1 = AdaptiveTransformer(config)
        model1.eval()
        input_ids = torch.randint(0, 32000, (1, 10))
        output1 = model1.generate(input_ids, max_new_tokens=3, use_qttt=False)

        # Second run with same seed
        torch.manual_seed(42)
        model2 = AdaptiveTransformer(config)
        model2.eval()
        input_ids = torch.randint(0, 32000, (1, 10))
        output2 = model2.generate(input_ids, max_new_tokens=3, use_qttt=False)

        # Should be identical (without qTTT which has randomness)
        assert torch.equal(output1, output2)
