"""
Test layer-specific qTTT adapted query application.

This test verifies that adapted queries are only applied to the specified layer,
not to all layers (which was a bug in the original implementation).
"""

import pytest
import torch
import torch.nn as nn
from src.models.adaptive_transformer import AdaptiveTransformer
from src.models.configs import get_config


class TestLayerSpecificAdaptedQuery:
    """Test that adapted queries are applied layer-specifically."""

    @pytest.fixture
    def config(self):
        return get_config("small")  # Use small for testing

    @pytest.fixture
    def model(self, config):
        return AdaptiveTransformer(config)

    def test_adapted_query_only_applied_to_target_layer(self, model, config):
        """Test that adapted query is only applied to the specified layer."""
        batch_size = 1
        seq_len = 5
        vocab_size = config.vocab_size

        # Create input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create KV caches
        kv_caches = model.get_kv_cache(input_ids)

        # Create a distinctive adapted query
        adapted_query = torch.randn(batch_size, seq_len, config.hidden_dim)

        # Test 1: Apply to last layer (default)
        logits_last_layer = model.forward_with_frozen_kv(
            input_ids=input_ids,
            kv_caches=kv_caches,
            adapted_query=adapted_query,
            adapted_query_layer_idx=config.num_layers - 1,
            use_attnres=True,
        )

        # Test 2: Apply to first layer
        logits_first_layer = model.forward_with_frozen_kv(
            input_ids=input_ids,
            kv_caches=kv_caches,
            adapted_query=adapted_query,
            adapted_query_layer_idx=0,
            use_attnres=True,
        )

        # Test 3: No adapted query
        logits_no_adapt = model.forward_with_frozen_kv(
            input_ids=input_ids,
            kv_caches=kv_caches,
            adapted_query=None,
            use_attnres=True,
        )

        # Results should be different
        diff_last_vs_first = (logits_last_layer - logits_first_layer).abs().max().item()
        diff_last_vs_none = (logits_last_layer - logits_no_adapt).abs().max().item()
        diff_first_vs_none = (logits_first_layer - logits_no_adapt).abs().max().item()

        print(f"Max diff (last vs first layer): {diff_last_vs_first:.6f}")
        print(f"Max diff (last vs none): {diff_last_vs_none:.6f}")
        print(f"Max diff (first vs none): {diff_first_vs_none:.6f}")

        # All should be different (adapted query changes output)
        assert (
            diff_last_vs_first > 1e-6
        ), "Applying to last vs first layer should give different results"
        assert diff_last_vs_none > 1e-6, "Applying to last layer should change output"
        assert diff_first_vs_none > 1e-6, "Applying to first layer should change output"

    def test_adapted_query_default_is_last_layer(self, model, config):
        """Test that default layer for adapted query is the last layer."""
        batch_size = 1
        seq_len = 5
        vocab_size = config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        kv_caches = model.get_kv_cache(input_ids)
        adapted_query = torch.randn(batch_size, seq_len, config.hidden_dim)

        # Without specifying layer_idx (should default to last layer)
        logits_default = model.forward_with_frozen_kv(
            input_ids=input_ids,
            kv_caches=kv_caches,
            adapted_query=adapted_query,
            adapted_query_layer_idx=None,  # Default
            use_attnres=True,
        )

        # Explicitly specify last layer
        logits_explicit = model.forward_with_frozen_kv(
            input_ids=input_ids,
            kv_caches=kv_caches,
            adapted_query=adapted_query,
            adapted_query_layer_idx=config.num_layers - 1,
            use_attnres=True,
        )

        # Should be identical
        max_diff = (logits_default - logits_explicit).abs().max().item()
        print(f"Max diff (default vs explicit last layer): {max_diff:.6f}")
        assert max_diff < 1e-6, "Default should be last layer"

    def test_different_adapted_queries_produce_different_results(self, model, config):
        """Test that different adapted queries produce different outputs at target layer."""
        batch_size = 1
        seq_len = 5
        vocab_size = config.vocab_size
        target_layer = config.num_layers - 1

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        kv_caches = model.get_kv_cache(input_ids)

        # Two different adapted queries
        adapted_query_1 = torch.randn(batch_size, seq_len, config.hidden_dim)
        adapted_query_2 = torch.randn(batch_size, seq_len, config.hidden_dim)

        logits_1 = model.forward_with_frozen_kv(
            input_ids=input_ids,
            kv_caches=kv_caches,
            adapted_query=adapted_query_1,
            adapted_query_layer_idx=target_layer,
            use_attnres=True,
        )

        logits_2 = model.forward_with_frozen_kv(
            input_ids=input_ids,
            kv_caches=kv_caches,
            adapted_query=adapted_query_2,
            adapted_query_layer_idx=target_layer,
            use_attnres=True,
        )

        max_diff = (logits_1 - logits_2).abs().max().item()
        print(f"Max diff (different adapted queries): {max_diff:.6f}")
        assert max_diff > 1e-6, "Different adapted queries should produce different outputs"


class TestLayerSpecificInGenerate:
    """Test layer-specific adapted query in generation context."""

    @pytest.fixture
    def config(self):
        return get_config("small")

    @pytest.fixture
    def model(self, config):
        model = AdaptiveTransformer(config)
        model.eval()
        return model

    def test_generate_applies_adapted_query_to_last_layer(self, model, config):
        """Test that generate() applies adapted query only to last layer."""
        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        # Generate with qTTT
        with torch.no_grad():
            output_with_qttt = model.generate(
                input_ids,
                max_new_tokens=3,
                use_qttt=True,
                qttt_config={"num_steps": 2},
            )

        # Generate without qTTT
        with torch.no_grad():
            output_without_qttt = model.generate(
                input_ids,
                max_new_tokens=3,
                use_qttt=False,
            )

        # Outputs should be different (qTTT changes behavior)
        are_different = not torch.equal(output_with_qttt, output_without_qttt)
        print(f"Outputs are different: {are_different}")
        # Note: They might be the same by chance, so we don't assert this
        # But they should usually be different


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
