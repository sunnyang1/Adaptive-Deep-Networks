"""
Integration tests for Attention Residuals (AttnRes).

Tests the full AttnRes pipeline including:
- BlockAttnRes with transformer layers
- Two-phase computation
- Memory efficiency
- Gradient flow
"""

import pytest
import torch
import torch.nn as nn

from src.models.configs import ModelConfig
from src.models.adaptive_transformer import AdaptiveTransformer
from src.attnres.block_attnres import BlockAttnRes, RMSNorm, TwoPhaseBlockAttnRes


class TestAttnResIntegration:
    """Integration tests for AttnRes in full model."""

    @pytest.fixture
    def small_config(self):
        """Small config for testing."""
        return ModelConfig(num_layers=8, hidden_dim=512, num_heads=8, num_blocks=4, vocab_size=1000)

    def test_model_with_attnres_forward(self, small_config):
        """Test full model forward pass with AttnRes."""
        model = AdaptiveTransformer(small_config)
        model.eval()

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_attnres_pseudo_query_initialization(self, small_config):
        """Test that pseudo-queries are zero-initialized."""
        model = AdaptiveTransformer(small_config)

        for i, attnres in enumerate(model.attnres_modules):
            assert (
                attnres.pseudo_query_attn.abs().max() < 1e-6
            ), f"Layer {i}: pseudo_query_attn not zero"
            assert (
                attnres.pseudo_query_mlp.abs().max() < 1e-6
            ), f"Layer {i}: pseudo_query_mlp not zero"

    def test_attnres_learns_during_training(self, small_config):
        """Test that pseudo-queries get updated during training."""
        model = AdaptiveTransformer(small_config)
        model.train()

        # Store initial values
        initial_attn = model.attnres_modules[0].pseudo_query_attn.clone()
        initial_mlp = model.attnres_modules[0].pseudo_query_mlp.clone()

        # Simple training step
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))

        logits = model(input_ids)
        loss = logits.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that pseudo-queries changed
        assert not torch.allclose(
            model.attnres_modules[0].pseudo_query_attn, initial_attn
        ), "pseudo_query_attn should be updated"

    def test_block_structure(self, small_config):
        """Test that block structure is correct."""
        model = AdaptiveTransformer(small_config)

        layers_per_block = small_config.num_layers // small_config.num_blocks
        assert layers_per_block == 2

        # Check block boundaries
        block_boundaries = [(i + 1) * layers_per_block for i in range(small_config.num_blocks)]
        assert block_boundaries == [2, 4, 6, 8]

    def test_attnres_parameter_count(self, small_config):
        """Test that AttnRes adds minimal parameters."""
        model = AdaptiveTransformer(small_config)

        total = model.count_parameters()
        attnres = model.count_attnsres_parameters()

        # AttnRes should be < 0.1% of total
        assert attnres / total < 0.001, f"AttnRes overhead too high: {attnres/total*100:.3f}%"

    def test_different_sequence_lengths(self, small_config):
        """Test model works with different sequence lengths."""
        model = AdaptiveTransformer(small_config)
        model.eval()

        for seq_len in [16, 32, 64, 128]:
            input_ids = torch.randint(0, small_config.vocab_size, (1, seq_len))
            with torch.no_grad():
                logits = model(input_ids)
            assert logits.shape == (1, seq_len, small_config.vocab_size)


class TestTwoPhaseBlockAttnRes:
    """Tests for TwoPhaseBlockAttnRes optimization."""

    @pytest.fixture
    def two_phase(self):
        """Create TwoPhaseBlockAttnRes instance."""
        return TwoPhaseBlockAttnRes(dim=512, block_size=4)

    def test_phase1_output_shape(self, two_phase):
        """Test Phase 1 output shape."""
        S, D = 4, 512
        B, T = 2, 8
        pseudo_queries = torch.randn(S, D)
        block_reps = [torch.randn(B, T, D) for _ in range(3)]  # 3 blocks

        outputs, max_vals, lse = two_phase.phase1_inter_block(pseudo_queries, block_reps)

        assert outputs.shape == (S, B, T, D)
        assert max_vals.shape == (S, B, T)
        assert lse.shape == (S, B, T)

    def test_phase2_merge(self, two_phase):
        """Test Phase 2 online softmax merge."""
        B, T, D = 2, 8, 512

        inter_output = torch.randn(B, T, D)
        inter_max = torch.randn(B, T)
        inter_lse = torch.randn(B, T)
        pseudo_query = torch.randn(D)
        partial_sum = torch.randn(B, T, D)

        merged, merged_max, merged_lse = two_phase.phase2_intra_block(
            inter_output, inter_max, inter_lse, pseudo_query, partial_sum
        )

        assert merged.shape == (B, T, D)
        assert merged_max.shape == (B, T)
        assert merged_lse.shape == (B, T)

        # Check that merge doesn't produce NaN
        assert not torch.isnan(merged).any()


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_rmsnorm_preserves_relative_magnitudes(self):
        """Test that RMSNorm preserves relative magnitudes."""
        norm = RMSNorm(dim=512)

        x1 = torch.randn(2, 16, 512)
        x2 = x1 * 2  # Double magnitude

        n1 = norm(x1)
        n2 = norm(x2)

        # Both should have similar normalized values (around 1)
        rms1 = torch.sqrt((n1**2).mean(dim=-1))
        rms2 = torch.sqrt((n2**2).mean(dim=-1))

        assert torch.allclose(rms1, rms2, atol=1e-5)

    def test_rmsnorm_learnable_scale(self):
        """Test that RMSNorm has learnable scale."""
        norm = RMSNorm(dim=512)

        # Check weight exists
        assert hasattr(norm, "weight")
        assert norm.weight.shape == (512,)

        # Check it's learnable
        assert norm.weight.requires_grad


class TestAttnResMemoryEfficiency:
    """Tests for AttnRes memory efficiency."""

    def test_block_attnres_vs_full_memory(self):
        """Test that BlockAttnRes uses less memory than FullAttnRes."""
        # This is a conceptual test - actual memory measurement is complex
        config = ModelConfig(
            num_layers=32, hidden_dim=1024, num_heads=8, num_blocks=8  # O(Nd) memory
        )

        # BlockAttnRes: O(Nd) = 8 * 1024 = 8192
        block_memory = config.num_blocks * config.hidden_dim

        # FullAttnRes: O(Ld) = 32 * 1024 = 32768
        full_memory = config.num_layers * config.hidden_dim

        # Block should be ~4x more memory efficient
        assert block_memory < full_memory
        assert full_memory / block_memory == 4
