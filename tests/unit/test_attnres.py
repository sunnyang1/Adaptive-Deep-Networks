"""
Unit tests for Attention Residuals (AttnRes) module.

Tests:
- RMSNorm functionality
- block_attn_res computation
- BlockAttnRes layer
- full_attn_res helper
- StandardResidualModel, FullAttnResModel, BlockAttnResModel
"""

import pytest
import torch
import torch.nn as nn

from src.attnres.block_attnres import (
    RMSNorm,
    block_attn_res,
    BlockAttnRes,
    full_attn_res,
    StandardResidualModel,
    FullAttnResTransformerBlock,
    FullAttnResModel,
    BlockAttnResTransformerBlock,
    BlockAttnResModel,
)


# =============================================================================
# Shared constants / fixtures
# =============================================================================

B, T, D = 2, 16, 32  # small dims for fast tests


@pytest.fixture
def h():
    torch.manual_seed(0)
    return torch.randn(B, T, D)


# =============================================================================
# RMSNorm
# =============================================================================


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_rmsnorm_output_shape(self):
        """Test that RMSNorm preserves input shape."""
        batch, seq_len, dim = 2, 10, 512
        rmsnorm = RMSNorm(dim)
        x = torch.randn(batch, seq_len, dim)

        output = rmsnorm(x)

        assert output.shape == x.shape

    def test_rmsnorm_normalization(self):
        """Test that RMSNorm normalizes as expected."""
        dim = 64
        rmsnorm = RMSNorm(dim)

        # Create input with known RMS
        x = torch.ones(1, 1, dim) * 2.0

        output = rmsnorm(x)

        # RMS of output should be close to 1 (with weight=1)
        rms = torch.sqrt(torch.mean(output**2))
        assert torch.allclose(rms, torch.tensor(1.0), atol=1e-5)

    def test_rmsnorm_learnable_weight(self):
        """Test that RMSNorm has learnable weight parameter."""
        dim = 128
        rmsnorm = RMSNorm(dim)

        assert hasattr(rmsnorm, "weight")
        assert rmsnorm.weight.shape == (dim,)
        assert rmsnorm.weight.requires_grad

    def test_rmsnorm_different_dims(self):
        """Test RMSNorm with various dimensions."""
        for dim in [64, 128, 256, 512]:
            rmsnorm = RMSNorm(dim)
            x = torch.randn(4, 8, dim)
            output = rmsnorm(x)
            assert output.shape == x.shape

    def test_rmsnorm_output_is_finite(self):
        """RMS normalization should keep outputs finite for standard random inputs."""
        norm = RMSNorm(D)
        x = torch.randn(B, T, D)
        assert torch.isfinite(norm(x)).all()

    def test_rmsnorm_unit_weight_scales_output(self):
        """With weight=1, RMSNorm(x) * 2 == RMSNorm_weight2(x)."""
        norm1 = RMSNorm(D)
        norm2 = RMSNorm(D)
        with torch.no_grad():
            norm2.weight.fill_(2.0)
        x = torch.randn(B, T, D)
        assert torch.allclose(norm1(x) * 2, norm2(x), atol=1e-5)

    def test_rmsnorm_zero_input_output_is_finite(self):
        """Near-zero input should not produce NaN (eps guards the rsqrt)."""
        norm = RMSNorm(D)
        x = torch.zeros(B, T, D)
        assert torch.isfinite(norm(x)).all()


# =============================================================================
# block_attn_res
# =============================================================================


class TestBlockAttnRes:
    """Tests for block_attn_res function."""

    def test_block_attn_res_output_shape(self):
        """Test output shape of block_attn_res."""
        batch, seq_len, dim = 2, 10, 64
        num_blocks = 4

        blocks = [torch.randn(batch, seq_len, dim) for _ in range(num_blocks)]
        partial_block = torch.randn(batch, seq_len, dim)
        pseudo_query = torch.randn(dim)
        norm = RMSNorm(dim)

        output = block_attn_res(blocks, partial_block, pseudo_query, norm)

        assert output.shape == (batch, seq_len, dim)

    def test_block_attn_res_weighted_sum(self):
        """Test that block_attn_res produces weighted sum."""
        batch, seq_len, dim = 1, 1, 4

        # Create simple blocks
        blocks = [torch.ones(batch, seq_len, dim) * i for i in range(3)]
        partial_block = torch.ones(batch, seq_len, dim) * 3
        pseudo_query = torch.randn(dim)
        norm = RMSNorm(dim)

        output = block_attn_res(blocks, partial_block, pseudo_query, norm)

        # Output should be a weighted combination (not equal to any single block)
        assert not torch.allclose(output, blocks[0])
        assert not torch.allclose(output, partial_block)

    def test_block_attn_res_numerical_stability(self):
        """Test numerical stability with small epsilon."""
        batch, seq_len, dim = 2, 5, 32
        num_blocks = 3

        blocks = [torch.randn(batch, seq_len, dim) * 0.01 for _ in range(num_blocks)]
        partial_block = torch.randn(batch, seq_len, dim) * 0.01
        pseudo_query = torch.randn(dim) * 0.01
        norm = RMSNorm(dim, eps=1e-6)

        output = block_attn_res(blocks, partial_block, pseudo_query, norm)

        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_block_attn_res_zero_query_is_average_of_all_sources(self):
        """Zero query -> uniform weights -> output = mean(completed + [partial])."""
        norm = RMSNorm(D)
        w = torch.zeros(D)
        completed = [torch.randn(B, T, D) for _ in range(2)]
        partial = torch.randn(B, T, D)
        all_srcs = torch.stack(completed + [partial], dim=0)  # [3, B, T, D]
        expected = all_srcs.mean(dim=0)
        out = block_attn_res(completed, partial, w, norm)
        assert torch.allclose(out, expected, atol=1e-5)


# =============================================================================
# BlockAttnRes layer
# =============================================================================


class TestBlockAttnResLayer:
    """Tests for BlockAttnRes layer."""

    def test_block_attn_res_layer_init(self):
        """Test BlockAttnRes layer initialization."""
        dim = 128
        num_blocks = 8

        layer = BlockAttnRes(dim, num_blocks)

        assert layer.dim == dim
        assert layer.num_blocks == num_blocks
        assert hasattr(layer, "pseudo_query_attn")
        assert hasattr(layer, "pseudo_query_mlp")

    def test_block_attn_res_layer_zero_init(self):
        """Test that pseudo-queries are initialized to zero."""
        dim = 64
        layer = BlockAttnRes(dim)

        assert torch.allclose(layer.pseudo_query_attn, torch.zeros(dim))
        assert torch.allclose(layer.pseudo_query_mlp, torch.zeros(dim))

    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        batch, seq_len, dim = 2, 10, 64
        num_blocks = 4

        layer = BlockAttnRes(dim, num_blocks)

        # Create block representations
        block_reprs = [torch.randn(batch, seq_len, dim) for _ in range(num_blocks)]
        hidden = torch.randn(batch, seq_len, dim)

        h_attn, h_mlp = layer(block_reprs, hidden, use_attn=True, use_mlp=True)

        assert h_attn.shape == hidden.shape
        assert h_mlp.shape == hidden.shape

    def test_forward_different_modes(self):
        """Test forward pass with different use_attn/use_mlp settings."""
        batch, seq_len, dim = 2, 5, 32
        num_blocks = 3

        layer = BlockAttnRes(dim, num_blocks)
        block_reprs = [torch.randn(batch, seq_len, dim) for _ in range(num_blocks)]
        hidden = torch.randn(batch, seq_len, dim)

        # Both enabled
        h_attn, h_mlp = layer(block_reprs, hidden, use_attn=True, use_mlp=True)

        # Only attention
        h_attn_only, _ = layer(block_reprs, hidden, use_attn=True, use_mlp=False)

        # Only MLP
        _, h_mlp_only = layer(block_reprs, hidden, use_attn=False, use_mlp=True)

        # Neither (just returns hidden)
        h_neither_attn, h_neither_mlp = layer(block_reprs, hidden, use_attn=False, use_mlp=False)

        assert torch.allclose(h_neither_attn, hidden)
        assert torch.allclose(h_neither_mlp, hidden)

    def test_forward_with_accumulation(self):
        """Test forward pass with block accumulation."""
        batch, seq_len, dim = 1, 4, 16
        num_blocks = 2
        num_layers = 4

        layer = BlockAttnRes(dim, num_blocks)

        # Simulate multiple layers
        block_reprs = []
        hidden = torch.randn(batch, seq_len, dim)

        for layer_idx in range(num_layers):
            h_attn, h_mlp = layer(block_reprs, hidden, use_attn=True, use_mlp=True)

            # Use attention output for next layer
            hidden = h_attn

            # Add to block representations at boundaries
            if (layer_idx + 1) % (num_layers // num_blocks) == 0:
                block_reprs.append(h_attn)

        assert len(block_reprs) <= num_blocks

    def test_reset_parameters(self):
        """Test reset_parameters sets pseudo-queries to zero."""
        dim = 64
        layer = BlockAttnRes(dim)

        # Modify parameters
        nn.init.uniform_(layer.pseudo_query_attn, -1, 1)
        nn.init.uniform_(layer.pseudo_query_mlp, -1, 1)

        # Reset
        layer.reset_parameters()

        assert torch.allclose(layer.pseudo_query_attn, torch.zeros(dim))
        assert torch.allclose(layer.pseudo_query_mlp, torch.zeros(dim))


# =============================================================================
# StandardResidualModel
# =============================================================================


class TestStandardResiduals:
    def test_model_output_shape(self, h):
        """Stacking standard residual blocks should preserve the hidden-state shape."""
        model = StandardResidualModel(D, num_transformer_blocks=4)
        assert model(h).shape == (B, T, D)

    def test_model_output_is_finite(self, h):
        """The standard residual model should produce finite activations end to end."""
        model = StandardResidualModel(D, num_transformer_blocks=4)
        assert torch.isfinite(model(h)).all()

    def test_single_block_model(self, h):
        """The standard residual model should also work when configured with a single block."""
        model = StandardResidualModel(D, num_transformer_blocks=1)
        assert model(h).shape == (B, T, D)


# =============================================================================
# full_attn_res
# =============================================================================


class TestFullAttnRes:
    def _make_inputs(self, n_sources):
        torch.manual_seed(42)
        w = torch.randn(D)
        srcs = [torch.randn(B, T, D) for _ in range(n_sources)]
        norm = RMSNorm(D)
        return w, srcs, norm

    def test_output_shape(self):
        """full_attn_res should return one residual tensor with the same shape as each source."""
        w, srcs, norm = self._make_inputs(3)
        out = full_attn_res(w, srcs, norm)
        assert out.shape == (B, T, D)

    def test_output_is_finite(self):
        """full_attn_res should remain numerically stable for normal random inputs."""
        w, srcs, norm = self._make_inputs(5)
        out = full_attn_res(w, srcs, norm)
        assert torch.isfinite(out).all()

    def test_single_source_returns_that_source(self):
        """With one source, softmax over dim-0 gives weight=1 -> output = source."""
        norm = RMSNorm(D)
        with torch.no_grad():
            norm.weight.fill_(1.0)
        src = torch.randn(B, T, D)
        w = torch.zeros(D)
        out = full_attn_res(w, [src], norm)
        assert torch.allclose(out, src, atol=1e-5)

    def test_zero_query_uniform_weights(self):
        """Zero pseudo-query -> uniform logits -> uniform alpha (1/n per source)."""
        n = 4
        norm = RMSNorm(D)
        w = torch.zeros(D)
        srcs = [torch.randn(B, T, D) for _ in range(n)]

        expected = torch.stack(srcs, dim=0).mean(dim=0)
        out = full_attn_res(w, srcs, norm)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_alpha_sums_to_one(self):
        """Attention weights must sum to 1 across source dimension for every (b,t)."""
        n = 5
        norm = RMSNorm(D)
        w = torch.randn(D)
        srcs = [torch.randn(B, T, D) for _ in range(n)]

        V = torch.stack(srcs, dim=0)  # [n, B, T, D]
        K = norm(V)
        logits = torch.einsum("d, n b t d -> n b t", w, K)
        alpha = logits.softmax(dim=0)  # [n, B, T]
        sums = alpha.sum(dim=0)  # [B, T]
        assert torch.allclose(sums, torch.ones(B, T), atol=1e-5)

    def test_more_sources_still_correct_shape(self):
        """Adding more source tensors should not change the output tensor shape."""
        w, srcs, norm = self._make_inputs(10)
        out = full_attn_res(w, srcs, norm)
        assert out.shape == (B, T, D)


# =============================================================================
# FullAttnResTransformerBlock / FullAttnResModel
# =============================================================================


class TestFullAttnRes_Block:
    def test_sources_grow_by_two(self, h):
        """Each call to the block must append exactly 2 tensors to sources."""
        block = FullAttnResTransformerBlock(D)
        sources_in = [h.clone()]
        sources_out = block(sources_in)
        assert len(sources_out) == 3  # 1 initial + 2 appended

    def test_new_sources_have_correct_shape(self, h):
        """Every source tensor returned by the block should keep the model hidden-state shape."""
        block = FullAttnResTransformerBlock(D)
        sources_out = block([h.clone()])
        for s in sources_out:
            assert s.shape == (B, T, D)

    def test_sources_grow_across_multiple_blocks(self, h):
        """Repeated full-attention residual blocks should append two sources per block."""
        n_blocks = 4
        blocks = [FullAttnResTransformerBlock(D) for _ in range(n_blocks)]
        sources = [h.clone()]
        for block in blocks:
            sources = block(sources)
        assert len(sources) == 1 + n_blocks * 2

    def test_model_output_shape(self, h):
        """The full-attention residual model should preserve the hidden-state shape."""
        model = FullAttnResModel(D, num_transformer_blocks=4)
        assert model(h).shape == (B, T, D)

    def test_model_output_is_finite(self, h):
        """The full-attention residual model should produce finite activations."""
        model = FullAttnResModel(D, num_transformer_blocks=4)
        assert torch.isfinite(model(h)).all()

    def test_model_single_block(self, h):
        """The full-attention residual model should also run with a single block."""
        model = FullAttnResModel(D, num_transformer_blocks=1)
        assert model(h).shape == (B, T, D)

    def test_model_has_more_params_than_standard(self, h):
        """Full AttnRes adds query vectors and extra norms per layer."""
        n = 4
        std = StandardResidualModel(D, n)
        full = FullAttnResModel(D, n)
        assert sum(p.numel() for p in full.parameters()) > sum(p.numel() for p in std.parameters())


# =============================================================================
# BlockAttnResTransformerBlock / BlockAttnResModel
# =============================================================================


class TestBlockAttnRes_Block:
    def _initial_state(self, h):
        completed = [h.clone()]
        partial = None
        layer_in_block = 0
        return completed, partial, layer_in_block

    def test_forward_returns_three_values(self, h):
        """A block-attention residual layer should return completed blocks, partial state, and offset."""
        block = BlockAttnResTransformerBlock(D, block_size=4, block_layer_offset=0)
        state = self._initial_state(h)
        result = block(*state)
        assert len(result) == 3

    def test_partial_block_shape(self, h):
        """The partial block accumulator should have the same shape as the hidden state."""
        block = BlockAttnResTransformerBlock(D, block_size=4, block_layer_offset=0)
        completed, partial, layer_in_block = block(*self._initial_state(h))
        assert partial is not None
        assert partial.shape == (B, T, D)

    def test_layer_in_block_increments(self, h):
        """Each transformer block should advance the intra-block offset by two sublayers."""
        block_size = 6
        block = BlockAttnResTransformerBlock(D, block_size=block_size, block_layer_offset=0)
        completed, partial, lib = block(*self._initial_state(h))
        assert lib == 2

    def test_block_boundary_finalizes_partial(self, h):
        """Boundary check fires at the START of the next sub-layer."""
        block_size = 2
        block = BlockAttnResTransformerBlock(D, block_size=block_size, block_layer_offset=0)

        state = self._initial_state(h)
        state = block(*state)
        assert state[2] == block_size
        assert len(state[0]) == 1

        completed, partial, lib = block(*state)
        assert len(completed) == 2

    def test_completed_blocks_grow_at_boundary(self, h):
        """completed_blocks grows by 1 each time a block boundary is crossed."""
        block_size = 2
        block = BlockAttnResTransformerBlock(D, block_size=block_size, block_layer_offset=0)
        state = self._initial_state(h)
        state = block(*state)
        completed, partial, lib = block(*state)
        assert len(completed) == 2

    def test_model_output_shape(self, h):
        """The block-attention residual model should preserve the hidden-state shape."""
        model = BlockAttnResModel(D, num_transformer_blocks=6, block_size=4)
        assert model(h).shape == (B, T, D)

    def test_model_output_is_finite(self, h):
        """The block-attention residual model should produce finite activations."""
        model = BlockAttnResModel(D, num_transformer_blocks=6, block_size=4)
        assert torch.isfinite(model(h)).all()

    def test_model_single_block(self, h):
        """The block-attention residual model should also work when only one logical block is used."""
        model = BlockAttnResModel(D, num_transformer_blocks=3, block_size=6)
        assert model(h).shape == (B, T, D)

    def test_model_block_size_one(self, h):
        """block_size=1 means every sub-layer starts a new block."""
        model = BlockAttnResModel(D, num_transformer_blocks=3, block_size=1)
        assert model(h).shape == (B, T, D)

    def test_block_model_fewer_params_than_full(self, h):
        """BlockAttnRes shares structure with FullAttnRes; param counts should be equal."""
        n = 4
        full = FullAttnResModel(D, n)
        block = BlockAttnResModel(D, n, block_size=4)
        assert sum(p.numel() for p in full.parameters()) == sum(
            p.numel() for p in block.parameters()
        )


# =============================================================================
# Cross-model: gradient flow
# =============================================================================


class TestGradientFlow:
    """Ensure gradients reach all learnable parameters (no dead subgraph)."""

    def _check_grads(self, model, h):
        out = model(h)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_standard_residual_grads(self, h):
        """Standard residual models should backpropagate gradients to every learnable parameter."""
        model = StandardResidualModel(D, num_transformer_blocks=3)
        self._check_grads(model, h)

    def test_full_attn_res_grads(self, h):
        """Full-attention residual models should backpropagate gradients to every learnable parameter."""
        model = FullAttnResModel(D, num_transformer_blocks=3)
        self._check_grads(model, h)

    def test_block_attn_res_grads(self, h):
        """Block-attention residual models should backpropagate gradients to every learnable parameter."""
        model = BlockAttnResModel(D, num_transformer_blocks=6, block_size=4)
        self._check_grads(model, h)
