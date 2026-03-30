"""
Unit tests for Attention Residuals (AttnRes) module.

Tests:
- RMSNorm functionality
- block_attn_res computation
- BlockAttnRes layer
"""

import pytest
import torch
import torch.nn as nn

from src.attnres.block_attnres import (
    RMSNorm, block_attn_res, BlockAttnRes
)


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
        rms = torch.sqrt(torch.mean(output ** 2))
        assert torch.allclose(rms, torch.tensor(1.0), atol=1e-5)
    
    def test_rmsnorm_learnable_weight(self):
        """Test that RMSNorm has learnable weight parameter."""
        dim = 128
        rmsnorm = RMSNorm(dim)
        
        assert hasattr(rmsnorm, 'weight')
        assert rmsnorm.weight.shape == (dim,)
        assert rmsnorm.weight.requires_grad
    
    def test_rmsnorm_different_dims(self):
        """Test RMSNorm with various dimensions."""
        for dim in [64, 128, 256, 512]:
            rmsnorm = RMSNorm(dim)
            x = torch.randn(4, 8, dim)
            output = rmsnorm(x)
            assert output.shape == x.shape


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
        blocks = [
            torch.ones(batch, seq_len, dim) * i
            for i in range(3)
        ]
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


class TestBlockAttnResLayer:
    """Tests for BlockAttnRes layer."""
    
    def test_block_attn_res_layer_init(self):
        """Test BlockAttnRes layer initialization."""
        dim = 128
        num_blocks = 8
        
        layer = BlockAttnRes(dim, num_blocks)
        
        assert layer.dim == dim
        assert layer.num_blocks == num_blocks
        assert hasattr(layer, 'pseudo_query_attn')
        assert hasattr(layer, 'pseudo_query_mlp')
    
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
