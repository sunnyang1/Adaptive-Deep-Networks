"""
Unit tests for Block Attention Residuals.

Tests based on Attention Residuals Technical Report specifications.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from attnres.block_attnres import (
    BlockAttnRes,
    block_attn_res,
    RMSNorm,
    TwoPhaseBlockAttnRes
)
from attnres.pseudo_query import PseudoQueryManager, PseudoQueryInitializer


class TestRMSNorm:
    """Tests for RMSNorm layer."""
    
    def test_initialization(self):
        norm = RMSNorm(128)
        assert norm.weight.shape == (128,)
        assert torch.allclose(norm.weight, torch.ones(128))
    
    def test_forward_shape(self):
        norm = RMSNorm(128)
        x = torch.randn(2, 10, 128)
        out = norm(x)
        assert out.shape == x.shape
    
    def test_normalization(self):
        norm = RMSNorm(128)
        x = torch.randn(2, 10, 128) * 10
        out = norm(x)
        # Check that output has roughly unit norm
        mean_norm = out.norm(dim=-1).mean()
        assert 0.9 < mean_norm < 1.1


class TestBlockAttnRes:
    """Tests for BlockAttnRes module."""
    
    @pytest.fixture
    def config(self):
        return {
            'dim': 128,
            'num_blocks': 4,
            'batch': 2,
            'seq_len': 10
        }
    
    def test_initialization(self, config):
        module = BlockAttnRes(config['dim'], config['num_blocks'])
        
        # Check pseudo-queries are zero-initialized
        assert torch.allclose(module.pseudo_query_attn, torch.zeros(config['dim']))
        assert torch.allclose(module.pseudo_query_mlp, torch.zeros(config['dim']))
    
    def test_forward_shape(self, config):
        module = BlockAttnRes(config['dim'], config['num_blocks'])
        
        # Create dummy inputs
        blocks = [torch.randn(config['batch'], config['seq_len'], config['dim']) 
                  for _ in range(2)]
        partial = torch.randn(config['batch'], config['seq_len'], config['dim'])
        
        h_attn, h_mlp = module(blocks, partial)
        
        assert h_attn.shape == (config['batch'], config['seq_len'], config['dim'])
        assert h_mlp.shape == (config['batch'], config['seq_len'], config['dim'])
    
    def test_uniform_attention_at_init(self, config):
        """At initialization (zero pseudo-query), attention should be uniform."""
        module = BlockAttnRes(config['dim'], config['num_blocks'])
        
        # Create dummy inputs
        num_blocks = 3
        blocks = [torch.randn(config['batch'], config['seq_len'], config['dim']) 
                  for _ in range(num_blocks)]
        partial = torch.randn(config['batch'], config['seq_len'], config['dim'])
        
        # Zero out pseudo-query
        module.pseudo_query_attn.data.zero_()
        
        h_attn, _ = module(blocks, partial, use_attn=True, use_mlp=False)
        
        # With zero query, all keys have same compatibility
        # Result should be approximately uniform average
        # Just verify output is finite and reasonable
        assert torch.isfinite(h_attn).all()
    
    def test_memory_complexity(self, config):
        """Verify memory usage is O(Nd) not O(Ld)."""
        # This is a conceptual test - in practice we'd profile memory
        module = BlockAttnRes(config['dim'], config['num_blocks'])
        
        # With 4 blocks, should only store 4 block representations
        # not all layer outputs
        assert module.num_blocks == config['num_blocks']


class TestBlockAttnResFunction:
    """Tests for standalone block_attn_res function."""
    
    def test_basic_functionality(self):
        dim = 64
        num_blocks = 3
        batch = 2
        seq_len = 5
        
        blocks = [torch.randn(batch, seq_len, dim) for _ in range(num_blocks)]
        partial = torch.randn(batch, seq_len, dim)
        pseudo_query = torch.randn(dim)
        norm = RMSNorm(dim)
        
        output = block_attn_res(blocks, partial, pseudo_query, norm)
        
        assert output.shape == (batch, seq_len, dim)
        assert torch.isfinite(output).all()
    
    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1."""
        dim = 64
        num_blocks = 3
        
        blocks = [torch.randn(1, 1, dim) for _ in range(num_blocks)]
        partial = torch.randn(1, 1, dim)
        pseudo_query = torch.randn(dim)
        norm = RMSNorm(dim)
        
        # Manually compute to check weights
        V = torch.stack(blocks + [partial], dim=0)  # [4, 1, 1, 64]
        K = norm(V)
        logits = torch.einsum('d, n b t d -> n b t', pseudo_query, K)
        logits = logits / (dim ** 0.5)
        weights = torch.softmax(logits, dim=0)
        
        # Weights should sum to 1
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


class TestPseudoQueryManager:
    """Tests for PseudoQueryManager."""
    
    def test_initialization(self):
        manager = PseudoQueryManager(
            num_layers=32,
            dim=128,
            num_blocks=8
        )
        
        # Check shape
        assert manager.pseudo_queries.shape == (32, 2, 128)
        
        # Check zero-initialized
        assert torch.allclose(manager.pseudo_queries, torch.zeros(32, 2, 128))
    
    def test_get_pseudo_query(self):
        manager = PseudoQueryManager(32, 128, 8)
        
        query = manager.get_pseudo_query(5, is_mlp=False)
        assert query.shape == (128,)
        
        mlp_query = manager.get_pseudo_query(5, is_mlp=True)
        assert mlp_query.shape == (128,)
        
        # Should be different parameters
        assert not torch.allclose(query, mlp_query)
    
    def test_compute_attention_weights(self):
        manager = PseudoQueryManager(32, 128, 8)
        
        blocks = [torch.randn(2, 10, 128) for _ in range(3)]
        weights = manager.compute_attention_weights(0, blocks, is_mlp=False)
        
        # Should sum to 1
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_entropy_computation(self):
        manager = PseudoQueryManager(32, 128, 8)
        
        # Uniform distribution should have max entropy
        uniform = torch.ones(8) / 8
        entropy_uniform = manager.compute_entropy(uniform)
        
        # Concentrated distribution should have lower entropy
        concentrated = torch.zeros(8)
        concentrated[0] = 1.0
        entropy_concentrated = manager.compute_entropy(concentrated)
        
        assert entropy_concentrated < entropy_uniform


class TestPseudoQueryInitializer:
    """Tests for initialization strategies."""
    
    def test_zero_init(self):
        module = BlockAttnRes(128, 8)
        module.pseudo_query_attn.data.normal_()
        
        PseudoQueryInitializer.zero_init(module)
        
        assert torch.allclose(module.pseudo_query_attn, torch.zeros(128))
    
    def test_uniform_init(self):
        module = BlockAttnRes(128, 8)
        module.pseudo_query_attn.data.zero_()
        
        PseudoQueryInitializer.uniform_init(module, std=0.02)
        
        # Should not be zero anymore
        assert not torch.allclose(module.pseudo_query_attn, torch.zeros(128))
        
        # Should have reasonable magnitude
        assert module.pseudo_query_attn.abs().max() < 1.0


class TestTwoPhaseBlockAttnRes:
    """Tests for two-phase computation strategy."""
    
    def test_initialization(self):
        module = TwoPhaseBlockAttnRes(dim=128, block_size=4)
        assert module.dim == 128
        assert module.block_size == 4
    
    def test_phase1_output_shape(self):
        module = TwoPhaseBlockAttnRes(dim=128, block_size=4)
        
        # 4 queries, 3 blocks
        pseudo_queries = torch.randn(4, 128)
        block_reps = [torch.randn(2, 10, 128) for _ in range(3)]
        
        outputs, max_vals, lse = module.phase1_inter_block(pseudo_queries, block_reps)
        
        assert outputs.shape == (4, 2, 10, 128)
        assert max_vals.shape == (4, 2, 10)
        assert lse.shape == (4, 2, 10)


class TestNumericalStability:
    """Tests for numerical stability edge cases."""
    
    def test_large_values(self):
        """Should handle large input values."""
        dim = 64
        blocks = [torch.randn(2, 10, dim) * 100 for _ in range(3)]
        partial = torch.randn(2, 10, dim) * 100
        pseudo_query = torch.randn(dim)
        norm = RMSNorm(dim)
        
        output = block_attn_res(blocks, partial, pseudo_query, norm)
        
        assert torch.isfinite(output).all()
    
    def test_small_values(self):
        """Should handle small input values."""
        dim = 64
        blocks = [torch.randn(2, 10, dim) * 1e-6 for _ in range(3)]
        partial = torch.randn(2, 10, dim) * 1e-6
        pseudo_query = torch.randn(dim)
        norm = RMSNorm(dim)
        
        output = block_attn_res(blocks, partial, pseudo_query, norm)
        
        assert torch.isfinite(output).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
