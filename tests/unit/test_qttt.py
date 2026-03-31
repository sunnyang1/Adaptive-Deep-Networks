"""
Unit tests for Query-Only Test-Time Training (qTTT) module.

Tests:
- KVCache management
- QueryOnlyTTT adaptation
- Margin maximization loss
"""

import pytest
import torch
import torch.nn as nn

from src.qttt.adaptation import QueryOnlyTTT, KVCache
from src.qttt.margin_loss import MarginMaximizationLoss


class TestKVCache:
    """Tests for KV cache management."""
    
    def test_kvcache_init(self):
        """Test KVCache initialization."""
        batch_size = 2
        num_heads = 8
        seq_len = 10
        head_dim = 64
        
        # Create sample keys and values
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        cache = KVCache(keys, values)
        
        assert cache.keys.shape == (batch_size, num_heads, seq_len, head_dim)
        assert cache.values.shape == (batch_size, num_heads, seq_len, head_dim)
        assert len(cache) == seq_len
    
    def test_kvcache_frozen(self):
        """Test that KVCache returns frozen (non-grad) tensors."""
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 32
        
        # Create tensors that require grad
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
        
        cache = KVCache(keys, values)
        
        k, v = cache.get_kv()
        
        # Retrieved tensors should not require grad
        assert not k.requires_grad
        assert not v.requires_grad
    
    def test_kvcache_detached(self):
        """Test that KVCache properly detaches tensors."""
        batch_size = 1
        num_heads = 2
        seq_len = 5
        head_dim = 16
        
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        cache = KVCache(keys, values)
        
        # Modify original tensors
        keys.add_(1.0)
        
        # Cache should still have original values (detached copy)
        cached_keys, _ = cache.get_kv()
        assert not torch.allclose(cached_keys, keys)


class TestQueryOnlyTTT:
    """Tests for Query-Only TTT adaptation."""
    
    def test_query_only_ttt_init(self):
        """Test QueryOnlyTTT initialization."""
        dim = 128
        learning_rate = 0.01
        
        ttt = QueryOnlyTTT(dim, learning_rate)
        
        assert ttt.dim == dim
        assert ttt.learning_rate == learning_rate
        assert ttt.query_adapt.shape == (dim,)
        assert ttt.query_adapt.requires_grad
    
    def test_query_only_ttt_zero_init(self):
        """Test that query_adapt is initialized to zero."""
        dim = 64
        ttt = QueryOnlyTTT(dim)
        
        assert torch.allclose(ttt.query_adapt, torch.zeros(dim))
    
    def test_adapt_returns_adapted_query(self):
        """Test that adapt returns adapted query."""
        dim = 32
        ttt = QueryOnlyTTT(dim)
        
        query = torch.randn(dim)
        adapted = ttt.adapt(query)
        
        # Adapted query should be different from original
        assert not torch.allclose(adapted, query)
        
        # Should be close to query (small perturbation)
        assert torch.allclose(adapted, query, atol=0.1)
    
    def test_adapt_gradient_flow(self):
        """Test that gradients flow through adaptation."""
        dim = 16
        ttt = QueryOnlyTTT(dim, learning_rate=0.1)
        
        query = torch.randn(dim)
        adapted = ttt.adapt(query)
        
        # Compute simple loss
        loss = adapted.sum()
        loss.backward()
        
        # query_adapt should have gradients
        assert ttt.query_adapt.grad is not None
        assert not torch.allclose(ttt.query_adapt.grad, torch.zeros(dim))
    
    def test_update_modifies_query_adapt(self):
        """Test that update modifies query_adapt."""
        dim = 32
        ttt = QueryOnlyTTT(dim, learning_rate=0.01)
        
        original = ttt.query_adapt.clone().detach()
        
        # Simulate update
        ttt.query_adapt.grad = torch.randn(dim)
        ttt.update()
        
        # Should have changed
        assert not torch.allclose(ttt.query_adapt, original)
    
    def test_get_stats(self):
        """Test get_stats returns expected metrics."""
        dim = 64
        ttt = QueryOnlyTTT(dim)
        
        stats = ttt.get_stats()
        
        assert 'norm' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert stats['norm'] >= 0


class TestMarginMaximizationLoss:
    """Tests for Margin Maximization Loss."""
    
    def test_margin_loss_init(self):
        """Test MarginMaximizationLoss initialization."""
        loss_fn = MarginMaximizationLoss(margin=0.5, temperature=1.0)
        
        assert loss_fn.margin == 0.5
        assert loss_fn.temperature == 1.0
    
    def test_margin_loss_computation(self):
        """Test margin loss computation."""
        loss_fn = MarginMaximizationLoss(margin=0.5, temperature=1.0)
        
        batch_size = 2
        seq_len = 5
        vocab_size = 100
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = loss_fn(logits, targets)
        
        # Loss should be a scalar
        assert loss.shape == ()
        assert loss.item() >= 0
    
    def test_margin_loss_with_target_positions(self):
        """Test margin loss with specific target positions."""
        loss_fn = MarginMaximizationLoss(margin=1.0, temperature=0.5)
        
        batch_size = 1
        seq_len = 3
        vocab_size = 50
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_positions = [0, 2]
        
        loss = loss_fn(logits, targets, target_positions)
        
        assert loss.shape == ()
        assert not torch.isnan(loss)
    
    def test_margin_loss_gradient(self):
        """Test that margin loss produces gradients."""
        loss_fn = MarginMaximizationLoss()
        
        logits = torch.randn(2, 4, 50, requires_grad=True)
        targets = torch.randint(0, 50, (2, 4))
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert not torch.allclose(logits.grad, torch.zeros_like(logits))
