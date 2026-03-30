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
        
        cache = KVCache(batch_size, num_heads, seq_len, head_dim)
        
        assert cache.k.shape == (batch_size, num_heads, seq_len, head_dim)
        assert cache.v.shape == (batch_size, num_heads, seq_len, head_dim)
        assert cache.current_length == 0
    
    def test_kvcache_update(self):
        """Test KVCache update mechanism."""
        batch_size = 1
        num_heads = 4
        seq_len = 8
        head_dim = 32
        
        cache = KVCache(batch_size, num_heads, seq_len, head_dim)
        
        # Initial state
        k, v = cache.get_kv()
        assert k.shape[1] == 0  # No tokens yet
        
        # Update with new keys and values
        new_k = torch.randn(batch_size, num_heads, 3, head_dim)
        new_v = torch.randn(batch_size, num_heads, 3, head_dim)
        cache.update(new_k, new_v)
        
        k, v = cache.get_kv()
        assert k.shape[2] == 3
        assert v.shape[2] == 3
        assert cache.current_length == 3
    
    def test_kvcache_frozen(self):
        """Test that KVCache returns frozen (non-grad) tensors."""
        batch_size = 2
        num_heads = 4
        seq_len = 6
        head_dim = 64
        
        cache = KVCache(batch_size, num_heads, seq_len, head_dim)
        
        # Update with tensors that require grad
        new_k = torch.randn(batch_size, num_heads, 3, head_dim, requires_grad=True)
        new_v = torch.randn(batch_size, num_heads, 3, head_dim, requires_grad=True)
        cache.update(new_k, new_v)
        
        k, v = cache.get_kv()
        
        # Retrieved tensors should not require grad
        assert not k.requires_grad
        assert not v.requires_grad
    
    def test_kvcache_sequence_length_tracking(self):
        """Test sequence length tracking."""
        batch_size = 1
        num_heads = 2
        seq_len = 10
        head_dim = 16
        
        cache = KVCache(batch_size, num_heads, seq_len, head_dim)
        
        # Multiple updates
        for i in range(1, 4):
            new_k = torch.randn(batch_size, num_heads, 2, head_dim)
            new_v = torch.randn(batch_size, num_heads, 2, head_dim)
            cache.update(new_k, new_v)
            
            assert cache.current_length == i * 2


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
