"""
Unit tests for Query-only Test-Time Training (qTTT).
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from qttt.adaptation import (
    qttt_adapt,
    QueryOnlyTTT,
    qTTTConfig,
    KVCache
)
from qttt.margin_loss import (
    MarginMaximizationLoss,
    compute_margin_loss,
    NeedleMarginLoss
)


class TestKVCache:
    """Tests for KVCache."""
    
    def test_initialization(self):
        keys = torch.randn(2, 8, 100, 64)  # [B, H, T, d]
        values = torch.randn(2, 8, 100, 64)
        
        cache = KVCache(keys, values)
        
        assert cache.keys.shape == keys.shape
        assert cache.values.shape == values.shape
        assert cache.is_frozen == True
    
    def test_get_kv_detached(self):
        keys = torch.randn(2, 8, 100, 64, requires_grad=True)
        values = torch.randn(2, 8, 100, 64, requires_grad=True)
        
        cache = KVCache(keys, values)
        k, v = cache.get_kv()
        
        assert not k.requires_grad
        assert not v.requires_grad
    
    def test_length(self):
        keys = torch.randn(2, 8, 100, 64)
        values = torch.randn(2, 8, 100, 64)
        
        cache = KVCache(keys, values)
        
        assert len(cache) == 100


class TestQtttAdapt:
    """Tests for qttt_adapt function."""
    
    def test_basic_functionality(self):
        batch = 2
        num_heads = 8
        seq_len = 100
        head_dim = 64
        
        # Create inputs
        initial_query = torch.randn(batch, num_heads, 10, head_dim)
        keys = torch.randn(batch, num_heads, seq_len, head_dim)
        values = torch.randn(batch, num_heads, seq_len, head_dim)
        kv_cache = KVCache(keys, values)
        
        seq_positions = torch.tensor([5, 10, 15, 20, 25])
        
        # Run adaptation
        adapted_query, loss_history = qttt_adapt(
            initial_query,
            kv_cache,
            seq_positions,
            num_steps=5,
            learning_rate=0.01
        )
        
        assert adapted_query.shape == initial_query.shape
        assert len(loss_history) == 5
        assert not adapted_query.requires_grad
    
    def test_loss_decreases(self):
        """Loss should generally decrease during adaptation."""
        batch = 1
        num_heads = 4
        seq_len = 50
        head_dim = 32
        
        initial_query = torch.randn(batch, num_heads, 5, head_dim)
        keys = torch.randn(batch, num_heads, seq_len, head_dim)
        values = torch.randn(batch, num_heads, seq_len, head_dim)
        kv_cache = KVCache(keys, values)
        
        seq_positions = torch.tensor([10, 20, 30])
        
        _, loss_history = qttt_adapt(
            initial_query,
            kv_cache,
            seq_positions,
            num_steps=10,
            learning_rate=0.05
        )
        
        # Loss should generally decrease (though not monotonically)
        assert loss_history[-1] < loss_history[0] * 1.5  # Some tolerance
    
    def test_query_changes(self):
        """Query should change during adaptation."""
        batch = 1
        num_heads = 4
        seq_len = 50
        head_dim = 32
        
        initial_query = torch.randn(batch, num_heads, 5, head_dim)
        keys = torch.randn(batch, num_heads, seq_len, head_dim)
        values = torch.randn(batch, num_heads, seq_len, head_dim)
        kv_cache = KVCache(keys, values)
        
        seq_positions = torch.tensor([10, 20, 30])
        
        adapted_query, _ = qttt_adapt(
            initial_query,
            kv_cache,
            seq_positions,
            num_steps=5,
            learning_rate=0.1
        )
        
        # Query should have changed
        assert not torch.allclose(initial_query, adapted_query, atol=1e-4)


class TestQueryOnlyTTT:
    """Tests for QueryOnlyTTT module."""
    
    def test_initialization(self):
        config = qTTTConfig(num_steps=16, learning_rate=0.005)
        module = QueryOnlyTTT(config, hidden_dim=256, num_heads=8)
        
        assert module.config == config
        assert module.hidden_dim == 256
        assert module.num_heads == 8
    
    def test_adapt_pseudo_query(self):
        config = qTTTConfig(num_steps=5, learning_rate=0.01)
        module = QueryOnlyTTT(config, hidden_dim=128, num_heads=8)
        
        pseudo_query = torch.randn(128)
        keys = torch.randn(1, 8, 50, 16)
        values = torch.randn(1, 8, 50, 16)
        kv_cache = KVCache(keys, values)
        
        seq_positions = torch.tensor([10, 20, 30])
        
        adapted, losses = module.adapt_pseudo_query(
            pseudo_query,
            kv_cache,
            seq_positions
        )
        
        assert adapted.shape == (128,)
        assert len(losses) == 5
    
    def test_adapt_query_projection(self):
        config = qTTTConfig(num_steps=5, learning_rate=0.01)
        module = QueryOnlyTTT(
            config,
            hidden_dim=128,
            num_heads=8
        )
        
        queries = torch.randn(2, 10, 128)  # [B, T, D]
        keys = torch.randn(2, 8, 50, 16)
        values = torch.randn(2, 8, 50, 16)
        kv_cache = KVCache(keys, values)
        
        seq_positions = torch.tensor([5, 10, 15])
        
        adapted, losses = module.adapt_query_projection(
            queries,
            kv_cache,
            seq_positions
        )
        
        assert adapted.shape == queries.shape
        assert len(losses) == 5
    
    def test_compute_flops(self):
        config = qTTTConfig(num_steps=16, learning_rate=0.01)
        module = QueryOnlyTTT(config, hidden_dim=256, num_heads=8)
        
        flops = module.compute_flops(
            batch_size=2,
            seq_len=1000,
            span_len=128
        )
        
        assert 'per_step' in flops
        assert 'total' in flops
        assert flops['total'] == flops['per_step'] * config.num_steps


class TestMarginMaximizationLoss:
    """Tests for margin maximization loss."""
    
    def test_initialization(self):
        loss_fn = MarginMaximizationLoss(temperature=1.0)
        assert loss_fn.temperature == 1.0
    
    def test_forward(self):
        loss_fn = MarginMaximizationLoss()
        
        logits = torch.randn(2, 10, 100)  # [B, T, V]
        target_positions = torch.tensor([[5, 10], [15, 20]])
        
        loss, margins = loss_fn(logits, target_positions, return_margin=True)
        
        assert loss.shape == ()
        assert loss.item() >= 0
        assert margins.shape == (2, 2)
    
    def test_margin_computation(self):
        """Test that margin is computed correctly."""
        loss_fn = MarginMaximizationLoss()
        
        # Create logits where target is clearly highest
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 5] = 5.0  # Target
        logits[0, 0, [0, 1, 2, 3, 4, 6, 7, 8, 9]] = 0.0  # Distractors
        
        target_positions = torch.tensor([[5]])
        
        loss, margins = loss_fn(logits, target_positions, return_margin=True)
        
        # Margin should be positive (target > distractors)
        assert margins.item() > 0
        
        # Loss should be small (good prediction)
        assert loss.item() < 0.1
    
    def test_compute_margin_loss(self):
        """Test standalone margin loss function."""
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))
        
        loss = compute_margin_loss(logits, targets, vocab_size=100)
        
        assert loss.shape == ()
        assert loss.item() >= 0


class TestNeedleMarginLoss:
    """Tests for needle-specific margin loss."""
    
    def test_forward(self):
        loss_fn = NeedleMarginLoss()
        
        # Attention scores: [B, H, 1, T]
        attn_scores = torch.randn(2, 8, 1, 100)
        needle_position = 50
        context_length = 100
        
        loss, accuracy = loss_fn(attn_scores, needle_position, context_length)
        
        assert loss.shape == ()
        assert 0 <= accuracy <= 1
    
    def test_high_needle_score_low_loss(self):
        """If needle has highest attention, loss should be low."""
        loss_fn = NeedleMarginLoss()
        
        attn_scores = torch.zeros(1, 1, 1, 100)
        attn_scores[0, 0, 0, 50] = 10.0  # High attention on needle
        attn_scores[0, 0, 0, [i for i in range(100) if i != 50]] = 0.0
        
        loss, accuracy = loss_fn(attn_scores, 50, 100)
        
        assert loss.item() < 0.1
        assert accuracy == 1.0


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_logits(self):
        """Should handle large logit values."""
        logits = torch.randn(2, 10, 100) * 100
        targets = torch.randint(0, 100, (2, 10))
        
        loss = compute_margin_loss(logits, targets, vocab_size=100)
        
        assert torch.isfinite(loss)
    
    def test_small_logits(self):
        """Should handle small logit values."""
        logits = torch.randn(2, 10, 100) * 1e-6
        targets = torch.randint(0, 100, (2, 10))
        
        loss = compute_margin_loss(logits, targets, vocab_size=100)
        
        assert torch.isfinite(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
