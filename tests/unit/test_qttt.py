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

from src.qttt.adaptation import (
    QueryOnlyTTT,
    KVCache,
    qTTTConfig,
    compute_attention_with_query,
    qttt_adapt,
)
from src.qttt.polar_adaptation import PolarQTTT, PolarQTTTConfig, QueryAdaptationPolarAdapter
from src.qttt.margin_loss import MarginMaximizationLoss, compute_margin_loss, NeedleMarginLoss


class TestKVCache:
    """Tests for KV cache management."""

    def test_kvcache_init(self):
        """Test KVCache initialization."""
        batch_size = 2
        num_heads = 8
        seq_len = 10
        head_dim = 64

        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)

        cache = KVCache(keys, values)

        assert cache.keys.shape == (batch_size, num_heads, seq_len, head_dim)
        assert cache.values.shape == (batch_size, num_heads, seq_len, head_dim)
        assert len(cache) == seq_len

    def test_kvcache_frozen(self):
        """Test that KVCache returns frozen (non-grad) tensors."""
        batch_size = 4
        num_heads = 4
        seq_len = 8
        head_dim = 32

        keys = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)

        cache = KVCache(keys, values)

        k, v = cache.get_kv()

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

        keys.add_(1.0)

        cached_keys, _ = cache.get_kv()
        assert not torch.allclose(cached_keys, keys)


class TestQueryOnlyTTT:
    """Tests for Query-Only TTT adaptation."""

    def test_query_only_ttt_init(self):
        """Test QueryOnlyTTT initialization."""
        hidden_dim = 128
        num_heads = 8
        config = qTTTConfig(num_steps=4, learning_rate=0.01)

        ttt = QueryOnlyTTT(config, hidden_dim, num_heads)

        assert ttt.hidden_dim == hidden_dim
        assert ttt.num_heads == num_heads
        assert ttt.config.num_steps == 4
        assert ttt.config.learning_rate == 0.01

    def test_query_only_ttt_query_projection_init(self):
        """Test QueryOnlyTTT creates projection head for query_projection target type."""
        hidden_dim = 64
        num_heads = 4
        config = qTTTConfig(target_type="query_projection")

        ttt = QueryOnlyTTT(config, hidden_dim, num_heads)

        assert ttt.query_projection is not None
        assert isinstance(ttt.query_projection, nn.Linear)

    def test_adapt_pseudo_query(self):
        """Test pseudo-query adaptation returns adapted query."""
        hidden_dim = 32
        num_heads = 4
        head_dim = hidden_dim // num_heads
        config = qTTTConfig(num_steps=2, learning_rate=0.05)

        ttt = QueryOnlyTTT(config, hidden_dim, num_heads)

        pseudo_query = torch.randn(hidden_dim)

        keys = torch.randn(1, num_heads, 8, head_dim)
        values = torch.randn(1, num_heads, 8, head_dim)
        kv_cache = KVCache(keys, values)
        seq_pos = torch.tensor([0])

        adapted, losses = ttt.adapt_pseudo_query(pseudo_query, kv_cache, seq_pos)

        assert adapted.shape == (hidden_dim,)
        assert len(losses) == config.num_steps
        assert not torch.allclose(adapted, pseudo_query, atol=1e-3)

    def test_adapt_query_projection(self):
        """Test query projection adaptation."""
        batch_size = 2
        seq_len = 4
        hidden_dim = 64
        num_heads = 4
        head_dim = hidden_dim // num_heads
        config = qTTTConfig(num_steps=2, learning_rate=0.05)

        ttt = QueryOnlyTTT(config, hidden_dim, num_heads)

        queries = torch.randn(batch_size, seq_len, hidden_dim)
        keys = torch.randn(batch_size, num_heads, 8, head_dim)
        values = torch.randn(batch_size, num_heads, 8, head_dim)
        kv_cache = KVCache(keys, values)
        seq_pos = torch.tensor([0, 1, 2, 3])

        adapted, losses = ttt.adapt_query_projection(queries, kv_cache, seq_positions=seq_pos)

        assert adapted.shape == (batch_size, seq_len, hidden_dim)
        assert len(losses) == config.num_steps

    def test_compute_flops(self):
        """Test FLOP computation."""
        config = qTTTConfig(num_steps=8)
        ttt = QueryOnlyTTT(config, hidden_dim=64, num_heads=4)

        flops = ttt.compute_flops(batch_size=2, seq_len=128, span_len=64)

        assert flops["num_steps"] == 8
        assert flops["total"] > 0
        assert flops["per_step"] > 0


class TestQtttAdapt:
    """Tests for the low-level qttt_adapt function."""

    def test_qttt_adapt_basic(self):
        """Test basic qTTT adaptation without projection head."""
        batch_size = 1
        num_heads = 2
        seq_len = 4
        head_dim = 16

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        keys = torch.randn(batch_size, num_heads, 8, head_dim)
        values = torch.randn(batch_size, num_heads, 8, head_dim)
        kv_cache = KVCache(keys, values)
        seq_pos = torch.arange(seq_len)

        adapted, losses = qttt_adapt(query, kv_cache, seq_pos, num_steps=2, learning_rate=0.05)

        assert adapted.shape == query.shape
        assert len(losses) == 2
        assert not torch.allclose(adapted, query, atol=1e-3)

    def test_qttt_adapt_with_projection(self):
        """Test qTTT adaptation with projection head and target tokens."""
        batch_size = 1
        num_heads = 2
        k_len = 3
        head_dim = 16
        vocab_size = 50

        query = torch.randn(batch_size, num_heads, k_len, head_dim)
        keys = torch.randn(batch_size, num_heads, 8, head_dim)
        values = torch.randn(batch_size, num_heads, 8, head_dim)
        kv_cache = KVCache(keys, values)
        seq_pos = torch.arange(k_len)
        target_token_ids = torch.randint(0, vocab_size, (k_len,))
        projection_head = nn.Linear(head_dim, vocab_size)

        adapted, losses = qttt_adapt(
            query,
            kv_cache,
            seq_pos,
            num_steps=2,
            learning_rate=0.05,
            projection_head=projection_head,
            target_token_ids=target_token_ids,
        )

        assert adapted.shape == query.shape
        assert len(losses) == 2


class TestComputeAttentionWithQuery:
    """Tests for compute_attention_with_query helper."""

    def test_attention_output_shape(self):
        """Test attention output has expected shape."""
        batch_size = 2
        num_heads = 4
        k_len = 3
        seq_len = 10
        head_dim = 16

        query = torch.randn(batch_size, num_heads, k_len, head_dim)
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        kv_cache = KVCache(keys, values)

        output = compute_attention_with_query(query, kv_cache)

        assert output.shape == (batch_size, num_heads, k_len, head_dim)


class TestMarginMaximizationLoss:
    """Tests for Margin Maximization Loss."""

    def test_margin_loss_init(self):
        """Test MarginMaximizationLoss initialization."""
        loss_fn = MarginMaximizationLoss(temperature=1.0, hard_negative_weight=1.5)

        assert loss_fn.temperature == 1.0
        assert loss_fn.hard_negative_weight == 1.5

    def test_margin_loss_computation(self):
        """Test margin loss computation."""
        loss_fn = MarginMaximizationLoss(temperature=1.0)

        batch_size = 2
        seq_len = 5
        vocab_size = 100
        k = 3

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, k))

        loss, margins = loss_fn(logits, targets, return_margin=True)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert margins is not None
        assert margins.shape == (batch_size, k)

    def test_margin_loss_with_distractors(self):
        """Test margin loss with explicit distractor positions."""
        loss_fn = MarginMaximizationLoss(temperature=0.5, hard_negative_weight=2.0)

        batch_size = 1
        seq_len = 4
        vocab_size = 50
        k = 2

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, k))
        distractors = torch.randint(0, vocab_size, (batch_size, k))

        loss, margins = loss_fn(
            logits, targets, distractor_positions=distractors, return_margin=True
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert margins is not None

    def test_margin_loss_gradient(self):
        """Test that margin loss produces gradients through logits."""
        loss_fn = MarginMaximizationLoss()

        logits = torch.randn(2, 4, 50, requires_grad=True)
        targets = torch.randint(0, 50, (2, 3))

        loss, _ = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.allclose(logits.grad, torch.zeros_like(logits))


class TestComputeMarginLoss:
    """Tests for compute_margin_loss helper."""

    def test_compute_margin_loss_basic(self):
        """Test simplified margin loss."""
        batch_size = 2
        seq_len = 4
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        target_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = compute_margin_loss(logits, target_token_ids, vocab_size)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestNeedleMarginLoss:
    """Tests for NeedleMarginLoss."""

    def test_needle_margin_loss(self):
        """Test needle retrieval margin loss."""
        loss_fn = NeedleMarginLoss(temperature=1.0)

        batch_size = 2
        num_heads = 4
        context_len = 16

        attention_scores = torch.randn(batch_size, num_heads, 1, context_len)
        needle_position = 5

        loss, accuracy = loss_fn(attention_scores, needle_position, context_len)

        assert loss.shape == ()
        assert 0.0 <= accuracy.item() <= 1.0
        assert not torch.isnan(loss)


class TestPolarQTTT:
    """Tests for Polar-coordinate qTTT."""

    def test_polar_qttt_init(self):
        """Test PolarQTTT initialization."""
        config = PolarQTTTConfig(num_steps=4, learning_rate=0.01)
        ttt = PolarQTTT(config, hidden_dim=64, num_heads=4)

        assert ttt.hidden_dim == 64
        assert ttt.num_heads == 4
        assert ttt.config.adapt_direction is True
        assert ttt.config.adapt_magnitude is False

    def test_polar_qttt_adapt_pseudo_query(self):
        """Test polar pseudo-query adaptation."""
        hidden_dim = 32
        num_heads = 4
        head_dim = hidden_dim // num_heads
        config = PolarQTTTConfig(num_steps=2, learning_rate=0.05)

        ttt = PolarQTTT(config, hidden_dim=hidden_dim, num_heads=num_heads)

        magnitude = torch.tensor(1.0)
        direction = torch.randn(hidden_dim)
        direction = direction / direction.norm()

        keys = torch.randn(1, num_heads, 8, head_dim)
        values = torch.randn(1, num_heads, 8, head_dim)
        kv_cache = KVCache(keys, values)

        adapted_dir, losses = ttt.adapt_pseudo_query(
            magnitude, direction, kv_cache, seq_positions=torch.tensor([0])
        )

        assert adapted_dir.shape == (hidden_dim,)
        assert len(losses) == config.num_steps
        assert torch.isclose(adapted_dir.norm(), torch.tensor(1.0), atol=1e-4)

    def test_polar_qttt_margin_loss_with_projection(self):
        """Test PolarQTTT margin loss indexing with explicit target/distractor IDs."""
        hidden_dim = 32
        num_heads = 4
        head_dim = hidden_dim // num_heads
        config = PolarQTTTConfig(num_steps=2, learning_rate=0.05)

        ttt = PolarQTTT(config, hidden_dim=hidden_dim, num_heads=num_heads)

        magnitude = torch.tensor(1.0)
        direction = torch.randn(hidden_dim)
        direction = direction / direction.norm()

        keys = torch.randn(1, num_heads, 8, head_dim)
        values = torch.randn(1, num_heads, 8, head_dim)
        kv_cache = KVCache(keys, values)

        projection_head = nn.Linear(head_dim, 50)
        target_token_ids = torch.randint(0, 50, (1,))
        distractor_positions = torch.randint(0, 8, (3,))  # seq positions of distractors

        adapted_dir, losses = ttt.adapt_pseudo_query(
            magnitude,
            direction,
            kv_cache,
            seq_positions=torch.tensor([0]),
            distractor_positions=distractor_positions,
            projection_head=projection_head,
            target_token_ids=target_token_ids,
        )

        assert adapted_dir.shape == (hidden_dim,)
        assert len(losses) == config.num_steps
        assert not torch.isnan(torch.tensor(losses)).any()

    def test_query_adaptation_polar_adapter(self):
        """Test QueryAdaptationPolarAdapter spherical update."""
        config = PolarQTTTConfig(num_steps=4, learning_rate=0.1, use_spherical_sgd=True)

        magnitude = torch.tensor(2.0)
        direction = torch.tensor([1.0, 0.0, 0.0])

        adapter = QueryAdaptationPolarAdapter(magnitude, direction, config)

        query = adapter.get_query()
        assert torch.allclose(query, magnitude * direction, atol=1e-4)

        # Simulate a loss that pushes direction toward y-axis
        loss = -adapter.get_direction()[1]
        adapter.update(loss)

        new_query = adapter.get_query()
        assert torch.isclose(new_query.norm(), magnitude, atol=1e-4)
        assert not torch.allclose(new_query, query, atol=1e-3)
