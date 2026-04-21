"""
Test qTTT self-supervised target handling in generation.

Based on paper §3.3.2: The algorithm uses cross_entropy(logits, context.targets)
which requires target tokens for self-supervised learning.
"""

import pytest
import torch
from src.qttt.polar_adaptation import PolarQTTT, PolarQTTTConfig
from src.models.adaptive_transformer import AdaptiveTransformer
from src.models.configs import get_config


class TestSelfSupervisedTarget:
    """Test self-supervised target handling for qTTT."""

    @pytest.fixture
    def config(self):
        return get_config("small")

    @pytest.fixture
    def model(self, config):
        return AdaptiveTransformer(config)

    def test_adaptation_loss_with_target(self, model, config):
        """Test that adaptation loss works with target_token_ids."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="cross_entropy",
        )
        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

        batch_size = 1
        seq_len = 5
        vocab_size = config.vocab_size

        # Create fake logits and targets
        logits = torch.randn(batch_size, seq_len, vocab_size)
        target_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute loss with targets
        loss = qttt._compute_adaptation_loss(
            logits,
            torch.arange(seq_len),
            target_token_ids=target_token_ids,
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Cross-entropy loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_adaptation_loss_without_target(self, model, config):
        """Test that adaptation loss falls back when no targets provided."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="cross_entropy",
        )
        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

        batch_size = 1
        seq_len = 5
        vocab_size = config.vocab_size

        # Create fake logits without targets
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Compute loss without targets (should use fallback)
        loss = qttt._compute_adaptation_loss(
            logits,
            torch.arange(seq_len),
            target_token_ids=None,
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_adaptation_loss_shape_mismatch(self, model, config):
        """Test handling of logits [B, T, V] vs target [B, 1] shape mismatch."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="cross_entropy",
        )
        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

        batch_size = 1
        seq_len = 10
        vocab_size = config.vocab_size

        # Create logits for full sequence but target only for last position
        logits = torch.randn(batch_size, seq_len, vocab_size)
        target_token_ids = torch.randint(0, vocab_size, (batch_size, 1))  # Only 1 target!

        # This should work - uses last position logits
        loss = qttt._compute_adaptation_loss(
            logits,
            torch.arange(seq_len),
            target_token_ids=target_token_ids,
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_full_adaptation_with_target_in_generate(self, model, config):
        """Test full qTTT adaptation with target in generation context."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="cross_entropy",
        )
        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

        batch_size = 1
        seq_len = 5

        # Setup
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        kv_caches = model.get_kv_cache(input_ids)
        queries = torch.randn(batch_size, seq_len, config.hidden_dim)

        # Create target (last token)
        target_token_ids = input_ids[:, -1:]

        # Adapt with target
        with torch.enable_grad():
            adapted_queries, loss_history = qttt.adapt_query_projection(
                queries,
                kv_cache=kv_caches[-1],
                seq_positions=torch.arange(seq_len),
                target_token_ids=target_token_ids,  # Provide target!
                model=model,
                input_ids=input_ids,
                kv_caches=kv_caches,
            )

        assert adapted_queries.shape == queries.shape
        assert len(loss_history) > 0
        assert all(not torch.isnan(torch.tensor(l)) for l in loss_history)
        assert all(l > 0 for l in loss_history), "Cross-entropy loss should be positive"

    def test_margin_loss_with_shape_mismatch(self, model, config):
        """Test margin loss handling of shape mismatch."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="margin_maximization",
            margin_temperature=1.0,
        )
        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

        batch_size = 1
        seq_len = 10
        vocab_size = config.vocab_size

        # Create logits for full sequence but target only for last position
        logits = torch.randn(batch_size, seq_len, vocab_size)
        target_token_ids = torch.randint(0, vocab_size, (batch_size, 1))

        # This should work for margin loss too
        loss = qttt._compute_adaptation_loss(
            logits,
            torch.arange(seq_len),
            target_token_ids=target_token_ids,
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Margin loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
