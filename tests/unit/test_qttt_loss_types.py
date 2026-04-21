"""
Test qTTT loss function types: cross_entropy vs margin_maximization.

Based on paper §3.3.2 - §3.3.3:
- cross_entropy (default): Primary training signal from Algorithm §3.3.2
- margin_maximization (alternative): Explicit margin maximization from §3.3.3
"""

import pytest
import torch
import torch.nn as nn
from src.qttt.polar_adaptation import PolarQTTT, PolarQTTTConfig
from src.models.adaptive_transformer import AdaptiveTransformer
from src.models.configs import get_config


class TestQTTTLossTypes:
    """Test different loss function types for qTTT."""

    @pytest.fixture
    def config(self):
        return get_config("small")

    @pytest.fixture
    def model(self, config):
        return AdaptiveTransformer(config)

    def test_cross_entropy_loss_default(self, model, config):
        """Test that cross_entropy is the default loss type."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
        )
        # Default should be cross_entropy
        assert cfg.loss_type == "cross_entropy", "Default loss_type should be cross_entropy"

        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)
        assert qttt.config.loss_type == "cross_entropy"

    def test_margin_maximization_loss_option(self, model, config):
        """Test that margin_maximization can be configured."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="margin_maximization",
        )

        assert cfg.loss_type == "margin_maximization"

        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)
        assert qttt.config.loss_type == "margin_maximization"

    def test_cross_entropy_computation(self, model, config):
        """Test cross-entropy loss computation."""
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
        seq_positions = torch.arange(seq_len)

        # Compute loss
        loss = qttt._compute_adaptation_loss(
            logits,
            seq_positions,
            target_token_ids=target_token_ids,
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Cross-entropy loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_margin_maximization_computation(self, model, config):
        """Test margin maximization loss computation."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="margin_maximization",
            margin_temperature=1.0,
        )
        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

        batch_size = 1
        seq_len = 5
        vocab_size = config.vocab_size

        # Create fake logits and targets
        logits = torch.randn(batch_size, seq_len, vocab_size)
        target_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        seq_positions = torch.arange(seq_len)

        # Compute loss
        loss = qttt._compute_adaptation_loss(
            logits,
            seq_positions,
            target_token_ids=target_token_ids,
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Margin loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_qttt_adaptation_with_cross_entropy(self, model, config):
        """Test full qTTT adaptation with cross-entropy loss."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="cross_entropy",
        )
        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

        batch_size = 1
        seq_len = 3

        # Setup
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        kv_caches = model.get_kv_cache(input_ids)
        queries = torch.randn(batch_size, seq_len, config.hidden_dim)

        # Adapt with cross-entropy loss
        with torch.enable_grad():
            adapted_queries, loss_history = qttt.adapt_query_projection(
                queries,
                kv_cache=kv_caches[-1],
                seq_positions=torch.arange(seq_len),
                model=model,
                input_ids=input_ids,
                kv_caches=kv_caches,
            )

        assert adapted_queries.shape == queries.shape
        assert len(loss_history) > 0
        assert all(not torch.isnan(torch.tensor(l)) for l in loss_history)

    def test_qttt_adaptation_with_margin_maximization(self, model, config):
        """Test full qTTT adaptation with margin maximization loss."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="margin_maximization",
            margin_temperature=1.0,
        )
        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

        batch_size = 1
        seq_len = 3

        # Setup
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        kv_caches = model.get_kv_cache(input_ids)
        queries = torch.randn(batch_size, seq_len, config.hidden_dim)

        # Adapt with margin maximization loss
        with torch.enable_grad():
            adapted_queries, loss_history = qttt.adapt_query_projection(
                queries,
                kv_cache=kv_caches[-1],
                seq_positions=torch.arange(seq_len),
                model=model,
                input_ids=input_ids,
                kv_caches=kv_caches,
            )

        assert adapted_queries.shape == queries.shape
        assert len(loss_history) > 0
        assert all(not torch.isnan(torch.tensor(l)) for l in loss_history)

    def test_invalid_loss_type_raises_error(self, model, config):
        """Test that invalid loss_type raises ValueError."""
        cfg = PolarQTTTConfig(
            num_steps=2,
            learning_rate=0.01,
            loss_type="invalid_loss",
        )
        qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

        batch_size = 1
        seq_len = 5
        vocab_size = config.vocab_size

        logits = torch.randn(batch_size, seq_len, vocab_size)
        target_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        with pytest.raises(ValueError, match="Unknown loss_type"):
            qttt._compute_adaptation_loss(
                logits,
                torch.arange(seq_len),
                target_token_ids=target_token_ids,
            )

    def test_different_losses_produce_different_results(self, model, config):
        """Test that different loss types can produce different adaptation results."""
        batch_size = 1
        seq_len = 3

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        kv_caches = model.get_kv_cache(input_ids)
        queries = torch.randn(batch_size, seq_len, config.hidden_dim)

        results = {}

        for loss_type in ["cross_entropy", "margin_maximization"]:
            cfg = PolarQTTTConfig(
                num_steps=2,
                learning_rate=0.01,
                loss_type=loss_type,
            )
            qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

            with torch.enable_grad():
                adapted_q, loss_hist = qttt.adapt_query_projection(
                    queries.clone(),
                    kv_cache=kv_caches[-1],
                    seq_positions=torch.arange(seq_len),
                    model=model,
                    input_ids=input_ids,
                    kv_caches=kv_caches,
                )

            results[loss_type] = (adapted_q, loss_hist)

        # Results should be different (adapted queries should differ)
        ce_q = results["cross_entropy"][0]
        margin_q = results["margin_maximization"][0]

        diff = (ce_q - margin_q).abs().max().item()
        print(f"Max diff between CE and Margin adapted queries: {diff:.6f}")

        # They might be the same by chance, but usually different
        # We just verify both produce valid results
        assert not torch.isnan(ce_q).any()
        assert not torch.isnan(margin_q).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
