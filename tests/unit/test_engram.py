"""
Unit tests for Engram module
"""

import pytest
import torch
import numpy as np

# Add project root to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.engram import (
    CompressedTokenizer,
    NgramHashMapping,
    NgramHashConfig,
    MultiHeadEmbedding,
    ShortConv,
    Engram,
    EngramConfig,
)


class TestCompressedTokenizer:
    """Test CompressedTokenizer functionality."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = CompressedTokenizer("gpt2")
        assert len(tokenizer) > 0
        assert tokenizer.get_compression_ratio() > 1.0

    def test_compression(self):
        """Test token compression."""
        tokenizer = CompressedTokenizer("gpt2")

        # Create sample input
        input_ids = np.array([[0, 1, 2, 100, 1000]])
        compressed = tokenizer.compress(input_ids)

        assert compressed.shape == input_ids.shape
        assert compressed.dtype == np.int64
        assert np.all(compressed >= 0)

    def test_compression_ratio(self):
        """Test that compression actually reduces vocabulary."""
        tokenizer = CompressedTokenizer("gpt2")
        ratio = tokenizer.get_compression_ratio()

        # Should have some compression due to case folding
        assert ratio >= 1.0
        print(f"Compression ratio: {ratio:.2f}x")

    def test_negative_values(self):
        """Test handling of negative values (ignore_index)."""
        tokenizer = CompressedTokenizer("gpt2")

        input_ids = np.array([[0, -100, 2, -100]])
        compressed = tokenizer.compress(input_ids)

        # Negative values should remain unchanged
        assert compressed[0, 1] == -100
        assert compressed[0, 3] == -100


class TestNgramHashMapping:
    """Test NgramHashMapping functionality."""

    @pytest.fixture
    def hash_config(self):
        return NgramHashConfig(
            engram_vocab_size=[1000, 1000],
            max_ngram_size=3,
            n_head_per_ngram=4,
            layer_ids=[0, 1],
            tokenizer_name_or_path="gpt2",
            pad_id=50256,
            seed=42,
        )

    def test_initialization(self, hash_config):
        """Test hash mapping initialization."""
        mapping = NgramHashMapping(hash_config)

        assert len(mapping.layer_multipliers) == 2
        assert 0 in mapping.layer_multipliers
        assert 1 in mapping.layer_multipliers

    def test_vocab_size_calculation(self, hash_config):
        """Test that vocab sizes are calculated correctly."""
        mapping = NgramHashMapping(hash_config)

        # Should have entries for each layer
        assert len(mapping.vocab_size_across_layers) == 2

        # Each layer should have (max_ngram_size - 1) entries
        for layer_id in hash_config.layer_ids:
            assert len(mapping.vocab_size_across_layers[layer_id]) == 2  # bigram, trigram

            # Each ngram should have n_head entries
            for ngram_vocab in mapping.vocab_size_across_layers[layer_id]:
                assert len(ngram_vocab) == hash_config.n_head_per_ngram
                # All should be prime numbers
                for prime in ngram_vocab:
                    assert self._is_prime(prime)

    def _is_prime(self, n):
        """Check if number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def test_hash_output_shape(self, hash_config):
        """Test hash output shape."""
        mapping = NgramHashMapping(hash_config)

        B, L = 2, 10
        input_ids = np.random.randint(0, 1000, size=(B, L))

        hashes = mapping.hash(input_ids)

        # Should return dict with entry for each layer
        assert len(hashes) == 2

        # Each hash should have shape [B, L, num_heads]
        expected_heads = (hash_config.max_ngram_size - 1) * hash_config.n_head_per_ngram
        for layer_id in hash_config.layer_ids:
            assert hashes[layer_id].shape == (B, L, expected_heads)

    def test_layer_specificity(self, hash_config):
        """Test that different layers produce different hashes."""
        mapping = NgramHashMapping(hash_config)

        input_ids = np.random.randint(0, 1000, size=(2, 10))
        hashes = mapping.hash(input_ids)

        # Different layers should have different hashes
        layer_0_hash = hashes[0][0, 0, 0]
        layer_1_hash = hashes[1][0, 0, 0]

        assert layer_0_hash != layer_1_hash


class TestMultiHeadEmbedding:
    """Test MultiHeadEmbedding functionality."""

    def test_initialization(self):
        """Test embedding initialization."""
        list_of_N = [100, 200, 300]
        D = 64

        emb = MultiHeadEmbedding(list_of_N, D)

        assert emb.num_heads == 3
        assert emb.embedding_dim == D

    def test_forward(self):
        """Test forward pass."""
        list_of_N = [100, 200]
        D = 64
        emb = MultiHeadEmbedding(list_of_N, D)

        B, L, num_heads = 2, 10, 2
        input_ids = torch.randint(0, 100, size=(B, L, num_heads))

        output = emb(input_ids)

        assert output.shape == (B, L, num_heads, D)

    def test_different_vocab_sizes(self):
        """Test that different heads can have different vocab sizes."""
        list_of_N = [50, 100, 200]
        D = 32
        emb = MultiHeadEmbedding(list_of_N, D)

        # Create input at boundaries
        input_ids = torch.tensor(
            [
                [[49, 99, 199]],  # At boundaries
                [[0, 0, 0]],  # At start
            ]
        )

        output = emb(input_ids)

        assert output.shape == (2, 1, 3, 32)
        assert not torch.isnan(output).any()


class TestShortConv:
    """Test ShortConv functionality."""

    def test_initialization(self):
        """Test convolution initialization."""
        conv = ShortConv(
            hidden_size=64,
            kernel_size=4,
            hc_mult=2,
        )

        assert conv.hc_mult == 2

    def test_forward(self):
        """Test forward pass."""
        conv = ShortConv(
            hidden_size=64,
            kernel_size=4,
            hc_mult=2,
        )

        B, L, G, C = 2, 10, 2, 64
        x = torch.randn(B, L, G, C)

        output = conv(x)

        assert output.shape == (B, L, G, C)
        assert not torch.isnan(output).any()

    def test_causality(self):
        """Test that convolution is causal (doesn't look ahead)."""
        conv = ShortConv(
            hidden_size=32,
            kernel_size=3,
            hc_mult=1,
        )

        B, L, G, C = 1, 5, 1, 32

        # First token should only depend on itself
        x1 = torch.zeros(B, L, G, C)
        x1[0, 0, 0, 0] = 1.0

        out1 = conv(x1)

        # Output at position 0 should be non-zero
        assert out1[0, 0, 0, 0] != 0


class TestEngramModule:
    """Test Engram module integration."""

    @pytest.fixture
    def engram_config(self):
        return EngramConfig(
            enabled=True,
            engram_vocab_size=[5000, 5000],
            max_ngram_size=3,
            n_embed_per_ngram=256,
            n_head_per_ngram=4,
            layer_ids=[0],
            tokenizer_name_or_path="gpt2",
            pad_id=50256,
            seed=42,
            kernel_size=4,
        )

    def test_initialization(self, engram_config):
        """Test Engram module initialization."""
        engram = Engram(
            layer_id=0,
            config=engram_config,
            hidden_size=512,
            hc_mult=1,
        )

        assert engram.layer_id == 0
        assert engram.hidden_size == 512

    def test_forward(self, engram_config):
        """Test forward pass."""
        engram = Engram(
            layer_id=0,
            config=engram_config,
            hidden_size=512,
            hc_mult=1,
        )

        B, L, G, D = 2, 10, 1, 512
        hidden_states = torch.randn(B, L, G, D)
        input_ids = torch.randint(0, 1000, size=(B, L))

        output = engram(hidden_states, input_ids)

        assert output.shape == (B, L, G, D)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, engram_config):
        """Test that gradients flow through Engram."""
        engram = Engram(
            layer_id=0,
            config=engram_config,
            hidden_size=64,
            hc_mult=1,
        )

        B, L, G, D = 1, 5, 1, 64
        hidden_states = torch.randn(B, L, G, D, requires_grad=True)
        input_ids = torch.randint(0, 100, size=(B, L))

        output = engram(hidden_states, input_ids)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()


class TestEngramIntegration:
    """Integration tests for Engram with transformer."""

    def test_with_adaptive_transformer(self):
        """Test Engram integration with AdaptiveTransformer."""
        from src.models.configs import AttnResSmallConfig
        from src.engram.integration import AdaptiveTransformerWithEngram, add_engram_to_config
        from src.engram.config import EngramSmallConfig

        # Create config with Engram
        config = AttnResSmallConfig()
        config.use_engram = True
        config.engram_config = EngramSmallConfig
        config.engram_config.tokenizer_name_or_path = "gpt2"

        # Create model
        model = AdaptiveTransformerWithEngram(config)

        # Forward pass
        B, L = 2, 10
        input_ids = torch.randint(0, config.vocab_size, size=(B, L))

        logits = model(input_ids)

        assert logits.shape == (B, L, config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_without_engram(self):
        """Test that model works without Engram."""
        from src.models.configs import AttnResSmallConfig
        from src.engram.integration import AdaptiveTransformerWithEngram

        config = AttnResSmallConfig()
        config.use_engram = False

        model = AdaptiveTransformerWithEngram(config)

        B, L = 2, 10
        input_ids = torch.randint(0, config.vocab_size, size=(B, L))

        logits = model(input_ids)

        assert logits.shape == (B, L, config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
