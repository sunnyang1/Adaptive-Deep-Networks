"""
Unit tests for Models module.

Tests:
- ModelConfig
- AdaptiveAttention
- AdaptiveMLP
- AdaptiveTransformer
"""

import pytest
import torch
import torch.nn as nn

from src.models.configs import ModelConfig
from src.models.adaptive_transformer import (
    AdaptiveAttention, AdaptiveMLP, AdaptiveLayer, AdaptiveTransformer
)
from src.qttt.adaptation import KVCache


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.num_layers == 32
        assert config.hidden_dim == 4096
        assert config.num_heads == 32
        assert config.vocab_size == 32000
        assert config.head_dim == 128  # 4096 / 32
    
    def test_head_dim_calculation(self):
        """Test head_dim is calculated correctly."""
        config = ModelConfig(hidden_dim=512, num_heads=8)
        
        assert config.head_dim == 64  # 512 / 8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            num_layers=12,
            hidden_dim=768,
            num_heads=12,
            vocab_size=10000
        )
        
        assert config.num_layers == 12
        assert config.hidden_dim == 768
        assert config.num_heads == 12
        assert config.vocab_size == 10000


class TestAdaptiveAttention:
    """Tests for AdaptiveAttention."""
    
    def test_adaptive_attention_init(self):
        """Test AdaptiveAttention initialization."""
        config = ModelConfig(hidden_dim=128, num_heads=4)
        attn = AdaptiveAttention(config)
        
        assert attn.head_dim == 32  # 128 / 4
        assert isinstance(attn.q_proj, nn.Linear)
        assert isinstance(attn.k_proj, nn.Linear)
        assert isinstance(attn.v_proj, nn.Linear)
        assert isinstance(attn.o_proj, nn.Linear)
    
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        config = ModelConfig(hidden_dim=128, num_heads=4)
        
        attn = AdaptiveAttention(config)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        output = attn(hidden_states)
        
        assert output.shape == hidden_states.shape
    
    def test_forward_with_kv_cache(self):
        """Test forward pass with KV cache."""
        batch_size = 1
        seq_len = 5
        config = ModelConfig(hidden_dim=64, num_heads=4)
        
        attn = AdaptiveAttention(config)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        # Create KV cache
        kv_cache = KVCache(batch_size, config.num_heads, seq_len * 2, config.head_dim)
        
        # First forward pass
        output1 = attn(hidden_states)
        
        # Forward with cache
        output2 = attn(hidden_states, kv_cache=kv_cache)
        
        assert output2.shape == hidden_states.shape
    
    def test_forward_with_adapted_query(self):
        """Test forward pass with adapted query."""
        batch_size = 2
        seq_len = 5
        config = ModelConfig(hidden_dim=64, num_heads=4)
        
        attn = AdaptiveAttention(config)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        adapted_query = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        output = attn(hidden_states, adapted_query=adapted_query)
        
        assert output.shape == hidden_states.shape


class TestAdaptiveMLP:
    """Tests for AdaptiveMLP."""
    
    def test_adaptive_mlp_init(self):
        """Test AdaptiveMLP initialization."""
        config = ModelConfig(hidden_dim=128, mlp_ratio=4)
        mlp = AdaptiveMLP(config)
        
        assert isinstance(mlp.gate_proj, nn.Linear)
        assert isinstance(mlp.up_proj, nn.Linear)
        assert isinstance(mlp.down_proj, nn.Linear)
    
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        config = ModelConfig(hidden_dim=128, mlp_ratio=4)
        
        mlp = AdaptiveMLP(config)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        output = mlp(hidden_states)
        
        assert output.shape == hidden_states.shape
    
    def test_forward_computation(self):
        """Test that forward pass computes SwiGLU."""
        batch_size = 1
        seq_len = 3
        config = ModelConfig(hidden_dim=64, mlp_ratio=2)
        
        mlp = AdaptiveMLP(config)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        output = mlp(hidden_states)
        
        # Output should be different from input
        assert not torch.allclose(output, hidden_states)
        
        # No NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestAdaptiveLayer:
    """Tests for AdaptiveLayer."""
    
    def test_adaptive_layer_init(self):
        """Test AdaptiveLayer initialization."""
        config = ModelConfig(hidden_dim=128, num_heads=4)
        layer = AdaptiveLayer(config)
        
        assert isinstance(layer.attention, AdaptiveAttention)
        assert isinstance(layer.mlp, AdaptiveMLP)
        assert isinstance(layer.attn_norm, nn.Module)
        assert isinstance(layer.mlp_norm, nn.Module)
    
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        config = ModelConfig(hidden_dim=128, num_heads=4)
        
        layer = AdaptiveLayer(config)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        output = layer(hidden_states)
        
        assert output.shape == hidden_states.shape
    
    def test_forward_residual_connection(self):
        """Test residual connections are applied."""
        batch_size = 1
        seq_len = 3
        config = ModelConfig(hidden_dim=64, num_heads=4)
        
        layer = AdaptiveLayer(config)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        output = layer(hidden_states)
        
        # Output should be close to input due to residual (at init)
        # But not exactly equal
        assert not torch.allclose(output, torch.zeros_like(output))


class TestAdaptiveTransformer:
    """Tests for AdaptiveTransformer."""
    
    def test_transformer_init(self):
        """Test AdaptiveTransformer initialization."""
        config = ModelConfig(num_layers=2, hidden_dim=128, num_heads=4)
        model = AdaptiveTransformer(config)
        
        assert len(model.layers) == 2
        assert isinstance(model.embed, nn.Embedding)
        assert isinstance(model.norm, nn.Module)
        assert isinstance(model.lm_head, nn.Linear)
    
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        config = ModelConfig(num_layers=2, hidden_dim=128, num_heads=4, vocab_size=1000)
        
        model = AdaptiveTransformer(config)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_forward_return_hidden(self):
        """Test forward pass with hidden states return."""
        batch_size = 2
        seq_len = 5
        config = ModelConfig(num_layers=2, hidden_dim=64, num_heads=4, vocab_size=100)
        
        model = AdaptiveTransformer(config)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids, return_hidden=True)
        
        assert hasattr(output, 'hidden_states')
        assert output.hidden_states.shape == (batch_size, seq_len, config.hidden_dim)
    
    def test_generate(self):
        """Test generate method."""
        batch_size = 1
        seq_len = 5
        config = ModelConfig(num_layers=2, hidden_dim=64, num_heads=4, vocab_size=100)
        
        model = AdaptiveTransformer(config)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        generated = model.generate(input_ids, max_new_tokens=3)
        
        # Should have generated new tokens
        assert generated.shape[1] == seq_len + 3
        assert generated.shape[0] == batch_size
    
    def test_model_parameters(self):
        """Test model has trainable parameters."""
        config = ModelConfig(num_layers=1, hidden_dim=64, num_heads=4)
        model = AdaptiveTransformer(config)
        
        params = list(model.parameters())
        
        assert len(params) > 0
        
        # Check all parameters require grad
        for param in params:
            assert param.requires_grad
