"""
Simplified unit tests for Models module.
"""

import pytest
import torch
import torch.nn as nn

from src.models.configs import ModelConfig
from src.models.adaptive_transformer import (
    AdaptiveAttention, AdaptiveMLP, AdaptiveLayer, AdaptiveTransformer
)


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.num_layers == 32
        assert config.hidden_dim == 4096
        assert config.num_heads == 32
    
    def test_head_dim_calculation(self):
        """Test head_dim is calculated correctly."""
        config = ModelConfig(hidden_dim=512, num_heads=8)
        
        assert config.head_dim == 64  # 512 / 8


class TestAdaptiveAttention:
    """Tests for AdaptiveAttention."""
    
    def test_adaptive_attention_init(self):
        """Test AdaptiveAttention initialization."""
        config = ModelConfig(hidden_dim=128, num_heads=4)
        attn = AdaptiveAttention(config)
        
        assert attn.head_dim == 32  # 128 / 4
    
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        config = ModelConfig(hidden_dim=128, num_heads=4)
        
        attn = AdaptiveAttention(config)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        output = attn(hidden_states)
        
        assert output.shape == hidden_states.shape


class TestAdaptiveMLP:
    """Tests for AdaptiveMLP."""
    
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        config = ModelConfig(hidden_dim=128, mlp_ratio=4)
        
        mlp = AdaptiveMLP(config)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        output = mlp(hidden_states)
        
        assert output.shape == hidden_states.shape


class TestAdaptiveLayer:
    """Tests for AdaptiveLayer."""
    
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        config = ModelConfig(hidden_dim=128, num_heads=4, num_blocks=4)
        
        layer = AdaptiveLayer(config, layer_idx=0)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
        
        output = layer(hidden_states)
        
        assert output.shape == hidden_states.shape


class TestAdaptiveTransformer:
    """Tests for AdaptiveTransformer."""
    
    def test_transformer_init(self):
        """Test AdaptiveTransformer initialization."""
        config = ModelConfig(num_layers=2, hidden_dim=128, num_heads=4, num_blocks=2)
        model = AdaptiveTransformer(config)
        
        assert len(model.layers) == 2
    
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        config = ModelConfig(
            num_layers=2, hidden_dim=128, num_heads=4, 
            vocab_size=1000, num_blocks=2
        )
        
        model = AdaptiveTransformer(config)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)
