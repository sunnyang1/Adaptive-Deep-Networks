"""
Tests for MNN-Inspired TurboQuant Implementation

Reference: https://github.com/alibaba/MNN/commit/244f5d10df5a95b4f4e6f3d9251c6fe3dc0e7c83
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.turboquant import (
    MNNTurboQuantConfig,
    MNNTurboQuantCompressor,
    AttentionMode,
    KVQuantMode,
    LloydMaxQuantizer,
    create_mnn_turboquant,
    CONFIG_RECOMMENDATIONS,
)


class TestAttentionMode:
    """Test attention mode encoding/decoding."""
    
    def test_encode_decode(self):
        """Test encoding and decoding roundtrip."""
        test_cases = [
            (False, KVQuantMode.FP16, 0),
            (True, KVQuantMode.FP16, 8),
            (True, KVQuantMode.KV_INT8, 10),
            (True, KVQuantMode.KV_TQ4, 14),
            (True, KVQuantMode.KV_TQ3, 12),
            (False, KVQuantMode.KEY_TQ4, 5),
        ]
        
        for flash, kv_mode, expected in test_cases:
            encoded = AttentionMode.encode(flash, kv_mode)
            assert encoded == expected, f"Encode failed for {(flash, kv_mode)}"
            
            decoded_flash, decoded_kv = AttentionMode.decode(expected)
            assert decoded_flash == flash, f"Decode flash failed for {expected}"
            assert decoded_kv == kv_mode, f"Decode KV failed for {expected}"
    
    def test_get_description(self):
        """Test mode descriptions."""
        assert "FlashAttn" in AttentionMode.get_description(8)
        assert "FP16" in AttentionMode.get_description(8)
        assert "KV-TQ4" in AttentionMode.get_description(14)
        assert "KV-TQ3" in AttentionMode.get_description(12)


class TestMNNTurboQuantConfig:
    """Test MNN TurboQuant configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MNNTurboQuantConfig()
        assert config.attention_mode == 8
        assert config.flash_attention == True
        assert config.kv_quant_mode == KVQuantMode.FP16
    
    def test_compression_ratios(self):
        """Test compression ratio calculations."""
        test_cases = [
            (8, 1.0),    # FP16
            (10, 2.0),   # KV-INT8
            (14, 3.0),   # KV-TQ4
            (12, 4.0),   # KV-TQ3
        ]
        
        for mode, expected_ratio in test_cases:
            config = MNNTurboQuantConfig(attention_mode=mode)
            assert config.compression_ratio == expected_ratio, \
                f"Mode {mode}: expected {expected_ratio}, got {config.compression_ratio}"
    
    def test_model_size_recommendations(self):
        """Test model-size aware recommendations."""
        # Small model (<4B) - TQ not recommended
        small_config = MNNTurboQuantConfig(attention_mode=14, min_params_for_tq=4e9)
        assert not small_config.is_recommended_for_model_size(2e9)
        
        # Large model (>=4B) - TQ recommended
        large_config = MNNTurboQuantConfig(attention_mode=14, min_params_for_tq=4e9)
        assert large_config.is_recommended_for_model_size(8e9)
        
        # INT8 always recommended
        int8_config = MNNTurboQuantConfig(attention_mode=10)
        assert int8_config.is_recommended_for_model_size(2e9)


class TestLloydMaxQuantizer:
    """Test Lloyd-Max quantizer."""
    
    def test_fit_3bit(self):
        """Test fitting 3-bit quantizer."""
        quantizer = LloydMaxQuantizer(num_bits=3)
        
        # Generate training data
        data = torch.randn(1000, 64)
        quantizer.fit(data)
        
        # Check codebook size
        assert len(quantizer.codebook) == 8  # 2^3 = 8 levels
        
        # Check boundaries
        assert len(quantizer.boundaries) == 7  # n-1 boundaries
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        quantizer = LloydMaxQuantizer(num_bits=3)
        
        # Fit on training data
        train_data = torch.randn(1000, 32)
        quantizer.fit(train_data)
        
        # Encode and decode test data
        test_data = torch.randn(100, 32)
        indices = quantizer.encode(test_data)
        decoded = quantizer.decode(indices)
        
        # Check shapes
        assert indices.shape == test_data.shape
        assert decoded.shape == test_data.shape
        
        # Check indices are in valid range
        assert indices.min() >= 0
        assert indices.max() < 8
    
    def test_reconstruction_error(self):
        """Test reconstruction error is reasonable."""
        quantizer = LloydMaxQuantizer(num_bits=4)  # 4-bit for better accuracy
        
        # Fit on Gaussian data
        train_data = torch.randn(10000, 64)
        quantizer.fit(train_data)
        
        # Test reconstruction
        test_data = torch.randn(1000, 64)
        indices = quantizer.encode(test_data)
        decoded = quantizer.decode(indices)
        
        # Calculate relative error
        error = (test_data - decoded).abs().mean()
        relative_error = error / test_data.abs().mean()
        
        # Should be < 10% for 4-bit
        assert relative_error < 0.1, f"Relative error too high: {relative_error}"


class TestMNNTurboQuantCompressor:
    """Test MNN TurboQuant compressor."""
    
    @pytest.fixture
    def sample_kv(self):
        """Create sample KV cache."""
        batch_size = 2
        num_heads = 4
        seq_len = 64
        head_dim = 32
        
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        return keys, values
    
    def test_fp16_mode(self, sample_kv):
        """Test FP16 mode (no compression)."""
        keys, values = sample_kv
        
        compressor = create_mnn_turboquant(attention_mode=8, head_dim=32)
        compressed = compressor.compress_kv(keys, values)
        keys_decomp, values_decomp = compressor.decompress_kv(compressed)
        
        # Should be nearly identical (just FP16 conversion)
        torch.testing.assert_close(keys, keys_decomp, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(values, values_decomp, rtol=1e-3, atol=1e-3)
    
    def test_int8_mode(self, sample_kv):
        """Test INT8 quantization mode."""
        keys, values = sample_kv
        
        compressor = create_mnn_turboquant(attention_mode=10, head_dim=32)
        compressed = compressor.compress_kv(keys, values)
        keys_decomp, values_decomp = compressor.decompress_kv(compressed)
        
        # Check compression happened
        assert compressed['key_indices'] is not None
        assert compressed['value_indices'] is not None
        
        # Check reasonable reconstruction
        key_error = (keys - keys_decomp).abs().mean()
        assert key_error < 0.1, f"Key error too high: {key_error}"
    
    def test_tq4_mode(self, sample_kv):
        """Test TQ4 quantization mode."""
        keys, values = sample_kv
        
        config = MNNTurboQuantConfig(attention_mode=14)
        compressor = MNNTurboQuantCompressor(config, head_dim=32)
        
        # Fit codebooks
        compressor.fit_codebooks(keys, values)
        
        # Compress and decompress
        compressed = compressor.compress_kv(keys, values)
        keys_decomp, values_decomp = compressor.decompress_kv(compressed)
        
        # Check shapes preserved
        assert keys_decomp.shape == keys.shape
        assert values_decomp.shape == values.shape
    
    def test_memory_stats(self):
        """Test memory statistics calculation."""
        compressor = create_mnn_turboquant(attention_mode=14, head_dim=128)
        
        stats = compressor.get_memory_stats(seq_len=32768, batch_size=1, num_heads=32)
        
        assert 'original_mb' in stats
        assert 'compressed_mb' in stats
        assert 'saving_ratio' in stats
        assert 'compression_ratio' in stats
        
        # For TQ4, should get ~3x compression
        assert stats['saving_ratio'] > 2.5
        assert stats['saving_ratio'] < 3.5


class TestIntegration:
    """Integration tests."""
    
    def test_config_recommendations(self):
        """Test configuration recommendations."""
        # Check all recommendation configs are valid
        for name, mode in CONFIG_RECOMMENDATIONS.items():
            config = MNNTurboQuantConfig(attention_mode=mode)
            assert config.flash_attention == (mode >= 8)
            print(f"{name}: Mode {mode} - {AttentionMode.get_description(mode)}")
    
    def test_different_head_dims(self):
        """Test with different head dimensions."""
        head_dims = [64, 128, 256]
        
        for head_dim in head_dims:
            compressor = create_mnn_turboquant(attention_mode=10, head_dim=head_dim)
            
            keys = torch.randn(1, 8, 32, head_dim)
            values = torch.randn(1, 8, 32, head_dim)
            
            compressed = compressor.compress_kv(keys, values)
            keys_decomp, values_decomp = compressor.decompress_kv(compressed)
            
            assert keys_decomp.shape == keys.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
