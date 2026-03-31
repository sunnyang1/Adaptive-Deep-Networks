"""
Tests for TurboQuant V3 (tonbistudio improvements)

Tests focus on core functionality that works with the current implementation.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.turboquant import (
    TurboQuantV3,
    TurboQuantConfig,
    MSECompressor,
    CompressorConfig,
    LloydMaxQuantizer,
    RandomRotation,
    pack_bits,
    unpack_bits,
    create_k4_v2,
    create_k3_v2,
    RECOMMENDED,
)


class TestBitPacking:
    """Test bit-packing utilities."""
    
    def test_pack_unpack_2bit(self):
        """Test 2-bit packing."""
        tensor = torch.randint(0, 4, (100,))  # 2-bit values
        packed, shape = pack_bits(tensor, bits=2)
        unpacked = unpack_bits(packed, bits=2, original_shape=shape)
        
        assert torch.equal(tensor, unpacked)
    
    def test_pack_unpack_4bit(self):
        """Test 4-bit packing."""
        tensor = torch.randint(0, 16, (100,))  # 4-bit values
        packed, shape = pack_bits(tensor, bits=4)
        unpacked = unpack_bits(packed, bits=4, original_shape=shape)
        
        assert torch.equal(tensor, unpacked)
    
    def test_pack_unpack_3bit(self):
        """Test 3-bit packing."""
        tensor = torch.randint(0, 8, (96,))  # 3-bit values
        packed, shape = pack_bits(tensor, bits=3)
        unpacked = unpack_bits(packed, bits=3, original_shape=shape)
        
        assert torch.equal(tensor, unpacked)


class TestRandomRotation:
    """Test random rotation."""
    
    def test_rotation_isometry(self):
        """Test that rotation preserves norms."""
        rot = RandomRotation(dim=64)
        x = torch.randn(10, 64)
        
        x_rot = rot.rotate(x)
        x_inv = rot.inverse(x_rot)
        
        # Check norm preservation
        assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), rtol=1e-4, atol=1e-5)
        
        # Check invertibility
        assert torch.allclose(x, x_inv, rtol=1e-4, atol=1e-4)
    
    def test_rotation_distribution(self):
        """Test that rotation creates bell-curve distribution."""
        rot = RandomRotation(dim=128)
        
        # Create uniform data
        x = torch.randn(1000, 128)
        x_rot = rot.rotate(x)
        
        # After rotation, each coordinate should be roughly Gaussian
        means = x_rot.mean(dim=0)
        stds = x_rot.std(dim=0)
        
        # Means should be close to 0
        assert means.abs().mean() < 0.1
        
        # Standard deviations should be similar
        assert stds.std() < 0.2


class TestLloydMaxQuantizer:
    """Test Lloyd-Max quantizer."""
    
    def test_fit_gaussian(self):
        """Test fitting on Gaussian data."""
        quantizer = LloydMaxQuantizer(num_bits=4)
        
        data = torch.randn(10000, 64)
        quantizer.fit(data)
        
        assert quantizer._is_fitted
        assert quantizer.centroids.shape[0] == 16  # 2^4
    
    def test_quantize_shape(self):
        """Test quantization output shape."""
        quantizer = LloydMaxQuantizer(num_bits=4)
        
        # Fit
        train_data = torch.randn(5000, 32)
        quantizer.fit(train_data)
        
        # Quantize
        test_data = torch.randn(100, 32)
        result = quantizer.quantize(test_data)
        
        # Result may be tuple or tensor depending on implementation
        if isinstance(result, tuple):
            indices = result[0]
        else:
            indices = result
        
        # Indices should have same shape as input
        assert indices.shape == test_data.shape


class TestMSECompressor:
    """Test MSE compressor."""
    
    def test_compress_decompress_4bit(self):
        """Test 4-bit compression roundtrip."""
        config = CompressorConfig(bits=4, use_rotation=True, pack_bits=True)
        compressor = MSECompressor(config)
        
        # Fit
        sample = torch.randn(100, 64)
        compressor.fit(sample)
        
        # Compress
        data = torch.randn(10, 64)
        compressed = compressor.compress(data)
        
        # Decompress
        decompressed = compressor.decompress(compressed)
        
        assert decompressed.shape == data.shape
        
        # Check error is reasonable
        error = (data - decompressed).abs().mean()
        assert error < 0.1
    
    def test_no_rotation(self):
        """Test compression without rotation."""
        config = CompressorConfig(bits=4, use_rotation=False, pack_bits=False)
        compressor = MSECompressor(config)
        
        sample = torch.randn(100, 64)
        compressor.fit(sample)
        
        data = torch.randn(10, 64)
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        
        assert decompressed.shape == data.shape


class TestTurboQuantV3:
    """Test V3 main class."""
    
    @pytest.fixture
    def sample_kv(self):
        """Create sample KV cache."""
        return torch.randn(2, 4, 128, 64), torch.randn(2, 4, 128, 64)
    
    def test_k4_v2_compression(self, sample_kv):
        """Test K4/V2 compression."""
        keys, values = sample_kv
        
        v3 = create_k4_v2(head_dim=64)
        v3.fit(keys[:1, :1, :32], values[:1, :1, :32])
        
        compressed = v3.compress(keys, values)
        keys_deq, values_deq = v3.decompress(compressed)
        
        assert keys_deq.shape == keys.shape
        assert values_deq.shape == values.shape
    
    def test_k3_v2_compression(self, sample_kv):
        """Test K3/V2 compression."""
        keys, values = sample_kv
        
        v3 = create_k3_v2(head_dim=64)
        v3.fit(keys[:1, :1, :32], values[:1, :1, :32])
        
        compressed = v3.compress(keys, values)
        keys_deq, values_deq = v3.decompress(compressed)
        
        assert keys_deq.shape == keys.shape
    
    def test_long_sequence(self, sample_kv):
        """Test with long sequences."""
        keys, values = sample_kv
        
        v3 = create_k4_v2(head_dim=64)
        v3.fit(keys[:, :, :64], values[:, :, :64])
        
        compressed = v3.compress(keys, values)
        keys_deq, values_deq = v3.decompress(compressed)
        
        assert keys_deq.shape == keys.shape


class TestV3RecommendedConfigs:
    """Test recommended V3 configurations."""
    
    def test_all_configs_valid(self):
        """Test all recommended configs create valid compressors."""
        for name, factory in RECOMMENDED.items():
            v3 = factory(head_dim=64, device='cpu')
            assert isinstance(v3, TurboQuantV3)
    
    def test_k4_v2_quality(self):
        """Test K4/V2 has good reconstruction quality."""
        keys = torch.randn(2, 4, 128, 64)
        values = torch.randn(2, 4, 128, 64)
        
        v3 = create_k4_v2(head_dim=64)
        v3.fit(keys[:1, :1, :32], values[:1, :1, :32])
        
        compressed = v3.compress(keys, values)
        keys_deq, values_deq = v3.decompress(compressed)
        
        key_error = (keys - keys_deq).abs().mean()
        value_error = (values - values_deq).abs().mean()
        
        # K4 should have low error
        assert key_error < 0.05
        assert value_error < 0.1


class TestIntegration:
    """Integration tests."""
    
    def test_v3_vs_quantized(self):
        """Compare V3 with standard quantization."""
        keys = torch.randn(2, 4, 256, 64)
        
        # V3 quantization
        v3 = create_k4_v2(head_dim=64)
        v3.fit(keys[:1, :1, :64], keys[:1, :1, :64])
        compressed = v3.compress(keys, keys)
        keys_v3, _ = v3.decompress(compressed)
        
        # Naive 4-bit quantization
        keys_scaled = keys * 7.5
        keys_int = keys_scaled.round().clamp(-8, 7).to(torch.int8)
        keys_naive = keys_int.float() / 7.5
        
        # V3 should have lower error due to rotation
        error_v3 = (keys - keys_v3).abs().mean()
        error_naive = (keys - keys_naive).abs().mean()
        
        assert error_v3 < error_naive


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
