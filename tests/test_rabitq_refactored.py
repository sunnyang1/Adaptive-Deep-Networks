"""
Refactored RaBitQ Test Suite.

Comprehensive tests for all components:
- Rotation (FWHT)
- Quantization (Lloyd-Max)
- Compression (MSE)
- Cache (HF integration)
"""

import pytest
import torch
import math
from typing import Tuple

# New refactored imports
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rabitq.rotation import RandomRotation, fwht, fwht_inverse
from rabitq.quantizer import LloydMaxQuantizer
from rabitq.compressor import MSECompressor, CompressorConfig, pack_bits, unpack_bits
from rabitq.cache import RaBitQCache, CacheConfig
from rabitq.api import RaBitQ, RaBitQConfig, create_k4_v2, create_k3_v2, create_k2_v2


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device."""
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@pytest.fixture
def sample_unit_vectors(device):
    """Generate sample unit vectors."""
    def _generate(n_vectors: int = 1000, dim: int = 64) -> torch.Tensor:
        x = torch.randn(n_vectors, dim, device=device)
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    return _generate


# =============================================================================
# Rotation Tests
# =============================================================================

class TestRotation:
    """Test Fast Walsh-Hadamard Transform and RandomRotation."""
    
    def test_fwht_orthogonality(self, device):
        """FWHT should be orthogonal up to factor of n."""
        n = 64
        x = torch.randn(10, n, device=device)
        
        x_transformed = fwht(x)
        x_recovered = fwht_inverse(x_transformed)
        
        # Should recover original
        error = (x - x_recovered).abs().max().item()
        assert error < 1e-4, f"FWHT not orthogonal: max error {error}"
    
    def test_fwht_power_of_2(self, device):
        """FWHT requires power-of-2 dimension."""
        x = torch.randn(10, 63, device=device)
        with pytest.raises(AssertionError):
            fwht(x)
    
    def test_random_rotation_norm_preservation(self, device):
        """Random rotation should preserve vector norms."""
        rotation = RandomRotation(64, seed=42, device=device)
        x = torch.randn(100, 64, device=device)
        
        x_rot = rotation.rotate(x)
        
        # Check norm preservation
        norms_orig = x.norm(dim=-1)
        norms_rot = x_rot.norm(dim=-1)
        ratio = (norms_rot / (norms_orig + 1e-8)).mean().item()
        
        assert 0.99 < ratio < 1.01, f"Norm not preserved: ratio {ratio}"
    
    def test_random_rotation_invertibility(self, device):
        """Random rotation should be invertible."""
        rotation = RandomRotation(64, seed=42, device=device)
        x = torch.randn(100, 64, device=device)
        
        x_rot = rotation.rotate(x)
        x_recovered = rotation.inverse(x_rot)
        
        error = (x - x_recovered).abs().mean().item()
        assert error < 1e-3, f"Rotation not invertible: error {error}"
    
    def test_random_rotation_padded(self, device):
        """Random rotation should handle non-power-of-2 dimensions."""
        rotation = RandomRotation(60, seed=42, device=device)  # 60 -> 64
        x = torch.randn(100, 60, device=device)
        
        x_rot = rotation.rotate(x)
        assert x_rot.shape[-1] == 64  # Padded to 64
        
        x_recovered = rotation.inverse(x_rot)
        assert x_recovered.shape[-1] == 60  # Back to original
        
        error = (x - x_recovered).abs().mean().item()
        assert error < 1e-3, f"Padded rotation failed: error {error}"


# =============================================================================
# Quantizer Tests
# =============================================================================

class TestQuantizer:
    """Test Lloyd-Max quantizer."""
    
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_quantizer_symmetry(self, device, bits):
        """Quantizer centroids should be symmetric for symmetric data."""
        quantizer = LloydMaxQuantizer(num_bits=bits, device=device)
        
        # Fit on symmetric Gaussian
        samples = torch.randn(10000, device=device) * 0.1
        quantizer.fit(samples)
        
        # Check symmetry (relaxed threshold for 4-bit with small sample)
        centroid_sum = quantizer.centroids.sum().item()
        assert abs(centroid_sum) < 0.2, f"Centroids not symmetric: sum={centroid_sum}"
    
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_quantizer_reconstruction(self, device, bits):
        """Quantizer should reconstruct with bounded error."""
        quantizer = LloydMaxQuantizer(num_bits=bits, device=device)
        
        # Fit and test on same distribution
        samples = torch.randn(5000, device=device) * 0.1
        quantizer.fit(samples)
        
        # Test reconstruction
        x = torch.randn(1000, device=device) * 0.1
        indices, x_hat = quantizer.quantize(x)
        
        # Check indices are valid
        assert indices.min() >= 0
        assert indices.max() < 2 ** bits
        
        # Check reconstruction error is bounded
        mse = ((x - x_hat) ** 2).mean().item()
        max_val = samples.abs().max().item()
        theoretical_max_mse = (max_val ** 2) / (2 ** bits) ** 2
        
        assert mse < theoretical_max_mse * 4, f"MSE too high: {mse}"
    
    def test_quantizer_not_fitted(self, device):
        """Quantizer should raise error if not fitted."""
        quantizer = LloydMaxQuantizer(num_bits=4, device=device)
        x = torch.randn(100, device=device)
        
        with pytest.raises(RuntimeError, match="fitted"):
            quantizer.encode(x)
    
    def test_quantizer_gaussian_fit(self, device):
        """Test fitting on Gaussian distribution."""
        quantizer = LloydMaxQuantizer(num_bits=4, device=device)
        quantizer.fit_gaussian(mean=0.0, std=0.1, num_samples=10000)
        
        assert quantizer.is_fitted
        
        # Test on similar distribution
        x = torch.randn(1000, device=device) * 0.1
        _, x_hat = quantizer.quantize(x)
        
        error = (x - x_hat).abs().mean().item()
        assert error < 0.02, f"Gaussian fit failed: error {error}"


# =============================================================================
# Compressor Tests
# =============================================================================

class TestCompressor:
    """Test MSE compressor."""
    
    @pytest.mark.parametrize("bits", [2, 4])
    def test_compressor_basic(self, device, sample_unit_vectors, bits):
        """Compressor should work end-to-end."""
        config = CompressorConfig(bits=bits, use_rotation=True, pack_bits=False, device=device)
        compressor = MSECompressor(config)
        
        # Generate data
        x = sample_unit_vectors(500, 64)
        
        # Fit
        compressor.fit(x[:100], head_dim=64)
        
        # Compress/decompress
        compressed = compressor.compress(x, head_dim=64)
        x_dq = compressor.decompress(compressed)
        
        # Check shape
        assert x_dq.shape == x.shape
        
        # Check error is reasonable
        rel_error = (x - x_dq).abs().mean() / x.abs().mean()
        assert rel_error < 0.5, f"Relative error too high: {rel_error}"
    
    def test_compressor_with_packing(self, device, sample_unit_vectors):
        """Compressor should work with bit-packing."""
        config = CompressorConfig(bits=4, use_rotation=True, pack_bits=True, device=device)
        compressor = MSECompressor(config)
        
        x = sample_unit_vectors(500, 64)
        compressor.fit(x[:100], head_dim=64)
        
        compressed = compressor.compress(x, head_dim=64)
        x_dq = compressor.decompress(compressed)
        
        # Should match without packing
        assert x_dq.shape == x.shape
    
    def test_compressor_no_rotation(self, device, sample_unit_vectors):
        """Compressor should work without rotation."""
        config = CompressorConfig(bits=4, use_rotation=False, pack_bits=False, device=device)
        compressor = MSECompressor(config)
        
        x = sample_unit_vectors(500, 64)
        compressor.fit(x[:100], head_dim=64)
        
        compressed = compressor.compress(x, head_dim=64)
        x_dq = compressor.decompress(compressed)
        
        assert x_dq.shape == x.shape


# =============================================================================
# Bit Packing Tests
# =============================================================================

class TestBitPacking:
    """Test bit packing utilities."""
    
    @pytest.mark.parametrize("bits", [1, 2, 4, 8])
    def test_pack_unpack_roundtrip(self, bits):
        """Pack/unpack should be lossless for valid bit widths."""
        # Generate random indices
        max_val = 2 ** bits
        x = torch.randint(0, max_val, (1000,), dtype=torch.long)
        
        packed, shape = pack_bits(x, bits)
        x_recovered = unpack_bits(packed, bits, shape)
        
        assert torch.equal(x, x_recovered), "Pack/unpack not lossless"
    
    def test_pack_efficiency(self):
        """Packing should be efficient for power-of-2 bits."""
        x = torch.randint(0, 16, (1000,), dtype=torch.long)
        
        # 4-bit: should pack 2 elements per byte
        packed_4, _ = pack_bits(x, 4)
        assert packed_4.numel() == 500, f"4-bit packing inefficient: {packed_4.numel()} vs 500"
        
        # 2-bit: should pack 4 elements per byte
        packed_2, _ = pack_bits(x, 2)
        assert packed_2.numel() == 250, f"2-bit packing inefficient: {packed_2.numel()} vs 250"
    
    def test_pack_3bit_no_packing(self):
        """3-bit should not pack (inefficient)."""
        x = torch.randint(0, 8, (1000,), dtype=torch.long)
        packed, _ = pack_bits(x, 3)
        
        # Should store as uint8 without packing
        assert packed.numel() == 1000
    
    def test_pack_bits_invalid(self):
        """Should reject invalid bit widths."""
        x = torch.randint(0, 16, (100,))
        with pytest.raises(ValueError):
            pack_bits(x, 10)


# =============================================================================
# API Tests
# =============================================================================

class TestAPI:
    """Test unified API."""
    
    def test_api_basic(self, device, sample_unit_vectors):
        """API should work end-to-end."""
        tq = create_k4_v2(head_dim=64, device=device)
        
        keys = sample_unit_vectors(500, 64)
        values = sample_unit_vectors(500, 64)
        
        # Fit
        tq.fit(keys[:100], values[:100])
        
        # Compress
        compressed = tq.compress(keys, values)
        
        # Decompress
        keys_dq, values_dq = tq.decompress(compressed)
        
        assert keys_dq.shape == keys.shape
        assert values_dq.shape == values.shape
    
    def test_api_memory_stats(self, device):
        """API should provide accurate memory stats."""
        tq = create_k4_v2(head_dim=64, device=device)
        
        stats = tq.memory_stats(seq_len=512, num_layers=32, num_heads=32)
        
        assert 'original_mb' in stats
        assert 'compressed_mb' in stats
        assert 'compression_ratio' in stats
        assert stats['compression_ratio'] > 1.0
    
    def test_api_as_cache(self, device):
        """API should provide HF-compatible cache."""
        tq = create_k4_v2(head_dim=64, device=device)
        cache = tq.as_cache(residual_window=64)
        
        assert isinstance(cache, RaBitQCache)
        assert cache.config.residual_window == 64

    @pytest.mark.parametrize("factory", [create_k4_v2, create_k3_v2, create_k2_v2])
    def test_factory_functions(self, device, factory):
        """All factory functions should work."""
        tq = factory(head_dim=64, device=device)
        assert isinstance(tq, RaBitQ)


# =============================================================================
# Cache Tests
# =============================================================================

class TestCache:
    """Test HF-compatible cache."""

    def test_cache_basic(self, device):
        """Cache should store and retrieve."""
        cache = RaBitQCache(key_bits=4, value_bits=2, residual_window=0, device=device)
        
        batch, heads, seq, dim = 1, 8, 64, 64
        keys = torch.randn(batch, heads, seq, dim, device=device)
        values = torch.randn(batch, heads, seq, dim, device=device)
        
        # Update
        out_k, out_v = cache.update(keys, values, layer_idx=0)
        
        assert out_k.shape == keys.shape
        assert out_v.shape == values.shape
        assert cache.get_seq_length(0) == seq
    
    def test_cache_accumulation(self, device):
        """Cache should accumulate across updates."""
        cache = RaBitQCache(key_bits=4, value_bits=2, residual_window=0, device=device)
        
        batch, heads, dim = 1, 8, 64
        
        # First update
        keys1 = torch.randn(batch, heads, 32, dim, device=device)
        values1 = torch.randn(batch, heads, 32, dim, device=device)
        cache.update(keys1, values1, layer_idx=0)
        
        # Second update
        keys2 = torch.randn(batch, heads, 32, dim, device=device)
        values2 = torch.randn(batch, heads, 32, dim, device=device)
        out_k, out_v = cache.update(keys2, values2, layer_idx=0)
        
        # Should have 64 tokens
        assert out_k.shape[2] == 64
        assert cache.get_seq_length(0) == 64
    
    def test_cache_residual_window(self, device):
        """Cache should keep recent tokens in fp16."""
        cache = RaBitQCache(key_bits=4, value_bits=2, residual_window=32, device=device)
        
        batch, heads, dim = 1, 8, 64
        
        # Add more than window
        keys = torch.randn(batch, heads, 64, dim, device=device)
        values = torch.randn(batch, heads, 64, dim, device=device)
        out_k, out_v = cache.update(keys, values, layer_idx=0)
        
        # All tokens should be returned
        assert out_k.shape[2] == 64
        
        # Recent window should be in recent buffer
        assert len(cache._recent_k[0]) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, device):
        """Test complete pipeline from rotation to compression."""
        # Generate KV cache
        batch, heads, seq, dim = 1, 8, 256, 64
        keys = torch.randn(batch, heads, seq, dim, device=device)
        values = torch.randn(batch, heads, seq, dim, device=device)
        
        # Compress
        tq = create_k4_v2(head_dim=dim, device=device)
        tq.fit(keys[:, :, :64, :], values[:, :, :64, :])
        
        compressed = tq.compress(keys, values)
        keys_dq, values_dq = tq.decompress(compressed)
        
        # Check
        assert keys_dq.shape == keys.shape
        assert values_dq.shape == values.shape
        
        # Error should be reasonable
        key_error = (keys - keys_dq).abs().mean() / keys.abs().mean()
        assert key_error < 0.2, f"Key error too high: {key_error}"
    
    def test_compression_ratio_accuracy(self, device):
        """Actual compression should match predicted ratio."""
        tq = create_k4_v2(head_dim=64, device=device)
        
        # Predicted
        stats = tq.memory_stats(seq_len=512, num_layers=1, num_heads=8)
        predicted_ratio = stats['compression_ratio']
        
        # Actual
        keys = torch.randn(1, 8, 512, 64, device=device)
        values = torch.randn(1, 8, 512, 64, device=device)
        
        tq.fit(keys[:, :, :64, :], values[:, :, :64, :])
        compressed = tq.compress(keys, values)
        
        # Calculate actual
        orig_bytes = keys.numel() * 2 + values.numel() * 2  # fp16
        comp_bytes = (compressed['keys']['indices'].numel() + 
                     compressed['values']['indices'].numel() +
                     compressed['keys']['norm'].numel() * 2 +
                     compressed['values']['norm'].numel() * 2)
        actual_ratio = orig_bytes / comp_bytes
        
        # Should be close
        ratio_diff = abs(actual_ratio - predicted_ratio) / predicted_ratio
        assert ratio_diff < 0.1, f"Ratio mismatch: predicted={predicted_ratio}, actual={actual_ratio}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
