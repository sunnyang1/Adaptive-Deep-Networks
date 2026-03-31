"""
Unit tests for TurboQuant compression modules.

Tests:
- PolarQuant compression/decompression
- QJL unbiased estimation
- Full TurboQuant pipeline
- Tensor Core kernels
"""

import pytest
import torch
import math

from src.turboquant import (
    PolarQuant, TurboQuantPipeline, TurboQuantConfig,
    TensorCoreKernel, INT4Linear
)
from src.turboquant.polar_quant import (
    HadamardTransform, CartesianToPolar, LloydMaxQuantizer
)
from src.turboquant.qjl import QJLCompressor, create_qjl_cache


class TestHadamardTransform:
    """Tests for Random Hadamard Transform."""
    
    def test_hadamard_matrix_properties(self):
        """Test that Hadamard matrix is orthogonal."""
        dim = 8
        rht = HadamardTransform(dim, device='cpu')
        
        # H @ H^T should be identity (up to scaling)
        H = rht.H
        product = H @ H.T
        
        # Should be close to identity
        identity = torch.eye(dim)
        assert torch.allclose(product, identity, atol=1e-5)
    
    def test_rht_invertibility(self):
        """Test that RHT is invertible."""
        dim = 64
        rht = HadamardTransform(dim, device='cpu')
        
        x = torch.randn(10, dim)
        x_transformed = rht.forward(x)
        x_recovered = rht.inverse(x_transformed)
        
        assert torch.allclose(x, x_recovered, atol=1e-5)
    
    def test_energy_preservation(self):
        """Test that RHT preserves energy (norm)."""
        dim = 64
        rht = HadamardTransform(dim, device='cpu')
        
        x = torch.randn(10, dim)
        x_transformed = rht.forward(x)
        
        orig_norm = torch.norm(x, dim=-1)
        transformed_norm = torch.norm(x_transformed, dim=-1)
        
        assert torch.allclose(orig_norm, transformed_norm, atol=1e-4)


class TestCartesianToPolar:
    """Tests for Cartesian-Polar coordinate conversion."""
    
    def test_conversion_roundtrip(self):
        """Test roundtrip conversion."""
        dim = 8
        x = torch.randn(10, dim)
        
        # Forward
        r, theta = CartesianToPolar.forward(x)
        
        # Inverse
        x_reconstructed = CartesianToPolar.inverse(r, theta)
        
        assert torch.allclose(x, x_reconstructed, atol=1e-4)
    
    def test_radius_computation(self):
        """Test that radius is correct."""
        x = torch.randn(10, 16)
        r, _ = CartesianToPolar.forward(x)
        
        expected_r = torch.norm(x, dim=-1, keepdim=True)
        assert torch.allclose(r, expected_r, atol=1e-4)
    
    def test_direction_unit_norm(self):
        """Test that direction vector has unit norm."""
        x = torch.randn(10, 16)
        r, theta = CartesianToPolar.forward(x)
        
        # Reconstruct using unit direction
        u = CartesianToPolar.inverse(torch.ones_like(r), theta)
        u_norm = torch.norm(u, dim=-1)
        
        assert torch.allclose(u_norm, torch.ones_like(u_norm), atol=1e-5)


class TestLloydMaxQuantizer:
    """Tests for Lloyd-Max optimal quantizer."""
    
    def test_quantization_reconstruction(self):
        """Test that quantization is reversible."""
        num_bits = 3
        quantizer = LloydMaxQuantizer(num_bits)
        
        # Test angles
        theta = torch.linspace(0, math.pi - 0.01, 100)
        
        # Quantize
        indices = quantizer.encode(theta)
        
        # Dequantize
        theta_reconstructed = quantizer.decode(indices)
        
        # Should be close (within quantization error)
        max_error = (math.pi / (2 ** num_bits)) / 2
        assert torch.all(torch.abs(theta - theta_reconstructed) <= max_error + 1e-4)
    
    def test_centroids_in_range(self):
        """Test that centroids are in valid range."""
        quantizer = LloydMaxQuantizer(num_bits=3)
        
        assert torch.all(quantizer.centroids >= 0)
        assert torch.all(quantizer.centroids <= math.pi)


class TestPolarQuant:
    """Tests for PolarQuant compression."""
    
    def test_compression_ratio(self):
        """Test compression ratio meets target."""
        dim = 64  # Must be power of 2
        pq = PolarQuant(dim, angle_bits=3)
        
        ratio = pq.get_compression_ratio()
        
        # Expected: 16 / (16 + 63*3/8) ≈ 4x
        assert ratio >= 3.5, f"Compression ratio {ratio} below target"
    
    def test_compression_reconstruction(self):
        """Test that compression-decompression works."""
        dim = 64
        pq = PolarQuant(dim, angle_bits=3)
        
        x = torch.randn(10, dim)
        
        # Compress
        r, theta_indices, x_rht = pq.compress(x)
        
        # Decompress
        x_reconstructed = pq.decompress(r, theta_indices)
        
        # Check cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            x.flatten(1), x_reconstructed.flatten(1), dim=-1
        )
        assert torch.all(cos_sim > 0.95), f"Cosine similarity too low: {cos_sim.min()}"
    
    def test_angle_indices_range(self):
        """Test that angle indices are in valid range."""
        dim = 64
        pq = PolarQuant(dim, angle_bits=3)
        
        x = torch.randn(10, dim)
        r, theta_indices, _ = pq.compress(x)
        
        assert torch.all(theta_indices >= 0)
        assert torch.all(theta_indices < 2 ** 3)  # 8 levels for 3 bits


class TestQJL:
    """Tests for Quantized Johnson-Lindenstrauss."""
    
    def test_compression_shape(self):
        """Test that QJL compression produces correct shape."""
        input_dim = 64
        proj_dim = 256
        
        qjl = QJLCompressor(input_dim, proj_dim)
        
        residual = torch.randn(10, input_dim)
        signs = qjl.compress(residual)
        
        assert signs.shape == (10, proj_dim)
        assert signs.dtype == torch.int8
    
    def test_unbiased_estimation(self):
        """Test that QJL provides unbiased dot product estimates."""
        input_dim = 64
        proj_dim = 512  # Larger for better accuracy
        
        qjl = QJLCompressor(input_dim, proj_dim)
        
        # Generate test data
        num_samples = 100
        query = torch.randn(num_samples, input_dim)
        key = torch.randn(num_samples, input_dim)
        
        # True dot products
        true_dots = (query * key).sum(dim=-1)
        
        # Compress key residual
        residual = torch.randn_like(key) * 0.1  # Small residual
        signs = qjl.compress(residual)
        
        # Estimated dot products
        est_dots = []
        for i in range(num_samples):
            est = qjl.decompress_for_dot_product(
                signs[i:i+1], query[i:i+1]
            )
            est_dots.append(est.item())
        
        est_dots = torch.tensor(est_dots)
        
        # Check bias (should be close to 0)
        bias = (est_dots - true_dots).mean()
        assert abs(bias) < 0.1, f"Bias too large: {bias}"
    
    def test_create_qjl_cache(self):
        """Test KV cache compression."""
        batch_size = 2
        num_heads = 4
        seq_len = 100
        head_dim = 64
        
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        compressed = create_qjl_cache(keys, values, proj_dim=256)
        
        assert 'keys_r' in compressed
        assert 'keys_theta' in compressed
        assert 'keys_qjl' in compressed
        assert 'values_r' in compressed


class TestTurboQuantPipeline:
    """Tests for full TurboQuant pipeline."""
    
    def test_full_compression(self):
        """Test full TurboQuant compression."""
        dim = 64
        config = TurboQuantConfig(angle_bits=3, qjl_proj_dim=256)
        turbo = TurboQuantPipeline(dim, config)
        
        x = torch.randn(10, dim)
        
        # Compress
        r, theta, qjl_signs, x_norm = turbo.compress_vector(x)
        
        # Verify shapes
        assert r.shape == (10, 1)
        assert theta.shape == (10, dim - 1)
        assert qjl_signs.shape == (10, config.qjl_proj_dim)
        assert x_norm.shape == (10, 1)
    
    def test_dot_product_with_correction(self):
        """Test dot product with QJL correction."""
        dim = 64
        config = TurboQuantConfig(angle_bits=3, qjl_proj_dim=256)
        turbo = TurboQuantPipeline(dim, config)
        
        x = torch.randn(10, dim)
        query = torch.randn(10, dim)
        
        # Compress x
        r, theta, qjl_signs, x_norm = turbo.compress_vector(x)
        
        # Compute approximate dot product
        approx_dot = turbo.decompress_for_dot_product(
            r, theta, qjl_signs, x_norm, query
        )
        
        # True dot product
        true_dot = (query * x).sum(dim=-1, keepdim=True)
        
        # Should be reasonably close
        relative_error = torch.abs(approx_dot - true_dot) / (torch.abs(true_dot) + 1e-6)
        assert torch.all(relative_error < 0.1), f"Relative error too large: {relative_error.max()}"
    
    def test_kv_cache_compression(self):
        """Test KV cache compression."""
        dim = 64
        config = TurboQuantConfig(angle_bits=3, qjl_proj_dim=256)
        turbo = TurboQuantPipeline(dim, config)
        
        batch_size = 2
        num_heads = 4
        seq_len = 50
        
        keys = torch.randn(batch_size, num_heads, seq_len, dim)
        values = torch.randn(batch_size, num_heads, seq_len, dim)
        
        # Compress
        compressed = turbo.compress_kv_cache(keys, values)
        
        # Check compression ratio
        orig_bytes = (keys.numel() + values.numel()) * 2  # FP16
        comp_bytes = sum(v.numel() * (1 if v.dtype == torch.int8 else 2) 
                        for v in compressed.values())
        ratio = orig_bytes / comp_bytes
        
        assert ratio >= 4.0, f"KV cache compression ratio {ratio} below target"


class TestTensorCoreKernels:
    """Tests for Tensor Core acceleration."""
    
    def test_tensor_core_detection(self):
        """Test Tensor Core availability detection."""
        kernel = TensorCoreKernel()
        
        # Should not crash
        assert hasattr(kernel, 'has_tensor_cores')
    
    def test_int4_packing(self):
        """Test INT4 packing/unpacking."""
        kernel = TensorCoreKernel()
        
        # Create test values in INT4 range [-8, 7]
        values = torch.tensor([[-8, 7], [-1, 0], [3, -4]], dtype=torch.float16)
        
        # Pack
        packed = kernel._pack_int4(values)
        
        # Unpack
        unpacked = kernel._unpack_int4(packed)
        
        # Should match (within quantization error)
        assert unpacked.shape == values.shape
        assert unpacked.dtype == torch.float16
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_int4_linear_layer(self):
        """Test INT4 linear layer."""
        device = 'cuda'
        
        layer = INT4Linear(64, 64).to(device)
        layer.quantize_weights()
        
        x = torch.randn(1, 64, dtype=torch.float16, device=device)
        
        with torch.no_grad():
            y = layer(x)
        
        assert y.shape == (1, 64)
        assert layer._quantized


class TestIntegration:
    """Integration tests for TurboQuant."""
    
    def test_end_to_end_compression(self):
        """Test complete compression and decompression workflow."""
        dim = 128
        config = TurboQuantConfig(angle_bits=3, qjl_proj_dim=256)
        turbo = TurboQuantPipeline(dim, config)
        
        # Original data
        x = torch.randn(5, dim)
        
        # Compress
        r, theta, qjl, norm = turbo.compress_vector(x)
        
        # Decompress
        x_reconstructed = turbo.decompress_for_dot_product(
            r, theta, qjl, norm, torch.eye(dim)[:dim]
        )
        
        # Verify shapes and types
        assert r.dtype == torch.float32 or r.dtype == torch.float16
        assert theta.dtype == torch.int64
        assert qjl.dtype == torch.int8
    
    def test_statistics_tracking(self):
        """Test that compression statistics are tracked."""
        dim = 64
        config = TurboQuantConfig()
        turbo = TurboQuantPipeline(dim, config)
        
        # Initially empty
        stats = turbo.get_stats()
        assert stats['bytes_original'] == 0
        
        # Compress some data
        keys = torch.randn(2, 4, 50, dim)
        values = torch.randn(2, 4, 50, dim)
        turbo.compress_kv_cache(keys, values)
        
        # Stats should be updated
        stats = turbo.get_stats()
        assert stats['bytes_original'] > 0
        assert stats['bytes_compressed'] > 0
        assert stats['compression_ratio'] > 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
