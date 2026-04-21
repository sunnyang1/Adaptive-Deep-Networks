"""
Unit tests for true RaBitQ implementation.

Tests cover:
- Rotation (FWHT-Kac, Matrix QR)
- Bit packing (binary + extended codes)
- Quantization (1-bit + extended bits)
- API (compress/decompress)
- Cache (HF integration)
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rabitq import (
    RaBitQ,
    RaBitQConfig,
    FhtKacRotator,
    MatrixRotator,
    IdentityRotator,
    QuantizedVector,
    RabitqConfig,
    quantize_vector,
    reconstruct_vector,
    compute_const_scaling_factor,
    quantize_scalar,
    dequantize_scalar,
    RaBitQCache,
    create_k1,
    create_k2,
    create_k3,
    RECOMMENDED,
    make_full_single_query,
    full_est_dist,
    split_single_estdist,
    split_single_fulldist,
)
from src.rabitq.packing import (
    pack_binary_code,
    unpack_binary_code,
    pack_ex_code_cpp_compat,
    unpack_ex_code_cpp_compat,
)


class TestRotation:
    """Test random orthogonal rotators."""

    def test_fht_kac_norm_preservation(self):
        rot = FhtKacRotator(dim=64, seed=42)
        x = torch.randn(10, 64)
        x_rot = rot.rotate(x)
        assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), rtol=1e-4, atol=1e-5)

    def test_fht_kac_invertibility(self):
        rot = FhtKacRotator(dim=64, seed=42)
        x = torch.randn(10, 64)
        x_rot = rot.rotate(x)
        x_rec = rot.inverse_rotate(x_rot)
        assert torch.allclose(x, x_rec, rtol=1e-4, atol=1e-4)

    def test_fht_kac_padding(self):
        rot = FhtKacRotator(dim=60, seed=42)
        assert rot.padded_dim() >= 64  # padded to power-of-2 and multiple of 64
        x = torch.randn(10, 60)
        x_rot = rot.rotate(x)
        x_rec = rot.inverse_rotate(x_rot)
        assert x_rec.shape[-1] == 60
        assert torch.allclose(x, x_rec, rtol=1e-4, atol=1e-4)

    def test_matrix_rotator_orthogonality(self):
        rot = MatrixRotator(dim=64, seed=42)
        x = torch.randn(10, 64)
        x_rot = rot.rotate(x)
        x_rec = rot.inverse_rotate(x_rot)
        assert torch.allclose(x, x_rec, rtol=1e-4, atol=1e-4)
        assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), rtol=1e-4, atol=1e-5)

    def test_identity_rotator(self):
        rot = IdentityRotator(dim=64)
        x = torch.randn(10, 64)
        assert torch.equal(rot.rotate(x), x)


class TestBitPacking:
    """Test bit-packing utilities."""

    def test_binary_pack_unpack(self):
        dim = 128
        binary = (torch.randn(dim) >= 0).to(torch.uint8)
        packed = pack_binary_code(binary)
        unpacked = unpack_binary_code(packed, dim)
        assert torch.equal(binary, unpacked)

    @pytest.mark.parametrize("ex_bits", [1, 2, 6])
    def test_ex_code_pack_unpack(self, ex_bits):
        dim = 128  # must be multiple of 16 for cpp compat
        ex_code = torch.randint(0, 1 << ex_bits, (dim,), dtype=torch.int16)
        packed = pack_ex_code_cpp_compat(ex_code.unsqueeze(0), ex_bits)
        unpacked = unpack_ex_code_cpp_compat(packed, dim, ex_bits).squeeze(0)
        assert torch.equal(ex_code, unpacked)

    def test_ex_code_generic_fallback(self):
        dim = 64
        ex_bits = 3
        ex_code = torch.randint(0, 1 << ex_bits, (dim,), dtype=torch.int16)
        from src.rabitq.packing import pack_ex_code_generic, unpack_ex_code_generic

        packed = pack_ex_code_generic(ex_code.unsqueeze(0), ex_bits)
        unpacked = unpack_ex_code_generic(packed, dim, ex_bits).squeeze(0)
        assert torch.equal(ex_code, unpacked)


class TestQuantizer:
    """Test true RaBitQ quantizer."""

    def test_1bit_quantize_reconstruct(self):
        dim = 64
        data = torch.randn(dim)
        centroid = torch.zeros(dim)
        config = create_k1(head_dim=dim).rabitq_config
        qv = quantize_vector(data, centroid, config)
        rec = reconstruct_vector(centroid, qv)
        assert rec.shape == data.shape
        # 1-bit has high distortion but should preserve direction roughly
        cos_sim = torch.nn.functional.cosine_similarity(data.unsqueeze(0), rec.unsqueeze(0), dim=-1)
        assert cos_sim.item() > 0.3

    def test_3bit_quantize_reconstruct(self):
        dim = 64
        data = torch.randn(dim)
        centroid = torch.zeros(dim)
        config = create_k3(head_dim=dim).rabitq_config
        qv = quantize_vector(data, centroid, config)
        rec = reconstruct_vector(centroid, qv)
        assert rec.shape == data.shape
        cos_sim = torch.nn.functional.cosine_similarity(data.unsqueeze(0), rec.unsqueeze(0), dim=-1)
        assert cos_sim.item() > 0.7

    def test_quantized_vector_fields(self):
        dim = 64
        data = torch.randn(dim)
        centroid = torch.zeros(dim)
        config = create_k2(head_dim=dim).rabitq_config
        qv = quantize_vector(data, centroid, config)
        assert isinstance(qv, QuantizedVector)
        assert qv.dim == dim
        assert qv.ex_bits == 1
        assert qv.binary_code_packed.numel() == (dim + 7) // 8

    def test_const_scaling_factor(self):
        t = compute_const_scaling_factor(dim=64, ex_bits=2, num_samples=20)
        assert t > 0.0


class TestRaBitQAPI:
    """Test RaBitQ main API."""

    @pytest.fixture
    def sample_kv(self):
        return torch.randn(2, 4, 128, 64), torch.randn(2, 4, 128, 64)

    @pytest.mark.parametrize("factory", [create_k1, create_k2, create_k3])
    def test_compress_decompress_roundtrip(self, factory, sample_kv):
        keys, values = sample_kv
        rq = factory(head_dim=64)
        rq.fit(keys[:1, :1, :32], values[:1, :1, :32])
        compressed = rq.compress(keys, values)
        keys_dq, values_dq = rq.decompress(compressed)
        assert keys_dq.shape == keys.shape
        assert values_dq.shape == values.shape

    def test_k1_compression_ratio(self, sample_kv):
        keys, values = sample_kv
        rq = create_k1(head_dim=64)
        stats = rq.memory_stats(seq_len=128, num_layers=1, batch_size=2, num_heads=4)
        # Note: true 1-bit compression with per-vector reconstruction metadata
        # achieves ~3-8x per-tensor vs fp16 depending on head_dim and overhead.
        assert stats["compression_ratio"] > 3.0

    def test_memory_stats_monotonic(self):
        rq = create_k2(head_dim=64)
        stats1 = rq.memory_stats(seq_len=128, num_layers=1, num_heads=8)
        stats2 = rq.memory_stats(seq_len=256, num_layers=1, num_heads=8)
        assert stats2["compressed_mb"] > stats1["compressed_mb"]

    def test_all_recommended_configs(self):
        for name, factory in RECOMMENDED.items():
            rq = factory(head_dim=64)
            assert isinstance(rq, RaBitQ)


class TestEstimatorAligned:
    """Tests for estimator.py aligned with RaBitQ-Library reference."""

    def test_full_est_dist_ip(self):
        dim = 64
        data = torch.randn(dim)
        query = torch.randn(dim)
        centroid = torch.zeros(dim)
        config = RabitqConfig(total_bits=1, metric_type="ip")
        qv = quantize_vector(data, centroid, config)
        q = make_full_single_query(query, centroid, metric_type="ip", total_bits=1)
        bin_code = unpack_binary_code(qv.binary_code_packed, qv.dim)
        ip = float((q.rotated_query * bin_code.float()).sum().item())
        est_neg_ip, _ = full_est_dist(ip, qv.f_add, qv.f_rescale, q)
        est_ip = -est_neg_ip
        true_ip = torch.dot(query, data).item()
        rel_err = abs(est_ip - true_ip) / (abs(true_ip) + 1e-6)
        assert rel_err < 0.15

    def test_full_est_dist_l2(self):
        dim = 64
        data = torch.randn(dim)
        query = torch.randn(dim)
        centroid = torch.zeros(dim)
        config = RabitqConfig(total_bits=1, metric_type="l2")
        qv = quantize_vector(data, centroid, config)
        q = make_full_single_query(query, centroid, metric_type="l2", total_bits=1)
        bin_code = unpack_binary_code(qv.binary_code_packed, qv.dim)
        ip = float((q.rotated_query * bin_code.float()).sum().item())
        est_dist, _ = full_est_dist(ip, qv.f_add, qv.f_rescale, q)
        true_dist = ((query - data) ** 2).sum().item()
        rel_err = abs(est_dist - true_dist) / (abs(true_dist) + 1e-6)
        assert rel_err < 0.15

    def test_split_single_incremental(self):
        dim = 64
        data = torch.randn(dim)
        query = torch.randn(dim)
        centroid = torch.zeros(dim)
        config = RabitqConfig(total_bits=3, metric_type="ip")
        qv = quantize_vector(data, centroid, config)
        q = make_full_single_query(query, centroid, metric_type="ip", total_bits=3)
        bin_code = unpack_binary_code(qv.binary_code_packed, qv.dim)
        ip_x0_qr, est_coarse, low_coarse = split_single_estdist(
            qv.f_add, qv.f_rescale, qv.f_error, bin_code, q
        )
        if qv.ex_bits > 0:
            ex_code = unpack_ex_code_cpp_compat(
                qv.ex_code_packed.unsqueeze(0), dim, qv.ex_bits
            ).squeeze(0)
            est_fine, low_fine = split_single_fulldist(
                qv.f_add_ex, qv.f_rescale_ex, qv.f_error, ex_code, qv.ex_bits, ip_x0_qr, q
            )
            true_ip = torch.dot(query, data).item()
            rel_err_fine = abs(-est_fine - true_ip) / (abs(true_ip) + 1e-6)
            assert rel_err_fine < 0.10

    def test_quantize_scalar_format1(self):
        dim = 64
        data = torch.randn(dim)
        bits = 4
        code, delta, vl = quantize_scalar(data, bits)
        recon = dequantize_scalar(code, delta, vl)
        max_err = (data - recon).abs().max().item()
        assert max_err <= delta * 1.1

    def test_faster_config(self):
        from src.rabitq.api import faster_config

        cfg = faster_config(64, 3)
        assert cfg.total_bits == 3
        assert cfg.t_const is not None
        assert cfg.t_const > 0


class TestRaBitQCache:
    """Test HF-compatible cache."""

    def test_cache_basic(self):
        cache = RaBitQCache(total_bits=1, head_dim=64, residual_window=0)
        batch, heads, seq, dim = 1, 8, 64, 64
        keys = torch.randn(batch, heads, seq, dim)
        values = torch.randn(batch, heads, seq, dim)
        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape == keys.shape
        assert out_v.shape == values.shape
        assert cache.get_seq_length(0) == seq

    def test_cache_accumulation(self):
        cache = RaBitQCache(total_bits=1, head_dim=64, residual_window=0)
        keys1 = torch.randn(1, 8, 32, 64)
        values1 = torch.randn(1, 8, 32, 64)
        cache.update(keys1, values1, layer_idx=0)
        keys2 = torch.randn(1, 8, 32, 64)
        values2 = torch.randn(1, 8, 32, 64)
        out_k, out_v = cache.update(keys2, values2, layer_idx=0)
        assert out_k.shape[2] == 64
        assert cache.get_seq_length(0) == 64

    def test_cache_residual_window(self):
        cache = RaBitQCache(total_bits=1, head_dim=64, residual_window=32)
        keys = torch.randn(1, 8, 64, 64)
        values = torch.randn(1, 8, 64, 64)
        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape[2] == 64
        assert len(cache._recent_k[0]) > 0

    def test_cache_multiple_layers(self):
        cache = RaBitQCache(total_bits=1, head_dim=64, residual_window=16)
        for layer in range(3):
            keys = torch.randn(1, 4, 32, 64)
            values = torch.randn(1, 4, 32, 64)
            cache.update(keys, values, layer_idx=layer)
        for layer in range(3):
            assert cache.get_seq_length(layer) == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
