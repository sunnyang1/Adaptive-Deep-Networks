"""
Tests for Refactored TurboQuant Core API
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.turboquant import (
    TurboQuant,
    TurboQuantConfig,
    QuantMode,
    LloydMaxQuantizer,
    INT8Quantizer,
    FP16Quantizer,
    RECOMMENDED_CONFIGS,
)


class TestTurboQuantConfig:
    """Test configuration class."""

    def test_mode_parsing(self):
        """Test mode string parsing."""
        # Basic modes
        config = TurboQuantConfig(mode="fp16")
        assert config.kv_quant_mode == QuantMode.FP16
        assert config.flash_attention == False

        config = TurboQuantConfig(mode="int8")
        assert config.kv_quant_mode == QuantMode.KV_INT8

        config = TurboQuantConfig(mode="tq4")
        assert config.kv_quant_mode == QuantMode.KV_TQ4

    def test_flash_attention_suffix(self):
        """Test flash attention suffix parsing."""
        config = TurboQuantConfig(mode="tq4_flash")
        assert config.flash_attention == True
        assert config.kv_quant_mode == QuantMode.KV_TQ4

        config = TurboQuantConfig(mode="flash_tq3")
        assert config.flash_attention == True
        assert config.kv_quant_mode == QuantMode.KV_TQ3

    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError):
            TurboQuantConfig(mode="invalid")

    def test_compression_ratios(self):
        """Test compression ratio calculations."""
        test_cases = [
            ("fp16", 1.0),
            ("int8", 2.0),
            ("tq4", 3.0),
            ("tq3", 4.0),
        ]

        for mode, expected in test_cases:
            config = TurboQuantConfig(mode=mode)
            assert config.compression_ratio == expected

    def test_memory_stats(self):
        """Test memory statistics calculation."""
        config = TurboQuantConfig(mode="int8")
        stats = config.memory_stats(seq_len=4096, batch_size=2, num_heads=8)

        assert "original_mb" in stats
        assert "compressed_mb" in stats
        assert "compression_ratio" in stats
        assert "memory_saved" in stats
        assert stats["compression_ratio"] == 2.0


class TestQuantizers:
    """Test individual quantizers."""

    def test_lloyd_max_fit(self):
        """Test Lloyd-Max quantizer fitting."""
        quantizer = LloydMaxQuantizer(num_bits=4)

        data = torch.randn(1000, 32)
        quantizer.fit(data)

        assert quantizer._fitted
        assert len(quantizer.codebook) == 16  # 2^4

    def test_lloyd_max_quantize(self):
        """Test Lloyd-Max quantization."""
        quantizer = LloydMaxQuantizer(num_bits=4)

        train_data = torch.randn(1000, 32)
        quantizer.fit(train_data)

        test_data = torch.randn(100, 32)
        indices, dequantized = quantizer.quantize(test_data)

        assert indices.shape == test_data.shape
        assert dequantized.shape == test_data.shape

        # Check error is reasonable
        error = (test_data - dequantized).abs().mean()
        assert error < 0.1

    def test_lloyd_max_beta_fit(self):
        """Test Beta distribution fitting."""
        quantizer = LloydMaxQuantizer(num_bits=3)
        quantizer.fit_beta(alpha=2.0, beta=2.0)

        assert quantizer._fitted
        assert len(quantizer.codebook) == 8

    def test_int8_quantizer(self):
        """Test INT8 quantizer."""
        quantizer = INT8Quantizer()

        data = torch.randn(10, 32)
        indices, dequantized = quantizer.quantize(data)

        assert indices.dtype == torch.int8
        assert dequantized.shape == data.shape

    def test_fp16_quantizer(self):
        """Test FP16 quantizer."""
        quantizer = FP16Quantizer()

        data = torch.randn(10, 32)
        indices, dequantized = quantizer.quantize(data)

        assert indices.dtype == torch.float16


class TestTurboQuant:
    """Test main TurboQuant class."""

    @pytest.fixture
    def sample_kv(self):
        """Create sample KV cache."""
        return torch.randn(2, 4, 64, 32), torch.randn(2, 4, 64, 32)

    def test_fp16_mode(self, sample_kv):
        """Test FP16 mode (no compression)."""
        keys, values = sample_kv

        quant = TurboQuant("fp16", head_dim=32)
        compressed = quant.compress_kv(keys, values)
        keys_deq, values_deq = quant.decompress_kv(compressed)

        # Should be nearly identical
        torch.testing.assert_close(keys, keys_deq, rtol=1e-3, atol=1e-3)

    def test_int8_mode(self, sample_kv):
        """Test INT8 mode."""
        keys, values = sample_kv

        quant = TurboQuant("int8", head_dim=32)
        compressed = quant.compress_kv(keys, values)
        keys_deq, values_deq = quant.decompress_kv(compressed)

        assert keys_deq.shape == keys.shape

        error = (keys - keys_deq).abs().mean()
        assert error < 0.1

    def test_tq4_mode(self, sample_kv):
        """Test TQ4 mode."""
        keys, values = sample_kv

        quant = TurboQuant("tq4", head_dim=32)
        quant.fit(keys, values)

        compressed = quant.compress_kv(keys, values)
        keys_deq, values_deq = quant.decompress_kv(compressed)

        assert keys_deq.shape == keys.shape

    def test_fit_beta(self, sample_kv):
        """Test Beta distribution fitting."""
        keys, values = sample_kv

        quant = TurboQuant("tq4", head_dim=32)
        quant.fit_beta()  # Fit on Beta distribution

        compressed = quant.compress_kv(keys, values)
        keys_deq, values_deq = quant.decompress_kv(compressed)

        assert keys_deq.shape == keys.shape

    def test_context_manager(self, sample_kv):
        """Test context manager API."""
        keys, values = sample_kv

        with TurboQuant("tq4", head_dim=32) as quant:
            quant.fit(keys, values)
            compressed = quant.compress_kv(keys, values)
            keys_deq, values_deq = quant.decompress_kv(compressed)

        assert keys_deq.shape == keys.shape

    def test_memory_stats(self):
        """Test memory statistics."""
        quant = TurboQuant("tq4", head_dim=128)
        stats = quant.memory_stats(seq_len=32768, batch_size=1, num_heads=32)

        assert stats["compression_ratio"] == 3.0
        assert stats["memory_saved"] > 60  # >60% saving

    def test_reset_stats(self, sample_kv):
        """Test statistics reset."""
        keys, values = sample_kv

        quant = TurboQuant("fp16")
        quant.compress_kv(keys, values)

        assert quant.stats["keys_compressed"] > 0

        quant.reset_stats()
        assert quant.stats["keys_compressed"] == 0

    def test_flash_attention_flag(self):
        """Test FlashAttention flag."""
        quant = TurboQuant("tq4")
        assert quant.config.flash_attention == False

        quant = TurboQuant("tq4_flash")
        assert quant.config.flash_attention == True


class TestRecommendedConfigs:
    """Test recommended configurations."""

    def test_all_configs_valid(self):
        """Test all recommended configs are valid."""
        for name, mode in RECOMMENDED_CONFIGS.items():
            quant = TurboQuant(mode)
            assert quant.config.compression_ratio > 0

    def test_small_model_config(self):
        """Test small model config."""
        quant = TurboQuant(RECOMMENDED_CONFIGS["small_model"])
        assert quant.config.compression_ratio == 1.0

    def test_fast_config(self):
        """Test fast config."""
        quant = TurboQuant(RECOMMENDED_CONFIGS["fast"])
        assert quant.config.kv_quant_mode == QuantMode.KV_TQ4


class TestIntegration:
    """Integration tests."""

    def test_different_head_dims(self):
        """Test with different head dimensions."""
        head_dims = [32, 64, 128, 256]

        for head_dim in head_dims:
            quant = TurboQuant("int8", head_dim=head_dim)

            keys = torch.randn(1, 4, 32, head_dim)
            values = torch.randn(1, 4, 32, head_dim)

            compressed = quant.compress_kv(keys, values)
            keys_deq, values_deq = quant.decompress_kv(compressed)

            assert keys_deq.shape == keys.shape

    def test_long_sequences(self):
        """Test with long sequences."""
        seq_lengths = [128, 512, 2048, 8192]

        for seq_len in seq_lengths:
            quant = TurboQuant("tq4", head_dim=64)

            keys = torch.randn(1, 8, seq_len, 64)
            values = torch.randn(1, 8, seq_len, 64)

            quant.fit(keys, values)
            compressed = quant.compress_kv(keys, values)
            keys_deq, values_deq = quant.decompress_kv(compressed)

            assert keys_deq.shape == keys.shape

    def test_reconstruction_quality(self):
        """Test reconstruction quality across modes."""
        keys = torch.randn(2, 4, 128, 64)
        values = torch.randn(2, 4, 128, 64)

        # TQ4 should have lower error than TQ3
        quant4 = TurboQuant("tq4", head_dim=64)
        quant4.fit(keys, values)
        compressed4 = quant4.compress_kv(keys, values)
        keys_deq4, _ = quant4.decompress_kv(compressed4)
        error4 = (keys - keys_deq4).abs().mean()

        quant3 = TurboQuant("tq3", head_dim=64)
        quant3.fit(keys, values)
        compressed3 = quant3.compress_kv(keys, values)
        keys_deq3, _ = quant3.decompress_kv(compressed3)
        error3 = (keys - keys_deq3).abs().mean()

        # TQ4 (4-bit) should have less error than TQ3 (3-bit)
        assert error4 < error3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
