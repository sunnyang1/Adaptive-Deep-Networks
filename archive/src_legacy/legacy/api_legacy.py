"""
Unified RaBitQ API.

Simple, clean interface for KV cache compression using RaBitQ
(Rapid and Accurate Bit-level Quantization).
"""

import torch
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from .compressor import MSECompressor, CompressorConfig
from .cache import RaBitQCache, CacheConfig


@dataclass
class RaBitQConfig:
    """Configuration for RaBitQ."""

    key_bits: int = 4
    value_bits: int = 2
    use_rotation: bool = True
    pack_bits: bool = True
    residual_window: int = 128
    device: str = "cpu"
    head_dim: int = 64


class RaBitQ:
    """
    Unified RaBitQ interface for KV cache compression.

    Simple API for compressing/decompressing KV caches:
        >>> rq = RaBitQ(key_bits=4, value_bits=2)
        >>> rq.fit(sample_keys, sample_values)
        >>> compressed = rq.compress(keys, values)
        >>> keys_dq, values_dq = rq.decompress(compressed)

    Or use as HF cache:
        >>> cache = rq.as_cache()
        >>> model.generate(..., past_key_values=cache)
    """

    def __init__(
        self,
        config: Optional[RaBitQConfig] = None,
        key_bits: int = 4,
        value_bits: int = 2,
        head_dim: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize RaBitQ.

        Args:
            config: Full configuration (or specify params)
            key_bits: Bits for key quantization (2-8)
            value_bits: Bits for value quantization (2-8)
            head_dim: Head dimension
            device: 'cpu', 'cuda', or 'mps'
        """
        if config is None:
            config = RaBitQConfig(
                key_bits=key_bits, value_bits=value_bits, head_dim=head_dim, device=device
            )

        self.config = config

        # Create compressors
        key_cfg = CompressorConfig(
            bits=config.key_bits,
            use_rotation=config.use_rotation,
            pack_bits=config.pack_bits,
            device=config.device,
        )
        val_cfg = CompressorConfig(
            bits=config.value_bits,
            use_rotation=config.use_rotation,
            pack_bits=config.pack_bits,
            device=config.device,
        )

        self.key_compressor = MSECompressor(key_cfg)
        self.val_compressor = MSECompressor(val_cfg)

    def fit(self, sample_keys: torch.Tensor, sample_values: torch.Tensor) -> "RaBitQ":
        """
        Fit quantizers on sample data.

        Args:
            sample_keys: Representative key samples
            sample_values: Representative value samples

        Returns:
            self for chaining
        """
        head_dim = sample_keys.shape[-1]
        self.key_compressor.fit(sample_keys, head_dim)
        self.val_compressor.fit(sample_values, head_dim)
        return self

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> Dict[str, any]:
        """
        Compress keys and values.

        Args:
            keys: Key tensor [..., head_dim]
            values: Value tensor [..., head_dim]

        Returns:
            Dict with 'keys' and 'values' compressed data
        """
        head_dim = keys.shape[-1]
        return {
            "keys": self.key_compressor.compress(keys, head_dim),
            "values": self.val_compressor.compress(values, head_dim),
        }

    def decompress(self, compressed: Dict[str, any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress keys and values.

        Args:
            compressed: Output from compress()

        Returns:
            (keys, values) tuple
        """
        keys = self.key_compressor.decompress(compressed["keys"])
        values = self.val_compressor.decompress(compressed["values"])
        return keys, values

    def as_cache(self, residual_window: Optional[int] = None) -> RaBitQCache:
        """
        Get HF-compatible cache using these compression settings.

        Args:
            residual_window: Number of recent tokens to keep in fp16

        Returns:
            RaBitQCache instance
        """
        rw = residual_window if residual_window is not None else self.config.residual_window
        return RaBitQCache(
            key_bits=self.config.key_bits,
            value_bits=self.config.value_bits,
            residual_window=rw,
            device=self.config.device,
        )

    def memory_stats(
        self, seq_len: int, num_layers: int = 1, batch_size: int = 1, num_heads: int = 32
    ) -> Dict[str, float]:
        """
        Calculate memory statistics.

        Args:
            seq_len: Sequence length
            num_layers: Number of layers
            batch_size: Batch size
            num_heads: Number of heads

        Returns:
            Dict with memory statistics
        """
        num_vectors = batch_size * num_heads * seq_len
        elements_per_tensor = num_vectors * self.config.head_dim

        total_original = 0
        total_compressed = 0

        for _ in range(num_layers):
            # Original: fp16 for K + V
            total_original += 2 * elements_per_tensor * 2  # 2 bytes per fp16

            # Compressed indices
            def index_bytes(bits, elements):
                # Efficient packing only for powers of 2
                return (elements * bits + 7) // 8 if (8 % bits == 0) else elements

            kb = index_bytes(self.config.key_bits, elements_per_tensor)
            vb = index_bytes(self.config.value_bits, elements_per_tensor)

            # Norms: fp16 per vector
            nb = 2 * num_vectors * 2  # K norms + V norms

            total_compressed += kb + vb + nb

        return {
            "original_mb": total_original / (1024**2),
            "compressed_mb": total_compressed / (1024**2),
            "compression_ratio": total_original / total_compressed,
            "memory_saved_percent": (1 - total_compressed / total_original) * 100,
        }


# Convenience factory functions


def create_k4_v2(head_dim: int = 64, device: str = "cpu") -> RaBitQ:
    """
    Create K4/V2 configuration (recommended).

    4-bit keys, 2-bit values. ~4.9x compression at 512 seq.
    Best quality with good compression.
    """
    return RaBitQ(key_bits=4, value_bits=2, head_dim=head_dim, device=device)


def create_k3_v2(head_dim: int = 64, device: str = "cpu") -> RaBitQ:
    """
    Create K3/V2 configuration.

    3-bit keys, 2-bit values. ~3.0x compression at 512 seq.
    Lower ratio due to 3-bit packing inefficiency.
    """
    return RaBitQ(key_bits=3, value_bits=2, head_dim=head_dim, device=device)


def create_k2_v2(head_dim: int = 64, device: str = "cpu") -> RaBitQ:
    """
    Create K2/V2 configuration (maximum compression).

    2-bit keys, 2-bit values. ~7.1x compression.
    Higher distortion, use only for memory-constrained scenarios.
    """
    return RaBitQ(key_bits=2, value_bits=2, head_dim=head_dim, device=device)


# Recommended configurations
RECOMMENDED = {
    "quality": create_k4_v2,  # Best quality
    "balanced": create_k4_v2,  # Same as quality
    "speed": create_k3_v2,  # Faster, less compression
    "memory": create_k2_v2,  # Max compression
}
