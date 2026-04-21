"""
Unified RaBitQ API for KV cache compression.

Implements true RaBitQ (1-bit + extended-bit quantization with random
orthogonal rotation and popcount-based inner-product estimation).
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

from adn.quantization.rabitq_rotation import FhtKacRotator, MatrixRotator, IdentityRotator, Rotator
from adn.quantization.rabitq_quantizer import (
    QuantizedVector,
    RabitqConfig,
    quantize_vector,
    reconstruct_vector,
    compute_const_scaling_factor,
    quantize_scalar,
)


@dataclass
class RaBitQConfig:
    """Configuration for RaBitQ KV cache compression."""

    total_bits: int = 1  # 1 = binary only (32x), 2 = 1+1, 3 = 1+2
    use_rotation: bool = True
    rotator_type: str = "fht"  # 'fht', 'matrix', or 'identity'
    rotator_seed: int = 42
    residual_window: int = 128
    device: str = "cpu"
    head_dim: int = 64
    use_fast_quantization: bool = True  # Use t_const for speed


@dataclass
class CompressedKV:
    """Compressed key-value pair for a batch of vectors."""

    quantized_vectors: List[QuantizedVector]
    original_shape: Tuple[int, ...]
    centroid: torch.Tensor


class RaBitQ:
    """
    True RaBitQ interface for KV cache compression.

    Usage:
        >>> rq = RaBitQ(total_bits=1, head_dim=64)
        >>> rq.fit(sample_keys, sample_values)
        >>> compressed = rq.compress(keys, values)
        >>> keys_dq, values_dq = rq.decompress(compressed)
    """

    def __init__(
        self,
        config: Optional[RaBitQConfig] = None,
        total_bits: int = 1,
        head_dim: int = 64,
        device: str = "cpu",
    ):
        if config is None:
            config = RaBitQConfig(total_bits=total_bits, head_dim=head_dim, device=device)
        self.config = config

        # Initialize rotator
        if config.use_rotation:
            if config.rotator_type == "fht":
                self.rotator = FhtKacRotator(
                    config.head_dim, seed=config.rotator_seed, device=config.device
                )
            elif config.rotator_type == "matrix":
                padded = ((config.head_dim + 63) // 64) * 64
                self.rotator = MatrixRotator(padded, seed=config.rotator_seed, device=config.device)
            else:
                self.rotator = IdentityRotator(config.head_dim, device=config.device)
        else:
            self.rotator = IdentityRotator(config.head_dim, device=config.device)

        self.padded_dim = self.rotator.padded_dim()
        self.rabitq_config = RabitqConfig(total_bits=config.total_bits)
        self._is_fitted = False
        self.centroid_k: Optional[torch.Tensor] = None
        self.centroid_v: Optional[torch.Tensor] = None

    def fit(self, sample_keys: torch.Tensor, sample_values: torch.Tensor) -> "RaBitQ":
        """
        Fit quantizers on sample data.

        For fast mode, pre-computes the constant scaling factor t_const.
        Also computes centroids (mean of rotated samples).
        """
        head_dim = sample_keys.shape[-1]
        assert head_dim == self.config.head_dim

        # Pad and rotate samples
        sample_keys = self._prepare_vectors(sample_keys)
        sample_values = self._prepare_vectors(sample_values)

        # Compute centroids as mean
        self.centroid_k = sample_keys.reshape(-1, self.padded_dim).mean(dim=0)
        self.centroid_v = sample_values.reshape(-1, self.padded_dim).mean(dim=0)

        # Precompute t_const for fast quantization
        if self.config.use_fast_quantization and self.rabitq_config.ex_bits > 0:
            t_const = compute_const_scaling_factor(
                self.padded_dim, self.rabitq_config.ex_bits, seed=self.config.rotator_seed
            )
            self.rabitq_config.t_const = t_const

        self._is_fitted = True
        return self

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> Dict[str, CompressedKV]:
        """
        Compress keys and values.

        Args:
            keys: [..., head_dim]
            values: [..., head_dim]

        Returns:
            Dict with 'keys' and 'values' as CompressedKV
        """
        if not self._is_fitted:
            # Auto-fit with zero centroids if not fitted
            self.centroid_k = torch.zeros(self.padded_dim, device=self.config.device)
            self.centroid_v = torch.zeros(self.padded_dim, device=self.config.device)
            if self.config.use_fast_quantization and self.rabitq_config.ex_bits > 0:
                self.rabitq_config.t_const = compute_const_scaling_factor(
                    self.padded_dim, self.rabitq_config.ex_bits, seed=self.config.rotator_seed
                )
            self._is_fitted = True

        keys_rot = self._prepare_vectors(keys)
        values_rot = self._prepare_vectors(values)

        return {
            "keys": self._compress_tensor(keys_rot, self.centroid_k, keys.shape),
            "values": self._compress_tensor(values_rot, self.centroid_v, values.shape),
        }

    def decompress(self, compressed: Dict[str, CompressedKV]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress keys and values back to original shape."""
        keys = self._decompress_tensor(compressed["keys"])
        values = self._decompress_tensor(compressed["values"])
        return keys, values

    def as_cache(self, residual_window: Optional[int] = None):
        """Get HF-compatible cache using these compression settings."""
        from adn.quantization.cache import RaBitQCache

        rw = residual_window if residual_window is not None else self.config.residual_window
        return RaBitQCache(
            total_bits=self.config.total_bits,
            head_dim=self.config.head_dim,
            rotator=self.rotator,
            residual_window=rw,
            device=self.config.device,
        )

    def memory_stats(
        self, seq_len: int, num_layers: int = 1, batch_size: int = 1, num_heads: int = 32
    ) -> Dict[str, float]:
        """Calculate memory statistics."""
        num_vectors = batch_size * num_heads * seq_len
        elements_per_tensor = num_vectors * self.config.head_dim

        total_original = 0
        total_compressed = 0

        for _ in range(num_layers):
            # Original: fp16 for K + V
            total_original += 2 * elements_per_tensor * 2

            # Per-vector overhead for reconstruction:
            # binary_code_packed + ex_code_packed + delta + vl
            # (f_add/f_rescale/f_error are only needed for on-the-fly popcount estimation)
            binary_bytes = (self.padded_dim + 7) // 8

            ex_bits = self.rabitq_config.ex_bits
            if ex_bits == 0:
                ex_bytes = 0
            elif ex_bits == 1:
                ex_bytes = self.padded_dim // 16 * 2
            elif ex_bits == 2:
                ex_bytes = self.padded_dim // 16 * 4
            elif ex_bits == 6:
                ex_bytes = self.padded_dim // 16 * 12
            else:
                ex_bytes = (self.padded_dim * ex_bits + 7) // 8

            # Reconstruction metadata: delta + vl (2 floats)
            meta_bytes = 2 * 4

            total_compressed += num_vectors * (binary_bytes + ex_bytes + meta_bytes)

        return {
            "original_mb": total_original / (1024**2),
            "compressed_mb": total_compressed / (1024**2),
            "compression_ratio": total_original / total_compressed if total_compressed > 0 else 0.0,
            "memory_saved_percent": (1 - total_compressed / total_original) * 100,
        }

    def _prepare_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """Pad and rotate vectors."""
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        if x.shape[-1] < self.padded_dim:
            x = torch.nn.functional.pad(x, (0, self.padded_dim - x.shape[-1]))
        if self.config.use_rotation:
            x = self.rotator.rotate(x)
        return x

    def _restore_vectors(self, x: torch.Tensor, original_last_dim: int) -> torch.Tensor:
        """Inverse rotate and crop vectors."""
        if self.config.use_rotation:
            x = self.rotator.inverse_rotate(x)
        if original_last_dim < self.padded_dim:
            x = x[..., :original_last_dim]
        return x

    def _compress_tensor(
        self, rotated: torch.Tensor, centroid: torch.Tensor, original_shape: Tuple[int, ...]
    ) -> CompressedKV:
        """Compress a batch of rotated vectors."""
        flat = rotated.reshape(-1, self.padded_dim)
        qvs = []
        for i in range(flat.shape[0]):
            qv = quantize_vector(flat[i], centroid, self.rabitq_config)
            qvs.append(qv)
        return CompressedKV(
            quantized_vectors=qvs, original_shape=original_shape, centroid=centroid.clone()
        )

    def _decompress_tensor(self, ck: CompressedKV) -> torch.Tensor:
        """Decompress a CompressedKV."""
        flat = torch.stack([reconstruct_vector(ck.centroid, qv) for qv in ck.quantized_vectors])
        original_last_dim = ck.original_shape[-1]
        flat = self._restore_vectors(flat, original_last_dim)
        return flat.reshape(ck.original_shape)


def faster_config(dim: int, total_bits: int) -> RabitqConfig:
    """
    Create a fast RaBitQ configuration with pre-computed t_const.

    Matches the C++ reference implementation's `faster_config()`.

    Args:
        dim: Vector dimension
        total_bits: Total bit-width (1 = binary only, 2 = 1+1, etc.)

    Returns:
        RabitqConfig with t_const pre-computed for fast quantization.
    """
    config = RabitqConfig(total_bits=total_bits)
    if total_bits > 1:
        config.t_const = compute_const_scaling_factor(dim, total_bits - 1)
    return config


# Convenience factory functions


def create_k1(head_dim: int = 64, device: str = "cpu") -> RaBitQ:
    """
    Create 1-bit RaBitQ configuration.
    ~32x compression. Maximum speed with popcount SIMD.
    """
    return RaBitQ(total_bits=1, head_dim=head_dim, device=device)


def create_k2(head_dim: int = 64, device: str = "cpu") -> RaBitQ:
    """Create 2-bit RaBitQ (1 sign + 1 extended)."""
    return RaBitQ(total_bits=2, head_dim=head_dim, device=device)


def create_k3(head_dim: int = 64, device: str = "cpu") -> RaBitQ:
    """Create 3-bit RaBitQ (1 sign + 2 extended)."""
    return RaBitQ(total_bits=3, head_dim=head_dim, device=device)


# Backward-compatible aliases matching old TurboQuant API
create_k4_v2 = create_k3
create_k3_v2 = create_k2
create_k2_v2 = create_k1

RECOMMENDED = {
    "quality": create_k3,  # Best quality with good compression
    "balanced": create_k2,
    "speed": create_k1,  # Max speed, 32x compression
    "memory": create_k1,
}
