"""
MSE Compressor: Rotation + Per-Vector Normalization + Lloyd-Max Quantization.

This is the core compression engine for TurboQuant V3.
Based on community findings that QJL hurts attention quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from .rotation import RandomRotation
from .quantizer import LloydMaxQuantizer


def pack_bits(tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Pack low-bit integers into compact storage.

    Only efficient when bits evenly divides 8 (1, 2, 4, 8).
    For other bit widths, stores as uint8 without packing.

    Args:
        tensor: Integer tensor to pack
        bits: Bits per element (1-8)

    Returns:
        (packed_tensor, original_shape) tuple
    """
    if bits > 8:
        raise ValueError(f"Bits must be <= 8, got {bits}")

    original_shape = tensor.shape
    flat = tensor.reshape(-1).cpu().numpy().astype(np.uint8)

    # Only pack if bits evenly divides 8
    if 8 % bits != 0:
        return torch.from_numpy(flat), original_shape

    # Pack bits
    elements_per_byte = 8 // bits
    packed_size = (len(flat) + elements_per_byte - 1) // elements_per_byte

    packed = np.zeros(packed_size, dtype=np.uint8)
    mask = (1 << bits) - 1

    for i, val in enumerate(flat):
        byte_idx = i // elements_per_byte
        bit_offset = (i % elements_per_byte) * bits
        packed[byte_idx] |= (val & mask) << bit_offset

    return torch.from_numpy(packed), original_shape


def unpack_bits(packed: torch.Tensor, bits: int, original_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Unpack bit-packed tensor.

    Args:
        packed: Bit-packed tensor (or uint8 if not packed)
        bits: Bits per element
        original_shape: Original shape before packing

    Returns:
        Unpacked uint8 tensor
    """
    # If bits doesn't divide 8, data was stored unpacked
    if 8 % bits != 0:
        return packed.reshape(original_shape)

    elements_per_byte = 8 // bits
    mask = (1 << bits) - 1

    packed_np = packed.cpu().numpy()
    total_elements = int(np.prod(original_shape))

    flat = np.zeros(total_elements, dtype=np.uint8)
    for i in range(total_elements):
        byte_idx = i // elements_per_byte
        bit_offset = (i % elements_per_byte) * bits
        if byte_idx < len(packed_np):
            flat[i] = (packed_np[byte_idx] >> bit_offset) & mask

    return torch.from_numpy(flat).reshape(original_shape)


@dataclass
class CompressorConfig:
    """Configuration for MSE Compressor."""

    bits: int = 4
    use_rotation: bool = True
    pack_bits: bool = True
    device: str = "cpu"


class MSECompressor:
    """
    MSE-only compressor with per-vector normalization.

    Pipeline:
        1. Per-vector normalization (store norms)
        2. Random rotation (if enabled)
        3. Lloyd-Max quantization
        4. Bit-packing (if enabled)

    Decompression reverses the pipeline and restores norms.
    """

    def __init__(self, config: CompressorConfig):
        self.config = config
        self.quantizer = LloydMaxQuantizer(config.bits, device=config.device)
        self.rotation: Optional[RandomRotation] = None
        self._is_fitted = False

    def fit(self, sample_data: torch.Tensor, head_dim: int) -> "MSECompressor":
        """Fit quantizer on sample data."""
        if self.config.use_rotation:
            from .rotation import _next_power_of_2

            dim = _next_power_of_2(head_dim)
            self.rotation = RandomRotation(dim, device=self.config.device)

            # Normalize then rotate sample data
            sample_norm = sample_data / (sample_data.norm(dim=-1, keepdim=True) + 1e-8)
            sample_padded = self._pad_to_dim(sample_norm, dim)
            sample_rotated = self.rotation.rotate(sample_padded)
        else:
            sample_rotated = sample_data

        # Fit quantizer
        self.quantizer.fit(sample_rotated)
        self._is_fitted = True
        return self

    def compress(self, x: torch.Tensor, head_dim: int) -> Dict[str, torch.Tensor]:
        """Compress tensor."""
        if not self._is_fitted:
            raise RuntimeError("Compressor must be fitted first. Call fit().")

        original_shape = x.shape

        # Per-vector normalization
        x_norm = x.norm(dim=-1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)

        # Rotation
        if self.rotation:
            dim = self.rotation.dim
            x_padded = self._pad_to_dim(x_normalized, dim)
            x_rotated = self.rotation.rotate(x_padded)
        else:
            x_rotated = x_normalized

        # Quantize
        indices, _ = self.quantizer.quantize(x_rotated)

        # Bit-packing
        if self.config.pack_bits:
            indices_packed, pack_shape = pack_bits(indices, self.config.bits)
        else:
            indices_packed = indices
            pack_shape = None

        return {
            "indices": indices_packed,
            "norm": x_norm.half(),
            "pack_shape": pack_shape,
            "original_shape": original_shape,
        }

    def decompress(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decompress tensor."""
        # Unpack bits
        if self.config.pack_bits and compressed.get("pack_shape") is not None:
            indices = unpack_bits(
                compressed["indices"], self.config.bits, compressed["pack_shape"]
            ).to(self.config.device)
        else:
            indices = compressed["indices"]

        # Decode
        x_rotated = self.quantizer.decode(indices)

        # Inverse rotation
        if self.rotation:
            x_normalized = self.rotation.inverse(x_rotated)
            x_normalized = x_normalized[..., : compressed["original_shape"][-1]]
        else:
            x_normalized = x_rotated

        # Restore norm
        x = x_normalized * compressed["norm"].float()
        return x.reshape(compressed["original_shape"])

    def _pad_to_dim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Pad tensor to target dimension."""
        if x.shape[-1] < dim:
            padding = dim - x.shape[-1]
            return F.pad(x, (0, padding))
        return x

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
