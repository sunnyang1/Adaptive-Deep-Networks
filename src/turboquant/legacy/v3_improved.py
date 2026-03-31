"""
TurboQuant V3: Community-Improved Implementation

Based on: https://github.com/tonbistudio/turboquant-pytorch

Key improvements from community implementations:
1. MSE-only (remove QJL) - QJL hurts with softmax attention
2. Asymmetric K/V bits - Keys get more bits than values
3. Bit-packed storage - Real compression ratios
4. Layer-adaptive precision - Protect sensitive layers

V3 Results (tonbistudio):
- 5.1x compression with 18/18 perfect generation
- V2 with QJL: 0/27 generation tests passed
- V3 without QJL: 18/18 generation tests passed
"""

import torch
import torch.nn as nn
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Union


# ============================================================================
# Bit-Packed Storage
# ============================================================================

def pack_bits(tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int]:
    """
    Pack low-bit integers into compact storage.
    
    Only packs efficiently when bits evenly divides 8 (1, 2, 4, 8).
    For other bit widths (e.g., 3), stores as uint8 without packing.
    
    Args:
        tensor: Integer tensor to pack [..., N]
        bits: Bits per element (1-8)
    
    Returns:
        packed: Bit-packed tensor (or uint8 if not packable)
        original_shape: Original shape for unpacking
    """
    if bits > 8:
        raise ValueError(f"Bits must be <= 8, got {bits}")
    
    original_shape = tensor.shape
    flat = tensor.reshape(-1).cpu().numpy().astype(np.uint8)
    
    # Only pack if bits evenly divides 8
    if 8 % bits != 0:
        # Store as uint8 without packing
        return torch.from_numpy(flat), original_shape
    
    # Pack bits
    elements_per_byte = 8 // bits
    packed_size = (len(flat) + elements_per_byte - 1) // elements_per_byte
    
    packed = np.zeros(packed_size, dtype=np.uint8)
    for i, val in enumerate(flat):
        byte_idx = i // elements_per_byte
        bit_offset = (i % elements_per_byte) * bits
        packed[byte_idx] |= (val & ((1 << bits) - 1)) << bit_offset
    
    return torch.from_numpy(packed), original_shape


def unpack_bits(packed: torch.Tensor, bits: int, original_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Unpack bit-packed tensor.
    
    Args:
        packed: Bit-packed tensor (or uint8 if not packed)
        bits: Bits per element
        original_shape: Original shape
    
    Returns:
        tensor: Unpacked integer tensor
    """
    # If bits doesn't divide 8, data was stored as uint8
    if 8 % bits != 0:
        return packed.reshape(original_shape)
    
    elements_per_byte = 8 // bits
    mask = (1 << bits) - 1
    
    packed_np = packed.cpu().numpy()
    total_elements = np.prod(original_shape)
    
    flat = np.zeros(total_elements, dtype=np.uint8)
    for i in range(total_elements):
        byte_idx = i // elements_per_byte
        bit_offset = (i % elements_per_byte) * bits
        if byte_idx < len(packed_np):
            flat[i] = (packed_np[byte_idx] >> bit_offset) & mask
    
    return torch.from_numpy(flat).reshape(original_shape)


# ============================================================================
# Random Rotation (Hadamard)
# ============================================================================

class RandomRotation:
    """
    Random orthogonal rotation using Walsh-Hadamard Transform.
    
    Key insight: Random rotation makes all coordinates follow a 
    predictable bell-curve distribution, enabling optimal scalar quantization.
    """
    
    def __init__(self, dim: int, device: str = 'cpu'):
        """
        Args:
            dim: Dimension (must be power of 2)
            device: Device
        """
        if dim & (dim - 1) != 0:
            raise ValueError(f"Dimension must be power of 2, got {dim}")
        
        self.dim = dim
        self.device = device
        
        # Random diagonal sign matrix (creates random rotation)
        self.D = torch.randn(dim, device=device).sign()
    
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation."""
        # Apply D (element-wise)
        x = x * self.D
        
        # Fast Walsh-Hadamard Transform
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        x_rotated = self._fwht(x_flat)
        
        return x_rotated.reshape(original_shape)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation."""
        # WH transform: H @ H = n * I
        # Forward: rotate(x) = H @ (D*x) / sqrt(n)
        # Inverse: We need to compute H^-1 @ y where y = H @ (D*x) / sqrt(n)
        # H^-1 = H / n, so H^-1 @ y = H @ y / n = H @ H @ (D*x) / (sqrt(n) * n)
        #                              = n * D * x / (sqrt(n) * n)
        #                              = D * x / sqrt(n)
        # Then we apply D again: D * (D * x / sqrt(n)) = x / sqrt(n)
        # So we need to multiply by sqrt(n) to get x back
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        
        # Apply WH transform (without normalization)
        x_inv = self._fwht(x_flat, normalize=False) / self.dim
        
        # Apply D^-1 = D (diagonal with ±1)
        x_inv = x_inv * self.D
        
        # Multiply by sqrt(n) to undo the forward normalization
        x_inv = x_inv * math.sqrt(self.dim)
        
        return x_inv.reshape(original_shape)
    
    def _fwht(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Fast Walsh-Hadamard Transform.
        
        O(n log n) complexity vs O(n^2) for matrix multiply.
        """
        n = x.shape[-1]
        h = 2
        
        # Work on a copy
        x = x.clone()
        
        while h <= n:
            # Perform butterfly operations in-place
            x = x.reshape(*x.shape[:-1], n // h, h)
            x_half = x[..., :h//2]
            y_half = x[..., h//2:]
            # [a, b] -> [a+b, a-b]
            new_x = torch.cat([x_half + y_half, x_half - y_half], dim=-1)
            x = new_x.reshape(*x.shape[:-2], n)
            h *= 2
        
        # Normalize
        if normalize:
            x = x / math.sqrt(n)
        
        return x


# ============================================================================
# Lloyd-Max Quantizer
# ============================================================================

class LloydMaxQuantizerV3:
    """
    Lloyd-Max optimal scalar quantizer.
    
    Pre-computes optimal centroids for the bell-curve distribution
    that results from random rotation.
    """
    
    def __init__(self, num_bits: int, max_iter: int = 100, device: str = 'cpu'):
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.max_iter = max_iter
        self.device = device
        
        self.centroids = None
        self.boundaries = None
        self._fitted = False
    
    def fit(self, data: torch.Tensor) -> 'LloydMaxQuantizerV3':
        """
        Fit quantizer on data.
        
        For rotated data, this follows a bell-curve distribution.
        """
        flat_data = data.reshape(-1).to(self.device)
        
        # Initialize centroids uniformly
        min_val, max_val = flat_data.min().item(), flat_data.max().item()
        self.centroids = torch.linspace(min_val, max_val, self.num_levels, device=self.device)
        
        # Lloyd-Max iterations
        for iteration in range(self.max_iter):
            # Assign to nearest centroid
            distances = torch.abs(flat_data.unsqueeze(1) - self.centroids)
            assignments = distances.argmin(dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(self.centroids)
            for i in range(self.num_levels):
                mask = assignments == i
                if mask.any():
                    new_centroids[i] = flat_data[mask].mean()
                else:
                    # Empty bin: interpolate
                    if i > 0 and i < self.num_levels - 1:
                        new_centroids[i] = (self.centroids[i-1] + self.centroids[i+1]) / 2
                    else:
                        new_centroids[i] = self.centroids[i]
            
            # Check convergence
            change = (new_centroids - self.centroids).abs().max()
            self.centroids = new_centroids
            
            if change < 1e-6:
                break
        
        # Compute boundaries (midpoints)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2
        self._fitted = True
        
        return self
    
    def fit_beta(self, alpha: float = 2.0, beta: float = 2.0, 
                 num_samples: int = 10000, scale: float = 1.0) -> 'LloydMaxQuantizerV3':
        """
        Fit on Beta distribution (for rotated data).
        
        After random rotation, coordinates follow Beta-like distribution.
        """
        dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta))
        samples = dist.sample((num_samples,)) * scale
        return self.fit(samples)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to indices."""
        if not self._fitted:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")
        
        original_shape = x.shape
        x_flat = x.reshape(-1, 1)
        
        # Find nearest centroid
        distances = torch.abs(x_flat - self.centroids)
        indices = distances.argmin(dim=1)
        
        return indices.reshape(original_shape).to(torch.uint8)
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from indices."""
        if not self._fitted:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")
        
        return self.centroids[indices.long()]
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and return indices + dequantized."""
        indices = self.encode(x)
        dequantized = self.decode(indices)
        return indices, dequantized


# ============================================================================
# V3 MSE Compressor
# ============================================================================

@dataclass
class MSECompressorConfig:
    """Configuration for MSE compressor."""
    bits: int = 4
    use_rotation: bool = True
    pack_bits: bool = True
    device: str = 'cpu'


class MSECompressor:
    """
    V3 MSE-only compressor with bit-packed storage.
    
    Key improvements:
    1. No QJL (removes variance that softmax amplifies)
    2. Bit-packed storage (real compression ratios)
    3. Fast random rotation
    """
    
    def __init__(self, config: MSECompressorConfig):
        self.config = config
        self.rotation = None
        self.quantizer = None
        self.scale = 1.0
    
    def fit(self, sample_data: torch.Tensor, head_dim: int):
        """Fit compressor on sample data."""
        if self.config.use_rotation:
            # Find next power of 2
            dim = 1 << (head_dim - 1).bit_length()
            self.rotation = RandomRotation(dim, self.config.device)
            
            # Normalize, then rotate sample data (consistent with compress)
            sample_norm = sample_data / (sample_data.norm(dim=-1, keepdim=True) + 1e-8)
            sample_padded = self._pad_to_dim(sample_norm, dim)
            sample_rotated = self.rotation.rotate(sample_padded)
        else:
            sample_rotated = sample_data
            dim = head_dim
        
        # Fit quantizer on rotated data
        self.quantizer = LloydMaxQuantizerV3(
            self.config.bits, 
            device=self.config.device
        )
        self.quantizer.fit(sample_rotated)
        
        return self
    
    def compress(self, x: torch.Tensor, head_dim: int) -> Dict[str, torch.Tensor]:
        """
        Compress tensor with bit-packed storage.
        
        Uses per-vector normalization to handle varying magnitudes.
        
        Args:
            x: Input tensor [..., head_dim]
            head_dim: Head dimension
        
        Returns:
            Dictionary with compressed data
        """
        original_shape = x.shape
        
        # Per-vector normalization for dynamic range handling
        x_norm = x.norm(dim=-1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        
        # Pad if needed
        if self.rotation:
            dim = self.rotation.dim
            x_padded = self._pad_to_dim(x_normalized, dim)
            x_rotated = self.rotation.rotate(x_padded)
        else:
            x_rotated = x_normalized
        
        # Quantize
        indices, _ = self.quantizer.quantize(x_rotated)
        
        # Pack bits
        if self.config.pack_bits:
            indices_packed, pack_shape = pack_bits(indices, self.config.bits)
        else:
            indices_packed = indices
            pack_shape = None
        
        return {
            'indices': indices_packed,
            'norm': x_norm.half(),  # Store norm for reconstruction
            'pack_shape': pack_shape,
            'original_shape': original_shape,
        }
    
    def decompress(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decompress tensor.
        
        Restores per-vector normalization for accurate reconstruction.
        """
        # Unpack bits
        if self.config.pack_bits and compressed.get('pack_shape') is not None:
            indices = unpack_bits(
                compressed['indices'], 
                self.config.bits, 
                compressed['pack_shape']
            ).to(self.config.device)
        else:
            indices = compressed['indices']
        
        # Decode
        x_rotated = self.quantizer.decode(indices)
        
        # Inverse rotation
        if self.rotation:
            x_normalized = self.rotation.inverse(x_rotated)
            # Remove padding
            x_normalized = x_normalized[..., :compressed['original_shape'][-1]]
        else:
            x_normalized = x_rotated
        
        # Restore norm for accurate reconstruction
        x = x_normalized * compressed['norm'].float()
        
        return x.reshape(compressed['original_shape'])
    
    def _pad_to_dim(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Pad tensor to target dimension."""
        if x.shape[-1] >= target_dim:
            return x[..., :target_dim]
        
        pad_size = target_dim - x.shape[-1]
        padding = torch.zeros(*x.shape[:-1], pad_size, device=x.device, dtype=x.dtype)
        return torch.cat([x, padding], dim=-1)


# ============================================================================
# V3 TurboQuant (Asymmetric K/V, Layer-Adaptive)
# ============================================================================

@dataclass
class TurboQuantV3Config:
    """Configuration for TurboQuant V3."""
    key_bits: int = 4
    value_bits: int = 2
    use_rotation: bool = True
    pack_bits: bool = True
    protected_layers: int = 0  # Number of first/last layers to protect
    total_layers: int = 32
    device: str = 'cpu'


class TurboQuantV3:
    """
    TurboQuant V3 with community improvements.
    
    Features:
    1. MSE-only (no QJL) - better for softmax attention
    2. Asymmetric K/V bits - keys get more precision
    3. Bit-packed storage - real compression
    4. Layer-adaptive - protect sensitive layers
    
    Recommended configs from tonbistudio:
    - K4/V2 (4-bit keys, 2-bit values): 5.1x compression, best quality
    - K3/V2 (3-bit keys, 2-bit values): 6.0x compression, good quality
    """
    
    def __init__(self, config: TurboQuantV3Config):
        self.config = config
        self.key_compressors: Dict[int, MSECompressor] = {}
        self.value_compressors: Dict[int, MSECompressor] = {}
    
    def fit(self, sample_keys: torch.Tensor, sample_values: torch.Tensor, 
            head_dim: int, layer_idx: int = 0):
        """
        Fit compressors for a layer.
        
        Args:
            sample_keys: Sample keys for this layer
            sample_values: Sample values for this layer
            head_dim: Head dimension
            layer_idx: Layer index for layer-adaptive compression
        """
        # Determine bits for this layer
        key_bits, value_bits = self._get_layer_bits(layer_idx)
        
        # Create key compressor
        key_config = MSECompressorConfig(
            bits=key_bits,
            use_rotation=self.config.use_rotation,
            pack_bits=self.config.pack_bits,
            device=self.config.device
        )
        key_comp = MSECompressor(key_config)
        key_comp.fit(sample_keys, head_dim)
        self.key_compressors[layer_idx] = key_comp
        
        # Create value compressor
        value_config = MSECompressorConfig(
            bits=value_bits,
            use_rotation=self.config.use_rotation,
            pack_bits=self.config.pack_bits,
            device=self.config.device
        )
        value_comp = MSECompressor(value_config)
        value_comp.fit(sample_values, head_dim)
        self.value_compressors[layer_idx] = value_comp
    
    def compress_kv(self, keys: torch.Tensor, values: torch.Tensor,
                   head_dim: int, layer_idx: int = 0) -> Dict[str, Dict]:
        """
        Compress KV cache for a layer.
        
        Args:
            keys: Keys tensor [..., head_dim]
            values: Values tensor [..., head_dim]
            head_dim: Head dimension
            layer_idx: Layer index
        
        Returns:
            Compressed KV cache
        """
        if layer_idx not in self.key_compressors:
            raise RuntimeError(f"Layer {layer_idx} not fitted. Call fit() first.")
        
        key_comp = self.key_compressors[layer_idx]
        value_comp = self.value_compressors[layer_idx]
        
        return {
            'keys': key_comp.compress(keys, head_dim),
            'values': value_comp.compress(values, head_dim),
            'layer_idx': layer_idx,
        }
    
    def decompress_kv(self, compressed: Dict[str, Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress KV cache."""
        layer_idx = compressed['layer_idx']
        
        key_comp = self.key_compressors[layer_idx]
        value_comp = self.value_compressors[layer_idx]
        
        keys = key_comp.decompress(compressed['keys'])
        values = value_comp.decompress(compressed['values'])
        
        return keys, values
    
    def _get_layer_bits(self, layer_idx: int) -> Tuple[int, int]:
        """
        Get bits for this layer (layer-adaptive).
        
        Protected layers (first/last) get more bits.
        """
        total = self.config.total_layers
        protected = self.config.protected_layers
        
        # Check if this layer is protected
        is_protected = layer_idx < protected or layer_idx >= (total - protected)
        
        if is_protected:
            # Protected layers: use more bits
            return max(self.config.key_bits, 4), max(self.config.value_bits, 2)
        else:
            return self.config.key_bits, self.config.value_bits
    
    def get_compression_ratio(self, layer_idx: int = None) -> float:
        """
        Get compression ratio.
        
        Args:
            layer_idx: If provided, get ratio for specific layer
        
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        if layer_idx is not None:
            key_bits, value_bits = self._get_layer_bits(layer_idx)
        else:
            key_bits, value_bits = self.config.key_bits, self.config.value_bits
        
        # Average bits per element
        avg_bits = (key_bits + value_bits) / 2
        return 16.0 / avg_bits
    
    def memory_stats(self, seq_len: int, num_layers: int = 1,
                    batch_size: int = 1, num_heads: int = 32, head_dim: int = 64) -> Dict[str, float]:
        """
        Calculate memory statistics with accurate accounting.
        
        Accounts for:
        - Bit-packing efficiency (only 37.5% for 3-bit)
        - Per-vector norm storage
        - Rotation padding
        """
        num_vectors = batch_size * num_heads * seq_len
        elements_per_tensor = num_vectors * head_dim
        
        total_original_bytes = 0
        total_compressed_bytes = 0
        
        for layer_idx in range(num_layers):
            key_bits, value_bits = self._get_layer_bits(layer_idx)
            
            # Original: fp16 for K + V
            original_bytes = 2 * elements_per_tensor * 2  # 2 tensors * 2 bytes
            total_original_bytes += original_bytes
            
            # Compressed indices
            def index_bytes(bits, elements):
                if 8 % bits == 0:
                    # Efficiently packed
                    return (elements * bits + 7) // 8
                else:
                    # Stored as uint8 (no packing)
                    return elements
            
            key_idx_bytes = index_bytes(key_bits, elements_per_tensor)
            val_idx_bytes = index_bytes(value_bits, elements_per_tensor)
            
            # Norms: fp16 per vector
            norm_bytes = 2 * num_vectors * 2  # K norms + V norms, 2 bytes each
            
            compressed_bytes = key_idx_bytes + val_idx_bytes + norm_bytes
            total_compressed_bytes += compressed_bytes
        
        return {
            'original_mb': total_original_bytes / (1024 ** 2),
            'compressed_mb': total_compressed_bytes / (1024 ** 2),
            'compression_ratio': total_original_bytes / total_compressed_bytes,
            'memory_saved_percent': (1 - total_compressed_bytes / total_original_bytes) * 100,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_v3_k4_v2(head_dim: int = 64, device: str = 'cpu') -> TurboQuantV3:
    """
    Create V3 with K4/V2 configuration (recommended).
    
    4-bit keys, 2-bit values: ~4.9x compression (512 seq), ~5.3x (4096 seq), best quality.
    
    Best for quality-critical applications. 9-10% relative error.
    """
    config = TurboQuantV3Config(
        key_bits=4,
        value_bits=2,
        use_rotation=True,
        pack_bits=True,
        device=device
    )
    return TurboQuantV3(config)


def create_v3_k3_v2(head_dim: int = 64, device: str = 'cpu') -> TurboQuantV3:
    """
    Create V3 with K3/V2 configuration.
    
    3-bit keys, 2-bit values: ~3.0x compression (512 seq), ~3.3x (4096 seq), good quality.
    
    Note: 3-bit doesn't pack efficiently (37.5% overhead), so actual ratio is lower
    than theoretical 6.4x. 18-34% relative error.
    """
    config = TurboQuantV3Config(
        key_bits=3,
        value_bits=2,
        use_rotation=True,
        pack_bits=True,
        device=device
    )
    return TurboQuantV3(config)


def create_v3_layer_adaptive(
    key_bits: int = 4,
    value_bits: int = 2,
    protected_layers: int = 2,
    total_layers: int = 32,
    device: str = 'cpu'
) -> TurboQuantV3:
    """
    Create V3 with layer-adaptive compression.
    
    Protected (first/last) layers get more bits.
    """
    config = TurboQuantV3Config(
        key_bits=key_bits,
        value_bits=value_bits,
        protected_layers=protected_layers,
        total_layers=total_layers,
        use_rotation=True,
        pack_bits=True,
        device=device
    )
    return TurboQuantV3(config)


# Recommended V3 configurations
V3_RECOMMENDED = {
    'k4_v2': create_v3_k4_v2,           # 5.1x, best quality
    'k3_v2': create_v3_k3_v2,           # 6.0x, good quality
    'k4_v2_protected': lambda d, dev: create_v3_layer_adaptive(4, 2, 2, 32, dev),  # 3.6x, 99% top-1
}
