"""
TurboQuant Core: Unified Compression API

Single, clean interface for all TurboQuant compression modes.

Usage:
    from src.turboquant import TurboQuant
    
    # Simple API
    quant = TurboQuant('fp16')                    # No compression
    quant = TurboQuant('int8')                    # INT8 quantization
    quant = TurboQuant('tq4')                     # TurboQuant 4-bit
    quant = TurboQuant('tq4_flash')               # With FlashAttention
    
    # Compress/decompress
    compressed = quant.compress_kv(keys, values)
    keys_deq, values_deq = quant.decompress_kv(compressed)
    
    # Or use context manager for automatic memory tracking
    with TurboQuant('tq4') as quant:
        compressed = quant.compress_kv(keys, values)
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Union, Literal
from enum import IntEnum
from contextlib import contextmanager


# ============================================================================
# Configuration
# ============================================================================

class QuantMode(IntEnum):
    """Quantization modes for KV cache."""
    FP16 = 0
    KEY_INT8 = 1
    KV_INT8 = 2
    KEY_TQ3 = 3
    KV_TQ3 = 4
    KEY_TQ4 = 5
    KV_TQ4 = 6


@dataclass
class TurboQuantConfig:
    """
    Unified TurboQuant configuration.
    
    Simple string-based configuration with optional advanced settings.
    """
    # Simple mode selection
    mode: str = 'fp16'  # 'fp16', 'int8', 'tq3', 'tq4', 'tq8'
    
    # Advanced settings (auto-populated from mode)
    flash_attention: bool = field(default=False, repr=False)
    kv_quant_mode: QuantMode = field(default=QuantMode.FP16, repr=False)
    
    # TurboQuant specific
    head_dim: int = 128
    lloyd_max_iterations: int = 100
    qjl_proj_dim: int = 256
    
    # Device
    device: Union[str, torch.device] = 'cpu'
    
    def __post_init__(self):
        """Parse mode string into components."""
        mode_lower = self.mode.lower()
        
        # Check for flash attention suffix
        if '_flash' in mode_lower or mode_lower.startswith('flash_'):
            self.flash_attention = True
            mode_lower = mode_lower.replace('_flash', '').replace('flash_', '')
        
        # Map mode string to quant mode
        mode_map = {
            'fp16': QuantMode.FP16,
            'fp32': QuantMode.FP16,  # FP16 storage
            'int8': QuantMode.KV_INT8,
            'key_int8': QuantMode.KEY_INT8,
            'tq3': QuantMode.KV_TQ3,
            'key_tq3': QuantMode.KEY_TQ3,
            'tq4': QuantMode.KV_TQ4,
            'key_tq4': QuantMode.KEY_TQ4,
            'tq8': QuantMode.KV_TQ3,  # Alias for extreme compression
        }
        
        if mode_lower not in mode_map:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from: {list(mode_map.keys())}")
        
        self.kv_quant_mode = mode_map[mode_lower]
    
    @property
    def compression_ratio(self) -> float:
        """Expected compression ratio."""
        ratios = {
            QuantMode.FP16: 1.0,
            QuantMode.KEY_INT8: 1.33,
            QuantMode.KV_INT8: 2.0,
            QuantMode.KEY_TQ3: 2.67,
            QuantMode.KV_TQ3: 4.0,
            QuantMode.KEY_TQ4: 2.0,
            QuantMode.KV_TQ4: 3.0,
        }
        return ratios.get(self.kv_quant_mode, 1.0)
    
    @property
    def bits_per_element(self) -> float:
        """Average bits per element."""
        return 16.0 / self.compression_ratio
    
    def memory_stats(self, seq_len: int, batch_size: int = 1, num_heads: int = 32) -> Dict[str, float]:
        """Calculate memory statistics."""
        elements = batch_size * num_heads * seq_len * self.head_dim
        
        original_bytes = elements * 2 * 2  # Keys + Values, FP16 = 2 bytes
        compressed_bytes = elements * 2 * 16.0 / self.compression_ratio / 8
        
        return {
            'original_mb': original_bytes / (1024 ** 2),
            'compressed_mb': compressed_bytes / (1024 ** 2),
            'compression_ratio': self.compression_ratio,
            'memory_saved': (1 - 1.0/self.compression_ratio) * 100,
        }


# ============================================================================
# Quantizers
# ============================================================================

class LloydMaxQuantizer:
    """
    Lloyd-Max optimal quantizer with iterative optimization.
    
    Supports both data-aware (fit) and data-oblivious modes.
    """
    
    def __init__(self, num_bits: int, max_iter: int = 100, device: str = 'cpu'):
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.max_iter = max_iter
        self.device = device
        
        self.codebook = None
        self.boundaries = None
        self._fitted = False
    
    def fit(self, data: torch.Tensor) -> 'LloydMaxQuantizer':
        """Fit codebook on data."""
        flat_data = data.reshape(-1).to(self.device)
        
        # Initialize with uniform quantization
        min_val, max_val = flat_data.min(), flat_data.max()
        self.codebook = torch.linspace(min_val, max_val, self.num_levels, device=self.device)
        
        # Lloyd-Max iterations
        for _ in range(self.max_iter):
            # Assign to nearest centroid
            distances = torch.abs(flat_data.unsqueeze(1) - self.codebook)
            assignments = distances.argmin(dim=1)
            
            # Update centroids
            new_codebook = torch.zeros_like(self.codebook)
            for i in range(self.num_levels):
                mask = assignments == i
                if mask.any():
                    new_codebook[i] = flat_data[mask].mean()
                else:
                    new_codebook[i] = self.codebook[i]
            
            # Check convergence
            if torch.allclose(new_codebook, self.codebook, rtol=1e-5):
                break
            
            self.codebook = new_codebook
        
        # Compute boundaries
        self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2
        self._fitted = True
        
        return self
    
    def fit_beta_distribution(self, alpha: float = 2.0, beta: float = 2.0, 
                               num_samples: int = 10000, scale: float = math.pi):
        """Fit on Beta distribution (for angle quantization after RHT)."""
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
        distances = torch.abs(x_flat - self.codebook)
        indices = distances.argmin(dim=1)
        
        return indices.reshape(original_shape)
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from indices."""
        if not self._fitted:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")
        
        return self.codebook[indices]
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and return indices + dequantized values."""
        indices = self.encode(x)
        dequantized = self.decode(indices)
        return indices, dequantized


class INT8Quantizer:
    """Simple INT8 symmetric quantization."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.scale = 127.0
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to INT8."""
        x_scaled = x * self.scale
        x_int8 = x_scaled.clamp(-128, 127).round().to(torch.int8)
        x_deq = x_int8.float() / self.scale
        return x_int8, x_deq


class FP16Quantizer:
    """FP16 quantization (no-op for dtype conversion)."""
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to FP16."""
        x_fp16 = x.half()
        return x_fp16, x_fp16.float()


# ============================================================================
# Main TurboQuant Class
# ============================================================================

class TurboQuant:
    """
    Unified TurboQuant interface.
    
    Simple, clean API for all quantization modes.
    
    Examples:
        >>> # FP16 (no compression)
        >>> quant = TurboQuant('fp16')
        
        >>> # INT8 quantization
        >>> quant = TurboQuant('int8')
        
        >>> # TurboQuant 4-bit with FlashAttention
        >>> quant = TurboQuant('tq4_flash')
        
        >>> # With custom settings
        >>> quant = TurboQuant('tq4', head_dim=64, lloyd_max_iterations=200)
    """
    
    def __init__(self, mode: str = 'fp16', **kwargs):
        """
        Initialize TurboQuant.
        
        Args:
            mode: Quantization mode
                - 'fp16', 'fp32': No compression
                - 'int8': INT8 quantization
                - 'tq3', 'tq4': TurboQuant 3-bit or 4-bit
                - Add '_flash' suffix for FlashAttention (e.g., 'tq4_flash')
            **kwargs: Additional config options (head_dim, device, etc.)
        
        Raises:
            ValueError: If mode is not recognized
        """
        self.config = TurboQuantConfig(mode=mode, **kwargs)
        self._init_quantizers()
        
        # Statistics
        self.stats = {
            'keys_compressed': 0,
            'values_compressed': 0,
            'bytes_original': 0,
            'bytes_compressed': 0,
        }
    
    def _init_quantizers(self):
        """Initialize quantizers based on mode."""
        mode = self.config.kv_quant_mode
        device = self.config.device
        
        # Key quantizer
        if mode in (QuantMode.KEY_INT8, QuantMode.KV_INT8):
            self.key_quantizer = INT8Quantizer(device)
        elif mode in (QuantMode.KEY_TQ3, QuantMode.KV_TQ3):
            self.key_quantizer = LloydMaxQuantizer(3, self.config.lloyd_max_iterations, device)
        elif mode in (QuantMode.KEY_TQ4, QuantMode.KV_TQ4):
            self.key_quantizer = LloydMaxQuantizer(4, self.config.lloyd_max_iterations, device)
        else:
            self.key_quantizer = FP16Quantizer()
        
        # Value quantizer
        if mode == QuantMode.KV_INT8:
            self.value_quantizer = INT8Quantizer(device)
        elif mode == QuantMode.KV_TQ3:
            self.value_quantizer = LloydMaxQuantizer(3, self.config.lloyd_max_iterations, device)
        elif mode == QuantMode.KV_TQ4:
            self.value_quantizer = LloydMaxQuantizer(4, self.config.lloyd_max_iterations, device)
        else:
            self.value_quantizer = FP16Quantizer()
    
    def fit(self, keys: torch.Tensor, values: torch.Tensor) -> 'TurboQuant':
        """
        Fit quantizers on sample data.
        
        Required for TQ3/TQ4 modes before compression.
        
        Args:
            keys: Sample keys [..., head_dim]
            values: Sample values [..., head_dim]
        
        Returns:
            self for chaining
        """
        # Fit key quantizer if it's Lloyd-Max
        if isinstance(self.key_quantizer, LloydMaxQuantizer):
            self.key_quantizer.fit(keys)
        
        # Fit value quantizer if it's Lloyd-Max
        if isinstance(self.value_quantizer, LloydMaxQuantizer):
            self.value_quantizer.fit(values)
        
        return self
    
    def fit_beta(self) -> 'TurboQuant':
        """
        Fit quantizers on Beta distribution (for RHT angles).
        
        Use this when you don't have sample data but know the
        distribution is Beta-concentrated (typical for RHT output).
        
        Returns:
            self for chaining
        """
        if isinstance(self.key_quantizer, LloydMaxQuantizer):
            self.key_quantizer.fit_beta_distribution()
        
        if isinstance(self.value_quantizer, LloydMaxQuantizer):
            self.value_quantizer.fit_beta_distribution()
        
        return self
    
    def compress_kv(self, keys: torch.Tensor, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compress KV cache.
        
        Args:
            keys: Key tensor [batch, heads, seq, head_dim]
            values: Value tensor [batch, heads, seq, head_dim]
        
        Returns:
            Dictionary with compressed representations
        """
        compressed = {}
        
        # Compress keys
        if isinstance(self.key_quantizer, (LloydMaxQuantizer, INT8Quantizer)):
            compressed['key_indices'], compressed['keys_deq'] = self.key_quantizer.quantize(keys)
        else:
            compressed['keys_fp16'], _ = self.key_quantizer.quantize(keys)
        
        # Compress values
        if isinstance(self.value_quantizer, (LloydMaxQuantizer, INT8Quantizer)):
            compressed['value_indices'], compressed['values_deq'] = self.value_quantizer.quantize(values)
        else:
            compressed['values_fp16'], _ = self.value_quantizer.quantize(values)
        
        # Update stats
        self.stats['keys_compressed'] += keys.numel()
        self.stats['values_compressed'] += values.numel()
        
        return compressed
    
    def decompress_kv(self, compressed: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress KV cache.
        
        Args:
            compressed: Output from compress_kv()
        
        Returns:
            keys, values: Decompressed tensors
        """
        # Decompress keys
        if 'key_indices' in compressed:
            if isinstance(self.key_quantizer, LloydMaxQuantizer):
                keys = self.key_quantizer.decode(compressed['key_indices'])
            else:
                keys = compressed['keys_deq']
        else:
            keys = compressed['keys_fp16'].float()
        
        # Decompress values
        if 'value_indices' in compressed:
            if isinstance(self.value_quantizer, LloydMaxQuantizer):
                values = self.value_quantizer.decode(compressed['value_indices'])
            else:
                values = compressed['values_deq']
        else:
            values = compressed['values_fp16'].float()
        
        return keys, values
    
    def memory_stats(self, seq_len: int, batch_size: int = 1, num_heads: int = 32) -> Dict[str, float]:
        """Get memory statistics for given sequence length."""
        return self.config.memory_stats(seq_len, batch_size, num_heads)
    
    def reset_stats(self):
        """Reset compression statistics."""
        self.stats = {
            'keys_compressed': 0,
            'values_compressed': 0,
            'bytes_original': 0,
            'bytes_compressed': 0,
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False
    
    def __repr__(self):
        return f"TurboQuant(mode='{self.config.mode}', ratio={self.config.compression_ratio:.2f}x)"


# ============================================================================
# Convenience Functions
# ============================================================================

def create_quantizer(mode: str, **kwargs) -> TurboQuant:
    """Create a TurboQuant instance."""
    return TurboQuant(mode, **kwargs)


# Recommended configurations
RECOMMENDED_CONFIGS = {
    'small_model': 'fp16',      # For <4B models
    'balanced': 'int8',         # 2x compression, near lossless
    'fast': 'tq4',              # 3x compression, 4B+ models
    'extreme': 'tq3',           # 4x compression, 4B+ models
    'flash_attention': 'tq4_flash',  # With FlashAttention
}


@contextmanager
def quantize_kv(mode: str, **kwargs):
    """Context manager for KV quantization."""
    quant = TurboQuant(mode, **kwargs)
    try:
        yield quant
    finally:
        pass
