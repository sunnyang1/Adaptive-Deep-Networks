"""
MNN-Inspired TurboQuant Improvements

Based on: https://github.com/alibaba/MNN/commit/244f5d10df5a95b4f4e6f3d9251c6fe3dc0e7c83

Key improvements from MNN:
1. attention_mode encoding: flash_attention * 8 + kv_quant_mode
2. Separate KV quantization modes (TQ3/TQ4)
3. FlashAttention integration
4. Optimized Lloyd-Max codebook

Quantization Modes (kv_quant_mode):
- 0: No quantization, FP16
- 1: Key INT8, Value FP16
- 2: Key and Value INT8
- 3: Key TQ3 (3-bit), Value FP16
- 4: Key and Value TQ3
- 5: Key TQ4 (4-bit), Value FP16
- 6: Key and Value TQ4

FlashAttention Modes (flash_attention):
- 0: Standard attention
- 1: FlashAttention enabled

Common Configurations:
- 8: FlashAttention + No quantization (default)
- 10: FlashAttention + KV-INT8 (near lossless)
- 14: FlashAttention + KV-TQ4 (>30% memory saving, 4B+ models)
- 12: FlashAttention + KV-TQ3 (extreme compression, 4B+ models)
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union
from enum import IntEnum


class KVQuantMode(IntEnum):
    """KV Cache quantization modes."""
    FP16 = 0           # No quantization
    KEY_INT8 = 1       # Key INT8, Value FP16
    KV_INT8 = 2        # Both INT8
    KEY_TQ3 = 3        # Key TQ3, Value FP16
    KV_TQ3 = 4         # Both TQ3
    KEY_TQ4 = 5        # Key TQ4, Value FP16
    KV_TQ4 = 6         # Both TQ4


class AttentionMode:
    """
    Attention mode encoding: attention_mode = flash_attention * 8 + kv_quant_mode
    
    Examples:
    - Mode 8: FlashAttention (1*8 + 0), FP16
    - Mode 10: FlashAttention + KV-INT8 (1*8 + 2)
    - Mode 14: FlashAttention + KV-TQ4 (1*8 + 6)
    - Mode 12: FlashAttention + KV-TQ3 (1*8 + 4)
    """
    
    @staticmethod
    def encode(flash_attention: bool, kv_mode: KVQuantMode) -> int:
        """Encode attention mode."""
        return (int(flash_attention) * 8) + int(kv_mode)
    
    @staticmethod
    def decode(mode: int) -> Tuple[bool, KVQuantMode]:
        """Decode attention mode."""
        flash_attention = mode >= 8
        kv_mode = KVQuantMode(mode % 8)
        return flash_attention, kv_mode
    
    @staticmethod
    def get_description(mode: int) -> str:
        """Get human-readable description."""
        flash, kv = AttentionMode.decode(mode)
        fa_str = "FlashAttn" if flash else "Standard"
        
        kv_str = {
            KVQuantMode.FP16: "FP16",
            KVQuantMode.KEY_INT8: "Key-INT8",
            KVQuantMode.KV_INT8: "KV-INT8",
            KVQuantMode.KEY_TQ3: "Key-TQ3",
            KVQuantMode.KV_TQ3: "KV-TQ3",
            KVQuantMode.KEY_TQ4: "Key-TQ4",
            KVQuantMode.KV_TQ4: "KV-TQ4",
        }[kv]
        
        return f"{fa_str} + {kv_str}"


@dataclass
class MNNTurboQuantConfig:
    """MNN-inspired TurboQuant configuration."""
    
    attention_mode: int = 8  # Default: FlashAttention + FP16
    
    # For TQ3/TQ4 quantization
    use_lloyd_max: bool = True  # Use Lloyd-Max codebook
    lloyd_max_iterations: int = 100
    
    # Tensor Core settings
    use_tensor_cores: bool = True
    
    # Model size threshold (MNN recommends TQ for 4B+ models)
    min_params_for_tq: float = 4e9  # 4B parameters
    
    @property
    def flash_attention(self) -> bool:
        """Check if FlashAttention is enabled."""
        return self.attention_mode >= 8
    
    @property
    def kv_quant_mode(self) -> KVQuantMode:
        """Get KV quantization mode."""
        return KVQuantMode(self.attention_mode % 8)
    
    @property
    def compression_ratio(self) -> float:
        """Expected compression ratio."""
        mode = self.kv_quant_mode
        ratios = {
            KVQuantMode.FP16: 1.0,
            KVQuantMode.KEY_INT8: 1.33,
            KVQuantMode.KV_INT8: 2.0,
            KVQuantMode.KEY_TQ3: 2.67,
            KVQuantMode.KV_TQ3: 4.0,
            KVQuantMode.KEY_TQ4: 2.0,
            KVQuantMode.KV_TQ4: 3.0,
        }
        return ratios.get(mode, 1.0)
    
    def is_recommended_for_model_size(self, num_params: int) -> bool:
        """
        Check if this config is recommended for model size.
        MNN recommends TQ3/TQ4 only for 4B+ models.
        """
        mode = self.kv_quant_mode
        if mode in (KVQuantMode.KEY_TQ3, KVQuantMode.KV_TQ3, 
                    KVQuantMode.KEY_TQ4, KVQuantMode.KV_TQ4):
            return num_params >= self.min_params_for_tq
        return True


class LloydMaxQuantizer:
    """
    Lloyd-Max optimal quantization using iterative optimization.
    
    Based on MNN's implementation for TurboQuant codebook generation.
    """
    
    def __init__(self, num_bits: int, max_iter: int = 100):
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.max_iter = max_iter
        self.codebook = None
        self.boundaries = None
    
    def fit(self, data: torch.Tensor) -> 'LloydMaxQuantizer':
        """
        Train Lloyd-Max quantizer on data.
        
        Args:
            data: Training data [N, D]
        
        Returns:
            self
        """
        # Flatten data
        flat_data = data.reshape(-1)
        
        # Initialize with uniform quantization
        min_val = flat_data.min()
        max_val = flat_data.max()
        
        # Initial codebook (centroids)
        self.codebook = torch.linspace(min_val, max_val, self.num_levels)
        
        # Lloyd-Max iterations
        for _ in range(self.max_iter):
            # Assign points to nearest centroid
            distances = torch.abs(flat_data.unsqueeze(1) - self.codebook)
            assignments = distances.argmin(dim=1)
            
            # Update centroids
            new_codebook = torch.zeros_like(self.codebook)
            for i in range(self.num_levels):
                mask = assignments == i
                if mask.any():
                    new_codebook[i] = flat_data[mask].mean()
                else:
                    new_codebook[i] = self.codebook[i]  # Keep old if empty
            
            # Check convergence
            if torch.allclose(new_codebook, self.codebook, rtol=1e-5):
                break
            
            self.codebook = new_codebook
        
        # Compute boundaries (midpoints between centroids)
        self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2
        
        return self
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to indices."""
        # Find nearest centroid
        distances = torch.abs(x.unsqueeze(-1) - self.codebook.to(x.device))
        indices = distances.argmin(dim=-1)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from indices."""
        return self.codebook[indices].to(indices.device)


class MNNTurboQuantCompressor:
    """
    MNN-inspired TurboQuant compressor for KV cache.
    
    Features:
    - Separate key/value quantization modes
    - FlashAttention integration
    - Lloyd-Max optimal codebooks
    - Model-size aware quantization
    """
    
    def __init__(
        self,
        config: MNNTurboQuantConfig,
        head_dim: int = 128,
        device: str = 'cpu'
    ):
        self.config = config
        self.head_dim = head_dim
        self.device = device
        
        # Initialize quantizers based on mode
        self._init_quantizers()
    
    def _init_quantizers(self):
        """Initialize quantizers based on KV mode."""
        mode = self.config.kv_quant_mode
        
        # Key quantizer
        if mode in (KVQuantMode.KEY_INT8, KVQuantMode.KV_INT8):
            self.key_quantizer = lambda x: (x * 127).to(torch.int8)
            self.key_dequantizer = lambda x: x.float() / 127
        elif mode in (KVQuantMode.KEY_TQ3, KVQuantMode.KV_TQ3):
            self.key_quantizer = LloydMaxQuantizer(3, self.config.lloyd_max_iterations)
        elif mode in (KVQuantMode.KEY_TQ4, KVQuantMode.KV_TQ4):
            self.key_quantizer = LloydMaxQuantizer(4, self.config.lloyd_max_iterations)
        else:
            self.key_quantizer = None
        
        # Value quantizer
        if mode in (KVQuantMode.KV_INT8,):
            self.value_quantizer = lambda x: (x * 127).to(torch.int8)
            self.value_dequantizer = lambda x: x.float() / 127
        elif mode in (KVQuantMode.KV_TQ3,):
            self.value_quantizer = LloydMaxQuantizer(3, self.config.lloyd_max_iterations)
        elif mode in (KVQuantMode.KV_TQ4,):
            self.value_quantizer = LloydMaxQuantizer(4, self.config.lloyd_max_iterations)
        else:
            self.value_quantizer = None
    
    def fit_codebooks(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Fit Lloyd-Max codebooks on sample data.
        
        Args:
            keys: Sample key tensors [..., head_dim]
            values: Sample value tensors [..., head_dim]
        """
        mode = self.config.kv_quant_mode
        
        if isinstance(self.key_quantizer, LloydMaxQuantizer):
            self.key_quantizer.fit(keys)
            print(f"Fitted key quantizer: {self.key_quantizer.codebook}")
        
        if isinstance(self.value_quantizer, LloydMaxQuantizer):
            self.value_quantizer.fit(values)
            print(f"Fitted value quantizer: {self.value_quantizer.codebook}")
    
    def compress_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compress KV cache.
        
        Args:
            keys: Key tensor [batch, heads, seq, head_dim]
            values: Value tensor [batch, heads, seq, head_dim]
        
        Returns:
            Dictionary with compressed representations
        """
        compressed = {
            'key_indices': None,
            'value_indices': None,
            'keys_fp16': None,
            'values_fp16': None,
        }
        
        # Compress keys
        if isinstance(self.key_quantizer, LloydMaxQuantizer):
            compressed['key_indices'] = self.key_quantizer.encode(keys)
        elif self.key_quantizer is not None:
            compressed['key_indices'] = self.key_quantizer(keys)
        else:
            compressed['keys_fp16'] = keys.half()
        
        # Compress values
        if isinstance(self.value_quantizer, LloydMaxQuantizer):
            compressed['value_indices'] = self.value_quantizer.encode(values)
        elif self.value_quantizer is not None:
            compressed['value_indices'] = self.value_quantizer(values)
        else:
            compressed['values_fp16'] = values.half()
        
        return compressed
    
    def decompress_kv(self, compressed: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress KV cache."""
        # Decompress keys
        if compressed['key_indices'] is not None:
            if isinstance(self.key_quantizer, LloydMaxQuantizer):
                keys = self.key_quantizer.decode(compressed['key_indices'])
            else:
                keys = self.key_quantizer(compressed['key_indices'])
        else:
            keys = compressed['keys_fp16'].float()
        
        # Decompress values
        if compressed['value_indices'] is not None:
            if isinstance(self.value_quantizer, LloydMaxQuantizer):
                values = self.value_quantizer.decode(compressed['value_indices'])
            else:
                values = self.value_quantizer(compressed['value_indices'])
        else:
            values = compressed['values_fp16'].float()
        
        return keys, values
    
    def get_memory_stats(
        self,
        seq_len: int,
        batch_size: int = 1,
        num_heads: int = 32
    ) -> Dict[str, float]:
        """Calculate memory statistics."""
        # Original memory (FP16)
        original_bytes = batch_size * num_heads * seq_len * self.head_dim * 2
        
        # Compressed memory
        mode = self.config.kv_quant_mode
        bits_per_element = {
            KVQuantMode.FP16: 16,
            KVQuantMode.KEY_INT8: 12,  # Key 8-bit, Value 16-bit
            KVQuantMode.KV_INT8: 8,
            KVQuantMode.KEY_TQ3: 11,   # Key 3-bit, Value 16-bit
            KVQuantMode.KV_TQ3: 6,
            KVQuantMode.KEY_TQ4: 12,   # Key 4-bit, Value 16-bit
            KVQuantMode.KV_TQ4: 8,
        }[mode]
        
        compressed_bytes = batch_size * num_heads * seq_len * self.head_dim * bits_per_element / 8
        
        return {
            'original_mb': original_bytes / (1024 ** 2),
            'compressed_mb': compressed_bytes / (1024 ** 2),
            'saving_ratio': original_bytes / compressed_bytes,
            'compression_ratio': compressed_bytes / original_bytes,
        }


# Convenience functions
def create_mnn_turboquant(
    attention_mode: int = 8,
    head_dim: int = 128,
    device: str = 'cpu'
) -> MNNTurboQuantCompressor:
    """
    Create MNN-style TurboQuant compressor.
    
    Args:
        attention_mode: Mode (8=FP16, 10=INT8, 14=TQ4, 12=TQ3)
        head_dim: Head dimension
        device: Device
    
    Returns:
        Configured compressor
    """
    config = MNNTurboQuantConfig(
        attention_mode=attention_mode,
        use_lloyd_max=True
    )
    return MNNTurboQuantCompressor(config, head_dim, device)


# Recommended configurations (from MNN)
CONFIG_RECOMMENDATIONS = {
    'default': 8,           # FlashAttention + FP16
    'near_lossless': 10,    # FlashAttention + KV-INT8
    'balanced': 14,         # FlashAttention + KV-TQ4 (4B+ models)
    'extreme': 12,          # FlashAttention + KV-TQ3 (4B+ models)
}
