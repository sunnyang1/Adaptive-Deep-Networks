"""
Compressed KV Cache for HuggingFace Transformers.

Provides V3Cache class compatible with HF's DynamicCache,
enabling transparent compression during generation.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .compressor import MSECompressor, CompressorConfig


@dataclass
class CacheConfig:
    """Configuration for compressed KV cache."""

    key_bits: int = 4
    value_bits: int = 2
    residual_window: int = 128
    device: str = "cpu"


class V3Cache:
    """
    Compressed KV cache compatible with HuggingFace Transformers.

    Uses chunked compression with residual window:
    - Recent `residual_window` tokens kept in fp16
    - Older tokens compressed and stored in chunks

    Usage:
        >>> from transformers import AutoModelForCausalLM
        >>> cache = V3Cache(key_bits=4, value_bits=2)
        >>> model.generate(..., past_key_values=cache)
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        key_bits: int = 4,
        value_bits: int = 2,
        residual_window: int = 128,
        device: str = "cpu",
    ):
        """
        Initialize compressed cache.

        Args:
            config: CacheConfig (or specify parameters individually)
            key_bits: Bits for key compression
            value_bits: Bits for value compression
            residual_window: Number of recent tokens to keep in fp16
            device: 'cpu', 'cuda', or 'mps'
        """
        if config is None:
            config = CacheConfig(
                key_bits=key_bits,
                value_bits=value_bits,
                residual_window=residual_window,
                device=device,
            )

        self.config = config

        # Compressors per layer
        self._key_compressors: Dict[int, MSECompressor] = {}
        self._val_compressors: Dict[int, MSECompressor] = {}

        # Compressed chunks per layer
        self._key_chunks: Dict[int, List[Dict]] = {}
        self._val_chunks: Dict[int, List[Dict]] = {}

        # Recent fp16 buffers per layer
        self._recent_k: Dict[int, List[torch.Tensor]] = {}
        self._recent_v: Dict[int, List[torch.Tensor]] = {}

        # Sequence tracking
        self._total_seq: Dict[int, int] = {}
        self._seen_layers = 0

    def _get_compressors(
        self, layer_idx: int, head_dim: int
    ) -> Tuple[MSECompressor, MSECompressor]:
        """Get or create compressors for layer."""
        if layer_idx not in self._key_compressors:
            key_config = CompressorConfig(
                bits=self.config.key_bits,
                use_rotation=True,
                pack_bits=True,
                device=self.config.device,
            )
            val_config = CompressorConfig(
                bits=self.config.value_bits,
                use_rotation=True,
                pack_bits=True,
                device=self.config.device,
            )
            self._key_compressors[layer_idx] = MSECompressor(key_config)
            self._val_compressors[layer_idx] = MSECompressor(val_config)

            # Initialize storage
            self._key_chunks[layer_idx] = []
            self._val_chunks[layer_idx] = []
            self._recent_k[layer_idx] = []
            self._recent_v[layer_idx] = []
            self._total_seq[layer_idx] = 0

        return self._key_compressors[layer_idx], self._val_compressors[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value states.

        Called by HF transformers during generation.

        Args:
            key_states: New key states [batch, heads, seq, head_dim]
            value_states: New value states [batch, heads, seq, head_dim]
            layer_idx: Layer index
            cache_kwargs: Additional kwargs (unused)

        Returns:
            (full_keys, full_values) tuple including compressed history
        """
        B, H, S_new, D = key_states.shape
        device = key_states.device

        key_comp, val_comp = self._get_compressors(layer_idx, D)

        # Update sequence count
        self._total_seq[layer_idx] = self._total_seq.get(layer_idx, 0) + S_new

        # Add to recent buffer
        self._recent_k[layer_idx].append(key_states)
        self._recent_v[layer_idx].append(value_states)

        # Concatenate recent
        recent_k = torch.cat(self._recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._recent_v[layer_idx], dim=2)

        rw = self.config.residual_window

        # Compress overflow if window exceeded
        if recent_k.shape[2] > rw and rw > 0:
            overflow = recent_k.shape[2] - rw

            # Compress overflow portion
            to_compress_k = recent_k[:, :, :overflow, :]
            to_compress_v = recent_v[:, :, :overflow, :]

            # Fit on first compression
            if not key_comp.is_fitted:
                sample_size = min(64, overflow)
                sample_k = to_compress_k[:, :, :sample_size, :]
                sample_v = to_compress_v[:, :, :sample_size, :]
                key_comp.fit(sample_k, D)
                val_comp.fit(sample_v, D)

            # Compress
            ck = key_comp.compress(to_compress_k, D)
            cv = val_comp.compress(to_compress_v, D)

            self._key_chunks[layer_idx].append(ck)
            self._val_chunks[layer_idx].append(cv)

            # Keep only window
            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._recent_k[layer_idx] = [recent_k]
            self._recent_v[layer_idx] = [recent_v]

        # Decompress all chunks + recent
        parts_k = []
        parts_v = []

        for ck, cv in zip(self._key_chunks[layer_idx], self._val_chunks[layer_idx]):
            dk = key_comp.decompress(ck)
            dv = val_comp.decompress(cv)
            parts_k.append(dk.to(key_states.dtype))
            parts_v.append(dv.to(value_states.dtype))

        # Add recent
        recent_k = torch.cat(self._recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._recent_v[layer_idx], dim=2)
        parts_k.append(recent_k)
        parts_v.append(recent_v)

        full_k = torch.cat(parts_k, dim=2)
        full_v = torch.cat(parts_v, dim=2)

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length for layer."""
        return self._total_seq.get(layer_idx, 0)

    def get_max_length(self) -> Optional[int]:
        """Get maximum sequence length (no limit for compressed cache)."""
        return None

    def reset(self):
        """Reset cache state."""
        self._key_compressors.clear()
        self._val_compressors.clear()
        self._key_chunks.clear()
        self._val_chunks.clear()
        self._recent_k.clear()
        self._recent_v.clear()
        self._total_seq.clear()
        self._seen_layers = 0


# Alias for backward compatibility
TurboQuantV3Cache = V3Cache
