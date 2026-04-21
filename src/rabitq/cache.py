"""
Compressed KV Cache for HuggingFace Transformers using true RaBitQ.

Compatible with HF's DynamicCache interface.
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .rotation import Rotator, FhtKacRotator, IdentityRotator
from .quantizer import RabitqConfig, QuantizedVector, quantize_vector, reconstruct_vector


@dataclass
class CacheConfig:
    """Configuration for compressed KV cache."""

    total_bits: int = 1
    residual_window: int = 128
    device: str = "cpu"
    head_dim: int = 64


class RaBitQCache:
    """
    Compressed KV cache compatible with HuggingFace Transformers.

    Uses true RaBitQ quantization:
    - Recent `residual_window` tokens kept in fp16
    - Older tokens compressed with 1-bit (+ extended-bit) RaBitQ
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        total_bits: int = 1,
        head_dim: int = 64,
        rotator: Optional[Rotator] = None,
        residual_window: int = 128,
        device: str = "cpu",
    ):
        if config is None:
            config = CacheConfig(
                total_bits=total_bits,
                head_dim=head_dim,
                residual_window=residual_window,
                device=device,
            )
        self.config = config
        self.rotator = rotator if rotator is not None else IdentityRotator(head_dim, device=device)
        self.padded_dim = self.rotator.padded_dim()
        self.rabitq_config = RabitqConfig(total_bits=config.total_bits)

        # Precompute t_const
        if self.rabitq_config.ex_bits > 0:
            from .quantizer import compute_const_scaling_factor

            self.rabitq_config.t_const = compute_const_scaling_factor(
                self.padded_dim, self.rabitq_config.ex_bits
            )

        # Per-layer state
        self._key_chunks: Dict[int, List[List[QuantizedVector]]] = {}
        self._val_chunks: Dict[int, List[List[QuantizedVector]]] = {}
        self._recent_k: Dict[int, List[torch.Tensor]] = {}
        self._recent_v: Dict[int, List[torch.Tensor]] = {}
        self._centroid_k: Dict[int, torch.Tensor] = {}
        self._centroid_v: Dict[int, torch.Tensor] = {}
        self._total_seq: Dict[int, int] = {}
        self._is_fitted: Dict[int, bool] = {}

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
        """
        B, H, S_new, D = key_states.shape
        device = key_states.device

        # Initialize layer state
        if layer_idx not in self._key_chunks:
            self._key_chunks[layer_idx] = []
            self._val_chunks[layer_idx] = []
            self._recent_k[layer_idx] = []
            self._recent_v[layer_idx] = []
            self._centroid_k[layer_idx] = torch.zeros(self.padded_dim, device=device)
            self._centroid_v[layer_idx] = torch.zeros(self.padded_dim, device=device)
            self._total_seq[layer_idx] = 0
            self._is_fitted[layer_idx] = False

        self._total_seq[layer_idx] = self._total_seq.get(layer_idx, 0) + S_new

        # Add to recent buffer
        self._recent_k[layer_idx].append(key_states)
        self._recent_v[layer_idx].append(value_states)

        # Concatenate recent
        recent_k = torch.cat(self._recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._recent_v[layer_idx], dim=2)

        rw = self.config.residual_window

        # Compress overflow
        if recent_k.shape[2] > rw and rw > 0:
            overflow = recent_k.shape[2] - rw

            to_compress_k = recent_k[:, :, :overflow, :]
            to_compress_v = recent_v[:, :, :overflow, :]

            # Fit centroids on first compression
            if not self._is_fitted[layer_idx]:
                self._fit_centroids(layer_idx, to_compress_k, to_compress_v)
                self._is_fitted[layer_idx] = True

            # Compress overflow tokens
            qvs_k = self._compress_tokens(to_compress_k, self._centroid_k[layer_idx])
            qvs_v = self._compress_tokens(to_compress_v, self._centroid_v[layer_idx])

            self._key_chunks[layer_idx].append(qvs_k)
            self._val_chunks[layer_idx].append(qvs_v)

            # Keep only window
            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._recent_k[layer_idx] = [recent_k]
            self._recent_v[layer_idx] = [recent_v]

        # Decompress all chunks + recent
        parts_k = []
        parts_v = []

        for chunk_k, chunk_v in zip(self._key_chunks[layer_idx], self._val_chunks[layer_idx]):
            dk = self._decompress_tokens(chunk_k, self._centroid_k[layer_idx], B, H, D)
            dv = self._decompress_tokens(chunk_v, self._centroid_v[layer_idx], B, H, D)
            parts_k.append(dk.to(key_states.dtype))
            parts_v.append(dv.to(value_states.dtype))

        recent_k = torch.cat(self._recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._recent_v[layer_idx], dim=2)
        parts_k.append(recent_k)
        parts_v.append(recent_v)

        full_k = torch.cat(parts_k, dim=2)
        full_v = torch.cat(parts_v, dim=2)

        return full_k, full_v

    def get_kv(
        self, layer_idx: int, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return decompressed full K and V for a layer without updating."""
        if layer_idx not in self._recent_k or len(self._recent_k[layer_idx]) == 0:
            # Empty cache
            device = self.config.device
            return (
                torch.zeros(1, 1, 0, self.config.head_dim, device=device, dtype=dtype),
                torch.zeros(1, 1, 0, self.config.head_dim, device=device, dtype=dtype),
            )

        recent_k = torch.cat(self._recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._recent_v[layer_idx], dim=2)
        B, H, _, D = recent_k.shape

        parts_k = []
        parts_v = []

        for chunk_k, chunk_v in zip(self._key_chunks[layer_idx], self._val_chunks[layer_idx]):
            dk = self._decompress_tokens(chunk_k, self._centroid_k[layer_idx], B, H, D)
            dv = self._decompress_tokens(chunk_v, self._centroid_v[layer_idx], B, H, D)
            parts_k.append(dk.to(dtype))
            parts_v.append(dv.to(dtype))

        parts_k.append(recent_k)
        parts_v.append(recent_v)

        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seq.get(layer_idx, 0)

    def get_max_length(self) -> Optional[int]:
        return None

    def reset(self):
        self._key_chunks.clear()
        self._val_chunks.clear()
        self._recent_k.clear()
        self._recent_v.clear()
        self._centroid_k.clear()
        self._centroid_v.clear()
        self._total_seq.clear()
        self._is_fitted.clear()

    def _fit_centroids(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        """Compute centroids from sample tokens."""
        k_rot = self._prepare_and_rotate(keys)
        v_rot = self._prepare_and_rotate(values)
        self._centroid_k[layer_idx] = k_rot.reshape(-1, self.padded_dim).mean(dim=0)
        self._centroid_v[layer_idx] = v_rot.reshape(-1, self.padded_dim).mean(dim=0)

    def _prepare_and_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape, pad, and rotate vectors."""
        B, H, S, D = x.shape
        x = x.reshape(B * H * S, D)
        if D < self.padded_dim:
            x = torch.nn.functional.pad(x, (0, self.padded_dim - D))
        x = self.rotator.rotate(x)
        return x

    def _inverse_rotate_and_crop(self, x: torch.Tensor, D: int) -> torch.Tensor:
        """Inverse rotate and crop to original dimension."""
        x = self.rotator.inverse_rotate(x)
        if D < self.padded_dim:
            x = x[..., :D]
        return x

    def _compress_tokens(
        self, states: torch.Tensor, centroid: torch.Tensor
    ) -> List[QuantizedVector]:
        """Compress a tensor of tokens into a list of QuantizedVectors."""
        rot = self._prepare_and_rotate(states)
        flat = rot.reshape(-1, self.padded_dim)
        return [
            quantize_vector(flat[i], centroid, self.rabitq_config) for i in range(flat.shape[0])
        ]

    def _decompress_tokens(
        self, qvs: List[QuantizedVector], centroid: torch.Tensor, B: int, H: int, D: int
    ) -> torch.Tensor:
        """Decompress a list of QuantizedVectors back to token tensor."""
        flat = torch.stack([reconstruct_vector(centroid, qv) for qv in qvs])
        flat = self._inverse_rotate_and_crop(flat, D)
        S = len(qvs) // (B * H)
        return flat.reshape(B, H, S, D)


# Alias for backward compatibility
V3Cache = RaBitQCache
