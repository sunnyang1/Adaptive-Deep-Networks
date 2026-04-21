from __future__ import annotations

from dataclasses import dataclass

import torch

from matdo_new.modeling.config import KVQuantizationConfig
from matdo_new.repo_imports import repo_root_on_path

with repo_root_on_path():
    from src.rabitq.api import CompressedKV, RaBitQ
    from src.rabitq.cache import RaBitQCache


@dataclass(frozen=True)
class QuantizedKVPackage:
    """Container returned by the KV quantization adapter."""

    compressed: dict[str, CompressedKV]
    original_key_shape: tuple[int, ...]
    original_value_shape: tuple[int, ...]


class KVQuantizationAdapter:
    """Thin MATDO-new wrapper around the repo's RaBitQ implementation."""

    def __init__(self, config: KVQuantizationConfig | None = None) -> None:
        self.config = config or KVQuantizationConfig()
        self._backend = RaBitQ(config=self.config.to_backend_config())

    @property
    def backend(self) -> RaBitQ:
        return self._backend

    def fit(self, sample_keys: torch.Tensor, sample_values: torch.Tensor) -> 'KVQuantizationAdapter':
        self._backend.fit(sample_keys, sample_values)
        return self

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> QuantizedKVPackage:
        self._validate_inputs(keys, values)
        compressed = self._backend.compress(keys, values)
        return QuantizedKVPackage(
            compressed=compressed,
            original_key_shape=tuple(keys.shape),
            original_value_shape=tuple(values.shape),
        )

    def decompress(self, package: QuantizedKVPackage) -> tuple[torch.Tensor, torch.Tensor]:
        return self._backend.decompress(package.compressed)

    def as_cache(self) -> RaBitQCache:
        return self._backend.as_cache(residual_window=self.config.residual_window)

    def memory_stats(
        self,
        *,
        seq_len: int,
        num_layers: int = 1,
        batch_size: int = 1,
        num_heads: int = 1,
    ) -> dict[str, float]:
        return self._backend.memory_stats(
            seq_len=seq_len,
            num_layers=num_layers,
            batch_size=batch_size,
            num_heads=num_heads,
        )

    def _validate_inputs(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.shape != values.shape:
            raise ValueError('keys and values must have matching shapes')
        if keys.shape[-1] != self.config.head_dim:
            raise ValueError(
                f'expected head_dim {self.config.head_dim}, got {keys.shape[-1]}'
            )
