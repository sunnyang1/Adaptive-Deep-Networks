from __future__ import annotations

from dataclasses import dataclass, field

from adn.matdo_e.repo_imports import repo_root_on_path

with repo_root_on_path():
    from src.models.configs import ModelConfig, get_config
    from src.qttt.adaptation import qTTTConfig as BackendQTTTConfig
    from src.rabitq.api import RaBitQConfig as BackendRaBitQConfig


@dataclass(frozen=True)
class KVQuantizationConfig:
    """Small adapter config for RaBitQ-backed KV compression."""

    total_bits: int = 2
    head_dim: int = 64
    residual_window: int = 128
    device: str = 'cpu'
    use_rotation: bool = True
    rotator_type: str = 'fht'
    rotator_seed: int = 42
    use_fast_quantization: bool = True

    def to_backend_config(self) -> BackendRaBitQConfig:
        return BackendRaBitQConfig(
            total_bits=self.total_bits,
            use_rotation=self.use_rotation,
            rotator_type=self.rotator_type,
            rotator_seed=self.rotator_seed,
            residual_window=self.residual_window,
            device=self.device,
            head_dim=self.head_dim,
            use_fast_quantization=self.use_fast_quantization,
        )


@dataclass(frozen=True)
class QueryAdaptationConfig:
    """Adapter config for the repo's qTTT implementation."""

    num_steps: int = 4
    learning_rate: float = 0.01
    span_length: int = 128
    target_type: str = 'pseudo_query'
    margin_temperature: float = 1.0
    early_stop_threshold: float | None = None

    def to_backend_config(self) -> BackendQTTTConfig:
        return BackendQTTTConfig(
            num_steps=self.num_steps,
            learning_rate=self.learning_rate,
            span_length=self.span_length,
            target_type=self.target_type,
            margin_temperature=self.margin_temperature,
            early_stop_threshold=self.early_stop_threshold,
        )


@dataclass(frozen=True)
class ExternalMemoryConfig:
    """Minimal external-memory toggle used by MATDOModel."""

    enabled: bool = False
    max_entries: int = 0


@dataclass(frozen=True)
class MATDOModelConfig:
    """Thin configuration surface for the MATDO modeling adapters."""

    model_size: str = 't4'
    use_attnres: bool = True
    use_qttt: bool = False
    use_engram: bool = False
    sampling_temperature: float = 1.0
    sampling_top_k: int | None = None
    quantization: KVQuantizationConfig = field(default_factory=KVQuantizationConfig)
    query_adaptation: QueryAdaptationConfig = field(default_factory=QueryAdaptationConfig)
    external_memory: ExternalMemoryConfig = field(default_factory=ExternalMemoryConfig)

    def build_backend_config(self) -> ModelConfig:
        backend = get_config(self.model_size)
        backend.use_engram = self.use_engram
        return backend
