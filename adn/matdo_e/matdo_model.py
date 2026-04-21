from __future__ import annotations

from dataclasses import replace

import torch

from adn.matdo_e.policy import PolicyDecision
from adn.matdo_e.matdo_attention import sample_next_token
from adn.matdo_e.matdo_blocks import RuntimeHandles
from adn.matdo_e.matdo_model_config import ExternalMemoryConfig, MATDOModelConfig
from adn.matdo_e.matdo_external_memory import ExternalMemoryHandle
from adn.matdo_e.matdo_kv_quantization import KVQuantizationAdapter
from adn.matdo_e.matdo_query_adaptation import QueryAdaptationAdapter
from adn.matdo_e.matdo_scope_memory import ScopeMemory
from adn.matdo_e.runtime_materialize import MaterializedPolicy, materialize_policy


class MATDOModel:
    """Minimal adapter-oriented model surface for MATDO-new."""

    def __init__(self, config: MATDOModelConfig | None = None, *, backend: object | None = None) -> None:
        self.config = config or MATDOModelConfig()
        self.backend = backend
        self.backend_config = self.config.build_backend_config()

    def prepare_runtime_handles(
        self,
        policy: PolicyDecision | MaterializedPolicy | None = None,
        *,
        device: str | None = None,
    ) -> RuntimeHandles:
        materialized = materialize_policy(policy)
        kv_config = replace(
            self.config.quantization,
            total_bits=(
                materialized.quantization_bits
                if materialized is not None
                else self.config.quantization.total_bits
            ),
            head_dim=self.backend_config.head_dim,
            device=device or self.config.quantization.device,
        )
        kv_quantization = KVQuantizationAdapter(kv_config)

        if materialized is not None:
            use_qttt = materialized.t_steps > 0
        else:
            use_qttt = self.config.use_qttt
        query_adaptation = None
        if use_qttt:
            query_config = replace(
                self.config.query_adaptation,
                num_steps=(
                    materialized.t_steps
                    if materialized is not None and materialized.t_steps > 0
                    else self.config.query_adaptation.num_steps
                ),
            )
            query_adaptation = QueryAdaptationAdapter(
                query_config,
                hidden_dim=self.backend_config.hidden_dim,
                num_heads=self.backend_config.num_heads,
            )

        external_entries = 0
        external_enabled = self.config.external_memory.enabled
        if materialized is not None and materialized.use_engram and materialized.engram_entries > 0:
            external_enabled = True
            external_entries = materialized.engram_entries
        elif self.config.external_memory.enabled:
            external_entries = self.config.external_memory.max_entries
        external_memory = ExternalMemoryHandle(
            ExternalMemoryConfig(enabled=external_enabled, max_entries=external_entries)
        )

        scope_capacity = (
            materialized.m_blocks if materialized is not None else self.backend_config.num_blocks
        )
        scope_memory = ScopeMemory(capacity=scope_capacity)

        return RuntimeHandles(
            policy=materialized,
            kv_quantization=kv_quantization,
            query_adaptation=query_adaptation,
            external_memory=external_memory,
            scope_memory=scope_memory,
        )

    def sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> int | torch.Tensor:
        sampled = sample_next_token(
            logits,
            temperature=(
                self.config.sampling_temperature if temperature is None else temperature
            ),
            top_k=self.config.sampling_top_k if top_k is None else top_k,
            generator=generator,
        )
        if logits.dim() == 1:
            return int(sampled.item())
        return sampled
