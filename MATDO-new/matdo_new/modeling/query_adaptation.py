from __future__ import annotations

from dataclasses import dataclass

import torch

from matdo_new.modeling.config import QueryAdaptationConfig
from matdo_new.repo_imports import repo_root_on_path

with repo_root_on_path():
    from src.qttt.adaptation import KVCache, QueryOnlyTTT


@dataclass(frozen=True)
class QueryAdaptationResult:
    """Normalized qTTT adapter output."""

    adapted_query: torch.Tensor
    loss_history: tuple[float, ...]


class QueryAdaptationAdapter:
    """Small adapter over the existing query-only TTT module."""

    def __init__(
        self,
        config: QueryAdaptationConfig | None = None,
        *,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        self.config = config or QueryAdaptationConfig()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self._backend = QueryOnlyTTT(
            self.config.to_backend_config(),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

    @property
    def backend(self) -> QueryOnlyTTT:
        return self._backend

    def make_kv_cache(self, keys: torch.Tensor, values: torch.Tensor) -> KVCache:
        return KVCache(keys=keys, values=values)

    def adapt_pseudo_query(
        self,
        pseudo_query: torch.Tensor,
        kv_cache: KVCache,
        seq_positions: torch.Tensor,
        distractor_positions: torch.Tensor | None = None,
    ) -> QueryAdaptationResult:
        adapted, losses = self._backend.adapt_pseudo_query(
            pseudo_query,
            kv_cache,
            seq_positions,
            distractor_positions,
        )
        return QueryAdaptationResult(
            adapted_query=adapted,
            loss_history=tuple(float(loss) for loss in losses),
        )

    def adapt_query_projection(
        self,
        queries: torch.Tensor,
        kv_cache: KVCache,
        seq_positions: torch.Tensor | None = None,
        distractor_positions: torch.Tensor | None = None,
    ) -> QueryAdaptationResult:
        adapted, losses = self._backend.adapt_query_projection(
            queries,
            kv_cache,
            seq_positions=seq_positions,
            distractor_positions=distractor_positions,
        )
        return QueryAdaptationResult(
            adapted_query=adapted,
            loss_history=tuple(float(loss) for loss in losses),
        )

    def compute_flops(self, *, batch_size: int, seq_len: int, span_len: int) -> dict[str, int]:
        return self._backend.compute_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            span_len=span_len,
        )
