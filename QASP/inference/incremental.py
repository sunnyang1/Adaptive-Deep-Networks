"""Incremental prefill / step API for QASP models.

``prefill`` runs one full forward pass while snapshotting per-layer K/V caches.
That pass matches the paper's **canonical full-sequence** semantics for
value-weighted AttnRes when AttnRes is enabled.

``step`` issues one new query against cached keys/values (cost scales with
model depth and cache length, as in standard cached attention).  If the underlying
model uses AttnRes with block statistics recomputed from a **prefix** history,
logits need not match a fresh ``forward`` over the extended sequence; the QASP
paper does not claim bit-identical equivalence between that incremental path
and the reference forward (see package docstring in ``QASP/__init__.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from QASP.inference.kv_cache import KVCache


class _IncrementalModel(Protocol):
    """Structural type for models that expose incremental prefill/step APIs."""

    def prefill(self, input_ids: Tensor) -> tuple[Tensor, KVCache]: ...

    def step(self, last_token: Tensor, cache: KVCache) -> Tensor: ...

    def eval(self) -> "_IncrementalModel": ...


@dataclass
class IncrementalState:
    """Mutable inference state carried across decoding steps."""

    cache: KVCache
    next_logits: Tensor

    @property
    def seq_len(self) -> int:
        return self.cache.seq_len


class IncrementalInference:
    """Thin wrapper around ``prefill`` / ``step`` for autoregressive loops.

    Prefer evaluating with :meth:`QASP.models.qasp_transformer.QASPTransformer.forward`
    when you need strict alignment with the paper's full-sequence definitions.
    """

    def __init__(self, model: _IncrementalModel) -> None:
        if not hasattr(model, "prefill") or not hasattr(model, "step"):
            raise TypeError(
                "`model` must expose `prefill(input_ids)` and "
                "`step(last_token, cache)` (see QASPTransformer)."
            )
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def prefill(self, input_ids: Tensor) -> IncrementalState:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        logits, cache = self.model.prefill(input_ids)
        return IncrementalState(cache=cache, next_logits=logits[:, -1, :])

    @torch.no_grad()
    def step(self, state: IncrementalState) -> Tensor:
        """Emit one token from ``state.next_logits`` and advance the cache."""

        if state.next_logits.ndim != 2:
            raise ValueError("state.next_logits must have shape [B, vocab_size]")

        next_token = torch.argmax(state.next_logits, dim=-1, keepdim=True)
        state.next_logits = self.model.step(next_token, state.cache)
        return next_token
