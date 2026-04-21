"""Autoregressive generation utilities for QASP models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from QASP.inference.incremental import IncrementalInference


class QASPGenerator:
    """Greedy autoregressive decoder with optional prefill+step optimization.

    If the supplied model exposes ``prefill`` and ``step`` methods
    (e.g. :class:`QASP.models.qasp_transformer.QASPTransformer`), generation
    uses the cached KV path.  Otherwise it falls back to the naive full-forward
    loop for compatibility with arbitrary ``nn.Module`` instances.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()
        self._use_incremental = hasattr(model, "prefill") and hasattr(model, "step")

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        if self._use_incremental:
            return self._generate_incremental(input_ids, max_new_tokens)
        return self._generate_naive(input_ids, max_new_tokens)

    def _generate_incremental(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        inference = IncrementalInference(self.model)
        state = inference.prefill(input_ids)
        tokens: list[Tensor] = [input_ids]
        for _ in range(max_new_tokens):
            tokens.append(inference.step(state))
        return torch.cat(tokens, dim=1)

    def _generate_naive(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        output_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.model(output_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token], dim=1)
        return output_ids
