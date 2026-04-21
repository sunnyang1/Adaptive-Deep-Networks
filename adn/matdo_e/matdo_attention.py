from __future__ import annotations

import torch

from adn.matdo_e.repo_imports import repo_root_on_path

with repo_root_on_path():
    from src.qttt.adaptation import KVCache, compute_attention_with_query


def apply_frozen_kv_attention(
    query: torch.Tensor,
    kv_cache: KVCache,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route attention through the existing qTTT frozen-KV helper."""
    return compute_attention_with_query(query, kv_cache, mask=mask)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token from logits with optional temperature and top-k filtering."""
    if temperature <= 0.0:
        raise ValueError('temperature must be positive')

    squeeze_result = logits.dim() == 1
    working_logits = logits.unsqueeze(0) if squeeze_result else logits.clone()
    working_logits = working_logits / temperature

    if top_k is not None:
        if top_k <= 0:
            raise ValueError('top_k must be positive when provided')
        k = min(int(top_k), working_logits.shape[-1])
        top_indices = torch.topk(working_logits, k=k, dim=-1).indices
        top_mask = torch.zeros_like(working_logits, dtype=torch.bool)
        top_mask.scatter_(dim=-1, index=top_indices, value=True)
        working_logits = working_logits.masked_fill(~top_mask, float('-inf'))

    probs = torch.softmax(working_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return sampled.squeeze(0) if squeeze_result else sampled
