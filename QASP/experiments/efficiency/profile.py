"""Real efficiency profiling for QASP models.

When a model is provided, measures actual tokens/sec, peak memory, and latency.
When ``model=None``, falls back to deterministic estimates.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn


def profile_qasp(
    model: nn.Module | None = None,
    prompt_len: int = 128,
    gen_len: int = 64,
    batch_size: int = 1,
    device: str = "cpu",
    quick: bool = True,
) -> dict[str, float]:
    """Profile generation throughput and memory.

    Args:
        model: QASP model to profile.  If ``None``, returns deterministic stubs.
        prompt_len: Number of tokens in the prompt.
        gen_len: Number of new tokens to generate.
        batch_size: Batch size for generation.
        device: torch device string.
        quick: If True, reduce ``gen_len`` to 8 and return stubs when
            ``model`` is missing.

    Returns:
        Dictionary with ``tokens_per_second``, ``memory_gb``, ``latency_ms``.
    """

    if quick:
        gen_len = min(gen_len, 8)

    if model is None:
        # Deterministic stub for CI
        if quick:
            return {
                "tokens_per_second": 112.0,
                "memory_gb": 2.5,
                "latency_ms": 9.3,
            }
        return {
            "tokens_per_second": 109.0,
            "memory_gb": 2.7,
            "latency_ms": 10.1,
        }

    if not hasattr(model, "config"):
        raise ValueError("model must have a config attribute with vocab_size")

    vocab_size = model.config.vocab_size
    model = model.to(device)
    model.eval()

    # Warm-up
    dummy = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
    with torch.no_grad():
        _ = model(dummy)

    # Synchronize before timing
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.no_grad():
        input_ids = dummy.clone()
        for _ in range(gen_len):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    total_tokens = batch_size * gen_len
    tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / gen_len) * 1000.0

    memory_gb = 0.0
    if device.startswith("cuda"):
        memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        # CPU memory estimate: model params in GB * 2 (activations rough guess)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        memory_gb = (param_bytes * 2) / (1024**3)

    return {
        "tokens_per_second": round(tokens_per_second, 1),
        "memory_gb": round(memory_gb, 2),
        "latency_ms": round(latency_ms, 2),
    }
