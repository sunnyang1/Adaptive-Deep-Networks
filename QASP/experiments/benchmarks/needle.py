"""Needle-in-a-haystack benchmark for QASP models.

Implements the standard long-context retrieval protocol:
1. Generate a random token sequence (haystack).
2. Insert a unique "needle" token sequence at a sampled position.
3. Append a query prompt.
4. Run the model and check whether the generated answer matches the needle.

Because the typical QASP test model is randomly initialized, absolute
accuracies will be near random.  The value of this runner is the *pipeline*:
end-to-end insertion, forward pass, and exact-match evaluation.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


PositionDistribution = Literal["uniform", "front", "end", "middle"]


def _sample_needle_position(seq_len: int, distribution: PositionDistribution) -> int:
    """Return an integer position in ``[0, seq_len)`` under the given distribution."""

    if distribution == "uniform":
        return int(torch.randint(0, seq_len, (1,)).item())
    if distribution == "front":
        # Front 25%
        max_pos = max(1, seq_len // 4)
        return int(torch.randint(0, max_pos, (1,)).item())
    if distribution == "end":
        # Last 25%
        min_pos = seq_len - max(1, seq_len // 4)
        return int(torch.randint(min_pos, seq_len, (1,)).item())
    # middle — central 50%
    quarter = max(1, seq_len // 4)
    return int(torch.randint(quarter, seq_len - quarter, (1,)).item())


def run_needle_benchmark(
    model: nn.Module | None = None,
    vocab_size: int = 128,
    context_length: int = 256,
    needle_len: int = 3,
    num_trials: int = 10,
    position_distribution: PositionDistribution = "uniform",
    device: str = "cpu",
    quick: bool = True,
) -> float:
    """Run needle-in-a-haystack retrieval and return exact-match accuracy.

    Args:
        model: A ``nn.Module`` with a ``forward(input_ids) -> [B, T, V]`` method.
            If ``None``, a tiny default transformer is created for smoke tests.
        vocab_size: Vocabulary size.  Needle tokens are drawn from the upper
            end of the vocabulary to minimize collisions with haystack tokens.
        context_length: Total sequence length including needle + query.
        needle_len: Length of the needle token sequence.
        num_trials: Number of independent trials to average over.
        position_distribution: Where to place the needle (see ``_sample_needle_position``).
        device: torch device string.
        quick: If True, reduce ``num_trials`` to 4 and ``context_length`` to 64
            for fast CI/smoke loops.

    Returns:
        Exact-match accuracy in ``[0.0, 1.0]``.
    """

    if quick:
        num_trials = min(num_trials, 4)
        context_length = min(context_length, 64)

    if model is None:
        from QASP.models import create_qasp_transformer

        model = create_qasp_transformer(
            vocab_size=vocab_size,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            max_position_embeddings=max(context_length * 2, 128),
            use_attnres=False,
            use_engram=False,
        )

    model = model.to(device)
    model.eval()

    # Reserve the top ``needle_len + 1`` tokens for needle + query
    query_token = vocab_size - 1
    needle_tokens = torch.arange(vocab_size - 1 - needle_len, vocab_size - 1)

    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_trials):
            # Haystack: random tokens from the lower vocabulary
            haystack = torch.randint(
                0, vocab_size - needle_len - 1, (1, context_length - needle_len - 1)
            )
            pos = _sample_needle_position(haystack.shape[1], position_distribution)

            # Insert needle
            prefix = haystack[:, :pos]
            suffix = haystack[:, pos:]
            sequence = torch.cat([prefix, needle_tokens.unsqueeze(0), suffix], dim=1)

            # Append query token
            input_ids = torch.cat([sequence, torch.tensor([[query_token]], device=sequence.device)], dim=1)
            input_ids = input_ids.to(device)

            logits = model(input_ids)
            # Predict the next token after the query
            pred = torch.argmax(logits[:, -1, :], dim=-1)

            # For a real trained model we would decode a longer answer.
            # With a random model we check if the first predicted token matches
            # the first needle token as a minimal signal.
            if pred.item() == needle_tokens[0].item():
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0
