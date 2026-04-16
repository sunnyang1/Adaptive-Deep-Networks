"""Shape tests for the minimal QASP model stack."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QASP.models import ValueWeightedAttnRes, ValueWeightedEngram, create_qasp_transformer


def test_qasp_transformer_forward_output_shape() -> None:
    """Transformer forward should return logits [B, T, V]."""

    model = create_qasp_transformer(
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        attnres_blocks=4,
    )
    input_ids = torch.randint(0, 128, (3, 11))

    logits = model(input_ids)

    assert logits.shape == (3, 11, 128)


def test_value_weighted_engram_fuse_shape() -> None:
    """Engram fusion should preserve hidden state shape."""

    module = ValueWeightedEngram(hidden_size=32)
    hidden = torch.randn(2, 7, 32)
    memory_vec = torch.randn(2, 32)
    memory_quality = torch.randn(2)

    fused = module(hidden, memory_vec, memory_quality)

    assert fused.shape == hidden.shape


def test_value_weighted_attnres_output_shape() -> None:
    """AttnRes aggregation should return [B, T, D]."""

    module = ValueWeightedAttnRes(hidden_size=48)
    hidden = torch.randn(2, 9, 48)
    block_repr = torch.randn(2, 5, 48)
    quality_scores = torch.randn(2, 5)

    residual = module(hidden, block_repr, quality_scores)

    assert residual.shape == hidden.shape

