"""Shape tests for the minimal QASP model stack."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
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


def test_value_weighted_engram_gate_scalar_tensor_shape() -> None:
    """Scalar tensor gate should broadcast over [B, T, D]."""

    module = ValueWeightedEngram(hidden_size=16)
    hidden = torch.randn(2, 5, 16)
    memory_vec = torch.randn(2, 16)
    memory_quality = torch.randn(2)
    gate = torch.tensor(0.75)

    fused = module(hidden, memory_vec, memory_quality, gate=gate)

    assert fused.shape == hidden.shape


def test_value_weighted_engram_gate_batch_shape() -> None:
    """Per-batch gate [B] should broadcast over sequence and hidden dims."""

    module = ValueWeightedEngram(hidden_size=16)
    hidden = torch.randn(2, 5, 16)
    memory_vec = torch.randn(2, 16)
    memory_quality = torch.randn(2)
    gate = torch.tensor([0.5, 1.25])

    fused = module(hidden, memory_vec, memory_quality, gate=gate)

    assert fused.shape == hidden.shape


def test_value_weighted_engram_gate_batch_time_shape() -> None:
    """Per-token gate [B, T] should broadcast over hidden dim."""

    module = ValueWeightedEngram(hidden_size=16)
    hidden = torch.randn(2, 5, 16)
    memory_vec = torch.randn(2, 16)
    memory_quality = torch.randn(2)
    gate = torch.full((2, 5), 0.8)

    fused = module(hidden, memory_vec, memory_quality, gate=gate)

    assert fused.shape == hidden.shape


def test_value_weighted_engram_invalid_gate_shape_raises() -> None:
    """Unsupported gate shape should raise ValueError."""

    module = ValueWeightedEngram(hidden_size=16)
    hidden = torch.randn(2, 5, 16)
    memory_vec = torch.randn(2, 16)
    memory_quality = torch.randn(2)
    invalid_gate = torch.ones(2, 5, 1)

    with pytest.raises(ValueError, match="gate"):
        module(hidden, memory_vec, memory_quality, gate=invalid_gate)


def test_value_weighted_attnres_output_shape() -> None:
    """AttnRes aggregation should return [B, T, D]."""

    module = ValueWeightedAttnRes(hidden_size=48)
    hidden = torch.randn(2, 9, 48)
    block_repr = torch.randn(2, 5, 48)
    quality_scores = torch.randn(2, 5)

    residual = module(hidden, block_repr, quality_scores)

    assert residual.shape == hidden.shape


def test_qasp_transformer_forward_without_attnres_or_engram() -> None:
    """Transformer should run with both optional hooks disabled."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=32,
        use_attnres=False,
        use_engram=False,
    )
    input_ids = torch.randint(0, 64, (2, 6))

    logits = model(input_ids)

    assert logits.shape == (2, 6, 64)

