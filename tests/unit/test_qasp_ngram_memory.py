"""Unit tests for the QASP n-gram external memory (paper Section 5.3)."""

from __future__ import annotations

import pytest
import torch

from QASP.models.ngram_memory import NgramMemory
from QASP.models.value_weighted_engram import ValueWeightedEngram


def test_ngram_memory_write_then_lookup_roundtrip() -> None:
    memory = NgramMemory(table_size=128, hidden_size=16, n_gram=3)
    tokens = (7, 11, 13)
    vector = torch.randn(16)
    idx = memory.write(tokens, vector, quality=0.75)

    retrieved_vec, retrieved_qual, populated = memory.lookup(tokens)

    assert populated is True
    assert idx == memory.hash_index(tokens)
    assert torch.allclose(retrieved_vec, vector)
    assert pytest.approx(float(retrieved_qual), abs=1e-6) == 0.75


def test_ngram_memory_unpopulated_slot_returns_zeros() -> None:
    memory = NgramMemory(table_size=32, hidden_size=8, n_gram=3)
    vec, qual, populated = memory.lookup((1, 2, 3))
    assert populated is False
    assert torch.allclose(vec, torch.zeros(8))
    assert float(qual) == 0.0


def test_ngram_memory_batch_lookup_fills_written_positions() -> None:
    memory = NgramMemory(table_size=64, hidden_size=4, n_gram=3)
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    memory.write((5, 6, 7), target, quality=0.9)

    input_ids = torch.tensor([[0, 5, 6, 7, 8]])
    vectors, qualities = memory.batch_lookup(input_ids)

    assert vectors.shape == (1, 5, 4)
    assert qualities.shape == (1, 5)
    assert torch.allclose(vectors[0, 0], torch.zeros(4))
    assert torch.allclose(vectors[0, 1], torch.zeros(4))
    assert torch.allclose(vectors[0, 3], target)
    assert float(qualities[0, 3]) == pytest.approx(0.9, abs=1e-6)
    assert float(qualities[0, 4]) == 0.0


def test_ngram_memory_rejects_bad_inputs() -> None:
    memory = NgramMemory(table_size=16, hidden_size=4, n_gram=2)
    with pytest.raises(ValueError):
        memory.hash_index(())
    with pytest.raises(ValueError):
        memory.write((1, 2), torch.zeros(5), quality=0.1)
    with pytest.raises(ValueError):
        memory.batch_lookup(torch.zeros(3, dtype=torch.long))


def test_value_weighted_engram_content_gate_default_uses_hidden_state() -> None:
    """When ``gate`` is None, alpha = sigmoid(w_g^T h); delta must be non-zero."""

    torch.manual_seed(0)
    engram = ValueWeightedEngram(hidden_size=8)
    hidden = torch.randn(2, 3, 8)
    memory_vec = torch.randn(2, 8)
    memory_quality = torch.tensor([2.0, 2.0])

    with torch.no_grad():
        engram.gate_proj.weight.fill_(0.1)
        engram.gate_proj.bias.zero_()

    fused = engram(hidden, memory_vec, memory_quality)
    delta = fused - hidden
    assert delta.abs().max() > 0.0


def test_value_weighted_engram_zero_memory_quality_has_half_gate() -> None:
    """sigmoid(rho=0) = 0.5, so contribution scales by 0.5 * alpha * m."""

    engram = ValueWeightedEngram(hidden_size=4)
    hidden = torch.zeros(1, 1, 4)
    memory_vec = torch.ones(1, 4)
    memory_quality = torch.zeros(1)

    with torch.no_grad():
        engram.gate_proj.weight.zero_()
        engram.gate_proj.bias.zero_()

    fused = engram(hidden, memory_vec, memory_quality)
    expected = torch.full((1, 1, 4), 0.5 * 0.5 * 1.0)
    assert torch.allclose(fused, expected, atol=1e-6)
