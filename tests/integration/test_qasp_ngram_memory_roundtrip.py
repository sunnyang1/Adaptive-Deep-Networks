"""Round-trip tests for NgramMemory write → lookup integration."""

from __future__ import annotations

import torch

from QASP.models import create_qasp_transformer
from QASP.models.ngram_memory import NgramMemory


def test_ngram_memory_roundtrip_write_then_lookup() -> None:
    """batch_write followed by batch_lookup must return stored vectors/qualities."""

    mem = NgramMemory(table_size=128, hidden_size=16, n_gram=3)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    vectors = torch.randn(1, 5, 16)
    qualities = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)

    mem.batch_write(input_ids, vectors, qualities)

    looked_up_vec, looked_up_qual = mem.batch_lookup(input_ids)

    # Positions t < n_gram - 1 should be zero
    assert torch.allclose(looked_up_vec[:, :2], torch.zeros(1, 2, 16))
    assert torch.allclose(looked_up_qual[:, :2], torch.zeros(1, 2))

    # Positions t >= n_gram - 1 should match what we wrote
    # For t=2, ngram=[1,2,3], mean_qual=(0.1+0.2+0.3)/3=0.2
    assert torch.allclose(looked_up_vec[:, 2], vectors[:, 2])
    assert looked_up_qual[0, 2].item() == pytest.approx(0.2, abs=1e-5)

    # For t=3, ngram=[2,3,4], mean_qual=(0.2+0.3+0.4)/3=0.3
    assert torch.allclose(looked_up_vec[:, 3], vectors[:, 3])
    assert looked_up_qual[0, 3].item() == pytest.approx(0.3, abs=1e-5)

    # For t=4, ngram=[3,4,5], mean_qual=(0.3+0.4+0.5)/3=0.4
    assert torch.allclose(looked_up_vec[:, 4], vectors[:, 4])
    assert looked_up_qual[0, 4].item() == pytest.approx(0.4, abs=1e-5)


def test_prefill_populates_engram_memory() -> None:
    """QASPTransformer.prefill must write non-zero entries to engram_memory."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        use_engram=True,
        engram_n_gram=3,
    )
    model.eval()

    input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    assert model.engram_memory is not None

    # Before prefill, memory should be empty
    vec_before, qual_before = model.engram_memory.batch_lookup(input_ids)
    assert torch.allclose(vec_before, torch.zeros_like(vec_before))
    assert torch.allclose(qual_before, torch.zeros_like(qual_before))

    with torch.no_grad():
        model.prefill(input_ids)

    # After prefill, at least some positions should be populated
    vec_after, qual_after = model.engram_memory.batch_lookup(input_ids)
    assert qual_after[:, 2:].sum().item() > 0.0, "Expected some non-zero qualities after prefill"


import pytest
