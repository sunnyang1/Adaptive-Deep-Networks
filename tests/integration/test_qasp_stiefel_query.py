"""Tests for the true matrix-level Stiefel query replacement (Story QASP-1)."""

from __future__ import annotations

import pytest
import torch

from QASP.adaptation.stiefel import project_to_stiefel
from QASP.configs.qasp import QASPConfig
from QASP.models import create_qasp_transformer


def _build_stiefel_query_model():
    torch.manual_seed(0)
    return create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=32,
        use_stiefel_query=True,
        use_attnres=False,
        use_engram=False,
    )


def test_stiefel_query_shape_is_square() -> None:
    """When use_stiefel_query=True, stiefel_query must be [d, d]."""

    model = _build_stiefel_query_model()
    for layer in model.layers:
        assert layer.stiefel_query.shape == (32, 32)


def test_stiefel_query_is_orthonormal() -> None:
    """Freshly initialized stiefel_query should have orthonormal columns."""

    model = _build_stiefel_query_model()
    for layer in model.layers:
        gram = layer.stiefel_query.data.transpose(0, 1) @ layer.stiefel_query.data
        identity = torch.eye(32)
        assert torch.allclose(gram, identity, atol=1e-2, rtol=1e-2)


def test_stiefel_query_forward_runs_end_to_end() -> None:
    """Forward pass must succeed with use_stiefel_query=True."""

    model = _build_stiefel_query_model()
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (1, 5, 64)


def test_stiefel_query_prefill_step_runs() -> None:
    """prefill and step must work with the Stiefel query path."""

    model = _build_stiefel_query_model()
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    with torch.no_grad():
        logits, cache = model.prefill(input_ids)
        assert logits.shape == (1, 4, 64)

        next_logits = model.step(torch.tensor([[2]], dtype=torch.long), cache)
        assert next_logits.shape == (1, 64)


def test_stiefel_query_adaptation_preserves_orthonormality() -> None:
    """After adapt_at_test_time, columns must remain orthonormal."""

    model = _build_stiefel_query_model()
    model.eval()

    uniform_logits = torch.zeros(1, 4, model.config.vocab_size)
    targets = [torch.randn_like(layer.stiefel_query.data) for layer in model.layers]

    def loss_fn(idx: int, weights: torch.Tensor) -> torch.Tensor:
        return ((weights - targets[idx]) ** 2).sum()

    fired = model.adapt_at_test_time(
        loss_fn,
        uniform_logits,
        qasp_config=QASPConfig(step_size=0.1, num_adapt_steps=3, ns_iters=15),
    )
    assert fired is True

    for layer in model.layers:
        gram = layer.stiefel_query.data.transpose(0, 1) @ layer.stiefel_query.data
        identity = torch.eye(32)
        assert torch.allclose(gram, identity, atol=1e-2, rtol=1e-2)


def test_stiefel_query_changes_forward_logits() -> None:
    """Mutating stiefel_query must change the forward output."""

    model = _build_stiefel_query_model()
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    with torch.no_grad():
        before = model(input_ids)

    for layer in model.layers:
        new_w = project_to_stiefel(torch.randn_like(layer.stiefel_query.data))
        layer.stiefel_query.data.copy_(new_w)

    with torch.no_grad():
        after = model(input_ids)

    assert not torch.allclose(before, after, atol=1e-4)


def test_stiefel_query_rejects_wrong_shape() -> None:
    """Using a non-square stiefel_query with use_stiefel_query=True must raise."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=32,
        use_stiefel_query=True,
        adapt_rank=8,  # != hidden_size
        use_attnres=False,
        use_engram=False,
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    # QASPLayer initializes stiefel_query as [d, d] when use_stiefel_query=True,
    # so we manually corrupt it to trigger the error.
    model.layers[0].stiefel_query.data = torch.randn(32, 8)

    with pytest.raises(ValueError, match="stiefel_query must have shape"):
        with torch.no_grad():
            model(input_ids)
