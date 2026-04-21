"""End-to-end test for QASPTransformer.adapt_at_test_time (paper Section 5.4)."""

from __future__ import annotations

import torch

from QASP.configs.qasp import QASPConfig
from QASP.models.qasp_transformer import create_qasp_transformer


def _make_small_model() -> "object":
    torch.manual_seed(0)
    return create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=32,
        attnres_blocks=2,
        adapt_rank=8,
    )


def test_adapt_at_test_time_noop_under_confident_logits() -> None:
    """Confident logits should keep ponder gate off and leave W_l unchanged."""

    model = _make_small_model()
    before = [layer.stiefel_query.detach().clone() for layer in model.layers]

    confident_logits = torch.full((1, 4, model.config.vocab_size), -1e3)
    confident_logits[..., 0] = 1e3

    def loss_fn(_idx: int, weights: torch.Tensor) -> torch.Tensor:
        return (weights**2).sum()

    fired = model.adapt_at_test_time(loss_fn, confident_logits)
    assert fired is False
    for layer, snapshot in zip(model.layers, before):
        assert torch.allclose(layer.stiefel_query.data, snapshot)


def test_adapt_at_test_time_updates_layers_and_preserves_stiefel() -> None:
    """When ponder gate fires, every W_l must change yet remain near-orthonormal."""

    model = _make_small_model()
    before = [layer.stiefel_query.detach().clone() for layer in model.layers]

    uniform_logits = torch.zeros(1, 4, model.config.vocab_size)

    targets = [torch.randn_like(w) for w in before]

    def loss_fn(idx: int, weights: torch.Tensor) -> torch.Tensor:
        return ((weights - targets[idx]) ** 2).sum()

    fired = model.adapt_at_test_time(
        loss_fn,
        uniform_logits,
        qasp_config=QASPConfig(step_size=0.1, num_adapt_steps=3, ns_iters=5),
    )
    assert fired is True

    for layer, snapshot in zip(model.layers, before):
        assert not torch.allclose(layer.stiefel_query.data, snapshot)
        gram = layer.stiefel_query.data.transpose(0, 1) @ layer.stiefel_query.data
        identity = torch.eye(layer.stiefel_query.shape[1])
        assert torch.allclose(gram, identity, atol=1e-2, rtol=1e-2)


def test_adapt_at_test_time_respects_quality_scores() -> None:
    """Zero-mean quality scores must shrink the effective update magnitude."""

    model = _make_small_model()
    before = [layer.stiefel_query.detach().clone() for layer in model.layers]

    uniform_logits = torch.zeros(1, 4, model.config.vocab_size)
    targets = [torch.randn_like(w) for w in before]

    def loss_fn(idx: int, weights: torch.Tensor) -> torch.Tensor:
        return ((weights - targets[idx]) ** 2).sum()

    zero_quality = torch.zeros(8)

    fired = model.adapt_at_test_time(
        loss_fn,
        uniform_logits,
        quality_scores=zero_quality,
        qasp_config=QASPConfig(step_size=0.5, num_adapt_steps=5, ns_iters=5),
    )
    assert fired is True

    for layer, snapshot in zip(model.layers, before):
        gram_before = snapshot.transpose(0, 1) @ snapshot
        gram_after = layer.stiefel_query.data.transpose(0, 1) @ layer.stiefel_query.data
        assert torch.allclose(gram_before, gram_after, atol=1e-2)
        assert torch.allclose(layer.stiefel_query.data, snapshot, atol=1e-2)
