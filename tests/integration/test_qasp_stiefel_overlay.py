"""End-to-end tests for the Stiefel query overlay (paper §5.2).

The overlay adds ``scale · h · W_ℓ · W_ℓᵀ`` to the attention query input.
When ``stiefel_overlay_scale == 0`` the overlay is a no-op; when it is
non-zero, the QASPTransformer forward becomes a function of ``W_ℓ``, so
:meth:`adapt_at_test_time` actually affects downstream logits.
"""

from __future__ import annotations

import torch

from QASP.adaptation.stiefel import project_to_stiefel
from QASP.configs.qasp import QASPConfig
from QASP.models.qasp_transformer import create_qasp_transformer


def _tiny(overlay_scale: float):
    torch.manual_seed(0)
    return create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=32,
        attnres_blocks=2,
        adapt_rank=8,
        stiefel_overlay_scale=overlay_scale,
        use_engram=False,
    )


def test_overlay_scale_zero_is_exact_noop_across_layers() -> None:
    """Mutating W_ℓ with scale=0 must leave every forward logit unchanged."""

    model = _tiny(overlay_scale=0.0)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    baseline_logits = model(input_ids)

    for layer in model.layers:
        new_w = project_to_stiefel(torch.randn_like(layer.stiefel_query.data))
        layer.stiefel_query.data.copy_(new_w)

    mutated_logits = model(input_ids)
    assert torch.allclose(baseline_logits, mutated_logits, atol=1e-6, rtol=1e-6)


def test_overlay_scale_nonzero_makes_forward_depend_on_W() -> None:
    """With scale>0, replacing W_ℓ must change the forward logits."""

    model = _tiny(overlay_scale=0.25)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    before_logits = model(input_ids)

    torch.manual_seed(42)
    for layer in model.layers:
        new_w = project_to_stiefel(torch.randn_like(layer.stiefel_query.data))
        layer.stiefel_query.data.copy_(new_w)

    after_logits = model(input_ids)
    assert not torch.allclose(before_logits, after_logits, atol=1e-4)


def test_adapt_at_test_time_changes_forward_when_overlay_is_active() -> None:
    """The adaptation → forward loop closes only when overlay_scale > 0."""

    model = _tiny(overlay_scale=0.1)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    logits_before = model(input_ids)

    uniform_logits = torch.zeros(1, 4, model.config.vocab_size)
    targets = [torch.randn_like(layer.stiefel_query) for layer in model.layers]

    def loss_fn(idx: int, weights: torch.Tensor) -> torch.Tensor:
        return ((weights - targets[idx]) ** 2).sum()

    fired = model.adapt_at_test_time(
        loss_fn,
        uniform_logits,
        qasp_config=QASPConfig(step_size=0.5, num_adapt_steps=5, ns_iters=5),
    )
    assert fired is True

    logits_after = model(input_ids)
    assert not torch.allclose(logits_before, logits_after, atol=1e-4)

    for layer in model.layers:
        gram = layer.stiefel_query.data.transpose(0, 1) @ layer.stiefel_query.data
        identity = torch.eye(layer.stiefel_query.shape[1])
        assert torch.allclose(gram, identity, atol=1e-2, rtol=1e-2)


def test_overlay_preserves_shapes_under_prefill_and_step() -> None:
    """Enabling the overlay must not break the incremental prefill/step path."""

    model = _tiny(overlay_scale=0.15)
    model.eval()
    input_ids = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)

    logits, cache = model.prefill(input_ids)
    assert logits.shape == (1, input_ids.shape[1], model.config.vocab_size)

    next_logits = model.step(torch.tensor([[2]], dtype=torch.long), cache)
    assert next_logits.shape == (1, model.config.vocab_size)
    for keys, values in zip(cache.layer_keys, cache.layer_values):
        assert keys.shape[2] == input_ids.shape[1] + 1
        assert values.shape[2] == input_ids.shape[1] + 1
