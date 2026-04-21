"""Smoke tests for lightweight QASP generation and incremental APIs."""

from __future__ import annotations

import torch

from QASP.inference import IncrementalInference, QASPGenerator
from QASP.models import create_qasp_transformer


def _build_tiny_model(use_engram: bool = True):
    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        use_engram=use_engram,
    )
    model.eval()
    return model


def test_generate_increases_output_length_by_max_new_tokens():
    model = _build_tiny_model()
    generator = QASPGenerator(model)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    max_new_tokens = 5

    output_ids = generator.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)

    assert output_ids.shape[1] == input_ids.shape[1] + max_new_tokens


def test_incremental_step_returns_batch_single_token_shape_after_prefill():
    model = _build_tiny_model()
    inference = IncrementalInference(model)
    input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)

    state = inference.prefill(input_ids)
    next_token = inference.step(state)

    assert next_token.shape == (input_ids.shape[0], 1)


def test_prefill_logits_match_full_forward():
    """prefill() must emit the same [B, T, V] logits as a plain forward pass."""

    torch.manual_seed(0)
    model = _build_tiny_model(use_engram=False)
    model.eval()

    input_ids = torch.tensor([[3, 14, 1, 5, 9]], dtype=torch.long)
    prefill_logits, cache = model.prefill(input_ids)
    forward_logits = model(input_ids)

    assert torch.allclose(prefill_logits, forward_logits, atol=1e-5, rtol=1e-5)
    assert cache.num_layers == len(model.layers)
    for keys, values in zip(cache.layer_keys, cache.layer_values):
        assert keys is not None and values is not None
        assert keys.shape[2] == input_ids.shape[1]
        assert values.shape[2] == input_ids.shape[1]


def test_step_extends_kv_cache_by_one_position_per_call():
    """Each step() grows every per-layer K/V tensor by exactly one token."""

    torch.manual_seed(0)
    model = _build_tiny_model()
    model.eval()

    input_ids = torch.tensor([[3, 14, 1]], dtype=torch.long)
    _, cache = model.prefill(input_ids)
    initial_lengths = [k.shape[2] for k in cache.layer_keys]

    next_logits = model.step(torch.tensor([[2]], dtype=torch.long), cache)

    assert next_logits.shape == (1, model.config.vocab_size)
    for idx, initial in enumerate(initial_lengths):
        assert cache.layer_keys[idx].shape[2] == initial + 1
        assert cache.layer_values[idx].shape[2] == initial + 1
    assert cache.seq_len == input_ids.shape[1] + 1


def test_incremental_generation_matches_full_autoregressive_loop():
    """Multi-step decoding via step() must match naive re-forward decoding."""

    torch.manual_seed(1)
    model = _build_tiny_model()
    model.eval()
    input_ids = torch.tensor([[11, 4, 7]], dtype=torch.long)
    num_new = 4

    inference = IncrementalInference(model)
    state = inference.prefill(input_ids)
    generated: list[torch.Tensor] = []
    for _ in range(num_new):
        generated.append(inference.step(state))
    incremental_output = torch.cat([input_ids] + generated, dim=1)

    naive_output = input_ids.clone()
    for _ in range(num_new):
        logits = model(naive_output)
        naive_output = torch.cat(
            [naive_output, torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)],
            dim=1,
        )

    assert torch.equal(incremental_output, naive_output)
