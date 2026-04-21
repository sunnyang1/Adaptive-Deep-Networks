"""Integration tests for QASP KV-cache RaBitQ quantization (paper §5.5)."""

from __future__ import annotations

import torch

from QASP.inference import IncrementalInference
from QASP.models import create_qasp_transformer


def _build_model(quantize: bool):
    torch.manual_seed(0)
    return create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        quantize_kv=quantize,
        kv_codec_seed=7,
    )


def test_quantized_model_instantiates_codec_with_head_dim() -> None:
    model = _build_model(quantize=True)
    assert model.kv_codec is not None
    assert model.kv_codec.dim == model.config.hidden_size // model.config.num_heads


def test_disabled_model_has_no_codec() -> None:
    model = _build_model(quantize=False)
    assert model.kv_codec is None


def test_quantized_prefill_matches_manual_codec_application() -> None:
    """Each cached K/V equals the codec-quantized version of its full-precision counterpart."""

    model_q = _build_model(quantize=True)
    model_q.eval()

    input_ids = torch.tensor([[3, 14, 1, 5, 9]], dtype=torch.long)
    _, cache = model_q.prefill(input_ids)

    for keys, values in zip(cache.layer_keys, cache.layer_values):
        assert keys is not None and values is not None
        round_trip_k = model_q.kv_codec.quantize(keys)
        round_trip_v = model_q.kv_codec.quantize(values)
        assert torch.allclose(round_trip_k, keys, atol=1e-5)
        assert torch.allclose(round_trip_v, values, atol=1e-5)


def test_quantized_step_grows_cache_and_stays_idempotent_under_requantization() -> None:
    """Every step extends cache by 1 token, and re-quantizing leaves tensors fixed."""

    model_q = _build_model(quantize=True)
    model_q.eval()

    input_ids = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)
    _, cache = model_q.prefill(input_ids)
    lengths_before = [k.shape[2] for k in cache.layer_keys]

    logits = model_q.step(torch.tensor([[2]], dtype=torch.long), cache)
    assert logits.shape == (1, model_q.config.vocab_size)
    for l_before, keys, values in zip(lengths_before, cache.layer_keys, cache.layer_values):
        assert keys.shape[2] == l_before + 1
        assert values.shape[2] == l_before + 1
        assert torch.allclose(model_q.kv_codec.quantize(keys), keys, atol=1e-5)
        assert torch.allclose(model_q.kv_codec.quantize(values), values, atol=1e-5)


def test_quantized_incremental_decoding_is_coherent() -> None:
    """Quantized decode should produce a well-formed token stream of the expected length."""

    model_q = _build_model(quantize=True)
    model_q.eval()

    input_ids = torch.tensor([[11, 4, 7]], dtype=torch.long)
    num_new = 3

    inference = IncrementalInference(model_q)
    state = inference.prefill(input_ids)
    generated = [inference.step(state) for _ in range(num_new)]

    output = torch.cat([input_ids] + generated, dim=1)
    assert output.shape == (1, input_ids.shape[1] + num_new)
    assert output.dtype == torch.long
    assert torch.all(output >= 0)
    assert torch.all(output < model_q.config.vocab_size)
