from __future__ import annotations

import torch
import pytest

from matdo_new.core.policy import PolicyDecision
from matdo_new.modeling.config import MATDOModelConfig
from matdo_new.runtime import decode_one_token, materialize_policy, prefill_prompt
from matdo_new.runtime.backend import AdaptiveTransformerBackendCache, AdaptiveTransformerRuntimeBackend
from src.models.incremental_state import IncrementalState
from src.models.configs import ModelConfig


def _tiny_backend_config() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        hidden_dim=16,
        num_heads=4,
        num_blocks=1,
        mlp_ratio=2,
        vocab_size=32,
        max_seq_len=32,
    )


def _policy_decision() -> PolicyDecision:
    return PolicyDecision(
        quantization_bits=2,
        m_blocks=3,
        t_steps=4,
        engram_entries=32,
        use_engram=False,
        is_arbitrage=False,
        estimated_error=0.01,
        target_error=0.05,
        reason="runtime-backend-test",
    )


def test_runtime_backend_prefill_and_decode_store_native_incremental_cache() -> None:
    torch.manual_seed(0)

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        _tiny_backend_config(),
        use_attnres=False,
    )
    policy = _policy_decision()

    state = prefill_prompt([1, 2, 3], backend=backend, policy=policy)
    next_state = decode_one_token(4, backend=backend, state=state)

    assert isinstance(state.cache, AdaptiveTransformerBackendCache)
    assert isinstance(next_state.cache, AdaptiveTransformerBackendCache)
    assert isinstance(state.cache.incremental_state, IncrementalState)
    assert isinstance(next_state.cache.incremental_state, IncrementalState)
    assert state.cache.sequence_position == 3
    assert next_state.cache.sequence_position == 4
    assert state.cache.policy_snapshot == materialize_policy(policy)
    assert next_state.cache.policy_snapshot == materialize_policy(policy)
    assert len(state.cache.incremental_state.kv_caches) == _tiny_backend_config().num_layers
    assert len(next_state.cache.incremental_state.kv_caches) == _tiny_backend_config().num_layers
    assert len(state.cache.incremental_state.kv_caches[0]) == 3
    assert len(next_state.cache.incremental_state.kv_caches[0]) == 4
    assert state.policy == materialize_policy(policy)
    assert next_state.policy == materialize_policy(policy)
    assert isinstance(state.last_logits, torch.Tensor)
    assert isinstance(next_state.last_logits, torch.Tensor)
    assert tuple(state.last_logits.shape) == (_tiny_backend_config().vocab_size,)
    assert tuple(next_state.last_logits.shape) == (_tiny_backend_config().vocab_size,)
    assert next_state.token_ids == (1, 2, 3, 4)
    assert next_state.metrics.incremental_decode_calls == 1
    assert next_state.metrics.decode_used_incremental is True


def test_runtime_backend_decode_step_avoids_full_sequence_forward_replay(monkeypatch) -> None:
    torch.manual_seed(0)

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        _tiny_backend_config(),
        use_attnres=False,
    )

    forward_inputs: list[tuple[int, ...]] = []
    original_forward = backend.model.forward

    def recording_forward(input_ids: torch.Tensor, *args, **kwargs):
        forward_inputs.append(tuple(int(token) for token in input_ids[0].tolist()))
        return original_forward(input_ids, *args, **kwargs)

    monkeypatch.setattr(backend.model, "forward", recording_forward)

    state = prefill_prompt([1, 2, 3], backend=backend)
    next_state = decode_one_token(4, backend=backend, state=state)

    assert forward_inputs == [(1, 2, 3)]
    assert isinstance(next_state.cache, AdaptiveTransformerBackendCache)
    assert next_state.cache.sequence_position == 4
    assert len(next_state.cache.incremental_state.kv_caches[0]) == 4


def test_runtime_backend_can_be_constructed_from_matdo_model_config(monkeypatch) -> None:
    tiny_config = _tiny_backend_config()

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: tiny_config,
    )

    backend = AdaptiveTransformerRuntimeBackend.from_model_config(
        MATDOModelConfig(model_size="t4", use_attnres=False),
    )

    state = prefill_prompt([5, 6], backend=backend)

    assert isinstance(state.cache, AdaptiveTransformerBackendCache)
    assert isinstance(state.cache.incremental_state, IncrementalState)
    assert state.cache.sequence_position == 2
    assert state.cache.policy_snapshot is None
    assert state.cache.runtime_handles is not None
    assert state.cache.active_scope_state == state.cache.runtime_handles.scope_memory.blocks()
    assert isinstance(state.last_logits, torch.Tensor)


def test_runtime_backend_from_model_config_rejects_unsupported_engram(monkeypatch) -> None:
    tiny_config = _tiny_backend_config()

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: tiny_config,
    )

    with pytest.raises(ValueError, match="Engram"):
        AdaptiveTransformerRuntimeBackend.from_model_config(
            MATDOModelConfig(model_size="t4", use_engram=True),
        )


def test_runtime_backend_reuses_runtime_handles_across_decode_steps(monkeypatch) -> None:
    tiny_config = _tiny_backend_config()

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: tiny_config,
    )

    backend = AdaptiveTransformerRuntimeBackend.from_model_config(
        MATDOModelConfig(model_size="t4", use_attnres=False),
    )

    state = prefill_prompt([5, 6], backend=backend)
    state_1 = decode_one_token(7, backend=backend, state=state)
    state_2 = decode_one_token(8, backend=backend, state=state_1)

    assert isinstance(state.cache, AdaptiveTransformerBackendCache)
    assert isinstance(state_1.cache, AdaptiveTransformerBackendCache)
    assert isinstance(state_2.cache, AdaptiveTransformerBackendCache)
    assert state.cache.runtime_handles is not None
    assert state_1.cache.runtime_handles is state.cache.runtime_handles
    assert state_2.cache.runtime_handles is state.cache.runtime_handles
    assert state_2.cache.sequence_position == 4
    assert len(state_2.cache.incremental_state.kv_caches[0]) == 4


def test_runtime_backend_attnres_decode_matches_full_forward_next_token_logits() -> None:
    torch.manual_seed(0)

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        _tiny_backend_config(),
        use_attnres=True,
    )

    state = prefill_prompt([1, 2, 3], backend=backend)
    next_state = decode_one_token(4, backend=backend, state=state)

    full_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=backend.device)
    with torch.no_grad():
        full_logits = backend.model.forward(
            full_input,
            use_attnres=True,
            use_engram=False,
            use_qttt=False,
        )[0, -1, :]

    assert isinstance(next_state.last_logits, torch.Tensor)
    assert torch.allclose(next_state.last_logits, full_logits, atol=1e-5, rtol=1e-5)


def test_runtime_backend_policy_change_rebuilds_runtime_handles(monkeypatch) -> None:
    tiny_config = _tiny_backend_config()

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: tiny_config,
    )

    backend = AdaptiveTransformerRuntimeBackend.from_model_config(
        MATDOModelConfig(model_size="t4", use_attnres=False),
    )
    policy_a = _policy_decision()
    policy_b = PolicyDecision(
        quantization_bits=4,
        m_blocks=2,
        t_steps=1,
        engram_entries=0,
        use_engram=False,
        is_arbitrage=False,
        estimated_error=0.02,
        target_error=0.05,
        reason="policy-b",
    )

    state = prefill_prompt([5, 6], backend=backend, policy=policy_a)
    next_state = decode_one_token(7, backend=backend, state=state)
    changed_state = decode_one_token(
        8,
        backend=backend,
        state=next_state,
    )

    changed_result = backend.forward_step(
        (9,),
        state=changed_state,
        policy=materialize_policy(policy_b),
    )

    assert isinstance(changed_state.cache, AdaptiveTransformerBackendCache)
    assert isinstance(changed_result.cache, AdaptiveTransformerBackendCache)
    assert next_state.cache.runtime_handles is state.cache.runtime_handles
    assert changed_result.cache.runtime_handles is not changed_state.cache.runtime_handles
    assert changed_result.cache.policy_snapshot == materialize_policy(policy_b)


def test_runtime_backend_replays_when_policy_or_cache_capability_changes() -> None:
    torch.manual_seed(0)

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        _tiny_backend_config(),
        use_attnres=True,
    )

    state = prefill_prompt([1, 2, 3], backend=backend)
    next_state = decode_one_token(4, backend=backend, state=state)

    assert next_state.metrics.incremental_decode_calls == 0
    assert next_state.metrics.decode_used_incremental is False
    assert next_state.metrics.decode_submitted_tokens == 4
    assert next_state.metrics.submitted_tokens == 7


def test_runtime_backend_prefill_runs_when_use_engram_is_enabled() -> None:
    """Engram-enabled prefill is allowed; replay still covers decode correctness."""
    torch.manual_seed(0)
    config = _tiny_backend_config()
    config.use_engram = True

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        config,
        use_attnres=False,
        use_engram=True,
    )

    state = prefill_prompt([1, 2, 3], backend=backend)

    assert isinstance(state.cache, AdaptiveTransformerBackendCache)
    assert tuple(state.last_logits.shape) == (config.vocab_size,)
    # Engram-on prefill falls back to replay on every decode step, so the
    # backend must still refuse to claim incremental decode capability.
    assert state.cache.supports_incremental_decode is False


def test_runtime_backend_engram_policy_forces_replay_on_decode() -> None:
    """When engram is on, decode always replays regardless of cache capability."""
    torch.manual_seed(0)
    config = _tiny_backend_config()
    config.use_engram = True

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        config,
        use_attnres=False,
        use_engram=True,
    )

    state = prefill_prompt([1, 2, 3], backend=backend)
    next_state = decode_one_token(4, backend=backend, state=state)

    assert next_state.metrics.decode_used_incremental is False


def _attnres_backend_config() -> ModelConfig:
    """A tiny AttnRes config with num_blocks=4 so block-cap is observable."""
    return ModelConfig(
        num_layers=4,
        hidden_dim=16,
        num_heads=4,
        num_blocks=4,
        mlp_ratio=2,
        vocab_size=32,
        max_seq_len=32,
    )


def _policy_with_m_blocks(m_blocks: int, *, t_steps: int = 0) -> PolicyDecision:
    return PolicyDecision(
        quantization_bits=2,
        m_blocks=m_blocks,
        t_steps=t_steps,
        engram_entries=0,
        use_engram=False,
        is_arbitrage=False,
        estimated_error=0.01,
        target_error=0.05,
        reason=f"m-blocks-cap-{m_blocks}",
    )


def test_runtime_backend_decode_without_m_blocks_policy_falls_back_to_replay() -> None:
    """With AttnRes on but no m_blocks policy, decode must keep the replay path."""
    torch.manual_seed(0)

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        _attnres_backend_config(),
        use_attnres=True,
    )

    state = prefill_prompt([1, 2, 3], backend=backend)
    next_state = decode_one_token(4, backend=backend, state=state)

    assert isinstance(next_state.cache, AdaptiveTransformerBackendCache)
    assert next_state.cache.active_scope_block_count == 0
    assert next_state.metrics.decode_used_incremental is False


def test_runtime_backend_decode_reports_attnres_active_block_count_under_generous_cap() -> None:
    """A generous m_blocks cap engages incremental decode and reports natural block count."""
    torch.manual_seed(0)

    config = _attnres_backend_config()
    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        config,
        use_attnres=True,
    )
    generous = _policy_with_m_blocks(config.num_blocks + 10)

    state = prefill_prompt([1, 2, 3], backend=backend, policy=generous)
    next_state = decode_one_token(4, backend=backend, state=state)

    # Natural per-step block stack is [hidden] + (num_blocks - 1) boundary
    # appends + final partial_block = num_blocks + 1 entries.
    assert next_state.cache.active_scope_block_count == config.num_blocks + 1
    assert next_state.metrics.decode_used_incremental is True


def test_runtime_backend_decode_caps_attnres_blocks_to_policy_m_blocks() -> None:
    """When policy.m_blocks < natural count, decode must drop oldest blocks."""
    torch.manual_seed(0)

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        _attnres_backend_config(),
        use_attnres=True,
    )
    policy_cap_two = _policy_with_m_blocks(2)

    state = prefill_prompt([1, 2, 3], backend=backend, policy=policy_cap_two)
    capped_state = decode_one_token(4, backend=backend, state=state)

    assert isinstance(capped_state.cache, AdaptiveTransformerBackendCache)
    assert capped_state.cache.active_scope_block_count == 2
    assert capped_state.cache.policy_snapshot is not None
    assert capped_state.cache.policy_snapshot.m_blocks == 2
    assert capped_state.metrics.decode_used_incremental is True


def test_runtime_backend_decode_ignores_m_blocks_when_use_attnres_is_off() -> None:
    """With use_attnres=False there is no block stack to cap."""
    torch.manual_seed(0)

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        _attnres_backend_config(),
        use_attnres=False,
    )
    policy = _policy_with_m_blocks(2)

    state = prefill_prompt([1, 2, 3], backend=backend, policy=policy)
    next_state = decode_one_token(4, backend=backend, state=state)

    assert next_state.cache.active_scope_block_count == 0


def _qttt_policy(t_steps: int) -> PolicyDecision:
    return PolicyDecision(
        quantization_bits=4,
        m_blocks=0,
        t_steps=t_steps,
        engram_entries=0,
        use_engram=False,
        is_arbitrage=False,
        estimated_error=0.02,
        target_error=0.05,
        reason=f"qttt-t-{t_steps}",
    )


def test_runtime_backend_prefill_invokes_qttt_when_policy_has_t_steps(monkeypatch) -> None:
    """With policy.t_steps > 0, prefill should run the qTTT adapter and record losses."""
    torch.manual_seed(0)
    tiny_config = _tiny_backend_config()

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: tiny_config,
    )

    backend = AdaptiveTransformerRuntimeBackend.from_model_config(
        MATDOModelConfig(model_size="t4", use_attnres=False, use_qttt=True),
    )

    state = prefill_prompt([1, 2, 3], backend=backend, policy=_qttt_policy(3))

    assert isinstance(state.cache, AdaptiveTransformerBackendCache)
    assert len(state.cache.qttt_loss_history) == 3
    assert all(isinstance(loss, float) for loss in state.cache.qttt_loss_history)


def test_runtime_backend_prefill_with_qttt_changes_prefill_logits(monkeypatch) -> None:
    """qTTT at prefill must actually affect the last-token logits."""
    torch.manual_seed(0)
    tiny_config = _tiny_backend_config()

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: tiny_config,
    )

    backend = AdaptiveTransformerRuntimeBackend.from_model_config(
        MATDOModelConfig(model_size="t4", use_attnres=False, use_qttt=True),
    )

    baseline = prefill_prompt([1, 2, 3], backend=backend, policy=_qttt_policy(0))
    adapted = prefill_prompt([1, 2, 3], backend=backend, policy=_qttt_policy(4))

    assert baseline.cache.qttt_loss_history == ()
    assert len(adapted.cache.qttt_loss_history) == 4
    assert not torch.allclose(baseline.last_logits, adapted.last_logits, atol=1e-6)


def test_runtime_backend_prefill_skips_qttt_without_adapter(monkeypatch) -> None:
    """Pure backend-config path (no MATDOModel) cannot run qTTT; must stay inert."""
    torch.manual_seed(0)

    backend = AdaptiveTransformerRuntimeBackend.from_backend_config(
        _tiny_backend_config(),
        use_attnres=False,
    )

    state = prefill_prompt([1, 2, 3], backend=backend, policy=_qttt_policy(3))

    assert state.cache.qttt_loss_history == ()


def _quantization_policy(quantization_bits: int) -> PolicyDecision:
    return PolicyDecision(
        quantization_bits=quantization_bits,
        m_blocks=0,
        t_steps=0,
        engram_entries=0,
        use_engram=False,
        is_arbitrage=False,
        estimated_error=0.02,
        target_error=0.05,
        reason=f"quant-{quantization_bits}",
    )


def test_runtime_backend_prefill_round_trips_kv_under_quantization_policy(
    monkeypatch,
) -> None:
    """Storing KV must respect policy.quantization_bits via the RaBitQ adapter."""
    torch.manual_seed(0)
    tiny_config = _tiny_backend_config()
    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: tiny_config,
    )

    backend = AdaptiveTransformerRuntimeBackend.from_model_config(
        MATDOModelConfig(model_size="t4", use_attnres=False),
    )

    baseline = prefill_prompt([1, 2, 3], backend=backend)
    quantized = prefill_prompt(
        [1, 2, 3], backend=backend, policy=_quantization_policy(1)
    )

    baseline_keys = baseline.cache.incremental_state.kv_caches[0].keys
    quantized_keys = quantized.cache.incremental_state.kv_caches[0].keys
    assert baseline_keys.shape == quantized_keys.shape
    assert not torch.allclose(baseline_keys, quantized_keys, atol=1e-4)


def test_runtime_backend_cache_stays_consistent_with_current_quantization_policy(
    monkeypatch,
) -> None:
    """Mid-run bit-width changes must leave the cache aligned with the current policy."""
    torch.manual_seed(0)
    tiny_config = _tiny_backend_config()
    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: tiny_config,
    )

    backend = AdaptiveTransformerRuntimeBackend.from_model_config(
        MATDOModelConfig(model_size="t4", use_attnres=False),
    )

    prefill_high = prefill_prompt([1, 2, 3], backend=backend, policy=_quantization_policy(4))
    mid_run = backend.forward_step(
        (4,), state=prefill_high, policy=_quantization_policy(1)
    )

    # Ground truth: a clean run under bits=1 from scratch.
    reference_prefill = prefill_prompt(
        [1, 2, 3], backend=backend, policy=_quantization_policy(1)
    )
    reference = backend.forward_step(
        (4,), state=reference_prefill, policy=_quantization_policy(1)
    )

    mid_keys = mid_run.cache.incremental_state.kv_caches[0].keys
    ref_keys = reference.cache.incremental_state.kv_caches[0].keys
    # The adapter is quantization-deterministic, so mid-run and reference must
    # agree on every token once the policy has settled on bits=1.
    assert mid_keys.shape == ref_keys.shape
    assert torch.allclose(mid_keys, ref_keys, atol=1e-6)

    # The stored cache must also differ from what bits=4 would have produced,
    # proving the old high-fidelity values were rebuilt under the new policy.
    baseline_high = prefill_prompt(
        [1, 2, 3], backend=backend, policy=_quantization_policy(4)
    )
    bits4_decode = backend.forward_step(
        (4,), state=baseline_high, policy=_quantization_policy(4)
    )
    bits4_keys = bits4_decode.cache.incremental_state.kv_caches[0].keys
    assert not torch.allclose(mid_keys, bits4_keys, atol=1e-6)

    # And the replay path must be the one that rebuilt the cache, not the
    # incremental path that only touches the newest token.
    assert mid_run.used_incremental_cache is False


def test_runtime_backend_decode_quantizes_new_kv_tokens_when_policy_active(
    monkeypatch,
) -> None:
    """Incremental decode must run new K/V through the quantization adapter."""
    torch.manual_seed(0)
    tiny_config = _tiny_backend_config()
    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: tiny_config,
    )

    backend = AdaptiveTransformerRuntimeBackend.from_model_config(
        MATDOModelConfig(model_size="t4", use_attnres=False),
    )
    policy = _quantization_policy(1)

    state = prefill_prompt([1, 2, 3], backend=backend, policy=policy)
    next_state = decode_one_token(4, backend=backend, state=state)

    # The new token's KV slice should differ from a pristine re-projection
    # because it has been through compress+decompress.
    input_ids = torch.tensor([[4]], dtype=torch.long, device=backend.device)
    with torch.no_grad():
        hidden = backend.model.token_embedding(input_ids)
        normed = backend.model.layers[0].attn_norm(hidden)
        raw_k, _ = backend._project_layer_kv(backend.model.layers[0], normed)

    stored_k = next_state.cache.incremental_state.kv_caches[0].keys
    new_token_slice = stored_k[..., -1:, :]
    assert not torch.allclose(new_token_slice, raw_k, atol=1e-4)
