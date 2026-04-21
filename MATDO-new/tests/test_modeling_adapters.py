from __future__ import annotations

import torch

from matdo_new.core.policy import PolicyDecision
from matdo_new.modeling.attention import apply_frozen_kv_attention, sample_next_token
from matdo_new.modeling.config import (
    ExternalMemoryConfig,
    KVQuantizationConfig,
    MATDOModelConfig,
    QueryAdaptationConfig,
)
from matdo_new.modeling.external_memory import ExternalMemoryHandle
from matdo_new.modeling.kv_quantization import KVQuantizationAdapter
from matdo_new.modeling.matdo_model import MATDOModel
from matdo_new.modeling.query_adaptation import QueryAdaptationAdapter
from matdo_new.modeling.scope_memory import ScopeMemory


def test_kv_quantization_adapter_round_trips_shapes_and_reports_stats() -> None:
    torch.manual_seed(0)
    adapter = KVQuantizationAdapter(
        KVQuantizationConfig(total_bits=2, head_dim=16, use_rotation=False)
    )
    keys = torch.randn(1, 2, 3, 16)
    values = torch.randn(1, 2, 3, 16)

    package = adapter.compress(keys, values)
    restored_keys, restored_values = adapter.decompress(package)
    stats = adapter.memory_stats(seq_len=3, num_layers=2, batch_size=1, num_heads=2)

    assert package.original_key_shape == tuple(keys.shape)
    assert restored_keys.shape == keys.shape
    assert restored_values.shape == values.shape
    assert stats['compressed_mb'] > 0.0
    assert stats['compression_ratio'] > 0.0


def test_query_adaptation_adapter_wraps_qttt_pseudo_query_path() -> None:
    torch.manual_seed(0)
    adapter = QueryAdaptationAdapter(
        QueryAdaptationConfig(num_steps=3, learning_rate=0.05),
        hidden_dim=8,
        num_heads=2,
    )
    kv_cache = adapter.make_kv_cache(
        keys=torch.randn(1, 2, 4, 4),
        values=torch.randn(1, 2, 4, 4),
    )
    pseudo_query = torch.randn(8)

    result = adapter.adapt_pseudo_query(
        pseudo_query,
        kv_cache,
        seq_positions=torch.tensor([0]),
    )
    attention_out = apply_frozen_kv_attention(
        result.adapted_query.view(1, 2, 1, 4),
        kv_cache,
    )

    assert result.adapted_query.shape == pseudo_query.shape
    assert len(result.loss_history) == 3
    assert attention_out.shape == (1, 2, 1, 4)


def test_external_and_scope_memory_handles_enforce_capacity() -> None:
    external = ExternalMemoryHandle(ExternalMemoryConfig(enabled=True, max_entries=2))
    external.put('a', 1)
    external.put('b', 2)
    external.put('c', 3)

    scope = ScopeMemory(capacity=2)
    scope.remember('block-0', block_index=0)
    scope.remember('block-1', block_index=1)
    scope.remember('block-2', block_index=2)

    assert [record.key for record in external.snapshot()] == ['b', 'c']
    assert [block.index for block in scope.blocks()] == [1, 2]
    assert scope.latest() is not None
    assert scope.latest().value == 'block-2'


def test_scope_memory_default_indices_stay_monotonic_after_eviction() -> None:
    scope = ScopeMemory(capacity=2)
    scope.remember('block-0')
    scope.remember('block-1')
    scope.remember('block-2')
    scope.remember('block-3')

    assert [block.index for block in scope.blocks()] == [2, 3]
    assert [block.value for block in scope.blocks()] == ['block-2', 'block-3']


def test_matdo_model_prepares_runtime_handles_from_policy() -> None:
    model = MATDOModel(
        MATDOModelConfig(
            model_size='t4',
            use_qttt=False,
            quantization=KVQuantizationConfig(use_rotation=False),
        )
    )
    policy = PolicyDecision(
        quantization_bits=4,
        m_blocks=3,
        t_steps=5,
        engram_entries=7,
        use_engram=True,
        is_arbitrage=True,
        estimated_error=0.02,
        target_error=0.05,
        reason='test',
    )

    handles = model.prepare_runtime_handles(policy, device='cpu')

    assert handles.policy is not None
    assert handles.policy.quantization_bits == 4
    assert handles.kv_quantization.config.total_bits == 4
    assert handles.kv_quantization.config.head_dim == model.backend_config.head_dim
    assert handles.query_adaptation is not None
    assert handles.query_adaptation.config.num_steps == 5
    assert handles.external_memory.enabled is True
    assert handles.external_memory.config.max_entries == 7
    assert handles.scope_memory.capacity == 3


def test_matdo_model_policy_can_disable_qttt_even_when_config_enabled() -> None:
    model = MATDOModel(
        MATDOModelConfig(
            model_size='t4',
            use_qttt=True,
            quantization=KVQuantizationConfig(use_rotation=False),
        )
    )
    policy = PolicyDecision(
        quantization_bits=2,
        m_blocks=2,
        t_steps=0,
        engram_entries=0,
        use_engram=False,
        is_arbitrage=False,
        estimated_error=0.04,
        target_error=0.05,
        reason='disable-qttt',
    )

    handles = model.prepare_runtime_handles(policy, device='cpu')

    assert handles.policy is not None
    assert handles.policy.t_steps == 0
    assert handles.query_adaptation is None


def test_matdo_model_sampler_respects_top_k() -> None:
    generator = torch.Generator().manual_seed(123)
    model = MATDOModel(MATDOModelConfig(sampling_top_k=2))
    logits = torch.tensor([-1000.0, -1000.0, 5.0, 4.0])

    sampled = model.sample_next_token(logits, generator=generator)
    direct = sample_next_token(logits, top_k=1, generator=torch.Generator().manual_seed(0))
    unrestricted = model.sample_next_token(logits, top_k=4, generator=torch.Generator().manual_seed(0))

    assert sampled in {2, 3}
    assert int(direct.item()) == 2
    assert unrestricted == 2


def test_sample_next_token_enforces_exact_top_k_under_ties() -> None:
    logits = torch.tensor([5.0, 5.0, 5.0, 1.0])
    allowed = set(torch.topk(logits, k=2).indices.tolist())

    observed = {
        int(sample_next_token(logits, top_k=2, generator=torch.Generator().manual_seed(seed)).item())
        for seed in range(64)
    }

    assert observed <= allowed
    assert len(observed) == 2
