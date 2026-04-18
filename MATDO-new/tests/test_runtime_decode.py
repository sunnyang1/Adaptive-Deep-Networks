from __future__ import annotations

from collections.abc import Sequence

from matdo_new.core.policy import PolicyDecision
from matdo_new.runtime.decode import decode_one_token
from matdo_new.runtime.materialize import MaterializedPolicy, materialize_policy
from matdo_new.runtime.prefill import prefill_prompt
from matdo_new.runtime.state import BackendResult, MATDOState


class FakeRuntimeBackend:
    def __init__(self) -> None:
        self.forward_inputs: list[tuple[int, ...]] = []
        self.forward_step_inputs: list[tuple[int, ...]] = []
        self.forward_step_state_lengths: list[int] = []
        self.forward_policies: list[MaterializedPolicy | None] = []
        self.forward_step_policies: list[MaterializedPolicy | None] = []
        self.forward_step_used_incremental: list[bool] = []
        self.forward_step_submitted_token_counts: list[int] = []

    def forward(
        self,
        token_ids: Sequence[int],
        *,
        policy: MaterializedPolicy | None = None,
    ) -> BackendResult:
        tokens = tuple(token_ids)
        self.forward_inputs.append(tokens)
        self.forward_policies.append(policy)
        return BackendResult(
            logits=("prefill", len(tokens)),
            cache={"seen_tokens": tokens},
        )

    def forward_step(
        self,
        token_ids: Sequence[int],
        *,
        state: MATDOState,
        policy: MaterializedPolicy | None = None,
    ) -> BackendResult:
        tokens = tuple(token_ids)
        self.forward_step_inputs.append(tokens)
        self.forward_step_state_lengths.append(state.sequence_length)
        self.forward_step_policies.append(policy)
        self.forward_step_used_incremental.append(True)
        self.forward_step_submitted_token_counts.append(len(tokens))
        return BackendResult(
            logits=("decode", len(tokens)),
            cache={"last_token": tokens[-1], "sequence_length": state.sequence_length + len(tokens)},
            submitted_token_count=len(tokens),
            used_incremental_cache=True,
        )


class MixedModeRuntimeBackend(FakeRuntimeBackend):
    def __init__(self) -> None:
        super().__init__()
        self._step_index = 0

    def forward_step(
        self,
        token_ids: Sequence[int],
        *,
        state: MATDOState,
        policy: MaterializedPolicy | None = None,
    ) -> BackendResult:
        tokens = tuple(token_ids)
        self.forward_step_inputs.append(tokens)
        self.forward_step_state_lengths.append(state.sequence_length)
        self.forward_step_policies.append(policy)

        self._step_index += 1
        used_incremental = self._step_index == 1
        submitted_token_count = len(tokens) if used_incremental else state.sequence_length + len(tokens)
        self.forward_step_used_incremental.append(used_incremental)
        self.forward_step_submitted_token_counts.append(submitted_token_count)

        return BackendResult(
            logits=("decode", len(tokens)),
            cache={
                "last_token": tokens[-1],
                "sequence_length": state.sequence_length + len(tokens),
                "mode": "incremental" if used_incremental else "replay",
            },
            submitted_token_count=submitted_token_count,
            used_incremental_cache=used_incremental,
        )


def test_prefill_uses_full_prompt_and_decode_uses_single_token_step() -> None:
    backend = FakeRuntimeBackend()
    decision = PolicyDecision(
        quantization_bits=2,
        m_blocks=8,
        t_steps=16,
        engram_entries=128,
        use_engram=True,
        is_arbitrage=True,
        estimated_error=0.02,
        target_error=0.05,
        reason="test",
    )

    state = prefill_prompt([11, 22, 33], backend=backend, policy=decision)
    next_state = decode_one_token(44, backend=backend, state=state)

    assert backend.forward_inputs == [(11, 22, 33)]
    assert backend.forward_step_inputs == [(44,)]
    assert backend.forward_step_state_lengths == [3]

    assert state.sequence_length == 3
    assert next_state.sequence_length == 4
    assert state.prompt_token_ids == (11, 22, 33)
    assert next_state.prompt_token_ids is state.prompt_token_ids
    assert state.decoded_tokens is None
    assert next_state.decoded_length == 1
    assert next_state.last_token_id == 44
    assert next_state.token_ids == (11, 22, 33, 44)

    assert state.metrics.prefill_calls == 1
    assert state.metrics.decode_calls == 0
    assert state.metrics.prefill_submitted_tokens == 3
    assert state.metrics.decode_submitted_tokens == 0
    assert state.metrics.incremental_decode_calls == 0
    assert state.metrics.decode_used_incremental is False
    assert state.metrics.submitted_tokens == 3

    assert next_state.metrics.prefill_calls == 1
    assert next_state.metrics.decode_calls == 1
    assert next_state.metrics.prompt_tokens == 3
    assert next_state.metrics.decode_tokens == 1
    assert next_state.metrics.prefill_submitted_tokens == 3
    assert next_state.metrics.decode_submitted_tokens == 1
    assert next_state.metrics.incremental_decode_calls == 1
    assert next_state.metrics.decode_used_incremental is True
    assert next_state.metrics.submitted_tokens == 4

    assert backend.forward_policies == [materialize_policy(decision)]
    assert backend.forward_step_policies == [materialize_policy(decision)]


def test_decode_history_stays_incremental_across_multiple_steps() -> None:
    backend = FakeRuntimeBackend()
    state = prefill_prompt([7, 8], backend=backend)

    state_1 = decode_one_token(9, backend=backend, state=state)
    state_2 = decode_one_token(10, backend=backend, state=state_1)

    assert state_2.prompt_token_ids is state.prompt_token_ids
    assert state_2.sequence_length == 4
    assert state_2.decoded_length == 2
    assert state_2.last_token_id == 10
    assert state_2.token_ids == (7, 8, 9, 10)
    assert backend.forward_step_inputs == [(9,), (10,)]
    assert backend.forward_step_state_lengths == [2, 3]
    assert state_2.metrics.prefill_submitted_tokens == 2
    assert state_2.metrics.decode_submitted_tokens == 2
    assert state_2.metrics.incremental_decode_calls == 2
    assert state_2.metrics.decode_used_incremental is True


def test_decode_metrics_reflect_latest_replay_fallback_honestly() -> None:
    backend = MixedModeRuntimeBackend()
    state = prefill_prompt([7, 8], backend=backend)

    state_1 = decode_one_token(9, backend=backend, state=state)
    state_2 = decode_one_token(10, backend=backend, state=state_1)

    assert backend.forward_step_used_incremental == [True, False]
    assert backend.forward_step_submitted_token_counts == [1, 4]
    assert state_1.metrics.incremental_decode_calls == 1
    assert state_1.metrics.decode_used_incremental is True
    assert state_2.metrics.incremental_decode_calls == 1
    assert state_2.metrics.decode_used_incremental is False
    assert state_2.metrics.decode_submitted_tokens == 5
    assert state_2.metrics.submitted_tokens == 7


def test_materialize_policy_preserves_existing_snapshot() -> None:
    materialized = MaterializedPolicy(
        quantization_bits=4,
        m_blocks=6,
        t_steps=12,
        engram_entries=0,
        use_engram=False,
        is_arbitrage=False,
        target_error=0.05,
        reason="already-materialized",
    )

    assert materialize_policy(materialized) is materialized
