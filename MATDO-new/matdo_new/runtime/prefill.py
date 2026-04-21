from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from matdo_new.core.policy import PolicyDecision
from matdo_new.runtime.materialize import MaterializedPolicy, materialize_policy
from matdo_new.runtime.metrics import RuntimeMetrics
from matdo_new.runtime.state import BackendResult, MATDOState


class RuntimeBackend(Protocol):
    def forward(
        self,
        token_ids: Sequence[int],
        *,
        policy: MaterializedPolicy | None = None,
    ) -> BackendResult: ...

    def forward_step(
        self,
        token_ids: Sequence[int],
        *,
        state: MATDOState,
        policy: MaterializedPolicy | None = None,
    ) -> BackendResult: ...


def prefill_prompt(
    prompt_token_ids: Sequence[int],
    *,
    backend: RuntimeBackend,
    policy: PolicyDecision | MaterializedPolicy | None = None,
) -> MATDOState:
    """Run a full-prompt prefill pass and capture the initial runtime state."""
    tokens = tuple(prompt_token_ids)
    if not tokens:
        raise ValueError("prompt_token_ids must not be empty")

    materialized = materialize_policy(policy)
    result = backend.forward(tokens, policy=materialized)
    return MATDOState(
        prompt_token_ids=tokens,
        cache=result.cache,
        last_logits=result.logits,
        policy=materialized,
        metrics=RuntimeMetrics.from_prefill(len(tokens)),
    )
