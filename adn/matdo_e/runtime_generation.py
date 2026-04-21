from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from adn.matdo_e.policy import PolicyDecision
from adn.matdo_e.runtime_decode import decode_one_token
from adn.matdo_e.runtime_materialize import MaterializedPolicy
from adn.matdo_e.runtime_prefill import RuntimeBackend, prefill_prompt
from adn.matdo_e.runtime_state import MATDOState

LogitsSampler = Callable[[object | None], int]


@dataclass(frozen=True)
class GenerationResult:
    """Small user-facing snapshot returned by the runtime generation loop."""

    state: MATDOState
    generated_token_ids: tuple[int, ...]

    @property
    def prompt_token_ids(self) -> tuple[int, ...]:
        return self.state.prompt_token_ids

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.state.token_ids

    @property
    def sequence_length(self) -> int:
        return self.state.sequence_length


def generate_tokens(
    prompt_token_ids: Sequence[int],
    *,
    backend: RuntimeBackend,
    sampler: LogitsSampler,
    max_new_tokens: int,
    policy: PolicyDecision | MaterializedPolicy | None = None,
) -> GenerationResult:
    """Generate a short continuation using prefill followed by incremental decode."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    state = prefill_prompt(prompt_token_ids, backend=backend, policy=policy)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        if state.last_logits is None:
            raise ValueError("backend must return logits to generate new tokens")
        next_token_id = int(sampler(state.last_logits))
        generated.append(next_token_id)
        state = decode_one_token(next_token_id, backend=backend, state=state)

    return GenerationResult(state=state, generated_token_ids=tuple(generated))
