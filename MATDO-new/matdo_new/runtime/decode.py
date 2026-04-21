from __future__ import annotations

from matdo_new.runtime.prefill import RuntimeBackend
from matdo_new.runtime.state import MATDOState


def decode_one_token(
    next_token_id: int,
    *,
    backend: RuntimeBackend,
    state: MATDOState,
) -> MATDOState:
    """Advance generation by one token using true incremental decode."""
    decode_input = (int(next_token_id),)
    result = backend.forward_step(
        decode_input,
        state=state,
        policy=state.policy,
    )
    return state.append_token(
        int(next_token_id),
        cache=result.cache,
        logits=result.logits,
        submitted_token_count=result.submitted_token_count or len(decode_input),
        used_incremental_cache=result.used_incremental_cache,
    )
