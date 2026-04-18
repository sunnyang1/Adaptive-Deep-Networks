from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matdo_new.runtime.decode import decode_one_token
from matdo_new.runtime.generation import GenerationResult, generate_tokens
from matdo_new.runtime.materialize import MaterializedPolicy, materialize_policy
from matdo_new.runtime.metrics import RuntimeMetrics
from matdo_new.runtime.prefill import RuntimeBackend, prefill_prompt
from matdo_new.runtime.state import BackendResult, MATDOState

if TYPE_CHECKING:
    from matdo_new.runtime.backend import (
        AdaptiveTransformerBackendCache,
        AdaptiveTransformerRuntimeBackend,
    )

__all__ = [
    "AdaptiveTransformerBackendCache",
    "AdaptiveTransformerRuntimeBackend",
    "BackendResult",
    "GenerationResult",
    "MATDOState",
    "MaterializedPolicy",
    "RuntimeBackend",
    "RuntimeMetrics",
    "decode_one_token",
    "generate_tokens",
    "materialize_policy",
    "prefill_prompt",
]


def __getattr__(name: str) -> Any:
    if name in {"AdaptiveTransformerBackendCache", "AdaptiveTransformerRuntimeBackend"}:
        from matdo_new.runtime.backend import (
            AdaptiveTransformerBackendCache,
            AdaptiveTransformerRuntimeBackend,
        )

        exported = {
            "AdaptiveTransformerBackendCache": AdaptiveTransformerBackendCache,
            "AdaptiveTransformerRuntimeBackend": AdaptiveTransformerRuntimeBackend,
        }
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

