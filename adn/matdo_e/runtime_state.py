from __future__ import annotations

from dataclasses import dataclass, field

from adn.matdo_e.runtime_materialize import MaterializedPolicy
from adn.matdo_e.runtime_metrics import RuntimeMetrics


@dataclass(frozen=True)
class BackendResult:
    """Backend output needed to continue incremental generation."""

    logits: object | None = None
    cache: object | None = None
    submitted_token_count: int | None = None
    used_incremental_cache: bool = False


@dataclass(frozen=True)
class DecodedTokenNode:
    """Persistent linked tail for decode tokens appended after prefill."""

    token_id: int
    previous: "DecodedTokenNode | None" = None


@dataclass(frozen=True)
class MATDOState:
    """Immutable runtime state carried across prompt prefill and decode."""

    prompt_token_ids: tuple[int, ...]
    decoded_tokens: DecodedTokenNode | None = None
    decoded_length: int = 0
    cache: object | None = None
    last_logits: object | None = None
    policy: MaterializedPolicy | None = None
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("MATDOState requires at least one token")
        if self.decoded_length < 0:
            raise ValueError("decoded_length must be non-negative")
        if (self.decoded_tokens is None) != (self.decoded_length == 0):
            raise ValueError("decoded token tail and decoded_length disagree")

    @property
    def sequence_length(self) -> int:
        return len(self.prompt_token_ids) + self.decoded_length

    @property
    def last_token_id(self) -> int:
        if self.decoded_tokens is not None:
            return self.decoded_tokens.token_id
        return self.prompt_token_ids[-1]

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Materialize the full sequence only when a caller explicitly needs it."""
        if self.decoded_tokens is None:
            return self.prompt_token_ids

        suffix = [0] * self.decoded_length
        current = self.decoded_tokens
        index = self.decoded_length - 1
        while current is not None:
            suffix[index] = current.token_id
            current = current.previous
            index -= 1
        return self.prompt_token_ids + tuple(suffix)

    def append_token(
        self,
        token_id: int,
        *,
        cache: object | None,
        logits: object | None,
        submitted_token_count: int = 1,
        used_incremental_cache: bool = False,
    ) -> "MATDOState":
        return MATDOState(
            prompt_token_ids=self.prompt_token_ids,
            decoded_tokens=DecodedTokenNode(int(token_id), previous=self.decoded_tokens),
            decoded_length=self.decoded_length + 1,
            cache=cache,
            last_logits=logits,
            policy=self.policy,
            metrics=self.metrics.record_decode(
                submitted_tokens=submitted_token_count,
                used_incremental=used_incremental_cache,
            ),
        )
