from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeMetrics:
    """Small bookkeeping snapshot for prompt prefill and incremental decode."""

    prefill_calls: int = 0
    decode_calls: int = 0
    prompt_tokens: int = 0
    decode_tokens: int = 0
    prefill_submitted_tokens: int = 0
    decode_submitted_tokens: int = 0
    submitted_tokens: int = 0
    incremental_decode_calls: int = 0
    last_decode_used_incremental: bool = False

    @property
    def decode_used_incremental(self) -> bool:
        return self.last_decode_used_incremental

    @classmethod
    def from_prefill(cls, prompt_tokens: int) -> "RuntimeMetrics":
        return cls(
            prefill_calls=1,
            prompt_tokens=prompt_tokens,
            prefill_submitted_tokens=prompt_tokens,
            submitted_tokens=prompt_tokens,
        )

    def record_decode(
        self,
        *,
        submitted_tokens: int = 1,
        used_incremental: bool = False,
    ) -> "RuntimeMetrics":
        return RuntimeMetrics(
            prefill_calls=self.prefill_calls,
            decode_calls=self.decode_calls + 1,
            prompt_tokens=self.prompt_tokens,
            decode_tokens=self.decode_tokens + 1,
            prefill_submitted_tokens=self.prefill_submitted_tokens,
            decode_submitted_tokens=self.decode_submitted_tokens + submitted_tokens,
            submitted_tokens=self.submitted_tokens + submitted_tokens,
            incremental_decode_calls=(
                self.incremental_decode_calls + (1 if used_incremental else 0)
            ),
            last_decode_used_incremental=used_incremental,
        )
