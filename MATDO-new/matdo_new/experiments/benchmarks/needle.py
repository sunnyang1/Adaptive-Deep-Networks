from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


VALID_DEPTH_DISTRIBUTIONS = {"uniform", "early", "late"}


@dataclass(frozen=True)
class NeedleBenchmark:
    context_lengths: tuple[int, ...]
    num_samples: int
    max_new_tokens: int
    tokenizer_name: str
    model_size: str
    depth_distribution: str = "uniform"
    use_attnres: bool = True
    seed: int = 42
    name: str = "needle"
    kind: str = "task"

    def __post_init__(self) -> None:
        context_lengths = tuple(int(context_length) for context_length in self.context_lengths)
        if not context_lengths:
            raise ValueError("NeedleBenchmark.context_lengths must not be empty")
        if any(context_length <= 0 for context_length in context_lengths):
            raise ValueError("NeedleBenchmark.context_lengths must contain positive values")
        if self.num_samples <= 0:
            raise ValueError("NeedleBenchmark.num_samples must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("NeedleBenchmark.max_new_tokens must be positive")
        if not self.tokenizer_name:
            raise ValueError("NeedleBenchmark.tokenizer_name must not be empty")
        if not self.model_size:
            raise ValueError("NeedleBenchmark.model_size must not be empty")
        if self.depth_distribution not in VALID_DEPTH_DISTRIBUTIONS:
            raise ValueError(
                "NeedleBenchmark.depth_distribution must be one of "
                f"{sorted(VALID_DEPTH_DISTRIBUTIONS)}"
            )
        object.__setattr__(self, "context_lengths", context_lengths)

    @property
    def config(self) -> dict[str, object]:
        return {
            "context_lengths": self.context_lengths,
            "num_samples": self.num_samples,
            "depth_distribution": self.depth_distribution,
            "max_new_tokens": self.max_new_tokens,
            "tokenizer_name": self.tokenizer_name,
            "model_size": self.model_size,
            "use_attnres": self.use_attnres,
            "seed": self.seed,
        }

    @classmethod
    def build(
        cls,
        *,
        context_lengths: Sequence[int],
        num_samples: int,
        max_new_tokens: int,
        tokenizer_name: str,
        model_size: str,
        depth_distribution: str = "uniform",
        use_attnres: bool = True,
        seed: int = 42,
        name: str = "needle",
    ) -> "NeedleBenchmark":
        return cls(
            context_lengths=tuple(int(context_length) for context_length in context_lengths),
            num_samples=int(num_samples),
            max_new_tokens=int(max_new_tokens),
            tokenizer_name=tokenizer_name,
            model_size=model_size,
            depth_distribution=depth_distribution,
            use_attnres=bool(use_attnres),
            seed=int(seed),
            name=name,
        )
