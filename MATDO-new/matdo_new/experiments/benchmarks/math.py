from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


VALID_SPLITS = {"test"}
VALID_PROMPT_STYLES = {"cot_boxed"}


@dataclass(frozen=True)
class MathBenchmark:
    split: str
    max_samples: int | None
    subjects: tuple[str, ...]
    levels: tuple[int, ...]
    prompt_style: str
    max_new_tokens: int
    tokenizer_name: str
    model_size: str
    use_attnres: bool = True
    seed: int = 42
    name: str = "math"
    kind: str = "task"

    def __post_init__(self) -> None:
        if self.name != "math":
            raise ValueError("MathBenchmark.name must be 'math'")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"MathBenchmark.split must be one of {sorted(VALID_SPLITS)}")
        if self.prompt_style not in VALID_PROMPT_STYLES:
            raise ValueError(
                "MathBenchmark.prompt_style must be one of "
                f"{sorted(VALID_PROMPT_STYLES)}"
            )
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("MathBenchmark.max_samples must be positive when provided")
        if self.max_new_tokens <= 0:
            raise ValueError("MathBenchmark.max_new_tokens must be positive")
        if not self.tokenizer_name:
            raise ValueError("MathBenchmark.tokenizer_name must not be empty")
        if not self.model_size:
            raise ValueError("MathBenchmark.model_size must not be empty")

        object.__setattr__(self, "subjects", tuple(str(subject) for subject in self.subjects))
        object.__setattr__(self, "levels", tuple(int(level) for level in self.levels))

    @property
    def config(self) -> dict[str, object]:
        return {
            "split": self.split,
            "max_samples": self.max_samples,
            "subjects": self.subjects,
            "levels": self.levels,
            "prompt_style": self.prompt_style,
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
        split: str = "test",
        max_samples: int | None = None,
        subjects: Sequence[str] = (),
        levels: Sequence[int] = (),
        prompt_style: str = "cot_boxed",
        max_new_tokens: int,
        tokenizer_name: str,
        model_size: str,
        use_attnres: bool = True,
        seed: int = 42,
    ) -> "MathBenchmark":
        normalized_max_samples = None if max_samples is None else int(max_samples)
        return cls(
            split=split,
            max_samples=normalized_max_samples,
            subjects=tuple(str(subject) for subject in subjects),
            levels=tuple(int(level) for level in levels),
            prompt_style=prompt_style,
            max_new_tokens=int(max_new_tokens),
            tokenizer_name=tokenizer_name,
            model_size=model_size,
            use_attnres=bool(use_attnres),
            seed=int(seed),
        )
