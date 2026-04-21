from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from matdo_new import REPO_ROOT


LEGACY_NEEDLE_DATASET_PATH = (
    REPO_ROOT / "experiments" / "real_model" / "datasets" / "needle_dataset.py"
)


@dataclass(frozen=True)
class NeedleEvaluationRule:
    """Normalized retrieval scoring rules carried with each adapted example."""

    allow_exact_match: bool = True
    allow_subsequence_match: bool = True
    case_sensitive: bool = False


@dataclass(frozen=True)
class NeedleExample:
    """MATDO-new normalized example built from the legacy dataset samples."""

    example_id: str
    prompt: str
    target_answer: str
    metadata: dict[str, object] = field(default_factory=dict)
    max_new_tokens: int = 16
    evaluation: NeedleEvaluationRule = field(default_factory=NeedleEvaluationRule)

    @property
    def prompt_token_count(self) -> int | None:
        value = self.metadata.get("prompt_token_count")
        if value is None:
            return None
        return int(value)


class LegacyNeedleDataset(Protocol):
    def create_dataset(
        self,
        context_tokens: int,
        num_samples: int,
        depth_distribution: str = "uniform",
    ) -> list[object]: ...


@lru_cache(maxsize=1)
def _load_legacy_needle_dataset_class() -> type[object]:
    module_path = LEGACY_NEEDLE_DATASET_PATH
    if not module_path.exists():
        raise FileNotFoundError(f"legacy NeedleDataset not found at {module_path}")

    spec = importlib.util.spec_from_file_location("matdo_new_legacy_needle_dataset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load legacy NeedleDataset module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    dataset_class = getattr(module, "NeedleDataset", None)
    if dataset_class is None:
        raise ImportError(f"NeedleDataset was not found in {module_path}")
    return dataset_class


class NeedleDatasetAdapter:
    """Adapt the legacy real Needle dataset into MATDO-new-native examples."""

    def __init__(
        self,
        dataset_factory: callable | None = None,
    ) -> None:
        self._dataset_factory = dataset_factory

    def build_examples(
        self,
        benchmark: object,
        *,
        tokenizer: object | None = None,
    ) -> tuple[NeedleExample, ...]:
        dataset = self._build_dataset(seed=int(getattr(benchmark, "seed", 42)))
        examples: list[NeedleExample] = []

        for requested_context_length in getattr(benchmark, "context_lengths"):
            samples = dataset.create_dataset(
                context_tokens=int(requested_context_length),
                num_samples=int(getattr(benchmark, "num_samples")),
                depth_distribution=str(getattr(benchmark, "depth_distribution")),
            )
            for sample_index, sample in enumerate(samples):
                prompt = str(sample.format_prompt())
                target_answer = str(getattr(sample, "secret")).strip()
                prompt_token_count = self._count_prompt_tokens(prompt, tokenizer=tokenizer)
                context_length = (
                    prompt_token_count
                    if prompt_token_count is not None
                    else int(requested_context_length)
                )
                examples.append(
                    NeedleExample(
                        example_id=f"needle:{int(requested_context_length)}:{sample_index}",
                        prompt=prompt,
                        target_answer=target_answer,
                        metadata={
                            "context_length": int(context_length),
                            "requested_context_length": int(requested_context_length),
                            "legacy_context_budget": int(requested_context_length),
                            "prompt_token_count": int(context_length),
                            "needle_depth_percent": float(getattr(sample, "needle_depth_percent")),
                            "sample_index": sample_index,
                            "depth_distribution": str(getattr(benchmark, "depth_distribution")),
                        },
                        max_new_tokens=int(getattr(benchmark, "max_new_tokens")),
                    )
                )

        return tuple(examples)

    def _build_dataset(self, *, seed: int) -> LegacyNeedleDataset:
        if self._dataset_factory is not None:
            try:
                return self._dataset_factory(seed)
            except TypeError:
                return self._dataset_factory()

        dataset_class = _load_legacy_needle_dataset_class()
        return dataset_class(seed=seed)

    @staticmethod
    def _count_prompt_tokens(prompt: str, *, tokenizer: object | None) -> int | None:
        if tokenizer is None:
            return None
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            return None
        token_ids = encode(prompt, add_special_tokens=False)
        return len(tuple(int(token_id) for token_id in token_ids))
