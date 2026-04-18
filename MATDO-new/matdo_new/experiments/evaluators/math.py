from __future__ import annotations

from functools import lru_cache
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Protocol

from matdo_new.experiments.benchmarks.math import MathBenchmark
from matdo_new.experiments.schema import EvaluatedBenchmark, ResultSummary, RuntimeEnvelope
from matdo_new.experiments.tasks.math import MathDatasetAdapter, MathExample
from matdo_new.modeling.config import MATDOModelConfig
from matdo_new.modeling.matdo_model import MATDOModel
from matdo_new.repo_imports import repo_root_on_path
from matdo_new.runtime.generation import LogitsSampler, generate_tokens


class TextTokenizer(Protocol):
    def encode(
        self,
        text: str,
        return_tensors: str | None = None,
        max_length: int | None = None,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> list[int] | object: ...

    def decode(self, tokens: Sequence[int] | object, skip_special_tokens: bool = True) -> str: ...


SamplerFactory = Callable[[MathExample, tuple[int, ...]], LogitsSampler]


@lru_cache(maxsize=1)
def _load_repo_bridge_classes() -> tuple[type[object], type[object]]:
    with repo_root_on_path():
        from src.models.adaptive_transformer import AdaptiveTransformer
        from src.models.tokenizer import TokenizerWrapper

    return TokenizerWrapper, AdaptiveTransformer


@lru_cache(maxsize=1)
def _load_runtime_backend_class() -> type[object]:
    from matdo_new.runtime.backend import AdaptiveTransformerRuntimeBackend

    return AdaptiveTransformerRuntimeBackend


@dataclass(frozen=True)
class _ExampleOutcome:
    example: MathExample
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    generated_text: str
    candidate_answer: str
    extraction_mode: str
    correct: bool
    runtime_metrics: dict[str, int]


@dataclass
class MathEvaluator:
    name: str = "math-real"
    tokenizer: TextTokenizer | None = None
    tokenizer_factory: Callable[[str], TextTokenizer] | None = None
    backend_factory: Callable[[MathBenchmark, TextTokenizer], object] | None = None
    dataset_adapter: MathDatasetAdapter | None = None
    sampler_factory: SamplerFactory | None = None

    def evaluate(
        self,
        benchmark: MathBenchmark,
        *,
        examples: Sequence[MathExample] | None = None,
    ) -> EvaluatedBenchmark:
        tokenizer = self._resolve_tokenizer(benchmark)
        backend = self._build_backend(benchmark, tokenizer)
        adapted_examples = (
            tuple(examples)
            if examples is not None
            else self._dataset_adapter().build_examples(benchmark)
        )
        outcomes = tuple(
            self._evaluate_example(example, tokenizer=tokenizer, backend=backend)
            for example in adapted_examples
        )
        return EvaluatedBenchmark(
            aggregate_metrics=self._aggregate_metrics(outcomes),
            slice_summaries=self._slice_summaries(outcomes),
            example_summaries=tuple(self._example_summary(outcome) for outcome in outcomes),
            runtime=self._runtime_envelope(benchmark, outcomes),
        )

    def _resolve_tokenizer(self, benchmark: MathBenchmark) -> TextTokenizer:
        if self.tokenizer is not None:
            return self.tokenizer
        if self.tokenizer_factory is not None:
            return self.tokenizer_factory(benchmark.tokenizer_name)
        try:
            tokenizer_wrapper_cls, _adaptive_transformer_cls = _load_repo_bridge_classes()
        except ImportError as exc:
            raise RuntimeError(
                "The default real-model MATH evaluator requires the repository bridge / "
                "source checkout modules (src.models.*). Provide a tokenizer or "
                "tokenizer_factory, or run from a source checkout."
            ) from exc
        return tokenizer_wrapper_cls(tokenizer_name=benchmark.tokenizer_name)

    @staticmethod
    def _resolve_tokenizer_vocab_size(tokenizer: TextTokenizer) -> int | None:
        length = getattr(tokenizer, "__len__", None)
        if length is None:
            return None
        vocab_size = length()
        if vocab_size is None:
            return None
        return int(vocab_size)

    def _build_backend(self, benchmark: MathBenchmark, tokenizer: TextTokenizer) -> object:
        if self.backend_factory is not None:
            return self.backend_factory(benchmark, tokenizer)
        try:
            _tokenizer_wrapper_cls, adaptive_transformer_cls = _load_repo_bridge_classes()
            runtime_backend_cls = _load_runtime_backend_class()
        except ImportError as exc:
            raise RuntimeError(
                "The default real-model MATH evaluator requires the repository bridge / "
                "source checkout modules (src.models.*). Provide a backend_factory, or "
                "run from a source checkout."
            ) from exc
        model_config = MATDOModelConfig(
            model_size=benchmark.model_size,
            use_attnres=benchmark.use_attnres,
        )
        backend_config = model_config.build_backend_config()
        tokenizer_vocab_size = self._resolve_tokenizer_vocab_size(tokenizer)
        if tokenizer_vocab_size is not None:
            backend_config.vocab_size = tokenizer_vocab_size
        model = adaptive_transformer_cls(backend_config)
        runtime_model = MATDOModel(config=model_config, backend=model)
        return runtime_backend_cls(
            model,
            runtime_model=runtime_model,
            use_attnres=model_config.use_attnres,
            use_engram=model_config.use_engram,
        )

    def _dataset_adapter(self) -> MathDatasetAdapter:
        if self.dataset_adapter is not None:
            return self.dataset_adapter
        return MathDatasetAdapter()

    def _evaluate_example(
        self,
        example: MathExample,
        *,
        tokenizer: TextTokenizer,
        backend: object,
    ) -> _ExampleOutcome:
        prompt_token_ids = tuple(
            int(token_id)
            for token_id in tokenizer.encode(example.prompt, add_special_tokens=False)
        )
        generation = generate_tokens(
            prompt_token_ids,
            backend=backend,
            sampler=self._build_sampler(example, prompt_token_ids=prompt_token_ids, backend=backend),
            max_new_tokens=example.max_new_tokens,
        )
        generated_token_ids = tuple(int(token_id) for token_id in generation.generated_token_ids)
        generated_text = tokenizer.decode(generated_token_ids)
        candidate_answer, extraction_mode = self._dataset_adapter().extract_candidate_answer(
            generated_text
        )
        correct = self._dataset_adapter().answers_match(candidate_answer, example.gold_answer)
        runtime_metrics = {
            "prefill_calls": int(generation.state.metrics.prefill_calls),
            "decode_calls": int(generation.state.metrics.decode_calls),
            "prompt_tokens": int(generation.state.metrics.prompt_tokens),
            "decode_tokens": int(generation.state.metrics.decode_tokens),
            "submitted_tokens": int(generation.state.metrics.submitted_tokens),
            "incremental_decode_calls": int(generation.state.metrics.incremental_decode_calls),
        }
        return _ExampleOutcome(
            example=example,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
            candidate_answer=candidate_answer,
            extraction_mode=extraction_mode,
            correct=correct,
            runtime_metrics=runtime_metrics,
        )

    def _build_sampler(
        self,
        example: MathExample,
        *,
        prompt_token_ids: tuple[int, ...],
        backend: object,
    ) -> LogitsSampler:
        if self.sampler_factory is not None:
            return self.sampler_factory(example, prompt_token_ids)

        return self._greedy_sampler

    @staticmethod
    def _greedy_sampler(logits: object | None) -> int:
        if logits is None:
            raise ValueError("backend must return logits to generate Math continuations")
        argmax = getattr(logits, "argmax", None)
        if argmax is None:
            return int(logits)
        value = argmax()
        return int(value.item() if hasattr(value, "item") else value)

    @staticmethod
    def _aggregate_metrics(outcomes: Sequence[_ExampleOutcome]) -> dict[str, float | int]:
        num_samples = len(outcomes)
        if num_samples == 0:
            return {
                "num_samples": 0,
                "accuracy": 0.0,
                "mean_prompt_tokens": 0.0,
                "mean_generated_tokens": 0.0,
            }

        prompt_lengths = [len(outcome.prompt_token_ids) for outcome in outcomes]
        generated_lengths = [len(outcome.generated_token_ids) for outcome in outcomes]
        return {
            "num_samples": num_samples,
            "accuracy": sum(1 for outcome in outcomes if outcome.correct) / num_samples,
            "mean_prompt_tokens": mean(prompt_lengths),
            "mean_generated_tokens": mean(generated_lengths),
        }

    def _slice_summaries(self, outcomes: Sequence[_ExampleOutcome]) -> tuple[ResultSummary, ...]:
        summaries: list[ResultSummary] = []
        by_level: dict[int, list[_ExampleOutcome]] = defaultdict(list)
        by_subject: dict[str, list[_ExampleOutcome]] = defaultdict(list)

        for outcome in outcomes:
            by_level[int(outcome.example.level)].append(outcome)
            by_subject[str(outcome.example.subject)].append(outcome)

        for level in sorted(by_level):
            bucket = by_level[level]
            summaries.append(
                ResultSummary(
                    summary_id=f"math-level:{level}",
                    metrics={
                        "num_samples": len(bucket),
                        "accuracy": self._accuracy(bucket),
                    },
                    metadata={"level": level},
                )
            )

        for subject in sorted(by_subject):
            bucket = by_subject[subject]
            summaries.append(
                ResultSummary(
                    summary_id=f"math-subject:{subject}",
                    metrics={
                        "num_samples": len(bucket),
                        "accuracy": self._accuracy(bucket),
                    },
                    metadata={"subject": subject},
                )
            )

        return tuple(summaries)

    def _example_summary(self, outcome: _ExampleOutcome) -> ResultSummary:
        metadata = dict(outcome.example.metadata)
        metadata.update(
            {
                "subject": outcome.example.subject,
                "level": outcome.example.level,
                "gold_answer": outcome.example.gold_answer,
                "candidate_answer": outcome.candidate_answer,
                "extraction_mode": outcome.extraction_mode,
                "generated_text": outcome.generated_text,
                "generated_token_ids": outcome.generated_token_ids,
            }
        )
        return ResultSummary(
            summary_id=outcome.example.example_id,
            metrics={
                "correct": outcome.correct,
                "prompt_tokens": len(outcome.prompt_token_ids),
                "generated_tokens": len(outcome.generated_token_ids),
            },
            metadata=metadata,
        )

    def _runtime_envelope(
        self,
        benchmark: MathBenchmark,
        outcomes: Sequence[_ExampleOutcome],
    ) -> RuntimeEnvelope:
        totals = {
            "prefill_calls": 0,
            "decode_calls": 0,
            "prompt_tokens": 0,
            "decode_tokens": 0,
            "submitted_tokens": 0,
            "incremental_decode_calls": 0,
        }
        for outcome in outcomes:
            for key, value in outcome.runtime_metrics.items():
                totals[key] += value
        return RuntimeEnvelope(
            policy={
                "tokenizer_name": benchmark.tokenizer_name,
                "model_size": benchmark.model_size,
                "use_attnres": benchmark.use_attnres,
                "split": benchmark.split,
                "subjects": benchmark.subjects,
                "levels": benchmark.levels,
                "prompt_style": benchmark.prompt_style,
                "max_samples": benchmark.max_samples,
                "max_new_tokens": benchmark.max_new_tokens,
                "seed": benchmark.seed,
            },
            metrics=totals,
        )

    @staticmethod
    def _accuracy(outcomes: Sequence[_ExampleOutcome]) -> float:
        if not outcomes:
            return 0.0
        return sum(1 for outcome in outcomes if outcome.correct) / len(outcomes)
