from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Protocol

from matdo_new.experiments.benchmarks.needle import NeedleBenchmark
from matdo_new.experiments.schema import EvaluatedBenchmark, ResultSummary, RuntimeEnvelope
from matdo_new.modeling.matdo_model import MATDOModel
from matdo_new.experiments.tasks.needle import NeedleDatasetAdapter, NeedleExample, NeedleEvaluationRule
from matdo_new.modeling.config import MATDOModelConfig
from matdo_new.repo_imports import repo_root_on_path
from matdo_new.runtime.backend import AdaptiveTransformerRuntimeBackend
from matdo_new.runtime.generation import LogitsSampler, generate_tokens

with repo_root_on_path():
    from src.models.adaptive_transformer import AdaptiveTransformer
    from src.models.tokenizer import TokenizerWrapper


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


SamplerFactory = Callable[[NeedleExample, tuple[int, ...]], LogitsSampler]


@dataclass(frozen=True)
class _ExampleOutcome:
    example: NeedleExample
    prompt_token_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    generated_text: str
    exact_match: bool
    retrieval_success: bool
    runtime_metrics: dict[str, int]


@dataclass
class NeedleEvaluator:
    name: str = "needle-real"
    tokenizer: TextTokenizer | None = None
    tokenizer_factory: Callable[[str], TextTokenizer] | None = None
    backend_factory: Callable[[NeedleBenchmark, TextTokenizer], object] | None = None
    dataset_adapter: NeedleDatasetAdapter | None = None
    sampler_factory: SamplerFactory | None = None

    def evaluate(
        self,
        benchmark: NeedleBenchmark,
        *,
        examples: Sequence[NeedleExample] | None = None,
    ) -> EvaluatedBenchmark:
        tokenizer = self._resolve_tokenizer(benchmark)
        backend = self._build_backend(benchmark, tokenizer)
        adapted_examples = (
            tuple(examples)
            if examples is not None
            else self._dataset_adapter().build_examples(benchmark, tokenizer=tokenizer)
        )
        outcomes = tuple(self._evaluate_example(example, tokenizer=tokenizer, backend=backend) for example in adapted_examples)

        aggregate_metrics = self._aggregate_metrics(outcomes)
        slice_summaries = self._slice_summaries(outcomes)
        example_summaries = tuple(self._example_summary(outcome) for outcome in outcomes)
        runtime = self._runtime_envelope(benchmark, outcomes)
        return EvaluatedBenchmark(
            aggregate_metrics=aggregate_metrics,
            slice_summaries=slice_summaries,
            example_summaries=example_summaries,
            runtime=runtime,
        )

    def _resolve_tokenizer(self, benchmark: NeedleBenchmark) -> TextTokenizer:
        if self.tokenizer is not None:
            return self.tokenizer
        if self.tokenizer_factory is not None:
            return self.tokenizer_factory(benchmark.tokenizer_name)
        return TokenizerWrapper(tokenizer_name=benchmark.tokenizer_name)

    @staticmethod
    def _resolve_tokenizer_vocab_size(tokenizer: TextTokenizer) -> int | None:
        length = getattr(tokenizer, "__len__", None)
        if length is None:
            return None
        vocab_size = length()
        if vocab_size is None:
            return None
        return int(vocab_size)

    def _build_backend(self, benchmark: NeedleBenchmark, tokenizer: TextTokenizer) -> object:
        if self.backend_factory is not None:
            return self.backend_factory(benchmark, tokenizer)
        model_config = MATDOModelConfig(
            model_size=benchmark.model_size,
            use_attnres=benchmark.use_attnres,
        )
        backend_config = model_config.build_backend_config()
        tokenizer_vocab_size = self._resolve_tokenizer_vocab_size(tokenizer)
        if tokenizer_vocab_size is not None:
            backend_config.vocab_size = tokenizer_vocab_size
        model = AdaptiveTransformer(backend_config)
        runtime_model = MATDOModel(config=model_config, backend=model)
        return AdaptiveTransformerRuntimeBackend(
            model,
            runtime_model=runtime_model,
            use_attnres=model_config.use_attnres,
            use_engram=model_config.use_engram,
        )

    def _dataset_adapter(self) -> NeedleDatasetAdapter:
        if self.dataset_adapter is not None:
            return self.dataset_adapter
        return NeedleDatasetAdapter()

    def _evaluate_example(
        self,
        example: NeedleExample,
        *,
        tokenizer: TextTokenizer,
        backend: object,
    ) -> _ExampleOutcome:
        prompt_token_ids = tuple(
            int(token_id)
            for token_id in tokenizer.encode(example.prompt, add_special_tokens=False)
        )
        answer_token_ids = tuple(
            int(token_id)
            for token_id in tokenizer.encode(example.target_answer, add_special_tokens=False)
        )
        sampler = self._build_sampler(example, answer_token_ids=answer_token_ids, backend=backend)
        generation = generate_tokens(
            prompt_token_ids,
            backend=backend,
            sampler=sampler,
            max_new_tokens=example.max_new_tokens,
        )
        generated_token_ids = tuple(int(token_id) for token_id in generation.generated_token_ids)
        generated_text = tokenizer.decode(generated_token_ids)
        exact_match = self._is_exact_match(
            generated_token_ids,
            answer_token_ids,
            generated_text=generated_text,
            target_answer=example.target_answer,
            rule=example.evaluation,
        )
        retrieval_success = exact_match or self._contains_answer_tokens(
            generated_token_ids,
            answer_token_ids,
            rule=example.evaluation,
        )
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
            answer_token_ids=answer_token_ids,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
            exact_match=exact_match,
            retrieval_success=retrieval_success,
            runtime_metrics=runtime_metrics,
        )

    def _build_sampler(
        self,
        example: NeedleExample,
        *,
        answer_token_ids: tuple[int, ...],
        backend: object,
    ) -> LogitsSampler:
        if self.sampler_factory is not None:
            return self.sampler_factory(example, answer_token_ids)

        runtime_model = getattr(backend, "runtime_model", None)
        if runtime_model is not None and hasattr(runtime_model, "sample_next_token"):
            return lambda logits: int(runtime_model.sample_next_token(logits, temperature=1.0))

        return self._greedy_sampler

    @staticmethod
    def _greedy_sampler(logits: object | None) -> int:
        if logits is None:
            raise ValueError("backend must return logits to generate Needle continuations")
        argmax = getattr(logits, "argmax", None)
        if argmax is None:
            return int(logits)
        value = argmax()
        return int(value.item() if hasattr(value, "item") else value)

    def _aggregate_metrics(self, outcomes: Sequence[_ExampleOutcome]) -> dict[str, float | int]:
        prompt_lengths = [len(outcome.prompt_token_ids) for outcome in outcomes]
        generated_lengths = [len(outcome.generated_token_ids) for outcome in outcomes]
        num_samples = len(outcomes)
        if num_samples == 0:
            return {
                "num_samples": 0,
                "exact_match_rate": 0.0,
                "retrieval_success_rate": 0.0,
                "mean_prompt_tokens": 0.0,
                "mean_generated_tokens": 0.0,
            }
        return {
            "num_samples": num_samples,
            "exact_match_rate": sum(1 for outcome in outcomes if outcome.exact_match) / num_samples,
            "retrieval_success_rate": sum(1 for outcome in outcomes if outcome.retrieval_success) / num_samples,
            "mean_prompt_tokens": mean(prompt_lengths),
            "mean_generated_tokens": mean(generated_lengths),
        }

    def _slice_summaries(self, outcomes: Sequence[_ExampleOutcome]) -> tuple[ResultSummary, ...]:
        summaries: list[ResultSummary] = []

        by_context_length: dict[int, list[_ExampleOutcome]] = defaultdict(list)
        by_depth_bucket: dict[str, list[_ExampleOutcome]] = defaultdict(list)
        for outcome in outcomes:
            context_length = int(outcome.example.metadata["context_length"])
            by_context_length[context_length].append(outcome)
            by_depth_bucket[self._depth_bucket(float(outcome.example.metadata["needle_depth_percent"]))].append(outcome)

        for context_length in sorted(by_context_length):
            bucket = by_context_length[context_length]
            summaries.append(
                ResultSummary(
                    summary_id=f"context-length:{context_length}",
                    metrics={
                        "context_length": context_length,
                        "num_samples": len(bucket),
                        "exact_match_rate": self._success_rate(bucket, exact=True),
                        "retrieval_success_rate": self._success_rate(bucket, exact=False),
                    },
                    metadata={"context_length": context_length},
                )
            )

        for bucket_name in ("early", "middle", "late"):
            bucket = by_depth_bucket.get(bucket_name)
            if not bucket:
                continue
            summaries.append(
                ResultSummary(
                    summary_id=f"needle-depth:{bucket_name}",
                    metrics={
                        "num_samples": len(bucket),
                        "exact_match_rate": self._success_rate(bucket, exact=True),
                        "retrieval_success_rate": self._success_rate(bucket, exact=False),
                    },
                    metadata={"depth_bucket": bucket_name},
                )
            )

        return tuple(summaries)

    def _example_summary(self, outcome: _ExampleOutcome) -> ResultSummary:
        metadata = dict(outcome.example.metadata)
        metadata.update(
            {
                "target_answer": outcome.example.target_answer,
                "generated_text": outcome.generated_text,
                "prompt_token_count": len(outcome.prompt_token_ids),
                "answer_token_count": len(outcome.answer_token_ids),
                "generated_token_ids": outcome.generated_token_ids,
            }
        )
        return ResultSummary(
            summary_id=outcome.example.example_id,
            metrics={
                "exact_match": outcome.exact_match,
                "retrieval_success": outcome.retrieval_success,
                "prompt_tokens": len(outcome.prompt_token_ids),
                "generated_tokens": len(outcome.generated_token_ids),
            },
            metadata=metadata,
        )

    def _runtime_envelope(
        self,
        benchmark: NeedleBenchmark,
        outcomes: Sequence[_ExampleOutcome],
    ) -> RuntimeEnvelope:
        totals: dict[str, int] = {
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
                "depth_distribution": benchmark.depth_distribution,
                "max_new_tokens": benchmark.max_new_tokens,
            },
            metrics=totals,
        )

    def _is_exact_match(
        self,
        generated_token_ids: tuple[int, ...],
        answer_token_ids: tuple[int, ...],
        *,
        generated_text: str,
        target_answer: str,
        rule: NeedleEvaluationRule,
    ) -> bool:
        if not rule.allow_exact_match:
            return False
        if generated_token_ids == answer_token_ids:
            return True
        return self._normalize(generated_text, case_sensitive=rule.case_sensitive) == self._normalize(
            target_answer,
            case_sensitive=rule.case_sensitive,
        )

    def _contains_answer_tokens(
        self,
        generated_token_ids: tuple[int, ...],
        answer_token_ids: tuple[int, ...],
        *,
        rule: NeedleEvaluationRule,
    ) -> bool:
        if not rule.allow_subsequence_match:
            return False
        if not answer_token_ids:
            return False
        if len(answer_token_ids) > len(generated_token_ids):
            return False
        window_size = len(answer_token_ids)
        return any(
            generated_token_ids[index : index + window_size] == answer_token_ids
            for index in range(len(generated_token_ids) - window_size + 1)
        )

    @staticmethod
    def _normalize(text: str, *, case_sensitive: bool) -> str:
        collapsed = " ".join(text.strip().split())
        return collapsed if case_sensitive else collapsed.lower()

    @staticmethod
    def _depth_bucket(depth_percent: float) -> str:
        if depth_percent < 33.334:
            return "early"
        if depth_percent < 66.667:
            return "middle"
        return "late"

    @staticmethod
    def _success_rate(outcomes: Sequence[_ExampleOutcome], *, exact: bool) -> float:
        if not outcomes:
            return 0.0
        if exact:
            return sum(1 for outcome in outcomes if outcome.exact_match) / len(outcomes)
        return sum(1 for outcome in outcomes if outcome.retrieval_success) / len(outcomes)
