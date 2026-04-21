from __future__ import annotations

from collections.abc import Sequence

from matdo_new.experiments.benchmarks.critical_points import CriticalPointsBenchmark
from matdo_new.experiments.benchmarks.needle import NeedleBenchmark
from matdo_new.experiments.evaluators.critical_points import CriticalPointsToyEvaluator
from matdo_new.experiments.run_experiments import (
    BenchmarkRunner,
    ExperimentRunContext,
    run_experiments,
)
from matdo_new.experiments.schema import EvaluatedBenchmark, ResultSummary, RuntimeEnvelope
from matdo_new.runtime.generation import generate_tokens
from matdo_new.runtime.state import BackendResult, MATDOState


class FakeGenerativeBackend:
    def __init__(self, planned_tokens: Sequence[int]) -> None:
        self.planned_tokens = tuple(planned_tokens)
        self.forward_inputs: list[tuple[int, ...]] = []
        self.forward_step_inputs: list[tuple[int, ...]] = []
        self.forward_step_state_lengths: list[int] = []

    def forward(
        self,
        token_ids: Sequence[int],
        *,
        policy: object | None = None,
    ) -> BackendResult:
        tokens = tuple(token_ids)
        self.forward_inputs.append(tokens)
        return BackendResult(
            logits=self.planned_tokens[0],
            cache={"seen_tokens": tokens, "policy": policy},
        )

    def forward_step(
        self,
        token_ids: Sequence[int],
        *,
        state: MATDOState,
        policy: object | None = None,
    ) -> BackendResult:
        tokens = tuple(token_ids)
        self.forward_step_inputs.append(tokens)
        self.forward_step_state_lengths.append(state.sequence_length)
        next_index = len(self.forward_step_inputs)
        next_logits = self.planned_tokens[next_index] if next_index < len(self.planned_tokens) else None
        return BackendResult(
            logits=next_logits,
            cache={
                "last_token": tokens[-1],
                "sequence_length": state.sequence_length + len(tokens),
                "policy": policy,
            },
        )


def take_planned_token(logits: object | None) -> int:
    assert logits is not None
    return int(logits)


class NeedleSchemaEvaluator:
    name = "needle-real"

    def evaluate(self, benchmark: NeedleBenchmark) -> EvaluatedBenchmark:
        metrics = {
            "num_samples": benchmark.num_samples,
            "exact_match_rate": 1.0,
            "retrieval_success_rate": 1.0,
        }
        return EvaluatedBenchmark(
            aggregate_metrics=metrics,
            slice_summaries=(
                ResultSummary(
                    summary_id="context-length:128",
                    metrics={"context_length": 128, "exact_match_rate": 1.0},
                    metadata={"context_length": 128},
                ),
            ),
            example_summaries=(
                ResultSummary(
                    summary_id="needle:128:0",
                    metrics={"exact_match": True, "retrieval_success": True},
                    metadata={
                        "context_length": 128,
                        "needle_depth_percent": 50.0,
                    },
                ),
            ),
            runtime=RuntimeEnvelope(
                policy={"tokenizer_name": benchmark.tokenizer_name},
                metrics={"prefill_calls": 1, "decode_calls": 1},
            ),
        )


def test_generate_tokens_extends_sequence_to_requested_length() -> None:
    backend = FakeGenerativeBackend([41, 42, 43])

    result = generate_tokens(
        [11, 22],
        backend=backend,
        sampler=take_planned_token,
        max_new_tokens=3,
    )

    assert result.prompt_token_ids == (11, 22)
    assert result.generated_token_ids == (41, 42, 43)
    assert result.token_ids == (11, 22, 41, 42, 43)
    assert result.sequence_length == 5
    assert result.state.metrics.prefill_calls == 1
    assert result.state.metrics.decode_calls == 3
    assert backend.forward_inputs == [(11, 22)]
    assert backend.forward_step_inputs == [(41,), (42,), (43,)]
    assert backend.forward_step_state_lengths == [2, 3, 4]


def test_benchmark_runners_return_rich_result_schema() -> None:
    run_context = ExperimentRunContext(
        app="tests",
        config_path="/tmp/experiments.yaml",
        root_config_path="/tmp/root.yaml",
    )

    results = run_experiments(
        BenchmarkRunner(
            benchmark=NeedleBenchmark(
                context_lengths=[128],
                num_samples=1,
                max_new_tokens=1,
                tokenizer_name="fake-tokenizer",
                model_size="t4",
            ),
            evaluator=NeedleSchemaEvaluator(),
        ),
        BenchmarkRunner(
            benchmark=CriticalPointsBenchmark(points=[0.25, 0.5]),
            evaluator=CriticalPointsToyEvaluator(),
        ),
        run_context=run_context,
    )

    assert len(results) == 2

    task_result = results[0]
    assert task_result.schema_version == "matdo_new.benchmark_result.v1"
    assert task_result.run.app == "tests"
    assert task_result.run.config_path == "/tmp/experiments.yaml"
    assert task_result.run.evaluator_name == "needle-real"
    assert task_result.benchmark.name == "needle"
    assert task_result.benchmark.kind == "task"
    assert task_result.benchmark.config == {
        "context_lengths": (128,),
        "depth_distribution": "uniform",
        "max_new_tokens": 1,
        "model_size": "t4",
        "num_samples": 1,
        "seed": 42,
        "tokenizer_name": "fake-tokenizer",
        "use_attnres": True,
    }
    assert task_result.aggregate_metrics == {
        "exact_match_rate": 1.0,
        "num_samples": 1,
        "retrieval_success_rate": 1.0,
    }
    assert len(task_result.slice_summaries) == 1
    assert task_result.slice_summaries[0].summary_id == "context-length:128"
    assert task_result.slice_summaries[0].metrics == {
        "context_length": 128,
        "exact_match_rate": 1.0,
    }
    assert len(task_result.example_summaries) == 1
    assert task_result.example_summaries[0].summary_id == "needle:128:0"
    assert task_result.example_summaries[0].metrics == {
        "exact_match": True,
        "retrieval_success": True,
    }
    assert task_result.example_summaries[0].metadata == {
        "context_length": 128,
        "needle_depth_percent": 50.0,
    }
    assert task_result.runtime is not None
    assert task_result.runtime.metrics == {"prefill_calls": 1, "decode_calls": 1}

    study_result = results[1]
    assert study_result.schema_version == "matdo_new.benchmark_result.v1"
    assert study_result.run.root_config_path == "/tmp/root.yaml"
    assert study_result.run.evaluator_name == "critical-points-toy"
    assert study_result.benchmark.name == "critical-points"
    assert study_result.benchmark.kind == "study"
    assert study_result.aggregate_metrics == {
        "max_score": 0.75,
        "mean_score": 0.625,
        "min_score": 0.5,
        "num_points": 2,
    }
    assert [summary.summary_id for summary in study_result.slice_summaries] == [
        "critical-point:0.25",
        "critical-point:0.5",
    ]
    assert study_result.slice_summaries[0].metrics == {"critical_point": 0.25, "score": 0.75}
    assert study_result.slice_summaries[1].metrics == {"critical_point": 0.5, "score": 0.5}
