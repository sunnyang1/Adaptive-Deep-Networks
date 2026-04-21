from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from matdo_new.experiments.schema import (
    BenchmarkIdentity,
    BenchmarkResult,
    EvaluatedBenchmark,
    RunMetadata,
)


@dataclass(frozen=True)
class ExperimentRunContext:
    app: str
    config_path: str
    root_config_path: str


class Benchmark(Protocol):
    name: str
    kind: str

    @property
    def config(self) -> dict[str, object]:
        ...


class BenchmarkEvaluator(Protocol):
    name: str

    def evaluate(self, benchmark: Benchmark) -> EvaluatedBenchmark:
        ...


@dataclass(frozen=True)
class BenchmarkRunner:
    benchmark: Benchmark
    evaluator: BenchmarkEvaluator

    def run(self, *, run_context: ExperimentRunContext) -> BenchmarkResult:
        evaluated = self.evaluator.evaluate(self.benchmark)
        return BenchmarkResult(
            run=RunMetadata(
                app=run_context.app,
                config_path=run_context.config_path,
                root_config_path=run_context.root_config_path,
                evaluator_name=self.evaluator.name,
            ),
            benchmark=BenchmarkIdentity(
                name=self.benchmark.name,
                kind=self.benchmark.kind,
                config=self.benchmark.config,
            ),
            aggregate_metrics=evaluated.aggregate_metrics,
            slice_summaries=evaluated.slice_summaries,
            example_summaries=evaluated.example_summaries,
            runtime=evaluated.runtime,
        )


def run_experiments(
    *runners: BenchmarkRunner,
    run_context: ExperimentRunContext,
) -> tuple[BenchmarkResult, ...]:
    """Execute benchmark runners and return normalized benchmark results."""
    return tuple(runner.run(run_context=run_context) for runner in runners)
