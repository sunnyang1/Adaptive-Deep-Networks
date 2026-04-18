from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

ScalarMetric: TypeAlias = bool | int | float | str


def _copy_dict(values: dict[str, object] | None) -> dict[str, object]:
    if values is None:
        return {}
    return dict(values)


def _copy_metrics(values: dict[str, ScalarMetric] | None) -> dict[str, ScalarMetric]:
    if values is None:
        return {}
    return dict(values)


@dataclass(frozen=True)
class RunMetadata:
    app: str
    config_path: str
    root_config_path: str
    evaluator_name: str | None = None

    def __post_init__(self) -> None:
        if not self.app:
            raise ValueError("RunMetadata.app must not be empty")
        if not self.config_path:
            raise ValueError("RunMetadata.config_path must not be empty")
        if not self.root_config_path:
            raise ValueError("RunMetadata.root_config_path must not be empty")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "app": self.app,
            "config_path": self.config_path,
            "root_config_path": self.root_config_path,
        }
        if self.evaluator_name is not None:
            payload["evaluator_name"] = self.evaluator_name
        return payload


@dataclass(frozen=True)
class BenchmarkIdentity:
    name: str
    kind: str
    config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BenchmarkIdentity.name must not be empty")
        if not self.kind:
            raise ValueError("BenchmarkIdentity.kind must not be empty")
        object.__setattr__(self, "config", _copy_dict(self.config))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class ResultSummary:
    summary_id: str
    metrics: dict[str, ScalarMetric]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.summary_id:
            raise ValueError("ResultSummary.summary_id must not be empty")
        object.__setattr__(self, "metrics", _copy_metrics(self.metrics))
        object.__setattr__(self, "metadata", _copy_dict(self.metadata))

    def to_dict(self, *, emit_metadata: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "summary_id": self.summary_id,
            "metrics": dict(self.metrics),
        }
        if emit_metadata and self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeEnvelope:
    policy: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, ScalarMetric] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _copy_dict(self.policy))
        object.__setattr__(self, "metrics", _copy_metrics(self.metrics))

    def is_empty(self) -> bool:
        return not self.policy and not self.metrics

    def to_dict(self, *, emit_metadata: bool) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.metrics:
            payload["metrics"] = dict(self.metrics)
        if emit_metadata and self.policy:
            payload["policy"] = dict(self.policy)
        return payload


@dataclass(frozen=True)
class EvaluatedBenchmark:
    aggregate_metrics: dict[str, ScalarMetric]
    slice_summaries: tuple[ResultSummary, ...] = ()
    example_summaries: tuple[ResultSummary, ...] = ()
    runtime: RuntimeEnvelope | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregate_metrics", _copy_metrics(self.aggregate_metrics))
        object.__setattr__(self, "slice_summaries", tuple(self.slice_summaries))
        object.__setattr__(self, "example_summaries", tuple(self.example_summaries))


@dataclass(frozen=True)
class BenchmarkResult:
    run: RunMetadata
    benchmark: BenchmarkIdentity
    aggregate_metrics: dict[str, ScalarMetric]
    slice_summaries: tuple[ResultSummary, ...] = ()
    example_summaries: tuple[ResultSummary, ...] = ()
    runtime: RuntimeEnvelope | None = None
    schema_version: str = "matdo_new.benchmark_result.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregate_metrics", _copy_metrics(self.aggregate_metrics))
        object.__setattr__(self, "slice_summaries", tuple(self.slice_summaries))
        object.__setattr__(self, "example_summaries", tuple(self.example_summaries))

    def to_dict(self, *, emit_metadata: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "aggregate_metrics": dict(self.aggregate_metrics),
            "slice_summaries": [
                summary.to_dict(emit_metadata=emit_metadata) for summary in self.slice_summaries
            ],
            "example_summaries": [
                summary.to_dict(emit_metadata=emit_metadata) for summary in self.example_summaries
            ],
        }
        if self.runtime is not None:
            runtime_payload = self.runtime.to_dict(emit_metadata=emit_metadata)
            if runtime_payload:
                payload["runtime"] = runtime_payload
        return payload
