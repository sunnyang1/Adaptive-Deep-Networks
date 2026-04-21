from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

from matdo_new.experiments.baselines import ExperimentResult, ScalarMetric

CriticalPoint: TypeAlias = int | float
CriticalPointEvaluator: TypeAlias = Callable[
    [CriticalPoint], ExperimentResult | Mapping[str, ScalarMetric]
]


def _make_study_result(
    *,
    point: CriticalPoint,
    name: str,
    metrics: Mapping[str, ScalarMetric],
    metadata: Mapping[str, object] | None = None,
) -> ExperimentResult:
    normalized_metrics = dict(metrics)
    normalized_metrics.setdefault("critical_point", point)

    normalized_metadata = {} if metadata is None else dict(metadata)
    normalized_metadata.setdefault("critical_point", point)
    normalized_metadata.setdefault("study", name)

    return ExperimentResult(
        name=f"{name}:{point:g}",
        kind="study",
        metrics=normalized_metrics,
        metadata=normalized_metadata,
    )


def run_critical_points_study(
    points: Sequence[CriticalPoint],
    *,
    evaluate: CriticalPointEvaluator,
    name: str = "critical-points",
) -> tuple[ExperimentResult, ...]:
    """Evaluate a small set of named critical points and normalize their outputs."""
    results: list[ExperimentResult] = []
    for point in points:
        outcome = evaluate(point)
        if isinstance(outcome, ExperimentResult):
            metadata = dict(outcome.metadata)
            metadata.setdefault("evaluator_name", outcome.name)
            metadata.setdefault("evaluator_kind", outcome.kind)
            results.append(
                _make_study_result(
                    point=point,
                    name=name,
                    metrics=outcome.metrics,
                    metadata=metadata,
                )
            )
            continue

        results.append(_make_study_result(point=point, name=name, metrics=outcome))
    return tuple(results)
