from __future__ import annotations

from dataclasses import dataclass

from matdo_new.experiments.benchmarks.critical_points import CriticalPointsBenchmark
from matdo_new.experiments.schema import EvaluatedBenchmark, ResultSummary


@dataclass(frozen=True)
class CriticalPointsToyEvaluator:
    name: str = "critical-points-toy"

    def evaluate(self, benchmark: CriticalPointsBenchmark) -> EvaluatedBenchmark:
        slice_summaries = tuple(
            ResultSummary(
                summary_id=f"critical-point:{point:g}",
                metrics={
                    "critical_point": point,
                    "score": round(1.0 - point, 3),
                },
                metadata={"critical_point": point},
            )
            for point in benchmark.points
        )
        scores = [float(summary.metrics["score"]) for summary in slice_summaries]
        return EvaluatedBenchmark(
            aggregate_metrics={
                "num_points": len(slice_summaries),
                "mean_score": round(sum(scores) / len(scores), 3),
                "min_score": min(scores),
                "max_score": max(scores),
            },
            slice_summaries=slice_summaries,
        )
