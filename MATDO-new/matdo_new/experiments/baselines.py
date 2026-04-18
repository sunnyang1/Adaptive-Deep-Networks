from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

ScalarMetric: TypeAlias = bool | int | float | str


@dataclass(frozen=True)
class ExperimentResult:
    """Minimal result record shared by task and study entrypoints."""

    name: str
    kind: str
    metrics: dict[str, ScalarMetric]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ExperimentResult.name must not be empty")
        if not self.kind:
            raise ValueError("ExperimentResult.kind must not be empty")
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "metadata", dict(self.metadata))
