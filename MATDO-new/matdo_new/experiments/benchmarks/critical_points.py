from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CriticalPointsBenchmark:
    points: tuple[float, ...]
    name: str = "critical-points"
    kind: str = "study"

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("CriticalPointsBenchmark.points must not be empty")
        object.__setattr__(self, "points", tuple(float(point) for point in self.points))

    @property
    def config(self) -> dict[str, object]:
        return {"points": self.points}

    @classmethod
    def build(cls, *, points: Sequence[float], name: str = "critical-points") -> "CriticalPointsBenchmark":
        return cls(points=tuple(float(point) for point in points), name=name)
