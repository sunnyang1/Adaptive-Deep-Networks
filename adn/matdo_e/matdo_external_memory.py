from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping

from adn.matdo_e.matdo_model_config import ExternalMemoryConfig


@dataclass(frozen=True)
class ExternalMemoryRecord:
    """Single external-memory entry."""

    key: str
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ExternalMemoryHandle:
    """Minimal bounded key-value store for MATDO memory hooks."""

    def __init__(self, config: ExternalMemoryConfig | None = None) -> None:
        self.config = config or ExternalMemoryConfig()
        if self.config.max_entries < 0:
            raise ValueError('max_entries must be non-negative')
        self._records: OrderedDict[str, ExternalMemoryRecord] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.max_entries > 0

    def put(self, key: str, value: Any, *, metadata: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        record = ExternalMemoryRecord(key=key, value=value, metadata=metadata or {})
        self._records.pop(key, None)
        self._records[key] = record
        while len(self._records) > self.config.max_entries:
            self._records.popitem(last=False)

    def get(self, key: str) -> ExternalMemoryRecord | None:
        return self._records.get(key)

    def snapshot(self) -> tuple[ExternalMemoryRecord, ...]:
        return tuple(self._records.values())

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)
