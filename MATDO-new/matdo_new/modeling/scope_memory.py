from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScopeBlock:
    """One retained scope-memory block."""

    index: int
    value: Any


class ScopeMemory:
    """Tiny bounded store for the active MATDO scope blocks."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError('capacity must be non-negative')
        self.capacity = capacity
        self._blocks: deque[ScopeBlock] = deque(maxlen=capacity or None)
        self._next_index = 0

    def remember(self, value: Any, *, block_index: int | None = None) -> None:
        if self.capacity == 0:
            return
        if block_index is None:
            next_index = self._next_index
            self._next_index += 1
        else:
            next_index = int(block_index)
            self._next_index = max(self._next_index, next_index + 1)
        self._blocks.append(ScopeBlock(index=next_index, value=value))
        while len(self._blocks) > self.capacity:
            self._blocks.popleft()

    def latest(self) -> ScopeBlock | None:
        return self._blocks[-1] if self._blocks else None

    def blocks(self) -> tuple[ScopeBlock, ...]:
        return tuple(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._blocks)
