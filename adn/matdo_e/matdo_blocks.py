from __future__ import annotations

from dataclasses import dataclass

from adn.matdo_e.matdo_external_memory import ExternalMemoryHandle
from adn.matdo_e.matdo_kv_quantization import KVQuantizationAdapter
from adn.matdo_e.matdo_query_adaptation import QueryAdaptationAdapter
from adn.matdo_e.matdo_scope_memory import ScopeMemory
from adn.matdo_e.runtime_materialize import MaterializedPolicy


@dataclass(frozen=True)
class RuntimeHandles:
    """Prepared adapter handles used by the minimal MATDO model surface."""

    policy: MaterializedPolicy | None
    kv_quantization: KVQuantizationAdapter
    query_adaptation: QueryAdaptationAdapter | None
    external_memory: ExternalMemoryHandle
    scope_memory: ScopeMemory
