from __future__ import annotations

from dataclasses import dataclass

from matdo_new.modeling.external_memory import ExternalMemoryHandle
from matdo_new.modeling.kv_quantization import KVQuantizationAdapter
from matdo_new.modeling.query_adaptation import QueryAdaptationAdapter
from matdo_new.modeling.scope_memory import ScopeMemory
from matdo_new.runtime.materialize import MaterializedPolicy


@dataclass(frozen=True)
class RuntimeHandles:
    """Prepared adapter handles used by the minimal MATDO model surface."""

    policy: MaterializedPolicy | None
    kv_quantization: KVQuantizationAdapter
    query_adaptation: QueryAdaptationAdapter | None
    external_memory: ExternalMemoryHandle
    scope_memory: ScopeMemory
