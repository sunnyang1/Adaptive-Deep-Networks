"""
Bridge between ``experiments.matdo.common.config.MATDOConfig`` (legacy US1–US6)
and ``matdo_new`` (MATDO-new paper policy: ``solve_policy`` / ``MaterializedPolicy``).

See ``MATDO_NEW_BRIDGE.md`` for field semantics and limitations.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from experiments.matdo.common.config import MATDOConfig as LegacyMATDOConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATDO_NEW_ROOT = _REPO_ROOT / "MATDO-new"


def _ensure_matdo_new_on_path() -> None:
    if _MATDO_NEW_ROOT.is_dir() and str(_MATDO_NEW_ROOT) not in sys.path:
        sys.path.insert(0, str(_MATDO_NEW_ROOT))


def legacy_to_paper_config(
    legacy: LegacyMATDOConfig,
    *,
    total_hbm_blocks: int | None = None,
    n_block: int | None = None,
    c_dram_entries: int | None = None,
    arbitrage_zone_rho: float = 0.93,
    critical_zone_rho: float = 0.98,
    dram_utilization_limit: float = 0.90,
) -> Any:
    """Map legacy experiment config to frozen :class:`matdo_new.core.config.MATDOConfig`.

    **Not mapped 1:1:** legacy ``N_block`` is *tokens per AttnRes block* for capacity
    formulas; MATDO-new ``n_block`` is the paper's :math:`N_{\\text{block}}` partition
    count. This bridge **does not** derive ``n_block`` from legacy ``N_block``; use
    ``n_block=`` override when calibrating to a specific architecture.

    **HBM scale:** ``total_hbm_blocks`` defaults to ``256`` (MATDO-new default). Tune
    together with ``n_block`` and ``c_unit_kv`` when aligning to a real deployment.
    """
    _ensure_matdo_new_on_path()
    from matdo_new.core.config import MATDOConfig as PaperMATDOConfig

    tb = int(total_hbm_blocks) if total_hbm_blocks is not None else 256
    nb = int(n_block) if n_block is not None else 8
    ce = int(c_dram_entries) if c_dram_entries is not None else int(legacy.E_max)

    return PaperMATDOConfig(
        quantization_bits=tuple(sorted(int(x) for x in legacy.R_options)),
        min_quantization_bits=int(legacy.R_min),
        scope_span=int(legacy.S),
        total_hbm_blocks=tb,
        min_scope_blocks=1,
        max_t_steps=int(legacy.T_max_hard),
        n_block=nb,
        c_unit_kv=1.0,
        c_dram_entries=ce,
        target_error=float(legacy.E_target),
        arbitrage_zone_rho=float(arbitrage_zone_rho),
        critical_zone_rho=float(critical_zone_rho),
        dram_utilization_limit=float(dram_utilization_limit),
        e_max=int(legacy.E_max),
        e0=float(legacy.E_0),
        zeta=float(legacy.zeta),
        eta=float(legacy.eta),
        alpha=float(legacy.alpha),
        beta=float(legacy.beta),
        gamma=float(legacy.gamma),
        delta=float(legacy.delta),
        epsilon=float(legacy.epsilon),
    )


def solve_policy_from_legacy(
    legacy: LegacyMATDOConfig,
    *,
    rho_hbm: float,
    rho_dram: float = 0.30,
    target_error: float | None = None,
    available_hbm_blocks: int | None = None,
    paper_config_overrides: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Run :func:`matdo_new.core.policy.solve_policy` using coefficients from ``legacy``.

    Returns:
        ``(PolicyDecision, MaterializedPolicy | None)`` — materialized is ``None`` only
        if the policy layer fails internally (should not happen for normal inputs).
    """
    _ensure_matdo_new_on_path()
    from matdo_new.core.policy import RuntimeObservation, solve_policy
    from matdo_new.runtime.materialize import MaterializedPolicy, materialize_policy

    paper = legacy_to_paper_config(legacy)
    if paper_config_overrides:
        paper = replace(paper, **paper_config_overrides)

    obs = RuntimeObservation(
        rho_hbm=float(rho_hbm),
        rho_dram=float(rho_dram),
        available_hbm_blocks=available_hbm_blocks,
        target_error=float(target_error) if target_error is not None else None,
    )
    decision = solve_policy(obs, config=paper)
    mat = materialize_policy(decision)
    return decision, mat


def _json_safe(x: Any) -> Any:
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def policy_payload_for_experiments(
    legacy: LegacyMATDOConfig,
    *,
    rho_hbm: float,
    rho_dram: float = 0.30,
    target_error: float | None = None,
    paper_config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable dict for ``experiment_summary.json`` / logging."""
    decision, materialized = solve_policy_from_legacy(
        legacy,
        rho_hbm=rho_hbm,
        rho_dram=rho_dram,
        target_error=target_error,
        paper_config_overrides=paper_config_overrides,
    )
    out: dict[str, Any] = {
        "rho_hbm": rho_hbm,
        "rho_dram": rho_dram,
        "policy_decision": _json_safe(asdict(decision)),
    }
    if materialized is not None:
        out["materialized_policy"] = _json_safe(asdict(materialized))
    return out


def dump_policy_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
