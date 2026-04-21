# Legacy MATDO experiments ↔ MATDO-new (`matdo_new`)

Two configuration objects share a name but **different roles**:

| | **Legacy** `experiments.matdo.common.config.MATDOConfig` | **Paper** `matdo_new.core.config.MATDOConfig` (in `MATDO-new/`) |
|---|----------------------------------------------------------|----------------------------------------------------------------|
| **Purpose** | US1–US6 orchestration, analytic helpers (`compute_M_min`, `MATDOESolver`), real-model knobs | Runtime policy + resource walls (`solve_policy`, discrete `R`, `M_min` closed form) |
| **Shape** | Mutable dataclass, full training/inference constants (`d_model`, `C_KV`, …) | Immutable dataclass, policy coefficients and normalized capacity |

## Bridge module

`experiments/matdo/paper_policy_bridge.py` provides:

- **`legacy_to_paper_config(legacy, …)`** — maps shared error/Engram fields and sets MATDO-new-specific defaults (`total_hbm_blocks=256`, `n_block=8`, …).
- **`solve_policy_from_legacy(legacy, rho_hbm, rho_dram, …)`** — runs `matdo_new.core.policy.solve_policy` and returns `(PolicyDecision, MaterializedPolicy)`.
- **`policy_payload_for_experiments(...)`** — JSON-safe dict for logging (used by the experiment driver).

### Semantic gaps (read before trusting numbers)

1. **Legacy `N_block`** (tokens per AttnRes block, used in `C_KV` formulas) is **not** the same as MATDO-new **`n_block`** (paper partition count in `hbm_kv_capacity`). The bridge does **not** auto-convert between them; override `n_block` / `total_hbm_blocks` when aligning to a specific model.
2. **Legacy `MATDOESolver`** (`matdo_e/solver.py`) and **`solve_policy`** use different search strategies; they should be **qualitatively** aligned, not necessarily identical on every `rho`.
3. **Install path:** ensure `MATDO-new` is importable (`pip install -e MATDO-new` from repo root, or rely on the bridge inserting `MATDO-new/` onto `sys.path`).

## Optional driver hook

`run_all_experiments.py` accepts:

```text
--paper-policy
--paper-rho-hbm FLOAT   (default 0.92)
--paper-rho-dram FLOAT  (default 0.30)
```

When set, the driver computes the paper policy once (for the current legacy `config`), prints it, stores it under `results["paper_policy_bridge"]` in the returned summary, and writes `paper_policy_bridge.json` under the run output directory.

### US4 real model (`us4_use_paper_runtime`)

When `experiments.matdo.common.config.MATDOConfig.us4_use_paper_runtime` is **True** (CLI: `--us4-paper-runtime` on `run_all_experiments.py` with `--use-real-model`), US4’s Needle evaluation calls `evaluate_needle_haystack(..., use_paper_runtime=True, rho_hbm=<trial ρ>)`. That path:

1. Runs `solve_policy_from_legacy` → `MaterializedPolicy`
2. Builds `AdaptiveTransformerRuntimeBackend` + `MATDOModel` around the loaded `AdaptiveTransformer`
3. Decodes with `matdo_new.runtime.generation.generate_tokens` (greedy), passing the materialized policy into prefill/decode

Engram is only activated if **both** the policy requests it **and** `model.config.use_engram` is true (weights present).
