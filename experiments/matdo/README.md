# MATDO experiments (`experiments/matdo/`)

Orchestration for the legacy MATDO user stories **US1–US6** (singularity, dual hierarchy, λ₂, SOTA comparison, ablation, online RLS). See [MATDO_E_IMPLEMENTATION.md](MATDO_E_IMPLEMENTATION.md) for module layout.

## Entry points

| Script | Purpose |
|--------|---------|
| `run_all_experiments.py` | Full US1–US6 driver |
| `run_smoke.py` | CPU-safe sanity: one real `AdaptiveTransformer` probe + simulation US4–US6 |

```bash
# Help (includes US4/US6 knobs)
python3 experiments/matdo/run_all_experiments.py --help

# Quick US4–US6, real model, small weights (GPU recommended)
python3 experiments/matdo/run_all_experiments.py \
  --use-real-model --skip-us1 --skip-us2 --skip-us3 \
  --size small --device cuda

# CPU: shorten US4 trials and disable per-token qTTT in US4
python3 experiments/matdo/run_all_experiments.py \
  --use-real-model --skip-us1 --skip-us2 --skip-us3 \
  --size small --device cpu \
  --us4-num-trials 1 --us4-no-qttt

# CPU: US6 RLS real-model grid with short prompts (overrides M×N_block)
python3 experiments/matdo/run_all_experiments.py \
  --use-real-model --skip-us1 --skip-us2 --skip-us3 --skip-us4 --skip-us5 \
  --size small --device cpu \
  --rls-ctx-lengths 128,256,512
```

## CLI flags (US4 / US6)

These map onto `experiments.matdo.common.config.config` (`MATDOConfig`):

| CLI | Config field | Effect |
|-----|----------------|--------|
| `--us4-num-trials N` | `us4_num_trials` | US4 `run_sota_comparison` trial count. If unset in config, the driver uses **10**. |
| `--us4-no-qttt` | `us4_enable_qttt=False` | US4 loads the real model with qTTT disabled (`load_matdo_model(..., enable_qttt=False)`). Much faster on CPU. |
| `--rls-ctx-lengths L1,L2,...` | `rls_ctx_lengths_override` | US6 `simulate_online_queries` uses `ctx_len = override[t % len]` instead of `min(M * N_block, max_seq_len)`. |

You can also set the same fields in Python before calling `run_all_matdo_experiments(...)`:

```python
from pathlib import Path
from experiments.matdo.common.config import config
from experiments.matdo.run_all_experiments import run_all_matdo_experiments

config.us4_num_trials = 2
config.us4_enable_qttt = False
config.rls_ctx_lengths_override = (128, 256)

run_all_matdo_experiments(
    skip_us1=True, skip_us2=True, skip_us3=True,
    output_dir=Path("experiments/matdo/results/dev"),
    use_real_model=True,
    model_size="small",
    device="cpu",
)
```

Or pass keyword-only arguments (same names as `MATDOConfig` where applicable):

```python
run_all_matdo_experiments(
    ...,
    us4_num_trials=2,
    us4_enable_qttt=False,
    rls_ctx_lengths_override=(128, 256),
)
```

## Related config (real model workload)

| Field | Default (typical) | Notes |
|-------|-------------------|--------|
| `real_model_context_lengths` | `(4096, 16384, 65536)` | Needle task context sizes for US4/US5-style evals |
| `real_model_num_samples` | `5` | Samples per length |

For laptop smoke runs, `run_smoke.py` sets `(128,)` and `num_samples=1`.

## MATDO-new paper policy bridge

Legacy US1–US6 use `experiments.matdo.common.config.MATDOConfig`. The paper-aligned package **`matdo_new`** lives under [`MATDO-new/`](../../MATDO-new/README.md). To compare knobs without duplicating heuristics by hand, see **[MATDO_NEW_BRIDGE.md](MATDO_NEW_BRIDGE.md)** and `paper_policy_bridge.py`. Optional CLI: `--paper-policy`, `--paper-rho-hbm`, `--paper-rho-dram` on `run_all_experiments.py`. For US4 real Needle runs through MATDO-new’s backend, add `--us4-paper-runtime` (with `--use-real-model`); see `MATDO_NEW_BRIDGE.md`.

## US5 ablation cache

`ablation/run_ablation.py` keeps **one** loaded `AdaptiveTransformer` per `(model_size, checkpoint, device)` and toggles `enable_rabitq` / `enable_attnres` / `enable_qttt` on the shared config. Do not reinstantiate one model per flag combination.
