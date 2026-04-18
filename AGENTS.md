# Adaptive Deep Networks Validation Framework

## Project Overview

This is the validation framework for the paper "Adaptive Deep Networks: Integrating Attention Residuals, Dynamic Computation, and Test-Time Adaptation".

## Repository layout

For the canonical directory map (root layout, `docs/`, `experiments/`, `src/`, `scripts/`, `tasks/`, tests, and where papers and guides live), see [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md).

**MATDO-E paper-aligned runtime:** the installable package `matdo_new` and its tests live under [`MATDO-new/`](MATDO-new/README.md) (`pip install -e MATDO-new`). It bridges to `src/` for `AdaptiveTransformer` at runtime.

**Submission manuscripts (keep at repo root):** `ADN_paper.md` and `matdo-e_paper.md` are the publication-target papers; when reorganizing folders, do **not** move them into `docs/` or elsewhere.

## Architecture

### Core Components

1. **Block AttnRes** (`src/attnres/`)
  - Block Attention Residuals implementation
  - Pseudo-query management
  - Two-phase computation strategy
2. **Dynamic Gating** (`src/gating/`)
  - Reconstruction loss computation
  - Threshold calibration (EMA, target-rate)
3. **qTTT** (`src/qttt/`)
  - Query-only test-time training
  - Margin maximization loss
  - KV cache management

### Model Configurations


| Size   | Params | Layers | Hidden | Heads | Blocks | d_model/L_b | H/L_b |
| ------ | ------ | ------ | ------ | ----- | ------ | ----------- | ----- |
| Small  | 1.1B   | 32     | 1408   | 8     | 8      | 44.0        | 0.25  |
| Medium | 5.7B   | 56     | 2496   | 16    | 8      | 44.6        | 0.29  |
| Large  | 23.0B  | 88     | 4032   | 18    | 11     | 45.8        | 0.21  |


> **Architecture Optimized for AttnRes (§5.4.1)**: Paper shows AttnRes shifts optimal `d_model/L_b` from ~60 (baseline) to ~45 (AttnRes), favoring deeper, narrower networks. Head ratio `H/L_b ≈ 0.3` is optimal for both.

## Key Design Decisions

### AttnRes

- **Zero Initialization**: Pseudo-queries initialize to zero for training stability (§5.3)
- **Block Structure**: Reduces memory from O(Ld) to O(Nd)
- **Two-Phase**: Phase 1 (parallel inter-block) + Phase 2 (sequential intra-block)
- **Optimal Block Count**: N≈8 recovers most FullAttnRes benefit (§5.3 Fig 6)
- **RMSNorm on Keys**: Critical for performance (without: +0.006/+0.004 loss)
- **Single-Head Depth Attention**: Multi-head hurts performance (1.752 vs 1.746)
- **Softmax over Sigmoid**: Competitive normalization forces sharper selection

### Gating

- **Signal**: Reconstruction loss as difficulty proxy
- **Calibration**: EMA or target-rate threshold
- **Ponder Gate**: Conditional qTTT triggering based on uncertainty
  - Uses entropy + max probability heuristics
  - Modes: 'strict', 'balanced', 'lenient'
  - Integrated into `generate(use_qttt='adaptive')`
- **Adaptive qTTT Config**: Dynamic adjustment of steps and LR
  - Scales num_steps based on sequence length
  - Adjusts learning rate based on gradient magnitude
  - Modes: 'fast', 'balanced', 'quality'

### Incremental KV Cache (NEW)

- **IncrementalState**: Explicit state management for generation
  - KV cache tracking per layer
  - Block representation management
  - Memory usage statistics
- **IncrementalGenerator**: High-level API for efficient generation
  - prefill() for O(T×L) initialization
  - step() for O(L) per-token generation (API ready)
  - generate() with performance stats
- **Target**: Maintain ~30% adaptation rate

### QASP (`QASP/`)

- **Canonical semantics (Path A):** Value-weighted AttnRes and block statistics in `QASP_paper.tex` are defined on a **single full-sequence forward** over a fixed context. Use `QASPTransformer.forward` or `prefill` for that definition.
- **Incremental `step`:** With `use_attnres=True`, block summaries use a **growing prefix**; logits need not match `forward` on the extended sequence. Documented in `QASP/__init__.py`, `QASP/models/qasp_transformer.py`, and `tests/integration/test_qasp_prefill_step_numeric_parity.py`.

### qTTT

- **Frozen KV**: Keys and values from prefill never change
- **Query-only**: Only query parameters updated
- **Margin Loss**: Explicit logit margin maximization

## Testing

Run all tests:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

Run specific module:

```bash
pytest tests/unit/test_attnres.py -v
```

## MATDO Real Model Testing

Run MATDO experiments (US1–US6) with actual `AdaptiveTransformer` model:

```bash
# Full suite with random initialization (small = 1.1B params)
python experiments/matdo/run_all_experiments.py --use-real-model --size small

# With pretrained weights
python experiments/matdo/run_all_experiments.py \
    --use-real-model \
    --checkpoint checkpoints/adb_medium.pt

# Quick validation (US4–US6 only)
python experiments/matdo/run_all_experiments.py \
    --use-real-model \
    --skip-us1 --skip-us2 --skip-us3 \
    --size small --device mps  # Apple Silicon
```

**Key flags:**

- `--use-real-model`: Enable real model mode (default: simulation)
- `--size {small,medium,large}`: Model configuration
- `--checkpoint PATH`: Load pretrained weights
- `--device {cuda,mps,cpu}`: Compute device

For CPU-friendly US4/US6 tuning (`--us4-num-trials`, `--us4-no-qttt`, `--rls-ctx-lengths`) and the shared `MATDOConfig` fields, see [experiments/matdo/README.md](experiments/matdo/README.md).

**Implementation notes:**

- US1–US3 default to simulation due to O(100) model evaluations each
- US4 MATDO uses real model; SnapKV/H2O are simulated baselines
- US5 uses `forward(use_attnres=..., use_qttt=...)` for component ablation
- US6 collects real model errors on sparse (R,M,T) grid for RLS

## Validation Benchmarks

1. **Needle-in-Haystack**: Long-context retrieval (target: 86.9% avg)
2. **MATH**: Mathematical reasoning (target: 52.3% @ 8.7B)
3. **FLOP Analysis**: Verify T_think ≈ 2 * N_qTTT * k
4. **Ablation Study**: Component contribution analysis

## Lambda AI Deployment

Setup:

```bash
bash scripts/setup/lambda_setup.sh
```

Run validation:

```bash
python scripts/evaluation/run_benchmarks.py --model-size medium --benchmarks all
```

Run experiments:

```bash
python experiments/run_experiments_unified.py --category paper
```

## Code Style

- Format: `black .`
- Lint: `flake8 .`
- Type check: `mypy src/`

## References

- Paper: Adaptive Deep Networks Final Draft
- Reference Code: Attention Residuals Technical Report (Chen et al., 2026)

## Cursor Cloud specific instructions

### Environment

- Python 3.12 (system default). No virtual environment needed; packages install to `~/.local`.
- `~/.local/bin` must be on `PATH` (already persisted in `~/.bashrc`).
- No Docker, databases, or external services required. This is a pure Python/PyTorch research codebase.

### Quick reference

Commands follow the `Makefile` and `README.md`. Key commands:


| Task                    | Command                                                                           |
| ----------------------- | --------------------------------------------------------------------------------- |
| Install deps            | `pip install -e ".[dev]"` then `pip install tqdm matplotlib seaborn pandas scipy` |
| Unit tests              | `pytest tests/unit/ -v --tb=short`                                                |
| All tests (skip legacy) | `pytest tests/ -v --tb=short --ignore=tests/legacy`                               |
| Lint (black)            | `black --check src/ experiments/ scripts/ tests/`                                 |
| Lint (ruff)             | `ruff check src/ experiments/ scripts/ tests/`                                    |
| Type check              | `mypy src/`                                                                       |
| List experiments        | `python3 experiments/run_experiments_unified.py --list`                           |
| Quick experiments       | `python3 experiments/run_experiments_unified.py --category core --quick`          |


### Known issues (pre-existing)

- `tests/legacy/` contains deprecated tests that fail on import; always pass `--ignore=tests/legacy` to pytest.
- `mypy` errors with `python_version = "3.8"` in `pyproject.toml` on Python 3.12; mypy requires ≥3.9 target. The error `Source file found twice under different module names` is also pre-existing.
- 11 unit tests fail due to pre-existing code/test mismatches (gating `DynamicThreshold` API changes, `TwoPhaseBlockAttnRes.norm` missing, `compute_reconstruction_loss` shape bug). These are not environment issues.
- The unified experiment runner passes `--output-dir` (hyphenated) but individual experiment scripts expect `--output_dir` (underscored), causing exit code 2. Running experiments individually with `--output_dir` works.
- Use `python3` not `python` (the latter is not linked by default).

