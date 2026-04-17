# QASP Big-Bang Migration Design

## Context

The paper `QASP_paper.tex` introduces a new end-to-end formulation that replaces legacy ADN adaptation assumptions. Existing code paths in `src/` and `experiments/` no longer match the paper-level algorithmic contract.  
This design defines a one-pass, big-bang migration to a new `QASP/` primary implementation for model, generation, inference, and experiments, with README updated to make QASP the default user path.

## Goal

Create a new `QASP/` top-level implementation that:

- implements paper-aligned QASP adaptation (matrix query optimization + Stiefel projection),
- integrates value-weighted AttnRes and value-weighted Engram in forward/inference paths,
- provides runnable generation, inference, and experiment entry points,
- makes QASP the default documented workflow in README.

## Non-Goals

- No incremental coexistence strategy as the primary plan (this is intentionally big-bang).
- No broad refactor of unrelated legacy modules outside what is required for entrypoint switch and migration wiring.
- No claim that full paper-scale 8.7B training is reproduced in this pass; this migration focuses on software architecture and runnable experimental pipeline.

## Selected Strategy

Single-step big-bang cutover:

- Build all new core functionality under `QASP/`.
- Redirect or deprecate old primary entrypoints to QASP-backed implementations.
- Update README so all main commands use `QASP/scripts/*`.

This minimizes long-term dual-maintenance cost at the expense of a larger one-time change.

## System Architecture

### Top-Level Structure

`QASP/` becomes the new primary package boundary:

- `QASP/__init__.py`
  - public factories and high-level runners.
- `QASP/configs/`
  - model, QASP, and experiment configuration objects.
- `QASP/models/`
  - QASP transformer and layer implementation.
- `QASP/adaptation/`
  - quality score, Stiefel projection, matrix adaptation loop, ponder gate.
- `QASP/inference/`
  - generation API, incremental state, KV cache interfaces.
- `QASP/experiments/`
  - unified QASP experiment runner + benchmarks + ablations + efficiency.
- `QASP/scripts/`
  - CLI entrypoints for generation, inference, experiments.

### Component Responsibilities

#### `QASP/models/`

- `qasp_transformer.py`
  - full model graph, embedding/layers/output projection, canonical `forward`.
- `qasp_layer.py`
  - single-layer attention + MLP + hooks for AttnRes/Engram integration.
- `value_weighted_attnres.py`
  - computes block-level weights with paper equation:
    - block quality average `rho_bar_m`,
    - quality-weighted logits and softmax aggregation.
- `value_weighted_engram.py`
  - stores memory tuple `(m, rho_mem)`,
  - retrieval fusion with `alpha * sigmoid(rho_mem) * m`.
- `components.py`
  - shared building blocks (norm, feed-forward, attention helpers).

#### `QASP/adaptation/`

- `quality_score.py`
  - computes token-level `rho(t)` and batch-level `rho_bar`.
- `stiefel.py`
  - Newton-Schulz based matrix sign projection `msign(W)` onto Stiefel manifold.
- `matrix_qasp.py`
  - five-step update loop:
    1. Euclidean gradient,
    2. value-weighted scaling by `rho_bar`,
    3. Euclidean update,
    4. Stiefel projection,
    5. iterative refinement for configured `N_iter`.
- `ponder_gate.py`
  - entropy/confidence decision to trigger adaptation.

#### `QASP/inference/`

- `generator.py`
  - token generation API (`generate`) with optional adaptive trigger.
- `incremental.py`
  - prefill + step incremental inference lifecycle and state transitions.
- `kv_cache.py`
  - common KV interfaces used by forward and incremental generation.

#### `QASP/experiments/`

- `runner.py`
  - unified command dispatcher for benchmarks, ablations, and efficiency tests.
- `benchmarks/needle.py`, `benchmarks/math_eval.py`
  - task-level evaluations.
- `ablations/qasp_ablation.py`
  - full QASP vs component-removal studies.
- `efficiency/profile.py`
  - throughput, memory, latency reporting and summary artifacts.

## Data Flow

### Forward Pass

1. `input_ids` -> embeddings.
2. Through each QASP layer:
  - attention/MLP backbone,
  - value-weighted AttnRes for block aggregation,
  - value-weighted Engram fusion for memory retrieval.
3. Final norm/projection -> logits.
4. Optional instrumentation output (quality statistics and orthogonality diagnostics) for experiment logging.

### Generation / Inference Loop

1. Prefill stage initializes sequence state and caches.
2. At each decode step:
  - run ponder gate on previous-step logits,
  - if triggered, run matrix QASP update (`matrix_qasp.py`),
  - continue forward with updated query matrix,
  - sample next token (temperature/top-k supported).
3. Persist trigger-rate and adaptation stats for analysis.

### Experiment Flow

`QASP/experiments/runner.py` orchestrates:

- benchmark runs,
- ablation runs,
- efficiency profiling,
- standardized outputs:
  - `metrics.json`,
  - `config_snapshot.json`,
  - `logs.txt`.

## Numerical Stability and Error Handling

- Protect all normalization denominators with epsilon.
- Guard Newton-Schulz iteration with convergence/fallback checks.
- Handle degenerate quality-score denominator safely with deterministic fallback behavior.
- Validate tensor shape compatibility in adaptation interfaces (especially `d x k` matrix updates).

## Testing Strategy

### Unit Tests

- `quality_score` correctness and bounds checks.
- `stiefel` projection invariants (orthogonality tolerance).
- `ponder_gate` trigger behavior under entropy/confidence scenarios.

### Integration Tests

- generation path with adaptation on/off.
- incremental inference prefill+step lifecycle.
- value-weighted AttnRes and Engram invoked in live forward pass.

### Smoke Tests

- `QASP/scripts/run_experiments.py --quick` full pipeline execution.
- output artifact presence and schema sanity checks.

## Migration/Cutover Plan

1. Add complete `QASP/` package and scripts.
2. Switch legacy primary entrypoints to invoke QASP equivalents (or explicit deprecation wrappers).
3. Keep legacy internal modules only as historical compatibility paths, not as primary runtime defaults.
4. Make README point all primary usage commands to QASP scripts.

## README Update Plan

README will be updated with:

- `Quick Start (QASP)` as first-class default.
- command examples for:
  - generation,
  - inference,
  - experiments.
- `Legacy ADN (Deprecated)` section with minimal historical guidance.
- `QASP vs ADN` concise differences:
  - vector qTTT -> matrix Stiefel adaptation,
  - standard AttnRes -> value-weighted AttnRes,
  - standard Engram -> value-weighted Engram.
- reproducibility guidance (seed, quick/full modes, output locations).

## Done Criteria

Migration is complete when all are true:

1. `QASP/scripts/run_generation.py` runs successfully.
2. `QASP/scripts/run_inference.py` runs successfully.
3. `QASP/scripts/run_experiments.py --quick` runs end-to-end and writes expected artifacts.
4. Matrix Stiefel adaptation path executes without numerical failure.
5. Value-weighted AttnRes and Engram are active in runtime path (not dead code).
6. README default commands use QASP entrypoints.
7. Legacy primary entrypoints are switched/deprecated in favor of QASP.

## Risks and Mitigations

- **Risk: regression due to broad one-shot cutover.**  
**Mitigation:** enforce quick smoke suite + unit/integration checks before declaring cutover complete.
- **Risk: numerical instability in Newton-Schulz projection.**  
**Mitigation:** bounded iteration count, normalization safeguards, fallback path and tests for ill-conditioned inputs.
- **Risk: mismatched experiment outputs after runner swap.**  
**Mitigation:** normalized output schema (`metrics.json`, `config_snapshot.json`, `logs.txt`) and compatibility checks in smoke tests.