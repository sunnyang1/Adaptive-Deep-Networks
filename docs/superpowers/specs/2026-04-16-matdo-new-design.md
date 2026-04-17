# MATDO-new Design

## Objective

Create a new top-level `MATDO-new/` implementation that reconstructs the MATDO-E system around the updated paper `matdo-e_paper.md`, covering:

- generation
- inference
- experiments
- documentation

The new package should follow the paper's `R / M / T / E` abstraction directly instead of extending the legacy `experiments/matdo/` and `src/models/adaptive_transformer.py` execution path.

## Decision Summary

The user selected a `full rewrite` direction with `reuse_modules_only` compatibility:

- `MATDO-new/` becomes a new execution stack
- old Python APIs do not need to remain compatible
- old experiment scripts do not need to remain compatible
- old checkpoints do not need to remain compatible
- existing repository modules may be reused as implementation ingredients or algorithm references

This design therefore prioritizes paper alignment, clear boundaries, and a correct runtime architecture over backward compatibility.

## Design Goals

1. Make the paper's four knobs `R`, `M`, `T`, and `E` first-class runtime concepts.
2. Replace the legacy full-sequence recomputation generation path with true stateful incremental decoding.
3. Separate policy, runtime, modeling, and experiment responsibilities so that theory changes do not require rewriting the generation kernel.
4. Reuse proven algorithmic components from the repo where they remain conceptually aligned with the updated paper.
5. Provide one coherent entrypoint for MATDO-new inference and one coherent entrypoint for MATDO-new experiments.
6. Update documentation so the new package is understandable without reading legacy MATDO code.

## Explicit Non-Goals

This design does not aim to:

- preserve old `experiments/matdo/` CLI compatibility
- preserve old `AdaptiveTransformer.generate()` behavior
- preserve checkpoint compatibility with the legacy model stack
- keep legacy user-story organization such as US1-US6
- reframe the old simulation-heavy MATDO code as the new source of truth

## Architectural Positioning

`MATDO-new/` should be treated as a new canonical subsystem inside the repository rather than a wrapper around the old MATDO implementation.

The central architectural rule is:

- the paper's knobs map to code objects directly

That means:

- `R` is not just a boolean compression flag
- `M` is not an accidental byproduct of the old block layout
- `T` is not an optional side action hidden inside generation
- `E` is not split across conflicting runtime and experiment-only implementations

## Top-Level Package Layout

`MATDO-new/` should start with the following structure:

```text
MATDO-new/
├── README.md
├── __init__.py
├── configs/
│   ├── default.yaml
│   ├── inference/
│   └── experiments/
├── core/
│   ├── config.py
│   ├── constraints.py
│   ├── error_model.py
│   ├── policy.py
│   ├── scheduler.py
│   └── online_estimation.py
├── modeling/
│   ├── config.py
│   ├── attention.py
│   ├── blocks.py
│   ├── matdo_model.py
│   ├── kv_quantization.py
│   ├── scope_memory.py
│   ├── external_memory.py
│   └── query_adaptation.py
├── runtime/
│   ├── state.py
│   ├── prefill.py
│   ├── decode.py
│   ├── materialize.py
│   ├── generation.py
│   └── metrics.py
├── experiments/
│   ├── run_experiments.py
│   ├── baselines.py
│   ├── tasks/
│   ├── studies/
│   └── reports/
├── data/
├── utils/
└── tests/
```

## Responsibility Boundaries

### `core/`

Owns the paper-facing resource model and decision logic:

- configuration for `R / M / T / E`
- hardware and resource constraints
- error model
- policy solve
- scheduling
- online parameter estimation

This layer must not perform tensor-heavy decoding or contain experiment protocol logic.

### `modeling/`

Owns the neural components and their MATDO-new composition:

- attention behavior
- block and scope representation logic
- KV quantization path
- external memory fusion
- query adaptation hooks
- the assembled `MATDOModel`

This layer defines what the model can do, but not when the policy chooses to do it.

### `runtime/`

Owns actual execution:

- prefill
- incremental state
- step-wise decode
- turning policy outputs into runnable runtime settings
- runtime metrics collection

This is the layer that replaces the current legacy generation path that still recomputes the full sequence on each decoding step.

### `experiments/`

Owns paper evaluation protocols:

- task runners
- study runners
- baselines
- result output and summaries

This layer must consume public MATDO-new entrypoints rather than reaching into model internals ad hoc.

## Key Runtime Objects

The design standardizes four core objects:

### `MATDOModel`

The neural execution object. It is responsible for model-level computation only.

Responsibilities:

- embed inputs
- compute attention and block interactions
- apply quantized or full-precision KV access
- incorporate external memory
- support query adaptation hooks

It must not own experiment orchestration or global policy search.

### `MATDOState`

The single source of truth for runtime state.

Responsibilities:

- per-layer KV state
- block summaries and scope state
- external memory state needed at runtime
- sequence position and decode bookkeeping
- resource accounting snapshots needed by the policy layer

All decode steps must read from and write to `MATDOState` rather than reconstructing state from the full token history.

### `MATDOPolicy`

The runtime decision object that expresses the paper's knob choices as executable configuration.

Responsibilities:

- solve current `R / M / T / E`
- expose the chosen strategy in a structured form
- indicate whether adaptation or external memory should be active
- communicate limits to the runtime layer

### `MATDOExperiment`

The evaluation object for tasks and studies.

Responsibilities:

- configure workloads
- run baselines and MATDO-new variants
- collect metrics
- persist outputs

It must not directly manage low-level tensors.

## End-To-End Data Flow

One MATDO-new generation should follow this pipeline:

1. `Prefill`
  The prompt is embedded and processed into an initial `MATDOState`.
2. `Policy Solve`
  The current resource pressure, targets, and online estimates are used to compute an executable `MATDOConfig(R, M, T, E)`.
3. `Runtime Materialization`
  The runtime layer maps the policy to concrete behavior:
  - `R` selects KV quantization behavior
  - `M` selects HBM-resident scope retention
  - `T` selects query adaptation budget
  - `E` selects external memory budget and use
4. `Incremental Decode`
  One token step is decoded from the prior state rather than by re-running the entire prompt.
5. `Observe And Update`
  Runtime metrics and error observations update online estimates for subsequent policy solves.

This loop is the core replacement for the legacy generation approach.

## Legacy Reuse Strategy

The reuse rule is:

- reuse algorithms and local implementations
- do not inherit legacy package boundaries as MATDO-new boundaries

### Reuse as algorithm or implementation sources

These areas are valid sources for adaptation or direct local reuse:

- `src/attnres/`
- `src/rabitq/`
- `src/qttt/`
- `src/engram/`

Expected mapping:

- `src/attnres/` informs `modeling/blocks.py` and `modeling/scope_memory.py`
- `src/rabitq/` informs `modeling/kv_quantization.py`
- `src/qttt/` informs `modeling/query_adaptation.py`
- `src/engram/` informs `modeling/external_memory.py`

### Do not treat as the new foundation

These legacy paths should not be used as the MATDO-new architectural skeleton:

- `src/models/adaptive_transformer.py`
- `src/models/incremental_generator.py`
- `src/models/incremental_kv_cache.py`
- `experiments/matdo/`

Reasons:

- legacy `generate()` still performs full-sequence recomputation
- `M` is entangled with the old architecture instead of being a clean scope knob
- `T` is embedded as an auxiliary generation behavior instead of a formal policy output
- `E` currently has mixed runtime-side and experiment-side meanings

## Generation And Inference Requirements

The new runtime must satisfy the following requirements:

1. Support explicit prefill and incremental decode as separate phases.
2. Keep a persistent runtime state object instead of rebuilding from token history.
3. Allow policy-controlled activation of quantization, scope restriction, adaptation, and external memory.
4. Make it possible to measure the operational effect of `R / M / T / E` at runtime.
5. Expose a user-facing generation entrypoint that is distinct from the legacy `AdaptiveTransformer.generate()`.

## Experiment Reorganization

`MATDO-new/experiments/` should be organized around paper questions, not legacy user stories.

### `tasks/`

For real inference tasks such as:

- needle-in-haystack
- long-context QA
- generation latency and throughput probes

### `studies/`

For paper studies such as:

- dual critical points
- quadratic blow-up near the context wall
- heterogeneous arbitrage inequality
- coupling sensitivity
- cross-architecture validation

### `baselines.py`

One shared baseline registry for:

- static baseline
- MATDO 3D
- MATDO-E 4D
- any representative comparison systems that remain part of the paper-facing evaluation contract

### `run_experiments.py`

One canonical experiment entrypoint that selects:

- model or architecture
- tasks or studies
- budgets and targets
- output directory
- result summary generation

## Validation Strategy

Validation should be delivered in three layers.

### 1. Unit Validation

Validate:

- policy solve behavior
- resource constraint calculations
- state updates
- KV quantization boundaries
- query adaptation scheduling
- external memory edge cases

### 2. Runtime Validation

Validate:

- prefill and incremental decode separation
- state consistency across decode steps
- absence of legacy-style full-sequence recomputation in the MATDO-new generation path
- correctness of policy materialization into runtime choices

### 3. Paper Validation

The first paper-facing target is trend reproduction rather than exact-number replication.

Initial success signals:

- both critical points can be measured
- the near-wall `T* ~ (rho_ctx - rho)^(-2)` trend is observable
- enabling `E` shifts the effective context wall
- latency and quality trade-offs are measurable under the new runtime

## Documentation Plan

Documentation should be updated in two places.

### Root `README.md`

Add a focused MATDO-new section that explains:

- what MATDO-new is
- why it differs from legacy `experiments/matdo/`
- minimal inference command
- minimal experiment command

### `MATDO-new/README.md`

This becomes the detailed source of truth for:

- architecture overview
- config layout
- generation and inference commands
- experiment commands
- output locations
- incompatibilities with legacy MATDO code

## Risks

1. The rewrite may accidentally reintroduce old couplings by wrapping legacy model entrypoints instead of defining new ones.
2. The first runtime iteration may still contain hidden full-sequence recomputation if state boundaries are not enforced strictly.
3. The theoretical `M` knob may not map cleanly to old AttnRes assumptions without an explicit MATDO-new scope memory abstraction.
4. External memory may drift into two implementations again unless one runtime meaning is chosen and enforced early.
5. Experiment migration may quietly preserve old narrative structure unless the new task-versus-study split is applied consistently.

## Success Criteria

The design is successful if:

- `MATDO-new/` is a standalone top-level package
- generation, inference, and experiments live on a new execution path
- `R / M / T / E` are first-class objects in code and runtime behavior
- legacy full-sequence decode is not the MATDO-new generation kernel
- experiments are organized around paper questions
- root and package documentation explain the new system clearly

## Implementation Phases

1. Create `MATDO-new/` package skeleton and core configuration objects.
2. Implement `MATDOState`, prefill, and incremental decode path.
3. Implement MATDO policy solve and runtime materialization of `R / M / T / E`.
4. Rebuild generation and inference entrypoints on top of the new runtime.
5. Rebuild experiments around tasks and studies.
6. Add focused tests for policy, runtime, and paper-facing trends.
7. Update root `README.md` and write `MATDO-new/README.md`.

