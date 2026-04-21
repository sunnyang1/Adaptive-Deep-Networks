MATDO-new is the paper-aligned package for the new MATDO-E runtime.

This package is the source of truth for the package-local MATDO-new surface. It owns:

- policy solve for `R / M / T / E`
- true incremental generation
- lightweight paper-facing task and study wiring
- package-scoped default configs and CLI entry modules

## Layout

- `matdo_new/core/`: policy config, constraints, error model, and online estimation
- `matdo_new/runtime/`: prompt prefill, decode, state tracking, and generation loop helpers
- `matdo_new/modeling/`: adapter-oriented modeling surfaces for quantization, memory, and attention
- `matdo_new/experiments/`: minimal task/study primitives and result normalization
- `matdo_new/apps/`: lightweight CLI entry modules for generation and experiments
- `configs/`: package-local default YAMLs for shared, inference, and experiment settings
- `tests/`: focused package tests

## Default Configs

The package-local defaults live under:

- `MATDO-new/configs/default.yaml`
- `MATDO-new/configs/inference/default.yaml`
- `MATDO-new/configs/experiments/default.yaml`

These files intentionally stay small. They document the default runtime-policy knobs, a baseline generation request shape, and the current task/study experiment defaults exposed by the package CLI entry modules.

## CLI Entry Modules

After installing the repo package, the package-local entry modules can be run directly:

```bash
python3 -m matdo_new.apps.generate --dry-run
python3 -m matdo_new.apps.generate
python3 -m matdo_new.apps.run_experiments
python3 -m matdo_new.apps.run_experiments --output tmp/matdo_new_experiments.json
```

`matdo_new.apps.generate` supports two modes:

- `--dry-run`: print the fully resolved generation request after config loading and CLI overrides
- live mode: resolve the MATDO policy, build a real runtime backend bridge, run generation, and emit generated token ids as JSON

`matdo_new.apps.run_experiments` runs the current lightweight `needle` task and `critical-points` study entrypoints, prints normalized JSON results, and can optionally write the same payload to a file with `--output`.

## Current Outputs

The current package CLI flows are still intentionally lightweight, but they now execute real package paths:

- `matdo_new.apps.generate --dry-run` writes the resolved request summary to standard output as JSON.
- `matdo_new.apps.generate` writes a live generation payload to standard output as JSON, including resolved config paths, runtime observation, policy decision, generated token ids, and final sequence length.
- `matdo_new.apps.run_experiments` writes normalized experiment results to standard output as JSON.
- `matdo_new.apps.run_experiments --output <path>` writes the same normalized JSON payload to the requested file.
- The experiments config currently supports `tasks.*.enabled`, `studies.*.enabled`, `output.emit_metadata`, and JSON-only `output.format`.
- Neither CLI currently creates package-managed result directories, plots, checkpoints, or benchmark bundles under `MATDO-new/`.

Examples:

```bash
python3 -m matdo_new.apps.generate --dry-run > tmp/matdo_new_generate_request.json
python3 -m matdo_new.apps.generate > tmp/matdo_new_generate_live.json
python3 -m matdo_new.apps.run_experiments --output tmp/matdo_new_experiments.json
```

That behavior is deliberate for this phase: the package entry modules now exercise real policy/runtime/result flows, while larger artifact management remains outside the package.

## Non-Goals and Incompatibilities

`MATDO-new/` does not aim to be a drop-in replacement for the legacy MATDO stack under `experiments/matdo/`.

Current non-goals and incompatibilities include:

- no compatibility promise for legacy `experiments/matdo/` CLIs, flags, or output directory layouts
- no attempt to mirror the legacy experiment suite, validation coverage, or real-model orchestration surface
- no direct reuse of legacy `run_all_experiments.py` behavior, simulation wrappers, or benchmark packaging conventions
- no package-local training pipeline, checkpoint manager, or benchmark artifact writer at this stage

When you need the established repo-level MATDO experiment workflows, continue using `experiments/matdo/`. When you need the new paper-aligned package surface for policy solve, incremental runtime wiring, and lightweight package-scoped task/study entrypoints, use `MATDO-new/`.

## Scope

MATDO-new is intentionally narrow in this phase. It provides the paper-aligned policy/runtime/experiment package surface without replacing the repository's broader training and benchmark orchestration.
