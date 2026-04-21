# PRD: Experiment Framework Refactoring & New Experiments

## Overview

Refactor the existing experiment framework to align with the Adaptive Deep Networks Experimental Plan (`Adaptive_Deep_Networks_Experimental_Plan.md`), fix remaining TurboQuant→RaBitQ references in experiment code, and develop missing experiments (Table 3, Table 5, Table 8, RaBitQ verification).

## Goals

1. Remove all remaining TurboQuant references in `experiments/` (active code only, not legacy)
2. Re-align experiment numbering with paper tables (Table 1-8)
3. Implement 4 missing experiments: Table 3 (Inference Benchmark), Table 5 (Logit Margin Distribution), Table 8 (Accuracy-Compute Pareto), RaBitQ Compression Verification
4. Register all experiments in `run_experiments_unified.py`
5. All existing + new experiments run successfully with `--quick` mode

## Refactoring Goals

### RG-1: Clean TurboQuant References in experiments/
- `experiments/__init__.py` mentions "TurboQuant" in docstring
- `experiments/turboquant/` → rename to `experiments/rabitq/`
- `experiments/docs/TURBOQUANT_EXPERIMENTS.md` → rename
- `experiments/validation/table7_synergy.py` mentions "w/o TurboQuant" in header
- `experiments/real_model/needle_haystack_real.py` mentions "ADB-TurboQuant"
- `experiments/real_model/datasets/needle_dataset.py` mentions "ADB-TurboQuant"
- `run_experiments_unified.py`: `val_turboquant` entry references legacy scripts

### RG-2: Fix Experiment-Table Mapping
Current `exp1-exp6` don't map cleanly to paper Table 1-8:
- `exp1` (Representation Burial) → Table 1 ✅
- `exp2` (Logit Margin) → Table 5 (currently mislabeled)
- `exp3` (Gradient Flow) → Table 2 (currently mislabeled)
- `exp4` (FLOP Equivalence) → Part of Table 8
- `exp5` (Component Synergy) → Table 7
- `exp6` (Auxiliary) → Supporting experiments

Action: Keep `exp1-exp6` numbering as-is (too much churn to rename), but add clear paper table references in descriptions. New experiments get `table_*` IDs.

## Functional Requirements

### FR-1: Table 5 - Logit Margin Distribution Experiment
Create `experiments/core/table5_margin_distribution/` with:
- `experiment.py`: Class-based implementation
- `run_table5.py`: Script entry point
- `config.yaml`: Configuration
- Extracts attention logit margins before/after qTTT adaptation
- Tests at context lengths [4K, 16K, 64K, 128K, 256K]
- Reports: mean margin before/after, Δ margin, % achieving theoretical minimum
- Piggybacks on needle-in-haystack data generation (reuse from exp2)

### FR-2: Table 3 - Inference Benchmark Experiment
Create `experiments/core/table3_inference_benchmark/` with:
- `experiment.py`: Class-based implementation
- `run_table3.py`: Script entry point
- `config.yaml`: Configuration
- Measures: tokens/sec, KV cache memory, p99 latency, max qTTT steps
- Tests at context lengths [4K, 16K, 64K, 128K] for batch_size=1
- Compares: Standard inference vs ADB+RaBitQ (depth-priority)
- Uses `torch.profiler` for FLOP measurement

### FR-3: Table 8 - Accuracy-Compute Pareto Experiment
Create `experiments/core/table8_pareto/` with:
- `experiment.py`: Class-based implementation
- `run_table8.py`: Script entry point
- `config.yaml`: Configuration with 5 configs (standard, static AttnRes, uniform qTTT, gated qTTT, oracle)
- Measures FLOPs via `torch.profiler` and accuracy
- Generates Pareto frontier plot

### FR-4: RaBitQ Compression Verification
Create `experiments/rabitq/` (renamed from `experiments/turboquant/`):
- `run_compression_verification.py`: Verifies 4.9× compression ratio
- `run_microbenchmarks.py`: Profiles RaBitQ decompression, AttnRes, polar conversion
- `config.yaml`: Configuration for compression targets

### FR-5: Update run_experiments_unified.py
- Replace `val_turboquant` with `val_rabitq` pointing to new RaBitQ validation script
- Add `table3`, `table5`, `table8`, `val_rabitq_compression` entries
- Update descriptions with paper table references

## Non-Goals

- Training real models on H100 cluster (Phase 3 of experimental plan)
- Implementing CUDA kernels for RaBitQ (infrastructure work)
- Modifying `src/` source code (already done in prior TurboQuant→RaBitQ migration)
- Updating legacy directories (`scripts/legacy/`, `experiments/turboquant/`, etc.)

## Technical Considerations

- Python 3.8.19 environment (no walrus operator, no match statement)
- PyTorch Profiler for FLOP measurement (available in PyTorch 2.0+)
- Existing experiment infrastructure: `common/`, `runner/`, `utils/measurement.py`
- Each experiment follows dual implementation pattern: `experiment.py` (class) + `run_*.py` (script)
- Scripts under `experiments/core/` use relative imports via `sys.path.insert`
- All experiments must support `--quick` and `--device` flags

## Success Metrics

1. `grep -ri "turboquant" experiments/ --exclude-dir=legacy --exclude-dir=turboquant` returns empty
2. All existing exp1-exp6 pass with `--quick` mode
3. New table3, table5, table8 experiments run with `--quick` mode and produce valid output
4. `python experiments/run_experiments_unified.py --list` shows all experiments
5. RaBitQ compression verification produces ratio within 10% of target (4.9×)
