# Adaptive Deep Networks (ADN)

[Validation](https://github.com/sunnyang1/Adaptive-Deep-Networks/actions)
[Python 3.9+](https://www.python.org/downloads/)
[License](LICENSE)

Adaptive Deep Networks is a research codebase for training and validating architectures that combine:

- `AttnRes` (Block Attention Residuals)
- `qTTT` (query-only test-time training)
- `RaBitQ` (KV-cache compression)
- dynamic gating / adaptive compute

The repository includes training scripts, benchmark/evaluation tooling, and real-model validation workflows used to reproduce paper-facing results.

## Quick Start (QASP)

`QASP/` is the primary path for the paper-aligned QASP workflow (generation, inference, experiments):

```bash
python3 QASP/scripts/run_generation.py
python3 QASP/scripts/run_inference.py
python3 QASP/scripts/run_experiments.py --quick
```

Artifacts from quick experiments are written to `results/qasp/quick/` by default.

## Legacy ADN (Deprecated)

The remaining README sections describe the legacy ADN workflow for compatibility and historical reference.

## Legacy ADN Quick Start

### 1) Install

```bash
git clone https://github.com/sunnyang1/Adaptive-Deep-Networks.git
cd Adaptive-Deep-Networks
pip install -e ".[dev]"
```

### 2) Train (canonical entrypoint)

All model sizes now use one entrypoint: `scripts/training/train_model.py`.

```bash
# Small
python3 scripts/training/train_model.py --model-size small --output-dir results/small

# Medium
python3 scripts/training/train_model.py --model-size medium --output-dir results/medium

# T4-friendly
python3 scripts/training/train_model.py --model-size t4 --output-dir results/t4 --paper-preset-t4

# Large (distributed)
torchrun --nproc_per_node=8 scripts/training/train_model.py --model-size large --output-dir results/large
```

**Resume training** (same `--output-dir` and model/data settings as the run you are continuing):

```bash
python3 scripts/training/train_model.py --model-size small --output-dir results/small --resume latest
# or: --resume results/small/checkpoints/checkpoint_epoch_1.pt
```

Checkpoints are written under `<output-dir>/checkpoints/` at the **end of each epoch** (full state: model, optimizer, LR scheduler, `global_step`, history). If `checkpoint_latest.pt` does not exist yet, `--resume latest` prints a warning and **starts from scratch** instead of failing. Use `checkpoint_best.pt` only for inference or evaluation—it is **weights-only** and cannot resume training.

Paper-aligned wrappers (strict alignment check):

```bash
make train-paper-small OUTPUT_DIR=results/small_paper
make train-paper-medium OUTPUT_DIR=results/medium_paper
make train-paper-large OUTPUT_DIR=results/large_paper
```

### 3) Run experiments and benchmarks

```bash
# List experiments
python3 experiments/run_experiments_unified.py --list

# Quick core experiments
python3 experiments/run_experiments_unified.py --category core --quick

# Paper metrics
python3 experiments/run_experiments_unified.py --category paper

# Benchmark suite
python3 scripts/evaluation/run_benchmarks.py --model-size medium --benchmarks all
```

### 4) AutoResearch loop (optional)

Use the lightweight autonomous loop under `experiments/autoresearch/` to run hypothesis -> edit -> quick eval -> score cycles in isolated git worktrees.

```bash
# One-time Cursor CLI auth
cursor-agent login

# Dry-run first (plans commands, no training execution)
python3 experiments/autoresearch/run_agent.py \
  --iterations 1 \
  --trial-cmd "python3 experiments/matdo/vllm_integration/latency_profiler.py" \
  --dry-run \
  --skip-clean-check

# Real run
python3 experiments/autoresearch/run_agent.py \
  --iterations 5 \
  --trial-cmd "python3 experiments/matdo/vllm_integration/latency_profiler.py" \
  --primary-metric p99_latency_ms \
  --primary-direction min \
  --constraint "p99_latency_ms<=1200"

# Dual objective (requires trial outputs with both throughput and P99 latency; see guide)
# python3 experiments/autoresearch/run_agent.py ... --objective dual --dual-w-throughput 1 --dual-w-latency 0.01
```

Each trial’s `score.json` includes `failure_reasons` when scoring fails (missing metrics, constraints, or non-zero agent/trial exit codes).

By default, `agent_driver.py` uses `cursor-agent` when `AUTORESEARCH_AGENT_CMD_TEMPLATE` is unset.

## Recommended Guides

- [A100 80G ADN Complete Guide](docs/A100_80G_ADN_COMPLETE_GUIDE.md)
- [MATDO-E A100 Beginner Guide](docs/guides/MATDO_E_A100_BEGINNER_GUIDE.md)
- [AutoResearch Guide](docs/guides/AUTORESEARCH_GUIDE.md)
- [Documentation Index](docs/README.md)

## MATDO-new Package

`MATDO-new/` is the new paper-aligned MATDO-E package surface. It is intentionally separate from the legacy MATDO orchestration under `experiments/matdo/`.

Current phase-2 behavior:

- `python3 -m matdo_new.apps.generate --dry-run` prints the fully resolved generation request
- `python3 -m matdo_new.apps.generate` runs policy solve plus the live runtime/backend path and prints generated token ids as JSON
- `python3 -m matdo_new.apps.run_experiments` runs the lightweight `needle` task and `critical-points` study entrypoints
- `python3 -m matdo_new.apps.run_experiments --output <path>` writes the normalized experiment payload to a JSON file

Use `MATDO-new/README.md` for the package-local layout, config defaults, and current non-goals.

## Project Layout

For the canonical, up-to-date layout and file placement rules, see:

- [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md)

High-level directories:

- `src/` core implementations (`attnres`, `qttt`, `gating`, `models`, `rabitq`)
- `scripts/` setup, training, evaluation, and utility scripts
- `experiments/` experiment runners and validation workflows
- `tests/` unit, integration, and e2e tests
- `docs/` guides, reports, and papers

## Common Commands

```bash
make install
make test
make lint
make quick
make full
make paper-metrics
```

Or directly:

```bash
pytest tests/ -v --tb=short --ignore=tests/legacy
black --check src/ experiments/ scripts/ tests/
ruff check src/ experiments/ scripts/ tests/
mypy src/
```

## Notes

- Use `python3` (not `python`) in this repo.
- `scripts/training/train_unified.py` and `scripts/training/train_refactored.py` are compatibility wrappers and dispatch to `train_model.py`.
- Submission manuscripts stay at repository root: `ADN_paper.md` and `matdo-e_paper.md`.

### Hugging Face downloads

Training loads tokenizers and (by default) streaming datasets from the Hugging Face Hub. If requests to `huggingface.co` fail or time out, point the client at a mirror (example for mainland China):

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

On fully offline machines you must rely on a populated Hub cache or provide local paths; see `src/models/tokenizer.py` and your dataset flags.

## License

Apache License 2.0