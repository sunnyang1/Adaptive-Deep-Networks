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

## Quick Start

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

## Recommended Guides

- [A100 80G ADN Complete Guide](docs/A100_80G_ADN_COMPLETE_GUIDE.md)
- [MATDO-E A100 Beginner Guide](docs/guides/MATDO_E_A100_BEGINNER_GUIDE.md)
- [Documentation Index](docs/README.md)

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

## License

Apache License 2.0