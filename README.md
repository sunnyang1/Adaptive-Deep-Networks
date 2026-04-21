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

This project is organized as a unified Python package `adn/` containing three main paper modules:

- **ADN Core**: `AttnRes`, `qTTT`, `RaBitQ`, `Engram`, `Gating`
- **QASP**: Quality-Aware Stiefel Projection
- **MATDO-E**: Unified Resource Model

## Quick Start

After installing the package (`pip install -e ".[dev]"`), use the CLI entry points:

```bash
# Training
adn-train --model-size small --output-dir results/small

# Evaluation
adn-eval --model-size medium --output-dir results/eval

# Generation
adn-generate --dry-run

# Benchmarking
adn-benchmark --model-size medium --benchmarks all

# Experiments
adn-experiment --list
adn-experiment --category core --quick
```

Legacy QASP scripts remain available under `QASP/`:

```bash
python3 QASP/scripts/run_generation.py
python3 QASP/scripts/run_inference.py
python3 QASP/scripts/run_experiments.py --quick
```

Artifacts from quick experiments are written to `results/qasp/quick/` by default.

## Install

```bash
git clone https://github.com/sunnyang1/Adaptive-Deep-Networks.git
cd Adaptive-Deep-Networks
pip install -e ".[dev]"
```

## Project Structure

### `adn/` - Unified Package

| Subpackage | Description |
|---|---|
| `adn.core` | Core configuration (`ModelConfig`, `ADNConfig`) and base utilities (`RMSNorm`) |
| `adn.attention` | `AttnRes` - Block Attention Residuals implementation |
| `adn.models` | Adaptive transformer and generator models |
| `adn.qttt` | qTTT (query-only test-time training) module |
| `adn.quantization` | `RaBitQ` KV-cache compression |
| `adn.memory` | `Engram` memory management |
| `adn.gating` | Dynamic gating / adaptive compute |
| `adn.qasp` | Quality-Aware Stiefel Projection (QASP) |
| `adn.matdo_e` | MATDO-E unified resource model |
| `adn.experiments` | Experiment runners |
| `adn.utils` | Shared utilities |

### Other Directories

- `src/` - Legacy core implementations (retained for reference; active code is in `adn/`)
- `QASP/` - QASP scripts and models (retained for reference)
- `MATDO-new/` - New paper-aligned MATDO-E package surface
- `scripts/` - Setup and training scripts
- `experiments/` - Experiment runners and validation workflows
- `tests/` - Unit, integration, and e2e tests
- `docs/` - Guides, reports, and papers
- `archive/` - Archived legacy code (see `archive/README_ARCHIVE.md`)

For the canonical, up-to-date layout and file placement rules, see [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md).

## Module Details

### ADN Core (`adn/`)

The core package provides:
- **Config** (`adn.core.config`): `ModelConfig`, `ADNConfig` for model and training configuration
- **Base** (`adn.core.base`): `RMSNorm` and other foundational utilities
- **Attention** (`adn.attention`): `AttnRes` block attention residuals
- **Models** (`adn.models`): Adaptive transformer and generator implementations
- **qTTT** (`adn.qttt`): Query-only test-time training
- **Quantization** (`adn.quantization`): `RaBitQ` KV-cache compression

### QASP (`QASP/`)

`QASP/` remains the primary path for the paper-aligned QASP workflow (generation, inference, experiments):

```bash
python3 QASP/scripts/run_generation.py
python3 QASP/scripts/run_inference.py
python3 QASP/scripts/run_experiments.py --quick
```

### MATDO-E (`MATDO-new/`)

`MATDO-new/` is the new paper-aligned MATDO-E package surface. It is intentionally separate from the legacy MATDO orchestration under `experiments/matdo/`.

Current phase-2 behavior:

- `python3 -m matdo_new.apps.generate --dry-run` prints the fully resolved generation request
- `python3 -m matdo_new.apps.generate` runs policy solve plus the live runtime/backend path and prints generated token ids as JSON
- `python3 -m matdo_new.apps.run_experiments` runs the lightweight `needle` task and `critical-points` study entrypoints
- `python3 -m matdo_new.apps.run_experiments --output <path>` writes the normalized experiment payload to a JSON file

Use `MATDO-new/README.md` for the package-local layout, config defaults, and current non-goals.

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
- Submission manuscripts stay at repository root: `ADN_paper.md` and `matdo-e_paper.md`.

### Hugging Face downloads

Training loads tokenizers and (by default) streaming datasets from the Hugging Face Hub. If requests to `huggingface.co` fail or time out, point the client at a mirror (example for mainland China):

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

On fully offline machines you must rely on a populated Hub cache or provide local paths; see `src/models/tokenizer.py` and your dataset flags.

## License

Apache License 2.0
