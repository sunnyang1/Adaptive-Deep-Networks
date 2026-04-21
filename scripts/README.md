# Scripts Directory

This directory contains all utility scripts organized by functionality.

## 📁 Directory Structure

### setup/
Environment setup and installation scripts.

| Script | Description |
|--------|-------------|
| `autodl_setup.sh` | AutoDL platform setup |
| `autodl_h20_setup.sh` | AutoDL H20 GPU setup |
| `lambda_setup.sh` | Lambda Cloud setup |
| `quick_start_h20.sh` | Quick start for H20 |

### model/
Model building, parameter calculation, and analysis scripts.

| Script | Description |
|--------|-------------|
| `build_and_benchmark_small.py` | Build and benchmark Small Model (2.2B) |
| `build_large_model.py` | Build Large Model configuration |
| `calculate_params.py` | Calculate model parameters (detailed) |
| `calculate_params_v2.py` | Calculate parameters (paper-matching) |
| `calculate_training_time.py` | Estimate training time |

### training/
Training scripts for different configurations.

| Script | Description |
|--------|-------------|
| `train_h20.py` | Training for H20 GPU |
| `train_model.py` | **Canonical unified training entrypoint** (`--model-size small|medium|large|t4`) |
| `train_refactored.py` | Deprecated compatibility wrapper (dispatches to `train_model.py`) |
| `train_streaming.py` | Streaming training for large datasets |
| `train_unified.py` | Deprecated compatibility wrapper (dispatches to `train_model.py`) |

### evaluation/
Benchmark and evaluation scripts.

| Script | Description |
|--------|-------------|
| `run_benchmarks.py` | Run all benchmarks |
| `eval_5_2.py` | Section 5.2 evaluation |
| `run_medium_model_eval.sh` | Medium model evaluation |
| `run_real_validation.sh` | Real validation run |
| `validate_models.py` | Model validation |

### experiments/
Paper experiments and validation scripts.

| Script | Description |
|--------|-------------|
| `run_small_experiments.py` | Run Small Model experiments |
| `run_small_model_experiments_fast.py` | Fast experiments (CPU-friendly) |
| `run_small_model_paper_experiments.py` | Full paper experiments |
| `paper_metrics_summary.py` | Generate paper metrics summary |
| `test_small_model_datasets.py` | Dataset testing |
| `test_turboquant_small.py` | TurboQuant testing |
| `validate_turboquant_setup.py` | TurboQuant setup validation |

### data/
Data processing, validation, and download scripts.

| Script | Description |
|--------|-------------|
| `dataset_info.py` | Dataset information |
| `dataset_validation.py` | Validate datasets |
| `check_datasets.sh` | Check dataset availability |
| `download_datasets.sh` | Download datasets |
| `download_zero_scrolls.sh` | Download ZeroSCROLLS |
| `validate_datasets.py` | Dataset validation |
| `validate_hf_datasets.py` | HuggingFace datasets validation |

### colab/
Google Colab specific scripts.

| Script | Description |
|--------|-------------|
| `test_colab.py` | Basic Colab test |
| `test_colab_complete.py` | Complete Colab test suite |

### common/
Shared utilities used by multiple scripts.

---

## Usage

Most scripts can be run directly:

```bash
# Setup
bash scripts/setup/lambda_setup.sh

# Training (recommended)
python3 scripts/training/train_model.py --model-size small --output-dir results/small

# Build model
python3 scripts/model/build_and_benchmark_small.py

# Run experiments
python3 scripts/experiments/run_small_model_experiments_fast.py

# Evaluate
python3 scripts/evaluation/run_benchmarks.py --model-size medium
```

## See Also

- [Main README](../README.md) - Project overview
- [Documentation Index](../docs/README.md) - All documentation
