# Adaptive Deep Networks (ADN)

[![Validation](https://github.com/sunnyang1/Adaptive-Deep-Networks/workflows/Validation/badge.svg)](https://github.com/sunnyang1/Adaptive-Deep-Networks/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

**Adaptive Deep Networks: Integrating Attention Residuals, Dynamic Computation, and Test-Time Adaptation**

This repository provides a complete implementation and validation framework for the ADN paper, featuring:
- **Block Attention Residuals (AttnRes)** - Prevents representation burial in deep transformers
- **Query-only Test-Time Training (qTTT)** - Dynamic adaptation with frozen KV cache
- **RaBitQ** - 5x KV cache compression with minimal quality loss
- **Dynamic Gating** - Allocates computation based on input difficulty

## Overview

This repository provides a complete validation framework for reproducing the paper's key results on Lambda AI infrastructure.

### Key Components

1. **Block Attention Residuals (AttnRes)**: Replaces fixed residual connections with learned attention over block-level representations
2. **Dynamic Computation Gating**: Allocates inference budget based on input difficulty
3. **Query-only Test-Time Training (qTTT)**: Performs targeted adaptation while keeping KV cache frozen

## Target Results

| Benchmark | Target | Paper Section |
|-----------|--------|---------------|
| Needle-in-Haystack (256K) | 86.9% avg | 5.4.1 |
| MATH (8.7B params) | 52.3% | 5.4.2 |
| Compute Efficiency | 40% reduction | 5.4.3 |
| AttnRes Overhead | <2% | 5.6.3 |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sunnyang1/Adaptive-Deep-Networks.git
cd Adaptive-Deep-Networks

# Setup environment
pip install -e ".[dev]"

# Or setup on Lambda AI
bash scripts/setup/lambda_setup.sh
```

### Run Experiments

```bash
# List available experiments
python experiments/run_experiments_unified.py --list

# Run a specific experiment
python experiments/run_experiments_unified.py --exp exp1_representation_burial

# Run with quick mode (reduced samples)
python experiments/run_experiments_unified.py --category core --quick

# Run paper metrics validation
python experiments/run_experiments_unified.py --category paper
```

### Training

```bash
# Small Model (1.1B params) - CPU-friendly, 1 GPU
python scripts/training/train_small.py --output-dir results/small --epochs 3

# Medium Model (5.7B params) - Requires 1-4 GPUs
python scripts/training/train_medium.py --output-dir results/medium --epochs 3

# Large Model (23B params) - Requires 8+ GPUs with distributed training
torchrun --nproc_per_node=8 scripts/training/train_large.py --output-dir results/large

# Multi-GPU distributed training (Medium/Large)
torchrun --nproc_per_node=4 scripts/training/train_medium.py --output-dir results/medium --distributed

# With DeepSpeed ZeRO-3 (recommended for Large model)
deepspeed --num_gpus=8 scripts/training/train_large.py \
    --output-dir results/large \
    --deepspeed configs/ds_config_h20.json

# Streaming training (for limited disk space)
python scripts/training/train_streaming.py --model-size small --max-steps 10000

# Build and validate Small Model
python scripts/model/build_and_benchmark_small.py

# Run Small Model experiments (150M experimental or 1.1B full)
python scripts/model/run_small_model_experiments.py --experimental
python scripts/model/run_small_model_experiments.py --full  # Requires GPU
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html

# Specific module
pytest tests/unit/test_attnres.py -v
pytest tests/unit/test_turboquant.py -v

# Run linting
black --check src/ experiments/ scripts/ tests/
ruff check src/ experiments/ scripts/ tests/
mypy src/
```

### TurboQuant V3 Usage

```python
from turboquant import create_k4_v2

# Create compressor (4-bit keys, 2-bit values)
tq = create_k4_v2(head_dim=64)

# Fit on representative samples
tq.fit(sample_keys, sample_values)

# Compress and decompress
compressed = tq.compress(keys, values)
keys_dq, values_dq = tq.decompress(compressed)

# Use with HuggingFace
cache = tq.as_cache(residual_window=128)
model.generate(..., past_key_values=cache)
```

**Recommended Configurations:**
- `create_k4_v2()`: 4-bit keys, 2-bit values (~4.9x compression) ⭐ Best quality
- `create_k3_v2()`: 3-bit keys, 2-bit values (~3.0x compression)
- `create_k2_v2()`: 2-bit keys, 2-bit values (~7.1x compression, max memory)

## Documentation

- **[Documentation Index](docs/README.md)** - All documentation organized by category
- **[Papers](docs/papers/)** - Research papers (V1 and TurboQuant versions)
- **[Validation Reports](docs/reports/validation/)** - Validation results and analysis
- **[Implementation Reports](docs/reports/implementation/)** - Refactoring and implementation details
- **[Guides](docs/guides/)** - How-to guides and build instructions

## Key Features

### 🧠 Block Attention Residuals (AttnRes)
Replaces fixed residual connections with learned attention over block representations:
- **Memory Efficient**: O(Nd) instead of O(Ld)
- **Prevents Representation Burial**: Maintains gradient flow in deep networks
- **Minimal Overhead**: <2% computation increase

### ⚡ Query-Only Test-Time Training (qTTT)
Dynamic adaptation while keeping KV cache frozen:
- **Fast Adaptation**: Only 0.5% of parameters updated
- **Frozen KV Cache**: No memory overhead
- **Margin Maximization**: Explicit logit margin optimization

### 🗜️ TurboQuant V3 Compression
~5x KV cache compression with minimal quality loss:
- **MSE-Only Pipeline**: No QJL (hurts attention quality)
- **Per-Vector Normalization**: Handles varying magnitudes
- **Fast Walsh-Hadamard Transform**: O(n log n) random rotation
- **Asymmetric K/V Bits**: 4-bit keys, 2-bit values for optimal quality/compression

### 🎛️ Dynamic Gating
Allocates computation based on input difficulty:
- **Reconstruction Loss Signal**: Difficulty proxy
- **EMA/Target-Rate Calibration**: Adaptive threshold
- **Depth-Priority Policy**: TurboQuant-aware allocation

## Project Structure

```
Adaptive-Deep-Networks/
├── src/                          # Core implementation
│   ├── attnres/                  # Block Attention Residuals
│   ├── qttt/                     # Query-Only Test-Time Training
│   ├── gating/                   # Dynamic gating controller
│   ├── models/                   # Model definitions
│   ├── rabitq/                  # RaBitQ compression
│   │   ├── api.py               # Main API
│   │   ├── compressor.py        # MSE compression
│   │   ├── quantizer.py         # Lloyd-Max quantization
│   │   ├── rotation.py          # FWHT random rotation
│   │   ├── cache.py             # HF-compatible cache
│   └── benchmarks/              # Evaluation benchmarks
│
├── experiments/                  # Experiment framework
│   ├── common/                   # Shared utilities (config, paths, logging)
│   ├── core/                     # Core paper experiments (exp1-6)
│   ├── validation/               # Paper table validation scripts
│   ├── real_model/              # Real model validation
│   └── runner/                  # Experiment execution framework
│
├── scripts/                      # Utility scripts (organized by function)
│   ├── setup/                   # Environment setup
│   ├── model/                   # Model building and analysis
│   ├── training/                # Training scripts
│   │   └── common/              # Shared training utilities
│   ├── evaluation/              # Benchmarks and evaluation
│   ├── experiments/             # Paper experiments
│   ├── data/                    # Data processing
│   ├── colab/                   # Google Colab scripts
│   └── README.md                # Script documentation
│
├── configs/                      # Configuration files
│   └── experiments/             # YAML configs for experiments
│
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── legacy/                  # Deprecated test files
│   ├── test_turboquant_v3_refactored.py  # V3 comprehensive tests
│   └── conftest.py              # Pytest fixtures
│
├── docs/                         # Documentation (organized by category)
│   ├── papers/                  # Research papers
│   ├── reports/                 # Validation and implementation reports
│   ├── guides/                  # How-to guides
│   ├── project/                 # Project management
│   └── README.md                # Documentation index
│
└── pyproject.toml               # Project configuration
```

## Core Algorithms

### Block AttnRes Forward Pass

```python
# Phase 1: Inter-block attention
V = stack(blocks + [partial_block])  # [N+1, B, T, D]
K = RMSNorm(V)
logits = w_attn · K                  # [N+1, B, T]
α = softmax(logits, dim=0)
h_attn = Σᵢ αᵢ · Vᵢ                 # [B, T, D]

# Phase 2: Standard transformer
attn_out = Attention(LayerNorm(h_attn))
partial_block = partial_block + attn_out
```

### qTTT Adaptation

```python
w_adapted = w_l.clone().requires_grad_(True)

for step in range(num_steps):
    attn_out = compute_attention(queries, kv_cache, w_adapted)
    logits = project_to_vocab(attn_out)
    
    # Margin maximization
    margin_loss = -logsigmoid(target_logits - max_distractor).mean()
    
    # Update only query
    grad = autograd.grad(margin_loss, w_adapted)[0]
    w_adapted = w_adapted - learning_rate * grad
```

## Model Configurations

| Config | Params | Layers | Hidden | Heads | Blocks | qTTT Steps |
|--------|--------|--------|--------|-------|--------|------------|
| Small  | 1.1B   | 32     | 1408   | 8     | 8      | 16         |
| Medium | 5.7B   | 56     | 2496   | 16    | 8      | 32         |
| Large  | 23.0B  | 88     | 4032   | 18    | 11     | 32         |

> **Architecture optimized for AttnRes** (Paper §5.4.1): d_model/L_b ≈ 45 and H/L_b ≈ 0.3

## Citation

```bibtex
@article{adn2025,
  title={Adaptive Deep Networks: Integrating Attention Residuals, 
         Dynamic Computation, and Test-Time Adaptation},
  year={2025}
}
```

## Makefile Targets

```bash
make install          # Install dependencies
make test             # Run all tests
make lint             # Run linting
make quick            # Quick validation
make full             # Full experiment suite
make paper-metrics    # Generate paper metrics
```

## License

Apache License 2.0
