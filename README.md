# Adaptive Deep Networks (ADN)

[![Validation](https://github.com/sunnyang1/Adaptive-Deep-Networks/workflows/Validation/badge.svg)](https://github.com/sunnyang1/Adaptive-Deep-Networks/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

**Adaptive Deep Networks: Integrating Attention Residuals, Dynamic Computation, and Test-Time Adaptation**

This repository provides a complete implementation and validation framework for the ADN paper, featuring:
- **Block Attention Residuals (AttnRes)** - Prevents representation burial in deep transformers
- **Query-only Test-Time Training (qTTT)** - Dynamic adaptation with frozen KV cache
- **TurboQuant** - 6x model compression with zero accuracy loss
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
bash scripts/lambda_setup.sh
```

### Run Experiments

```bash
# List available experiments
python experiments/run_refactored.py --list

# Run a specific experiment
python experiments/run_refactored.py exp1_representation_burial

# Run with quick mode (reduced samples)
python experiments/run_refactored.py exp3_gradient_flow --quick

# Run with custom config
python experiments/run_refactored.py exp2_margin_analysis --config configs/experiments/exp2.yaml
```

### Training

```bash
# Unified training script (works on all platforms)
python scripts/train_refactored.py --model-size medium --epochs 3

# Multi-GPU distributed training
torchrun --nproc_per_node=4 scripts/train_refactored.py --model-size medium --distributed

# Streaming training (for limited disk space)
python scripts/train_streaming.py --model-size small --max-steps 10000
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
black --check src/ experiments/ scripts/
ruff check src/ experiments/ scripts/
mypy src/
```

## Documentation

- **[API Documentation](docs/api/README.md)** - Complete API reference
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System architecture and diagrams
- **[Migration Guide](experiments/MIGRATION_GUIDE.md)** - How to migrate experiments
- **[Refactoring Summary](experiments/REFACTORING_SUMMARY.md)** - Architecture refactoring details

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

### 🗜️ TurboQuant Compression
6x model compression with zero accuracy loss:
- **Two-Stage Pipeline**: PolarQuant + QJL
- **Tensor Core Ready**: INT4 quantization
- **5.7x KV Cache Reduction**: Enables 1M+ context lengths

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
│   ├── turboquant/              # TurboQuant compression
│   └── benchmarks/              # Evaluation benchmarks
│
├── experiments/                  # Experiment framework
│   ├── common/                   # Shared utilities (config, paths, logging)
│   ├── core/                     # Core paper experiments (exp1-6)
│   ├── validation/               # Paper table validation scripts
│   ├── real_model/              # Real model validation
│   └── runner/                  # Experiment execution framework
│
├── scripts/                      # Training and setup scripts
│   ├── common/                  # Shared training utilities
│   ├── train_refactored.py      # Unified training script
│   └── lambda_setup.sh          # Lambda AI setup
│
├── configs/                      # Configuration files
│   └── experiments/             # YAML configs for experiments
│
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   └── conftest.py              # Pytest fixtures
│
├── docs/                         # Documentation
│   ├── api/                     # API documentation
│   └── ARCHITECTURE.md          # Architecture diagrams
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

| Config | Params | Layers | Hidden | Blocks | qTTT Steps |
|--------|--------|--------|--------|--------|------------|
| Small  | 2.2B   | 32     | 2048   | 8      | 16         |
| Medium | 8.7B   | 32     | 4096   | 8      | 32         |
| Large  | 27B    | 64     | 5120   | 16     | 32         |

## Citation

```bibtex
@article{adn2025,
  title={Adaptive Deep Networks: Integrating Attention Residuals, 
         Dynamic Computation, and Test-Time Adaptation},
  year={2025}
}
```

## License

Apache License 2.0
