# Adaptive Deep Networks: Validation Framework

[![Validation](https://github.com/sunnyang1/Adaptive-Deep-Networks-Integrating-Attention-Residuals-Dynamic-Computation-and-Test-Time-Adaptation/workflows/Validation/badge.svg)](https://github.com/sunnyang1/Adaptive-Deep-Networks-Integrating-Attention-Residuals-Dynamic-Computation-and-Test-Time-Adaptation/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Validation framework for the paper "Adaptive Deep Networks: Integrating Attention Residuals, Dynamic Computation, and Test-Time Adaptation".

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
| MATH (7B params) | 52.3% | 5.4.2 |
| Compute Efficiency | 40% reduction | 5.4.3 |
| AttnRes Overhead | <2% | 5.6.3 |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sunnyang1/Adaptive-Deep-Networks-Integrating-Attention-Residuals-Dynamic-Computation-and-Test-Time-Adaptation.git
cd Adaptive-Deep-Networks

# Setup on Lambda AI
bash scripts/lambda_setup.sh

# Or local installation
pip install -r requirements.txt
```

### Run Validation

```bash
# Run all benchmarks
python scripts/run_benchmarks.py --model-size medium --benchmarks all

# Run specific benchmark
python scripts/run_benchmarks.py --benchmarks flop

# Skip model tests (FLOP analysis only)
python scripts/run_benchmarks.py --skip-model-tests
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Specific module
pytest tests/unit/test_attnres.py -v
```

## Project Structure

```
.
├── src/
│   ├── attnres/          # Block AttnRes implementation
│   ├── gating/           # Dynamic gating
│   ├── qttt/             # Query-only TTT
│   ├── models/           # Model definitions
│   └── benchmarks/       # Evaluation benchmarks
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── benchmarks/       # Benchmark tests
├── scripts/
│   ├── lambda_setup.sh   # Lambda AI setup
│   └── run_benchmarks.py # Main runner
├── tasks/
│   ├── product-brief.md
│   └── prd-*.md
└── prd.json              # Task tracking
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
| Small  | 1.5B   | 32     | 2048   | 8      | 16         |
| Medium | 7B     | 32     | 4096   | 8      | 32         |
| Large  | 50B    | 64     | 5120   | 16     | 32         |

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
