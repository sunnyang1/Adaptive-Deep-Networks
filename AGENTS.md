# Adaptive Deep Networks Validation Framework

## Project Overview

This is the validation framework for the paper "Adaptive Deep Networks: Integrating Attention Residuals, Dynamic Computation, and Test-Time Adaptation".

## Architecture

### Core Components

1. **Block AttnRes** (`src/attnres/`)
   - Block Attention Residuals implementation
   - Pseudo-query management
   - Two-phase computation strategy

2. **Dynamic Gating** (`src/gating/`)
   - Reconstruction loss computation
   - Threshold calibration (EMA, target-rate)

3. **qTTT** (`src/qttt/`)
   - Query-only test-time training
   - Margin maximization loss
   - KV cache management

### Model Configurations

| Size | Params | Layers | Hidden | Blocks |
|------|--------|--------|--------|--------|
| Small | 1.5B | 32 | 2048 | 8 |
| Medium | 7B | 32 | 4096 | 8 |
| Large | 50B | 64 | 5120 | 16 |

## Key Design Decisions

### AttnRes

- **Zero Initialization**: Pseudo-queries initialize to zero for training stability
- **Block Structure**: Reduces memory from O(Ld) to O(Nd)
- **Two-Phase**: Phase 1 (parallel inter-block) + Phase 2 (sequential intra-block)

### Gating

- **Signal**: Reconstruction loss as difficulty proxy
- **Calibration**: EMA or target-rate threshold
- **Target**: Maintain ~30% adaptation rate

### qTTT

- **Frozen KV**: Keys and values from prefill never change
- **Query-only**: Only query parameters updated
- **Margin Loss**: Explicit logit margin maximization

## Testing

Run all tests:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

Run specific module:
```bash
pytest tests/unit/test_attnres.py -v
```

## Validation Benchmarks

1. **Needle-in-Haystack**: Long-context retrieval (target: 86.9% avg)
2. **MATH**: Mathematical reasoning (target: 52.3% @ 7B)
3. **FLOP Analysis**: Verify T_think ≈ 2 * N_qTTT * k
4. **Ablation Study**: Component contribution analysis

## Lambda AI Deployment

Setup:
```bash
bash scripts/lambda_setup.sh
```

Run validation:
```bash
python scripts/run_benchmarks.py --model-size medium --benchmarks all
```

## Code Style

- Format: `black .`
- Lint: `flake8 .`
- Type check: `mypy src/`

## References

- Paper: Adaptive Deep Networks Final Draft
- Reference Code: Attention Residuals Technical Report (Chen et al., 2026)
