# TurboQuant V3

Clean, modular implementation of TurboQuant for extreme KV cache compression.

## Overview

TurboQuant V3 compresses key-value caches in transformer models using:
- **MSE-only compression** (no QJL - hurts attention quality)
- **Per-vector normalization** (handles varying magnitudes)
- **Fast Walsh-Hadamard Transform** (O(n log n) random rotation)
- **Lloyd-Max quantization** (optimal scalar quantization)

## Installation

```bash
pip install torch numpy
```

## Quick Start

```python
from turboquant import create_k4_v2

# Create compressor (4-bit keys, 2-bit values)
tq = create_k4_v2(head_dim=64)

# Fit on representative samples
tq.fit(sample_keys, sample_values)

# Compress
compressed = tq.compress(keys, values)

# Decompress
keys_dq, values_dq = tq.decompress(compressed)
```

## Configuration Presets

| Function | Key Bits | Value Bits | Compression | Quality |
|----------|----------|------------|-------------|---------|
| `create_k4_v2()` | 4 | 2 | ~4.9x | ⭐ Best |
| `create_k3_v2()` | 3 | 2 | ~3.0x | Good |
| `create_k2_v2()` | 2 | 2 | ~7.1x | Fair |

## HuggingFace Integration

```python
# Create cache with residual window
cache = tq.as_cache(residual_window=128)

# Use in generation
model.generate(
    input_ids,
    past_key_values=cache,
    use_cache=True
)
```

## Architecture

```
turboquant/
├── api.py        # Main API (TurboQuantV3, factory functions)
├── compressor.py # MSE compression with normalization
├── quantizer.py  # Lloyd-Max optimal quantization
├── rotation.py   # FWHT random rotation
├── cache.py      # HF-compatible compressed cache
└── legacy/       # Deprecated implementations
```

## Performance

| Config | Key Error | Attention CosSim | Top-1 Match |
|--------|-----------|------------------|-------------|
| K4/V2 | ~9-10% | >0.99 | >90% |
| K3/V2 | ~18-20% | >0.98 | >85% |
| K2/V2 | ~30-35% | >0.95 | >70% |

## Testing

```bash
# Run all tests
pytest tests/test_turboquant_v3_refactored.py -v

# Validate on synthetic data
python scripts/validate_turboquant_v3.py --skip-model --seq-len 512

# Validate with real model
python scripts/validate_turboquant_v3.py --model Qwen/Qwen2.5-0.5B
```

## Advanced Usage

### Custom Configuration

```python
from turboquant import TurboQuantV3, TurboQuantConfig

config = TurboQuantConfig(
    key_bits=5,
    value_bits=3,
    use_rotation=True,
    residual_window=256
)
tq = TurboQuantV3(config)
```

### Low-level Components

```python
from turboquant import RandomRotation, LloydMaxQuantizer, MSECompressor

# Use components directly
rotation = RandomRotation(dim=128)
quantizer = LloydMaxQuantizer(num_bits=4)
```

## References

- **Paper**: "TurboQuant: Optimal High-Precision Quantization for Attention" (ICLR 2026)
- **Reference Implementation**: [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)

## License

MIT License - See LICENSE file
