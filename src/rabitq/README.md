# RaBitQ - Rapid and Accurate Bit-level Quantization

Clean, modular implementation of RaBitQ for extreme KV cache compression.

## Overview

RaBitQ compresses key-value caches in transformer models using:
- **MSE-only compression** with per-vector normalization
- **Fast Walsh-Hadamard Transform** (O(n log n) random rotation)
- **Lloyd-Max quantization** (optimal scalar quantization)
- **Bit-packing** for efficient storage

## Installation

```bash
pip install torch numpy
```

## Quick Start

```python
from rabitq import create_k4_v2

# Create compressor (4-bit keys, 2-bit values)
rq = create_k4_v2(head_dim=64)

# Fit on representative samples
rq.fit(sample_keys, sample_values)

# Compress
compressed = rq.compress(keys, values)

# Decompress
keys_dq, values_dq = rq.decompress(compressed)
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
cache = rq.as_cache(residual_window=128)

# Use in generation
model.generate(
    input_ids,
    past_key_values=cache,
    use_cache=True
)
```

## Architecture

```
rabitq/
├── api.py        # Main API (RaBitQ, factory functions)
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
pytest tests/unit/test_rabitq.py -v

# Validate on synthetic data
python scripts/validate_rabitq.py --skip-model --seq-len 512
```

## Advanced Usage

### Custom Configuration

```python
from rabitq import RaBitQ, RaBitQConfig

config = RaBitQConfig(
    key_bits=5,
    value_bits=3,
    use_rotation=True,
    residual_window=256
)
rq = RaBitQ(config)
```

### Low-level Components

```python
from rabitq import RandomRotation, LloydMaxQuantizer, MSECompressor

# Use components directly
rotation = RandomRotation(dim=128)
quantizer = LloydMaxQuantizer(num_bits=4)
```

## References

- **RaBitQ**: "RaBitQ: Rotation-aligned Bias-free Quantization" (Li et al.)
- **TurboQuant**: Original inspiration for the compression pipeline

## License

MIT License - See LICENSE file
