# RaBitQ - Rapid and Accurate Bit-level Quantization

True implementation of RaBitQ (SIGMOD 2024/2025) for transformer KV cache compression.

## Overview

RaBitQ compresses key-value caches in transformer models using:
- **1-bit binary quantization** of rotated residual vectors
- **Extended-bit refinement** (optional multi-bit correction codes)
- **Random orthogonal rotation** (FWHT-Kac or QR-based matrix)
- **Popcount-based asymmetric inner-product estimation**
- **Per-vector factor computation** for unbiased distance estimation

This implementation is based on the original C++ reference by Gao & Long (SIGMOD 2024) and the follow-up Extended-RaBitQ (SIGMOD 2025).

## Installation

```bash
pip install torch numpy
```

## Quick Start

```python
from rabitq import create_k3

# Create compressor (3-bit total = 1 sign + 2 extended bits)
rq = create_k3(head_dim=64)

# Fit on representative samples
rq.fit(sample_keys, sample_values)

# Compress
compressed = rq.compress(keys, values)

# Decompress
keys_dq, values_dq = rq.decompress(compressed)
```

## Configuration Presets

| Function | Total Bits | Ex Bits | Compression* | Quality |
|----------|------------|---------|--------------|---------|
| `create_k1()` | 1 | 0 | ~8x | Fastest |
| `create_k2()` | 2 | 1 | ~5.3x | Good |
| `create_k3()` | 3 | 2 | ~4.0x | Best |

\*Compression ratios are per-tensor vs fp16 baseline for head_dim=64, counting only reconstruction-effective storage (binary + ex codes + delta/vl metadata).

Backward-compatible aliases:
- `create_k4_v2()` → `create_k3()`
- `create_k3_v2()` → `create_k2()`
- `create_k2_v2()` → `create_k1()`

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
├── api.py          # Main API (RaBitQ, factory functions)
├── rotation.py     # Random orthogonal rotators (FWHT-Kac, Matrix QR)
├── packing.py      # Bit-packing for binary and extended codes
├── quantizer.py    # True RaBitQ quantization (binary + ex_bits)
├── estimator.py    # Asymmetric popcount-based IP estimator
├── cache.py        # HF-compatible compressed cache
└── legacy/         # Deprecated TurboQuant-based implementations
```

## Testing

```bash
# Run all tests
pytest tests/unit/test_rabitq.py tests/test_rabitq_refactored.py -v

# Validate compression
python experiments/rabitq/run_compression_verification.py --quick
```

## Advanced Usage

### Custom Configuration

```python
from rabitq import RaBitQ, RaBitQConfig

config = RaBitQConfig(
    total_bits=3,           # 1 sign + 2 extended bits
    use_rotation=True,
    rotator_type='fht',     # 'fht' or 'matrix'
    residual_window=256
)
rq = RaBitQ(config)
```

### Low-level Components

```python
from rabitq import FhtKacRotator, quantize_vector, reconstruct_vector

# Rotate vectors
rotator = FhtKacRotator(dim=128)
x_rot = rotator.rotate(x)

# Quantize a single vector (in rotated space)
qv = quantize_vector(x_rot, centroid=torch.zeros(128), config=...)

# Reconstruct
x_rec = reconstruct_vector(centroid, qv)
```

### Asymmetric Inner-Product Estimation

```python
from rabitq import estimate_inner_product

# Estimate q @ k^T without full decompression
ip_est = estimate_inner_product(query_rot, centroid, qv, query_bits=8)
```

## References

- **RaBitQ**: Gao, J. & Long, C. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." SIGMOD, 2024.
- **Extended-RaBitQ**: VectorDB-NTU. "Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search." SIGMOD, 2025.
- **Reference Implementation**: [VectorDB-NTU/RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library)

## License

MIT License - See LICENSE file
