# Legacy TurboQuant Implementations

This directory contains older implementations of TurboQuant that are kept for backward compatibility but are no longer actively maintained.

## Files

| File | Description | Status |
|------|-------------|--------|
| `v3_improved.py` | Original monolithic V3 implementation | Superseded by modular version |
| `core.py` | Pre-refactor unified API | Superseded by `api.py` |
| `mnn_improved.py` | MNN-inspired quantization modes | Experimental |
| `polar_quant.py` | Polar coordinate quantization | Not used in V3 |
| `qjl.py` | Quantized Johnson-Lindenstrauss | Removed from V3 (hurts quality) |
| `tensor_core.py` | Tensor core optimizations | Experimental |
| `turbo_quant.py` | Original TurboQuant implementation | Superseded |

## Migration Guide

### Old API (Deprecated)
```python
from turboquant import TurboQuantV3, create_v3_k4_v2

v3 = create_v3_k4_v2(head_dim=64)
v3.fit(keys, values, head_dim=64, layer_idx=0)
compressed = v3.compress_kv(keys, values, head_dim=64, layer_idx=0)
```

### New API (Recommended)
```python
from turboquant import create_k4_v2

tq = create_k4_v2(head_dim=64)
tq.fit(sample_keys, sample_values)
compressed = tq.compress(keys, values)
keys_dq, values_dq = tq.decompress(compressed)
```

## Deprecation Timeline

These legacy modules will be removed in version 4.0.0. Please migrate your code to use the new refactored API.
