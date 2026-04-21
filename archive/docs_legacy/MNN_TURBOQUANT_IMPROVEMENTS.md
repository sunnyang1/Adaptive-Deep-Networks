# MNN-Inspired TurboQuant Improvements

**Reference**: [Alibaba MNN Commit 244f5d1](https://github.com/alibaba/MNN/commit/244f5d10df5a95b4f4e6f3d9251c6fe3dc0e7c83)

## Overview

This document describes the improvements to TurboQuant based on Alibaba's MNN (Mobile Neural Network) framework implementation. MNN's approach provides a more flexible and production-ready KV cache quantization system.

## Key Improvements

### 1. Attention Mode Encoding

MNN introduces a unified encoding scheme for attention configuration:

```
attention_mode = flash_attention * 8 + kv_quant_mode
```

Where:
- **flash_attention**: 0 (standard) or 1 (FlashAttention)
- **kv_quant_mode**: 0-6 (quantization level)

| Mode | FlashAttn | KV Quant | Description |
|------|-----------|----------|-------------|
| 0 | No | FP16 | Baseline |
| 8 | Yes | FP16 | Default (FlashAttention only) |
| 10 | Yes | INT8 | Near lossless |
| 12 | Yes | TQ3 | Extreme compression |
| 14 | Yes | TQ4 | Balanced (recommended for 4B+) |

### 2. KV Quantization Modes

MNN supports separate quantization for keys and values:

```python
class KVQuantMode(IntEnum):
    FP16 = 0           # No quantization
    KEY_INT8 = 1       # Key INT8, Value FP16
    KV_INT8 = 2        # Both INT8
    KEY_TQ3 = 3        # Key TQ3, Value FP16
    KV_TQ3 = 4         # Both TQ3
    KEY_TQ4 = 5        # Key TQ4, Value FP16
    KV_TQ4 = 6         # Both TQ4
```

### 3. Lloyd-Max Quantization

MNN uses optimized Lloyd-Max quantization for TQ3/TQ4 modes:

```python
from src.turboquant import LloydMaxQuantizer

# Create 4-bit quantizer
quantizer = LloydMaxQuantizer(num_bits=4, max_iter=100)

# Train on data
quantizer.fit(training_data)

# Encode/decode
indices = quantizer.encode(data)
decoded = quantizer.decode(indices)
```

### 4. Model-Size Aware Recommendations

MNN recommends different modes based on model size:

| Model Size | Recommended Mode | Compression | Notes |
|------------|------------------|-------------|-------|
| <4B | 8 or 10 | 1x-2x | TQ not recommended |
| 4B-10B | 14 | 3x | FlashAttention + TQ4 |
| >10B | 12 or 14 | 3x-4x | TQ3 for extreme compression |

## Usage

### Basic Usage

```python
from src.turboquant import (
    MNNTurboQuantConfig,
    MNNTurboQuantCompressor,
    create_mnn_turboquant,
)

# Create compressor with recommended config
compressor = create_mnn_turboquant(
    attention_mode=14,  # FlashAttention + KV-TQ4
    head_dim=128,
)

# Compress KV cache
compressed = compressor.compress_kv(keys, values)
keys_decomp, values_decomp = compressor.decompress_kv(compressed)
```

### Custom Configuration

```python
from src.turboquant import MNNTurboQuantConfig

config = MNNTurboQuantConfig(
    attention_mode=14,
    use_lloyd_max=True,
    lloyd_max_iterations=100,
    min_params_for_tq=4e9,  # 4B threshold
)

compressor = MNNTurboQuantCompressor(config, head_dim=128)

# Fit codebooks on sample data
compressor.fit_codebooks(sample_keys, sample_values)
```

### Memory Calculation

```python
stats = compressor.get_memory_stats(
    seq_len=32768,
    batch_size=1,
    num_heads=32
)

print(f"Original: {stats['original_mb']:.1f} MB")
print(f"Compressed: {stats['compressed_mb']:.1f} MB")
print(f"Saving ratio: {stats['saving_ratio']:.2f}x")
```

## Performance Comparison

### Compression Ratios

| Mode | Compression | Memory Saved | Accuracy Impact |
|------|-------------|--------------|-----------------|
| FP16 (8) | 1.0x | 0% | None |
| KV-INT8 (10) | 2.0x | 50% | Near lossless |
| KV-TQ4 (14) | 3.0x | 67% | Minimal (4B+) |
| KV-TQ3 (12) | 4.0x | 75% | Acceptable (4B+) |

### Recommended Configurations

```python
from src.turboquant import CONFIG_RECOMMENDATIONS

# Pre-defined configs
CONFIG_RECOMMENDATIONS = {
    'default': 8,           # FlashAttention + FP16
    'near_lossless': 10,    # FlashAttention + KV-INT8
    'balanced': 14,         # FlashAttention + KV-TQ4
    'extreme': 12,          # FlashAttention + KV-TQ3
}
```

## Demo Script

Run the demo to see all features:

```bash
python scripts/experiments/mnn_turboquant_demo.py --all
```

Or specific demos:

```bash
python scripts/experiments/mnn_turboquant_demo.py --modes
python scripts/experiments/mnn_turboquant_demo.py --compression
python scripts/experiments/mnn_turboquant_demo.py --memory
python scripts/experiments/mnn_turboquant_demo.py --quantization
python scripts/experiments/mnn_turboquant_demo.py --recommendations
```

## Testing

Run tests:

```bash
pytest tests/unit/test_mnn_turboquant.py -v
```

## Migration from Original TurboQuant

The MNN-inspired implementation is compatible with the original TurboQuant but offers more flexibility:

```python
# Original
from src.turboquant import TurboQuantPipeline, TurboQuantConfig

# MNN-inspired (new)
from src.turboquant import (
    MNNTurboQuantCompressor,
    MNNTurboQuantConfig,
)

# Key differences:
# 1. attention_mode replaces separate flags
# 2. Separate KV quantization
# 3. Model-size awareness
# 4. Optimized Lloyd-Max
```

## References

1. [MNN Commit 244f5d1](https://github.com/alibaba/MNN/commit/244f5d10df5a95b4f4e6f3d9251c6fe3dc0e7c83)
2. [MNN Documentation](https://www.yuque.com/mnn/en)
3. TurboQuant Paper (ICLR 2026)
