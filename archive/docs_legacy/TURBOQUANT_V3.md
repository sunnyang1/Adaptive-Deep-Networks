# TurboQuant V3 - Community Improvements

**Reference**: [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)

## Overview

TurboQuant V3 incorporates key findings from 8+ independent community implementations. The main discovery: **QJL hurts for KV cache compression** because softmax amplifies its variance.

## Key Findings

### QJL Problem (V2)

The original paper's QJL (Quantized Johnson-Lindenstrauss) adds a second stage for unbiased inner products. **This doesn't work for attention:**

1. QJL provides **unbiased** inner products
2. Attention applies **softmax** to scores
3. Softmax **exponentially amplifies variance**
4. QJL's random noise gets magnified → garbage output

**Results (Qwen2.5-3B):**
- V2 (MSE+QJL): **0/27** generation tests passed
- V3 (MSE-only): **18/18** generation tests passed

### V3 Solution

Remove QJL, use all bits for reconstruction quality (MSE-only).

## V3 Features

### 1. MSE-Only Compression

```python
from src.turboquant import create_v3_k4_v2

# No QJL - all bits go to reconstruction quality
v3 = create_v3_k4_v2(head_dim=128)
```

### 2. Asymmetric K/V Bits

Keys need more precision than values:

| Config | Key Bits | Value Bits | Avg Bits | Compression |
|--------|----------|------------|----------|-------------|
| K4/V2 | 4 | 2 | 3.0 | **5.1x** ⭐ |
| K3/V2 | 3 | 2 | 2.5 | 6.0x |
| K4/V3 | 4 | 3 | 3.5 | 3.8x |

**Why?**
- Keys: Decide WHICH tokens to attend to (needs precision)
- Values: Content that gets averaged (errors cancel)

### 3. Bit-Packed Storage

Real compression ratios, not theoretical:

```python
# Before: 38% larger than uncompressed (padding waste)
# After: True compression with bit-packing
```

### 4. Layer-Adaptive Precision

Protect sensitive first/last layers:

```python
v3 = create_v3_layer_adaptive(
    key_bits=3,
    value_bits=2,
    protected_layers=2,  # First/last 2 layers get more bits
    total_layers=32
)
```

## Usage

### Quick Start (Recommended: K4/V2)

```python
from src.turboquant import create_v3_k4_v2

# Create compressor
v3 = create_v3_k4_v2(head_dim=128, device='cuda')

# Fit on sample data for each layer
for layer_idx in range(num_layers):
    v3.fit(
        sample_keys[layer_idx],
        sample_values[layer_idx],
        head_dim=128,
        layer_idx=layer_idx
    )

# Compress during inference
compressed = v3.compress_kv(
    keys, values, 
    head_dim=128, 
    layer_idx=layer_idx
)

# Decompress
keys_deq, values_deq = v3.decompress_kv(compressed)
```

### Memory Statistics

```python
stats = v3.memory_stats(
    seq_len=32768,
    num_layers=32,
    batch_size=1,
    num_heads=32,
    head_dim=128
)

print(f"Original: {stats['original_mb']:.1f} MB")
print(f"Compressed: {stats['compressed_mb']:.1f} MB")
print(f"Saved: {stats['memory_saved_percent']:.0f}%")
```

### Custom Configuration

```python
from src.turboquant import TurboQuantV3, TurboQuantV3Config

config = TurboQuantV3Config(
    key_bits=4,
    value_bits=2,
    use_rotation=True,
    pack_bits=True,
    protected_layers=2,
    total_layers=32,
    device='cuda'
)

v3 = TurboQuantV3(config)
```

## Results

### Generation Test (Needle in Haystack)

Hidden fact retrieval across context lengths:

| Config | 2K ctx | 4K ctx | 8K ctx | Score |
|--------|--------|--------|--------|-------|
| FP16 | FOUND | FOUND | FOUND | 3/3 |
| **K4/V2** | **FOUND** | **FOUND** | **FOUND** | **3/3** |
| K3/V2 | FOUND | FOUND | FOUND | 3/3 |

### Attention Score Accuracy

| Config | Compression | Cosine Similarity | Top-1 Match |
|--------|-------------|-------------------|-------------|
| K4/V2 | 5.1x | 0.9996 | 94% |
| K4/V2 Protected | 3.6x | 0.9997 | 99% |
| V2 (MSE+QJL) | 5.0x | 0.9945 | 86% |

## Recommended Configurations

```python
from src.turboquant import V3_RECOMMENDED

# Best overall (recommended)
v3 = V3_RECOMMENDED['k4_v2'](head_dim=128)

# Maximum compression
v3 = V3_RECOMMENDED['k3_v2'](head_dim=128)

# Layer-adaptive (protect first/last)
v3 = V3_RECOMMENDED['k4_v2_protected'](head_dim=128)
```

| Name | Ratio | Quality | Use Case |
|------|-------|---------|----------|
| k4_v2 | 5.1x | 18/18 perfect | Best overall |
| k3_v2 | 6.0x | 18/18 perfect | Max compression |
| k4_v2_protected | 3.6x | 99% top-1 | Sensitive tasks |

## Implementation Details

### Algorithm

1. **Random Rotation**: Fast Walsh-Hadamard Transform (FWHT)
2. **Normalization**: L2 normalize vector
3. **Quantization**: Lloyd-Max optimal scalar quantizer
4. **Bit-Packing**: Pack low-bit integers compactly

### Why Remove QJL?

```
V2 (with QJL):
  Unbiased inner product → Softmax → AMPLIFIED VARIANCE → ❌

V3 (MSE-only):
  Biased but LOW VARIANCE → Softmax → BETTER RESULTS → ✅
```

### Fast Walsh-Hadamard Transform

```python
# O(n log n) instead of O(n^2)
x_rotated = fwht(x * D)  # D = random diagonal signs
```

## Demo

```bash
# Run V3 demo
python scripts/experiments/turboquant_v3_demo.py --all

# Run tests
pytest tests/unit/test_turboquant_v3.py -v
```

## Community Implementations

- **scos-lab/turboquant**: 8-model benchmark, K/V norm ratio analysis
- **0xSero/turboquant**: Triton kernels, vLLM integration
- **back2matching/turboquant**: HuggingFace drop-in, residual windowing
- **TheTom/turboquant_plus**: Layer-adaptive, Apple Silicon optimized
- **RecursiveIntell/turbo-quant**: Rust implementation
- **SCJedi/entropy-adaptive-kv-cache**: + token eviction for 12x compression

## References

1. [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)
2. TurboQuant Paper (ICLR 2026)
3. [MNN TurboQuant](https://github.com/alibaba/MNN)
