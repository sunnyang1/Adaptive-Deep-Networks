"""
RaBitQ: Rapid and Accurate Bit-level Quantization for KV Cache Compression

A true implementation of RaBitQ (SIGMOD 2024/2025) for transformer KV cache
compression, featuring:
- Random orthogonal rotation (FWHT-Kac or QR-based)
- 1-bit binary quantization with optional extended-bit refinement
- Per-vector factor computation for unbiased inner-product estimation
- Popcount-based asymmetric distance estimation
- HuggingFace-compatible compressed cache

Quick Start:
    >>> from rabitq import create_k1
    >>> 
    >>> rq = create_k1(head_dim=64)
    >>> rq.fit(sample_keys, sample_values)
    >>> compressed = rq.compress(keys, values)
    >>> keys_dq, values_dq = rq.decompress(compressed)

Recommended Configurations:
    - create_k1(): 1-bit binary (~32x compression) ⭐ Maximum speed
    - create_k2(): 2-bit total (1 sign + 1 ex) (~16x compression)
    - create_k3(): 3-bit total (1 sign + 2 ex) (~10x compression)

For HuggingFace Integration:
    >>> cache = rq.as_cache(residual_window=128)
    >>> model.generate(..., past_key_values=cache)
"""

__version__ = "2.0.0"

# ============================================================================
# Main API (Recommended)
# ============================================================================

from .api import (
    RaBitQ,
    RaBitQConfig,
    CompressedKV,
    create_k1,
    create_k2,
    create_k3,
    create_k4_v2,
    create_k3_v2,
    create_k2_v2,
    RECOMMENDED,
)

# ============================================================================
# Low-level Components (For Advanced Usage)
# ============================================================================

from .rotation import (
    FhtKacRotator,
    MatrixRotator,
    IdentityRotator,
)

from .quantizer import (
    QuantizedVector,
    RabitqConfig,
    quantize_vector,
    reconstruct_vector,
    compute_const_scaling_factor,
    quantize_scalar,
    dequantize_scalar,
)

from .estimator import (
    estimate_inner_product,
    FullSingleQuery,
    SplitSingleQuery,
    SplitBatchQuery,
    make_full_single_query,
    full_est_dist,
    split_single_estdist,
    split_single_fulldist,
    split_distance_boosting,
)

from .cache import (
    RaBitQCache,
    CacheConfig,
)

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main API
    "RaBitQ",
    "RaBitQConfig",
    "CompressedKV",
    "create_k1",
    "create_k2",
    "create_k3",
    "create_k4_v2",
    "create_k3_v2",
    "create_k2_v2",
    "RECOMMENDED",
    # Components
    "FhtKacRotator",
    "MatrixRotator",
    "IdentityRotator",
    "QuantizedVector",
    "RabitqConfig",
    "quantize_vector",
    "reconstruct_vector",
    "compute_const_scaling_factor",
    "estimate_inner_product",
    "FullSingleQuery",
    "SplitSingleQuery",
    "SplitBatchQuery",
    "make_full_single_query",
    "full_est_dist",
    "split_single_estdist",
    "split_single_fulldist",
    "split_distance_boosting",
    "quantize_scalar",
    "dequantize_scalar",
    "RaBitQCache",
    "CacheConfig",
]


# ============================================================================
# Legacy import helpers
# ============================================================================


def __getattr__(name):
    """Provide helpful error messages for legacy imports."""
    legacy_imports = {
        "LloydMaxQuantizer": "This class has been removed. RaBitQ uses binary+extended quantization.",
        "MSECompressor": "This class has been removed. Use RaBitQ instead.",
        "CompressorConfig": "This class has been removed. Use RaBitQConfig instead.",
        "pack_bits": "Moved to rabitq.packing module.",
        "unpack_bits": "Moved to rabitq.packing module.",
        "V3Cache": "This class has been renamed to RaBitQCache.",
    }

    if name in legacy_imports:
        raise ImportError(f"'{name}' is no longer available in rabitq. {legacy_imports[name]}")

    raise AttributeError(f"module 'rabitq' has no attribute '{name}'")
