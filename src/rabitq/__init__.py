"""
RaBitQ: Rapid and Accurate Bit-level Quantization for KV Cache Compression

A clean, modular implementation using Hadamard transform + per-vector normalization
+ Lloyd-Max quantization + bit-packing for efficient KV cache compression.

Quick Start:
    >>> from rabitq import create_k4_v2
    >>> 
    >>> rq = create_k4_v2(head_dim=64)
    >>> rq.fit(sample_keys, sample_values)
    >>> compressed = rq.compress(keys, values)
    >>> keys_dq, values_dq = rq.decompress(compressed)

Recommended Configurations:
    - create_k4_v2(): 4-bit keys, 2-bit values (~4.9x compression) ⭐ Recommended
    - create_k3_v2(): 3-bit keys, 2-bit values (~3.0x compression)
    - create_k2_v2(): 2-bit keys, 2-bit values (~7.1x compression, max memory)

For HuggingFace Integration:
    >>> cache = rq.as_cache(residual_window=128)
    >>> model.generate(..., past_key_values=cache)
"""

__version__ = '1.0.0'

# ============================================================================
# Main API (Recommended)
# ============================================================================

from .api import (
    RaBitQ,
    RaBitQConfig,
    create_k4_v2,
    create_k3_v2,
    create_k2_v2,
    RECOMMENDED,
)

# ============================================================================
# Low-level Components (For Advanced Usage)
# ============================================================================

from .rotation import (
    RandomRotation,
    fwht,
    fwht_inverse,
)

from .quantizer import (
    LloydMaxQuantizer,
)

from .compressor import (
    MSECompressor,
    CompressorConfig,
    pack_bits,
    unpack_bits,
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
    'RaBitQ',
    'RaBitQConfig',
    'create_k4_v2',
    'create_k3_v2',
    'create_k2_v2',
    'RECOMMENDED',
    
    # Components
    'RandomRotation',
    'fwht',
    'fwht_inverse',
    'LloydMaxQuantizer',
    'MSECompressor',
    'CompressorConfig',
    'pack_bits',
    'unpack_bits',
    'RaBitQCache',
    'CacheConfig',
]


def __getattr__(name):
    """
    Provide helpful error messages for legacy imports.
    """
    legacy_imports = {
        'TurboQuant': 'This class has been replaced by RaBitQ.',
        'TurboQuantV3': 'This class has been replaced by RaBitQ.',
        'PolarQuant': 'This class has been moved to rabitq.legacy.polar_quant',
        'QJLCompressor': 'QJL has been removed. Use MSECompressor instead.',
        'TurboQuantCompressorV2': 'This class has been replaced by RaBitQ.',
        'TurboQuantPipeline': 'This class has been replaced by RaBitQ.',
        'V3Cache': 'This class has been renamed to RaBitQCache.',
    }
    
    if name in legacy_imports:
        raise ImportError(
            f"'{name}' is no longer available in rabitq. {legacy_imports[name]}"
        )
    
    raise AttributeError(f"module 'rabitq' has no attribute '{name}'")
