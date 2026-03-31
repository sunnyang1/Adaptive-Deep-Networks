"""
TurboQuant V3: Clean, Modular KV Cache Compression

A refactored implementation based on community findings that QJL hurts
attention quality. Uses MSE-only compression with per-vector normalization.

Quick Start:
    >>> from turboquant import create_k4_v2
    >>> 
    >>> tq = create_k4_v2(head_dim=64)
    >>> tq.fit(sample_keys, sample_values)
    >>> compressed = tq.compress(keys, values)
    >>> keys_dq, values_dq = tq.decompress(compressed)

Recommended Configurations:
    - create_k4_v2(): 4-bit keys, 2-bit values (~4.9x compression) ⭐ Recommended
    - create_k3_v2(): 3-bit keys, 2-bit values (~3.0x compression)
    - create_k2_v2(): 2-bit keys, 2-bit values (~7.1x compression, max memory)

For HuggingFace Integration:
    >>> cache = tq.as_cache(residual_window=128)
    >>> model.generate(..., past_key_values=cache)
"""

__version__ = '3.0.0'

# ============================================================================
# Main API (Recommended)
# ============================================================================

from .api import (
    TurboQuantV3,
    TurboQuantConfig,
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
    V3Cache,
    CacheConfig,
)

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main API
    'TurboQuantV3',
    'TurboQuantConfig',
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
    'V3Cache',
    'CacheConfig',
]


def __getattr__(name):
    """
    Provide helpful error messages for legacy imports.
    """
    legacy_imports = {
        'TurboQuant': 'This class has been removed. Use TurboQuantV3 instead.',
        'PolarQuant': 'This class has been moved to turboquant.legacy.polar_quant',
        'QJLCompressor': 'QJL has been removed from V3. Use MSECompressor instead.',
        'TurboQuantCompressorV2': 'This class has been removed. Use TurboQuantV3.',
        'TurboQuantPipeline': 'This class has been removed. Use TurboQuantV3.',
    }
    
    if name in legacy_imports:
        raise ImportError(
            f"'{name}' is no longer available in turboquant. {legacy_imports[name]}"
        )
    
    raise AttributeError(f"module 'turboquant' has no attribute '{name}'")
