"""
TurboQuant: Unified Extreme Compression API

Simple, clean interface for all quantization modes.

Quick Start:
    >>> from src.turboquant import TurboQuant
    
    >>> # No compression (FP16)
    >>> quant = TurboQuant('fp16')
    
    >>> # INT8 quantization (2x compression)
    >>> quant = TurboQuant('int8')
    
    >>> # TurboQuant 4-bit (3x compression)
    >>> quant = TurboQuant('tq4')
    >>> quant.fit(sample_keys, sample_values)  # Fit on data
    >>> compressed = quant.compress_kv(keys, values)
    
    >>> # With FlashAttention
    >>> quant = TurboQuant('tq4_flash')

Recommended Configs:
    - 'fp16': No compression (default)
    - 'int8': 2x compression, near lossless
    - 'tq4': 3x compression (4B+ models)
    - 'tq3': 4x compression (4B+ models)

Full API:
    >>> quant = TurboQuant('tq4', head_dim=128, device='cuda')
    >>> 
    >>> # Fit quantizers
    >>> quant.fit(sample_keys, sample_values)
    >>> # Or use Beta distribution for RHT
    >>> quant.fit_beta()
    >>> 
    >>> # Compress/decompress
    >>> compressed = quant.compress_kv(keys, values)
    >>> keys_deq, values_deq = quant.decompress_kv(compressed)
    >>> 
    >>> # Memory stats
    >>> stats = quant.memory_stats(seq_len=32768)
    >>> print(f"Saving: {stats['memory_saved']:.1f}%")
"""

# ============================================================================
# Unified API (Recommended)
# ============================================================================

from .core import (
    TurboQuant,
    TurboQuantConfig,
    QuantMode,
    create_quantizer,
    RECOMMENDED_CONFIGS,
    quantize_kv,
)

# ============================================================================
# Quantizers (Advanced)
# ============================================================================

from .core import (
    LloydMaxQuantizer,
    INT8Quantizer,
    FP16Quantizer,
)

# ============================================================================
# Legacy API (Backward Compatibility)
# ============================================================================

from .polar_quant import (
    PolarQuant,
    CartesianToPolar,
    HadamardTransform,
)

from .qjl import (
    QJLCompressor,
    QJLDecompressor,
    BatchQJL,
)

from .turbo_quant import (
    TurboQuantPipeline,
    TurboQuantConfig as LegacyTurboQuantConfig,
)

from .tensor_core import (
    TensorCoreKernel,
    INT4Linear,
)

# ============================================================================
# MNN-Inspired Improvements
# ============================================================================

from .mnn_improved import (
    MNNTurboQuantConfig,
    MNNTurboQuantCompressor,
    AttentionMode,
    KVQuantMode,
    create_mnn_turboquant,
    CONFIG_RECOMMENDATIONS as MNN_RECOMMENDATIONS,
)

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Unified API (New - Recommended)
    'TurboQuant',
    'TurboQuantConfig',
    'QuantMode',
    'create_quantizer',
    'RECOMMENDED_CONFIGS',
    'quantize_kv',
    
    # Quantizers
    'LloydMaxQuantizer',
    'INT8Quantizer',
    'FP16Quantizer',
    
    # Legacy API (Backward Compatibility)
    'PolarQuant',
    'CartesianToPolar',
    'HadamardTransform',
    'QJLCompressor',
    'QJLDecompressor',
    'BatchQJL',
    'TurboQuantPipeline',
    'LegacyTurboQuantConfig',
    'TensorCoreKernel',
    'INT4Linear',
    
    # MNN-Inspired
    'MNNTurboQuantConfig',
    'MNNTurboQuantCompressor',
    'AttentionMode',
    'KVQuantMode',
    'create_mnn_turboquant',
    'MNN_RECOMMENDATIONS',
]

__version__ = '2.0.0'
