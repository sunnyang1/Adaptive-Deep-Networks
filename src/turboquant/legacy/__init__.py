"""
Legacy TurboQuant implementations.

These modules are kept for backward compatibility but are no longer
actively maintained. New code should use the refactored API.

Warning:
    These modules may be removed in a future version.
    Please migrate to the new API in turboquant.
"""

import warnings

warnings.warn(
    "turboquant.legacy is deprecated. Use turboquant (the refactored API) instead.",
    DeprecationWarning,
    stacklevel=2
)

# Make legacy modules available
from . import v3_improved
from . import core
from . import mnn_improved
from . import polar_quant
from . import qjl
from . import tensor_core
from . import turbo_quant

__all__ = [
    'v3_improved',
    'core',
    'mnn_improved',
    'polar_quant',
    'qjl',
    'tensor_core',
    'turbo_quant',
]
