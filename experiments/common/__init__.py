"""
Experiments Common Utilities

Shared infrastructure for all experiments across:
- core/: Main paper experiments (exp1-exp6)
- validation/: Paper table validation
- real_model/: Real model validation
"""

import sys
from pathlib import Path

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import ExperimentConfig, ModelSizeConfig, MODEL_SIZES, VALIDATION_TARGETS
from .paths import OutputPaths, get_project_root
from .device import get_device, DeviceManager
from .logging_config import setup_logging, get_logger
from .visualization import (
    ARCHITECTURE_COLORS,
    ARCHITECTURE_LABELS,
    FigureManager,
    plot_architecture_comparison,
    plot_training_curves,
    plot_heatmap,
)

__all__ = [
    'ExperimentConfig',
    'ModelSizeConfig', 
    'MODEL_SIZES',
    'OutputPaths',
    'get_project_root',
    'get_device',
    'DeviceManager',
    'setup_logging',
    'get_logger',
    'ARCHITECTURE_COLORS',
    'ARCHITECTURE_LABELS',
    'FigureManager',
    'plot_architecture_comparison',
    'plot_training_curves',
    'plot_heatmap',
]

__version__ = "1.0.0"
