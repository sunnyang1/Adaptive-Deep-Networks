"""
Pytest configuration and shared fixtures for Adaptive Deep Networks tests.
"""

import pytest

# Legacy tests target removed/renamed APIs (TurboQuant V1/V2, old MNNTurboQuant).
# Run explicitly with: pytest tests/legacy/ -v
collect_ignore = ["legacy"]
import sys
from pathlib import Path
import torch
import numpy as np
from typing import Tuple

# Ensure repository root is importable so `import src.*` works
# regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def device():
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def rng():
    """Return a seeded random number generator."""
    return np.random.RandomState(42)


@pytest.fixture
def torch_rng():
    """Return a seeded torch random number generator."""
    torch.manual_seed(42)
    return torch


@pytest.fixture
def sample_tensor():
    """Return a sample tensor for testing."""

    def _create(shape: Tuple[int, ...], dtype=torch.float32):
        return torch.randn(shape, dtype=dtype)

    return _create


@pytest.fixture
def model_config_small():
    """Return a small model configuration for fast testing."""
    return {
        "vocab_size": 1000,
        "dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "mlp_ratio": 4,
        "num_blocks": 2,
    }


@pytest.fixture
def model_config_medium():
    """Return a medium model configuration."""
    return {
        "vocab_size": 32000,
        "dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "mlp_ratio": 4,
        "num_blocks": 4,
    }


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset torch random seed before each test."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "unit: marks unit tests")
    config.addinivalue_line("markers", "integration: marks integration tests")
