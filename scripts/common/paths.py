"""
Path Management for Scripts

Handle project paths and environment detection.
"""

import sys
import os
from pathlib import Path
from typing import Dict


def add_project_to_path():
    """
    Add project root and src to sys.path.

    Call this at the start of every script.
    """
    script_dir = Path(__file__).parent.parent.parent
    project_dir = script_dir.parent

    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    src_dir = project_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def get_default_paths() -> Dict[str, Path]:
    """
    Get default paths based on detected environment.

    Returns:
        Dictionary with 'output', 'data', 'checkpoints' paths
    """
    # Detect environment
    if os.path.exists("/root/autodl-tmp"):
        # AutoDL environment
        base = Path("/root/autodl-tmp")
    elif os.path.exists("/home/ubuntu"):
        # Lambda Labs / AWS
        base = Path("/home/ubuntu")
    else:
        # Local development
        base = Path(__file__).parent.parent.parent

    return {
        "output": base / "results",
        "data": base / "data",
        "checkpoints": base / "checkpoints",
        "cache": base / ".cache",
    }


def ensure_directories(paths: Dict[str, Path]):
    """Ensure all directories exist."""
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


# Environment detection
class Environment:
    """Detect and provide environment information."""

    @staticmethod
    def is_autodl() -> bool:
        return os.path.exists("/root/autodl-tmp")

    @staticmethod
    def is_lambda() -> bool:
        return os.path.exists("/home/ubuntu") and not Environment.is_autodl()

    @staticmethod
    def is_local() -> bool:
        return not (Environment.is_autodl() or Environment.is_lambda())

    @staticmethod
    def get_name() -> str:
        if Environment.is_autodl():
            return "autodl"
        elif Environment.is_lambda():
            return "lambda"
        return "local"
