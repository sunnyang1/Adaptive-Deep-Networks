from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# MATDO-new/matdo_new/ -> repository root (Adaptive-Deep-Networks)
REPO_ROOT = PACKAGE_ROOT.parent.parent

__all__ = ["PACKAGE_ROOT", "REPO_ROOT"]
