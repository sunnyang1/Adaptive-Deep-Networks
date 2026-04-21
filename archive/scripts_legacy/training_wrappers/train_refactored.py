"""
Refactored Training Script

Deprecated entrypoint.

This file is kept for backwards compatibility. The canonical training entrypoint
is now:

  python3 scripts/training/train_model.py ...
"""

from scripts.training import train_model


def main() -> None:
    print(
        "NOTE: `scripts/training/train_refactored.py` is deprecated. "
        "Dispatching to `scripts/training/train_model.py`."
    )
    train_model.main()


if __name__ == "__main__":
    main()
