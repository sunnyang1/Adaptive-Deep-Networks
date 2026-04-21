from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from adn.matdo_e import REPO_ROOT


@contextmanager
def repo_root_on_path() -> Iterator[None]:
    """Temporarily expose the parent repo root for bridge imports."""
    repo_root = str(Path(REPO_ROOT))
    inserted = False
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(repo_root)
            except ValueError:
                pass
