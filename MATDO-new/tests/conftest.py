from __future__ import annotations

import sys
from pathlib import Path

# Allow source-tree test runs to import `matdo_new` from MATDO-new/ and `src.*` from repo root.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
