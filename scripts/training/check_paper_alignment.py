#!/usr/bin/env python3
"""
Check paper alignment from training_results.json.

Usage:
  python3 scripts/training/check_paper_alignment.py \
    --results results/medium_paper_train/training_results.json \
    --strict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser(description="Validate paper_alignment in training_results.json")
    p.add_argument("--results", required=True, help="Path to training_results.json")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if not aligned")
    args = p.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"FAIL: results file not found: {path}")
        raise SystemExit(2)

    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    pa = obj.get("paper_alignment")
    if not isinstance(pa, dict):
        print("FAIL: missing paper_alignment field")
        raise SystemExit(2 if args.strict else 0)

    checks = pa.get("checks", {})
    is_aligned = bool(pa.get("is_aligned", False))
    expected = pa.get("expected", {})
    actual = pa.get("actual", {})
    preset = bool(pa.get("paper_preset_enabled", False))

    print("Paper Alignment Check")
    print("=" * 60)
    print(f"results: {path}")
    print(f"paper_preset_enabled: {preset}")
    print(f"is_aligned: {is_aligned}")
    print("-" * 60)

    if isinstance(checks, dict) and checks:
        keys = sorted(checks.keys())
        for k in keys:
            ok = bool(checks[k])
            e = expected.get(k, "N/A")
            a = actual.get(k, "N/A")
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {k}: expected={e} actual={a}")
    else:
        print("No detailed checks found.")

    print("=" * 60)
    if is_aligned:
        print("OVERALL: PASS")
        raise SystemExit(0)
    else:
        print("OVERALL: FAIL")
        raise SystemExit(1 if args.strict else 0)


if __name__ == "__main__":
    main()

