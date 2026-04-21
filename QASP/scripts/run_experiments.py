"""Run QASP quick experiment sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QASP.experiments.runner import run_quick_experiments


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QASP experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick experiment suite")
    parser.add_argument(
        "--output-dir",
        default="results/qasp/quick",
        help="Output directory for experiment artifacts",
    )
    args = parser.parse_args()

    if not args.quick:
        parser.error("Only --quick mode is currently supported in this runner.")

    output_dir = ROOT / args.output_dir
    code = run_quick_experiments(output_dir=output_dir)
    if code == 0:
        print(f"QASP quick experiments completed. Artifacts: {output_dir}")
    else:
        print(f"QASP quick experiments failed with code {code}.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

