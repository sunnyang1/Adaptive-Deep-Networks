"""Unified QASP experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

from QASP.experiments.ablations import run_qasp_ablation
from QASP.experiments.benchmarks import run_math_eval, run_needle_benchmark
from QASP.experiments.efficiency import profile_qasp


def run_quick_experiments(output_dir: Path) -> int:
    """Execute quick QASP experiment sweep and write normalized artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "needle_accuracy": run_needle_benchmark(quick=True),
        "math_score": run_math_eval(quick=True),
        "ablation": run_qasp_ablation(quick=True),
        "efficiency": profile_qasp(quick=True),
    }
    config = {"mode": "quick", "framework": "QASP"}

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output_dir / "logs.txt").write_text("QASP quick experiment run completed.\n", encoding="utf-8")
    return 0

