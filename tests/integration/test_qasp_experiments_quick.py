"""Smoke test for QASP quick experiment runner artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from QASP.experiments.runner import run_quick_experiments


def test_qasp_experiment_runner_quick_writes_outputs(tmp_path: Path) -> None:
    code = run_quick_experiments(output_dir=tmp_path)
    assert code == 0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "config_snapshot.json").exists()
    assert (tmp_path / "logs.txt").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "needle_accuracy" in metrics
    assert "math_score" in metrics
    assert "ablation" in metrics
    assert "efficiency" in metrics
    assert isinstance(metrics["needle_accuracy"], float)
    assert 0.0 <= metrics["needle_accuracy"] <= 1.0
    assert isinstance(metrics["math_score"], float)
    assert 0.0 <= metrics["math_score"] <= 1.0
    assert isinstance(metrics["ablation"], dict)
    assert isinstance(metrics["efficiency"], dict)
    assert metrics["efficiency"]["tokens_per_second"] > 0.0
    assert metrics["efficiency"]["memory_gb"] > 0.0
    assert metrics["efficiency"]["latency_ms"] > 0.0
