from __future__ import annotations

import json
from pathlib import Path

import pytest

from matdo_new.apps import run_experiments as app
from matdo_new.apps.run_experiments import main
from matdo_new.experiments.schema import (
    BenchmarkIdentity,
    BenchmarkResult,
    ResultSummary,
    RunMetadata,
    RuntimeEnvelope,
)


def write_config(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def benchmark_names(payload: dict[str, object]) -> list[str]:
    results = payload["results"]
    assert isinstance(results, list)
    return [result["benchmark"]["name"] for result in results]


def fake_results() -> tuple[BenchmarkResult, BenchmarkResult]:
    needle_result = BenchmarkResult(
        run=RunMetadata(
            app="run_experiments",
            config_path="/tmp/experiments.yaml",
            root_config_path="/tmp/root.yaml",
            evaluator_name="needle-real",
        ),
        benchmark=BenchmarkIdentity(
            name="needle",
            kind="task",
            config={
                "context_lengths": (128, 256),
                "depth_distribution": "uniform",
                "max_new_tokens": 2,
                "model_size": "t4",
                "num_samples": 2,
                "seed": 42,
                "tokenizer_name": "fake-tokenizer",
                "use_attnres": False,
            },
        ),
        aggregate_metrics={
            "exact_match_rate": 0.5,
            "num_samples": 2,
            "retrieval_success_rate": 1.0,
        },
        slice_summaries=(
            ResultSummary(
                summary_id="context-length:128",
                metrics={"context_length": 128, "exact_match_rate": 1.0},
                metadata={"context_length": 128},
            ),
        ),
        example_summaries=(
            ResultSummary(
                summary_id="needle:128:0",
                metrics={"exact_match": True, "retrieval_success": True},
                metadata={"context_length": 128, "needle_depth_percent": 10.0},
            ),
        ),
        runtime=RuntimeEnvelope(
            policy={"tokenizer_name": "fake-tokenizer"},
            metrics={"prefill_calls": 1, "decode_calls": 2},
        ),
    )
    critical_points_result = BenchmarkResult(
        run=RunMetadata(
            app="run_experiments",
            config_path="/tmp/experiments.yaml",
            root_config_path="/tmp/root.yaml",
            evaluator_name="critical-points-toy",
        ),
        benchmark=BenchmarkIdentity(
            name="critical-points",
            kind="study",
            config={"points": (0.1, 0.2)},
        ),
        aggregate_metrics={
            "max_score": 0.9,
            "mean_score": 0.85,
            "min_score": 0.8,
            "num_points": 2,
        },
    )
    return needle_result, critical_points_result


def fake_math_result() -> BenchmarkResult:
    return BenchmarkResult(
        run=RunMetadata(
            app="run_experiments",
            config_path="/tmp/experiments.yaml",
            root_config_path="/tmp/root.yaml",
            evaluator_name="math-real",
        ),
        benchmark=BenchmarkIdentity(
            name="math",
            kind="task",
            config={
                "split": "test",
                "max_samples": 4,
                "subjects": ("algebra",),
                "levels": (3,),
                "prompt_style": "cot_boxed",
                "max_new_tokens": 128,
                "tokenizer_name": "fake-tokenizer",
                "model_size": "small",
                "use_attnres": False,
                "seed": 42,
            },
        ),
        aggregate_metrics={
            "accuracy": 0.75,
            "num_samples": 4,
        },
        slice_summaries=(
            ResultSummary(
                summary_id="math-subject:algebra",
                metrics={"accuracy": 0.75, "num_samples": 4},
                metadata={"subject": "algebra"},
            ),
        ),
        example_summaries=(
            ResultSummary(
                summary_id="math:algebra:3:0",
                metrics={"correct": True},
                metadata={"subject": "algebra", "level": 3},
            ),
        ),
    )


def test_main_writes_real_needle_payload_to_stdout_and_file(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    root_config = write_config(tmp_path / "root.yaml", "config_version: 1\n")
    experiments_config = write_config(
        tmp_path / "experiments.yaml",
        """
runtime:
  root_config: root.yaml
benchmarks:
  needle:
    enabled: true
    dataset:
      context_lengths: [128, 256]
      num_samples: 2
      depth_distribution: uniform
      seed: 42
    generation:
      max_new_tokens: 2
    runtime:
      tokenizer_name: fake-tokenizer
      model_size: t4
      use_attnres: false
  critical-points:
    enabled: true
    dataset:
      points: [0.1, 0.2]
output:
  emit_metadata: true
  format: json
""".strip()
        + "\n",
    )
    output_path = tmp_path / "results" / "experiments.json"
    captured: list[tuple[str, dict[str, object]]] = []

    def fake_run_experiments(*runners, **kwargs):
        captured.extend((runner.benchmark.name, runner.benchmark.config) for runner in runners)
        return fake_results()

    monkeypatch.setattr(app, "run_experiments", fake_run_experiments)

    assert main(["--config", str(experiments_config), "--output", str(output_path)]) == 0

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload == file_payload
    assert stdout_payload["schema_version"] == "matdo_new.cli_payload.v1"
    assert stdout_payload["app"] == "run_experiments"
    assert stdout_payload["config_path"] == str(experiments_config.resolve())
    assert stdout_payload["root_config_path"] == str(root_config.resolve())
    assert benchmark_names(stdout_payload) == ["needle", "critical-points"]
    assert captured == [
        (
            "needle",
            {
                "context_lengths": (128, 256),
                "depth_distribution": "uniform",
                "max_new_tokens": 2,
                "model_size": "t4",
                "num_samples": 2,
                "seed": 42,
                "tokenizer_name": "fake-tokenizer",
                "use_attnres": False,
            },
        ),
        ("critical-points", {"points": (0.1, 0.2)}),
    ]

    needle_result = stdout_payload["results"][0]
    assert needle_result["run"]["evaluator_name"] == "needle-real"
    assert needle_result["benchmark"]["config"] == {
        "context_lengths": [128, 256],
        "depth_distribution": "uniform",
        "max_new_tokens": 2,
        "model_size": "t4",
        "num_samples": 2,
        "seed": 42,
        "tokenizer_name": "fake-tokenizer",
        "use_attnres": False,
    }
    assert needle_result["aggregate_metrics"] == {
        "exact_match_rate": 0.5,
        "num_samples": 2,
        "retrieval_success_rate": 1.0,
    }
    assert needle_result["example_summaries"][0]["metadata"] == {
        "context_length": 128,
        "needle_depth_percent": 10.0,
    }


def test_main_applies_cli_benchmark_selection_and_overrides(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    write_config(tmp_path / "root.yaml", "config_version: 1\n")
    experiments_config = write_config(
        tmp_path / "experiments.yaml",
        """
runtime:
  root_config: root.yaml
benchmarks:
  needle:
    enabled: false
    dataset:
      context_lengths: [128]
      num_samples: 1
      depth_distribution: uniform
      seed: 42
    generation:
      max_new_tokens: 2
    runtime:
      tokenizer_name: config-tokenizer
      model_size: t4
      use_attnres: true
  critical-points:
    enabled: true
    dataset:
      points: [0.2, 0.4]
output:
  emit_metadata: true
  format: json
""".strip()
        + "\n",
    )
    captured: list[tuple[str, dict[str, object]]] = []

    def fake_run_experiments(*runners, **kwargs):
        captured.extend((runner.benchmark.name, runner.benchmark.config) for runner in runners)
        return ()

    monkeypatch.setattr(app, "run_experiments", fake_run_experiments)

    assert (
        main(
            [
                "--config",
                str(experiments_config),
                "--benchmark",
                "needle",
                "--needle-context-lengths",
                "512",
                "--needle-num-samples",
                "3",
                "--needle-depth-distribution",
                "late",
                "--needle-max-new-tokens",
                "4",
                "--needle-tokenizer-name",
                "cli-tokenizer",
                "--needle-model-size",
                "small",
                "--no-needle-use-attnres",
                "--needle-seed",
                "9",
                "--no-emit-metadata",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["results"] == []
    assert captured == [
        (
            "needle",
            {
                "context_lengths": (512,),
                "depth_distribution": "late",
                "max_new_tokens": 4,
                "model_size": "small",
                "num_samples": 3,
                "seed": 9,
                "tokenizer_name": "cli-tokenizer",
                "use_attnres": False,
            },
        )
    ]


def test_main_respects_enabled_flags_and_configured_root(monkeypatch, tmp_path: Path, capsys) -> None:
    configured_root = write_config(tmp_path / "configured-root.yaml", "config_version: 2\n")
    experiments_config = write_config(
        tmp_path / "experiments.yaml",
        """
runtime:
  root_config: configured-root.yaml
benchmarks:
  needle:
    enabled: false
    dataset:
      context_lengths: [128]
  critical-points:
    enabled: true
    dataset:
      points: [0.4]
output:
  emit_metadata: true
  format: json
""".strip()
        + "\n",
    )
    captured: list[str] = []

    def fake_run_experiments(*runners, **kwargs):
        captured.extend(runner.benchmark.name for runner in runners)
        return ()

    monkeypatch.setattr(app, "run_experiments", fake_run_experiments)

    assert main(["--config", str(experiments_config)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert configured_root.exists()
    assert payload["root_config_path"] == str(configured_root.resolve())
    assert captured == ["critical-points"]


def test_main_rejects_unsupported_output_format(tmp_path: Path) -> None:
    write_config(tmp_path / "root.yaml", "config_version: 1\n")
    experiments_config = write_config(
        tmp_path / "experiments.yaml",
        """
runtime:
  root_config: root.yaml
benchmarks:
  needle:
    enabled: true
    dataset:
      context_lengths: [128]
output:
  emit_metadata: true
  format: json
""".strip()
        + "\n",
    )

    with pytest.raises(SystemExit, match="2"):
        main(["--config", str(experiments_config), "--output-format", "yaml"])


def test_main_writes_math_payload_with_normalized_benchmark_shape(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    write_config(tmp_path / "root.yaml", "config_version: 1\n")
    experiments_config = write_config(
        tmp_path / "experiments.yaml",
        """
runtime:
  root_config: root.yaml
benchmarks:
  math:
    enabled: true
    dataset:
      split: test
      max_samples: 4
      subjects:
        - algebra
      levels:
        - 3
      seed: 42
    generation:
      prompt_style: cot_boxed
      max_new_tokens: 128
    runtime:
      tokenizer_name: fake-tokenizer
      model_size: small
      use_attnres: false
output:
  emit_metadata: true
  format: json
""".strip()
        + "\n",
    )
    captured: list[tuple[str, dict[str, object]]] = []

    def fake_run_experiments(*runners, **kwargs):
        captured.extend((runner.benchmark.name, runner.benchmark.config) for runner in runners)
        return (fake_math_result(),)

    monkeypatch.setattr(app, "run_experiments", fake_run_experiments)

    assert main(["--config", str(experiments_config)]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert benchmark_names(payload) == ["math"]
    assert captured == [
        (
            "math",
            {
                "split": "test",
                "max_samples": 4,
                "subjects": ("algebra",),
                "levels": (3,),
                "prompt_style": "cot_boxed",
                "max_new_tokens": 128,
                "tokenizer_name": "fake-tokenizer",
                "model_size": "small",
                "use_attnres": False,
                "seed": 42,
            },
        )
    ]

    math_result = payload["results"][0]
    assert math_result["run"]["evaluator_name"] == "math-real"
    assert math_result["benchmark"]["name"] == "math"
    assert math_result["benchmark"]["config"] == {
        "split": "test",
        "max_samples": 4,
        "subjects": ["algebra"],
        "levels": [3],
        "prompt_style": "cot_boxed",
        "max_new_tokens": 128,
        "tokenizer_name": "fake-tokenizer",
        "model_size": "small",
        "use_attnres": False,
        "seed": 42,
    }


def test_main_rejects_mixed_schema_config(tmp_path: Path) -> None:
    write_config(tmp_path / "root.yaml", "config_version: 1\n")
    experiments_config = write_config(
        tmp_path / "experiments.yaml",
        """
runtime:
  root_config: root.yaml
benchmarks:
  needle:
    enabled: true
    dataset:
      context_lengths: [128]
tasks:
  needle:
    enabled: true
    context_lengths: [64]
output:
  emit_metadata: true
  format: json
""".strip()
        + "\n",
    )

    with pytest.raises(ValueError, match="mixed experiment config schemas"):
        main(["--config", str(experiments_config)])
