from __future__ import annotations

from pathlib import Path

import pytest

from matdo_new.experiments.config import (
    ExperimentCLIOverrides,
    load_experiments_config,
    resolve_execution_plan,
)


def test_resolve_execution_plan_builds_enabled_benchmarks_from_benchmark_config() -> None:
    plan = resolve_execution_plan(
        {
            "benchmarks": {
                "needle": {
                    "enabled": True,
                    "dataset": {
                        "context_lengths": [128, 256],
                        "num_samples": 2,
                        "depth_distribution": "late",
                        "seed": 7,
                    },
                    "generation": {"max_new_tokens": 4},
                    "runtime": {
                        "tokenizer_name": "fake-tokenizer",
                        "model_size": "small",
                        "use_attnres": False,
                    },
                },
                "critical-points": {
                    "enabled": False,
                    "dataset": {"points": [0.1, 0.2]},
                },
            },
            "output": {"emit_metadata": False, "format": "json"},
        }
    )

    assert plan.output.emit_metadata is False
    assert plan.output.output_format == "json"
    assert len(plan.runners) == 1
    assert plan.runners[0].benchmark.name == "needle"
    assert plan.runners[0].benchmark.config == {
        "context_lengths": (128, 256),
        "depth_distribution": "late",
        "max_new_tokens": 4,
        "model_size": "small",
        "num_samples": 2,
        "seed": 7,
        "tokenizer_name": "fake-tokenizer",
        "use_attnres": False,
    }


def test_resolve_execution_plan_prefers_cli_selection_and_overrides() -> None:
    plan = resolve_execution_plan(
        {
            "benchmarks": {
                "needle": {
                    "enabled": False,
                    "dataset": {
                        "context_lengths": [128],
                        "num_samples": 1,
                    },
                    "generation": {"max_new_tokens": 2},
                    "runtime": {
                        "tokenizer_name": "config-tokenizer",
                        "model_size": "t4",
                        "use_attnres": True,
                    },
                },
                "critical-points": {
                    "enabled": True,
                    "dataset": {"points": [0.1, 0.2]},
                },
            },
            "output": {"emit_metadata": True, "format": "json"},
        },
        cli_overrides=ExperimentCLIOverrides(
            benchmarks=("needle",),
            needle_context_lengths=(512,),
            needle_num_samples=3,
            needle_depth_distribution="early",
            needle_max_new_tokens=6,
            needle_tokenizer_name="cli-tokenizer",
            needle_model_size="medium",
            needle_use_attnres=False,
            needle_seed=99,
            emit_metadata=False,
        ),
    )

    assert plan.output.emit_metadata is False
    assert [runner.benchmark.name for runner in plan.runners] == ["needle"]
    assert plan.runners[0].benchmark.config == {
        "context_lengths": (512,),
        "depth_distribution": "early",
        "max_new_tokens": 6,
        "model_size": "medium",
        "num_samples": 3,
        "seed": 99,
        "tokenizer_name": "cli-tokenizer",
        "use_attnres": False,
    }


def test_resolve_execution_plan_builds_math_runner_from_benchmark_config() -> None:
    plan = resolve_execution_plan(
        {
            "benchmarks": {
                "math": {
                    "enabled": True,
                    "dataset": {
                        "split": "test",
                        "max_samples": 12,
                        "subjects": ["algebra", "geometry"],
                        "levels": [3, 5],
                        "seed": 123,
                    },
                    "generation": {
                        "prompt_style": "cot_boxed",
                        "max_new_tokens": 192,
                    },
                    "runtime": {
                        "tokenizer_name": "math-tokenizer",
                        "model_size": "small",
                        "use_attnres": False,
                    },
                }
            }
        }
    )

    assert len(plan.runners) == 1
    assert plan.runners[0].benchmark.name == "math"
    assert plan.runners[0].benchmark.config == {
        "split": "test",
        "max_samples": 12,
        "subjects": ("algebra", "geometry"),
        "levels": (3, 5),
        "prompt_style": "cot_boxed",
        "max_new_tokens": 192,
        "tokenizer_name": "math-tokenizer",
        "model_size": "small",
        "use_attnres": False,
        "seed": 123,
    }


def test_resolve_execution_plan_supports_math_cli_selection_and_overrides() -> None:
    plan = resolve_execution_plan(
        {
            "benchmarks": {
                "needle": {
                    "enabled": True,
                    "dataset": {
                        "context_lengths": [128],
                        "num_samples": 1,
                    },
                    "generation": {"max_new_tokens": 2},
                    "runtime": {
                        "tokenizer_name": "config-tokenizer",
                        "model_size": "t4",
                        "use_attnres": True,
                    },
                },
                "math": {
                    "enabled": False,
                    "dataset": {
                        "split": "test",
                        "max_samples": 8,
                        "subjects": ["number_theory"],
                        "levels": [2],
                        "seed": 7,
                    },
                    "generation": {
                        "prompt_style": "cot_boxed",
                        "max_new_tokens": 64,
                    },
                    "runtime": {
                        "tokenizer_name": "config-math-tokenizer",
                        "model_size": "small",
                        "use_attnres": True,
                    },
                },
            }
        },
        cli_overrides=ExperimentCLIOverrides(
            benchmarks=("math",),
            math_subjects=("algebra", "geometry"),
            math_levels=(4, 5),
            math_max_new_tokens=256,
            math_tokenizer_name="cli-math-tokenizer",
            math_model_size="medium",
            math_use_attnres=False,
            math_seed=99,
        ),
    )

    assert [runner.benchmark.name for runner in plan.runners] == ["math"]
    assert plan.runners[0].benchmark.config == {
        "split": "test",
        "max_samples": 8,
        "subjects": ("algebra", "geometry"),
        "levels": (4, 5),
        "prompt_style": "cot_boxed",
        "max_new_tokens": 256,
        "tokenizer_name": "cli-math-tokenizer",
        "model_size": "medium",
        "use_attnres": False,
        "seed": 99,
    }


def test_load_experiments_config_strips_inline_comments_from_scalars(tmp_path: Path) -> None:
    config_path = tmp_path / "experiments.yaml"
    config_path.write_text(
        """
benchmarks:
  needle:
    enabled: true
    dataset:
      num_samples: 2 # keep this integer
    runtime:
      use_attnres: false # must stay boolean false
      tokenizer_name: gpt2 # comment after scalar
output:
  emit_metadata: true # trailing note
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_experiments_config(config_path)

    assert config["benchmarks"]["needle"]["dataset"]["num_samples"] == 2
    assert config["benchmarks"]["needle"]["runtime"]["use_attnres"] is False
    assert config["benchmarks"]["needle"]["runtime"]["tokenizer_name"] == "gpt2"
    assert config["output"]["emit_metadata"] is True


def test_load_experiments_config_parses_block_sequence_lists(tmp_path: Path) -> None:
    config_path = tmp_path / "experiments.yaml"
    config_path.write_text(
        """
benchmarks:
  needle:
    dataset:
      context_lengths:
        - 128
        - 256
  math:
    dataset:
      subjects:
        - algebra
        - geometry
      levels:
        - 3
        - 5
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_experiments_config(config_path)

    assert config["benchmarks"]["needle"]["dataset"]["context_lengths"] == [128, 256]
    assert config["benchmarks"]["math"]["dataset"]["subjects"] == ["algebra", "geometry"]
    assert config["benchmarks"]["math"]["dataset"]["levels"] == [3, 5]


def test_resolve_execution_plan_rejects_missing_benchmarks_section() -> None:
    with pytest.raises(ValueError, match="missing required 'benchmarks' section"):
        resolve_execution_plan({"output": {"emit_metadata": True, "format": "json"}})


def test_resolve_execution_plan_rejects_mixed_benchmark_and_legacy_schema() -> None:
    with pytest.raises(ValueError, match="mixed experiment config schemas"):
        resolve_execution_plan(
            {
                "benchmarks": {
                    "needle": {
                        "enabled": True,
                        "dataset": {"context_lengths": [128]},
                    }
                },
                "tasks": {
                    "needle": {
                        "enabled": True,
                        "context_lengths": [64],
                    }
                },
            }
        )


def test_resolve_execution_plan_rejects_legacy_only_schema() -> None:
    with pytest.raises(ValueError, match="legacy experiment config schema is no longer supported"):
        resolve_execution_plan(
            {
                "tasks": {
                    "needle": {
                        "enabled": True,
                        "context_lengths": [64],
                        "num_samples": 1,
                        "max_new_tokens": 3,
                        "tokenizer_name": "legacy-tokenizer",
                        "model_size": "t4",
                    }
                },
                "studies": {
                    "critical_points": {
                        "enabled": True,
                        "points": [0.25],
                    }
                },
            }
        )
