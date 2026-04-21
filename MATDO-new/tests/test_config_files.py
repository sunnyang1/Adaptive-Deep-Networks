from __future__ import annotations

from pathlib import Path

from matdo_new.apps.generate import DEFAULT_CONFIG_PATH as DEFAULT_INFERENCE_CONFIG_PATH
from matdo_new.apps.generate import DEFAULT_ROOT_CONFIG_PATH
from matdo_new.apps.generate import build_parser as build_generate_parser
from matdo_new.apps.run_experiments import DEFAULT_CONFIG_PATH as DEFAULT_EXPERIMENTS_CONFIG_PATH
from matdo_new.apps.run_experiments import build_parser as build_experiments_parser


def read_top_level_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line[:1].isspace():
            continue
        key, _, _ = stripped.partition(":")
        if key:
            keys.add(key)
    return keys


def test_package_default_config_files_exist_with_expected_top_level_keys() -> None:
    expected_keys = {
        DEFAULT_ROOT_CONFIG_PATH: {"config_version", "matdo_new", "policy"},
        DEFAULT_INFERENCE_CONFIG_PATH: {"runtime", "model", "prompt", "generation", "observation", "output"},
        DEFAULT_EXPERIMENTS_CONFIG_PATH: {"runtime", "benchmarks", "output"},
    }

    for path, keys in expected_keys.items():
        assert path.exists()
        assert path.is_file()
        assert keys.issubset(read_top_level_keys(path))


def test_cli_parsers_default_to_package_config_files() -> None:
    generate_args = build_generate_parser().parse_args([])
    experiments_args = build_experiments_parser().parse_args([])

    assert generate_args.root_config == DEFAULT_ROOT_CONFIG_PATH
    assert generate_args.config == DEFAULT_INFERENCE_CONFIG_PATH
    assert experiments_args.root_config == DEFAULT_ROOT_CONFIG_PATH
    assert experiments_args.config == DEFAULT_EXPERIMENTS_CONFIG_PATH


def test_default_experiments_config_includes_disabled_math_block() -> None:
    config_text = DEFAULT_EXPERIMENTS_CONFIG_PATH.read_text(encoding="utf-8")

    assert "benchmarks:" in config_text
    assert "math:" in config_text
    assert "enabled: false" in config_text
    assert "split: test" in config_text
    assert "max_samples:" in config_text
    assert "subjects:" in config_text
    assert "levels:" in config_text
    assert "seed: 42" in config_text
    assert "prompt_style: cot_boxed" in config_text
    assert "tokenizer_name: gpt2" in config_text
    assert "model_size: t4" in config_text
    assert "use_attnres:" in config_text
