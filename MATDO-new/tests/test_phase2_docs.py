from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_phase2_readme_mentions_live_generation_and_experiment_output_files() -> None:
    package_readme = (_PACKAGE_ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "live mode" in package_readme
    assert "--output" in package_readme
    assert "generated token ids" in package_readme
    assert "output.emit_metadata" in package_readme


def test_phase2_experiments_default_config_uses_portable_root_reference() -> None:
    experiments_config = (_PACKAGE_ROOT / "configs/experiments/default.yaml").read_text(
        encoding="utf-8"
    )

    assert "root_config: ../default.yaml" in experiments_config
