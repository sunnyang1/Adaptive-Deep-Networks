from __future__ import annotations

import json
from pathlib import Path

import torch

from matdo_new.apps.generate import main
from matdo_new.modeling.config import MATDOModelConfig
from matdo_new.modeling.matdo_model import MATDOModel
from src.models.configs import ModelConfig


def _tiny_backend_config() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        hidden_dim=16,
        num_heads=4,
        num_blocks=1,
        mlp_ratio=2,
        vocab_size=32,
        max_seq_len=32,
    )


def test_generate_main_dry_run_prints_request_payload(capsys) -> None:
    exit_code = main(["--dry-run"])

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["app"] == "generate"
    assert payload["dry_run"] is True
    assert "note" not in payload
    assert Path(payload["config_path"]).is_absolute()
    assert Path(payload["root_config_path"]).is_absolute()
    assert payload["prompt_token_ids"] == [11, 22]
    assert payload["max_new_tokens"] == 8
    assert payload["observation"] == {
        "rho_hbm": 0.92,
        "rho_dram": 0.3,
        "target_error": 0.05,
    }


def test_generate_main_live_prints_generation_payload(monkeypatch, capsys) -> None:
    torch.manual_seed(0)

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: _tiny_backend_config(),
    )
    monkeypatch.setattr(
        MATDOModel,
        "sample_next_token",
        lambda self, logits, **kwargs: int(torch.argmax(logits).item()),
    )

    exit_code = main(["--max-new-tokens", "3"])

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["app"] == "generate"
    assert payload["dry_run"] is False
    assert "note" not in payload
    assert Path(payload["config_path"]).is_absolute()
    assert Path(payload["root_config_path"]).is_absolute()
    assert isinstance(payload["generated_token_ids"], list)
    assert len(payload["generated_token_ids"]) == 3
    assert payload["sequence_length"] == len(payload["prompt_token_ids"]) + 3
    assert payload["policy"]["target_error"] == 0.05
    assert payload["runtime"]["metrics"]["prefill_calls"] == 1
    assert payload["runtime"]["metrics"]["decode_calls"] == 3
    assert payload["runtime"]["metrics"]["incremental_decode_calls"] == 0
    assert payload["runtime"]["metrics"]["decode_used_incremental"] is False
    assert payload["runtime"]["cache"]["supports_incremental_decode"] is False


def test_generate_main_live_uses_inference_config_file(monkeypatch, tmp_path: Path, capsys) -> None:
    torch.manual_seed(0)

    seen_model_sizes: list[str] = []

    original_build_backend_config = MATDOModelConfig.build_backend_config

    def patched_build_backend_config(self: MATDOModelConfig) -> ModelConfig:
        seen_model_sizes.append(self.model_size)
        return _tiny_backend_config()

    monkeypatch.setattr(MATDOModelConfig, "build_backend_config", patched_build_backend_config)
    monkeypatch.setattr(
        MATDOModel,
        "sample_next_token",
        lambda self, logits, **kwargs: int(torch.argmax(logits).item()),
    )

    inference_config = tmp_path / "custom-inference.yaml"
    custom_root = tmp_path / "custom-root.yaml"
    custom_root.write_text(
        "\n".join(
            [
                "config_version: 1",
                "policy:",
                "  quantization_bits: [2, 4, 8]",
                "  min_quantization_bits: 2",
                "  scope_span: 4",
                "  total_hbm_blocks: 256",
                "  min_scope_blocks: 1",
                "  max_t_steps: 4096",
                "  target_error: 0.02",
                "  arbitrage_zone_rho: 0.93",
                "  critical_zone_rho: 0.98",
                "  dram_utilization_limit: 0.90",
                "  e_max: 128000",
                "  e0: 10000.0",
                "  zeta: 0.35",
                "  eta: 0.5",
                "  alpha: 0.015",
                "  beta: 2.0",
                "  gamma: 0.10",
                "  delta: 0.005",
                "  epsilon: 0.002",
            ]
        ),
        encoding="utf-8",
    )
    inference_config.write_text(
        "\n".join(
            [
                "runtime:",
                "  mode: generate",
                f"  root_config: {custom_root.name}",
                "",
                "model:",
                "  model_size: tiny-config-model",
                "  use_attnres: false",
                "  use_qttt: false",
                "  use_engram: false",
                "",
                "prompt:",
                "  token_ids: [3, 5, 7]",
                "",
                "generation:",
                "  max_new_tokens: 2",
                "  stop_token_ids: []",
                "",
                "observation:",
                "  rho_hbm: 0.96",
                "  rho_dram: 0.25",
                "  target_error: 0.01",
                "",
                "output:",
                "  emit_metrics: true",
                "  format: json",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(inference_config)])

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["prompt_token_ids"] == [3, 5, 7]
    assert payload["max_new_tokens"] == 2
    assert payload["observation"] == {
        "rho_hbm": 0.96,
        "rho_dram": 0.25,
        "target_error": 0.01,
    }
    assert Path(payload["root_config_path"]) == custom_root.resolve()
    assert len(payload["generated_token_ids"]) == 2
    assert payload["policy"]["target_error"] == 0.01
    assert payload["runtime"]["metrics"]["prefill_calls"] == 1
    assert payload["runtime"]["metrics"]["decode_calls"] == 2
    assert payload["runtime"]["metrics"]["incremental_decode_calls"] == 2
    assert payload["runtime"]["metrics"]["decode_used_incremental"] is True
    assert payload["runtime"]["cache"]["supports_incremental_decode"] is True
    assert seen_model_sizes
    assert set(seen_model_sizes) == {"tiny-config-model"}


def test_generate_main_cli_overrides_inference_config(monkeypatch, tmp_path: Path, capsys) -> None:
    torch.manual_seed(0)

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: _tiny_backend_config(),
    )
    monkeypatch.setattr(
        MATDOModel,
        "sample_next_token",
        lambda self, logits, **kwargs: int(torch.argmax(logits).item()),
    )

    root_config = tmp_path / "override-root.yaml"
    root_config.write_text(
        "\n".join(
            [
                "config_version: 1",
                "policy:",
                "  quantization_bits: [2, 4, 8]",
                "  min_quantization_bits: 2",
                "  scope_span: 4",
                "  total_hbm_blocks: 256",
                "  min_scope_blocks: 1",
                "  max_t_steps: 4096",
                "  target_error: 0.05",
                "  arbitrage_zone_rho: 0.93",
                "  critical_zone_rho: 0.98",
                "  dram_utilization_limit: 0.90",
                "  e_max: 128000",
                "  e0: 10000.0",
                "  zeta: 0.35",
                "  eta: 0.5",
                "  alpha: 0.015",
                "  beta: 2.0",
                "  gamma: 0.10",
                "  delta: 0.005",
                "  epsilon: 0.002",
            ]
        ),
        encoding="utf-8",
    )
    inference_config = tmp_path / "override-inference.yaml"
    inference_config.write_text(
        "\n".join(
            [
                "runtime:",
                "  mode: generate",
                f"  root_config: {root_config.name}",
                "",
                "model:",
                "  model_size: tiny-config-model",
                "",
                "prompt:",
                "  token_ids: [3, 5, 7]",
                "",
                "generation:",
                "  max_new_tokens: 2",
                "",
                "observation:",
                "  rho_hbm: 0.96",
                "  rho_dram: 0.25",
                "  target_error: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(inference_config),
            "--prompt-token-ids",
            "9,10",
            "--max-new-tokens",
            "1",
            "--rho-hbm",
            "0.5",
            "--rho-dram",
            "0.4",
            "--target-error",
            "0.02",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["prompt_token_ids"] == [9, 10]
    assert payload["max_new_tokens"] == 1
    assert payload["observation"] == {
        "rho_hbm": 0.5,
        "rho_dram": 0.4,
        "target_error": 0.02,
    }
    assert len(payload["generated_token_ids"]) == 1
    assert payload["policy"]["target_error"] == 0.02


def test_generate_main_cli_root_config_overrides_inference_root_config(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    torch.manual_seed(0)

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: _tiny_backend_config(),
    )
    monkeypatch.setattr(
        MATDOModel,
        "sample_next_token",
        lambda self, logits, **kwargs: int(torch.argmax(logits).item()),
    )

    bad_root = tmp_path / "bad-root.yaml"
    bad_root.write_text(
        "\n".join(
            [
                "config_version: 1",
                "policy:",
                "  not_a_real_policy_field: 1",
            ]
        ),
        encoding="utf-8",
    )
    good_root = tmp_path / "good-root.yaml"
    good_root.write_text(
        "\n".join(
            [
                "config_version: 1",
                "policy:",
                "  quantization_bits: [2, 4, 8]",
                "  min_quantization_bits: 2",
                "  scope_span: 4",
                "  total_hbm_blocks: 256",
                "  min_scope_blocks: 1",
                "  max_t_steps: 4096",
                "  target_error: 0.02",
                "  arbitrage_zone_rho: 0.93",
                "  critical_zone_rho: 0.98",
                "  dram_utilization_limit: 0.90",
                "  e_max: 128000",
                "  e0: 10000.0",
                "  zeta: 0.35",
                "  eta: 0.5",
                "  alpha: 0.015",
                "  beta: 2.0",
                "  gamma: 0.10",
                "  delta: 0.005",
                "  epsilon: 0.002",
            ]
        ),
        encoding="utf-8",
    )
    inference_config = tmp_path / "root-override-inference.yaml"
    inference_config.write_text(
        "\n".join(
            [
                "runtime:",
                f"  root_config: {bad_root.name}",
                "model:",
                "  model_size: tiny-config-model",
                "prompt:",
                "  token_ids: [1, 2]",
                "generation:",
                "  max_new_tokens: 1",
                "observation:",
                "  rho_hbm: 0.96",
                "  rho_dram: 0.25",
                "  target_error: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(inference_config),
            "--root-config",
            str(good_root),
            "--max-new-tokens",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert Path(payload["root_config_path"]) == good_root.resolve()
    assert payload["policy"]["target_error"] == 0.01


def test_generate_main_real_entrypoint_argv_honors_root_config_override(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    torch.manual_seed(0)

    monkeypatch.setattr(
        MATDOModelConfig,
        "build_backend_config",
        lambda self: _tiny_backend_config(),
    )
    monkeypatch.setattr(
        MATDOModel,
        "sample_next_token",
        lambda self, logits, **kwargs: int(torch.argmax(logits).item()),
    )

    bad_root = tmp_path / "bad-root.yaml"
    bad_root.write_text(
        "\n".join(
            [
                "config_version: 1",
                "policy:",
                "  not_a_real_policy_field: 1",
            ]
        ),
        encoding="utf-8",
    )
    good_root = tmp_path / "good-root.yaml"
    good_root.write_text(
        "\n".join(
            [
                "config_version: 1",
                "policy:",
                "  quantization_bits: [2, 4, 8]",
                "  min_quantization_bits: 2",
                "  scope_span: 4",
                "  total_hbm_blocks: 256",
                "  min_scope_blocks: 1",
                "  max_t_steps: 4096",
                "  target_error: 0.02",
                "  arbitrage_zone_rho: 0.93",
                "  critical_zone_rho: 0.98",
                "  dram_utilization_limit: 0.90",
                "  e_max: 128000",
                "  e0: 10000.0",
                "  zeta: 0.35",
                "  eta: 0.5",
                "  alpha: 0.015",
                "  beta: 2.0",
                "  gamma: 0.10",
                "  delta: 0.005",
                "  epsilon: 0.002",
            ]
        ),
        encoding="utf-8",
    )
    inference_config = tmp_path / "entrypoint-inference.yaml"
    inference_config.write_text(
        "\n".join(
            [
                "runtime:",
                f"  root_config: {bad_root.name}",
                "model:",
                "  model_size: tiny-config-model",
                "prompt:",
                "  token_ids: [1, 2]",
                "generation:",
                "  max_new_tokens: 1",
                "observation:",
                "  rho_hbm: 0.96",
                "  rho_dram: 0.25",
                "  target_error: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "python3 -m matdo_new.apps.generate",
            "--config",
            str(inference_config),
            "--root-config",
            str(good_root),
        ],
    )

    exit_code = main()

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert Path(payload["root_config_path"]) == good_root.resolve()
    assert payload["policy"]["target_error"] == 0.01
