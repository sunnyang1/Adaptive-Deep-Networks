from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from matdo_new import PACKAGE_ROOT
from matdo_new.core.config import MATDOConfig
from matdo_new.core.policy import RuntimeObservation, solve_policy
from matdo_new.modeling.config import MATDOModelConfig
from matdo_new.runtime.backend import AdaptiveTransformerRuntimeBackend
from matdo_new.runtime.generation import GenerationResult, generate_tokens

DEFAULT_ROOT_CONFIG_PATH = PACKAGE_ROOT.parent / "configs" / "default.yaml"
DEFAULT_CONFIG_PATH = PACKAGE_ROOT.parent / "configs" / "inference" / "default.yaml"
DEFAULT_PROMPT_TOKEN_IDS = (11, 22)
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_RHO_HBM = 0.92
DEFAULT_RHO_DRAM = 0.30
DEFAULT_TARGET_ERROR = 0.05


def parse_token_ids(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integer token ids."""
    raw_items = [item.strip() for item in value.split(",")]
    token_ids = tuple(int(item) for item in raw_items if item)
    if not token_ids:
        raise argparse.ArgumentTypeError("expected at least one token id")
    return token_ids


def build_parser() -> argparse.ArgumentParser:
    """Build the lightweight MATDO-new generation entrypoint parser."""
    parser = argparse.ArgumentParser(
        prog="python3 -m matdo_new.apps.generate",
        description="Resolve and execute a MATDO-new generation request.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the inference config YAML.",
    )
    parser.add_argument(
        "--root-config",
        type=Path,
        default=DEFAULT_ROOT_CONFIG_PATH,
        help="Path to the shared MATDO-new root config YAML.",
    )
    parser.add_argument(
        "--prompt-token-ids",
        type=parse_token_ids,
        default=None,
        help="Comma-separated prompt token ids.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to request from the runtime.",
    )
    parser.add_argument("--rho-hbm", type=float, default=None, help="Observed HBM utilization ratio.")
    parser.add_argument("--rho-dram", type=float, default=None, help="Observed DRAM utilization ratio.")
    parser.add_argument(
        "--target-error",
        type=float,
        default=None,
        help="Target runtime error budget for policy selection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the resolved request without attaching a backend.",
    )
    return parser


def build_request(
    *,
    config_path: Path,
    root_config_path: Path,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    observation: RuntimeObservation,
    dry_run: bool,
) -> dict[str, object]:
    """Normalize CLI arguments into a serializable request summary."""
    return {
        "app": "generate",
        "config_path": str(config_path),
        "root_config_path": str(root_config_path),
        "prompt_token_ids": list(prompt_token_ids),
        "max_new_tokens": int(max_new_tokens),
        "observation": {
            "rho_hbm": float(observation.rho_hbm),
            "rho_dram": float(observation.rho_dram),
            "target_error": float(observation.target_error if observation.target_error is not None else DEFAULT_TARGET_ERROR),
        },
        "dry_run": bool(dry_run),
    }


def load_simple_config(path: Path) -> dict[str, object]:
    """Load the limited YAML subset used by MATDO-new package configs."""
    config: dict[str, object] = {}
    current_section: dict[str, object] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not raw_line[:1].isspace():
            key, _, raw_value = stripped.partition(":")
            if raw_value.strip():
                config[key] = _parse_scalar(raw_value.strip())
                current_section = None
            else:
                current_section = {}
                config[key] = current_section
            continue

        if current_section is None:
            continue

        key, _, raw_value = stripped.partition(":")
        if key:
            current_section[key] = _parse_scalar(raw_value.strip())

    return config


def load_policy_config(path: Path) -> MATDOConfig:
    """Load the policy section from the root MATDO-new config."""
    raw_config = load_simple_config(path)
    policy_config = raw_config.get("policy", {})
    if not isinstance(policy_config, dict):
        raise ValueError(f"expected mapping at policy section in {path}")
    return MATDOConfig(**policy_config)


def _parse_scalar(raw_value: str) -> object:
    """Parse the small subset of YAML scalars used in package configs."""
    if raw_value == "":
        return None
    if raw_value in {"true", "false"}:
        return raw_value == "true"
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def build_runtime_model_config(model_config_section: dict[str, object] | None) -> MATDOModelConfig:
    """Build the runtime model config from inference config values."""
    resolved = model_config_section or {}
    return MATDOModelConfig(
        model_size=str(resolved.get("model_size", "t4")),
        use_attnres=bool(resolved.get("use_attnres", True)),
        use_qttt=bool(resolved.get("use_qttt", False)),
        use_engram=bool(resolved.get("use_engram", False)),
        sampling_temperature=float(resolved.get("sampling_temperature", 1.0)),
        sampling_top_k=(
            None
            if resolved.get("sampling_top_k") is None
            else int(resolved["sampling_top_k"])
        ),
    )


def build_live_payload(
    *,
    request: dict[str, object],
    result: GenerationResult,
    policy: object,
) -> dict[str, object]:
    """Serialize the real generation result for CLI output."""
    runtime_metrics = {
        "prefill_calls": int(result.state.metrics.prefill_calls),
        "decode_calls": int(result.state.metrics.decode_calls),
        "prompt_tokens": int(result.state.metrics.prompt_tokens),
        "decode_tokens": int(result.state.metrics.decode_tokens),
        "prefill_submitted_tokens": int(result.state.metrics.prefill_submitted_tokens),
        "decode_submitted_tokens": int(result.state.metrics.decode_submitted_tokens),
        "submitted_tokens": int(result.state.metrics.submitted_tokens),
        "incremental_decode_calls": int(result.state.metrics.incremental_decode_calls),
        "decode_used_incremental": bool(result.state.metrics.decode_used_incremental),
    }
    payload = dict(request)
    payload.update(
        {
            "generated_token_ids": list(result.generated_token_ids),
            "sequence_length": int(result.sequence_length),
            "token_ids": list(result.token_ids),
            "policy": asdict(policy),
            "runtime": {
                "metrics": runtime_metrics,
            },
        }
    )
    supports_incremental_decode = getattr(result.state.cache, "supports_incremental_decode", None)
    if supports_incremental_decode is not None:
        payload["runtime"]["cache"] = {
            "supports_incremental_decode": bool(supports_incremental_decode),
        }
    payload.pop("note", None)
    return payload


def sample_from_runtime_model(backend: AdaptiveTransformerRuntimeBackend, logits: object | None) -> int:
    """Sample one token id using the MATDO runtime model helper."""
    if backend.runtime_model is None:
        raise RuntimeError("runtime backend did not initialize a MATDO runtime model")
    return int(backend.runtime_model.sample_next_token(logits))


def resolve_root_config_path(
    args: argparse.Namespace,
    inference_config: dict[str, object],
    *,
    argv: Sequence[str] | None,
) -> Path:
    """Resolve the effective root config path, allowing inference config defaults."""
    runtime_config = inference_config.get("runtime", {})
    configured_root = None
    if isinstance(runtime_config, dict):
        configured_root = runtime_config.get("root_config")

    if _flag_was_provided(argv, "--root-config") or configured_root is None:
        return args.root_config
    return resolve_config_reference(Path(str(configured_root)), base_path=args.config)


def resolve_prompt_token_ids(
    args: argparse.Namespace,
    inference_config: dict[str, object],
) -> tuple[int, ...]:
    if args.prompt_token_ids is not None:
        return tuple(int(token_id) for token_id in args.prompt_token_ids)
    prompt_config = inference_config.get("prompt", {})
    if isinstance(prompt_config, dict) and "token_ids" in prompt_config:
        return tuple(int(token_id) for token_id in prompt_config["token_ids"])
    return DEFAULT_PROMPT_TOKEN_IDS


def resolve_runtime_observation(
    args: argparse.Namespace,
    inference_config: dict[str, object],
) -> RuntimeObservation:
    observation_config = inference_config.get("observation", {})
    if not isinstance(observation_config, dict):
        observation_config = {}

    return RuntimeObservation(
        rho_hbm=float(
            args.rho_hbm if args.rho_hbm is not None else observation_config.get("rho_hbm", DEFAULT_RHO_HBM)
        ),
        rho_dram=float(
            args.rho_dram if args.rho_dram is not None else observation_config.get("rho_dram", DEFAULT_RHO_DRAM)
        ),
        target_error=float(
            args.target_error
            if args.target_error is not None
            else observation_config.get("target_error", DEFAULT_TARGET_ERROR)
        ),
    )


def resolve_max_new_tokens(args: argparse.Namespace, inference_config: dict[str, object]) -> int:
    if args.max_new_tokens is not None:
        return int(args.max_new_tokens)
    generation_config = inference_config.get("generation", {})
    if isinstance(generation_config, dict) and "max_new_tokens" in generation_config:
        return int(generation_config["max_new_tokens"])
    return DEFAULT_MAX_NEW_TOKENS


def resolve_model_section(inference_config: dict[str, object]) -> dict[str, object]:
    model_config = inference_config.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("expected mapping at model section in inference config")
    return model_config


def _flag_was_provided(argv: Sequence[str] | None, flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in effective_argv(argv))


def effective_argv(argv: Sequence[str] | None) -> tuple[str, ...]:
    """Return the effective CLI argv, including real process args when needed."""
    if argv is None:
        return tuple(sys.argv[1:])
    return tuple(argv)


def resolve_config_reference(path: Path, *, base_path: Path) -> Path:
    """Resolve config references relative to the active config file or package root."""
    if path.is_absolute():
        return path

    candidate_roots = (
        base_path.parent,
        PACKAGE_ROOT.parent,
        PACKAGE_ROOT.parent.parent,
    )
    for root in candidate_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate

    return (base_path.parent / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    cli_argv = effective_argv(argv)
    args = parser.parse_args(cli_argv)

    for path in (args.config, args.root_config):
        if not path.exists():
            parser.error(f"config file not found: {path}")

    resolved_config_path = args.config.resolve()
    inference_config = load_simple_config(resolved_config_path)
    resolved_root_config = resolve_root_config_path(args, inference_config, argv=cli_argv).resolve()
    if not resolved_root_config.exists():
        parser.error(f"config file not found: {resolved_root_config}")

    prompt_token_ids = resolve_prompt_token_ids(args, inference_config)
    max_new_tokens = resolve_max_new_tokens(args, inference_config)
    if max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    observation = resolve_runtime_observation(args, inference_config)

    request = build_request(
        config_path=resolved_config_path,
        root_config_path=resolved_root_config,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        observation=observation,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(request, indent=2, sort_keys=True))
        return 0

    policy = solve_policy(observation, config=load_policy_config(resolved_root_config))
    model_config = build_runtime_model_config(resolve_model_section(inference_config))
    backend = AdaptiveTransformerRuntimeBackend.from_model_config(model_config)

    result = generate_tokens(
        prompt_token_ids,
        backend=backend,
        sampler=lambda logits: sample_from_runtime_model(backend, logits),
        max_new_tokens=max_new_tokens,
        policy=policy,
    )

    print(json.dumps(build_live_payload(request=request, result=result, policy=policy), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
