from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from matdo_new import PACKAGE_ROOT
from matdo_new.experiments.config import (
    ExperimentCLIOverrides,
    add_cli_override_arguments,
    flag_was_provided,
    load_experiments_config,
    resolve_execution_plan,
    resolve_root_config_path,
)
from matdo_new.experiments.io import write_json_payload
from matdo_new.experiments.run_experiments import (
    ExperimentRunContext,
    run_experiments,
)
from matdo_new.experiments.schema import BenchmarkResult

DEFAULT_ROOT_CONFIG_PATH = PACKAGE_ROOT.parent / "configs" / "default.yaml"
DEFAULT_CONFIG_PATH = PACKAGE_ROOT.parent / "configs" / "experiments" / "default.yaml"
CLI_PAYLOAD_SCHEMA_VERSION = "matdo_new.cli_payload.v1"


def build_parser() -> argparse.ArgumentParser:
    """Build the MATDO-new experiments entrypoint parser."""
    parser = argparse.ArgumentParser(
        prog="python3 -m matdo_new.apps.run_experiments",
        description="Run MATDO-new benchmark and study entrypoints.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiments config YAML.",
    )
    parser.add_argument(
        "--root-config",
        type=Path,
        default=DEFAULT_ROOT_CONFIG_PATH,
        help="Path to the shared MATDO-new root config YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the normalized JSON payload.",
    )
    add_cli_override_arguments(parser)
    return parser


def effective_argv(argv: Sequence[str] | None) -> tuple[str, ...]:
    if argv is None:
        return tuple(sys.argv[1:])
    return tuple(argv)


def build_payload(
    *,
    config_path: Path,
    root_config_path: Path,
    results: Sequence[BenchmarkResult],
    emit_metadata: bool,
) -> dict[str, object]:
    return {
        "schema_version": CLI_PAYLOAD_SCHEMA_VERSION,
        "app": "run_experiments",
        "config_path": str(config_path),
        "root_config_path": str(root_config_path),
        "results": [result.to_dict(emit_metadata=emit_metadata) for result in results],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    cli_argv = effective_argv(argv)
    args = parser.parse_args(cli_argv)

    resolved_config_path = args.config.resolve()
    if not resolved_config_path.exists():
        parser.error(f"config file not found: {resolved_config_path}")

    experiments_config = load_experiments_config(resolved_config_path)
    resolved_root_config_path = resolve_root_config_path(
        cli_root_config_path=args.root_config,
        experiments_config=experiments_config,
        default_root_config_path=DEFAULT_ROOT_CONFIG_PATH,
        config_path=resolved_config_path,
        cli_root_config_provided=flag_was_provided(cli_argv, "--root-config"),
    ).resolve()
    if not resolved_root_config_path.exists():
        parser.error(f"config file not found: {resolved_root_config_path}")

    execution_plan = resolve_execution_plan(
        experiments_config,
        cli_overrides=ExperimentCLIOverrides.from_namespace(args),
    )
    if execution_plan.output.output_format != "json":
        parser.error(f"unsupported output.format: {execution_plan.output.output_format}")

    results = run_experiments(
        *execution_plan.runners,
        run_context=ExperimentRunContext(
            app="run_experiments",
            config_path=str(resolved_config_path),
            root_config_path=str(resolved_root_config_path),
        ),
    )

    payload = build_payload(
        config_path=resolved_config_path,
        root_config_path=resolved_root_config_path,
        results=results,
        emit_metadata=execution_plan.output.emit_metadata,
    )
    if args.output is not None:
        write_json_payload(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
