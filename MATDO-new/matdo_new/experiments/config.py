from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from matdo_new import PACKAGE_ROOT
from matdo_new.experiments.builders import (
    build_critical_points_runner,
    build_math_runner,
    build_needle_runner,
)
from matdo_new.experiments.run_experiments import BenchmarkRunner

DEFAULT_NEEDLE_CONTEXT_LENGTHS = (4096,)
DEFAULT_NEEDLE_NUM_SAMPLES = 1
DEFAULT_NEEDLE_DEPTH_DISTRIBUTION = "uniform"
DEFAULT_NEEDLE_MAX_NEW_TOKENS = 16
DEFAULT_NEEDLE_TOKENIZER_NAME = "gpt2"
DEFAULT_NEEDLE_MODEL_SIZE = "t4"
DEFAULT_NEEDLE_USE_ATTNRES = True
DEFAULT_NEEDLE_SEED = 42
DEFAULT_MATH_SPLIT = "test"
DEFAULT_MATH_MAX_SAMPLES: int | None = None
DEFAULT_MATH_PROMPT_STYLE = "cot_boxed"
DEFAULT_MATH_MAX_NEW_TOKENS = 256
DEFAULT_MATH_TOKENIZER_NAME = "gpt2"
DEFAULT_MATH_MODEL_SIZE = "t4"
DEFAULT_MATH_USE_ATTNRES = True
DEFAULT_MATH_SEED = 42
DEFAULT_CRITICAL_POINTS = (0.25, 0.5, 0.75)
SUPPORTED_BENCHMARKS = ("needle", "math", "critical-points")
BENCHMARK_ALIASES = {
    "needle": "needle",
    "math": "math",
    "critical-points": "critical-points",
    "critical_points": "critical-points",
}


def add_cli_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark",
        dest="benchmarks",
        action="append",
        choices=tuple(BENCHMARK_ALIASES),
        default=None,
        help="Benchmark name to run. May be provided multiple times.",
    )
    parser.add_argument(
        "--needle-context-lengths",
        nargs="+",
        type=int,
        default=None,
        help="Override Needle context lengths.",
    )
    parser.add_argument(
        "--needle-num-samples",
        type=int,
        default=None,
        help="Override Needle sample count.",
    )
    parser.add_argument(
        "--needle-depth-distribution",
        type=str,
        default=None,
        help="Override Needle depth distribution.",
    )
    parser.add_argument(
        "--needle-max-new-tokens",
        type=int,
        default=None,
        help="Override Needle max_new_tokens.",
    )
    parser.add_argument(
        "--needle-tokenizer-name",
        type=str,
        default=None,
        help="Override Needle tokenizer name.",
    )
    parser.add_argument(
        "--needle-model-size",
        type=str,
        default=None,
        help="Override Needle model size.",
    )
    needle_attnres_group = parser.add_mutually_exclusive_group()
    needle_attnres_group.add_argument(
        "--needle-use-attnres",
        dest="needle_use_attnres",
        action="store_true",
        default=None,
        help="Force Needle to enable AttnRes.",
    )
    needle_attnres_group.add_argument(
        "--no-needle-use-attnres",
        dest="needle_use_attnres",
        action="store_false",
        help="Force Needle to disable AttnRes.",
    )
    parser.add_argument(
        "--needle-seed",
        type=int,
        default=None,
        help="Override Needle sampling seed.",
    )
    parser.add_argument(
        "--math-split",
        type=str,
        default=None,
        help="Override MATH split.",
    )
    parser.add_argument(
        "--math-max-samples",
        type=int,
        default=None,
        help="Override MATH max sample count.",
    )
    parser.add_argument(
        "--math-subjects",
        nargs="+",
        type=str,
        default=None,
        help="Override MATH subjects filter.",
    )
    parser.add_argument(
        "--math-levels",
        nargs="+",
        type=int,
        default=None,
        help="Override MATH levels filter.",
    )
    parser.add_argument(
        "--math-prompt-style",
        type=str,
        default=None,
        help="Override MATH prompt style.",
    )
    parser.add_argument(
        "--math-max-new-tokens",
        type=int,
        default=None,
        help="Override MATH max_new_tokens.",
    )
    parser.add_argument(
        "--math-tokenizer-name",
        type=str,
        default=None,
        help="Override MATH tokenizer name.",
    )
    parser.add_argument(
        "--math-model-size",
        type=str,
        default=None,
        help="Override MATH model size.",
    )
    math_attnres_group = parser.add_mutually_exclusive_group()
    math_attnres_group.add_argument(
        "--math-use-attnres",
        dest="math_use_attnres",
        action="store_true",
        default=None,
        help="Force MATH to enable AttnRes.",
    )
    math_attnres_group.add_argument(
        "--no-math-use-attnres",
        dest="math_use_attnres",
        action="store_false",
        help="Force MATH to disable AttnRes.",
    )
    parser.add_argument(
        "--math-seed",
        type=int,
        default=None,
        help="Override MATH sampling seed.",
    )
    parser.add_argument(
        "--critical-points",
        nargs="+",
        type=float,
        default=None,
        help="Override Critical Points study locations.",
    )
    emit_metadata_group = parser.add_mutually_exclusive_group()
    emit_metadata_group.add_argument(
        "--emit-metadata",
        dest="emit_metadata",
        action="store_true",
        default=None,
        help="Include metadata fields in the output payload.",
    )
    emit_metadata_group.add_argument(
        "--no-emit-metadata",
        dest="emit_metadata",
        action="store_false",
        help="Omit metadata fields in the output payload.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default=None,
        help="Override output format.",
    )


def effective_argv(argv: Sequence[str] | None) -> tuple[str, ...]:
    if argv is None:
        return ()
    return tuple(argv)


def flag_was_provided(argv: Sequence[str] | None, flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in effective_argv(argv))


def _parse_scalar(raw_value: str) -> object:
    if raw_value == "":
        return None
    if raw_value in {"true", "false"}:
        return raw_value == "true"
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def _strip_inline_comment(raw_value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for index, character in enumerate(raw_value):
        if character == "\\" and not escaped:
            escaped = True
            continue
        if character == "'" and not in_double_quote and not escaped:
            in_single_quote = not in_single_quote
        elif character == '"' and not in_single_quote and not escaped:
            in_double_quote = not in_double_quote
        elif character == "#" and not in_single_quote and not in_double_quote:
            if index == 0 or raw_value[index - 1].isspace():
                return raw_value[:index].rstrip()
        escaped = False

    return raw_value.strip()


def load_experiments_config(path: Path) -> dict[str, object]:
    config: dict[str, object] = {}
    stack: list[tuple[int, object]] = [(-1, config)]
    pending_list_keys: dict[int, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))

        while indent <= stack[-1][0]:
            stack.pop()
        pending_list_keys = {
            pending_indent: key for pending_indent, key in pending_list_keys.items() if pending_indent < indent
        }

        current = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(current, list):
                parent_indent = stack[-1][0]
                list_key = pending_list_keys.get(parent_indent)
                if list_key is None:
                    raise ValueError(f"unsupported YAML list item at indent {indent}: {stripped}")
                new_list: list[object] = []
                if len(stack) < 2 or not isinstance(stack[-2][1], dict):
                    raise ValueError(f"unsupported YAML parent for list key '{list_key}'")
                parent = stack[-2][1]
                parent[list_key] = new_list
                stack[-1] = (parent_indent, new_list)
                current = new_list

            current.append(_parse_scalar(_strip_inline_comment(stripped[2:].strip())))
            continue

        key, _, raw_value = stripped.partition(":")
        if not key:
            continue

        value = _strip_inline_comment(raw_value.strip())
        if value:
            if not isinstance(current, dict):
                raise ValueError(f"unsupported YAML mapping entry: {stripped}")
            current[key] = _parse_scalar(value)
            continue

        if not isinstance(current, dict):
            raise ValueError(f"unsupported YAML nested mapping: {stripped}")
        child: dict[str, object] = {}
        current[key] = child
        stack.append((indent, child))
        pending_list_keys[indent] = key

    return config


def resolve_config_reference(path: Path, *, base_path: Path) -> Path:
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


def resolve_root_config_path(
    *,
    cli_root_config_path: Path,
    experiments_config: Mapping[str, object],
    default_root_config_path: Path,
    config_path: Path,
    cli_root_config_provided: bool,
) -> Path:
    runtime_config = _as_dict(experiments_config.get("runtime"))
    configured_root = runtime_config.get("root_config")

    if cli_root_config_provided or configured_root is None:
        return cli_root_config_path or default_root_config_path

    return resolve_config_reference(Path(str(configured_root)), base_path=config_path)


@dataclass(frozen=True)
class ExperimentOutputConfig:
    emit_metadata: bool = True
    output_format: str = "json"


@dataclass(frozen=True)
class ExperimentExecutionPlan:
    runners: tuple[BenchmarkRunner, ...]
    output: ExperimentOutputConfig


@dataclass(frozen=True)
class ExperimentCLIOverrides:
    benchmarks: tuple[str, ...] = ()
    needle_context_lengths: tuple[int, ...] | None = None
    needle_num_samples: int | None = None
    needle_depth_distribution: str | None = None
    needle_max_new_tokens: int | None = None
    needle_tokenizer_name: str | None = None
    needle_model_size: str | None = None
    needle_use_attnres: bool | None = None
    needle_seed: int | None = None
    math_split: str | None = None
    math_max_samples: int | None = None
    math_subjects: tuple[str, ...] | None = None
    math_levels: tuple[int, ...] | None = None
    math_prompt_style: str | None = None
    math_max_new_tokens: int | None = None
    math_tokenizer_name: str | None = None
    math_model_size: str | None = None
    math_use_attnres: bool | None = None
    math_seed: int | None = None
    critical_points: tuple[float, ...] | None = None
    emit_metadata: bool | None = None
    output_format: str | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "ExperimentCLIOverrides":
        benchmarks = tuple(normalize_benchmark_name(name) for name in (args.benchmarks or ()))
        needle_context_lengths = (
            tuple(int(value) for value in args.needle_context_lengths)
            if args.needle_context_lengths is not None
            else None
        )
        math_subjects = (
            tuple(str(value) for value in args.math_subjects)
            if args.math_subjects is not None
            else None
        )
        math_levels = (
            tuple(int(value) for value in args.math_levels)
            if args.math_levels is not None
            else None
        )
        critical_points = (
            tuple(float(value) for value in args.critical_points)
            if args.critical_points is not None
            else None
        )
        return cls(
            benchmarks=benchmarks,
            needle_context_lengths=needle_context_lengths,
            needle_num_samples=args.needle_num_samples,
            needle_depth_distribution=args.needle_depth_distribution,
            needle_max_new_tokens=args.needle_max_new_tokens,
            needle_tokenizer_name=args.needle_tokenizer_name,
            needle_model_size=args.needle_model_size,
            needle_use_attnres=args.needle_use_attnres,
            needle_seed=args.needle_seed,
            math_split=args.math_split,
            math_max_samples=args.math_max_samples,
            math_subjects=math_subjects,
            math_levels=math_levels,
            math_prompt_style=args.math_prompt_style,
            math_max_new_tokens=args.math_max_new_tokens,
            math_tokenizer_name=args.math_tokenizer_name,
            math_model_size=args.math_model_size,
            math_use_attnres=args.math_use_attnres,
            math_seed=args.math_seed,
            critical_points=critical_points,
            emit_metadata=args.emit_metadata,
            output_format=args.output_format,
        )


def normalize_benchmark_name(name: str) -> str:
    canonical = BENCHMARK_ALIASES.get(name)
    if canonical is None:
        raise ValueError(f"unsupported benchmark: {name}")
    return canonical


def resolve_execution_plan(
    experiments_config: Mapping[str, object],
    *,
    cli_overrides: ExperimentCLIOverrides | None = None,
) -> ExperimentExecutionPlan:
    overrides = cli_overrides or ExperimentCLIOverrides()
    benchmark_configs = _resolve_benchmark_configs(experiments_config)
    selected_benchmarks = _resolve_selected_benchmarks(benchmark_configs, overrides)
    runners = tuple(
        _build_runner(name, benchmark_configs.get(name, {}), overrides) for name in selected_benchmarks
    )
    return ExperimentExecutionPlan(
        runners=runners,
        output=_resolve_output_config(experiments_config, overrides),
    )


def _resolve_output_config(
    experiments_config: Mapping[str, object], overrides: ExperimentCLIOverrides
) -> ExperimentOutputConfig:
    output_config = _as_dict(experiments_config.get("output"))
    emit_metadata = bool(output_config.get("emit_metadata", True))
    output_format = str(output_config.get("format", "json"))

    if overrides.emit_metadata is not None:
        emit_metadata = overrides.emit_metadata
    if overrides.output_format is not None:
        output_format = overrides.output_format

    return ExperimentOutputConfig(
        emit_metadata=emit_metadata,
        output_format=output_format,
    )


def _resolve_selected_benchmarks(
    benchmark_configs: Mapping[str, Mapping[str, object]],
    overrides: ExperimentCLIOverrides,
) -> tuple[str, ...]:
    if overrides.benchmarks:
        ordered = []
        for name in overrides.benchmarks:
            if name not in ordered:
                ordered.append(name)
        return tuple(ordered)

    if benchmark_configs:
        return tuple(
            name
            for name in SUPPORTED_BENCHMARKS
            if name in benchmark_configs and bool(_as_dict(benchmark_configs.get(name)).get("enabled", True))
        )
    return ()


def _resolve_benchmark_configs(
    experiments_config: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    benchmark_entries = _as_dict(experiments_config.get("benchmarks"))
    tasks = _as_dict(experiments_config.get("tasks"))
    studies = _as_dict(experiments_config.get("studies"))
    has_legacy_entries = bool(tasks) or bool(studies)

    if benchmark_entries:
        if has_legacy_entries:
            raise ValueError(
                "mixed experiment config schemas are not supported; use only 'benchmarks'"
            )
        resolved: dict[str, dict[str, object]] = {}
        for raw_name, raw_config in benchmark_entries.items():
            resolved[normalize_benchmark_name(raw_name)] = _as_dict(raw_config)
        return resolved

    if has_legacy_entries:
        raise ValueError(
            "legacy experiment config schema is no longer supported; migrate to 'benchmarks'"
        )

    raise ValueError("missing required 'benchmarks' section in experiments config")


def _build_runner(
    benchmark_name: str,
    benchmark_config: Mapping[str, object],
    overrides: ExperimentCLIOverrides,
) -> BenchmarkRunner:
    if benchmark_name == "needle":
        return _build_needle_runner(benchmark_config, overrides)
    if benchmark_name == "math":
        return _build_math(benchmark_config, overrides)
    if benchmark_name == "critical-points":
        return _build_critical_points(benchmark_config, overrides)
    raise ValueError(f"unsupported benchmark: {benchmark_name}")


def _build_needle_runner(
    benchmark_config: Mapping[str, object],
    overrides: ExperimentCLIOverrides,
) -> BenchmarkRunner:
    dataset_config = _as_dict(benchmark_config.get("dataset"))
    generation_config = _as_dict(benchmark_config.get("generation"))
    runtime_config = _as_dict(benchmark_config.get("runtime"))

    context_lengths = overrides.needle_context_lengths
    if context_lengths is None:
        raw_context_lengths = _first_non_none(
            dataset_config.get("context_lengths"),
            benchmark_config.get("context_lengths"),
        )
        context_lengths = _coerce_int_tuple(
            raw_context_lengths,
            default=DEFAULT_NEEDLE_CONTEXT_LENGTHS,
        )

    num_samples = overrides.needle_num_samples
    if num_samples is None:
        num_samples = int(
            _first_non_none(
                dataset_config.get("num_samples"),
                benchmark_config.get("num_samples"),
                DEFAULT_NEEDLE_NUM_SAMPLES,
            )
        )

    depth_distribution = overrides.needle_depth_distribution
    if depth_distribution is None:
        depth_distribution = str(
            _first_non_none(
                dataset_config.get("depth_distribution"),
                benchmark_config.get("depth_distribution"),
                DEFAULT_NEEDLE_DEPTH_DISTRIBUTION,
            )
        )

    seed = overrides.needle_seed
    if seed is None:
        seed = int(
            _first_non_none(
                dataset_config.get("seed"),
                benchmark_config.get("seed"),
                DEFAULT_NEEDLE_SEED,
            )
        )

    max_new_tokens = overrides.needle_max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(
            _first_non_none(
                generation_config.get("max_new_tokens"),
                benchmark_config.get("max_new_tokens"),
                DEFAULT_NEEDLE_MAX_NEW_TOKENS,
            )
        )

    tokenizer_name = overrides.needle_tokenizer_name
    if tokenizer_name is None:
        tokenizer_name = str(
            _first_non_none(
                runtime_config.get("tokenizer_name"),
                benchmark_config.get("tokenizer_name"),
                DEFAULT_NEEDLE_TOKENIZER_NAME,
            )
        )

    model_size = overrides.needle_model_size
    if model_size is None:
        model_size = str(
            _first_non_none(
                runtime_config.get("model_size"),
                benchmark_config.get("model_size"),
                DEFAULT_NEEDLE_MODEL_SIZE,
            )
        )

    use_attnres = overrides.needle_use_attnres
    if use_attnres is None:
        use_attnres = bool(
            _first_non_none(
                runtime_config.get("use_attnres"),
                benchmark_config.get("use_attnres"),
                DEFAULT_NEEDLE_USE_ATTNRES,
            )
        )

    return build_needle_runner(
        context_lengths=context_lengths,
        num_samples=num_samples,
        depth_distribution=depth_distribution,
        max_new_tokens=max_new_tokens,
        tokenizer_name=tokenizer_name,
        model_size=model_size,
        use_attnres=use_attnres,
        seed=seed,
    )


def _build_critical_points(
    benchmark_config: Mapping[str, object],
    overrides: ExperimentCLIOverrides,
) -> BenchmarkRunner:
    dataset_config = _as_dict(benchmark_config.get("dataset"))
    points = overrides.critical_points
    if points is None:
        points = _coerce_float_tuple(
            _first_non_none(
                dataset_config.get("points"),
                benchmark_config.get("points"),
            ),
            default=DEFAULT_CRITICAL_POINTS,
        )
    return build_critical_points_runner(critical_points=points)


def _build_math(
    benchmark_config: Mapping[str, object],
    overrides: ExperimentCLIOverrides,
) -> BenchmarkRunner:
    dataset_config = _as_dict(benchmark_config.get("dataset"))
    generation_config = _as_dict(benchmark_config.get("generation"))
    runtime_config = _as_dict(benchmark_config.get("runtime"))

    split = overrides.math_split
    if split is None:
        split = str(
            _first_non_none(
                dataset_config.get("split"),
                benchmark_config.get("split"),
                DEFAULT_MATH_SPLIT,
            )
        )

    max_samples = overrides.math_max_samples
    if max_samples is None:
        raw_max_samples = _first_non_none(
            dataset_config.get("max_samples"),
            benchmark_config.get("max_samples"),
            DEFAULT_MATH_MAX_SAMPLES,
        )
        max_samples = None if raw_max_samples is None else int(raw_max_samples)

    subjects = overrides.math_subjects
    if subjects is None:
        subjects = _coerce_str_tuple(
            _first_non_none(
                dataset_config.get("subjects"),
                benchmark_config.get("subjects"),
            ),
            default=(),
        )

    levels = overrides.math_levels
    if levels is None:
        levels = _coerce_int_tuple(
            _first_non_none(
                dataset_config.get("levels"),
                benchmark_config.get("levels"),
            ),
            default=(),
        )

    prompt_style = overrides.math_prompt_style
    if prompt_style is None:
        prompt_style = str(
            _first_non_none(
                generation_config.get("prompt_style"),
                benchmark_config.get("prompt_style"),
                DEFAULT_MATH_PROMPT_STYLE,
            )
        )

    max_new_tokens = overrides.math_max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(
            _first_non_none(
                generation_config.get("max_new_tokens"),
                benchmark_config.get("max_new_tokens"),
                DEFAULT_MATH_MAX_NEW_TOKENS,
            )
        )

    tokenizer_name = overrides.math_tokenizer_name
    if tokenizer_name is None:
        tokenizer_name = str(
            _first_non_none(
                runtime_config.get("tokenizer_name"),
                benchmark_config.get("tokenizer_name"),
                DEFAULT_MATH_TOKENIZER_NAME,
            )
        )

    model_size = overrides.math_model_size
    if model_size is None:
        model_size = str(
            _first_non_none(
                runtime_config.get("model_size"),
                benchmark_config.get("model_size"),
                DEFAULT_MATH_MODEL_SIZE,
            )
        )

    use_attnres = overrides.math_use_attnres
    if use_attnres is None:
        use_attnres = bool(
            _first_non_none(
                runtime_config.get("use_attnres"),
                benchmark_config.get("use_attnres"),
                DEFAULT_MATH_USE_ATTNRES,
            )
        )

    seed = overrides.math_seed
    if seed is None:
        seed = int(
            _first_non_none(
                dataset_config.get("seed"),
                benchmark_config.get("seed"),
                DEFAULT_MATH_SEED,
            )
        )

    return build_math_runner(
        split=split,
        max_samples=max_samples,
        subjects=subjects,
        levels=levels,
        prompt_style=prompt_style,
        max_new_tokens=max_new_tokens,
        tokenizer_name=tokenizer_name,
        model_size=model_size,
        use_attnres=use_attnres,
        seed=seed,
    )


def _coerce_int_tuple(value: object, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(item) for item in value)


def _coerce_str_tuple(value: object, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    return tuple(str(item) for item in value)


def _coerce_float_tuple(value: object, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(item) for item in value)


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _first_non_none(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None
