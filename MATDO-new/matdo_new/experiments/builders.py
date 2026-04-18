from __future__ import annotations

from collections.abc import Sequence

from matdo_new.experiments.benchmarks.critical_points import CriticalPointsBenchmark
from matdo_new.experiments.benchmarks.math import MathBenchmark
from matdo_new.experiments.benchmarks.needle import NeedleBenchmark
from matdo_new.experiments.evaluators.critical_points import CriticalPointsToyEvaluator
from matdo_new.experiments.evaluators.math import MathEvaluator
from matdo_new.experiments.evaluators.needle import NeedleEvaluator
from matdo_new.experiments.run_experiments import BenchmarkRunner


def build_needle_runner(
    *,
    context_lengths: Sequence[int],
    num_samples: int,
    depth_distribution: str,
    max_new_tokens: int,
    tokenizer_name: str,
    model_size: str,
    use_attnres: bool,
    seed: int = 42,
) -> BenchmarkRunner:
    return BenchmarkRunner(
        benchmark=NeedleBenchmark.build(
            context_lengths=context_lengths,
            num_samples=num_samples,
            depth_distribution=depth_distribution,
            max_new_tokens=max_new_tokens,
            tokenizer_name=tokenizer_name,
            model_size=model_size,
            use_attnres=use_attnres,
            seed=seed,
        ),
        evaluator=NeedleEvaluator(),
    )


def build_critical_points_runner(*, critical_points: Sequence[float]) -> BenchmarkRunner:
    return BenchmarkRunner(
        benchmark=CriticalPointsBenchmark.build(points=critical_points),
        evaluator=CriticalPointsToyEvaluator(),
    )


def build_math_runner(
    *,
    split: str,
    max_samples: int | None,
    subjects: Sequence[str],
    levels: Sequence[int],
    prompt_style: str,
    max_new_tokens: int,
    tokenizer_name: str,
    model_size: str,
    use_attnres: bool,
    seed: int = 42,
) -> BenchmarkRunner:
    return BenchmarkRunner(
        benchmark=MathBenchmark.build(
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
        ),
        evaluator=MathEvaluator(),
    )


def build_configured_runners(
    *,
    enable_needle: bool,
    needle_context_lengths: Sequence[int],
    needle_num_samples: int,
    needle_depth_distribution: str,
    needle_max_new_tokens: int,
    needle_tokenizer_name: str,
    needle_model_size: str,
    needle_use_attnres: bool,
    needle_seed: int = 42,
    enable_math: bool,
    math_split: str,
    math_max_samples: int | None,
    math_subjects: Sequence[str],
    math_levels: Sequence[int],
    math_prompt_style: str,
    math_max_new_tokens: int,
    math_tokenizer_name: str,
    math_model_size: str,
    math_use_attnres: bool,
    math_seed: int = 42,
    enable_critical_points: bool,
    critical_points: Sequence[float],
) -> tuple[BenchmarkRunner, ...]:
    runners: list[BenchmarkRunner] = []
    if enable_needle:
        runners.append(
            build_needle_runner(
                context_lengths=needle_context_lengths,
                num_samples=needle_num_samples,
                depth_distribution=needle_depth_distribution,
                max_new_tokens=needle_max_new_tokens,
                tokenizer_name=needle_tokenizer_name,
                model_size=needle_model_size,
                use_attnres=needle_use_attnres,
                seed=needle_seed,
            )
        )
    if enable_math:
        runners.append(
            build_math_runner(
                split=math_split,
                max_samples=math_max_samples,
                subjects=math_subjects,
                levels=math_levels,
                prompt_style=math_prompt_style,
                max_new_tokens=math_max_new_tokens,
                tokenizer_name=math_tokenizer_name,
                model_size=math_model_size,
                use_attnres=math_use_attnres,
                seed=math_seed,
            )
        )
    if enable_critical_points:
        runners.append(build_critical_points_runner(critical_points=critical_points))
    return tuple(runners)
