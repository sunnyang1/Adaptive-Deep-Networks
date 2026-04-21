#!/usr/bin/env python3
"""
Score a trial run from metrics files in a run directory.

This script is intentionally lightweight and robust:
- works with an explicit metrics file, or
- auto-discovers known ADN/MATDO-E experiment outputs.

Objectives:
- single: one primary metric + direction (max/min)
- dual: maximize weighted sum  w_tps * throughput - w_lat * p99_latency
  (requires both throughput_tokens_per_sec and p99_latency_ms)
"""

from __future__ import annotations

import argparse
import json
import math
import operator
from pathlib import Path
from typing import Any, Callable


OPS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}

DUAL_TPS = "throughput_tokens_per_sec"
DUAL_LAT = "p99_latency_ms"

METRIC_HINTS: dict[str, str] = {
    DUAL_TPS: (
        "throughput_result.json:throughput_tokens_per_sec, or "
        "validation_results.json:throughput_test.throughput_tokens_per_sec"
    ),
    DUAL_LAT: (
        "throughput_result.json:p99_latency_ms, latency_breakdown.json:results.with_prefetch.*, "
        "or validation_results.json-derived timings if present"
    ),
    "needle_128k_accuracy": (
        "needle_haystack_real.json or validation_results.json:needle_haystack.results"
    ),
}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_known_metrics(run_dir: Path) -> dict[str, float]:
    """
    Best-effort extraction from known report files in this repo.
    """
    metrics: dict[str, float] = {}

    # throughput_result.json
    for path in run_dir.rglob("throughput_result.json"):
        data = _load_json(path) or {}
        tps = data.get("throughput_tokens_per_sec")
        p99 = data.get("p99_latency_ms")
        mem = data.get("peak_memory_gb") or data.get("peak_allocated_gb")
        if isinstance(tps, (int, float)):
            metrics["throughput_tokens_per_sec"] = float(tps)
        if isinstance(p99, (int, float)):
            metrics["p99_latency_ms"] = float(p99)
        if isinstance(mem, (int, float)):
            metrics["peak_memory_gb"] = float(mem)

    # needle_haystack_real.json
    for path in run_dir.rglob("needle_haystack_real.json"):
        data = _load_json(path) or {}
        results = data.get("results", {})
        for key in ("131072", 131072):
            row = results.get(key) if isinstance(results, dict) else None
            if not row and isinstance(results, dict):
                row = results.get(str(key))
            if isinstance(row, dict):
                acc = row.get("accuracy")
                if isinstance(acc, (int, float)):
                    metrics["needle_128k_accuracy"] = float(acc)
                    break

    # p0_summary.json style pass counts
    for path in run_dir.rglob("p0_summary.json"):
        data = _load_json(path) or {}
        checks = data.get("checks")
        if isinstance(checks, dict):
            total = 0
            passed = 0
            for v in checks.values():
                if isinstance(v, dict) and "passed" in v:
                    total += 1
                    passed += int(bool(v["passed"]))
            if total > 0:
                metrics["p0_pass_rate"] = passed / total

    # experiments/real_model/validator.py output
    for path in run_dir.rglob("validation_results.json"):
        data = _load_json(path) or {}
        if not isinstance(data, dict):
            continue

        # needle metric
        needle = data.get("needle_haystack")
        if isinstance(needle, dict):
            results = needle.get("results", {})
            if isinstance(results, dict):
                row = results.get("131072") or results.get(131072)
                if isinstance(row, dict):
                    acc = row.get("accuracy")
                    if isinstance(acc, (int, float)):
                        metrics["needle_128k_accuracy"] = float(acc)

        # throughput metric
        t = data.get("throughput_test")
        if isinstance(t, dict):
            v = t.get("throughput_tokens_per_sec")
            if isinstance(v, (int, float)):
                metrics["throughput_tokens_per_sec"] = float(v)

        # memory metric from profiling
        mp = data.get("memory_profiling")
        if isinstance(mp, dict):
            measurements = mp.get("measurements")
            if isinstance(measurements, list):
                peaks: list[float] = []
                for m in measurements:
                    if isinstance(m, dict):
                        pv = m.get("peak_memory_gb")
                        if isinstance(pv, (int, float)):
                            peaks.append(float(pv))
                if peaks:
                    metrics["peak_memory_gb"] = max(peaks)

    # latency profiler output
    for path in run_dir.rglob("latency_breakdown.json"):
        data = _load_json(path) or {}
        if not isinstance(data, dict):
            continue
        results = data.get("results")
        if isinstance(results, dict):
            with_prefetch = results.get("with_prefetch")
            if isinstance(with_prefetch, dict) and with_prefetch:
                # choose largest E entry if possible
                best_key = None
                best_e = -1
                for k in with_prefetch.keys():
                    try:
                        e = int(k)
                    except Exception:
                        continue
                    if e > best_e:
                        best_e = e
                        best_key = k
                row = with_prefetch.get(best_key) if best_key is not None else None
                if isinstance(row, dict):
                    p99 = row.get("p99_latency_ms")
                    avg = row.get("avg_latency_ms")
                    if isinstance(p99, (int, float)):
                        metrics["p99_latency_ms"] = float(p99)
                    if isinstance(avg, (int, float)):
                        metrics["avg_latency_ms"] = float(avg)

        acceptance = data.get("acceptance")
        if isinstance(acceptance, dict):
            speedup = acceptance.get("speedup_with_prefetch")
            mask = acceptance.get("masking_efficiency")
            if isinstance(speedup, (int, float)):
                metrics["speedup_with_prefetch"] = float(speedup)
            if isinstance(mask, (int, float)):
                metrics["masking_efficiency"] = float(mask)

    # table6_math-like accuracy key if present
    for path in run_dir.rglob("*.json"):
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        for k in ("math_accuracy", "accuracy", "val_accuracy"):
            v = data.get(k)
            if isinstance(v, (int, float)):
                # keep only if a more specific key wasn't already set
                metrics.setdefault("math_accuracy", float(v))

    return metrics


def _parse_constraint(text: str) -> tuple[str, str, float]:
    for op in ("<=", ">=", "==", "<", ">"):
        if op in text:
            left, right = text.split(op, 1)
            return left.strip(), op, float(right.strip())
    raise ValueError(f"Invalid constraint: {text}")


def _evaluate_constraints(
    metrics: dict[str, float],
    constraints: list[str],
) -> tuple[bool, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    ok_all = True
    for raw in constraints:
        key, op, bound = _parse_constraint(raw)
        value = metrics.get(key)
        if value is None or not math.isfinite(value):
            ok = False
        else:
            ok = OPS[op](value, bound)
        details.append(
            {
                "constraint": raw,
                "metric": key,
                "op": op,
                "bound": bound,
                "value": value,
                "passed": ok,
            }
        )
        ok_all = ok_all and ok
    return ok_all, details


def _finite_metric(metrics: dict[str, float], key: str) -> float | None:
    v = metrics.get(key)
    if v is None or not math.isfinite(v):
        return None
    return float(v)


def _build_failure_reasons(
    objective: str,
    metrics: dict[str, float],
    primary_metric: str,
    primary_found: bool,
    constraints_ok: bool,
    constraint_details: list[dict[str, Any]],
    agent_returncode: int | None,
    trial_returncode: int | None,
    invalidate_on_trial_failure: bool,
) -> list[str]:
    reasons: list[str] = []

    if agent_returncode is not None and agent_returncode != 0:
        reasons.append(f"Agent command exited with code {agent_returncode}.")

    if trial_returncode is not None and trial_returncode != 0:
        reasons.append(f"Trial command exited with code {trial_returncode}.")
        if invalidate_on_trial_failure:
            reasons.append(
                "Trial failure invalidates the score (--invalidate-on-trial-failure is enabled)."
            )

    if objective == "single":
        if not primary_found:
            hint = METRIC_HINTS.get(primary_metric, "see experiments/autoresearch/score_trial.py")
            reasons.append(
                f"Primary metric {primary_metric!r} was not found or was non-finite in run_dir. "
                f"Hint: {hint}"
            )
    elif objective == "dual":
        tps_ok = _finite_metric(metrics, DUAL_TPS) is not None
        lat_ok = _finite_metric(metrics, DUAL_LAT) is not None
        if not tps_ok:
            reasons.append(
                f"Dual objective requires {DUAL_TPS}; not found or non-finite. "
                f"Hint: {METRIC_HINTS[DUAL_TPS]}"
            )
        if not lat_ok:
            reasons.append(
                f"Dual objective requires {DUAL_LAT}; not found or non-finite. "
                f"Hint: {METRIC_HINTS[DUAL_LAT]}"
            )

    if not constraints_ok:
        for d in constraint_details:
            if d.get("passed"):
                continue
            raw = d.get("constraint", "")
            val = d.get("value")
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                reasons.append(
                    f"Constraint {raw!r} failed: metric {d.get('metric')!r} missing or non-finite."
                )
            else:
                reasons.append(
                    f"Constraint {raw!r} failed: got {val}, needed {d.get('op')} {d.get('bound')}."
                )

    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Score an autoresearch trial.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument(
        "--objective",
        choices=("single", "dual"),
        default="single",
        help="single: one --primary-metric; dual: throughput + p99 latency composite.",
    )
    parser.add_argument("--primary-metric", type=str, default="needle_128k_accuracy")
    parser.add_argument(
        "--primary-direction",
        choices=("max", "min"),
        default="max",
    )
    parser.add_argument(
        "--dual-w-throughput",
        type=float,
        default=1.0,
        help="Weight on throughput_tokens_per_sec in dual objective (maximize contribution).",
    )
    parser.add_argument(
        "--dual-w-latency",
        type=float,
        default=0.01,
        help="Weight on p99_latency_ms in dual objective (minimize by subtracting w * latency).",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Constraint like p99_latency_ms<=400 (can repeat).",
    )
    parser.add_argument(
        "--agent-returncode",
        type=int,
        default=None,
        help="If set, recorded in score.json; non-zero adds failure_reasons.",
    )
    parser.add_argument(
        "--trial-returncode",
        type=int,
        default=None,
        help="If set, recorded in score.json; non-zero adds failure_reasons.",
    )
    parser.add_argument(
        "--invalidate-on-trial-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When trial-returncode is non-zero, mark valid=false (default: true).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics: dict[str, float] = {}

    if args.metrics_file:
        data = _load_json(args.metrics_file.resolve()) or {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
    else:
        metrics = _extract_known_metrics(run_dir)

    primary_value: float | None = None
    score = float("-inf")
    primary_found = False
    dual_block: dict[str, Any] | None = None

    if args.objective == "single":
        pv = _finite_metric(metrics, args.primary_metric)
        if pv is not None:
            primary_found = True
            primary_value = pv
            score = pv if args.primary_direction == "max" else -pv
    else:
        tps = _finite_metric(metrics, DUAL_TPS)
        lat = _finite_metric(metrics, DUAL_LAT)
        primary_found = tps is not None and lat is not None
        if primary_found and tps is not None and lat is not None:
            primary_value = args.dual_w_throughput * tps - args.dual_w_latency * lat
            score = primary_value
            dual_block = {
                "throughput_tokens_per_sec": tps,
                "p99_latency_ms": lat,
                "weights": {
                    "throughput": args.dual_w_throughput,
                    "latency": args.dual_w_latency,
                },
                "formula": "w_throughput * throughput_tokens_per_sec - w_latency * p99_latency_ms",
            }

    constraints_ok, constraint_details = _evaluate_constraints(metrics, args.constraint)

    trial_invalid = (
        args.trial_returncode is not None
        and args.trial_returncode != 0
        and args.invalidate_on_trial_failure
    )
    agent_invalid = args.agent_returncode is not None and args.agent_returncode != 0

    valid = (
        primary_found
        and constraints_ok
        and not trial_invalid
        and not agent_invalid
        and math.isfinite(score)
    )

    failure_reasons = _build_failure_reasons(
        objective=args.objective,
        metrics=metrics,
        primary_metric=args.primary_metric,
        primary_found=primary_found,
        constraints_ok=constraints_ok,
        constraint_details=constraint_details,
        agent_returncode=args.agent_returncode,
        trial_returncode=args.trial_returncode,
        invalidate_on_trial_failure=bool(args.invalidate_on_trial_failure),
    )

    if primary_found and not math.isfinite(score):
        failure_reasons.append("Computed score is non-finite (check metric values and weights).")

    display_primary = args.primary_metric
    if args.objective == "dual":
        display_primary = "dual(throughput_tokens_per_sec,p99_latency_ms)"

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "objective": args.objective,
        "primary_metric": display_primary,
        "primary_direction": args.primary_direction,
        "primary_value": primary_value,
        "score": score,
        "primary_found": primary_found,
        "constraints_ok": constraints_ok,
        "constraint_details": constraint_details,
        "failure_reasons": failure_reasons,
        "valid": valid,
        "agent_returncode": args.agent_returncode,
        "trial_returncode": args.trial_returncode,
    }
    if dual_block is not None:
        result["dual_objective"] = dual_block

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
