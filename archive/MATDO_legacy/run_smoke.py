"""End-to-end MATDO sanity smoke runner (CPU-safe).

Purpose
-------
Exercise the US4–US6 pipeline against the *real* ``AdaptiveTransformer`` just
enough to verify that the unit-test-level QASP contracts (Stiefel adaptation,
ponder gate, RaBitQ KV, Stiefel overlay, prefill/step parity) integrate with
the outer pipeline driver. This is **not** an acceptance or benchmark run;
weights are randomly initialised so accuracy will be near zero.

Scope
-----
Two stages:

1. ``real-model plumbing probe``:
   - Load ``AdaptiveTransformer`` (small, random weights, CPU).
   - Run a **single** generate through ``evaluate_on_task`` with
     ``context_lengths=(128,)``, ``num_samples=1``, ``use_qttt=False``.
   - This is enough to flush the real-model code path end-to-end.

2. ``simulation-mode driver run``:
   - Call ``run_all_matdo_experiments`` skipping US1–US3 and forcing
     ``use_real_model=False`` so US4 / US5 / US6 execute in their fast
     simulation branches. This validates the driver plumbing and JSON
     emission contract.

CPU / wall-clock notes
----------------------
- US4 with ``enable_qttt=True`` runs query-only TTT per generated token; use
  ``config.us4_enable_qttt=False`` or ``--us4-no-qttt`` for tractable CPU runs.
- US5 shares one model instance and toggles ``enable_*`` flags (see README).
- US6 defaults to ``ctx_len = M * N_block`` (16K–64K); set
  ``rls_ctx_lengths_override`` or ``--rls-ctx-lengths`` for short prompts.

See ``experiments/matdo/README.md`` for CLI and config fields.

Run
---
    python3 experiments/matdo/run_smoke.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.matdo.common.config import config as matdo_config


def _apply_cpu_safe_overrides() -> Dict[str, Any]:
    """Reduce the real-model workload to something finishable on a laptop CPU."""

    overrides = {
        "real_model_context_lengths": (128,),
        "real_model_num_samples": 1,
    }
    for key, value in overrides.items():
        setattr(matdo_config, key, value)
    return overrides


def _real_model_plumbing_probe(output_dir: Path) -> Dict[str, Any]:
    """Load AdaptiveTransformer once and run one tiny generate to flush the path."""

    from experiments.matdo.common.real_model_bridge import (
        evaluate_on_task,
        load_matdo_model,
    )

    matdo_config.use_real_model = True
    matdo_config.checkpoint_path = None
    matdo_config.model_size = "small"
    matdo_config.device = "cpu"

    print("[smoke] probe: loading AdaptiveTransformer small on CPU (~20-40s)...")
    t0 = time.perf_counter()
    model, cfg = load_matdo_model(
        checkpoint_path=None,
        model_size="small",
        device="cpu",
        enable_rabitq=True,
        enable_attnres=True,
        enable_qttt=False,
    )
    load_s = time.perf_counter() - t0

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[smoke] loaded model in {load_s:.1f}s "
        f"({total_params / 1e9:.2f}B params, ~{total_params * 4 / 1024**3:.1f}GB fp32)"
    )

    print("[smoke] probe: running one needle generate (ctx=128, samples=1, qTTT=off)...")
    t0 = time.perf_counter()
    result = evaluate_on_task(
        model,
        "needle",
        cfg,
        device="cpu",
        context_lengths=(128,),
        num_samples=1,
    )
    gen_s = time.perf_counter() - t0
    print(f"[smoke] probe: generate completed in {gen_s:.1f}s")
    print(f"[smoke] probe result: {result}")

    del model

    return {
        "load_s": load_s,
        "generate_s": gen_s,
        "model_params_B": total_params / 1e9,
        "result": result,
        "notes": (
            "Accuracy is expected to be 0 because weights are random. "
            "The point is that load + generate with use_attnres=True "
            "completes without raising."
        ),
    }


def _us6_ctx_override_contract(output_dir: Path) -> Dict[str, Any]:
    """Verify ``rls_ctx_lengths_override`` replaces ``M*N_block`` in US6.

    Without the override ``rls_estimator`` forces ``ctx_len=M*N_block`` →
    16K–64K tokens on ``small``, which is not reachable on CPU. We don't
    want to spend a generate per grid point just to verify wiring, so we
    monkey-patch ``evaluate_on_task`` and assert the context lengths it
    receives come from the override tuple rather than the physical
    derivation.
    """

    from experiments.matdo.online_identification import rls_estimator

    matdo_config.use_real_model = True
    matdo_config.checkpoint_path = None
    matdo_config.model_size = "small"
    matdo_config.device = "cpu"
    matdo_config.rls_ctx_lengths_override = (64, 96, 128)

    observed_ctx_lengths: list[int] = []
    real_evaluate = rls_estimator.evaluate_on_task
    real_ensure_model = rls_estimator._ensure_us6_model

    from types import SimpleNamespace

    def _fake_ensure_model():
        # Skip the 20s + 8GB real load. `simulate_online_queries` only
        # touches ``cfg.max_qttt_steps``; everything else flows through
        # ``evaluate_on_task`` which we also monkey-patch below.
        return object(), SimpleNamespace(max_qttt_steps=16)

    def _fake_evaluate(model, task, cfg, *, context_lengths, num_samples, **_kw):
        assert len(context_lengths) == 1, context_lengths
        observed_ctx_lengths.append(int(context_lengths[0]))
        return {
            "task": task,
            "context_lengths": {context_lengths[0]: {"accuracy": 0.9, "correct": 0, "total": 1}},
            "average_accuracy": 90.0,
            "error": 0.10,
        }

    rls_estimator.evaluate_on_task = _fake_evaluate
    rls_estimator._ensure_us6_model = _fake_ensure_model
    try:
        t0 = time.perf_counter()
        data = rls_estimator.simulate_online_queries(num_queries=9)
        elapsed = time.perf_counter() - t0
    finally:
        rls_estimator.evaluate_on_task = real_evaluate
        rls_estimator._ensure_us6_model = real_ensure_model
        matdo_config.rls_ctx_lengths_override = None

    expected = [
        matdo_config.rls_ctx_lengths_override[t % 3] if False else (64, 96, 128)[t % 3]
        for t in range(9)
    ]
    print(
        f"[smoke] us6-ctx: 9 RLS queries produced ctx lengths {observed_ctx_lengths} "
        f"(expected cycle through {(64, 96, 128)})"
    )

    if observed_ctx_lengths != expected:
        raise AssertionError(
            f"rls_ctx_lengths_override did not replace M*N_block. "
            f"Expected {expected}, got {observed_ctx_lengths}."
        )

    return {
        "elapsed_s": elapsed,
        "num_queries": len(data),
        "observed_ctx_lengths": observed_ctx_lengths,
        "note": (
            "Synthetic probe with evaluate_on_task monkey-patched; only the "
            "ctx-length routing path is under test here."
        ),
    }


def _us4_knob_contract(output_dir: Path) -> Dict[str, Any]:
    """Verify ``us4_num_trials`` and ``us4_enable_qttt`` knobs land.

    We don't run 10 × multi-minute real MATDO generates just to check
    wiring. Instead we:
      * monkey-patch ``load_matdo_model`` so we can record the
        ``enable_qttt`` it receives without paying for a real load;
      * monkey-patch ``evaluate_on_task`` so we can count MATDO-E trials
        without paying for real generates;
      * call the driver in real-model mode with ``us4_num_trials=2`` and
        ``us4_enable_qttt=False`` and assert the observations match.
    """

    from experiments.matdo.sota_comparison import compare_baselines
    from experiments.matdo.run_all_experiments import run_all_matdo_experiments

    matdo_config.us4_num_trials = 2
    matdo_config.us4_enable_qttt = False

    compare_baselines._global_matdo_model = None
    compare_baselines._global_matdo_cfg = None

    captured_qttt: list[bool] = []
    matdo_trial_count = 0

    real_load = compare_baselines.load_matdo_model
    real_evaluate = compare_baselines.evaluate_on_task

    from types import SimpleNamespace

    def _fake_load(**kwargs):
        captured_qttt.append(bool(kwargs.get("enable_qttt", True)))
        cfg = SimpleNamespace(max_seq_len=4096, max_qttt_steps=16)
        cfg.enable_rabitq = kwargs.get("enable_rabitq", True)
        cfg.enable_attnres = kwargs.get("enable_attnres", True)
        cfg.enable_qttt = kwargs.get("enable_qttt", True)
        return object(), cfg

    def _fake_evaluate(model, task, cfg, **_kw):
        nonlocal matdo_trial_count
        matdo_trial_count += 1
        return {
            "task": task,
            "context_lengths": {},
            "average_accuracy": 95.0,
            "error": 0.05,
        }

    compare_baselines.load_matdo_model = _fake_load
    compare_baselines.evaluate_on_task = _fake_evaluate
    try:
        t0 = time.perf_counter()
        summary = run_all_matdo_experiments(
            skip_us1=True,
            skip_us2=True,
            skip_us3=True,
            skip_us4=False,
            skip_us5=True,
            skip_us6=True,
            output_dir=output_dir / "us4_knob_probe",
            use_real_model=True,
        )
        elapsed = time.perf_counter() - t0
    finally:
        compare_baselines.load_matdo_model = real_load
        compare_baselines.evaluate_on_task = real_evaluate
        matdo_config.us4_num_trials = None
        matdo_config.us4_enable_qttt = True
        matdo_config.use_real_model = False

    us4 = summary.get("results", {}).get("US4", {})
    recorded_trials = us4.get("test_conditions", {}).get("num_trials")
    print(
        f"[smoke] us4-knob: elapsed {elapsed:.2f}s, "
        f"enable_qttt seen by load_matdo_model={captured_qttt}, "
        f"test_conditions.num_trials={recorded_trials}, "
        f"evaluate_on_task call count={matdo_trial_count}"
    )

    errors: list[str] = []
    if not captured_qttt or any(q for q in captured_qttt):
        errors.append(
            f"us4_enable_qttt=False should propagate as enable_qttt=False, "
            f"got {captured_qttt}"
        )
    if recorded_trials != 2:
        errors.append(
            f"us4_num_trials=2 should yield MATDO-E num_trials=2, got {recorded_trials}"
        )
    if errors:
        raise AssertionError("; ".join(errors))

    return {
        "elapsed_s": elapsed,
        "enable_qttt_observations": captured_qttt,
        "matdo_e_trials_recorded": recorded_trials,
        "matdo_trial_count_total": matdo_trial_count,
        "note": (
            "Synthetic probe with load_matdo_model + evaluate_on_task "
            "monkey-patched; only the num_trials/enable_qttt knob wiring is "
            "under test here."
        ),
    }


def _simulation_driver_run(output_dir: Path) -> Dict[str, Any]:
    """Drive US4 / US5 / US6 through the unified runner in simulation mode."""

    from experiments.matdo.run_all_experiments import run_all_matdo_experiments

    matdo_config.use_real_model = False
    matdo_config.checkpoint_path = None

    print("[smoke] driver: US4/US5/US6 in simulation mode via run_all_matdo_experiments")
    t0 = time.perf_counter()
    summary = run_all_matdo_experiments(
        skip_us1=True,
        skip_us2=True,
        skip_us3=True,
        skip_us4=False,
        skip_us5=False,
        skip_us6=False,
        output_dir=output_dir / "sim_driver",
        use_real_model=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"[smoke] driver: completed in {elapsed:.1f}s")
    return {"elapsed_s": elapsed, "summary": summary}


def _us5_shared_cache_contract(output_dir: Path) -> Dict[str, Any]:
    """Verify the US5 shared-model-cache contract without paying for real generates.

    Running four real ``evaluate_on_task`` calls through the 2.2B CPU model
    takes 5–30 minutes once qTTT is toggled on (query-only TTT per generated
    token). That is far outside a sanity-check budget and buys us nothing
    extra: the contract we care about is "all four ``evaluate_*_only``
    functions share a single cached :class:`AdaptiveTransformer` instance
    and only mutate the ``enable_*`` flags". We can verify that by
    monkey-patching ``evaluate_on_task`` to return a synthetic result and
    snapshotting the config flags it sees on each call.

    A separate end-to-end real-forward call was already exercised by
    :func:`_real_model_plumbing_probe`; there is no reason to repeat it per
    ablation config.
    """

    from experiments.matdo.ablation import run_ablation as ablation_mod
    from experiments.matdo.common import real_model_bridge

    matdo_config.use_real_model = True
    matdo_config.checkpoint_path = None
    matdo_config.model_size = "small"
    matdo_config.device = "cpu"

    ablation_mod._global_model_cache.clear()

    observed_flags: list[tuple[bool, bool, bool]] = []
    real_evaluate = ablation_mod.evaluate_on_task

    def _fake_evaluate(model, task, cfg, **_kwargs):
        observed_flags.append(
            (
                bool(getattr(cfg, "enable_rabitq", False)),
                bool(getattr(cfg, "enable_attnres", False)),
                bool(getattr(cfg, "enable_qttt", False)),
            )
        )
        return {
            "task": task,
            "context_lengths": {},
            "average_accuracy": 0.0,
            "error": 1.0,
        }

    ablation_mod.evaluate_on_task = _fake_evaluate
    try:
        t0 = time.perf_counter()
        configs = [
            ("rabitq_only", ablation_mod.evaluate_rabitq_only, (True, False, False)),
            ("attnres_only", ablation_mod.evaluate_attnres_only, (False, True, False)),
            ("qttt_only", ablation_mod.evaluate_qttt_only, (False, False, True)),
            ("matdo_full", ablation_mod.evaluate_matdo_full, (True, True, True)),
        ]
        expected_flags = [cfg_flags for _, _, cfg_flags in configs]
        for name, fn, _ in configs:
            fn(rho=0.9)
        elapsed = time.perf_counter() - t0
    finally:
        ablation_mod.evaluate_on_task = real_evaluate

    cache_size = len(ablation_mod._global_model_cache)
    print(
        f"[smoke] us5-cache: 4 evaluate_* calls done in {elapsed:.2f}s, "
        f"shared cache size = {cache_size} (expected 1)"
    )
    print(f"[smoke] us5-cache: observed flags per call = {observed_flags}")

    if cache_size != 1:
        raise AssertionError(
            f"Expected exactly 1 cached model instance after 4 ablation evals, "
            f"got {cache_size}. The shared-model refactor is regressing."
        )
    if observed_flags != expected_flags:
        raise AssertionError(
            f"Flag-toggle contract broken: expected {expected_flags}, got {observed_flags}."
        )

    return {
        "elapsed_s": elapsed,
        "cache_size": cache_size,
        "observed_flags": [list(f) for f in observed_flags],
        "expected_flags": [list(f) for f in expected_flags],
        "note": (
            "Synthetic probe with evaluate_on_task monkey-patched. Real-forward "
            "end-to-end behaviour was already exercised by real_model_probe."
        ),
    }


def main() -> int:
    output_dir = Path(__file__).parent / "results" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = _apply_cpu_safe_overrides()
    print(f"[smoke] applied CPU-safe overrides: {overrides}")
    print(f"[smoke] output dir: {output_dir}")

    summary: Dict[str, Any] = {
        "mode": "cpu-safe-smoke",
        "overrides": overrides,
        "stages": {},
        "findings": [],
    }

    try:
        summary["stages"]["real_model_probe"] = _real_model_plumbing_probe(output_dir)
    except Exception as exc:
        traceback.print_exc()
        summary["stages"]["real_model_probe"] = {"error": repr(exc)}
        summary["findings"].append(
            "Real-model plumbing probe raised; unit tests pass but the AdaptiveTransformer "
            "real-model path does not integrate cleanly on CPU without further changes."
        )

    try:
        summary["stages"]["us5_shared_cache"] = _us5_shared_cache_contract(output_dir)
    except Exception as exc:
        traceback.print_exc()
        summary["stages"]["us5_shared_cache"] = {"error": repr(exc)}
        summary["findings"].append(
            "US5 shared-cache contract probe raised; the ablation refactor is "
            "regressing either the cache-size invariant or the flag-toggle invariant."
        )

    try:
        summary["stages"]["us6_ctx_override"] = _us6_ctx_override_contract(output_dir)
    except Exception as exc:
        traceback.print_exc()
        summary["stages"]["us6_ctx_override"] = {"error": repr(exc)}
        summary["findings"].append(
            "US6 ctx override probe raised; `rls_ctx_lengths_override` is "
            "not flowing through to `evaluate_on_task` as expected."
        )

    try:
        summary["stages"]["us4_knob_contract"] = _us4_knob_contract(output_dir)
    except Exception as exc:
        traceback.print_exc()
        summary["stages"]["us4_knob_contract"] = {"error": repr(exc)}
        summary["findings"].append(
            "US4 knob contract probe raised; either `us4_num_trials` or "
            "`us4_enable_qttt` is not flowing through the driver."
        )

    try:
        summary["stages"]["simulation_driver"] = _simulation_driver_run(output_dir)
    except Exception as exc:
        traceback.print_exc()
        summary["stages"]["simulation_driver"] = {"error": repr(exc)}
        summary["findings"].append(
            "Simulation driver run raised; the US4-6 default simulation branch itself is broken."
        )

    # All three integration-gap findings have now been addressed via
    # ``us4_num_trials`` / ``us4_enable_qttt`` / ``rls_ctx_lengths_override``
    # / the shared ablation cache. Nothing further to flag here.

    summary_file = output_dir / "smoke_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[smoke] summary written to {summary_file}")

    had_errors = any("error" in stage for stage in summary["stages"].values())
    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())
