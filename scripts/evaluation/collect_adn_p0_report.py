#!/usr/bin/env python3
"""
Collect P0 verification outputs into a single markdown report.

Expected input directory: results/paper_verify_<timestamp>/
Produced files:
  - P0_verification_report.md
  - p0_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


NEEDLE_TARGETS = {
    4096: 98.5,    # 4K
    32768: 91.8,   # 32K
    131072: 79.5,  # 128K
    262144: 69.0,  # 256K
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick_latest_run(base: Path) -> Path | None:
    runs = sorted(base.glob("paper_verify_*"))
    return runs[-1] if runs else None


def _fmt(v: Any, digits: int = 2) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return "N/A"


def _status(ok: bool) -> str:
    return "PASS" if ok else "FLAG"


def build_report(run_dir: Path, needle_tol_pp: float, throughput_target: float) -> tuple[str, dict[str, Any]]:
    static_consistency = _load_json(run_dir / "static_consistency.json")
    needle = _load_json(run_dir / "needle_haystack_real.json")
    throughput = _load_json(run_dir / "throughput_result.json")
    flop = _load_json(run_dir / "flop_analysis.json")
    memory = _load_json(run_dir / "memory_profile.json")
    table4 = _load_json(run_dir / "table4_bitsweep.json")
    table5 = _load_json(run_dir / "table5_components.json")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    summary: dict[str, Any] = {
        "generated_at": now,
        "run_dir": str(run_dir),
        "checks": {},
    }

    lines.append("# ADN P0 Verification Report")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append("")

    lines.append("## P0 Claim Checks")
    lines.append("")
    lines.append("### Needle-in-Haystack (Table 5 targets)")
    lines.append("")
    lines.append(f"- Target tolerance: `±{needle_tol_pp:.1f} pp`")
    lines.append("")
    lines.append("| Context | Target (%) | Actual (%) | Delta (pp) | Status |")
    lines.append("|---|---:|---:|---:|---|")

    needle_results = (needle or {}).get("results", {})
    needle_statuses = []
    for ctx, target in NEEDLE_TARGETS.items():
        # JSON keys can be str
        row = needle_results.get(str(ctx)) or needle_results.get(ctx) or {}
        actual = row.get("accuracy")
        if actual is None:
            lines.append(f"| {ctx//1024}K | {target:.1f} | N/A | N/A | FLAG |")
            needle_statuses.append(False)
            continue
        delta = actual - target
        ok = abs(delta) <= needle_tol_pp
        needle_statuses.append(ok)
        lines.append(f"| {ctx//1024}K | {target:.1f} | {actual:.1f} | {delta:+.1f} | {_status(ok)} |")

    needle_overall = all(needle_statuses) if needle_statuses else False
    summary["checks"]["needle_table5"] = {"passed": needle_overall}
    lines.append("")
    lines.append(f"- Needle check overall: **{_status(needle_overall)}** (tolerance: ±{needle_tol_pp:.1f} pp)")
    lines.append("")

    lines.append("### Throughput (Table 4 setting)")
    lines.append("")
    lines.append(f"- Target throughput: `{throughput_target:.1f} tok/s`")
    lines.append("")
    tps = (throughput or {}).get("throughput_tokens_per_sec")
    if tps is None:
        lines.append("- Actual throughput: `N/A`")
        lines.append("- Status: **FLAG**")
        summary["checks"]["throughput"] = {"passed": False}
    else:
        # Paper target is 115 tok/s, user can override.
        ok = tps >= 0.9 * throughput_target
        summary["checks"]["throughput"] = {"passed": ok, "actual": tps, "target": throughput_target}
        lines.append(f"- Actual throughput: `{tps:.2f} tok/s`")
        lines.append(f"- Target throughput: `{throughput_target:.1f} tok/s`")
        lines.append(f"- Status: **{_status(ok)}** (pass threshold: >= {0.9 * throughput_target:.1f})")
    lines.append("")

    lines.append("### FLOP Equivalence (Section 4 support)")
    lines.append("")
    eq = ((flop or {}).get("equivalence", {}) or {}).get("equivalence", {})
    ratio = eq.get("ratio")
    eq_ok = eq.get("is_equivalent")
    if ratio is None or eq_ok is None:
        lines.append("- Ratio `T_think / (2*N_qTTT*k)`: `N/A`")
        lines.append("- Status: **FLAG**")
        summary["checks"]["flop_equivalence"] = {"passed": False}
    else:
        lines.append(f"- Ratio `T_think / (2*N_qTTT*k)`: `{ratio:.3f}`")
        lines.append(f"- Status: **{_status(bool(eq_ok))}**")
        summary["checks"]["flop_equivalence"] = {"passed": bool(eq_ok), "ratio": ratio}
    lines.append("")

    lines.append("## Supporting Measurements")
    lines.append("")

    lines.append("### Table 4 Bitwidth Sweep (measured prefill throughput)")
    lines.append("")
    if not table4:
        lines.append("- `table4_bitsweep.json` not found.")
    else:
        lines.append("| Setting | Paper tok/s | Measured tok/s | Peak mem (GB) | Time/forward (s) | Note |")
        lines.append("|---|---:|---:|---:|---:|---|")
        meas = table4.get("measurements", {})
        for key in ["fp16", "3bit", "2bit", "1bit"]:
            row = meas.get(key, {})
            paper = (row.get("paper") or {}).get("tokens_per_sec")
            m = row.get("measured") or {}
            tps = m.get("tokens_per_sec")
            peak = m.get("peak_allocated_gb")
            tmean = m.get("time_mean_s")
            note = "prefill-style (full forward over T)" if key == "fp16" else "RaBitQ prefill-style"
            lines.append(
                f"| {key} | {paper if paper is not None else 'N/A'} | "
                f"{_fmt(tps)} | {_fmt(peak)} | {_fmt(tmean)} | {note} |"
            )
        lines.append("")
        lines.append("- This section measures **prefill throughput**, not incremental decode throughput with KV reuse.")
    lines.append("")

    lines.append("### Table 5 Component Sweep (Needle-in-Haystack ablations)")
    lines.append("")
    if not table5:
        lines.append("- `table5_components.json` not found.")
    else:
        lines.append("- Note: This sweep uses the real-model NeedleDataset protocol with runtime toggles.")
        lines.append("")
        lengths = (table5.get("meta") or {}).get("lengths") or []
        runs = table5.get("runs") or {}
        # Print compact table: rows=configs, cols=lengths
        header = "| Config | " + " | ".join([f"{int(L)//1024}K (%)" for L in lengths]) + " |"
        sep = "|---|" + "|".join(["---:"] * len(lengths)) + "|"
        lines.append(header)
        lines.append(sep)
        for cfg_name, cfg_obj in runs.items():
            res = (cfg_obj.get("results") or {}).get("results") or {}
            vals = []
            for L in lengths:
                row = res.get(str(L)) or res.get(L) or {}
                acc = (row or {}).get("accuracy")
                vals.append("N/A" if acc is None else f"{acc:.1f}")
            lines.append("| " + cfg_name + " | " + " | ".join(vals) + " |")
        lines.append("")
        lines.append("#### Table 5-style reconstruction (measured)")
        lines.append("")
        lines.append("| Context | Baseline | +RaBitQ | +AttnRes | +Engram | +qTTT (Full) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for L in lengths:
            r_base = ((runs.get("baseline") or {}).get("results") or {}).get("results") or {}
            r_rq = ((runs.get("+rabitq") or {}).get("results") or {}).get("results") or {}
            r_ar = ((runs.get("+attnres") or {}).get("results") or {}).get("results") or {}
            r_en = ((runs.get("+engram") or {}).get("results") or {}).get("results") or {}
            r_full = ((runs.get("full(+qttt)") or {}).get("results") or {}).get("results") or {}

            def get_acc(obj, key):
                row = obj.get(str(key)) or obj.get(key) or {}
                v = row.get("accuracy")
                return None if v is None else float(v)

            b = get_acc(r_base, L)
            q = get_acc(r_rq, L)
            a = get_acc(r_ar, L)
            e = get_acc(r_en, L)
            f = get_acc(r_full, L)
            ctx = f"{int(L)//1024}K"
            lines.append(
                f"| {ctx} | "
                f"{'N/A' if b is None else f'{b:.1f}'} | "
                f"{'N/A' if q is None else f'{q:.1f}'} | "
                f"{'N/A' if a is None else f'{a:.1f}'} | "
                f"{'N/A' if e is None else f'{e:.1f}'} | "
                f"{'N/A' if f is None else f'{f:.1f}'} |"
            )

        lines.append("")
        lines.append("#### Progressive Gains (measured, 128K if available else max tested)")
        lines.append("")

        # Choose 128K if present; otherwise the largest tested length.
        chosen = 131072 if 131072 in lengths else (max(lengths) if lengths else None)
        if chosen is not None:
            r_base = ((runs.get("baseline") or {}).get("results") or {}).get("results") or {}
            r_rq = ((runs.get("+rabitq") or {}).get("results") or {}).get("results") or {}
            r_ar = ((runs.get("+attnres") or {}).get("results") or {}).get("results") or {}
            r_en = ((runs.get("+engram") or {}).get("results") or {}).get("results") or {}
            r_full = ((runs.get("full(+qttt)") or {}).get("results") or {}).get("results") or {}

            def get_acc(obj, key):
                row = obj.get(str(key)) or obj.get(key) or {}
                v = row.get("accuracy")
                return None if v is None else float(v)

            b = get_acc(r_base, chosen)
            q = get_acc(r_rq, chosen)
            a = get_acc(r_ar, chosen)
            e = get_acc(r_en, chosen)
            f = get_acc(r_full, chosen)

            lines.append(f"- Context used for gains: `{chosen//1024}K`")
            if None in (b, q, a, e, f):
                lines.append("- Progressive gains unavailable (missing one or more component results).")
            else:
                g_rq = q - b
                g_ar = a - q
                g_en = e - a
                g_qt = f - e
                lines.append(f"- RaBitQ gain: `{g_rq:+.1f} pp`")
                lines.append(f"- AttnRes gain: `{g_ar:+.1f} pp`")
                lines.append(f"- Engram gain: `{g_en:+.1f} pp`")
                lines.append(f"- qTTT gain: `{g_qt:+.1f} pp`")

                # Compare with paper reference if available
                paper_ref = (table5.get("paper_table5_reference") or {}).get(chosen) or (table5.get("paper_table5_reference") or {}).get(str(chosen))
                if isinstance(paper_ref, dict):
                    pr_b = float(paper_ref.get("baseline", 0))
                    pr_q = float(paper_ref.get("rabitq", 0))
                    pr_a = float(paper_ref.get("attnres", 0))
                    pr_e = float(paper_ref.get("engram", 0))
                    pr_f = float(paper_ref.get("full", 0))
                    lines.append("")
                    lines.append("Paper-vs-measured deltas at this context:")
                    lines.append(f"- Baseline delta: `{(b - pr_b):+.1f} pp`")
                    lines.append(f"- +RaBitQ delta: `{(q - pr_q):+.1f} pp`")
                    lines.append(f"- +AttnRes delta: `{(a - pr_a):+.1f} pp`")
                    lines.append(f"- +Engram delta: `{(e - pr_e):+.1f} pp`")
                    lines.append(f"- Full(+qTTT) delta: `{(f - pr_f):+.1f} pp`")
        else:
            lines.append("- Progressive gains unavailable (no context lengths found).")
    lines.append("")

    lines.append("### Memory Profile Snapshot")
    lines.append("")
    lines.append("| Context | Peak Memory (GB) | KV Cache Est. (GB) | Time (s) |")
    lines.append("|---|---:|---:|---:|")
    mem_rows = (memory or {}).get("measurements", [])
    if not mem_rows:
        lines.append("| N/A | N/A | N/A | N/A |")
    else:
        for m in mem_rows:
            ctx = m.get("context_length", "N/A")
            ctx_s = f"{int(ctx)//1024}K" if isinstance(ctx, int) and ctx >= 1024 else str(ctx)
            lines.append(
                f"| {ctx_s} | {_fmt(m.get('peak_memory_gb'))} | {_fmt(m.get('kv_cache_est_gb'))} | {_fmt(m.get('time'))} |"
            )
    lines.append("")

    lines.append("### Static Consistency Checks")
    lines.append("")
    if not static_consistency:
        lines.append("- `static_consistency.json` not found.")
    else:
        kv = static_consistency.get("kv_cache", {})
        comp = static_consistency.get("compression_table4", {})
        ponder = static_consistency.get("ponder_amortized", {})
        lines.append(f"- KV total (80 layers @128K): `{_fmt(kv.get('gb_total_80_layers'))} GB` (paper text: 40 GB)")
        lines.append(
            "- Compression-derived storage: "
            f"`3-bit={_fmt(comp.get('3bit_gb'))} GB`, "
            f"`2-bit={_fmt(comp.get('2bit_gb'))} GB`, "
            f"`1-bit={_fmt(comp.get('1bit_gb'))} GB`"
        )
        lines.append(
            "- Ponder amortized factors: "
            f"`0.30*3.6={_fmt(ponder.get('trigger_30pct_step2_overhead3p6x'), 3)}x`, "
            f"`0.30*12.8={_fmt(ponder.get('trigger_30pct_step10_overhead12p8x'), 3)}x`"
        )
    lines.append("")

    lines.append("## Coverage Notes")
    lines.append("")
    lines.append("- This report aggregates outputs from the A100 verification runner.")
    lines.append("- Full reproduction of Table 4 bit-width sweep, Table 8 MATH, and Table 9 LongBench ablations requires dedicated benchmark runs with trained checkpoints.")
    lines.append("- Scripts in `experiments/validation/` and `scripts/evaluation/eval_5_2.py` are target-replay/simulated and should not be used as primary evidence.")
    lines.append("")

    overall = all(v.get("passed", False) for v in summary["checks"].values()) if summary["checks"] else False
    summary["overall_passed"] = overall
    lines.append(f"## Overall P0 Status: **{_status(overall)}**")
    lines.append("")

    return "\n".join(lines), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ADN P0 verification outputs into markdown report.")
    parser.add_argument("--run-dir", type=str, default="", help="Path to results/paper_verify_<timestamp> directory")
    parser.add_argument("--base-dir", type=str, default="results", help="Base directory for auto-discovery")
    parser.add_argument("--needle-tol-pp", type=float, default=5.0, help="Tolerance in percentage points for Needle checks")
    parser.add_argument("--throughput-target", type=float, default=115.0, help="Throughput target in tok/s")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _pick_latest_run(Path(args.base_dir))
    if run_dir is None:
        raise SystemExit("No run directory found. Provide --run-dir or run verification first.")
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    report_text, summary = build_report(run_dir, args.needle_tol_pp, args.throughput_target)
    report_path = run_dir / "P0_verification_report.md"
    summary_path = run_dir / "p0_summary.json"
    report_path.write_text(report_text, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote report: {report_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Overall: {'PASS' if summary.get('overall_passed') else 'FLAG'}")


if __name__ == "__main__":
    main()
