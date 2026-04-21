#!/usr/bin/env python3
"""
First-version autoresearch loop for ADN/MATDO-E.

Loop:
1) create isolated git worktree
2) run agent command to edit code in that worktree
3) run fixed trial command
4) score with score_trial.py
5) keep best patch artifact and full logs

This v0 runner is intentionally conservative:
- it does NOT auto-cherry-pick into main by default
- it stores best patch in experiments/autoresearch/best/
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ad = str(Path(__file__).resolve().parent)
if _ad not in sys.path:
    sys.path.insert(0, _ad)
from hypothesis_stub import write_hypothesis_stub  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTO_DIR = REPO_ROOT / "experiments" / "autoresearch"
TRIALS_DIR = AUTO_DIR / "trials"
BEST_DIR = AUTO_DIR / "best"
SCORE_SCRIPT = AUTO_DIR / "score_trial.py"
PROGRAM_FILE = AUTO_DIR / "program.md"
AGENT_DRIVER = AUTO_DIR / "agent_driver.py"


@dataclass
class TrialRecord:
    trial_id: str
    branch: str
    workspace: str
    started_at: str
    ended_at: str
    agent_returncode: int
    trial_returncode: int
    score_returncode: int
    score: float | None
    valid: bool
    improved: bool
    patch_file: str
    score_json: str
    logs_dir: str
    notes: str = ""


def _run(
    cmd: str,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        capture_output=True,
        text=True,
        env=merged_env,
        timeout=timeout,
    )


def _require_clean_main() -> None:
    status = _run("git status --porcelain", REPO_ROOT)
    if status.returncode != 0:
        raise RuntimeError(f"Failed to check git status: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError(
            "Working tree is not clean. Commit/stash changes before running autoresearch."
        )


def _now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _score_trial(
    workspace: Path,
    output_json: Path,
    primary_metric: str,
    primary_direction: str,
    constraints: list[str],
    objective: str,
    dual_w_throughput: float,
    dual_w_latency: float,
    agent_returncode: int | None,
    trial_returncode: int | None,
    invalidate_on_trial_failure: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        shlex.quote(sys.executable),
        shlex.quote(str(SCORE_SCRIPT)),
        "--run-dir",
        shlex.quote(str(workspace)),
        "--output-json",
        shlex.quote(str(output_json)),
        "--objective",
        shlex.quote(objective),
        "--primary-metric",
        shlex.quote(primary_metric),
        "--primary-direction",
        shlex.quote(primary_direction),
        "--dual-w-throughput",
        str(dual_w_throughput),
        "--dual-w-latency",
        str(dual_w_latency),
    ]
    if agent_returncode is not None:
        cmd.extend(["--agent-returncode", str(agent_returncode)])
    if trial_returncode is not None:
        cmd.extend(["--trial-returncode", str(trial_returncode)])
    if invalidate_on_trial_failure:
        cmd.append("--invalidate-on-trial-failure")
    else:
        cmd.append("--no-invalidate-on-trial-failure")
    for c in constraints:
        cmd.extend(["--constraint", shlex.quote(c)])
    return _run(" ".join(cmd), REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ADN autoresearch loop (v0).")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--program",
        type=Path,
        default=PROGRAM_FILE,
        help="Path to program.md context file.",
    )
    parser.add_argument(
        "--agent-cmd",
        type=str,
        default="",
        help=(
            "Command that performs code edits inside each trial worktree. "
            "Template variables: {program}, {workspace}, {trial_dir}, {trial_id}"
        ),
    )
    parser.add_argument(
        "--trial-cmd",
        type=str,
        required=True,
        help="Fixed trial command to execute in each worktree.",
    )
    parser.add_argument("--trial-timeout-sec", type=int, default=1800)
    parser.add_argument("--agent-timeout-sec", type=int, default=600)
    parser.add_argument(
        "--objective",
        choices=("single", "dual"),
        default="single",
        help="single: one primary metric; dual: w_tps*throughput - w_lat*p99 (both required).",
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
        help="Dual objective: weight on throughput_tokens_per_sec.",
    )
    parser.add_argument(
        "--dual-w-latency",
        type=float,
        default=0.01,
        help="Dual objective: weight on p99_latency_ms (subtracted).",
    )
    parser.add_argument(
        "--invalidate-on-trial-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When trial command exits non-zero, mark score invalid (default: true).",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Constraint like p99_latency_ms<=400 (can repeat).",
    )
    parser.add_argument(
        "--baseline-score",
        type=float,
        default=None,
        help="Optional starting best score; if omitted, best starts at -inf.",
    )
    parser.add_argument(
        "--keep-worktrees",
        action="store_true",
        help="Keep trial worktrees for debugging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and log each trial without running agent/trial/score commands.",
    )
    parser.add_argument(
        "--skip-clean-check",
        action="store_true",
        help="Skip requiring a clean git working tree (recommended only with --dry-run).",
    )
    args = parser.parse_args()

    if not args.skip_clean_check:
        _require_clean_main()
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    program_path = args.program.resolve()
    if not program_path.exists():
        raise FileNotFoundError(f"Program file not found: {program_path}")

    run_id = _now()
    ledger_path = TRIALS_DIR / f"ledger-{run_id}.jsonl"
    run_summary_path = TRIALS_DIR / f"summary-{run_id}.json"

    best_score = args.baseline_score if args.baseline_score is not None else float("-inf")
    best_trial_id: str | None = None
    records: list[TrialRecord] = []

    for i in range(1, args.iterations + 1):
        trial_id = f"{run_id}-t{i:03d}"
        branch = f"autoresearch/{trial_id}"
        trial_dir = TRIALS_DIR / trial_id
        workspace = trial_dir / "workspace"
        logs_dir = trial_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        write_hypothesis_stub(trial_dir, trial_id)

        started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        add = _run(
            f"git worktree add -b {shlex.quote(branch)} {shlex.quote(str(workspace))} HEAD",
            REPO_ROOT,
        )
        if add.returncode != 0:
            raise RuntimeError(f"Failed to create worktree for {trial_id}: {add.stderr.strip()}")

        agent_rc = 0
        trial_rc = 0
        score_rc = 0
        score_value: float | None = None
        valid = False
        improved = False

        patch_file = trial_dir / "trial.patch"
        score_json = trial_dir / "score.json"

        try:
            planned_agent_cmd = ""
            if args.agent_cmd.strip():
                planned_agent_cmd = args.agent_cmd.format(
                    program=str(program_path),
                    workspace=str(workspace),
                    trial_dir=str(trial_dir),
                    trial_id=trial_id,
                )
            else:
                planned_agent_cmd = (
                    f"{shlex.quote(sys.executable)} {shlex.quote(str(AGENT_DRIVER))} "
                    f"--workspace {shlex.quote(str(workspace))} "
                    f"--program {shlex.quote(str(program_path))} "
                    f"--trial-id {shlex.quote(trial_id)} "
                    f"--trial-dir {shlex.quote(str(trial_dir))}"
                )
            if args.dry_run:
                trial_rc = 0
                score_rc = 0
                _write_text(
                    logs_dir / "agent.stdout.log",
                    f"[dry-run] planned agent command:\n{planned_agent_cmd}\n",
                )
                _write_text(logs_dir / "agent.stderr.log", "")
                _write_text(
                    logs_dir / "trial.stdout.log",
                    f"[dry-run] planned trial command:\n{args.trial_cmd}\n",
                )
                _write_text(logs_dir / "trial.stderr.log", "")
                _write_text(
                    logs_dir / "score.stdout.log",
                    (
                        "[dry-run] planned score invocation:\n"
                        f"{sys.executable} {SCORE_SCRIPT} --run-dir {workspace} "
                        f"--output-json {score_json} --objective {args.objective} "
                        f"--primary-metric {args.primary_metric} "
                        f"--primary-direction {args.primary_direction} "
                        f"--dual-w-throughput {args.dual_w_throughput} "
                        f"--dual-w-latency {args.dual_w_latency} "
                        + " ".join([f'--constraint "{c}"' for c in args.constraint])
                        + "\n"
                    ),
                )
                _write_text(logs_dir / "score.stderr.log", "")
                _write_text(patch_file, "[dry-run] patch not generated.\n")
                _write_text(
                    score_json,
                    json.dumps(
                        {
                            "valid": False,
                            "score": None,
                            "dry_run": True,
                            "failure_reasons": ["dry-run: scoring not executed"],
                        },
                        indent=2,
                    ),
                )
            else:
                agent_run = _run(planned_agent_cmd, workspace, timeout=args.agent_timeout_sec)
                agent_rc = agent_run.returncode
                _write_text(logs_dir / "agent.stdout.log", agent_run.stdout)
                _write_text(logs_dir / "agent.stderr.log", agent_run.stderr)

                trial_run = _run(args.trial_cmd, workspace, timeout=args.trial_timeout_sec)
                trial_rc = trial_run.returncode
                _write_text(logs_dir / "trial.stdout.log", trial_run.stdout)
                _write_text(logs_dir / "trial.stderr.log", trial_run.stderr)

                patch_run = _run("git diff HEAD", workspace)
                _write_text(patch_file, patch_run.stdout)

                score_run = _score_trial(
                    workspace=workspace,
                    output_json=score_json,
                    primary_metric=args.primary_metric,
                    primary_direction=args.primary_direction,
                    constraints=args.constraint,
                    objective=args.objective,
                    dual_w_throughput=args.dual_w_throughput,
                    dual_w_latency=args.dual_w_latency,
                    agent_returncode=agent_rc,
                    trial_returncode=trial_rc,
                    invalidate_on_trial_failure=args.invalidate_on_trial_failure,
                )
                score_rc = score_run.returncode
                _write_text(logs_dir / "score.stdout.log", score_run.stdout)
                _write_text(logs_dir / "score.stderr.log", score_run.stderr)

                score_data: dict[str, Any] = {}
                if score_json.exists():
                    score_data = json.loads(score_json.read_text(encoding="utf-8"))
                    score_value = score_data.get("score")
                    valid = bool(score_data.get("valid", False))

                if valid and score_value is not None and score_value > best_score:
                    best_score = float(score_value)
                    best_trial_id = trial_id
                    improved = True
                    (BEST_DIR / "best.patch").write_text(
                        patch_file.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                    (BEST_DIR / "best_score.json").write_text(
                        json.dumps(
                            {
                                "trial_id": trial_id,
                                "score": best_score,
                                "objective": args.objective,
                                "primary_metric": args.primary_metric,
                                "primary_direction": args.primary_direction,
                                "dual_w_throughput": args.dual_w_throughput,
                                "dual_w_latency": args.dual_w_latency,
                                "constraints": args.constraint,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
        finally:
            ended = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            rec = TrialRecord(
                trial_id=trial_id,
                branch=branch,
                workspace=str(workspace),
                started_at=started,
                ended_at=ended,
                agent_returncode=agent_rc,
                trial_returncode=trial_rc,
                score_returncode=score_rc,
                score=score_value,
                valid=valid,
                improved=improved,
                patch_file=str(patch_file),
                score_json=str(score_json),
                logs_dir=str(logs_dir),
                notes="dry-run" if args.dry_run else "",
            )
            records.append(rec)
            with ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

            if not args.keep_worktrees:
                _run(f"git worktree remove --force {shlex.quote(str(workspace))}", REPO_ROOT)
                _run(f"git branch -D {shlex.quote(branch)}", REPO_ROOT)

    summary = {
        "run_id": run_id,
        "iterations": args.iterations,
        "program": str(program_path),
        "trial_cmd": args.trial_cmd,
        "objective": args.objective,
        "primary_metric": args.primary_metric,
        "primary_direction": args.primary_direction,
        "dual_w_throughput": args.dual_w_throughput,
        "dual_w_latency": args.dual_w_latency,
        "constraints": args.constraint,
        "best_score": best_score,
        "best_trial_id": best_trial_id,
        "ledger": str(ledger_path),
        "records": [asdict(r) for r in records],
    }
    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
