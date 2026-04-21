"""
Shared HYPOTHESIS.md template for autoresearch trials.

Aligned with experiments/autoresearch/program.md (Explainable hypothesis + small-step edits).
"""

from __future__ import annotations

from pathlib import Path


def hypothesis_markdown(trial_id: str) -> str:
    return f"""# Trial hypothesis — {trial_id}

Fill this in **before** making code changes (see `program.md` → Trial Policy).

## 1. Hypothesis (one sentence)

What behavior or metric change do you expect from this trial?

> _Replace this line._

## 2. Mechanism

Why should this work in ADN/MATDO terms? Which knob (AttnRes, qTTT, gating, Engram-side glue, etc.)?

> _Replace this block._

## 3. Falsifiable prediction

What outcome on the fixed `trial-cmd` would **refute** this hypothesis?

> _Replace this block._

## 4. Scope link

How does the planned edit map to sections 1–3? List intended files/modules.

> _Replace this block._

## 5. After the trial run

Did results **match** or **contradict** the prediction? One short paragraph.

> _Fill after scoring._

---
Small-step rule: one primary intent, minimal diff, no bundled refactors (see `program.md`).
"""


def write_hypothesis_stub(trial_dir: Path, trial_id: str) -> Path:
    """Write `trial_dir/HYPOTHESIS.md` (overwrites if present)."""
    trial_dir = trial_dir.resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)
    path = trial_dir / "HYPOTHESIS.md"
    path.write_text(hypothesis_markdown(trial_id), encoding="utf-8")
    return path
