# ADN AutoResearch Program (v0)

This file defines the first research loop policy for autonomous improvement of ADN/MATDO-E.

## Mission

Improve long-context inference quality-efficiency trade-offs for the ADN stack under fixed compute/runtime budgets.

Primary optimization targets:

1. Increase long-context quality (especially Needle 128K behavior).
2. Maintain or improve serving efficiency (throughput / latency / memory).
3. Avoid regressions in stability and reproducibility.

## Scope

Allowed edit regions (first phase):

- `src/attnres/`
- `src/qttt/`
- `src/gating/`
- `experiments/matdo/`
- lightweight config/runner glue required to test a change

Disallowed edits (first phase):

- broad repo refactors
- dependency upgrades unrelated to trial goal
- destructive git history operations

## Trial Policy

Each trial must follow the workflow below. **Small steps** and an **explicit, explainable hypothesis** are mandatory: trials that bundle unrelated edits or omit a clear causal story are out of policy.

### Explainable hypothesis (required)

Before coding, the agent must state in the trial artifact (e.g. `trials/<id>/HYPOTHESIS.md` or the top of the agent transcript summary):

1. **Hypothesis** — One sentence: what change in behavior or metric do you expect?
2. **Mechanism** — Why should this work in ADN/MATDO terms? (which knob: AttnRes, qTTT, gating, Engram-side glue, etc.)
3. **Falsifiable prediction** — What outcome would *refute* the hypothesis on the fixed `trial-cmd`?
4. **Scope link** — How does the planned code edit map to (1)–(3)?

If any of the above is missing, treat the trial as **failed policy compliance** even if metrics move.

### Small-step edits (required)

1. **One primary intent per trial** — A single coherent change (e.g. adjust one threshold family, or one scheduling rule), not a mixed refactor + feature + config sweep.
2. **Minimal diff** — Touch the fewest files and lines needed to test the hypothesis; prefer one file or one module unless the hypothesis explicitly spans two layers.
3. **Bounded surface** — Avoid drive-by renames, formatting-only sweeps, and “while we’re here” edits. No dependency or toolchain upgrades unless the hypothesis requires them.
4. **No multi-hypothesis bundles** — If you need a second idea, run a **new** trial with a new hypothesis document.

### Standard trial steps

1. Propose a **single focused hypothesis** (sections above).
2. Implement the **smallest** change that tests it.
3. Run the fixed, short evaluation command.
4. Score against the current best checkpoint/metric policy.
5. Keep only if it improves objective and passes constraints.
6. In logs or `HYPOTHESIS.md`, briefly note whether results **match** or **contradict** the prediction (one short paragraph).

## Objective and Constraints

Default objective for v0:

- Primary metric: `p99_latency_ms` (minimize)

Default soft secondary signal:

- `throughput_tokens_per_sec` (maximize)
- `masking_efficiency` (maximize)

Default hard constraints:

- (optional) `p99_latency_ms<=1200`

## Experiment hygiene

- One hypothesis per trial, documented per **Explainable hypothesis** and **Small-step edits** above.
- The runner creates `trials/<trial_id>/HYPOTHESIS.md` from a template at trial start; the agent must fill it before coding (see `agent_driver.py` prompt).
- Log every trial in JSONL.
- Keep a patch artifact for every trial.
- Promote only tested improvements.

## Suggested first search space

1. qTTT step schedule:
   - adaptive cap
   - warm-start / early-stop thresholds
2. AttnRes block/window policy:
   - modest block count adjustments
   - read-window strategy
3. Gating threshold calibration:
   - target-rate smoothing
   - entropy/max-probability blending

## Example run command

From repo root:

```bash
# One-time login for Cursor CLI agent:
cursor-agent login

# Optional: configure your external agent command template once.
# Placeholders: {workspace} {program} {trial_id} {trial_dir} {prompt}
export AUTORESEARCH_AGENT_CMD_TEMPLATE='echo "Plug in your real agent CLI here: {workspace}"'

python3 experiments/autoresearch/run_agent.py \
  --iterations 5 \
  --trial-cmd "python3 experiments/matdo/vllm_integration/latency_profiler.py" \
  --primary-metric p99_latency_ms \
  --primary-direction min \
  --constraint "p99_latency_ms<=1200"
```

Dry-run check (plans commands without executing agent/training/scoring):

```bash
python3 experiments/autoresearch/run_agent.py \
  --iterations 1 \
  --trial-cmd "python3 experiments/matdo/vllm_integration/latency_profiler.py" \
  --dry-run
```

If `AUTORESEARCH_AGENT_CMD_TEMPLATE` is not set, `run_agent.py` invokes
`agent_driver.py`, which defaults to:
`cursor-agent -p --trust --workspace <trial_worktree> "<generated_prompt>"`.

Scoring: `score_trial.py` writes `failure_reasons` in each trial’s `score.json` when a run is invalid.
For **dual objective** (throughput + P99 latency), pass `--objective dual` and ensure the trial emits
both `throughput_tokens_per_sec` and `p99_latency_ms` (see `docs/guides/AUTORESEARCH_GUIDE.md`).
