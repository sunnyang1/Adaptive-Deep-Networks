# AutoResearch Guide (ADN + Cursor CLI)

This guide explains how to run the first-version autonomous research loop in this repository.

The loop is implemented in:

- `experiments/autoresearch/program.md` (policy + objectives)
- `experiments/autoresearch/run_agent.py` (orchestrator)
- `experiments/autoresearch/agent_driver.py` (agent invocation bridge)
- `experiments/autoresearch/score_trial.py` (metric extraction + scoring)

---

## 1. What AutoResearch does

For each trial, AutoResearch:

1. Creates an isolated git worktree.
2. Invokes an agent to apply a small code/config change.
3. Runs a fixed evaluation command.
4. Scores the trial against a primary metric and constraints.
5. Logs everything and stores the best patch artifact.

This keeps the main workspace clean and makes trial diffs auditable.

---

## 2. Requirements

- Python 3.9+
- Repo dependencies installed (`pip install -e ".[dev]"`)
- `cursor-agent` CLI installed and authenticated

Check agent availability:

```bash
which cursor-agent
```

Authenticate once:

```bash
cursor-agent login
```

---

## 3. Quick start

### 3.1 Dry-run (recommended first)

Dry-run validates orchestration and command wiring without spending compute.

```bash
python3 experiments/autoresearch/run_agent.py \
  --iterations 1 \
  --trial-cmd "python3 experiments/matdo/vllm_integration/latency_profiler.py" \
  --dry-run \
  --skip-clean-check
```

What dry-run does:

- creates and removes trial worktree(s)
- writes planned agent/trial/score commands to logs
- does **not** run agent edits or experiments

### 3.2 Real run

```bash
python3 experiments/autoresearch/run_agent.py \
  --iterations 5 \
  --trial-cmd "python3 experiments/matdo/vllm_integration/latency_profiler.py" \
  --primary-metric p99_latency_ms \
  --primary-direction min \
  --constraint "p99_latency_ms<=1200"
```

### 3.2b Dual objective (throughput + latency)

Use when your evaluation produces **both** throughput and P99 latency under the trial workspace:

```bash
python3 experiments/autoresearch/run_agent.py \
  --iterations 5 \
  --trial-cmd "python3 your_eval_that_writes_both_metrics.py" \
  --objective dual \
  --dual-w-throughput 1.0 \
  --dual-w-latency 0.01 \
  --constraint "p99_latency_ms<=1200"
```

The latency-only profiler alone does **not** emit throughput; dual scoring will fail with reasons listed in `score.json` unless you add a throughput-producing step or `--metrics-file`.

### 3.3 Optional periodic deep validation stage

Run a heavier model-facing validator every N accepted trials (manual trigger):

```bash
python3 -m experiments.real_model.validator \
  --size small \
  --device mps \
  --test throughput \
  --output-dir results/real_model_autoresearch
```

This stage is intentionally decoupled from the per-trial fast loop to keep iteration speed high.

---

## 4. Agent invocation behavior

`run_agent.py` supports two paths:

1. **Custom command template** via `AUTORESEARCH_AGENT_CMD_TEMPLATE`
2. **Default Cursor path** via `cursor-agent` (if template is unset)

### 4.1 Default Cursor behavior

If no template is provided, `agent_driver.py` calls:

```text
cursor-agent -p --trust --workspace <trial_worktree> "<generated_prompt>"
```

### 4.2 Custom template (optional)

Set your own command template:

```bash
export AUTORESEARCH_AGENT_CMD_TEMPLATE='my-agent-cli --cwd "{workspace}" --prompt-file "{prompt}"'
```

Supported placeholders:

- `{workspace}`
- `{program}`
- `{trial_id}`
- `{trial_dir}`
- `{prompt}`

---

## 5. Scoring model

`score_trial.py` computes:

### Single objective (default)

- `--objective single`
- One `--primary-metric` and `--primary-direction` (`max` or `min`)
- Recommended fast-loop default: `p99_latency_ms` with `min`

### Dual objective (throughput + latency)

- `--objective dual` (via `run_agent.py` or `score_trial.py`)
- Requires **both** `throughput_tokens_per_sec` and `p99_latency_ms` in extracted metrics (or `--metrics-file`).
- Composite score (maximize):
`w_throughput * throughput_tokens_per_sec - w_latency * p99_latency_ms`
- Tune balance with `--dual-w-throughput` and `--dual-w-latency` (defaults: `1.0` and `0.01`).

Dual is only meaningful when your `trial-cmd` writes outputs that include **both** metrics (for example a validator that emits `validation_results.json` **and** timing JSON, or a wrapper that merges results into one directory).

### Constraints

- Constraint pass/fail checks (e.g. latency/memory) use the same metric names as in extracted JSON.

### Validity and `failure_reasons`

Trial is **valid** only when:

- the objective’s required metrics are found (and finite),
- all constraints pass,
- agent and trial commands succeeded (non-zero exit codes are recorded and, by default, invalidate the trial),
- the score is finite.

Every `score.json` includes `**failure_reasons`**: a list of human-readable strings explaining why a trial was invalid (missing metrics, failed constraints, non-zero agent/trial exit codes, etc.). Empty list means no failure was attributed at the scoring stage.

It writes per-trial score JSON (default path generated by `run_agent.py`).

---

## 6. Files and artifacts

AutoResearch writes to:

- `experiments/autoresearch/trials/<trial-id>/`
  - `HYPOTHESIS.md` (template filled by the agent per `program.md`; created at trial start)
  - `logs/agent.stdout.log`
  - `logs/agent.stderr.log`
  - `logs/trial.stdout.log`
  - `logs/trial.stderr.log`
  - `logs/score.stdout.log`
  - `logs/score.stderr.log`
  - `trial.patch`
  - `score.json`
- `experiments/autoresearch/trials/ledger-<run-id>.jsonl`
- `experiments/autoresearch/trials/summary-<run-id>.json`
- `experiments/autoresearch/best/best.patch`
- `experiments/autoresearch/best/best_score.json`

---

## 7. Safety and workflow recommendations

1. Start with small search spaces (qTTT steps, gating thresholds, AttnRes block settings).
2. Keep trial command fixed for comparability.
3. Use dry-run before every command/policy change.
4. Run periodic full benchmarks outside quick loop before promoting ideas.
5. Keep your main branch clean for production runs (unless intentionally using `--skip-clean-check` with dry-run).

---

## 8. Troubleshooting

### `cursor-agent` auth error

Symptom:

```text
Authentication required. Please run 'agent login' first...
```

Fix:

```bash
cursor-agent login
```

### Dirty working tree blocked

Symptom:

```text
Working tree is not clean...
```

Fix options:

- commit/stash local changes (recommended for real runs)
- for dry-run only, use `--skip-clean-check`

### No primary metric found

Cause: `score_trial.py` could not extract expected metric from trial outputs.

Fix:

- verify trial command actually produces known report JSONs
- switch `--primary-metric` to a metric present in your outputs
- optionally pass `--metrics-file` directly when calling `score_trial.py`

---

## 9. Suggested next upgrades

- Add `--promote-best` mode to auto-cherry-pick best patch to a review branch.
- Extend beyond dual composite (e.g. Pareto frontier or three-way objectives).
- Add mandatory smoke tests before scoring.
- Add per-trial budget guardrails (max edited files, max LOC touched).

