# Implementation Readiness Check: QASP Gap Fixes

## Checklist

| Criterion | Status | Notes |
|---|---|---|
| PRD has clear acceptance criteria per story | PASS | All 7 stories have 4-5 ACs each |
| Dependencies are identified and available | PASS | PyTorch, pytest already in project; scipy may be needed for stats (optional fallback) |
| Story sizes are reasonable for single iteration | PASS | 8-15 min each = medium complexity |
| Technical feasibility is high | PASS | All stories are local to `QASP/`; no external APIs or training infra needed |
| No story blocked by another | CONCERNS | Story 1 (Stiefel query) changes attention API; Story 6 (generator) depends on prefill/step which is already implemented. Stories 2,5,7 are independent. |
| Tests can be run locally | PASS | `pytest tests/ -v` works per AGENTS.md |
| Existing contracts preserved | CONCERNS | Story 1 must gate changes behind `use_stiefel_query=False` default to avoid breaking parity tests |

## Risk Mitigation
- **Story 1 risk:** Add `use_stiefel_query: bool = False` to config. Default path is unchanged.
- **Cross-story risk:** Execute Story 1 first, verify tests, then proceed to others.
- **scipy risk:** If unavailable, implement bootstrap CI with numpy only; paired t-test uses standard formulas.

## Verdict
**PASS with guardrails** — Proceed with execution, defaulting all behavioral changes to off.
