# Pre-mortem Analysis: QASP Gap Fixes

## Assumed Failure
We finish all 7 stories, run the tests, and a paper reviewer still dismisses QASP as "not credible" because:
1. The benchmark numbers are still synthetic (no trained model)
2. The Stiefel query path diverges from standard Transformers enough that parity tests break
3. Quality-score caching introduces a subtle bug where stale scores leak across batches
4. NgramMemory CPU-side writes make 128K-context prefill unbearably slow

## Root Causes & Mitigations

### Root Cause 1: No trained model → synthetic accuracy
- **Mitigation:** Be explicit in documentation that these fixes make the *artifact runnable*, not the *model trained*. The benchmark runner is real; the model weights are random. Add a prominent `README.md` in `QASP/experiments/` stating this.

### Root Cause 2: Stiefel query changes attention shape contract
- **Mitigation:** Keep `q_proj` as a fallback. Add a `use_stiefel_query=False` default so existing tests are untouched. New tests validate the Stiefel path independently.

### Root Cause 3: Stale quality cache across batches
- **Mitigation:** Never cache across `forward` calls. Cache only *within* a single `forward/prefill` call, across layers. Use a local variable or dataclass, not instance state that persists.

### Root Cause 4: CPU n-gram writes on long contexts
- **Mitigation:** Batch the writes. Instead of per-token Python loops inside the model, accumulate n-gram tuples in a tensor and write after the forward pass. Document the performance limitation.

## Inversion Check
How do we guarantee failure?
- Change `q_proj` globally without a flag → breaks all existing tests
- Store quality scores as a persistent buffer → batch leak
- Call `compute_quality_score` eagerly in every layer → no efficiency gain
- Remove synthetic stubs without adding real runner → CI fails because nothing returns metrics

## Conclusion
The main risk is over-promising what the code can demonstrate. The PRD is scoped to "runnable artifact," not "trained model." We will gate every behavioral change behind config flags and invalidate caches aggressively.
