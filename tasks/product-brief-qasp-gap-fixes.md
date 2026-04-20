# Product Brief: QASP Code-Paper Alignment

## Strategic Vision
Close the critical gaps between the QASP paper claims (`QASP_paper_cn.md`) and the `QASP/` implementation so that the codebase becomes a credible, runnable research artifact. The paper presents QASP as an ADN extension with matrix-level Stiefel query adaptation, spectral quality weighting, and benchmark results. The code currently has the algorithmic skeleton but lacks correct semantics, real benchmarks, and integration hooks.

## Target Users
- Paper reviewers who may inspect code
- Internal researchers running ablations
- Future developers extending QASP into the main ADN pipeline

## Success Metrics
1. `QASPTransformer` can run a real needle-in-haystack evaluation end-to-end
2. Stiefel query participates in the differentiable attention path (not a disabled overlay)
3. Quality scores are computed only when the ponder gate permits (~30% tokens)
4. NgramMemory is written during prefill and read during generation
5. All changes pass existing tests + new tests for added functionality
6. No synthetic stub data in the primary benchmark paths

## Constraints
- Must preserve existing test contracts (especially `test_qasp_prefill_step_numeric_parity.py`)
- Must not break the ADN project structure; QASP remains under `QASP/`
- Training a 1.5B model is **out of scope** — we fix the research artifact, not train
- Must work on CPU for smoke tests; CUDA is optional

## Non-Goals
- Full LLaMA architecture rewrite (GQA, RoPE, SwiGLU) — listed as future work
- Training pipeline or data loaders
- Integration with `src/` ADN modules in this iteration
- FlashAttention-2 kernel implementation
