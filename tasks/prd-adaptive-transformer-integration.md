# PRD: Adaptive Transformer Algorithm Integration Fix

## Objective
Fix the integration gaps between RaBitQ, AttnRes, and Polar qTTT inside `AdaptiveTransformer` so that the three query-optimization stages compose correctly as described in the paper.

## User Stories

### US-101: Fix PolarQTTT margin loss indexing bug
- **What**: `PolarQTTT._compute_margin_loss()` uses `logits[..., target_token_ids].max(dim=-1)` which computes the max over target positions instead of gathering the exact target logit per position.
- **Acceptance**: `PolarQTTT` unit test passes with proper margin loss semantics.

### US-102: Replace Euclidean qTTT with PolarQTTT in generate()
- **What**: `AdaptiveTransformer.generate()` currently instantiates `QueryOnlyTTT` (Euclidean). It must use `PolarQTTT` with `PolarQTTTConfig` to match paper Section 4.4.
- **Acceptance**: `generate(use_qttt=True)` calls `PolarQTTT.adapt_query_projection()` or `adapt_pseudo_query()`.

### US-103: Incremental KV cache in generation
- **What**: `generate()` currently rebuilds the full KV cache every token via `get_kv_cache(output_ids)`. This is O(T×L) per step and unusable.
- **Acceptance**: Generation uses an incremental KV cache (only compute K/V for the newest token and append).

### US-104: Integrate RaBitQ into AdaptiveTransformer KV cache
- **What**: `get_kv_cache()` returns full-precision `List[KVCache]`. Replace with `RaBitQCache` (or a per-layer RaBitQ quantizer) so KV states are compressed after the residual window.
- **Acceptance**: `AdaptiveTransformer` can be configured with `use_rabitq=True`; memory stats show >4× compression.

### US-105: Wire RaBitQ cache into qTTT attention
- **What**: qTTT's `compute_attention_with_query()` expects `KVCache` with `keys/values` tensors. It needs to transparently decompress RaBitQ quantized keys/values before attention.
- **Acceptance**: `PolarQTTT` can accept a `RaBitQCache` wrapper that decompresses on `get_kv()`.

### US-106: Green test suite after all changes
- **What**: Ensure `pytest tests/unit/test_qttt.py tests/unit/test_models.py tests/unit/test_rabitq.py tests/unit/test_attnres_integration.py` all pass.
- **Acceptance**: ≥120 tests pass, zero regressions in core modules.

### US-107: Adversarial review & final integration check
- **What**: Review `AdaptiveTransformer.forward()`, `generate()`, gating controller wiring, and AttnRes final aggregation for correctness.
- **Acceptance**: Document any remaining gaps; fix P0/P1 issues found.
