# PRD: QASP Open Gap Fixes

## 1. Introduction

The QASP package implements the algorithmic skeleton described in the QASP paper (Stiefel projection, spectral quality score, value-weighted AttnRes, etc.) but has 6 documented open gaps vs. the paper's claims. This PRD closes all of them.

## 2. Goals

1. Bring model architecture in line with paper (1.5B-scale config, GQA, RoPE, SwiGLU)
2. Add a minimal but real training infrastructure (`train.py`, data loader, optimizer setup)
3. Create optional bridge modules so QASP can reuse `src/` ADN implementations
4. Replace all stub benchmark runners with real evaluation pipelines
5. Gate quality-score computation behind the ponder gate to realize the claimed ~70% overhead reduction

## 3. Non-Goals

- We will not train a real 1.5B checkpoint (no GPU budget in scope)
- We will not implement full distributed training (FSDP, DeepSpeed)
- We will not re-implement `src/` modules; we bridge to existing ones

## 4. User Stories

### Story QASP-ARCH-1: GQA Support
**As a** researcher, **I want** `CausalSelfAttention` to support Grouped Query Attention, **so that** KV-cache memory and param count match the paper's 4 KV-head design.

- Acceptance:
  - `QASPTransformerConfig` gains `num_key_value_heads` (default = `num_heads` for backward compatibility)
  - `CausalSelfAttention` reshapes K/V projections to `num_key_value_heads * head_dim`
  - K/V are repeated/broadcast to `num_heads` before `scaled_dot_product_attention`
  - All existing tests still pass when `num_key_value_heads == num_heads`
  - New test verifies shape correctness when `num_key_value_heads < num_heads`

### Story QASP-ARCH-2: RoPE Position Encoding
**As a** researcher, **I want** Rotary Position Embeddings (RoPE) instead of learned absolute embeddings, **so that** long-context extrapolation works as claimed.

- Acceptance:
  - Add `QASP.models.rope` module with `RotaryEmbedding` and `apply_rotary_pos_emb`
  - `QASPTransformerConfig` gains `use_rope: bool = False` (default false for backward compatibility)
  - When `use_rope=True`, `QASPTransformer` skips `position_embedding`, and `CausalSelfAttention` applies RoPE to Q/K inside `forward`/`forward_with_cache`/`step`
  - Existing tests pass with `use_rope=False`
  - New test verifies RoPE changes Q/K phases but preserves norms

### Story QASP-ARCH-3: 1.5B Paper Config
**As a** researcher, **I want** a factory preset that instantiates the paper's 1.5B model, **so that** experiments use the right dimensions.

- Acceptance:
  - Add `QASPTransformerConfig.paper_1_5b()` classmethod returning:
    - 24 layers, hidden 2048, 16 heads, 4 KV heads, head dim 128, intermediate 5504, vocab 32000, context 4096, 8 AttnRes blocks
  - `create_qasp_transformer(preset="paper_1_5b")` works
  - Test confirms param count is within ±5% of 1.5B

### Story QASP-TRAIN-1: Training Skeleton
**As a** researcher, **I want** a `train.py` script with AdamW, cosine schedule, gradient clipping, and BF16 support, **so that** I can start training on SlimPajama or any HuggingFace dataset.

- Acceptance:
  - `QASP/scripts/train.py` exists with argparse for: data path, output dir, epochs, batch size, LR, warmup, max seq len, gradient clipping, mixed precision
  - Uses HuggingFace `datasets` + `transformers` tokenizers (no custom loader required)
  - Logs loss to console and TensorBoard
  - Saves checkpoints every N steps
  - Includes `--quick` smoke-test mode (tiny model, 10 steps)
  - A test runs `--quick` to completion without error

### Story QASP-BRIDGE-1: Optional src/ Reuse
**As a** maintainer, **I want** QASP to optionally import `src.attnres`, `src.rabitq`, and `src.engram`, **so that** we avoid maintaining two copies of the same logic.

- Acceptance:
  - Add `QASP.integration` module with:
    - `try_import_src_attnres()` → returns `BlockAttnRes` if available, else `None`
    - `try_import_src_rabitq()` → returns `RaBitQCodec` from `src/rabitq/` if available
    - `try_import_src_engram()` → returns `Engram` from `src/engram/` if available
  - `QASPTransformerConfig` gains `use_src_bridge: bool = False`
  - When `use_src_bridge=True`, `QASPTransformer` uses `src.attnres.BlockAttnRes` instead of `ValueWeightedAttnRes`, and `src.engram.Engram` instead of `NgramMemory`+`ValueWeightedEngram`
  - Fallback gracefully when `src/` is not on PYTHONPATH
  - Test verifies bridge works when src is available and falls back when not

### Story QASP-BENCH-1: Real Benchmark Runners
**As a** researcher, **I want** `run_math_eval`, `run_qasp_ablation`, and `profile_qasp` to run actual model evaluations instead of returning hard-coded numbers.

- Acceptance:
  - `run_math_eval(model, dataset, num_samples)` runs the model on a HuggingFace math dataset (e.g. `hendrycks/competition_math`) and returns exact-match accuracy
  - `run_qasp_ablation(model, dataset, num_samples)` runs the model with each component toggled and returns real accuracy deltas
  - `profile_qasp(model, prompt_len, gen_len)` measures actual tokens/sec, peak memory, and latency via `torch.cuda.Event` or `time.perf_counter`
  - All three accept `model=None` and fall back to deterministic stubs for CI (preserve existing behaviour)
  - New integration tests verify real-mode runs without error on tiny random model

### Story QASP-OPT-1: Ponder-Gated Quality
**As a** researcher, **I want** `compute_quality_score` to run only when adaptation is likely, **so that** generation overhead drops ~70%.

- Acceptance:
  - `QASPTransformerConfig.gate_quality_computation` already exists (default False); wire it up
  - When `gate_quality_computation=True`:
    - `forward()` and `prefill()` still compute quality (needed for AttnRes/Engram correctness)
    - `step()` skips `compute_quality_score` unless `PonderGate.should_adapt(logits)` fires on the *previous* token's logits
    - When skipped, `step()` uses cached `per_token_quality` from prefill or last computed window
  - New test verifies `step()` calls `compute_quality_score` fewer times when gating is on
  - Existing parity tests still pass (with gating off)

## 5. Design & Technical Considerations

- **Backward compatibility**: All changes gated behind default-false flags or new optional args. Existing tests must pass without modification.
- **RoPE integration**: Use standard LLaMA-style RoPE with base 10000. Apply to Q/K head dim pairs.
- **GQA integration**: K/V projection output dim = `num_key_value_heads * head_dim`. Use `tensor.repeat_interleave(num_heads // num_key_value_heads, dim=1)` to expand to full head count before SDPA.
- **src/ bridge**: Use lazy imports inside functions, not top-level, to avoid import errors when `src/` is unavailable.
- **Benchmarks**: Use `datasets.load_dataset` with streaming for memory efficiency. For math, parse the last integer/latex answer and compare.

## 6. Success Metrics

- All 95 existing QASP tests pass
- At least 15 new tests added, all passing
- `pytest tests/ -k qasp` passes in under 10 seconds
- `python QASP/scripts/train.py --quick` completes in under 30 seconds
- QASP README updated listing which benchmarks are real vs. stub
