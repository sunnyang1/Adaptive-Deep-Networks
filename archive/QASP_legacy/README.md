# QASP — Query-Adaptive Spectral Projection

This package implements the algorithmic skeleton of the QASP paper:
*Stiefel projection, spectral quality score, value-weighted AttnRes, value-weighted Engram, ponder gate, KV cache, and incremental prefill/step decoding.*

## Architecture

| Component | Status | Notes |
|---|---|---|
| GQA (Grouped Query Attention) | ✅ | `num_key_value_heads` config field; K/V head repetition via `_repeat_kv` |
| RoPE (Rotary Position Embedding) | ✅ | Optional (`use_rope=True`); replaces learned absolute embeddings |
| SwiGLU MLP | ✅ | Already implemented as `FeedForward` |
| 1.5B Paper Preset | ✅ | `QASPTransformerConfig.paper_1_5b()` + `create_qasp_transformer(preset="paper_1_5b")` |

Default configs remain backward-compatible (GQA off, RoPE off, learned embeddings).

## Training

A minimal training script is provided:

```bash
python -m QASP.scripts.train --quick  # smoke test
python -m QASP.scripts.train \
    --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
    --output_dir ./checkpoints \
    --num_epochs 1 \
    --batch_size 4 \
    --max_seq_len 512 \
    --use_rope \
    --learning_rate 3e-4
```

Features: AdamW, cosine schedule with warmup, gradient clipping, mixed-precision support, checkpoint saving.

## Integration with `src/` (ADN)

QASP can optionally reuse modules from the parent `src/` package via `QASP.integration`:

```python
from QASP.integration import try_import_src_attnres, SrcAttnResAdapter

if try_import_src_attnres() is not None:
    adapter = SrcAttnResAdapter(hidden_size=2048, num_blocks=8)
```

Adapters are provided for `src.attnres.BlockAttnRes`, `src.engram.Engram`, and `src.rabitq.RaBitQ`.
Set `QASPTransformerConfig.use_src_bridge=True` to enable (adapters are instantiated lazily).

## Benchmarks

| Benchmark | Real / Stub | Entry Point |
|---|---|---|
| Needle-in-Haystack | ✅ Real | `QASP.experiments.benchmarks.needle.run_needle_benchmark` |
| MATH | ✅ Real (with fallback) | `QASP.experiments.benchmarks.math_eval.run_math_eval` |
| Component Ablation | ✅ Real (with fallback) | `QASP.experiments.ablations.qasp_ablation.run_qasp_ablation` |
| Efficiency Profile | ✅ Real (with fallback) | `QASP.experiments.efficiency.profile.profile_qasp` |

All benchmark runners accept `model=None` and fall back to deterministic stubs for CI.

## Quality Computation Gating

To realize the ~70% overhead reduction claimed in the paper, set:

```python
config.gate_quality_computation = True
```

During incremental `step()`, `compute_quality_score` is skipped when the ponder gate indicates low uncertainty. AttnRes is also skipped for that step. `forward()` and `prefill()` always compute quality for correctness.

## Tests

```bash
pytest tests/ -k qasp -v
```

113 tests passing (95 original + 18 new).

## Canonical Semantics

As documented in `QASP/__init__.py`:

- `forward()` and `prefill()` implement the paper's **full-sequence** value-weighted AttnRes definition.
- `step()` uses prefix-growing block statistics; logits need not match a hypothetical full forward over the extended sequence.
