#!/usr/bin/env python3
"""
Table 4 bitwidth sweep (REAL runtime measurement).

Measures:
- peak GPU memory (allocated)
- wall time per forward pass at long context
- tokens/sec (context_len / elapsed)

Notes:
- This measures *prefill-style* throughput (full forward over T tokens),
  not incremental decode throughput with KV reuse.
- It exercises the project's RaBitQ path by passing use_rabitq + rabitq_caches
  into AdaptiveTransformer.forward().
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from experiments.real_model.model_loader import load_adb_model


PAPER_TABLE4 = {
    "fp16": {"compression_ratio": 1.0, "storage_gb": 40.0, "tokens_per_sec": 25},
    "3bit": {"compression_ratio": 5.3, "storage_gb": 7.5, "tokens_per_sec": 89},
    "2bit": {"compression_ratio": 8.0, "storage_gb": 5.0, "tokens_per_sec": 105},
    "1bit": {"compression_ratio": 16.0, "storage_gb": 2.5, "tokens_per_sec": 115},
}


def measure_forward(
    model,
    input_ids: torch.Tensor,
    *,
    use_attnres: bool,
    use_rabitq: bool,
    total_bits: int | None,
    repeats: int,
) -> dict[str, Any]:
    if use_rabitq:
        if total_bits is None:
            raise ValueError("total_bits must be provided when use_rabitq=True")
        model.init_rabitq_caches(total_bits=total_bits, residual_window=128)

    # Warmup (build caches, kernels, etc.)
    torch.cuda.synchronize() if input_ids.is_cuda else None
    if input_ids.is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model(
            input_ids,
            use_attnres=use_attnres,
            use_rabitq=use_rabitq,
            rabitq_caches=getattr(model, "rabitq_caches", None) if use_rabitq else None,
        )

    torch.cuda.synchronize() if input_ids.is_cuda else None
    if input_ids.is_cuda:
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            torch.cuda.synchronize() if input_ids.is_cuda else None
            t0 = time.time()
            _ = model(
                input_ids,
                use_attnres=use_attnres,
                use_rabitq=use_rabitq,
                rabitq_caches=getattr(model, "rabitq_caches", None) if use_rabitq else None,
            )
            torch.cuda.synchronize() if input_ids.is_cuda else None
            times.append(time.time() - t0)

    peak_gb = None
    if input_ids.is_cuda:
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    mean_s = sum(times) / len(times)
    tps = input_ids.shape[1] / mean_s
    return {
        "repeats": repeats,
        "time_mean_s": mean_s,
        "time_min_s": min(times),
        "time_max_s": max(times),
        "tokens_per_sec": tps,
        "peak_allocated_gb": peak_gb,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark Table 4 bitwidth sweep (prefill throughput)."
    )
    p.add_argument("--checkpoint", type=str, default="", help="Checkpoint path (recommended)")
    p.add_argument("--size", type=str, default="medium", choices=["small", "medium", "large"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--context-len", type=int, default=131072)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--use-attnres", action="store_true", help="Enable AttnRes (default off for isolation)"
    )
    p.add_argument("--output", type=str, required=True, help="Output JSON path")
    args = p.parse_args()

    device = args.device
    ckpt = args.checkpoint.strip() or None
    model, config = load_adb_model(checkpoint_path=ckpt, model_size=args.size, device=device)
    model.eval()

    vocab_size = getattr(config, "vocab_size", 32000)
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.context_len), device=device)

    results: dict[str, Any] = {
        "meta": {
            "checkpoint": args.checkpoint or None,
            "size": args.size,
            "device": device,
            "context_len": args.context_len,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "use_attnres": bool(args.use_attnres),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "paper_table4_reference": PAPER_TABLE4,
        "measurements": {},
    }

    # FP16 baseline
    results["measurements"]["fp16"] = {
        "label": "FP16 baseline (no RaBitQ)",
        "use_rabitq": False,
        "total_bits": None,
        "paper": PAPER_TABLE4["fp16"],
        "measured": measure_forward(
            model,
            input_ids,
            use_attnres=bool(args.use_attnres),
            use_rabitq=False,
            total_bits=None,
            repeats=args.repeats,
        ),
    }

    # RaBitQ variants
    for name, bits in [("3bit", 3), ("2bit", 2), ("1bit", 1)]:
        results["measurements"][name] = {
            "label": f"RaBitQ total_bits={bits}",
            "use_rabitq": True,
            "total_bits": bits,
            "paper": PAPER_TABLE4[name],
            "measured": measure_forward(
                model,
                input_ids,
                use_attnres=bool(args.use_attnres),
                use_rabitq=True,
                total_bits=bits,
                repeats=args.repeats,
            ),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
