#!/usr/bin/env python3
"""
Table 5 component sweep on Needle-in-Haystack (REAL path).

Runs a small ablation grid on the same NeedleDataset protocol used by
experiments/real_model/needle_haystack_real.py, but with controlled
generation flags:
- baseline (no RaBitQ, no AttnRes, no qTTT)
- +RaBitQ
- +AttnRes
- +qTTT (with AttnRes and optionally RaBitQ)

Important:
- "+Engram" requires an Engram-enabled checkpoint/architecture; this script will
  report Engram as UNSUPPORTED unless the model exposes compatible hooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.real_model.model_loader import load_adb_model
from experiments.real_model.needle_haystack_real import NeedleHaystackValidator


PAPER_TABLE5 = {
    4096: {"baseline": 87.5, "rabitq": 96.8, "attnres": 97.2, "engram": 98.0, "full": 98.5},
    32768: {"baseline": 22.1, "rabitq": 68.4, "attnres": 78.9, "engram": 86.2, "full": 91.8},
    131072: {"baseline": 3.2, "rabitq": 42.1, "attnres": 64.5, "engram": 75.8, "full": 79.5},
    262144: {"baseline": 1.5, "rabitq": 28.7, "attnres": 51.2, "engram": 64.3, "full": 69.0},
}


def run_one(model, device: str, lengths: list[int], num_samples: int, generate_kwargs: dict[str, Any]) -> dict[str, Any]:
    v = NeedleHaystackValidator(model, device=device, generate_kwargs=generate_kwargs)
    return v.run_test(context_lengths=lengths, num_samples=num_samples)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark Table 5 component sweep (Needle-in-Haystack).")
    p.add_argument("--checkpoint", type=str, default="", help="Checkpoint path (recommended)")
    p.add_argument("--size", type=str, default="medium", choices=["small", "medium", "large"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768, 131072], help="Context lengths to test")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--rabitq-bits", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--qttt-steps", type=int, default=10)
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    ckpt = args.checkpoint.strip() or None
    model, config = load_adb_model(checkpoint_path=ckpt, model_size=args.size, device=args.device)
    model.eval()

    lengths = list(args.lengths)

    # Define sweep configs (Table 5 style)
    configs: list[tuple[str, dict[str, Any]]] = [
        ("baseline", {"use_attnres": False, "use_engram": False, "use_qttt": False, "use_rabitq": False}),
        ("+rabitq", {"use_attnres": False, "use_engram": False, "use_qttt": False, "use_rabitq": True}),
        ("+attnres", {"use_attnres": True, "use_engram": False, "use_qttt": False, "use_rabitq": True}),
        ("+engram", {"use_attnres": True, "use_engram": True, "use_qttt": False, "use_rabitq": True}),
        ("full(+qttt)", {
            "use_attnres": True,
            "use_engram": True,
            "use_qttt": True,
            "use_rabitq": True,
            "qttt_config": {"num_steps": int(args.qttt_steps)},
        }),
    ]

    # Prepare RaBitQ caches once if any config needs it
    if any(cfg.get("use_rabitq") for _, cfg in configs):
        model.init_rabitq_caches(total_bits=int(args.rabitq_bits), residual_window=128)

    results: dict[str, Any] = {
        "meta": {
            "checkpoint": args.checkpoint or None,
            "size": args.size,
            "device": args.device,
            "lengths": lengths,
            "num_samples": args.num_samples,
            "rabitq_bits": args.rabitq_bits,
            "qttt_steps": args.qttt_steps,
            "torch": torch.__version__,
        },
        "paper_table5_reference": PAPER_TABLE5,
        "runs": {},
        "engram": {"status": "RUNTIME_TOGGLE", "reason": "Engram enabled/disabled via use_engram flag."},
    }

    for name, gen_kwargs in configs:
        print("\n" + "=" * 70)
        print(f"Table 5 config: {name}")
        print("=" * 70)
        out = run_one(model, args.device, lengths, args.num_samples, gen_kwargs)
        results["runs"][name] = {"generate_kwargs": gen_kwargs, "results": out}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

