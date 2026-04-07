#!/usr/bin/env python3
"""
Upgrade an existing ADN checkpoint to an Engram-enabled checkpoint.

This script:
1) Loads the original checkpoint state_dict
2) Builds an Engram-enabled model (AdaptiveTransformerWithEngram) using the same base ModelConfig
3) Loads original weights with strict=False (missing Engram weights are randomly initialized)
4) Saves a new checkpoint containing the full state_dict, plus a sidecar config JSON

Example:
  python3 scripts/model/upgrade_checkpoint_add_engram.py \
    --in checkpoints/adb_medium.pt \
    --size medium \
    --out checkpoints/adb_medium_engram.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.models.configs import get_config
from src.models.adaptive_transformer import AdaptiveTransformer
from src.engram.integration import add_engram_to_config
from src.engram.config import EngramSmallConfig, EngramMediumConfig, EngramLargeConfig


def _extract_state_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # raw state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError("Unrecognized checkpoint format (expected state_dict or wrapper dict).")


def main() -> None:
    p = argparse.ArgumentParser(description="Upgrade ADN checkpoint to Engram-enabled checkpoint.")
    p.add_argument("--in", dest="in_path", required=True, help="Input checkpoint path")
    p.add_argument("--out", dest="out_path", required=True, help="Output checkpoint path")
    p.add_argument("--size", choices=["small", "medium", "large"], default="medium", help="Model size config")
    p.add_argument("--device", default="cpu", help="cpu|cuda (loading/saving can be done on cpu)")
    p.add_argument("--strict-base", action="store_true", help="If set, require base weights to match strictly (excluding Engram)")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_path, map_location=args.device)
    base_sd = _extract_state_dict(ckpt)

    base_cfg = get_config(args.size)
    engram_cfg = {"small": EngramSmallConfig, "medium": EngramMediumConfig, "large": EngramLargeConfig}[args.size]
    cfg = add_engram_to_config(base_cfg, engram_cfg)

    model = AdaptiveTransformer(cfg).to(args.device)
    model.eval()

    # Load base weights (ignore missing engram.* keys)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print(f"Loaded base checkpoint with strict=False.")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    # Optional: ensure we didn't drop many non-engram keys
    if args.strict_base:
        non_engram_missing = [k for k in missing if "engram" not in k]
        if non_engram_missing:
            raise RuntimeError(f"Missing non-Engram keys under --strict-base: {non_engram_missing[:10]}")

    # Save new checkpoint as raw state_dict for compatibility with loader
    torch.save(model.state_dict(), out_path)
    print(f"Saved upgraded checkpoint: {out_path}")

    # Save a sidecar config for transparency
    config_sidecar = out_path.with_suffix(".engram_config.json")
    config_sidecar.write_text(
        json.dumps(
            {
                "base_size": args.size,
                "use_engram": True,
                "engram_config": {
                    "enabled": cfg.engram_config.enabled if cfg.engram_config else None,
                    "engram_vocab_size": cfg.engram_config.engram_vocab_size if cfg.engram_config else None,
                    "max_ngram_size": cfg.engram_config.max_ngram_size if cfg.engram_config else None,
                    "n_embed_per_ngram": cfg.engram_config.n_embed_per_ngram if cfg.engram_config else None,
                    "n_head_per_ngram": cfg.engram_config.n_head_per_ngram if cfg.engram_config else None,
                    "layer_ids": cfg.engram_config.layer_ids if cfg.engram_config else None,
                    "tokenizer_name_or_path": cfg.engram_config.tokenizer_name_or_path if cfg.engram_config else None,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote sidecar config: {config_sidecar}")


if __name__ == "__main__":
    main()

