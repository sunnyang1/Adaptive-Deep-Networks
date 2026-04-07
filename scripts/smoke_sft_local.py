#!/usr/bin/env python3
"""Local SFT smoke test: tiny Qwen2.5-0.5B + LoRA, a few steps (no Hub push)."""

from __future__ import annotations

import sys


def _pick_device() -> str:
    """Prefer CUDA; use CPU otherwise.

    MPS is skipped: some PyTorch/macOS combos pass a trivial MPS smoke test but still
    fail inside Transformers' TrainingArguments device setup.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    device = _pick_device()
    use_fp16 = device == "cuda"

    ds = load_dataset("trl-lib/Capybara", split="train")
    n = min(48, len(ds))
    train_ds = ds.select(range(n))

    no_cuda = device != "cuda"

    trainer = SFTTrainer(
        model="Qwen/Qwen2.5-0.5B",
        train_dataset=train_ds,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        ),
        args=SFTConfig(
            output_dir="smoke-sft-out-local",
            max_steps=5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="no",
            push_to_hub=False,
            bf16=False,
            fp16=use_fp16,
            report_to="none",
            dataloader_num_workers=0,
            no_cuda=no_cuda,
            use_mps_device=False,
        ),
    )

    trainer.train()
    print("LOCAL_SMOKE_OK device=", device, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as e:
        print(
            "Missing deps. Install with (Python 3.9+ recommended):\n"
            "  python3 -m pip install 'trl>=0.12.0' peft datasets accelerate "
            "'transformers>=4.44.0' torch",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        raise SystemExit(1) from e
