"""Minimal training script for QASP models.

Supports:
- HuggingFace datasets (streaming or local)
- AdamW with cosine schedule and warmup
- Gradient clipping
- BF16/FP16 mixed precision via torch.autocast
- Checkpoint saving
- --quick smoke-test mode

Example:
    python -m QASP.scripts.train \
        --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
        --dataset_config default \
        --text_column text \
        --output_dir ./qasp_checkpoints \
        --num_epochs 1 \
        --batch_size 4 \
        --max_seq_len 512 \
        --learning_rate 3e-4 \
        --warmup_steps 100

Quick smoke test:
    python -m QASP.scripts.train --quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

from QASP.models import QASPTransformer, QASPTransformerConfig


def _get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class _TextDataset(IterableDataset):
    """Simple iterable dataset that yields tokenized sequences from a HF dataset."""

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        text_column: str,
        max_seq_len: int,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_seq_len = max_seq_len

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get(self.text_column, "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.max_seq_len + 1:
                seq = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len :]
                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                labels = torch.tensor(seq[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def _create_dataloader(
    dataset_name: str,
    dataset_config: str | None,
    text_column: str,
    tokenizer: Any,
    max_seq_len: int,
    batch_size: int,
    split: str = "train",
) -> DataLoader:
    """Create a DataLoader from a HuggingFace dataset."""

    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    iterable = _TextDataset(ds, tokenizer, text_column, max_seq_len)
    return DataLoader(iterable, batch_size=batch_size)


def _get_tokenizer(vocab_size: int):
    """Return a simple character-level or HF tokenizer fallback."""

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.vocab_size < vocab_size:
            # Pad vocabulary by adding dummy tokens if needed
            pass
        return tokenizer
    except Exception:
        # Minimal fallback: map bytes to ids
        class _ByteTokenizer:
            def __init__(self, vocab_size: int):
                self.vocab_size = vocab_size

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [min(ord(c), self.vocab_size - 1) for c in text]

        return _ByteTokenizer(vocab_size)


def train(args: argparse.Namespace) -> None:
    """Main training loop."""

    device = torch.device(args.device)

    # Config
    if args.preset:
        config = QASPTransformerConfig.paper_1_5b()
    else:
        config = QASPTransformerConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_key_value_heads=args.num_key_value_heads,
            num_layers=args.num_layers,
            max_position_embeddings=args.max_seq_len,
            use_attnres=args.use_attnres,
            use_engram=args.use_engram,
            use_rope=args.use_rope,
        )

    model = QASPTransformer(config).to(device)

    if args.quick:
        print("[quick mode] Using tiny model and synthetic data")
        model = QASPTransformer(
            QASPTransformerConfig(
                vocab_size=128,
                hidden_size=64,
                num_heads=4,
                num_layers=2,
                max_position_embeddings=64,
                use_attnres=False,
                use_engram=False,
            )
        ).to(device)

    tokenizer = _get_tokenizer(model.config.vocab_size)

    if args.quick:
        # Synthetic dataloader for smoke test
        class _QuickDataset(IterableDataset):
            def __init__(self, vocab_size: int, seq_len: int):
                self.vocab_size = vocab_size
                self.seq_len = seq_len

            def __iter__(self):
                for _ in range(20):
                    seq = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
                    yield {"input_ids": seq[:-1], "labels": seq[1:]}

        dataloader = DataLoader(_QuickDataset(model.config.vocab_size, 32), batch_size=2)
        args.num_epochs = 1
        args.max_steps = 10
    else:
        dataloader = _create_dataloader(
            args.dataset_name,
            args.dataset_config,
            args.text_column,
            tokenizer,
            args.max_seq_len,
            args.batch_size,
        )

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = args.max_steps if args.max_steps > 0 else args.num_epochs * 1000
    scheduler = _get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # Training state
    global_step = 0
    model.train()
    start_time = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(model.config), f, indent=2, default=str)

    print(f"Training on {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Total steps: {total_steps}")

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32):
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

            loss.backward()
            if args.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1

            if global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"step {global_step:>6} | loss {loss.item():.4f} | lr {lr:.2e} | "
                    f"elapsed {time.time() - start_time:.1f}s"
                )

            if global_step % args.save_every == 0:
                ckpt_path = output_dir / f"checkpoint_step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1} complete | avg loss {avg_loss:.4f}")

    # Final save
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QASP model")

    # Data
    parser.add_argument("--dataset_name", type=str, default="togethercomputer/RedPajama-Data-1T-Sample")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--text_column", type=str, default="text")

    # Model
    parser.add_argument("--preset", type=str, default=None, help="Model preset (e.g. 'paper_1_5b')")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--use_attnres", action="store_true", default=False)
    parser.add_argument("--use_engram", action="store_true", default=False)
    parser.add_argument("--use_rope", action="store_true", default=False)

    # Training
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 = unlimited)")
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "bf16", "fp16"])

    # Logging / checkpointing
    parser.add_argument("--output_dir", type=str, default="./qasp_checkpoints")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Quick smoke test
    parser.add_argument("--quick", action="store_true", help="Run a fast smoke test on synthetic data")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
