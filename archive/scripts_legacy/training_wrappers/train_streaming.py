#!/usr/bin/env python3
"""
Streaming Training Script for Adaptive Deep Networks
支持流式加载，适用于有限磁盘空间 (如 AutoDL 50GB 数据盘)

Features:
- 流式加载大数据集 (FineWeb, SlimPajama 等)
- 支持 4x H20-NVLink 分布式训练
- 支持 checkpoint 断点续训
- 零本地数据存储

Usage:
    # 单机单卡 Small 模型
    python scripts/train_streaming.py --model-size small --max-steps 10000
    
    # 单机4卡 Medium 模型
    torchrun --nproc_per_node=4 scripts/train_streaming.py --model-size medium
    
    # Large 模型 + DeepSpeed
    deepspeed --num_gpus=4 scripts/train_streaming.py --model-size large --use-deepspeed
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import numpy as np

from models.configs import get_config, get_model_size_params
from models.adaptive_transformer import create_adaptive_transformer


# ==============================================================================
# 流式数据集
# ==============================================================================


class StreamingTextDataset(IterableDataset):
    """
    流式文本数据集

    支持 HuggingFace datasets 的流式加载，不占用本地磁盘空间
    """

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "sample-10BT",  # 10B tokens 样本
        split: str = "train",
        seq_len: int = 2048,
        vocab_size: int = 32000,
        tokenizer=None,
        buffer_size: int = 100000,  # 预缓冲的 token 数
        streaming: bool = True,
    ):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.buffer_size = buffer_size

        # 延迟导入，避免没有安装 datasets 时出错
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        print(f"Loading streaming dataset: {dataset_name} ({dataset_config})")
        print(f"This may take a moment to initialize the stream...")

        # 流式加载数据集
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
        )

        # 简单的字符级 tokenizer（如果没有提供）
        self.tokenizer = tokenizer or self._simple_tokenizer

    def _simple_tokenizer(self, text: str) -> list:
        """简单的字符级 tokenizer（用于测试）"""
        # 实际使用时应该使用真正的 tokenizer
        return [ord(c) % self.vocab_size for c in text[: self.seq_len + 1]]

    def __iter__(self) -> Iterator[tuple]:
        """
        迭代生成训练样本

        Yields:
            (input_ids, targets): 输入序列和目标序列
        """
        buffer = []

        for item in self.dataset:
            # 获取文本
            text = item.get("text", item.get("content", ""))
            if not text:
                continue

            # Tokenize
            tokens = self.tokenizer(text)
            buffer.extend(tokens)

            # 当缓冲区足够时，生成样本
            while len(buffer) >= self.seq_len + 1:
                input_ids = buffer[: self.seq_len]
                targets = buffer[1 : self.seq_len + 1]

                yield (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(targets, dtype=torch.long),
                )

                # 滑动窗口（可配置 stride）
                buffer = buffer[self.seq_len :]  # 无重叠
                # buffer = buffer[self.seq_len // 2:]  # 50% 重叠


class DummyStreamingDataset(IterableDataset):
    """
    虚拟流式数据集（用于测试，无需网络）
    """

    def __init__(self, seq_len: int = 2048, vocab_size: int = 32000, seed: int = 42):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        self.step = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        while True:
            tokens = rng.randint(0, self.vocab_size, size=self.seq_len + 1)
            yield (
                torch.tensor(tokens[:-1], dtype=torch.long),
                torch.tensor(tokens[1:], dtype=torch.long),
            )
            self.step += 1


# ==============================================================================
# 分布式训练工具
# ==============================================================================


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ==============================================================================
# 训练函数
# ==============================================================================


def compute_loss(model, input_ids, targets, device):
    """Compute cross-entropy loss."""
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
    return loss


def train_step(model, input_ids, targets, optimizer, device, scaler=None):
    """Single training step with optional mixed precision."""
    optimizer.zero_grad()

    if scaler is not None:
        # Mixed precision training
        with torch.cuda.amp.autocast():
            loss = compute_loss(model, input_ids, targets, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss = compute_loss(model, input_ids, targets, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item()


def save_checkpoint(model, optimizer, step, loss, path, scaler=None, is_main_process=True):
    """Save training checkpoint."""
    if not is_main_process:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": (
            model.state_dict() if not hasattr(model, "module") else model.module.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, scaler=None, device=None):
    """Load training checkpoint."""
    if not os.path.exists(path):
        return 0, float("inf")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    print(f"Checkpoint loaded: {path}")
    return checkpoint.get("step", 0), checkpoint.get("loss", float("inf"))


# ==============================================================================
# 主函数
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Streaming Training for Adaptive Deep Networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Model size to train",
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length for training")

    # Training arguments
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Maximum training steps (流式数据集用 steps 代替 epochs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per GPU (default: auto based on model)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--warmup-steps", type=int, default=2000, help="Warmup steps for learning rate schedule"
    )

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="dummy",
        choices=["dummy", "fineweb", "slimpajama", "openwebtext"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="",
        help="Dataset configuration (e.g., sample-10BT for FineWeb)",
    )

    # System arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/autodl-tmp/checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save-every", type=int, default=5000, help="Save checkpoint every N steps"
    )
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    parser.add_argument("--use-deepspeed", action="store_true", help="Use DeepSpeed for training")
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default="scripts/ds_config_h20.json",
        help="DeepSpeed config file",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Use mixed precision (BF16/FP16) training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process:
        print("=" * 70)
        print("Adaptive Deep Networks - Streaming Training")
        print("=" * 70)
        print(f"Model size: {args.model_size}")
        print(f"Dataset: {args.dataset}")
        print(f"Max steps: {args.max_steps}")
        print(f"World size: {world_size} GPUs")
        print(f"Device: {device}")
        print("=" * 70)

    # Load model config
    config = get_config(args.model_size)
    param_count = get_model_size_params(config)

    if is_main_process:
        print(f"\nModel parameters: {param_count / 1e9:.2f}B")
        print(f"Layers: {config.num_layers}, Hidden: {config.hidden_dim}")

    # Auto-determine batch size if not specified
    if args.batch_size is None:
        batch_sizes = {
            "small": 8,
            "medium": 2,
            "large": 1,
        }
        args.batch_size = batch_sizes[args.model_size]
        if is_main_process:
            print(f"Auto batch size: {args.batch_size} per GPU")

    # Create model
    if is_main_process:
        print("\nInitializing model...")

    model = create_adaptive_transformer(args.model_size)
    model = model.to(device)

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    # Setup mixed precision scaler
    scaler = (
        torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    )

    # Load checkpoint if resuming
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        start_step, _ = load_checkpoint(model, optimizer, args.resume, scaler, device)
        if is_main_process:
            print(f"Resuming from step {start_step}")

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create streaming dataset
    if is_main_process:
        print(f"\nInitializing streaming dataset: {args.dataset}")

    # Dataset mapping
    dataset_map = {
        "dummy": ("dummy", ""),
        "fineweb": ("HuggingFaceFW/fineweb-edu", args.dataset_config or "sample-10BT"),
        "slimpajama": ("cerebras/SlimPajama-627B", ""),
        "openwebtext": ("openwebtext", ""),
    }

    dataset_name, dataset_config = dataset_map[args.dataset]

    if dataset_name == "dummy":
        dataset = DummyStreamingDataset(seq_len=args.seq_len, vocab_size=config.vocab_size)
    else:
        try:
            dataset = StreamingTextDataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                seq_len=args.seq_len,
                vocab_size=config.vocab_size,
            )
        except Exception as e:
            if is_main_process:
                print(f"Failed to load {dataset_name}: {e}")
                print("Falling back to dummy dataset")
            dataset = DummyStreamingDataset(seq_len=args.seq_len, vocab_size=config.vocab_size)

    # Create dataloader
    # 注意：流式数据集不能用普通 DataLoader 的 shuffle
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,  # 流式加载不支持多进程
    )

    # Training loop
    if is_main_process:
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)

    model.train()
    total_loss = 0
    step = start_step
    start_time = time.time()

    # Progress bar (only on main process)
    if is_main_process:
        pbar = tqdm(total=args.max_steps, initial=start_step, desc="Training")
    else:
        pbar = None

    # Infinite dataloader iteration
    data_iter = iter(dataloader)

    while step < args.max_steps:
        try:
            input_ids, targets = next(data_iter)
        except StopIteration:
            # Restart iterator if exhausted (shouldn't happen with streaming)
            data_iter = iter(dataloader)
            input_ids, targets = next(data_iter)

        # Training step
        loss = train_step(model, input_ids, targets, optimizer, device, scaler)
        total_loss += loss
        step += 1

        # Logging
        if is_main_process:
            if step % args.log_every == 0:
                avg_loss = total_loss / args.log_every
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    {"loss": f"{loss:.4f}", "avg": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"}
                )
                total_loss = 0

            pbar.update(1)

            # Save checkpoint
            if step % args.save_every == 0:
                checkpoint_path = os.path.join(args.output_dir, f"{args.model_size}_step{step}.pt")
                save_checkpoint(
                    model, optimizer, step, loss, checkpoint_path, scaler, is_main_process
                )

    # Final checkpoint
    if is_main_process:
        final_path = os.path.join(args.output_dir, f"{args.model_size}_final.pt")
        save_checkpoint(model, optimizer, step, loss, final_path, scaler, is_main_process)

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("Training Complete")
        print("=" * 70)
        print(f"Total steps: {step}")
        print(f"Total time: {elapsed / 3600:.2f} hours")
        print(f"Average speed: {step / elapsed:.2f} steps/sec")
        print(f"Final checkpoint: {final_path}")
        print("=" * 70)

    cleanup_distributed()


if __name__ == "__main__":
    main()
