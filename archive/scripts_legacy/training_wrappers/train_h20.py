#!/usr/bin/env python3
"""
H20-NVLink 4卡训练脚本
针对 H20 96GB × 4 配置优化
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "src"))

from models.configs import get_config, get_model_size_params
from models.adaptive_transformer import create_adaptive_transformer


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


def get_gpu_memory_info():
    """Get GPU memory info for H20."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1e9
        return {
            "total_gb": total_memory,
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
        }
    return None


def estimate_batch_size(model_size, num_gpus):
    """
    根据 H20 96GB 显存估算 batch size
    """
    configs = {
        "small": {
            "params": 2.2e9,
            "single_gpu_batch": 8,  # 96GB 单卡可承载
            "gpu_memory_per_sample": 10,  # GB per sample
        },
        "medium": {
            "params": 8.7e9,
            "single_gpu_batch": 2,  # 96GB 单卡可承载
            "gpu_memory_per_sample": 40,
        },
        "large": {
            "params": 27e9,
            "single_gpu_batch": 1,  # 需 DeepSpeed ZeRO-3 + 4卡
            "gpu_memory_per_sample": 80,
            "min_gpus": 4,
        },
    }

    if model_size not in configs:
        return 1

    config = configs[model_size]

    # 单 GPU 估算
    if num_gpus == 1:
        return config["single_gpu_batch"]

    # 多 GPU (数据并行)
    if model_size in ["small", "medium"]:
        # 数据并行可以线性扩展 batch size
        return config["single_gpu_batch"] * num_gpus
    else:
        # Large 模型需要至少 4 卡
        if num_gpus < 4:
            return None  # 返回 None 表示需要更多 GPU
        return config["single_gpu_batch"] * num_gpus


def main():
    parser = argparse.ArgumentParser(description="Train on H20 cluster")
    parser.add_argument(
        "--model-size", type=str, default="medium", choices=["small", "medium", "large"]
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Auto-estimated if not specified"
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/checkpoints")
    parser.add_argument(
        "--use-deepspeed", action="store_true", help="Use DeepSpeed for Large model"
    )
    parser.add_argument("--deepspeed-config", type=str, default="scripts/ds_config_h20.json")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    if is_main_process:
        print("=" * 70)
        print("H20-NVLink Training")
        print("=" * 70)
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"GPU: {gpu_info['name']}")
            print(f"Total GPUs: {gpu_info['count']}")
            print(f"Per GPU Memory: {gpu_info['total_gb']:.0f} GB")
            print(f"Total Memory: {gpu_info['total_gb'] * gpu_info['count']:.0f} GB")
        print(f"World Size: {world_size}")
        print("=" * 70)

    # Get config
    config = get_config(args.model_size)
    param_count = get_model_size_params(config)

    if is_main_process:
        print(f"\nModel: {args.model_size}")
        print(f"Parameters: {param_count / 1e9:.2f}B")
        print(f"Layers: {config.num_layers}, Hidden: {config.hidden_dim}")

    # Auto-estimate batch size if not specified
    if args.batch_size is None:
        args.batch_size = estimate_batch_size(args.model_size, world_size)
        if is_main_process:
            print(
                f"Auto-estimated batch size: {args.batch_size} (per GPU: {args.batch_size // world_size})"
            )

    # Create model
    if is_main_process:
        print("\nInitializing model...")

    model = create_adaptive_transformer(args.model_size)

    # Move to GPU
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    # Wrap with DDP if multi-GPU
    if world_size > 1 and not args.use_deepspeed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total_params / 1e9:.2f}B parameters")
        print(f"Device: {device}")

    # Training would continue here...
    # For now, just print configuration
    if is_main_process:
        print("\n" + "=" * 70)
        print("Training Configuration:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Global Batch Size: {args.batch_size}")
        print(f"  Per-GPU Batch Size: {args.batch_size // world_size}")
        print(f"  Sequence Length: {args.seq_len}")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Gradient Accumulation: {args.gradient_accumulation}")
        print(f"  Output Dir: {args.output_dir}")
        print("=" * 70)
        print("\nReady for training!")
        print("\nExample command to actually train:")
        if world_size > 1:
            print(f"  torchrun --nproc_per_node=4 scripts/training/train_model.py \\")
            print(f"    --model-size {args.model_size} \\")
            print(f"    --batch-size {args.batch_size // world_size} \\")
            print(f"    --epochs {args.epochs}")
        else:
            print(f"  python scripts/training/train_model.py \\")
            print(f"    --model-size {args.model_size} \\")
            print(f"    --batch-size {args.batch_size} \\")
            print(f"    --epochs {args.epochs}")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
