"""
Distributed Training Utilities

Handle multi-GPU setup and synchronization.
"""

import os
import torch
import torch.distributed as dist
from typing import Tuple, Optional


def setup_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
    """
    Setup distributed training.

    Args:
        backend: Backend to use ('nccl', 'gloo', 'mpi')

    Returns:
        (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: Optional[int] = None) -> bool:
    """Check if current process is main process."""
    if rank is not None:
        return rank == 0
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def barrier():
    """Synchronization barrier."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM):
    """All-reduce operation."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather_object(obj):
    """Gather object from all processes."""
    if not dist.is_initialized():
        return [obj]

    world_size = dist.get_world_size()
    output = [None] * world_size
    dist.all_gather_object(output, obj)
    return output
