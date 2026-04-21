"""
Incremental State Management for Efficient Generation

Provides data structures and utilities for managing incremental generation state,
including KV caches and AttnRes block representations.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from src.qttt.adaptation import KVCache


@dataclass
class IncrementalState:
    """
    Complete state for incremental generation.

    This class encapsulates all mutable state that needs to be maintained
    across generation steps, enabling O(L) incremental updates instead of
    O(T×L) full recomputation.

    Attributes:
        kv_caches: List of KVCache, one per layer [num_layers]
        block_representations: List of completed block outputs [num_blocks]
        partial_block: Current incomplete block accumulation [B, T, D]
        seq_len: Current sequence length
        num_layers: Total number of layers (for validation)
        num_blocks: Total number of blocks (for validation)

    Example:
        >>> state = IncrementalState(
        ...     kv_caches=[KVCache(k, v) for _ in range(num_layers)],
        ...     block_representations=[hidden],
        ...     partial_block=torch.zeros_like(hidden),
        ...     seq_len=10,
        ...     num_layers=32,
        ...     num_blocks=8
        ... )
    """

    kv_caches: List[KVCache]
    block_representations: List[torch.Tensor]
    partial_block: torch.Tensor
    seq_len: int
    num_layers: int
    num_blocks: int

    def __post_init__(self):
        """Validate state consistency."""
        if not validate_state(self, raise_on_error=False):
            raise ValueError("Invalid IncrementalState: validation failed")

    def get_cache_for_layer(self, layer_idx: int) -> KVCache:
        """
        Get KV cache for specific layer.

        Args:
            layer_idx: Layer index (0 to num_layers-1)

        Returns:
            KVCache for the layer

        Raises:
            IndexError: If layer_idx is out of range
        """
        if layer_idx < 0 or layer_idx >= len(self.kv_caches):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {len(self.kv_caches)})")
        return self.kv_caches[layer_idx]

    def update_cache_for_layer(self, layer_idx: int, new_kv: KVCache) -> None:
        """
        Update KV cache for specific layer.

        Args:
            layer_idx: Layer index
            new_kv: New KV cache to replace old one
        """
        if layer_idx < 0 or layer_idx >= len(self.kv_caches):
            raise IndexError(f"Layer index {layer_idx} out of range")
        self.kv_caches[layer_idx] = new_kv

    def append_to_cache(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """
        Append new K, V to existing cache for a layer.

        This is the core operation for incremental updates.

        Args:
            layer_idx: Layer index
            new_k: New key tensor [B, H, 1, D]
            new_v: New value tensor [B, H, 1, D]
        """
        if layer_idx < 0 or layer_idx >= len(self.kv_caches):
            raise IndexError(f"Layer index {layer_idx} out of range")

        old_cache = self.kv_caches[layer_idx]
        updated_cache = concat_kv_cache(old_cache, new_k, new_v)
        self.kv_caches[layer_idx] = updated_cache

    def add_block_representation(self, block_output: torch.Tensor) -> None:
        """
        Add a completed block representation.

        Args:
            block_output: Output tensor for completed block [B, T, D]
        """
        self.block_representations.append(block_output)

    def get_latest_block_representation(self) -> Optional[torch.Tensor]:
        """
        Get the most recent block representation.

        Returns:
            Latest block representation or None if no blocks
        """
        if not self.block_representations:
            return None
        return self.block_representations[-1]

    def increment_seq_len(self, num_tokens: int = 1) -> None:
        """
        Increment sequence length counter.

        Args:
            num_tokens: Number of new tokens (default: 1)
        """
        self.seq_len += num_tokens

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of state
        """
        return {
            "kv_caches": [
                (cache.keys.cpu().numpy(), cache.values.cpu().numpy()) for cache in self.kv_caches
            ],
            "block_representations": [b.cpu().numpy() for b in self.block_representations],
            "partial_block": self.partial_block.cpu().numpy(),
            "seq_len": self.seq_len,
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
        }

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory stats in bytes
        """
        kv_memory = sum(
            cache.keys.numel() * cache.keys.element_size()
            + cache.values.numel() * cache.values.element_size()
            for cache in self.kv_caches
        )

        block_memory = sum(b.numel() * b.element_size() for b in self.block_representations)

        partial_memory = self.partial_block.numel() * self.partial_block.element_size()

        return {
            "kv_caches": kv_memory,
            "block_representations": block_memory,
            "partial_block": partial_memory,
            "total": kv_memory + block_memory + partial_memory,
        }


def concat_kv_cache(existing: KVCache, new_k: torch.Tensor, new_v: torch.Tensor) -> KVCache:
    """
    Concatenate new K, V to existing KV cache.

    This is the fundamental operation for incremental updates:
    cache_{t+1} = concat([cache_t, new_kv])

    Args:
        existing: Existing KV cache with shape [B, H, T, D]
        new_k: New key tensor with shape [B, H, 1, D] or [B, H, N, D]
        new_v: New value tensor with shape [B, H, 1, D] or [B, H, N, D]

    Returns:
        Updated KV cache with shape [B, H, T+N, D]

    Example:
        >>> cache = KVCache(keys=torch.randn(1, 4, 10, 32), values=torch.randn(1, 4, 10, 32))
        >>> new_k = torch.randn(1, 4, 1, 32)
        >>> new_v = torch.randn(1, 4, 1, 32)
        >>> updated = concat_kv_cache(cache, new_k, new_v)
        >>> updated.keys.shape
        torch.Size([1, 4, 11, 32])
    """
    # Concatenate along sequence dimension (dim=2)
    updated_k = torch.cat([existing.keys, new_k], dim=2)
    updated_v = torch.cat([existing.values, new_v], dim=2)
    return KVCache(updated_k, updated_v)


def validate_state(state: IncrementalState, raise_on_error: bool = True) -> bool:
    """
    Validate IncrementalState consistency.

    Args:
        state: State to validate
        raise_on_error: If True, raise exception on invalid state

    Returns:
        True if valid, False otherwise (if raise_on_error=False)

    Raises:
        ValueError: If state is invalid and raise_on_error=True
    """
    errors = []

    # Check KV cache count
    if len(state.kv_caches) != state.num_layers:
        errors.append(f"KV cache count ({len(state.kv_caches)}) != num_layers ({state.num_layers})")

    # Check sequence length is positive
    if state.seq_len < 0:
        errors.append(f"seq_len ({state.seq_len}) must be non-negative")

    # Check block representations don't exceed num_blocks
    if len(state.block_representations) > state.num_blocks:
        errors.append(
            f"Block rep count ({len(state.block_representations)}) > num_blocks ({state.num_blocks})"
        )

    # Check all KV caches have same batch size
    if state.kv_caches:
        batch_sizes = [cache.keys.shape[0] for cache in state.kv_caches]
        if len(set(batch_sizes)) > 1:
            errors.append(f"Inconsistent batch sizes in KV caches: {batch_sizes}")

    # Check device consistency
    if state.kv_caches:
        devices = [cache.keys.device for cache in state.kv_caches]
        devices.append(state.partial_block.device)
        if len(set(str(d) for d in devices)) > 1:
            errors.append(f"Inconsistent devices: {devices}")

    if errors:
        if raise_on_error:
            raise ValueError("; ".join(errors))
        else:
            return False

    return True


def create_empty_state(
    batch_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    num_blocks: int,
    device: str = "cpu",
) -> IncrementalState:
    """
    Create empty initial state for incremental generation.

    Args:
        batch_size: Batch size
        num_layers: Number of layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        hidden_dim: Hidden dimension
        num_blocks: Number of AttnRes blocks
        device: Device for tensors

    Returns:
        Empty IncrementalState
    """
    # Empty KV caches (seq_len=0)
    kv_caches = [
        KVCache(
            keys=torch.empty(batch_size, num_heads, 0, head_dim, device=device),
            values=torch.empty(batch_size, num_heads, 0, head_dim, device=device),
        )
        for _ in range(num_layers)
    ]

    # Empty block representations
    block_representations = []

    # Empty partial block
    partial_block = torch.empty(batch_size, 0, hidden_dim, device=device)

    return IncrementalState(
        kv_caches=kv_caches,
        block_representations=block_representations,
        partial_block=partial_block,
        seq_len=0,
        num_layers=num_layers,
        num_blocks=num_blocks,
    )
