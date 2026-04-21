"""Generic KV cache compression interface"""
from typing import Protocol, Tuple
import torch

from adn.quantization.rabitq_api import RaBitQ, RaBitQConfig


class KVCompressor(Protocol):
    """KV cache compressor protocol"""
    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> dict: ...
    def decompress(self, compressed: dict) -> Tuple[torch.Tensor, torch.Tensor]: ...


class RaBitQCompressor:
    """RaBitQ compressor implementation"""
    def __init__(self, config: RaBitQConfig = None, head_dim: int = 64):
        self.rabitq = RaBitQ(config=config, head_dim=head_dim)

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> dict:
        return self.rabitq.compress(keys, values)

    def decompress(self, compressed: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rabitq.decompress(compressed)
