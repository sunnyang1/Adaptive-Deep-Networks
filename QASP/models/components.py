"""Core components shared by QASP model modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(slots=True)
class QASPTransformerConfig:
    """Configuration for the lightweight QASP transformer stack."""

    vocab_size: int = 32000
    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_ratio: float = 4.0
    max_position_embeddings: int = 2048
    attnres_blocks: int = 4
    use_attnres: bool = True
    use_engram: bool = True


class RMSNorm(nn.Module):
    """Simple RMSNorm used for stable transformer blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class CausalSelfAttention(nn.Module):
    """Minimal multi-head causal self-attention."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def _shape(self, tensor: Tensor) -> Tensor:
        batch, seq, _ = tensor.shape
        tensor = tensor.view(batch, seq, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        q = self._shape(self.q_proj(hidden_states))
        k = self._shape(self.k_proj(hidden_states))
        v = self._shape(self.v_proj(hidden_states))

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(hidden_states.size(0), hidden_states.size(1), self.hidden_size)
        return self.o_proj(attn_output)


class FeedForward(nn.Module):
    """SwiGLU-style MLP block."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        inner_dim = int(config.hidden_size * config.mlp_ratio)
        self.gate_proj = nn.Linear(config.hidden_size, inner_dim)
        self.up_proj = nn.Linear(config.hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(hidden_states))
        values = self.up_proj(hidden_states)
        return self.down_proj(gated * values)


def compute_block_representations(hidden_states: Tensor, num_blocks: int) -> tuple[Tensor, Tensor]:
    """Pool hidden states into block-level vectors and quality scores."""

    chunks = torch.chunk(hidden_states, chunks=max(1, num_blocks), dim=1)
    block_vectors = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)
    quality_scores = block_vectors.norm(dim=-1)
    return block_vectors, quality_scores

