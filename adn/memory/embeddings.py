"""
Multi-Head Embedding and ShortConv for Engram
"""

import math
from typing import List
import torch
import torch.nn as nn


class MultiHeadEmbedding(nn.Module):
    """
    Multi-head embedding layer that supports different vocabulary sizes per head.

    Each head can have a different vocabulary size, allowing flexible embedding
    dimensions for different n-gram sizes.

    Args:
        list_of_N: List of vocabulary sizes for each head
        D: Embedding dimension per head
    """

    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        # Calculate cumulative offsets for each head
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        # Total embedding table size
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with normal distribution."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for multi-head input.

        Args:
            input_ids: Token IDs, shape [..., num_heads]
                      Each position corresponds to one head's token ID

        Returns:
            Embeddings, shape [..., num_heads, D]
        """
        # Add offsets to get absolute indices
        shifted_input_ids = input_ids + self.offsets

        # Look up embeddings
        output = self.embedding(shifted_input_ids)

        return output


class ShortConv(nn.Module):
    """
    Short convolution for processing local dependencies.

    Applies depthwise (grouped) 1D convolution with RMS normalization.
    Designed to work with hyper-connection format (B, L, G, C).

    Args:
        hidden_size: Hidden dimension per group
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        norm_eps: RMSNorm epsilon
        hc_mult: Hyper-connection multiplier (number of groups)
        activation: Whether to apply SiLU activation
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult

        # Depthwise convolution (groups = channels)
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,  # Depthwise
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        # RMSNorm for each group
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)])

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply short convolution.

        Args:
            x: Input tensor, shape [B, L, hc_mult, C]

        Returns:
            Output tensor, shape [B, L, hc_mult, C]
        """
        B, T, G, C = x.shape

        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        # Apply RMSNorm to each group
        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))

        # Concatenate along feature dimension: [B, L, G*C]
        x_norm = torch.cat(normed_chunks, dim=-1)

        # Transpose for Conv1d: [B, G*C, L]
        x_bct = x_norm.transpose(1, 2)

        # Apply convolution
        y_bct = self.conv(x_bct)

        # Truncate to original length (remove padding)
        y_bct = y_bct[..., :T]

        # Apply activation
        if self.activation:
            y_bct = self.act_fn(y_bct)

        # Transpose back and reshape: [B, L, G, C]
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

        return y


# Alias for public API
EmbeddingManager = MultiHeadEmbedding
