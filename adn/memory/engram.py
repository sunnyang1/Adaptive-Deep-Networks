"""
Engram Module - Main implementation

Integrates n-gram hash embeddings with gating mechanism and short convolution
to enhance Transformer layers with explicit memory.
"""

import math
from typing import Optional, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np

from adn.memory.ngram_hash import NgramHashMapping, NgramHashConfig
from adn.memory.embeddings import MultiHeadEmbedding, ShortConv


@dataclass
class EngramConfig:
    """
    Configuration for Engram module.

    Engram enhances Transformer with explicit n-gram memory through
    hash-based embeddings at specific layers.

    Attributes:
        enabled: Whether to enable Engram
        engram_vocab_size: Vocabulary sizes for each n-gram level
        max_ngram_size: Maximum n-gram size (e.g., 3 for up to trigrams)
        n_embed_per_ngram: Embedding dimension per n-gram type
        n_head_per_ngram: Number of heads per n-gram type
        layer_ids: Which layers to apply Engram
        tokenizer_name_or_path: Tokenizer to use
        pad_id: Padding token ID
        seed: Random seed for hashing
        kernel_size: ShortConv kernel size
    """

    enabled: bool = False
    engram_vocab_size: List[int] = field(default_factory=lambda: [100000, 100000])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    tokenizer_name_or_path: str = "gpt2"
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4

    def __post_init__(self):
        """Validate configuration."""
        assert self.max_ngram_size >= 2, "max_ngram_size must be >= 2"
        assert (
            len(self.engram_vocab_size) == self.max_ngram_size - 1
        ), f"engram_vocab_size must have {self.max_ngram_size - 1} elements"
        assert (
            self.n_embed_per_ngram % self.n_head_per_ngram == 0
        ), "n_embed_per_ngram must be divisible by n_head_per_ngram"


# Predefined configurations
EngramSmallConfig = EngramConfig(
    enabled=True,
    engram_vocab_size=[50000, 50000],
    max_ngram_size=3,
    n_embed_per_ngram=256,
    n_head_per_ngram=4,
    layer_ids=[1, 7],
)

EngramMediumConfig = EngramConfig(
    enabled=True,
    engram_vocab_size=[100000, 100000],
    max_ngram_size=3,
    n_embed_per_ngram=512,
    n_head_per_ngram=8,
    layer_ids=[1, 15],
)

EngramLargeConfig = EngramConfig(
    enabled=True,
    engram_vocab_size=[200000, 200000, 150000],
    max_ngram_size=4,
    n_embed_per_ngram=768,
    n_head_per_ngram=12,
    layer_ids=[1, 15, 30],
)


class Engram(nn.Module):
    """
    Engram module for explicit n-gram memory in Transformers.

    Architecture:
    1. N-gram hash mapping from input tokens
    2. Multi-head embedding lookup
    3. Key-query gating mechanism
    4. Short convolution for local dependencies

    Args:
        layer_id: Which layer this Engram belongs to
        config: EngramConfig
        hidden_size: Backbone hidden size
        hc_mult: Hyper-connection multiplier
    """

    def __init__(
        self,
        layer_id: int,
        config: EngramConfig,
        hidden_size: int,
        hc_mult: int = 1,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult

        # Initialize hash mapping
        hash_config = NgramHashConfig(
            engram_vocab_size=config.engram_vocab_size,
            max_ngram_size=config.max_ngram_size,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.layer_ids,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            pad_id=config.pad_id,
            seed=config.seed,
        )
        self.hash_mapping = NgramHashMapping(hash_config)

        # Multi-head embedding
        # Each (ngram_size, head) pair gets its own vocab size
        list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[layer_id] for x in y]
        embed_dim_per_head = config.n_embed_per_ngram // config.n_head_per_ngram
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=list_of_N,
            D=embed_dim_per_head,
        )

        # Short convolution
        self.short_conv = ShortConv(
            hidden_size=hidden_size,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
            hc_mult=hc_mult,
        )

        # Projection layers
        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)

        # Key projections for each hyper-connection group
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, hidden_size) for _ in range(hc_mult)]
        )

        # Layer norms for keys and queries
        self.norm_keys = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.norm_queries = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])

        self._init_weights()

    def _init_weights(self):
        """Initialize projection layers."""
        for proj in [self.value_proj] + list(self.key_projs):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of Engram module.

        Args:
            hidden_states: Backbone hidden states, shape [B, L, hc_mult, D]
            input_ids: Input token IDs, shape [B, L]

        Returns:
            Enhanced hidden states, shape [B, L, hc_mult, D]
        """
        B, L, G, D = hidden_states.shape
        assert G == self.hc_mult

        # Compute n-gram hashes
        # hash_ids shape: [B, L, num_heads]
        hash_ids = torch.from_numpy(
            self.hash_mapping.hash(input_ids.cpu().numpy())[self.layer_id]
        ).to(hidden_states.device)

        # Look up embeddings
        # embeddings shape: [B, L, num_heads, embed_dim_per_head]
        embeddings = self.multi_head_embedding(hash_ids)

        # Flatten last two dimensions
        # shape: [B, L, (max_ngram-1) * n_embed_per_ngram]
        embeddings = embeddings.flatten(start_dim=-2)

        # Compute gates for each hyper-connection group
        gates = []
        for hc_idx in range(self.hc_mult):
            # Project to get keys
            key = self.key_projs[hc_idx](embeddings)

            # Normalize key and query
            normed_key = self.norm_keys[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm_queries[hc_idx](query)

            # Compute gate as sigmoid of scaled dot product
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(D)

            # Signed square root for stable gradients
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()

            # Apply sigmoid and reshape
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)

        # Stack gates: [B, L, hc_mult, 1]
        gates = torch.stack(gates, dim=2)

        # Project embeddings to get values
        # value shape: [B, L, D] -> [B, L, 1, D] -> [B, L, hc_mult, D]
        value = self.value_proj(embeddings).unsqueeze(2)
        value = value.expand(-1, -1, self.hc_mult, -1)

        # Apply gating
        gated_value = gates * value

        # Apply short convolution
        output = gated_value + self.short_conv(gated_value)

        return output

    def get_compression_ratio(self) -> float:
        """Get tokenizer compression ratio."""
        return self.hash_mapping.compressed_tokenizer.get_compression_ratio()
