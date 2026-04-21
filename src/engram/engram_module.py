"""
Engram Module - Main implementation

Integrates n-gram hash embeddings with gating mechanism and short convolution
to enhance Transformer layers with explicit memory.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import numpy as np

from .config import EngramConfig
from .ngram_hash import NgramHashMapping, NgramHashConfig
from .embeddings import MultiHeadEmbedding, ShortConv


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
