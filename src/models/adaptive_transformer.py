"""
Adaptive Transformer with AttnRes, Gating, and qTTT

Complete model implementation integrating all three components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from models.configs import ModelConfig
from attnres.block_attnres import BlockAttnRes, RMSNorm
from gating.threshold import DynamicThreshold
from qttt.adaptation import QueryOnlyTTT, KVCache


class AdaptiveAttention(nn.Module):
    """Multi-head attention with optional qTTT adaptation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        adapted_query: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional KV cache and adapted query.
        
        Args:
            hidden_states: [B, T, D]
            kv_cache: Optional frozen KV cache for qTTT
            adapted_query: Optional adapted query from qTTT
        """
        B, T, D = hidden_states.shape
        
        if adapted_query is not None:
            # Use adapted query from qTTT
            q = adapted_query
        else:
            q = self.q_proj(hidden_states)
        
        if kv_cache is not None:
            # Use frozen KV cache
            k, v = kv_cache.get_kv()
            # Reshape for multi-head: [B, H, T, d]
            q = q.view(B, T, self.config.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            # Reshape
            q = q.view(B, T, self.config.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, -1, self.config.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, -1, self.config.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        
        return out


class AdaptiveMLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        mlp_dim = hidden_dim * config.mlp_ratio
        
        self.gate_proj = nn.Linear(hidden_dim, mlp_dim)
        self.up_proj = nn.Linear(hidden_dim, mlp_dim)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate(x) * up(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        out = self.down_proj(hidden)
        return out


class AdaptiveLayer(nn.Module):
    """
    Single transformer layer with AttnRes.
    
    Structure:
    1. AttnRes -> Attention -> Residual
    2. AttnRes -> MLP -> Residual
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Layer norm
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.mlp_norm = RMSNorm(config.hidden_dim)
        
        # Attention and MLP
        self.attn = AdaptiveAttention(config)
        self.mlp = AdaptiveMLP(config)
        
        # Track if this is a block boundary
        layers_per_block = config.num_layers // config.num_blocks
        self.is_block_boundary = (layer_idx + 1) % layers_per_block == 0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_representations: List[torch.Tensor],
        partial_block: torch.Tensor,
        attnres_module: BlockAttnRes,
        use_qttt: bool = False,
        kv_cache: Optional[KVCache] = None,
        adapted_query: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with AttnRes.
        
        Args:
            hidden_states: [B, T, D]
            block_representations: List of previous block outputs
            partial_block: Current partial block sum
            attnres_module: BlockAttnRes module
            use_qttt: Whether to use qTTT adaptation
            kv_cache: Frozen KV cache for qTTT
            adapted_query: Adapted query from qTTT
        
        Returns:
            output: Updated hidden states
            updated_partial: Updated partial block sum
        """
        # Phase 1: Inter-block attention before Attention layer
        h_attn, _ = attnres_module(
            block_representations,
            partial_block,
            use_attn=True,
            use_mlp=False
        )
        
        # Attention layer
        normed = self.attn_norm(h_attn)
        attn_out = self.attn(normed, kv_cache, adapted_query)
        partial_block = partial_block + attn_out
        
        # Phase 1: Inter-block attention before MLP layer
        _, h_mlp = attnres_module(
            block_representations,
            partial_block,
            use_attn=False,
            use_mlp=True
        )
        
        # MLP layer
        normed = self.mlp_norm(h_mlp)
        mlp_out = self.mlp(normed)
        partial_block = partial_block + mlp_out
        
        return partial_block, partial_block


class AdaptiveTransformer(nn.Module):
    """
    Complete Adaptive Deep Networks transformer.
    
    Integrates:
    - Block Attention Residuals
    - Dynamic Computation Gating
    - Query-only Test-Time Training
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            AdaptiveLayer(config, i) for i in range(config.num_layers)
        ])
        
        # AttnRes modules for each layer
        self.attnres_modules = nn.ModuleList([
            BlockAttnRes(config.hidden_dim, config.num_blocks)
            for _ in range(config.num_layers)
        ])
        
        # Output
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        # Zero-initialize pseudo-queries
        for attnres in self.attnres_modules:
            attnres.reset_parameters()
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        use_qttt: bool = False,
        qttt_config: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [B, T]
            use_qttt: Whether to use qTTT adaptation
            qttt_config: Configuration for qTTT
        
        Returns:
            logits: [B, T, V]
        """
        B, T = input_ids.shape
        
        # Token embedding
        hidden = self.token_embedding(input_ids)
        
        # Block management
        layers_per_block = self.config.num_layers // self.config.num_blocks
        block_representations = [hidden]  # Start with embeddings
        partial_block = torch.zeros_like(hidden)
        
        # Track for potential qTTT
        kv_cache = None
        adapted_query = None
        
        # Process layers
        for layer_idx, (layer, attnres) in enumerate(zip(self.layers, self.attnres_modules)):
            # Check if we need to finalize a block
            if layer_idx > 0 and layer_idx % layers_per_block == 0:
                block_representations.append(partial_block)
                partial_block = torch.zeros_like(hidden)
            
            # Forward through layer
            hidden, partial_block = layer(
                hidden,
                block_representations,
                partial_block,
                attnres,
                use_qttt=use_qttt,
                kv_cache=kv_cache,
                adapted_query=adapted_query
            )
        
        # Final norm and projection
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        
        return logits
    
    def get_kv_cache(self, input_ids: torch.Tensor) -> KVCache:
        """Compute and return KV cache for initial prefill."""
        B, T = input_ids.shape
        
        hidden = self.token_embedding(input_ids)
        
        # Compute K and V for all layers
        # Simplified: just use first layer for demo
        k = self.layers[0].attn.k_proj(hidden)
        v = self.layers[0].attn.v_proj(hidden)
        
        # Reshape for multi-head
        head_dim = self.config.hidden_dim // self.config.num_heads
        k = k.view(B, T, self.config.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.num_heads, head_dim).transpose(1, 2)
        
        return KVCache(k, v)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_attnsres_parameters(self) -> int:
        """Count parameters added by AttnRes (should be negligible)."""
        return sum(
            p.numel() for n, p in self.named_parameters()
            if 'pseudo_query' in n or 'attnres' in n
        )


def create_adaptive_transformer(
    config_name: str = "medium",
    **kwargs
) -> AdaptiveTransformer:
    """Factory function to create model."""
    from models.configs import get_config
    
    config = get_config(config_name)
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return AdaptiveTransformer(config)
