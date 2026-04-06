"""
Engram Integration with AdaptiveTransformer

This module provides integration utilities for adding Engram to the
existing AdaptiveTransformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.models.configs import ModelConfig
from src.models.adaptive_transformer import AdaptiveTransformer, AdaptiveLayer
from src.attnres.block_attnres import RMSNorm
from .engram_module import Engram
from .config import EngramConfig


class AdaptiveLayerWithEngram(AdaptiveLayer):
    """
    AdaptiveLayer extended with Engram support.
    
    Engram is applied before attention, using the input token IDs
    to retrieve n-gram embeddings that enhance the hidden states.
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Initialize Engram if enabled for this layer
        self.engram = None
        if config.use_engram and config.engram_config:
            if layer_idx in config.engram_config.layer_ids:
                self.engram = Engram(
                    layer_id=layer_idx,
                    config=config.engram_config,
                    hidden_size=config.hidden_dim,
                    hc_mult=1,  # Standard transformer doesn't use hc_mult
                )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional Engram.
        
        Args:
            hidden_states: [B, T, D]
            input_ids: [B, T] - Required if Engram is enabled
            **kwargs: Additional arguments passed to parent
        
        Returns:
            output: Updated hidden states
            partial_block: Updated partial block (for AttnRes)
        """
        # Apply Engram if enabled
        if self.engram is not None and input_ids is not None:
            # Engram expects [B, L, hc_mult, D] but we have [B, L, D]
            # So we add a dummy dimension
            hidden_expanded = hidden_states.unsqueeze(2)  # [B, L, 1, D]
            engram_output = self.engram(hidden_expanded, input_ids)
            # Squeeze back and add as residual
            hidden_states = hidden_states + engram_output.squeeze(2)
        
        # Call parent forward (without input_ids to avoid conflict)
        kwargs.pop('input_ids', None)
        return super().forward(hidden_states, **kwargs)


class AdaptiveTransformerWithEngram(AdaptiveTransformer):
    """
    AdaptiveTransformer with Engram integration.
    
    Usage:
        config = AttnResMediumConfig()
        config.use_engram = True
        config.engram_config = EngramMediumConfig
        
        model = AdaptiveTransformerWithEngram(config)
        logits = model(input_ids)
    """
    
    def __init__(self, config: ModelConfig):
        # Store config
        self.config = config
        
        # Initialize as nn.Module but don't call AdaptiveTransformer.__init__
        # because we need to override the layers
        nn.Module.__init__(self)
        
        # Embeddings
        from src.models.adaptive_transformer import AdaptiveTransformer
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Transformer layers with Engram support
        self.layers = nn.ModuleList([
            AdaptiveLayerWithEngram(config, i) 
            for i in range(config.num_layers)
        ])
        
        # AttnRes modules
        from src.attnres.block_attnres import TwoPhaseBlockAttnRes
        block_size = max(config.num_layers // max(config.num_blocks, 1), 1)
        self.attnres_modules = nn.ModuleList([
            TwoPhaseBlockAttnRes(config.hidden_dim, block_size)
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
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with Engram support.
        
        Args:
            input_ids: [B, T]
            **kwargs: Additional arguments
        
        Returns:
            logits: [B, T, V]
        """
        B, T = input_ids.shape
        
        # Token embedding
        hidden = self.token_embedding(input_ids)
        
        # Block management
        layers_per_block = max(self.config.num_layers // max(self.config.num_blocks, 1), 1)
        use_attnres = kwargs.get('use_attnres', True)
        block_representations = [hidden] if use_attnres else []
        partial_block = torch.zeros_like(hidden) if use_attnres else hidden
        
        # Process layers (match AdaptiveTransformer: keep `hidden` as residual stream)
        for layer_idx, (layer, attnres) in enumerate(zip(self.layers, self.attnres_modules)):
            # Check if we need to finalize a block
            if use_attnres and layer_idx > 0 and layer_idx % layers_per_block == 0:
                block_representations.append(partial_block)
                partial_block = torch.zeros_like(hidden)
            
            # Prepare layer arguments
            layer_kwargs = {
                'block_representations': block_representations if use_attnres else None,
                'partial_block': partial_block,
                'attnres_module': attnres if use_attnres else None,
                'use_attnres': use_attnres,
            }
            
            # Pass input_ids for Engram
            if hasattr(layer, 'engram') and layer.engram is not None:
                layer_kwargs['input_ids'] = input_ids
            
            # Forward through layer
            hidden, partial_block = layer(hidden, **layer_kwargs)
        
        # Final AttnRes aggregation (same as AdaptiveTransformer.forward)
        if use_attnres:
            all_blocks = block_representations + [partial_block]
            V = torch.stack(all_blocks, dim=0)
            attnres = self.attnres_modules[-1]
            K = attnres.norm_mlp(V)
            w = attnres.pseudo_query_mlp
            logits_attn = torch.einsum("d, n b t d -> n b t", w, K)
            alpha = F.softmax(logits_attn, dim=0)
            hidden = torch.einsum("n b t, n b t d -> b t d", alpha, V)
        else:
            hidden = partial_block
        
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        
        return logits


def add_engram_to_config(
    config: ModelConfig,
    engram_config: Optional[EngramConfig] = None
) -> ModelConfig:
    """
    Add Engram configuration to an existing ModelConfig.
    
    Args:
        config: Existing ModelConfig
        engram_config: EngramConfig (if None, uses default)
    
    Returns:
        Modified config with Engram enabled
    """
    config.use_engram = True
    
    if engram_config is None:
        # Auto-select based on model size
        if config.hidden_dim <= 1408:
            from src.engram.config import EngramSmallConfig
            config.engram_config = EngramSmallConfig
        elif config.hidden_dim <= 2496:
            from src.engram.config import EngramMediumConfig
            config.engram_config = EngramMediumConfig
        else:
            from src.engram.config import EngramLargeConfig
            config.engram_config = EngramLargeConfig
    else:
        config.engram_config = engram_config
    
    # Update tokenizer path to match model
    config.engram_config.tokenizer_name_or_path = "gpt2"  # Default, should be updated
    
    return config
