"""
Block Attention Residuals (AttnRes) Implementation

Based on: Chen et al. "Attention Residuals" Technical Report, 2026
Reference: Page 5, Figure 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


def block_attn_res(
    blocks: List[torch.Tensor],
    partial_block: torch.Tensor,
    pseudo_query: torch.Tensor,
    norm: RMSNorm,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Inter-block attention: attend over block representations + partial sum.
    
    Based on Attention Residuals Technical Report, Algorithm on Page 5.
    
    Args:
        blocks: List of N completed block representations, each [B, T, D]
        partial_block: Current partial block sum [B, T, D] (b_n^i)
        pseudo_query: Learned pseudo-query vector [D] (w_l)
        norm: RMSNorm layer for normalization
        eps: Small constant for numerical stability
    
    Returns:
        AttnRes-augmented hidden state [B, T, D]
    
    Memory: O(Nd), Communication: O(Nd), Computation: O(N^2d)
    """
    # Stack block representations: [N+1, B, T, D]
    V = torch.stack(blocks + [partial_block], dim=0)
    
    # Normalize keys
    K = norm(V)  # [N+1, B, T, D]
    
    # Compute compatibility scores: logits [N+1, B, T]
    # einsum: 'd, n b t d -> n b t'
    logits = torch.einsum('d, n b t d -> n b t', pseudo_query, K)
    logits = logits / (pseudo_query.size(-1) ** 0.5)  # Scale by sqrt(dim)
    
    # Softmax normalization over blocks (dim=0)
    attn_weights = F.softmax(logits, dim=0)  # [N+1, B, T]
    
    # Weighted aggregation: h [B, T, D]
    # einsum: 'n b t, n b t d -> b t d'
    h = torch.einsum('n b t, n b t d -> b t d', attn_weights, V)
    
    return h


class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals layer.
    
    Implements the two-phase computation strategy:
    - Phase 1: Inter-block attention (batched)
    - Phase 2: Intra-block processing (sequential)
    
    Reference: Attention Residuals Technical Report, Section 4.2
    """
    
    def __init__(
        self,
        dim: int,
        num_blocks: int = 8,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.eps = eps
        
        # Pseudo-query vectors (one for attention, one for MLP)
        # Zero initialization for stable training
        self.pseudo_query_attn = nn.Parameter(torch.zeros(dim))
        self.pseudo_query_mlp = nn.Parameter(torch.zeros(dim))
        
        # RMSNorm for key normalization
        self.norm_attn = RMSNorm(dim, eps)
        self.norm_mlp = RMSNorm(dim, eps)
    
    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
        use_attn: bool = True,
        use_mlp: bool = True
    ) -> torch.Tensor:
        """
        Compute AttnRes-augmented hidden state.
        
        Args:
            blocks: List of completed block representations
            partial_block: Current partial block sum
            use_attn: Whether to apply AttnRes before attention layer
            use_mlp: Whether to apply AttnRes before MLP layer
        
        Returns:
            Tuple of (attn_res_input, mlp_res_input)
        """
        if use_attn and len(blocks) > 0:
            h_attn = block_attn_res(
                blocks, partial_block, self.pseudo_query_attn, self.norm_attn
            )
        else:
            h_attn = partial_block
        
        if use_mlp and len(blocks) > 0:
            h_mlp = block_attn_res(
                blocks, partial_block, self.pseudo_query_mlp, self.norm_mlp
            )
        else:
            h_mlp = partial_block
        
        return h_attn, h_mlp
    
    def reset_parameters(self):
        """Reset pseudo-queries to zero (standard residual behavior)."""
        nn.init.zeros_(self.pseudo_query_attn)
        nn.init.zeros_(self.pseudo_query_mlp)


class TwoPhaseBlockAttnRes(nn.Module):
    """
    Two-phase computation strategy for efficient Block AttnRes.
    
    Phase 1: Parallel inter-block attention (batched across layers in block)
    Phase 2: Sequential intra-block attention + online softmax merge
    
    Reference: Attention Residuals Technical Report, Algorithm 1
    """
    
    def __init__(
        self,
        dim: int,
        block_size: int,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.eps = eps
        
        self.norm = RMSNorm(dim, eps)
    
    def phase1_inter_block(
        self,
        pseudo_queries: torch.Tensor,  # [S, D] - queries for all layers in block
        block_representations: List[torch.Tensor]  # [b0, ..., b_{n-1}]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase 1: Compute inter-block attention for all layers simultaneously.
        
        Args:
            pseudo_queries: [S, D] where S = block_size
            block_representations: List of N tensors, each [B, T, D]
        
        Returns:
            outputs: [S, B, T, D]
            max_vals: [S, B, T] for online softmax
            lse: [S, B, T] log-sum-exp for online softmax
        """
        if len(block_representations) == 0:
            # First block: return zeros
            S = pseudo_queries.size(0)
            B, T, D = 1, 1, self.dim  # Will be broadcasted
            return (
                torch.zeros(S, B, T, D, device=pseudo_queries.device),
                torch.zeros(S, B, T, device=pseudo_queries.device),
                torch.ones(S, B, T, device=pseudo_queries.device)
            )
        
        # Stack block representations: [N, B, T, D]
        V = torch.stack(block_representations, dim=0)
        K = self.norm(V)  # [N, B, T, D]
        
        # Batched attention: [S, D] @ [N, B, T, D] -> [S, N, B, T]
        logits = torch.einsum('s d, n b t d -> s n b t', pseudo_queries, K)
        logits = logits / (self.dim ** 0.5)
        
        # Compute softmax stats
        max_vals = logits.max(dim=1, keepdim=True).values  # [S, 1, B, T]
        exp_logits = torch.exp(logits - max_vals)  # [S, N, B, T]
        lse = torch.log(exp_logits.sum(dim=1)) + max_vals.squeeze(1)  # [S, B, T]
        
        # Weighted sum: [S, N, B, T] @ [N, B, T, D] -> [S, B, T, D]
        attn_weights = F.softmax(logits, dim=1)  # [S, N, B, T]
        outputs = torch.einsum('s n b t, n b t d -> s b t d', attn_weights, V)
        
        return outputs, max_vals.squeeze(1), lse
    
    def phase2_intra_block(
        self,
        inter_output: torch.Tensor,  # [B, T, D]
        inter_max: torch.Tensor,     # [B, T]
        inter_lse: torch.Tensor,     # [B, T]
        pseudo_query: torch.Tensor,  # [D]
        partial_sum: torch.Tensor    # [B, T, D] (b_n^i)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase 2: Merge intra-block attention with online softmax.
        
        Args:
            inter_output: Output from inter-block attention
            inter_max: Max value from inter-block
            inter_lse: Log-sum-exp from inter-block
            pseudo_query: [D]
            partial_sum: [B, T, D]
        
        Returns:
            merged_output: [B, T, D]
            merged_max: [B, T]
            merged_lse: [B, T]
        """
        # Intra-block attention (single key-value = partial_sum)
        K = self.norm(partial_sum)  # [B, T, D]
        logits = torch.einsum('d, b t d -> b t', pseudo_query, K)
        logits = logits / (self.dim ** 0.5)
        intra_max = logits  # [B, T]
        
        # Online softmax merge
        merged_max = torch.maximum(inter_max, intra_max)  # [B, T]
        
        # Compute merged output
        exp_inter = torch.exp(inter_max - merged_max)  # [B, T]
        exp_intra = torch.exp(intra_max - merged_max)  # [B, T]
        
        merged_lse = torch.log(
            exp_inter * torch.exp(inter_lse - inter_max) + exp_intra
        ) + merged_max  # [B, T]
        
        # Weighted combination
        weight_inter = torch.exp(inter_max - merged_max)  # [B, T]
        weight_intra = torch.exp(intra_max - merged_max)  # [B, T]
        
        norm = weight_inter + weight_intra  # [B, T]
        merged_output = (
            weight_inter.unsqueeze(-1) * inter_output + 
            weight_intra.unsqueeze(-1) * partial_sum
        ) / norm.unsqueeze(-1)  # [B, T, D]
        
        return merged_output, merged_max, merged_lse


# Convenience function for simple usage
def create_block_attnres(
    dim: int,
    num_blocks: int = 8,
    use_two_phase: bool = True
) -> nn.Module:
    """Factory function to create BlockAttnRes module."""
    if use_two_phase:
        block_size = 32 // num_blocks  # Assuming 32 layers
        return TwoPhaseBlockAttnRes(dim, block_size)
    return BlockAttnRes(dim, num_blocks)
