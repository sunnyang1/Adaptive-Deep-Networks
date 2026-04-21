"""
Block Attention Residuals (AttnRes) Implementation

This module implements Block Attention Residuals, a method for maintaining
distributed representations across deep transformer layers to prevent
representation burial.

Key Features:
    - Block-based attention over layer representations
    - Two-phase computation (inter-block and intra-block)
    - Memory efficient: O(Nd) instead of O(Ld)
    - Learnable pseudo-queries for block aggregation

Based on: Chen et al. "Attention Residuals" Technical Report, 2026
Paper Reference: Page 5, Figure 2, Algorithm 1

Example:
    >>> import torch
    >>> from src.attnres.block_attnres import BlockAttnRes, RMSNorm
    >>>
    >>> # Initialize layer
    >>> dim = 512
    >>> num_blocks = 8
    >>> layer = BlockAttnRes(dim, num_blocks)
    >>>
    >>> # Create block representations
    >>> batch_size, seq_len = 2, 10
    >>> blocks = [torch.randn(batch_size, seq_len, dim) for _ in range(4)]
    >>> hidden = torch.randn(batch_size, seq_len, dim)
    >>>
    >>> # Forward pass
    >>> h_attn, h_mlp = layer(blocks, hidden, use_attn=True, use_mlp=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes inputs by their root mean square instead of mean and variance,
    providing better stability for deep networks without the mean-centering overhead
    of LayerNorm.

    Formula:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Args:
        dim: Dimension to normalize over
        eps: Small constant for numerical stability (default: 1e-6)

    Attributes:
        weight: Learnable scale parameter of shape [dim]

    Example:
        >>> import torch
        >>> norm = RMSNorm(dim=512)
        >>> x = torch.randn(2, 10, 512)
        >>> normalized = norm(x)
        >>> normalized.shape
        torch.Size([2, 10, 512])

    References:
        Zhang and Sennrich (2019): "Root Mean Square Layer Normalization"
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Computes in float32 for numerical stability and casts back to the
        input dtype. This is important for mixed-precision training.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of shape [..., dim]
        """
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def block_attn_res(
    blocks: List[torch.Tensor],
    partial_block: torch.Tensor,
    pseudo_query: torch.Tensor,
    norm: RMSNorm,
    eps: float = 1e-6,
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
    logits = torch.einsum("d, n b t d -> n b t", pseudo_query, K)
    logits = logits / (pseudo_query.size(-1) ** 0.5)  # Scale by sqrt(dim)

    # Softmax normalization over blocks (dim=0)
    attn_weights = F.softmax(logits, dim=0)  # [N+1, B, T]

    # Weighted aggregation: h [B, T, D]
    # einsum: 'n b t, n b t d -> b t d'
    h = torch.einsum("n b t, n b t d -> b t d", attn_weights, V)

    return h


class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals layer.

    Implements the two-phase computation strategy:
    - Phase 1: Inter-block attention (batched)
    - Phase 2: Intra-block processing (sequential)

    Reference: Attention Residuals Technical Report, Section 4.2
    """

    def __init__(self, dim: int, num_blocks: int = 8, eps: float = 1e-6):
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
        use_mlp: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            h_attn = block_attn_res(blocks, partial_block, self.pseudo_query_attn, self.norm_attn)
        else:
            h_attn = partial_block

        if use_mlp and len(blocks) > 0:
            h_mlp = block_attn_res(blocks, partial_block, self.pseudo_query_mlp, self.norm_mlp)
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

    def __init__(self, dim: int, block_size: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.eps = eps

        # Pseudo-query vectors (one for attention, one for MLP)
        # Zero initialization for stable training (as per paper §3.2.2)
        self.pseudo_query_attn = nn.Parameter(torch.zeros(dim))
        self.pseudo_query_mlp = nn.Parameter(torch.zeros(dim))

        # RMSNorm for key normalization (critical for performance)
        self.norm_attn = RMSNorm(dim, eps)
        self.norm_mlp = RMSNorm(dim, eps)
        # Alias for two-phase helpers (keys use same norm as BlockAttnRes keys)
        self.norm = self.norm_attn

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
        use_attn: bool = True,
        use_mlp: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute AttnRes-augmented hidden state.

        This is a compatibility wrapper that uses the standard block_attn_res
        function. The full two-phase computation is managed by the model's
        forward loop (see AdaptiveTransformer).

        Args:
            blocks: List of completed block representations
            partial_block: Current partial block sum
            use_attn: Whether to apply AttnRes before attention layer
            use_mlp: Whether to apply AttnRes before MLP layer

        Returns:
            Tuple of (attn_res_input, mlp_res_input)
        """
        if use_attn and len(blocks) > 0:
            h_attn = block_attn_res(blocks, partial_block, self.pseudo_query_attn, self.norm_attn)
        else:
            h_attn = partial_block

        if use_mlp and len(blocks) > 0:
            h_mlp = block_attn_res(blocks, partial_block, self.pseudo_query_mlp, self.norm_mlp)
        else:
            h_mlp = partial_block

        return h_attn, h_mlp

    def reset_parameters(self):
        """Reset pseudo-queries to zero (standard residual behavior)."""
        nn.init.zeros_(self.pseudo_query_attn)
        nn.init.zeros_(self.pseudo_query_mlp)

    def phase1_inter_block(
        self,
        pseudo_queries: torch.Tensor,  # [S, D] - queries for all layers in block
        block_representations: List[torch.Tensor],  # [b0, ..., b_{n-1}]
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
                torch.ones(S, B, T, device=pseudo_queries.device),
            )

        # Stack block representations: [N, B, T, D]
        V = torch.stack(block_representations, dim=0)
        # Keys use the same RMSNorm as inter-block attention in block_attn_res
        K = self.norm_attn(V)  # [N, B, T, D]

        # Batched attention: [S, D] @ [N, B, T, D] -> [S, N, B, T]
        logits = torch.einsum("s d, n b t d -> s n b t", pseudo_queries, K)
        logits = logits / (self.dim**0.5)

        # Compute softmax stats
        max_vals = logits.max(dim=1, keepdim=True).values  # [S, 1, B, T]
        exp_logits = torch.exp(logits - max_vals)  # [S, N, B, T]
        lse = torch.log(exp_logits.sum(dim=1)) + max_vals.squeeze(1)  # [S, B, T]

        # Weighted sum: [S, N, B, T] @ [N, B, T, D] -> [S, B, T, D]
        attn_weights = F.softmax(logits, dim=1)  # [S, N, B, T]
        outputs = torch.einsum("s n b t, n b t d -> s b t d", attn_weights, V)

        return outputs, max_vals.squeeze(1), lse

    def phase2_intra_block(
        self,
        inter_output: torch.Tensor,  # [B, T, D]
        inter_max: torch.Tensor,  # [B, T]
        inter_lse: torch.Tensor,  # [B, T]
        pseudo_query: torch.Tensor,  # [D]
        partial_sum: torch.Tensor,  # [B, T, D] (b_n^i)
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
        K = self.norm_attn(partial_sum)  # [B, T, D]
        logits = torch.einsum("d, b t d -> b t", pseudo_query, K)
        logits = logits / (self.dim**0.5)
        intra_max = logits  # [B, T]

        # Online softmax merge
        merged_max = torch.maximum(inter_max, intra_max)  # [B, T]

        # Compute merged output using the full log-sum-exp for the inter block.
        # inter_lse encodes the total weight of all inter-block values, whereas
        # inter_max is only the per-block maximum. Using inter_lse is required
        # for a correct weighted combination.
        weight_inter = torch.exp(inter_lse - merged_max)  # [B, T]
        weight_intra = torch.exp(intra_max - merged_max)  # [B, T]

        norm = weight_inter + weight_intra  # [B, T]
        merged_output = (
            weight_inter.unsqueeze(-1) * inter_output + weight_intra.unsqueeze(-1) * partial_sum
        ) / norm.unsqueeze(
            -1
        )  # [B, T, D]

        merged_lse = torch.log(norm) + merged_max  # [B, T]

        return merged_output, merged_max, merged_lse


# =============================================================================
# Full Attention Residuals (baseline / ablation)
# =============================================================================


def full_attn_res(
    w_l: torch.Tensor,
    sources: List[torch.Tensor],
    norm: RMSNorm,
) -> torch.Tensor:
    """
    Softmax attention over all previous layer outputs (Eq. 2-4).

    phi(q, k) = exp(q^T RMSNorm(k))
    alpha_{i->l} = phi(w_l, k_i) / sum_j phi(w_l, k_j)
    h_l = sum_i alpha_{i->l} * v_i
    """
    V = torch.stack(sources, dim=0)  # [num_sources, B, T, d]
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", w_l, K)
    alpha = logits.softmax(dim=0)
    return torch.einsum("n b t, n b t d -> b t d", alpha, V)


class FullAttnResTransformerBlock(nn.Module):
    """
    Transformer block with Full Attention Residuals.

    Before each sub-layer, replaces the standard residual with softmax
    attention over all preceding layer outputs via a learned pseudo-query w_l.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = nn.Linear(dim, dim, bias=False)

        self.attn_res_query = nn.Parameter(torch.zeros(dim))
        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_query = nn.Parameter(torch.zeros(dim))
        self.mlp_res_norm = RMSNorm(dim)

    def forward(self, sources: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            sources: all previous layer outputs; sources[0] = embedding

        Returns:
            sources with two new entries appended (attn_out, mlp_out)
        """
        h = full_attn_res(self.attn_res_query, sources, self.attn_res_norm)
        sources.append(self.attn(self.attn_norm(h)))

        h = full_attn_res(self.mlp_res_query, sources, self.mlp_res_norm)
        sources.append(self.mlp(self.mlp_norm(h)))

        return sources


class FullAttnResModel(nn.Module):
    """
    Full Attention Residuals (Section 3.1).
    Memory: O(L*d). Compute: O(L^2*d).
    """

    def __init__(self, dim: int, num_transformer_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [FullAttnResTransformerBlock(dim) for _ in range(num_transformer_blocks)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        sources: List[torch.Tensor] = [h]
        for block in self.blocks:
            sources = block(sources)
        return sources[-1]


# =============================================================================
# Block Attention Residuals — Transformer block integration
# =============================================================================


class BlockAttnResTransformerBlock(nn.Module):
    """
    Transformer block for Block AttnRes.

    Tracks completed_blocks (finalized block reps b_0..b_{n-1}) and
    partial_block (running intra-block sum). At block boundaries, partial_block
    is committed to completed_blocks.
    """

    def __init__(self, dim: int, block_size: int, block_layer_offset: int = 0):
        super().__init__()
        self.block_size = block_size
        self.block_layer_offset = block_layer_offset

        self.attn_norm = RMSNorm(dim)
        self.attn = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = nn.Linear(dim, dim, bias=False)

        self.attn_res_query = nn.Parameter(torch.zeros(dim))
        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_query = nn.Parameter(torch.zeros(dim))
        self.mlp_res_norm = RMSNorm(dim)

    def _inter_block_attn(
        self,
        completed_blocks: List[torch.Tensor],
        partial_block: Optional[torch.Tensor],
        w_l: torch.Tensor,
        norm: RMSNorm,
    ) -> torch.Tensor:
        if partial_block is None:
            V = torch.stack(completed_blocks, dim=0)
            K = norm(V)
            logits = torch.einsum("d, n b t d -> n b t", w_l, K)
            alpha = logits.softmax(dim=0)
            return torch.einsum("n b t, n b t d -> b t d", alpha, V)
        return block_attn_res(completed_blocks, partial_block, w_l, norm)

    def forward(
        self,
        completed_blocks: List[torch.Tensor],
        partial_block: Optional[torch.Tensor],
        layer_in_block: int,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], int]:
        # Attn sub-layer
        if layer_in_block == self.block_size:
            completed_blocks.append(partial_block)  # type: ignore[arg-type]
            partial_block = None
            layer_in_block = 0

        h = self._inter_block_attn(
            completed_blocks, partial_block, self.attn_res_query, self.attn_res_norm
        )
        attn_out = self.attn(self.attn_norm(h))
        partial_block = attn_out if partial_block is None else partial_block + attn_out
        layer_in_block += 1

        # MLP sub-layer
        if layer_in_block == self.block_size:
            completed_blocks.append(partial_block)  # type: ignore[arg-type]
            partial_block = None
            layer_in_block = 0

        h = self._inter_block_attn(
            completed_blocks, partial_block, self.mlp_res_query, self.mlp_res_norm
        )
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = mlp_out if partial_block is None else partial_block + mlp_out
        layer_in_block += 1

        return completed_blocks, partial_block, layer_in_block


class BlockAttnResModel(nn.Module):
    """
    Block Attention Residuals (Section 3.2).

    L layers grouped into N = L/S blocks.
    Intra-block: standard residual -> b_n = sum_{j in B_n} f_j(h_j)
    Inter-block: softmax attention over [b_0, ..., b_{n-1}, b_n^i]

    Paper's 48B model: L=54, S=6, N=9 blocks + embedding = 10 sources.
    Memory: O(N*d) vs O(L*d) for Full AttnRes.
    """

    def __init__(self, dim: int, num_transformer_blocks: int, block_size: int = 6):
        super().__init__()
        self.block_size = block_size
        self.blocks = nn.ModuleList(
            [
                BlockAttnResTransformerBlock(dim, block_size, i * 2)
                for i in range(num_transformer_blocks)
            ]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        completed_blocks: List[torch.Tensor] = [h]
        partial_block: Optional[torch.Tensor] = None
        layer_in_block = 0

        for block in self.blocks:
            completed_blocks, partial_block, layer_in_block = block(
                completed_blocks, partial_block, layer_in_block
            )

        if partial_block is not None:
            completed_blocks.append(partial_block)

        V_final = torch.stack(completed_blocks, dim=0)
        w_out = self.blocks[-1].mlp_res_query
        norm_out = self.blocks[-1].mlp_res_norm
        K_final = norm_out(V_final)
        logits = torch.einsum("d, n b t d -> n b t", w_out, K_final)
        alpha = logits.softmax(dim=0)
        return torch.einsum("n b t, n b t d -> b t d", alpha, V_final)


# =============================================================================
# Standard Residuals (baseline)
# =============================================================================


class StandardTransformerBlock(nn.Module):
    """Single Transformer block: h = h + f(norm(h))."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.attn_norm(h))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class StandardResidualModel(nn.Module):
    """
    h_l = h_{l-1} + f_{l-1}(h_{l-1})
    Hidden state magnitude grows as O(L).
    """

    def __init__(self, dim: int, num_transformer_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [StandardTransformerBlock(dim) for _ in range(num_transformer_blocks)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h


# Convenience function for simple usage
def create_block_attnres(
    dim: int,
    num_blocks: int = 8,
    use_two_phase: bool = True,
) -> nn.Module:
    """Factory function to create BlockAttnRes module."""
    if use_two_phase:
        block_size = 32 // num_blocks  # Assuming 32 layers
        return TwoPhaseBlockAttnRes(dim, block_size)
    return BlockAttnRes(dim, num_blocks)
