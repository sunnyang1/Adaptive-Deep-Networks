"""
Query-only Test-Time Training (qTTT) Implementation

Based on: Section 4.4 of Adaptive Deep Networks paper
Key feature: Only query parameters are updated, KV cache remains frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass


@dataclass
class qTTTConfig:
    """Configuration for qTTT adaptation."""
    num_steps: int = 16
    learning_rate: float = 0.005
    span_length: int = 128
    target_type: str = "pseudo_query"  # "pseudo_query" or "query_projection"
    margin_temperature: float = 1.0
    early_stop_threshold: Optional[float] = None


class KVCache:
    """
    Frozen Key-Value Cache from initial prefill.
    
    Critical for efficiency: keys and values are computed once
    and reused across all qTTT steps.
    """
    
    def __init__(
        self,
        keys: torch.Tensor,  # [B, num_heads, T, head_dim]
        values: torch.Tensor  # [B, num_heads, T, head_dim]
    ):
        self.keys = keys.detach()
        self.values = values.detach()
        self.is_frozen = True
    
    def __len__(self):
        return self.keys.size(2)  # T
    
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return keys and values (detached, no grad)."""
        return self.keys, self.values


def compute_attention_with_query(
    query: torch.Tensor,  # [B, num_heads, k, head_dim]
    kv_cache: KVCache,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute attention with query against frozen KV cache.
    
    Args:
        query: Query tensor [B, num_heads, k, head_dim]
        kv_cache: Frozen key-value cache
        mask: Optional attention mask
    
    Returns:
        Attention output [B, num_heads, k, head_dim]
    """
    keys, values = kv_cache.get_kv()
    
    # Scaled dot-product attention
    # Q @ K^T: [B, H, k, d] @ [B, H, T, d]^T -> [B, H, k, T]
    scores = torch.matmul(query, keys.transpose(-2, -1))
    scores = scores / (query.size(-1) ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)  # [B, H, k, T]
    
    # Attn @ V: [B, H, k, T] @ [B, H, T, d] -> [B, H, k, d]
    output = torch.matmul(attn_weights, values)
    
    return output


def qttt_adapt(
    initial_query: torch.Tensor,
    kv_cache: KVCache,
    target_positions: torch.Tensor,
    distractor_positions: Optional[torch.Tensor] = None,
    num_steps: int = 16,
    learning_rate: float = 0.005,
    projection_head: Optional[nn.Module] = None,
    targets: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, List[float]]:
    """
    Query-only Test-Time Training adaptation.
    
    Algorithm from Section 4.4 and Appendix A.1:
    1. Clone query and enable gradients
    2. For each step:
       a. Compute attention with adapted query
       b. Compute margin maximization loss
       c. Update only query parameters
    3. Return adapted query
    
    Args:
        initial_query: Initial query tensor [B, H, k, D] or pseudo-query [D]
        kv_cache: Frozen key-value cache
        target_positions: Positions of target tokens in context
        distractor_positions: Positions of distractor tokens
        num_steps: Number of adaptation steps
        learning_rate: Step size for adaptation
        projection_head: Optional projection to vocab
        targets: Target token IDs for loss computation
    
    Returns:
        adapted_query: Optimized query
        loss_history: Loss values at each step
    """
    # Clone and enable gradients
    query = initial_query.clone().detach().requires_grad_(True)
    
    loss_history = []
    
    for step in range(num_steps):
        # Forward pass with current adapted query
        attn_output = compute_attention_with_query(query, kv_cache)
        
        # Compute margin maximization loss if projection available
        if projection_head is not None and targets is not None:
            # Project to logits
            logits = projection_head(attn_output)  # [B, H, k, V]
            
            # Margin maximization loss
            target_logits = logits[..., target_positions, :]
            
            if distractor_positions is not None:
                distractor_logits = logits[..., distractor_positions, :]
                max_distractor = distractor_logits.max(dim=-1, keepdim=True).values
            else:
                # Use all other positions as distractors
                max_distractor = logits.max(dim=-1, keepdim=True).values
            
            margin = target_logits - max_distractor
            loss = -F.logsigmoid(margin).mean()
        else:
            # Simple reconstruction-like loss if no targets
            loss = attn_output.pow(2).mean()
        
        loss_history.append(loss.item())
        
        # Backward and update (only query!)
        grad = torch.autograd.grad(loss, query)[0]
        
        with torch.no_grad():
            query = query - learning_rate * grad
        
        # Re-enable gradients for next step
        query = query.detach().requires_grad_(True)
    
    return query.detach(), loss_history


class QueryOnlyTTT(nn.Module):
    """
    Query-only Test-Time Training module.
    
    Provides a clean interface for qTTT with support for:
    - Pseudo-query adaptation (per-layer)
    - Query projection adaptation (per-token)
    - Mixed adaptation (both)
    """
    
    def __init__(
        self,
        config: qTTTConfig,
        hidden_dim: int,
        num_heads: int = 32
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Optional projection head for margin maximization
        if config.target_type == "query_projection":
            self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.query_projection = None
    
    def adapt_pseudo_query(
        self,
        pseudo_query: torch.Tensor,  # [D]
        kv_cache: KVCache,
        target_positions: torch.Tensor,
        distractor_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Adapt a pseudo-query vector.
        
        Args:
            pseudo_query: Learned pseudo-query [D]
            kv_cache: Frozen KV cache
            target_positions: Target token positions
            distractor_positions: Distractor positions
        
        Returns:
            adapted_query: Optimized pseudo-query
            loss_history: Loss trajectory
        """
        # Expand pseudo-query for attention computation
        # [D] -> [1, 1, 1, D] (will be broadcasted)
        query_expanded = pseudo_query.view(1, 1, 1, -1)
        
        adapted, losses = qttt_adapt(
            query_expanded,
            kv_cache,
            target_positions,
            distractor_positions,
            num_steps=self.config.num_steps,
            learning_rate=self.config.learning_rate
        )
        
        # Squeeze back to [D]
        return adapted.view(-1), losses
    
    def adapt_query_projection(
        self,
        queries: torch.Tensor,  # [B, T, D]
        kv_cache: KVCache,
        target_positions: torch.Tensor,
        distractor_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Adapt query projection matrix.
        
        This allows per-token query modification at higher cost.
        
        Args:
            queries: Query tensor [B, T, D]
            kv_cache: Frozen KV cache
            target_positions: Target positions
            distractor_positions: Distractor positions
        
        Returns:
            adapted_queries: Optimized queries
            loss_history: Loss trajectory
        """
        # Reshape for multi-head attention: [B, T, D] -> [B, H, T, d]
        B, T, D = queries.shape
        queries_mha = queries.view(B, T, self.num_heads, self.head_dim)
        queries_mha = queries_mha.transpose(1, 2)  # [B, H, T, d]
        
        adapted, losses = qttt_adapt(
            queries_mha,
            kv_cache,
            target_positions,
            distractor_positions,
            num_steps=self.config.num_steps,
            learning_rate=self.config.learning_rate,
            projection_head=self.query_projection if self.query_projection else None
        )
        
        # Reshape back: [B, H, T, d] -> [B, T, D]
        adapted = adapted.transpose(1, 2).contiguous()
        adapted = adapted.view(B, T, D)
        
        return adapted, losses
    
    def compute_flops(
        self,
        batch_size: int,
        seq_len: int,
        span_len: int
    ) -> dict:
        """
        Compute FLOPs for qTTT adaptation.
        
        For comparison with thinking token generation.
        
        Returns:
            Dictionary with FLOP counts per component
        """
        H = self.num_heads
        d = self.head_dim
        k = span_len
        T = seq_len
        N = self.config.num_steps
        
        # Per-step FLOPs
        # Query projection: B * H * k * d^2
        query_proj_flops = batch_size * H * k * d * d
        
        # Attention: B * H * k * T * d
        attn_flops = batch_size * H * k * T * d
        
        # Backward (roughly 2x forward for query-only)
        backward_flops = 2 * (query_proj_flops + attn_flops)
        
        step_flops = query_proj_flops + attn_flops + backward_flops
        total_flops = N * step_flops
        
        return {
            'per_step': step_flops,
            'total': total_flops,
            'num_steps': N,
            'query_projection': query_proj_flops * N,
            'attention': attn_flops * N,
            'backward': backward_flops * N
        }


class AdaptiveInference:
    """
    High-level interface for adaptive inference with qTTT.
    
    Combines gating decision with qTTT execution.
    """
    
    def __init__(
        self,
        qttt_module: QueryOnlyTTT,
        gating_controller: Optional[nn.Module] = None
    ):
        self.qttt = qttt_module
        self.gating = gating_controller
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        reconstruction_loss: Optional[float] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Conditional forward pass with optional qTTT.
        
        Args:
            hidden_states: Input hidden states
            kv_cache: Frozen KV cache
            reconstruction_loss: For gating decision
        
        Returns:
            output_states: Possibly adapted hidden states
            metadata: Information about adaptation
        """
        metadata = {
            'adapted': False,
            'num_steps': 0,
            'loss_history': []
        }
        
        # Gating decision
        if self.gating is not None and reconstruction_loss is not None:
            should_adapt, num_steps, threshold = self.gating.decide(
                reconstruction_loss
            )
        else:
            should_adapt = True
            num_steps = self.qttt.config.num_steps
        
        if should_adapt and num_steps > 0:
            # Perform qTTT adaptation
            adapted_states, losses = self.qttt.adapt_query_projection(
                hidden_states,
                kv_cache
            )
            
            metadata['adapted'] = True
            metadata['num_steps'] = num_steps
            metadata['loss_history'] = losses
            
            return adapted_states, metadata
        
        return hidden_states, metadata
