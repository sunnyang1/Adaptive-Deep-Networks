"""
Reconstruction Loss for Dynamic Gating

Computes self-supervised reconstruction loss as a difficulty signal.
Based on: Section 4.1.2 of Adaptive Deep Networks paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ReconstructionLoss(nn.Module):
    """
    TTT Reconstruction Loss for gating signal.
    
    Uses frozen KV cache from initial prefill to compute
    reconstruction loss on a span of tokens.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        span_length: int = 128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.span_length = span_length
        
        # Projection head for reconstruction
        self.reconstruction_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, D]
        target_tokens: torch.Tensor,  # [B, T]
        mask: Optional[torch.Tensor] = None,  # [B, T]
        span_start: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            hidden_states: Hidden states from model
            target_tokens: Target token IDs to reconstruct
            mask: Optional mask for valid positions
            span_start: Start position of span (if None, use random)
        
        Returns:
            Scalar loss value
        """
        B, T, D = hidden_states.shape
        
        # Select span
        if span_start is None:
            if T > self.span_length:
                span_start = torch.randint(0, T - self.span_length, (1,)).item()
            else:
                span_start = 0
        
        span_end = min(span_start + self.span_length, T)
        
        # Get span hidden states and targets
        span_hidden = hidden_states[:, span_start:span_end, :]  # [B, k, D]
        span_targets = target_tokens[:, span_start:span_end]  # [B, k]
        
        # Project to vocab
        logits = self.reconstruction_head(span_hidden)  # [B, k, V]
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            span_targets.reshape(-1),
            reduction='mean' if mask is None else 'none'
        )
        
        if mask is not None:
            span_mask = mask[:, span_start:span_end]
            loss = (loss * span_mask.reshape(-1)).sum() / span_mask.sum()
        
        return loss


def compute_reconstruction_loss(
    logits: torch.Tensor,  # [B, T, V]
    targets: torch.Tensor,  # [B, T]
    kv_cache: Optional[Tuple] = None,
    span_length: int = 128
) -> torch.Tensor:
    """
    Compute reconstruction loss from logits and targets.
    
    This is the core gating signal from Section 4.1.2.
    
    Args:
        logits: Model output logits
        targets: Target token IDs
        kv_cache: Frozen KV cache (for documentation, not used in computation)
        span_length: Length of reconstruction span
    
    Returns:
        Reconstruction loss (scalar)
    """
    B, T, V = logits.shape
    
    # Select a span for reconstruction
    if T > span_length:
        start_idx = torch.randint(0, T - span_length, (1,)).item()
    else:
        start_idx = 0
    
    end_idx = min(start_idx + span_length, T)
    
    # Extract span
    span_logits = logits[:, start_idx:end_idx, :]  # [B, k, V]
    span_targets = targets[:, start_idx:end_idx]  # [B, k]
    
    # Compute loss
    loss = F.cross_entropy(
        span_logits.reshape(-1, V),
        span_targets.reshape(-1),
        reduction='mean'
    )
    
    return loss


class GatingLossComputer:
    """
    Computes gating loss with various options.
    
    Supports:
    - Standard reconstruction loss
    - Multi-span reconstruction (more robust)
    - Contrastive reconstruction (experimental)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_spans: int = 1,
        span_length: int = 128
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_spans = num_spans
        self.span_length = span_length
        
        self.reconstruction_head = nn.Linear(hidden_dim, vocab_size)
    
    def compute_multi_span(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss over multiple random spans (more robust signal).
        
        Args:
            hidden_states: [B, T, D]
            targets: [B, T]
        
        Returns:
            Average loss across spans
        """
        losses = []
        
        for _ in range(self.num_spans):
            # Random span
            T = hidden_states.size(1)
            if T > self.span_length:
                start = torch.randint(0, T - self.span_length, (1,)).item()
            else:
                start = 0
            end = min(start + self.span_length, T)
            
            span_hidden = hidden_states[:, start:end, :]
            span_targets = targets[:, start:end]
            
            logits = self.reconstruction_head(span_hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                span_targets.reshape(-1)
            )
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def compute_with_confidence(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss and confidence estimate.
        
        Returns:
            loss: Reconstruction loss
            confidence: Model confidence (1 - normalized entropy)
        """
        logits = self.reconstruction_head(hidden_states)
        
        # Standard loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        
        # Compute confidence (negative entropy)
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(self.vocab_size, dtype=torch.float))
        confidence = 1.0 - (entropy / max_entropy)
        
        return loss, confidence
