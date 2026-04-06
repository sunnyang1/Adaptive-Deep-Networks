"""
Polar-Coordinate Query-only Test-Time Training (qTTT)

Based on: Section 3.3 of Adaptive Deep Networks TurboQuant version

Key innovations:
1. Polar decomposition: w = r * u(θ)
2. Freeze magnitude r, adapt only direction θ (50% parameter reduction)
3. Spherical gradient descent for natural geometry
4. Integration with TurboQuant for 8× cost reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .adaptation import KVCache, compute_attention_with_query


@dataclass
class PolarQTTTConfig:
    """Configuration for Polar-coordinate qTTT.
    
    Based on Section 3.3 of Adaptive Deep Networks paper.
    
    Loss Function (§3.3.2 - §3.3.3):
    - "cross_entropy" (default): Self-supervised next-token prediction.
      This is the primary training signal described in the algorithm.
    - "margin_maximization" (alternative): Explicit margin maximization.
      Useful for fine-grained confidence calibration (§3.3.3).
    
    Paper Defaults (§3.3.4):
    - num_steps=10 for quality (can reduce to 2-4 for speed)
    - learning_rate=0.01 for short sequences, 0.005 for medium, 0.002 for long
    - early_stop_threshold=0.001 to avoid unnecessary iterations
    """
    
    # Paper default: 10 steps (§3.3.4)
    num_steps: int = 10
    # Paper default: 0.01 for short sequences (§3.3.4)
    learning_rate: float = 0.01
    span_length: int = 128
    
    # Loss function selection (§3.3.2 - §3.3.3)
    loss_type: str = "cross_entropy"  # "cross_entropy" or "margin_maximization"
    
    # Polar-specific settings
    adapt_magnitude: bool = False  # False = freeze r, adapt only θ
    adapt_direction: bool = True   # True = adapt θ
    use_spherical_sgd: bool = True  # Use Riemannian optimization on sphere
    
    # RaBitQ integration
    use_rabitq: bool = True        # Enable compressed execution
    rabitq_bits: int = 1           # Total bits: 1=16×, 2=8×, 3=5.3× (vs FP16)
    
    # Early stopping (disabled for paper defaults to ensure full steps)
    early_stop_threshold: Optional[float] = None  # Stop if loss change < threshold
    
    @property
    def compression_ratio(self) -> float:
        """Return compression ratio vs FP16 baseline."""
        # FP16 = 16 bits, rabitq_bits = actual bits per dimension
        return 16.0 / self.rabitq_bits
    
    # Margin maximization (alternative loss, §3.3.3)
    margin_temperature: float = 1.0


# JIT-compiled spherical step for performance
@torch.jit.script
def spherical_step_jit(
    point: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
    momentum: float,
    velocity: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled spherical gradient step.
    
    Args:
        point: Current point on sphere [d]
        gradient: Gradient in ambient space [d]
        learning_rate: Step size
        momentum: Momentum coefficient
        velocity: Current velocity [d]
    
    Returns:
        new_point: Updated point on sphere [d]
        new_velocity: Updated velocity [d]
    """
    # Project gradient to tangent space
    grad_parallel = (gradient * point).sum() * point
    grad_tangent = gradient - grad_parallel
    
    # Momentum update in tangent space
    new_velocity = momentum * velocity - learning_rate * grad_tangent
    
    # Exponential map
    v_norm = torch.norm(new_velocity)
    if v_norm > 1e-8:
        new_point = point * torch.cos(v_norm) + (new_velocity / v_norm) * torch.sin(v_norm)
    else:
        new_point = point
    
    # Renormalize for numerical stability
    new_point = new_point / (torch.norm(new_point) + 1e-8)
    
    return new_point, new_velocity


class SphericalSGD:
    """
    Stochastic Gradient Descent on the unit sphere.
    
    Uses exponential map for updates on the manifold.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        use_jit: bool = True  # NEW: Enable JIT by default
    ):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.use_jit = use_jit
    
    def step(
        self,
        point: torch.Tensor,
        gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Take one step on the sphere using exponential map.
        
        Args:
            point: Current point on sphere (unit vector) [d]
            gradient: Gradient in ambient space [d]
        
        Returns:
            new_point: Updated point on sphere [d]
        """
        # Use JIT version for better performance
        if self.use_jit and point.dim() == 1:
            if self.velocity is None or self.velocity.shape != point.shape:
                self.velocity = torch.zeros_like(point)
            
            new_point, self.velocity = spherical_step_jit(
                point, gradient, self.lr, self.momentum, self.velocity
            )
            return new_point
        
        # Fallback to original implementation
        grad_parallel = (gradient * point).sum() * point
        grad_tangent = gradient - grad_parallel
        
        # Initialize velocity
        if self.velocity is None or self.velocity.shape != point.shape:
            self.velocity = torch.zeros_like(point)
        
        # Momentum update in tangent space
        self.velocity = self.momentum * self.velocity - self.lr * grad_tangent
        
        # Exponential map
        v_norm = torch.norm(self.velocity)
        if v_norm > 1e-8:
            new_point = (point * torch.cos(v_norm) + 
                        (self.velocity / v_norm) * torch.sin(v_norm))
        else:
            new_point = point
        
        # Renormalize for numerical stability
        return F.normalize(new_point, dim=-1)
    
    def reset(self):
        """Reset optimizer state."""
        self.velocity = None


class QueryAdaptationPolarAdapter:
    """
    Adapter for pseudo-queries in polar coordinates.
    
    Handles the decomposition w = r * u(θ) and provides
    gradients for qTTT adaptation.
    """
    
    def __init__(
        self,
        magnitude: torch.Tensor,
        direction: torch.Tensor,
        config: PolarQTTTConfig
    ):
        """
        Args:
            magnitude: Scalar r
            direction: Unit vector u(θ) [d]
            config: Polar qTTT configuration
        """
        self.r = magnitude
        self.u = direction
        self.config = config
        
        # Clone for adaptation (enable gradients)
        self.r_adapt = magnitude.clone().detach()
        self.u_adapt = direction.clone().detach()
        
        # Freeze magnitude if configured
        if config.adapt_magnitude:
            self.r_adapt.requires_grad = True
        else:
            self.r_adapt.requires_grad = False
        
        self.u_adapt.requires_grad = True
        
        # Spherical optimizer for direction
        if config.use_spherical_sgd:
            self.spherical_opt = SphericalSGD(
                learning_rate=config.learning_rate,
                momentum=0.9
            )
        else:
            self.spherical_opt = None
    
    def get_query(self) -> torch.Tensor:
        """Get current adapted query w = r * u."""
        return self.r_adapt * F.normalize(self.u_adapt, dim=-1)
    
    def get_direction(self) -> torch.Tensor:
        """Get current direction (unit vector)."""
        return F.normalize(self.u_adapt, dim=-1)
    
    def update(self, loss: torch.Tensor):
        """
        Update adapted parameters based on loss.
        
        Args:
            loss: Scalar loss tensor
        """
        # Compute gradients for parameters that require grad
        params = []
        if self.config.adapt_magnitude:
            params.append(self.r_adapt)
        params.append(self.u_adapt)
        
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=False,
            allow_unused=True
        )
        
        grad_r = grads[0] if self.config.adapt_magnitude else None
        grad_u = grads[-1]
        
        # Update magnitude (if trainable)
        if self.config.adapt_magnitude and grad_r is not None:
            with torch.no_grad():
                self.r_adapt = self.r_adapt - self.config.learning_rate * grad_r
                self.r_adapt.requires_grad_(True)
        
        # Update direction
        if grad_u is not None:
            if self.spherical_opt is not None:
                # Riemannian update on sphere
                with torch.no_grad():
                    u_normalized = F.normalize(self.u_adapt, dim=-1)
                    new_u = self.spherical_opt.step(u_normalized, grad_u)
                    self.u_adapt = new_u.clone().requires_grad_(True)
            else:
                # Standard gradient descent with renormalization
                with torch.no_grad():
                    self.u_adapt = self.u_adapt - self.config.learning_rate * grad_u
                    self.u_adapt = F.normalize(self.u_adapt, dim=-1)
                    self.u_adapt.requires_grad_(True)


class PolarQTTT(nn.Module):
    """
    Polar-coordinate Query-only Test-Time Training.
    
    Integrates with RaBitQ for 8× cost reduction.
    """
    
    def __init__(
        self,
        config: PolarQTTTConfig,
        hidden_dim: int,
        num_heads: int = 32
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Statistics tracking
        self.adaptation_stats = {
            'num_calls': 0,
            'total_steps': 0,
            'avg_loss_reduction': 0.0
        }
    
    def adapt_pseudo_query(
        self,
        magnitude: torch.Tensor,      # Scalar r
        direction: torch.Tensor,      # Unit vector u(θ) [d]
        kv_cache: KVCache,
        seq_positions: torch.Tensor,
        distractor_positions: Optional[torch.Tensor] = None,
        projection_head: Optional[nn.Module] = None,
        target_token_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Adapt polar pseudo-query via qTTT.
        
        Args:
            magnitude: Magnitude r (frozen)
            direction: Direction u(θ) [d] (adapted)
            kv_cache: Frozen KV cache
            seq_positions: Target token positions
            distractor_positions: Distractor positions for margin loss
            projection_head: Optional projection to vocab
            target_token_ids: Target token IDs
        
        Returns:
            adapted_direction: Optimized direction [d]
            loss_history: Loss trajectory
        """
        # Create adapter
        adapter = QueryAdaptationPolarAdapter(magnitude, direction, self.config)
        
        loss_history = []
        
        for step in range(self.config.num_steps):
            # Forward pass with current adapted query
            query = adapter.get_query()
            
            # Reshape for multi-head attention: [d] -> [1, H, 1, d_h]
            H = self.num_heads
            d_h = self.head_dim
            query_mha = query.view(H, d_h).unsqueeze(0).unsqueeze(2)
            
            # Compute attention
            attn_output = compute_attention_with_query(query_mha, kv_cache)
            
            # Apply projection head if available to get logits
            if projection_head is not None:
                logits = projection_head(attn_output)
            else:
                logits = attn_output
            
            # Compute adaptation loss (cross_entropy default, margin_maximization alternative)
            loss = self._compute_adaptation_loss(
                logits,
                seq_positions,
                target_token_ids=target_token_ids,
                distractor_positions=distractor_positions,
            )
            
            loss_history.append(loss.item())
            
            # Update adapter
            adapter.update(loss)
            
            # Early stopping
            if (self.config.early_stop_threshold is not None and
                len(loss_history) > 1 and
                abs(loss_history[-1] - loss_history[-2]) < self.config.early_stop_threshold):
                break
        
        # Update statistics
        self.adaptation_stats['num_calls'] += 1
        self.adaptation_stats['total_steps'] += len(loss_history)
        if len(loss_history) > 1:
            reduction = loss_history[0] - loss_history[-1]
            self.adaptation_stats['avg_loss_reduction'] = (
                (self.adaptation_stats['avg_loss_reduction'] * 
                 (self.adaptation_stats['num_calls'] - 1) + reduction) /
                self.adaptation_stats['num_calls']
            )
        
        return adapter.get_direction(), loss_history
    
    def adapt_query_projection(
        self,
        queries: torch.Tensor,  # [B, T, D]
        kv_cache: KVCache,
        seq_positions: Optional[torch.Tensor] = None,
        distractor_positions: Optional[torch.Tensor] = None,
        projection_head: Optional[nn.Module] = None,
        target_token_ids: Optional[torch.Tensor] = None,
        model: Optional['AdaptiveTransformer'] = None,
        input_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[List['KVCache']] = None,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Adapt query projection vectors in polar coordinates.
        
        Decomposes each query into magnitude (frozen) and direction (adapted),
        then performs Riemannian gradient descent on the unit sphere.
        
        Args:
            queries: Query tensor [B, T, D]
            kv_cache: Frozen KV cache (legacy, single cache)
            seq_positions: Target positions in the sequence
            distractor_positions: Distractor positions for margin loss
            projection_head: Optional projection to vocab
            target_token_ids: Target token IDs
            model: Optional model for full forward pass (recommended)
            input_ids: Input token IDs for full forward (required if model provided)
            kv_caches: List of KV caches for all layers (required if model provided)
        
        Returns:
            adapted_queries: Optimized queries [B, T, D]
            loss_history: Loss trajectory
        
        Note:
            If model is provided, uses forward_with_frozen_kv for complete
            forward pass. Otherwise falls back to attention-only computation.
        """
        B, T, D = queries.shape
        
        # Polar decomposition: freeze magnitude, adapt direction
        r = queries.norm(dim=-1, keepdim=True).detach()  # [B, T, 1]
        u = queries / (r + 1e-8)  # [B, T, D]
        u_adapt = u.clone().detach().requires_grad_(True)
        
        loss_history = []
        
        for step in range(self.config.num_steps):
            query = r * F.normalize(u_adapt, dim=-1)  # [B, T, D]
            
            # Use full model forward if available (recommended)
            if model is not None and input_ids is not None and kv_caches is not None:
                # Reshape query for all layers: [B, T, D] broadcast to all positions
                # For qTTT, we adapt the query at specific positions
                adapted_query_full = query  # [B, T, D]
                
                # Forward with frozen KV and adapted query
                # Apply adapted query only to the last layer (where it was computed)
                logits = model.forward_with_frozen_kv(
                    input_ids=input_ids,
                    kv_caches=kv_caches,
                    adapted_query=adapted_query_full,
                    adapted_query_layer_idx=model.config.num_layers - 1,
                    use_attnres=True,  # Enable AttnRes for full pipeline
                )
                
                # Compute loss on logits (full output distribution)
                # Default: cross_entropy (§3.3.2), alternative: margin_maximization (§3.3.3)
                loss = self._compute_adaptation_loss(
                    logits,
                    seq_positions if seq_positions is not None else torch.arange(T, device=queries.device),
                    target_token_ids,
                    distractor_positions,
                )
            else:
                # Fallback: attention-only computation (legacy)
                # Reshape for multi-head attention: [B, T, D] -> [B, H, T, d]
                query_mha = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                attn_output = compute_attention_with_query(query_mha, kv_cache)
                
                # For legacy path, apply projection head to get logits if available
                if projection_head is not None:
                    logits = projection_head(attn_output)
                else:
                    # Use attention output directly (fallback)
                    logits = attn_output
                
                # Compute loss
                if seq_positions is None:
                    seq_positions_loss = torch.arange(T, device=queries.device)
                else:
                    seq_positions_loss = seq_positions
                
                loss = self._compute_adaptation_loss(
                    logits,
                    seq_positions_loss,
                    target_token_ids,
                    distractor_positions,
                )
            
            loss_history.append(loss.item())
            
            # Gradient w.r.t. direction
            grad_u = torch.autograd.grad(loss, u_adapt)[0]
            
            with torch.no_grad():
                u_norm = F.normalize(u_adapt, dim=-1)
                
                # Project gradient to tangent space
                grad_parallel = (grad_u * u_norm).sum(dim=-1, keepdim=True) * u_norm
                grad_tangent = grad_u - grad_parallel
                
                # Riemannian exponential map step
                step_vec = self.config.learning_rate * grad_tangent
                step_norm = step_vec.norm(dim=-1, keepdim=True)
                
                mask = step_norm.squeeze(-1) > 1e-8
                new_u = u_norm.clone()
                if mask.any():
                    new_u[mask] = (
                        u_norm[mask] * torch.cos(step_norm[mask])
                        + (step_vec[mask] / (step_norm[mask] + 1e-8)) * torch.sin(step_norm[mask])
                    )
                
                u_adapt = new_u.requires_grad_(True)
            
            # Early stopping
            if (self.config.early_stop_threshold is not None and
                len(loss_history) > 1 and
                abs(loss_history[-1] - loss_history[-2]) < self.config.early_stop_threshold):
                break
        
        # Update statistics
        self.adaptation_stats['num_calls'] += 1
        self.adaptation_stats['total_steps'] += len(loss_history)
        if len(loss_history) > 1:
            reduction = loss_history[0] - loss_history[-1]
            self.adaptation_stats['avg_loss_reduction'] = (
                (self.adaptation_stats['avg_loss_reduction'] *
                 (self.adaptation_stats['num_calls'] - 1) + reduction) /
                self.adaptation_stats['num_calls']
            )
        
        return r * F.normalize(u_adapt, dim=-1), loss_history
    
    def _compute_adaptation_loss(
        self,
        logits: torch.Tensor,
        seq_positions: torch.Tensor,
        target_token_ids: Optional[torch.Tensor] = None,
        distractor_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute adaptation loss based on config.loss_type.
        
        Supports two loss functions as described in §3.3.2 - §3.3.3:
        
        1. "cross_entropy" (default): Self-supervised next-token prediction.
           This is the primary training signal described in Algorithm §3.3.2.
           
        2. "margin_maximization" (alternative): Explicit margin maximization.
           Useful for fine-grained confidence calibration (§3.3.3).
        
        Args:
            logits: Model output logits [B, T, V]
            seq_positions: Target sequence positions
            target_token_ids: Target token IDs for cross-entropy or margin
            distractor_positions: Optional distractor positions for margin loss
            
        Returns:
            Scalar loss tensor
        """
        # Handle different input shapes
        if logits.dim() == 4:  # [B, H, T, V] -> [B, T, V]
            logits = logits.mean(dim=1)
        
        B, T, V = logits.shape
        device = logits.device
        
        # Default to reconstruction loss if no targets provided
        if target_token_ids is None:
            return logits.pow(2).mean()
        
        # Ensure target_token_ids has correct shape [B, T]
        if target_token_ids.dim() == 0:
            target_token_ids = target_token_ids.unsqueeze(0).unsqueeze(0).expand(B, T)
        elif target_token_ids.dim() == 1:
            if target_token_ids.size(0) == T:
                target_token_ids = target_token_ids.unsqueeze(0).expand(B, -1)
            else:
                target_token_ids = target_token_ids.unsqueeze(0).unsqueeze(0).expand(B, T)
        
        # Safe token ids for cross-entropy (vocab may differ from sampling range)
        target_token_ids = target_token_ids.clamp(0, V - 1)
        
        # Compute loss based on type
        if self.config.loss_type == "cross_entropy":
            # Cross-entropy loss (default, §3.3.2)
            # If target has fewer positions than logits, use only the last positions
            if target_token_ids.size(1) < T:
                # Use only the last |target| positions from logits
                logits_subset = logits[:, -target_token_ids.size(1):, :]  # [B, target_T, V]
                logits_flat = logits_subset.reshape(-1, V)
                targets_flat = target_token_ids.reshape(-1)
            else:
                logits_flat = logits.view(-1, V)
                targets_flat = target_token_ids.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        elif self.config.loss_type == "margin_maximization":
            # Margin maximization loss (alternative, §3.3.3)
            # If target has fewer positions than logits, use only the last positions
            if target_token_ids.size(1) < T:
                logits_subset = logits[:, -target_token_ids.size(1):, :]  # [B, target_T, V]
                T_subset = target_token_ids.size(1)
                batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, T_subset)
                time_idx = torch.arange(T_subset, device=device).unsqueeze(0).expand(B, -1)
                target_logits = logits_subset[batch_idx, time_idx, target_token_ids]
                
                # Compute max distractor
                mask = torch.ones_like(logits_subset, dtype=torch.bool)
                mask[batch_idx, time_idx, target_token_ids] = False
                max_distractor = logits_subset.masked_fill(~mask, float('-inf')).max(dim=-1).values
            else:
                batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, T)
                time_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
                target_logits = logits[batch_idx, time_idx, target_token_ids]
                
                # Compute max distractor
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[batch_idx, time_idx, target_token_ids] = False
                max_distractor = logits.masked_fill(~mask, float('-inf')).max(dim=-1).values
            
            # Margin maximization
            margin = target_logits - max_distractor
            loss = -F.logsigmoid(margin / self.config.margin_temperature).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.config.loss_type}. "
                           f"Choose from 'cross_entropy' or 'margin_maximization'")
        
        return loss
    
    # Backward compatibility alias
    _compute_margin_loss = _compute_adaptation_loss
    
    def _compute_margin_loss_legacy(
        self,
        attn_output: torch.Tensor,
        seq_positions: torch.Tensor,
        distractor_positions: Optional[torch.Tensor],
        projection_head: Optional[nn.Module],
        target_token_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute margin maximization loss with proper 4D indexing."""
        if projection_head is None or target_token_ids is None:
            # Simple reconstruction loss fallback
            return attn_output.pow(2).mean()
        
        # Project to logits: [B, H, k, V]
        logits = projection_head(attn_output)
        B, H, k, V = logits.shape
        
        # Ensure indices are 2D [B, k]
        if target_token_ids.dim() == 0:
            target_token_ids = target_token_ids.unsqueeze(0).unsqueeze(0).expand(B, k)
        elif target_token_ids.dim() == 1:
            target_token_ids = target_token_ids.unsqueeze(0).expand(B, -1)
        
        if seq_positions.dim() == 0:
            seq_positions = seq_positions.unsqueeze(0).unsqueeze(0).expand(B, k)
        elif seq_positions.dim() == 1:
            seq_positions = seq_positions.unsqueeze(0).expand(B, -1)
        
        # Gather target logits: [B, H, k]
        target_logits = logits[
            torch.arange(B, device=logits.device).view(-1, 1, 1),
            torch.arange(H, device=logits.device).view(1, -1, 1),
            seq_positions.unsqueeze(1).expand(-1, H, -1),
            target_token_ids.unsqueeze(1).expand(-1, H, -1)
        ]
        
        # Max distractor
        if distractor_positions is not None:
            if distractor_positions.dim() == 1:
                distractor_positions = distractor_positions.unsqueeze(0).expand(B, -1)
            # Gather distractor logits at given sequence positions: [B, H, num_distractors, V]
            num_d = distractor_positions.size(1)
            dp_expanded = distractor_positions.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, V)
            # Need to index into [B, H, seq_len, V] where seq_len = logits.size(2)
            # Ensure we only index valid positions
            dp_expanded = torch.clamp(dp_expanded, 0, logits.size(2) - 1)
            b_idx = torch.arange(B, device=logits.device).view(-1, 1, 1, 1).expand(-1, H, num_d, V)
            h_idx = torch.arange(H, device=logits.device).view(1, -1, 1, 1).expand(B, -1, num_d, V)
            v_idx = torch.arange(V, device=logits.device).view(1, 1, 1, -1).expand(B, H, num_d, -1)
            distractor_logits = logits[b_idx, h_idx, dp_expanded, v_idx]
            max_distractor = distractor_logits.max(dim=-1).values  # [B, H, num_d]
            max_distractor = max_distractor.max(dim=-1).values     # [B, H]
            # Expand to match target_logits [B, H, k]
            max_distractor = max_distractor.unsqueeze(-1).expand(-1, -1, k)
        else:
            max_distractor = logits.max(dim=-1).values  # [B, H, k]
        
        # Margin maximization
        margin = target_logits - max_distractor
        loss = -F.logsigmoid(margin / self.config.margin_temperature).mean()
        
        return loss
    
    def compute_effective_cost(
        self,
        batch_size: int,
        seq_len: int,
        use_turboquant: bool = True
    ) -> Dict[str, float]:
        """
        Compute effective FLOP cost of polar qTTT.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            use_turboquant: Account for TurboQuant acceleration
        
        Returns:
            Cost breakdown dictionary
        """
        H = self.num_heads
        d = self.head_dim
        k = self.config.span_length
        N = self.config.num_steps
        
        # Base cost per step
        query_proj = batch_size * H * k * d * d
        attn = batch_size * H * k * seq_len * d
        backward = 2 * (query_proj + attn) * 0.5  # 50% for polar (only direction)
        
        step_cost = query_proj + attn + backward
        total_cost = N * step_cost
        
        # RaBitQ discount - matches paper §3.4: 16× for 1-bit vs FP16
        if self.config.use_rabitq:
            # 16× reduction for 1-bit (vs FP16 baseline)
            discount = 16.0 / self.config.rabitq_bits
            total_cost = total_cost / discount
        
        return {
            'per_step_flops': step_cost,
            'total_flops': total_cost,
            'num_steps': N,
            'rabitq_discount': 8.0 if self.config.use_rabitq else 1.0,
            'parameter_reduction': 0.5  # 50% fewer adapted params
        }
    
    def get_stats(self) -> Dict:
        """Get adaptation statistics."""
        return self.adaptation_stats.copy()


class DepthPriorityController:
    """
    Controller for depth-priority computation allocation.
    
    Under TurboQuant acceleration, strictly prioritizes depth (qTTT)
    over width (thinking tokens) when gating activates.
    """
    
    def __init__(
        self,
        max_qttt_steps: int = 32,
        think_tokens_budget: int = 0,  # Set to 0 for strict depth priority
        turboquant_enabled: bool = True
    ):
        self.max_qttt_steps = max_qttt_steps
        self.think_tokens_budget = think_tokens_budget
        self.turboquant_enabled = turboquant_enabled
        
        # Policy: With TurboQuant, depth is 8× cheaper
        self.depth_priority_factor = 8.0 if turboquant_enabled else 2.0
    
    def allocate(
        self,
        gating_active: bool,
        budget_constraint: str = 'moderate'
    ) -> Tuple[int, int]:
        """
        Allocate computation budget between depth and width.
        
        Args:
            gating_active: Whether gating triggered adaptation
            budget_constraint: 'constrained', 'moderate', or 'abundant'
        
        Returns:
            (num_qttt_steps, num_think_tokens)
        """
        if not gating_active:
            return 0, 0
        
        # Strict depth priority under TurboQuant
        if budget_constraint == 'constrained':
            return self.max_qttt_steps, 0
        elif budget_constraint == 'moderate':
            # Minimal thinking tokens, maximize qTTT
            return self.max_qttt_steps, min(128, self.think_tokens_budget)
        else:  # abundant
            # Hybrid with learned balance (future work)
            return self.max_qttt_steps, self.think_tokens_budget
    
    def get_policy_summary(self) -> Dict:
        """Get summary of allocation policy."""
        return {
            'max_qttt_steps': self.max_qttt_steps,
            'think_tokens_budget': self.think_tokens_budget,
            'rabitq_enabled': self.turboquant_enabled,
            'depth_priority_factor': self.depth_priority_factor,
            'policy': 'strict_depth_priority' if self.turboquant_enabled else 'balanced'
        }
