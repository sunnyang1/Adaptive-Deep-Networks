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
    """Configuration for Polar-coordinate qTTT."""
    
    num_steps: int = 16
    learning_rate: float = 0.005
    span_length: int = 128
    
    # Polar-specific settings
    adapt_magnitude: bool = False  # False = freeze r, adapt only θ
    adapt_direction: bool = True   # True = adapt θ
    use_spherical_sgd: bool = True  # Use Riemannian optimization on sphere
    
    # TurboQuant integration
    use_turboquant: bool = True    # Enable 4-bit execution
    turboquant_bits: int = 4       # Total bits (3 for angles + 1 QJL)
    
    # Margin maximization
    margin_temperature: float = 1.0
    early_stop_threshold: Optional[float] = None


class SphericalSGD:
    """
    Stochastic Gradient Descent on the unit sphere.
    
    Uses exponential map for updates on the manifold.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0
    ):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
    
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
        # Project gradient to tangent space
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
        # Compute gradients
        grad_r, grad_u = torch.autograd.grad(
            loss, [self.r_adapt, self.u_adapt],
            retain_graph=False,
            allow_unused=True
        )
        
        # Update magnitude (if trainable)
        if self.config.adapt_magnitude and grad_r is not None:
            with torch.no_grad():
                self.r_adapt = self.r_adapt - self.config.learning_rate * grad_r
                self.r_adapt.requires_grad = True
        
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
                    self.u_adapt.requires_grad = True


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
        adapter = PolarQueryAdapter(magnitude, direction, self.config)
        
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
            
            # Compute margin maximization loss
            loss = self._compute_margin_loss(
                attn_output,
                seq_positions,
                distractor_positions,
                projection_head,
                target_token_ids
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
    
    def _compute_margin_loss(
        self,
        attn_output: torch.Tensor,
        seq_positions: torch.Tensor,
        distractor_positions: Optional[torch.Tensor],
        projection_head: Optional[nn.Module],
        target_token_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute margin maximization loss."""
        if projection_head is None or target_token_ids is None:
            # Simple reconstruction loss fallback
            return attn_output.pow(2).mean()
        
        # Project to logits
        logits = projection_head(attn_output)
        
        # Gather target logits
        # Simplified - actual implementation would use proper indexing
        target_logits = logits[..., target_token_ids].max(dim=-1).values
        
        # Max distractor
        if distractor_positions is not None:
            distractor_logits = logits[..., distractor_positions]
            max_distractor = distractor_logits.max(dim=-1).values
        else:
            max_distractor = logits.max(dim=-1).values
        
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
        
        # TurboQuant discount
        if use_turboquant and self.config.use_turboquant:
            # 8× reduction: 2× arithmetic + 4× memory
            discount = 8.0
            total_cost = total_cost / discount
        
        return {
            'per_step_flops': step_cost,
            'total_flops': total_cost,
            'num_steps': N,
            'turboquant_discount': 8.0 if (use_turboquant and self.config.use_turboquant) else 1.0,
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
            'turboquant_enabled': self.turboquant_enabled,
            'depth_priority_factor': self.depth_priority_factor,
            'policy': 'strict_depth_priority' if self.turboquant_enabled else 'balanced'
        }
