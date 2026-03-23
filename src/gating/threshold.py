"""
Dynamic Threshold Calibration for Gating

Implements EMA-based and target-rate-based threshold calibration.
Based on: Section 4.2 of Adaptive Deep Networks paper
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from collections import deque


class DynamicThreshold(nn.Module):
    """Base class for dynamic threshold calibration."""
    
    def __init__(self, initial_threshold: float = 2.0):
        super().__init__()
        self.register_buffer('threshold', torch.tensor(initial_threshold))
        self.history = deque(maxlen=1000)
    
    def update(self, loss_value: float) -> float:
        """Update threshold based on new loss value."""
        raise NotImplementedError
    
    def should_adapt(self, loss_value: float) -> bool:
        """Return True if input difficulty warrants adaptation."""
        return loss_value > self.threshold.item()
    
    def get_stats(self) -> dict:
        """Return statistics about threshold calibration."""
        if len(self.history) == 0:
            return {"threshold": self.threshold.item(), "history_size": 0}
        
        return {
            "threshold": self.threshold.item(),
            "history_size": len(self.history),
            "mean_loss": sum(self.history) / len(self.history),
            "min_loss": min(self.history),
            "max_loss": max(self.history)
        }


class EMAThreshold(DynamicThreshold):
    """
    Exponential Moving Average threshold calibration.
    
    Formula from Section 4.2.1:
        τ_{t+1} = β * τ_t + (1-β) * percentile(L_rec^(t), p_target)
    
    Default β = 0.99 for slow adaptation to distribution shifts.
    """
    
    def __init__(
        self,
        initial_threshold: float = 2.0,
        beta: float = 0.99,
        percentile: float = 70.0
    ):
        super().__init__(initial_threshold)
        self.beta = beta
        self.percentile = percentile
        self.loss_buffer = deque(maxlen=100)  # Recent losses for percentile
    
    def update(self, loss_value: float) -> float:
        """
        Update threshold using EMA on percentile.
        
        Args:
            loss_value: New reconstruction loss value
        
        Returns:
            Updated threshold
        """
        self.history.append(loss_value)
        self.loss_buffer.append(loss_value)
        
        if len(self.loss_buffer) < 10:
            # Not enough data yet
            return self.threshold.item()
        
        # Compute percentile of recent losses
        losses_tensor = torch.tensor(list(self.loss_buffer))
        p_value = torch.quantile(losses_tensor, self.percentile / 100.0)
        
        # EMA update
        new_threshold = self.beta * self.threshold + (1 - self.beta) * p_value
        self.threshold.copy_(new_threshold)
        
        return new_threshold.item()


class TargetRateThreshold(DynamicThreshold):
    """
    Maintain target adaptation rate through threshold adjustment.
    
    Formula from Section 4.2.2:
        τ_{t+1} = τ_t + η * (ρ_target - 1[L_rec^(t) > τ_t])
    
    This ensures predictable computational budgeting.
    """
    
    def __init__(
        self,
        initial_threshold: float = 2.0,
        target_rate: float = 0.3,
        learning_rate: float = 0.01,
        window_size: int = 100
    ):
        super().__init__(initial_threshold)
        self.target_rate = target_rate
        self.lr = learning_rate
        self.window_size = window_size
        self.recent_decisions = deque(maxlen=window_size)
        self.adaptation_count = 0
        self.total_count = 0
    
    def update(self, loss_value: float) -> float:
        """
        Update threshold to maintain target adaptation rate.
        
        Args:
            loss_value: New reconstruction loss value
        
        Returns:
            Updated threshold
        """
        self.history.append(loss_value)
        
        # Binary decision
        should_adapt = loss_value > self.threshold.item()
        self.recent_decisions.append(1 if should_adapt else 0)
        
        self.total_count += 1
        if should_adapt:
            self.adaptation_count += 1
        
        # Compute current rate
        if len(self.recent_decisions) > 0:
            current_rate = sum(self.recent_decisions) / len(self.recent_decisions)
        else:
            current_rate = 0.0
        
        # Proportional control
        error = self.target_rate - current_rate
        new_threshold = self.threshold.item() + self.lr * error
        
        # Ensure threshold stays positive
        new_threshold = max(0.01, new_threshold)
        
        self.threshold.copy_(torch.tensor(new_threshold))
        
        return new_threshold
    
    def get_stats(self) -> dict:
        """Get statistics including oracle recovery estimation."""
        stats = super().get_stats()
        
        if len(self.recent_decisions) > 0:
            current_rate = sum(self.recent_decisions) / len(self.recent_decisions)
        else:
            current_rate = 0.0
        
        stats.update({
            "target_rate": self.target_rate,
            "current_rate": current_rate,
            "rate_error": abs(self.target_rate - current_rate),
            "total_adaptations": self.adaptation_count,
            "total_samples": self.total_count
        })
        
        return stats


class HybridThreshold(DynamicThreshold):
    """
    Hybrid approach combining EMA and target rate.
    
    Uses EMA for smooth tracking but adjusts percentile based on
    deviation from target rate.
    """
    
    def __init__(
        self,
        initial_threshold: float = 2.0,
        beta: float = 0.99,
        target_rate: float = 0.3,
        rate_lr: float = 0.1
    ):
        super().__init__(initial_threshold)
        self.beta = beta
        self.target_rate = target_rate
        self.rate_lr = rate_lr
        
        self.percentile = 70.0  # Will be adjusted
        self.decisions = deque(maxlen=100)
    
    def update(self, loss_value: float) -> float:
        """Update using hybrid approach."""
        self.history.append(loss_value)
        
        # Track decision
        should_adapt = loss_value > self.threshold.item()
        self.decisions.append(1 if should_adapt else 0)
        
        # Adjust percentile based on rate error
        if len(self.decisions) >= 50:
            current_rate = sum(self.decisions) / len(self.decisions)
            rate_error = self.target_rate - current_rate
            
            # Adjust percentile (higher percentile -> higher threshold -> lower rate)
            self.percentile += self.rate_lr * rate_error * 100
            self.percentile = max(10.0, min(90.0, self.percentile))
        
        # EMA update with current percentile
        recent_losses = list(self.history)[-100:]
        if len(recent_losses) > 0:
            losses_tensor = torch.tensor(recent_losses)
            p_value = torch.quantile(losses_tensor, self.percentile / 100.0)
            
            new_threshold = self.beta * self.threshold + (1 - self.beta) * p_value
            self.threshold.copy_(new_threshold)
        
        return self.threshold.item()
    
    def get_stats(self) -> dict:
        """Get detailed statistics."""
        stats = super().get_stats()
        stats["current_percentile"] = self.percentile
        if len(self.decisions) > 0:
            stats["current_rate"] = sum(self.decisions) / len(self.decisions)
        return stats


class GatingController:
    """
    High-level controller for dynamic computation gating.
    
    Combines reconstruction loss computation with threshold calibration
to make adaptation decisions.
    """
    
    def __init__(
        self,
        threshold_calibrator: DynamicThreshold,
        reconstruction_computer: Optional[nn.Module] = None,
        min_adaptation_steps: int = 1,
        max_adaptation_steps: int = 32
    ):
        self.threshold_calibrator = threshold_calibrator
        self.reconstruction_computer = reconstruction_computer
        self.min_steps = min_adaptation_steps
        self.max_steps = max_adaptation_steps
        
        self.decision_history = []
    
    def decide(
        self,
        reconstruction_loss: float,
        input_features: Optional[torch.Tensor] = None
    ) -> Tuple[bool, int, float]:
        """
        Make adaptation decision.
        
        Args:
            reconstruction_loss: Current reconstruction loss
            input_features: Optional input features for additional context
        
        Returns:
            should_adapt: Whether to trigger adaptation
            num_steps: Number of adaptation steps to perform
            threshold: Current threshold value
        """
        # Update threshold
        threshold = self.threshold_calibrator.update(reconstruction_loss)
        
        # Make decision
        should_adapt = self.threshold_calibrator.should_adapt(reconstruction_loss)
        
        # Determine number of steps (can be based on loss magnitude)
        if should_adapt:
            # Scale steps based on how far above threshold
            excess = reconstruction_loss / max(threshold, 0.01) - 1.0
            num_steps = min(
                self.max_steps,
                max(self.min_steps, int(self.min_steps * (1 + excess)))
            )
        else:
            num_steps = 0
        
        # Log decision
        self.decision_history.append({
            'loss': reconstruction_loss,
            'threshold': threshold,
            'adapt': should_adapt,
            'steps': num_steps
        })
        
        return should_adapt, num_steps, threshold
    
    def get_oracle_recovery(self, oracle_labels: list) -> float:
        """
        Compute oracle recovery rate (for evaluation).
        
        Args:
            oracle_labels: List of boolean indicating true need for adaptation
        
        Returns:
            Fraction of decisions matching oracle
        """
        if len(self.decision_history) == 0 or len(oracle_labels) == 0:
            return 0.0
        
        n = min(len(self.decision_history), len(oracle_labels))
        matches = sum(
            1 for i in range(n)
            if self.decision_history[i]['adapt'] == oracle_labels[i]
        )
        
        return matches / n
