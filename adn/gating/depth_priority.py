"""
Depth-Priority Gating Controller

Based on: Section 4.2 of Adaptive Deep Networks RaBitQ version

When RaBitQ acceleration is enabled:
- Depth (qTTT) becomes 8× cheaper than standard precision
- Policy strictly prioritizes depth over width (thinking tokens)
- This transforms the FLOP equivalence to decisively favor depth
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from collections import deque

from adn.gating.threshold import DynamicThreshold, EMAThreshold, TargetRateThreshold


class DepthPriorityGatingController:
    """
    Gating controller with strict depth-priority policy.

    Under TurboQuant acceleration:
    - C_qTTT^Turbo ≈ (1/8) * C_qTTT^Standard
    - T_think ≈ 16 * N_qTTT * k (vs 2 * N_qTTT * k without TurboQuant)
    - Policy: When d_t = 1, allocate 100% to depth
    """

    def __init__(
        self,
        threshold_calibrator: DynamicThreshold,
        max_qttt_steps: int = 32,
        min_qttt_steps: int = 4,
        think_tokens_if_forced: int = 0,  # Set to 0 for strict depth priority
        rabitq_enabled: bool = True,
        reconstruction_computer: Optional[nn.Module] = None,
    ):
        """
        Args:
            threshold_calibrator: Threshold calibration strategy
            max_qttt_steps: Maximum qTTT adaptation steps
            min_qttt_steps: Minimum qTTT steps when adapting
            think_tokens_if_forced: Thinking tokens budget (should be 0)
            rabitq_enabled: Whether RaBitQ acceleration is active
            reconstruction_computer: Module for computing reconstruction loss
        """
        self.threshold_calibrator = threshold_calibrator
        self.reconstruction_computer = reconstruction_computer

        self.max_qttt_steps = max_qttt_steps
        self.min_qttt_steps = min_qttt_steps
        self.think_tokens = think_tokens_if_forced
        self.rabitq_enabled = rabitq_enabled

        # Cost ratio: depth vs width
        if rabitq_enabled:
            # With RaBitQ: 8× arithmetic + 4× memory = 8× total
            self.depth_cost_factor = 1 / 8
            self.flop_equivalence_multiplier = 16  # T_think = 16 * N_qTTT * k
        else:
            self.depth_cost_factor = 1.0
            self.flop_equivalence_multiplier = 2

        # Decision history
        self.decision_history = []
        self.adaptation_stats = {
            "total_decisions": 0,
            "adaptation_triggers": 0,
            "total_qttt_steps": 0,
            "total_think_tokens": 0,
            "avg_qttt_steps_per_adaptation": 0.0,
        }

    def decide(
        self, reconstruction_loss: float, input_complexity: Optional[float] = None
    ) -> Tuple[bool, int, int, float]:
        """
        Make depth-priority adaptation decision.

        Args:
            reconstruction_loss: Current reconstruction loss value
            input_complexity: Optional additional complexity signal

        Returns:
            should_adapt: Whether to trigger adaptation
            num_qttt_steps: Number of qTTT steps (depth)
            num_think_tokens: Number of thinking tokens (width)
            threshold: Current threshold value
        """
        # Update threshold
        threshold = self.threshold_calibrator.update(reconstruction_loss)

        # Binary gating decision
        should_adapt = self.threshold_calibrator.should_adapt(reconstruction_loss)

        # Update statistics
        self.adaptation_stats["total_decisions"] += 1

        if should_adapt:
            self.adaptation_stats["adaptation_triggers"] += 1

            # Scale steps based on loss magnitude
            excess = reconstruction_loss / max(threshold, 0.01) - 1.0
            num_steps = min(
                self.max_qttt_steps,
                max(self.min_qttt_steps, int(self.min_qttt_steps * (1 + excess * 2))),
            )

            # Strict depth priority: minimize thinking tokens
            if self.rabitq_enabled:
                think_tokens = 0  # All budget to depth
            else:
                think_tokens = self.think_tokens

            self.adaptation_stats["total_qttt_steps"] += num_steps
            self.adaptation_stats["total_think_tokens"] += think_tokens

            # Update average
            n = self.adaptation_stats["adaptation_triggers"]
            self.adaptation_stats["avg_qttt_steps_per_adaptation"] = (
                self.adaptation_stats["total_qttt_steps"] / n
            )
        else:
            num_steps = 0
            think_tokens = 0

        # Log decision
        self.decision_history.append(
            {
                "loss": reconstruction_loss,
                "threshold": threshold,
                "adapt": should_adapt,
                "qttt_steps": num_steps,
                "think_tokens": think_tokens,
                "rabitq_enabled": self.rabitq_enabled,
            }
        )

        return should_adapt, num_steps, think_tokens, threshold

    def get_allocation_report(self) -> Dict:
        """
        Generate report on computation allocation.

        Returns:
            Dictionary with allocation statistics
        """
        total = self.adaptation_stats["total_decisions"]
        triggers = self.adaptation_stats["adaptation_triggers"]

        if triggers == 0:
            return {
                "adaptation_rate": 0.0,
                "avg_qttt_steps": 0.0,
                "avg_think_tokens": 0.0,
                "depth_priority_ratio": 0.0,
            }

        qttt_total = self.adaptation_stats["total_qttt_steps"]
        think_total = self.adaptation_stats["total_think_tokens"]

        # Effective FLOPs (normalized)
        qttt_flops = qttt_total * self.depth_cost_factor
        think_flops = think_total

        total_flops = qttt_flops + think_flops
        depth_ratio = qttt_flops / total_flops if total_flops > 0 else 0

        return {
            "adaptation_rate": triggers / total,
            "total_decisions": total,
            "total_adaptations": triggers,
            "avg_qttt_steps": qttt_total / triggers,
            "avg_think_tokens": think_total / triggers,
            "depth_priority_ratio": depth_ratio,
            "rabitq_savings": 8.0 if self.rabitq_enabled else 1.0,
            "flop_equivalence_multiplier": self.flop_equivalence_multiplier,
        }

    def get_policy_comparison(self) -> Dict:
        """
        Compare depth-priority vs width-priority policies.

        Shows why TurboQuant makes depth-priority optimal.
        """
        # Standard policy (without TurboQuant)
        standard_equiv = 2  # T_think = 2 * N_qTTT * k

        # TurboQuant policy
        turbo_equiv = self.flop_equivalence_multiplier

        # Cost for equivalent computation
        N = 16  # Example: 16 qTTT steps with k=128
        k = 128

        standard_think = standard_equiv * N * k
        turbo_think = turbo_equiv * N * k

        return {
            "standard_policy": {
                "equivalence": f"T_think = {standard_equiv} * N_qTTT * k",
                "example_16_steps_128_span": f"{standard_think} tokens",
            },
            "rabitq_policy": {
                "equivalence": f"T_think = {turbo_equiv} * N_qTTT * k",
                "example_16_steps_128_span": f"{turbo_think} tokens",
                "savings_vs_standard": f"{turbo_equiv / standard_equiv:.1f}x",
            },
            "recommendation": "Strict depth priority under RaBitQ",
        }


class AdaptiveThresholdWithDepthPriority(EMAThreshold):
    """
    EMA threshold with automatic depth-priority adjustment.

    When adaptation rate exceeds target, automatically shifts
    more budget to depth (which is cheaper under RaBitQ).
    """

    def __init__(
        self,
        initial_threshold: float = 2.0,
        beta: float = 0.99,
        target_rate: float = 0.3,
        rate_tolerance: float = 0.05,
        max_depth_steps: int = 32,
    ):
        super().__init__(initial_threshold, beta, percentile=70.0)
        self.target_rate = target_rate
        self.rate_tolerance = rate_tolerance
        self.max_depth_steps = max_depth_steps

        self.recent_decisions = deque(maxlen=100)
        self.recommended_depth_steps = max_depth_steps // 2

    def update(self, loss_value: float) -> float:
        """Update threshold and adjust depth recommendations."""
        threshold = super().update(loss_value)

        # Track decision
        should_adapt = self.should_adapt(loss_value)
        self.recent_decisions.append(1 if should_adapt else 0)

        # Adjust depth recommendation based on rate
        if len(self.recent_decisions) >= 50:
            current_rate = sum(self.recent_decisions) / len(self.recent_decisions)

            if current_rate > self.target_rate + self.rate_tolerance:
                # Adaptation rate too high - reduce steps to save budget
                self.recommended_depth_steps = max(4, self.recommended_depth_steps - 2)
            elif current_rate < self.target_rate - self.rate_tolerance:
                # Adaptation rate too low - can afford more steps
                self.recommended_depth_steps = min(
                    self.max_depth_steps, self.recommended_depth_steps + 2
                )

        return threshold

    def get_recommended_depth_steps(self) -> int:
        """Get dynamically adjusted depth steps recommendation."""
        return self.recommended_depth_steps


def create_depth_priority_controller(
    target_rate: float = 0.3,
    max_qttt_steps: int = 32,
    rabitq_enabled: bool = True,
    use_adaptive_threshold: bool = True,
) -> DepthPriorityGatingController:
    """
    Factory for creating depth-priority gating controller.

    Args:
        target_rate: Target adaptation rate (ρ_target)
        max_qttt_steps: Maximum qTTT steps per adaptation
        rabitq_enabled: Whether RaBitQ is active
        use_adaptive_threshold: Use adaptive threshold with depth adjustment

    Returns:
        Configured DepthPriorityGatingController
    """
    if use_adaptive_threshold:
        threshold = AdaptiveThresholdWithDepthPriority(
            target_rate=target_rate, max_depth_steps=max_qttt_steps
        )
    else:
        threshold = TargetRateThreshold(target_rate=target_rate)

    return DepthPriorityGatingController(
        threshold_calibrator=threshold, max_qttt_steps=max_qttt_steps, rabitq_enabled=rabitq_enabled
    )


# Alias for public API
DepthPriorityGate = DepthPriorityGatingController
