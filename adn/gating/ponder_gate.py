"""
Ponder Gate for Conditional qTTT Execution

Implements uncertainty-based triggering for query adaptation.
Based on: Section 3.4 of Adaptive Deep Networks paper

The Ponder Gate triggers qTTT only when query uncertainty is high,
reducing unnecessary computation.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing import Union


@dataclass
class PonderGateConfig:
    """Configuration for PonderGate."""

    entropy_threshold: float = 2.0
    min_prob_threshold: float = 0.3
    mode: str = "balanced"

    def create_gate(self) -> "PonderGate":
        """Create a PonderGate from this config."""
        return PonderGate(
            entropy_threshold=self.entropy_threshold,
            min_prob_threshold=self.min_prob_threshold,
        )


class PonderGate:
    """
    Gate that decides whether to trigger qTTT based on output uncertainty.

    Uses two heuristics:
    1. Entropy: High entropy = uncertain distribution
    2. Max probability: Low max prob = no clear winner

    Either condition triggers adaptation.

    Args:
        entropy_threshold: Entropy threshold (default: 2.0)
            Higher = more tolerant of uncertainty
        min_prob_threshold: Minimum max probability (default: 0.3)
            Lower = more tolerant of uncertainty

    Example:
        >>> gate = PonderGate(entropy_threshold=2.0, min_prob_threshold=0.3)
        >>> logits = model(input_ids)
        >>> if gate.should_adapt(logits[:, -1, :]):
        ...     adapted_query = qttt.adapt(query, kv_cache)
    """

    def __init__(self, entropy_threshold: float = 2.0, min_prob_threshold: float = 0.3):
        self.entropy_threshold = entropy_threshold
        self.min_prob_threshold = min_prob_threshold

    def should_adapt(self, logits: torch.Tensor) -> Union[bool, torch.Tensor]:
        """
        Decide whether to trigger qTTT adaptation.

        Args:
            logits: Model output logits [..., vocab_size]

        Returns:
            bool or tensor of bools: True if adaptation should be triggered
        """
        # Use last sequence position when given full LM logits [B, T, V]
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        # Compute metrics
        entropy = self.compute_entropy(logits)
        max_prob = self.compute_max_probability(logits)

        # Trigger if either:
        # 1. Entropy is high (uncertain)
        # 2. Max probability is low (no clear winner)
        high_entropy = entropy > self.entropy_threshold
        low_confidence = max_prob < self.min_prob_threshold

        should_trigger = high_entropy | low_confidence

        # Single-row distributions [1, V] should return a Python bool (common in tests / B=1 gen)
        if should_trigger.shape[0] == 1:
            return bool(should_trigger.squeeze().item())
        return should_trigger

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy of output distribution.

        H = -sum(p * log(p))

        Args:
            logits: Model output logits [..., vocab_size]

        Returns:
            Entropy value [..., 1]
        """
        # Stabilize softmax (critical for sharp peaks, e.g. one-hot-like logits)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        return entropy

    def compute_max_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute maximum probability in output distribution.

        Args:
            logits: Model output logits [..., vocab_size]

        Returns:
            Max probability [..., 1]
        """
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        return max_prob

    def get_uncertainty_metrics(self, logits: torch.Tensor) -> dict:
        """
        Get detailed uncertainty metrics.

        Args:
            logits: Model output logits

        Returns:
            Dictionary with entropy, max_prob, and decision
        """
        entropy = self.compute_entropy(logits)
        max_prob = self.compute_max_probability(logits)
        should_trigger = self.should_adapt(logits)

        return {
            "entropy": entropy.item() if entropy.numel() == 1 else entropy,
            "max_probability": max_prob.item() if max_prob.numel() == 1 else max_prob,
            "entropy_threshold": self.entropy_threshold,
            "min_prob_threshold": self.min_prob_threshold,
            "should_adapt": should_trigger,
        }


def create_ponder_gate(mode: str = "balanced") -> PonderGate:
    """
    Factory function to create PonderGate with preset configurations.

    Args:
        mode: One of 'strict', 'balanced', 'lenient'
            - strict: Trigger easily (high quality, more compute)
            - balanced: Moderate triggering (~30% trigger rate)
            - lenient: Rarely trigger (fast, may miss edge cases)

    Returns:
        Configured PonderGate

    Example:
        >>> gate = create_ponder_gate('strict')
        >>> # Use for high-stakes scenarios
    """
    presets = {
        "strict": {"entropy_threshold": 1.0, "min_prob_threshold": 0.5},
        "balanced": {"entropy_threshold": 2.0, "min_prob_threshold": 0.3},
        "lenient": {"entropy_threshold": 3.0, "min_prob_threshold": 0.2},
    }

    if mode not in presets:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(presets.keys())}")

    return PonderGate(**presets[mode])


def calibrate_ponder_gate(
    model,
    val_dataloader,
    target_trigger_rate: float = 0.30,
    tolerance: float = 0.05,
    max_iterations: int = 10,
) -> PonderGate:
    """
    Calibrate Ponder Gate thresholds on held-out validation set.

    As described in §3.3.4, thresholds should be "calibrated on a held-out
    validation set to achieve ~30% trigger rate while maintaining accuracy."

    Args:
        model: Model to evaluate
        val_dataloader: Validation data loader
        target_trigger_rate: Target trigger rate (default 0.30)
        tolerance: Acceptable deviation from target
        max_iterations: Maximum calibration iterations

    Returns:
        Calibrated PonderGate
    """
    import torch

    device = next(model.parameters()).device

    # Binary search for optimal thresholds
    entropy_low, entropy_high = 0.5, 4.0
    prob_low, prob_high = 0.1, 0.6

    for iteration in range(max_iterations):
        entropy_mid = (entropy_low + entropy_high) / 2
        prob_mid = (prob_low + prob_high) / 2

        gate = PonderGate(entropy_threshold=entropy_mid, min_prob_threshold=prob_mid)

        # Evaluate trigger rate on validation set
        trigger_count = 0
        total_count = 0

        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                logits = model(input_ids)

                # Check last token
                should_trigger = gate.should_adapt(logits[:, -1, :])
                if isinstance(should_trigger, bool):
                    trigger_count += int(should_trigger)
                else:
                    trigger_count += should_trigger.sum().item()
                total_count += input_ids.size(0)

        trigger_rate = trigger_count / total_count if total_count > 0 else 0

        # Check convergence
        if abs(trigger_rate - target_trigger_rate) < tolerance:
            print(f"[Calibration] Converged at iteration {iteration + 1}")
            print(f"  Entropy threshold: {entropy_mid:.2f}")
            print(f"  Min prob threshold: {prob_mid:.2f}")
            print(f"  Trigger rate: {trigger_rate:.2%}")
            return gate

        # Adjust thresholds
        if trigger_rate > target_trigger_rate:
            # Triggering too often, raise thresholds
            entropy_low = entropy_mid
            prob_low = prob_mid
        else:
            # Triggering too rarely, lower thresholds
            entropy_high = entropy_mid
            prob_high = prob_mid

    # Return best found
    print(f"[Calibration] Max iterations reached. Best:")
    print(f"  Entropy threshold: {entropy_mid:.2f}")
    print(f"  Min prob threshold: {prob_mid:.2f}")
    print(f"  Trigger rate: {trigger_rate:.2%}")
    return PonderGate(entropy_threshold=entropy_mid, min_prob_threshold=prob_mid)
