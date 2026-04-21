from __future__ import annotations

from dataclasses import dataclass

from matdo_new.core.policy import PolicyDecision


@dataclass(frozen=True)
class MaterializedPolicy:
    """Concrete runtime controls handed to the backend."""

    quantization_bits: int
    m_blocks: int
    t_steps: int
    engram_entries: int
    use_engram: bool
    is_arbitrage: bool
    target_error: float
    reason: str


def materialize_policy(
    policy: PolicyDecision | MaterializedPolicy | None,
) -> MaterializedPolicy | None:
    """Normalize policy-like inputs into a backend-facing snapshot."""
    if policy is None:
        return None
    if isinstance(policy, MaterializedPolicy):
        return policy
    return MaterializedPolicy(
        quantization_bits=policy.quantization_bits,
        m_blocks=policy.m_blocks,
        t_steps=policy.t_steps,
        engram_entries=policy.engram_entries,
        use_engram=policy.use_engram,
        is_arbitrage=policy.is_arbitrage,
        target_error=policy.target_error,
        reason=policy.reason,
    )
