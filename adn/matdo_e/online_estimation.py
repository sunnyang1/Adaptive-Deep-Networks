from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from adn.matdo_e.config import MATDOConfig


@dataclass(frozen=True)
class OnlineEstimate:
    """Online estimates for the coupling terms in the paper error model."""

    delta: float
    epsilon: float
    forgetting_factor: float = 0.95
    sample_count: int = 0

    def apply(self, config: MATDOConfig) -> MATDOConfig:
        """Inject the latest estimates into a config snapshot."""
        return replace(config, delta=self.delta, epsilon=self.epsilon)


@dataclass
class OnlineRLSEstimator:
    """Appendix C: RLS for (delta, epsilon) with forgetting factor lambda.

    Regress y ~ delta * x0 + epsilon * x1 with
    ``x0 = 2^(-2R) / M``, ``x1 = ln(M) / T`` (paper feature map).
    """

    lambda_: float = 0.98
    theta: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    P: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=float) * 100.0)
    n: int = 0

    def update(self, x: np.ndarray, y: float) -> OnlineRLSEstimator:
        x = np.asarray(x, dtype=float).reshape(2)
        denom = self.lambda_ + float(x @ self.P @ x)
        k = self.P @ x / denom
        err = y - float(x @ self.theta)
        self.theta = self.theta + k * err
        self.P = (self.P - np.outer(k, x @ self.P)) / self.lambda_
        self.n += 1
        return self

    def to_online_estimate(self) -> OnlineEstimate:
        d = float(self.theta[0])
        e = float(self.theta[1])
        return OnlineEstimate(
            delta=max(0.0, d),
            epsilon=max(0.0, e),
            forgetting_factor=self.lambda_,
            sample_count=int(self.n),
        )
