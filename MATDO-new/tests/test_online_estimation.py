from __future__ import annotations

import numpy as np

from matdo_new.core.config import MATDOConfig
from matdo_new.core.online_estimation import OnlineRLSEstimator
from matdo_new.core.policy import RuntimeObservation, solve_policy


def test_online_rls_recovers_coefficients_on_synthetic_data() -> None:
    rng = np.random.default_rng(0)
    true_d, true_e = 0.005, 0.002
    est = OnlineRLSEstimator(lambda_=0.99)
    for _ in range(500):
        R = int(rng.choice([2, 4, 8]))
        M = int(rng.integers(16, 64))
        T = int(rng.choice([8, 16, 32]))
        m_safe = max(M, 1)
        t_safe = max(T, 1)
        x1 = (2.0 ** (-2 * R)) / m_safe
        x2 = np.log(m_safe) / t_safe
        x = np.array([x1, x2])
        y = true_d * x1 + true_e * x2 + rng.normal(0.0, 1e-5)
        est.update(x, y)

    assert abs(est.theta[0] - true_d) < 5e-3
    assert abs(est.theta[1] - true_e) < 5e-3


def test_online_estimate_apply_and_solve_policy() -> None:
    est = OnlineRLSEstimator(lambda_=0.98)
    est.update(np.array([1e-4, 1e-3]), 0.01)
    oe = est.to_online_estimate()
    cfg = MATDOConfig()
    patched = oe.apply(cfg)
    assert patched.delta == oe.delta
    assert patched.epsilon == oe.epsilon

    dec = solve_policy(
        RuntimeObservation(rho_hbm=0.92, rho_dram=0.3),
        MATDOConfig(),
        oe,
    )
    assert dec.t_steps >= 0
