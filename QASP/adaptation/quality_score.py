"""Spectral information-quality score ``rho(t)`` (QASP Sec.~3.2, ``eq:quality-score``).

This module implements the **per-token** spectral score: one FFT along the hidden
dimension per token position. The paper also discusses sliding-window amortization
(``sec:sliding-window`` in ``QASP_paper.tex``) for complexity; that batched path is
not implemented here and is optional for mathematical equivalence.

Implements the paper's definition:

    s(t) = || F^{-1} ( F(h_t) ⊙ g_LP ) ||_2 / || h_t ||_2,
    rho(t) = 1 - s(t),

with a Gaussian low-pass filter in the frequency domain:

    g_{LP, k} = exp( -k^2 / (2 f_c^2) ),   f_c = low_pass_ratio * d.

``rho(t) ≈ 1`` marks information-rich (high-frequency) representations while
``rho(t) ≈ 0`` marks semantically stable (low-frequency) tokens such as stop
words. The DFT is applied along the last (channel) axis of ``signal`` so the
function returns a score per token, matching how ``rho(t)`` is consumed by the
value-weighted AttnRes and matrix QASP update.
"""

from __future__ import annotations

from typing import cast

import torch


def compute_quality_score(
    signal: torch.Tensor,
    low_pass_ratio: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ``rho(t) = 1 - s(t)`` along the last dimension of ``signal``.

    Args:
        signal: Tensor of shape ``[..., d]`` holding hidden representations.
            Any leading batch / sequence dimensions are preserved in the output.
        low_pass_ratio: Gaussian cut-off ``f_c / d``. The QASP paper (Table 2)
            recommends ``f_c = d / 4``, corresponding to ``low_pass_ratio=0.25``.
        eps: Numerical guard on the denominator ``||h_t||_2``.

    Returns:
        Tensor of shape ``signal.shape[:-1]`` with entries in ``[0, 1]``.
    """

    if low_pass_ratio <= 0.0 or low_pass_ratio > 1.0:
        raise ValueError("`low_pass_ratio` must be in (0, 1].")
    if signal.ndim < 1:
        raise ValueError("`signal` must have at least one dimension.")

    d = signal.shape[-1]
    if d < 2:
        raise ValueError("feature dimension must be >= 2.")

    fft_vals = torch.fft.rfft(signal, dim=-1)
    n_bins = fft_vals.shape[-1]

    fc = float(low_pass_ratio) * float(d)
    freq_idx = torch.arange(n_bins, device=signal.device, dtype=torch.float32)
    gaussian_lpf = torch.exp(-(freq_idx * freq_idx) / (2.0 * fc * fc))
    gaussian_lpf = gaussian_lpf.to(signal.dtype)

    filtered = torch.fft.irfft(fft_vals * gaussian_lpf, n=d, dim=-1)

    signal_norm = signal.norm(dim=-1).clamp_min(eps)
    filtered_norm = filtered.norm(dim=-1)
    stability = (filtered_norm / signal_norm).clamp(0.0, 1.0)

    return cast(torch.Tensor, (1.0 - stability).clamp(0.0, 1.0))
