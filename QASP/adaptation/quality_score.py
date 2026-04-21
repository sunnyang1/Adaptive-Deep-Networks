"""Spectral information-quality score ``rho(t)`` (QASP Sec.~3.2, ``eq:quality-score``).

This module implements the spectral score with one FFT along the hidden dimension
per token. Optional **sliding-window batching** (``sec:sliding-window`` in
``QASP_paper.tex``) splits the sequence into contiguous chunks of ``W`` tokens and
runs the same vectorized FFT on each chunk; this matches the per-token definition
and is equivalent to a single ``rfft`` over the full sequence (see
:func:`compute_quality_score` ``window_size``).

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


def _compute_quality_score_fft(
    signal: torch.Tensor,
    low_pass_ratio: float,
    eps: float,
) -> torch.Tensor:
    """Single-pass ``rho`` on ``signal`` (FFT along channel dim ``d``)."""

    fft_vals = torch.fft.rfft(signal, dim=-1)
    n_bins = fft_vals.shape[-1]
    d = signal.shape[-1]

    fc = float(low_pass_ratio) * float(d)
    freq_idx = torch.arange(n_bins, device=signal.device, dtype=torch.float32)
    gaussian_lpf = torch.exp(-(freq_idx * freq_idx) / (2.0 * fc * fc))
    gaussian_lpf = gaussian_lpf.to(signal.dtype)

    filtered = torch.fft.irfft(fft_vals * gaussian_lpf, n=d, dim=-1)

    signal_norm = signal.norm(dim=-1).clamp_min(eps)
    filtered_norm = filtered.norm(dim=-1)
    stability = (filtered_norm / signal_norm).clamp(0.0, 1.0)

    return cast(torch.Tensor, (1.0 - stability).clamp(0.0, 1.0))


def compute_quality_score(
    signal: torch.Tensor,
    low_pass_ratio: float = 0.25,
    eps: float = 1e-8,
    *,
    window_size: int | None = None,
) -> torch.Tensor:
    """Compute ``rho(t) = 1 - s(t)`` along the last dimension of ``signal``.

    Args:
        signal: Tensor of shape ``[..., d]`` holding hidden representations.
            Any leading batch / sequence dimensions are preserved in the output.
        low_pass_ratio: Gaussian cut-off ``f_c / d``. The QASP paper (Table 2)
            recommends ``f_c = d / 4``, corresponding to ``low_pass_ratio=0.25``.
        eps: Numerical guard on the denominator ``||h_t||_2``.
        window_size: If ``None`` (default), compute ``rho`` in one kernel over the
            full tensor. If a positive integer ``W`` (e.g. 512 from the paper),
            split the **sequence** axis (dimension ``-2``) into contiguous chunks
            of at most ``W`` tokens and evaluate each chunk with the same FFT
            pipeline. Results match the single-pass path up to floating-point
            non-associativity; ``W >=`` sequence length falls back to one chunk.
            Ignored when ``signal.ndim < 2`` (no sequence axis).

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

    if window_size is not None:
        if window_size < 1:
            raise ValueError("`window_size` must be >= 1 when provided.")
        if window_size == 1:
            window_size = None

    if signal.ndim < 2 or window_size is None:
        return _compute_quality_score_fft(signal, low_pass_ratio, eps)

    seq_len = signal.shape[-2]
    if seq_len <= window_size:
        return _compute_quality_score_fft(signal, low_pass_ratio, eps)

    pieces: list[torch.Tensor] = []
    for start in range(0, seq_len, window_size):
        end = min(start + window_size, seq_len)
        slc = signal[..., start:end, :]
        pieces.append(_compute_quality_score_fft(slc, low_pass_ratio, eps))

    return torch.cat(pieces, dim=-1)
