"""
RaBitQ distance / inner-product estimators aligned with VectorDB-NTU/RaBitQ-Library.

Reference: https://github.com/VectorDB-NTU/RaBitQ-Library

This module provides Python equivalents of the C++ estimator functions:
- full_est_dist      (Format 2: full single-vector estimation)
- split_single_estdist / split_single_fulldist  (Format 4: incremental single)
- split_batch_estdist / split_distance_boosting (Format 5: incremental batch)

All formulas follow the official derivation exactly:
    est = F_add + G_add + F_rescale * (ip + G_kBxSumq)
    error_bound = F_error * G_error
"""

import math
import torch
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class FullSingleQuery:
    """Query preprocessing for Format-2 full single-vector estimation."""

    rotated_query: torch.Tensor  # [dim] float, already inverse-rotated (P^{-1} q)
    g_add: float
    g_error: float
    k1xsumq: float  # c_1 * S_q  (for 1-bit)
    kbxsumq: float  # c_B * S_q  (for multi-bit)


@dataclass
class SplitSingleQuery:
    """Query preprocessing for Format-4 split single-vector estimation."""

    rotated_query: torch.Tensor
    query_bin: torch.Tensor  # binary code of rotated_query
    delta: float
    vl: float
    g_add: float
    g_error: float
    k1xsumq: float
    kbxsumq: float


@dataclass
class SplitBatchQuery:
    """Query preprocessing for Format-5 split batch estimation (FastScan LUT)."""

    rotated_query: torch.Tensor
    lut: torch.Tensor  # lookup table for FastScan
    delta_lut: float
    sum_vl_lut: float
    g_add: float
    g_error: float
    k1xsumq: float
    kbxsumq: float


def _mask_ip_x0_q(query: torch.Tensor, bin_code: torch.Tensor) -> float:
    """Inner product between float query and compact binary code."""
    # bin_code is uint8 {0,1}
    return float((query * (bin_code.float() - 0.5) * 2.0).sum().item())


def preprocess_query(
    query: torch.Tensor, centroid: torch.Tensor, metric_type: str = "ip"
) -> Tuple[float, float]:
    """
    Compute G_add and G_error for a query against a centroid.

    Args:
        query: [dim] float tensor (in rotated space)
        centroid: [dim] float tensor
        metric_type: 'l2' or 'ip'

    Returns:
        (g_add, g_error)
    """
    diff = query - centroid
    norm_diff = diff.norm().item()
    if metric_type == "l2":
        g_add = norm_diff * norm_diff
    elif metric_type == "ip":
        g_add = -float((query * centroid).sum().item())
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")
    g_error = norm_diff
    return g_add, g_error


def make_full_single_query(
    rotated_query: torch.Tensor,
    centroid: torch.Tensor,
    metric_type: str = "ip",
    total_bits: int = 1,
) -> FullSingleQuery:
    """Create a FullSingleQuery object."""
    g_add, g_error = preprocess_query(rotated_query, centroid, metric_type)
    sq = float(rotated_query.sum().item())
    k1xsumq = -0.5 * sq
    c_B = -((1 << total_bits) - 1) / 2.0
    kbxsumq = c_B * sq
    return FullSingleQuery(
        rotated_query=rotated_query,
        g_add=g_add,
        g_error=g_error,
        k1xsumq=k1xsumq,
        kbxsumq=kbxsumq,
    )


def full_est_dist(
    ip: float, f_add: float, f_rescale: float, query: FullSingleQuery
) -> Tuple[float, float]:
    """
    Full-bit distance estimation for a single vector (Format 2).

    Returns:
        (est_dist, error_bound)
    """
    est_dist = f_add + query.g_add + f_rescale * (ip + query.kbxsumq)
    error_bound = 0.0  # Full-bit has no F_error by default in this interface
    return est_dist, error_bound


def split_single_estdist(
    f_add: float, f_rescale: float, f_error: float, bin_code: torch.Tensor, query: SplitSingleQuery
) -> Tuple[float, float, float]:
    """
    1-bit coarse distance estimation for a split single vector (Format 4).

    Returns:
        (ip_x0_qr, est_dist, low_dist)
    """
    # ip_x0_qr = <query, x_u> where x_u is binary code {0,1}
    ip_x0_qr = float((query.rotated_query * (bin_code.float())).sum().item())
    est_dist = f_add + query.g_add + f_rescale * (ip_x0_qr + query.k1xsumq)
    low_dist = est_dist - f_error * query.g_error
    return ip_x0_qr, est_dist, low_dist


def split_single_fulldist(
    f_add_ex: float,
    f_rescale_ex: float,
    f_error: float,
    ex_code: torch.Tensor,
    ex_bits: int,
    ip_x0_qr: float,
    query: FullSingleQuery,
) -> Tuple[float, float]:
    """
    Boost 1-bit estimate to full-bit accuracy using ex-code (Format 4).

    Returns:
        (est_dist, low_dist)
    """
    ip_ex = float((query.rotated_query * ex_code.float()).sum().item())
    est_dist = (
        f_add_ex + query.g_add + f_rescale_ex * ((1 << ex_bits) * ip_x0_qr + ip_ex + query.kbxsumq)
    )
    low_dist = est_dist - (f_error * query.g_error / (1 << ex_bits))
    return est_dist, low_dist


def split_distance_boosting(
    f_add_ex: float,
    f_rescale_ex: float,
    ex_code: torch.Tensor,
    ex_bits: int,
    ip_x0_qr: float,
    query: FullSingleQuery,
) -> float:
    """
    Generic distance boosting with ex-data (works with SplitSingleQuery too).

    Returns:
        est_dist
    """
    ip_ex = float((query.rotated_query * ex_code.float()).sum().item())
    est_dist = (
        f_add_ex + query.g_add + f_rescale_ex * ((1 << ex_bits) * ip_x0_qr + ip_ex + query.kbxsumq)
    )
    return est_dist


# ============================================================================
# Legacy compatibility wrapper
# ============================================================================


def estimate_inner_product(
    query: torch.Tensor, centroid: torch.Tensor, qv, query_bits: int = 8
) -> float:
    """
    Backward-compatible inner-product estimator.

    Uses the corrected RaBitQ formula aligned with the reference C++ library.
    The query_bits argument is kept for API compatibility but is ignored,
    because the reference implementation keeps queries in full precision.
    """
    from .quantizer import QuantizedVector

    assert isinstance(qv, QuantizedVector)

    q = make_full_single_query(query, centroid, metric_type="ip")

    # ip = <query_rotated, x_u> where x_u is the binary code {0,1}
    from .packing import unpack_binary_code

    bin_code = unpack_binary_code(qv.binary_code_packed, qv.dim)
    ip = float((q.rotated_query * bin_code.float()).sum().item())

    est_dist, _ = full_est_dist(ip, qv.f_add, qv.f_rescale, q)
    # full_est_dist returns negative inner product (matching Faiss convention).
    # We negate back to return raw inner product for backward compatibility.
    return -est_dist
