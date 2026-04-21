"""
True RaBitQ quantizer for KV cache compression.

Based on:
- Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
  Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024.
- VectorDB-NTU/RaBitQ-Library (C++)
- lqhl/rabitq-rs (Rust)

Implements:
- 1-bit binary quantization of residuals
- Optional extended-bit refinement (total_bits = 1 + ex_bits)
- Per-vector factor computation for unbiased inner-product estimation
- Fast constant-scaling mode for high-throughput KV cache compression
"""

import math
import heapq
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, List


# Constants from reference implementations
K_TIGHT_START = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81]
K_EPS = 1e-5
K_NENUM = 10.0
K_CONST_EPSILON = 1.9


@dataclass
class QuantizedVector:
    """
    RaBitQ quantized representation of a single vector.

    For batch efficiency, the API stores lists/tensors of these fields,
    but this dataclass defines the conceptual per-vector structure.
    """

    binary_code_packed: torch.Tensor  # uint8, ceil(dim/8)
    ex_code_packed: torch.Tensor  # uint8, variable size
    ex_bits: int
    dim: int
    delta: float
    vl: float
    f_add: float
    f_rescale: float
    f_error: float
    residual_norm: float
    f_add_ex: float
    f_rescale_ex: float


@dataclass
class RabitqConfig:
    """Configuration for RaBitQ quantization."""

    total_bits: int = 1  # 1 means binary-only; 2 means 1+1; 3 means 1+2, etc.
    t_const: Optional[float] = None  # Constant scaling factor for fast quantization
    metric_type: str = "ip"  # 'l2' or 'ip'

    @property
    def ex_bits(self) -> int:
        return max(0, self.total_bits - 1)


def compute_const_scaling_factor(
    dim: int, ex_bits: int, seed: int = 42, num_samples: int = 100
) -> float:
    """
    Pre-compute a constant scaling factor t for fast quantization.

    Sampling random Gaussian vectors and averaging their optimal t values
    yields a t_const that is 100-500x faster than per-vector optimization
    with <1% accuracy loss.
    """
    generator = torch.Generator().manual_seed(seed)
    sum_t = 0.0
    count = 0
    for _ in range(num_samples):
        vec = torch.randn(dim, generator=generator)
        norm = vec.norm()
        if norm <= 1e-8:
            continue
        o_abs = (vec / norm).abs().tolist()
        t = _best_rescale_factor(o_abs, ex_bits)
        sum_t += t
        count += 1
    return float(sum_t / count) if count > 0 else 1.0


def _best_rescale_factor(o_abs: List[float], ex_bits: int) -> float:
    """
    Find optimal rescaling factor t for extended-bit quantization.

    Translated from the heap-based C++ algorithm in rabitq-rs.
    """
    dim = len(o_abs)
    max_o = max(o_abs)
    if max_o <= 1e-12 or ex_bits <= 0:
        return 1.0

    table_idx = min(ex_bits, len(K_TIGHT_START) - 1)
    t_end = (((1 << ex_bits) - 1) + K_NENUM) / max_o
    t_start = t_end * K_TIGHT_START[table_idx]

    cur_o_bar = [0] * dim
    sqr_denominator = dim * 0.25
    numerator = 0.0

    for idx, val in enumerate(o_abs):
        cur = int(t_start * val + K_EPS)
        cur_o_bar[idx] = cur
        sqr_denominator += cur * cur + cur
        numerator += (cur + 0.5) * val

    # Min-heap of (t, idx)
    heap = []
    for idx, val in enumerate(o_abs):
        if val > 0.0:
            next_t = (cur_o_bar[idx] + 1) / val
            heapq.heappush(heap, (next_t, idx))

    max_ip = 0.0
    best_t = t_start

    while heap:
        cur_t, idx = heapq.heappop(heap)
        if cur_t >= t_end:
            continue

        cur_o_bar[idx] += 1
        update = cur_o_bar[idx]
        sqr_denominator += 2.0 * update
        numerator += o_abs[idx]

        cur_ip = numerator / math.sqrt(sqr_denominator)
        if cur_ip > max_ip:
            max_ip = cur_ip
            best_t = cur_t

        if update < (1 << ex_bits) - 1 and o_abs[idx] > 0.0:
            t_next = (update + 1) / o_abs[idx]
            if t_next < t_end:
                heapq.heappush(heap, (t_next, idx))

    return best_t if best_t > 0.0 else t_start


def _quantize_ex_with_inv(
    residual: torch.Tensor, ex_bits: int, t: float
) -> Tuple[torch.Tensor, float]:
    """
    Quantize absolute values of residual into ex_bits extended code.

    Returns:
        ex_code: [dim] uint16 tensor
        ipnorm_inv: inverse of inner product norm factor
    """
    dim = residual.shape[-1]
    normalized_abs = residual.abs()
    norm = normalized_abs.norm()

    if norm <= 1e-8:
        return torch.zeros(dim, dtype=torch.int16, device=residual.device), 1.0

    normalized_abs = normalized_abs / norm
    max_val = (1 << ex_bits) - 1

    code = torch.clamp((t * normalized_abs + K_EPS).long(), 0, max_val).to(torch.int16)

    ipnorm = ((code.float() + 0.5) * normalized_abs).sum().item()
    ipnorm_inv = 1.0 / ipnorm if ipnorm > 0.0 else 1.0

    # Flip codes for negative residuals
    mask = residual < 0
    code = torch.where(mask, (~code) & max_val, code)

    if not math.isfinite(ipnorm_inv):
        ipnorm_inv = 1.0

    return code, float(ipnorm_inv)


def _compute_one_bit_factors(
    residual: torch.Tensor,
    centroid: torch.Tensor,
    binary_code: torch.Tensor,
    metric_type: str = "ip",
) -> Tuple[float, float, float, float]:
    """
    Compute factors for 1-bit RaBitQ under L2 or InnerProduct metric.

    Args:
        residual: residual vector
        centroid: centroid vector
        binary_code: binary quantization code
        metric_type: 'l2' or 'ip'

    Returns:
        (f_add, f_rescale, f_error, l2_norm)
    """
    dim = residual.shape[-1]
    xu_cb = binary_code.float() - 0.5  # {0,1} -> {-0.5, +0.5}

    l2_sqr = residual.pow(2).sum().item()
    l2_norm = math.sqrt(l2_sqr)
    xu_cb_norm_sqr = xu_cb.pow(2).sum().item()
    ip_resi_xucb = (residual * xu_cb).sum().item()
    ip_cent_xucb = (centroid * xu_cb).sum().item()
    dot_residual_centroid = (residual * centroid).sum().item()

    denom = ip_resi_xucb
    if abs(denom) <= 1e-12:
        denom = float("inf")

    tmp_error = 0.0
    if dim > 1:
        ratio = ((l2_sqr * xu_cb_norm_sqr) / (denom * denom)) - 1.0
        if math.isfinite(ratio) and ratio > 0.0:
            tmp_error = l2_norm * K_CONST_EPSILON * math.sqrt(max(0.0, ratio / (dim - 1)))

    if metric_type == "l2":
        f_add = l2_sqr + (2 * l2_sqr * ip_cent_xucb / denom)
        f_rescale = -2 * l2_sqr / denom
        f_error = 2 * tmp_error
    elif metric_type == "ip":
        f_add = 1.0 - dot_residual_centroid + l2_sqr * ip_cent_xucb / denom
        f_rescale = -l2_sqr / denom
        f_error = 1 * tmp_error
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    return f_add, f_rescale, f_error, l2_norm


def _compute_extended_factors(
    residual: torch.Tensor,
    centroid: torch.Tensor,
    binary_code: torch.Tensor,
    ex_code: torch.Tensor,
    ipnorm_inv: float,
    ex_bits: int,
    metric_type: str = "ip",
) -> Tuple[float, float]:
    """
    Compute extended factors for L2 or InnerProduct metric.

    Args:
        metric_type: 'l2' or 'ip'

    Returns:
        (f_add_ex, f_rescale_ex)
    """
    dim = residual.shape[-1]
    cb = -((1 << ex_bits) - 0.5)
    total = ex_code.float() + (binary_code.float() * (1 << ex_bits))
    xu_cb = total + cb

    l2_sqr = residual.pow(2).sum().item()
    l2_norm = math.sqrt(l2_sqr)
    ip_resi_xucb = (residual * xu_cb).sum().item()
    ip_cent_xucb = (centroid * xu_cb).sum().item()
    dot_residual_centroid = (residual * centroid).sum().item()

    safe_denom = ip_resi_xucb if abs(ip_resi_xucb) > 1e-12 else float("inf")

    if metric_type == "l2":
        f_add_ex = l2_sqr + (2 * l2_sqr * ip_cent_xucb / safe_denom)
        f_rescale_ex = -2 * l2_norm * ipnorm_inv
    elif metric_type == "ip":
        f_add_ex = 1.0 - dot_residual_centroid + l2_sqr * ip_cent_xucb / safe_denom
        f_rescale_ex = -l2_norm * ipnorm_inv
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    return f_add_ex, f_rescale_ex


def quantize_vector(
    data: torch.Tensor, centroid: torch.Tensor, config: RabitqConfig
) -> QuantizedVector:
    """
    Quantize a single vector with RaBitQ.

    Args:
        data: [dim] float tensor (in rotated space)
        centroid: [dim] float tensor
        config: RabitqConfig

    Returns:
        QuantizedVector
    """
    assert data.dim() == 1 and centroid.dim() == 1
    dim = data.shape[0]
    ex_bits = config.ex_bits

    residual = data - centroid

    # 1-bit binary code
    binary_code = (residual >= 0).to(torch.uint8)

    # Extended code
    if ex_bits > 0:
        t = (
            config.t_const
            if config.t_const is not None
            else compute_const_scaling_factor(dim, ex_bits)
        )
        ex_code, ipnorm_inv = _quantize_ex_with_inv(residual, ex_bits, t)
    else:
        ex_code = torch.zeros(dim, dtype=torch.int16, device=data.device)
        ipnorm_inv = 1.0

    # Total code for delta/vl computation
    cb = -((1 << ex_bits) - 0.5)
    total_code = ex_code.float() + (binary_code.float() * (1 << ex_bits))
    quantized_shifted = total_code + cb

    norm_quan_sqr = quantized_shifted.pow(2).sum().item()
    dot_residual_quant = (residual * quantized_shifted).sum().item()
    norm_residual = residual.norm().item()
    norm_quant = math.sqrt(norm_quan_sqr) if norm_quan_sqr > 0 else 0.0

    denom = norm_residual * norm_quant
    cos_similarity = (dot_residual_quant / denom) if denom > 1e-12 else 0.0
    cos_similarity = max(-1.0, min(1.0, cos_similarity))

    delta = (norm_residual / norm_quant) * cos_similarity if norm_quant > 1e-12 else 0.0
    vl = delta * cb

    # Factors
    f_add, f_rescale, f_error, residual_norm = _compute_one_bit_factors(
        residual, centroid, binary_code, config.metric_type
    )

    f_add_ex = 0.0
    f_rescale_ex = 0.0
    if ex_bits > 0:
        f_add_ex, f_rescale_ex = _compute_extended_factors(
            residual, centroid, binary_code, ex_code, ipnorm_inv, ex_bits, config.metric_type
        )

    # Pack codes
    from .packing import pack_binary_code, pack_ex_code_cpp_compat

    binary_packed = pack_binary_code(binary_code)
    ex_packed = (
        pack_ex_code_cpp_compat(ex_code.unsqueeze(0), ex_bits).squeeze(0)
        if ex_bits > 0
        else torch.empty(0, dtype=torch.uint8, device=data.device)
    )

    return QuantizedVector(
        binary_code_packed=binary_packed,
        ex_code_packed=ex_packed,
        ex_bits=ex_bits,
        dim=dim,
        delta=delta,
        vl=vl,
        f_add=f_add,
        f_rescale=f_rescale,
        f_error=f_error,
        residual_norm=residual_norm,
        f_add_ex=f_add_ex,
        f_rescale_ex=f_rescale_ex,
    )


def quantize_scalar(
    data: torch.Tensor, bits: int, config: Optional[RabitqConfig] = None
) -> Tuple[torch.Tensor, float, float]:
    """
    Format 1: Scalar quantization as a drop-in replacement.

    Quantizes a vector uniformly into `bits` per dimension.
    This matches the RaBitQ-Library `quantize_scalar` API.

    Args:
        data: [dim] float tensor
        bits: number of bits per dimension
        config: optional config (not used for scalar quant, kept for API consistency)

    Returns:
        code: [dim] uint32 tensor of quantized codes
        delta: rescaling factor
        vl: lower value offset
    """
    dim = data.shape[0]
    vl = data.min().item()
    vr = data.max().item()
    max_val = (1 << bits) - 1
    if max_val == 0:
        delta = 1.0
    else:
        delta = (vr - vl) / max_val
    if delta < 1e-12:
        delta = 1.0

    code = torch.clamp(((data - vl) / delta).long(), 0, max_val).to(torch.int32)
    return code, float(delta), float(vl)


def dequantize_scalar(code: torch.Tensor, delta: float, vl: float) -> torch.Tensor:
    """Dequantize a scalar-quantized vector."""
    return code.float() * delta + vl


def reconstruct_vector(centroid: torch.Tensor, qv: QuantizedVector) -> torch.Tensor:
    """
    Reconstruct a vector from its RaBitQ quantized form.

    Args:
        centroid: [dim] float tensor
        qv: QuantizedVector

    Returns:
        reconstructed: [dim] float tensor
    """
    from .packing import unpack_binary_code, unpack_ex_code_cpp_compat

    binary_code = unpack_binary_code(qv.binary_code_packed, qv.dim)
    if qv.ex_bits > 0:
        ex_code = unpack_ex_code_cpp_compat(
            qv.ex_code_packed.unsqueeze(0), qv.dim, qv.ex_bits
        ).squeeze(0)
    else:
        ex_code = torch.zeros(qv.dim, dtype=torch.int16, device=centroid.device)

    total_code = ex_code.float() + (binary_code.float() * (1 << qv.ex_bits))
    return centroid + qv.delta * total_code + qv.vl
