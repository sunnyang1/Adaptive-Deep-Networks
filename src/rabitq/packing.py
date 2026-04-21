"""
Bit-packing utilities for RaBitQ binary codes and extended codes.

Aligned with the C++ reference implementation for interoperability.
"""

import torch
from typing import Tuple


def pack_binary_code(binary_code: torch.Tensor) -> torch.Tensor:
    """
    Pack a 0/1 uint8 tensor into compact uint8 bytes.

    Args:
        binary_code: [..., dim] uint8 tensor with values 0 or 1

    Returns:
        packed: [..., ceil(dim/8)] uint8 tensor
    """
    *leading, dim = binary_code.shape
    packed_size = (dim + 7) // 8
    device = binary_code.device

    # Fallback to CPU for bitwise ops on devices that don't support them (e.g., MPS)
    if device.type == "mps":
        binary_code = binary_code.cpu()

    packed = torch.zeros(*leading, packed_size, dtype=torch.uint8)

    for i in range(8):
        bit_plane = binary_code[..., i::8]
        if bit_plane.shape[-1] < packed_size:
            # Pad last chunk if needed
            pad = packed_size - bit_plane.shape[-1]
            bit_plane = torch.nn.functional.pad(bit_plane, (0, pad))
        packed |= bit_plane << i

    return packed.to(device)


def unpack_binary_code(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Unpack uint8 bytes back to 0/1 uint8 tensor.

    Args:
        packed: [..., ceil(dim/8)] uint8 tensor
        dim: Original dimension

    Returns:
        binary_code: [..., dim] uint8 tensor
    """
    *leading, packed_size = packed.shape
    device = packed.device
    if device.type == "mps":
        packed = packed.cpu()

    binary_code = torch.zeros(*leading, dim, dtype=torch.uint8)

    for i in range(dim):
        byte_idx = i // 8
        bit_offset = i % 8
        if byte_idx < packed_size:
            binary_code[..., i] = (packed[..., byte_idx] >> bit_offset) & 1

    return binary_code.to(device)


def pack_ex_code_generic(ex_code: torch.Tensor, ex_bits: int) -> torch.Tensor:
    """
    Generic bit-packing for extended codes (fallback for any ex_bits).

    Args:
        ex_code: [..., dim] uint16 tensor with values in [0, 2^ex_bits - 1]
        ex_bits: Bits per element

    Returns:
        packed: [..., ceil(dim * ex_bits / 8)] uint8 tensor
    """
    *leading, dim = ex_code.shape
    packed_size = (dim * ex_bits + 7) // 8
    packed = torch.zeros(*leading, packed_size, dtype=torch.uint8, device=ex_code.device)

    flat_ex = ex_code.reshape(-1, dim)
    flat_packed = packed.reshape(-1, packed_size)

    for b in range(ex_bits):
        bit_plane = ((flat_ex >> b) & 1).to(torch.uint8)
        for i in range(dim):
            bit_idx = i * ex_bits + b
            byte_idx = bit_idx // 8
            bit_offset = bit_idx % 8
            if byte_idx < packed_size:
                flat_packed[:, byte_idx] |= bit_plane[:, i] << bit_offset

    return flat_packed.reshape(*leading, packed_size)


def unpack_ex_code_generic(packed: torch.Tensor, dim: int, ex_bits: int) -> torch.Tensor:
    """
    Generic unpacking for extended codes.

    Args:
        packed: [..., packed_size] uint8 tensor
        dim: Original dimension
        ex_bits: Bits per element

    Returns:
        ex_code: [..., dim] uint16 tensor
    """
    *leading, packed_size = packed.shape
    ex_code = torch.zeros(*leading, dim, dtype=torch.int16, device=packed.device)

    flat_packed = packed.reshape(-1, packed_size)
    flat_ex = ex_code.reshape(-1, dim)

    for b in range(ex_bits):
        for i in range(dim):
            bit_idx = i * ex_bits + b
            byte_idx = bit_idx // 8
            bit_offset = bit_idx % 8
            if byte_idx < packed_size:
                flat_ex[:, i] |= ((flat_packed[:, byte_idx] >> bit_offset) & 1).to(torch.int16) << b

    return flat_ex.reshape(*leading, dim)


def pack_ex_code_cpp_compat(ex_code: torch.Tensor, ex_bits: int) -> torch.Tensor:
    """
    C++-compatible packing for common ex_bits values (1, 2, 6).
    Falls back to generic packing for other values.

    Args:
        ex_code: [..., dim] uint16 tensor, dim must be multiple of 16
        ex_bits: Extended bits (1, 2, or 6)

    Returns:
        packed: uint8 tensor
    """
    dim = ex_code.shape[-1]
    if ex_bits not in (1, 2, 6):
        return pack_ex_code_generic(ex_code, ex_bits)

    device = ex_code.device
    if device.type == "mps":
        ex_code = ex_code.cpu()

    num_groups = dim // 16
    bytes_per_group = {1: 2, 2: 4, 6: 12}[ex_bits]
    packed_size = num_groups * bytes_per_group

    *leading, _ = ex_code.shape
    packed = torch.zeros(*leading, packed_size, dtype=torch.uint8)

    flat_ex = ex_code.reshape(-1, dim)
    flat_packed = packed.reshape(-1, packed_size)

    for g in range(num_groups):
        group = flat_ex[:, g * 16 : (g + 1) * 16]  # [B, 16]
        out = flat_packed[:, g * bytes_per_group : (g + 1) * bytes_per_group]

        if ex_bits == 1:
            for b in range(2):
                sub = group[:, b * 8 : (b + 1) * 8]  # [B, 8]
                byte_val = torch.zeros(sub.shape[0], dtype=torch.uint8)
                for i in range(8):
                    byte_val |= (sub[:, i] & 1).to(torch.uint8) << i
                out[:, b] = byte_val

        elif ex_bits == 2:
            for b in range(4):
                sub = group[:, b * 4 : (b + 1) * 4]  # [B, 4]
                byte_val = torch.zeros(sub.shape[0], dtype=torch.uint8)
                for i in range(4):
                    byte_val |= (sub[:, i] & 0x3).to(torch.uint8) << (i * 2)
                out[:, b] = byte_val

        elif ex_bits == 6:
            for sub_start in range(4):
                sub = group[:, sub_start * 4 : (sub_start + 1) * 4]  # [B, 4]
                byte0 = ((sub[:, 0] & 0x3F) | ((sub[:, 1] & 0x03) << 6)).to(torch.uint8)
                byte1 = (((sub[:, 1] & 0x3C) >> 2) | ((sub[:, 2] & 0x0F) << 4)).to(torch.uint8)
                byte2 = (((sub[:, 2] & 0x30) >> 4) | ((sub[:, 3] & 0x3F) << 2)).to(torch.uint8)
                out[:, sub_start * 3 + 0] = byte0
                out[:, sub_start * 3 + 1] = byte1
                out[:, sub_start * 3 + 2] = byte2

    return packed.to(device)


def unpack_ex_code_cpp_compat(packed: torch.Tensor, dim: int, ex_bits: int) -> torch.Tensor:
    """Unpack C++-compatible extended codes."""
    if ex_bits not in (1, 2, 6):
        return unpack_ex_code_generic(packed, dim, ex_bits)

    device = packed.device
    if device.type == "mps":
        packed = packed.cpu()

    num_groups = dim // 16
    bytes_per_group = {1: 2, 2: 4, 6: 12}[ex_bits]
    packed_size = num_groups * bytes_per_group

    *leading, _ = packed.shape
    ex_code = torch.zeros(*leading, dim, dtype=torch.int16)

    flat_packed = packed.reshape(-1, packed_size)
    flat_ex = ex_code.reshape(-1, dim)

    for g in range(num_groups):
        inp = flat_packed[:, g * bytes_per_group : (g + 1) * bytes_per_group]
        group = flat_ex[:, g * 16 : (g + 1) * 16]

        if ex_bits == 1:
            for b in range(2):
                byte_val = inp[:, b].to(torch.int16)
                for i in range(8):
                    group[:, b * 8 + i] = (byte_val >> i) & 1

        elif ex_bits == 2:
            for b in range(4):
                byte_val = inp[:, b].to(torch.int16)
                for i in range(4):
                    group[:, b * 4 + i] = (byte_val >> (i * 2)) & 0x3

        elif ex_bits == 6:
            for sub_start in range(4):
                b0 = inp[:, sub_start * 3 + 0].to(torch.int16)
                b1 = inp[:, sub_start * 3 + 1].to(torch.int16)
                b2 = inp[:, sub_start * 3 + 2].to(torch.int16)
                group[:, sub_start * 4 + 0] = b0 & 0x3F
                group[:, sub_start * 4 + 1] = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
                group[:, sub_start * 4 + 2] = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
                group[:, sub_start * 4 + 3] = (b2 >> 2) & 0x3F

    return ex_code.to(device)
