"""Unit tests for the RaBitQ 1-bit KV codec (paper §5.5)."""

from __future__ import annotations

import math

import pytest
import torch

from QASP.inference.rabitq import (
    RaBitQCodec,
    pack_sign_bits_pm1,
    packed_sign_dim,
    unpack_sign_bits_pm1,
)


def test_rotation_buffer_is_orthonormal() -> None:
    codec = RaBitQCodec(dim=16, seed=0)
    q = codec.rotation
    identity = torch.eye(16)
    assert torch.allclose(q.transpose(-2, -1) @ q, identity, atol=1e-5)


def test_encode_preserves_norm_and_returns_packed_uint8_by_default() -> None:
    torch.manual_seed(0)
    codec = RaBitQCodec(dim=32, seed=1)
    x = torch.randn(4, 32)

    signs, norms = codec.encode(x)

    assert signs.dtype == torch.uint8
    assert signs.shape == (4, packed_sign_dim(32))
    assert torch.allclose(norms, x.norm(dim=-1), atol=1e-5)


def test_encode_packed_false_returns_int8_signs() -> None:
    torch.manual_seed(0)
    codec = RaBitQCodec(dim=32, seed=1)
    x = torch.randn(2, 32)

    signs, norms = codec.encode(x, packed=False)

    assert signs.dtype == torch.int8
    assert signs.shape == (2, 32)
    assert set(signs.unique().tolist()).issubset({-1, 1})
    assert torch.allclose(norms, x.norm(dim=-1), atol=1e-5)


def test_decode_preserves_norm_and_direction() -> None:
    torch.manual_seed(0)
    codec = RaBitQCodec(dim=64, seed=2)
    x = torch.randn(8, 64) * 3.5

    x_hat = codec.quantize(x)

    assert torch.allclose(x_hat.norm(dim=-1), x.norm(dim=-1), atol=1e-4)

    cos = (x * x_hat).sum(dim=-1) / (x.norm(dim=-1) * x_hat.norm(dim=-1))
    assert (cos > 0.5).all()


def test_encode_rejects_wrong_last_dim() -> None:
    codec = RaBitQCodec(dim=8)
    with pytest.raises(ValueError):
        codec.encode(torch.randn(2, 7))


def test_decode_rejects_mismatched_shapes() -> None:
    codec = RaBitQCodec(dim=8)
    signs = torch.ones(2, 3, 8, dtype=torch.int8)
    bad_norms = torch.ones(2, 4)
    with pytest.raises(ValueError):
        codec.decode(signs, bad_norms)


def test_pack_unpack_round_trip_random_pm1() -> None:
    torch.manual_seed(9)
    for d in (1, 3, 7, 8, 63, 64):
        signs = torch.randint(0, 2, (5, 11, d), dtype=torch.int8) * 2 - 1
        packed = pack_sign_bits_pm1(signs)
        assert packed.shape == (5, 11, packed_sign_dim(d))
        back = unpack_sign_bits_pm1(packed, d)
        assert torch.equal(signs, back)


def test_packed_encode_decode_matches_quantize() -> None:
    torch.manual_seed(10)
    codec = RaBitQCodec(dim=40, seed=3)
    x = torch.randn(6, 40)
    packed, norms = codec.encode(x, packed=True)
    from_int8, _ = codec.encode(x, packed=False)
    out_packed = codec.decode(packed, norms)
    out_int8 = codec.decode(from_int8, norms)
    assert torch.allclose(out_packed, out_int8, atol=0.0, rtol=0.0)
    assert torch.allclose(out_packed, codec.quantize(x), atol=0.0, rtol=0.0)


def test_decode_rejects_wrong_packed_width() -> None:
    codec = RaBitQCodec(dim=16, seed=0)
    bad = torch.zeros(2, 1, dtype=torch.uint8)
    norms = torch.ones(2)
    with pytest.raises(ValueError, match="packed_last_dim"):
        codec.decode(bad, norms)


def test_reconstruction_scale_matches_closed_form() -> None:
    """When signs perfectly agree with a rotated axis, decode must recover the axis."""

    codec = RaBitQCodec(dim=4, seed=5)
    rotated = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    x = rotated @ codec.rotation.transpose(0, 1)
    x_hat = codec.quantize(x)

    expected_norm = rotated.norm(dim=-1)
    assert torch.allclose(x_hat.norm(dim=-1), expected_norm, atol=1e-5)

    scale = expected_norm / math.sqrt(codec.dim)
    expected_rotated = torch.ones(1, 4) * scale
    assert torch.allclose((x_hat @ codec.rotation), expected_rotated, atol=1e-5)
