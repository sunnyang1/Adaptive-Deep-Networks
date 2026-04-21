"""Tests for QASP -> src/ integration bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QASP.integration import src_modules_available, try_import_src_attnres, try_import_src_rabitq, try_import_src_engram


def test_src_modules_availability_map() -> None:
    """Availability map should contain the three src/ submodules."""

    avail = src_modules_available()
    assert set(avail.keys()) == {"src.attnres", "src.rabitq", "src.engram"}
    # Values are booleans — we don't assert True/False because it depends on PYTHONPATH
    assert all(isinstance(v, bool) for v in avail.values())


def test_try_import_returns_module_or_none() -> None:
    """Lazy imports should return a module object when src/ is available."""

    mod = try_import_src_attnres()
    if mod is not None:
        assert hasattr(mod, "BlockAttnRes")


def test_src_attnres_adapter_forward_shape() -> None:
    """SrcAttnResAdapter should match ValueWeightedAttnRes output shape when src/ is available."""

    from QASP.integration import SrcAttnResAdapter

    if try_import_src_attnres() is None:
        pytest.skip("src.attnres not available")

    adapter = SrcAttnResAdapter(hidden_size=32, num_blocks=4)
    hidden = torch.randn(2, 5, 32)
    block_repr = torch.randn(2, 4, 32)
    block_quality = torch.randn(2, 4)

    out = adapter(hidden, block_repr, block_quality)
    assert out.shape == hidden.shape


def test_src_engram_adapter_forward_shape() -> None:
    """SrcEngramAdapter should match ValueWeightedEngram output shape when src/ is available."""

    from QASP.integration import SrcEngramAdapter

    if try_import_src_engram() is None:
        pytest.skip("src.engram not available")

    adapter = SrcEngramAdapter(layer_id=0, hidden_size=32)
    hidden = torch.randn(2, 5, 32)
    memory_vec = torch.randn(2, 32)
    memory_quality = torch.randn(2)

    out = adapter(hidden, memory_vec, memory_quality)
    assert out.shape == hidden.shape


def test_src_rabitq_codec_quantize_shape() -> None:
    """SrcRaBitQCodec should preserve tensor shape through quantize/dequantize."""

    from QASP.integration import SrcRaBitQCodec

    if try_import_src_rabitq() is None:
        pytest.skip("src.rabitq not available")

    codec = SrcRaBitQCodec(head_dim=16, device="cpu")
    x = torch.randn(2, 4, 10, 16)
    out = codec.quantize(x)
    assert out.shape == x.shape
