"""Optional bridge from QASP to ``src/`` ADN modules.

This module provides lazy import helpers and adapter wrappers so that QASP
 can reuse implementations from the parent ``src/`` package when it is on
 ``PYTHONPATH``.  When ``src/`` is unavailable the imports return ``None`` and
 QASP falls back to its own local reimplementations.

Usage::

    from QASP.integration import try_import_src_attnres, SrcAttnResAdapter
    BlockAttnRes = try_import_src_attnres()  # None if src/ unavailable

The adapters translate between QASP's functional API and ``src/'s`` object-
oriented APIs.  They are not zero-cost abstractions — the goal is code reuse,
not bit-identical equivalence.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _try_import(module_path: str) -> Any | None:
    try:
        import importlib

        return importlib.import_module(module_path)
    except Exception:
        return None


def try_import_src_attnres() -> Any | None:
    """Return ``src.attnres`` module if available, else ``None``."""
    return _try_import("src.attnres")


def try_import_src_rabitq() -> Any | None:
    """Return ``src.rabitq`` module if available, else ``None``."""
    return _try_import("src.rabitq")


def try_import_src_engram() -> Any | None:
    """Return ``src.engram`` module if available, else ``None``."""
    return _try_import("src.engram")


# ---------------------------------------------------------------------------
# Adapter wrappers
# ---------------------------------------------------------------------------

class SrcAttnResAdapter(torch.nn.Module):
    """Adapter that exposes QASP's ``ValueWeightedAttnRes`` interface backed by ``src.attnres.BlockAttnRes``.

    ``src.attnres.BlockAttnRes`` expects a *list* of block representations and a
    partial block, whereas QASP pre-computes a single tensor of block vectors.
    This adapter reshapes the QASP tensor into the list format that ``BlockAttnRes``
    expects and uses a single pseudo-query for the attention path.
    """

    def __init__(self, hidden_size: int, num_blocks: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        src_attnres = try_import_src_attnres()
        if src_attnres is None:
            raise ImportError("src.attnres is not available; cannot use SrcAttnResAdapter")

        self._block_attnres = src_attnres.BlockAttnRes(dim=hidden_size, num_blocks=num_blocks)

    def forward(
        self,
        hidden_states: Tensor,
        block_representations: Tensor,
        block_quality: Tensor,
    ) -> Tensor:
        """Match ``ValueWeightedAttnRes.forward`` signature.

        Args:
            hidden_states: ``[B, T, D]``
            block_representations: ``[B, N, D]``
            block_quality: ``[B, N]``

        Returns:
            Residual of shape ``[B, T, D]``.
        """

        # src.attnres.BlockAttnRes works with a *list* of completed blocks,
        # each expected to be [B, T, D].  Our block_representations are [B, N, D]
        # (one vector per block).  We unsqueeze a time dim of 1.
        blocks = [block_representations[:, i : i + 1, :] for i in range(block_representations.size(1))]

        # QASP uses quality-weighted affinity; src/ uses raw pseudo-query dot product.
        # We approximate by scaling each block vector by its quality score so that
        # the pseudo-query attention naturally favours high-quality blocks.
        scaled_blocks = [blocks[i] * block_quality[:, i : i + 1, None] for i in range(len(blocks))]

        # Use the last block as the "partial" input (standard residual)
        partial = scaled_blocks[-1] if scaled_blocks else hidden_states[:, 0:1, :]
        if len(scaled_blocks) > 1:
            completed = scaled_blocks[:-1]
        else:
            completed = []

        h_attn, _ = self._block_attnres(completed, partial, use_attn=True, use_mlp=False)

        # h_attn is [B, T, D] where T=1; broadcast across the full sequence
        return h_attn.expand_as(hidden_states)


class SrcEngramAdapter(torch.nn.Module):
    """Adapter that exposes QASP's ``ValueWeightedEngram`` interface backed by ``src.engram.Engram``.

    ``src.engram.Engram`` expects ``[B, L, G, D]`` tensors with hyper-connection
    groups.  This adapter squeezes/unsqueezes the group dimension and handles
    the n-gram hash mapping internally.
    """

    def __init__(self, layer_id: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        src_engram = try_import_src_engram()
        if src_engram is None:
            raise ImportError("src.engram is not available; cannot use SrcEngramAdapter")

        from src.engram.config import EngramConfig

        config = EngramConfig(
            enabled=True,
            engram_vocab_size=[50000, 50000],
            max_ngram_size=3,
            n_embed_per_ngram=min(256, hidden_size),
            n_head_per_ngram=4,
            layer_ids=[layer_id],
            tokenizer_name_or_path="gpt2",
        )
        self._engram = src_engram.Engram(layer_id=layer_id, config=config, hidden_size=hidden_size, hc_mult=1)

    def forward(
        self,
        hidden_states: Tensor,
        memory_vector: Tensor,
        memory_quality: Tensor,
    ) -> Tensor:
        """Match ``ValueWeightedEngram.forward`` signature.

        Args:
            hidden_states: ``[B, T, D]``
            memory_vector: ``[B, D]`` (ignored in src/ path; engram computes its own)
            memory_quality: ``[B]`` (used as a scalar gate multiplier)

        Returns:
            Residual of shape ``[B, T, D]``.
        """

        # src/ engram expects [B, L, G, D] where G = hc_mult = 1
        x = hidden_states.unsqueeze(2)  # [B, T, 1, D]

        # Need input_ids for n-gram hashing — fallback to zeros if not provided
        B, T, _ = hidden_states.shape
        dummy_input_ids = torch.zeros(B, T, dtype=torch.long, device=hidden_states.device)

        out = self._engram(x, dummy_input_ids)  # [B, T, 1, D]
        residual = out.squeeze(2)  # [B, T, D]

        # Scale by average quality if provided
        if memory_quality is not None:
            scale = memory_quality.mean().sigmoid()
            residual = residual * scale

        return residual


class SrcRaBitQCodec:
    """Adapter that exposes QASP's ``_KVCodec`` protocol backed by ``src.rabitq.RaBitQ``.

    This is a stateful codec: :meth:`fit` must be called once on representative
    K/V data before :meth:`quantize` can be used during generation.
    """

    def __init__(self, head_dim: int, device: str = "cpu") -> None:
        src_rabitq = try_import_src_rabitq()
        if src_rabitq is None:
            raise ImportError("src.rabitq is not available; cannot use SrcRaBitQCodec")

        self._rq = src_rabitq.RaBitQ(total_bits=1, head_dim=head_dim, device=device)
        self._head_dim = head_dim
        self._fitted = False

    def fit(self, sample_k: Tensor, sample_v: Tensor) -> None:
        """Fit quantizers on sample K/V tensors of shape ``[..., head_dim]``."""

        self._rq.fit(sample_k, sample_v)
        self._fitted = True

    def quantize(self, x: Tensor) -> Tensor:
        """Quantize and immediately decompress to match QASP codec semantics.

        Args:
            x: Tensor of shape ``[B, H, T, head_dim]``.

        Returns:
            Dequantized tensor of the same shape.
        """

        if not self._fitted:
            # Auto-fit with zeros if user skipped explicit fitting
            zeros = torch.zeros(self._head_dim, device=x.device)
            self._rq.fit(zeros, zeros)
            self._fitted = True

        original_shape = x.shape
        flat = x.reshape(-1, self._head_dim)
        compressed = self._rq.compress(flat, flat)
        dequantized, _ = self._rq.decompress(compressed)
        return dequantized.reshape(original_shape)


def src_modules_available() -> dict[str, bool]:
    """Return availability map for src/ submodules."""

    return {
        "src.attnres": try_import_src_attnres() is not None,
        "src.rabitq": try_import_src_rabitq() is not None,
        "src.engram": try_import_src_engram() is not None,
    }
