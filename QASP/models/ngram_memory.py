"""Minimal n-gram hashed external memory for QASP Engram (Section 5.3).

Each entry stores ``(m, ρ_mem)``: the embedding vector ``m ∈ R^d`` and the
average information-quality score ``ρ_mem ∈ [0, 1]`` over the n-gram tokens
(see the Engram memory tuple in Sec.~5.3 / ``eq:value-weighted-engram``). Lookup is O(1) via a deterministic FNV-1a hash
modulo the table size; unpopulated slots return zero vectors and zero quality.

This is a CPU-side reference implementation suitable for unit testing and
small-scale experiments. A production system would mirror this layout in a
GPU-resident embedding table with batched gather kernels.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor


_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_FNV_MASK = 0xFFFFFFFFFFFFFFFF


def _fnv1a64(values: Iterable[int]) -> int:
    """Deterministic 64-bit FNV-1a hash over a sequence of integers."""

    h = _FNV_OFFSET
    for value in values:
        h ^= int(value) & _FNV_MASK
        h = (h * _FNV_PRIME) & _FNV_MASK
    return h


class NgramMemory:
    """Hash-addressed table of ``(memory_vector, memory_quality)`` entries."""

    def __init__(
        self,
        table_size: int,
        hidden_size: int,
        n_gram: int = 3,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        if table_size < 1:
            raise ValueError("`table_size` must be >= 1.")
        if hidden_size < 1:
            raise ValueError("`hidden_size` must be >= 1.")
        if n_gram < 1:
            raise ValueError("`n_gram` must be >= 1.")

        self.table_size = int(table_size)
        self.hidden_size = int(hidden_size)
        self.n_gram = int(n_gram)
        self.device = torch.device(device)
        self.dtype = dtype

        self.values = torch.zeros(self.table_size, self.hidden_size, dtype=dtype, device=self.device)
        self.qualities = torch.zeros(self.table_size, dtype=dtype, device=self.device)
        self.populated = torch.zeros(self.table_size, dtype=torch.bool, device=self.device)

    def hash_index(self, tokens: Sequence[int]) -> int:
        """Hash ``tokens`` to a slot in ``[0, table_size)``."""

        if len(tokens) == 0:
            raise ValueError("`tokens` must be non-empty.")
        return _fnv1a64(tokens) % self.table_size

    def write(self, tokens: Sequence[int], vector: Tensor, quality: float) -> int:
        """Insert ``(vector, quality)`` at the hashed slot and return its index."""

        if vector.shape[-1] != self.hidden_size:
            raise ValueError("`vector` last dim must equal `hidden_size`.")
        idx = self.hash_index(tokens)
        self.values[idx] = vector.to(device=self.device, dtype=self.dtype)
        self.qualities[idx] = float(quality)
        self.populated[idx] = True
        return idx

    def lookup(self, tokens: Sequence[int]) -> tuple[Tensor, Tensor, bool]:
        """Read ``(vector, quality, populated)`` from the slot for ``tokens``."""

        idx = self.hash_index(tokens)
        return self.values[idx], self.qualities[idx], bool(self.populated[idx].item())

    @torch.no_grad()
    def batch_write(
        self,
        input_ids: Tensor,
        vectors: Tensor,
        qualities: Tensor,
    ) -> None:
        """Write n-gram memories for every valid position in ``input_ids``.

        Args:
            input_ids: ``[B, T]`` token indices.
            vectors: ``[B, T, D]`` per-token vectors to store (typically hidden
                states at the last token of each n-gram).
            qualities: ``[B, T]`` per-token quality scores.  The mean over the
                n-gram window is stored alongside each entry.
        """

        if input_ids.ndim != 2:
            raise ValueError("`input_ids` must have shape [B, T].")
        if vectors.shape[:2] != input_ids.shape:
            raise ValueError("`vectors` batch/seq dims must match `input_ids`.")
        if qualities.shape != input_ids.shape:
            raise ValueError("`qualities` must have shape [B, T].")

        batch_size, seq_len = input_ids.shape
        ids_cpu = input_ids.detach().cpu().tolist()
        quals_cpu = qualities.detach().cpu().tolist()

        for b in range(batch_size):
            row = ids_cpu[b]
            row_quals = quals_cpu[b]
            for t in range(self.n_gram - 1, seq_len):
                ngram = row[t - self.n_gram + 1 : t + 1]
                qual_window = row_quals[t - self.n_gram + 1 : t + 1]
                mean_qual = sum(qual_window) / len(qual_window)
                vec = vectors[b, t]
                self.write(ngram, vec, mean_qual)

    @torch.no_grad()
    def batch_lookup(
        self,
        input_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Per-token n-gram lookup over ``input_ids`` of shape ``[B, T]``.

        Returns ``(memory_vectors, memory_qualities)`` of shapes ``[B, T, D]``
        and ``[B, T]``. For positions ``t < n_gram - 1`` (insufficient context)
        and unpopulated slots the result is zero.
        """

        if input_ids.ndim != 2:
            raise ValueError("`input_ids` must have shape [B, T].")
        batch_size, seq_len = input_ids.shape

        out_vec = torch.zeros(
            batch_size, seq_len, self.hidden_size, dtype=self.dtype, device=self.device
        )
        out_qual = torch.zeros(batch_size, seq_len, dtype=self.dtype, device=self.device)

        ids_cpu = input_ids.detach().cpu().tolist()
        for b in range(batch_size):
            row = ids_cpu[b]
            for t in range(self.n_gram - 1, seq_len):
                ngram = row[t - self.n_gram + 1 : t + 1]
                idx = self.hash_index(ngram)
                if bool(self.populated[idx].item()):
                    out_vec[b, t] = self.values[idx]
                    out_qual[b, t] = self.qualities[idx]
        return out_vec, out_qual
