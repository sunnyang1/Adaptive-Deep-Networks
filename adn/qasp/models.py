"""QASP model stack — transformer layers, components, and top-level model.

Merged modules (original QASP locations):
    * ``QASP/models/rope.py``             → RoPE helpers
    * ``QASP/models/ngram_memory.py``     → NgramMemory
    * ``QASP/inference/kv_cache.py``      → KVCache
    * ``QASP/inference/rabitq.py``        → RaBitQCodec
    * ``QASP/models/components.py``       → QASPTransformerConfig, RMSNorm, CausalSelfAttention, FeedForward, compute_block_representations
    * ``QASP/models/qasp_layer.py``       → QASPLayer
    * ``QASP/models/qasp_transformer.py`` → QASPTransformer, create_qasp_transformer

**Evaluation semantics.**  :meth:`QASPTransformer.forward` and
:meth:`QASPTransformer.prefill` run a full-sequence pass and, when AttnRes is
enabled, compute block summaries and ``ρ̄_m`` from the complete hidden states —
this matches the QASP paper's canonical (Path~A) definition.

:meth:`QASPTransformer.step` is an engineering API for autoregressive decoding.
With ``use_attnres=True``, block statistics are recomputed from a growing
``layer_input_history`` prefix; that information set differs from the full
sequence in the paper's equations, so **step logits are not guaranteed** to
match ``forward(cat(prefix, new_token))`` at intermediate positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    cast,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adn.qasp.matrix_qasp import QASPConfig, PonderGate, matrix_qasp_update
from adn.qasp.quality_score import compute_quality_score
from adn.qasp.stiefel import project_to_stiefel
from adn.qasp.value_weighted_attnres import ValueWeightedAttnRes
from adn.qasp.value_weighted_engram import ValueWeightedEngram


# ===========================================================================
# 1. RoPE (from QASP/models/rope.py)
# ===========================================================================

def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape [B, H, T, d_h]
        k: Key tensor of shape [B, H_kv, T, d_h]
        cos: Cosine precomputations of shape [1, 1, T, d_h]
        sin: Sine precomputations of shape [1, 1, T, d_h]

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    """

    def rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Precompute and cache RoPE frequency tensors."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cached_seq_len: int = 0
        self._cached_cos: Tensor | None = None
        self._cached_sin: Tensor | None = None

    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        if seq_len > self._cached_seq_len or self._cached_cos is None or self._cached_cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
            emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
            self._cached_cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
            self._cached_sin = emb.sin()[None, None, :, :]  # [1, 1, seq_len, dim]
            self._cached_seq_len = seq_len
        return self._cached_cos, self._cached_sin

    def forward(self, x: Tensor, seq_len: int | None = None) -> tuple[Tensor, Tensor]:
        """Return cos and sin tensors for the given sequence length.

        Args:
            x: Reference tensor to infer device/dtype from. Shape irrelevant.
            seq_len: Sequence length. If None, inferred from x's time dim.

        Returns:
            (cos, sin) each of shape [1, 1, seq_len, dim]
        """

        if seq_len is None:
            seq_len = x.shape[-2]
        cos, sin = self._compute_cos_sin(seq_len, x.device)
        return cos.to(x.dtype), sin.to(x.dtype)


# ===========================================================================
# 2. N-gram memory (from QASP/models/ngram_memory.py)
# ===========================================================================

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


# ===========================================================================
# 3. KV Cache (from QASP/inference/kv_cache.py)
# ===========================================================================

@dataclass
class KVCache:
    """Mutable cache of generated tokens and per-layer attention state."""

    input_ids: Tensor
    layer_keys: List[Optional[Tensor]] = field(default_factory=list)
    layer_values: List[Optional[Tensor]] = field(default_factory=list)
    layer_inputs: List[Optional[Tensor]] = field(default_factory=list)
    per_token_quality: Optional[Tensor] = None
    last_logits: Optional[Tensor] = None

    @classmethod
    def from_input_ids(cls, input_ids: Tensor, num_layers: int = 0) -> "KVCache":
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        return cls(
            input_ids=input_ids.clone(),
            layer_keys=[None] * num_layers,
            layer_values=[None] * num_layers,
            layer_inputs=[None] * num_layers,
            per_token_quality=None,
            last_logits=None,
        )

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_layers(self) -> int:
        return len(self.layer_keys)

    def append(self, token: Tensor) -> None:
        """Append one generated token per batch element to ``input_ids``."""

        if token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("token must have shape [B, 1]")
        if token.shape[0] != self.input_ids.shape[0]:
            raise ValueError("token batch size must match cache batch size")
        self.input_ids = torch.cat([self.input_ids, token], dim=1)


# ===========================================================================
# 4. RaBitQ codec (from QASP/inference/rabitq.py)
# ===========================================================================

def packed_sign_dim(feature_dim: int) -> int:
    """Number of bytes needed to store ``feature_dim`` sign bits (8 bits per byte)."""

    if feature_dim < 1:
        raise ValueError("`feature_dim` must be >= 1.")
    return (int(feature_dim) + 7) // 8


def pack_sign_bits_pm1(signs_pm1: Tensor) -> Tensor:
    """Pack a ``±1`` sign tensor along its last dimension into ``uint8`` bytes.

    Channel ``i`` maps to byte ``i // 8``, bit ``i % 8`` (LSB = smallest ``i`` in
    that byte). The final byte is zero-padded when ``d`` is not a multiple of 8.
    """

    if signs_pm1.shape[-1] < 1:
        raise ValueError("last dimension must be >= 1.")
    d = signs_pm1.shape[-1]
    pack_len = packed_sign_dim(d)
    pos = (signs_pm1 >= 0).to(torch.uint8)
    remainder = d % 8
    if remainder != 0:
        pos = F.pad(pos, (0, 8 - remainder), value=0)
    blocks = pos.reshape(*pos.shape[:-1], pack_len, 8)
    shifts = torch.arange(8, device=signs_pm1.device, dtype=torch.uint8)
    packed = (blocks << shifts).sum(dim=-1).to(torch.uint8)
    return packed


def unpack_sign_bits_pm1(packed: Tensor, feature_dim: int) -> Tensor:
    """Unpack ``uint8`` sign bytes to ``int8`` ``±1`` with ``feature_dim`` channels."""

    d = int(feature_dim)
    if d < 1:
        raise ValueError("`feature_dim` must be >= 1.")
    if packed.dtype != torch.uint8:
        raise ValueError("`packed` must have dtype torch.uint8.")
    expected = packed_sign_dim(d)
    if packed.shape[-1] != expected:
        raise ValueError(
            f"last dim of `packed` must be {expected} for feature_dim={d}, got {packed.shape[-1]}."
        )
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32).view(1, 1, 8)
    bits = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 1
    flat = bits.reshape(*packed.shape[:-1], -1)[..., :d]
    return torch.where(flat > 0, torch.ones_like(flat, dtype=torch.int8), -torch.ones_like(flat, dtype=torch.int8))


class RaBitQCodec(nn.Module):
    """Shared 1-bit sign codec with a fixed random orthonormal rotation."""

    rotation: Tensor

    def __init__(self, dim: int, *, seed: int = 0) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("`dim` must be >= 1.")
        self.dim = int(dim)

        generator = torch.Generator().manual_seed(int(seed))
        gaussian = torch.randn(self.dim, self.dim, generator=generator)
        q, _ = torch.linalg.qr(gaussian)
        self.register_buffer("rotation", q, persistent=True)

    @property
    def packed_last_dim(self) -> int:
        """Length of the packed sign axis (bytes) for this codec."""

        return packed_sign_dim(self.dim)

    def encode(self, x: Tensor, *, packed: bool = True) -> tuple[Tensor, Tensor]:
        """Encode ``x`` (shape ``[..., d]``) to signs and per-vector L2 norms of ``y = xQ``.

        Args:
            packed: If ``True`` (default), return signs as ``uint8`` with last dim
                ``ceil(d/8)``. If ``False``, return ``int8`` ``±1`` per channel
                (legacy layout).
        """

        if x.shape[-1] != self.dim:
            raise ValueError("last dim of `x` must equal codec.dim")

        rotated = x @ self.rotation
        norms = rotated.norm(dim=-1)
        signs = torch.where(rotated >= 0, torch.ones_like(rotated), -torch.ones_like(rotated))
        signs_i8 = signs.to(torch.int8)
        if not packed:
            return signs_i8, norms
        return pack_sign_bits_pm1(signs_i8), norms

    def decode(self, signs: Tensor, norms: Tensor) -> Tensor:
        """Reconstruct an approximation of ``x`` from ``(signs, norms)``.

        ``signs`` may be ``uint8`` packed bytes (last dim ``ceil(d/8)``) or ``int8``
        ``±1`` values (last dim ``d``). :meth:`encode` uses packed ``uint8`` by
        default; distinguish layouts by dtype.
        """

        if signs.dtype == torch.uint8:
            if signs.shape[-1] != self.packed_last_dim:
                raise ValueError(
                    "last dim of packed `signs` must equal codec.packed_last_dim "
                    f"({self.packed_last_dim}), got {signs.shape[-1]}."
                )
            signs_i8 = unpack_sign_bits_pm1(signs, self.dim)
        elif signs.dtype == torch.int8:
            if signs.shape[-1] != self.dim:
                raise ValueError("last dim of `signs` must equal codec.dim")
            signs_i8 = signs
        else:
            raise ValueError("`signs` must be torch.uint8 (packed) or torch.int8 (unpacked).")

        if signs_i8.shape[:-1] != norms.shape:
            raise ValueError("`norms` must match all-but-last dims of `signs`.")

        scale = (norms / math.sqrt(float(self.dim))).unsqueeze(-1)
        rotated_approx = signs_i8.to(scale.dtype) * scale
        return rotated_approx @ self.rotation.transpose(-2, -1)

    def quantize(self, x: Tensor) -> Tensor:
        """Convenience: ``decode(encode(x))`` — returns same-shape approximation."""

        signs, norms = self.encode(x)
        return self.decode(signs, norms)


# ===========================================================================
# 5. Core components (from QASP/models/components.py)
# ===========================================================================

class _KVCodec(Protocol):
    """Structural type for KV-cache codecs (see :class:`RaBitQCodec`)."""

    def quantize(self, x: Tensor) -> Tensor: ...


@dataclass
class QASPTransformerConfig:
    """Configuration for the lightweight QASP transformer stack."""

    vocab_size: int = 32000
    hidden_size: int = 256
    num_heads: int = 8
    num_key_value_heads: int | None = None
    num_layers: int = 4
    mlp_ratio: float = 4.0
    max_position_embeddings: int = 2048
    attnres_blocks: int = 4
    use_attnres: bool = True
    use_engram: bool = True
    engram_table_size: int = 4096
    engram_n_gram: int = 3
    adapt_rank: int = 32
    stiefel_overlay_scale: float = 0.0
    quantize_kv: bool = False
    kv_codec_seed: int = 0
    quality_window_size: int | None = None
    gate_quality_computation: bool = False
    use_stiefel_query: bool = False
    use_rope: bool = False

    @classmethod
    def paper_1_5b(cls) -> "QASPTransformerConfig":
        """Return the 1.5B-parameter architecture described in the QASP paper (Sec 6.1.1, Table 1)."""

        return cls(
            vocab_size=32000,
            hidden_size=2048,
            num_heads=16,
            num_key_value_heads=4,
            num_layers=24,
            mlp_ratio=5504 / 2048,  # SwiGLU intermediate = 5504
            max_position_embeddings=4096,
            attnres_blocks=8,
            use_attnres=True,
            use_engram=True,
            use_rope=True,
        )


class RMSNorm(nn.Module):
    """Simple RMSNorm used for stable transformer blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional Grouped Query Attention (GQA)."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_key_value_heads = config.num_key_value_heads or config.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.overlay_scale = float(config.stiefel_overlay_scale)
        self.use_stiefel_query = config.use_stiefel_query
        self.use_rope = config.use_rope

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def _apply_stiefel_overlay(
        self,
        hidden_states: Tensor,
        stiefel_query: Tensor | None,
    ) -> Tensor:
        """Apply ``h + scale · h W W^T`` where ``W ∈ St(k, d)`` (Stiefel query path; see ``sec:qasp-matrix`` in ``QASP_paper.tex``)."""

        if stiefel_query is None or self.overlay_scale == 0.0:
            return hidden_states
        if stiefel_query.ndim != 2 or stiefel_query.shape[0] != self.hidden_size:
            raise ValueError(
                "stiefel_query must have shape [hidden_size, k] to form a Stiefel overlay."
            )
        projected = hidden_states @ stiefel_query
        overlay = projected @ stiefel_query.transpose(-2, -1)
        return hidden_states + self.overlay_scale * overlay

    def _shape(self, tensor: Tensor, num_heads: int) -> Tensor:
        batch, seq, _ = tensor.shape
        tensor = tensor.view(batch, seq, num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _repeat_kv(self, hidden_states: Tensor, n_rep: int) -> Tensor:
        """Repeat K/V heads to match query head count for GQA."""

        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _compute_q(
        self,
        hidden_states: Tensor,
        stiefel_query: Tensor | None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> Tensor:
        """Compute query tensor, optionally via Stiefel query replacement or RoPE."""

        if self.use_stiefel_query and stiefel_query is not None:
            if stiefel_query.shape != (self.hidden_size, self.hidden_size):
                raise ValueError(
                    f"stiefel_query must have shape ({self.hidden_size}, {self.hidden_size}) "
                    f"when use_stiefel_query=True, got {tuple(stiefel_query.shape)}"
                )
            q = hidden_states @ stiefel_query
            q = self._shape(q, self.num_heads)
            if self.use_rope and rope_cos is not None:
                q, _ = apply_rotary_pos_emb(q, q, rope_cos, rope_sin)
            return q

        q_input = self._apply_stiefel_overlay(hidden_states, stiefel_query)
        q = self._shape(self.q_proj(q_input), self.num_heads)
        if self.use_rope and rope_cos is not None:
            q, _ = apply_rotary_pos_emb(q, q, rope_cos, rope_sin)
        return q

    def forward(
        self,
        hidden_states: Tensor,
        stiefel_query: Tensor | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> Tensor:
        q = self._compute_q(hidden_states, stiefel_query, rope_cos, rope_sin)
        k = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        v = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)
        if self.use_rope and rope_cos is not None:
            _, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        k = self._repeat_kv(k, self.num_key_value_groups)
        v = self._repeat_kv(v, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(hidden_states.size(0), hidden_states.size(1), self.hidden_size)
        return cast(Tensor, self.o_proj(attn_output))

    def forward_with_cache(
        self,
        hidden_states: Tensor,
        codec: _KVCodec | None = None,
        stiefel_query: Tensor | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Same as :meth:`forward` but also returns full K/V for the KV cache.

        When ``codec`` is provided, ``K`` and ``V`` are passed through
        ``codec.quantize`` before attention so the attention computation
        matches what subsequent :meth:`step` calls will observe. When
        ``stiefel_query`` is provided, the Stiefel overlay (``sec:qasp-matrix``) is applied to
        the query side before ``q_proj``.
        """

        q = self._compute_q(hidden_states, stiefel_query, rope_cos, rope_sin)
        k = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        v = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)
        if self.use_rope and rope_cos is not None:
            _, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        if codec is not None:
            k = codec.quantize(k)
            v = codec.quantize(v)

        k_repeated = self._repeat_kv(k, self.num_key_value_groups)
        v_repeated = self._repeat_kv(v, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            q,
            k_repeated,
            v_repeated,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(hidden_states.size(0), hidden_states.size(1), self.hidden_size)
        return cast(Tensor, self.o_proj(attn_output)), k, v

    def step(
        self,
        hidden_new: Tensor,
        cached_k: Tensor | None,
        cached_v: Tensor | None,
        codec: _KVCodec | None = None,
        stiefel_query: Tensor | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run a single-token incremental attention update with KV caching.

        ``hidden_new`` has shape ``[B, 1, D]`` (the embedding of the newly
        generated token). ``cached_k`` and ``cached_v`` are the previously
        accumulated keys / values, each of shape ``[B, H_kv, T_prev, d_h]`` or
        ``None`` for the very first step.

        Returns ``(output, new_k, new_v)``, where ``new_k`` / ``new_v`` are the
        updated per-head caches of shape ``[B, H_kv, T_prev + 1, d_h]``.
        """

        if hidden_new.ndim != 3 or hidden_new.size(1) != 1:
            raise ValueError("`hidden_new` must have shape [B, 1, D].")

        q = self._compute_q(hidden_new, stiefel_query, rope_cos, rope_sin)
        k_new = self._shape(self.k_proj(hidden_new), self.num_key_value_heads)
        v_new = self._shape(self.v_proj(hidden_new), self.num_key_value_heads)
        if self.use_rope and rope_cos is not None:
            _, k_new = apply_rotary_pos_emb(q, k_new, rope_cos, rope_sin)

        if codec is not None:
            k_new = codec.quantize(k_new)
            v_new = codec.quantize(v_new)

        if cached_k is None or cached_v is None:
            if cached_k is not None or cached_v is not None:
                raise ValueError("`cached_k` and `cached_v` must both be None or both be Tensors.")
            full_k = k_new
            full_v = v_new
        else:
            full_k = torch.cat([cached_k, k_new], dim=2)
            full_v = torch.cat([cached_v, v_new], dim=2)

        k_repeated = self._repeat_kv(full_k, self.num_key_value_groups)
        v_repeated = self._repeat_kv(full_v, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            q,
            k_repeated,
            v_repeated,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(hidden_new.size(0), 1, self.hidden_size)
        return cast(Tensor, self.o_proj(attn_output)), full_k, full_v


class FeedForward(nn.Module):
    """SwiGLU-style MLP block."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        inner_dim = int(config.hidden_size * config.mlp_ratio)
        self.gate_proj = nn.Linear(config.hidden_size, inner_dim)
        self.up_proj = nn.Linear(config.hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(hidden_states))
        values = self.up_proj(hidden_states)
        return cast(Tensor, self.down_proj(gated * values))


def compute_block_representations(
    hidden_states: Tensor,
    num_blocks: int,
    low_pass_ratio: float = 0.25,
    *,
    quality_window_size: int | None = None,
    per_token_quality: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Pool hidden states into block summaries and block-level quality.

    Implements label ``eq:block-quality`` in ``QASP_paper.tex`` (block-level mean
    quality ``ρ̄_m = (1/|B_m|) Σ_{t∈B_m} ρ(t)``), with ``ρ(t)`` from the spectral
    quality score (``eq:quality-score``, Sec.~3.2).

    ``quality_window_size`` forwards to :func:`compute_quality_score` (optional
    sliding-window batching along ``T``; ``None`` = one FFT over the full sequence).

    **Canonical use.**  Pass the **entire** sequence tensor ``[B, T, D]`` from
    one forward pass so that ``B_m`` and ``ρ̄_m`` match the paper's
    full-context definition.  For prefix-only histories (e.g. incremental
    ``step``), statistics differ from that definition; the manuscript does not
    claim bit-identical equivalence between the two.

    Args:
        per_token_quality: Optional pre-computed ``rho(t)`` of shape
            ``[B, T]``. When provided, the function skips the FFT call and
            pools the supplied scores directly.  This is used to cache quality
            across layers within a single forward pass.
    """

    chunks = torch.chunk(hidden_states, chunks=max(1, num_blocks), dim=1)
    block_vectors = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)

    if per_token_quality is None:
        per_token_quality = compute_quality_score(
            hidden_states,
            low_pass_ratio=low_pass_ratio,
            window_size=quality_window_size,
        )
    quality_chunks = torch.chunk(per_token_quality, chunks=max(1, num_blocks), dim=1)
    block_quality = torch.stack([chunk.mean(dim=1) for chunk in quality_chunks], dim=1)
    return block_vectors, block_quality


# ===========================================================================
# 6. QASP Layer (from QASP/models/qasp_layer.py)
# ===========================================================================

class QASPLayer(nn.Module):
    """Single QASP layer with self-attention, MLP, and optional hooks."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        self.use_attnres = config.use_attnres
        self.use_engram = config.use_engram

        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = FeedForward(config)

        self.attnres = ValueWeightedAttnRes(config.hidden_size) if self.use_attnres else None
        self.engram = ValueWeightedEngram(config.hidden_size) if self.use_engram else None

        if config.use_stiefel_query:
            rank = config.hidden_size
            init_iters = 15  # square matrices need more Newton-Schulz iterations
        else:
            rank = max(1, min(config.adapt_rank, config.hidden_size))
            init_iters = 5
        with torch.no_grad():
            init_matrix = project_to_stiefel(torch.randn(config.hidden_size, rank), num_iters=init_iters)
        self.stiefel_query = nn.Parameter(init_matrix, requires_grad=False)

    def forward(
        self,
        hidden_states: Tensor,
        block_representations: Tensor | None = None,
        block_quality: Tensor | None = None,
        memory_vector: Tensor | None = None,
        memory_quality: Tensor | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states),
            stiefel_query=self.stiefel_query,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        if self.attnres is not None and block_representations is not None and block_quality is not None:
            hidden_states = hidden_states + self.attnres(
                hidden_states,
                block_representations,
                block_quality,
            )

        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))

        if self.engram is not None and memory_vector is not None and memory_quality is not None:
            hidden_states = self.engram(hidden_states, memory_vector, memory_quality)

        return hidden_states

    def forward_with_cache(
        self,
        hidden_states: Tensor,
        block_representations: Tensor | None = None,
        block_quality: Tensor | None = None,
        memory_vector: Tensor | None = None,
        memory_quality: Tensor | None = None,
        kv_codec: _KVCodec | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full-sequence forward that also emits per-layer K/V for cache prefill."""

        attn_out, k, v = self.attn.forward_with_cache(
            self.attn_norm(hidden_states),
            codec=kv_codec,
            stiefel_query=self.stiefel_query,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        hidden_states = hidden_states + attn_out

        if self.attnres is not None and block_representations is not None and block_quality is not None:
            hidden_states = hidden_states + self.attnres(
                hidden_states,
                block_representations,
                block_quality,
            )

        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))

        if self.engram is not None and memory_vector is not None and memory_quality is not None:
            hidden_states = self.engram(hidden_states, memory_vector, memory_quality)

        return hidden_states, k, v

    def step(
        self,
        hidden_new: Tensor,
        cached_k: Tensor | None,
        cached_v: Tensor | None,
        block_representations: Tensor | None = None,
        block_quality: Tensor | None = None,
        memory_vector: Tensor | None = None,
        memory_quality: Tensor | None = None,
        kv_codec: _KVCodec | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Incremental single-token forward that reuses cached K/V.

        ``hidden_new`` has shape ``[B, 1, D]``. ``block_representations`` /
        ``block_quality`` should already reflect the full layer-input history
        (including the new token); the AttnRes residual is broadcast to length
        1 to match ``hidden_new``. ``memory_vector`` / ``memory_quality`` are
        expected to be the lookup results for the new token only.
        """

        attn_out, new_k, new_v = self.attn.step(
            self.attn_norm(hidden_new),
            cached_k=cached_k,
            cached_v=cached_v,
            codec=kv_codec,
            stiefel_query=self.stiefel_query,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        hidden_new = hidden_new + attn_out

        if self.attnres is not None and block_representations is not None and block_quality is not None:
            hidden_new = hidden_new + self.attnres(
                hidden_new,
                block_representations,
                block_quality,
            )

        hidden_new = hidden_new + self.mlp(self.mlp_norm(hidden_new))

        if self.engram is not None and memory_vector is not None and memory_quality is not None:
            hidden_new = self.engram(hidden_new, memory_vector, memory_quality)

        return hidden_new, new_k, new_v


# ===========================================================================
# 7. Top-level transformer (from QASP/models/qasp_transformer.py)
# ===========================================================================

LayerLossFn = Callable[[int, Tensor], Tensor]


class QASPTransformer(nn.Module):
    """Minimal runnable transformer with QASP AttnRes / Engram hooks.

    Use :meth:`forward` (or :meth:`prefill` for logits+KV cache) for behaviour
    aligned with the paper's **full-sequence** value-weighted AttnRes.  Use
    :meth:`step` only when you need incremental decoding and accept the
    prefix-based block statistics described in the module docstring.
    """

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = (
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if not config.use_rope
            else None
        )
        self.layers = nn.ModuleList([QASPLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rope: RotaryEmbedding | None
        if config.use_rope:
            if config.hidden_size % config.num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads for RoPE.")
            head_dim = config.hidden_size // config.num_heads
            self.rope = RotaryEmbedding(head_dim, max_position_embeddings=config.max_position_embeddings)
        else:
            self.rope = None

        self.engram_memory: NgramMemory | None
        if config.use_engram:
            self.engram_memory = NgramMemory(
                table_size=config.engram_table_size,
                hidden_size=config.hidden_size,
                n_gram=config.engram_n_gram,
            )
        else:
            self.engram_memory = None

        self.kv_codec: RaBitQCodec | None
        if config.quantize_kv:
            if config.hidden_size % config.num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads for KV quantization.")
            head_dim = config.hidden_size // config.num_heads
            self.kv_codec = RaBitQCodec(head_dim, seed=config.kv_codec_seed)
        else:
            self.kv_codec = None

    def forward(self, input_ids: Tensor) -> Tensor:
        """Full-sequence forward pass; canonical definition for AttnRes + QASP."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError("input sequence length exceeds max_position_embeddings")

        hidden_states = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            hidden_states = hidden_states + self.position_embedding(positions)

        rope_cos = rope_sin = None
        if self.rope is not None:
            rope_cos, rope_sin = self.rope(hidden_states, seq_len)

        memory_vector: Tensor | None = None
        memory_quality: Tensor | None = None
        if self.config.use_engram and self.engram_memory is not None:
            memory_vector, memory_quality = self.engram_memory.batch_lookup(input_ids)

        per_token_quality: Tensor | None = None
        if self.config.use_attnres or self.config.use_engram:
            per_token_quality = compute_quality_score(
                hidden_states,
                low_pass_ratio=0.25,
                window_size=self.config.quality_window_size,
            )

        for module in self.layers:
            layer = cast(QASPLayer, module)
            block_repr = block_quality = None
            if self.config.use_attnres:
                block_repr, block_quality = compute_block_representations(
                    hidden_states,
                    num_blocks=self.config.attnres_blocks,
                    quality_window_size=self.config.quality_window_size,
                    per_token_quality=per_token_quality,
                )

            hidden_states = layer(
                hidden_states,
                block_representations=block_repr,
                block_quality=block_quality,
                memory_vector=memory_vector,
                memory_quality=memory_quality,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        hidden_states = self.norm(hidden_states)

        if self.config.use_engram and self.engram_memory is not None:
            self.engram_memory.batch_write(
                input_ids=input_ids,
                vectors=hidden_states,
                qualities=per_token_quality,
            )

        return cast(Tensor, self.lm_head(hidden_states))

    @torch.no_grad()
    def prefill(self, input_ids: Tensor) -> tuple[Tensor, KVCache]:
        """Run the full forward pass while capturing per-layer K/V caches.

        Uses the same block pooling as :meth:`forward` (full-sequence, paper
        Path~A).  Returns ``(logits, cache)`` where ``logits`` matches
        :meth:`forward` shape ``[B, T, V]`` and ``cache`` is populated for
        :meth:`step`.  Subsequent :meth:`step` calls use prefix statistics for
        AttnRes; see module docstring.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError("input sequence length exceeds max_position_embeddings")

        hidden_states = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            hidden_states = hidden_states + self.position_embedding(positions)

        rope_cos = rope_sin = None
        if self.rope is not None:
            rope_cos, rope_sin = self.rope(hidden_states, seq_len)

        memory_vector = memory_quality = None
        if self.config.use_engram and self.engram_memory is not None:
            memory_vector, memory_quality = self.engram_memory.batch_lookup(input_ids)

        cache = KVCache.from_input_ids(input_ids, num_layers=len(self.layers))

        per_token_quality: Tensor | None = None
        if self.config.use_attnres or self.config.use_engram:
            per_token_quality = compute_quality_score(
                hidden_states,
                low_pass_ratio=0.25,
                window_size=self.config.quality_window_size,
            )

        for idx, module in enumerate(self.layers):
            layer = cast(QASPLayer, module)
            cache.layer_inputs[idx] = hidden_states

            block_repr = block_quality = None
            if self.config.use_attnres:
                block_repr, block_quality = compute_block_representations(
                    hidden_states,
                    num_blocks=self.config.attnres_blocks,
                    quality_window_size=self.config.quality_window_size,
                    per_token_quality=per_token_quality,
                )

            hidden_states, k, v = layer.forward_with_cache(
                hidden_states,
                block_representations=block_repr,
                block_quality=block_quality,
                memory_vector=memory_vector,
                memory_quality=memory_quality,
                kv_codec=self.kv_codec,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            cache.layer_keys[idx] = k
            cache.layer_values[idx] = v

        hidden_states = self.norm(hidden_states)

        if self.config.use_engram and self.engram_memory is not None:
            self.engram_memory.batch_write(
                input_ids=input_ids,
                vectors=hidden_states,
                qualities=per_token_quality,
            )

        logits = cast(Tensor, self.lm_head(hidden_states))
        cache.per_token_quality = per_token_quality
        return logits, cache

    @torch.no_grad()
    def step(self, last_token: Tensor, cache: KVCache) -> Tensor:
        """Incrementally decode one token using cached K/V; returns ``[B, V]`` logits.

        When ``use_attnres`` is True, block summaries are computed from the
        concatenated prefix history, **not** from a hypothetical full forward over
        the extended sequence.          This is an implementation choice for streaming
        decode and is **not** claimed to match the paper's full-sequence
        operator at every position (see QASP paper, Value-Weighted AttnRes,
        canonical evaluation semantics).
        """

        if last_token.ndim != 2 or last_token.shape[1] != 1:
            raise ValueError("last_token must have shape [B, 1]")
        if cache.num_layers != len(self.layers):
            raise ValueError("cache was not initialised for this model's layer count")
        if last_token.shape[0] != cache.batch_size:
            raise ValueError("last_token batch size must match cache batch size")

        new_position = cache.seq_len
        if new_position + 1 > self.config.max_position_embeddings:
            raise ValueError("cache length would exceed max_position_embeddings")

        cache.append(last_token)

        batch_size = last_token.size(0)
        hidden = self.token_embedding(last_token)
        if self.position_embedding is not None:
            pos_idx = torch.full(
                (batch_size, 1),
                new_position,
                dtype=torch.long,
                device=last_token.device,
            )
            hidden = hidden + self.position_embedding(pos_idx)

        rope_cos = rope_sin = None
        if self.rope is not None:
            rope_cos, rope_sin = self.rope(hidden, new_position + 1)
            # Slice to the single position we need
            rope_cos = rope_cos[:, :, new_position : new_position + 1, :]
            rope_sin = rope_sin[:, :, new_position : new_position + 1, :]

        memory_vec_new: Optional[Tensor] = None
        memory_qual_new: Optional[Tensor] = None
        if self.config.use_engram and self.engram_memory is not None:
            mem_vec_full, mem_qual_full = self.engram_memory.batch_lookup(cache.input_ids)
            memory_vec_new = mem_vec_full[:, -1:, :]
            memory_qual_new = mem_qual_full[:, -1:]

        # Ponder-gated quality: decide whether to compute quality this step
        should_compute_quality = True
        if self.config.gate_quality_computation and cache.last_logits is not None:
            gate = PonderGate(
                entropy_threshold=0.8,
                confidence_threshold=0.6,
            )
            should_compute_quality = gate.should_adapt(cache.last_logits)

        for idx, module in enumerate(self.layers):
            layer = cast(QASPLayer, module)
            previous_inputs = cache.layer_inputs[idx]
            if previous_inputs is None:
                layer_input_history = hidden
            else:
                layer_input_history = torch.cat([previous_inputs, hidden], dim=1)
            cache.layer_inputs[idx] = layer_input_history

            block_repr = block_quality = None
            if self.config.use_attnres:
                if should_compute_quality:
                    block_repr, block_quality = compute_block_representations(
                        layer_input_history,
                        num_blocks=self.config.attnres_blocks,
                        quality_window_size=self.config.quality_window_size,
                    )
                else:
                    # Gate blocked: skip AttnRes for this step to avoid stale quality
                    pass

            hidden, new_k, new_v = layer.step(
                hidden,
                cached_k=cache.layer_keys[idx],
                cached_v=cache.layer_values[idx],
                block_representations=block_repr,
                block_quality=block_quality,
                memory_vector=memory_vec_new,
                memory_quality=memory_qual_new,
                kv_codec=self.kv_codec,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            cache.layer_keys[idx] = new_k
            cache.layer_values[idx] = new_v

        hidden = self.norm(hidden)
        logits = cast(Tensor, self.lm_head(hidden).squeeze(1))

        # Store logits for next step's gate decision
        if self.config.gate_quality_computation:
            cache.last_logits = logits.clone()

        return logits

    def adapt_at_test_time(
        self,
        loss_fn_for_layer: LayerLossFn,
        logits: Tensor,
        quality_scores: Optional[Tensor] = None,
        hidden_states: Optional[Tensor] = None,
        qasp_config: Optional[QASPConfig] = None,
    ) -> bool:
        """Run the paper's ponder-gated QASP update on every layer's ``W_ℓ``.

        Returns ``True`` if the ponder gate fired and adaptation ran,
        ``False`` otherwise. When it runs, each ``layer.stiefel_query`` is
        replaced with the Stiefel-projected result of
        :func:`adn.qasp.matrix_qasp.matrix_qasp_update`.

        If ``quality_scores`` is not provided but ``hidden_states`` is, quality
        scores are computed lazily inside this method **only when the ponder
        gate fires**, avoiding the FFT cost on the no-adapt path.
        """

        cfg = qasp_config or QASPConfig()
        gate = PonderGate(
            entropy_threshold=cfg.entropy_threshold,
            confidence_threshold=cfg.confidence_threshold,
        )
        if not gate.should_adapt(logits):
            return False

        resolved_quality = quality_scores
        if resolved_quality is None and hidden_states is not None:
            resolved_quality = compute_quality_score(
                hidden_states,
                low_pass_ratio=cfg.low_pass_ratio,
            )

        for idx, module in enumerate(self.layers):
            layer = cast(QASPLayer, module)
            current = cast(Tensor, layer.stiefel_query.data)

            def layer_loss(w: Tensor, _idx: int = idx) -> Tensor:
                return loss_fn_for_layer(_idx, w)

            updated = matrix_qasp_update(
                matrix=current,
                loss_fn=layer_loss,
                quality_scores=resolved_quality,
                step_size=cfg.step_size,
                num_adapt_steps=cfg.num_adapt_steps,
                ns_iters=cfg.ns_iters,
                eps=cfg.epsilon,
            )
            cast(Tensor, layer.stiefel_query.data).copy_(updated)
        return True


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_qasp_transformer(preset: str | None = None, **kwargs: Any) -> QASPTransformer:
    """Factory helper for quick model creation in tests and experiments.

    Args:
        preset: Optional preset name. ``"paper_1_5b"`` returns the paper's 1.5B config.
        **kwargs: Overrides applied on top of the preset (or default) config.
    """

    if preset == "paper_1_5b":
        config = QASPTransformerConfig.paper_1_5b()
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                raise TypeError(f"QASPTransformerConfig has no field {k!r}")
    else:
        config = QASPTransformerConfig(**kwargs)
    return QASPTransformer(config)
