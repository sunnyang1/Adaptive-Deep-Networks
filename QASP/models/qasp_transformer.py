"""Top-level QASP transformer model and factory helper.

**Evaluation semantics.**  :meth:`QASPTransformer.forward` and
:meth:`QASPTransformer.prefill` run a full-sequence pass and, when AttnRes is
enabled, compute block summaries and ``ρ̄_m`` from the complete hidden states —
this matches the QASP paper's canonical (Path~A) definition.

:meth:`QASPTransformer.step` is an engineering API for autoregressive decoding.
With ``use_attnres=True``, block statistics are recomputed from a growing
``layer_input_history`` prefix; that information set differs from the full
sequence in the paper's equations, so **step logits are not guaranteed** to
match ``forward(cat(prefix, new_token))`` at intermediate positions.  Tests in
``tests/integration/test_qasp_prefill_step_numeric_parity.py`` document the
expected parity when AttnRes is disabled vs. the intentional gap when enabled.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import torch
import torch.nn as nn
from torch import Tensor

from QASP.adaptation.matrix_qasp import matrix_qasp_update
from QASP.adaptation.ponder_gate import PonderGate
from QASP.configs.qasp import QASPConfig
from QASP.inference.kv_cache import KVCache
from QASP.inference.rabitq import RaBitQCodec
from QASP.adaptation.quality_score import compute_quality_score
from QASP.models.components import QASPTransformerConfig, RMSNorm, compute_block_representations
from QASP.models.ngram_memory import NgramMemory
from QASP.models.qasp_layer import QASPLayer
from QASP.models.rope import RotaryEmbedding


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
        :func:`QASP.adaptation.matrix_qasp.matrix_qasp_update`.

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
