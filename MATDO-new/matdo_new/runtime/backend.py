from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from matdo_new.modeling.config import MATDOModelConfig
from matdo_new.modeling.matdo_model import MATDOModel
from matdo_new.repo_imports import repo_root_on_path
from matdo_new.runtime.materialize import MaterializedPolicy
from matdo_new.runtime.state import BackendResult, MATDOState

with repo_root_on_path():
    from src.models.adaptive_transformer import AdaptiveTransformer
    from src.models.configs import ModelConfig
    from src.models.incremental_state import IncrementalState, concat_kv_cache, create_empty_state

if TYPE_CHECKING:
    from matdo_new.modeling.blocks import RuntimeHandles


@dataclass(frozen=True)
class AdaptiveTransformerBackendCache:
    """Cache envelope produced by the real ADN runtime bridge."""

    incremental_state: IncrementalState
    backend_id: int
    sequence_position: int
    policy_snapshot: MaterializedPolicy | None = None
    runtime_handles: RuntimeHandles | None = None
    active_scope_state: tuple[object, ...] = ()
    device: str | None = None
    supports_incremental_decode: bool = False
    active_scope_block_count: int = 0
    qttt_loss_history: tuple[float, ...] = ()


class AdaptiveTransformerRuntimeBackend:
    """RuntimeBackend bridge backed by the repository's AdaptiveTransformer."""

    def __init__(
        self,
        model: AdaptiveTransformer,
        *,
        runtime_model: MATDOModel | None = None,
        device: str | torch.device | None = None,
        use_attnres: bool = True,
        use_engram: bool | None = None,
    ) -> None:
        self.model = model.eval()
        self.runtime_model = runtime_model

        resolved_device = torch.device(device) if device is not None else next(model.parameters()).device
        self.device = resolved_device
        self.model.to(self.device)

        self.use_attnres = bool(use_attnres)
        if use_engram is None:
            use_engram = bool(getattr(model.config, "use_engram", False))
        self.use_engram = bool(use_engram)

    @classmethod
    def from_backend_config(
        cls,
        backend_config: ModelConfig,
        *,
        device: str | torch.device | None = None,
        use_attnres: bool = True,
        use_engram: bool | None = None,
    ) -> "AdaptiveTransformerRuntimeBackend":
        model = AdaptiveTransformer(backend_config)
        return cls(
            model,
            device=device,
            use_attnres=use_attnres,
            use_engram=use_engram,
        )

    @classmethod
    def from_model_config(
        cls,
        config: MATDOModelConfig,
        *,
        device: str | torch.device | None = None,
    ) -> "AdaptiveTransformerRuntimeBackend":
        backend_config = config.build_backend_config()
        if config.use_engram and getattr(backend_config, "engram_config", None) is None:
            raise ValueError(
                "Engram is not supported by AdaptiveTransformerRuntimeBackend.from_model_config() "
                "without a real backend engram_config."
            )
        model = AdaptiveTransformer(backend_config)
        runtime_model = MATDOModel(config=config, backend=model)
        return cls(
            model,
            runtime_model=runtime_model,
            device=device,
            use_attnres=config.use_attnres,
            use_engram=config.use_engram,
        )

    def forward(
        self,
        token_ids: Sequence[int],
        *,
        policy: MaterializedPolicy | None = None,
    ) -> BackendResult:
        return self._prefill_sequence(token_ids, policy=policy)

    def forward_step(
        self,
        token_ids: Sequence[int],
        *,
        state: MATDOState,
        policy: MaterializedPolicy | None = None,
    ) -> BackendResult:
        step_tokens = tuple(int(token_id) for token_id in token_ids)
        if not step_tokens:
            raise ValueError("token_ids must not be empty")

        resolved_cache = self._resolve_backend_cache(state.cache)
        resolved_policy = (
            policy if policy is not None
            else (resolved_cache.policy_snapshot if resolved_cache is not None else state.policy)
        )

        if not self._can_decode_incrementally(resolved_cache, resolved_policy):
            return self._replay_decode(
                state=state,
                step_tokens=step_tokens,
                policy=resolved_policy,
            )

        cache = resolved_cache
        logits: torch.Tensor | None = None
        for step_token in step_tokens:
            logits, cache = self._decode_single_token(
                step_token,
                cache=cache,
                policy=resolved_policy,
            )

        return BackendResult(
            logits=logits,
            cache=cache,
            submitted_token_count=len(step_tokens),
            used_incremental_cache=True,
        )

    def _prefill_sequence(
        self,
        token_ids: Sequence[int],
        *,
        policy: MaterializedPolicy | None,
    ) -> BackendResult:
        tokens = tuple(int(token_id) for token_id in token_ids)
        if not tokens:
            raise ValueError("token_ids must not be empty")
        self._validate_prefill_semantics(policy)

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        resolved_handles = self._resolve_runtime_handles(policy)

        adapted_query: torch.Tensor | None = None
        qttt_loss_history: tuple[float, ...] = ()
        kv_caches_for_forward: list | None = None

        if self._should_run_prefill_qttt(policy, resolved_handles):
            kv_caches_for_forward, adapted_query, qttt_loss_history = self._run_prefill_qttt(
                input_ids,
                runtime_handles=resolved_handles,
            )

        with torch.no_grad():
            kv_caches = (
                kv_caches_for_forward
                if kv_caches_for_forward is not None
                else self.model.get_kv_cache(input_ids)
            )
            kv_caches = self._maybe_quantize_kv_caches(
                kv_caches,
                policy=policy,
                runtime_handles=resolved_handles,
            )
            logits = self.model.forward(
                input_ids,
                use_attnres=self.use_attnres,
                use_engram=self._resolve_use_engram(policy),
                use_qttt=adapted_query is not None,
                kv_caches=kv_caches if adapted_query is not None else None,
                adapted_query=adapted_query,
            )

        incremental_state = self._create_incremental_state(
            kv_caches=kv_caches,
            seq_len=len(tokens),
        )
        self._remember_scope_event(
            resolved_handles,
            phase="prefill",
            sequence_position=len(tokens),
            token_id=tokens[-1],
        )

        return BackendResult(
            logits=logits[0, -1, :].detach().clone(),
            cache=self._build_backend_cache(
                incremental_state=incremental_state,
                policy=policy,
                runtime_handles=resolved_handles,
                qttt_loss_history=qttt_loss_history,
            ),
            submitted_token_count=len(tokens),
            used_incremental_cache=False,
        )

    def _should_run_prefill_qttt(
        self,
        policy: MaterializedPolicy | None,
        runtime_handles: RuntimeHandles | None,
    ) -> bool:
        if policy is None or policy.t_steps <= 0:
            return False
        if runtime_handles is None or runtime_handles.query_adaptation is None:
            return False
        return True

    def _run_prefill_qttt(
        self,
        input_ids: torch.Tensor,
        *,
        runtime_handles: RuntimeHandles,
    ) -> tuple[list, torch.Tensor, tuple[float, ...]]:
        """Build kv_caches + adapted query for the last token via the qTTT adapter.

        Returns (kv_caches, adapted_query [B, T, D], loss_history).
        """
        batch_size, seq_len = input_ids.shape
        hidden_dim = self.model.config.hidden_dim

        with torch.no_grad():
            kv_caches = self.model.get_kv_cache(input_ids)

        # Build the last-token query from the raw token embedding so we do not
        # have to rerun the whole forward pass.  This mirrors the pattern used
        # inside AdaptiveTransformer.generate() for qTTT bootstrapping.
        with torch.enable_grad():
            embeddings = self.model.token_embedding(input_ids)
            last_hidden = embeddings[:, -1:, :]
            query = self.model.layers[-1].attn.q_proj(last_hidden)
            seq_positions = torch.tensor(
                [seq_len - 1], dtype=torch.long, device=input_ids.device
            )
            result = runtime_handles.query_adaptation.adapt_query_projection(
                query,
                kv_caches[-1],
                seq_positions=seq_positions,
            )

        pad = torch.zeros(
            batch_size,
            seq_len - 1,
            hidden_dim,
            device=input_ids.device,
            dtype=result.adapted_query.dtype,
        )
        adapted_full = torch.cat([pad, result.adapted_query.detach()], dim=1)
        return kv_caches, adapted_full, tuple(result.loss_history)

    def _resolve_use_engram(self, policy: MaterializedPolicy | None) -> bool:
        if policy is None:
            return self.use_engram
        return bool(policy.use_engram)

    def _resolve_backend_cache(self, cache: object | None) -> AdaptiveTransformerBackendCache | None:
        if not isinstance(cache, AdaptiveTransformerBackendCache):
            return None
        if cache.backend_id != id(self):
            return None
        return cache

    def _resolve_runtime_handles(
        self,
        policy: MaterializedPolicy | None,
        cached_handles: RuntimeHandles | None = None,
        cached_policy: MaterializedPolicy | None = None,
    ) -> RuntimeHandles | None:
        if cached_handles is not None and cached_policy == policy:
            return cached_handles
        if self.runtime_model is None:
            return None
        return self.runtime_model.prepare_runtime_handles(
            policy,
            device=str(self.device),
        )

    def _create_incremental_state(
        self,
        *,
        kv_caches: Sequence[object],
        seq_len: int,
        block_representations: list[torch.Tensor] | None = None,
        partial_block: torch.Tensor | None = None,
    ) -> IncrementalState:
        config = self.model.config
        state = create_empty_state(
            batch_size=1,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.hidden_dim // config.num_heads,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            device=str(self.device),
        )
        state.kv_caches = list(kv_caches)
        state.seq_len = seq_len
        if block_representations is not None:
            state.block_representations = list(block_representations)
        if partial_block is not None:
            state.partial_block = partial_block
        return state

    def _build_backend_cache(
        self,
        *,
        incremental_state: IncrementalState,
        policy: MaterializedPolicy | None,
        runtime_handles: RuntimeHandles | None,
        active_scope_block_count: int = 0,
        qttt_loss_history: tuple[float, ...] = (),
    ) -> AdaptiveTransformerBackendCache:
        return AdaptiveTransformerBackendCache(
            incremental_state=incremental_state,
            backend_id=id(self),
            sequence_position=incremental_state.seq_len,
            policy_snapshot=policy,
            runtime_handles=runtime_handles,
            active_scope_state=(
                runtime_handles.scope_memory.blocks() if runtime_handles is not None else ()
            ),
            device=str(self.device),
            supports_incremental_decode=self._supports_incremental_decode(policy),
            active_scope_block_count=int(active_scope_block_count),
            qttt_loss_history=tuple(float(loss) for loss in qttt_loss_history),
        )

    def _should_quantize_kv(
        self,
        policy: MaterializedPolicy | None,
        runtime_handles: RuntimeHandles | None,
    ) -> bool:
        if policy is None:
            return False
        bits = getattr(policy, "quantization_bits", None)
        if bits is None or int(bits) <= 0:
            return False
        if runtime_handles is None or runtime_handles.kv_quantization is None:
            return False
        return True

    def _round_trip_kv_pair(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        runtime_handles: RuntimeHandles,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        package = runtime_handles.kv_quantization.compress(keys, values)
        restored_keys, restored_values = runtime_handles.kv_quantization.decompress(package)
        return restored_keys.to(keys.dtype), restored_values.to(values.dtype)

    def _maybe_quantize_kv_caches(
        self,
        kv_caches: Sequence[object],
        *,
        policy: MaterializedPolicy | None,
        runtime_handles: RuntimeHandles | None,
    ) -> list[object]:
        if not self._should_quantize_kv(policy, runtime_handles):
            return list(kv_caches)

        from src.qttt.adaptation import KVCache as _KVCache  # local import for lazy bridge

        quantized: list[object] = []
        for layer_cache in kv_caches:
            keys = getattr(layer_cache, "keys", None)
            values = getattr(layer_cache, "values", None)
            if keys is None or values is None:
                quantized.append(layer_cache)
                continue
            new_keys, new_values = self._round_trip_kv_pair(
                keys, values, runtime_handles=runtime_handles
            )
            quantized.append(_KVCache(new_keys, new_values))
        return quantized

    @staticmethod
    def _resolve_scope_cap(policy: MaterializedPolicy | None) -> int | None:
        """Translate policy.m_blocks into a per-step block-list cap (or None)."""
        if policy is None:
            return None
        m_blocks = getattr(policy, "m_blocks", 0)
        if m_blocks is None or int(m_blocks) <= 0:
            return None
        return int(m_blocks)

    @staticmethod
    def _cap_scope_blocks(
        blocks: list[torch.Tensor],
        scope_cap: int | None,
    ) -> list[torch.Tensor]:
        """Keep at most ``scope_cap`` most-recent block representations."""
        if scope_cap is None or len(blocks) <= scope_cap:
            return blocks
        return blocks[-scope_cap:]

    def _remember_scope_event(
        self,
        runtime_handles: RuntimeHandles | None,
        *,
        phase: str,
        sequence_position: int,
        token_id: int,
    ) -> None:
        if runtime_handles is None:
            return
        if phase == "prefill":
            runtime_handles.scope_memory.clear()
        runtime_handles.scope_memory.remember(
            {
                "phase": phase,
                "sequence_position": sequence_position,
                "token_id": int(token_id),
            }
        )

    def _decode_single_token(
        self,
        token_id: int,
        *,
        cache: AdaptiveTransformerBackendCache,
        policy: MaterializedPolicy | None,
    ) -> tuple[torch.Tensor, AdaptiveTransformerBackendCache]:
        input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        use_engram = self._resolve_use_engram(policy)
        previous_state = cache.incremental_state
        scope_cap = self._resolve_scope_cap(policy)
        runtime_handles_for_decode = cache.runtime_handles
        quantize_kv = self._should_quantize_kv(policy, runtime_handles_for_decode)

        with torch.no_grad():
            hidden = self.model.token_embedding(input_ids)
            layers_per_block = max(
                self.model.config.num_layers // max(self.model.config.num_blocks, 1),
                1,
            )
            block_representations = [hidden] if self.use_attnres else []
            partial_block = torch.zeros_like(hidden) if self.use_attnres else hidden
            updated_kv_caches = []

            for layer_idx, (layer, attnres) in enumerate(
                zip(self.model.layers, self.model.attnres_modules)
            ):
                if self.use_attnres and layer_idx > 0 and layer_idx % layers_per_block == 0:
                    block_representations.append(partial_block)
                    block_representations = self._cap_scope_blocks(block_representations, scope_cap)
                    partial_block = torch.zeros_like(hidden)

                if use_engram and layer.engram is not None:
                    hidden_expanded = hidden.unsqueeze(2)
                    engram_out = layer.engram(hidden_expanded, input_ids).squeeze(2)
                    hidden = hidden + engram_out
                    if block_representations:
                        block_representations = list(block_representations)
                        block_representations[0] = hidden
                    else:
                        partial_block = hidden

                if self.use_attnres and attnres is not None:
                    h_attn, _ = attnres(
                        block_representations,
                        partial_block,
                        use_attn=True,
                        use_mlp=False,
                    )
                else:
                    h_attn = partial_block

                normed = layer.attn_norm(h_attn)
                new_k, new_v = self._project_layer_kv(layer, normed)
                if quantize_kv:
                    new_k, new_v = self._round_trip_kv_pair(
                        new_k, new_v, runtime_handles=runtime_handles_for_decode
                    )
                previous_kv = previous_state.get_cache_for_layer(layer_idx)
                updated_kv = concat_kv_cache(previous_kv, new_k, new_v)
                updated_kv_caches.append(updated_kv)

                attn_out = layer.attn(normed, updated_kv, None)
                partial_block = partial_block + attn_out

                if self.use_attnres and attnres is not None:
                    _, h_mlp = attnres(
                        block_representations,
                        partial_block,
                        use_attn=False,
                        use_mlp=True,
                    )
                else:
                    h_mlp = partial_block

                normed = layer.mlp_norm(h_mlp)
                mlp_out = layer.mlp(normed)
                partial_block = partial_block + mlp_out
                hidden = partial_block

            active_block_count = 0
            if self.use_attnres:
                all_blocks = self._cap_scope_blocks(
                    block_representations + [partial_block],
                    scope_cap,
                )
                active_block_count = len(all_blocks)
                stacked = torch.stack(all_blocks, dim=0)
                final_attnres = self.model.attnres_modules[-1]
                normalized = final_attnres.norm_mlp(stacked)
                weights = final_attnres.pseudo_query_mlp
                attn_logits = torch.einsum("d, n b t d -> n b t", weights, normalized)
                alpha = torch.softmax(attn_logits, dim=0)
                hidden = torch.einsum("n b t, n b t d -> b t d", alpha, stacked)

            hidden = self.model.norm(hidden)
            logits = self.model.lm_head(hidden)

        updated_state = self._create_incremental_state(
            kv_caches=updated_kv_caches,
            seq_len=previous_state.seq_len + 1,
            block_representations=list(block_representations),
            partial_block=partial_block,
        )
        runtime_handles = self._resolve_runtime_handles(
            policy,
            cached_handles=cache.runtime_handles,
            cached_policy=cache.policy_snapshot,
        )
        self._remember_scope_event(
            runtime_handles,
            phase="decode",
            sequence_position=updated_state.seq_len,
            token_id=token_id,
        )
        updated_cache = self._build_backend_cache(
            incremental_state=updated_state,
            policy=policy,
            runtime_handles=runtime_handles,
            active_scope_block_count=active_block_count,
        )
        return logits[0, -1, :].detach().clone(), updated_cache

    def _project_layer_kv(
        self,
        layer: object,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, _ = hidden_states.shape
        head_dim = self.model.config.hidden_dim // self.model.config.num_heads
        keys = layer.attn.k_proj(hidden_states)
        values = layer.attn.v_proj(hidden_states)
        keys = keys.view(batch_size, token_count, self.model.config.num_heads, head_dim)
        values = values.view(batch_size, token_count, self.model.config.num_heads, head_dim)
        return keys.transpose(1, 2), values.transpose(1, 2)

    def _validate_prefill_semantics(self, policy: MaterializedPolicy | None) -> None:
        """Enforce invariants that prefill cannot currently honour.

        Engram-aware prefill now runs through the real model forward, but the
        stored KV cache is taken from a non-engram pass. That mismatch is
        harmless today because ``_supports_incremental_decode`` always forces
        replay when engram is active, so the cached KV is never read.
        If that gate changes, this validator must be tightened accordingly.
        """

        return None

    def _supports_incremental_decode(self, policy: MaterializedPolicy | None) -> bool:
        if self._resolve_use_engram(policy):
            return False
        if not self.use_attnres:
            return True
        # Native AttnRes incremental decode is opt-in via policy.m_blocks.
        # Without a policy, we keep the safer full-forward replay path so that
        # logits remain bit-for-bit identical to the legacy model forward.
        return self._resolve_scope_cap(policy) is not None

    def _policies_match(
        self,
        left: MaterializedPolicy | None,
        right: MaterializedPolicy | None,
    ) -> bool:
        return left == right

    def _can_decode_incrementally(
        self,
        cache: AdaptiveTransformerBackendCache | None,
        policy: MaterializedPolicy | None,
    ) -> bool:
        if cache is None:
            return False
        if not cache.supports_incremental_decode:
            return False
        if not self._supports_incremental_decode(policy):
            return False
        if not self._policies_match(cache.policy_snapshot, policy):
            return False
        return True

    def _replay_decode(
        self,
        *,
        state: MATDOState,
        step_tokens: tuple[int, ...],
        policy: MaterializedPolicy | None,
    ) -> BackendResult:
        replay_tokens = state.token_ids + step_tokens
        replay_result = self._prefill_sequence(replay_tokens, policy=policy)
        return BackendResult(
            logits=replay_result.logits,
            cache=replay_result.cache,
            submitted_token_count=len(replay_tokens),
            used_incremental_cache=False,
        )
