"""Shape tests for the minimal QASP model stack."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QASP.models import ValueWeightedAttnRes, ValueWeightedEngram, create_qasp_transformer


def test_qasp_transformer_forward_output_shape() -> None:
    """Transformer forward should return logits [B, T, V]."""

    model = create_qasp_transformer(
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        attnres_blocks=4,
    )
    input_ids = torch.randint(0, 128, (3, 11))

    logits = model(input_ids)

    assert logits.shape == (3, 11, 128)


def test_value_weighted_engram_fuse_shape() -> None:
    """Engram fusion should preserve hidden state shape."""

    module = ValueWeightedEngram(hidden_size=32)
    hidden = torch.randn(2, 7, 32)
    memory_vec = torch.randn(2, 32)
    memory_quality = torch.randn(2)

    fused = module(hidden, memory_vec, memory_quality)

    assert fused.shape == hidden.shape


def test_value_weighted_engram_gate_scalar_tensor_shape() -> None:
    """Scalar tensor gate should broadcast over [B, T, D]."""

    module = ValueWeightedEngram(hidden_size=16)
    hidden = torch.randn(2, 5, 16)
    memory_vec = torch.randn(2, 16)
    memory_quality = torch.randn(2)
    gate = torch.tensor(0.75)

    fused = module(hidden, memory_vec, memory_quality, gate=gate)

    assert fused.shape == hidden.shape


def test_value_weighted_engram_gate_batch_shape() -> None:
    """Per-batch gate [B] should broadcast over sequence and hidden dims."""

    module = ValueWeightedEngram(hidden_size=16)
    hidden = torch.randn(2, 5, 16)
    memory_vec = torch.randn(2, 16)
    memory_quality = torch.randn(2)
    gate = torch.tensor([0.5, 1.25])

    fused = module(hidden, memory_vec, memory_quality, gate=gate)

    assert fused.shape == hidden.shape


def test_value_weighted_engram_gate_batch_time_shape() -> None:
    """Per-token gate [B, T] should broadcast over hidden dim."""

    module = ValueWeightedEngram(hidden_size=16)
    hidden = torch.randn(2, 5, 16)
    memory_vec = torch.randn(2, 16)
    memory_quality = torch.randn(2)
    gate = torch.full((2, 5), 0.8)

    fused = module(hidden, memory_vec, memory_quality, gate=gate)

    assert fused.shape == hidden.shape


def test_value_weighted_engram_invalid_gate_shape_raises() -> None:
    """Unsupported gate shape should raise ValueError."""

    module = ValueWeightedEngram(hidden_size=16)
    hidden = torch.randn(2, 5, 16)
    memory_vec = torch.randn(2, 16)
    memory_quality = torch.randn(2)
    invalid_gate = torch.ones(2, 5, 1)

    with pytest.raises(ValueError, match="gate"):
        module(hidden, memory_vec, memory_quality, gate=invalid_gate)


def test_value_weighted_attnres_output_shape() -> None:
    """AttnRes aggregation should return [B, T, D]."""

    module = ValueWeightedAttnRes(hidden_size=48)
    hidden = torch.randn(2, 9, 48)
    block_repr = torch.randn(2, 5, 48)
    quality_scores = torch.randn(2, 5)

    residual = module(hidden, block_repr, quality_scores)

    assert residual.shape == hidden.shape


def test_value_weighted_attnres_matches_paper_eq8() -> None:
    """Pre-projection pooled vector follows softmax((w_l . B_m) * rho_m / sqrt(d))."""
    import math

    torch.manual_seed(0)
    hidden_size = 16
    module = ValueWeightedAttnRes(hidden_size=hidden_size)

    with torch.no_grad():
        module.pseudo_query.copy_(torch.randn(hidden_size))
        module.output_proj.weight.copy_(torch.eye(hidden_size))
        module.output_proj.bias.zero_()

    hidden = torch.zeros(1, 1, hidden_size)
    block_repr = torch.randn(1, 4, hidden_size)
    block_quality = torch.tensor([[0.9, 0.1, 0.5, 0.7]])

    residual = module(hidden, block_repr, block_quality).squeeze()

    affinity = block_repr[0] @ module.pseudo_query
    scores = affinity * block_quality[0] / math.sqrt(hidden_size)
    expected_weights = torch.softmax(scores, dim=-1)
    expected_pooled = (expected_weights.unsqueeze(-1) * block_repr[0]).sum(dim=0)

    assert torch.allclose(residual, expected_pooled, atol=1e-5)


def test_value_weighted_attnres_zero_quality_block_is_suppressed() -> None:
    """A block with rho=0 must not dominate the softmax over the others."""
    import math

    torch.manual_seed(1)
    hidden_size = 8
    module = ValueWeightedAttnRes(hidden_size=hidden_size)

    with torch.no_grad():
        module.pseudo_query.copy_(torch.ones(hidden_size))
        module.output_proj.weight.copy_(torch.eye(hidden_size))
        module.output_proj.bias.zero_()

    hidden = torch.zeros(1, 1, hidden_size)
    block_repr = torch.zeros(1, 3, hidden_size)
    block_repr[0, 0] = 100.0
    block_repr[0, 1] = 1.0
    block_repr[0, 2] = 1.0
    block_quality = torch.tensor([[0.0, 0.5, 0.5]])

    residual = module(hidden, block_repr, block_quality).squeeze()

    affinity = block_repr[0] @ module.pseudo_query
    scores = affinity * block_quality[0] / math.sqrt(hidden_size)
    weights = torch.softmax(scores, dim=-1)
    assert weights[0] < 0.5
    assert torch.allclose(residual, (weights.unsqueeze(-1) * block_repr[0]).sum(dim=0), atol=1e-4)


def test_qasp_transformer_forward_without_attnres_or_engram() -> None:
    """Transformer should run with both optional hooks disabled."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=32,
        use_attnres=False,
        use_engram=False,
    )
    input_ids = torch.randint(0, 64, (2, 6))

    logits = model(input_ids)

    assert logits.shape == (2, 6, 64)


def test_gqa_kv_heads_smaller_than_query_heads() -> None:
    """GQA should reduce K/V param count and still produce correct output shape."""

    from QASP.models.components import CausalSelfAttention, QASPTransformerConfig

    config = QASPTransformerConfig(
        hidden_size=64,
        num_heads=8,
        num_key_value_heads=2,
    )
    attn = CausalSelfAttention(config)
    hidden = torch.randn(2, 5, 64)

    out = attn(hidden)
    assert out.shape == (2, 5, 64)

    # K/V projection should have fewer parameters
    assert attn.k_proj.weight.shape[0] == config.num_key_value_heads * (config.hidden_size // config.num_heads)
    assert attn.v_proj.weight.shape[0] == config.num_key_value_heads * (config.hidden_size // config.num_heads)


def test_gqa_forward_with_cache_shape() -> None:
    """GQA forward_with_cache should return K/V with reduced head count."""

    from QASP.models.components import CausalSelfAttention, QASPTransformerConfig

    config = QASPTransformerConfig(
        hidden_size=64,
        num_heads=8,
        num_key_value_heads=2,
    )
    attn = CausalSelfAttention(config)
    hidden = torch.randn(2, 5, 64)

    out, k, v = attn.forward_with_cache(hidden)
    assert out.shape == (2, 5, 64)
    assert k.shape == (2, 2, 5, 8)  # [B, H_kv, T, d_h]
    assert v.shape == (2, 2, 5, 8)


def test_gqa_step_shape() -> None:
    """GQA step should accumulate K/V with reduced head count."""

    from QASP.models.components import CausalSelfAttention, QASPTransformerConfig

    config = QASPTransformerConfig(
        hidden_size=64,
        num_heads=8,
        num_key_value_heads=2,
    )
    attn = CausalSelfAttention(config)
    hidden = torch.randn(2, 1, 64)

    out, k, v = attn.step(hidden, cached_k=None, cached_v=None)
    assert out.shape == (2, 1, 64)
    assert k.shape == (2, 2, 1, 8)
    assert v.shape == (2, 2, 1, 8)

    # Second step should concatenate along time dim
    out2, k2, v2 = attn.step(hidden, cached_k=k, cached_v=v)
    assert k2.shape == (2, 2, 2, 8)
    assert v2.shape == (2, 2, 2, 8)


def test_rope_changes_qk_phases() -> None:
    """RoPE should rotate Q/K embeddings and produce different logits than no-RoPE."""

    from QASP.models.components import CausalSelfAttention, QASPTransformerConfig

    config = QASPTransformerConfig(
        hidden_size=64,
        num_heads=4,
        use_rope=True,
    )
    attn = CausalSelfAttention(config)
    hidden = torch.randn(1, 5, 64)

    from QASP.models.rope import RotaryEmbedding, apply_rotary_pos_emb
    rope = RotaryEmbedding(dim=16, max_position_embeddings=8)
    cos, sin = rope(hidden, seq_len=5)

    out = attn(hidden, rope_cos=cos, rope_sin=sin)
    assert out.shape == (1, 5, 64)

    # Output should differ from no-RoPE path
    out_no_rope = attn(hidden, rope_cos=None, rope_sin=None)
    assert not torch.allclose(out, out_no_rope, atol=1e-3)


def test_rope_preserves_norms() -> None:
    """RoPE rotation should preserve vector norms."""

    from QASP.models.rope import RotaryEmbedding, apply_rotary_pos_emb

    rope = RotaryEmbedding(dim=16, max_position_embeddings=8)
    q = torch.randn(1, 4, 5, 16)
    k = torch.randn(1, 2, 5, 16)
    cos, sin = rope(q, seq_len=5)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


def test_transformer_forward_with_rope() -> None:
    """QASPTransformer should work with use_rope=True and no position embeddings."""

    model = create_qasp_transformer(
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        use_rope=True,
        use_attnres=False,
        use_engram=False,
    )
    input_ids = torch.randint(0, 128, (2, 11))
    logits = model(input_ids)
    assert logits.shape == (2, 11, 128)
    assert model.position_embedding is None
    assert model.rope is not None


def test_paper_1_5b_preset() -> None:
    """The paper_1_5b preset should instantiate the claimed architecture."""

    model = create_qasp_transformer(preset="paper_1_5b")
    cfg = model.config

    assert cfg.hidden_size == 2048
    assert cfg.num_layers == 24
    assert cfg.num_heads == 16
    assert cfg.num_key_value_heads == 4
    assert cfg.vocab_size == 32000
    assert cfg.max_position_embeddings == 4096
    assert cfg.use_rope is True

    params = sum(p.numel() for p in model.parameters())
    # Paper claims ~1.5B; actual count with these dims is ~1.3B (depends on exact SwiGLU sizing)
    assert 1.2e9 < params < 1.6e9, f"Expected ~1.5B params, got {params/1e9:.2f}B"
