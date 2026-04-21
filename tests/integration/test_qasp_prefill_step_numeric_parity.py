"""Numerical-parity contract between ``prefill + step`` and ``forward``.

Paper alignment (Path A)
------------------------
The QASP manuscript defines value-weighted AttnRes on a **full-sequence**
forward pass.  The implementation matches that in :meth:`QASPTransformer.forward`
and :meth:`QASPTransformer.prefill`.  :meth:`QASPTransformer.step` uses
prefix-based block statistics when AttnRes is on, so it is **not** part of the
canonical definition; parity with ``forward`` is only required where the paper
commits to it (see strict-causal configs below).

Context
-------
``QASPTransformer.step`` is designed to produce the same logits at the new
position as a fresh ``forward(extended_ids)[:, -1, :]`` would. Whether this
holds **strictly** (i.e. to floating-point noise) depends on two things:

1. **SDPA kernel selection.** ``torch.nn.functional.scaled_dot_product_attention``
   may dispatch to different optimized kernels for full-sequence causal
   attention vs. single-token attention against a cached history, which can
   introduce small numerical drift (~1e-3) even when the mathematics is
   identical. Forcing ``SDPBackend.MATH`` eliminates that drift.
2. **Module causality.** ``ValueWeightedAttnRes`` (paper Eq. 8) aggregates a
   *global* pseudo-query against block summaries computed over **all** tokens
   and broadcasts the same residual to every position. That residual at
   position ``t`` therefore depends on future tokens ``[t+1 .. T-1]``, so
   ``layer_inputs`` cached during ``prefill(T)`` cannot match what
   ``forward(T+1)`` would compute at the same positions once a new token is
   appended. This is a modelled invariant of the paper's architecture, not a
   numerical accident.

These tests pin down the contract precisely:

* Under ``use_attnres=False`` and ``use_engram=False`` (the strict-causal
  configuration), ``prefill + step`` must match ``forward`` to ~1e-5 under the
  MATH SDPA backend, even across multiple autoregressive steps.
* Under ``use_attnres=True`` (the default), a *measurable* gap is expected and
  is asserted here as a regression guard — if AttnRes is ever refactored to be
  causally consistent, this test will flag the behavioural change and the
  contract above should be strengthened.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from QASP.models import create_qasp_transformer


def _build_causal_strict_model() -> torch.nn.Module:
    """Tiny transformer with AttnRes / Engram disabled so step == forward."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        use_attnres=False,
        use_engram=False,
    )
    model.eval()
    return model


def _build_attnres_model() -> torch.nn.Module:
    """Tiny transformer with AttnRes (paper-default) to probe the known gap."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
        use_attnres=True,
        use_engram=True,
    )
    model.eval()
    return model


def test_math_backend_strict_parity_single_step_without_attnres() -> None:
    """MATH backend + no AttnRes ⇒ ``step`` matches ``forward`` to ~1e-5."""

    torch.manual_seed(0)
    model = _build_causal_strict_model()

    prefix = torch.tensor([[3, 14, 1, 5, 9]], dtype=torch.long)
    new_token = torch.tensor([[7]], dtype=torch.long)
    extended = torch.cat([prefix, new_token], dim=1)

    with sdpa_kernel(SDPBackend.MATH):
        _, cache = model.prefill(prefix)
        step_logits = model.step(new_token, cache)
        reference = model(extended)[:, -1, :]

    assert step_logits.shape == reference.shape
    assert torch.allclose(step_logits, reference, atol=1e-5, rtol=1e-5), (
        "prefill+step must match forward(extended)[:, -1] under MATH backend "
        "when AttnRes/Engram are disabled (strict-causal config); "
        f"max abs diff = {(step_logits - reference).abs().max().item():.2e}"
    )


def test_math_backend_strict_parity_multi_step_without_attnres() -> None:
    """Multiple consecutive ``step`` calls must stay within ~1e-5 of forward."""

    torch.manual_seed(1)
    model = _build_causal_strict_model()

    prefix = torch.tensor([[2, 11, 4]], dtype=torch.long)
    new_tokens = [
        torch.tensor([[7]], dtype=torch.long),
        torch.tensor([[13]], dtype=torch.long),
        torch.tensor([[21]], dtype=torch.long),
    ]

    with sdpa_kernel(SDPBackend.MATH):
        _, cache = model.prefill(prefix)

        running_ids = prefix.clone()
        for tok in new_tokens:
            step_logits = model.step(tok, cache)
            running_ids = torch.cat([running_ids, tok], dim=1)
            reference = model(running_ids)[:, -1, :]

            assert torch.allclose(step_logits, reference, atol=1e-5, rtol=1e-5), (
                "Multi-step incremental decoding must stay within 1e-5 of "
                "full forward under MATH + strict-causal config; "
                f"step at length {running_ids.shape[1]} diverged by "
                f"{(step_logits - reference).abs().max().item():.2e}"
            )


def test_strict_parity_also_holds_with_kv_quantization_off_default_backend() -> None:
    """Default backend + strict-causal config should already be ~1e-6."""

    torch.manual_seed(2)
    model = _build_causal_strict_model()

    prefix = torch.tensor([[8, 2, 19, 3, 4]], dtype=torch.long)
    new_token = torch.tensor([[5]], dtype=torch.long)
    extended = torch.cat([prefix, new_token], dim=1)

    with torch.no_grad():
        _, cache = model.prefill(prefix)
        step_logits = model.step(new_token, cache)
        reference = model(extended)[:, -1, :]

    assert torch.allclose(step_logits, reference, atol=1e-5, rtol=1e-5), (
        "Even without forcing MATH backend, a strict-causal model should match "
        f"to 1e-5 on CPU; got {(step_logits - reference).abs().max().item():.2e}"
    )


def test_attnres_introduces_measurable_prefill_step_drift_by_design() -> None:
    """Regression guard: AttnRes aggregates globally, so drift must be > 1e-3.

    If this test ever fails with a *smaller* diff, it means AttnRes became
    causally consistent between ``prefill`` and ``forward``. That is a real
    behavioural change and the strict-parity tests above should be promoted
    to cover AttnRes too.
    """

    torch.manual_seed(3)
    model = _build_attnres_model()

    prefix = torch.tensor([[3, 14, 1, 5, 9]], dtype=torch.long)
    new_token = torch.tensor([[7]], dtype=torch.long)
    extended = torch.cat([prefix, new_token], dim=1)

    with sdpa_kernel(SDPBackend.MATH):
        _, cache = model.prefill(prefix)
        step_logits = model.step(new_token, cache)
        reference = model(extended)[:, -1, :]

    diff = (step_logits - reference).abs().max().item()
    assert diff > 1e-3, (
        "AttnRes is acausal by construction (paper Eq. 8 aggregates over all "
        "block representations). A drift <= 1e-3 means the model became "
        "causally consistent — strengthen the parity contract if so. "
        f"Observed max abs diff = {diff:.2e}"
    )


def test_math_and_default_backends_agree_on_strict_causal_model() -> None:
    """MATH vs. default backend must produce identical step output for the
    strict-causal config (this is the weak form of kernel-independence)."""

    torch.manual_seed(4)
    model = _build_causal_strict_model()

    prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    new_token = torch.tensor([[6]], dtype=torch.long)

    with torch.no_grad():
        _, cache_default = model.prefill(prefix)
        step_default = model.step(new_token, cache_default)

    with sdpa_kernel(SDPBackend.MATH):
        _, cache_math = model.prefill(prefix)
        step_math = model.step(new_token, cache_math)

    assert torch.allclose(step_default, step_math, atol=1e-5, rtol=1e-5), (
        "Default and MATH SDPA backends should agree on the strict-causal "
        f"model; max abs diff = {(step_default - step_math).abs().max().item():.2e}"
    )


if __name__ == "__main__":  # pragma: no cover - manual sanity run
    pytest.main([__file__, "-v"])
