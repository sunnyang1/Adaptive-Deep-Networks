"""Real math benchmark runner for QASP models.

When a model is provided, runs exact-match evaluation on a HuggingFace math
dataset (default: ``hendrycks/competition_math``).  When ``model=None``,
falls back to a deterministic stub for CI smoke tests.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn


def _extract_final_answer(text: str) -> str:
    """Extract the final numeric or boxed answer from model output."""

    # Look for LaTeX boxed answer
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # Look for "ANSWER: X" pattern
    ans = re.search(r"ANSWER[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
    if ans:
        return ans.group(1).strip()

    # Last line fallback
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison: remove whitespace, commas, dollar signs."""

    text = text.replace("$", "").replace(",", "").replace("\\", "").strip().lower()
    # Remove fractions like \frac{1}{2} -> 1/2 (basic)
    text = re.sub(r"\\frac\{(.*?)\}\{(.*?)\}", r"\1/\2", text)
    return text


def run_math_eval(
    model: nn.Module | None = None,
    dataset_name: str = "hendrycks/competition_math",
    dataset_config: str = "main",
    split: str = "test",
    text_column: str = "problem",
    answer_column: str = "solution",
    num_samples: int | None = None,
    max_answer_tokens: int = 256,
    device: str = "cpu",
    quick: bool = True,
) -> float:
    """Run exact-match math evaluation and return accuracy in ``[0.0, 1.0]``.

    Args:
        model: A ``nn.Module`` with ``forward(input_ids) -> [B, T, V]``.  If
            ``None``, returns a deterministic stub score.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration name.
        split: Dataset split to evaluate on.
        text_column: Column containing the problem text.
        answer_column: Column containing the reference solution.
        num_samples: Maximum number of examples to evaluate.  ``None`` = all.
        max_answer_tokens: Maximum tokens to generate for the answer.
        device: torch device string.
        quick: If True, reduce ``num_samples`` to 8 for fast CI loops.

    Returns:
        Exact-match accuracy.
    """

    if quick:
        num_samples = min(num_samples or 8, 8)

    if model is None:
        # Deterministic stub for CI
        torch.manual_seed(11 if quick else 19)
        base = torch.tensor([0.49, 0.52, 0.50, 0.53], dtype=torch.float32)
        jitter = (torch.rand_like(base) - 0.5) * (0.015 if quick else 0.03)
        score = (base + jitter).clamp(min=0.0, max=1.0).mean().item()
        return float(score)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("`datasets` is required for real math evaluation. Install: pip install datasets") from exc

    model = model.to(device)
    model.eval()

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    correct = 0
    total = 0

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass  # Will use character-level fallback

    with torch.no_grad():
        for example in ds:
            if num_samples is not None and total >= num_samples:
                break

            problem = example.get(text_column, "")
            ref_solution = example.get(answer_column, "")

            if not problem or not ref_solution:
                continue

            # Tokenize prompt
            if tokenizer is not None:
                prompt_ids = tokenizer.encode(problem, return_tensors="pt", truncation=True, max_length=512)
            else:
                prompt_ids = torch.tensor([[min(ord(c), model.config.vocab_size - 1) for c in problem[:512]]])

            prompt_ids = prompt_ids.to(device)

            # Greedy generation
            generated = prompt_ids.clone()
            for _ in range(max_answer_tokens):
                logits = model(generated)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop at EOS if tokenizer knows it
                if tokenizer is not None and next_token.item() == tokenizer.eos_token_id:
                    break

            # Decode output
            if tokenizer is not None:
                output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            else:
                output_text = "".join(chr(min(t, 127)) for t in generated[0].tolist())

            pred_answer = _extract_final_answer(output_text[len(problem):])
            ref_answer = _extract_final_answer(ref_solution)

            if _normalize_answer(pred_answer) == _normalize_answer(ref_answer):
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0
