"""
Synthetic needle-in-a-haystack dataset for long-context retrieval smoke tests.

Each sample is a long filler prompt with a unique \"needle\" fact buried at a
configurable depth; the model is asked to recall the fact from the prompt.
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass
from typing import List, Literal


DepthDistribution = Literal["uniform", "early", "late"]


@dataclass
class NeedleSample:
    """One needle task instance."""

    needle_depth_percent: float
    secret: str
    _prompt: str

    def format_prompt(self) -> str:
        return self._prompt

    def evaluate(self, generated_text: str) -> dict:
        """Return whether the generated text contains the secret (case-insensitive)."""
        gen = (generated_text or "").strip()
        ok = self.secret.lower() in gen.lower()
        return {
            "correct": ok,
            "score": 1.0 if ok else 0.0,
        }


class NeedleDataset:
    """Builds needle-in-haystack prompts for a target token/character budget."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def create_dataset(
        self,
        context_tokens: int,
        num_samples: int,
        depth_distribution: DepthDistribution = "uniform",
    ) -> List[NeedleSample]:
        """
        Args:
            context_tokens: Target approximate length of the prompt in characters
                (callers using per-char tokenization treat this as token count).
            num_samples: Number of independent samples.
            depth_distribution: Where to bury the needle in the haystack.
        """
        samples: List[NeedleSample] = []
        for _ in range(num_samples):
            depth_pct = self._sample_depth(depth_distribution)
            secret = self._random_secret()
            prompt = self._build_prompt(context_tokens, depth_pct, secret)
            samples.append(
                NeedleSample(
                    needle_depth_percent=depth_pct,
                    secret=secret,
                    _prompt=prompt,
                )
            )
        return samples

    def _sample_depth(self, dist: DepthDistribution) -> float:
        if dist == "uniform":
            return 100.0 * self.rng.random()
        if dist == "early":
            return 100.0 * self.rng.uniform(0.0, 0.33)
        if dist == "late":
            return 100.0 * self.rng.uniform(0.66, 1.0)
        return 100.0 * self.rng.random()

    def _random_secret(self, length: int = 10) -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "".join(self.rng.choice(alphabet) for _ in range(length))

    def _build_prompt(self, context_tokens: int, depth_percent: float, secret: str) -> str:
        needle_line = (
            f"The secret passcode you must remember is: {secret}. "
            "Do not forget this passcode.\n\n"
        )
        question = (
            "\n\nQuestion: What is the secret passcode? "
            "Answer with the passcode only.\n"
        )

        budget = max(256, context_tokens)
        # Reserve space for needle + question + small margin
        overhead = len(needle_line) + len(question) + 64
        filler_target = max(128, budget - overhead)

        # Split filler so the needle sits at depth_percent along the haystack
        frac = max(0.01, min(0.99, depth_percent / 100.0))
        pre_len = int(filler_target * frac)
        post_len = filler_target - pre_len

        pre = self._filler_block(pre_len, seed=b"pre")
        post = self._filler_block(post_len, seed=b"post")

        return pre + needle_line + post + question

    def _filler_block(self, length: int, seed: bytes) -> str:
        """Repeatable pseudo-text of approximately `length` characters."""
        if length <= 0:
            return ""
        unit = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        )
        h = hashlib.sha256(seed).hexdigest()
        out = []
        total = 0
        i = 0
        while total < length:
            chunk = f"[{h[:8]}:{i}] " + unit
            out.append(chunk)
            total += len(chunk)
            i += 1
        text = "".join(out)
        return text[:length]

