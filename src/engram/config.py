"""
Configuration for Engram module
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EngramConfig:
    """
    Configuration for Engram module.

    Engram enhances Transformer with explicit n-gram memory through
    hash-based embeddings at specific layers.

    Attributes:
        enabled: Whether to enable Engram
        engram_vocab_size: Vocabulary sizes for each n-gram level
        max_ngram_size: Maximum n-gram size (e.g., 3 for up to trigrams)
        n_embed_per_ngram: Embedding dimension per n-gram type
        n_head_per_ngram: Number of heads per n-gram type
        layer_ids: Which layers to apply Engram
        tokenizer_name_or_path: Tokenizer to use
        pad_id: Padding token ID
        seed: Random seed for hashing
        kernel_size: ShortConv kernel size
    """

    enabled: bool = False
    engram_vocab_size: List[int] = field(default_factory=lambda: [100000, 100000])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    tokenizer_name_or_path: str = "gpt2"
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4

    def __post_init__(self):
        """Validate configuration."""
        assert self.max_ngram_size >= 2, "max_ngram_size must be >= 2"
        assert (
            len(self.engram_vocab_size) == self.max_ngram_size - 1
        ), f"engram_vocab_size must have {self.max_ngram_size - 1} elements"
        assert (
            self.n_embed_per_ngram % self.n_head_per_ngram == 0
        ), "n_embed_per_ngram must be divisible by n_head_per_ngram"


# Predefined configurations
EngramSmallConfig = EngramConfig(
    enabled=True,
    engram_vocab_size=[50000, 50000],
    max_ngram_size=3,
    n_embed_per_ngram=256,
    n_head_per_ngram=4,
    layer_ids=[1, 7],
)

EngramMediumConfig = EngramConfig(
    enabled=True,
    engram_vocab_size=[100000, 100000],
    max_ngram_size=3,
    n_embed_per_ngram=512,
    n_head_per_ngram=8,
    layer_ids=[1, 15],
)

EngramLargeConfig = EngramConfig(
    enabled=True,
    engram_vocab_size=[200000, 200000, 150000],
    max_ngram_size=4,
    n_embed_per_ngram=768,
    n_head_per_ngram=12,
    layer_ids=[1, 15, 30],
)
