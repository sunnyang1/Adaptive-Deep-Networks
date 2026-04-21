"""
Engram Module for Adaptive Deep Networks

Explicit n-gram memory mechanism for enhanced long-range dependency modeling.
"""

from .config import EngramConfig, EngramSmallConfig, EngramMediumConfig, EngramLargeConfig
from .compressed_tokenizer import CompressedTokenizer
from .ngram_hash import NgramHashMapping, NgramHashConfig
from .embeddings import MultiHeadEmbedding, ShortConv
from .engram_module import Engram

__all__ = [
    "EngramConfig",
    "EngramSmallConfig",
    "EngramMediumConfig",
    "EngramLargeConfig",
    "CompressedTokenizer",
    "NgramHashMapping",
    "NgramHashConfig",
    "MultiHeadEmbedding",
    "ShortConv",
    "Engram",
]
