"""
Compressed Tokenizer for Engram

Compresses vocabulary by merging semantically equivalent tokens
using Unicode normalization.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Union, List
from transformers import AutoTokenizer


class CompressedTokenizer:
    """
    Compresses tokenizer vocabulary by normalizing and merging equivalent tokens.

    Uses NFKC normalization + case folding to merge tokens like:
    - "Hello" and "hello" → same compressed ID
    - Unicode variants of same character → same compressed ID

    Args:
        tokenizer_name_or_path: HuggingFace tokenizer name or local path
    """

    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )
        self.lookup_table, self.num_new_tokens = self._build_lookup_table()

    def __len__(self) -> int:
        """Return compressed vocabulary size."""
        return self.num_new_tokens

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text using NFKC + lowercase.

        NFKC: Compatibility decomposition followed by canonical composition
        - Decomposes compatibility characters
        - Recomposes canonical equivalents
        - Lowercases for case insensitivity
        """
        import unicodedata

        # NFKC normalization
        text = unicodedata.normalize("NFKC", text)
        # Lowercase
        text = text.lower()
        return text

    def _build_lookup_table(self) -> tuple:
        """
        Build mapping from original token IDs to compressed token IDs.

        Returns:
            lookup_table: numpy array mapping old_id → new_id
            num_new_tokens: size of compressed vocabulary
        """
        old2new = {}
        key2new = {}

        vocab_size = len(self.tokenizer)

        for tid in range(vocab_size):
            # Decode token
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            # Handle replacement character (decode failure)
            if "\ufffd" in text:
                # Use token representation for special/invalid tokens
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                # Normalize text
                norm = self._normalize_text(text)
                key = norm if norm else text

            # Check if we've seen this key before
            nid = key2new.get(key)
            if nid is None:
                nid = len(key2new)
                key2new[key] = nid

            old2new[tid] = nid

        # Build lookup array
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(key2new)

    def compress(self, input_ids: Union[np.ndarray, List[int], torch.Tensor]) -> np.ndarray:
        """
        Compress input IDs using the lookup table.

        Args:
            input_ids: Original token IDs, shape [...]

        Returns:
            Compressed token IDs, same shape as input
        """
        arr = np.asarray(input_ids, dtype=np.int64)

        # Handle negative values (typically -100 for ignore_index)
        pos_mask = arr >= 0
        out = arr.copy()

        if pos_mask.any():
            valid_ids = arr[pos_mask]
            # Clip to valid range to avoid index errors
            valid_ids = np.clip(valid_ids, 0, len(self.lookup_table) - 1)
            out[pos_mask] = self.lookup_table[valid_ids]

        return out

    def __call__(self, input_ids: Union[np.ndarray, List[int], torch.Tensor]) -> np.ndarray:
        """Alias for compress method."""
        return self.compress(input_ids)

    def get_compression_ratio(self) -> float:
        """Return compression ratio (original / compressed)."""
        original_size = len(self.tokenizer)
        compressed_size = self.num_new_tokens
        return original_size / compressed_size
