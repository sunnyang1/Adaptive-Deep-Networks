"""
N-gram Hash Mapping for Engram

Computes layer-specific n-gram hashes using prime-based hashing.
"""

import numpy as np
from typing import List, Dict, Set
from dataclasses import dataclass

from .compressed_tokenizer import CompressedTokenizer


def find_next_prime(start: int, seen_primes: Set[int]) -> int:
    """
    Find the next prime number after 'start' that is not in seen_primes.

    Uses simple trial division for primality testing.
    """
    candidate = start + 1
    while True:
        if candidate not in seen_primes:
            # Check if prime
            if candidate < 2:
                candidate += 1
                continue
            is_prime = True
            for i in range(2, int(candidate**0.5) + 1):
                if candidate % i == 0:
                    is_prime = False
                    break
            if is_prime:
                return candidate
        candidate += 1


@dataclass
class NgramHashConfig:
    """Configuration for NgramHashMapping."""

    engram_vocab_size: List[int]
    max_ngram_size: int
    n_head_per_ngram: int
    layer_ids: List[int]
    tokenizer_name_or_path: str
    pad_id: int
    seed: int = 0


class NgramHashMapping:
    """
    Computes layer-specific n-gram hash mappings.

    Each layer uses different random multipliers to generate distinct hashes.
    For each n-gram size and head, uses a different prime number for modulo.

    Args:
        config: NgramHashConfig with all parameters
    """

    def __init__(self, config: NgramHashConfig):
        self.config = config
        self.vocab_size_per_ngram = config.engram_vocab_size
        self.max_ngram_size = config.max_ngram_size
        self.n_head_per_ngram = config.n_head_per_ngram
        self.layer_ids = config.layer_ids
        self.pad_id = config.pad_id

        # Initialize compressed tokenizer
        self.compressed_tokenizer = CompressedTokenizer(config.tokenizer_name_or_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)

        # Convert pad_id to compressed ID
        if self.pad_id is not None:
            compressed_pad = self.compressed_tokenizer.compress(np.array([self.pad_id]))[0]
            self.pad_id = int(compressed_pad)

        # Generate layer-specific multipliers
        self.layer_multipliers = self._generate_layer_multipliers()

        # Calculate vocab sizes for each layer/ngram/head
        self.vocab_size_across_layers = self._calculate_vocab_sizes()

    def _generate_layer_multipliers(self) -> Dict[int, np.ndarray]:
        """
        Generate random multipliers for each layer.

        Each layer gets different multipliers to ensure layer-specific hashing.
        Uses PRIME_1 as a base to spread out random seeds.
        """
        PRIME_1 = 10007
        multipliers = {}

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)

        for layer_id in self.layer_ids:
            # Layer-specific seed
            base_seed = int(self.config.seed + PRIME_1 * int(layer_id))
            rng = np.random.default_rng(base_seed)

            # Generate odd multipliers
            r = rng.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            layer_multipliers = r * 2 + 1
            multipliers[layer_id] = layer_multipliers

        return multipliers

    def _calculate_vocab_sizes(self) -> Dict[int, List[List[int]]]:
        """
        Calculate vocab sizes for each layer, ngram size, and head.

        Returns:
            Dict mapping layer_id -> [
                [head1_prime, head2_prime, ...],  # for ngram=2
                [head1_prime, head2_prime, ...],  # for ngram=3
                ...
            ]
        """
        seen_primes: Set[int] = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []

            # For each n-gram size (starting from 2)
            for ngram in range(2, self.max_ngram_size + 1):
                ngram_index = ngram - 2
                current_ngram_head_sizes = []

                # Get target vocab size for this ngram
                vocab_size = self.vocab_size_per_ngram[ngram_index]
                current_prime_search_start = vocab_size - 1

                # Find a unique prime for each head
                for _ in range(self.n_head_per_ngram):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_head_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_head_sizes)

            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        """
        Compute n-gram hashes for a specific layer.

        Args:
            input_ids: Compressed token IDs, shape [B, T]
            layer_id: Which layer to compute hashes for

        Returns:
            Hash values, shape [B, T, num_heads]
            where num_heads = (max_ngram_size - 1) * n_head_per_ngram
        """
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        # Helper to shift sequence by k positions with padding
        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            # Pad at the beginning, truncate at the end
            shifted = np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id)[
                :, :T
            ]
            return shifted

        # Precompute shifted versions for each position in n-gram
        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []

        # For each n-gram size (2, 3, ..., max_ngram_size)
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]

            # Compute mixed hash using XOR of token * multiplier
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

            # Get vocab sizes for this ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            # Compute hash for each head using different primes
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        # Stack all head hashes: [B, T, num_heads]
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute n-gram hashes for all layers.

        Args:
            input_ids: Original token IDs, shape [B, T]

        Returns:
            Dict mapping layer_id -> hash array [B, T, num_heads]
        """
        # First compress the input
        compressed_ids = self.compressed_tokenizer.compress(input_ids)

        # Compute hashes for each layer
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(
                compressed_ids, layer_id=layer_id
            )

        return hash_ids_for_all_layers
