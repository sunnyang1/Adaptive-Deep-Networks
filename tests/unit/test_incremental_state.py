"""
Unit tests for IncrementalState data structure.

TDD: Red → Green → Refactor
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.incremental_state import IncrementalState, validate_state, concat_kv_cache


class TestIncrementalState:
    """Test IncrementalState dataclass."""

    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        batch_size = 1
        num_layers = 4
        num_heads = 4
        head_dim = 32
        seq_len = 10
        hidden_dim = 128

        from src.qttt.adaptation import KVCache

        kv_caches = []
        for _ in range(num_layers):
            k = torch.randn(batch_size, num_heads, seq_len, head_dim)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim)
            kv_caches.append(KVCache(k, v))

        block_reps = [torch.randn(batch_size, seq_len, hidden_dim)]
        partial_block = torch.randn(batch_size, seq_len, hidden_dim)

        return IncrementalState(
            kv_caches=kv_caches,
            block_representations=block_reps,
            partial_block=partial_block,
            seq_len=seq_len,
            num_layers=num_layers,
            num_blocks=2,
        )

    def test_initialization(self, sample_state):
        """Test state can be created."""
        assert sample_state.seq_len == 10
        assert len(sample_state.kv_caches) == 4
        assert len(sample_state.block_representations) == 1

    def test_validation_passes(self, sample_state):
        """Test validation with correct state."""
        assert validate_state(sample_state) is True

    def test_validation_fails_on_mismatched_kv(self):
        """Test validation catches mismatched KV cache count."""
        from src.qttt.adaptation import KVCache

        # Create state with wrong number of caches
        kv_caches = [KVCache(torch.randn(1, 4, 5, 32), torch.randn(1, 4, 5, 32)) for _ in range(3)]
        block_reps = [torch.randn(1, 5, 128)]
        partial = torch.randn(1, 5, 128)

        # This should fail validation (3 caches but claims 4 layers)
        try:
            state = IncrementalState(
                kv_caches=kv_caches,
                block_representations=block_reps,
                partial_block=partial,
                seq_len=5,
                num_layers=4,  # Mismatch!
                num_blocks=2,
            )
            # If we get here, validation passed when it should have failed
            assert False, "Expected validation to fail"
        except ValueError:
            pass  # Expected

    def test_validation_fails_on_negative_seq_len(self):
        """Test validation catches negative seq_len."""
        from src.qttt.adaptation import KVCache

        kv_caches = [KVCache(torch.randn(1, 4, 5, 32), torch.randn(1, 4, 5, 32)) for _ in range(4)]
        block_reps = [torch.randn(1, 5, 128)]
        partial = torch.randn(1, 5, 128)

        try:
            state = IncrementalState(
                kv_caches=kv_caches,
                block_representations=block_reps,
                partial_block=partial,
                seq_len=-1,  # Invalid!
                num_layers=4,
                num_blocks=2,
            )
            assert False, "Expected validation to fail"
        except ValueError:
            pass

    def test_get_cache_for_layer(self, sample_state):
        """Test getting cache for specific layer."""
        cache = sample_state.get_cache_for_layer(0)
        assert cache is not None
        assert cache.keys.shape == (1, 4, 10, 32)

    def test_get_cache_invalid_layer(self, sample_state):
        """Test getting cache for invalid layer raises error."""
        with pytest.raises(IndexError):
            sample_state.get_cache_for_layer(10)

    def test_append_kv_cache(self):
        """Test appending to KV cache."""
        from src.qttt.adaptation import KVCache

        k1 = torch.randn(1, 4, 10, 32)
        v1 = torch.randn(1, 4, 10, 32)
        cache = KVCache(k1, v1)

        k_new = torch.randn(1, 4, 1, 32)
        v_new = torch.randn(1, 4, 1, 32)

        updated = concat_kv_cache(cache, k_new, v_new)

        assert updated.keys.shape == (1, 4, 11, 32)
        assert updated.values.shape == (1, 4, 11, 32)

    def test_state_device_consistency(self, sample_state):
        """Test all tensors on same device."""
        devices = [cache.keys.device for cache in sample_state.kv_caches]
        devices.append(sample_state.partial_block.device)

        assert all(d == devices[0] for d in devices)


class TestIncrementalStateSerialization:
    """Test state serialization."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        from src.qttt.adaptation import KVCache

        kv_caches = [KVCache(torch.randn(1, 4, 5, 32), torch.randn(1, 4, 5, 32))]
        block_reps = [torch.randn(1, 5, 128)]
        partial = torch.randn(1, 5, 128)

        state = IncrementalState(
            kv_caches=kv_caches,
            block_representations=block_reps,
            partial_block=partial,
            seq_len=5,
            num_layers=1,
            num_blocks=1,
        )

        state_dict = state.to_dict()

        assert "seq_len" in state_dict
        assert state_dict["seq_len"] == 5
        assert "kv_caches" in state_dict

    def test_from_dict(self):
        """Test loading from dictionary."""
        # This would require tensor serialization
        pass  # Complex, skip for now
