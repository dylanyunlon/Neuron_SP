"""
Unit tests for deepspeed.runtime.microbatch_guard.

Tests the microbatch uniformity guard (fix #591 Blocker 2) without
requiring a real distributed environment.

Test matrix:
  - broadcast_uniform_microbatch_count: non-distributed fallback
  - PaddedMicrobatchIterator: real + dummy batch iteration
  - PaddedMicrobatchIterator: all-real case (no padding)
  - PaddedMicrobatchIterator: all-dummy case (edge)
  - log_microbatch_guard_stats: stats dict shape
"""

import importlib.util
import os
import sys

import pytest
import torch

# Direct-import to bypass the heavy deepspeed/__init__.py chain
# (cpuinfo, tqdm, pydantic, msgpack, einops, regex).
# The module under test only needs torch + torch.distributed.
_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "deepspeed", "runtime", "microbatch_guard.py"
)
_spec = importlib.util.spec_from_file_location("microbatch_guard", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

broadcast_uniform_microbatch_count = _mod.broadcast_uniform_microbatch_count
PaddedMicrobatchIterator = _mod.PaddedMicrobatchIterator
log_microbatch_guard_stats = _mod.log_microbatch_guard_stats


class TestBroadcastUniformMicrobatchCount:
    """Tests for broadcast_uniform_microbatch_count without dist."""

    def test_non_distributed_passthrough(self):
        """Without torch.distributed, should return local_count unchanged."""
        result = broadcast_uniform_microbatch_count(local_count=5)
        assert result == 5

    def test_zero_count(self):
        """Edge case: local_count=0 should pass through."""
        result = broadcast_uniform_microbatch_count(local_count=0)
        assert result == 0

    def test_large_count(self):
        """Large count should pass through unchanged."""
        result = broadcast_uniform_microbatch_count(local_count=10000)
        assert result == 10000


class TestPaddedMicrobatchIterator:
    """Tests for PaddedMicrobatchIterator."""

    def _make_data_iter(self, n_batches, seq_len=16, vocab_size=100):
        """Create a simple data iterator yielding (input_ids, labels) tuples."""
        batches = []
        for i in range(n_batches):
            ids = torch.randint(0, vocab_size, (1, seq_len))
            labels = torch.randint(0, vocab_size, (1, seq_len))
            batches.append((ids, labels))
        return iter(batches)

    def test_no_padding_needed(self):
        """When real_count == padded_count, no dummies should appear."""
        data_iter = self._make_data_iter(3)
        padded = PaddedMicrobatchIterator(
            real_iter=data_iter,
            real_count=3,
            padded_count=3,
            seq_len=16,
            vocab_size=100,
            device=torch.device("cpu"),
        )
        results = list(padded)
        assert len(results) == 3
        for ids, labels, is_dummy in results:
            assert is_dummy is False
            assert ids.shape == (1, 16)

    def test_with_padding(self):
        """When padded_count > real_count, extra iterations are dummy."""
        data_iter = self._make_data_iter(2)
        padded = PaddedMicrobatchIterator(
            real_iter=data_iter,
            real_count=2,
            padded_count=5,
            seq_len=16,
            vocab_size=100,
            device=torch.device("cpu"),
        )
        results = list(padded)
        assert len(results) == 5

        # First 2 are real
        for i in range(2):
            ids, labels, is_dummy = results[i]
            assert is_dummy is False

        # Last 3 are dummy
        for i in range(2, 5):
            ids, labels, is_dummy = results[i]
            assert is_dummy is True
            assert ids.shape == (1, 16)
            # Dummy labels should be -100 (CrossEntropyLoss ignore index)
            assert (labels == -100).all()

    def test_all_dummy(self):
        """Edge case: real_count=0, all iterations are dummy."""
        data_iter = iter([])
        padded = PaddedMicrobatchIterator(
            real_iter=data_iter,
            real_count=0,
            padded_count=3,
            seq_len=32,
            vocab_size=100,
            device=torch.device("cpu"),
        )
        results = list(padded)
        assert len(results) == 3
        for ids, labels, is_dummy in results:
            assert is_dummy is True

    def test_dict_format_input(self):
        """Should handle dict-format batches (tokens/labels keys)."""
        batches = [
            {"tokens": torch.randint(0, 100, (1, 16)), "labels": torch.randint(0, 100, (1, 16))}
            for _ in range(2)
        ]
        padded = PaddedMicrobatchIterator(
            real_iter=iter(batches),
            real_count=2,
            padded_count=3,
            seq_len=16,
            device=torch.device("cpu"),
        )
        results = list(padded)
        assert len(results) == 3
        # First 2 are real (dict format)
        assert results[0][2] is False
        assert results[1][2] is False
        # Last is dummy
        assert results[2][2] is True

    def test_len(self):
        """__len__ should return padded_count."""
        padded = PaddedMicrobatchIterator(
            real_iter=iter([]),
            real_count=0,
            padded_count=7,
            seq_len=16,
            device=torch.device("cpu"),
        )
        assert len(padded) == 7

    def test_dummy_tensors_reused(self):
        """Dummy tensors should be allocated once and reused."""
        padded = PaddedMicrobatchIterator(
            real_iter=iter([]),
            real_count=0,
            padded_count=3,
            seq_len=16,
            device=torch.device("cpu"),
        )
        results = list(padded)
        # Same tensor objects for all dummies
        assert results[0][0] is results[1][0]
        assert results[1][0] is results[2][0]


class TestLogMicrobatchGuardStats:
    """Tests for log_microbatch_guard_stats."""

    def test_stats_dict_keys(self):
        stats = log_microbatch_guard_stats(
            step=10, local_count=2, uniform_count=5, rank=1,
        )
        assert stats["gate/step"] == 10
        assert stats["gate/rank"] == 1
        assert stats["gate/local_microbatches"] == 2
        assert stats["gate/uniform_microbatches"] == 5
        assert stats["gate/dummy_microbatches"] == 3
        assert stats["gate/is_padded"] is True

    def test_no_padding_stats(self):
        stats = log_microbatch_guard_stats(
            step=0, local_count=4, uniform_count=4, rank=0,
        )
        assert stats["gate/dummy_microbatches"] == 0
        assert stats["gate/is_padded"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
