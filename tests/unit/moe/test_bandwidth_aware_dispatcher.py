# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Unit tests for deepspeed/moe/bandwidth_aware_dispatcher.py.

Mirrors Megatron tests/unit_tests/inference/test_moe_dispatching_and_routing.py
structure and intent, adapted for the DES-LOC BandwidthAwareAllGatherDispatcher.

Tests use ep_size=1 (no-op AllGather path) or mock torch.distributed to avoid
requiring a multi-GPU environment in CI.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from deepspeed.moe.bandwidth_aware_dispatcher import (
    AllGatherStrategy,
    BandwidthAwareAllGatherDispatcher,
    BandwidthAwareDispatcherConfig,
    BandwidthProbe,
    EPBatchCoordinator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensors(local_tokens: int, hidden: int, topk: int, device="cpu"):
    """Return (hidden_states, probs, routing_map) test fixtures."""
    hidden_states = torch.randn(local_tokens, hidden, dtype=torch.bfloat16, device=device)
    probs         = torch.rand(local_tokens, topk,   dtype=torch.float32,  device=device)
    routing_map   = torch.randint(0, 8, (local_tokens, topk), dtype=torch.int64, device=device)
    return hidden_states, probs, routing_map


# ---------------------------------------------------------------------------
# BandwidthProbe tests
# ---------------------------------------------------------------------------

class TestBandwidthProbe(unittest.TestCase):

    def test_cache_hit_skips_measure(self):
        probe = BandwidthProbe(src_device=0, dst_device=0, probe_mb=1.0)
        probe._cached_bw_gbps = 99.0
        self.assertEqual(probe.estimate(), 99.0)

    def test_invalidate_clears_cache(self):
        probe = BandwidthProbe(src_device=0, dst_device=0, probe_mb=1.0)
        probe._cached_bw_gbps = 99.0
        probe.invalidate()
        self.assertIsNone(probe._cached_bw_gbps)

    def test_measure_failure_returns_conservative(self):
        """If the CUDA copy fails (no GPU), _measure returns a safe fallback."""
        probe = BandwidthProbe(src_device=0, dst_device=0, probe_mb=1.0)
        with patch("torch.zeros", side_effect=RuntimeError("no CUDA")):
            result = probe._measure()
        self.assertLessEqual(result, 16.0,
            "Fallback bandwidth should be <= 16 GB/s to force CHUNKED strategy")


# ---------------------------------------------------------------------------
# Strategy selection tests
# ---------------------------------------------------------------------------

class TestStrategySelection(unittest.TestCase):

    def _dispatcher(self, bw_gbps, manual=None):
        cfg = BandwidthAwareDispatcherConfig(
            fast_link_threshold_gbps=50.0,
            manual_strategy=manual,
        )
        d = BandwidthAwareAllGatherDispatcher(
            ep_group=None,
            hidden_size=128,
            topk=2,
            config=cfg,
        )
        # override probe result without touching CUDA
        if manual is None:
            with patch.object(BandwidthProbe, "estimate", return_value=bw_gbps):
                d2 = BandwidthAwareAllGatherDispatcher(
                    ep_group=None, hidden_size=128, topk=2, config=cfg,
                )
            return d2
        return d

    def test_slow_link_selects_chunked(self):
        cfg = BandwidthAwareDispatcherConfig(fast_link_threshold_gbps=50.0)
        with patch.object(BandwidthProbe, "estimate", return_value=16.0):
            d = BandwidthAwareAllGatherDispatcher(
                ep_group=None, hidden_size=128, topk=2, config=cfg)
        self.assertEqual(d.strategy, AllGatherStrategy.CHUNKED)

    def test_fast_link_selects_bulk(self):
        cfg = BandwidthAwareDispatcherConfig(fast_link_threshold_gbps=50.0)
        with patch.object(BandwidthProbe, "estimate", return_value=128.0):
            d = BandwidthAwareAllGatherDispatcher(
                ep_group=None, hidden_size=128, topk=2, config=cfg)
        self.assertEqual(d.strategy, AllGatherStrategy.BULK)

    def test_manual_strategy_skips_probe(self):
        cfg = BandwidthAwareDispatcherConfig(manual_strategy=AllGatherStrategy.CHUNKED)
        with patch.object(BandwidthProbe, "_measure") as mock_measure:
            d = BandwidthAwareAllGatherDispatcher(
                ep_group=None, hidden_size=128, topk=2, config=cfg)
        mock_measure.assert_not_called()
        self.assertEqual(d.strategy, AllGatherStrategy.CHUNKED)

    def test_threshold_boundary_slow(self):
        # exactly at threshold → CHUNKED (not strictly greater than)
        cfg = BandwidthAwareDispatcherConfig(fast_link_threshold_gbps=50.0)
        with patch.object(BandwidthProbe, "estimate", return_value=50.0):
            d = BandwidthAwareAllGatherDispatcher(
                ep_group=None, hidden_size=128, topk=2, config=cfg)
        self.assertEqual(d.strategy, AllGatherStrategy.CHUNKED)

    def test_update_strategy_flip(self):
        cfg = BandwidthAwareDispatcherConfig(fast_link_threshold_gbps=50.0,
                                              manual_strategy=AllGatherStrategy.CHUNKED)
        d = BandwidthAwareAllGatherDispatcher(ep_group=None, hidden_size=64, topk=2, config=cfg)
        self.assertEqual(d.strategy, AllGatherStrategy.CHUNKED)
        with patch.object(BandwidthProbe, "estimate", return_value=200.0):
            flipped = d.update_strategy()
        self.assertTrue(flipped)
        self.assertEqual(d.strategy, AllGatherStrategy.BULK)

    def test_update_strategy_no_flip(self):
        cfg = BandwidthAwareDispatcherConfig(fast_link_threshold_gbps=50.0,
                                              manual_strategy=AllGatherStrategy.CHUNKED)
        d = BandwidthAwareAllGatherDispatcher(ep_group=None, hidden_size=64, topk=2, config=cfg)
        with patch.object(BandwidthProbe, "estimate", return_value=10.0):
            flipped = d.update_strategy()
        self.assertFalse(flipped)
        self.assertEqual(d.strategy, AllGatherStrategy.CHUNKED)


# ---------------------------------------------------------------------------
# ep_size=1 pass-through tests (no distributed)
# ---------------------------------------------------------------------------

class TestEpSizeOne(unittest.TestCase):
    """ep_size=1 → dispatch/combine are no-ops (identity up to dtype cast)."""

    def setUp(self):
        self.cfg = BandwidthAwareDispatcherConfig(manual_strategy=AllGatherStrategy.BULK)
        self.d = BandwidthAwareAllGatherDispatcher(
            ep_group=None, hidden_size=64, topk=2, config=self.cfg
        )

    def test_dispatch_returns_same_tensors(self):
        h, p, r = _make_tensors(8, 64, 2)
        h_g, p_g, r_g = self.d.dispatch(h, p, r)
        self.assertTrue(torch.allclose(h, h_g))
        self.assertTrue(torch.allclose(p, p_g))
        self.assertTrue(torch.allclose(r, r_g))

    def test_combine_casts_to_bf16(self):
        expert_out = torch.randn(8, 64, dtype=torch.float32)
        result = self.d.combine(expert_out)
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_dispatch_sets_tokens_per_rank(self):
        h, p, r = _make_tensors(5, 64, 2)
        self.d.dispatch(h, p, r)
        self.assertEqual(self.d._tokens_per_rank, [5])


# ---------------------------------------------------------------------------
# _pad / _compact round-trip tests
# ---------------------------------------------------------------------------

class TestPadCompact(unittest.TestCase):

    def setUp(self):
        cfg = BandwidthAwareDispatcherConfig(manual_strategy=AllGatherStrategy.BULK)
        self.d = BandwidthAwareAllGatherDispatcher(
            ep_group=None, hidden_size=16, topk=2, config=cfg
        )

    def test_pad_no_op_when_full(self):
        t = torch.randn(4, 16)
        padded = self.d._pad(t, 4)
        self.assertIs(padded, t)  # should be the same object

    def test_pad_extends_token_dim(self):
        t = torch.randn(3, 16)
        padded = self.d._pad(t, 6)
        self.assertEqual(padded.shape, (6, 16))
        # Original rows preserved
        self.assertTrue(torch.allclose(padded[:3], t))

    def test_compact_strips_padding(self):
        # Simulate 2-rank gather with ranks having [3, 5] tokens, max_tokens=5
        max_tokens = 5
        ep_size = 2
        tokens_per_rank = [3, 5]
        gathered = torch.arange(ep_size * max_tokens * 4).float().view(ep_size * max_tokens, 4)
        compacted = self.d._compact(gathered, tokens_per_rank, max_tokens)
        expected_rows = tokens_per_rank[0] + tokens_per_rank[1]
        self.assertEqual(compacted.shape[0], expected_rows)
        # rank-0 rows 0..2
        self.assertTrue(torch.allclose(compacted[:3], gathered[:3]))
        # rank-1 rows 5..9
        self.assertTrue(torch.allclose(compacted[3:], gathered[5:10]))

    def test_compact_uniform_tokens(self):
        # When all ranks have the same token count, compact == reshape then cat
        tokens_per_rank = [4, 4]
        max_tokens = 4
        gathered = torch.randn(8, 16)
        compacted = self.d._compact(gathered, tokens_per_rank, max_tokens)
        self.assertEqual(compacted.shape, (8, 16))
        self.assertTrue(torch.allclose(compacted, gathered))


# ---------------------------------------------------------------------------
# EPBatchCoordinator tests
# ---------------------------------------------------------------------------

class TestEPBatchCoordinator(unittest.TestCase):

    def test_no_group_returns_local(self):
        token_count, eager = EPBatchCoordinator.adjust(
            local_token_count=10, is_prefill=False, ep_group=None
        )
        self.assertEqual(token_count, 10)
        self.assertFalse(eager)

    def test_no_group_prefill_still_no_eager(self):
        # Without a group there is no distributed sync; prefill flag is ignored.
        token_count, eager = EPBatchCoordinator.adjust(
            local_token_count=10, is_prefill=True, ep_group=None
        )
        self.assertFalse(eager)

    def _mock_ep_group(self, world_size, rank):
        """Return a minimal mock that satisfies get_world_size / get_rank."""
        grp = MagicMock()
        return grp, world_size, rank

    def test_prefill_any_rank_forces_eager(self):
        """If any rank reports is_non_decode=1, adjust must return eager_mode=True."""
        grp = MagicMock()
        # Simulate all_reduce result: [max_token_count=10, is_non_decode=1]
        def fake_all_reduce(tensor, op, group):
            tensor[0] = 10  # max tokens
            tensor[1] = 1   # prefill present
        with patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.get_rank",       return_value=0), \
             patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            token_count, eager = EPBatchCoordinator.adjust(
                local_token_count=8, is_prefill=False, ep_group=grp
            )
        self.assertTrue(eager)
        self.assertEqual(token_count, 10)

    def test_decode_only_returns_max_count(self):
        """All ranks in decode → eager_mode=False, adjusted_count = max."""
        grp = MagicMock()
        def fake_all_reduce(tensor, op, group):
            tensor[0] = 12  # max tokens across ranks
            tensor[1] = 0   # no prefill
        with patch("torch.distributed.get_world_size", return_value=4), \
             patch("torch.distributed.get_rank",       return_value=0), \
             patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            token_count, eager = EPBatchCoordinator.adjust(
                local_token_count=10, is_prefill=False, ep_group=grp
            )
        self.assertFalse(eager)
        self.assertEqual(token_count, 12)


# ---------------------------------------------------------------------------
# Mock-distributed dispatch/combine tests (ep_size=2)
# ---------------------------------------------------------------------------

class TestMockedDistributed(unittest.TestCase):
    """Test dispatch+combine logic with mocked dist calls (no real GPU comms)."""

    def _make_dispatcher(self, strategy: AllGatherStrategy, num_chunks: int = 2):
        cfg = BandwidthAwareDispatcherConfig(
            manual_strategy=strategy,
            num_chunks=num_chunks,
        )
        # Create dispatcher with ep_group mock but bypass real probe
        grp = MagicMock()
        d = BandwidthAwareAllGatherDispatcher.__new__(BandwidthAwareAllGatherDispatcher)
        d.ep_group   = grp
        d.ep_size    = 2
        d.ep_rank    = 0
        d.hidden_size = 16
        d.topk       = 2
        d.config     = cfg
        d.strategy   = strategy
        d._local_token_count = 0
        d._tokens_per_rank   = None
        return d

    def _setup_sync_mock(self, d, tokens_per_rank):
        """Patch _sync_token_counts to return fixed per-rank counts."""
        d._sync_token_counts = MagicMock(return_value=tokens_per_rank)

    def _setup_allgather_mock(self, d, ep_size, tokens_per_rank):
        """Patch dist.all_gather_into_tensor to simulate rank-0-only gather."""
        max_tokens = max(tokens_per_rank)

        def fake_ag(output, input_tensor, group):
            # Simulate: each rank copies its slice; rank 0 fills with arange
            # For testing we just replicate input_tensor ep_size times.
            chunk_size = input_tensor.shape[0]
            for i in range(ep_size):
                output[i * chunk_size : (i + 1) * chunk_size].copy_(input_tensor)

        return fake_ag

    def test_bulk_dispatch_compact_shape(self):
        tokens_per_rank = [3, 5]  # unequal
        d = self._make_dispatcher(AllGatherStrategy.BULK)
        self._setup_sync_mock(d, tokens_per_rank)
        h, p, r = _make_tensors(3, 16, 2)  # rank-0 local

        def fake_ag_into(output, inp, group):
            chunk = inp.shape[0]
            for i in range(2):
                output[i * chunk : (i + 1) * chunk].copy_(inp)

        with patch("torch.distributed.all_gather_into_tensor", side_effect=fake_ag_into):
            h_g, p_g, r_g = d.dispatch(h, p, r)

        # compact: rank-0 contributes 3 tokens, rank-1 contributes 5 tokens;
        # but since we replicated inp, both slices come from rank-0's 3-row tensor.
        # We just validate shape: total = 3 + 5 = 8 from compact.
        # compact indexes: rank0[0:3] + rank1[5:10] from a [10, 16] buffer.
        # Since rank1 slice is max_tokens=5 rows repeated, rank1[5:10] = rows 5..9 of
        # the gathered buffer.  With our fake_ag the buffer is [h; h; padding...].
        # Shape assertion is the key invariant.
        total_tokens = sum(tokens_per_rank)
        self.assertEqual(h_g.shape[0], total_tokens)
        self.assertEqual(p_g.shape[0], total_tokens)
        self.assertEqual(r_g.shape[0], total_tokens)

    def test_combine_expand_reduce_truncate(self):
        tokens_per_rank = [4, 4]
        d = self._make_dispatcher(AllGatherStrategy.BULK)
        d._tokens_per_rank   = tokens_per_rank
        d._local_token_count = 4

        expert_output = torch.randn(8, 16, dtype=torch.float32)

        def fake_reduce_scatter(output, input_tensor, group):
            # Simulate: sum halves; for uniform tokens each rank gets rows 0..3
            output.copy_(input_tensor[:output.shape[0]])

        with patch("torch.distributed.reduce_scatter_tensor", side_effect=fake_reduce_scatter):
            result = d.combine(expert_output)

        self.assertEqual(result.shape, (4, 16))
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_combine_truncates_padding(self):
        """combine() must truncate to local_tokens, not max_tokens."""
        tokens_per_rank = [3, 5]  # rank-0 has 3 tokens, max=5
        d = self._make_dispatcher(AllGatherStrategy.BULK)
        d._tokens_per_rank   = tokens_per_rank
        d._local_token_count = 3
        d.ep_rank = 0

        expert_output = torch.randn(8, 16, dtype=torch.float32)  # 3+5=8

        def fake_reduce_scatter(output, input_tensor, group):
            output.copy_(input_tensor[:output.shape[0]])  # returns max_tokens rows

        with patch("torch.distributed.reduce_scatter_tensor", side_effect=fake_reduce_scatter):
            result = d.combine(expert_output)

        # rank-0 has 3 tokens, so result should be [3, 16], not [5, 16].
        self.assertEqual(result.shape[0], 3)

    def test_chunked_dispatch_chunk_count(self):
        """CHUNKED path must call all_gather_into_tensor num_chunks * num_tensors times."""
        tokens_per_rank = [6, 6]
        d = self._make_dispatcher(AllGatherStrategy.CHUNKED, num_chunks=3)
        self._setup_sync_mock(d, tokens_per_rank)
        h, p, r = _make_tensors(6, 16, 2)

        call_count = [0]
        def fake_ag_into(output, inp, group):
            call_count[0] += 1
            chunk = inp.shape[0]
            for i in range(2):
                output[i * chunk : (i + 1) * chunk].copy_(inp)

        with patch("torch.distributed.all_gather_into_tensor", side_effect=fake_ag_into):
            h_g, p_g, r_g = d.dispatch(h, p, r)

        # 3 chunks × 3 tensors (h, p, r) = 9 calls
        self.assertEqual(call_count[0], 9)

    def test_ep_skew_logging(self):
        """EP_SKEW event should be triggered when max/min ratio > threshold."""
        cfg = BandwidthAwareDispatcherConfig(
            manual_strategy=AllGatherStrategy.BULK,
            ep_skew_warn_ratio=1.5,
        )
        d = self._make_dispatcher(AllGatherStrategy.BULK)
        d.config = cfg
        d.ep_rank = 0

        with patch.object(d, '_log_chunk_stall') as mock_log, \
             patch("deepspeed.moe.bandwidth_aware_dispatcher.ds_logger") as mock_logger:
            # tokens_per_rank = [2, 10]: ratio = 5.0 > 1.5 → should log EP_SKEW
            d._maybe_log_ep_skew([2, 10])
            mock_logger.warning.assert_called_once()
            logged_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("EP_SKEW", logged_msg)

    def test_no_ep_skew_below_threshold(self):
        d = self._make_dispatcher(AllGatherStrategy.BULK)
        d.ep_rank = 0
        with patch("deepspeed.moe.bandwidth_aware_dispatcher.ds_logger") as mock_logger:
            d._maybe_log_ep_skew([8, 10])  # ratio = 1.25 < 1.5
            mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# Config defaults tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):

    def test_default_config(self):
        cfg = BandwidthAwareDispatcherConfig()
        self.assertEqual(cfg.fast_link_threshold_gbps, 50.0)
        self.assertEqual(cfg.num_chunks, 4)
        self.assertIsNone(cfg.manual_strategy)
        self.assertEqual(cfg.ep_skew_warn_ratio, 1.5)

    def test_manual_strategy_none_by_default(self):
        cfg = BandwidthAwareDispatcherConfig()
        self.assertIsNone(cfg.manual_strategy)


if __name__ == "__main__":
    unittest.main()
