# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Tests for deepspeed/inference/dynamic_batch_context.py

Covers all four bug classes from Megatron f8becec65:
  Bug 1: SSM seq_idx normalization
  Bug 2: Chunked prefill double-decrement (unified add_request)
  Bug 3: get_chunked_prefill_slot safe/unsafe boundary
  Bug 4a: swap_bookkeeping None guard for next_tokens
  Bug 4b: compute_chunk_length skip-when-1-slot-remains
  Bug 4c: reset_chunked_requests for KV-cache recompute fallback

Also covers the TierPaddingBudget DES-LOC-specific decision boundary.
"""

import pytest
import torch

from deepspeed.inference.dynamic_batch_context import (
    DynamicBatchContext,
    RequestPhase,
    RequestSlot,
    TierPaddingBudget,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ctx(
    max_requests: int = 16,
    max_tokens: int = 512,
    num_speculative_tokens: int = 0,
    tier_label: str = "TEST",
    is_hybrid: bool = False,
) -> DynamicBatchContext:
    return DynamicBatchContext(
        max_requests=max_requests,
        max_tokens=max_tokens,
        num_speculative_tokens=num_speculative_tokens,
        tier_label=tier_label,
        is_hybrid_model=is_hybrid,
    )


# ---------------------------------------------------------------------------
# Bug 1: SSM seq_idx normalization
# ---------------------------------------------------------------------------

class TestSSMSeqIdxNormalization:
    """Bug 1: normalize_seq_idx should subtract token_to_request_idx[start],
    not start itself."""

    def test_contiguous_batch_same_result(self):
        """When request IDs start at 0, old and new approaches agree."""
        ctx = make_ctx(is_hybrid=True)
        # Simulate: tokens 0-5 belong to requests 0,0,1,1,2,2
        t2r = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        result = ctx.normalize_seq_idx(t2r, start_token_idx=0, end_token_idx=6)
        # All values should be 0-based: 0,0,1,1,2,2
        expected = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_non_zero_base_corrected(self):
        """When prefill window starts mid-batch, normalization must use the
        value at start_token_idx, not the integer start_token_idx."""
        ctx = make_ctx(is_hybrid=True)
        # Decode requests occupy slots 0,1 (request IDs 0,1).
        # Prefill window starts at token 4 (request ID 2).
        # token_to_request_idx = [0,1,1,2,2,2,3,3]
        # start_token_idx=4 → value=2, not 4
        t2r = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3], dtype=torch.long)
        result = ctx.normalize_seq_idx(t2r, start_token_idx=4, end_token_idx=8)
        # segment = [2,2,3,3], base = t2r[4] = 2 → result = [0,0,1,1]
        expected = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        assert torch.equal(result, expected), (
            f"Bug 1 regression: got {result.tolist()}, expected {expected.tolist()}"
        )

    def test_old_base_would_be_wrong(self):
        """Illustrate that the old subtraction (start_token_idx) gives wrong values."""
        ctx = make_ctx(is_hybrid=True)
        t2r = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3], dtype=torch.long)
        # Old code would subtract 4 (the integer position):
        segment = t2r[4:8]
        old_result = segment - 4  # old bug
        new_result = ctx.normalize_seq_idx(t2r, start_token_idx=4, end_token_idx=8)
        assert not torch.equal(old_result, new_result), (
            "Test setup error: old and new should differ when base != start_token_idx"
        )

    def test_empty_segment(self):
        """Empty segment returns empty tensor."""
        ctx = make_ctx(is_hybrid=True)
        t2r = torch.tensor([0, 1, 2], dtype=torch.long)
        result = ctx.normalize_seq_idx(t2r, start_token_idx=1, end_token_idx=1)
        assert result.numel() == 0


# ---------------------------------------------------------------------------
# Bug 2: Unified add_request — total_request_count always increments by 1
# ---------------------------------------------------------------------------

class TestAddRequestUnified:
    """Bug 2: add_request must always increment total_request_count by 1."""

    def test_fresh_request_increments_count(self):
        ctx = make_ctx()
        slot = ctx.add_request(req_id=10, chunk_length=100, total_prompt_tokens=100)
        assert ctx.total_request_count == 1
        assert slot.slot_idx == 0
        assert ctx.active_token_count == 100

    def test_two_fresh_requests(self):
        ctx = make_ctx()
        ctx.add_request(req_id=1, chunk_length=50, total_prompt_tokens=50)
        ctx.add_request(req_id=2, chunk_length=60, total_prompt_tokens=60)
        assert ctx.total_request_count == 2
        assert ctx.active_token_count == 110

    def test_continuation_after_hide_is_plus1(self):
        """After update_requests hides the chunked request (decrements count),
        add_request for the continuation should increment back to the same count."""
        ctx = make_ctx(max_tokens=1000)
        # Add a chunked request (chunk < total)
        ctx.add_request(req_id=42, chunk_length=100, total_prompt_tokens=300)
        assert ctx.total_request_count == 1
        assert ctx._chunked_prefill_request_id == 42

        # Simulate update_requests hiding the chunked request:
        # It decrements total_request_count by 1
        ctx._total_request_count -= 1  # simulates the hide step
        assert ctx.total_request_count == 0

        # Now update the slot's finished_chunk_tokens to make it a continuation
        slot = ctx._slots[42]
        slot.finished_chunk_tokens = 100
        slot.remaining_prompt_tokens = 200

        # add_request for continuation should increment to 1
        new_slot = ctx.add_request(req_id=42, chunk_length=100, total_prompt_tokens=300)
        assert ctx.total_request_count == 1, (
            f"Bug 2 regression: expected 1 after continuation add, got {ctx.total_request_count}"
        )

    def test_overflow_raises(self):
        ctx = make_ctx(max_requests=2)
        ctx.add_request(req_id=1, chunk_length=10, total_prompt_tokens=10)
        ctx.add_request(req_id=2, chunk_length=10, total_prompt_tokens=10)
        with pytest.raises(ValueError, match="RequestOverflow"):
            ctx.add_request(req_id=3, chunk_length=10, total_prompt_tokens=10)

    def test_token_overflow_raises(self):
        ctx = make_ctx(max_tokens=100)
        ctx.add_request(req_id=1, chunk_length=80, total_prompt_tokens=80)
        with pytest.raises(ValueError, match="TokenOverflow"):
            ctx.add_request(req_id=2, chunk_length=30, total_prompt_tokens=30)


# ---------------------------------------------------------------------------
# Bug 3: get_chunked_prefill_slot safe vs unsafe
# ---------------------------------------------------------------------------

class TestGetChunkedPrefillSlot:
    """Bug 3: safe=True must not find the request if it is hidden past boundary."""

    def test_no_chunked_request_returns_minus1(self):
        ctx = make_ctx()
        assert ctx.get_chunked_prefill_slot(safe=True) == -1
        assert ctx.get_chunked_prefill_slot(safe=False) == -1

    def test_active_chunked_found_with_safe_true(self):
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=7, chunk_length=50, total_prompt_tokens=200)
        # Slot 0 is within total_request_count=1
        assert ctx.get_chunked_prefill_slot(safe=True) == 0
        assert ctx.get_chunked_prefill_slot(safe=False) == 0

    def test_hidden_chunked_not_found_with_safe_true(self):
        """After hiding (slot >= total_request_count), safe=True returns -1."""
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=7, chunk_length=50, total_prompt_tokens=200)
        # Simulate hiding: move total_request_count to 0 so slot_idx=0 is out of window
        ctx._total_request_count = 0
        # safe=True should not find it
        assert ctx.get_chunked_prefill_slot(safe=True) == -1, (
            "Bug 3 regression: safe=True found hidden request"
        )
        # safe=False should still find it
        assert ctx.get_chunked_prefill_slot(safe=False) == 0

    def test_no_false_positive_from_stale_slots(self):
        """Simulates the LIGHT-tier stale-slot scenario: a previous batch's
        request_id value matches the new chunked_prefill_request_id but lives
        beyond the active window."""
        ctx = make_ctx(max_tokens=1000)
        # Add two requests to fill slots 0 and 1
        ctx.add_request(req_id=1, chunk_length=50, total_prompt_tokens=50)
        ctx.add_request(req_id=7, chunk_length=50, total_prompt_tokens=200)
        # req_id=7 is at slot 1, total_request_count=2
        # Move total_request_count to 1 (hiding slot 1)
        ctx._total_request_count = 1
        # safe=True should not return slot 1 (it is hidden)
        assert ctx.get_chunked_prefill_slot(safe=True) == -1


# ---------------------------------------------------------------------------
# Bug 4a: swap_bookkeeping — next_tokens None guard
# ---------------------------------------------------------------------------

class TestSwapBookkeeping:
    """Bug 4a: swap_bookkeeping with swap_next_tokens=False must not crash."""

    def test_swap_moves_slot_indices(self):
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=1, chunk_length=50, total_prompt_tokens=50)
        ctx.add_request(req_id=2, chunk_length=60, total_prompt_tokens=60)
        # Slots: req_id=1 at slot 0, req_id=2 at slot 1
        ctx.swap_bookkeeping(src_idx=0, dst_idx=1, swap_next_tokens=True)
        assert ctx._slots[1].slot_idx == 1  # moved from 0 to 1
        assert ctx._slots[2].slot_idx == 0  # moved from 1 to 0

    def test_swap_with_no_next_tokens(self):
        """swap_next_tokens=False must not raise even for out-of-bounds indices."""
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=1, chunk_length=50, total_prompt_tokens=50)
        ctx.add_request(req_id=2, chunk_length=60, total_prompt_tokens=60)
        # Simulate out-of-bounds hidden slot (total_request_count=1, slot 1 is hidden)
        ctx._total_request_count = 1
        # Should not raise
        ctx.swap_bookkeeping(src_idx=1, dst_idx=1, swap_next_tokens=False)

    def test_noop_when_src_equals_dst(self):
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=5, chunk_length=30, total_prompt_tokens=30)
        original_idx = ctx._slots[5].slot_idx
        ctx.swap_bookkeeping(src_idx=0, dst_idx=0, swap_next_tokens=True)
        assert ctx._slots[5].slot_idx == original_idx


# ---------------------------------------------------------------------------
# Bug 4b: compute_chunk_length — skip when only 1 slot remains
# ---------------------------------------------------------------------------

class TestComputeChunkLength:
    """Bug 4b: skip scheduling when remaining=2 but available=1 (FA kernel bug)."""

    def test_normal_chunking(self):
        ctx = make_ctx()
        length, can_schedule = ctx.compute_chunk_length(
            req_id=1, remaining_len=100, available_tokens=80
        )
        assert can_schedule is True
        assert length == 80

    def test_reduce_by_1_when_would_leave_1(self):
        """remaining=10, available=9 → remaining - chunk = 1 with chunk > 1 → reduce by 1."""
        ctx = make_ctx()
        length, can_schedule = ctx.compute_chunk_length(
            req_id=1, remaining_len=10, available_tokens=9
        )
        assert can_schedule is True
        assert length == 8, f"Expected 8 (reduced from 9), got {length}"

    def test_skip_when_remaining_2_available_1(self):
        """remaining=2, available=1 → remaining - chunk = 1, chunk = 1: skip scheduling."""
        ctx = make_ctx()
        length, can_schedule = ctx.compute_chunk_length(
            req_id=1, remaining_len=2, available_tokens=1
        )
        assert can_schedule is False, "Bug 4b regression: should skip scheduling"
        assert length is None

    def test_skip_when_exact_match_triggers_boundary(self):
        """remaining=2, available=2 → chunk=2, remaining-chunk=0: normal, no skip."""
        ctx = make_ctx()
        length, can_schedule = ctx.compute_chunk_length(
            req_id=1, remaining_len=2, available_tokens=2
        )
        assert can_schedule is True
        assert length == 2

    def test_no_available_returns_false(self):
        ctx = make_ctx()
        length, can_schedule = ctx.compute_chunk_length(
            req_id=1, remaining_len=5, available_tokens=0
        )
        assert can_schedule is False

    def test_light_tier_emits_steer_pressure(self, capsys):
        """LIGHT tier emits STEER_PRESSURE when skip_sched triggers."""
        ctx = make_ctx(tier_label="LIGHT")
        ctx.compute_chunk_length(req_id=1, remaining_len=2, available_tokens=1)
        captured = capsys.readouterr()
        assert "STEER_PRESSURE" in captured.out or "SKIP_SCHED" in captured.out


# ---------------------------------------------------------------------------
# Bug 4c: reset_chunked_requests — recompute fallback
# ---------------------------------------------------------------------------

class TestResetChunkedRequests:
    """Bug 4c: reset_chunked_requests must restore initial prompt state."""

    def test_reset_clears_finished_chunk_tokens(self):
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=5, chunk_length=100, total_prompt_tokens=500)
        slot = ctx._slots[5]
        slot.finished_chunk_tokens = 100
        slot.remaining_prompt_tokens = 400

        ctx.reset_chunked_requests([5])

        assert slot.finished_chunk_tokens == 0
        assert slot.remaining_prompt_tokens == 500
        assert slot.phase == RequestPhase.PREFILL

    def test_reset_clears_chunked_prefill_tracker(self):
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=5, chunk_length=100, total_prompt_tokens=500)
        ctx._chunked_prefill_request_id = 5
        ctx.reset_chunked_requests([5])
        assert ctx.chunked_prefill_request_id == -1

    def test_reset_noop_for_untracked_request(self):
        ctx = make_ctx()
        # Should not raise for unknown request IDs
        ctx.reset_chunked_requests([999])

    def test_reset_only_affects_chunked_requests(self):
        """Requests with finished_chunk_tokens == 0 should be unchanged."""
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=1, chunk_length=50, total_prompt_tokens=50)
        slot = ctx._slots[1]
        original_finished = slot.finished_chunk_tokens  # 0
        ctx.reset_chunked_requests([1])
        # Should remain unchanged
        assert slot.finished_chunk_tokens == original_finished


# ---------------------------------------------------------------------------
# TierPaddingBudget: DES-LOC specific decision boundary
# ---------------------------------------------------------------------------

class TestTierPaddingBudget:
    """Test the LIGHT vs HEAVY padding waste steering logic."""

    def test_waste_fraction_light(self):
        budget = TierPaddingBudget(light_max_tokens=4096, heavy_max_tokens=32768)
        frac = budget.padding_waste_fraction("LIGHT", 1)
        assert abs(frac - 1 / 4096) < 1e-9

    def test_waste_fraction_heavy(self):
        budget = TierPaddingBudget(light_max_tokens=4096, heavy_max_tokens=32768)
        frac = budget.padding_waste_fraction("HEAVY", 1)
        assert abs(frac - 1 / 32768) < 1e-9

    def test_steer_to_heavy_when_waste_exceeds_threshold(self):
        budget = TierPaddingBudget(
            light_max_tokens=4096, heavy_max_tokens=32768, steer_threshold=0.001
        )
        # 10 padding tokens on 4096 → 10/4096 ≈ 0.0024 > 0.001
        should_steer = budget.should_steer_to_heavy(
            num_padding_tokens=10, light_utilization=0.8, heavy_utilization=0.5
        )
        assert should_steer is True

    def test_no_steer_when_heavy_saturated(self):
        budget = TierPaddingBudget(
            light_max_tokens=4096, heavy_max_tokens=32768, steer_threshold=0.001
        )
        should_steer = budget.should_steer_to_heavy(
            num_padding_tokens=100, light_utilization=0.8, heavy_utilization=0.98
        )
        assert should_steer is False

    def test_no_steer_when_waste_below_threshold(self):
        budget = TierPaddingBudget(
            light_max_tokens=4096, heavy_max_tokens=32768, steer_threshold=0.01
        )
        # 1 token / 4096 ≈ 0.00024 < 0.01
        should_steer = budget.should_steer_to_heavy(
            num_padding_tokens=1, light_utilization=0.5, heavy_utilization=0.3
        )
        assert should_steer is False

    def test_heavy_waste_fraction_is_8x_smaller_than_light(self):
        """Validate the 8x amplification ratio documented in the module."""
        budget = TierPaddingBudget(light_max_tokens=4096, heavy_max_tokens=32768)
        light_frac = budget.padding_waste_fraction("LIGHT", 1)
        heavy_frac = budget.padding_waste_fraction("HEAVY", 1)
        ratio = light_frac / heavy_frac
        assert abs(ratio - 8.0) < 0.01, f"Expected 8x ratio, got {ratio}"


# ---------------------------------------------------------------------------
# Integration: full add → update → continuation cycle
# ---------------------------------------------------------------------------

class TestFullCycle:
    """Integration test covering the add → update → continuation path."""

    def test_single_request_fresh_then_decode(self):
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=1, chunk_length=50, total_prompt_tokens=50)
        assert ctx.total_request_count == 1
        assert ctx.chunked_prefill_request_id == -1  # not chunked (full prompt)

    def test_chunked_request_hidden_and_restored(self):
        """Full cycle: add chunked → update (hide) → add continuation (restore)."""
        ctx = make_ctx(max_tokens=1000)
        # Step 1: Add chunked request (chunk < total)
        ctx.add_request(req_id=42, chunk_length=100, total_prompt_tokens=300)
        assert ctx.total_request_count == 1
        assert ctx.chunked_prefill_request_id == 42

        # Step 2: Simulate update_requests hiding the request
        # (update_requests decrements total_request_count after swapping to end)
        # The request at slot 0 gets swapped to slot (total-1) = 0, then hidden.
        ctx._total_request_count -= 1   # hide
        slot = ctx._slots[42]
        assert ctx.get_chunked_prefill_slot(safe=True) == -1   # hidden
        assert ctx.get_chunked_prefill_slot(safe=False) == 0   # still findable

        # Step 3: Update slot to reflect finished chunk
        slot.finished_chunk_tokens = 100
        slot.remaining_prompt_tokens = 200

        # Step 4: add_request continuation → total_request_count back to 1
        ctx.add_request(req_id=42, chunk_length=100, total_prompt_tokens=300)
        assert ctx.total_request_count == 1, (
            f"Expected total=1 after continuation, got {ctx.total_request_count}"
        )

    def test_reset_then_add_fresh(self):
        ctx = make_ctx(max_tokens=1000)
        ctx.add_request(req_id=1, chunk_length=100, total_prompt_tokens=100)
        ctx.reset()
        assert ctx.total_request_count == 0
        assert ctx.active_token_count == 0
        ctx.add_request(req_id=2, chunk_length=50, total_prompt_tokens=50)
        assert ctx.total_request_count == 1
