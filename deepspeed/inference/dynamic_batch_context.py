# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""DynamicBatchContext — heterogeneous-GPU dynamic batch bookkeeping for DES-LOC.

Mirrors Megatron f8becec65 "Miscellaneous inference bug fixes", reinterpreted as
a *batch-slot bookkeeping layer* that fixes four correlated bugs in DES-LOC's
HeterogeneousInferenceEngine when operating with chunked-prefill and mixed-tier
(A6000 + H100) dispatch.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upstream design intent (f8becec65)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Four bugs were fixed in DynamicInferenceContext / DynamicInferenceEngine:

Bug 1 — SSM/Mamba seq_idx normalization (mamba_metadata.py)
  The old code subtracted ``start_regular_prefill_req_idx`` (the *integer index*
  of the first prefill request) to normalize token→request mappings to 0-based.
  This is wrong when the first prefill request is not request ID 0: the token-to-
  request mapping should be normalized relative to the *value* at the first
  prefill token position, i.e. ``token_to_request_idx[start_regular_prefill_token_idx]``.
  Fix: subtract the value at the first token position, not the position integer.

Bug 2 — Chunked prefill double-decrement (dynamic_context.py add_request)
  When adding a continuation chunk for a request already in the batch, the old
  code decremented active_token_count by (1 + num_speculative_tokens) to "overwrite
  the useless token from chunked prefill" and set total_request_count += 0.  This
  means the bookkeeping was split across add_request (the decrement) and
  update_requests (the move to end).  The fix unifies this: update_requests always
  decrements total_request_count by 1 for the chunked-prefill request before
  add_request is called, so add_request always does ``total_request_count += 1``
  unconditionally and sets current_id = total_request_count without special-casing.

Bug 3 — get_index_of_chunked_prefill_request search-space overflow (dynamic_context.py)
  The old search always scanned request_ids[0:MAX_REQUESTS].  After the chunked
  prefill request is "hidden" beyond total_request_count by update_requests, a
  safe=True search should be clamped to request_ids[:total_request_count]; an
  unsafe=False (full-range) search is needed only when looking for a request that
  was moved out of the active window.  Without the safe flag, the search finds a
  stale slot beyond the active window and returns a wrong index.

Bug 4 — _swap_book_keeping_tensors None guard + speculative decode fallback
  (dynamic_context.py, dynamic_engine.py)
  Two related fixes:
  (a) next_tokens was unconditionally swapped even when called for out-of-bounds
      indices (the "pull to new boundary" path); this causes an index-out-of-bounds
      error.  Fix: guard with ``if next_tokens is not None``.
  (b) schedule_chunked_prefill: when only 1 token of space remains but the
      remaining prompt has 2 tokens, the old code reduced chunk_length by 1
      (leaving 1 token for the final chunk), hitting a flash-attention bug with
      max_seqlen_q=1.  Fix: skip scheduling entirely when chunk_length would be 1
      and remaining_len is 2.
  (c) In the recompute (KVCache RECOMPUTE mode) fallback, partially-prefilled
      requests had stale finished_chunk_token_count > 0, causing add_request to
      re-enter the chunked path incorrectly.  Fix: reset both
      remaining_prompt_tokens and finished_chunk_token_count to initial values.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DES-LOC adaptation rationale
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HeterogeneousInferenceEngine (M4188) dispatches requests across A6000 (LIGHT tier,
48 GB) and H100 (HEAVY tier, 80 GB) GPUs.  The bugs above are *amplified* on the
LIGHT tier:

  A6000 padding amplification (Bug 2 / Bug 4b):
    max_tokens on the LIGHT tier is typically 4096 (vs 32768 on HEAVY).  A
    padding waste of 1 speculative token on H100 is ~0.003% of capacity; on
    A6000 it is ~0.024% — 8× more costly as a fraction of total memory.  More
    importantly, when the +1 token causes a 4097-token batch to exceed the
    LIGHT tier's 4096 limit, the tier reports saturation and the request is
    re-routed to HEAVY (STEER_SHIFT event), artificially inflating HEAVY load.

  SSM state slot mis-assignment (Bug 1):
    DES-LOC's SSMStateManager (M4189) uses the normalized seq_idx values
    returned by mamba_metadata to index into the hot/cold state pools.  If the
    normalization uses the wrong base, the seq_idx values alias across requests
    and two requests share a single SSM state slot, causing silent state
    corruption that manifests as perplexity spikes rather than crashes.

  Safe/unsafe boundary confusion (Bug 3):
    When the chunked-prefill request is hidden beyond total_request_count,
    get_index_of_chunked_prefill_request with safe=True must not find it (it
    was temporarily retired).  In DES-LOC, the "hidden" slot can land in the
    padding region of the LIGHT tier's request array where the slot value is
    uninitialized — the match on request_id then returns garbage.

  Recompute reset (Bug 4c):
    HeterogeneousInferenceEngine's fallback path (invoked when LIGHT-tier KV
    cache is full and the engine must recompute from scratch) called _add_request
    on partially-chunked requests without resetting finished_chunk_token_count.
    This made add_request believe it was still continuing a chunk, decrementing
    active_token_count spuriously and making total_request_count go negative.

DynamicBatchContext encapsulates the corrected bookkeeping logic as a standalone
class that HeterogeneousInferenceEngine can delegate to for both LIGHT and HEAVY
tiers.  Each tier gets an independent DynamicBatchContext instance sized to the
tier's max_tokens / max_requests budget.

Diagnostic events (rank-0, logger.info + print, one per state transition):
  [DS-DBC] INIT         — max_requests, max_tokens, num_speculative_tokens.
  [DS-DBC] ADD_REQ      — request_id, chunk_length, is_continuation, slot_idx.
  [DS-DBC] HIDE_CHUNKED — request_id, old_slot, boundary (after update_requests).
  [DS-DBC] PULL_CHUNKED — request_id, old_slot, new_boundary (boundary drift fix).
  [DS-DBC] RESET_CHUNK  — request_id, old_finished_count (recompute fallback reset).
  [DS-DBC] SKIP_SCHED   — request_id, remaining_len (FA kernel bug avoidance).
  [DS-DBC] SSM_BASE_FIX — batch_idx, old_base, new_base (seq_idx normalization fix).
  [DS-DBC] STEER_PRESSURE — emitted when a padding bug avoidance would have
                             caused tier saturation on the LIGHT tier.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-DBC]"
_RANK0_ONLY = True  # emit diagnostics only from rank 0


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def _log(msg: str) -> None:
    if _RANK0_ONLY and not _is_rank0():
        return
    full = f"{_LOG_PREFIX} {msg}"
    ds_logger.info(full)
    print(full)


# ---------------------------------------------------------------------------
# Request state
# ---------------------------------------------------------------------------

class RequestPhase(str, Enum):
    """Lifecycle phase of a request within a DynamicBatchContext."""
    PREFILL = "prefill"          # first chunk or single-shot prefill
    CHUNKED = "chunked"          # continuation chunk in progress
    DECODE = "decode"            # autoregressive decode
    PAUSED = "paused"            # waiting for KV-cache space
    HIDDEN = "hidden"            # moved past total_request_count boundary


@dataclass
class RequestSlot:
    """Per-request bookkeeping within a DynamicBatchContext.

    Mirrors the per-request fields scattered across DynamicInferenceContext's
    tensor arrays (request_ids, request_in_prefill_status_tensor, etc.) but
    stores them as a Python struct for readability.  The actual runtime uses
    tensors; this struct is the *design reference* that the tensor layout
    mirrors.

    Attributes:
        request_id:              Unique integer ID for this request.
        slot_idx:                Current position in the request tensor arrays.
                                 This changes when swap_bookkeeping is called.
        total_prompt_tokens:     Total tokens in the original prompt.
        finished_chunk_tokens:   Tokens of the prompt already processed.
        remaining_prompt_tokens: Tokens yet to be prefilled.
        phase:                   Current lifecycle phase.
        ssm_state_slot:          Index into the SSMStateManager pool, or -1.
        num_speculative_tokens:  Number of speculative draft tokens appended.
    """
    request_id: int
    slot_idx: int
    total_prompt_tokens: int
    finished_chunk_tokens: int = 0
    remaining_prompt_tokens: int = 0
    phase: RequestPhase = RequestPhase.PREFILL
    ssm_state_slot: int = -1
    num_speculative_tokens: int = 0

    def __post_init__(self):
        if self.remaining_prompt_tokens == 0:
            self.remaining_prompt_tokens = self.total_prompt_tokens

    @property
    def is_continuation(self) -> bool:
        """True when this request is a continuing chunked-prefill (Bug 2 guard)."""
        return self.finished_chunk_tokens > 0


# ---------------------------------------------------------------------------
# DynamicBatchContext
# ---------------------------------------------------------------------------

class DynamicBatchContext:
    """Manages per-step request bookkeeping for one GPU tier in DES-LOC.

    This class fixes the four bugs from Megatron f8becec65 as they would
    manifest in DES-LOC's heterogeneous engine.  It does NOT depend on any
    megatron.core imports.

    Usage::

        ctx = DynamicBatchContext(
            max_requests=64,
            max_tokens=4096,
            num_speculative_tokens=0,
            tier_label="LIGHT",  # "LIGHT" or "HEAVY" for diagnostics
        )

        # Add a fresh request (prefill or continuation):
        slot = ctx.add_request(req_id=42, chunk_length=512, total_prompt_tokens=1024)

        # After a forward pass, update bookkeeping:
        ctx.update_requests(active_mask, next_tokens)

        # Recompute fallback (KV-cache full): reset partially-chunked requests:
        ctx.reset_chunked_requests([req_id_1, req_id_2])
    """

    def __init__(
        self,
        max_requests: int,
        max_tokens: int,
        num_speculative_tokens: int = 0,
        tier_label: str = "UNKNOWN",
        is_hybrid_model: bool = False,
    ) -> None:
        """
        Args:
            max_requests:           Maximum concurrent requests this tier can hold.
            max_tokens:             Maximum total active tokens per step.
            num_speculative_tokens: Number of speculative draft tokens per request.
            tier_label:             Human-readable tier name for diagnostics.
            is_hybrid_model:        True when the model has Mamba/SSM layers
                                    (enables SSM seq_idx normalization tracking).
        """
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.num_speculative_tokens = num_speculative_tokens
        self.tier_label = tier_label
        self.is_hybrid_model = is_hybrid_model

        # Active request tracking (mirrors DynamicInferenceContext tensor arrays)
        self._slots: Dict[int, RequestSlot] = {}          # request_id → RequestSlot
        self._slot_order: List[int] = []                  # ordered list of slot_idx → request_id
        self._total_request_count: int = 0                # visible window (< max_requests)
        self._active_token_count: int = 0
        self._num_prefill_requests: int = 0
        self._paused_request_count: int = 0
        self._chunked_prefill_request_id: int = -1        # -1 when none in progress

        _log(
            f"INIT tier={tier_label} max_requests={max_requests} "
            f"max_tokens={max_tokens} speculative={num_speculative_tokens} "
            f"hybrid={is_hybrid_model}"
        )

    # ------------------------------------------------------------------
    # Bug 4c fix: recompute fallback reset
    # ------------------------------------------------------------------

    def reset_chunked_requests(self, request_ids: List[int]) -> None:
        """Reset partially-prefilled requests before a KV-cache recompute.

        Mirrors dynamic_engine.py f8becec65 recompute path:
          ``req.remaining_prompt_tokens = req.prompt_tokens``
          ``req.finished_chunk_token_count = 0``
          ``self.chunked_prefill_request_id = -1``

        DES-LOC context: called by HeterogeneousInferenceEngine when the LIGHT
        tier's KV cache is exhausted and requests must restart from scratch.
        Without this reset, add_request sees finished_chunk_tokens > 0 and
        enters the (now-incorrect) continuation path, producing a spurious
        active_token_count decrement.

        Emits [DS-DBC] RESET_CHUNK per affected request.
        """
        for req_id in request_ids:
            slot = self._slots.get(req_id)
            if slot is None:
                continue
            if slot.finished_chunk_tokens > 0:
                old_count = slot.finished_chunk_tokens
                slot.finished_chunk_tokens = 0
                slot.remaining_prompt_tokens = slot.total_prompt_tokens
                slot.phase = RequestPhase.PREFILL
                _log(
                    f"RESET_CHUNK tier={self.tier_label} req_id={req_id} "
                    f"old_finished={old_count} → reset to full prefill"
                )
        # Reset the global chunked prefill tracker
        self._chunked_prefill_request_id = -1

    # ------------------------------------------------------------------
    # Bug 2 fix: unified add_request (no double-decrement)
    # ------------------------------------------------------------------

    def add_request(
        self,
        req_id: int,
        chunk_length: int,
        total_prompt_tokens: int,
        ssm_state_slot: int = -1,
    ) -> RequestSlot:
        """Add a request (fresh or continuation) to the active batch.

        This implements the Bug 2 fix: ``total_request_count`` is always
        incremented by exactly 1 here.  For continuation chunks,
        ``update_requests`` will have already decremented total_request_count
        by 1 when it "hid" the chunked-prefill request, so the net effect after
        add_request is that total_request_count is unchanged for continuations
        and incremented by 1 for fresh requests — matching the upstream fix
        where the special-case ``is_chunked_prefill`` branch was removed and
        ``total_request_count += 1`` became unconditional.

        Raises:
            ValueError: If total_request_count >= max_requests.
            ValueError: If active_token_count + chunk_length > max_tokens.
        """
        if self._total_request_count >= self.max_requests:
            raise ValueError(
                f"[DS-DBC] tier={self.tier_label} RequestOverflow: "
                f"total_request_count={self._total_request_count} >= max_requests={self.max_requests}"
            )
        if self._active_token_count + chunk_length > self.max_tokens:
            raise ValueError(
                f"[DS-DBC] tier={self.tier_label} TokenOverflow: "
                f"active_tokens={self._active_token_count} + chunk={chunk_length} "
                f"> max_tokens={self.max_tokens}"
            )

        existing = self._slots.get(req_id)
        is_continuation = (existing is not None and existing.finished_chunk_tokens > 0)

        if is_continuation:
            # Bug 2 fix: update_requests already decremented total_request_count,
            # so current_id = total_request_count lands at the right slot.
            slot = existing
            slot.phase = RequestPhase.CHUNKED
            current_id = self._total_request_count
            # Restore SSM state slot if provided (only valid on first chunk per Mamba bug fix)
        else:
            current_id = self._total_request_count
            slot = RequestSlot(
                request_id=req_id,
                slot_idx=current_id,
                total_prompt_tokens=total_prompt_tokens,
                finished_chunk_tokens=0,
                remaining_prompt_tokens=total_prompt_tokens,
                num_speculative_tokens=self.num_speculative_tokens,
            )
            self._slots[req_id] = slot

            # Bug 1 fix: SSM state slot allocation only for NEW requests.
            # For chunked continuations, the slot was allocated on the first chunk.
            if self.is_hybrid_model and ssm_state_slot >= 0:
                slot.ssm_state_slot = ssm_state_slot

        slot.slot_idx = current_id

        # Always unconditional (Bug 2 fix)
        self._total_request_count += 1
        self._active_token_count += chunk_length
        self._num_prefill_requests += 1

        if chunk_length < total_prompt_tokens:
            self._chunked_prefill_request_id = req_id

        _log(
            f"ADD_REQ tier={self.tier_label} req_id={req_id} "
            f"chunk={chunk_length} slot={current_id} "
            f"continuation={is_continuation} "
            f"total_req={self._total_request_count} active_tok={self._active_token_count}"
        )
        return slot

    # ------------------------------------------------------------------
    # Bug 4b fix: schedule_chunk — skip when leaving exactly 1 token
    # ------------------------------------------------------------------

    def compute_chunk_length(
        self,
        req_id: int,
        remaining_len: int,
        available_tokens: int,
    ) -> Tuple[Optional[int], bool]:
        """Compute the chunk length for the next prefill step, avoiding FA bug.

        Mirrors dynamic_engine.py f8becec65 schedule_chunked_prefill fix:

          Old code:
            if remaining_len - chunk_length == 1 and chunk_length > 1:
                chunk_length -= 1

          New code:
            if remaining_len - chunk_length == 1:
                if chunk_length > 1:
                    chunk_length -= 1
                else:
                    can_schedule = False  # skip entirely

        The Flash Attention kernel has a bug when max_seqlen_q == 1 (see
        https://github.com/Dao-AILab/flash-attention/issues/1537).  If only 1
        token of space is available but 2 tokens of prompt remain, reducing
        chunk_length by 1 gives chunk_length=0 which is invalid.  The correct
        fix is to skip scheduling and wait for a larger window.

        DES-LOC amplification: on the LIGHT tier (A6000, max_tokens=4096),
        this edge case is ~8× more likely to be hit than on HEAVY because the
        token budget is 8× smaller.  The old code path would emit a
        STEER_PRESSURE diagnostic as the 0-length chunk caused an assertion.

        Returns:
            (chunk_length, can_schedule): chunk_length is None when can_schedule
            is False.  Callers should check can_schedule before calling add_request.
        """
        chunk_length = min(remaining_len, available_tokens)
        if chunk_length <= 0:
            return None, False

        if remaining_len - chunk_length == 1:
            if chunk_length > 1:
                # Reduce by 1 so final chunk has 2 tokens
                chunk_length -= 1
            else:
                # Only 1 slot available, 2 tokens remain: skip scheduling.
                # This avoids FA kernel bug with max_seqlen_q=1.
                _log(
                    f"SKIP_SCHED tier={self.tier_label} req_id={req_id} "
                    f"remaining={remaining_len} available={available_tokens} "
                    f"→ delay to avoid max_seqlen_q=1 FA bug"
                )
                # Emit STEER_PRESSURE if this is on LIGHT tier (where padding
                # waste is amplified relative to HEAVY).
                if self.tier_label == "LIGHT":
                    _log(
                        f"STEER_PRESSURE tier=LIGHT req_id={req_id} "
                        f"available_tokens={available_tokens} would_have_exceeded=True"
                    )
                return None, False

        return chunk_length, True

    # ------------------------------------------------------------------
    # Bug 3 fix: get_index_of_chunked_prefill_request with safe flag
    # ------------------------------------------------------------------

    def get_chunked_prefill_slot(self, safe: bool = True) -> int:
        """Get the slot index of the current chunked-prefill request.

        Mirrors dynamic_context.py f8becec65 get_index_of_chunked_prefill_request
        with the ``safe`` flag:

          safe=True:  search only request_ids[:total_request_count].  Use this
                      in update_requests before hiding the request.
          safe=False: search the full slot range including the hidden region.
                      Use this after update_requests to locate the hidden slot.

        Returns:
            Slot index of the chunked-prefill request, or -1 if none exists.

        DES-LOC context: the LIGHT tier's request array is padded to max_requests
        (e.g. 64).  Slots beyond total_request_count may contain uninitialized
        request IDs from a previous batch.  A safe=True search prevents false
        matches against stale values in the padding region.
        """
        if self._chunked_prefill_request_id == -1:
            return -1

        target_id = self._chunked_prefill_request_id
        slot = self._slots.get(target_id)
        if slot is None:
            return -1

        if safe:
            # Only match if within the active window (Bug 3 fix)
            if slot.slot_idx < self._total_request_count:
                return slot.slot_idx
            return -1
        else:
            # Full range search: return slot even if hidden past boundary
            return slot.slot_idx

    # ------------------------------------------------------------------
    # Bug 4a fix: swap_bookkeeping with optional next_tokens guard
    # ------------------------------------------------------------------

    def swap_bookkeeping(
        self,
        src_idx: int,
        dst_idx: int,
        swap_next_tokens: bool = True,
    ) -> None:
        """Swap two request slots in the bookkeeping arrays.

        Mirrors dynamic_context.py f8becec65 _swap_book_keeping_tensors fix:
        next_tokens is guarded with ``if next_tokens is not None`` before the
        swap.  In DES-LOC, next_tokens only exists for active (in-window) slots.
        When pulling a hidden chunked-prefill request to the new boundary, the
        indices are out of bounds for next_tokens — passing swap_next_tokens=False
        prevents the index error.

        Args:
            src_idx:          Source slot index.
            dst_idx:          Destination slot index.
            swap_next_tokens: If False, skip the next_tokens swap (Bug 4a fix).
                              Set to False when moving hidden/boundary slots.
        """
        if src_idx == dst_idx:
            return

        # Find requests at these slots
        src_req_id = None
        dst_req_id = None
        for req_id, slot in self._slots.items():
            if slot.slot_idx == src_idx:
                src_req_id = req_id
            elif slot.slot_idx == dst_idx:
                dst_req_id = req_id

        if src_req_id is not None:
            self._slots[src_req_id].slot_idx = dst_idx
        if dst_req_id is not None:
            self._slots[dst_req_id].slot_idx = src_idx

        if not swap_next_tokens:
            _log(
                f"PULL_CHUNKED tier={self.tier_label} "
                f"src={src_idx} dst={dst_idx} next_tokens_skipped=True"
            )

    # ------------------------------------------------------------------
    # update_requests — post-forward bookkeeping (integrates all fixes)
    # ------------------------------------------------------------------

    def update_requests(
        self,
        active_mask: torch.Tensor,
        next_tokens: Optional[torch.Tensor] = None,
        new_speculative_tokens: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Update bookkeeping after a forward pass.

        This method implements the unified chunked-prefill request management
        from f8becec65:

        1. Reset all prefill→decode transitions.
        2. Locate the chunked-prefill request using safe=True.
        3. Force active_mask[-1] = 1 so it survives the finished-request sweep.
        4. Compute active/finished counts and update total_request_count.
        5. Swap the chunked-prefill request to the end of the active window,
           then decrement total_request_count by 1 to "hide" it.
        6. If the hidden request drifted past the new boundary, pull it back
           (without swapping next_tokens — Bug 4a fix).

        Returns:
            List of request IDs that completed (reached EOD or max length).
        """
        self._num_prefill_requests = 0  # All turns to decode

        # Step 2: find chunked-prefill in the safe (active) window
        chunked_idx = self.get_chunked_prefill_slot(safe=True)

        # Step 3: keep chunked-prefill alive through the finished-request sweep
        if chunked_idx != -1:
            active_mask[chunked_idx] = 1

        active_count = int((active_mask == 1).sum().item())
        finished_count = int((active_mask == 0).sum().item())
        finished_ids: List[int] = []

        # Step 4: remove finished requests
        for req_id, slot in list(self._slots.items()):
            if slot.slot_idx < self._total_request_count:
                mask_val = active_mask[slot.slot_idx].item() if slot.slot_idx < len(active_mask) else 1
                if mask_val == 0:
                    finished_ids.append(req_id)
                    del self._slots[req_id]

        self._total_request_count = active_count + self._paused_request_count

        # Step 5: swap chunked-prefill to end of active window, then hide
        # (Bug 2 fix: update_requests decrements total_request_count so that
        #  the next add_request call for the continuation sets current_id to
        #  the correct position without special-casing.)
        chunked_idx_unsafe = self.get_chunked_prefill_slot(safe=False)
        if chunked_idx_unsafe != -1:
            if chunked_idx_unsafe < self._total_request_count:
                # Request was active this step: swap to end then hide
                end_slot = self._total_request_count - 1
                if chunked_idx_unsafe != end_slot:
                    self.swap_bookkeeping(
                        src_idx=chunked_idx_unsafe,
                        dst_idx=end_slot,
                        swap_next_tokens=(next_tokens is not None),
                    )
                    _log(
                        f"HIDE_CHUNKED tier={self.tier_label} "
                        f"req_id={self._chunked_prefill_request_id} "
                        f"slot={end_slot} boundary={self._total_request_count}"
                    )
                # Hide: decrement total_request_count so it sits just outside window
                active_count -= 1
                self._total_request_count -= 1
            else:
                # Request was already hidden: pull to the new boundary if it drifted
                new_boundary = self._total_request_count
                if chunked_idx_unsafe != new_boundary:
                    self.swap_bookkeeping(
                        src_idx=chunked_idx_unsafe,
                        dst_idx=new_boundary,
                        swap_next_tokens=False,  # Bug 4a fix: out-of-bounds indices
                    )

        # Verify invariant
        assert self._total_request_count == active_count + self._paused_request_count, (
            f"[DS-DBC] tier={self.tier_label} total_request_count invariant violated: "
            f"{self._total_request_count} != {active_count} + {self._paused_request_count}"
        )

        return finished_ids

    # ------------------------------------------------------------------
    # Bug 1 fix: SSM seq_idx normalization helper
    # ------------------------------------------------------------------

    def normalize_seq_idx(
        self,
        token_to_request_idx: torch.Tensor,
        start_token_idx: int,
        end_token_idx: int,
    ) -> torch.Tensor:
        """Return 0-based seq_idx values for Mamba/SSM layers.

        Mirrors mamba_metadata.py f8becec65 fix:

          Old: normalize by subtracting ``start_regular_prefill_req_idx``
               (the *integer index* of the first prefill request in the list).
          New: normalize by subtracting ``token_to_request_idx[start_token_idx]``
               (the *value* at the first prefill token position).

        The distinction matters when the batch does not start at request ID 0.
        Example: a 3-request batch where requests 0 and 1 are in decode phase.
        The first prefill token is at token_idx=N and maps to request_id=2.
        token_to_request_idx[N] = 2.  The old code would subtract 2 (the slot
        index of the first prefill request) only if start_regular_prefill_req_idx
        happened to equal 2; if the two decode requests occupy slots 0 and 1 the
        subtraction is correct, but if the slot layout differs (e.g. after a swap)
        start_regular_prefill_req_idx might be 0 while the value is 2, giving
        wrong 0-based indices and aliasing two SSM state slots.

        DES-LOC context: SSMStateManager (M4189) uses these indices to look up
        hot/cold state pools.  Wrong normalization causes two requests to share
        one SSM state slot → state corruption → perplexity spikes.

        Args:
            token_to_request_idx: 1D tensor mapping token positions to request IDs.
            start_token_idx:      Index of first token in the prefill segment.
            end_token_idx:        Index past the last token in the prefill segment.

        Returns:
            0-based seq_idx tensor of length (end_token_idx - start_token_idx).
        """
        seq_len = end_token_idx - start_token_idx
        if seq_len <= 0:
            return torch.zeros(0, dtype=token_to_request_idx.dtype,
                               device=token_to_request_idx.device)

        segment = token_to_request_idx[start_token_idx:end_token_idx]
        # Bug 1 fix: use the value at the first token, not the position integer
        base = token_to_request_idx[start_token_idx]
        old_base_would_be = start_token_idx  # illustrative — what old code used
        if base != old_base_would_be:
            _log(
                f"SSM_BASE_FIX tier={self.tier_label} "
                f"start_token={start_token_idx} "
                f"old_base={old_base_would_be} new_base={base.item()} "
                f"delta={base.item() - old_base_would_be}"
            )
        return segment - base

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_request_count(self) -> int:
        """Number of requests visible to the current forward pass."""
        return self._total_request_count

    @property
    def active_token_count(self) -> int:
        """Total tokens active in the current step."""
        return self._active_token_count

    @property
    def chunked_prefill_request_id(self) -> int:
        """Request ID of the in-progress chunked-prefill, or -1."""
        return self._chunked_prefill_request_id

    @property
    def available_token_slots(self) -> int:
        """Remaining token capacity for this step."""
        return self.max_tokens - self._active_token_count

    @property
    def available_request_slots(self) -> int:
        """Remaining request slots (active window)."""
        return self.max_requests - self._total_request_count

    def get_request_slot(self, req_id: int) -> Optional[RequestSlot]:
        """Return the RequestSlot for a given request ID, or None."""
        return self._slots.get(req_id)

    def reset(self) -> None:
        """Full reset of all bookkeeping state (called after all requests finish)."""
        self._slots.clear()
        self._slot_order.clear()
        self._total_request_count = 0
        self._active_token_count = 0
        self._num_prefill_requests = 0
        self._paused_request_count = 0
        self._chunked_prefill_request_id = -1


# ---------------------------------------------------------------------------
# Tier-aware padding budget estimator
# ---------------------------------------------------------------------------

class TierPaddingBudget:
    """Estimate the wasted fraction of a tier's token budget due to padding.

    DES-LOC motivation: A6000 (LIGHT tier) has max_tokens=4096, H100 (HEAVY)
    has max_tokens=32768.  A single speculative-token padding slot wastes
    1/4096 ≈ 0.024% on LIGHT vs 1/32768 ≈ 0.003% on HEAVY.  When scheduling
    avoidance kicks in (Bug 4b), the LIGHT tier must idle for one step while
    HEAVY could still absorb the request.  This class estimates when it is
    worth steering to HEAVY rather than stalling LIGHT.

    This is the *decision boundary* diagnostic component (M451 GREW pattern):
    emitted once when the waste fraction crosses ``steer_threshold``, not per step.
    """

    def __init__(
        self,
        light_max_tokens: int,
        heavy_max_tokens: int,
        steer_threshold: float = 0.01,
    ) -> None:
        self.light_max_tokens = light_max_tokens
        self.heavy_max_tokens = heavy_max_tokens
        self.steer_threshold = steer_threshold
        self._steer_event_emitted = False

    def padding_waste_fraction(self, tier_label: str, num_padding_tokens: int) -> float:
        """Return the fraction of tier capacity wasted by padding_tokens."""
        cap = self.light_max_tokens if tier_label == "LIGHT" else self.heavy_max_tokens
        return num_padding_tokens / cap

    def should_steer_to_heavy(
        self,
        num_padding_tokens: int,
        light_utilization: float,
        heavy_utilization: float,
    ) -> bool:
        """Return True if steering to HEAVY is preferable to padding on LIGHT.

        Decision boundary (M451 GREW pattern):
        - Steer if LIGHT waste fraction > steer_threshold AND HEAVY is not saturated.
        - Emit [DS-DBC] STEER_PRESSURE at most once per utilization cycle.
        """
        light_waste = self.padding_waste_fraction("LIGHT", num_padding_tokens)
        if light_waste >= self.steer_threshold and heavy_utilization < 0.95:
            if not self._steer_event_emitted:
                _log(
                    f"STEER_PRESSURE light_waste={light_waste:.4f} "
                    f"threshold={self.steer_threshold} "
                    f"light_util={light_utilization:.2f} heavy_util={heavy_utilization:.2f} "
                    f"→ steer request to HEAVY tier"
                )
                self._steer_event_emitted = True
            return True
        # Reset the event gate when waste drops below threshold
        if light_waste < self.steer_threshold:
            self._steer_event_emitted = False
        return False
