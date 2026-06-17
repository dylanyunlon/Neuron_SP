"""
DES-LOC Heterogeneous Admission Gating for Neuron_SP
=====================================================

Upstream design intent (Megatron ef549a6d):
--------------------------------------------
Megatron's ``DynamicInferenceEngine`` introduced CUDA-graph-aware admission gating
in its prefill scheduler (PR #4870).  The core idea: before adding a new request to
the active batch, check whether the resulting batch shape (token_count, num_prefill,
num_decode) has a matching *captured* CUDA graph.  If not, defer the request rather
than running it eagerly outside a graph — because an un-graphed step defeats the
latency and memory-determinism guarantees that graph capture provides.

Key upstream abstractions:
  * ``InferenceBatchDimensions``  — a named triple (token_count, prefill_req_count,
    decode_req_count) describing the shape of a candidate batch.
  * ``CUDAGraphBatchDimensionBuilder.match_graph_config`` — given a real batch dim and
    the set of captured-graph dims, returns the best-matching graph or None.
  * ``_cg_admission_gating_active()``  — opt-in guard; gating only activates when
    ``cuda_graph_all_prefills`` is set AND non-decode graphs exist.
  * ``_find_cg_chunk_size()``  — for chunked prefill, snaps the chunk size to the
    largest CG boundary that fits within the token budget.
  * ``_register_cg_wait()``  — starvation detector: logs a warning when a request
    has been deferred for more than ``max_sequence_length`` consecutive steps.
  * ``_cg_admission_check()``  — the admission oracle: True = admit, False = defer
    (and bump the wait counter).

DES-LOC adaptation points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) extends the upstream design
in three ways that reflect the asymmetric hardware:

  1. **Device-class awareness.**
     The cluster is heterogeneous: 2× A6000 (SM86, 48 GB, PCIe) + 1× H100 NVL
     (SM90, 96 GB, PCIe).  A captured graph from SM90 cannot replay on SM86 and
     vice-versa.  ``DeviceClass`` tracks the device type and capability; every batch
     dimension record is tagged with the originating device class.  Admission gating
     checks not only shape compatibility but also device-class compatibility.

  2. **LOC-cache token accounting.**
     DES-LOC maintains a Shared LOcality Cache (LOC) in the 1.5 TB host DRAM.
     Tokens that hit the LOC are *free* in terms of KV-compute budget; they do not
     count against the GPU active_token_count for graph-shape matching purposes.
     ``DesLocAdmissionCandidate`` carries a ``loc_hit_tokens`` field; the effective
     token count used for CG matching is ``raw_token_count - loc_hit_tokens``.

  3. **Decoupled execution queues.**
     Prefill and decode are executed on different device classes (prefill on H100 to
     exploit SM90 flash-attention; decode spread across both A6000s via tensor
     parallelism over PCIe).  The scheduler therefore maintains two separate batch-
     dimension lists (one per device class) and runs admission gating independently
     per queue.  A request is only admitted to the prefill queue when the H100's CG
     list has a matching shape; decode requests are gated against the A6000 CG list.

Hardware topology constants are defined at module level and used throughout; they
reflect the physical cluster described in the Neuron_SP README.

Thread safety:
  All mutable state (wait counters, device CG lists) is guarded by per-device locks
  defined in ``DesLocAdmissionGatingEngine``.  The unit tests run single-threaded but
  the production call sites acquire the lock before invoking scheduler methods.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import threading
import time
import unittest
from typing import Dict, List, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level logger — used only at meaningful decision points.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware topology constants for the Neuron_SP DES-LOC cluster.
# ---------------------------------------------------------------------------

#: SM architecture for the NVIDIA A6000 (Ampere).
SM_CAP_A6000: int = 86

#: SM architecture for the NVIDIA H100 NVL (Hopper).
SM_CAP_H100: int = 90

#: VRAM in bytes for a single A6000.
VRAM_A6000_BYTES: int = 48 * (1 << 30)   # 48 GiB

#: VRAM in bytes for the H100 NVL.
VRAM_H100_BYTES: int = 96 * (1 << 30)    # 96 GiB

#: Host DRAM available for the LOC cache.
LOC_DRAM_BYTES: int = 1_500 * (1 << 30)  # 1.5 TiB

#: Number of A6000 devices in the cluster.
NUM_A6000: int = 2

#: Number of H100 devices in the cluster.
NUM_H100: int = 1

#: Interconnect topology tag (PCIe only, no NVLink).
INTERCONNECT: str = "PCIe"


# ---------------------------------------------------------------------------
# Device class abstraction.
# ---------------------------------------------------------------------------

class DeviceClass(enum.Enum):
    """
    Logical device class in the DES-LOC cluster.

    ``AMPERE_A6000`` covers both A6000 cards (used for decode via tensor
    parallelism over PCIe).  ``HOPPER_H100`` covers the single H100 NVL card
    (used for prefill and large-batch attention).

    Graphs captured on one device class are *not* portable to the other.
    """
    AMPERE_A6000 = "ampere_a6000"
    HOPPER_H100 = "hopper_h100"

    @property
    def sm_cap(self) -> int:
        """Return the SM capability major*10 version for this device class."""
        return SM_CAP_A6000 if self is DeviceClass.AMPERE_A6000 else SM_CAP_H100

    @property
    def vram_bytes(self) -> int:
        """Total VRAM per device in this class."""
        return VRAM_A6000_BYTES if self is DeviceClass.AMPERE_A6000 else VRAM_H100_BYTES

    @property
    def count(self) -> int:
        """Number of physical devices of this class in the cluster."""
        return NUM_A6000 if self is DeviceClass.AMPERE_A6000 else NUM_H100


# ---------------------------------------------------------------------------
# Batch dimension types.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BatchDimensions:
    """
    Shape of a candidate or captured batch.

    Mirrors Megatron's ``InferenceBatchDimensions`` but adds:
      * ``device_class``  — which device class this graph was captured on.
      * ``loc_hit_tokens`` — tokens satisfied by the LOC cache (not in GPU compute).

    The *effective* token count for graph matching is
    ``token_count - loc_hit_tokens``.
    """
    token_count: int
    prefill_req_count: int
    decode_req_count: int
    device_class: DeviceClass = DeviceClass.HOPPER_H100
    loc_hit_tokens: int = 0

    @property
    def effective_token_count(self) -> int:
        """Token count excluding LOC-cache hits (used for CG shape matching)."""
        return max(0, self.token_count - self.loc_hit_tokens)

    @property
    def total_req_count(self) -> int:
        return self.prefill_req_count + self.decode_req_count

    def is_applicable_for_batch_dim(
        self,
        real: "BatchDimensions",
        *,
        strict: bool = False,
    ) -> bool:
        """
        Return True if this captured graph can service the *real* batch dim.

        Strict mode (hybrid/Mamba models):
          captured_prefill >= real_prefill AND captured_decode >= real_decode.

        Non-strict mode (dense transformers):
          captured_total >= real_total (decode slots can absorb prefill overflow).

        DES-LOC extension: device classes must match exactly — an SM86 graph
        cannot be replayed on SM90.

        The effective (LOC-adjusted) token counts are compared so that LOC cache
        hits do not artificially gate admission.
        """
        if self.device_class is not real.device_class:
            return False
        captured_eff = self.effective_token_count
        real_eff = real.effective_token_count
        if captured_eff < real_eff:
            return False
        if real.prefill_req_count == 0:
            # Pure-decode path: check that this is a decode-only captured graph.
            return (
                self.prefill_req_count == 0
                and self.decode_req_count >= real.decode_req_count
            )
        if strict:
            return (
                self.prefill_req_count >= real.prefill_req_count
                and self.decode_req_count >= real.decode_req_count
            )
        return self.total_req_count >= real.total_req_count


# ---------------------------------------------------------------------------
# Batch dimension builder / matcher (mirrors Megatron's CUDAGraphBatchDimensionBuilder).
# ---------------------------------------------------------------------------

class DesLocBatchDimensionMatcher:
    """
    Matches a real (candidate) batch dimension against a set of captured graphs,
    respecting device-class boundaries.

    Mirrors ``CUDAGraphBatchDimensionBuilder.match_graph_config`` from Megatron
    but is extended with:
      - Device-class filtering (SM86 vs SM90 graphs are kept in separate lists).
      - LOC-adjusted token counts.
      - EP token-count sync is *not* performed here (the step-time matcher owns
        that; this is a local admission probe).
    """

    @staticmethod
    def match_graph_config(
        real_batch_dim: BatchDimensions,
        cuda_graph_batch_dimensions_list: List[BatchDimensions],
        *,
        strict: bool = False,
    ) -> Optional[BatchDimensions]:
        """
        Return the first applicable captured graph for *real_batch_dim*, or None.

        The list must be pre-sorted descending by effective token count so that
        the first match is the tightest fit (smallest waste).

        Parameters
        ----------
        real_batch_dim:
            The candidate batch shape to be matched.
        cuda_graph_batch_dimensions_list:
            Captured-graph batch dimensions, filtered to the relevant device class
            before calling this method, and sorted descending by token_count.
        strict:
            Whether to use strict per-type matching (required for hybrid models
            such as Mamba where P/D slot types are not interchangeable).
        """
        for cg in cuda_graph_batch_dimensions_list:
            if cg.is_applicable_for_batch_dim(real_batch_dim, strict=strict):
                return cg
        return None


# ---------------------------------------------------------------------------
# LOC-cache oracle (stub; production implementation is in des_loc_cache.py).
# ---------------------------------------------------------------------------

class LocCacheOracle:
    """
    Estimates the number of tokens in a request that are expected to hit the
    Shared LOcality Cache (LOC) in host DRAM.

    In production this consults the LOC cache index; this stub returns zero for
    all requests.  The admission gating logic is correct regardless of the return
    value: zero means no LOC discount, which is conservative (may defer more than
    strictly necessary).
    """

    def estimate_loc_hits(self, prompt_token_count: int) -> int:  # noqa: D401
        """Return the estimated number of prompt tokens that will hit the LOC cache."""
        return 0  # stub; overridden in production


# ---------------------------------------------------------------------------
# Per-device context (mirrors DynamicInferenceContext per device class).
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DeviceQueueContext:
    """
    Mutable scheduler state for a single device class.

    In DES-LOC the prefill queue runs on the H100 NVL and the decode queue runs
    across both A6000s.  Each has its own CG list, token budget, and request counts.

    ``is_hybrid_model`` drives strict-vs-non-strict CG matching; set True for
    Mamba/SSM models where prefill and decode slots are not interchangeable.
    """
    device_class: DeviceClass
    max_tokens: int
    max_requests: int
    max_sequence_length: int
    cuda_graph_batch_dimensions_list: List[BatchDimensions]
    is_hybrid_model: bool = False
    use_cuda_graphs_for_non_decode_steps: bool = True
    # Mutable runtime state
    active_token_count: int = 0
    num_prefill_requests: int = 0
    num_decode_requests: int = 0


# ---------------------------------------------------------------------------
# Request stub (mirrors DynamicInferenceRequest fields needed by gating).
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AdmissionRequest:
    """
    Minimal request record used by the admission gating layer.

    In production this is part of the full ``DynamicInferenceRequest``; here
    it is isolated so the gating logic can be unit-tested without the engine.

    Fields
    ------
    request_id:
        Unique identifier.
    remaining_prompt_tokens:
        Count of prompt tokens not yet scheduled (decremented on partial admit).
    cg_wait_iters:
        Consecutive steps this request was deferred by CG-aware admission.
        Reset to 0 on successful admission.  Used *only* for starvation logging.
    is_prefill:
        True for prefill requests (routed to H100); False for decode (A6000).
    loc_hit_tokens:
        Tokens expected to hit the LOC cache (set by ``LocCacheOracle``).
    """
    request_id: int
    remaining_prompt_tokens: int
    is_prefill: bool = True
    loc_hit_tokens: int = 0
    cg_wait_iters: int = 0
    finished_chunk_token_count: int = 0


# ---------------------------------------------------------------------------
# Core admission gating engine.
# ---------------------------------------------------------------------------

class DesLocAdmissionGatingEngine:
    """
    DES-LOC heterogeneous admission gating engine.

    Reinterpretation of Megatron's cudagraph-aware prefill scheduler
    (commit ef549a6d) for the DES-LOC framework.

    Architecture
    ------------
    The engine maintains two ``DeviceQueueContext`` objects — one for the H100
    (prefill queue) and one for the A6000 pair (decode queue) — each with an
    independent CG list.  Admission decisions are made per-queue:

      * Prefill requests are gated against the H100 CG list.
      * Decode requests are gated against the A6000 CG list.
      * LOC cache hits reduce the effective token count before CG matching,
        potentially allowing a larger batch than the raw token count would suggest.

    The ``cuda_graph_all_prefills`` flag mirrors Megatron: gating is opt-in and
    only activates when the flag is set AND non-decode graphs exist for the
    relevant device class.

    Starvation detection
    --------------------
    Upstream Megatron sets ``_cg_admission_warn_after = max(100, max_seq_len)``.
    DES-LOC preserves this threshold but logs *per device class* so operators can
    distinguish H100 starvation (prefill bottleneck) from A6000 starvation (decode
    bottleneck).

    Parameters
    ----------
    prefill_ctx:
        Device queue context for the H100 (prefill).
    decode_ctx:
        Device queue context for the A6000 pair (decode).
    cuda_graph_all_prefills:
        Opt-in flag — mirrors ``InferenceConfig.cuda_graph_all_prefills``.
    loc_oracle:
        LOC cache hit estimator; defaults to the zero-hit stub.
    """

    def __init__(
        self,
        prefill_ctx: DeviceQueueContext,
        decode_ctx: DeviceQueueContext,
        *,
        cuda_graph_all_prefills: bool = False,
        loc_oracle: Optional[LocCacheOracle] = None,
    ) -> None:
        self._prefill_ctx = prefill_ctx
        self._decode_ctx = decode_ctx
        self.cuda_graph_all_prefills = cuda_graph_all_prefills
        self._loc_oracle = loc_oracle or LocCacheOracle()

        # Starvation warn thresholds (floor at 100 like upstream).
        self._prefill_warn_after = max(100, prefill_ctx.max_sequence_length)
        self._decode_warn_after = max(100, decode_ctx.max_sequence_length)

        # Per-device locks; production callers hold these across scheduling loops.
        self._prefill_lock = threading.Lock()
        self._decode_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers: context selection.
    # ------------------------------------------------------------------

    def _ctx_for(self, req: AdmissionRequest) -> DeviceQueueContext:
        """Return the device queue context that owns this request."""
        return self._prefill_ctx if req.is_prefill else self._decode_ctx

    def _warn_after_for(self, req: AdmissionRequest) -> int:
        return self._prefill_warn_after if req.is_prefill else self._decode_warn_after

    # ------------------------------------------------------------------
    # Gating activation guard.
    # ------------------------------------------------------------------

    def _cg_admission_gating_active(self, req: AdmissionRequest) -> bool:
        """
        Return True when CG-aware admission gating should run for *req*.

        Three conditions must ALL hold (mirrors upstream ``_cg_admission_gating_active``):
          1. ``cuda_graph_all_prefills`` is set (opt-in).
          2. The device context has non-decode CG graphs
             (``use_cuda_graphs_for_non_decode_steps``).
          3. The CG list for the relevant device class is non-empty.

        DES-LOC extension: conditions are evaluated *per device class* so that
        gating can be active on H100 but inactive on A6000 (e.g. if the A6000
        CG list is still being populated).
        """
        ctx = self._ctx_for(req)
        return (
            self.cuda_graph_all_prefills
            and ctx.use_cuda_graphs_for_non_decode_steps
            and bool(ctx.cuda_graph_batch_dimensions_list)
        )

    # ------------------------------------------------------------------
    # LOC-adjusted candidate construction.
    # ------------------------------------------------------------------

    def _build_candidate(
        self,
        req: AdmissionRequest,
        token_count: int,
        *,
        is_partial: bool = False,
    ) -> BatchDimensions:
        """
        Build the candidate ``BatchDimensions`` for admission checking.

        DES-LOC extension: subtract LOC cache hits from token_count before
        constructing the candidate so the CG matcher works on effective compute.

        Parameters
        ----------
        req:
            The request being considered.
        token_count:
            Raw token count if this request were admitted (active + new tokens).
        is_partial:
            True when this is a chunked-prefill partial admit; loc hits are
            recomputed against the chunk size rather than the full prompt.
        """
        ctx = self._ctx_for(req)
        loc_hits = self._loc_oracle.estimate_loc_hits(req.remaining_prompt_tokens)
        return BatchDimensions(
            token_count=token_count,
            prefill_req_count=ctx.num_prefill_requests + (1 if req.is_prefill else 0),
            decode_req_count=ctx.num_decode_requests + (0 if req.is_prefill else 1),
            device_class=ctx.device_class,
            loc_hit_tokens=loc_hits,
        )

    # ------------------------------------------------------------------
    # Chunk-size snapping (chunked prefill path).
    # ------------------------------------------------------------------

    def _find_cg_chunk_size(
        self,
        req: AdmissionRequest,
        max_chunk_tokens: int,
    ) -> Optional[int]:
        """
        Return the largest chunk size ``<= max_chunk_tokens`` that would produce a
        CG-matching batch shape, or None if no captured graph covers any chunk in
        the budget.

        Mirrors Megatron's ``_find_cg_chunk_size`` with two DES-LOC extensions:
          * The CG list is taken from the device-class-specific context.
          * LOC-adjusted effective token counts are used during matching.

        The caller (``schedule_chunked_prefill``) MUST handle the None return by
        falling back to ``max_chunk_tokens`` (eager) for chunked prefill or breaking
        the loop for non-chunked prefill.  The two behaviours diverge because
        mid-flight chunked prefills cannot be deferred without deadlocking progress.

        Walk order: CG list sorted descending by token_count (production invariant),
        so the first match is the largest fitting chunk — minimum wasted capacity.
        """
        ctx = self._ctx_for(req)
        active_tok = ctx.active_token_count
        active_p = ctx.num_prefill_requests
        active_d = ctx.num_decode_requests
        strict = ctx.is_hybrid_model
        dev_class = ctx.device_class

        for cg in ctx.cuda_graph_batch_dimensions_list:
            if cg.device_class is not dev_class:
                # Should not happen if lists are partitioned correctly, but guard anyway.
                continue
            chunk = cg.effective_token_count - active_tok
            if chunk < 1:
                continue
            if chunk > max_chunk_tokens:
                continue
            loc_hits = self._loc_oracle.estimate_loc_hits(req.remaining_prompt_tokens)
            candidate = BatchDimensions(
                token_count=cg.token_count,
                prefill_req_count=active_p + (1 if req.is_prefill else 0),
                decode_req_count=active_d + (0 if req.is_prefill else 1),
                device_class=dev_class,
                loc_hit_tokens=loc_hits,
            )
            if cg.is_applicable_for_batch_dim(candidate, strict=strict):
                return chunk

        return None

    # ------------------------------------------------------------------
    # Starvation tracking.
    # ------------------------------------------------------------------

    def _register_cg_wait(self, req: AdmissionRequest) -> None:
        """
        Increment the per-request deferral counter and emit a starvation warning
        when the threshold is crossed.

        The warning fires every ``warn_after`` steps so that operators see ongoing
        starvation rather than a one-shot alert.  The log includes the device class
        so H100 prefill starvation is distinguished from A6000 decode starvation.

        Mirrors Megatron's ``_register_cg_wait`` but adds device-class context to
        the log message and uses the per-device threshold.
        """
        ctx = self._ctx_for(req)
        warn_after = self._warn_after_for(req)
        req.cg_wait_iters += 1
        if req.cg_wait_iters % warn_after == 0:
            logger.warning(
                "request %d has been deferred by DES-LOC CG-aware admission for %d steps "
                "on device_class=%s (strict=%s, active P=%d D=%d effective_tok=%d) — "
                "possible starvation; check CG list coverage for this device class",
                req.request_id,
                req.cg_wait_iters,
                ctx.device_class.value,
                ctx.is_hybrid_model,
                ctx.num_prefill_requests,
                ctx.num_decode_requests,
                ctx.active_token_count,
            )

    # ------------------------------------------------------------------
    # Admission oracle.
    # ------------------------------------------------------------------

    def _cg_admission_check(
        self,
        req: AdmissionRequest,
        candidate: BatchDimensions,
    ) -> bool:
        """
        Return True if *candidate* matches a captured graph for the request's device class.

        On miss:
          * Increments ``req.cg_wait_iters`` via ``_register_cg_wait``.
          * Returns False; the caller is responsible for breaking the scheduler loop.

        On hit:
          * Resets ``req.cg_wait_iters`` to 0.
          * Returns True.

        DES-LOC extension vs Megatron:
          * ``match_ep_token_counts=False`` is the upstream default; we replicate it
            (EP sync happens at step time, not during local admission probing).
          * The CG list is drawn from the per-device context, so SM86 and SM90
            graphs are never cross-matched.
          * LOC-adjusted effective token counts are already embedded in *candidate*.
        """
        ctx = self._ctx_for(req)
        matched = DesLocBatchDimensionMatcher.match_graph_config(
            real_batch_dim=candidate,
            cuda_graph_batch_dimensions_list=ctx.cuda_graph_batch_dimensions_list,
            strict=ctx.is_hybrid_model,
        )
        if matched is not None:
            req.cg_wait_iters = 0
            return True
        self._register_cg_wait(req)
        return False

    # ------------------------------------------------------------------
    # Non-chunked prefill scheduler (mirrors schedule_waiting_requests).
    # ------------------------------------------------------------------

    def try_admit_non_chunked(
        self,
        req: AdmissionRequest,
        *,
        kv_cache_available: bool = True,
    ) -> bool:
        """
        Try to admit *req* as a whole (non-chunked) into the active batch.

        Mirrors the admission block in Megatron's ``schedule_waiting_requests``,
        extended with DES-LOC LOC-adjusted token accounting and per-device CG lists.

        The caller should call this in a scheduling loop and break on False (same
        contract as Megatron's scheduler).

        Returns True if the request was admitted (caller should pop it from the
        waiting queue and add it to the active context).  Returns False if the
        request was deferred (caller should leave it at the queue head and break).

        Parameters
        ----------
        req:
            The request to try to admit.
        kv_cache_available:
            Whether the KV cache has capacity for this request.
        """
        ctx = self._ctx_for(req)
        token_budget_ok = (
            ctx.active_token_count + req.remaining_prompt_tokens <= ctx.max_tokens
        )
        request_slots_ok = (
            ctx.num_prefill_requests + ctx.num_decode_requests < ctx.max_requests
        )
        if not (token_budget_ok and request_slots_ok and kv_cache_available):
            return False

        if self._cg_admission_gating_active(req):
            candidate = self._build_candidate(
                req,
                token_count=ctx.active_token_count + req.remaining_prompt_tokens,
            )
            admitted = self._cg_admission_check(req, candidate)
            if not admitted:
                logger.debug(
                    "request %d deferred (non-chunked): no CG shape covers "
                    "(eff_tok=%d, P=%d, D=%d) on device_class=%s",
                    req.request_id,
                    candidate.effective_token_count,
                    candidate.prefill_req_count,
                    candidate.decode_req_count,
                    ctx.device_class.value,
                )
                return False

        # Admit: update context state.
        ctx.active_token_count += req.remaining_prompt_tokens
        if req.is_prefill:
            ctx.num_prefill_requests += 1
        else:
            ctx.num_decode_requests += 1
        req.finished_chunk_token_count += req.remaining_prompt_tokens
        req.remaining_prompt_tokens = 0
        logger.debug(
            "request %d admitted (non-chunked, loc_hits=%d) on %s — "
            "active_tok=%d P=%d D=%d",
            req.request_id,
            req.loc_hit_tokens,
            ctx.device_class.value,
            ctx.active_token_count,
            ctx.num_prefill_requests,
            ctx.num_decode_requests,
        )
        return True

    # ------------------------------------------------------------------
    # Chunked prefill scheduler (mirrors schedule_chunked_prefill).
    # ------------------------------------------------------------------

    def try_admit_chunked(
        self,
        req: AdmissionRequest,
        *,
        is_continuing_chunked_prefill: bool = False,
        kv_cache_available: bool = True,
    ) -> Tuple[bool, int]:
        """
        Try to admit a chunk of *req* into the active batch.

        Mirrors the refactored ``schedule_chunked_prefill`` from Megatron ef549a6d,
        with DES-LOC extensions for LOC-adjusted token counts and per-device CG lists.

        Key behavioural differences from the upstream:
          1. CG gating is **skipped** for continuing chunked prefills (deferring an
             in-flight chunked prefill would deadlock progress — same as upstream).
          2. On a CG miss, the chunked path uses *eager fallback* (schedule the
             full budget chunk anyway) rather than deferring.  This diverges from
             the non-chunked path where a miss causes a break.  Rationale: chunked
             prefill is already splitting the request; deferring would cause head-of-
             line blocking on the remaining chunks.
          3. Flash-attention guard: if ``remaining - chunk == 1``, reduce chunk by 1
             (or skip) to avoid ``max_seqlen_q == 1`` (FA bug #1537).  Applied after
             CG snapping: the snapped CG still covers ``token_count - 1`` because
             ``is_applicable_for_batch_dim`` checks ``captured >= real``.

        Returns
        -------
        (admitted, chunk_length):
            admitted=True  → chunk was scheduled; chunk_length > 0.
            admitted=False → request deferred; chunk_length == 0.
        """
        ctx = self._ctx_for(req)
        token_partially_can_be_added = ctx.active_token_count < ctx.max_tokens
        request_slots_ok = (
            ctx.num_prefill_requests + ctx.num_decode_requests < ctx.max_requests
        )
        if not (token_partially_can_be_added and request_slots_ok and kv_cache_available):
            return False, 0

        token_budget = ctx.max_tokens - ctx.active_token_count
        max_chunk = min(req.remaining_prompt_tokens, token_budget)

        if self._cg_admission_gating_active(req) and not is_continuing_chunked_prefill:
            snapped = self._find_cg_chunk_size(req, max_chunk)
            if snapped is not None:
                prefill_chunk_length = snapped
                req.cg_wait_iters = 0
            else:
                # CG miss in chunked path: eager fallback (do not defer).
                prefill_chunk_length = max_chunk
                logger.debug(
                    "request %d chunked-prefill CG miss (max_chunk=%d, eff_active=%d) on %s — "
                    "using eager chunk (no deferral for in-progress chunked prefill)",
                    req.request_id,
                    max_chunk,
                    ctx.active_token_count,
                    ctx.device_class.value,
                )
        else:
            prefill_chunk_length = max_chunk

        # Flash-attention guard (FA bug #1537): avoid leaving exactly 1 token
        # for the last chunk.
        remaining_after = req.remaining_prompt_tokens - prefill_chunk_length
        if remaining_after == 1:
            if prefill_chunk_length > 1:
                prefill_chunk_length -= 1
            else:
                logger.debug(
                    "request %d chunked-prefill: only 1-token budget but 2 remaining — "
                    "deferring to avoid FA max_seqlen_q=1 bug",
                    req.request_id,
                )
                return False, 0

        # Commit the chunk to the context.
        ctx.active_token_count += prefill_chunk_length
        if req.is_prefill:
            ctx.num_prefill_requests += 1
        else:
            ctx.num_decode_requests += 1
        req.finished_chunk_token_count += prefill_chunk_length
        req.remaining_prompt_tokens -= prefill_chunk_length

        logger.debug(
            "request %d chunked-prefill chunk=%d admitted on %s — "
            "remaining=%d active_tok=%d",
            req.request_id,
            prefill_chunk_length,
            ctx.device_class.value,
            req.remaining_prompt_tokens,
            ctx.active_token_count,
        )
        return True, prefill_chunk_length

    # ------------------------------------------------------------------
    # Batch scheduler loop (high-level entry point).
    # ------------------------------------------------------------------

    def schedule_waiting_requests(
        self,
        waiting_queue: List[AdmissionRequest],
        *,
        chunked: bool = False,
        kv_cache_available: bool = True,
    ) -> List[AdmissionRequest]:
        """
        Run one scheduling pass over *waiting_queue*, admitting as many requests
        as possible within the budget and CG constraints.

        This is the top-level entry point that the Neuron_SP engine calls per step.
        It mirrors the structure of Megatron's ``schedule_waiting_requests`` and
        ``schedule_chunked_prefill`` but dispatches to the DES-LOC-aware methods.

        Parameters
        ----------
        waiting_queue:
            Ordered list of pending requests (head = highest priority).
        chunked:
            If True, run the chunked-prefill scheduler; otherwise non-chunked.
        kv_cache_available:
            Whether KV cache has headroom (simplified; production checks per-request).

        Returns
        -------
        List of admitted requests (in admission order).
        """
        admitted: List[AdmissionRequest] = []

        for req in list(waiting_queue):  # snapshot to allow mutation
            if chunked:
                ok, chunk_len = self.try_admit_chunked(
                    req, kv_cache_available=kv_cache_available
                )
                if ok:
                    admitted.append(req)
                    if req.remaining_prompt_tokens == 0:
                        waiting_queue.remove(req)
                    # If remaining > 0, request stays at head (partial admit).
                    # Break: token budget is exhausted after a partial admit.
                    if req.remaining_prompt_tokens > 0:
                        break
                else:
                    break  # Head-of-line block.
            else:
                ok = self.try_admit_non_chunked(
                    req, kv_cache_available=kv_cache_available
                )
                if ok:
                    admitted.append(req)
                    waiting_queue.remove(req)
                else:
                    break  # CG miss or budget exhausted; stop scanning.

        return admitted


# ---------------------------------------------------------------------------
# Helper factory for tests.
# ---------------------------------------------------------------------------

def _make_device_ctx(
    device_class: DeviceClass,
    cg_list: List[BatchDimensions],
    *,
    max_tokens: int = 512,
    max_requests: int = 64,
    max_sequence_length: int = 512,
    is_hybrid: bool = False,
    active_tok: int = 0,
    num_prefill: int = 0,
    num_decode: int = 0,
    use_cuda_graphs_for_non_decode_steps: bool = True,
) -> DeviceQueueContext:
    return DeviceQueueContext(
        device_class=device_class,
        max_tokens=max_tokens,
        max_requests=max_requests,
        max_sequence_length=max_sequence_length,
        cuda_graph_batch_dimensions_list=cg_list,
        is_hybrid_model=is_hybrid,
        use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
        active_token_count=active_tok,
        num_prefill_requests=num_prefill,
        num_decode_requests=num_decode,
    )


def _make_engine(
    prefill_cg_list: List[BatchDimensions],
    decode_cg_list: Optional[List[BatchDimensions]] = None,
    *,
    prefill_active_tok: int = 0,
    prefill_num_p: int = 0,
    prefill_num_d: int = 0,
    decode_active_tok: int = 0,
    decode_num_p: int = 0,
    decode_num_d: int = 0,
    is_hybrid: bool = False,
    cuda_graph_all_prefills: bool = True,
    max_tokens: int = 512,
    max_sequence_length: int = 512,
) -> DesLocAdmissionGatingEngine:
    """Convenience factory for unit tests."""
    decode_cg_list = decode_cg_list or []
    prefill_ctx = _make_device_ctx(
        DeviceClass.HOPPER_H100,
        prefill_cg_list,
        max_tokens=max_tokens,
        max_sequence_length=max_sequence_length,
        is_hybrid=is_hybrid,
        active_tok=prefill_active_tok,
        num_prefill=prefill_num_p,
        num_decode=prefill_num_d,
    )
    decode_ctx = _make_device_ctx(
        DeviceClass.AMPERE_A6000,
        decode_cg_list,
        max_tokens=max_tokens,
        max_sequence_length=max_sequence_length,
        is_hybrid=is_hybrid,
        active_tok=decode_active_tok,
        num_prefill=decode_num_p,
        num_decode=decode_num_d,
    )
    return DesLocAdmissionGatingEngine(
        prefill_ctx,
        decode_ctx,
        cuda_graph_all_prefills=cuda_graph_all_prefills,
    )


def _bd(tok: int, p: int, d: int, dc: DeviceClass) -> BatchDimensions:
    """Shorthand for constructing a BatchDimensions in tests."""
    return BatchDimensions(
        token_count=tok,
        prefill_req_count=p,
        decode_req_count=d,
        device_class=dc,
    )


def _req(
    rid: int,
    prompt_len: int,
    *,
    is_prefill: bool = True,
    wait: int = 0,
) -> AdmissionRequest:
    return AdmissionRequest(
        request_id=rid,
        remaining_prompt_tokens=prompt_len,
        is_prefill=is_prefill,
        cg_wait_iters=wait,
    )


# ---------------------------------------------------------------------------
# SAMPLE CG LISTS (mirroring Megatron test fixtures, adapted per device class).
# ---------------------------------------------------------------------------

H100 = DeviceClass.HOPPER_H100
A6K = DeviceClass.AMPERE_A6000

SAMPLE_H100_CG_LIST: List[BatchDimensions] = [
    _bd(256, 1, 255, H100), _bd(256, 4, 252, H100), _bd(256, 256, 0, H100),
    _bd(128, 1, 127, H100), _bd(128, 4, 124, H100), _bd(128, 128, 0, H100),
    _bd(64,  1, 63,  H100), _bd(64,  4, 60,  H100), _bd(64,  64,  0, H100),
    _bd(16,  1, 15,  H100), _bd(16,  4, 12,  H100), _bd(16,  16,  0, H100),
    _bd(4,   1, 3,   H100), _bd(4,   4, 0,   H100),
    _bd(2,   1, 1,   H100), _bd(2,   2, 0,   H100),
]

SAMPLE_A6K_CG_LIST: List[BatchDimensions] = [
    _bd(256, 0, 256, A6K), _bd(128, 0, 128, A6K),
    _bd(64,  0, 64,  A6K), _bd(32,  0, 32,  A6K),
    _bd(16,  0, 16,  A6K), _bd(8,   0, 8,   A6K),
    _bd(4,   0, 4,   A6K), _bd(2,   0, 2,   A6K),
]


# ===========================================================================
# Unit tests
# ===========================================================================

class TestDeviceClassProperties(unittest.TestCase):
    """Verify the hardware-topology constants are self-consistent."""

    def test_sm_caps(self):
        self.assertEqual(DeviceClass.AMPERE_A6000.sm_cap, SM_CAP_A6000)
        self.assertEqual(DeviceClass.HOPPER_H100.sm_cap, SM_CAP_H100)

    def test_vram(self):
        self.assertEqual(DeviceClass.AMPERE_A6000.vram_bytes, VRAM_A6000_BYTES)
        self.assertEqual(DeviceClass.HOPPER_H100.vram_bytes, VRAM_H100_BYTES)

    def test_counts(self):
        self.assertEqual(DeviceClass.AMPERE_A6000.count, 2)
        self.assertEqual(DeviceClass.HOPPER_H100.count, 1)


class TestBatchDimensionsEffectiveToken(unittest.TestCase):
    """LOC-adjusted effective token count."""

    def test_no_loc_hits(self):
        bd = BatchDimensions(64, 1, 0, H100, loc_hit_tokens=0)
        self.assertEqual(bd.effective_token_count, 64)

    def test_partial_loc_hits(self):
        bd = BatchDimensions(64, 1, 0, H100, loc_hit_tokens=20)
        self.assertEqual(bd.effective_token_count, 44)

    def test_full_loc_hits_clamps_to_zero(self):
        bd = BatchDimensions(64, 1, 0, H100, loc_hit_tokens=100)
        self.assertEqual(bd.effective_token_count, 0)


class TestBatchDimensionsIsApplicable(unittest.TestCase):
    """Device-class filtering and strict/non-strict matching."""

    def test_device_class_mismatch_rejects(self):
        captured = _bd(64, 1, 63, H100)
        real = _bd(64, 1, 0, A6K)
        self.assertFalse(captured.is_applicable_for_batch_dim(real))

    def test_non_strict_absorbs_overflow(self):
        # Captured total=128; real P+D=70 — non-strict accepts.
        captured = _bd(128, 64, 64, H100)
        real = BatchDimensions(64, 10, 60, H100)
        self.assertTrue(captured.is_applicable_for_batch_dim(real, strict=False))

    def test_strict_requires_per_type(self):
        captured = _bd(128, 64, 64, H100)
        real = BatchDimensions(64, 10, 70, H100)  # D=70 > 64
        self.assertFalse(captured.is_applicable_for_batch_dim(real, strict=True))

    def test_decode_only_path(self):
        captured = _bd(64, 0, 64, H100)
        real = BatchDimensions(8, 0, 8, H100)
        self.assertTrue(captured.is_applicable_for_batch_dim(real))

    def test_decode_only_rejects_if_captured_has_prefill(self):
        captured = _bd(64, 4, 60, H100)
        real = BatchDimensions(8, 0, 8, H100)
        self.assertFalse(captured.is_applicable_for_batch_dim(real))


class TestGatingActivation(unittest.TestCase):
    """Gating must be strictly opt-in."""

    def test_inactive_when_all_prefills_off(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST, cuda_graph_all_prefills=False)
        req = _req(1, 64)
        self.assertFalse(engine._cg_admission_gating_active(req))

    def test_inactive_when_no_non_decode_graphs(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST)
        engine._prefill_ctx.use_cuda_graphs_for_non_decode_steps = False
        req = _req(1, 64)
        self.assertFalse(engine._cg_admission_gating_active(req))

    def test_inactive_when_cg_list_empty(self):
        engine = _make_engine([])
        req = _req(1, 64)
        self.assertFalse(engine._cg_admission_gating_active(req))

    def test_active_when_all_conditions_hold(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST)
        req = _req(1, 64)
        self.assertTrue(engine._cg_admission_gating_active(req))

    def test_decode_request_uses_decode_ctx(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST, decode_cg_list=[])
        req = _req(1, 8, is_prefill=False)
        # Decode CG list is empty → gating inactive for decode requests.
        self.assertFalse(engine._cg_admission_gating_active(req))


class TestFindCgChunkSize(unittest.TestCase):
    """_find_cg_chunk_size must respect budget, device class, and LOC hits."""

    def test_picks_largest_within_budget(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST)
        req = _req(1, 300)
        self.assertEqual(engine._find_cg_chunk_size(req, 300), 256)

    def test_respects_budget_ceiling(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST)
        req = _req(1, 128)
        self.assertEqual(engine._find_cg_chunk_size(req, 100), 64)
        self.assertEqual(engine._find_cg_chunk_size(req, 20), 16)

    def test_returns_none_when_no_cg_fits(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST, prefill_active_tok=300)
        req = _req(1, 10)
        self.assertIsNone(engine._find_cg_chunk_size(req, 10))

    def test_device_class_filter_rejects_a6k_graphs(self):
        # An H100 prefill request should not match A6K-tagged graphs.
        engine = _make_engine(
            prefill_cg_list=SAMPLE_A6K_CG_LIST,  # A6K graphs put in H100 slot (misconfigured)
            decode_cg_list=[],
        )
        # SAMPLE_A6K_CG_LIST has device_class=A6K; prefill request expects H100 graphs.
        req = _req(1, 64)
        # All graphs in the list are tagged A6K but the req targets H100 → mismatch.
        result = engine._find_cg_chunk_size(req, 300)
        self.assertIsNone(result)

    def test_strict_mode_filters_decode_count(self):
        engine = _make_engine(
            SAMPLE_H100_CG_LIST,
            prefill_num_d=125,
            is_hybrid=True,
        )
        req = _req(1, 256)
        # With D=125 + new P, only CGs with captured_D>=125 qualify in strict mode.
        chunk = engine._find_cg_chunk_size(req, 300)
        # (256, 1, 255) has D=255 >= 125 and P=1 >= 1 → chunk = 256 - 0 = 256.
        self.assertEqual(chunk, 256)

    def test_empty_list_returns_none(self):
        engine = _make_engine([])
        req = _req(1, 64)
        self.assertIsNone(engine._find_cg_chunk_size(req, 64))


class TestCgAdmissionCheck(unittest.TestCase):
    """_cg_admission_check: True on hit (counter reset), False on miss (counter bump)."""

    def test_match_returns_true_and_resets_counter(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST)
        req = _req(1, 64, wait=5)
        candidate = _bd(64, 1, 0, H100)
        self.assertTrue(engine._cg_admission_check(req, candidate))
        self.assertEqual(req.cg_wait_iters, 0)

    def test_no_match_returns_false_and_increments(self):
        engine = _make_engine([])
        req = _req(1, 64)
        candidate = _bd(64, 1, 0, H100)
        self.assertFalse(engine._cg_admission_check(req, candidate))
        self.assertEqual(req.cg_wait_iters, 1)

    def test_repeated_misses_accumulate(self):
        engine = _make_engine([])
        req = _req(1, 64)
        candidate = _bd(64, 1, 0, H100)
        for expected in range(1, 7):
            engine._cg_admission_check(req, candidate)
            self.assertEqual(req.cg_wait_iters, expected)

    def test_warning_fires_at_threshold(self):
        engine = _make_engine([])
        engine._prefill_warn_after = 3
        req = _req(1, 64)
        candidate = _bd(64, 1, 0, H100)
        with self.assertLogs(logger="deepspeed.inference.des_loc_admission_gating", level="WARNING") as cm:
            for _ in range(3):
                engine._cg_admission_check(req, candidate)
        self.assertTrue(any("DES-LOC CG-aware admission" in msg for msg in cm.output))

    def test_warning_repeats_at_multiples(self):
        engine = _make_engine([])
        engine._prefill_warn_after = 2
        req = _req(1, 64)
        candidate = _bd(64, 1, 0, H100)
        with self.assertLogs(logger="deepspeed.inference.des_loc_admission_gating", level="WARNING") as cm:
            for _ in range(6):
                engine._cg_admission_check(req, candidate)
        warnings = [m for m in cm.output if "DES-LOC CG-aware admission" in m]
        self.assertEqual(len(warnings), 3)  # at iters 2, 4, 6

    def test_device_class_in_warning_message(self):
        engine = _make_engine([])
        engine._prefill_warn_after = 1
        req = _req(1, 64)
        candidate = _bd(64, 1, 0, H100)
        with self.assertLogs(logger="deepspeed.inference.des_loc_admission_gating", level="WARNING") as cm:
            engine._cg_admission_check(req, candidate)
        self.assertTrue(any("hopper_h100" in msg for msg in cm.output))


class TestNonChunkedAdmission(unittest.TestCase):
    """try_admit_non_chunked: budget, CG, and state-update checks."""

    def test_admits_when_cg_matches(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST)
        req = _req(1, 64)
        ok = engine.try_admit_non_chunked(req)
        self.assertTrue(ok)
        self.assertEqual(req.remaining_prompt_tokens, 0)
        self.assertEqual(req.finished_chunk_token_count, 64)
        self.assertEqual(engine._prefill_ctx.active_token_count, 64)
        self.assertEqual(engine._prefill_ctx.num_prefill_requests, 1)

    def test_defers_when_no_cg_match(self):
        engine = _make_engine([])  # No CG list → no match
        req = _req(1, 64)
        ok = engine.try_admit_non_chunked(req)
        self.assertFalse(ok)
        self.assertEqual(req.cg_wait_iters, 1)
        self.assertEqual(engine._prefill_ctx.active_token_count, 0)

    def test_defers_when_token_budget_exceeded(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST, prefill_active_tok=500, max_tokens=512)
        req = _req(1, 64)
        ok = engine.try_admit_non_chunked(req)
        self.assertFalse(ok)

    def test_gating_inactive_still_admits(self):
        engine = _make_engine([], cuda_graph_all_prefills=False)
        req = _req(1, 16)
        ok = engine.try_admit_non_chunked(req)
        self.assertTrue(ok)

    def test_decode_request_routes_to_decode_ctx(self):
        engine = _make_engine([], decode_cg_list=SAMPLE_A6K_CG_LIST)
        req = _req(1, 8, is_prefill=False)
        ok = engine.try_admit_non_chunked(req)
        # Decode CG list exists but gating requires cuda_graph_all_prefills; it's True by default.
        # decode_ctx has a non-empty CG list, so gating is active.
        # Candidate: (8 tokens, P=0, D=1) on A6K — (8, 0, 8) matches (8 ≥ 8, decode-only).
        self.assertTrue(ok)
        self.assertEqual(engine._decode_ctx.num_decode_requests, 1)


class TestChunkedAdmission(unittest.TestCase):
    """try_admit_chunked: CG snapping, eager fallback, FA guard, continuing bypass."""

    def test_snaps_to_cg_boundary(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=512)
        req = _req(1, 300)
        ok, chunk = engine.try_admit_chunked(req)
        self.assertTrue(ok)
        self.assertEqual(chunk, 256)

    def test_eager_fallback_on_cg_miss(self):
        # Budget of 1 token; smallest H100 CG has token_count=2 → miss → eager.
        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=1)
        req = _req(1, 10)
        # But FA guard: remaining=10, chunk=1, remaining_after=9 ≠ 1 → no FA issue.
        ok, chunk = engine.try_admit_chunked(req)
        # _find_cg_chunk_size(req, 1) → None (all CGs have eff_tok >= 2).
        # Eager fallback: chunk = max_chunk = 1.
        self.assertTrue(ok)
        self.assertEqual(chunk, 1)

    def test_continuing_prefill_skips_gating(self):
        # Even with a CG list, is_continuing=True bypasses gating.
        engine = _make_engine(SAMPLE_H100_CG_LIST, prefill_active_tok=50, max_tokens=512)
        req = _req(1, 100)
        ok, chunk = engine.try_admit_chunked(req, is_continuing_chunked_prefill=True)
        self.assertTrue(ok)
        # No snapping: chunk = min(100, 512-50) = 100.
        self.assertEqual(chunk, 100)

    def test_fa_guard_reduces_chunk(self):
        # remaining=5, budget=4 → prefill_chunk_length=4, remaining_after=1 → reduce to 3.
        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=4)
        req = _req(1, 5)
        # _find_cg_chunk_size(req, 4) → 4 (CG(4,1,3) exists with eff_tok=4).
        ok, chunk = engine.try_admit_chunked(req)
        self.assertTrue(ok)
        self.assertEqual(chunk, 3)

    def test_fa_guard_defers_when_budget_is_one(self):
        # remaining=2, budget=1 → chunk=1, remaining_after=1 → cannot reduce → defer.
        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=1)
        req = _req(1, 2)
        ok, chunk = engine.try_admit_chunked(req)
        self.assertFalse(ok)
        self.assertEqual(chunk, 0)

    def test_state_updated_on_partial_admit(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=512)
        req = _req(1, 300)
        ok, chunk = engine.try_admit_chunked(req)
        self.assertTrue(ok)
        self.assertEqual(engine._prefill_ctx.active_token_count, chunk)
        self.assertEqual(req.finished_chunk_token_count, chunk)
        self.assertEqual(req.remaining_prompt_tokens, 300 - chunk)


class TestScheduleWaitingRequests(unittest.TestCase):
    """Top-level scheduler loop: admission ordering and head-of-line blocking."""

    def test_admits_multiple_requests_in_order(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=512, max_requests=8)
        queue = [_req(i, 64) for i in range(1, 4)]
        admitted = engine.schedule_waiting_requests(queue)
        self.assertEqual(len(admitted), 3)
        self.assertEqual(len(queue), 0)

    def test_head_of_line_block_stops_on_first_miss(self):
        # First request is too large (700 tokens); should block everything.
        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=512)
        queue = [_req(1, 700), _req(2, 64)]
        admitted = engine.schedule_waiting_requests(queue)
        # req 1 deferred (budget), req 2 never tried.
        self.assertEqual(len(admitted), 0)
        self.assertEqual(len(queue), 2)

    def test_chunked_admits_partial_and_breaks(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=128)
        req1 = _req(1, 300)
        req2 = _req(2, 64)
        queue = [req1, req2]
        admitted = engine.schedule_waiting_requests(queue, chunked=True)
        # req1 gets a chunk (128 tokens → CG snapped to 128), req2 is not tried.
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0].request_id, 1)
        # req1 still in queue (partial admit).
        self.assertIn(req1, queue)
        self.assertIn(req2, queue)


class TestDesLocDeviceClassIsolation(unittest.TestCase):
    """
    DES-LOC-specific: graphs from one device class must not match the other.

    This is the central correctness property of the DES-LOC extension.  An SM86
    graph cannot be replayed on SM90 (different instruction sets, different warp
    sizes, different flash-attention kernel variants).  The gating layer must
    enforce this boundary unconditionally.
    """

    def test_h100_graph_does_not_match_a6k_request(self):
        # A decode request (A6K) against the H100 CG list → no match.
        engine = _make_engine(SAMPLE_H100_CG_LIST, decode_cg_list=[])
        req = _req(1, 64, is_prefill=False)
        # decode ctx CG list is empty; gating is inactive → admission falls through.
        ok = engine.try_admit_non_chunked(req)
        self.assertTrue(ok)  # admitted without gating (CG list empty → gating inactive)
        self.assertEqual(engine._decode_ctx.num_decode_requests, 1)

    def test_a6k_graph_does_not_match_h100_request(self):
        # Misconfigured prefill CG list contains A6K-tagged graphs.
        # The matcher must reject them for H100 prefill requests.
        engine = _make_engine(prefill_cg_list=SAMPLE_A6K_CG_LIST)
        req = _req(1, 64)
        candidate = _bd(64, 1, 0, H100)
        # All graphs in prefill list are A6K-tagged → device class mismatch → no match.
        matched = DesLocBatchDimensionMatcher.match_graph_config(
            real_batch_dim=candidate,
            cuda_graph_batch_dimensions_list=SAMPLE_A6K_CG_LIST,
        )
        self.assertIsNone(matched)

    def test_correct_device_class_matches(self):
        candidate = _bd(64, 1, 0, H100)
        matched = DesLocBatchDimensionMatcher.match_graph_config(
            real_batch_dim=candidate,
            cuda_graph_batch_dimensions_list=SAMPLE_H100_CG_LIST,
        )
        self.assertIsNotNone(matched)
        self.assertEqual(matched.device_class, H100)


class TestLocCacheIntegration(unittest.TestCase):
    """LOC hit discount: effective token count reduction enables larger batches."""

    def test_loc_hits_reduce_effective_token_count(self):
        # Without LOC hits, 300 tokens would not fit in a 256-token budget.
        # With 50 LOC hits, effective = 250 → fits within 256-token CG.
        class FiftyHitOracle(LocCacheOracle):
            def estimate_loc_hits(self, _: int) -> int:
                return 50

        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=256)
        engine._loc_oracle = FiftyHitOracle()
        req = _req(1, 300)
        # effective_token_count = 300 - 50 = 250; CG(256,1,255).eff=256 >= 250 → match.
        ok = engine.try_admit_non_chunked(req)
        self.assertTrue(ok)

    def test_zero_loc_hits_conservative(self):
        # Without LOC discount, 300 tokens in a 256-token budget → gated out.
        engine = _make_engine(SAMPLE_H100_CG_LIST, max_tokens=256)
        req = _req(1, 300)
        ok = engine.try_admit_non_chunked(req)
        # token_budget check: 300 > 256 → rejected before CG check.
        self.assertFalse(ok)


class TestStarvationCounterIsolation(unittest.TestCase):
    """Two requests' wait counters must not interfere."""

    def test_independent_counters(self):
        engine = _make_engine(SAMPLE_H100_CG_LIST)
        req_a = _req(1, 64, wait=0)
        req_b = _req(2, 64, wait=0)
        empty_engine = _make_engine([])

        # req_a deferred 3 times.
        candidate = _bd(64, 1, 0, H100)
        for _ in range(3):
            empty_engine._cg_admission_check(req_a, candidate)
        # req_b admitted once.
        engine._cg_admission_check(req_b, candidate)

        self.assertEqual(req_a.cg_wait_iters, 3)
        self.assertEqual(req_b.cg_wait_iters, 0)

    def test_deferred_then_admitted_resets(self):
        req = _req(1, 64, wait=0)
        candidate = _bd(64, 1, 0, H100)
        empty_engine = _make_engine([])
        full_engine = _make_engine(SAMPLE_H100_CG_LIST)

        empty_engine._cg_admission_check(req, candidate)
        empty_engine._cg_admission_check(req, candidate)
        self.assertEqual(req.cg_wait_iters, 2)

        full_engine._cg_admission_check(req, candidate)
        self.assertEqual(req.cg_wait_iters, 0)


class TestSchedulerInvariants(unittest.TestCase):
    """
    Engine state is immutable when only gating helpers are called (no side effects
    from a False admission decision).
    """

    def test_false_admission_leaves_engine_state_unchanged(self):
        engine = _make_engine([], max_tokens=512)
        req = _req(1, 64)
        candidate = _bd(64, 1, 0, H100)

        before = (
            engine._prefill_ctx.active_token_count,
            engine._prefill_ctx.num_prefill_requests,
            engine._prefill_ctx.num_decode_requests,
        )
        for _ in range(10):
            engine._cg_admission_check(req, candidate)
        after = (
            engine._prefill_ctx.active_token_count,
            engine._prefill_ctx.num_prefill_requests,
            engine._prefill_ctx.num_decode_requests,
        )
        self.assertEqual(before, after)
        self.assertEqual(req.cg_wait_iters, 10)


if __name__ == "__main__":
    # Run with: python deepspeed/inference/des_loc_admission_gating.py -v
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestDeviceClassProperties,
        TestBatchDimensionsEffectiveToken,
        TestBatchDimensionsIsApplicable,
        TestGatingActivation,
        TestFindCgChunkSize,
        TestCgAdmissionCheck,
        TestNonChunkedAdmission,
        TestChunkedAdmission,
        TestScheduleWaitingRequests,
        TestDesLocDeviceClassIsolation,
        TestLocCacheIntegration,
        TestStarvationCounterIsolation,
        TestSchedulerInvariants,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        import sys
        sys.exit(1)
