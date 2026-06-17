"""
DES-LOC Heterogeneous CUDAGraph-Aware Admission Gating
=======================================================

Upstream design intent (Megatron ef549a6d):
    Megatron's dynamic inference engine gained "cudagraph-aware admission gating" for its prefill
    scheduler. The core insight: when a model is compiled under CUDA graph capture, every batch
    shape (token_count, num_prefill_reqs, num_decode_reqs) must exactly match a previously captured
    graph. Admitting a request that produces an uncaptured shape forces an eager fallback — or
    worse, a runtime crash on hybrid (Mamba/SSM) models where strict shape matching is mandatory.

    Megatron's solution: before admitting a request into the active batch, probe the captured-graph
    registry to confirm the resulting batch shape has a matching CG. If no match exists, defer the
    request (increment a starvation counter, warn at threshold) rather than schedule eagerly.
    For chunked prefill, snap the chunk size to the largest CG-aligned boundary within the token
    budget; fall back to eager only if no CG covers the budget at all.

DES-LOC adaptation rationale:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) runs inference across a heterogeneous
    device pool: in the Neuron_SP reference deployment, two A6000 (48 GB, SM86) and one H100 NVL
    (96 GB, SM90) connected over PCIe with 1.5 TB of CPU DRAM as a locality cache tier.

    This heterogeneity introduces three complications absent from Megatron's homogeneous NVLink
    setup:

    1. **Per-device CG registries**: Each device archetype (SM86 vs SM90) captures its own graph
       set at different token granularities reflecting different arithmetic throughput and memory
       bandwidth. A batch shape valid for the H100 may not be captured on the A6000s.

    2. **PCIe-bottlenecked EP sync**: Megatron's `match_ep_token_counts=False` avoids an NCCL
       all-reduce during the admission probe. In DES-LOC, even the lightweight probe must be
       device-aware: probing the A6000 registry for an H100-bound batch is wrong.

    3. **LOC cache pressure**: The Shared LOcality Cache (CPU DRAM) holds KV blocks evicted from
       GPU HBM. Admitting a batch that misses on all device CG registries wastes a full PCIe
       round-trip fetching LOC blocks into GPU memory, only to stall on an eager-mode forward.
       CG-aware admission prevents this by deferring until a shape with a matched graph (and thus
       predictable execution time) can be formed.

    This module re-implements Megatron's admission gating as a device-topology-aware subsystem
    suitable for DeepSpeed's inference pipeline under DES-LOC. Key differences:

    - ``DeviceRole`` classifies each GPU by SM architecture and HBM capacity.
    - ``HeteroCudagraphRegistry`` maintains per-device-role CG lists; lookup is always
      scoped to the role assigned to the current batch's placement plan.
    - ``HeteroAdmissionGate`` replaces Megatron's three inline methods
      (``_cg_admission_gating_active``, ``_find_cg_chunk_size``, ``_cg_admission_check``) with
      a single, device-aware gate object that can be queried by DeepSpeed's scheduler.
    - Starvation warnings carry device-role context so operators know *which* device archetype
      is the bottleneck.
    - The LOC cache tier is consulted during admission: if a deferred request has all its KV
      blocks resident in the LOC cache, the gate may grant a "LOC-priority bypass" — admitting
      the request even without a perfect CG match, trading graph efficiency for cache locality.
      This bypass is rate-limited and logged explicitly.

References:
    - Megatron commit ef549a6d0792485d1e5c4db049e82dc00f38ecca
    - DeepSpeed inference pipeline: deepspeed/inference/engine.py
    - Neuron_SP DES-LOC design doc: docs/des_loc_design.md
    - Flash Attention edge case: https://github.com/Dao-AILab/flash-attention/issues/1537
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device taxonomy
# ---------------------------------------------------------------------------

class DeviceRole(Enum):
    """Coarse-grained device taxonomy sufficient for CG-registry partitioning.

    DES-LOC deployment: A6000 × 2 (SM86, 48 GB) + H100 NVL × 1 (SM90, 96 GB).
    The SM version determines which CUDA instructions are available; capacity
    determines the maximum KV cache size and thus the maximum batch shapes that
    can be captured on that device.
    """
    A6000_SM86 = auto()   # 48 GB, PCIe, compute cap 8.6
    H100_SM90  = auto()   # 96 GB, PCIe NVL, compute cap 9.0
    UNKNOWN    = auto()   # fallback for unrecognised devices

    @classmethod
    def from_device(cls, device: torch.device) -> "DeviceRole":
        """Classify a torch.device by querying CUDA device properties.

        Falls back to UNKNOWN for non-CUDA devices or unrecognised architectures
        so the rest of the admission logic degrades gracefully.
        """
        if device.type != "cuda":
            return cls.UNKNOWN
        try:
            props = torch.cuda.get_device_properties(device)
            major, minor = props.major, props.minor
            total_mb = props.total_memory // (1024 * 1024)
            if major == 9 and minor == 0 and total_mb >= 80_000:
                return cls.H100_SM90
            if major == 8 and minor == 6 and total_mb <= 52_000:
                return cls.A6000_SM86
        except Exception:
            pass
        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# Batch dimension primitives
# ---------------------------------------------------------------------------

class BatchDimensions(NamedTuple):
    """Minimal description of a batch shape for CG matching.

    Mirrors Megatron's ``InferenceBatchDimensions`` with an added ``device_role``
    so the registry lookup is always device-scoped.

    Attributes:
        token_count:        Total number of tokens in the batch (prefill + decode).
        prefill_req_count:  Number of requests currently in prefill phase.
        decode_req_count:   Number of requests currently in decode phase.
        device_role:        Device archetype this batch is destined for.
    """
    token_count: int
    prefill_req_count: int
    decode_req_count: int
    device_role: DeviceRole = DeviceRole.UNKNOWN


@dataclass
class CapturedGraph:
    """A single captured CUDA graph entry in the registry.

    Attributes:
        token_count:        The token_count this graph was captured at.
        prefill_req_count:  The prefill_req_count at capture time.
        decode_req_count:   The decode_req_count at capture time.
        device_role:        The device archetype this graph lives on.
        is_hybrid:          Whether the model is a hybrid (SSM/Mamba) requiring strict matching.
    """
    token_count: int
    prefill_req_count: int
    decode_req_count: int
    device_role: DeviceRole
    is_hybrid: bool = False

    # ------------------------------------------------------------------
    # Matching predicates
    # ------------------------------------------------------------------

    def is_applicable_for_batch_dim(
        self,
        candidate: BatchDimensions,
        strict: bool = False,
    ) -> bool:
        """Return True if this captured graph can service ``candidate``.

        Non-strict (transformer-only) mode:
            The graph can service the candidate if its total token capacity is at
            least as large as the candidate's token count.  Prefill/decode slots
            are interchangeable — the graph's total req capacity is
            ``prefill_req_count + decode_req_count``, and the candidate's total
            is ``prefill_req_count + decode_req_count``.  A match requires:
                captured.token_count >= candidate.token_count
                captured.total_reqs  >= candidate.total_reqs

        Strict (hybrid/SSM) mode:
            SSM state is maintained per-request-type: a decode state cannot be
            reused for a prefill step.  Match requires:
                captured.token_count      >= candidate.token_count
                captured.prefill_req_count >= candidate.prefill_req_count
                captured.decode_req_count  >= candidate.decode_req_count

        Device role must always match (device isolation invariant of DES-LOC).
        """
        if self.device_role != candidate.device_role:
            return False
        if self.token_count < candidate.token_count:
            return False
        if strict:
            return (
                self.prefill_req_count >= candidate.prefill_req_count
                and self.decode_req_count >= candidate.decode_req_count
            )
        # Non-strict: total slot capacity check.
        captured_total = self.prefill_req_count + self.decode_req_count
        candidate_total = candidate.prefill_req_count + candidate.decode_req_count
        return captured_total >= candidate_total


# ---------------------------------------------------------------------------
# Per-device CG registry
# ---------------------------------------------------------------------------

class HeteroCudagraphRegistry:
    """Maintains per-device-role captured-graph lists, sorted descending by token_count.

    In a DES-LOC deployment the H100 can capture graphs at larger token counts
    (more HBM) than the A6000s.  Keeping separate sorted lists lets the
    admission gate do O(n_graphs) lookup scoped to the target device without
    cross-device noise.

    Usage::

        registry = HeteroCudagraphRegistry()
        registry.register(CapturedGraph(token_count=256, prefill_req_count=1,
                                        decode_req_count=255, device_role=DeviceRole.H100_SM90))
        graphs = registry.get_sorted(DeviceRole.H100_SM90)
    """

    def __init__(self) -> None:
        # device_role -> list of CapturedGraph sorted descending by token_count
        self._graphs: Dict[DeviceRole, List[CapturedGraph]] = defaultdict(list)
        self._dirty: Dict[DeviceRole, bool] = defaultdict(bool)

    def register(self, cg: CapturedGraph) -> None:
        """Add a captured graph to the registry.

        Marks the role's list as dirty so it is re-sorted on the next ``get_sorted`` call.
        Sorting is deferred to avoid O(n log n) cost on every register during startup when
        all graphs are registered in bulk.
        """
        self._graphs[cg.device_role].append(cg)
        self._dirty[cg.device_role] = True
        logger.debug(
            "Registered CG: device_role=%s tok=%d P=%d D=%d hybrid=%s",
            cg.device_role.name, cg.token_count,
            cg.prefill_req_count, cg.decode_req_count, cg.is_hybrid,
        )

    def register_bulk(self, graphs: Sequence[CapturedGraph]) -> None:
        """Register multiple graphs at once and sort once per role touched."""
        touched: set = set()
        for cg in graphs:
            self._graphs[cg.device_role].append(cg)
            touched.add(cg.device_role)
        for role in touched:
            self._dirty[role] = True

    def get_sorted(self, role: DeviceRole) -> List[CapturedGraph]:
        """Return the list of captured graphs for ``role``, sorted descending by token_count.

        Sort is performed lazily on dirty lists.
        """
        if self._dirty.get(role, False):
            self._graphs[role].sort(key=lambda g: g.token_count, reverse=True)
            self._dirty[role] = False
        return self._graphs[role]

    def all_roles(self) -> List[DeviceRole]:
        """Return all device roles that have at least one registered graph."""
        return [role for role, graphs in self._graphs.items() if graphs]

    def __len__(self) -> int:
        return sum(len(g) for g in self._graphs.values())


# ---------------------------------------------------------------------------
# LOC cache tier interface
# ---------------------------------------------------------------------------

@dataclass
class LOCCacheStatus:
    """Summary of a request's KV block residency in the LOC (CPU DRAM) cache tier.

    Attributes:
        request_id:          Identifies the request.
        blocks_in_loc:       Number of KV blocks currently resident in the LOC tier.
        total_blocks:        Total KV blocks required by this request.
        loc_hit_ratio:       ``blocks_in_loc / total_blocks``, or 0.0 if total is 0.
        last_access_ts:      Unix timestamp of the most recent LOC access for this request.
    """
    request_id: int
    blocks_in_loc: int
    total_blocks: int
    loc_hit_ratio: float = 0.0
    last_access_ts: float = 0.0

    def __post_init__(self) -> None:
        if self.total_blocks > 0:
            self.loc_hit_ratio = self.blocks_in_loc / self.total_blocks


class NullLOCCache:
    """Null object for deployments where LOC cache querying is not wired up.

    Returns a LOCCacheStatus indicating 0 blocks resident, disabling the
    LOC-priority bypass path in ``HeteroAdmissionGate``.
    """

    def query(self, request_id: int) -> LOCCacheStatus:
        return LOCCacheStatus(
            request_id=request_id,
            blocks_in_loc=0,
            total_blocks=1,
            loc_hit_ratio=0.0,
            last_access_ts=0.0,
        )


# ---------------------------------------------------------------------------
# Admission request descriptor
# ---------------------------------------------------------------------------

@dataclass
class AdmissionRequest:
    """Lightweight descriptor passed to ``HeteroAdmissionGate``.

    Carries the request identity, the current per-request wait state, and
    the candidate batch dimensions that would result from admission.

    Attributes:
        request_id:          Unique request identifier (for logging).
        cg_wait_iters:       Consecutive steps this request has been deferred.
                             Reset to 0 on successful admission.
        candidate:           The ``BatchDimensions`` the batch would have if this
                             request is admitted.
        target_device_role:  Which device archetype will execute this batch.
        is_continuing_chunk: True if this is the continuation of an in-flight
                             chunked prefill.  Gating is bypassed in this case
                             to avoid deadlocking progress.
        remaining_tokens:    Tokens still to prefill for this request.
    """
    request_id: int
    cg_wait_iters: int = 0
    candidate: Optional[BatchDimensions] = None
    target_device_role: DeviceRole = DeviceRole.UNKNOWN
    is_continuing_chunk: bool = False
    remaining_tokens: int = 0


@dataclass
class AdmissionDecision:
    """Result returned by ``HeteroAdmissionGate.evaluate``.

    Attributes:
        admit:              True → the request may be added to the active batch.
        chunk_size:         For chunked prefill, the CG-aligned token count to
                            admit this step.  -1 means "admit all remaining tokens".
        matched_graph:      The ``CapturedGraph`` that matched, or None on a miss.
        loc_bypass_used:    True if admission was granted via the LOC-priority bypass
                            rather than a clean CG match.
        deferred_reason:    Human-readable reason for deferral (empty on admit).
    """
    admit: bool
    chunk_size: int = -1
    matched_graph: Optional[CapturedGraph] = None
    loc_bypass_used: bool = False
    deferred_reason: str = ""


# ---------------------------------------------------------------------------
# Core admission gate
# ---------------------------------------------------------------------------

class HeteroAdmissionGate:
    """CUDAGraph-aware admission gate for DES-LOC heterogeneous inference.

    This class is the primary DES-LOC adaptation of Megatron's inline admission
    methods.  It replaces the three methods scattered through
    ``DynamicInferenceEngine`` with a single, testable, device-topology-aware
    object.

    Design contract (mirrors Megatron's invariants):
        1. On a CG miss, ``evaluate`` returns ``AdmissionDecision(admit=False)``.
           The caller **must** break its scheduling loop — there is no internal
           "schedule eagerly anyway" side channel.
        2. ``cg_wait_iters`` on the ``AdmissionRequest`` is the only mutable
           state written by the gate.  The engine's active_token_count,
           num_prefill_requests, and num_decode_requests are never modified here.
        3. The LOC-priority bypass is the sole exception to rule 1.  It is
           rate-limited, logged at WARNING level, and only activated when the
           request's LOC hit ratio exceeds ``loc_bypass_threshold``.

    Args:
        registry:               The ``HeteroCudagraphRegistry`` shared with the engine.
        is_hybrid_model:        If True, strict P/D matching is applied (SSM/Mamba models).
        warn_after_steps:       Emit a starvation warning every N consecutive deferrals.
                                Floor is 100 to avoid noise in short test configs.
        loc_cache:              LOC cache query interface.  Pass a ``NullLOCCache`` to
                                disable the bypass.
        loc_bypass_threshold:   Minimum LOC hit ratio to trigger the bypass.  1.0 means
                                all KV blocks must be in the LOC tier.
        loc_bypass_rate_limit:  Minimum seconds between consecutive bypass admissions
                                (per-gate, not per-request) to prevent LOC bypass from
                                dominating the scheduling queue.
        cuda_graph_all_prefills: Gate is only active when this flag is True (mirrors
                                 Megatron's ``--inference-cuda-graph-all-prefills``).
        use_graphs_for_non_decode: Whether the engine uses CGs for prefill/mixed steps.
    """

    def __init__(
        self,
        registry: HeteroCudagraphRegistry,
        is_hybrid_model: bool = False,
        warn_after_steps: int = 100,
        loc_cache: Optional[object] = None,
        loc_bypass_threshold: float = 1.0,
        loc_bypass_rate_limit: float = 0.1,
        cuda_graph_all_prefills: bool = False,
        use_graphs_for_non_decode: bool = True,
    ) -> None:
        self.registry = registry
        self.is_hybrid_model = is_hybrid_model
        self.warn_after_steps = max(100, warn_after_steps)
        self.loc_cache = loc_cache if loc_cache is not None else NullLOCCache()
        self.loc_bypass_threshold = loc_bypass_threshold
        self.loc_bypass_rate_limit = loc_bypass_rate_limit
        self.cuda_graph_all_prefills = cuda_graph_all_prefills
        self.use_graphs_for_non_decode = use_graphs_for_non_decode

        # Rate-limiting state for LOC bypass.
        self._last_loc_bypass_ts: float = 0.0
        # Per-device starvation counters for metrics (not per-request).
        self._device_starvation_counts: Dict[DeviceRole, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True when CG-aware gating should be applied.

        Gating is opt-in: all three conditions must hold:
            1. ``cuda_graph_all_prefills`` is enabled by the operator.
            2. The engine uses CGs for prefill/mixed steps.
            3. The registry has at least one graph for some device role.

        Mirrors Megatron's ``_cg_admission_gating_active``, extended with the
        device-registry liveness check instead of a single list-length test.
        """
        return (
            self.cuda_graph_all_prefills
            and self.use_graphs_for_non_decode
            and len(self.registry) > 0
        )

    def evaluate(
        self,
        req: AdmissionRequest,
        active_token_count: int,
        active_prefill_count: int,
        active_decode_count: int,
    ) -> AdmissionDecision:
        """Top-level admission evaluation for a non-chunked prefill request.

        Corresponds to the inline gating block added by Megatron ef549a6d inside
        the ``schedule_waiting_requests`` loop, adapted for device-aware lookup.

        The candidate batch shape is constructed here from ``req.candidate`` (which
        should already include the +1 prefill the engine would add), but the
        device_role is injected from the gate's target-device context.

        Args:
            req:                  The admission request descriptor.
            active_token_count:   Current total tokens in the active batch.
            active_prefill_count: Current prefill request count.
            active_decode_count:  Current decode request count.

        Returns:
            ``AdmissionDecision`` with ``admit=True`` or ``admit=False``.
        """
        if not self.is_active():
            return AdmissionDecision(admit=True)

        if req.is_continuing_chunk:
            # In-flight chunked prefill: bypassing is mandatory to avoid deadlock.
            return AdmissionDecision(admit=True)

        if req.candidate is None:
            # Caller did not supply a candidate; construct from active state.
            candidate = BatchDimensions(
                token_count=active_token_count + req.remaining_tokens,
                prefill_req_count=active_prefill_count + 1,
                decode_req_count=active_decode_count,
                device_role=req.target_device_role,
            )
        else:
            candidate = req.candidate

        matched = self._match_graph(candidate)
        if matched is not None:
            req.cg_wait_iters = 0
            return AdmissionDecision(admit=True, matched_graph=matched)

        # CG miss: check LOC bypass before deferring.
        bypass_decision = self._try_loc_bypass(req, candidate)
        if bypass_decision is not None:
            req.cg_wait_iters = 0
            return bypass_decision

        self._register_wait(req, candidate)
        return AdmissionDecision(
            admit=False,
            deferred_reason=(
                f"No captured CG for device_role={candidate.device_role.name} "
                f"tok={candidate.token_count} P={candidate.prefill_req_count} "
                f"D={candidate.decode_req_count} strict={self.is_hybrid_model}"
            ),
        )

    def find_chunked_chunk_size(
        self,
        req: AdmissionRequest,
        max_chunk_tokens: int,
        active_token_count: int,
        active_prefill_count: int,
        active_decode_count: int,
    ) -> Optional[int]:
        """Find the largest CG-aligned chunk size within the token budget for chunked prefill.

        Mirrors Megatron's ``_find_cg_chunk_size``, extended for device-role scoping.

        Walks the captured-CG list for ``req.target_device_role`` (sorted descending by
        token_count) and returns the first chunk size that:
            1. Falls within ``[1, max_chunk_tokens]``.
            2. When added to ``active_token_count``, lands on a captured token_count boundary.
            3. Produces an applicable batch_dim under the engine's matching mode (strict
               for hybrid models).

        Returns None if no CG covers any chunk in the budget.  The caller must then decide
        between eager fallback and deferral (chunked prefill always falls back to eager to
        avoid deadlocking mid-flight requests).

        Note: Continuing chunked prefills bypass this method entirely (handled in ``evaluate``).

        Args:
            req:                  The admission request (for device_role).
            max_chunk_tokens:     The token budget available this step.
            active_token_count:   Current total tokens in the active batch.
            active_prefill_count: Current prefill request count.
            active_decode_count:  Current decode request count.

        Returns:
            An integer chunk size, or None on miss.
        """
        graphs = self.registry.get_sorted(req.target_device_role)
        if not graphs:
            return None

        for cg in graphs:
            chunk = cg.token_count - active_token_count
            if chunk < 1:
                continue
            if chunk > max_chunk_tokens:
                continue
            candidate = BatchDimensions(
                token_count=cg.token_count,
                prefill_req_count=active_prefill_count + 1,
                decode_req_count=active_decode_count,
                device_role=req.target_device_role,
            )
            if cg.is_applicable_for_batch_dim(candidate, strict=self.is_hybrid_model):
                return chunk

        return None

    def apply_flash_attn_guard(
        self,
        prefill_chunk_length: int,
        remaining_len: int,
    ) -> Tuple[int, bool]:
        """Apply the Flash Attention single-token-final-chunk guard.

        Flash Attention raises when max_seqlen_q == 1 on the final chunk.
        See: https://github.com/Dao-AILab/flash-attention/issues/1537

        If admitting ``prefill_chunk_length`` tokens would leave exactly 1 token
        for the final chunk (``remaining_len - prefill_chunk_length == 1``):
            - If budget > 1: reduce chunk by 1 (safe after CG snapping because
              ``is_applicable_for_batch_dim`` matches on ``>=``, so the snapped
              CG still covers ``token_count - 1``).
            - If budget == 1: the request cannot be safely scheduled this step;
              return ``(0, False)`` to signal deferral.

        Returns:
            (adjusted_chunk_length, can_schedule):
                adjusted_chunk_length: The (possibly reduced) chunk length.
                can_schedule:          False if the request must be deferred.
        """
        if remaining_len - prefill_chunk_length == 1:
            if prefill_chunk_length > 1:
                return prefill_chunk_length - 1, True
            else:
                return 0, False
        return prefill_chunk_length, True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_graph(self, candidate: BatchDimensions) -> Optional[CapturedGraph]:
        """Find the first captured graph applicable for ``candidate``.

        Iterates the sorted list for the candidate's device role.  Returns the
        first match (largest token_count first), or None.

        Passes ``match_ep_token_counts=False`` equivalent: no NCCL sync is
        triggered here.  The step-time matcher performs its own EP synchronisation.
        """
        graphs = self.registry.get_sorted(candidate.device_role)
        for cg in graphs:
            if cg.is_applicable_for_batch_dim(candidate, strict=self.is_hybrid_model):
                return cg
        return None

    def _try_loc_bypass(
        self,
        req: AdmissionRequest,
        candidate: BatchDimensions,
    ) -> Optional[AdmissionDecision]:
        """Attempt a LOC-priority bypass admission for a CG-miss request.

        LOC bypass rationale:
            When a request's KV blocks are fully resident in the LOC (CPU DRAM) tier,
            prefetching them into GPU HBM is a sunk PCIe cost regardless of whether
            we use graph or eager mode.  In this situation the marginal cost of
            admitting eagerly is lower: the PCIe transfer dominates the step latency,
            not the CUDA graph overhead.  By admitting the request now, we avoid
            an extra scheduling round-trip that would keep the PCIe bus idle.

            Rate-limiting prevents this bypass from being exercised on every miss
            (which would defeat CG-aware scheduling entirely).

        Returns ``AdmissionDecision(admit=True, loc_bypass_used=True)`` if bypass
        criteria are met, otherwise None.
        """
        now = time.monotonic()
        if now - self._last_loc_bypass_ts < self.loc_bypass_rate_limit:
            return None

        loc_status = self.loc_cache.query(req.request_id)
        if loc_status.loc_hit_ratio < self.loc_bypass_threshold:
            return None

        self._last_loc_bypass_ts = now
        logger.warning(
            "LOC-priority bypass granted for request %d: "
            "loc_hit_ratio=%.2f device_role=%s tok=%d P=%d D=%d "
            "— admitting without matching CG (eager execution)",
            req.request_id,
            loc_status.loc_hit_ratio,
            candidate.device_role.name,
            candidate.token_count,
            candidate.prefill_req_count,
            candidate.decode_req_count,
        )
        return AdmissionDecision(admit=True, loc_bypass_used=True)

    def _register_wait(
        self,
        req: AdmissionRequest,
        candidate: BatchDimensions,
    ) -> None:
        """Increment the request's deferral counter and emit a starvation warning at threshold.

        Mirrors Megatron's ``_register_cg_wait``, with device-role context added so
        operators can identify which device archetype is the bottleneck.

        The warning threshold is ``warn_after_steps`` (floored at 100) to avoid noise in
        test environments with short max_sequence_length configs.
        """
        req.cg_wait_iters += 1
        self._device_starvation_counts[candidate.device_role] += 1

        if req.cg_wait_iters % self.warn_after_steps == 0:
            logger.warning(
                "Request %d deferred by CG-aware admission for %d consecutive steps — "
                "possible starvation (device_role=%s strict=%s active P=%d D=%d tok=%d). "
                "Consider expanding the captured-graph set for this device.",
                req.request_id,
                req.cg_wait_iters,
                candidate.device_role.name,
                self.is_hybrid_model,
                candidate.prefill_req_count - 1,   # active P before this request
                candidate.decode_req_count,
                candidate.token_count - req.remaining_tokens,
            )

    def starvation_stats(self) -> Dict[str, int]:
        """Return cumulative per-device starvation deferral counts.

        Useful for metrics export.  Does not reset counters.
        """
        return {role.name: count for role, count in self._device_starvation_counts.items()}


# ---------------------------------------------------------------------------
# Scheduler integration helper
# ---------------------------------------------------------------------------

class HeteroSchedulerAdmissionMixin:
    """Mixin for DeepSpeed inference scheduler classes to integrate HeteroAdmissionGate.

    Provides the three entry points that map 1-to-1 onto the scheduler's decision
    points, translating between DeepSpeed request objects and the gate's
    ``AdmissionRequest`` / ``AdmissionDecision`` vocabulary.

    Concrete schedulers should inherit from this mixin alongside the base scheduler
    class and call ``_init_hetero_admission`` from their ``__init__``.

    Example::

        class MyDeepSpeedScheduler(HeteroSchedulerAdmissionMixin, BaseScheduler):
            def __init__(self, config, registry, ...):
                super().__init__(config)
                self._init_hetero_admission(
                    registry=registry,
                    is_hybrid=config.is_hybrid_model,
                    cuda_graph_all_prefills=config.cuda_graph_all_prefills,
                )
    """

    def _init_hetero_admission(
        self,
        registry: HeteroCudagraphRegistry,
        is_hybrid: bool = False,
        warn_after_steps: int = 100,
        loc_cache: Optional[object] = None,
        loc_bypass_threshold: float = 1.0,
        loc_bypass_rate_limit: float = 0.1,
        cuda_graph_all_prefills: bool = False,
        use_graphs_for_non_decode: bool = True,
    ) -> None:
        """Initialise the admission gate.  Call from ``__init__`` of the concrete scheduler."""
        self._hetero_gate = HeteroAdmissionGate(
            registry=registry,
            is_hybrid_model=is_hybrid,
            warn_after_steps=warn_after_steps,
            loc_cache=loc_cache,
            loc_bypass_threshold=loc_bypass_threshold,
            loc_bypass_rate_limit=loc_bypass_rate_limit,
            cuda_graph_all_prefills=cuda_graph_all_prefills,
            use_graphs_for_non_decode=use_graphs_for_non_decode,
        )

    def _gate_evaluate_non_chunked(
        self,
        request_id: int,
        cg_wait_iters: int,
        remaining_tokens: int,
        target_device_role: DeviceRole,
        active_token_count: int,
        active_prefill_count: int,
        active_decode_count: int,
    ) -> AdmissionDecision:
        """Evaluate admission for a non-chunked prefill request."""
        req = AdmissionRequest(
            request_id=request_id,
            cg_wait_iters=cg_wait_iters,
            remaining_tokens=remaining_tokens,
            target_device_role=target_device_role,
            is_continuing_chunk=False,
        )
        decision = self._hetero_gate.evaluate(
            req,
            active_token_count=active_token_count,
            active_prefill_count=active_prefill_count,
            active_decode_count=active_decode_count,
        )
        return decision

    def _gate_find_chunked_chunk(
        self,
        request_id: int,
        max_chunk_tokens: int,
        target_device_role: DeviceRole,
        active_token_count: int,
        active_prefill_count: int,
        active_decode_count: int,
        is_continuing: bool = False,
    ) -> int:
        """Return the CG-aligned chunk size for chunked prefill, or max_chunk_tokens on miss.

        Mirrors Megatron's scheduler logic:
            - If gating is inactive or request is continuing: return max_chunk_tokens.
            - If a CG match exists: return the snapped chunk size.
            - If no CG match: return max_chunk_tokens (eager fallback — never defer mid-flight).
        """
        if not self._hetero_gate.is_active() or is_continuing:
            return max_chunk_tokens
        req = AdmissionRequest(
            request_id=request_id,
            target_device_role=target_device_role,
            is_continuing_chunk=is_continuing,
        )
        snapped = self._hetero_gate.find_chunked_chunk_size(
            req=req,
            max_chunk_tokens=max_chunk_tokens,
            active_token_count=active_token_count,
            active_prefill_count=active_prefill_count,
            active_decode_count=active_decode_count,
        )
        return snapped if snapped is not None else max_chunk_tokens


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_registry_for_deployment(
    a6000_token_counts: Sequence[int],
    h100_token_counts: Sequence[int],
    a6000_prefill_grid: Sequence[int],
    h100_prefill_grid: Sequence[int],
    is_hybrid: bool = False,
) -> HeteroCudagraphRegistry:
    """Build a ``HeteroCudagraphRegistry`` for the DES-LOC reference deployment.

    Generates a cross-product of (token_count, prefill_count) for each device role,
    filling decode slots as ``token_count - prefill_count`` (clamped to 0).

    Args:
        a6000_token_counts:   Token-count breakpoints for A6000 graph capture.
        h100_token_counts:    Token-count breakpoints for H100 graph capture.
        a6000_prefill_grid:   Prefill-count grid for A6000 (e.g. [0, 1, 2, 4, 8]).
        h100_prefill_grid:    Prefill-count grid for H100.
        is_hybrid:            Whether the model is SSM/Mamba (stored in each CapturedGraph).

    Returns:
        A fully populated ``HeteroCudagraphRegistry``.
    """
    registry = HeteroCudagraphRegistry()
    graphs: List[CapturedGraph] = []

    for tok in a6000_token_counts:
        for p in a6000_prefill_grid:
            d = max(0, tok - p)
            graphs.append(CapturedGraph(
                token_count=tok,
                prefill_req_count=p,
                decode_req_count=d,
                device_role=DeviceRole.A6000_SM86,
                is_hybrid=is_hybrid,
            ))

    for tok in h100_token_counts:
        for p in h100_prefill_grid:
            d = max(0, tok - p)
            graphs.append(CapturedGraph(
                token_count=tok,
                prefill_req_count=p,
                decode_req_count=d,
                device_role=DeviceRole.H100_SM90,
                is_hybrid=is_hybrid,
            ))

    registry.register_bulk(graphs)
    logger.info(
        "Built CG registry: %d A6000 graphs, %d H100 graphs (hybrid=%s)",
        len(a6000_token_counts) * len(a6000_prefill_grid),
        len(h100_token_counts) * len(h100_prefill_grid),
        is_hybrid,
    )
    return registry


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback
    import types as _types

    _PASS = "\033[92mPASS\033[0m"
    _FAIL = "\033[91mFAIL\033[0m"

    _results: List[Tuple[str, bool, str]] = []

    def _run(name: str, fn):
        try:
            fn()
            _results.append((name, True, ""))
            print(f"  {_PASS}  {name}")
        except Exception as exc:
            tb = traceback.format_exc()
            _results.append((name, False, tb))
            print(f"  {_FAIL}  {name}")
            print(f"        {exc}")

    def _assert(cond: bool, msg: str = "") -> None:
        if not cond:
            raise AssertionError(msg or "assertion failed")

    # ------------------------------------------------------------------
    # Helpers shared across tests
    # ------------------------------------------------------------------

    def _make_a6000_registry() -> HeteroCudagraphRegistry:
        """Small registry covering A6000 with token counts {4, 16, 64, 128, 256}."""
        return build_registry_for_deployment(
            a6000_token_counts=[256, 128, 64, 16, 4],
            h100_token_counts=[],
            a6000_prefill_grid=[0, 1, 2, 4, 8],
            h100_prefill_grid=[],
            is_hybrid=False,
        )

    def _make_h100_registry() -> HeteroCudagraphRegistry:
        return build_registry_for_deployment(
            a6000_token_counts=[],
            h100_token_counts=[512, 256, 128, 64, 16, 4],
            a6000_prefill_grid=[],
            h100_prefill_grid=[0, 1, 2, 4, 8],
            is_hybrid=False,
        )

    def _make_full_registry(is_hybrid: bool = False) -> HeteroCudagraphRegistry:
        return build_registry_for_deployment(
            a6000_token_counts=[256, 128, 64, 16, 4],
            h100_token_counts=[512, 256, 128, 64, 16, 4],
            a6000_prefill_grid=[0, 1, 2, 4, 8],
            h100_prefill_grid=[0, 1, 2, 4, 8],
            is_hybrid=is_hybrid,
        )

    def _make_gate(
        registry: HeteroCudagraphRegistry,
        is_hybrid: bool = False,
        warn_after: int = 100,
        cuda_graph_all_prefills: bool = True,
        use_graphs_for_non_decode: bool = True,
        loc_cache=None,
    ) -> HeteroAdmissionGate:
        return HeteroAdmissionGate(
            registry=registry,
            is_hybrid_model=is_hybrid,
            warn_after_steps=warn_after,
            loc_cache=loc_cache,
            cuda_graph_all_prefills=cuda_graph_all_prefills,
            use_graphs_for_non_decode=use_graphs_for_non_decode,
        )

    def _req(rid: int = 1, cg_wait: int = 0, remaining: int = 64,
             role: DeviceRole = DeviceRole.A6000_SM86) -> AdmissionRequest:
        return AdmissionRequest(
            request_id=rid,
            cg_wait_iters=cg_wait,
            remaining_tokens=remaining,
            target_device_role=role,
        )

    # ------------------------------------------------------------------
    # Test suite
    # ------------------------------------------------------------------

    print("\n=== DeviceRole classification ===")

    def test_device_role_unknown_for_cpu():
        cpu_dev = torch.device("cpu")
        role = DeviceRole.from_device(cpu_dev)
        _assert(role == DeviceRole.UNKNOWN, f"Expected UNKNOWN, got {role}")
    _run("device_role_unknown_for_cpu", test_device_role_unknown_for_cpu)

    print("\n=== CapturedGraph.is_applicable_for_batch_dim ===")

    def test_non_strict_match_total_slots():
        cg = CapturedGraph(128, 4, 124, DeviceRole.A6000_SM86)
        candidate = BatchDimensions(64, 2, 60, DeviceRole.A6000_SM86)
        _assert(cg.is_applicable_for_batch_dim(candidate, strict=False))
    _run("non_strict_match_total_slots", test_non_strict_match_total_slots)

    def test_non_strict_rejects_wrong_device():
        cg = CapturedGraph(128, 4, 124, DeviceRole.H100_SM90)
        candidate = BatchDimensions(64, 2, 60, DeviceRole.A6000_SM86)
        _assert(not cg.is_applicable_for_batch_dim(candidate, strict=False))
    _run("non_strict_rejects_wrong_device", test_non_strict_rejects_wrong_device)

    def test_non_strict_rejects_insufficient_token_count():
        cg = CapturedGraph(32, 4, 28, DeviceRole.A6000_SM86)
        candidate = BatchDimensions(64, 2, 60, DeviceRole.A6000_SM86)
        _assert(not cg.is_applicable_for_batch_dim(candidate, strict=False))
    _run("non_strict_rejects_insufficient_token_count", test_non_strict_rejects_insufficient_token_count)

    def test_strict_requires_per_type_slots():
        cg = CapturedGraph(128, 2, 126, DeviceRole.A6000_SM86)
        # Candidate wants P=4 but captured P=2: strict miss.
        candidate = BatchDimensions(128, 4, 60, DeviceRole.A6000_SM86)
        _assert(not cg.is_applicable_for_batch_dim(candidate, strict=True))
        # Reduce P to 2: strict hit.
        candidate2 = BatchDimensions(128, 2, 60, DeviceRole.A6000_SM86)
        _assert(cg.is_applicable_for_batch_dim(candidate2, strict=True))
    _run("strict_requires_per_type_slots", test_strict_requires_per_type_slots)

    def test_strict_requires_decode_slots():
        cg = CapturedGraph(128, 4, 100, DeviceRole.A6000_SM86)
        # Candidate needs D=124 but captured D=100: strict miss.
        candidate = BatchDimensions(128, 2, 124, DeviceRole.A6000_SM86)
        _assert(not cg.is_applicable_for_batch_dim(candidate, strict=True))
    _run("strict_requires_decode_slots", test_strict_requires_decode_slots)

    print("\n=== HeteroCudagraphRegistry ===")

    def test_registry_sorted_descending():
        reg = _make_a6000_registry()
        graphs = reg.get_sorted(DeviceRole.A6000_SM86)
        tok_counts = [g.token_count for g in graphs]
        _assert(tok_counts == sorted(tok_counts, reverse=True),
                "Registry not sorted descending by token_count")
    _run("registry_sorted_descending", test_registry_sorted_descending)

    def test_registry_device_isolation():
        reg = _make_full_registry()
        a6000_graphs = reg.get_sorted(DeviceRole.A6000_SM86)
        h100_graphs = reg.get_sorted(DeviceRole.H100_SM90)
        _assert(all(g.device_role == DeviceRole.A6000_SM86 for g in a6000_graphs))
        _assert(all(g.device_role == DeviceRole.H100_SM90 for g in h100_graphs))
    _run("registry_device_isolation", test_registry_device_isolation)

    def test_registry_empty_role_returns_empty_list():
        reg = _make_a6000_registry()
        _assert(reg.get_sorted(DeviceRole.H100_SM90) == [])
    _run("registry_empty_role_returns_empty_list", test_registry_empty_role_returns_empty_list)

    def test_registry_len_counts_all_roles():
        reg = _make_full_registry()
        # A6000: 5 tok × 5 p = 25; H100: 6 tok × 5 p = 30 → total 55
        _assert(len(reg) == 55, f"Expected 55, got {len(reg)}")
    _run("registry_len_counts_all_roles", test_registry_len_counts_all_roles)

    print("\n=== HeteroAdmissionGate.is_active ===")

    def test_gate_inactive_when_all_prefills_off():
        reg = _make_a6000_registry()
        gate = _make_gate(reg, cuda_graph_all_prefills=False)
        _assert(not gate.is_active())
    _run("gate_inactive_when_all_prefills_off", test_gate_inactive_when_all_prefills_off)

    def test_gate_inactive_when_no_graphs_for_non_decode():
        reg = _make_a6000_registry()
        gate = _make_gate(reg, use_graphs_for_non_decode=False)
        _assert(not gate.is_active())
    _run("gate_inactive_when_no_graphs_for_non_decode", test_gate_inactive_when_no_graphs_for_non_decode)

    def test_gate_inactive_when_registry_empty():
        reg = HeteroCudagraphRegistry()
        gate = _make_gate(reg)
        _assert(not gate.is_active())
    _run("gate_inactive_when_registry_empty", test_gate_inactive_when_registry_empty)

    def test_gate_active_when_all_conditions_hold():
        reg = _make_a6000_registry()
        gate = _make_gate(reg)
        _assert(gate.is_active())
    _run("gate_active_when_all_conditions_hold", test_gate_active_when_all_conditions_hold)

    print("\n=== HeteroAdmissionGate.evaluate (non-chunked) ===")

    def test_evaluate_admits_matching_candidate():
        reg = _make_a6000_registry()
        gate = _make_gate(reg)
        req = _req(remaining=64)
        decision = gate.evaluate(req, active_token_count=0, active_prefill_count=0, active_decode_count=0)
        _assert(decision.admit, "Expected admit=True for matching candidate")
        _assert(req.cg_wait_iters == 0)
    _run("evaluate_admits_matching_candidate", test_evaluate_admits_matching_candidate)

    def test_evaluate_defers_when_no_matching_cg():
        reg = HeteroCudagraphRegistry()  # empty
        gate = _make_gate(reg)
        req = _req(remaining=64)
        decision = gate.evaluate(req, active_token_count=0, active_prefill_count=0, active_decode_count=0)
        _assert(not decision.admit, "Expected admit=False for empty registry")
        _assert(req.cg_wait_iters == 1)
    _run("evaluate_defers_when_no_matching_cg", test_evaluate_defers_when_no_matching_cg)

    def test_evaluate_continuing_chunk_always_admits():
        reg = HeteroCudagraphRegistry()  # empty — would block a fresh request
        gate = _make_gate(reg)
        req = _req(remaining=32)
        req.is_continuing_chunk = True
        decision = gate.evaluate(req, active_token_count=50, active_prefill_count=1, active_decode_count=0)
        _assert(decision.admit, "Continuing chunk must always be admitted")
    _run("evaluate_continuing_chunk_always_admits", test_evaluate_continuing_chunk_always_admits)

    def test_evaluate_bypasses_when_gate_inactive():
        reg = _make_a6000_registry()
        gate = _make_gate(reg, cuda_graph_all_prefills=False)
        req = _req(remaining=999)  # no matching CG for 999 tokens either way
        decision = gate.evaluate(req, active_token_count=0, active_prefill_count=0, active_decode_count=0)
        _assert(decision.admit, "Inactive gate must pass all requests")
    _run("evaluate_bypasses_when_gate_inactive", test_evaluate_bypasses_when_gate_inactive)

    def test_evaluate_resets_wait_counter_on_admit():
        reg = _make_a6000_registry()
        gate = _make_gate(reg)
        req = _req(cg_wait=17, remaining=16)
        decision = gate.evaluate(req, active_token_count=0, active_prefill_count=0, active_decode_count=0)
        _assert(decision.admit)
        _assert(req.cg_wait_iters == 0, f"Expected cg_wait_iters=0, got {req.cg_wait_iters}")
    _run("evaluate_resets_wait_counter_on_admit", test_evaluate_resets_wait_counter_on_admit)

    def test_evaluate_accumulates_wait_counter_on_miss():
        reg = HeteroCudagraphRegistry()
        gate = _make_gate(reg)
        req = _req(remaining=64)
        for expected in range(1, 6):
            gate.evaluate(req, active_token_count=0, active_prefill_count=0, active_decode_count=0)
            _assert(req.cg_wait_iters == expected,
                    f"Expected cg_wait_iters={expected}, got {req.cg_wait_iters}")
    _run("evaluate_accumulates_wait_counter_on_miss", test_evaluate_accumulates_wait_counter_on_miss)

    def test_evaluate_starvation_warning_at_threshold(capsule=None):
        """Starvation warning fires at warn_after_steps and at each multiple thereafter."""
        import io
        reg = HeteroCudagraphRegistry()
        gate = _make_gate(reg, warn_after=3)
        req = _req(remaining=64)

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.WARNING)

        try:
            for _ in range(6):
                gate.evaluate(req, 0, 0, 0)
            warnings = [l for l in log_stream.getvalue().splitlines()
                        if "deferred by CG-aware admission" in l]
            _assert(len(warnings) == 2,
                    f"Expected 2 starvation warnings (at step 3 and 6), got {len(warnings)}")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
    _run("evaluate_starvation_warning_at_threshold", test_evaluate_starvation_warning_at_threshold)

    print("\n=== HeteroAdmissionGate.find_chunked_chunk_size ===")

    def test_find_chunked_chunk_picks_largest_in_budget():
        reg = _make_a6000_registry()
        gate = _make_gate(reg)
        req = _req(remaining=300)
        chunk = gate.find_chunked_chunk_size(req, max_chunk_tokens=300,
                                              active_token_count=0, active_prefill_count=0,
                                              active_decode_count=0)
        _assert(chunk == 256, f"Expected 256, got {chunk}")
    _run("find_chunked_chunk_picks_largest_in_budget", test_find_chunked_chunk_picks_largest_in_budget)

    def test_find_chunked_chunk_respects_budget_ceiling():
        reg = _make_a6000_registry()
        gate = _make_gate(reg)
        req = _req(remaining=200)
        chunk = gate.find_chunked_chunk_size(req, max_chunk_tokens=100,
                                              active_token_count=0, active_prefill_count=0,
                                              active_decode_count=0)
        _assert(chunk == 64, f"Expected 64 (within 100 budget), got {chunk}")
    _run("find_chunked_chunk_respects_budget_ceiling", test_find_chunked_chunk_respects_budget_ceiling)

    def test_find_chunked_chunk_accounts_for_active_tokens():
        reg = _make_a6000_registry()
        gate = _make_gate(reg)
        req = _req(remaining=256)
        # active=50 → need cg.token_count-50 in [1,300]; first match: 256-50=206
        chunk = gate.find_chunked_chunk_size(req, max_chunk_tokens=300,
                                              active_token_count=50, active_prefill_count=0,
                                              active_decode_count=0)
        _assert(chunk == 206, f"Expected 206, got {chunk}")
    _run("find_chunked_chunk_accounts_for_active_tokens", test_find_chunked_chunk_accounts_for_active_tokens)

    def test_find_chunked_chunk_returns_none_when_no_match():
        reg = _make_a6000_registry()
        gate = _make_gate(reg)
        req = _req(remaining=5)
        # active=300, max_chunk=10: need cg in (300,310]; none exist
        chunk = gate.find_chunked_chunk_size(req, max_chunk_tokens=10,
                                              active_token_count=300, active_prefill_count=0,
                                              active_decode_count=0)
        _assert(chunk is None, f"Expected None, got {chunk}")
    _run("find_chunked_chunk_returns_none_when_no_match", test_find_chunked_chunk_returns_none_when_no_match)

    def test_find_chunked_chunk_strict_filters_by_decode():
        reg = _make_full_registry(is_hybrid=True)
        gate = _make_gate(reg, is_hybrid=True)
        # A6000, active D=200; need captured D>=200 AND valid P.  A6000 max D=256-0=256.
        req = _req(remaining=300, role=DeviceRole.A6000_SM86)
        chunk = gate.find_chunked_chunk_size(req, max_chunk_tokens=300,
                                              active_token_count=0, active_prefill_count=0,
                                              active_decode_count=200)
        # Strict: P_candidate=1, D_candidate=200. Need captured P>=1 AND D>=200.
        # A6000 graphs: (256,1,255)→D=255>=200, P=1>=1 ✓ → chunk=256.
        _assert(chunk == 256, f"Expected 256, got {chunk}")
    _run("find_chunked_chunk_strict_filters_by_decode", test_find_chunked_chunk_strict_filters_by_decode)

    def test_find_chunked_chunk_strict_no_match_returns_none():
        # A6000 max D=255 (token=256, P=1). If active_D=300, strict requires D>=300 → no A6000 graph.
        reg = _make_a6000_registry()
        gate = _make_gate(reg, is_hybrid=True)
        req = _req(remaining=256, role=DeviceRole.A6000_SM86)
        chunk = gate.find_chunked_chunk_size(req, max_chunk_tokens=300,
                                              active_token_count=0, active_prefill_count=0,
                                              active_decode_count=300)
        _assert(chunk is None, f"Expected None for out-of-range D, got {chunk}")
    _run("find_chunked_chunk_strict_no_match_returns_none", test_find_chunked_chunk_strict_no_match_returns_none)

    print("\n=== apply_flash_attn_guard ===")

    def test_flash_guard_reduces_chunk_by_one():
        gate = _make_gate(_make_a6000_registry())
        # remaining=65, chunk=64 → leftover=1 → reduce chunk to 63
        adjusted, can_schedule = gate.apply_flash_attn_guard(64, 65)
        _assert(adjusted == 63, f"Expected 63, got {adjusted}")
        _assert(can_schedule)
    _run("flash_guard_reduces_chunk_by_one", test_flash_guard_reduces_chunk_by_one)

    def test_flash_guard_defers_when_only_one_token_budget():
        gate = _make_gate(_make_a6000_registry())
        # remaining=2, chunk=1 → leftover=1 and chunk==1 → cannot schedule
        adjusted, can_schedule = gate.apply_flash_attn_guard(1, 2)
        _assert(adjusted == 0)
        _assert(not can_schedule)
    _run("flash_guard_defers_when_only_one_token_budget", test_flash_guard_defers_when_only_one_token_budget)

    def test_flash_guard_passthrough_when_no_edge_case():
        gate = _make_gate(_make_a6000_registry())
        adjusted, can_schedule = gate.apply_flash_attn_guard(64, 128)
        _assert(adjusted == 64)
        _assert(can_schedule)
    _run("flash_guard_passthrough_when_no_edge_case", test_flash_guard_passthrough_when_no_edge_case)

    print("\n=== LOC-priority bypass ===")

    def test_loc_bypass_granted_when_fully_resident():
        class FullLOCCache:
            def query(self, rid):
                return LOCCacheStatus(rid, 16, 16)

        reg = HeteroCudagraphRegistry()  # empty → would normally defer
        gate = _make_gate(reg, loc_cache=FullLOCCache(), loc_bypass_threshold=1.0)
        gate._last_loc_bypass_ts = 0.0  # ensure rate limit is clear
        req = _req(remaining=64)
        decision = gate.evaluate(req, 0, 0, 0)
        _assert(decision.admit, "Expected LOC bypass to admit request")
        _assert(decision.loc_bypass_used, "Expected loc_bypass_used=True")
    _run("loc_bypass_granted_when_fully_resident", test_loc_bypass_granted_when_fully_resident)

    def test_loc_bypass_denied_when_partially_resident():
        class PartialLOCCache:
            def query(self, rid):
                return LOCCacheStatus(rid, 8, 16)  # 50% hit ratio

        reg = HeteroCudagraphRegistry()
        gate = _make_gate(reg, loc_cache=PartialLOCCache(), loc_bypass_threshold=1.0)
        gate._last_loc_bypass_ts = 0.0
        req = _req(remaining=64)
        decision = gate.evaluate(req, 0, 0, 0)
        _assert(not decision.admit, "Partial LOC residency must not trigger bypass")
    _run("loc_bypass_denied_when_partially_resident", test_loc_bypass_denied_when_partially_resident)

    def test_loc_bypass_rate_limited():
        class FullLOCCache:
            def query(self, rid):
                return LOCCacheStatus(rid, 16, 16)

        reg = HeteroCudagraphRegistry()
        # Very long rate limit: second bypass must be denied.
        gate = _make_gate(reg, loc_cache=FullLOCCache(), loc_bypass_threshold=1.0,
                          loc_bypass_rate_limit=9999.0)
        gate._last_loc_bypass_ts = 0.0

        req1 = _req(request_id=1, remaining=64)
        req2 = _req(request_id=2, remaining=64)

        d1 = gate.evaluate(req1, 0, 0, 0)
        d2 = gate.evaluate(req2, 0, 0, 0)
        _assert(d1.admit and d1.loc_bypass_used, "First bypass should be granted")
        _assert(not d2.admit, "Second bypass within rate limit window must be denied")
    _run("loc_bypass_rate_limited", test_loc_bypass_rate_limited)

    print("\n=== HeteroSchedulerAdmissionMixin ===")

    def test_mixin_gate_find_chunked_chunk_skips_continuing():
        class StubScheduler(HeteroSchedulerAdmissionMixin):
            pass
        sched = StubScheduler()
        sched._init_hetero_admission(registry=_make_a6000_registry())
        # is_continuing=True → always returns max_chunk regardless of gating.
        chunk = sched._gate_find_chunked_chunk(
            request_id=1, max_chunk_tokens=50,
            target_device_role=DeviceRole.A6000_SM86,
            active_token_count=0, active_prefill_count=0, active_decode_count=0,
            is_continuing=True,
        )
        _assert(chunk == 50, f"Expected 50 (max_chunk), got {chunk}")
    _run("mixin_gate_find_chunked_chunk_skips_continuing", test_mixin_gate_find_chunked_chunk_skips_continuing)

    def test_mixin_gate_find_chunked_chunk_snaps_to_cg():
        class StubScheduler(HeteroSchedulerAdmissionMixin):
            pass
        sched = StubScheduler()
        sched._init_hetero_admission(
            registry=_make_a6000_registry(),
            cuda_graph_all_prefills=True,
            use_graphs_for_non_decode=True,
        )
        chunk = sched._gate_find_chunked_chunk(
            request_id=1, max_chunk_tokens=300,
            target_device_role=DeviceRole.A6000_SM86,
            active_token_count=0, active_prefill_count=0, active_decode_count=0,
        )
        _assert(chunk == 256, f"Expected 256 (largest CG within 300 budget), got {chunk}")
    _run("mixin_gate_find_chunked_chunk_snaps_to_cg", test_mixin_gate_find_chunked_chunk_snaps_to_cg)

    def test_mixin_evaluate_non_chunked_defers_on_empty_registry():
        class StubScheduler(HeteroSchedulerAdmissionMixin):
            pass
        sched = StubScheduler()
        sched._init_hetero_admission(registry=HeteroCudagraphRegistry())
        decision = sched._gate_evaluate_non_chunked(
            request_id=5, cg_wait_iters=0, remaining_tokens=64,
            target_device_role=DeviceRole.A6000_SM86,
            active_token_count=0, active_prefill_count=0, active_decode_count=0,
        )
        _assert(not decision.admit, "Empty registry must defer")
    _run("mixin_evaluate_non_chunked_defers_on_empty_registry", test_mixin_evaluate_non_chunked_defers_on_empty_registry)

    print("\n=== build_registry_for_deployment ===")

    def test_build_registry_correct_device_assignment():
        reg = build_registry_for_deployment(
            a6000_token_counts=[64, 128],
            h100_token_counts=[256, 512],
            a6000_prefill_grid=[0, 1],
            h100_prefill_grid=[0, 1, 2],
        )
        a6000_graphs = reg.get_sorted(DeviceRole.A6000_SM86)
        h100_graphs = reg.get_sorted(DeviceRole.H100_SM90)
        # A6000: 2 tok × 2 p = 4; H100: 2 tok × 3 p = 6
        _assert(len(a6000_graphs) == 4, f"A6000: expected 4, got {len(a6000_graphs)}")
        _assert(len(h100_graphs) == 6, f"H100: expected 6, got {len(h100_graphs)}")
        _assert(all(g.device_role == DeviceRole.A6000_SM86 for g in a6000_graphs))
        _assert(all(g.device_role == DeviceRole.H100_SM90 for g in h100_graphs))
    _run("build_registry_correct_device_assignment", test_build_registry_correct_device_assignment)

    def test_build_registry_decode_slots_clamped_to_zero():
        # When P > token_count, D should be clamped to 0 not go negative.
        reg = build_registry_for_deployment(
            a6000_token_counts=[4],
            h100_token_counts=[],
            a6000_prefill_grid=[8],  # P=8 > tok=4
            h100_prefill_grid=[],
        )
        graphs = reg.get_sorted(DeviceRole.A6000_SM86)
        _assert(len(graphs) == 1)
        _assert(graphs[0].decode_req_count == 0,
                f"D should be clamped to 0, got {graphs[0].decode_req_count}")
    _run("build_registry_decode_slots_clamped_to_zero", test_build_registry_decode_slots_clamped_to_zero)

    print("\n=== starvation_stats ===")

    def test_starvation_stats_tracks_per_device():
        reg = HeteroCudagraphRegistry()
        gate = _make_gate(reg)
        req_a = _req(request_id=1, role=DeviceRole.A6000_SM86)
        req_h = _req(request_id=2, role=DeviceRole.H100_SM90)
        # 3 misses on A6000, 2 misses on H100
        for _ in range(3):
            gate.evaluate(req_a, 0, 0, 0)
        for _ in range(2):
            gate.evaluate(req_h, 0, 0, 0)
        stats = gate.starvation_stats()
        _assert(stats.get("A6000_SM86", 0) == 3, f"A6000 count: {stats}")
        _assert(stats.get("H100_SM90", 0) == 2, f"H100 count: {stats}")
    _run("starvation_stats_tracks_per_device", test_starvation_stats_tracks_per_device)

    print("\n=== cross-device isolation (DES-LOC invariant) ===")

    def test_h100_candidate_not_matched_by_a6000_graphs():
        """An H100-bound batch must never be matched against A6000 graphs."""
        reg = _make_a6000_registry()  # only A6000 graphs
        gate = _make_gate(reg)
        req = _req(remaining=64, role=DeviceRole.H100_SM90)
        decision = gate.evaluate(req, 0, 0, 0)
        _assert(not decision.admit,
                "H100-bound batch must not match A6000 CGs — device isolation violated")
    _run("h100_candidate_not_matched_by_a6000_graphs", test_h100_candidate_not_matched_by_a6000_graphs)

    def test_a6000_candidate_not_matched_by_h100_graphs():
        reg = _make_h100_registry()  # only H100 graphs
        gate = _make_gate(reg)
        req = _req(remaining=64, role=DeviceRole.A6000_SM86)
        decision = gate.evaluate(req, 0, 0, 0)
        _assert(not decision.admit,
                "A6000-bound batch must not match H100 CGs — device isolation violated")
    _run("a6000_candidate_not_matched_by_h100_graphs", test_a6000_candidate_not_matched_by_h100_graphs)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print(f"Results: {passed} passed, {failed} failed out of {len(_results)} tests.")
    if failed:
        print("\nFailed tests:")
        for name, ok, tb in _results:
            if not ok:
                print(f"  {name}")
                print(tb)
        sys.exit(1)
    else:
        print("All tests passed.")
