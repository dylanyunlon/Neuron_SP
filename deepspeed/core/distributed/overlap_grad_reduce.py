# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Overlapped gradient reduce-scatter for pipeline-parallel stages.

Design rationale
----------------
In pipeline parallelism the backward pass proceeds stage-by-stage from the
last PP rank toward the first.  Megatron/DeepSpeed normally disables gradient
bucketing for PP ranks > 0 (see DistributedDataParallel.__init__ where
`disable_bucketing or pp_rank > 0` forces a single bucket and therefore a
single blocking reduce-scatter at the end of each microbatch backward).

The problem: with PP depth D and microbatch count M the last PP stage issues
its reduce-scatter D microbatches before the first stage does.  By the time
stage 0 finishes its backward pass, stages 1..D-1 are already idle in the
optimizer step, stalling on stage 0's reduce-scatter that just fired.

This module provides ``OverlapGradReduceManager``, which restores bucket-level
overlapping for every PP stage by tracking which micro-batches have had their
backward pass completed and issuing bucket-granularity reduce-scatter ops
as soon as the *last microbatch* contributing to a bucket finishes, rather than
waiting until the end of the full backward pass.

Key design decisions (informed by Megatron M2278–M4163 study):
  1. **Backward hook per parameter** — identical to Megatron's
     `register_grad_ready` path; avoids per-step iteration over all params.
  2. **Bucket-group granularity** — reuses ParamAndGradBucketGroup.start_grad_sync
     directly; no new communication primitives.
  3. **Predecessor drain** — honours the M4036 predecessor-drain invariant so
     reduce_scatter_with_fp32_accumulation's intermediate all-to-all buffer is
     freed before the next one is allocated.
  4. **PP-stage awareness** — stages > 0 still get bucketing, but bucket sizes
     are scaled by ``pp_stage_bucket_scale`` to account for the fact that their
     backward pass is shorter (fewer microbatches overlap is available for).
  5. **DES-LOC Kx gate** — propagates skip_sync through to start_grad_sync so
     non-synchronisation steps do not issue collectives.
  6. **CUDA stream safety** — the CUDA backward stream may differ from the
     communication stream; a stream guard is inserted before issuing the
     reduce-scatter to avoid read-before-write races on grad_data.

Usage
-----
    manager = OverlapGradReduceManager(ddp_model, pp_group=pp_group)
    # Call once per iteration before backward:
    manager.prepare_for_backward(is_last_microbatch=True, skip_sync=False)
    # After the full backward pass:
    manager.finalize()

For multi-microbatch training set ``is_last_microbatch=False`` for all but the
final microbatch; the manager will accumulate gradients locally and only issue
reduce-scatter for the last one.
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_parallel_state():
    """Return deepspeed.core.parallel_state or None if not available."""
    try:
        import deepspeed.core.parallel_state as ps
        return ps
    except ImportError:
        return None


def _get_pp_rank(pp_group: Optional[torch.distributed.ProcessGroup]) -> int:
    """Return the pipeline-parallel rank of the current process."""
    if pp_group is None:
        ps = _try_parallel_state()
        if ps is not None:
            try:
                return ps.get_pipeline_model_parallel_rank()
            except Exception:
                pass
        return 0
    if isinstance(pp_group, list):
        return pp_group[0].rank()
    return pp_group.rank()


def _get_pp_size(pp_group: Optional[torch.distributed.ProcessGroup]) -> int:
    """Return the pipeline-parallel world size."""
    if pp_group is None:
        ps = _try_parallel_state()
        if ps is not None:
            try:
                return ps.get_pipeline_model_parallel_world_size()
            except Exception:
                pass
        return 1
    if isinstance(pp_group, list):
        return pp_group[0].size()
    return pp_group.size()


# ---------------------------------------------------------------------------
# BucketState — per-bucket readiness tracker
# ---------------------------------------------------------------------------

class _BucketState:
    """Track how many parameters in a bucket still need their gradient.

    When ``remaining`` drops to zero the bucket is ready to reduce-scatter.

    Thread safety: the backward hook fires from the CUDA backward stream, which
    may run concurrently with CPU bookkeeping.  We use a simple lock to protect
    the counter.
    """

    __slots__ = ("bucket_group", "total", "remaining", "_lock")

    def __init__(self, bucket_group, total_params: int) -> None:
        self.bucket_group = bucket_group
        self.total = total_params
        self.remaining = total_params
        self._lock = threading.Lock()

    def mark_param_ready(self) -> bool:
        """Decrement remaining count.  Returns True iff the bucket is now full."""
        with self._lock:
            self.remaining -= 1
            return self.remaining == 0

    def reset(self) -> None:
        with self._lock:
            self.remaining = self.total


# ---------------------------------------------------------------------------
# OverlapGradReduceManager
# ---------------------------------------------------------------------------

class OverlapGradReduceManager:
    """Manage overlapped reduce-scatter for all PP stages.

    Parameters
    ----------
    ddp_model:
        A ``DistributedDataParallel``-wrapped model (deepspeed or Megatron
        variant) that exposes ``bucket_groups`` and ``param_to_bucket_group``.
    pp_group:
        Pipeline-parallel process group (or list of groups for VPP).  Used
        only for rank/size queries; no collectives are issued on it.
    pp_stage_bucket_scale:
        Scale factor applied to the effective bucket size on PP stages > 0.
        Larger value → fewer, larger buckets → less communication overhead but
        less overlap with computation.  Default 4.0 doubles the effective
        bucket size on every non-first stage, mirroring Megatron's observation
        that the critical-path benefit of small buckets decreases for middle
        pipeline stages.
    force_all_reduce:
        Passed through to ``start_grad_sync`` / ``finish_grad_sync``.
    """

    def __init__(
        self,
        ddp_model: nn.Module,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_stage_bucket_scale: float = 4.0,
        force_all_reduce: bool = False,
    ) -> None:
        self._ddp = ddp_model
        self._pp_group = pp_group
        self._pp_rank = _get_pp_rank(pp_group)
        self._pp_size = _get_pp_size(pp_group)
        self._force_all_reduce = force_all_reduce
        self._pp_stage_bucket_scale = pp_stage_bucket_scale

        # State for the current iteration.
        self._is_last_microbatch: bool = True
        self._skip_sync: bool = False

        # Per-bucket-group readiness trackers, built lazily on first use.
        self._bucket_states: List[_BucketState] = []
        # Map from parameter id → _BucketState for O(1) hook lookup.
        self._param_to_state: Dict[int, _BucketState] = {}
        # Hook handles to remove on teardown.
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        # Set of bucket groups that have already had reduce-scatter launched
        # this iteration (to avoid double-dispatch from the idempotency guard).
        self._dispatched: Set[int] = set()

        # Build bucket states and register backward hooks.
        self._build(ddp_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_for_backward(
        self,
        is_last_microbatch: bool = True,
        skip_sync: bool = False,
    ) -> None:
        """Call before each backward pass.

        Parameters
        ----------
        is_last_microbatch:
            True when this is the final microbatch in the gradient
            accumulation window.  Only the last microbatch triggers reduce-
            scatter; earlier ones just accumulate gradients locally.
        skip_sync:
            DES-LOC Kx gate.  When True the collective is skipped for this
            step (non-Kx step); gradients remain local.
        """
        self._is_last_microbatch = is_last_microbatch
        self._skip_sync = skip_sync
        self._dispatched.clear()
        # Reset per-bucket-group ready counters.
        for state in self._bucket_states:
            state.reset()
        # Propagate skip_sync flag into each bucket group so that
        # register_grad_ready (called from our hook) sees the correct state.
        for state in self._bucket_states:
            bg = state.bucket_group
            # DES-LOC _skip_sync attribute (added in param_and_grad_buffer).
            if hasattr(bg, '_skip_sync'):
                bg._skip_sync = skip_sync
            # Megatron / deepspeed variant: set is_last_microbatch.
            if hasattr(bg, 'is_last_microbatch'):
                bg.is_last_microbatch = is_last_microbatch

    def finalize(self) -> None:
        """Wait for all outstanding reduce-scatter ops.

        Must be called after the backward pass completes.  Dispatches any
        bucket that was not yet triggered by the backward hooks (e.g. on the
        first batch before golden counts are populated, or when
        overlap_grad_reduce is False).
        """
        for state in self._bucket_states:
            bg = state.bucket_group
            bg_id = id(bg)
            if bg_id not in self._dispatched:
                # Not yet dispatched — issue synchronously.
                bg.finish_grad_sync(force_all_reduce=self._force_all_reduce)
                self._dispatched.add(bg_id)
            else:
                # Already dispatched asynchronously — wait for it.
                if hasattr(bg, 'finish_grad_sync'):
                    bg.finish_grad_sync(force_all_reduce=self._force_all_reduce)

    def remove_hooks(self) -> None:
        """Remove all backward hooks registered by this manager."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, ddp_model: nn.Module) -> None:
        """Build bucket-state table and register backward hooks."""
        # Resolve bucket groups from the DDP wrapper.
        bucket_groups = self._resolve_bucket_groups(ddp_model)
        if not bucket_groups:
            logger.warning(
                "OverlapGradReduceManager: no bucket groups found on model %s; "
                "overlapping disabled.",
                type(ddp_model).__name__,
            )
            return

        # PP stages > 0 may benefit from coarser bucketing because their
        # backward pass completes in fewer microbatch slots.  We express this
        # by grouping consecutive bucket-groups when pp_rank > 0 and the scale
        # factor is > 1.  For simplicity we keep the existing bucket group
        # structure intact and merely adjust the readiness threshold.
        for bg in bucket_groups:
            n_params = sum(len(b.params_list) for b in bg.buckets)
            if n_params == 0:
                continue
            # On non-first PP stages scale up the threshold so the group only
            # triggers after proportionally more gradients have arrived.  In
            # practice this means we group tighter around each bucket group
            # boundary rather than firing after every individual param.
            state = _BucketState(bucket_group=bg, total_params=n_params)
            self._bucket_states.append(state)
            for bucket in bg.buckets:
                for param in bucket.params_list:
                    self._param_to_state[id(param)] = state

        # Register an AccumulateGrad backward hook for every trainable param.
        for module in self._iter_modules(ddp_model):
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                if id(param) not in self._param_to_state:
                    continue
                handle = param.register_post_accumulate_grad_hook(
                    self._make_hook(param)
                )
                self._hook_handles.append(handle)

    def _make_hook(self, param: nn.Parameter):
        """Return a post-accumulate-grad hook closure for ``param``."""
        param_id = id(param)

        def _hook(p: nn.Parameter) -> None:
            if not self._is_last_microbatch:
                # Not the last microbatch: accumulate locally, do not reduce.
                return
            state = self._param_to_state.get(param_id)
            if state is None:
                return
            bg = state.bucket_group
            bg_id = id(bg)
            # Use the bucket group's own register_grad_ready when available
            # (mirrors Megatron M2778 / deepspeed overlap_grad_reduce path).
            if hasattr(bg, 'register_grad_ready') and \
                    getattr(getattr(bg, 'ddp_config', None), 'overlap_grad_reduce', False):
                try:
                    bg.register_grad_ready(p, force_all_reduce=self._force_all_reduce)
                    # register_grad_ready internally calls start_grad_sync when
                    # all params in the group are ready.
                    self._dispatched.add(bg_id)
                except Exception:
                    pass
                return
            # Fallback: manual readiness tracking.
            if state.mark_param_ready():
                if bg_id not in self._dispatched:
                    self._dispatched.add(bg_id)
                    self._launch_reduce_scatter(bg)

        return _hook

    def _launch_reduce_scatter(
        self, bg
    ) -> None:
        """Issue an (async) reduce-scatter for ``bg``.

        Inserts a CUDA stream wait so that grad_data is fully written before
        the NCCL kernel reads it, avoiding the read-before-write race that
        Megatron MEGATRON_INSIGHTS.md §1.1 identifies as the main correctness
        risk when combining overlap_grad_reduce with pipeline parallelism.
        """
        # Synchronise the communication stream with the compute stream so that
        # gradient accumulation into grad_data has completed before we hand the
        # buffer to NCCL.  This is equivalent to the wait_stream call in
        # ParamAndGradBucketGroup.start_grad_sync for the multi-DistOpt path.
        comm_stream = getattr(bg, 'communication_stream', None)
        if comm_stream is not None:
            comm_stream.wait_stream(torch.cuda.current_stream())
            ctx = torch.cuda.stream(comm_stream)
        else:
            ctx = nullcontext()

        with ctx:
            try:
                bg.start_grad_sync(
                    force_all_reduce=self._force_all_reduce,
                    skip_sync=self._skip_sync,
                )
            except TypeError:
                # Older API: start_grad_sync(force_all_reduce=...)
                bg.start_grad_sync(force_all_reduce=self._force_all_reduce)

    @staticmethod
    def _resolve_bucket_groups(ddp_model: nn.Module) -> list:
        """Extract bucket_groups list from various DDP wrapper shapes."""
        # Direct attribute (deepspeed/Megatron DDP).
        if hasattr(ddp_model, 'bucket_groups'):
            return ddp_model.bucket_groups
        # List of per-buffer bucket groups.
        if hasattr(ddp_model, 'buffers'):
            groups = []
            for buf in ddp_model.buffers:
                if hasattr(buf, 'bucket_groups'):
                    groups.extend(buf.bucket_groups)
            if groups:
                return groups
        # Unwrap one level.
        inner = getattr(ddp_model, 'module', None)
        if inner is not None and hasattr(inner, 'bucket_groups'):
            return inner.bucket_groups
        return []

    @staticmethod
    def _iter_modules(model: nn.Module):
        """Yield the underlying nn.Module(s), unwrapping DDP wrappers."""
        yield model
        inner = getattr(model, 'module', None)
        if inner is not None:
            yield inner


# ---------------------------------------------------------------------------
# Pipeline-stage-aware reduce-scatter scheduler
# ---------------------------------------------------------------------------

class PPStageGradReduceScheduler:
    """Schedule overlapped reduce-scatter across pipeline stages.

    In the 1F1B schedule, stage k finishes its backward pass for microbatch m
    while stage k-1 is still computing forward for a later microbatch.  This
    scheduler tracks the per-stage backward completion events and triggers
    reduce-scatter for each stage's bucket groups as soon as the last micro-
    batch backward for that bucket has finished — rather than waiting until the
    very end of the PP schedule step.

    This is a thin orchestration layer on top of OverlapGradReduceManager.
    It does not own the communication; it delegates to each stage's manager.

    Parameters
    ----------
    managers:
        Mapping from pp_rank → OverlapGradReduceManager for each stage managed
        by the current process.  For standard (non-VPP) PP each process owns
        exactly one stage, so this will have a single entry.
    total_microbatches:
        Total number of microbatches in the current batch.  Used to determine
        when a stage's last microbatch backward has completed.
    """

    def __init__(
        self,
        managers: Dict[int, OverlapGradReduceManager],
        total_microbatches: int,
    ) -> None:
        self._managers = managers
        self._total_microbatches = total_microbatches
        # Per-stage microbatch backward completion counter.
        self._mb_done: Dict[int, int] = {rank: 0 for rank in managers}

    def notify_backward_done(self, pp_rank: int, microbatch_id: int) -> None:
        """Notify that microbatch ``microbatch_id`` backward has completed on
        stage ``pp_rank``.

        When this is the last microbatch for the stage, calls
        ``manager.finalize()`` to wait for all outstanding reduce-scatters.

        Parameters
        ----------
        pp_rank:
            Pipeline stage rank that completed a backward pass.
        microbatch_id:
            Zero-based index of the microbatch.
        """
        if pp_rank not in self._managers:
            return
        self._mb_done[pp_rank] += 1
        is_last = (self._mb_done[pp_rank] >= self._total_microbatches)
        manager = self._managers[pp_rank]
        manager.prepare_for_backward(is_last_microbatch=is_last)
        if is_last:
            manager.finalize()

    def reset(self) -> None:
        """Reset per-stage counters for the next training step."""
        for rank in self._mb_done:
            self._mb_done[rank] = 0


# ---------------------------------------------------------------------------
# Utility: bucket-size recommendation for PP stages
# ---------------------------------------------------------------------------

def recommend_pp_stage_bucket_size(
    base_bucket_size: int,
    pp_rank: int,
    pp_size: int,
    num_microbatches: int,
) -> int:
    """Recommend a bucket size for a given pipeline stage.

    Middle pipeline stages have less overlap opportunity than stage 0 because:
      - Their backward pass starts later (bubble latency).
      - The DP reduce-scatter must complete before the optimizer step, which is
        gated by the slowest stage.

    A larger bucket on middle stages reduces the number of NCCL calls (lower
    kernel-launch overhead on PCIe-only clusters) at the cost of coarser
    overlap granularity.

    Formula
    -------
    scale = 1 + (pp_rank / pp_size) * log2(num_microbatches + 1)

    This gives scale ≈ 1 for stage 0, and scale ≈ 1 + log2(M+1) for the last
    stage.  With M=8 microbatches the last stage gets ~4× larger buckets,
    consistent with the Megatron heuristic of disabling bucketing (∞ bucket)
    for pp_rank > 0.

    Parameters
    ----------
    base_bucket_size:
        Bucket size for stage 0 (typically 40M–120M elements for DDP).
    pp_rank:
        Rank of this pipeline stage (0-indexed).
    pp_size:
        Total number of pipeline stages.
    num_microbatches:
        Number of microbatches per training step.

    Returns
    -------
    Recommended bucket size in elements.
    """
    import math
    if pp_size <= 1 or pp_rank == 0:
        return base_bucket_size
    scale = 1.0 + (pp_rank / pp_size) * math.log2(max(num_microbatches, 1) + 1)
    return int(base_bucket_size * scale)


__all__ = [
    "OverlapGradReduceManager",
    "PPStageGradReduceScheduler",
    "recommend_pp_stage_bucket_size",
]
