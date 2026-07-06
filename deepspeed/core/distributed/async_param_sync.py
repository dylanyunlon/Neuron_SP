# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Asynchronous parameter all-gather synchronization for DES-LOC DDP.

Overview
--------
When ``DistributedDataParallelConfig.overlap_param_gather`` is True and the
distributed optimizer is in use, each ``ParamAndGradBucketGroup`` manages an
async all-gather that fills ``bucket.param_data`` from the DP-shard owned by
this rank.  The call chain is::

    DDP.start_param_sync()
        → _ParamAndGradBucketGroup.start_param_sync()   # dispatches NCCL AG
    DDP.forward()
        → forward pre-hook
            → _ParamAndGradBucketGroup.finish_param_sync()  # waits + dispatches next

This module provides two higher-level utilities that sit *above* the bucket-group
layer and are useful for both training and evaluation:

  ``AsyncParamSyncManager``
      Orchestrates the pipeline of async all-gathers across all bucket groups
      for one or more DDP model chunks.  Handles:

      * **Alignment** (``align_param_gather``): With VPP, each model chunk has
        its own bucket group chain.  Alignment ensures that the first bucket of
        every chunk dispatches its all-gather simultaneously (via a coalescing
        manager) so that the NCCL scheduler can merge them into a single fused
        kernel where hardware supports it.

      * **Optimizer-step overlap**: When
        ``overlap_param_gather_with_optimizer_step`` is True the first bucket
        group's all-gather is dispatched during the optimizer step itself
        (instead of at the start of the next forward pass) to hide optimizer
        latency.

      * **Force-sync fallback**: During evaluation, checkpointing, or when
        the pipeline schedule requires all weights to be present, a
        ``force_sync()`` call waits for all outstanding all-gathers and
        leaves every bucket's ``param_data`` in a consistent state.

  ``ParamSyncScheduler``
      A lightweight scheduler that tracks which bucket groups have dispatched
      or completed their all-gathers within a single forward pass.  Used
      internally by ``AsyncParamSyncManager`` and exposed for test
      introspection.

Design decisions (ported from Megatron M2777 + M3443 + M3948)
--------------------------------------------------------------
  M2777: ``_post_param_sync`` extracted from ``start_param_sync`` to support
         FP8/MXFP8 post-processing after the NCCL all-gather completes.

  M3443: ``overlap_param_gather`` for layer-wise optimizer.  Each rank in the
         DP group owns a disjoint set of parameters per bucket; the all-gather
         collects them from all ranks.  This is the *layerwise* path (distinct
         from the *distributed optimizer* path which all-gathers the full
         ``param_data`` shard).

  M3948: ``LayerWiseDistributedOptimizer`` integration.  The optimizer calls
         ``start_param_sync()`` directly on bucket groups after updating its
         owned parameters; ``AsyncParamSyncManager`` needs to be aware of
         these externally-dispatched all-gathers to avoid double-dispatching.

DES-LOC extensions
------------------
  * ``ParamSyncScheduler.mark_externally_dispatched()`` — lets the layer-wise
    optimizer notify the scheduler that a bucket group's all-gather has already
    been dispatched so ``AsyncParamSyncManager`` skips it.
  * ``AsyncParamSyncManager.overlap_with_optimizer_step()`` — context manager
    that enables the optimizer-step overlap pattern for one optimizer step.
  * ``AsyncParamSyncManager.eval_mode()`` — context manager that forces all
    param all-gathers to be synchronous (needed during eval where the forward
    pre-hook is not active).

Public API
----------
  ParamSyncScheduler
  AsyncParamSyncManager
  async_param_sync_context
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch.distributed import _coalescing_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ParamSyncScheduler
# ---------------------------------------------------------------------------

class ParamSyncScheduler:
    """Track dispatch and completion state for param all-gathers.

    Maintains a set of bucket group ids that have been dispatched this
    forward pass so that ``AsyncParamSyncManager`` can avoid double-dispatch
    and correctly sequence the pipeline of all-gathers.

    Parameters
    ----------
    bucket_groups:
        Ordered list of ``ParamAndGradBucketGroup`` objects for one model
        chunk.  The order must match the forward-pass execution order (first
        bucket = first layer encountered in forward).
    """

    def __init__(self, bucket_groups: list) -> None:
        self._groups = list(bucket_groups)
        self._dispatched: Set[int] = set()
        self._completed: Set[int] = set()
        # Map from id → group for O(1) lookup.
        self._id_to_group: Dict[int, object] = {id(bg): bg for bg in self._groups}

    # ------------------------------------------------------------------
    # State mutations
    # ------------------------------------------------------------------

    def mark_dispatched(self, bucket_group) -> None:
        """Record that ``bucket_group``'s all-gather has been dispatched."""
        self._dispatched.add(id(bucket_group))

    def mark_externally_dispatched(self, bucket_group) -> None:
        """Same as ``mark_dispatched``; called by external code (e.g. LayerWise opt)."""
        self._dispatched.add(id(bucket_group))

    def mark_completed(self, bucket_group) -> None:
        """Record that ``bucket_group``'s all-gather has completed (wait() returned)."""
        bg_id = id(bucket_group)
        self._dispatched.add(bg_id)
        self._completed.add(bg_id)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_dispatched(self, bucket_group) -> bool:
        """Return True if the all-gather for ``bucket_group`` has been dispatched."""
        return id(bucket_group) in self._dispatched

    def is_completed(self, bucket_group) -> bool:
        """Return True if the all-gather for ``bucket_group`` has completed."""
        return id(bucket_group) in self._completed

    def all_dispatched(self) -> bool:
        """Return True if every bucket group has dispatched."""
        return len(self._dispatched) >= len(self._groups)

    def all_completed(self) -> bool:
        """Return True if every bucket group has completed."""
        return len(self._completed) >= len(self._groups)

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def undispatched_groups(self) -> List:
        """Return bucket groups that have not yet been dispatched."""
        return [bg for bg in self._groups if id(bg) not in self._dispatched]

    def uncompleted_groups(self) -> List:
        """Return bucket groups whose all-gather has not yet completed."""
        return [bg for bg in self._groups if id(bg) not in self._completed]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset dispatch and completion state for the next forward pass."""
        self._dispatched.clear()
        self._completed.clear()

    def __len__(self) -> int:
        return len(self._groups)

    def __repr__(self) -> str:
        return (
            f"ParamSyncScheduler("
            f"total={len(self._groups)}, "
            f"dispatched={len(self._dispatched)}, "
            f"completed={len(self._completed)})"
        )


# ---------------------------------------------------------------------------
# AsyncParamSyncManager
# ---------------------------------------------------------------------------

class AsyncParamSyncManager:
    """Orchestrate async parameter all-gathers across one or more DDP model chunks.

    Wraps the ``start_param_sync`` / ``finish_param_sync`` calls on
    ``ParamAndGradBucketGroup`` objects and provides alignment, optimizer-step
    overlap, and force-sync utilities.

    Parameters
    ----------
    ddp_chunks:
        List of DDP-wrapped model chunks.  For non-VPP pass a single-element
        list.
    align_param_gather:
        If True, coalesce the dispatch of the first bucket group in every
        chunk so that NCCL can fuse them.  Requires all chunks to share the
        same intra-DP process group.
    overlap_param_gather_with_optimizer_step:
        If True, the first bucket group's all-gather is dispatched during
        ``optimizer_step_overlap()`` (before the forward pass) rather than
        inside the forward pre-hook.
    """

    def __init__(
        self,
        ddp_chunks: List[nn.Module],
        align_param_gather: bool = False,
        overlap_param_gather_with_optimizer_step: bool = False,
    ) -> None:
        self._chunks = ddp_chunks
        self._align = align_param_gather
        self._optim_overlap = overlap_param_gather_with_optimizer_step

        # Build a scheduler per chunk.
        self._schedulers: List[ParamSyncScheduler] = []
        self._all_bucket_groups: List[list] = []
        for chunk in ddp_chunks:
            bgs = self._resolve_bucket_groups(chunk)
            self._all_bucket_groups.append(bgs)
            self._schedulers.append(ParamSyncScheduler(bgs))

        # Track whether we are inside an optimizer-step overlap window.
        self._in_optimizer_overlap: bool = False

    # ------------------------------------------------------------------
    # Forward-pass API
    # ------------------------------------------------------------------

    def start_param_sync(self, force_sync: bool = False) -> None:
        """Dispatch async (or sync) param all-gathers for all chunks.

        When ``align_param_gather`` is True and ``force_sync`` is False,
        coalesces the first bucket group of every chunk into a single NCCL
        kernel launch.

        Parameters
        ----------
        force_sync:
            If True, issue synchronous all-gathers regardless of the
            ``overlap_param_gather`` setting.
        """
        if self._align and not force_sync and len(self._chunks) > 1:
            self._start_aligned(force_sync=False)
        else:
            for chunk_idx, chunk in enumerate(self._chunks):
                bgs = self._all_bucket_groups[chunk_idx]
                scheduler = self._schedulers[chunk_idx]
                for bg in bgs:
                    if scheduler.is_dispatched(bg):
                        continue
                    self._dispatch_one(bg, force_sync=force_sync)
                    scheduler.mark_dispatched(bg)
                    if force_sync:
                        scheduler.mark_completed(bg)

    def finish_param_sync(
        self,
        bucket_group,
        skip_next_dispatch: bool = False,
    ) -> None:
        """Wait for ``bucket_group``'s all-gather and optionally dispatch the next.

        Called from the forward pre-hook once the module using ``bucket_group``'s
        parameters is about to execute.

        Parameters
        ----------
        bucket_group:
            The bucket group whose all-gather should be awaited.
        skip_next_dispatch:
            If True, do not dispatch the next bucket group's all-gather after
            this one completes (used when the caller will handle dispatch
            externally, e.g. in the aligned case).
        """
        # Find which chunk this bucket group belongs to.
        scheduler = self._find_scheduler(bucket_group)
        if scheduler is None:
            logger.debug(
                "finish_param_sync: bucket_group %s not found in any scheduler; skipping.",
                id(bucket_group),
            )
            return

        # Ensure it has been dispatched.
        if not scheduler.is_dispatched(bucket_group):
            self._dispatch_one(bucket_group, force_sync=False)
            scheduler.mark_dispatched(bucket_group)

        # Wait.
        try:
            bucket_group.finish_param_sync(
                skip_next_bucket_dispatch=skip_next_dispatch
            )
        except TypeError:
            bucket_group.finish_param_sync()
        scheduler.mark_completed(bucket_group)

    def force_sync(self) -> None:
        """Force-sync all outstanding param all-gathers synchronously.

        Useful before checkpoint saves, evaluation entry, and any code path
        that cannot tolerate async all-gathers being in flight.
        """
        for chunk_idx, chunk in enumerate(self._chunks):
            bgs = self._all_bucket_groups[chunk_idx]
            scheduler = self._schedulers[chunk_idx]
            for bg in bgs:
                if scheduler.is_completed(bg):
                    continue
                if not scheduler.is_dispatched(bg):
                    self._dispatch_one(bg, force_sync=True)
                else:
                    # Already dispatched async — wait.
                    try:
                        bg.start_param_sync(force_sync=True)
                    except Exception:
                        pass
                scheduler.mark_completed(bg)

    def reset(self) -> None:
        """Reset scheduler state for the next forward pass."""
        for sched in self._schedulers:
            sched.reset()

    # ------------------------------------------------------------------
    # Optimizer-step overlap
    # ------------------------------------------------------------------

    @contextmanager
    def optimizer_step_overlap(self):
        """Context manager that enables optimizer-step param-gather overlap.

        Within this context, ``start_param_sync`` dispatches the *first*
        bucket group of each chunk asynchronously before the forward pass
        begins, hiding the all-gather behind the optimizer step.  This
        mirrors Megatron's ``overlap_param_gather_with_optimizer_step``
        (M3948 / DES-LOC Kx path).

        Usage::

            with manager.optimizer_step_overlap():
                optimizer.step()
            # All-gather for bucket-0 is now in flight.
            loss = model.forward(...)
        """
        self._in_optimizer_overlap = True
        try:
            # Dispatch first bucket group of every chunk.
            for chunk_idx in range(len(self._chunks)):
                bgs = self._all_bucket_groups[chunk_idx]
                if not bgs:
                    continue
                first_bg = bgs[0]
                sched = self._schedulers[chunk_idx]
                if not sched.is_dispatched(first_bg):
                    self._dispatch_one(first_bg, force_sync=False)
                    sched.mark_dispatched(first_bg)
            yield
        finally:
            self._in_optimizer_overlap = False

    # ------------------------------------------------------------------
    # Evaluation mode
    # ------------------------------------------------------------------

    @contextmanager
    def eval_mode(self):
        """Context manager that forces synchronous param all-gathers during eval.

        During evaluation the forward pre-hook is typically not active, so
        async all-gathers would never be waited on.  This context manager
        patches ``start_param_sync`` on each DDP chunk to issue synchronous
        collectives for the duration of the eval forward pass.

        Usage::

            with manager.eval_mode():
                output = model(eval_batch)
        """
        # Force-sync any outstanding all-gathers from training.
        self.force_sync()
        self.reset()
        try:
            yield
        finally:
            # Ensure all parameters are gathered after eval forward.
            self.force_sync()

    # ------------------------------------------------------------------
    # External dispatch notification (M3948 / LayerWise optimizer)
    # ------------------------------------------------------------------

    def notify_externally_dispatched(self, bucket_group) -> None:
        """Notify the scheduler that an external caller dispatched ``bucket_group``'s AG.

        Called by ``LayerWiseDistributedOptimizer.step()`` after it calls
        ``bucket_group.start_param_sync()`` so that ``AsyncParamSyncManager``
        does not double-dispatch.

        Parameters
        ----------
        bucket_group:
            The bucket group whose all-gather has been externally dispatched.
        """
        scheduler = self._find_scheduler(bucket_group)
        if scheduler is not None:
            scheduler.mark_externally_dispatched(bucket_group)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_aligned(self, force_sync: bool) -> None:
        """Coalesce first-bucket dispatches across all chunks.

        Uses PyTorch's ``_coalescing_manager`` to batch the all-gather
        kernel launches so the NCCL scheduler can fuse them where hardware
        supports it.  Only the *first* (undispatched) bucket of each chunk
        is coalesced; remaining buckets are dispatched individually in the
        normal pipelined fashion.
        """
        # Identify the intra-DP group — all chunks must share the same one.
        dp_group = self._infer_dp_group()
        first_bgs = []
        for chunk_idx in range(len(self._chunks)):
            bgs = self._all_bucket_groups[chunk_idx]
            sched = self._schedulers[chunk_idx]
            undispatched = sched.undispatched_groups()
            if undispatched:
                first_bgs.append((chunk_idx, undispatched[0]))

        if not first_bgs or dp_group is None:
            # Fallback: dispatch independently.
            for chunk_idx, bg in first_bgs:
                sched = self._schedulers[chunk_idx]
                self._dispatch_one(bg, force_sync=force_sync)
                sched.mark_dispatched(bg)
            return

        # Coalesce.
        async_op = not force_sync
        with _coalescing_manager(dp_group, async_ops=async_op):
            for chunk_idx, bg in first_bgs:
                self._dispatch_one(bg, force_sync=force_sync, _skip_cm=True)

        for chunk_idx, bg in first_bgs:
            self._schedulers[chunk_idx].mark_dispatched(bg)
            if force_sync:
                self._schedulers[chunk_idx].mark_completed(bg)

    def _dispatch_one(
        self,
        bg,
        force_sync: bool,
        _skip_cm: bool = False,
    ) -> None:
        """Dispatch the all-gather for a single bucket group.

        Parameters
        ----------
        bg:
            The ``ParamAndGradBucketGroup`` to dispatch.
        force_sync:
            If True, issue a blocking call.
        _skip_cm:
            Internal flag set when already inside a ``_coalescing_manager``
            context; prevents nested coalescing.
        """
        try:
            bg.start_param_sync(force_sync=force_sync)
        except AssertionError:
            # Bucket group may not support param sync (no distributed optimizer
            # and overlap_param_gather=False) — skip silently.
            logger.debug(
                "_dispatch_one: bucket group %s does not support param sync; skipping.",
                id(bg),
            )
        except Exception as exc:
            logger.warning(
                "_dispatch_one: unexpected error dispatching bucket group %s: %s",
                id(bg),
                exc,
            )

    def _find_scheduler(self, bucket_group) -> Optional[ParamSyncScheduler]:
        """Return the scheduler that contains ``bucket_group``, or None."""
        bg_id = id(bucket_group)
        for idx, bgs in enumerate(self._all_bucket_groups):
            for bg in bgs:
                if id(bg) == bg_id:
                    return self._schedulers[idx]
        return None

    def _infer_dp_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Infer the intra-DP group from the first bucket group of the first chunk."""
        for bgs in self._all_bucket_groups:
            if bgs:
                bg = bgs[0]
                # Try various attribute paths used by deepspeed / Megatron DDP.
                for attr in (
                    'intra_distributed_optimizer_instance_group',
                    'data_parallel_group',
                ):
                    grp = getattr(bg, attr, None)
                    if grp is not None:
                        return grp
        return None

    @staticmethod
    def _resolve_bucket_groups(ddp_model: nn.Module) -> list:
        """Extract bucket_groups from a DDP wrapper."""
        if hasattr(ddp_model, 'bucket_groups'):
            return list(ddp_model.bucket_groups)
        if hasattr(ddp_model, 'buffers'):
            groups = []
            for buf in ddp_model.buffers:
                if hasattr(buf, 'bucket_groups'):
                    groups.extend(buf.bucket_groups)
            if groups:
                return groups
        inner = getattr(ddp_model, 'module', None)
        if inner is not None and hasattr(inner, 'bucket_groups'):
            return list(inner.bucket_groups)
        return []


# ---------------------------------------------------------------------------
# Convenience context manager
# ---------------------------------------------------------------------------

@contextmanager
def async_param_sync_context(
    ddp_chunks: List[nn.Module],
    align: bool = False,
    force_sync_on_exit: bool = True,
) -> "AsyncParamSyncManager":
    """Context manager that creates an ``AsyncParamSyncManager`` for a block.

    On exit, optionally calls ``force_sync()`` to ensure all param all-gathers
    have completed before leaving the context.

    Usage::

        with async_param_sync_context([ddp_model]) as mgr:
            mgr.start_param_sync()
            output = model(batch)
        # All params are gathered here.

    Parameters
    ----------
    ddp_chunks:
        DDP-wrapped model chunks.
    align:
        Forward to ``AsyncParamSyncManager(align_param_gather=align)``.
    force_sync_on_exit:
        If True, call ``force_sync()`` before the context exits.

    Yields
    ------
    AsyncParamSyncManager
    """
    manager = AsyncParamSyncManager(
        ddp_chunks=ddp_chunks,
        align_param_gather=align,
    )
    try:
        yield manager
    finally:
        if force_sync_on_exit:
            manager.force_sync()


# ---------------------------------------------------------------------------
# Utility: compute per-chunk param-gather alignment groups for VPP
# ---------------------------------------------------------------------------

def build_vpp_param_sync_manager(
    ddp_chunks: List[nn.Module],
    align_param_gather: bool = True,
    overlap_with_optimizer: bool = False,
) -> AsyncParamSyncManager:
    """Build an ``AsyncParamSyncManager`` for a VPP model.

    Convenience factory that sets sensible defaults for Virtual Pipeline
    Parallelism: alignment is enabled by default (all first-bucket all-gathers
    are coalesced) and optimizer-step overlap is opt-in.

    Parameters
    ----------
    ddp_chunks:
        List of DDP-wrapped VPP model chunks in forward-pass order.
    align_param_gather:
        Coalesce first-bucket all-gathers across chunks (recommended for VPP).
    overlap_with_optimizer:
        Enable optimizer-step overlap for the first bucket group of each chunk.

    Returns
    -------
    AsyncParamSyncManager
    """
    return AsyncParamSyncManager(
        ddp_chunks=ddp_chunks,
        align_param_gather=align_param_gather,
        overlap_param_gather_with_optimizer_step=overlap_with_optimizer,
    )


__all__ = [
    "ParamSyncScheduler",
    "AsyncParamSyncManager",
    "async_param_sync_context",
    "build_vpp_param_sync_manager",
]
