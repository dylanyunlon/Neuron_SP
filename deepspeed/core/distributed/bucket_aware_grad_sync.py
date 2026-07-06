# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Bucket-aware gradient synchronization for DES-LOC heterogeneous GPU clusters.

Overview
--------
On NVLink-connected clusters (e.g., DGX) the default Megatron DDP overlap
strategy works well: every bucket dispatches an async reduce-scatter as soon
as all its gradients are ready, and the NCCL collective overlaps with the
continuing backward pass.

On PCIe-only or mixed NVLink + PCIe clusters (DES-LOC heterogeneous topology:
A6000 × 2 + H100-NVL × 1 connected over PCIe) there are two complications:

  1. **Latency dominance for small buckets** — PCIe round-trip latency (~10 µs)
     is 5-10× higher than NVLink.  Small bucket collectives whose transfer time
     is shorter than the latency overhead are more efficiently run *synchronously*
     (no async queue depth, no CUDA stream stall).

  2. **Bandwidth asymmetry** — A6000 PCIe (16 GB/s) vs H100 NVLink (400 GB/s).
     Optimal bucket size for A6000 is ~0.5× the default; H100 can absorb 1.5×.
     compute_tier_bucket_sizes() in param_and_grad_buffer.py handles this.

This module implements **Insight I6** (PCIe-aware overlap, Megatron aa-3.5
pattern) as a standalone synchronization policy that wraps
``ParamAndGradBucketGroup.start_grad_sync`` / ``finish_grad_sync``:

  * ``BucketAwareGradSync`` — context manager that drives per-bucket collective
    dispatch for one backward pass, selecting async vs sync mode based on the
    PCIe overlap threshold.
  * ``pcie_overlap_trigger_elems()`` — compute the minimum bucket element count
    at which async dispatch pays off over PCIe.
  * ``pcie_bucket_size()`` — recommend a PCIe-optimal bucket size for DDP init.
  * ``should_use_async_op()`` — predicate used by BucketAwareGradSync and the
    distributed optimizer's ``_reduce_scatter_grads()``.

Design notes
------------
* The trigger-threshold computation mirrors the one in
  ``deepspeed/core/optimizer/optimizer_config.py :: pcie_overlap_trigger_elems``
  so both the DDP path and the DistributedOptimizer path share the same
  bandwidth/latency model.

* ``BucketAwareGradSync`` is a *context manager* so that callers can bracket a
  complete backward pass and automatically flush any remaining pending collectives
  on exit without boilerplate.

* When ``overlap_grad_reduce=False`` (synchronous mode), this module degenerates
  to a thin wrapper that calls ``finish_grad_sync`` sequentially — no async state
  is needed.

Public API
----------
  pcie_overlap_trigger_elems(pcie_bw_gbps, pcie_latency_us)  → int
  pcie_bucket_size(pcie_bw_gbps, pcie_latency_us, dp_world_size, dtype_bytes) → int
  should_use_async_op(bucket_numel, trigger_elems, overlap_grad_reduce) → bool
  BucketAwareGradSync
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, List, Optional

import torch

logger = logging.getLogger(__name__)

# Default PCIe 4.0 ×16 unidirectional bandwidth and typical host↔device
# round-trip latency.  These match the defaults in DistributedDataParallelConfig
# and optimizer_config.OptimizerConfig.
_DEFAULT_PCIE_BW_GBPS: float = 16.0
_DEFAULT_PCIE_LATENCY_US: float = 10.0

# NVLink bucket-size default used as an upper cap (element count of float32).
# Mirrors Megatron DDP_BUCKET_SIZE = 40_000_000 elements × 4 bytes = 160 MB.
_NVLINK_DEFAULT_BUCKET_ELEMS: int = 40_000_000


# ---------------------------------------------------------------------------
# Bandwidth / latency helpers
# ---------------------------------------------------------------------------

def pcie_overlap_trigger_elems(
    pcie_bw_gbps: float = _DEFAULT_PCIE_BW_GBPS,
    pcie_latency_us: float = _DEFAULT_PCIE_LATENCY_US,
    dtype_bytes: int = 2,
) -> int:
    """Compute the minimum bucket element count for async overlap over PCIe.

    An async collective is only useful when its transfer time is at least as
    long as the PCIe round-trip latency; otherwise the async-launch overhead
    dominates.  The threshold is:

        trigger_bytes = pcie_latency_us × pcie_bw_gbps × 1e3

    (where 1e3 = 1e9 bytes/GB × 1e-6 s/µs).

    Args:
        pcie_bw_gbps:  Effective unidirectional PCIe bandwidth in GB/s.
        pcie_latency_us: PCIe round-trip latency in microseconds.
        dtype_bytes:   Bytes per gradient element (default 2 = BF16/FP16).

    Returns:
        Minimum number of elements in a bucket to justify async dispatch.
        Always ≥ 1.

    Examples
    --------
    >>> pcie_overlap_trigger_elems(16.0, 10.0, dtype_bytes=2)
    80000   # 160 KB / 2 bytes per BF16 element
    """
    pcie_bw_bytes_per_s = pcie_bw_gbps * 1e9
    latency_s = pcie_latency_us * 1e-6
    trigger_bytes = latency_s * pcie_bw_bytes_per_s
    trigger_elems = max(1, int(trigger_bytes / dtype_bytes))
    return trigger_elems


def pcie_bucket_size(
    pcie_bw_gbps: float = _DEFAULT_PCIE_BW_GBPS,
    pcie_latency_us: float = _DEFAULT_PCIE_LATENCY_US,
    dp_world_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Recommend a PCIe-optimal bucket size for DDP initialization.

    On PCIe-only topologies the NVLink default of ~40 M elements is too large:
    the collective takes much longer than the backward pass, destroying overlap.
    A good PCIe bucket should satisfy:

      transfer_time ≥ 4 × latency  (so compute meaningfully overlaps)

    The formula therefore targets 4× the trigger threshold, then clamps to
    [500 K × dp_world_size, NVLink-default].

    Args:
        pcie_bw_gbps:    PCIe bandwidth GB/s.
        pcie_latency_us: PCIe latency µs.
        dp_world_size:   Data-parallel world size (scales minimum bucket).
        dtype_bytes:     Bytes per element.

    Returns:
        Recommended bucket size in elements.
    """
    pcie_bw_bytes_per_s = pcie_bw_gbps * 1e9
    latency_s = pcie_latency_us * 1e-6
    min_bucket_bytes = 4.0 * latency_s * pcie_bw_bytes_per_s
    min_bucket_elems = max(1, int(min_bucket_bytes / dtype_bytes))
    pcie_bucket = max(min_bucket_elems, 500_000 * dp_world_size)
    return min(pcie_bucket, _NVLINK_DEFAULT_BUCKET_ELEMS)


def should_use_async_op(
    bucket_numel: int,
    trigger_elems: int,
    overlap_grad_reduce: bool,
) -> bool:
    """Decide whether to dispatch an async collective for a bucket.

    An async collective is dispatched when:
    1. ``overlap_grad_reduce=True`` (async overlap is enabled globally), **and**
    2. the bucket has at least ``trigger_elems`` elements (transfer time ≥ latency).

    When ``overlap_grad_reduce=False`` this always returns ``False``, making
    every collective synchronous regardless of bucket size.

    Args:
        bucket_numel:       Element count of the bucket's grad tensor.
        trigger_elems:      Minimum element count for async (from
                            ``pcie_overlap_trigger_elems()``).
        overlap_grad_reduce: Whether async overlap is globally enabled.

    Returns:
        ``True`` if an async collective should be launched.
    """
    if not overlap_grad_reduce:
        return False
    return bucket_numel >= trigger_elems


# ---------------------------------------------------------------------------
# BucketAwareGradSync
# ---------------------------------------------------------------------------

class BucketAwareGradSync:
    """PCIe-aware per-bucket gradient synchronization manager.

    Wraps the gradient-sync lifecycle for a list of ``ParamAndGradBucketGroup``
    objects, selecting async vs sync collective dispatch on a per-bucket basis
    according to the PCIe overlap threshold (Insight I6).

    Usage
    -----
    Typically constructed once per training iteration in the DDP wrapper's
    ``finish_grad_sync`` path:

    .. code-block:: python

        sync = BucketAwareGradSync(
            bucket_groups=ddp.bucket_groups,
            ddp_config=ddp.ddp_config,
        )
        with sync:
            model(inputs).backward()
        # All pending collectives are flushed on context-manager exit.

    Alternatively, use the ``flush()`` method to drain pending handles at
    the end of a backward pass without the context-manager protocol.

    Attributes
    ----------
    bucket_groups : list of ParamAndGradBucketGroup
    ddp_config    : DistributedDataParallelConfig
    _pending_handles : list of handles from async collectives
    _trigger_elems   : per-bucket async trigger threshold
    """

    def __init__(
        self,
        bucket_groups: List,
        ddp_config,
    ) -> None:
        self.bucket_groups = bucket_groups
        self.ddp_config = ddp_config

        # Compute PCIe overlap trigger once at construction.
        use_pcie = getattr(ddp_config, 'use_pcie_aware_overlap', False)
        if use_pcie:
            bw = getattr(ddp_config, 'pcie_bw_gbps', _DEFAULT_PCIE_BW_GBPS)
            lat = getattr(ddp_config, 'pcie_latency_us', _DEFAULT_PCIE_LATENCY_US)
            self._trigger_elems = pcie_overlap_trigger_elems(bw, lat)
            logger.info(
                "[BucketAwareGradSync] PCIe-aware overlap enabled: "
                "bw=%.1f GB/s, latency=%.1f µs, trigger=%d elems",
                bw, lat, self._trigger_elems,
            )
        else:
            # Non-PCIe path: threshold of 0 means always async (if overlap_grad_reduce=True).
            self._trigger_elems = 0

        self._pending_handles: List = []

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "BucketAwareGradSync":
        """Reset state for a new backward pass."""
        self._pending_handles.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Flush all pending async collectives on exit.

        Does not suppress exceptions (always returns False).
        """
        if exc_type is None:
            self.flush()
        return False

    # ------------------------------------------------------------------
    # Per-bucket dispatch helpers
    # ------------------------------------------------------------------

    def dispatch_bucket_group(
        self,
        bucket_group,
        force_all_reduce: bool = False,
        skip_sync: bool = False,
    ) -> None:
        """Dispatch gradient sync for a single bucket group.

        Selects async vs sync mode per PCIe overlap policy:
        - If overlap_grad_reduce is False → synchronous (calls finish_grad_sync
          which dispatches + waits inline).
        - If overlap_grad_reduce is True and bucket is above threshold → async
          (calls start_grad_sync then remembers the handle for later wait).
        - If overlap_grad_reduce is True but bucket is below threshold → sync.

        Args:
            bucket_group:     The ``ParamAndGradBucketGroup`` to sync.
            force_all_reduce: Force all-reduce even in distributed-optimizer mode.
            skip_sync:        DES-LOC Kx gate — skip collective on non-Kx steps.
        """
        overlap = getattr(self.ddp_config, 'overlap_grad_reduce', False)
        # Compute total numel for this group's first bucket (representative).
        total_numel = sum(b.grad_data.numel() for b in bucket_group.buckets)
        use_async = should_use_async_op(total_numel, self._trigger_elems, overlap)

        if not use_async:
            # Synchronous path: dispatch + wait inline via finish_grad_sync.
            # Pass skip_sync through for DES-LOC Kx gating.
            if overlap:
                # In overlap mode finish_grad_sync dispatches if needed and waits.
                bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)
            else:
                # Non-overlap: finish_grad_sync → start_grad_sync (sync) → return.
                bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)
            return

        # Async path: dispatch the collective and record it.
        bucket_group.start_grad_sync(
            force_all_reduce=force_all_reduce,
            skip_sync=skip_sync,
        )
        if bucket_group.grad_reduce_handle is not None:
            self._pending_handles.append(bucket_group)

    def flush(self, force_all_reduce: bool = False) -> None:
        """Wait on all pending async collectives and clear the pending list.

        Called automatically on context-manager ``__exit__`` and may also be
        called explicitly at the end of a backward pass before the optimizer step.

        Args:
            force_all_reduce: Forward to ``finish_grad_sync``.
        """
        for bucket_group in self._pending_handles:
            bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)
        self._pending_handles.clear()

    # ------------------------------------------------------------------
    # Batch dispatch (all-at-once, non-overlap mode)
    # ------------------------------------------------------------------

    def finish_all_grad_syncs(self, force_all_reduce: bool = False) -> None:
        """Synchronously finish gradient sync for every bucket group.

        This is the end-of-step call equivalent to Megatron DDP's
        ``finish_grad_sync`` loop.  It is idempotent for groups that have
        already been synced via the async path.

        Args:
            force_all_reduce: Force all-reduce even in distributed-optimizer mode.
        """
        for bucket_group in self.bucket_groups:
            bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)

    # ------------------------------------------------------------------
    # Iterator support
    # ------------------------------------------------------------------

    def iter_bucket_groups(self) -> Iterator:
        """Yield bucket groups in reverse backward-pass order."""
        yield from reversed(self.bucket_groups)


# ---------------------------------------------------------------------------
# Context manager factory (convenience)
# ---------------------------------------------------------------------------

@contextmanager
def bucket_aware_grad_sync_context(
    bucket_groups: List,
    ddp_config,
    force_all_reduce: bool = False,
) -> Iterator[BucketAwareGradSync]:
    """Context manager factory for a single backward pass.

    Equivalent to constructing ``BucketAwareGradSync`` and using it as a
    context manager, but also calls ``finish_all_grad_syncs()`` on exit to
    guarantee all bucket groups are drained.

    Args:
        bucket_groups:    List of ``ParamAndGradBucketGroup``.
        ddp_config:       ``DistributedDataParallelConfig`` instance.
        force_all_reduce: Forward to ``finish_all_grad_syncs``.

    Yields:
        The active ``BucketAwareGradSync`` instance.

    Example
    -------
    .. code-block:: python

        with bucket_aware_grad_sync_context(ddp.bucket_groups, ddp.ddp_config) as sync:
            loss.backward()
        # All collectives are finished here.
    """
    sync = BucketAwareGradSync(bucket_groups=bucket_groups, ddp_config=ddp_config)
    with sync:
        yield sync
    sync.finish_all_grad_syncs(force_all_reduce=force_all_reduce)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def log_bucket_sync_plan(
    bucket_groups: List,
    ddp_config,
    log_level: int = logging.DEBUG,
) -> None:
    """Log the async/sync assignment for each bucket group.

    Useful for debugging overlap efficiency on a new cluster topology.

    Args:
        bucket_groups: List of ``ParamAndGradBucketGroup``.
        ddp_config:    ``DistributedDataParallelConfig`` instance.
        log_level:     Python logging level (default DEBUG).
    """
    use_pcie = getattr(ddp_config, 'use_pcie_aware_overlap', False)
    overlap = getattr(ddp_config, 'overlap_grad_reduce', False)
    bw = getattr(ddp_config, 'pcie_bw_gbps', _DEFAULT_PCIE_BW_GBPS)
    lat = getattr(ddp_config, 'pcie_latency_us', _DEFAULT_PCIE_LATENCY_US)
    trigger = pcie_overlap_trigger_elems(bw, lat) if use_pcie else 0

    logger.log(
        log_level,
        "[BucketAwareGradSync] sync plan: overlap=%s, pcie_aware=%s, trigger=%d",
        overlap, use_pcie, trigger,
    )
    for i, bg in enumerate(bucket_groups):
        numel = sum(b.grad_data.numel() for b in bg.buckets)
        mode = "async" if should_use_async_op(numel, trigger, overlap) else "sync"
        logger.log(
            log_level,
            "  bucket_group[%d]: numel=%d, num_buckets=%d, mode=%s",
            i, numel, len(bg.buckets), mode,
        )
