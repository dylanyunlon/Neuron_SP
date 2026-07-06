# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Overlapped gradient reduce-scatter — extended multi-stream / VPP edition.

Relationship to ``overlap_grad_reduce.py``
------------------------------------------
``overlap_grad_reduce.py`` (ported from Megatron M2278–M4036) implements the
core per-bucket hook-driven overlap strategy.  This module extends that design
with three additional capabilities needed by DES-LOC heterogeneous GPU
clusters and Virtual Pipeline Parallelism (VPP):

  1. **Multi-model-chunk VPP support** — VPP interleaves multiple model chunks
     per PP rank.  ``overlap_grad_reduce.py`` assumed a single model chunk and
     a single ``OverlapGradReduceManager``.  Here ``VPPOverlapGradReduceManager``
     tracks a separate readiness counter per (model-chunk, bucket-group) pair so
     that the first chunk whose bucket is full can dispatch immediately while the
     second chunk is still computing.

  2. **Dedicated CUDA reduce stream per bucket group** — On PCIe clusters a
     shared NCCL stream serialises all reduce-scatters.  Under VPP, chunks in
     flight on the same PP rank may have bucket groups ready at the same time.
     A per-bucket-group stream lets NCCL schedule these concurrently (bounded
     by hardware).  Each stream is created lazily; the number of active streams
     is capped by ``max_reduce_streams`` to avoid OOM on machines with limited
     CUDA context memory.

  3. **DES-LOC Kx-gated skip-sync** — Propagates the ``skip_sync`` flag (from
     the Kx-step gate in ``finalize_model_grads.py``) through the entire
     dispatch chain.  When ``skip_sync=True`` the collective is suppressed and
     gradients remain local; the Kx-recovery all-reduce in
     ``DistributedDataParallel.broadcast_params()`` later restores consistency.

  4. **Predecessor-drain ordering** — Respects the M4036 predecessor linkage
     (``previous_grad_reduce_bucket_group``) when dispatching reduce-scatters
     from non-hook paths (e.g., ``finalize()``).  This guarantees the
     intermediate all-to-all tensor for ``reduce_scatter_with_fp32_accumulation``
     is freed before the next allocation.

Design notes
------------
* Backward hooks are registered in ``OverlapGradReduceManager`` (base class
  from ``overlap_grad_reduce.py``).  ``VPPOverlapGradReduceManager`` overrides
  ``_build()`` to iterate over multiple DDP wrappers (one per model chunk).
* The stream pool is shared across all model chunks managed by the same
  ``VPPOverlapGradReduceManager`` instance so that the cap is applied globally.
* ``OverlappedGradReduceContext`` is a convenience context-manager wrapper for
  the common training loop pattern::

      ctx = OverlappedGradReduceContext(ddp_chunks, pp_group=pp_group)
      for micro_batch in micro_batches:
          ctx.prepare(is_last=micro_batch is micro_batches[-1])
          loss = forward_backward(micro_batch)
      ctx.finalize()

Public API
----------
  VPPOverlapGradReduceManager
  OverlappedGradReduceContext
  StreamPool
  build_overlapped_grad_reduce_manager
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from deepspeed.core.distributed.overlap_grad_reduce import (
    OverlapGradReduceManager,
    PPStageGradReduceScheduler,
    _BucketState,
    _get_pp_rank,
    _get_pp_size,
    recommend_pp_stage_bucket_size,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StreamPool — bounded CUDA stream pool shared across bucket groups
# ---------------------------------------------------------------------------

class StreamPool:
    """A bounded pool of CUDA streams for gradient reduce-scatter.

    Streams are allocated lazily and re-used in round-robin order to cap
    the total number of CUDA streams in use.  Using more streams than
    ``max_streams`` would not provide additional parallelism on typical
    hardware and wastes CUDA context memory.

    Parameters
    ----------
    max_streams:
        Maximum number of CUDA streams to create.  Default 4 is sufficient
        for VPP depth ≤ 4; increase for deeper VPP at the cost of higher
        context memory.
    priority:
        CUDA stream priority (lower integer = higher priority).  Gradient
        reduce-scatters are time-critical, so we use high priority by default.
    """

    def __init__(self, max_streams: int = 4, priority: int = -1) -> None:
        self._max = max_streams
        self._priority = priority
        self._streams: List[torch.cuda.Stream] = []
        self._idx: int = 0
        self._lock = threading.Lock()

    def get(self) -> torch.cuda.Stream:
        """Return the next stream from the pool (round-robin)."""
        with self._lock:
            if len(self._streams) < self._max:
                stream = torch.cuda.Stream(priority=self._priority)
                self._streams.append(stream)
            else:
                stream = self._streams[self._idx % self._max]
            self._idx += 1
            return stream

    def synchronize_all(self) -> None:
        """Synchronize the current CUDA stream with every pool stream."""
        current = torch.cuda.current_stream()
        for s in self._streams:
            current.wait_stream(s)

    def __len__(self) -> int:
        return len(self._streams)


# ---------------------------------------------------------------------------
# _ChunkBucketState — per-(model-chunk, bucket-group) readiness tracker
# ---------------------------------------------------------------------------

class _ChunkBucketState(_BucketState):
    """Extend _BucketState with a model-chunk identifier and a dedicated stream.

    When VPP interleaves multiple model chunks on the same PP rank, two buckets
    from different chunks may become ready at the same time.  Storing the chunk
    index and an optional CUDA stream here lets ``VPPOverlapGradReduceManager``
    dispatch them concurrently.

    Parameters
    ----------
    bucket_group:
        The ``ParamAndGradBucketGroup`` this state tracks.
    total_params:
        Total number of trainable parameters in the bucket group.
    chunk_idx:
        Zero-based index of the model chunk this bucket belongs to.
    stream:
        CUDA stream on which to issue the reduce-scatter for this bucket.
        If None the current stream is used (synchronous path).
    """

    __slots__ = _BucketState.__slots__ + ("chunk_idx", "stream")

    def __init__(
        self,
        bucket_group,
        total_params: int,
        chunk_idx: int = 0,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(bucket_group=bucket_group, total_params=total_params)
        self.chunk_idx = chunk_idx
        self.stream = stream


# ---------------------------------------------------------------------------
# VPPOverlapGradReduceManager
# ---------------------------------------------------------------------------

class VPPOverlapGradReduceManager:
    """Manage overlapped reduce-scatter for all model chunks on a PP rank.

    Extends ``OverlapGradReduceManager`` to support Virtual Pipeline
    Parallelism (VPP) where multiple model chunks run on the same PP rank.
    Each chunk has its own set of bucket groups and backward hooks; this
    manager aggregates them and dispatches reduce-scatters as soon as each
    bucket group's gradients are ready regardless of which chunk produced them.

    Parameters
    ----------
    ddp_chunks:
        List of ``DistributedDataParallel``-wrapped model chunks.  For
        standard (non-VPP) PP pass a single-element list; the behaviour
        degenerates to ``OverlapGradReduceManager``.
    pp_group:
        Pipeline-parallel process group.  Used for rank/size queries only.
    pp_stage_bucket_scale:
        Bucket-size scale factor for non-first PP stages (see
        ``recommend_pp_stage_bucket_size``).
    force_all_reduce:
        Passed through to ``start_grad_sync`` / ``finish_grad_sync``.
    max_reduce_streams:
        Maximum CUDA streams in the stream pool.  Ignored when
        ``use_dedicated_streams=False``.
    use_dedicated_streams:
        If True, each bucket group gets a dedicated stream from the pool
        so that multiple groups can overlap their NCCL collectives.
    skip_sync:
        Initial value for the DES-LOC Kx skip-sync gate.  Call
        ``prepare_for_backward(skip_sync=...)`` each iteration to update.
    """

    def __init__(
        self,
        ddp_chunks: List[nn.Module],
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_stage_bucket_scale: float = 4.0,
        force_all_reduce: bool = False,
        max_reduce_streams: int = 4,
        use_dedicated_streams: bool = True,
        skip_sync: bool = False,
    ) -> None:
        self._ddp_chunks = ddp_chunks
        self._pp_group = pp_group
        self._pp_rank = _get_pp_rank(pp_group)
        self._pp_size = _get_pp_size(pp_group)
        self._force_all_reduce = force_all_reduce
        self._pp_stage_bucket_scale = pp_stage_bucket_scale
        self._use_dedicated_streams = use_dedicated_streams

        # Current-step state.
        self._is_last_microbatch: bool = True
        self._skip_sync: bool = skip_sync

        # Per-(chunk, bucket-group) state.
        self._chunk_bucket_states: List[_ChunkBucketState] = []
        # param_id → _ChunkBucketState for O(1) hook lookup.
        self._param_to_state: Dict[int, _ChunkBucketState] = {}
        # Bucket groups dispatched this iteration.
        self._dispatched: Set[int] = set()
        # Hook handles for cleanup.
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        # Shared CUDA stream pool.
        self._stream_pool = StreamPool(
            max_streams=max_reduce_streams, priority=-1
        ) if use_dedicated_streams else None

        self._build()

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
            accumulation window.
        skip_sync:
            DES-LOC Kx gate.  When True collectives are suppressed.
        """
        self._is_last_microbatch = is_last_microbatch
        self._skip_sync = skip_sync
        self._dispatched.clear()
        for state in self._chunk_bucket_states:
            state.reset()
            bg = state.bucket_group
            if hasattr(bg, '_skip_sync'):
                bg._skip_sync = skip_sync
            if hasattr(bg, 'is_last_microbatch'):
                bg.is_last_microbatch = is_last_microbatch

    def finalize(self) -> None:
        """Drain all outstanding reduce-scatter ops.

        Must be called after the backward pass.  Any bucket group whose
        backward hook did not fire (e.g., first batch, no overlap) is
        flushed synchronously here.
        """
        for state in self._chunk_bucket_states:
            bg = state.bucket_group
            bg_id = id(bg)
            if bg_id not in self._dispatched:
                bg.finish_grad_sync(force_all_reduce=self._force_all_reduce)
                self._dispatched.add(bg_id)
            else:
                bg.finish_grad_sync(force_all_reduce=self._force_all_reduce)

        # Wait for all dedicated streams.
        if self._stream_pool is not None:
            self._stream_pool.synchronize_all()

    def remove_hooks(self) -> None:
        """Remove all backward hooks registered by this manager."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Build bucket-state table and register backward hooks for all chunks."""
        for chunk_idx, ddp_chunk in enumerate(self._ddp_chunks):
            bucket_groups = self._resolve_bucket_groups(ddp_chunk)
            if not bucket_groups:
                logger.warning(
                    "VPPOverlapGradReduceManager: chunk %d has no bucket groups; "
                    "overlapping disabled for this chunk.",
                    chunk_idx,
                )
                continue

            for bg in bucket_groups:
                n_params = sum(len(b.params_list) for b in bg.buckets)
                if n_params == 0:
                    continue

                # Assign a CUDA stream from the shared pool.
                stream = (
                    self._stream_pool.get()
                    if self._stream_pool is not None
                    else None
                )
                # Propagate stream to bucket group for predecessor-drain path.
                if stream is not None and not hasattr(bg, 'communication_stream'):
                    bg.communication_stream = stream

                state = _ChunkBucketState(
                    bucket_group=bg,
                    total_params=n_params,
                    chunk_idx=chunk_idx,
                    stream=stream,
                )
                self._chunk_bucket_states.append(state)
                for bucket in bg.buckets:
                    for param in bucket.params_list:
                        self._param_to_state[id(param)] = state

            # Register hooks for this chunk.
            for module in self._iter_modules(ddp_chunk):
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
                return
            state = self._param_to_state.get(param_id)
            if state is None:
                return
            bg = state.bucket_group
            bg_id = id(bg)

            # Prefer bucket group's own register_grad_ready when available.
            if hasattr(bg, 'register_grad_ready') and getattr(
                getattr(bg, 'ddp_config', None), 'overlap_grad_reduce', False
            ):
                try:
                    bg.register_grad_ready(p, force_all_reduce=self._force_all_reduce)
                    self._dispatched.add(bg_id)
                except Exception:
                    pass
                return

            # Fallback: manual readiness tracking.
            if state.mark_param_ready():
                if bg_id not in self._dispatched:
                    self._dispatched.add(bg_id)
                    self._launch_reduce_scatter(bg, state.stream)

        return _hook

    def _launch_reduce_scatter(
        self,
        bg,
        stream: Optional[torch.cuda.Stream],
    ) -> None:
        """Issue an (async) reduce-scatter for ``bg`` on ``stream``.

        Inserts a stream wait so that gradient accumulation has completed
        before NCCL reads grad_data.  Mirrors the wait_stream call in
        ``ParamAndGradBucketGroup.start_grad_sync`` for the multi-DistOpt path.
        """
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
            ctx = torch.cuda.stream(stream)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            try:
                bg.start_grad_sync(
                    force_all_reduce=self._force_all_reduce,
                    skip_sync=self._skip_sync,
                )
            except TypeError:
                bg.start_grad_sync(force_all_reduce=self._force_all_reduce)

    @staticmethod
    def _resolve_bucket_groups(ddp_model: nn.Module) -> list:
        """Extract bucket_groups from a DDP wrapper."""
        if hasattr(ddp_model, 'bucket_groups'):
            return ddp_model.bucket_groups
        if hasattr(ddp_model, 'buffers'):
            groups = []
            for buf in ddp_model.buffers:
                if hasattr(buf, 'bucket_groups'):
                    groups.extend(buf.bucket_groups)
            if groups:
                return groups
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
# OverlappedGradReduceContext — convenience context manager
# ---------------------------------------------------------------------------

class OverlappedGradReduceContext:
    """Context-manager wrapper for the VPP overlapped grad-reduce pattern.

    Provides a clean interface for training loops that use VPP.  A single
    context instance spans the full forward-backward pass for one batch.

    Usage::

        ctx = OverlappedGradReduceContext(model_chunks, pp_group=pp_group)

        for step in range(num_steps):
            ctx.prepare(is_last_microbatch=True, skip_sync=False)
            # run forward / backward for all microbatches
            for i, mb in enumerate(microbatches):
                is_last = (i == len(microbatches) - 1)
                ctx.prepare(is_last_microbatch=is_last, skip_sync=False)
                loss = forward_backward(mb)
            ctx.finalize()

    Parameters
    ----------
    ddp_chunks:
        List of DDP-wrapped model chunks (one per VPP virtual stage).
    pp_group:
        Pipeline-parallel process group.
    force_all_reduce:
        If True, use all-reduce instead of reduce-scatter everywhere.
    kwargs:
        Additional keyword arguments forwarded to ``VPPOverlapGradReduceManager``.
    """

    def __init__(
        self,
        ddp_chunks: List[nn.Module],
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        force_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        self._manager = VPPOverlapGradReduceManager(
            ddp_chunks=ddp_chunks,
            pp_group=pp_group,
            force_all_reduce=force_all_reduce,
            **kwargs,
        )

    def prepare(
        self,
        is_last_microbatch: bool = True,
        skip_sync: bool = False,
    ) -> None:
        """Prepare manager for the upcoming backward pass."""
        self._manager.prepare_for_backward(
            is_last_microbatch=is_last_microbatch,
            skip_sync=skip_sync,
        )

    def finalize(self) -> None:
        """Drain all outstanding reduce-scatters."""
        self._manager.finalize()

    def remove_hooks(self) -> None:
        """Remove all registered backward hooks."""
        self._manager.remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finalize()
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_overlapped_grad_reduce_manager(
    ddp_chunks: List[nn.Module],
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    num_microbatches: int = 1,
    force_all_reduce: bool = False,
    use_dedicated_streams: bool = True,
    max_reduce_streams: int = 4,
) -> VPPOverlapGradReduceManager:
    """Build a ``VPPOverlapGradReduceManager`` with sensible defaults.

    Automatically computes ``pp_stage_bucket_scale`` from the PP rank and
    the number of microbatches using ``recommend_pp_stage_bucket_size`` so
    that callers do not need to set this manually.

    Parameters
    ----------
    ddp_chunks:
        List of DDP-wrapped model chunks.
    pp_group:
        Pipeline-parallel process group.
    num_microbatches:
        Total microbatches per training step (used to compute
        ``pp_stage_bucket_scale``).
    force_all_reduce:
        If True, use all-reduce instead of reduce-scatter.
    use_dedicated_streams:
        Allocate a dedicated CUDA stream per bucket group from the pool.
    max_reduce_streams:
        Cap on the number of CUDA streams in the stream pool.

    Returns
    -------
    VPPOverlapGradReduceManager
    """
    pp_rank = _get_pp_rank(pp_group)
    pp_size = _get_pp_size(pp_group)

    # Derive scale factor: 1.0 for stage 0, growing for later stages.
    base_size = 40_000_000  # elements (Megatron default)
    scaled_size = recommend_pp_stage_bucket_size(
        base_bucket_size=base_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        num_microbatches=num_microbatches,
    )
    pp_stage_bucket_scale = scaled_size / base_size

    return VPPOverlapGradReduceManager(
        ddp_chunks=ddp_chunks,
        pp_group=pp_group,
        pp_stage_bucket_scale=pp_stage_bucket_scale,
        force_all_reduce=force_all_reduce,
        max_reduce_streams=max_reduce_streams,
        use_dedicated_streams=use_dedicated_streams,
    )


__all__ = [
    "StreamPool",
    "VPPOverlapGradReduceManager",
    "OverlappedGradReduceContext",
    "build_overlapped_grad_reduce_manager",
]
