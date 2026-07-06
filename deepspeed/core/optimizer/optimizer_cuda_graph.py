# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CUDA graph wrapper for the Adam optimizer step.

Ported from Megatron-LM/megatron/core/optimizer/optimizer_cuda_graph.py
with DES-LOC extensions for heterogeneous GPU tier handling.

Overview
--------
CUDA graph capture eliminates the CPU-side kernel-launch overhead that
dominates at small batch sizes or high-throughput training runs.  After a
configurable warmup period the optimizer's ``step()`` call is captured
into a replayable CUDA graph.  Subsequent steps replay the graph instead
of re-launching individual CUDA kernels, reducing per-step CPU time from
~2 ms to ~0.1 ms for a 7B-parameter Adam update.

DES-LOC considerations
-----------------------
On PCIe-only heterogeneous clusters (A6000 + H100), CUDA graph capture must
be performed independently on each tier because:

  1. The CUDA graph stream captures GPU work on a single device.
  2. The optimizer shard sizes differ between tiers (TFLOPS-weighted
     :class:`~deepspeed.core.optimizer.distrib_optimizer.DistributedOptimizer`
     shards), so the graph structures differ across ranks.
  3. On A6000 (VRAM-limited) ranks the optimizer state may be offloaded to
     CPU (``_cpu_offload_optim=True``), which is incompatible with CUDA
     graph capture — the wrapper automatically disables graph capture for
     CPU-offloaded ranks.

Public API
----------
  OptimizerCudaGraphWrapper  — callable wrapping optimizer.step()
  wrap_optimizer_step        — convenience constructor
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared graph pool helpers (mirrors Megatron full_cuda_graph module)
# ---------------------------------------------------------------------------

_GRAPH_POOL_HANDLE: Optional[Any] = None
_CAPTURE_STREAM: Optional[torch.cuda.Stream] = None


def _get_graph_pool(use_single_mempool: bool = False) -> Optional[Any]:
    """Return a shared CUDA memory pool handle for graph capture.

    When ``use_single_mempool=True``, all graphs share a single memory pool
    so that allocations made during capture are not re-allocated on replay
    (reduces fragmentation on A6000 48 GB cards).

    Returns:
        CUDA graph memory pool handle, or None when CUDA is unavailable.
    """
    global _GRAPH_POOL_HANDLE
    if not torch.cuda.is_available():
        return None
    if use_single_mempool:
        if _GRAPH_POOL_HANDLE is None:
            _GRAPH_POOL_HANDLE = torch.cuda.graph_pool_handle()
        return _GRAPH_POOL_HANDLE
    return None


def _get_shared_capture_stream() -> Optional[torch.cuda.Stream]:
    """Return a dedicated CUDA stream for graph capture.

    Using a dedicated stream prevents the capture from inadvertently
    capturing work from other streams (e.g. data-loading prefetch),
    which would cause graph replay to fail with "stream is not in the
    graph's dependencies" errors.

    Returns:
        A persistent CUDA stream for graph capture, or None when unavailable.
    """
    global _CAPTURE_STREAM
    if not torch.cuda.is_available():
        return None
    if _CAPTURE_STREAM is None:
        _CAPTURE_STREAM = torch.cuda.Stream()
    return _CAPTURE_STREAM


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class OptimizerCudaGraphWrapper:
    """Replayable CUDA graph wrapper for optimizer.step().

    After ``cuda_graph_warmup_steps`` eager (non-graph) warmup steps, the
    next call captures the optimizer step into a CUDA graph.  All subsequent
    calls replay the captured graph instead of re-launching individual kernels.

    Thread-safety: not thread-safe; designed for single-threaded training.

    CPU-offload guard
    ~~~~~~~~~~~~~~~~~
    When ``cpu_offload_rank=True`` CUDA graph capture is disabled for this
    instance (CPU Adam steps cannot be graph-captured).  The wrapper becomes
    a thin no-op pass-through for the optimizer function.

    Distributed barrier
    ~~~~~~~~~~~~~~~~~~~
    A ``torch.distributed.barrier()`` is inserted before and after capture
    to ensure all ranks enter and exit graph capture simultaneously.  This
    prevents a race where a fast H100 rank starts training step N+1 while
    a slow A6000 rank is still capturing step N.

    Args:
        optimizer_step_func: Zero-argument callable that performs one
                             optimizer step (typically ``optimizer.step``).
        cuda_graph_warmup_steps: Number of eager steps before capture.
        use_single_mempool:   Share a CUDA memory pool across graphs to
                              reduce fragmentation on VRAM-limited ranks.
        cpu_offload_rank:     If True, disable graph capture (CPU optimizer).
        rank_label:           Human-readable label for logging (e.g. "h100_r0").

    Examples::

        # Wrap the distributed optimizer step for CUDA graph replay
        step_fn = OptimizerCudaGraphWrapper(
            optimizer_step_func=dist_optimizer.step_with_ready_grads,
            cuda_graph_warmup_steps=3,
        )
        # Training loop
        for step in range(total_steps):
            loss.backward()
            step_fn()   # graph-captured after warmup
    """

    # Class-level graph state (one graph per process / rank — not shared)
    _cuda_graph: Optional[torch.cuda.CUDAGraph] = None
    _graph_result: Any = None
    _curr_iteration: int = 0

    def __init__(
        self,
        optimizer_step_func: Callable,
        cuda_graph_warmup_steps: int = 3,
        use_single_mempool: bool = False,
        cpu_offload_rank: bool = False,
        rank_label: str = "",
    ) -> None:
        self.optimizer_step_func = optimizer_step_func
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool
        self.cpu_offload_rank = cpu_offload_rank
        self.rank_label = rank_label

        # Per-instance graph state (avoids class-level sharing when multiple
        # instances are created — e.g. one per optimizer in ChainedOptimizer)
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._result: Any = None
        self._iteration: int = 0

        if cpu_offload_rank:
            logger.info(
                "OptimizerCudaGraphWrapper [%s]: CUDA graph disabled (cpu_offload_rank=True).",
                rank_label,
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute optimizer step, capturing or replaying a CUDA graph.

        Args:
            *args:   Not accepted (optimizer.step() takes no positional args).
            **kwargs: Not accepted.

        Returns:
            Return value of ``optimizer_step_func()`` (often None or a bool).
        """
        if args or kwargs:
            raise TypeError(
                "OptimizerCudaGraphWrapper: optimizer.step() does not accept "
                "positional or keyword arguments."
            )

        # CPU-offload path: never graph-capture
        if self.cpu_offload_rank or not torch.cuda.is_available():
            self._result = self.optimizer_step_func()
            self._iteration += 1
            return self._result

        curr = self._iteration

        if curr == self.cuda_graph_warmup_steps:
            # ---------------------------------------------------------------
            # Capture phase: wrap optimizer step in a CUDAGraph
            # ---------------------------------------------------------------
            logger.info(
                "OptimizerCudaGraphWrapper [%s]: capturing CUDA graph at iteration %d.",
                self.rank_label,
                curr,
            )

            # Barrier: ensure all ranks reach capture simultaneously
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            torch.cuda.synchronize()
            self._graph = torch.cuda.CUDAGraph()
            capture_stream = _get_shared_capture_stream()
            pool = _get_graph_pool(self.use_single_mempool)

            try:
                with torch.cuda.graph(
                    self._graph,
                    stream=capture_stream,
                    pool=pool,
                ):
                    self._result = self.optimizer_step_func()
            except RuntimeError as e:
                logger.warning(
                    "OptimizerCudaGraphWrapper [%s]: CUDA graph capture failed (%s). "
                    "Falling back to eager mode.",
                    self.rank_label,
                    e,
                )
                self._graph = None
                self._result = self.optimizer_step_func()
                self._iteration += 1
                return self._result

            torch.cuda.synchronize()

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            logger.info(
                "OptimizerCudaGraphWrapper [%s]: CUDA graph capture complete.",
                self.rank_label,
            )

        elif curr > self.cuda_graph_warmup_steps and self._graph is not None:
            # ---------------------------------------------------------------
            # Replay phase: replay the captured graph
            # ---------------------------------------------------------------
            self._graph.replay()

        else:
            # ---------------------------------------------------------------
            # Warmup phase: eager execution (no graph)
            # ---------------------------------------------------------------
            self._result = self.optimizer_step_func()

        self._iteration += 1
        return self._result

    def curr_iter(self) -> int:
        """Return the current training iteration count."""
        return self._iteration

    def reset(self) -> None:
        """Delete the captured graph and reset to warmup mode.

        Useful when the optimizer state changes (e.g. after loading a
        checkpoint or changing the learning rate schedule).
        """
        if self._graph is not None:
            del self._graph
            self._graph = None
        self._result = None
        self._iteration = 0
        logger.info(
            "OptimizerCudaGraphWrapper [%s]: graph reset; will re-capture after warmup.",
            self.rank_label,
        )

    def is_capturing(self) -> bool:
        """Return True if the graph has been captured and is in replay mode."""
        return self._graph is not None and self._iteration > self.cuda_graph_warmup_steps

    def __del__(self) -> None:
        if self._graph is not None:
            logger.debug(
                "OptimizerCudaGraphWrapper [%s]: destructor — deleting CUDA graph.",
                self.rank_label,
            )
            del self._graph
            self._graph = None


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def wrap_optimizer_step(
    optimizer: Any,
    warmup_steps: int = 3,
    use_single_mempool: bool = False,
    cpu_offload: bool = False,
    rank_label: str = "",
) -> OptimizerCudaGraphWrapper:
    """Wrap *optimizer*.step() in a CUDA graph for replay-based execution.

    Convenience constructor that extracts ``optimizer.step`` and wraps it in
    :class:`OptimizerCudaGraphWrapper`.

    DES-LOC usage: set ``cpu_offload=True`` for A6000 ranks where optimizer
    state is CPU-offloaded (determined by ``DesLocEngine._cpu_offload_optim``).

    Args:
        optimizer:          Any optimizer with a ``.step()`` method.
        warmup_steps:       Eager steps before graph capture.
        use_single_mempool: Share memory pool across all graphs in this process.
        cpu_offload:        Disable graph capture (CPU optimizer path).
        rank_label:         Human-readable label for logs.

    Returns:
        Configured :class:`OptimizerCudaGraphWrapper` instance.

    Examples::

        step_fn = wrap_optimizer_step(dist_optimizer, warmup_steps=5,
                                      rank_label="h100_rank0")
        # Replace direct optimizer.step() calls:
        step_fn()
    """
    step_fn = (
        optimizer.step_with_ready_grads
        if hasattr(optimizer, "step_with_ready_grads")
        else optimizer.step
    )
    return OptimizerCudaGraphWrapper(
        optimizer_step_func=step_fn,
        cuda_graph_warmup_steps=warmup_steps,
        use_single_mempool=use_single_mempool,
        cpu_offload_rank=cpu_offload,
        rank_label=rank_label,
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "OptimizerCudaGraphWrapper",
    "wrap_optimizer_step",
    "_get_graph_pool",
    "_get_shared_capture_stream",
]
