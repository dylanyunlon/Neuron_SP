# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""GradBuffer — legacy grad-only buffer view for distributed optimizer compatibility.

Background
----------
Megatron-LM historically exposed ``model.grad_buffers`` as a ``Dict[dtype, GradBuffer]``
where ``GradBuffer`` was a thin wrapper around a flat ``data`` tensor plus a
``param_index_map`` for locating per-parameter slices.  The distributed optimizer
(``compile/megatron_optimizer.py``) and the runtime engine
(``runtime/engine.py :: DeslocDistributedOptimizerShardManager``) rely on this API
to compute shard ranges.

In DES-LOC the contiguous buffer is owned by ``ParamAndGradBuffer`` (which holds both
param and grad data in the same class).  ``GradBuffer`` is therefore a *view* over
that buffer rather than an independent allocation.

Key commit references (Megatron-LM)
-------------------------------------
  M1820 (4feb2b0d): Public rename ``_grad_buffers`` → ``grad_buffers``.
  M1835 (54b41689): Fix param_index_map coordinate order for DistributedOptimizer.
  M2278 (1e9e94cc): Initial contiguous GradBuffer allocator for DDP.
  M3238 (a3ec4b02): 64-element alignment at param start.
  M3811 (55b8111ad): Extract param layout computation from DDP __init__.

DES-LOC extensions
------------------
  - ``GradBuffer.from_param_and_grad_buffer()``: factory that wraps an
    existing ``ParamAndGradBuffer`` so the old ``model.grad_buffers[dtype]``
    access pattern keeps working without re-allocation.
  - ``GradBufferRegistry``: helper attached to ``DistributedDataParallel``
    that maintains the ``{grad_dtype: GradBuffer}`` mapping expected by
    ``compile/megatron_optimizer.py``.
  - ``build_grad_buffer_registry()``: build the registry from a list of
    ``ParamAndGradBuffer`` objects.

Public API
----------
  GradBuffer
  GradBufferRegistry
  build_grad_buffer_registry
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GradBuffer
# ---------------------------------------------------------------------------

class GradBuffer:
    """A grad-only view over a contiguous flat tensor.

    Wraps a ``ParamAndGradBuffer``'s ``grad_data`` tensor and exposes the
    interface that ``compile/megatron_optimizer.py`` and
    ``runtime/engine.py`` depend on:

      - ``self.data``          — the flat grad tensor (1-D, ``grad_dtype``).
      - ``self.numel``         — total padded element count.
      - ``self.numel_unpadded`` — unpadded element count.
      - ``self.param_index_map`` — ``{param: (start, end, bucket_id)}`` mapping
                                   using *full-numel* (unpacked) offsets, consistent
                                   with Megatron M3238 + M3781 conventions.
      - ``self.dtype``         — grad dtype of this buffer.
      - ``self.buckets``       — ordered list of ``ParamAndGradBucket`` objects.
      - ``self.data_parallel_group`` — DP process group.

    Notes
    -----
    The ``data`` tensor is a *view* into ``ParamAndGradBuffer.grad_data``.
    Modifications to ``data`` affect the live training buffer.  Callers that
    need to zero the grad buffer should call ``reset()`` (which delegates to
    the parent buffer).
    """

    def __init__(
        self,
        data: torch.Tensor,
        numel: int,
        numel_unpadded: int,
        param_index_map: Dict[nn.Parameter, Tuple[int, int, int]],
        buckets,
        data_parallel_group: torch.distributed.ProcessGroup,
        dtype: torch.dtype,
    ) -> None:
        self.data = data
        self.numel = numel
        self.numel_unpadded = numel_unpadded
        self.param_index_map = param_index_map
        self.buckets = buckets
        self.data_parallel_group = data_parallel_group
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_param_and_grad_buffer(cls, buf) -> "GradBuffer":
        """Create a GradBuffer view from a ParamAndGradBuffer.

        Args:
            buf: A ``ParamAndGradBuffer`` instance.

        Returns:
            A ``GradBuffer`` wrapping ``buf.grad_data``.
        """
        return cls(
            data=buf.grad_data,
            numel=buf.numel,
            numel_unpadded=buf.numel_unpadded,
            param_index_map=buf.param_index_map,
            buckets=buf.buckets,
            data_parallel_group=buf.data_parallel_group,
            dtype=buf.grad_dtype,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero the grad buffer data in place."""
        self.data.zero_()

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all gradient data by *scaling_factor*."""
        self.data *= scaling_factor

    def get_tensor_for_param(self, param: nn.Parameter) -> torch.Tensor:
        """Return a contiguous view of the grad slice for *param*.

        Args:
            param: A model parameter that must exist in ``param_index_map``.

        Returns:
            A 1-D view into ``self.data`` covering the param's gradient region.

        Raises:
            KeyError: if *param* is not registered in this buffer.
        """
        start, end, _ = self.param_index_map[param]
        return self.data[start:end]

    def __repr__(self) -> str:
        return (
            f"GradBuffer(dtype={self.dtype}, numel={self.numel}, "
            f"numel_unpadded={self.numel_unpadded}, "
            f"num_params={len(self.param_index_map)}, "
            f"num_buckets={len(self.buckets)})"
        )


# ---------------------------------------------------------------------------
# GradBufferRegistry
# ---------------------------------------------------------------------------

class GradBufferRegistry:
    """Dict-like mapping from grad_dtype to GradBuffer.

    Provides the ``model.grad_buffers[dtype]`` and
    ``model.grad_buffer_param_index_map[dtype]`` access patterns that
    the Megatron distributed optimizer expects (M1820 public rename).

    The registry is constructed by ``build_grad_buffer_registry()`` and
    attached to ``DistributedDataParallel`` during ``__init__``.

    Attributes
    ----------
    _registry : Dict[torch.dtype, GradBuffer]
        Maps each unique ``grad_dtype`` to its aggregated GradBuffer.
        When multiple ``ParamAndGradBuffer`` objects share the same
        ``grad_dtype`` (e.g. expert-parallel + non-expert buffers both
        using FP32 grads), they are merged into a single GradBuffer entry
        with a unified ``param_index_map``.
    """

    def __init__(self) -> None:
        self._registry: Dict[torch.dtype, GradBuffer] = {}

    # ------------------------------------------------------------------
    # Dict-like protocol
    # ------------------------------------------------------------------

    def __getitem__(self, dtype: torch.dtype) -> GradBuffer:
        return self._registry[dtype]

    def __contains__(self, dtype: object) -> bool:
        return dtype in self._registry

    def __iter__(self) -> Iterator[torch.dtype]:
        return iter(self._registry)

    def __len__(self) -> int:
        return len(self._registry)

    def items(self):
        return self._registry.items()

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    # ------------------------------------------------------------------
    # param_index_map accessor
    # ------------------------------------------------------------------

    @property
    def param_index_map(self) -> Dict[torch.dtype, Dict[nn.Parameter, Tuple[int, int, int]]]:
        """Return ``{grad_dtype: {param: (start, end, bucket_id)}}`` mapping.

        Mirrors ``model.grad_buffer_param_index_map`` from Megatron M1820.
        """
        return {dtype: gbuf.param_index_map for dtype, gbuf in self._registry.items()}

    # ------------------------------------------------------------------
    # Internal mutation (used by build_grad_buffer_registry)
    # ------------------------------------------------------------------

    def _register(self, dtype: torch.dtype, grad_buffer: GradBuffer) -> None:
        """Register a GradBuffer under *dtype*."""
        self._registry[dtype] = grad_buffer

    def __repr__(self) -> str:
        dtypes = list(self._registry.keys())
        return f"GradBufferRegistry(dtypes={dtypes})"


# ---------------------------------------------------------------------------
# build_grad_buffer_registry
# ---------------------------------------------------------------------------

def build_grad_buffer_registry(
    buffers: List,
    expert_parallel_buffers: Optional[List] = None,
) -> GradBufferRegistry:
    """Build a GradBufferRegistry from ParamAndGradBuffer lists.

    Called once from ``DistributedDataParallel.__init__`` after all buffers
    have been allocated.  Provides the backward-compatible
    ``model.grad_buffers[dtype]`` interface needed by the distributed
    optimizer sharding logic in ``compile/megatron_optimizer.py``.

    When multiple buffers share the same ``grad_dtype`` (e.g., an expert-
    parallel FP32 buffer and a non-expert FP32 buffer), their param_index_maps
    are *merged* into a single GradBuffer with a combined param_index_map.
    The ``data`` tensor of the *first* encountered buffer is stored; callers
    that need all raw grad tensors should iterate ``DDP.buffers`` directly.

    Args:
        buffers: List of ``ParamAndGradBuffer`` (non-expert-parallel).
        expert_parallel_buffers: List of ``ParamAndGradBuffer`` for expert
            params (may be None or empty).

    Returns:
        A populated ``GradBufferRegistry``.

    Notes
    -----
    Megatron M1835: param_index_map coordinates are in *reverse* allocation
    order (last param in forward order = first in index map).  This is
    already correct because ``_compute_default_per_buffer_param_layout``
    iterates ``params[::-1]``.  No additional reordering is needed here.
    """
    all_bufs = list(buffers or []) + list(expert_parallel_buffers or [])

    # dtype → (grad_data_list, numel, numel_unpadded, combined_param_index_map, buckets, dp_group)
    dtype_to_accum: Dict[torch.dtype, dict] = {}

    for buf in all_bufs:
        gd = buf.grad_dtype
        if gd not in dtype_to_accum:
            dtype_to_accum[gd] = {
                "data": buf.grad_data,
                "numel": buf.numel,
                "numel_unpadded": buf.numel_unpadded,
                "param_index_map": dict(buf.param_index_map),
                "buckets": list(buf.buckets),
                "dp_group": buf.data_parallel_group,
            }
        else:
            # Merge subsequent buffers with the same grad_dtype.
            accum = dtype_to_accum[gd]
            accum["numel"] += buf.numel
            accum["numel_unpadded"] += buf.numel_unpadded
            accum["param_index_map"].update(buf.param_index_map)
            accum["buckets"].extend(buf.buckets)
            # Keep the dp_group from the first buffer (all should be equal).

    registry = GradBufferRegistry()
    for dtype, accum in dtype_to_accum.items():
        gbuf = GradBuffer(
            data=accum["data"],
            numel=accum["numel"],
            numel_unpadded=accum["numel_unpadded"],
            param_index_map=accum["param_index_map"],
            buckets=accum["buckets"],
            data_parallel_group=accum["dp_group"],
            dtype=dtype,
        )
        registry._register(dtype, gbuf)
        logger.debug(
            "GradBufferRegistry: registered grad_dtype=%s, numel=%d, params=%d",
            dtype,
            accum["numel"],
            len(accum["param_index_map"]),
        )

    return registry
