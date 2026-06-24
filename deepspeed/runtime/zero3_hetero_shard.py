"""
ZeRO-3 Heterogeneous Parameter Sharding utilities for DesLocEngine.

Implements per-rank flat parameter sharding such that each rank holds only
``1/N`` of the total flattened parameter buffer (or a VRAM-proportional
fraction in the heterogeneous case), and provides on-demand all-gather
and reduce-scatter primitives for forward / backward.

Design notes
------------
* The model's parameters are conceptually concatenated into a single 1-D
  buffer of dtype FP32 (the master copy used for optimizer math). Each
  rank keeps a contiguous slice of that buffer in ``self.param_shard``.

* ``param_offsets`` records, for every original parameter, the
  ``(global_start, global_end, shape)`` triple inside the flat buffer so
  that we can scatter / gather back into the right ``nn.Parameter``.

* ``gather_full_params(module)`` is a context manager that materializes
  the full FP32 → BF16 buffer via ``dist.all_gather_into_tensor`` and
  rewrites the storage of every parameter to view into that buffer for
  the duration of the ``with`` block. On exit the storage is released.

* ``scatter_grads(model)`` reduces gradients across all ranks with
  ``dist.reduce_scatter_tensor`` and keeps only this rank's local slice,
  matching the layout of ``param_shard``.

* Heterogeneous shard sizing: when ``vram_weights`` is supplied the flat
  buffer is partitioned proportionally (an H100 with 96 GB receives
  twice the share of an A6000 with 48 GB, etc.). Otherwise the buffer
  is partitioned evenly.

* Backwards compatible: ``ShardState.build`` returns ``None`` for
  ``world_size == 1`` and callers must treat that as "no sharding".
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ParamSlice:
    """Locator for an ``nn.Parameter`` inside the flat parameter buffer."""
    name: str
    global_start: int
    global_end: int            # exclusive
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class ShardState:
    """
    Container for ZeRO-3 sharding state attached to the engine.

    Attributes:
        rank: This process's rank.
        world_size: Total number of ranks.
        total_numel: Total number of params across the whole model
                     (after any per-rank padding alignment).
        shard_sizes: List of length ``world_size``; ``shard_sizes[r]`` is
                     the number of FP32 elements held by rank ``r``.
        shard_offsets: Cumulative offsets ``[0, shard_sizes[0],
                       shard_sizes[0]+shard_sizes[1], ...]``.
        param_shard: This rank's local FP32 master shard (1-D tensor of
                     size ``shard_sizes[rank]``) living on CUDA.
        param_offsets: Mapping ``param-name -> ParamSlice``.
        param_order: Ordered list of (name, parameter) tuples used to
                     re-materialize the flat buffer.
        pad: Number of zero-padding elements appended to make
             ``total_numel`` evenly divisible across ranks when using
             ``all_gather_into_tensor`` (which requires equal shard
             sizes). For heterogeneous splits this is also non-zero so
             that the flat buffer length is the sum of ``shard_sizes``.
    """
    rank: int
    world_size: int
    total_numel: int
    shard_sizes: List[int]
    shard_offsets: List[int]
    param_shard: torch.Tensor
    param_offsets: Dict[str, ParamSlice]
    param_order: List[Tuple[str, nn.Parameter]] = field(default_factory=list)
    pad: int = 0

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        model: nn.Module,
        rank: int,
        world_size: int,
        device: torch.device,
        vram_weights: Optional[Sequence[float]] = None,
    ) -> Optional["ShardState"]:
        """
        Build the sharding plan for ``model``.

        Returns ``None`` when ``world_size <= 1`` (no sharding needed —
        backward-compatible single-GPU path).

        Args:
            model:        Module whose ``.parameters()`` will be sharded.
            rank:         This process's rank in [0, world_size).
            world_size:   Total number of ranks.
            device:       Local CUDA device.
            vram_weights: Optional per-rank positive weights. When given
                          the flat buffer is partitioned proportionally
                          (heterogeneous case: H100 gets a larger share).
                          Must have length ``world_size``.
        """
        if world_size is None or world_size <= 1:
            return None

        # ---- 1. Enumerate params in a deterministic order. ----------
        param_order: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]
        if not param_order:
            return None

        # ---- 2. Compute shard sizes. --------------------------------
        raw_total = sum(p.numel() for _, p in param_order)

        if vram_weights is not None:
            if len(vram_weights) != world_size:
                raise ValueError(
                    f"vram_weights length ({len(vram_weights)}) "
                    f"!= world_size ({world_size})"
                )
            if any(w <= 0 for w in vram_weights):
                raise ValueError("vram_weights must be strictly positive")
            wsum = float(sum(vram_weights))
            # First (N-1) ranks take floor(weight * total / sum); last
            # rank absorbs the remainder so the sum matches exactly.
            shard_sizes: List[int] = []
            assigned = 0
            for r in range(world_size - 1):
                size_r = int((vram_weights[r] / wsum) * raw_total)
                shard_sizes.append(size_r)
                assigned += size_r
            shard_sizes.append(max(0, raw_total - assigned))
            total_numel = raw_total
            pad = 0
        else:
            # Even split: pad up to a multiple of world_size so that
            # all_gather_into_tensor (equal shards) works cleanly.
            shard_size = (raw_total + world_size - 1) // world_size
            total_numel = shard_size * world_size
            pad = total_numel - raw_total
            shard_sizes = [shard_size] * world_size

        # Cumulative offsets — useful both for indexing and for
        # interpreting the layout after all-gather.
        shard_offsets: List[int] = [0]
        for s in shard_sizes:
            shard_offsets.append(shard_offsets[-1] + s)
        assert shard_offsets[-1] == total_numel, (
            f"shard layout mismatch: {shard_offsets[-1]} vs {total_numel}"
        )

        # ---- 3. Build param_offsets (flat layout map). --------------
        param_offsets: Dict[str, ParamSlice] = {}
        cursor = 0
        for name, p in param_order:
            n = p.numel()
            param_offsets[name] = ParamSlice(
                name=name,
                global_start=cursor,
                global_end=cursor + n,
                shape=p.shape,
                dtype=p.dtype,
            )
            cursor += n
        assert cursor == raw_total

        # ---- 4. Initialise this rank's FP32 master shard. -----------
        lo = shard_offsets[rank]
        hi = shard_offsets[rank + 1]
        local_size = hi - lo

        param_shard = torch.zeros(local_size, dtype=torch.float32, device=device)

        # Fill the slice by copying the relevant region of each param.
        # We walk the flat layout and intersect with [lo, hi).
        for name, p in param_order:
            sl = param_offsets[name]
            # Skip params entirely outside this rank's window.
            if sl.global_end <= lo or sl.global_start >= hi:
                continue
            # Intersection in global coordinates.
            g_start = max(sl.global_start, lo)
            g_end   = min(sl.global_end, hi)
            # Indices into the param's own flat view.
            p_start = g_start - sl.global_start
            p_end   = g_end   - sl.global_start
            # Indices into the local shard buffer.
            s_start = g_start - lo
            s_end   = g_end   - lo
            with torch.no_grad():
                flat = p.detach().reshape(-1).to(torch.float32)
                param_shard[s_start:s_end].copy_(flat[p_start:p_end])

        logger.info(
            "[zero3] rank=%d/%d shard=%d elems (of %d total, pad=%d, hetero=%s)",
            rank, world_size, local_size, total_numel, pad,
            vram_weights is not None,
        )

        return cls(
            rank=rank,
            world_size=world_size,
            total_numel=total_numel,
            shard_sizes=shard_sizes,
            shard_offsets=shard_offsets,
            param_shard=param_shard,
            param_offsets=param_offsets,
            param_order=param_order,
            pad=pad,
        )

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------
    def _build_full_buffer(self, dtype: torch.dtype) -> torch.Tensor:
        """
        All-gather all rank shards into a single contiguous flat buffer.

        For even splits we can use ``dist.all_gather_into_tensor``
        directly. For heterogeneous (variable-size) splits we fall back
        to ``dist.all_gather`` into per-rank tensors and concat.
        """
        device = self.param_shard.device
        full = torch.empty(self.total_numel, dtype=dtype, device=device)

        # Cast our shard to the target gather dtype (typically BF16 for
        # forward, FP32 for diagnostic / debug).
        local = self.param_shard.to(dtype)

        even = all(s == self.shard_sizes[0] for s in self.shard_sizes)
        if even:
            dist.all_gather_into_tensor(full, local)
        else:
            chunks = [
                torch.empty(s, dtype=dtype, device=device)
                for s in self.shard_sizes
            ]
            dist.all_gather(chunks, local)
            for r, ch in enumerate(chunks):
                lo = self.shard_offsets[r]
                hi = self.shard_offsets[r + 1]
                full[lo:hi].copy_(ch)
        return full

    @contextmanager
    def gather_full_params(self, module: nn.Module) -> Iterator[None]:
        """
        Context manager: materialize full params via all-gather, rewrite
        each ``nn.Parameter`` to view into the gathered buffer, run the
        ``with`` block, then release the buffer and restore the local
        sharded storage.

        The gathered buffer's dtype matches the first parameter's dtype
        (usually BF16), so forward/backward run in low precision.
        """
        if self.world_size <= 1:
            yield
            return

        # Choose dtype to gather in — match the live parameter dtype.
        gather_dtype = self.param_order[0][1].dtype

        full = self._build_full_buffer(gather_dtype)

        # Save originals so we can restore after the block.
        saved: List[Tuple[nn.Parameter, torch.Tensor]] = []
        try:
            for name, p in self.param_order:
                sl = self.param_offsets[name]
                view = full[sl.global_start:sl.global_end].view(sl.shape)
                saved.append((p, p.data))
                p.data = view
            yield
        finally:
            for p, orig in saved:
                p.data = orig
            del full

    def scatter_grads(self, model: nn.Module) -> torch.Tensor:
        """
        Reduce-scatter gradients across all ranks.

        Returns the local FP32 gradient shard (1-D, same length as
        ``param_shard``). Each rank's returned tensor corresponds to its
        own slice of the flat parameter layout, so it can be used
        directly to update ``param_shard`` via the optimizer.

        After this call the per-parameter ``.grad`` fields are cleared
        to free memory.
        """
        device = self.param_shard.device

        # Assemble a flat gradient buffer in FP32 from each param's grad.
        flat_grad = torch.zeros(self.total_numel, dtype=torch.float32, device=device)
        for name, p in self.param_order:
            if p.grad is None:
                continue
            sl = self.param_offsets[name]
            flat_grad[sl.global_start:sl.global_end].copy_(
                p.grad.detach().reshape(-1).to(torch.float32)
            )
            p.grad = None

        local_grad = torch.empty(
            self.shard_sizes[self.rank], dtype=torch.float32, device=device
        )

        even = all(s == self.shard_sizes[0] for s in self.shard_sizes)
        if even:
            dist.reduce_scatter_tensor(local_grad, flat_grad, op=dist.ReduceOp.SUM)
            # Average across ranks (standard data-parallel reduction).
            local_grad.div_(self.world_size)
        else:
            # Variable-size fallback: all-reduce then slice.
            dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
            flat_grad.div_(self.world_size)
            lo = self.shard_offsets[self.rank]
            hi = self.shard_offsets[self.rank + 1]
            local_grad.copy_(flat_grad[lo:hi])

        return local_grad


# ---------------------------------------------------------------------------
# Heterogeneous helpers
# ---------------------------------------------------------------------------
def vram_weights_from_tiers(tiers: Sequence[object]) -> List[float]:
    """
    Build a per-rank ``vram_weights`` list from a list of ``TierSpec``
    objects. Each weight is the GPU's total memory in GB; the caller
    decides whether to pass these into ``ShardState.build`` for a
    heterogeneous split.

    Accepts any object exposing ``total_mem_gb`` and ``device_index``.
    """
    by_idx: Dict[int, float] = {}
    for t in tiers:
        idx = int(getattr(t, "device_index"))
        mem = float(getattr(t, "total_mem_gb"))
        by_idx[idx] = mem
    if not by_idx:
        return []
    # Order by device index so that rank r maps to the rth lowest GPU.
    ordered = [by_idx[k] for k in sorted(by_idx)]
    return ordered
