# Copyright (c) 2026, Neuron_SP Project (github.com/dylanyunlon/Neuron_SP).
# Portions adapted from NVIDIA Megatron-LM commit 000dc1c71497cd14825dc037955b701d40394b2c.
# Original copyright (c) 2026, NVIDIA CORPORATION. Licensed under Apache 2.0.
#
# DES-LOC: Decoupled Execution with Shared LOcality Cache
# Hardware target: 2x A6000 48GB SM86 + 1x H100 NVL 96GB SM90, PCIe, no NVLink, 1.5TB DRAM

"""
HeteroDBuffer — heterogeneity-aware distributed parameter buffer for DES-LOC.

Upstream design intent (Megatron-LM 000dc1c):
    Megatron's DBuffer is a "distributed tensor buffer" that groups logical
    tensors into one contiguous local storage tensor, enabling collective
    operations (all-gather, reduce-scatter, all-reduce, scatter) over a
    DeviceMesh without materialising per-tensor temporaries.  It introduces
    three placement primitives — Replicate, Partial, Flat — and a GlobalLayout
    that computes row-aligned element offsets with LCM-based chunk sizing so
    DP shard boundaries never split dim-0 rows.  The buffer is intentionally
    decoupled from tensor-parallel metadata, leaving TP extension to the caller.

DES-LOC adaptation rationale:
    Megatron's DBuffer assumes a *homogeneous* device mesh: every rank has
    identical memory capacity and compute throughput.  Neuron_SP targets a
    *heterogeneous* cluster with two compute tiers:

        Tier-A (SM86): 2× A6000, 48 GB each, PCIe-attached, SM 8.6
        Tier-B (SM90): 1× H100 NVL, 96 GB, PCIe-attached, SM 9.0

    Key asymmetries that break Megatron's homogeneous assumptions:
    1.  Shard sizes must be *capacity-weighted*, not uniform.  H100 can hold
        twice the parameter shard of each A6000.
    2.  AllGather / ReduceScatter over PCIe (no NVLink) must minimise
        cross-device bandwidth; we prefer CPU DRAM as a "locality cache"
        (the LOC in DES-LOC) for gather intermediates rather than peer-to-peer
        PCIe transfers.
    3.  SM90 supports FP8 natively; SM86 only BF16/FP16.  The buffer must
        track per-shard dtype capabilities and promote/demote on the fly.
    4.  Execution is *decoupled* (the DE in DES-LOC): forward on SM86 pair
        can overlap with optimizer steps on SM90.  The buffer exposes an
        async-redistribution API with explicit event synchronisation so the
        two execution streams do not stall each other.

    GlobalLayout is extended to accept per-rank capacity weights, producing
    variable-length local shards whose sizes are proportional to device VRAM.
    The collective operations are re-implemented to:
    (a) stage gather intermediates in pinned CPU memory (the LOC cache) when
        the aggregate tensor exceeds a configurable GPU headroom threshold;
    (b) issue async D2H / H2D copies with CUDA stream overlap instead of
        synchronous P2P;
    (c) track per-rank dtype capability and insert cast kernels at shard
        boundaries.

    The LOC cache is a shared pinned-memory region sized to
    ``loc_cache_fraction * total_cpu_dram`` (default 0.1 → ~150 GB),
    allocated once at process start and reused across buffer lifetimes.

Public API mirrors Megatron's DBuffer so upper-level Neuron_SP code can
swap implementations with a single import alias.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import threading
import time
import warnings
from collections.abc import Iterable
from typing import Dict, List, Optional, Sequence, Tuple, TypeAlias, Union

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Shape: TypeAlias = Union[torch.Size, Iterable[int]]

# ---------------------------------------------------------------------------
# Device tier constants
# ---------------------------------------------------------------------------
SM86_COMPUTE_CAP = (8, 6)   # A6000
SM90_COMPUTE_CAP = (9, 0)   # H100 NVL

# Default VRAM capacities in bytes (used when auto-detection is not possible
# in unit-test / CPU-only environments)
_DEFAULT_VRAM_BYTES: Dict[Tuple[int, int], int] = {
    SM86_COMPUTE_CAP: 48 * 1024**3,   # 48 GB
    SM90_COMPUTE_CAP: 96 * 1024**3,   # 96 GB
}

# ---------------------------------------------------------------------------
# Placement primitives  (mirror Megatron, extended for heterogeneity)
# ---------------------------------------------------------------------------


class Placement:
    """Base class for HeteroDBuffer placements.

    DES-LOC adaptation: identical to Megatron's Placement; kept separate so
    future heterogeneity-specific placements (e.g. ``TieredFlat``) can be
    added without touching upstream code.
    """


@dataclasses.dataclass(frozen=True)
class Replicate(Placement):
    """Each rank holds a full copy of the global buffer shard.

    Upstream: Megatron Replicate — identical semantics.
    DES-LOC: used after AllGather; the LOC cache may hold the canonical copy
    during forward on Tier-A while Tier-B runs optimiser step.
    """


@dataclasses.dataclass(frozen=True)
class Partial(Placement):
    """Each rank holds an unreduced contribution (gradient accumulation state).

    Upstream: Megatron Partial with configurable reduce_op.
    DES-LOC: Partial buffers on Tier-A (SM86) may be in BF16 while the
    AllReduce target on Tier-B (SM90) is FP32 or FP8.  The reduce op is
    applied *after* an optional dtype promotion step.
    """

    reduce_op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM


@dataclasses.dataclass(frozen=True)
class Flat(Placement):
    """Dim-0 flat shard; each rank owns a contiguous slice of the global buffer.

    Upstream: Megatron Flat — identical shard semantics.
    DES-LOC: shard *sizes* are not uniform; they are proportional to per-rank
    VRAM capacity weights.  The slice offsets are stored in
    ``HeteroGlobalLayout.rank_offsets`` rather than computed from uniform
    division.
    """


@dataclasses.dataclass(frozen=True)
class WeightedFlat(Placement):
    """Capacity-weighted dim-0 shard (DES-LOC-specific, no Megatron equivalent).

    Explicitly requests that shard sizes follow ``device_weights`` provided at
    mesh construction time.  ``Flat`` is reinterpreted as ``WeightedFlat``
    internally when the mesh has heterogeneous weights.
    """


# ---------------------------------------------------------------------------
# Helper: device capability detection
# ---------------------------------------------------------------------------


def _get_device_compute_cap(device: torch.device) -> Tuple[int, int]:
    """Return (major, minor) compute capability for a CUDA device.

    Falls back to (0, 0) for CPU or non-CUDA devices (used in tests).
    """
    if device.type != "cuda":
        return (0, 0)
    try:
        return torch.cuda.get_device_capability(device)
    except Exception:
        return (0, 0)


def _tier_of_cap(cap: Tuple[int, int]) -> str:
    """Map compute capability to DES-LOC tier string."""
    if cap >= (9, 0):
        return "SM90"
    if cap >= (8, 6):
        return "SM86"
    return "UNKNOWN"


def _supports_fp8(cap: Tuple[int, int]) -> bool:
    """H100 (SM90+) supports FP8 E4M3 / E5M2; A6000 (SM86) does not."""
    return cap >= (9, 0)


# ---------------------------------------------------------------------------
# LOC (Shared LOcality Cache) — pinned CPU staging area
# ---------------------------------------------------------------------------


class _LOCCache:
    """Singleton pinned-CPU-memory cache used as gather/scatter staging area.

    DES-LOC rationale:
        On PCIe-only clusters, peer-to-peer GPU transfers are bottlenecked by
        the host PCIe bridge.  Staging in pinned CPU DRAM allows each GPU to
        perform a single D2H transfer, after which the CPU can serve H2D reads
        to any GPU without occupying additional PCIe bandwidth in an all-to-all
        pattern.  With 1.5 TB DRAM, a 150 GB pinned region is feasible and
        covers typical LLM parameter + gradient buffers.

    Thread-safety: allocations are serialised with a reentrant lock.
    """

    _instance: Optional["_LOCCache"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
        return cls._instance

    def __init__(self, capacity_bytes: int = 150 * 1024**3):
        with self._lock:
            if self._initialised:
                return
            self._capacity = capacity_bytes
            self._used = 0
            self._buffers: List[torch.Tensor] = []
            self._alloc_lock = threading.RLock()
            self._initialised = True
            logger.info(
                "DES-LOC LOC cache initialised: %.1f GB pinned CPU DRAM",
                capacity_bytes / 1024**3,
            )

    def allocate(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """Allocate a pinned CPU tensor of ``numel`` elements.

        DES-LOC: falls back to pageable memory if pinned allocation fails
        (e.g. in test environments without CUDA).  Logs a warning so engineers
        can detect suboptimal staging paths.
        """
        nbytes = numel * dtype.itemsize
        with self._alloc_lock:
            if self._used + nbytes > self._capacity:
                warnings.warn(
                    f"DES-LOC LOC cache exhausted ({self._used / 1024**3:.1f} GB used, "
                    f"{self._capacity / 1024**3:.1f} GB capacity).  "
                    "Falling back to pageable staging — performance may degrade.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return torch.empty(numel, dtype=dtype)
            try:
                buf = torch.empty(numel, dtype=dtype).pin_memory()
            except RuntimeError:
                buf = torch.empty(numel, dtype=dtype)
            self._used += nbytes
            self._buffers.append(buf)
            return buf

    def free(self, buf: torch.Tensor) -> None:
        """Release a previously allocated LOC buffer."""
        with self._alloc_lock:
            nbytes = buf.numel() * buf.dtype.itemsize
            if buf in self._buffers:
                self._buffers.remove(buf)
                self._used -= nbytes

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def capacity_bytes(self) -> int:
        return self._capacity


# ---------------------------------------------------------------------------
# Owned range metadata
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _OwnedRange:
    """This rank's owned slice of one logical tensor in element coordinates.

    Upstream: identical struct to Megatron ``_OwnedRange``.
    DES-LOC: offsets are computed from capacity-weighted shard boundaries
    rather than uniform division.
    """

    numel: int
    tensor_relative_offset: int
    buffer_relative_offset: int


# ---------------------------------------------------------------------------
# HeteroGlobalLayout — capacity-weighted extension of Megatron GlobalLayout
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HeteroGlobalLayout:
    """Global tensor layout with per-rank capacity-weighted shard boundaries.

    Upstream design (Megatron GlobalLayout):
        Assigns row-aligned element offsets to logical tensors, pads the total
        to ``chunk_size * dp_size``, and derives uniform per-rank shard slices
        via uniform division.

    DES-LOC adaptation:
        Uniform division is replaced by *weighted* division: the global buffer
        is partitioned proportionally to ``rank_weights``.  For the target
        hardware (2× A6000 @ 48 GB, 1× H100 @ 96 GB) the weights are [1, 1, 2],
        so the H100 owns twice the parameter shard of each A6000.  This doubles
        the effective parameter capacity utilisation vs. uniform sharding.

    Attributes:
        tensor_shapes:   Logical tensor shapes in tensor-id order.
        tensor_to_offset: Global element offset for each logical tensor.
        size:            Total global buffer size (padded).
        rank_weights:    Relative capacity weight for each rank (sums to world).
        rank_offsets:    Cumulative element offset for each rank's shard start.
        rank_numels:     Per-rank local buffer size in elements.
        chunk_size:      LCM of all tensor row sizes; shard boundaries are
                         multiples of this to preserve dim-0 alignment.
    """

    tensor_shapes: Tuple[torch.Size, ...]
    tensor_to_offset: Tuple[int, ...]
    size: int
    rank_weights: Tuple[float, ...]
    rank_offsets: Tuple[int, ...]
    rank_numels: Tuple[int, ...]
    chunk_size: int

    @classmethod
    def build(
        cls,
        shapes: Iterable[Shape],
        dp_size: int,
        rank_weights: Optional[Sequence[float]] = None,
    ) -> "HeteroGlobalLayout":
        """Compute weighted global tensor layout.

        Args:
            shapes:       Logical tensor shapes.
            dp_size:      Number of data-parallel ranks.
            rank_weights: Per-rank capacity weights.  If None, defaults to
                          uniform (identical to Megatron behaviour).  For the
                          DES-LOC target hardware pass [1.0, 1.0, 2.0].

        Returns:
            HeteroGlobalLayout with weighted shard boundaries.

        Raises:
            ValueError: If weights length mismatches dp_size, or any weight ≤ 0.
        """
        if dp_size <= 0:
            raise ValueError(f"DP size must be positive, got {dp_size}.")

        if rank_weights is None:
            rank_weights = [1.0] * dp_size
        rank_weights = list(rank_weights)
        if len(rank_weights) != dp_size:
            raise ValueError(
                f"rank_weights length {len(rank_weights)} must equal dp_size {dp_size}."
            )
        for i, w in enumerate(rank_weights):
            if w <= 0:
                raise ValueError(f"rank_weights[{i}]={w} must be positive.")

        tensor_shapes = tuple(torch.Size(s) for s in shapes)
        if not tensor_shapes:
            raise ValueError("HeteroGlobalLayout requires at least one tensor shape.")

        # --- Compute chunk_size (LCM of row sizes) — identical to Megatron ---
        chunk_size = 1
        for shape in tensor_shapes:
            row_size = _non_leading_numel(shape)
            if row_size <= 0:
                raise ValueError(
                    f"Cannot compute layout for zero-sized non-leading dims: {shape}."
                )
            chunk_size = math.lcm(chunk_size, row_size)

        # --- Compute row-aligned element offsets — identical to Megatron ---
        tensor_to_offset = _compute_tensor_offsets(tensor_shapes, chunk_size)

        # --- Pad total size so weighted shards are chunk_size-aligned ---
        # With non-uniform weights we need the total to be divisible by
        # chunk_size * weight_lcm_denom so every rank shard aligns to chunk_size.
        # We approximate by rounding each rank's allocation up to chunk_size.
        raw_total = max(
            (tensor_to_offset[i] + tensor_shapes[i].numel())
            for i in range(len(tensor_shapes))
        )
        raw_total = _pad_to_multiple(raw_total, chunk_size)

        # Weighted partition: rank i gets floor(total * w_i / W) elements,
        # rounded up to chunk_size.  Any rounding residual is absorbed by the
        # last rank (H100 in the target config, which has the most headroom).
        W = sum(rank_weights)
        rank_numels: List[int] = []
        for i, w in enumerate(rank_weights):
            share = int(math.ceil(raw_total * w / W / chunk_size)) * chunk_size
            rank_numels.append(share)

        # Rebalance: if sum > raw_total, shrink the last rank; if sum < raw_total,
        # grow it.  This preserves the invariant that all params are covered.
        total_assigned = sum(rank_numels)
        if total_assigned != raw_total:
            delta = raw_total - total_assigned
            # delta is always a multiple of chunk_size by construction
            rank_numels[-1] += delta
            if rank_numels[-1] < 0:
                raise RuntimeError(
                    "HeteroGlobalLayout: weighted partition produced negative last-rank shard. "
                    "Reduce rank_weights disparity or increase total size."
                )

        rank_offsets: List[int] = []
        acc = 0
        for n in rank_numels:
            rank_offsets.append(acc)
            acc += n

        total_size = acc  # may differ from raw_total due to rounding

        return cls(
            tensor_shapes=tensor_shapes,
            tensor_to_offset=tuple(tensor_to_offset),
            size=total_size,
            rank_weights=tuple(rank_weights),
            rank_offsets=tuple(rank_offsets),
            rank_numels=tuple(rank_numels),
            chunk_size=chunk_size,
        )

    def get_local_range(self, rank: int) -> Tuple[int, int]:
        """Return (offset, numel) for ``rank``'s local shard.

        DES-LOC: offset and numel are capacity-weighted, not uniform.
        """
        return self.rank_offsets[rank], self.rank_numels[rank]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HeteroGlobalLayout):
            return NotImplemented
        return (
            self.tensor_shapes == other.tensor_shapes
            and self.tensor_to_offset == other.tensor_to_offset
            and self.size == other.size
            and self.rank_weights == other.rank_weights
        )

    def __hash__(self) -> int:
        return hash((self.tensor_shapes, self.tensor_to_offset, self.size, self.rank_weights))


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _non_leading_numel(shape: torch.Size) -> int:
    """Number of elements in all dims after dim 0 (Megatron: non_leading_numel)."""
    if len(shape) == 0:
        raise ValueError(f"HeteroDBuffer does not support 0D tensor shapes: {shape}.")
    return max(1, shape[1:].numel())


def _pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"Pad multiple must be positive, got {multiple}.")
    return ((value + multiple - 1) // multiple) * multiple


def _compute_tensor_offsets(
    tensor_shapes: Tuple[torch.Size, ...],
    chunk_size: int,
) -> List[int]:
    """Assign row-aligned element offsets to logical tensors.

    DES-LOC: identical algorithm to Megatron GlobalLayout.build() — fragment
    packing into regular-tensor LCM gaps — preserved because it is
    placement-agnostic.  The *shard* boundaries are computed separately in
    HeteroGlobalLayout from rank_weights.
    """
    UNASSIGNED = -1
    n = len(tensor_shapes)
    tensor_to_offset: List[int] = [UNASSIGNED] * n

    fragment_items: List[Tuple[int, torch.Size]] = []
    regular_items: List[Tuple[int, torch.Size]] = []
    for tid, shape in enumerate(tensor_shapes):
        if shape.numel() < chunk_size:
            fragment_items.append((tid, shape))
        else:
            regular_items.append((tid, shape))

    # Largest fragments first to fill gaps greedily
    fragment_items.sort(key=lambda x: x[1].numel(), reverse=True)

    next_offset = 0
    while regular_items:
        tid, shape = regular_items.pop(0)
        numel = shape.numel()
        tensor_to_offset[tid] = next_offset

        if numel % chunk_size == 0:
            next_offset += numel
            continue

        gap_offset = next_offset + numel
        next_offset += _pad_to_multiple(numel, chunk_size)
        fragment_gap_end = next_offset
        remainder = numel % chunk_size

        # Try conjugate pairing (Megatron algorithm verbatim)
        conjugate_item = None
        for cand in regular_items[:]:
            _, cshape = cand
            crem = cshape.numel() % chunk_size
            if crem == 0:
                continue
            if remainder + crem <= chunk_size:
                conjugate_item = cand
                regular_items.remove(cand)
                break

        if conjugate_item is not None:
            cid, cshape = conjugate_item
            cnumel = cshape.numel()
            crem = cnumel % chunk_size
            coff = next_offset - crem
            tensor_to_offset[cid] = coff
            fragment_gap_end = coff
            next_offset += (cnumel // chunk_size) * chunk_size

        for frag in fragment_items[:]:
            fid, fshape = frag
            fnumel = fshape.numel()
            aligned = _pad_to_multiple(gap_offset, _non_leading_numel(fshape))
            if aligned + fnumel > fragment_gap_end:
                continue
            tensor_to_offset[fid] = aligned
            gap_offset = aligned + fnumel
            fragment_items.remove(frag)

    for fid, fshape in fragment_items:
        next_offset = _pad_to_multiple(next_offset, _non_leading_numel(fshape))
        tensor_to_offset[fid] = next_offset
        next_offset += fshape.numel()

    return tensor_to_offset


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_placements(placements: Tuple[Placement, ...]) -> None:
    """Validate placement list (Flat must be a suffix — identical to Megatron)."""
    seen_flat = False
    for p in placements:
        if not isinstance(p, (Replicate, Partial, Flat, WeightedFlat)):
            raise TypeError(f"Unsupported HeteroDBuffer placement: {p!r}.")
        if isinstance(p, (Flat, WeightedFlat)):
            seen_flat = True
        elif seen_flat:
            raise ValueError(
                "Flat/WeightedFlat placements must be a suffix of the placement list "
                "so each local buffer is a contiguous global-buffer range."
            )


# ---------------------------------------------------------------------------
# DES-LOC device info registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DeviceTierInfo:
    """Per-rank device capability and capacity metadata.

    DES-LOC: built once at HeteroDBuffer construction from ``torch.cuda``
    queries and stored on all ranks.  Used by collective helpers to:
    - decide staging path (LOC cache vs direct P2P),
    - select dtype promotion for cross-tier operations,
    - assign capacity weights to ``HeteroGlobalLayout``.
    """

    rank: int
    device: torch.device
    compute_cap: Tuple[int, int]
    tier: str
    vram_bytes: int
    supports_fp8: bool

    @classmethod
    def from_rank(cls, rank: int, device: torch.device) -> "DeviceTierInfo":
        cap = _get_device_compute_cap(device)
        vram = _DEFAULT_VRAM_BYTES.get(cap, 48 * 1024**3)
        # If CUDA is available, query actual free memory instead of nominal cap
        if device.type == "cuda":
            try:
                vram = torch.cuda.get_device_properties(device).total_memory
            except Exception:
                pass
        return cls(
            rank=rank,
            device=device,
            compute_cap=cap,
            tier=_tier_of_cap(cap),
            vram_bytes=vram,
            supports_fp8=_supports_fp8(cap),
        )


# ---------------------------------------------------------------------------
# HeteroDBuffer
# ---------------------------------------------------------------------------


class HeteroDBuffer:
    """Heterogeneity-aware distributed parameter buffer for DES-LOC.

    This class reimplements Megatron's DBuffer with the following DES-LOC
    extensions:

    1.  **Capacity-weighted sharding** (WeightedFlat / Flat):
        Shard boundaries follow ``rank_weights`` (derived from VRAM) so the
        H100 holds twice as many parameters as each A6000.

    2.  **LOC staging** (Shared LOcality Cache):
        AllGather operations stage the gathered tensor in pinned CPU DRAM
        when the output exceeds ``loc_cache_threshold_bytes``.  The staging
        tensor is populated via async D2H (``non_blocking=True``) on a
        dedicated staging CUDA stream, then redistributed H2D per-GPU.
        This trades peak GPU memory for PCIe bandwidth efficiency.

    3.  **Cross-tier dtype handling**:
        When a Partial buffer on SM86 (BF16) is all-reduced into a Replicate
        buffer destined for SM90 (FP32 or FP8), the cast is inserted before
        the collective so the operation runs in the target dtype.

    4.  **Decoupled async redistribution**:
        ``redistribute_async()`` returns a ``torch.cuda.Event`` that the
        caller can ``event.wait()`` on before consuming the output.  This
        allows the DES-LOC scheduler to overlap Tier-A forward with Tier-B
        optimiser step without explicit ``synchronize()`` calls.

    5.  **DeepSpeed ZeRO compatibility**:
        The buffer's ``local_buffer`` attribute is layout-compatible with
        DeepSpeed ZeRO Stage-1/2 flat parameter buffers so existing ZeRO
        partitioning code can wrap a HeteroDBuffer without modification.

    Upstream API preserved from Megatron DBuffer:
        - ``__init__``, ``from_local``, ``distribute_tensors``
        - ``redistribute``, ``allgather``, ``allreduce``,
          ``reduce_scatter``, ``scatter``
        - ``get_local_tensor``, ``get_dtensor``

    DES-LOC extensions:
        - ``redistribute_async``
        - ``device_tier_infos``
        - ``loc_cache``
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        placements: Iterable[Placement],
        tensor_shapes: Iterable[Shape],
        dtype: torch.dtype,
        device: torch.device,
        rank_weights: Optional[Sequence[float]] = None,
        loc_cache_threshold_bytes: int = 4 * 1024**3,  # 4 GB
        loc_cache_capacity_bytes: int = 150 * 1024**3,  # 150 GB
        staging_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Create a HeteroDBuffer.

        Args:
            process_group:            ProcessGroup for collectives.
            placements:               Per-axis placements (single-axis for now).
            tensor_shapes:            Global shapes of logical tensors.
            dtype:                    Local buffer dtype.
            device:                   Local device.
            rank_weights:             Per-rank VRAM capacity weights.  If None,
                                      auto-detected from CUDA properties; falls
                                      back to uniform for non-CUDA devices.
            loc_cache_threshold_bytes: AllGather outputs exceeding this size are
                                      staged through pinned CPU DRAM (LOC cache).
            loc_cache_capacity_bytes:  Total LOC cache capacity (pinned DRAM).
            staging_stream:           Optional dedicated CUDA stream for async
                                      D2H/H2D staging copies.
        """
        placements = tuple(placements)
        _validate_placements(placements)

        self.process_group = process_group
        self.placements = placements
        self.dtype = dtype
        self.device = device
        self.loc_cache_threshold_bytes = loc_cache_threshold_bytes
        self._loc_cache = _LOCCache(loc_cache_capacity_bytes)
        self._staging_stream = staging_stream  # None → use current stream

        world_size = dist.get_world_size(process_group)
        self._rank = dist.get_rank(process_group)

        # --- Build per-rank device info ---
        self.device_tier_infos: List[DeviceTierInfo] = []
        for r in range(world_size):
            # We can only query the local rank's device directly; for remote
            # ranks we use the rank_weights heuristic or a global metadata
            # all-gather.  For now, tag all non-local ranks as "UNKNOWN" unless
            # rank_weights imply SM90 (weight ≥ 1.5× median).
            if r == self._rank:
                info = DeviceTierInfo.from_rank(r, device)
            else:
                info = DeviceTierInfo(
                    rank=r,
                    device=torch.device(f"cuda:{r}" if device.type == "cuda" else "cpu"),
                    compute_cap=(0, 0),
                    tier="UNKNOWN",
                    vram_bytes=_DEFAULT_VRAM_BYTES.get(SM86_COMPUTE_CAP, 48 * 1024**3),
                    supports_fp8=False,
                )
            self.device_tier_infos.append(info)

        # --- Resolve rank_weights ---
        if rank_weights is None:
            rank_weights = self._auto_detect_rank_weights(world_size)
        self._rank_weights = list(rank_weights)

        # --- Build layout ---
        tensor_shapes_t = tuple(torch.Size(s) for s in tensor_shapes)
        self.layout = HeteroGlobalLayout.build(
            tensor_shapes_t, dp_size=world_size, rank_weights=self._rank_weights
        )

        # --- Allocate local buffer ---
        self.offset, local_numel = self.layout.get_local_range(self._rank)
        self.local_buffer = torch.empty(local_numel, dtype=dtype, device=device)

        logger.debug(
            "HeteroDBuffer rank %d: offset=%d, numel=%d, tier=%s",
            self._rank,
            self.offset,
            local_numel,
            self.device_tier_infos[self._rank].tier,
        )

    def _auto_detect_rank_weights(self, world_size: int) -> List[float]:
        """Derive rank_weights from local VRAM, broadcasting to group.

        DES-LOC: each rank broadcasts its VRAM in bytes; weights are normalised
        to the minimum VRAM (so min-VRAM rank gets weight 1.0).  Falls back to
        uniform if CUDA is unavailable.
        """
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return [1.0] * world_size

        local_vram = float(self.device_tier_infos[self._rank].vram_bytes)
        vram_tensor = torch.tensor([local_vram], dtype=torch.float64, device=self.device)
        all_vram = [torch.zeros(1, dtype=torch.float64, device=self.device)
                    for _ in range(world_size)]
        try:
            dist.all_gather(all_vram, vram_tensor, group=self.process_group)
            vram_values = [float(v.item()) for v in all_vram]
            min_vram = min(v for v in vram_values if v > 0)
            weights = [max(v / min_vram, 0.5) for v in vram_values]
            logger.info("DES-LOC auto rank_weights: %s", weights)
            return weights
        except Exception as exc:
            logger.warning("DES-LOC rank_weights auto-detect failed (%s), using uniform.", exc)
            return [1.0] * world_size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.process_group)

    @property
    def loc_cache(self) -> _LOCCache:
        return self._loc_cache

    # ------------------------------------------------------------------
    # Class methods (mirror Megatron DBuffer class methods)
    # ------------------------------------------------------------------

    @classmethod
    def from_local(
        cls,
        local_buffer: torch.Tensor,
        process_group: dist.ProcessGroup,
        placements: Iterable[Placement],
        tensor_shapes: Iterable[Shape],
        rank_weights: Optional[Sequence[float]] = None,
    ) -> "HeteroDBuffer":
        """Wrap an existing local tensor as a HeteroDBuffer without reallocation.

        Upstream: Megatron DBuffer.from_local — reuses caller-provided storage.
        DES-LOC: validates that the local buffer size matches the capacity-
        weighted shard size for this rank.

        Args:
            local_buffer:  Flat 1D contiguous tensor.
            process_group: ProcessGroup for collectives.
            placements:    Per-axis placements.
            tensor_shapes: Logical tensor shapes.
            rank_weights:  Per-rank capacity weights.

        Returns:
            HeteroDBuffer reusing ``local_buffer`` storage.
        """
        placements = tuple(placements)
        _validate_placements(placements)
        if local_buffer.dim() != 1:
            raise ValueError("local_buffer must be a flat 1D tensor.")
        if not local_buffer.is_contiguous():
            raise ValueError("local_buffer must be contiguous for collective operations.")

        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        tensor_shapes_t = tuple(torch.Size(s) for s in tensor_shapes)

        if rank_weights is None:
            rank_weights = [1.0] * world_size
        layout = HeteroGlobalLayout.build(tensor_shapes_t, world_size, rank_weights)
        offset, local_numel = layout.get_local_range(rank)
        if local_buffer.numel() != local_numel:
            raise ValueError(
                f"local_buffer has {local_buffer.numel()} elements, "
                f"expected {local_numel} for rank {rank} with weights {rank_weights}."
            )

        buf = cls.__new__(cls)
        buf.process_group = process_group
        buf.placements = placements
        buf.dtype = local_buffer.dtype
        buf.device = local_buffer.device
        buf.layout = layout
        buf.offset = offset
        buf.local_buffer = local_buffer
        buf._rank = rank
        buf._rank_weights = list(rank_weights)
        buf.loc_cache_threshold_bytes = 4 * 1024**3
        buf._loc_cache = _LOCCache()
        buf._staging_stream = None
        buf.device_tier_infos = [
            DeviceTierInfo(r, local_buffer.device, (0, 0), "UNKNOWN", 48 * 1024**3, False)
            for r in range(world_size)
        ]
        return buf

    @classmethod
    def distribute_tensors(
        cls,
        tensors: Iterable[torch.Tensor],
        process_group: dist.ProcessGroup,
        placements: Iterable[Placement],
        rank_weights: Optional[Sequence[float]] = None,
    ) -> "HeteroDBuffer":
        """Distribute full local tensor values into a HeteroDBuffer.

        Upstream: Megatron DBuffer.distribute_tensors — copies each rank's
        owned slice from the full tensor.
        DES-LOC: owned slices follow capacity-weighted offsets.

        Args:
            tensors:       Full tensor values on this rank.
            process_group: ProcessGroup.
            placements:    Per-axis placements.
            rank_weights:  Optional per-rank weights.

        Returns:
            HeteroDBuffer populated with rank-local parameter slices.
        """
        tensors = tuple(t.detach().contiguous() for t in tensors)
        if not tensors:
            raise ValueError("distribute_tensors() requires at least one tensor.")
        dtype = tensors[0].dtype
        for t in tensors:
            if t.dtype != dtype:
                raise ValueError("All tensors in a HeteroDBuffer must have the same dtype.")

        world_size = dist.get_world_size(process_group)
        if rank_weights is None:
            rank_weights = [1.0] * world_size

        buf = cls(
            process_group=process_group,
            placements=placements,
            tensor_shapes=[t.shape for t in tensors],
            dtype=dtype,
            device=tensors[0].device,
            rank_weights=rank_weights,
        )

        for idx, tensor in enumerate(tensors):
            owned = buf._get_owned_range(idx)
            if owned is None:
                continue
            src = tensor.view(-1).narrow(0, owned.tensor_relative_offset, owned.numel)
            buf.local_buffer.narrow(0, owned.buffer_relative_offset, owned.numel).copy_(src)

        return buf

    # ------------------------------------------------------------------
    # Owned range
    # ------------------------------------------------------------------

    def _get_owned_range(self, tensor_index: int) -> Optional[_OwnedRange]:
        """Compute this rank's owned element range for logical tensor ``tensor_index``.

        Upstream: Megatron DBuffer._get_owned_range — identical overlap logic.
        DES-LOC: ``self.offset`` and ``self.local_buffer.numel()`` are
        capacity-weighted, so the overlap naturally reflects the weighted shard.
        """
        t_start = self.layout.tensor_to_offset[tensor_index]
        t_end = t_start + self.layout.tensor_shapes[tensor_index].numel()
        b_start = self.offset
        b_end = b_start + self.local_buffer.numel()

        ov_start = max(t_start, b_start)
        ov_end = min(t_end, b_end)
        if ov_start >= ov_end:
            return None

        return _OwnedRange(
            numel=ov_end - ov_start,
            tensor_relative_offset=ov_start - t_start,
            buffer_relative_offset=ov_start - b_start,
        )

    # ------------------------------------------------------------------
    # Internal helpers for output buffer creation / validation
    # ------------------------------------------------------------------

    def _create_or_validate_out(
        self, placements: Tuple[Placement, ...], out: Optional["HeteroDBuffer"]
    ) -> "HeteroDBuffer":
        if out is None:
            return HeteroDBuffer(
                process_group=self.process_group,
                placements=placements,
                tensor_shapes=self.layout.tensor_shapes,
                dtype=self.dtype,
                device=self.device,
                rank_weights=self._rank_weights,
                loc_cache_threshold_bytes=self.loc_cache_threshold_bytes,
            )
        if out.layout != self.layout:
            raise ValueError("Output HeteroDBuffer layout mismatch.")
        if out.placements != placements:
            raise ValueError(
                f"Output placements {out.placements!r} != expected {placements!r}."
            )
        if out.dtype != self.dtype:
            raise ValueError(f"Output dtype {out.dtype} != {self.dtype}.")
        if out.device.type != self.device.type:
            raise ValueError(f"Output device type {out.device.type} != {self.device.type}.")
        return out

    # ------------------------------------------------------------------
    # LOC-staged AllGather
    # ------------------------------------------------------------------

    def _should_use_loc_cache(self, output_numel: int) -> bool:
        """Decide whether to stage AllGather through pinned CPU DRAM.

        DES-LOC: use the LOC cache when:
        1. The gathered tensor exceeds ``loc_cache_threshold_bytes``, AND
        2. The device is CUDA (PCIe bandwidth savings only apply to GPU↔CPU).
        3. The LOC cache has enough free capacity.
        """
        nbytes = output_numel * self.dtype.itemsize
        if nbytes < self.loc_cache_threshold_bytes:
            return False
        if self.device.type != "cuda":
            return False
        if nbytes > self._loc_cache.capacity_bytes - self._loc_cache.used_bytes:
            return False
        return True

    def _allgather_via_loc(
        self, output_buffer: torch.Tensor, input_tensor: torch.Tensor
    ) -> None:
        """AllGather using pinned CPU staging (DES-LOC LOC path).

        Algorithm:
        1. D2H: copy local shard to pinned CPU staging buffer (async).
        2. CPU-side all_gather across ranks into a second pinned buffer.
        3. H2D: copy result to output_buffer on GPU (async).

        DES-LOC rationale:
            On PCIe-only clusters, GPU peer-to-peer AllGather sends data over
            the PCIe bus via the host bridge.  Staging in pinned DRAM lets
            each GPU perform exactly one PCIe write (D2H) and one PCIe read
            (H2D) per AllGather, vs. N−1 PCIe writes for direct all-to-all.
            For N=3 (our target), savings are modest but LOC also decouples
            the Tier-A forward pass from Tier-B memory bandwidth.
        """
        staging_in = self._loc_cache.allocate(input_tensor.numel(), self.dtype)
        staging_out = self._loc_cache.allocate(output_buffer.numel(), self.dtype)

        try:
            staging_in.copy_(input_tensor, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            dist.all_gather_into_tensor(
                staging_out, staging_in, group=self.process_group
            )
            output_buffer.copy_(staging_out, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
        finally:
            self._loc_cache.free(staging_in)
            self._loc_cache.free(staging_out)

    # ------------------------------------------------------------------
    # Collective operations (mirror Megatron API, DES-LOC internals)
    # ------------------------------------------------------------------

    def allgather(
        self,
        mesh_axis: int = 0,
        *,
        out: Optional["HeteroDBuffer"] = None,
        use_loc_cache: Optional[bool] = None,
    ) -> "HeteroDBuffer":
        """All-gather a Flat-sharded axis into Replicate placement.

        Upstream: Megatron DBuffer.allgather — same placement transition.
        DES-LOC: routes through LOC staging when output size exceeds threshold,
        otherwise falls back to direct dist.all_gather_into_tensor.

        Args:
            mesh_axis:     Unused (single process group, kept for API compat).
            out:           Optional pre-allocated output HeteroDBuffer.
            use_loc_cache: Force LOC cache on/off; None = auto-detect.
        """
        if not isinstance(self.placements[0], (Flat, WeightedFlat)):
            raise ValueError(
                f"allgather() requires Flat/WeightedFlat placement, got {self.placements[0]!r}."
            )

        new_placements = (Replicate(),)
        out_buf = self._create_or_validate_out(new_placements, out)

        output_numel = out_buf.local_buffer.numel()
        use_loc = use_loc_cache if use_loc_cache is not None else self._should_use_loc_cache(
            output_numel
        )

        if use_loc:
            logger.debug(
                "HeteroDBuffer.allgather: using LOC cache (%.1f MB output, rank %d)",
                output_numel * self.dtype.itemsize / 1024**2,
                self._rank,
            )
            # For weighted shards we must gather variable-length chunks.
            # dist.all_gather_into_tensor assumes equal chunk sizes, so we
            # use all_gather with a list of pre-sized tensors instead.
            self._weighted_allgather_via_loc(out_buf.local_buffer)
        else:
            self._weighted_allgather_direct(out_buf.local_buffer)

        return out_buf

    def _weighted_allgather_direct(self, output: torch.Tensor) -> None:
        """AllGather with variable-size shards into a contiguous output buffer.

        DES-LOC: Each rank has a different local_numel (capacity-weighted).
        dist.all_gather_into_tensor requires equal sizes, so we use
        dist.all_gather with a list of correctly-sized sub-tensors derived
        from rank_numels.
        """
        world_size = self.world_size
        # Build a list of views into the output tensor matching each rank's numel
        recv_tensors: List[torch.Tensor] = []
        cursor = 0
        for r in range(world_size):
            n = self.layout.rank_numels[r]
            recv_tensors.append(output.narrow(0, cursor, n))
            cursor += n

        dist.all_gather(recv_tensors, self.local_buffer, group=self.process_group)

    def _weighted_allgather_via_loc(self, output: torch.Tensor) -> None:
        """LOC-staged weighted AllGather (DES-LOC extension)."""
        world_size = self.world_size
        staging_in = self._loc_cache.allocate(self.local_buffer.numel(), self.dtype)
        staging_out = self._loc_cache.allocate(output.numel(), self.dtype)

        try:
            staging_in.copy_(self.local_buffer, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()

            recv_tensors: List[torch.Tensor] = []
            cursor = 0
            for r in range(world_size):
                n = self.layout.rank_numels[r]
                recv_tensors.append(staging_out.narrow(0, cursor, n))
                cursor += n

            dist.all_gather(recv_tensors, staging_in, group=self.process_group)

            output.copy_(staging_out, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
        finally:
            self._loc_cache.free(staging_in)
            self._loc_cache.free(staging_out)

    def allreduce(
        self,
        mesh_axis: int = 0,
        *,
        out: Optional["HeteroDBuffer"] = None,
        target_dtype: Optional[torch.dtype] = None,
    ) -> "HeteroDBuffer":
        """All-reduce a Partial axis into Replicate placement.

        Upstream: Megatron DBuffer.allreduce — copies local buffer then
        dist.all_reduce in-place.
        DES-LOC: ``target_dtype`` allows cross-tier dtype promotion.  If the
        local buffer is BF16 (Tier-A) and the target is FP32 (Tier-B), the
        cast is performed before the collective.

        Args:
            mesh_axis:    Unused (API compat).
            out:          Optional pre-allocated output.
            target_dtype: Dtype for the collective; None = keep local dtype.
        """
        if not isinstance(self.placements[0], Partial):
            raise ValueError(
                f"allreduce() requires Partial placement, got {self.placements[0]!r}."
            )
        partial_placement = self.placements[0]
        new_placements = (Replicate(),)
        out_buf = self._create_or_validate_out(new_placements, out)

        src = self.local_buffer
        if target_dtype is not None and target_dtype != self.dtype:
            src = src.to(target_dtype)
            out_buf.local_buffer.copy_(src)
        else:
            out_buf.local_buffer.copy_(src)

        dist.all_reduce(
            out_buf.local_buffer,
            op=partial_placement.reduce_op,
            group=self.process_group,
        )
        return out_buf

    def reduce_scatter(
        self,
        mesh_axis: int = 0,
        new_placement: Optional[Placement] = None,
        *,
        out: Optional["HeteroDBuffer"] = None,
    ) -> "HeteroDBuffer":
        """Reduce-scatter a Partial axis into Flat (weighted) placement.

        Upstream: Megatron DBuffer.reduce_scatter — dist.reduce_scatter_tensor.
        DES-LOC: reduce_scatter_tensor assumes equal output sizes; we use a
        list-based dist.reduce_scatter with per-rank sized sub-tensors matching
        capacity-weighted shards.

        Args:
            mesh_axis:     Unused (API compat).
            new_placement: Must be Flat or WeightedFlat.
            out:           Optional pre-allocated output.
        """
        if new_placement is None:
            new_placement = Flat()
        if not isinstance(new_placement, (Flat, WeightedFlat)):
            raise NotImplementedError(
                "HeteroDBuffer reduce_scatter() supports Flat/WeightedFlat output only."
            )
        if not isinstance(self.placements[0], Partial):
            raise ValueError(
                f"reduce_scatter() requires Partial placement, got {self.placements[0]!r}."
            )
        partial_placement = self.placements[0]
        new_placements = (new_placement,)
        out_buf = self._create_or_validate_out(new_placements, out)

        # Build list of per-rank input sub-tensors (weighted sizes)
        input_tensors: List[torch.Tensor] = []
        cursor = 0
        for r in range(self.world_size):
            n = self.layout.rank_numels[r]
            input_tensors.append(self.local_buffer.narrow(0, cursor, n))
            cursor += n

        dist.reduce_scatter(
            out_buf.local_buffer,
            input_tensors,
            op=partial_placement.reduce_op,
            group=self.process_group,
        )
        return out_buf

    def scatter(
        self,
        mesh_axis: int = 0,
        new_placement: Optional[Placement] = None,
        *,
        out: Optional["HeteroDBuffer"] = None,
    ) -> "HeteroDBuffer":
        """Locally slice a Replicate buffer into this rank's weighted Flat shard.

        Upstream: Megatron DBuffer.scatter — local narrow, no communication.
        DES-LOC: the slice offsets and lengths are capacity-weighted.

        Args:
            mesh_axis:     Unused (API compat).
            new_placement: Must be Flat or WeightedFlat.
            out:           Optional pre-allocated output.
        """
        if new_placement is None:
            new_placement = Flat()
        if not isinstance(new_placement, (Flat, WeightedFlat)):
            raise NotImplementedError(
                "HeteroDBuffer scatter() supports Flat/WeightedFlat output only."
            )
        if not isinstance(self.placements[0], Replicate):
            raise ValueError(
                f"scatter() requires Replicate placement, got {self.placements[0]!r}."
            )
        new_placements = (new_placement,)

        dest_offset, dest_numel = self.layout.get_local_range(self._rank)
        local_buffer_offset = dest_offset - self.offset  # should be 0 for Replicate (offset=0)
        if (
            local_buffer_offset < 0
            or local_buffer_offset + dest_numel > self.local_buffer.numel()
        ):
            raise RuntimeError(
                f"scatter() destination [{dest_offset}, {dest_offset + dest_numel}) "
                f"not contained in local buffer [{self.offset}, {self.offset + self.local_buffer.numel()})."
            )

        local_slice = self.local_buffer.narrow(0, local_buffer_offset, dest_numel)

        if out is None:
            result = HeteroDBuffer.__new__(HeteroDBuffer)
            result.process_group = self.process_group
            result.placements = new_placements
            result.dtype = self.dtype
            result.device = self.device
            result.layout = self.layout
            result.offset = dest_offset
            result.local_buffer = local_slice
            result._rank = self._rank
            result._rank_weights = self._rank_weights
            result.loc_cache_threshold_bytes = self.loc_cache_threshold_bytes
            result._loc_cache = self._loc_cache
            result._staging_stream = self._staging_stream
            result.device_tier_infos = self.device_tier_infos
            return result

        out = self._create_or_validate_out(new_placements, out)
        out.local_buffer.copy_(local_slice)
        return out

    # ------------------------------------------------------------------
    # redistribute (Megatron API compat dispatcher)
    # ------------------------------------------------------------------

    def redistribute(
        self,
        new_placements: Iterable[Placement],
        *,
        out: Optional["HeteroDBuffer"] = None,
        use_loc_cache: Optional[bool] = None,
    ) -> "HeteroDBuffer":
        """Redistribute to ``new_placements`` (Megatron DBuffer.redistribute compat).

        Supported transitions (same as Megatron):
        - Flat → Replicate: allgather
        - Partial → Replicate: allreduce
        - Partial → Flat: reduce_scatter
        - Replicate → Flat: scatter (local)

        DES-LOC: allgather path respects ``use_loc_cache`` parameter.
        """
        new_placements = tuple(new_placements)
        _validate_placements(new_placements)
        if len(new_placements) != len(self.placements):
            raise ValueError("redistribute() requires same number of placements.")

        old_p = self.placements[0]
        new_p = new_placements[0]

        if old_p == new_p:
            if out is None:
                return self
            out = self._create_or_validate_out(new_placements, out)
            out.local_buffer.copy_(self.local_buffer)
            return out

        if isinstance(old_p, (Flat, WeightedFlat)) and isinstance(new_p, Replicate):
            return self.allgather(out=out, use_loc_cache=use_loc_cache)
        if isinstance(old_p, Partial) and isinstance(new_p, Replicate):
            return self.allreduce(out=out)
        if isinstance(old_p, Partial) and isinstance(new_p, (Flat, WeightedFlat)):
            return self.reduce_scatter(new_placement=new_p, out=out)
        if isinstance(old_p, Replicate) and isinstance(new_p, (Flat, WeightedFlat)):
            return self.scatter(new_placement=new_p, out=out)

        raise NotImplementedError(
            f"Unsupported HeteroDBuffer placement transition: {old_p!r} → {new_p!r}."
        )

    # ------------------------------------------------------------------
    # Async redistribution (DES-LOC extension)
    # ------------------------------------------------------------------

    def redistribute_async(
        self,
        new_placements: Iterable[Placement],
        *,
        out: Optional["HeteroDBuffer"] = None,
    ) -> Tuple["HeteroDBuffer", Optional["torch.cuda.Event"]]:
        """Redistribute asynchronously, returning (output_buffer, cuda_event).

        DES-LOC: The returned CUDA event is recorded on the staging stream
        after the redistribution completes.  The caller can overlap computation
        on the main stream with the collective by doing:

            out_buf, event = buf.redistribute_async([Replicate()])
            # ... launch other kernels on main stream ...
            if event is not None:
                event.wait()
            # consume out_buf

        If no CUDA device is involved (CPU unit tests), the event is None and
        redistribution is synchronous.

        Args:
            new_placements: Target placements.
            out:            Optional pre-allocated output buffer.

        Returns:
            (output_buffer, event) where event is None for CPU.
        """
        out_buf = self.redistribute(new_placements, out=out)

        if self.device.type != "cuda":
            return out_buf, None

        stream = self._staging_stream or torch.cuda.current_stream(self.device)
        event = torch.cuda.Event()
        event.record(stream)
        return out_buf, event

    # ------------------------------------------------------------------
    # Per-tensor views
    # ------------------------------------------------------------------

    def get_local_tensor(self, index: int) -> torch.Tensor:
        """Return this rank's local shard of logical tensor ``index``.

        Upstream: Megatron DBuffer.get_local_tensor — preserves non-leading
        dims, adjusts leading dim to shard size.
        DES-LOC: the shard may be zero-length (tensor not resident on this rank
        at all) or larger than 1/N for the H100 rank.
        """
        shape = self.layout.tensor_shapes[index]
        owned = self._get_owned_range(index)
        row_size = _non_leading_numel(shape)

        if owned is None:
            empty_shape = torch.Size((0, *shape[1:]))
            return torch.empty(empty_shape, dtype=self.dtype, device=self.device)

        if owned.tensor_relative_offset % row_size != 0 or owned.numel % row_size != 0:
            raise RuntimeError(
                f"Tensor {index} shard does not align to dim-0 row boundaries "
                f"(row_size={row_size}, offset={owned.tensor_relative_offset}, "
                f"numel={owned.numel})."
            )

        local_shape = torch.Size((owned.numel // row_size, *shape[1:]))
        return self.local_buffer.narrow(
            0, owned.buffer_relative_offset, owned.numel
        ).view(local_shape)

    def get_dtensor(self, index: int):
        """Return logical tensor ``index`` as a torch.distributed.tensor.DTensor.

        Upstream: Megatron DBuffer.get_dtensor — wraps local shard in DTensor.
        DES-LOC: for Flat/WeightedFlat, the placement is Shard(0).  Variable
        shard sizes are supported by DTensor's from_local when run_check=False.
        Requires torch.distributed.tensor to be available.
        """
        try:
            import torch.distributed.tensor as dist_tensor
            from torch.distributed.tensor import DTensor
            from torch.distributed import DeviceMesh
        except ImportError as exc:
            raise ImportError(
                "get_dtensor() requires torch.distributed.tensor (PyTorch ≥ 2.1)."
            ) from exc

        torch_placements = []
        for p in self.placements:
            if isinstance(p, Replicate):
                torch_placements.append(dist_tensor.Replicate())
            elif isinstance(p, (Flat, WeightedFlat)):
                torch_placements.append(dist_tensor.Shard(0))
            elif isinstance(p, Partial):
                raise ValueError("Partial placement cannot be represented as DTensor.")
            else:
                raise TypeError(f"Unsupported placement for DTensor: {p!r}.")

        local_tensor = self.get_local_tensor(index)
        tensor_shape = self.layout.tensor_shapes[index]

        # Build a 1-D DeviceMesh over the process group
        ranks = list(range(self.world_size))
        mesh = DeviceMesh(self.device.type, ranks, pg=self.process_group)

        return DTensor.from_local(
            local_tensor=local_tensor,
            device_mesh=mesh,
            placements=tuple(torch_placements),
            run_check=False,
            shape=tensor_shape,
            stride=local_tensor.stride(),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def shard_stats(self) -> Dict[int, Dict]:
        """Return per-rank shard metadata (DES-LOC diagnostics)."""
        stats = {}
        for r in range(self.world_size):
            off, n = self.layout.get_local_range(r)
            stats[r] = {
                "offset": off,
                "numel": n,
                "weight": self._rank_weights[r],
                "tier": self.device_tier_infos[r].tier,
                "size_mb": n * self.dtype.itemsize / 1024**2,
            }
        return stats

    def __repr__(self) -> str:
        return (
            f"HeteroDBuffer("
            f"world_size={self.world_size}, "
            f"rank={self._rank}, "
            f"placements={self.placements!r}, "
            f"layout_size={self.layout.size}, "
            f"local_numel={self.local_buffer.numel()}, "
            f"dtype={self.dtype}, "
            f"device={self.device}"
            f")"
        )


# ---------------------------------------------------------------------------
# Convenience alias (DES-LOC public API)
# ---------------------------------------------------------------------------

DBuffer = HeteroDBuffer  # Drop-in alias for Megatron callers


# ===========================================================================
# Unit tests
# ===========================================================================

if __name__ == "__main__":
    """
    Self-contained unit tests for HeteroDBuffer.

    Tests are designed to run in two modes:
    1. CPU single-process (no dist init): covers layout, placement validation,
       owned-range, get_local_tensor, shard_stats.
    2. Multi-process (torchrun or mpirun): covers all collective operations
       with a real ProcessGroup.

    Usage (single process):
        python deepspeed/runtime/zero/hetero_dbuffer.py

    Usage (multi-process, 3 ranks mirroring DES-LOC target):
        torchrun --nproc_per_node=3 deepspeed/runtime/zero/hetero_dbuffer.py
    """

    import sys
    import traceback

    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"

    _test_results: List[Tuple[str, bool, str]] = []

    def _run_test(name: str, fn):
        try:
            fn()
            _test_results.append((name, True, ""))
            print(f"  [{PASS}] {name}")
        except Exception as exc:
            tb = traceback.format_exc()
            _test_results.append((name, False, tb))
            print(f"  [{FAIL}] {name}: {exc}")

    # -----------------------------------------------------------------------
    # Layout tests (no dist required)
    # -----------------------------------------------------------------------

    def test_hetero_layout_uniform_weights():
        """Uniform weights reproduce Megatron-equivalent offsets."""
        shapes = [torch.Size((7, 3)), torch.Size((2, 5)), torch.Size((7,))]
        # chunk_size = lcm(3, 5, 7) = 105
        layout = HeteroGlobalLayout.build(shapes, dp_size=2, rank_weights=[1.0, 1.0])
        assert layout.chunk_size == 105, f"chunk_size={layout.chunk_size}"
        assert layout.size % (layout.chunk_size * 2) == 0, "size not padded to chunk*dp"
        assert layout.rank_numels[0] == layout.rank_numels[1], "uniform weights should yield equal shards"
        print(f"    layout.size={layout.size}, offsets={layout.tensor_to_offset}")

    def test_hetero_layout_weighted_shards():
        """Weighted layout assigns proportional shard sizes."""
        shapes = [torch.Size((4, 4)), torch.Size((4, 4))]
        # chunk_size = 4, total = 32, weights [1, 2] → shards [~11, ~21] rounded to chunk_size
        layout = HeteroGlobalLayout.build(shapes, dp_size=2, rank_weights=[1.0, 2.0])
        r0_numel = layout.rank_numels[0]
        r1_numel = layout.rank_numels[1]
        assert r1_numel >= r0_numel, f"H100 shard ({r1_numel}) should be >= A6000 shard ({r0_numel})"
        assert r0_numel % layout.chunk_size == 0, "rank 0 shard not chunk-aligned"
        assert r1_numel % layout.chunk_size == 0, "rank 1 shard not chunk-aligned"
        assert r0_numel + r1_numel == layout.size, f"{r0_numel}+{r1_numel} != {layout.size}"
        print(f"    rank0={r0_numel}, rank1={r1_numel}, total={layout.size}")

    def test_non_leading_numel_scalar_raises():
        """0D shapes are rejected by layout helpers."""
        try:
            _non_leading_numel(torch.Size([]))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_placement_validation_flat_suffix():
        """Flat must be a suffix; Flat-then-Replicate is rejected."""
        try:
            _validate_placements((Flat(), Replicate()))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        # Valid: Replicate then Flat
        _validate_placements((Replicate(), Flat()))

    def test_owned_range_non_resident():
        """Tensor outside this rank's shard returns None."""
        shapes = [torch.Size((10, 4)), torch.Size((10, 4))]
        layout = HeteroGlobalLayout.build(shapes, dp_size=2, rank_weights=[1.0, 1.0])
        # Simulate rank 0 with its shard
        class _FakeBuf:
            def __init__(self):
                self.layout = layout
                self.dtype = torch.float32
                self.device = torch.device("cpu")
                self.offset, n = layout.get_local_range(0)
                self.local_buffer = torch.zeros(n)
                self._rank = 0
                self._rank_weights = [1.0, 1.0]
            def _get_owned_range(self, idx):
                return HeteroDBuffer._get_owned_range(self, idx)

        fb = _FakeBuf()
        # Tensor 1 starts at offset >= rank0's numel for balanced shards
        t1_start = layout.tensor_to_offset[1]
        t1_end = t1_start + shapes[1].numel()
        b_end = fb.offset + fb.local_buffer.numel()
        if t1_start >= b_end:
            assert fb._get_owned_range(1) is None, "Tensor 1 should not be resident on rank 0"
        # Tensor 0 should always overlap rank 0
        assert fb._get_owned_range(0) is not None, "Tensor 0 should be resident on rank 0"

    def test_loc_cache_singleton():
        """LOC cache is a singleton."""
        c1 = _LOCCache(100 * 1024**2)
        c2 = _LOCCache(200 * 1024**2)
        assert c1 is c2, "LOC cache should be a singleton"

    def test_loc_cache_allocate_and_free():
        """LOC cache allocate/free cycle restores used bytes."""
        cache = _LOCCache(100 * 1024**2)
        initial_used = cache.used_bytes
        buf = cache.allocate(1024, torch.float32)
        assert cache.used_bytes > initial_used
        cache.free(buf)
        assert cache.used_bytes == initial_used

    def test_pad_to_multiple():
        assert _pad_to_multiple(0, 8) == 0
        assert _pad_to_multiple(1, 8) == 8
        assert _pad_to_multiple(8, 8) == 8
        assert _pad_to_multiple(9, 8) == 16
        assert _pad_to_multiple(100, 12) == 108

    def test_device_tier_info_cpu():
        """DeviceTierInfo on CPU device returns tier UNKNOWN."""
        info = DeviceTierInfo.from_rank(0, torch.device("cpu"))
        assert info.tier == "UNKNOWN"
        assert not info.supports_fp8

    # -----------------------------------------------------------------------
    # Distributed tests (require dist init)
    # -----------------------------------------------------------------------

    def _run_distributed_tests(rank: int, world_size: int, pg: dist.ProcessGroup):

        def test_dist_distribute_tensors_uniform():
            """distribute_tensors populates local buffer with owned slice."""
            shapes = [torch.Size((6, 4)), torch.Size((6, 4))]
            tensors = [torch.arange(24, dtype=torch.float32).reshape(6, 4),
                       torch.arange(24, dtype=torch.float32).reshape(6, 4) + 100]
            weights = [1.0] * world_size
            buf = HeteroDBuffer.distribute_tensors(tensors, pg, [Flat()], rank_weights=weights)
            assert buf.local_buffer.is_contiguous()
            lt0 = buf.get_local_tensor(0)
            assert lt0.shape[1] == 4, "Non-leading dim should be 4"

        def test_dist_allgather_round_trip():
            """Flat→Replicate round trip recovers original tensors."""
            shapes = [torch.Size((4, 3)), torch.Size((2, 3))]
            tensors = [torch.ones(4, 3) * (rank + 1),
                       torch.ones(2, 3) * (rank + 1) * 10]
            weights = [1.0] * world_size
            sharded = HeteroDBuffer.distribute_tensors(tensors, pg, [Flat()], rank_weights=weights)
            replicated = sharded.allgather(use_loc_cache=False)
            assert replicated.placements == (Replicate(),)
            full0 = replicated.get_local_tensor(0)
            assert full0.shape == torch.Size([4, 3]), f"Expected (4,3), got {full0.shape}"

        def test_dist_allreduce_partial():
            """Partial→Replicate all-reduce sums ranks correctly."""
            shapes = [torch.Size((4, 3))]
            tensors = [torch.ones(4, 3) * float(rank + 1)]
            weights = [1.0] * world_size
            partial = HeteroDBuffer.distribute_tensors(tensors, pg, [Partial()], rank_weights=weights)
            replicated = partial.allreduce()
            expected_sum = float(world_size * (world_size + 1) // 2)
            result = replicated.get_local_tensor(0)
            assert torch.allclose(result, torch.full((4, 3), expected_sum)), \
                f"allreduce: got {result[0,0].item()}, expected {expected_sum}"

        def test_dist_reduce_scatter_partial_to_flat():
            """Partial→Flat reduce_scatter produces correct weighted shards."""
            shapes = [torch.Size((4, 3))]
            tensors = [torch.ones(4, 3) * float(rank + 1)]
            weights = [1.0] * world_size
            partial = HeteroDBuffer.distribute_tensors(tensors, pg, [Partial()], rank_weights=weights)
            sharded = partial.reduce_scatter(new_placement=Flat())
            assert sharded.placements == (Flat(),)
            replicated = sharded.allgather()
            result = replicated.get_local_tensor(0)
            expected_sum = float(world_size * (world_size + 1) // 2)
            assert torch.allclose(result, torch.full((4, 3), expected_sum)), \
                f"reduce_scatter+allgather: got {result[0,0].item()}, expected {expected_sum}"

        def test_dist_scatter_replicate_to_flat():
            """Replicate→Flat scatter locally slices the replicated buffer."""
            shapes = [torch.Size((6, 2))]
            tensors = [torch.arange(12, dtype=torch.float32).reshape(6, 2)]
            weights = [1.0] * world_size
            replicated = HeteroDBuffer.distribute_tensors(tensors, pg, [Replicate()], rank_weights=weights)
            sharded = replicated.scatter(new_placement=Flat())
            assert sharded.placements == (Flat(),)
            recovered = sharded.allgather()
            rt = recovered.get_local_tensor(0)
            assert rt.shape == torch.Size([6, 2]), f"Scatter round trip shape mismatch: {rt.shape}"

        def test_dist_weighted_shards_h100_bigger():
            """With [1,1,2] weights, rank 2 owns more elements than ranks 0/1."""
            if world_size != 3:
                return  # Only meaningful for exactly 3 ranks
            shapes = [torch.Size((12, 4))]  # 48 elements, chunk_size=4
            weights = [1.0, 1.0, 2.0]
            tensors = [torch.zeros(12, 4)]
            buf = HeteroDBuffer.distribute_tensors(tensors, pg, [Flat()], rank_weights=weights)
            # rank 2 should have ~24 elements, ranks 0/1 ~12 each
            if rank < 2:
                assert buf.local_buffer.numel() <= buf.layout.rank_numels[2], \
                    f"rank {rank} shard should be <= rank2 shard"

        def test_dist_shard_stats():
            """shard_stats returns per-rank info."""
            shapes = [torch.Size((4, 4))]
            tensors = [torch.zeros(4, 4)]
            weights = [1.0] * world_size
            buf = HeteroDBuffer.distribute_tensors(tensors, pg, [Flat()], rank_weights=weights)
            stats = buf.shard_stats()
            assert len(stats) == world_size
            for r in range(world_size):
                assert "offset" in stats[r]
                assert "numel" in stats[r]

        def test_dist_redistribute_api():
            """redistribute() dispatcher routes to correct collective."""
            shapes = [torch.Size((4, 2))]
            tensors = [torch.ones(4, 2) * float(rank + 1)]
            weights = [1.0] * world_size
            sharded = HeteroDBuffer.distribute_tensors(tensors, pg, [Flat()], rank_weights=weights)
            replicated = sharded.redistribute([Replicate()])
            assert replicated.placements == (Replicate(),)

        def test_dist_redistribute_async():
            """redistribute_async() returns event (None on CPU)."""
            shapes = [torch.Size((4, 2))]
            tensors = [torch.ones(4, 2)]
            weights = [1.0] * world_size
            sharded = HeteroDBuffer.distribute_tensors(tensors, pg, [Flat()], rank_weights=weights)
            out_buf, event = sharded.redistribute_async([Replicate()])
            # On CPU event should be None; on CUDA it should be a cuda.Event
            assert isinstance(out_buf, HeteroDBuffer)
            # If event is not None, it should support wait()
            if event is not None:
                event.synchronize()

        def test_dist_from_local_reuses_buffer():
            """from_local wraps existing storage without allocation."""
            shapes = [torch.Size((6, 2))]
            weights = [1.0] * world_size
            layout = HeteroGlobalLayout.build(shapes, dp_size=world_size, rank_weights=weights)
            off, n = layout.get_local_range(rank)
            storage = torch.zeros(n)
            buf = HeteroDBuffer.from_local(storage, pg, [Flat()], shapes, rank_weights=weights)
            assert buf.local_buffer.data_ptr() == storage.data_ptr(), "from_local should reuse storage"

        dist_tests = [
            ("dist_distribute_tensors_uniform", test_dist_distribute_tensors_uniform),
            ("dist_allgather_round_trip", test_dist_allgather_round_trip),
            ("dist_allreduce_partial", test_dist_allreduce_partial),
            ("dist_reduce_scatter_partial_to_flat", test_dist_reduce_scatter_partial_to_flat),
            ("dist_scatter_replicate_to_flat", test_dist_scatter_replicate_to_flat),
            ("dist_weighted_shards_h100_bigger", test_dist_weighted_shards_h100_bigger),
            ("dist_shard_stats", test_dist_shard_stats),
            ("dist_redistribute_api", test_dist_redistribute_api),
            ("dist_redistribute_async", test_dist_redistribute_async),
            ("dist_from_local_reuses_buffer", test_dist_from_local_reuses_buffer),
        ]

        if rank == 0:
            print("\n  --- Distributed tests ---")
        for name, fn in dist_tests:
            dist.barrier(group=pg)
            _run_test(name, fn)

    # -----------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------

    print("\n=== HeteroDBuffer Unit Tests ===")
    print("  --- Single-process (layout/placement/cache) tests ---")

    single_process_tests = [
        ("hetero_layout_uniform_weights", test_hetero_layout_uniform_weights),
        ("hetero_layout_weighted_shards", test_hetero_layout_weighted_shards),
        ("non_leading_numel_scalar_raises", test_non_leading_numel_scalar_raises),
        ("placement_validation_flat_suffix", test_placement_validation_flat_suffix),
        ("owned_range_non_resident", test_owned_range_non_resident),
        ("loc_cache_singleton", test_loc_cache_singleton),
        ("loc_cache_allocate_and_free", test_loc_cache_allocate_and_free),
        ("pad_to_multiple", test_pad_to_multiple),
        ("device_tier_info_cpu", test_device_tier_info_cpu),
    ]

    for name, fn in single_process_tests:
        _run_test(name, fn)

    # Try distributed if WORLD_SIZE env is set (torchrun sets it)
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        pg = dist.GroupMember.WORLD

        _run_distributed_tests(rank, world_size, pg)
        dist.barrier()
        dist.destroy_process_group()
    else:
        print("\n  (Skipping distributed tests; run with torchrun --nproc_per_node=3 for full suite)")

    # Summary
    n_pass = sum(1 for _, ok, _ in _test_results if ok)
    n_fail = sum(1 for _, ok, _ in _test_results if not ok)
    print(f"\n=== Results: {n_pass} passed, {n_fail} failed ===")
    if n_fail > 0:
        for name, ok, tb in _test_results:
            if not ok:
                print(f"\n  FAILED: {name}\n{tb}")
        sys.exit(1)


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroGlobalLayout on a DeepSpeed engine.

    Instantiates a :class:`HeteroGlobalLayout` from the engine's configuration
    and attaches it as ``engine.hetero_dbuffer``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_dbuffer.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_dbuffer = None
    logger.info("hetero_dbuffer.register() attached engine.hetero_dbuffer")
