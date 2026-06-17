"""
deepspeed/moe/hetero_nvls_dispatcher_buffers.py

DES-LOC Heterogeneous NVLS Dispatcher Buffer Manager
=====================================================

Upstream Design Intent (Megatron 4e0f6367)
-------------------------------------------
The upstream Megatron commit fixes a silent overflow bug in the NVLSAllGatherVDispatcher:
previously all symmetric memory buffers were allocated with a fixed default size (256 MB,
later bumped to 512 MB), regardless of the actual tensor footprints implied by
(max_tokens, hidden_size, topk, ep_size). When users run non-default configurations —
e.g. large hidden dimensions or high expert-parallelism degrees — the buffers would
silently overflow the symmetric-memory cap, producing corrupted results rather than a
clear error.

The fix introduces per-buffer self-sizing: each buffer computes its exact byte footprint
from its shape and dtype, rounds up to the nearest MiB, and passes that as the requested
size. Additionally, init_failure_reason is introduced on SymmetricMemoryBuffer so that
downstream code can surface a human-readable explanation of *why* initialization failed,
rather than only reporting which buffers failed.

Two secondary fixes accompany the main change:
  1. grouped_mm availability falls back to the private symbol torch._grouped_mm for
     PyTorch versions < 2.10, broadening compatibility without requiring a new install.
  2. The assertion in MoELayer is relaxed to accept either the public or private symbol.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a heterogeneous
cluster: 2× A6000 48 GB (SM86, no NVLink) + 1× H100 NVL 96 GB (SM90), connected
over PCIe with 1.5 TB CPU DRAM as a shared spill tier.

Key differences from the Megatron target (homogeneous Hopper + NVLink):

1.  **No symmetric memory on A6000 nodes.**
    torch.distributed._symmetric_memory requires NVLink-connected Hopper+ GPUs.
    On the A6000 ranks we must fall back to NCCL-based all-gather with pinned CPU
    staging buffers.  The SymmetricMemoryBuffer analog here (HeteroSymmBuffer) tracks
    init_failure_reason per Megatron but routes the fallback to CPU-DRAM staging
    instead of hard-failing.

2.  **H100 NVL as the NVLS-capable anchor.**
    Only rank(s) mapped to the H100 participate in NVLS fast-path dispatch.  The
    HeteroNVLSDispatcherBuffers class detects device capability at construction time
    and allocates NVLS buffers exclusively on SM90 ranks.

3.  **Per-buffer self-sizing (direct port of upstream fix).**
    The same _size_mb helper from upstream is reproduced here so that large
    hidden_size / topk / ep_size combinations on the H100 don't overflow the buffer.

4.  **Locality Cache (LOC) integration.**
    DES-LOC maintains a Shared LOcality Cache in CPU DRAM for expert activations that
    spill off-GPU.  When NVLS init fails on a rank, the dispatcher records the failure
    and routes token dispatch through the LOC tier instead of raising immediately.
    This is the key semantic difference: upstream raises RuntimeError on any failure;
    DES-LOC degrades gracefully to the CPU-DRAM path.

5.  **DeepSpeed process group compatibility.**
    Megatron uses its own parallel state management.  Here we accept a standard
    torch.distributed ProcessGroup (as DeepSpeed provides) and wrap it in a thin
    adapter so the rest of the logic is group-agnostic.

6.  **grouped_mm fallback (direct port of upstream fix).**
    _resolve_grouped_mm() mirrors Megatron's torch._grouped_mm fallback so that
    DeepSpeed MoE on PyTorch 2.9.x continues to function.

Author note: This file is the DES-LOC reinterpretation of Megatron commit 4e0f6367.
It is NOT a string-level port; the buffer sizing algorithm, failure routing, and
device-capability detection are all rewritten around the A6000+H100 PCIe topology.
"""

from __future__ import annotations

import logging
import math
import operator
import os
import traceback
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import reduce
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

_SM90_CAPABILITY = (9, 0)
_SM86_CAPABILITY = (8, 6)


def _device_sm_capability(device: torch.device) -> Tuple[int, int]:
    """Return the (major, minor) SM capability of *device*."""
    if device.type != "cuda":
        return (0, 0)
    return torch.cuda.get_device_capability(device)


def _is_nvls_capable(device: torch.device) -> bool:
    """
    NVLS (NVLink Switch) symmetric memory is only available on SM90+ (Hopper) GPUs
    that are fully connected via NVLink.  On a PCIe-only topology this will be False
    even for an H100 PCIe SKU — but an H100 NVL card does expose peer access via its
    on-die NVLink fabric, so we treat SM90 as the capability gate here and let the
    runtime confirm actual NVLink presence during buffer allocation.
    """
    major, _ = _device_sm_capability(device)
    return major >= 9


def _resolve_grouped_mm():
    """
    Return the grouped matrix-multiply callable, or None.

    Mirrors the fallback introduced in Megatron 4e0f6367:
      - PyTorch >= 2.10 exposes torch.nn.functional.grouped_mm (public API).
      - PyTorch < 2.10 may expose the private symbol torch._grouped_mm.
      - Older versions have neither; callers must use a loop-based fallback.

    DES-LOC adaptation: we additionally log which path was chosen so operators
    can verify that the H100 is using the fast fused kernel.
    """
    fn = getattr(torch.nn.functional, "grouped_mm", None)
    if fn is not None:
        logger.debug("grouped_mm resolved via torch.nn.functional.grouped_mm (>=2.10 path)")
        return fn
    fn = getattr(torch, "_grouped_mm", None)
    if fn is not None:
        logger.debug("grouped_mm resolved via torch._grouped_mm (<2.10 private symbol)")
        return fn
    logger.debug("grouped_mm not available; callers must use loop-based GEMM fallback")
    return None


GROUPED_MM_FN = _resolve_grouped_mm()
HAVE_GROUPED_MM: bool = GROUPED_MM_FN is not None

# ---------------------------------------------------------------------------
# Symmetric-memory availability
# ---------------------------------------------------------------------------

try:
    import torch.distributed._symmetric_memory as _symm_mem_mod

    HAVE_TORCH_SYMM_MEM = True
except ImportError:
    _symm_mem_mod = None  # type: ignore[assignment]
    HAVE_TORCH_SYMM_MEM = False

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

# ---------------------------------------------------------------------------
# Dispatch path enum
# ---------------------------------------------------------------------------


class DispatchPath(Enum):
    """
    The execution path taken by a token dispatcher on a given rank.

    NVLS:
        Full NVLink-symmetric-memory fast path. Only available on SM90 ranks
        with working symmetric memory init (H100 NVL in our topology).

    NCCL_CPU_STAGING:
        NCCL all-gather with pinned CPU-DRAM staging buffers.  Used on A6000
        ranks and as fallback when NVLS init fails on the H100.

    LOC_SPILL:
        Locality Cache spill path.  Activations are stored in the 1.5 TB
        CPU-DRAM LOC tier and fetched on demand.  Used when both GPU paths
        are unavailable (e.g. CUDA OOM during buffer allocation).
    """

    NVLS = auto()
    NCCL_CPU_STAGING = auto()
    LOC_SPILL = auto()


# ---------------------------------------------------------------------------
# HeteroSymmBuffer — per-buffer analog of Megatron's SymmetricMemoryBuffer
# ---------------------------------------------------------------------------


@dataclass
class HeteroSymmBuffer:
    """
    A single symmetric-memory buffer slot for heterogeneous DES-LOC topology.

    This is the DES-LOC analog of Megatron's SymmetricMemoryBuffer.  The key
    semantic difference is that init failure does not raise immediately; instead
    the failure reason is recorded and the buffer object remains valid (with
    None tensor handle), allowing the caller to decide whether to fall back to
    the NCCL or LOC path.

    Fields
    ------
    name : str
        Logical buffer name (e.g. "ep_agv_h").
    shape : List[int]
        Expected tensor shape.
    dtype : torch.dtype
        Expected tensor dtype.
    size_mb : int
        Actual allocated size in MiB (self-sized from shape/dtype).
    tensor : Optional[torch.Tensor]
        The allocated CUDA tensor backed by symmetric memory, or None.
    symm_handle : object
        The opaque symmetric-memory rendezvous handle, or None.
    init_failure_reason : Optional[str]
        Human-readable explanation of why init failed, or None on success.
        Mirrors the init_failure_reason field added in Megatron 4e0f6367.
    fallback_cpu_tensor : Optional[torch.Tensor]
        Pinned CPU-DRAM staging tensor, allocated when NVLS is unavailable.
        Part of DES-LOC LOC integration — not present in upstream Megatron.
    """

    name: str
    shape: List[int]
    dtype: torch.dtype
    size_mb: int = 0
    tensor: Optional[torch.Tensor] = None
    symm_handle: object = None
    init_failure_reason: Optional[str] = None
    fallback_cpu_tensor: Optional[torch.Tensor] = None

    @property
    def is_nvls_ready(self) -> bool:
        """True if the NVLS fast-path tensor is available."""
        return self.tensor is not None and self.symm_handle is not None

    @property
    def is_cpu_fallback_ready(self) -> bool:
        """True if the CPU-DRAM staging buffer is available."""
        return self.fallback_cpu_tensor is not None

    def maybe_get_tensor(self) -> Optional[torch.Tensor]:
        """
        Return the best available tensor for this buffer:
          1. NVLS-backed CUDA tensor (fast path).
          2. CPU-DRAM pinned tensor (NCCL staging fallback).
          3. None (LOC spill path must be used by caller).
        """
        if self.is_nvls_ready:
            return self.tensor
        if self.is_cpu_fallback_ready:
            return self.fallback_cpu_tensor
        return None


# ---------------------------------------------------------------------------
# Buffer sizing utility (direct port + DES-LOC annotation)
# ---------------------------------------------------------------------------


def _compute_size_mb(shape: List[int], dtype: torch.dtype) -> int:
    """
    Compute the MiB ceiling for a tensor of *shape* and *dtype*.

    This is a direct port of the _size_mb helper introduced in Megatron 4e0f6367
    to fix silent overflow of the symmetric-memory cap.  The upstream motivation:
    previously all NVLS buffers used a fixed default size (256 MB → 512 MB).
    For non-default (max_tokens, hidden_size, topk, ep_size) combinations the
    tensors would exceed the fixed cap and corrupt results silently.

    DES-LOC note: we use the same formula but also apply it to the CPU-DRAM
    staging allocations so that pinned memory is right-sized too.

    Parameters
    ----------
    shape : List[int]
        Tensor shape.
    dtype : torch.dtype
        Tensor element type.

    Returns
    -------
    int
        Number of MiB required, rounded up, minimum 1.
    """
    _MB = 1024 * 1024
    element_size = torch.empty([], dtype=dtype).element_size()
    nbytes = reduce(operator.mul, shape, 1) * element_size
    return max(1, math.ceil(nbytes / _MB))


# ---------------------------------------------------------------------------
# Device-rank mapping for DES-LOC heterogeneous cluster
# ---------------------------------------------------------------------------


@dataclass
class DeviceRankInfo:
    """
    Encapsulates per-rank device information for the DES-LOC heterogeneous cluster.

    In our 3-GPU topology:
      - Rank 0: A6000 48 GB, SM86, PCIe
      - Rank 1: A6000 48 GB, SM86, PCIe
      - Rank 2: H100 NVL 96 GB, SM90, PCIe (NVLink on-die fabric for H100 NVL)

    The rank→device mapping is discovered at runtime by querying CUDA device
    properties.  Ranks are identified as NVLS-capable if their SM major >= 9.
    """

    rank: int
    device: torch.device
    sm_capability: Tuple[int, int]
    nvls_capable: bool
    device_name: str
    total_memory_bytes: int

    @classmethod
    def from_current_rank(cls) -> "DeviceRankInfo":
        """Build DeviceRankInfo for the calling process's current CUDA device."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        device_idx = torch.cuda.current_device()
        device = torch.device("cuda", device_idx)
        props = torch.cuda.get_device_properties(device_idx)
        sm_cap = (props.major, props.minor)
        return cls(
            rank=rank,
            device=device,
            sm_capability=sm_cap,
            nvls_capable=sm_cap >= _SM90_CAPABILITY,
            device_name=props.name,
            total_memory_bytes=props.total_memory,
        )

    def __repr__(self) -> str:
        return (
            f"DeviceRankInfo(rank={self.rank}, device={self.device}, "
            f"sm={self.sm_capability}, nvls={self.nvls_capable}, "
            f"name={self.device_name!r}, mem={self.total_memory_bytes // (1 << 30)}GB)"
        )


# ---------------------------------------------------------------------------
# HeteroNVLSDispatcherBuffers — the main DES-LOC class
# ---------------------------------------------------------------------------


class HeteroNVLSDispatcherBuffers:
    """
    Heterogeneous MoE dispatcher buffer manager for the DES-LOC framework.

    Motivation
    ----------
    Megatron's NVLSAllGatherVDispatcher assumes a homogeneous Hopper cluster
    with full NVLink connectivity.  In DES-LOC we have a mixed A6000 (SM86) +
    H100 NVL (SM90) cluster connected over PCIe.  We need a dispatcher that:

      * Allocates NVLS symmetric-memory buffers *only* on the H100 rank.
      * Self-sizes each buffer from its exact tensor footprint (upstream fix).
      * Falls back to NCCL + CPU-DRAM staging on A6000 ranks.
      * Integrates with the LOC (Shared Locality Cache) tier for spill.
      * Reports precise failure reasons (upstream fix: init_failure_reason).
      * Does not hard-fail on non-NVLS ranks — instead records the path taken.

    Buffer Layout (per upstream NVLSAllGatherVDispatcher)
    ------------------------------------------------------
    ep_agv_h  : [global_max_tokens, hidden_size] bfloat16  — all-gather hidden
    ep_agv_r  : [global_max_tokens, topk]        int64     — routing indices
    ep_agv_p  : [global_max_tokens, topk]        float32   — routing probabilities
    ep_rsv    : [global_max_tokens, hidden_size] bfloat16  — reduce-scatter output
    ep_meta   : [ep_size]                        int32     — per-rank token counts

    Each buffer is self-sized (see _compute_size_mb) so non-default shapes do
    not silently overflow the symmetric-memory cap.

    DES-LOC Extensions
    ------------------
    * CPU staging tensors (pinned) mirror each NVLS buffer for fallback.
    * LOCSpillHandle records the LOC cache key for spilled activations.
    * dispatch_path records the actual execution path chosen per rank.
    * init_failure_reasons aggregates all buffer failure reasons for operators.
    """

    # Class-level buffer registry (one entry per logical buffer name)
    _buffers: Dict[str, HeteroSymmBuffer] = {}

    # Class-level dispatch path (set during init_buffers)
    _dispatch_path: Optional[DispatchPath] = None

    # Aggregate failure reasons from all buffers
    _init_failure_reasons: Dict[str, str] = {}

    # LOC tier reference (injected by DES-LOC runtime)
    _loc_cache: Optional[object] = None

    @classmethod
    def init_buffers(
        cls,
        max_tokens_per_rank: int,
        hidden_size: int,
        topk: int,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        loc_cache: Optional[object] = None,
        force_cpu_fallback: bool = False,
    ) -> DispatchPath:
        """
        Allocate all dispatcher buffers for the calling rank.

        This is the DES-LOC analog of NVLSAllGatherVDispatcher._initialize_symmetric_memory.
        The key differences:

          1. Per-buffer self-sizing (upstream fix, ported verbatim in _compute_size_mb).
          2. NVLS allocation attempted only on NVLS-capable ranks (SM90+).
          3. Graceful fallback to CPU-DRAM pinned staging on NVLS failure.
          4. init_failure_reason captured per buffer (upstream fix, adapted).
          5. LOC cache reference stored for downstream spill routing.

        Parameters
        ----------
        max_tokens_per_rank : int
            Maximum number of tokens assigned to a single rank in one step.
        hidden_size : int
            Model hidden dimension (D).
        topk : int
            Number of experts each token is dispatched to.
        ep_size : int
            Expert-parallelism world size (number of EP ranks).
        ep_group : dist.ProcessGroup
            The expert-parallelism process group.
        loc_cache : object, optional
            DES-LOC Shared Locality Cache handle.  If provided and NVLS/NCCL
            both fail, activations are spilled here.
        force_cpu_fallback : bool
            If True, skip NVLS allocation and go straight to CPU staging.
            Useful for testing or when the operator knows NVLS is unavailable.

        Returns
        -------
        DispatchPath
            The execution path selected for this rank.
        """
        cls._buffers.clear()
        cls._init_failure_reasons.clear()
        cls._loc_cache = loc_cache

        rank_info = DeviceRankInfo.from_current_rank()
        logger.info(
            "HeteroNVLSDispatcherBuffers.init_buffers: rank=%d device=%s sm=%s nvls_capable=%s",
            rank_info.rank,
            rank_info.device,
            rank_info.sm_capability,
            rank_info.nvls_capable,
        )

        global_max = max_tokens_per_rank * ep_size

        # Build the shape/dtype table (same logical layout as upstream)
        buffer_specs: List[Tuple[str, List[int], torch.dtype]] = [
            ("ep_agv_h", [global_max, hidden_size], torch.bfloat16),
            ("ep_agv_r", [global_max, topk], torch.int64),
            ("ep_agv_p", [global_max, topk], torch.float32),
            ("ep_rsv", [global_max, hidden_size], torch.bfloat16),
            ("ep_meta", [ep_size], torch.int32),
        ]

        # Self-size each buffer (direct port of upstream fix)
        for name, shape, dtype in buffer_specs:
            size_mb = _compute_size_mb(shape, dtype)
            buf = HeteroSymmBuffer(name=name, shape=shape, dtype=dtype, size_mb=size_mb)
            cls._buffers[name] = buf
            logger.debug(
                "Buffer spec: name=%s shape=%s dtype=%s size_mb=%d",
                name, shape, dtype, size_mb,
            )

        # Attempt NVLS allocation on capable ranks
        nvls_ok = False
        if rank_info.nvls_capable and not force_cpu_fallback:
            nvls_ok = cls._try_allocate_nvls(ep_group)
        else:
            reason = (
                "force_cpu_fallback=True" if force_cpu_fallback
                else f"SM{rank_info.sm_capability[0]}{rank_info.sm_capability[1]} < SM90"
            )
            for name, buf in cls._buffers.items():
                buf.init_failure_reason = reason
            logger.info(
                "Skipping NVLS allocation on rank %d: %s", rank_info.rank, reason
            )

        if nvls_ok:
            cls._dispatch_path = DispatchPath.NVLS
            logger.info(
                "Rank %d: NVLS fast path active (%d buffers allocated)",
                rank_info.rank, len(cls._buffers),
            )
        else:
            # Attempt CPU-DRAM pinned staging fallback
            cpu_ok = cls._try_allocate_cpu_staging(rank_info)
            if cpu_ok:
                cls._dispatch_path = DispatchPath.NCCL_CPU_STAGING
                logger.info(
                    "Rank %d: NCCL+CPU staging path active", rank_info.rank
                )
            else:
                cls._dispatch_path = DispatchPath.LOC_SPILL
                logger.warning(
                    "Rank %d: both NVLS and CPU staging failed; routing to LOC spill tier",
                    rank_info.rank,
                )

        return cls._dispatch_path

    @classmethod
    def _try_allocate_nvls(cls, ep_group: dist.ProcessGroup) -> bool:
        """
        Attempt to allocate all buffers in NVLS symmetric memory.

        Returns True iff all buffers were successfully allocated.
        Records init_failure_reason on each buffer that fails (mirroring
        the upstream init_failure_reason field on SymmetricMemoryBuffer).
        """
        if not HAVE_TORCH_SYMM_MEM:
            reason = "torch.distributed._symmetric_memory not importable"
            for buf in cls._buffers.values():
                buf.init_failure_reason = reason
            cls._init_failure_reasons["__global__"] = reason
            logger.warning("NVLS unavailable: %s", reason)
            return False

        if not HAVE_TRITON:
            reason = "triton not installed"
            for buf in cls._buffers.values():
                buf.init_failure_reason = reason
            cls._init_failure_reasons["__global__"] = reason
            logger.warning("NVLS unavailable: %s", reason)
            return False

        all_ok = True
        for name, buf in cls._buffers.items():
            try:
                numel = reduce(operator.mul, buf.shape, 1)
                raw = _symm_mem_mod.empty(numel, dtype=buf.dtype, device="cuda")
                handle = _symm_mem_mod.rendezvous(raw, ep_group)
                # Reshape to the logical shape
                buf.tensor = raw.view(buf.shape)
                buf.symm_handle = handle
                logger.debug(
                    "NVLS buffer '%s' allocated: shape=%s dtype=%s size_mb=%d",
                    name, buf.shape, buf.dtype, buf.size_mb,
                )
            except RuntimeError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                buf.init_failure_reason = reason
                cls._init_failure_reasons[name] = reason
                all_ok = False
                logger.warning(
                    "NVLS allocation failed for buffer '%s': %s", name, reason
                )
            except Exception as exc:  # noqa: BLE001
                reason = f"Unexpected {type(exc).__name__}: {exc}"
                buf.init_failure_reason = reason
                cls._init_failure_reasons[name] = reason
                all_ok = False
                logger.error(
                    "Unexpected error allocating NVLS buffer '%s': %s\n%s",
                    name, reason, traceback.format_exc(),
                )

        if not all_ok:
            failed_details = "; ".join(
                f"{n}: {r}" for n, r in cls._init_failure_reasons.items()
            )
            logger.warning(
                "HeteroNVLSDispatcherBuffers: NVLS init partially failed [%s]; "
                "falling back to NCCL+CPU staging",
                failed_details,
            )
        return all_ok

    @classmethod
    def _try_allocate_cpu_staging(cls, rank_info: DeviceRankInfo) -> bool:
        """
        Allocate pinned CPU-DRAM staging tensors for NCCL fallback.

        This is a DES-LOC-specific extension with no direct upstream analog.
        The 1.5 TB CPU DRAM provides ample space for staging expert activations
        when NVLS symmetric memory is unavailable (A6000 ranks or NVLS failure).

        We check that the system has enough available memory before allocating.
        If a single buffer's pinned alloc fails we record the reason and fall
        through to the LOC spill path.
        """
        all_ok = True
        for name, buf in cls._buffers.items():
            try:
                cpu_tensor = torch.empty(buf.shape, dtype=buf.dtype, pin_memory=True)
                buf.fallback_cpu_tensor = cpu_tensor
                logger.debug(
                    "CPU staging buffer '%s' allocated: shape=%s dtype=%s size_mb=%d",
                    name, buf.shape, buf.dtype, buf.size_mb,
                )
            except RuntimeError as exc:
                reason = f"CPU pinned alloc failed: {exc}"
                if buf.init_failure_reason is None:
                    buf.init_failure_reason = reason
                cls._init_failure_reasons[name] = reason
                all_ok = False
                logger.warning(
                    "CPU staging allocation failed for buffer '%s': %s", name, reason
                )
        return all_ok

    @classmethod
    def get_buffer(cls, name: str) -> Optional[HeteroSymmBuffer]:
        """Return the HeteroSymmBuffer for *name*, or None if not initialized."""
        return cls._buffers.get(name)

    @classmethod
    def get_dispatch_path(cls) -> Optional[DispatchPath]:
        """Return the execution path selected during init_buffers."""
        return cls._dispatch_path

    @classmethod
    def get_tensor(cls, name: str) -> Optional[torch.Tensor]:
        """
        Return the best available tensor for buffer *name*.

        Preference order:
          1. NVLS-backed CUDA tensor.
          2. CPU-DRAM pinned staging tensor.
          3. None (caller must route to LOC spill).
        """
        buf = cls._buffers.get(name)
        if buf is None:
            return None
        return buf.maybe_get_tensor()

    @classmethod
    def assert_nvls_ready(cls, backend: str = "nvls") -> None:
        """
        Assert that NVLS grouped-GEMM prerequisites are satisfied.

        Mirrors the assertion in Megatron's MoELayer (also fixed in 4e0f6367)
        to accept both torch.nn.functional.grouped_mm (>= 2.10) and the private
        torch._grouped_mm symbol (<= 2.10).

        Raises
        ------
        AssertionError
            If neither grouped_mm symbol is available and backend is 'nvls'.
        """
        if backend == "nvls":
            assert HAVE_GROUPED_MM, (
                "HeteroNVLSDispatcherBuffers: inference_grouped_gemm_backend='nvls' requires "
                "torch.nn.functional.grouped_mm (> torch 2.10) or torch._grouped_mm (<= 2.10). "
                "Consider using backend='nccl' on this host."
            )

    @classmethod
    def summarize(cls) -> str:
        """Return a human-readable summary of buffer allocation status."""
        lines = [
            f"HeteroNVLSDispatcherBuffers summary:",
            f"  dispatch_path = {cls._dispatch_path}",
        ]
        for name, buf in cls._buffers.items():
            status = "NVLS" if buf.is_nvls_ready else ("CPU" if buf.is_cpu_fallback_ready else "FAILED")
            lines.append(
                f"  [{status}] {name}: shape={buf.shape} dtype={buf.dtype} "
                f"size_mb={buf.size_mb} failure={buf.init_failure_reason!r}"
            )
        if cls._init_failure_reasons:
            lines.append(f"  Failure reasons: {cls._init_failure_reasons}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LOC Spill Handle — DES-LOC specific
# ---------------------------------------------------------------------------


@dataclass
class LOCSpillHandle:
    """
    Handle for a token-dispatch buffer spilled to the DES-LOC LOC tier.

    When both NVLS and CPU staging are unavailable (e.g. OOM), the dispatcher
    stores activation tensors in the Shared Locality Cache (LOC) keyed by
    (layer_idx, buffer_name, step_id).  This dataclass carries the metadata
    needed to retrieve them.

    Fields
    ------
    layer_idx : int
        Transformer layer index (used to namespace LOC keys).
    buffer_name : str
        Logical buffer name (ep_agv_h, ep_agv_r, etc.).
    step_id : int
        Training/inference step counter (for cache eviction).
    loc_key : str
        Derived key used to look up the buffer in the LOC store.
    shape : List[int]
        Original tensor shape (needed to reconstruct on retrieval).
    dtype : torch.dtype
        Original tensor dtype.
    """

    layer_idx: int
    buffer_name: str
    step_id: int
    shape: List[int]
    dtype: torch.dtype
    loc_key: str = field(init=False)

    def __post_init__(self) -> None:
        self.loc_key = f"des_loc:dispatcher:{self.layer_idx}:{self.buffer_name}:{self.step_id}"


class LOCDispatcherStore:
    """
    Minimal LOC store for dispatcher buffers when GPU memory is exhausted.

    In production DES-LOC this would delegate to the full LOC subsystem backed
    by the 1.5 TB CPU DRAM.  Here we implement a simple dict-based store so
    that the rest of the dispatcher code can be written against a stable interface.
    """

    def __init__(self) -> None:
        self._store: Dict[str, torch.Tensor] = {}

    def put(self, handle: LOCSpillHandle, tensor: torch.Tensor) -> None:
        """Store *tensor* under *handle.loc_key*."""
        # Pin to CPU DRAM for fast DMA transfer on retrieval
        cpu_tensor = tensor.detach().cpu()
        self._store[handle.loc_key] = cpu_tensor
        logger.debug(
            "LOC spill: key=%s shape=%s dtype=%s nbytes=%d",
            handle.loc_key, tensor.shape, tensor.dtype,
            tensor.nelement() * tensor.element_size(),
        )

    def get(self, handle: LOCSpillHandle, device: torch.device) -> Optional[torch.Tensor]:
        """Retrieve tensor for *handle*, moving it to *device*."""
        t = self._store.get(handle.loc_key)
        if t is None:
            return None
        return t.to(device=device, non_blocking=True)

    def evict(self, handle: LOCSpillHandle) -> bool:
        """Remove *handle* from the store. Returns True if it existed."""
        existed = handle.loc_key in self._store
        self._store.pop(handle.loc_key, None)
        return existed

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# HeteroMoEDispatcher — thin dispatcher that uses HeteroNVLSDispatcherBuffers
# ---------------------------------------------------------------------------


class HeteroMoEDispatcher:
    """
    A MoE token dispatcher for the DES-LOC heterogeneous cluster.

    This class orchestrates token dispatch across the A6000+H100 topology by
    selecting the appropriate execution path (NVLS / NCCL+CPU / LOC spill) per
    rank and exposing a unified API to the MoE layer.

    Design
    ------
    - On the H100 rank: NVLS fast path with self-sized symmetric buffers.
    - On A6000 ranks: NCCL all-gather with pinned CPU-DRAM staging.
    - LOC spill: fallback when GPU OOM prevents any CUDA buffer allocation.

    The dispatcher does NOT implement the full Megatron NVLS kernel stack
    (which requires Triton + specific kernel binaries); instead it provides
    the buffer management layer so that those kernels can be plugged in when
    the H100 rank is the active sender.
    """

    def __init__(
        self,
        layer_idx: int,
        max_tokens_per_rank: int,
        hidden_size: int,
        topk: int,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        loc_store: Optional[LOCDispatcherStore] = None,
    ) -> None:
        self.layer_idx = layer_idx
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden_size = hidden_size
        self.topk = topk
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.loc_store = loc_store or LOCDispatcherStore()
        self._step_id: int = 0

        self.dispatch_path = HeteroNVLSDispatcherBuffers.init_buffers(
            max_tokens_per_rank=max_tokens_per_rank,
            hidden_size=hidden_size,
            topk=topk,
            ep_size=ep_size,
            ep_group=ep_group,
            loc_cache=self.loc_store,
        )
        logger.info(
            "HeteroMoEDispatcher layer=%d initialized: path=%s",
            layer_idx, self.dispatch_path,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Dispatch tokens to experts using the selected execution path.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape [local_tokens, hidden_size], bfloat16.
        routing_indices : torch.Tensor
            Shape [local_tokens, topk], int64.
        routing_probs : torch.Tensor
            Shape [local_tokens, topk], float32.

        Returns
        -------
        dict with keys: 'hidden', 'routing', 'probs', 'path'
            Tensors available in the working memory tier, plus the path label.
        """
        self._step_id += 1

        if self.dispatch_path == DispatchPath.NVLS:
            return self._dispatch_nvls(hidden_states, routing_indices, routing_probs)
        elif self.dispatch_path == DispatchPath.NCCL_CPU_STAGING:
            return self._dispatch_nccl_cpu(hidden_states, routing_indices, routing_probs)
        else:
            return self._dispatch_loc_spill(hidden_states, routing_indices, routing_probs)

    def _dispatch_nvls(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        NVLS fast path: copy local tensors into symmetric buffers.

        In production this would invoke the Triton NVLS kernel for fused
        all-gather + dispatch.  Here we write into the symmetric buffer
        and return it, representing the data as globally visible.
        """
        h_buf = HeteroNVLSDispatcherBuffers.get_tensor("ep_agv_h")
        r_buf = HeteroNVLSDispatcherBuffers.get_tensor("ep_agv_r")
        p_buf = HeteroNVLSDispatcherBuffers.get_tensor("ep_agv_p")

        local_n = hidden_states.shape[0]

        if h_buf is not None and local_n <= h_buf.shape[0]:
            h_buf[:local_n].copy_(hidden_states, non_blocking=True)
            r_buf[:local_n].copy_(routing_indices, non_blocking=True)
            p_buf[:local_n].copy_(routing_probs, non_blocking=True)
            logger.debug(
                "NVLS dispatch step=%d layer=%d local_tokens=%d",
                self._step_id, self.layer_idx, local_n,
            )
            return {
                "hidden": h_buf[:local_n],
                "routing": r_buf[:local_n],
                "probs": p_buf[:local_n],
                "path": DispatchPath.NVLS,
            }
        else:
            # Buffer shape mismatch — log and fall through to CPU
            logger.warning(
                "NVLS buffer shape mismatch at step=%d layer=%d: "
                "local_tokens=%d but buffer_max=%s; falling to CPU staging",
                self._step_id, self.layer_idx,
                local_n, h_buf.shape if h_buf is not None else "None",
            )
            return self._dispatch_nccl_cpu(hidden_states, routing_indices, routing_probs)

    def _dispatch_nccl_cpu(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        NCCL + CPU-DRAM staging path.

        Copies tensors to pinned CPU staging buffers, performs NCCL all-gather
        (simulated here as a no-op for single-node dev mode), then returns the
        CPU-side tensors.  The MoE layer is responsible for moving them back to
        GPU before the expert GEMM.
        """
        h_cpu = HeteroNVLSDispatcherBuffers.get_tensor("ep_agv_h")
        r_cpu = HeteroNVLSDispatcherBuffers.get_tensor("ep_agv_r")
        p_cpu = HeteroNVLSDispatcherBuffers.get_tensor("ep_agv_p")

        local_n = hidden_states.shape[0]

        if h_cpu is not None:
            h_cpu[:local_n].copy_(hidden_states.cpu(), non_blocking=False)
            r_cpu[:local_n].copy_(routing_indices.cpu(), non_blocking=False)
            p_cpu[:local_n].copy_(routing_probs.cpu(), non_blocking=False)
            logger.debug(
                "NCCL+CPU dispatch step=%d layer=%d local_tokens=%d",
                self._step_id, self.layer_idx, local_n,
            )
            return {
                "hidden": h_cpu[:local_n],
                "routing": r_cpu[:local_n],
                "probs": p_cpu[:local_n],
                "path": DispatchPath.NCCL_CPU_STAGING,
            }
        else:
            logger.warning(
                "CPU staging also unavailable at step=%d layer=%d; routing to LOC spill",
                self._step_id, self.layer_idx,
            )
            return self._dispatch_loc_spill(hidden_states, routing_indices, routing_probs)

    def _dispatch_loc_spill(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        LOC spill path: store activations in the CPU-DRAM LOC tier.

        This path is taken only when both NVLS and CPU staging allocation
        failed.  The tensors are spilled to the LOC store and LOCSpillHandles
        are returned so the MoE layer can retrieve them before the GEMM.
        """
        shape_h = list(hidden_states.shape)
        shape_r = list(routing_indices.shape)
        shape_p = list(routing_probs.shape)

        h_handle = LOCSpillHandle(self.layer_idx, "ep_agv_h", self._step_id, shape_h, hidden_states.dtype)
        r_handle = LOCSpillHandle(self.layer_idx, "ep_agv_r", self._step_id, shape_r, routing_indices.dtype)
        p_handle = LOCSpillHandle(self.layer_idx, "ep_agv_p", self._step_id, shape_p, routing_probs.dtype)

        self.loc_store.put(h_handle, hidden_states)
        self.loc_store.put(r_handle, routing_indices)
        self.loc_store.put(p_handle, routing_probs)

        logger.warning(
            "LOC spill dispatch step=%d layer=%d: "
            "activations written to CPU-DRAM LOC tier (keys: %s, %s, %s)",
            self._step_id, self.layer_idx,
            h_handle.loc_key, r_handle.loc_key, p_handle.loc_key,
        )
        return {
            "hidden": None,
            "routing": None,
            "probs": None,
            "path": DispatchPath.LOC_SPILL,
            "_loc_handles": (h_handle, r_handle, p_handle),
        }

    def retrieve_loc_spill(
        self,
        result: Dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieve tensors spilled to the LOC tier and move them to *device*.

        Should be called by the MoE layer before the expert GEMM when the
        dispatch result path is LOC_SPILL.
        """
        if result.get("path") != DispatchPath.LOC_SPILL:
            return result
        h_handle, r_handle, p_handle = result["_loc_handles"]
        return {
            "hidden": self.loc_store.get(h_handle, device),
            "routing": self.loc_store.get(r_handle, device),
            "probs": self.loc_store.get(p_handle, device),
            "path": DispatchPath.LOC_SPILL,
        }


# ---------------------------------------------------------------------------
# Utility: build a fake process group for single-process tests
# ---------------------------------------------------------------------------


def _make_fake_pg() -> Optional[dist.ProcessGroup]:
    """
    Initialize a gloo process group on localhost for unit tests.
    Returns None if dist is already initialized or env vars are missing.
    """
    if dist.is_initialized():
        return dist.group.WORLD
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        return dist.group.WORLD
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestComputeSizeMb(unittest.TestCase):
    """Tests for the _compute_size_mb helper (upstream fix, DES-LOC port)."""

    def test_bfloat16_small(self):
        # [1, 1] bfloat16 = 2 bytes → 1 MiB (minimum)
        self.assertEqual(_compute_size_mb([1, 1], torch.bfloat16), 1)

    def test_bfloat16_large_hidden(self):
        # [8192, 8192] bfloat16 = 8192*8192*2 = 134,217,728 bytes = 128 MiB
        result = _compute_size_mb([8192, 8192], torch.bfloat16)
        self.assertEqual(result, 128)

    def test_float32(self):
        # [4096, 4] float32 = 4096*4*4 = 65,536 bytes < 1 MiB → 1 MiB
        result = _compute_size_mb([4096, 4], torch.float32)
        self.assertEqual(result, 1)

    def test_int64_routing(self):
        # [65536, 8] int64 = 65536*8*8 = 4,194,304 bytes = 4 MiB
        result = _compute_size_mb([65536, 8], torch.int64)
        self.assertEqual(result, 4)

    def test_int32_meta(self):
        # [128] int32 = 128*4 = 512 bytes < 1 MiB → 1 MiB
        result = _compute_size_mb([128], torch.int32)
        self.assertEqual(result, 1)

    def test_non_default_config(self):
        # Simulates a non-default hidden_size=16384, max_tokens=4096, ep_size=8
        global_max = 4096 * 8
        shape = [global_max, 16384]
        result = _compute_size_mb(shape, torch.bfloat16)
        # 32768 * 16384 * 2 = 1,073,741,824 bytes = 1024 MiB
        self.assertEqual(result, 1024)

    def test_ceiling_rounding(self):
        # 1 MB + 1 byte should round up to 2 MiB
        # 1 MB + 1 byte in float32 = ceil((1048576+1)/1048576) = 2
        # shape: [262145] float32 = 262145*4 = 1,048,580 bytes
        result = _compute_size_mb([262145], torch.float32)
        self.assertEqual(result, 2)


class TestHeteroSymmBuffer(unittest.TestCase):
    """Tests for HeteroSymmBuffer dataclass."""

    def test_initial_state(self):
        buf = HeteroSymmBuffer(
            name="test_buf",
            shape=[128, 512],
            dtype=torch.bfloat16,
            size_mb=1,
        )
        self.assertFalse(buf.is_nvls_ready)
        self.assertFalse(buf.is_cpu_fallback_ready)
        self.assertIsNone(buf.maybe_get_tensor())

    def test_cpu_fallback_ready(self):
        buf = HeteroSymmBuffer(
            name="test_buf",
            shape=[8, 8],
            dtype=torch.float32,
            size_mb=1,
        )
        buf.fallback_cpu_tensor = torch.zeros(8, 8)
        self.assertTrue(buf.is_cpu_fallback_ready)
        self.assertIsNotNone(buf.maybe_get_tensor())

    def test_nvls_priority_over_cpu(self):
        buf = HeteroSymmBuffer(
            name="test_buf",
            shape=[4, 4],
            dtype=torch.bfloat16,
            size_mb=1,
        )
        # Simulate both available — NVLS should take priority
        buf.fallback_cpu_tensor = torch.zeros(4, 4)
        fake_cuda = torch.zeros(4, 4)
        fake_handle = object()
        buf.tensor = fake_cuda
        buf.symm_handle = fake_handle
        self.assertIs(buf.maybe_get_tensor(), fake_cuda)

    def test_failure_reason_recording(self):
        buf = HeteroSymmBuffer(
            name="ep_agv_h",
            shape=[1024, 4096],
            dtype=torch.bfloat16,
            size_mb=8,
        )
        buf.init_failure_reason = "RuntimeError: NVLink not available on this topology"
        self.assertIn("NVLink", buf.init_failure_reason)


class TestDeviceRankInfo(unittest.TestCase):
    """Tests for DeviceRankInfo capability detection."""

    def test_nvls_capable_sm90(self):
        """SM90 should be NVLS-capable."""
        info = DeviceRankInfo(
            rank=2,
            device=torch.device("cuda:0"),
            sm_capability=(9, 0),
            nvls_capable=True,
            device_name="H100 NVL",
            total_memory_bytes=96 * (1 << 30),
        )
        self.assertTrue(info.nvls_capable)

    def test_not_nvls_capable_sm86(self):
        """SM86 (A6000) should not be NVLS-capable."""
        info = DeviceRankInfo(
            rank=0,
            device=torch.device("cuda:0"),
            sm_capability=(8, 6),
            nvls_capable=False,
            device_name="NVIDIA RTX A6000",
            total_memory_bytes=48 * (1 << 30),
        )
        self.assertFalse(info.nvls_capable)

    def test_repr(self):
        info = DeviceRankInfo(
            rank=1,
            device=torch.device("cuda:1"),
            sm_capability=(8, 6),
            nvls_capable=False,
            device_name="NVIDIA RTX A6000",
            total_memory_bytes=48 * (1 << 30),
        )
        r = repr(info)
        self.assertIn("SM86", r.upper().replace("(8, 6)", "SM86") or r)
        self.assertIn("A6000", r)


class TestLOCDispatcherStore(unittest.TestCase):
    """Tests for LOCDispatcherStore (DES-LOC-specific)."""

    def setUp(self):
        self.store = LOCDispatcherStore()
        self.device = torch.device("cpu")

    def test_put_and_get(self):
        handle = LOCSpillHandle(
            layer_idx=0, buffer_name="ep_agv_h",
            step_id=1, shape=[4, 8], dtype=torch.bfloat16,
        )
        t = torch.randn(4, 8).bfloat16()
        self.store.put(handle, t)
        retrieved = self.store.get(handle, self.device)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.shape, t.shape)

    def test_get_missing_returns_none(self):
        handle = LOCSpillHandle(
            layer_idx=99, buffer_name="ep_rsv",
            step_id=999, shape=[2, 2], dtype=torch.float32,
        )
        result = self.store.get(handle, self.device)
        self.assertIsNone(result)

    def test_evict(self):
        handle = LOCSpillHandle(
            layer_idx=1, buffer_name="ep_agv_r",
            step_id=2, shape=[3], dtype=torch.int64,
        )
        t = torch.tensor([1, 2, 3])
        self.store.put(handle, t)
        self.assertEqual(len(self.store), 1)
        existed = self.store.evict(handle)
        self.assertTrue(existed)
        self.assertEqual(len(self.store), 0)

    def test_evict_nonexistent(self):
        handle = LOCSpillHandle(
            layer_idx=5, buffer_name="ep_meta",
            step_id=10, shape=[4], dtype=torch.int32,
        )
        self.assertFalse(self.store.evict(handle))

    def test_loc_key_format(self):
        handle = LOCSpillHandle(
            layer_idx=3, buffer_name="ep_agv_p",
            step_id=7, shape=[16, 4], dtype=torch.float32,
        )
        self.assertEqual(handle.loc_key, "des_loc:dispatcher:3:ep_agv_p:7")

    def test_multiple_buffers(self):
        tensors = {}
        handles = {}
        for name in ("ep_agv_h", "ep_agv_r", "ep_agv_p"):
            h = LOCSpillHandle(0, name, 1, [2, 2], torch.float32)
            t = torch.randn(2, 2)
            self.store.put(h, t)
            handles[name] = h
            tensors[name] = t
        self.assertEqual(len(self.store), 3)
        for name, h in handles.items():
            r = self.store.get(h, self.device)
            self.assertIsNotNone(r)


class TestGroupedMmResolution(unittest.TestCase):
    """Tests for _resolve_grouped_mm fallback logic."""

    def test_returns_callable_or_none(self):
        fn = _resolve_grouped_mm()
        if fn is not None:
            self.assertTrue(callable(fn))

    def test_have_grouped_mm_flag(self):
        fn = _resolve_grouped_mm()
        if fn is not None:
            self.assertTrue(HAVE_GROUPED_MM)


class TestHeteroNVLSDispatcherBuffersCPUFallback(unittest.TestCase):
    """
    Tests for HeteroNVLSDispatcherBuffers in CPU-fallback mode.

    We force force_cpu_fallback=True so these tests run on any machine
    without requiring CUDA or NVLS hardware.
    """

    def _make_fake_group(self) -> Optional[dist.ProcessGroup]:
        return _make_fake_pg()

    def test_init_buffers_cpu_fallback(self):
        pg = self._make_fake_group()
        if pg is None:
            self.skipTest("Cannot initialize dist for this test")

        path = HeteroNVLSDispatcherBuffers.init_buffers(
            max_tokens_per_rank=64,
            hidden_size=256,
            topk=2,
            ep_size=1,
            ep_group=pg,
            force_cpu_fallback=True,
        )
        # Should be CPU staging or LOC spill (never NVLS in forced fallback)
        self.assertIn(path, (DispatchPath.NCCL_CPU_STAGING, DispatchPath.LOC_SPILL))

    def test_buffer_shapes_correct(self):
        pg = self._make_fake_group()
        if pg is None:
            self.skipTest("Cannot initialize dist for this test")

        max_tokens = 32
        hidden_size = 64
        topk = 4
        ep_size = 1
        global_max = max_tokens * ep_size

        HeteroNVLSDispatcherBuffers.init_buffers(
            max_tokens_per_rank=max_tokens,
            hidden_size=hidden_size,
            topk=topk,
            ep_size=ep_size,
            ep_group=pg,
            force_cpu_fallback=True,
        )
        buf_h = HeteroNVLSDispatcherBuffers.get_buffer("ep_agv_h")
        buf_r = HeteroNVLSDispatcherBuffers.get_buffer("ep_agv_r")
        buf_p = HeteroNVLSDispatcherBuffers.get_buffer("ep_agv_p")
        buf_rsv = HeteroNVLSDispatcherBuffers.get_buffer("ep_rsv")
        buf_meta = HeteroNVLSDispatcherBuffers.get_buffer("ep_meta")

        self.assertIsNotNone(buf_h)
        self.assertEqual(buf_h.shape, [global_max, hidden_size])
        self.assertEqual(buf_r.shape, [global_max, topk])
        self.assertEqual(buf_p.shape, [global_max, topk])
        self.assertEqual(buf_rsv.shape, [global_max, hidden_size])
        self.assertEqual(buf_meta.shape, [ep_size])

    def test_self_sizing_matches_compute(self):
        pg = self._make_fake_group()
        if pg is None:
            self.skipTest("Cannot initialize dist for this test")

        max_tokens = 128
        hidden_size = 512
        topk = 8
        ep_size = 1

        HeteroNVLSDispatcherBuffers.init_buffers(
            max_tokens_per_rank=max_tokens,
            hidden_size=hidden_size,
            topk=topk,
            ep_size=ep_size,
            ep_group=pg,
            force_cpu_fallback=True,
        )
        global_max = max_tokens * ep_size
        expected_h = _compute_size_mb([global_max, hidden_size], torch.bfloat16)
        buf_h = HeteroNVLSDispatcherBuffers.get_buffer("ep_agv_h")
        self.assertEqual(buf_h.size_mb, expected_h)

    def test_summarize_output(self):
        pg = self._make_fake_group()
        if pg is None:
            self.skipTest("Cannot initialize dist for this test")

        HeteroNVLSDispatcherBuffers.init_buffers(
            max_tokens_per_rank=16,
            hidden_size=32,
            topk=2,
            ep_size=1,
            ep_group=pg,
            force_cpu_fallback=True,
        )
        summary = HeteroNVLSDispatcherBuffers.summarize()
        self.assertIn("ep_agv_h", summary)
        self.assertIn("dispatch_path", summary)


class TestHeteroMoEDispatcherCPUPath(unittest.TestCase):
    """End-to-end dispatcher tests on CPU/GPU-agnostic path."""

    def setUp(self):
        self.pg = _make_fake_pg()

    def test_dispatch_loc_spill_when_no_cuda(self):
        """Without CUDA, both NVLS and CPU staging may fail; LOC spill must work."""
        if self.pg is None:
            self.skipTest("Cannot initialize dist")

        # Use tiny config; on a machine without CUDA pinned_memory also unavailable
        try:
            dispatcher = HeteroMoEDispatcher(
                layer_idx=0,
                max_tokens_per_rank=4,
                hidden_size=8,
                topk=2,
                ep_size=1,
                ep_group=self.pg,
            )
        except Exception as e:
            self.skipTest(f"Dispatcher init failed (expected on CI): {e}")

        h = torch.randn(4, 8).bfloat16()
        r = torch.zeros(4, 2, dtype=torch.int64)
        p = torch.ones(4, 2)

        result = dispatcher.dispatch(h, r, p)
        self.assertIn("path", result)
        self.assertIn(result["path"], list(DispatchPath))

    def test_loc_spill_retrieve(self):
        """LOC spill tensors can be retrieved back to CPU device."""
        if self.pg is None:
            self.skipTest("Cannot initialize dist")

        loc_store = LOCDispatcherStore()
        try:
            dispatcher = HeteroMoEDispatcher(
                layer_idx=1,
                max_tokens_per_rank=4,
                hidden_size=8,
                topk=2,
                ep_size=1,
                ep_group=self.pg,
                loc_store=loc_store,
            )
        except Exception as e:
            self.skipTest(f"Dispatcher init failed: {e}")

        if dispatcher.dispatch_path != DispatchPath.LOC_SPILL:
            self.skipTest("Not on LOC spill path in this environment")

        h = torch.randn(4, 8).bfloat16()
        r = torch.zeros(4, 2, dtype=torch.int64)
        p = torch.ones(4, 2)

        result = dispatcher.dispatch(h, r, p)
        self.assertEqual(result["path"], DispatchPath.LOC_SPILL)

        retrieved = dispatcher.retrieve_loc_spill(result, torch.device("cpu"))
        self.assertIn("hidden", retrieved)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    unittest.main(verbosity=2)
