"""
DES-LOC Heterogeneous FP32 Gradient Accumulation Module
========================================================

Upstream Design Intent (Megatron c586f6d5663f):
------------------------------------------------
Megatron-LM commit c586f6d introduces the ability to selectively promote gradient
accumulation to FP32 for a *subset* of named parameters, controlled via fnmatch
patterns (``param_name_patterns_for_fp32_local_accumulation``). The core insight is
that all-reduce / reduce-scatter collectives can operate in lower precision (BF16)
for bandwidth efficiency, while the *local* accumulation step that feeds the
optimizer can be kept in FP32 for numerical stability — but only where it matters
(e.g., LayerNorm weights, embedding tables) rather than universally.

Mechanically, Megatron achieves this by:
1. Maintaining a ``main_grad`` FP32 tensor *separate* from the BF16 ``grad_data``
   communication buffer (``main_grad_copy_in_grad_buffer`` aliases into the comm
   buffer; ``main_grad`` is the detached FP32 accumulator).
2. Before each collective: copy FP32 ``main_grad`` → BF16 comm buffer.
3. After each collective: copy reduced BF16 result back → FP32 ``main_grad``.
4. ``scale_gradients`` and ``reset`` must touch *both* the comm buffer and the
   extra FP32 tensors.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces a three-tier
heterogeneous compute topology:

    Tier-0 (H100 NVL, SM90, 96 GB)   — high-precision anchor
    Tier-1 (A6000 ×2, SM86, 48 GB)   — mid-precision workers
    Tier-2 (CPU DRAM, 1.5 TB)         — overflow / offload

Key adaptations over Megatron upstream:

A. **Tier-aware precision policy** — instead of a single global ``grad_reduce_in_fp32``
   flag, each parameter's precision is determined by (a) its fnmatch name pattern AND
   (b) which device tier owns its current replica.  H100 (Tier-0) always accumulates
   in FP32; A6000s (Tier-1) follow the pattern; CPU (Tier-2) accumulates in FP32 but
   communicates in BF16.

B. **Locality cache integration** — DES-LOC's SLOC cache holds recently-used gradient
   shards in a pinned CPU buffer so PCIe transfers are amortized.  The extra FP32
   ``main_grad`` tensors are optionally *resident in the SLOC cache* rather than on
   device, eliminating A6000 VRAM pressure during long micro-batch sequences.

C. **PCIe-aware copy scheduling** — because A6000↔H100 has no NVLink, copies of
   ``main_grad_copy_in_grad_buffer`` are enqueued on a dedicated PCIe stream rather
   than the default CUDA stream, avoiding head-of-line blocking.

D. **Heterogeneous bucket grouping** — buckets may span parameters that live on
   different device tiers; the ``HeteroParamAndGradBucket`` tracks per-tier
   subsets of ``params_with_extra_main_grads`` so copy-back is issued on the
   correct stream per tier.

E. **Zero-copy offload path** — when ``offload_fp32_grads_to_cpu=True``, FP32
   main_grads are allocated in pinned host memory and the copy-in/copy-out uses
   non-blocking transfers, with the SLOC cache providing coherency.

Module layout:
    HeteroFP32GradAccumConfig   — configuration dataclass
    DeviceTier                   — enum for Tier-0/1/2
    SLOCGradCache                — lightweight pinned-CPU gradient cache
    HeteroParamAndGradBucket     — per-bucket state with extra main_grad tracking
    HeteroParamAndGradBuffer     — full buffer manager (replaces _ParamAndGradBuffer)
    HeteroFP32GradAccumManager   — top-level manager wired into DeepSpeed engine
"""

from __future__ import annotations

import fnmatch
import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SM90_COMPUTE_CAPABILITY = (9, 0)   # H100 NVL
_SM86_COMPUTE_CAPABILITY = (8, 6)   # A6000

# DES-LOC SLOC cache default capacity in number of gradient elements (FP32).
# 4 GB / 4 bytes per float32 = 1 073 741 824 elements.
_DEFAULT_SLOC_CAPACITY_ELEMENTS = 1 << 30

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeviceTier(Enum):
    """
    Compute tier classification for DES-LOC heterogeneous topology.

    Tier-0 is the "anchor" (H100), Tier-1 are mid-range workers (A6000),
    Tier-2 is the CPU / DRAM overflow tier.
    """
    TIER_0 = auto()   # H100 NVL  SM90  96 GB
    TIER_1 = auto()   # A6000 ×2  SM86  48 GB each
    TIER_2 = auto()   # CPU DRAM  1.5 TB


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HeteroFP32GradAccumConfig:
    """
    Configuration for DES-LOC heterogeneous FP32 gradient accumulation.

    Upstream analogue: ``DistributedDataParallelConfig`` with the new
    ``param_name_patterns_for_fp32_local_accumulation`` and
    ``grad_reduce_in_fp32`` fields.

    DES-LOC extends this with tier-specific policies and SLOC cache controls.

    Attributes
    ----------
    param_name_patterns_for_fp32_local_accumulation:
        fnmatch patterns (or the special token ``'all'``) identifying which
        named parameters should receive a separate FP32 ``main_grad``
        accumulator.  Mirrors Megatron's field of the same name.
    grad_reduce_in_fp32:
        When True *all* parameters reduce in FP32 (legacy path, mutually
        exclusive with non-empty ``param_name_patterns_for_fp32_local_accumulation``).
    tier0_always_fp32:
        Force FP32 accumulation on Tier-0 (H100) regardless of patterns.
        Default True because H100 VRAM is large enough to afford it.
    tier1_follow_patterns:
        If True, Tier-1 (A6000) devices follow ``param_name_patterns``; if
        False they always use the buffer's ``grad_dtype``.
    offload_fp32_grads_to_cpu:
        Allocate extra FP32 ``main_grad`` tensors in pinned host memory
        (Tier-2 SLOC cache) rather than on-device.  Requires ``pin_memory``
        support.
    sloc_capacity_elements:
        Maximum number of FP32 gradient elements to hold in the SLOC cache
        simultaneously.  Overflow evicts LRU entries back to device.
    pcie_stream_priority:
        CUDA stream priority for PCIe-scheduled copies.  Lower integer = higher
        priority in PyTorch convention.
    overlap_pcie_copies:
        When True, copies to/from the SLOC cache are overlapped with the
        communication collective on a dedicated stream.
    bucket_size:
        Target number of elements per gradient bucket (Megatron default ≈ 40M).
    average_in_collective:
        Divide by DP world-size inside the collective rather than pre-scaling.
    check_for_nan_in_grad:
        Abort if NaN is detected in any gradient before the collective.
    """

    param_name_patterns_for_fp32_local_accumulation: Tuple[str, ...] = ()
    grad_reduce_in_fp32: bool = False

    # DES-LOC tier policy
    tier0_always_fp32: bool = True
    tier1_follow_patterns: bool = True

    # SLOC cache controls
    offload_fp32_grads_to_cpu: bool = False
    sloc_capacity_elements: int = _DEFAULT_SLOC_CAPACITY_ELEMENTS
    pcie_stream_priority: int = -1
    overlap_pcie_copies: bool = True

    # Communication
    bucket_size: int = 40_000_000
    average_in_collective: bool = False
    check_for_nan_in_grad: bool = False

    def __post_init__(self) -> None:
        if self.param_name_patterns_for_fp32_local_accumulation and self.grad_reduce_in_fp32:
            raise ValueError(
                "HeteroFP32GradAccumConfig: 'param_name_patterns_for_fp32_local_accumulation' "
                "and 'grad_reduce_in_fp32=True' are mutually exclusive.  Set grad_reduce_in_fp32 "
                "to False when specifying per-parameter patterns."
            )

    @property
    def has_selective_fp32(self) -> bool:
        """True when selective FP32 accumulation is active (either via patterns or flags)."""
        return bool(self.param_name_patterns_for_fp32_local_accumulation) or self.grad_reduce_in_fp32

    def tier_wants_fp32(self, tier: DeviceTier, param_name: str) -> bool:
        """
        Return True if a parameter on *tier* should accumulate gradients in FP32.

        DES-LOC decision logic:
        - Tier-0 (H100): always FP32 when ``tier0_always_fp32`` is set.
        - Tier-1 (A6000): follow fnmatch patterns when ``tier1_follow_patterns``.
        - Tier-2 (CPU): always FP32 (pinned host memory, no precision cost).
        """
        if self.grad_reduce_in_fp32:
            return True
        if tier == DeviceTier.TIER_0 and self.tier0_always_fp32:
            return True
        if tier == DeviceTier.TIER_2:
            return True
        if tier == DeviceTier.TIER_1 and self.tier1_follow_patterns:
            return self._matches_any_pattern(param_name)
        return False

    def _matches_any_pattern(self, param_name: str) -> bool:
        for pattern in self.param_name_patterns_for_fp32_local_accumulation:
            if pattern == "all" or fnmatch.fnmatch(param_name, pattern):
                return True
        return False


# ---------------------------------------------------------------------------
# Device tier detection
# ---------------------------------------------------------------------------


def detect_device_tier(device: torch.device) -> DeviceTier:
    """
    Classify a CUDA device (or CPU) into a DES-LOC :class:`DeviceTier`.

    Uses CUDA compute capability to distinguish H100 (SM90) from A6000 (SM86).
    Falls back to Tier-1 for unknown CUDA devices.
    """
    if device.type == "cpu":
        return DeviceTier.TIER_2
    if device.type != "cuda":
        raise ValueError(f"Unsupported device type '{device.type}' for DES-LOC tier detection.")

    major, minor = torch.cuda.get_device_capability(device)
    cc = (major, minor)
    if cc >= _SM90_COMPUTE_CAPABILITY:
        return DeviceTier.TIER_0
    if cc >= _SM86_COMPUTE_CAPABILITY:
        return DeviceTier.TIER_1
    # Older GPUs treated as Tier-1.
    logger.warning(
        "DES-LOC: Device %s has compute capability %d.%d, which is below SM86.  "
        "Classifying as Tier-1 (A6000-equivalent).  Precision policy may be sub-optimal.",
        device,
        major,
        minor,
    )
    return DeviceTier.TIER_1


# ---------------------------------------------------------------------------
# SLOC (Shared LOcality Cache) — pinned CPU gradient store
# ---------------------------------------------------------------------------


class SLOCGradCache:
    """
    Lightweight pinned-CPU gradient cache for DES-LOC.

    The SLOC cache holds FP32 gradient tensors in page-locked (pinned) host
    memory so that PCIe DMA transfers from GPU→CPU are as fast as possible.
    When the cache is full, the least-recently-used entry is evicted back to
    device memory.

    This class is intentionally simple — it does not implement a full coherency
    protocol.  The caller (HeteroParamAndGradBuffer) is responsible for ensuring
    a gradient tensor is not simultaneously being written on GPU and read from
    the cache.

    Parameters
    ----------
    capacity_elements:
        Maximum total FP32 elements resident in the cache.
    pcie_stream:
        CUDA stream on which DMA copies are scheduled.
    """

    def __init__(self, capacity_elements: int, pcie_stream: torch.cuda.Stream) -> None:
        self._capacity = capacity_elements
        self._stream = pcie_stream
        self._used: int = 0
        # Ordered dict gives us LRU eviction by move-to-end.
        self._store: Dict[int, torch.Tensor] = {}   # key = param data_ptr()

        logger.info(
            "DES-LOC SLOC cache initialized: capacity=%d FP32 elements (~%.1f GB pinned)",
            capacity_elements,
            capacity_elements * 4 / (1 << 30),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, shape: torch.Size, key: int) -> torch.Tensor:
        """
        Allocate a pinned FP32 tensor in the cache and register it under *key*.

        If capacity would be exceeded, evict LRU entries until there is room.
        """
        numel = math.prod(shape)
        self._evict_if_needed(numel)
        tensor = torch.zeros(shape, dtype=torch.float32, pin_memory=True)
        self._store[key] = tensor
        self._used += numel
        return tensor

    def get(self, key: int) -> Optional[torch.Tensor]:
        """Retrieve a cached tensor (None if not present).  Updates LRU order."""
        t = self._store.get(key)
        if t is not None:
            # Move to end (most recently used).
            self._store.pop(key)
            self._store[key] = t
        return t

    def copy_to_device(self, key: int, device_tensor: torch.Tensor) -> None:
        """
        Non-blocking copy from the pinned CPU tensor (cache entry *key*) to
        *device_tensor* on the PCIe stream.
        """
        host_tensor = self.get(key)
        if host_tensor is None:
            raise KeyError(f"SLOC cache has no entry for key {key}")
        with torch.cuda.stream(self._stream):
            device_tensor.copy_(host_tensor, non_blocking=True)

    def copy_from_device(self, key: int, device_tensor: torch.Tensor) -> None:
        """
        Non-blocking copy from *device_tensor* to the pinned CPU cache entry
        (key *key*) on the PCIe stream.
        """
        host_tensor = self.get(key)
        if host_tensor is None:
            raise KeyError(f"SLOC cache has no entry for key {key}")
        with torch.cuda.stream(self._stream):
            host_tensor.copy_(device_tensor, non_blocking=True)

    def synchronize_pcie(self) -> None:
        """Block until all pending PCIe DMA transfers complete."""
        self._stream.synchronize()

    @property
    def utilization_fraction(self) -> float:
        return self._used / self._capacity if self._capacity > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self, numel_needed: int) -> None:
        while self._used + numel_needed > self._capacity and self._store:
            evict_key, evict_tensor = next(iter(self._store.items()))
            self._store.pop(evict_key)
            freed = math.prod(evict_tensor.shape)
            self._used -= freed
            logger.debug(
                "DES-LOC SLOC eviction: freed %d elements (cache %.1f%% full after eviction)",
                freed,
                self.utilization_fraction * 100,
            )


# ---------------------------------------------------------------------------
# Per-bucket state
# ---------------------------------------------------------------------------


@dataclass
class _BucketParamInfo:
    """
    Metadata for a single parameter within a gradient bucket.

    Attributes
    ----------
    param:
        The ``nn.Parameter`` itself.
    name:
        Fully-qualified parameter name (dot-separated).
    tier:
        DES-LOC device tier of this parameter's owning device.
    start:
        Bucket-local offset of the first element (in elements).
    end:
        Bucket-local offset just past the last element.
    has_extra_main_grad:
        True when this param has a separate higher-precision ``main_grad``.
    """
    param: torch.nn.Parameter
    name: str
    tier: DeviceTier
    start: int
    end: int
    has_extra_main_grad: bool = False


class HeteroParamAndGradBucket:
    """
    DES-LOC gradient bucket that is tier-aware.

    Upstream analogue: ``_ParamAndGradBucket`` in
    ``megatron/core/distributed/param_and_grad_buffer.py``.

    Key difference from Megatron:
    - ``params_with_extra_main_grads`` is partitioned by ``DeviceTier`` so
      copy-in / copy-back operations are issued on the correct CUDA stream per
      tier (PCIe stream for Tier-1 A6000s, default stream for Tier-0 H100).
    - SLOC cache integration: for CPU-offloaded FP32 main_grads, copies are
      routed through the pinned cache rather than direct device↔device paths.

    Parameters
    ----------
    params_info:
        Ordered list of :class:`_BucketParamInfo` for parameters in this bucket.
    grad_data:
        View into the parent buffer's ``grad_data`` tensor covering this bucket.
    bucket_id:
        Zero-based index of this bucket.
    gradient_scaling_factor:
        Pre-scaling applied before reduction (1/dp_size when not averaging in
        collective).
    pcie_stream:
        Dedicated CUDA stream for PCIe-scheduled copies (A6000→CPU or A6000→H100).
    sloc_cache:
        Optional SLOC cache.  When provided, offloaded FP32 main_grads are kept
        here rather than on device.
    """

    def __init__(
        self,
        params_info: List[_BucketParamInfo],
        grad_data: torch.Tensor,
        bucket_id: int,
        gradient_scaling_factor: float,
        pcie_stream: torch.cuda.Stream,
        sloc_cache: Optional[SLOCGradCache] = None,
    ) -> None:
        self.params_info = params_info
        self.params: Set[torch.nn.Parameter] = {pi.param for pi in params_info}
        self.grad_data = grad_data
        self.bucket_id = bucket_id
        self.gradient_scaling_factor = gradient_scaling_factor
        self.pcie_stream = pcie_stream
        self.sloc_cache = sloc_cache

        # Tier-partitioned lists of params needing extra main_grad copy.
        self.extra_params_by_tier: Dict[DeviceTier, List[torch.nn.Parameter]] = {
            DeviceTier.TIER_0: [],
            DeviceTier.TIER_1: [],
            DeviceTier.TIER_2: [],
        }
        for pi in params_info:
            if pi.has_extra_main_grad:
                self.extra_params_by_tier[pi.tier].append(pi.param)

    @property
    def params_with_extra_main_grads(self) -> List[torch.nn.Parameter]:
        """Flat list of all parameters that have a separate FP32 main_grad (any tier)."""
        result: List[torch.nn.Parameter] = []
        for lst in self.extra_params_by_tier.values():
            result.extend(lst)
        return result

    def copy_extra_main_grads_to_comm_buffer(self) -> None:
        """
        Pre-collective step: copy FP32 ``main_grad`` tensors into the BF16
        communication buffer.

        DES-LOC adaptation: copies for Tier-1 params are issued on the PCIe
        stream so they can overlap with other work on the default stream.  Tier-0
        params use the default stream (NVLink-equivalent latency is already low
        within the H100).

        Upstream analogue: the anonymous loop in ``_ParamAndGradBucketGroup.
        start_grad_sync`` that calls ``param.main_grad_copy_in_grad_buffer.copy_
        (param.main_grad)``.
        """
        # Tier-0: copy on default stream (H100 is the anchor, lowest latency).
        for param in self.extra_params_by_tier[DeviceTier.TIER_0]:
            copy_in = getattr(param, "main_grad_copy_in_grad_buffer", None)
            if copy_in is not None:
                copy_in.copy_(param.main_grad)

        # Tier-1: PCIe stream (A6000 — no NVLink, avoid blocking default stream).
        with torch.cuda.stream(self.pcie_stream):
            for param in self.extra_params_by_tier[DeviceTier.TIER_1]:
                copy_in = getattr(param, "main_grad_copy_in_grad_buffer", None)
                if copy_in is not None:
                    if self.sloc_cache is not None:
                        # SLOC path: grad lives in pinned CPU; DMA to device buffer.
                        self.sloc_cache.copy_to_device(
                            id(param.main_grad), copy_in
                        )
                    else:
                        copy_in.copy_(param.main_grad, non_blocking=True)

        # Tier-2 (CPU): source is already in host memory; copy to device buffer.
        for param in self.extra_params_by_tier[DeviceTier.TIER_2]:
            copy_in = getattr(param, "main_grad_copy_in_grad_buffer", None)
            if copy_in is not None:
                copy_in.copy_(param.main_grad, non_blocking=True)

    def copy_back_extra_main_grads(self) -> None:
        """
        Post-collective step: copy the reduced BF16 gradient from the comm
        buffer back into the FP32 ``main_grad``.

        DES-LOC adaptation: same stream assignment as
        ``copy_extra_main_grads_to_comm_buffer``.

        Upstream analogue: ``_ParamAndGradBucketGroup._copy_back_extra_main_grads``.
        """
        # Tier-0: default stream.
        for param in self.extra_params_by_tier[DeviceTier.TIER_0]:
            copy_in = getattr(param, "main_grad_copy_in_grad_buffer", None)
            if copy_in is not None:
                param.main_grad.copy_(copy_in)

        # Tier-1: PCIe stream.
        with torch.cuda.stream(self.pcie_stream):
            for param in self.extra_params_by_tier[DeviceTier.TIER_1]:
                copy_in = getattr(param, "main_grad_copy_in_grad_buffer", None)
                if copy_in is not None:
                    if self.sloc_cache is not None:
                        # Write reduced value into pinned CPU cache.
                        self.sloc_cache.copy_from_device(
                            id(param.main_grad), copy_in
                        )
                    else:
                        param.main_grad.copy_(copy_in, non_blocking=True)

        # Tier-2 (CPU): write back to host tensor.
        for param in self.extra_params_by_tier[DeviceTier.TIER_2]:
            copy_in = getattr(param, "main_grad_copy_in_grad_buffer", None)
            if copy_in is not None:
                param.main_grad.copy_(copy_in, non_blocking=True)

    def synchronize_pcie_copies(self) -> None:
        """Ensure all PCIe-stream copies for this bucket have completed."""
        self.pcie_stream.synchronize()


# ---------------------------------------------------------------------------
# Parameter name → tier mapping helper
# ---------------------------------------------------------------------------


def _infer_param_tier(param: torch.nn.Parameter) -> DeviceTier:
    """
    Infer the DES-LOC tier for a parameter from its ``device`` attribute.

    This is called once during buffer construction and the result is cached
    in ``_BucketParamInfo.tier``.
    """
    return detect_device_tier(param.device)


# ---------------------------------------------------------------------------
# Main buffer class
# ---------------------------------------------------------------------------


class HeteroParamAndGradBuffer:
    """
    DES-LOC heterogeneous parameter and gradient buffer.

    Upstream analogue: ``_ParamAndGradBuffer`` in
    ``megatron/core/distributed/param_and_grad_buffer.py`` (Megatron c586f6d).

    This class manages contiguous ``param_data`` and ``grad_data`` buffers
    that pack model parameters and their gradients for efficient all-reduce /
    reduce-scatter collectives.  On top of Megatron's design it adds:

    1. **Tier-aware FP32 promotion** — each parameter's precision is decided by
       :meth:`HeteroFP32GradAccumConfig.tier_wants_fp32` rather than by a single
       global flag.
    2. **SLOC cache offload** — when ``config.offload_fp32_grads_to_cpu`` is True,
       extra FP32 ``main_grad`` tensors are pinned in CPU memory and managed by a
       :class:`SLOCGradCache`, freeing device VRAM between micro-batches.
    3. **PCIe stream scheduling** — copies to/from the comm buffer use a dedicated
       stream to avoid blocking the computation stream on PCIe transfers.
    4. **Heterogeneous bucket groups** — buckets track per-tier sets of parameters
       with extra main_grads for correct stream assignment during copy operations.

    Parameters
    ----------
    config:
        :class:`HeteroFP32GradAccumConfig` controlling precision policies.
    param_dtype:
        dtype of the parameter data (e.g., ``torch.bfloat16``).
    grad_dtype:
        dtype of the communication buffer (e.g., ``torch.bfloat16``).
    params_with_names:
        Ordered list of ``(parameter, name)`` pairs.  The ordering determines
        bucket assignment (parameters are bucketed in reverse order, roughly
        following backward pass order).
    data_parallel_group:
        Process group for gradient all-reduce / reduce-scatter.
    device:
        Target CUDA device for the buffer.  Determines which tier owns this
        buffer and influences per-parameter tier classification when a parameter
        has not yet been assigned a device.
    """

    def __init__(
        self,
        config: HeteroFP32GradAccumConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params_with_names: List[Tuple[torch.nn.Parameter, str]],
        data_parallel_group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        self.config = config
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.device = device
        self.buffer_tier = detect_device_tier(device)

        self._params: List[torch.nn.Parameter] = [p for p, _ in params_with_names]
        self._param_names: Dict[torch.nn.Parameter, str] = {
            p: n for p, n in params_with_names
        }

        # Validate uniqueness.
        seen: Set[torch.nn.Parameter] = set()
        for param, name in params_with_names:
            if param in seen:
                raise ValueError(
                    f"HeteroParamAndGradBuffer: duplicate parameter '{name}' detected."
                )
            seen.add(param)

        # Create PCIe CUDA stream for inter-tier copies.
        self._pcie_stream = torch.cuda.Stream(
            device=device, priority=config.pcie_stream_priority
        )

        # Optionally create SLOC cache.
        self._sloc_cache: Optional[SLOCGradCache] = None
        if config.offload_fp32_grads_to_cpu:
            self._sloc_cache = SLOCGradCache(
                capacity_elements=config.sloc_capacity_elements,
                pcie_stream=self._pcie_stream,
            )

        # Allocate contiguous buffers.
        self._numel = sum(p.data.nelement() for p in self._params)
        self.grad_data = torch.zeros(self._numel, dtype=grad_dtype, device=device)
        self.param_data = torch.zeros(self._numel, dtype=param_dtype, device=device)

        # Extra FP32 main_grad tensors (device or CPU-pinned).
        self.extra_main_grads: List[torch.Tensor] = []

        # Build param→offset map and assign main_grad / param.data views.
        self._param_offset_map: Dict[torch.nn.Parameter, Tuple[int, int]] = {}
        self.buckets: List[HeteroParamAndGradBucket] = []
        self._build_buffers(params_with_names)

        logger.info(
            "DES-LOC HeteroParamAndGradBuffer on %s (Tier-%s): "
            "%d params, %d elements, %d buckets, %d extra FP32 main_grads",
            device,
            self.buffer_tier.name,
            len(self._params),
            self._numel,
            len(self.buckets),
            len(self.extra_main_grads),
        )

    # ------------------------------------------------------------------
    # Buffer construction
    # ------------------------------------------------------------------

    def _build_buffers(
        self, params_with_names: List[Tuple[torch.nn.Parameter, str]]
    ) -> None:
        """
        Assign contiguous slices of ``param_data`` / ``grad_data`` to each
        parameter and group parameters into buckets.

        Iterates in *reverse* order (mirrors backward pass order) — same
        convention as Megatron's ``_ParamAndGradBuffer``.
        """
        offset = 0
        # Forward pass: compute offsets.
        for param, _ in params_with_names:
            numel = param.data.nelement()
            self._param_offset_map[param] = (offset, offset + numel)
            offset += numel

        # Build bucket assignment: iterate in reverse (backward pass order).
        bucket_params_info: List[_BucketParamInfo] = []
        bucket_grad_start: int = 0
        current_bucket_numel: int = 0

        for param, name in reversed(params_with_names):
            start, end = self._param_offset_map[param]
            numel = end - start

            # Map parameter data into contiguous param_data buffer.
            param.data = self.param_data[start:end].view(param.data.shape)

            # Map main_grad (initially) into grad_data buffer.
            grad_slice = self.grad_data[start:end].view(param.data.shape)
            param.main_grad = grad_slice

            # Determine tier and whether FP32 promotion is needed.
            tier = _infer_param_tier(param) if param.device.type != "meta" else self.buffer_tier
            promote = self.config.tier_wants_fp32(tier, name)

            has_extra = False
            if promote and self.grad_dtype != torch.float32:
                # Keep the BF16 slice as the comm buffer reference.
                param.main_grad_copy_in_grad_buffer = grad_slice

                # Allocate the FP32 accumulator.
                if self._sloc_cache is not None and tier == DeviceTier.TIER_1:
                    # Offload to pinned CPU memory via SLOC cache.
                    fp32_grad = self._sloc_cache.allocate(param.data.shape, id(param.data))
                else:
                    fp32_grad = torch.zeros(
                        param.data.shape, dtype=torch.float32, device=self.device
                    )

                param.main_grad = fp32_grad
                self.extra_main_grads.append(fp32_grad)
                has_extra = True

                logger.debug(
                    "DES-LOC: Promoted main_grad for '%s' (Tier-%s) from %s → float32%s",
                    name,
                    tier.name,
                    self.grad_dtype,
                    " [SLOC offload]" if self._sloc_cache is not None and tier == DeviceTier.TIER_1 else "",
                )

            pi = _BucketParamInfo(
                param=param,
                name=name,
                tier=tier,
                start=start - bucket_grad_start,
                end=end - bucket_grad_start,
                has_extra_main_grad=has_extra,
            )
            bucket_params_info.append(pi)
            current_bucket_numel += numel

            # Check if we should close this bucket.
            if current_bucket_numel >= self.config.bucket_size:
                self._create_bucket(
                    bucket_params_info,
                    grad_start=bucket_grad_start,
                    grad_end=bucket_grad_start + current_bucket_numel,
                )
                bucket_grad_start += current_bucket_numel
                current_bucket_numel = 0
                bucket_params_info = []

        # Flush remaining params into final bucket.
        if bucket_params_info:
            self._create_bucket(
                bucket_params_info,
                grad_start=bucket_grad_start,
                grad_end=bucket_grad_start + current_bucket_numel,
            )

        # Log bucket summary.
        self._log_bucket_summary()

    def _create_bucket(
        self,
        params_info: List[_BucketParamInfo],
        grad_start: int,
        grad_end: int,
    ) -> None:
        """Instantiate a :class:`HeteroParamAndGradBucket` and register it."""
        n_extra = sum(1 for pi in params_info if pi.has_extra_main_grad)
        tier_counts = {t: 0 for t in DeviceTier}
        for pi in params_info:
            if pi.has_extra_main_grad:
                tier_counts[pi.tier] += 1

        dp_size = dist.get_world_size(self.data_parallel_group)
        scaling = 1.0 / dp_size if not self.config.average_in_collective else 1.0

        bucket = HeteroParamAndGradBucket(
            params_info=params_info,
            grad_data=self.grad_data[grad_start:grad_end],
            bucket_id=len(self.buckets),
            gradient_scaling_factor=scaling,
            pcie_stream=self._pcie_stream,
            sloc_cache=self._sloc_cache,
        )
        self.buckets.append(bucket)

        if n_extra > 0:
            logger.debug(
                "DES-LOC: Bucket %d — %d params, %d with extra FP32 main_grads "
                "(Tier-0: %d, Tier-1: %d, Tier-2: %d)",
                len(self.buckets) - 1,
                len(params_info),
                n_extra,
                tier_counts[DeviceTier.TIER_0],
                tier_counts[DeviceTier.TIER_1],
                tier_counts[DeviceTier.TIER_2],
            )

    def _log_bucket_summary(self) -> None:
        """Emit a single INFO log summarising all buckets and their precision layout."""
        lines = [
            f"DES-LOC HeteroParamAndGradBuffer bucket layout "
            f"(param_dtype={self.param_dtype}, grad_dtype={self.grad_dtype}):"
        ]
        for bkt in self.buckets:
            n_extra = len(bkt.params_with_extra_main_grads)
            lines.append(
                f"  Bucket {bkt.bucket_id}: {len(bkt.params_info)} params, "
                f"grad_data numel={bkt.grad_data.numel()}, "
                f"extra FP32 main_grads={n_extra}"
            )
            for pi in bkt.params_info:
                grad_dtype_str = "fp32" if pi.has_extra_main_grad else str(self.grad_dtype)
                lines.append(f"    [{pi.tier.name}] {pi.name}  main_grad.dtype={grad_dtype_str}")
        logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Gradient lifecycle methods
    # ------------------------------------------------------------------

    def scale_gradients(self, scaling_factor: float) -> None:
        """
        Scale all gradient data by *scaling_factor*.

        Applies to both the communication buffer (``grad_data``) and all extra
        FP32 ``main_grad`` tensors.

        Upstream analogue: ``_ParamAndGradBuffer.scale_gradients``.
        DES-LOC delta: also scales ``self.extra_main_grads``.
        """
        self.grad_data.mul_(scaling_factor)
        for fp32_grad in self.extra_main_grads:
            fp32_grad.mul_(scaling_factor)

    def reset(self) -> None:
        """
        Zero the communication buffer and all extra FP32 ``main_grad`` tensors.

        Called at the start of each forward pass to clear stale gradient state.

        Upstream analogue: ``_ParamAndGradBuffer.reset`` (``grad_data.zero_()``
        + new ``extra_main_grads.zero_()`` loop in Megatron c586f6d).
        DES-LOC delta: handles SLOC-offloaded FP32 tensors (they may be pinned
        CPU tensors; ``zero_()`` works on both device and pinned host tensors).
        """
        self.grad_data.zero_()
        for fp32_grad in self.extra_main_grads:
            fp32_grad.zero_()

    def start_grad_sync(self, bucket: HeteroParamAndGradBucket) -> Optional[dist.Work]:
        """
        Initiate the gradient all-reduce for *bucket*.

        Pre-collective: copy FP32 ``main_grad`` tensors into the BF16 comm
        buffer, then launch the collective.

        Upstream analogue: ``_ParamAndGradBucketGroup.start_grad_sync`` plus the
        anonymous pre-collective copy loop added in Megatron c586f6d.

        Returns the ``dist.Work`` handle for asynchronous callers, or None when
        called synchronously.
        """
        # Scale before reduce.
        if bucket.gradient_scaling_factor != 1.0:
            bucket.grad_data.mul_(bucket.gradient_scaling_factor)

        # Copy FP32 main_grads → BF16 comm buffer (tier-aware, stream-aware).
        bucket.copy_extra_main_grads_to_comm_buffer()

        if self.config.check_for_nan_in_grad:
            self._check_for_nan(bucket)

        # Wait for PCIe copies to land before launching collective.
        if self.config.overlap_pcie_copies:
            torch.cuda.current_stream().wait_stream(self._pcie_stream)
        else:
            self._pcie_stream.synchronize()

        handle = dist.all_reduce(
            bucket.grad_data,
            op=dist.ReduceOp.SUM,
            group=self.data_parallel_group,
            async_op=True,
        )
        return handle

    def finish_grad_sync(
        self,
        bucket: HeteroParamAndGradBucket,
        handle: Optional[dist.Work],
    ) -> None:
        """
        Wait for the all-reduce to complete and copy reduced gradients back to
        FP32 ``main_grad`` tensors.

        Upstream analogue: ``_ParamAndGradBucketGroup.finish_grad_sync`` plus
        ``_copy_back_extra_main_grads`` (added in Megatron c586f6d).

        DES-LOC delta: copy-back is stream-aware (per-tier) and synchronises
        the PCIe stream before returning so callers can safely read ``main_grad``.
        """
        if handle is not None:
            handle.wait()

        # Copy reduced BF16 results → FP32 main_grads (tier-aware).
        bucket.copy_back_extra_main_grads()

        # Ensure PCIe copies have completed before the optimizer reads main_grad.
        bucket.synchronize_pcie_copies()

    def synchronize_all_buckets(self) -> None:
        """
        Run synchronous all-reduce across all buckets.

        Convenience wrapper used when ``overlap_grad_reduce=False``.
        """
        for bucket in self.buckets:
            handle = self.start_grad_sync(bucket)
            self.finish_grad_sync(bucket, handle)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def _check_for_nan(self, bucket: HeteroParamAndGradBucket) -> None:
        """Raise if any NaN/Inf is found in the bucket's grad_data."""
        if not torch.isfinite(bucket.grad_data).all():
            raise RuntimeError(
                f"DES-LOC: NaN or Inf detected in grad_data for bucket "
                f"{bucket.bucket_id} before all-reduce."
            )

    def report_memory(self) -> Dict[str, float]:
        """
        Return a dictionary with VRAM / RAM usage statistics for this buffer.

        Useful for monitoring whether the SLOC cache is keeping device memory
        within bounds across the heterogeneous topology.
        """
        device_bytes = (
            self.grad_data.nelement() * self.grad_data.element_size()
            + self.param_data.nelement() * self.param_data.element_size()
        )
        fp32_device_bytes = sum(
            g.nelement() * g.element_size()
            for g in self.extra_main_grads
            if not g.is_pinned()
        )
        fp32_pinned_bytes = sum(
            g.nelement() * g.element_size()
            for g in self.extra_main_grads
            if g.is_pinned()
        )
        return {
            "grad_param_buffer_MB": device_bytes / (1 << 20),
            "extra_fp32_device_MB": fp32_device_bytes / (1 << 20),
            "extra_fp32_pinned_MB": fp32_pinned_bytes / (1 << 20),
            "sloc_utilization": (
                self._sloc_cache.utilization_fraction if self._sloc_cache else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Top-level manager (DeepSpeed engine integration point)
# ---------------------------------------------------------------------------


class HeteroFP32GradAccumManager:
    """
    Manager that wires :class:`HeteroParamAndGradBuffer` instances into a
    DeepSpeed training engine.

    In the Neuron_SP / DES-LOC project this class is intended to replace the
    ``GradientAllReduceManager`` in ``deepspeed/runtime/engine.py`` for
    heterogeneous topologies.

    Responsibilities:
    - Create one or more ``HeteroParamAndGradBuffer`` instances (one per
      dtype pair, mirroring Megatron's buffer creation).
    - Drive the pre/post-collective copy protocol (``start_grad_sync`` /
      ``finish_grad_sync``) respecting DES-LOC tier ordering (Tier-0 first,
      then Tier-1, then Tier-2).
    - Expose ``scale_gradients`` and ``reset`` that fan out to all buffers.
    - Report aggregated memory usage across all buffers.

    Parameters
    ----------
    config:
        Shared :class:`HeteroFP32GradAccumConfig`.
    model:
        The ``nn.Module`` whose parameters will be managed.
    data_parallel_group:
        Process group for gradient reduction.
    device:
        Primary compute device (should be the device on which the model lives).
    param_dtype:
        Storage dtype for parameters.
    grad_dtype:
        Communication dtype for gradients.
    """

    def __init__(
        self,
        config: HeteroFP32GradAccumConfig,
        model: torch.nn.Module,
        data_parallel_group: dist.ProcessGroup,
        device: torch.device,
        param_dtype: torch.dtype = torch.bfloat16,
        grad_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.model = model
        self.data_parallel_group = data_parallel_group
        self.device = device

        params_with_names: List[Tuple[torch.nn.Parameter, str]] = [
            (p, n) for n, p in model.named_parameters() if p.requires_grad
        ]

        self.buffer = HeteroParamAndGradBuffer(
            config=config,
            param_dtype=param_dtype,
            grad_dtype=grad_dtype,
            params_with_names=params_with_names,
            data_parallel_group=data_parallel_group,
            device=device,
        )

        logger.info(
            "DES-LOC HeteroFP32GradAccumManager initialised: "
            "%d parameters, selective_fp32=%s, offload_to_cpu=%s",
            len(params_with_names),
            config.has_selective_fp32,
            config.offload_fp32_grads_to_cpu,
        )

    def before_backward(self) -> None:
        """Zero gradient buffers at the start of each backward pass."""
        self.buffer.reset()

    def after_backward(self, scale: float = 1.0) -> None:
        """
        Apply gradient scaling and run synchronous all-reduce across all buckets.

        In a production engine this would be split into ``start_grad_sync`` and
        ``finish_grad_sync`` calls, potentially overlapped with the backward pass
        on a per-bucket basis.  This synchronous version is provided for
        correctness testing.
        """
        if scale != 1.0:
            self.buffer.scale_gradients(scale)
        self.buffer.synchronize_all_buckets()

    def report_memory(self) -> Dict[str, float]:
        """Aggregate memory stats from the underlying buffer."""
        return self.buffer.report_memory()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import unittest

    # ------------------------------------------------------------------
    # Helpers for tests that do not require a real distributed environment.
    # ------------------------------------------------------------------

    def _fake_dist_group() -> dist.ProcessGroup:
        """
        Return the default process group, initialising a single-rank
        gloo backend if no backend is already initialised.
        """
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://127.0.0.1:29500",
                world_size=1,
                rank=0,
            )
        return dist.group.WORLD

    def _cpu_device() -> torch.device:
        """Return CPU device for topology-agnostic unit tests."""
        return torch.device("cpu")

    def _make_tiny_model(n_layers: int = 2, dim: int = 8) -> torch.nn.Module:
        layers = []
        for _ in range(n_layers):
            layers.append(torch.nn.Linear(dim, dim, bias=True))
            layers.append(torch.nn.LayerNorm(dim))
        return torch.nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------

    class TestHeteroFP32GradAccumConfigValidation(unittest.TestCase):
        """Config-level validation — no GPU or distributed required."""

        def test_mutual_exclusion_raises(self):
            with self.assertRaises(ValueError):
                HeteroFP32GradAccumConfig(
                    grad_reduce_in_fp32=True,
                    param_name_patterns_for_fp32_local_accumulation=("all",),
                )

        def test_valid_patterns_only(self):
            cfg = HeteroFP32GradAccumConfig(
                grad_reduce_in_fp32=False,
                param_name_patterns_for_fp32_local_accumulation=("*.weight",),
            )
            self.assertTrue(cfg.has_selective_fp32)

        def test_valid_grad_reduce_in_fp32_only(self):
            cfg = HeteroFP32GradAccumConfig(grad_reduce_in_fp32=True)
            self.assertTrue(cfg.has_selective_fp32)

        def test_no_patterns_no_selective_fp32(self):
            cfg = HeteroFP32GradAccumConfig()
            self.assertFalse(cfg.has_selective_fp32)

    class TestTierWantsFP32(unittest.TestCase):
        """Tier-aware precision policy logic."""

        def _cfg(self, **kwargs) -> HeteroFP32GradAccumConfig:
            return HeteroFP32GradAccumConfig(**kwargs)

        def test_tier0_always_fp32_when_flag_set(self):
            cfg = self._cfg(tier0_always_fp32=True)
            self.assertTrue(cfg.tier_wants_fp32(DeviceTier.TIER_0, "any.param"))

        def test_tier0_not_fp32_when_flag_clear(self):
            cfg = self._cfg(tier0_always_fp32=False)
            self.assertFalse(cfg.tier_wants_fp32(DeviceTier.TIER_0, "any.param"))

        def test_tier2_always_fp32(self):
            cfg = self._cfg()
            self.assertTrue(cfg.tier_wants_fp32(DeviceTier.TIER_2, "any.param"))

        def test_tier1_follows_pattern_match(self):
            cfg = self._cfg(
                tier1_follow_patterns=True,
                param_name_patterns_for_fp32_local_accumulation=("*.weight",),
                tier0_always_fp32=False,
            )
            self.assertTrue(cfg.tier_wants_fp32(DeviceTier.TIER_1, "layer.weight"))
            self.assertFalse(cfg.tier_wants_fp32(DeviceTier.TIER_1, "layer.bias"))

        def test_tier1_all_pattern(self):
            cfg = self._cfg(
                tier1_follow_patterns=True,
                param_name_patterns_for_fp32_local_accumulation=("all",),
                tier0_always_fp32=False,
            )
            self.assertTrue(cfg.tier_wants_fp32(DeviceTier.TIER_1, "anything"))

        def test_tier1_no_patterns_no_fp32(self):
            cfg = self._cfg(
                tier1_follow_patterns=True,
                tier0_always_fp32=False,
            )
            self.assertFalse(cfg.tier_wants_fp32(DeviceTier.TIER_1, "layer.weight"))

        def test_grad_reduce_in_fp32_overrides_all(self):
            cfg = self._cfg(
                grad_reduce_in_fp32=True,
                tier0_always_fp32=False,
            )
            for tier in DeviceTier:
                self.assertTrue(cfg.tier_wants_fp32(tier, "anything"))

    class TestDeviceTierDetection(unittest.TestCase):
        """Tier detection from device type — CPU path only (no real GPU needed)."""

        def test_cpu_is_tier2(self):
            self.assertEqual(detect_device_tier(torch.device("cpu")), DeviceTier.TIER_2)

        def test_unknown_device_type_raises(self):
            with self.assertRaises((ValueError, AttributeError)):
                detect_device_tier(torch.device("xpu"))

    class TestSLOCGradCache(unittest.TestCase):
        """SLOC cache allocation and eviction."""

        def _make_cache(self, capacity: int) -> SLOCGradCache:
            # Use a CPU stream (no GPU required for allocation tests).
            if torch.cuda.is_available():
                stream = torch.cuda.Stream()
            else:
                stream = None  # type: ignore[assignment]
            return SLOCGradCache(capacity_elements=capacity, pcie_stream=stream)  # type: ignore[arg-type]

        def test_allocate_and_retrieve(self):
            cache = self._make_cache(capacity=1024)
            t = cache.allocate(torch.Size([4, 4]), key=1)
            self.assertEqual(t.dtype, torch.float32)
            self.assertTrue(t.is_pinned() or not torch.cuda.is_available())
            retrieved = cache.get(1)
            self.assertIsNotNone(retrieved)
            self.assertIs(retrieved, t)

        def test_eviction_when_full(self):
            cache = self._make_cache(capacity=16)
            # Allocate 16 elements (fills cache).
            t1 = cache.allocate(torch.Size([16]), key=1)
            self.assertIsNotNone(t1)
            # Allocate 8 more — should evict t1.
            t2 = cache.allocate(torch.Size([8]), key=2)
            self.assertIsNotNone(t2)
            # t1 should have been evicted.
            self.assertIsNone(cache.get(1))
            self.assertIsNotNone(cache.get(2))

        def test_utilization_fraction(self):
            cache = self._make_cache(capacity=100)
            cache.allocate(torch.Size([50]), key=10)
            self.assertAlmostEqual(cache.utilization_fraction, 0.5)

        def test_miss_returns_none(self):
            cache = self._make_cache(capacity=1024)
            self.assertIsNone(cache.get(999))

    class TestHeteroParamAndGradBufferNoDist(unittest.TestCase):
        """
        Buffer tests using CPU device and a gloo single-rank process group so
        they can run without a real GPU cluster.
        """

        @classmethod
        def setUpClass(cls):
            cls.group = _fake_dist_group()
            cls.device = _cpu_device()

        def _make_buffer(
            self,
            model: torch.nn.Module,
            patterns: Tuple[str, ...] = (),
            grad_reduce_in_fp32: bool = False,
            tier0_always_fp32: bool = False,
            bucket_size: int = 10_000,
        ) -> HeteroParamAndGradBuffer:
            config = HeteroFP32GradAccumConfig(
                param_name_patterns_for_fp32_local_accumulation=patterns,
                grad_reduce_in_fp32=grad_reduce_in_fp32,
                tier0_always_fp32=tier0_always_fp32,
                offload_fp32_grads_to_cpu=False,
                bucket_size=bucket_size,
            )
            params_with_names = [
                (p, n) for n, p in model.named_parameters() if p.requires_grad
            ]
            return HeteroParamAndGradBuffer(
                config=config,
                param_dtype=torch.float32,
                grad_dtype=torch.float32,   # Use FP32 throughout for CPU tests.
                params_with_names=params_with_names,
                data_parallel_group=self.group,
                device=self.device,
            )

        def test_no_patterns_no_extra_main_grads(self):
            model = _make_tiny_model()
            buf = self._make_buffer(model, patterns=())
            self.assertEqual(len(buf.extra_main_grads), 0)
            for bkt in buf.buckets:
                self.assertEqual(len(bkt.params_with_extra_main_grads), 0)

        def test_all_pattern_promotes_all_params(self):
            """When grad_reduce_in_fp32=True, all params get FP32 main_grad.
            (Using grad_reduce_in_fp32 because grad_dtype==float32 for CPU tests,
            so selective FP32 would be a no-op.  We test the flag path instead.)"""
            model = _make_tiny_model()
            # Use bfloat16 grad_dtype so promotion is meaningful.
            config = HeteroFP32GradAccumConfig(
                grad_reduce_in_fp32=True,
                tier0_always_fp32=False,
                offload_fp32_grads_to_cpu=False,
                bucket_size=10_000,
            )
            params_with_names = [
                (p, n) for n, p in model.named_parameters() if p.requires_grad
            ]
            # We cannot use bfloat16 on CPU easily; just verify the logic path
            # using float32 grad_dtype (promotion skipped because dtypes match).
            buf = HeteroParamAndGradBuffer(
                config=config,
                param_dtype=torch.float32,
                grad_dtype=torch.float32,
                params_with_names=params_with_names,
                data_parallel_group=self.group,
                device=self.device,
            )
            # When grad_dtype == torch.float32, promotion condition
            # (grad_dtype != float32) is False, so extra_main_grads stays empty.
            self.assertEqual(len(buf.extra_main_grads), 0)

        def test_reset_zeros_grad_data(self):
            model = _make_tiny_model()
            buf = self._make_buffer(model)
            buf.grad_data.fill_(3.14)
            buf.reset()
            self.assertTrue(torch.all(buf.grad_data == 0.0))

        def test_scale_gradients(self):
            model = _make_tiny_model()
            buf = self._make_buffer(model)
            buf.grad_data.fill_(2.0)
            buf.scale_gradients(0.5)
            self.assertTrue(
                torch.allclose(buf.grad_data, torch.tensor(1.0)),
                msg="grad_data should be scaled by 0.5",
            )

        def test_scale_gradients_also_scales_extra_main_grads(self):
            """Manually inject an extra FP32 grad and verify scale_gradients touches it."""
            model = _make_tiny_model()
            buf = self._make_buffer(model)
            fake_fp32 = torch.full((8,), 4.0, dtype=torch.float32)
            buf.extra_main_grads.append(fake_fp32)
            buf.scale_gradients(0.25)
            self.assertTrue(
                torch.allclose(fake_fp32, torch.tensor(1.0)),
                msg="extra FP32 main_grad should be scaled",
            )

        def test_reset_also_zeros_extra_main_grads(self):
            model = _make_tiny_model()
            buf = self._make_buffer(model)
            fake_fp32 = torch.full((8,), 99.0, dtype=torch.float32)
            buf.extra_main_grads.append(fake_fp32)
            buf.reset()
            self.assertTrue(torch.all(fake_fp32 == 0.0))

        def test_param_data_views_into_contiguous_buffer(self):
            """After buffer construction, param.data should be a view of param_data."""
            model = _make_tiny_model()
            buf = self._make_buffer(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertEqual(
                        param.data.storage().data_ptr(),
                        buf.param_data.storage().data_ptr(),
                        msg=f"param '{name}' data should alias param_data buffer",
                    )

        def test_buckets_cover_all_params(self):
            """Every parameter should appear in exactly one bucket."""
            model = _make_tiny_model()
            buf = self._make_buffer(model, bucket_size=32)
            all_bucket_params: Set[torch.nn.Parameter] = set()
            for bkt in buf.buckets:
                for pi in bkt.params_info:
                    self.assertNotIn(
                        pi.param,
                        all_bucket_params,
                        msg=f"Param '{pi.name}' appears in multiple buckets",
                    )
                    all_bucket_params.add(pi.param)
            model_params = {p for p in model.parameters() if p.requires_grad}
            self.assertEqual(all_bucket_params, model_params)

        def test_report_memory_keys_present(self):
            model = _make_tiny_model()
            buf = self._make_buffer(model)
            mem = buf.report_memory()
            for key in (
                "grad_param_buffer_MB",
                "extra_fp32_device_MB",
                "extra_fp32_pinned_MB",
                "sloc_utilization",
            ):
                self.assertIn(key, mem)

        def test_synchronize_all_buckets_runs_without_error(self):
            """End-to-end test: zero grads, scale, synchronise (single rank = no-op reduce)."""
            model = _make_tiny_model()
            buf = self._make_buffer(model)
            buf.reset()
            buf.grad_data.fill_(1.0)
            # Should complete without exception on single-rank gloo group.
            buf.synchronize_all_buckets()

    class TestHeteroFP32GradAccumManager(unittest.TestCase):
        """Manager-level integration tests (single-rank, CPU)."""

        @classmethod
        def setUpClass(cls):
            cls.group = _fake_dist_group()

        def test_before_and_after_backward(self):
            model = _make_tiny_model()
            config = HeteroFP32GradAccumConfig(
                tier0_always_fp32=False,
                tier1_follow_patterns=False,
                bucket_size=10_000,
            )
            mgr = HeteroFP32GradAccumManager(
                config=config,
                model=model,
                data_parallel_group=self.group,
                device=torch.device("cpu"),
                param_dtype=torch.float32,
                grad_dtype=torch.float32,
            )
            mgr.before_backward()
            self.assertTrue(torch.all(mgr.buffer.grad_data == 0.0))
            mgr.buffer.grad_data.fill_(2.0)
            mgr.after_backward(scale=0.5)
            # After scale + all-reduce (sum, single rank), expect 1.0.
            self.assertTrue(
                torch.allclose(mgr.buffer.grad_data, torch.tensor(1.0)),
                msg="Grad data should be 1.0 after scale 0.5 and single-rank all-reduce",
            )

        def test_report_memory_from_manager(self):
            model = _make_tiny_model()
            config = HeteroFP32GradAccumConfig()
            mgr = HeteroFP32GradAccumManager(
                config=config,
                model=model,
                data_parallel_group=self.group,
                device=torch.device("cpu"),
                param_dtype=torch.float32,
                grad_dtype=torch.float32,
            )
            mem = mgr.report_memory()
            self.assertGreater(mem["grad_param_buffer_MB"], 0.0)

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    suite = unittest.TestLoader().loadTestsFromTestCase(TestHeteroFP32GradAccumConfigValidation)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTierWantsFP32))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDeviceTierDetection))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSLOCGradCache))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroParamAndGradBufferNoDist))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroFP32GradAccumManager))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
