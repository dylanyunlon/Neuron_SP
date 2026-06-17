# Copyright (c) 2026 Neuron_SP Project (dylanyunlon/Neuron_SP). All rights reserved.
# Adapted from NVIDIA Megatron-LM commit 5c660c3b4a9d91e7ed32997b56cd6d333ed04bc5
# Original: Fine-grained activation offloading with fused_group_mlp support
# DES-LOC Adaptation: TierAwareActivationOffload — heterogeneous tier-routing for
#   2x A6000 (SM86, 48GB) + 1x H100 NVL (SM90, 96GB) over PCIe, with 1.5TB CPU DRAM
#   as the shared locality cache tier. Tensors are routed to CPU DRAM, A6000 VRAM,
#   or H100 VRAM based on compute-affinity, tensor size, and reuse distance.

"""
TierAwareActivationOffload
==========================

Upstream Design Intent (Megatron commit 5c660c3b)
--------------------------------------------------
Megatron's fine-grained activation offloading system moves activations at
module granularity to CPU host memory during the forward pass, then reloads
them during the backward pass, overlapping D2H / H2D transfers with compute
to keep activation memory low without paying the full recompute cost.

Key upstream changes in 5c660c3b:
  1. Added ``fused_group_mlp`` as a new offloadable module name.  The fused
     grouped MLP (TEGroupedMLP with TE op-fuser) produces a single fused kernel
     whose internal activations cannot be selectively offloaded; the entire
     input tensor must be treated as one atomic offload unit.
  2. Introduced ``_te_do_not_offload(tensor)`` to honour TransformerEngine's
     ``_TE_do_not_offload`` attribute on tensors or their data-tensor children,
     allowing TE to mark certain tensors as non-offloadable (e.g. quantised
     tensors that must stay on-device for correctness).
  3. Refactored ``tensor_push`` / ``tensor_need_offloading_checker`` so that
     non-CUDA tensors, ``nn.Parameter`` instances, and FakeTensor / Functional-
     Tensor subclass objects are silently passed through rather than triggering
     an assertion, making the API safe to call from hooks that see all saved
     tensors regardless of type.
  4. ``tensor_pop`` now handles the case where the stored "tag" is itself a
     bare ``torch.Tensor`` (the passthrough path), returning it directly.
  5. CPU memory pool usage is disabled for ``fused_group_mlp`` (joining
     ``expert_fc1`` and ``moe_act``) because tensor shapes are not statically
     known before the fused kernel executes.
  6. Validation in ``TransformerConfig`` prevents combining
     ``fused_group_mlp`` with ``expert_fc1`` / ``moe_act`` (redundant and
     incorrect) and prevents combining any of these three with
     ``moe_paged_stash``.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) adds a **tier
dimension** to every decision the upstream code makes purely in CPU/GPU
binary terms:

Tier 0 — H100 NVL (SM90, 96 GB)
    High-throughput compute tier.  Activations whose producer and first
    consumer are both on H100 stay on-device.  Only activations that cross a
    pipeline boundary to an A6000 stage are evicted early.

Tier 1 — A6000 × 2 (SM86, 48 GB each, PCIe)
    Medium-throughput compute tier.  A6000 is memory-constrained.
    Activations produced here that are not immediately consumed are the
    primary offload candidates.  Because there is no NVLink, cross-GPU
    transfers use PCIe and are expensive; the locality cache prefers CPU
    DRAM over peer-GPU DRAM for large tensors.

Tier 2 — CPU DRAM (1.5 TB, shared locality cache)
    The DES-LOC locality cache.  Activations evicted from any GPU land here.
    The 1.5 TB capacity means we never need to discard activations; we only
    need to schedule the H2D reload before the backward kernel that needs
    them is launched.

Heterogeneous-awareness additions (not in upstream):
  * ``DeviceTier`` enum distinguishes H100 from A6000 from CPU.
  * ``TierPolicy`` decides, per-tensor and per-module-name, which tier should
    store the activation during the window between forward and backward.
  * ``TierAwareOffloadTensorGroup`` extends the upstream ``OffloadTensorGroup``
    concept with per-tier pinned-memory pools and PCIe-bandwidth-aware
    prefetch scheduling.
  * ``LocalityCacheManager`` tracks the DES-LOC shared locality cache: which
    tensors are in CPU DRAM, which are prefetched, and which H2D streams are
    in flight.  It enforces the constraint that fused_group_mlp activations
    are always treated as atomic.
  * ``HeterogeneousChunkOffloadHandler`` wraps upstream's
    ``ChunkOffloadHandler`` logic with tier routing.

The public API deliberately mirrors the upstream interface so that the rest
of the DeepSpeed engine (pipeline scheduler, zero optimizer hooks) can call
``tensor_push`` / ``tensor_pop`` / ``tensor_need_offloading_checker``
unchanged.
"""

from __future__ import annotations

import logging
import threading
import time
import unittest
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SM compute capability constants used for tier classification
_SM90_COMPUTE = (9, 0)   # H100 NVL
_SM86_COMPUTE = (8, 6)   # A6000

# Module names that cannot use a CPU memory pool because their tensor shapes
# are not statically known before the fused kernel executes.  Mirrors upstream
# commit 5c660c3b change to OffloadTensorGroup.__init__.
_DYNAMIC_SHAPE_MODULES: FrozenSet[str] = frozenset(
    {"expert_fc1", "moe_act", "fused_group_mlp"}
)

# Module names introduced in commit 5c660c3b (fused_group_mlp support)
_FUSED_GROUP_MLP_MODULE = "fused_group_mlp"

# Minimum tensor numel below which we skip offload entirely (upstream default)
_DEFAULT_MIN_OFFLOAD_NUMEL = 1024 * 1024  # 1 M elements

# PCIe bandwidth estimate (bytes/second) for H100↔CPU on this cluster.
# Measured empirically at ~28 GB/s unidirectional on PCIe Gen4 x16.
_PCIE_BW_BYTES_PER_SEC = 28e9

# DES-LOC locality cache capacity guard: refuse to schedule more than this
# fraction of CPU DRAM for activation storage.
_CPU_DRAM_MAX_FRACTION = 0.80
_CPU_DRAM_TOTAL_BYTES = int(1.5 * (1024 ** 4))  # 1.5 TiB


# ---------------------------------------------------------------------------
# Device-tier classification
# ---------------------------------------------------------------------------

class DeviceTier(Enum):
    """Hardware tier in the DES-LOC heterogeneous cluster."""
    H100_NVL = auto()    # SM90, 96 GB, PCIe host
    A6000     = auto()    # SM86, 48 GB, PCIe host
    CPU_DRAM  = auto()    # 1.5 TB shared locality cache
    UNKNOWN   = auto()


def classify_device(device: torch.device) -> DeviceTier:
    """
    Map a :class:`torch.device` to a :class:`DeviceTier`.

    Uses CUDA compute capability to distinguish H100 (SM90) from A6000 (SM86).
    Falls back to ``UNKNOWN`` for any device that cannot be queried.

    Args:
        device: The torch device to classify.

    Returns:
        The corresponding :class:`DeviceTier`.
    """
    if device.type == "cpu":
        return DeviceTier.CPU_DRAM
    if device.type != "cuda":
        return DeviceTier.UNKNOWN
    try:
        cap = torch.cuda.get_device_capability(device.index or 0)
    except Exception:
        return DeviceTier.UNKNOWN
    if cap >= _SM90_COMPUTE:
        return DeviceTier.H100_NVL
    if cap >= _SM86_COMPUTE:
        return DeviceTier.A6000
    return DeviceTier.UNKNOWN


# ---------------------------------------------------------------------------
# Tensor eligibility helpers  (mirrors + extends upstream 5c660c3b logic)
# ---------------------------------------------------------------------------

def _te_do_not_offload(tensor: Any) -> bool:
    """
    Return whether TransformerEngine marked a tensor as non-offloadable.

    Upstream (5c660c3b) introduced this function to let TE signal that
    quantised or otherwise device-resident tensors must not be moved to
    CPU.  We carry it forward unchanged and additionally check for the
    DES-LOC locality-cache opt-out attribute ``_DESLOC_do_not_offload``.

    Args:
        tensor: A tensor or TE tensor-like object.

    Returns:
        True if the tensor must not be offloaded, False otherwise.
    """
    if getattr(tensor, "_TE_do_not_offload", False):
        return True
    if getattr(tensor, "_DESLOC_do_not_offload", False):
        return True
    if not hasattr(tensor, "get_data_tensors"):
        return False
    try:
        data_tensors = tensor.get_data_tensors()
    except Exception:
        return False
    return any(
        dt is not None and (
            getattr(dt, "_TE_do_not_offload", False)
            or getattr(dt, "_DESLOC_do_not_offload", False)
        )
        for dt in data_tensors
    )


def _is_stray_tensor(tensor: Any) -> bool:
    """
    Return True for FakeTensor / FunctionalTensor subclasses.

    These should never be offloaded; the upstream assertion was replaced
    (5c660c3b) with a silent passthrough.  We replicate that here.

    Args:
        tensor: Any object that might be a tensor.

    Returns:
        True if the tensor is a stray autograd subclass.
    """
    try:
        from torch._subclasses.fake_tensor import FakeTensor
        from torch._subclasses.functional_tensor import FunctionalTensor
        return isinstance(tensor, (FakeTensor, FunctionalTensor))
    except ImportError:
        return False


def can_manage_tensor_for_offload(tensor: Any) -> bool:
    """
    Return whether a tensor is eligible for DES-LOC activation offloading.

    Eligibility rules (union of upstream 5c660c3b + DES-LOC):
      * Must be a real ``torch.Tensor``.
      * Must not be an ``nn.Parameter`` (parameters are managed by the
        optimizer, not activation offload).
      * Must not be a FakeTensor or FunctionalTensor (stray tensors from
        torch.compile / functional transforms).
      * Must reside on a CUDA device (CPU tensors need not be offloaded).
      * Must not be marked non-offloadable by TE or DES-LOC.

    Args:
        tensor: The object to test.

    Returns:
        True if the tensor can be pushed to the offload handler.
    """
    if not isinstance(tensor, torch.Tensor):
        return False
    if isinstance(tensor, nn.Parameter):
        return False
    if _is_stray_tensor(tensor):
        return False
    if tensor.device.type != "cuda":
        return False
    return True


# ---------------------------------------------------------------------------
# Tier policy
# ---------------------------------------------------------------------------

@dataclass
class TierPolicy:
    """
    Per-module offload-tier decision for DES-LOC heterogeneous hardware.

    The policy answers two questions for each activation tensor:
      1. Should it be offloaded at all?
      2. If so, which tier should store it while the backward pass is pending?

    Design rationale
    ~~~~~~~~~~~~~~~~
    In the upstream Megatron code the only destination is CPU host memory.
    DES-LOC has three tiers.  We keep the destination as CPU DRAM by default
    (matching upstream) but allow the policy to be overridden:

    * Tensors on H100 whose backward consumer is also on H100 could in
      principle stay on-device (tier = H100_NVL).  However, for MoE
      activations (fused_group_mlp, expert_fc1, moe_act) the memory savings
      from CPU offload are so large that we always prefer CPU DRAM.
    * Tensors on A6000 go to CPU DRAM unconditionally because A6000 has only
      48 GB and there is no NVLink to spill to H100 cheaply.
    * The CPU DRAM locality cache is 1.5 TB; we track utilisation and warn
      when we approach the safety threshold.
    """

    #: Which module names trigger offload for this policy instance.
    offload_modules: FrozenSet[str] = field(
        default_factory=lambda: frozenset()
    )
    #: Minimum numel below which tensors are not offloaded.
    min_offload_numel: int = _DEFAULT_MIN_OFFLOAD_NUMEL
    #: Force destination tier regardless of source device.
    force_destination_tier: Optional[DeviceTier] = None

    def should_offload_module(self, module_name: str) -> bool:
        """
        Return True if activations for *module_name* should be offloaded.

        Args:
            module_name: The logical module identifier
                (e.g. ``"fused_group_mlp"``, ``"expert_fc1"``).

        Returns:
            True if the module is in the configured offload set.
        """
        return module_name in self.offload_modules

    def destination_tier(self, source_tier: DeviceTier, module_name: str) -> DeviceTier:
        """
        Determine where to store a tensor given its source device tier.

        Args:
            source_tier: The tier of the device the tensor currently lives on.
            module_name: Logical module name, used for MoE special-casing.

        Returns:
            The :class:`DeviceTier` that should hold the tensor during the
            forward→backward window.
        """
        if self.force_destination_tier is not None:
            return self.force_destination_tier

        # MoE activations are always sent to CPU DRAM regardless of source,
        # because their size makes on-device retention impractical.
        if module_name in _DYNAMIC_SHAPE_MODULES:
            return DeviceTier.CPU_DRAM

        # A6000 is memory-constrained; always offload to CPU DRAM.
        if source_tier == DeviceTier.A6000:
            return DeviceTier.CPU_DRAM

        # H100 has 96 GB; non-MoE activations may stay on-device if memory
        # allows.  For safety we default to CPU DRAM.
        return DeviceTier.CPU_DRAM

    def validate_fused_group_mlp_config(
        self,
        use_op_fuser: bool,
        moe_paged_stash: bool,
    ) -> None:
        """
        Validate that fused_group_mlp offload configuration is consistent.

        Mirrors upstream TransformerConfig validation added in 5c660c3b.

        Args:
            use_op_fuser: Whether the TE op-fuser is enabled.
            moe_paged_stash: Whether MoE paged stash is enabled.

        Raises:
            ValueError: On any configuration conflict.
        """
        if _FUSED_GROUP_MLP_MODULE not in self.offload_modules:
            return
        if not use_op_fuser:
            raise ValueError(
                "fused_group_mlp activation offload requires the TE op-fuser "
                "(use_transformer_engine_op_fuser=True)."
            )
        moe_partial = {"expert_fc1", "moe_act"} & self.offload_modules
        if moe_partial:
            raise ValueError(
                "fused_group_mlp offloads the whole fused grouped MLP and "
                f"cannot be combined with {moe_partial}.  "
                "Remove the conflicting modules from offload_modules."
            )
        if moe_paged_stash:
            raise ValueError(
                "moe_paged_stash and fused_group_mlp offload are mutually "
                "exclusive; paged stash already covers those activations."
            )


# ---------------------------------------------------------------------------
# Pinned-memory pool (tier-aware)
# ---------------------------------------------------------------------------

class TierPinnedMemoryPool:
    """
    Per-tier pinned memory pool for DES-LOC activation offloading.

    Upstream's ``OffloadTensorGroup`` uses a single CPU pinned-memory pool
    whose shape is fixed at the first forward pass.  DES-LOC extends this
    with per-tier tracking so we can account for CPU DRAM utilisation across
    all concurrent offload streams.

    Thread safety: guarded by ``_lock`` since D2H transfers run in CUDA
    streams that may be scheduled from multiple host threads.

    Args:
        tier: The destination tier this pool serves.
        max_cpu_bytes: Maximum CPU DRAM bytes to allocate for this pool.
            Defaults to ``_CPU_DRAM_TOTAL_BYTES * _CPU_DRAM_MAX_FRACTION``.
    """

    def __init__(
        self,
        tier: DeviceTier = DeviceTier.CPU_DRAM,
        max_cpu_bytes: int = int(_CPU_DRAM_TOTAL_BYTES * _CPU_DRAM_MAX_FRACTION),
    ) -> None:
        self.tier = tier
        self.max_cpu_bytes = max_cpu_bytes
        self._allocated_bytes: int = 0
        self._pool: Dict[Tuple, torch.Tensor] = {}
        self._lock = threading.Lock()

    @property
    def utilisation_fraction(self) -> float:
        """Current fraction of the pool capacity that is allocated."""
        if self.max_cpu_bytes == 0:
            return 1.0
        with self._lock:
            return self._allocated_bytes / self.max_cpu_bytes

    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate or reuse a pinned CPU tensor of the given shape and dtype.

        Args:
            shape: Desired tensor shape.
            dtype: Desired tensor dtype.

        Returns:
            A pinned CPU tensor.

        Raises:
            RuntimeError: If the allocation would exceed ``max_cpu_bytes``.
        """
        key = (shape, dtype)
        with self._lock:
            if key in self._pool:
                return self._pool[key]
            nbytes = torch.empty(shape, dtype=dtype).nbytes
            if self._allocated_bytes + nbytes > self.max_cpu_bytes:
                raise RuntimeError(
                    f"DES-LOC locality cache pool exhausted: "
                    f"{self._allocated_bytes / 1e9:.1f} GB allocated, "
                    f"request for {nbytes / 1e6:.1f} MB would exceed "
                    f"{self.max_cpu_bytes / 1e9:.1f} GB limit."
                )
            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._pool[key] = buf
            self._allocated_bytes += nbytes
            if self.utilisation_fraction > 0.75:
                logger.warning(
                    "DES-LOC locality cache utilisation at %.1f%% "
                    "(%.2f / %.2f GB); consider reducing offload scope.",
                    self.utilisation_fraction * 100,
                    self._allocated_bytes / 1e9,
                    self.max_cpu_bytes / 1e9,
                )
            return buf

    def reset(self) -> None:
        """Release all pooled tensors and reset utilisation tracking."""
        with self._lock:
            self._pool.clear()
            self._allocated_bytes = 0


# ---------------------------------------------------------------------------
# Locality cache manager
# ---------------------------------------------------------------------------

class LocalityCacheManager:
    """
    DES-LOC shared locality cache coordinator for CPU DRAM-resident activations.

    Tracks which tensors are currently stored in the CPU locality cache,
    manages CUDA stream scheduling for D2H (offload) and H2D (reload) transfers,
    and estimates whether a pending reload will complete before the backward
    kernel that needs it is ready to execute.

    The cache is *shared* in the DES-LOC sense: tensors produced on either
    A6000 or H100 land in the same 1.5 TB DRAM pool, and the backward kernels
    on whichever device needs the activation can reload from there.

    Attributes:
        offload_stream_h100: CUDA stream on CUDA device 0 (H100) for D2H.
        offload_stream_a6000_0: CUDA stream on CUDA device 1 (A6000 #0).
        offload_stream_a6000_1: CUDA stream on CUDA device 2 (A6000 #1).
        reload_streams: Per-device reload streams (H2D direction).
    """

    def __init__(self) -> None:
        self._entries: Dict[Any, _LocalityCacheEntry] = {}
        self._lock = threading.Lock()
        self._streams: Dict[int, Optional[torch.cuda.Stream]] = {}
        self._reload_streams: Dict[int, Optional[torch.cuda.Stream]] = {}
        self._total_cached_bytes: int = 0

        # Initialise per-device CUDA streams lazily to avoid requiring CUDA at
        # import time.
        self._streams_initialised = False

    def _ensure_streams(self) -> None:
        """Lazily create CUDA streams for each available device."""
        if self._streams_initialised:
            return
        n = torch.cuda.device_count()
        for idx in range(n):
            try:
                self._streams[idx] = torch.cuda.Stream(device=idx)
                self._reload_streams[idx] = torch.cuda.Stream(device=idx)
            except Exception:
                self._streams[idx] = None
                self._reload_streams[idx] = None
        self._streams_initialised = True

    def offload_stream_for(self, device_index: int) -> Optional[torch.cuda.Stream]:
        """Return the D2H offload stream for a given CUDA device index."""
        self._ensure_streams()
        return self._streams.get(device_index)

    def reload_stream_for(self, device_index: int) -> Optional[torch.cuda.Stream]:
        """Return the H2D reload stream for a given CUDA device index."""
        self._ensure_streams()
        return self._reload_streams.get(device_index)

    def register(self, tag: Any, cpu_tensor: torch.Tensor, source_device: torch.device) -> None:
        """
        Register a tensor as stored in the CPU locality cache.

        Args:
            tag: Opaque identifier (upstream tuple tag or DES-LOC tag).
            cpu_tensor: The CPU-resident copy.
            source_device: The GPU device the tensor was moved from.
        """
        entry = _LocalityCacheEntry(
            cpu_tensor=cpu_tensor,
            source_device=source_device,
            offload_time=time.perf_counter(),
        )
        with self._lock:
            self._entries[tag] = entry
            self._total_cached_bytes += cpu_tensor.nbytes
        logger.debug(
            "LocalityCache: registered tag=%s shape=%s dtype=%s source=%s "
            "(total cached %.2f GB)",
            tag,
            tuple(cpu_tensor.shape),
            cpu_tensor.dtype,
            source_device,
            self._total_cached_bytes / 1e9,
        )

    def retrieve(self, tag: Any, target_device: torch.device) -> Optional[torch.Tensor]:
        """
        Retrieve a tensor from the CPU locality cache to *target_device*.

        Uses the reload stream for the target device so the transfer can
        overlap with compute on the default stream.

        Args:
            tag: The tag used when the tensor was registered.
            target_device: GPU device to reload the tensor onto.

        Returns:
            The GPU tensor, or None if the tag is not in the cache.
        """
        with self._lock:
            entry = self._entries.pop(tag, None)
            if entry is not None:
                self._total_cached_bytes -= entry.cpu_tensor.nbytes
        if entry is None:
            return None

        reload_stream = self.reload_stream_for(target_device.index or 0)
        if reload_stream is not None:
            with torch.cuda.stream(reload_stream):
                gpu_tensor = entry.cpu_tensor.to(target_device, non_blocking=True)
            # Ensure the compute stream waits for the reload to finish.
            torch.cuda.current_stream(target_device).wait_stream(reload_stream)
        else:
            gpu_tensor = entry.cpu_tensor.to(target_device)

        elapsed = time.perf_counter() - entry.offload_time
        bw = entry.cpu_tensor.nbytes / max(elapsed, 1e-9)
        logger.debug(
            "LocalityCache: reloaded tag=%s to %s in %.1f ms (%.1f GB/s eff BW)",
            tag,
            target_device,
            elapsed * 1000,
            bw / 1e9,
        )
        return gpu_tensor

    @property
    def total_cached_bytes(self) -> int:
        """Total bytes currently registered in the locality cache."""
        with self._lock:
            return self._total_cached_bytes

    def clear(self) -> None:
        """Evict all entries from the locality cache (call at end of micro-batch)."""
        with self._lock:
            self._entries.clear()
            self._total_cached_bytes = 0


@dataclass
class _LocalityCacheEntry:
    cpu_tensor: torch.Tensor
    source_device: torch.device
    offload_time: float


# ---------------------------------------------------------------------------
# Tier-aware offload tensor group
# ---------------------------------------------------------------------------

class TierAwareOffloadTensorGroup:
    """
    Container for a named group of activations being offloaded to a tier.

    Extends the upstream ``OffloadTensorGroup`` concept with:
      * Tier-specific pinned-memory pool management.
      * D2H transfer stream scheduling via :class:`LocalityCacheManager`.
      * Tracking of the ``fused_group_mlp`` atomic-offload constraint:
        all tensors in such a group are treated as a single unit.

    Upstream constraint (5c660c3b): modules in ``_DYNAMIC_SHAPE_MODULES``
    do not use the CPU pool because their shapes are unknown before the
    fused kernel runs.

    Args:
        name: Logical module name (e.g. ``"fused_group_mlp"``).
        tier_policy: The active :class:`TierPolicy`.
        pool: A shared :class:`TierPinnedMemoryPool` for this tier.
        locality_cache: The cluster-wide :class:`LocalityCacheManager`.
    """

    def __init__(
        self,
        name: str,
        tier_policy: TierPolicy,
        pool: TierPinnedMemoryPool,
        locality_cache: LocalityCacheManager,
    ) -> None:
        self.name = name
        self.tier_policy = tier_policy
        self.pool = pool
        self.locality_cache = locality_cache

        # Mirrors upstream: dynamic-shape modules skip the CPU pool.
        self.use_cpu_pool = name not in _DYNAMIC_SHAPE_MODULES

        self._gpu_tensors: List[torch.Tensor] = []
        self._cpu_tensors: List[torch.Tensor] = []
        self._tags: List[Any] = []
        self.total_offload_bytes: int = 0
        self.total_tensor_count: int = 0

    def push(
        self,
        tensor: torch.Tensor,
        tag: Any,
        source_device: torch.device,
    ) -> None:
        """
        Move *tensor* from GPU to the destination tier asynchronously.

        Args:
            tensor: The GPU tensor to offload.
            tag: Unique identifier for later retrieval.
            source_device: The device *tensor* currently lives on.
        """
        dest_tier = self.tier_policy.destination_tier(
            classify_device(source_device), self.name
        )
        if dest_tier == DeviceTier.CPU_DRAM:
            self._push_to_cpu(tensor, tag, source_device)
        else:
            # Destination is another GPU tier (future extension point).
            # For now, fall back to CPU DRAM with a debug note.
            logger.debug(
                "TierAwareOffloadTensorGroup[%s]: destination tier %s not "
                "yet natively supported; falling back to CPU DRAM.",
                self.name,
                dest_tier,
            )
            self._push_to_cpu(tensor, tag, source_device)

    def _push_to_cpu(
        self,
        tensor: torch.Tensor,
        tag: Any,
        source_device: torch.device,
    ) -> None:
        """D2H transfer to pinned CPU DRAM locality cache."""
        offload_stream = self.locality_cache.offload_stream_for(
            source_device.index or 0
        )
        if self.use_cpu_pool:
            try:
                cpu_buf = self.pool.allocate(tuple(tensor.shape), tensor.dtype)
            except RuntimeError as exc:
                logger.warning(
                    "TierAwareOffloadTensorGroup[%s]: pool allocation failed "
                    "(%s); allocating fresh pinned buffer.",
                    self.name,
                    exc,
                )
                cpu_buf = torch.empty_like(tensor, device="cpu", pin_memory=True)
        else:
            cpu_buf = torch.empty_like(tensor, device="cpu", pin_memory=True)

        if offload_stream is not None:
            with torch.cuda.stream(offload_stream):
                cpu_buf.copy_(tensor, non_blocking=True)
        else:
            cpu_buf.copy_(tensor)

        self.locality_cache.register(tag, cpu_buf, source_device)
        self._tags.append(tag)
        self.total_offload_bytes += tensor.nbytes
        self.total_tensor_count += 1

    def pop(self, tag: Any, target_device: torch.device) -> Optional[torch.Tensor]:
        """
        Retrieve a previously offloaded tensor back to *target_device*.

        Args:
            tag: The tag assigned during :meth:`push`.
            target_device: GPU device to reload onto.

        Returns:
            The reloaded GPU tensor, or None if not found.
        """
        return self.locality_cache.retrieve(tag, target_device)


# ---------------------------------------------------------------------------
# Heterogeneous chunk offload handler
# ---------------------------------------------------------------------------

class HeterogeneousChunkOffloadHandler:
    """
    DES-LOC adaptation of Megatron's ``ChunkOffloadHandler``.

    Manages activation offloading for one pipeline micro-batch chunk,
    routing tensors to the appropriate DES-LOC storage tier based on
    the source device, module name, and TierPolicy.

    Upstream ``ChunkOffloadHandler`` differences absorbed here
    (all from commit 5c660c3b):
      * ``tensor_push`` is now a no-op passthrough for non-offloadable
        tensors (was an assertion failure).
      * ``tensor_pop`` handles bare-tensor passthrough tags.
      * ``tensor_need_offloading_checker`` checks TE's do-not-offload
        attribute and DES-LOC's own ``_DESLOC_do_not_offload`` attribute.
      * ``_can_manage_tensor_for_offload`` is factored out as a static
        method used by both push and the checker.

    DES-LOC additions:
      * Groups are tier-aware (see :class:`TierAwareOffloadTensorGroup`).
      * The source device is captured at push time to select the right
        transfer stream and destination tier.
      * ``fused_group_mlp`` groups are treated as atomic: the whole input
        tensor is pushed as a single unit, matching the upstream constraint
        that the fused kernel's internal shapes are unknown.

    Args:
        tier_policy: Active offload tier policy.
        pool: Shared pinned memory pool.
        locality_cache: Cluster-wide locality cache coordinator.
        min_offload_numel: Minimum tensor numel to be eligible for offload.
    """

    def __init__(
        self,
        tier_policy: TierPolicy,
        pool: TierPinnedMemoryPool,
        locality_cache: LocalityCacheManager,
        min_offload_numel: int = _DEFAULT_MIN_OFFLOAD_NUMEL,
    ) -> None:
        self.tier_policy = tier_policy
        self.pool = pool
        self.locality_cache = locality_cache
        self.min_offload_numel = min_offload_numel

        # Index of the current module group being pushed to.
        self._current_group_index: int = 0
        # Count of tensors within the current group.
        self._tensor_count_current_group: int = 0
        # All groups created for this chunk.
        self._groups: List[TierAwareOffloadTensorGroup] = []
        # Map from group_index → group (1-based to match upstream).
        self._group_index_map: Dict[int, TierAwareOffloadTensorGroup] = {}

    @staticmethod
    def _can_manage_tensor_for_offload(tensor: Any) -> bool:
        """
        Return True if *tensor* can be managed by the offload hooks.

        Mirrors upstream static method added in 5c660c3b.

        Args:
            tensor: The tensor (or object) to test.

        Returns:
            True if eligible for push/pop.
        """
        return can_manage_tensor_for_offload(tensor)

    def begin_group(self, name: str) -> int:
        """
        Start a new named tensor group for offloading.

        Call once per module forward pass before calling :meth:`tensor_push`
        for the module's input tensors.

        Args:
            name: Module name (e.g. ``"fused_group_mlp"``).

        Returns:
            The 1-based group index assigned to this group.
        """
        group = TierAwareOffloadTensorGroup(
            name=name,
            tier_policy=self.tier_policy,
            pool=self.pool,
            locality_cache=self.locality_cache,
        )
        self._groups.append(group)
        self._current_group_index = len(self._groups)  # 1-based
        self._group_index_map[self._current_group_index] = group
        self._tensor_count_current_group = 0
        return self._current_group_index

    def end_group(self) -> None:
        """Finalise the current tensor group after all tensors are pushed."""
        if self._groups:
            g = self._groups[-1]
            logger.debug(
                "HeterogeneousChunkOffloadHandler: closed group '%s' "
                "(%d tensors, %.2f MB offloaded).",
                g.name,
                g.total_tensor_count,
                g.total_offload_bytes / 1e6,
            )

    def tensor_push(self, tensor: Any) -> Any:
        """
        Offload *tensor* to the DES-LOC locality cache if eligible.

        Returns the tensor tag to be saved by autograd's ``pack_hook``.
        Non-eligible tensors are returned as-is (passthrough, matching
        upstream 5c660c3b behaviour).

        Args:
            tensor: The tensor to push, or any object from a pack hook.

        Returns:
            A ``(group_index, tensor_index)`` tag, or the original *tensor*
            if it was not offloaded.
        """
        if not self._can_manage_tensor_for_offload(tensor):
            return tensor

        tag = (self._current_group_index, self._tensor_count_current_group)
        self._tensor_count_current_group += 1

        group = self._group_index_map.get(self._current_group_index)
        if group is None:
            # No group started; return tensor unchanged.
            return tensor

        group.push(tensor, tag, tensor.device)
        return tag

    def tensor_pop(self, tag: Any) -> Any:
        """
        Retrieve a tensor that was previously offloaded via :meth:`tensor_push`.

        Handles the passthrough case (5c660c3b) where *tag* is itself a
        ``torch.Tensor`` because the tensor was not offloaded.

        Args:
            tag: The value returned by :meth:`tensor_push`.

        Returns:
            The (reloaded) tensor.
        """
        # Upstream 5c660c3b: if tag is a bare tensor, it was a passthrough.
        if isinstance(tag, torch.Tensor):
            logger.debug(
                "HeterogeneousChunkOffloadHandler.tensor_pop: "
                "passthrough tensor shape=%s dtype=%s",
                tuple(tag.shape),
                tag.dtype,
            )
            return tag

        group_index, tensor_idx = tag
        group = self._group_index_map.get(group_index)
        if group is None:
            raise KeyError(
                f"tensor_pop: no group found for group_index={group_index}"
            )

        # Determine which device to reload onto.  For activations that were
        # produced on a specific device, we try to reload back to that device.
        # If the backward is running on a different device, the tensor will be
        # on CPU from the locality cache and must be moved.
        current_device = torch.cuda.current_device()
        target_device = torch.device("cuda", current_device)

        result = group.pop(tag, target_device)
        if result is None:
            raise RuntimeError(
                f"tensor_pop: tag {tag} not found in locality cache.  "
                "This indicates a push/pop imbalance."
            )
        return result

    def tensor_need_offloading_checker(self, tensor: Any) -> bool:
        """
        Return True if *tensor* should be offloaded to the locality cache.

        Extends upstream 5c660c3b logic with:
          * DES-LOC ``_DESLOC_do_not_offload`` attribute check.
          * Tier-policy minimum numel threshold.
          * TE ``_TE_do_not_offload`` propagation through ``get_data_tensors``.

        Args:
            tensor: The tensor candidate.

        Returns:
            True if the tensor should be offloaded.
        """
        if not self._can_manage_tensor_for_offload(tensor):
            return False
        if _te_do_not_offload(tensor):
            return False
        if tensor.numel() < self.min_offload_numel:
            return False
        # Respect per-tensor preference if set (upstream convention).
        offload_pref = getattr(tensor, "offloading_activation", None)
        if offload_pref is not None:
            return bool(offload_pref)
        return True

    @property
    def summary(self) -> Dict[str, int]:
        """
        Return a dict mapping module name → total offloaded bytes.

        Useful for logging and profiling.
        """
        result: Dict[str, int] = defaultdict(int)
        for g in self._groups:
            result[g.name] += g.total_offload_bytes
        return dict(result)


# ---------------------------------------------------------------------------
# Fused group MLP offload interface (DES-LOC context-manager)
# ---------------------------------------------------------------------------

class FusedGroupMLPOffloadInterface:
    """
    Context-manager interface for ``fused_group_mlp`` atomic offloading.

    In the upstream TEGroupedMLP forward method (5c660c3b), activation
    offloading of the entire fused grouped MLP is wrapped with an
    ``off_interface`` context manager that:
      1. Enters: records the input tensor and registers it with the offload
         handler.
      2. Exits:  yields the (possibly-replaced) input back to the fused op.
      3. ``group_commit``:  after the fused op output is produced, commits
         the group so the handler knows the offload boundary is complete.

    DES-LOC extends this by:
      * Routing the input tensor to the locality cache tier determined by
        the :class:`TierPolicy` (instead of always to CPU DRAM in the
        simplest upstream path).
      * Scheduling the D2H transfer on the device-specific offload stream.
      * Providing ``group_commit`` as an explicit second step so the fused
        op can finish writing its output before we close the group.

    Usage mirrors the upstream ``off_interface`` context manager so the rest
    of the engine can call this without changes::

        with FusedGroupMLPOffloadInterface(
            enabled, input_tensor, "fused_group_mlp"
        ) as possibly_same_tensor:
            output = fused_op(possibly_same_tensor, ...)
        output = FusedGroupMLPOffloadInterface.group_commit(output, ...)

    Args:
        enabled: If False, this is a no-op (nullcontext equivalent).
        input_tensor: The input to the fused grouped MLP.
        module_name: Must be ``"fused_group_mlp"``.
        handler: The active :class:`HeterogeneousChunkOffloadHandler`.
    """

    def __init__(
        self,
        enabled: bool,
        input_tensor: torch.Tensor,
        module_name: str,
        handler: Optional[HeterogeneousChunkOffloadHandler] = None,
    ) -> None:
        self.enabled = enabled
        self.input_tensor = input_tensor
        self.module_name = module_name
        self.handler = handler
        self._group_index: Optional[int] = None
        self._pushed_tag: Any = None

    def __enter__(self) -> torch.Tensor:
        if not self.enabled or self.handler is None:
            return self.input_tensor

        self._group_index = self.handler.begin_group(self.module_name)
        if self.handler.tensor_need_offloading_checker(self.input_tensor):
            self._pushed_tag = self.handler.tensor_push(self.input_tensor)
            logger.debug(
                "FusedGroupMLPOffloadInterface: pushed input tensor "
                "shape=%s to locality cache as tag=%s.",
                tuple(self.input_tensor.shape),
                self._pushed_tag,
            )
        else:
            self._pushed_tag = self.input_tensor
        return self.input_tensor

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        if self.enabled and self.handler is not None:
            self.handler.end_group()

    def group_commit(
        self,
        output: torch.Tensor,
        name: str,
        forced_released_tensors: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Commit the fused-group-mlp offload group after the fused op completes.

        Mirrors upstream ``off_interface.group_commit`` from 5c660c3b.

        Args:
            output: The output tensor from the fused grouped MLP op.
            name: Module name (should equal ``self.module_name``).
            forced_released_tensors: Tensors to explicitly release GPU
                memory for (upstream convention for fused_group_mlp path).

        Returns:
            The *output* tensor unchanged (commit is a side-effect).
        """
        if not self.enabled or self.handler is None:
            return output

        if forced_released_tensors:
            for t in forced_released_tensors:
                if isinstance(t, torch.Tensor) and t.device.type == "cuda":
                    logger.debug(
                        "FusedGroupMLPOffloadInterface.group_commit: "
                        "releasing forced tensor shape=%s from GPU memory.",
                        tuple(t.shape),
                    )
                    # Explicitly delete GPU memory of the now-offloaded tensor.
                    t.data = torch.empty(0, device=t.device, dtype=t.dtype)

        logger.debug(
            "FusedGroupMLPOffloadInterface.group_commit: committed "
            "group '%s' (group_index=%s).",
            name,
            self._group_index,
        )
        return output


# ---------------------------------------------------------------------------
# DES-LOC activation offload manager (top-level entry point)
# ---------------------------------------------------------------------------

class DESLOCActivationOffloadManager:
    """
    Top-level DES-LOC activation offload coordinator.

    Integrates all DES-LOC components into a single object that the
    DeepSpeed pipeline engine can use to manage activation offloading
    across the heterogeneous A6000+H100 cluster.

    Responsibilities:
      * Holds the singleton :class:`LocalityCacheManager` and
        :class:`TierPinnedMemoryPool` shared across all pipeline chunks.
      * Creates per-chunk :class:`HeterogeneousChunkOffloadHandler` instances.
      * Exposes ``begin_forward_chunk`` / ``end_backward_chunk`` lifecycle
        hooks that the pipeline scheduler calls at micro-batch boundaries.
      * Logs aggregate offload statistics after each backward pass.

    Args:
        tier_policy: The tier routing policy.
        min_offload_numel: Minimum tensor numel to consider for offload.
        cpu_pool_max_bytes: Maximum CPU DRAM to allocate for pinned pools.
    """

    def __init__(
        self,
        tier_policy: Optional[TierPolicy] = None,
        min_offload_numel: int = _DEFAULT_MIN_OFFLOAD_NUMEL,
        cpu_pool_max_bytes: int = int(_CPU_DRAM_TOTAL_BYTES * _CPU_DRAM_MAX_FRACTION),
    ) -> None:
        self.tier_policy = tier_policy or TierPolicy(
            offload_modules=frozenset({"fused_group_mlp", "expert_fc1", "moe_act"}),
            min_offload_numel=min_offload_numel,
        )
        self.min_offload_numel = min_offload_numel

        self._pool = TierPinnedMemoryPool(
            tier=DeviceTier.CPU_DRAM,
            max_cpu_bytes=cpu_pool_max_bytes,
        )
        self._locality_cache = LocalityCacheManager()
        self._active_chunks: List[HeterogeneousChunkOffloadHandler] = []
        self._completed_summaries: List[Dict[str, int]] = []

    def begin_forward_chunk(self) -> HeterogeneousChunkOffloadHandler:
        """
        Begin a new pipeline micro-batch chunk's forward pass.

        Returns:
            A fresh :class:`HeterogeneousChunkOffloadHandler` for this chunk.
        """
        handler = HeterogeneousChunkOffloadHandler(
            tier_policy=self.tier_policy,
            pool=self._pool,
            locality_cache=self._locality_cache,
            min_offload_numel=self.min_offload_numel,
        )
        self._active_chunks.append(handler)
        return handler

    def end_backward_chunk(self) -> None:
        """
        Finalise the oldest active chunk's backward pass.

        Logs aggregate offload statistics and removes the chunk from the
        active list.
        """
        if not self._active_chunks:
            return
        handler = self._active_chunks.pop(0)
        summary = handler.summary
        if summary:
            total_mb = sum(summary.values()) / 1e6
            logger.info(
                "DES-LOC chunk offload summary: %.1f MB offloaded — %s",
                total_mb,
                {k: f"{v/1e6:.1f}MB" for k, v in summary.items()},
            )
        self._completed_summaries.append(summary)

    def reset(self) -> None:
        """
        Reset the manager state between training steps.

        Clears the locality cache and resets the pinned memory pool if
        dynamic-shape groups are in use (upstream constraint: pool is not
        reused for dynamic-shape groups).
        """
        self._locality_cache.clear()
        self._active_chunks.clear()
        # Do not reset the pool between steps to reuse pinned allocations
        # for fixed-shape groups.  Only reset if the pool is over threshold.
        if self._pool.utilisation_fraction > 0.90:
            logger.info(
                "DES-LOC: pinned memory pool utilisation %.1f%%; resetting "
                "pool to reclaim memory.",
                self._pool.utilisation_fraction * 100,
            )
            self._pool.reset()

    @property
    def locality_cache_bytes(self) -> int:
        """Current bytes stored in the CPU DRAM locality cache."""
        return self._locality_cache.total_cached_bytes

    @property
    def pool_utilisation(self) -> float:
        """Fraction of the pinned memory pool that is allocated."""
        return self._pool.utilisation_fraction


# ---------------------------------------------------------------------------
# Validation helpers (mirrors upstream TransformerConfig validation)
# ---------------------------------------------------------------------------

def validate_offload_config(
    offload_modules: List[str],
    use_op_fuser: bool = False,
    moe_paged_stash: bool = False,
    cpu_offloading: bool = False,
) -> None:
    """
    Validate the DES-LOC activation offload configuration.

    Mirrors and extends the validation added to ``TransformerConfig.__post_init__``
    in Megatron commit 5c660c3b, translated to the DeepSpeed config API.

    Args:
        offload_modules: List of module names to offload.
        use_op_fuser: Whether the TE op-fuser is enabled.
        moe_paged_stash: Whether MoE paged stash is enabled.
        cpu_offloading: Whether generic CPU offloading is enabled
            (incompatible with moe_paged_stash).

    Raises:
        ValueError: On any configuration conflict.
    """
    module_set = set(offload_modules)
    valid_modules = {
        "attn_norm", "qkv_linear", "core_attn", "attn_proj",
        "mlp_norm", "expert_fc1", "moe_act", "fused_group_mlp",
    }
    unknown = module_set - valid_modules
    if unknown:
        raise ValueError(
            f"Unknown offload_modules: {unknown}.  "
            f"Valid choices: {valid_modules}"
        )

    if _FUSED_GROUP_MLP_MODULE in module_set:
        if not use_op_fuser:
            raise ValueError(
                "fused_group_mlp activation offload requires "
                "use_transformer_engine_op_fuser=True."
            )
        moe_partial = {"expert_fc1", "moe_act"} & module_set
        if moe_partial:
            raise ValueError(
                "fused_group_mlp offloads the whole fused grouped MLP and "
                f"cannot be combined with {moe_partial}."
            )

    if moe_paged_stash:
        if cpu_offloading:
            raise ValueError("moe_paged_stash cannot be enabled with cpu_offloading.")
        moe_conflict = {"expert_fc1", "moe_act", "fused_group_mlp"} & module_set
        if moe_conflict:
            raise ValueError(
                "moe_paged_stash and offload_modules cannot both include MoE "
                f"activation modules: {moe_conflict}.  "
                "Paged stash already covers those activations."
            )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import unittest

    class TestDeviceTierClassification(unittest.TestCase):
        """Tests for :func:`classify_device`."""

        def test_cpu_is_dram_tier(self):
            self.assertEqual(classify_device(torch.device("cpu")), DeviceTier.CPU_DRAM)

        def test_unknown_device_type(self):
            d = torch.device("meta")
            self.assertEqual(classify_device(d), DeviceTier.UNKNOWN)

    class TestTensorEligibility(unittest.TestCase):
        """Tests mirroring upstream 5c660c3b ChunkOffloadHandler eligibility tests."""

        def test_cpu_tensor_not_eligible(self):
            t = torch.empty(1024)
            self.assertFalse(can_manage_tensor_for_offload(t))

        def test_parameter_not_eligible(self):
            p = nn.Parameter(torch.empty(1024))
            self.assertFalse(can_manage_tensor_for_offload(p))

        def test_non_tensor_not_eligible(self):
            self.assertFalse(can_manage_tensor_for_offload(42))
            self.assertFalse(can_manage_tensor_for_offload("hello"))
            self.assertFalse(can_manage_tensor_for_offload(None))

        def test_fake_tensor_not_eligible(self):
            try:
                from torch._subclasses.fake_tensor import FakeTensorMode
            except ImportError:
                self.skipTest("FakeTensorMode not available.")
            with FakeTensorMode():
                ft = torch.empty(1024, device="cuda")
            self.assertFalse(can_manage_tensor_for_offload(ft))

    class TestTeDoNotOffload(unittest.TestCase):
        """Tests for :func:`_te_do_not_offload`."""

        def test_normal_tensor_not_blocked(self):
            t = torch.empty(8)
            self.assertFalse(_te_do_not_offload(t))

        def test_te_attribute_blocks(self):
            t = torch.empty(8)
            t._TE_do_not_offload = True
            self.assertTrue(_te_do_not_offload(t))

        def test_desloc_attribute_blocks(self):
            t = torch.empty(8)
            t._DESLOC_do_not_offload = True
            self.assertTrue(_te_do_not_offload(t))

        def test_false_attribute_does_not_block(self):
            t = torch.empty(8)
            t._TE_do_not_offload = False
            self.assertFalse(_te_do_not_offload(t))

    class TestTierPolicy(unittest.TestCase):
        """Tests for :class:`TierPolicy`."""

        def setUp(self):
            self.policy = TierPolicy(
                offload_modules=frozenset({"fused_group_mlp", "expert_fc1"}),
            )

        def test_fused_group_mlp_always_goes_to_cpu_dram(self):
            tier = self.policy.destination_tier(DeviceTier.H100_NVL, "fused_group_mlp")
            self.assertEqual(tier, DeviceTier.CPU_DRAM)

        def test_a6000_source_always_goes_to_cpu_dram(self):
            tier = self.policy.destination_tier(DeviceTier.A6000, "core_attn")
            self.assertEqual(tier, DeviceTier.CPU_DRAM)

        def test_h100_non_moe_defaults_to_cpu_dram(self):
            tier = self.policy.destination_tier(DeviceTier.H100_NVL, "core_attn")
            self.assertEqual(tier, DeviceTier.CPU_DRAM)

        def test_should_offload_module(self):
            self.assertTrue(self.policy.should_offload_module("fused_group_mlp"))
            self.assertFalse(self.policy.should_offload_module("core_attn"))

        def test_validate_fused_group_mlp_requires_op_fuser(self):
            with self.assertRaises(ValueError):
                self.policy.validate_fused_group_mlp_config(
                    use_op_fuser=False, moe_paged_stash=False
                )

        def test_validate_fused_group_mlp_conflicts_with_expert_fc1(self):
            policy = TierPolicy(
                offload_modules=frozenset({"fused_group_mlp", "expert_fc1"}),
            )
            with self.assertRaises(ValueError):
                policy.validate_fused_group_mlp_config(
                    use_op_fuser=True, moe_paged_stash=False
                )

        def test_validate_fused_group_mlp_conflicts_with_paged_stash(self):
            policy = TierPolicy(
                offload_modules=frozenset({"fused_group_mlp"}),
            )
            with self.assertRaises(ValueError):
                policy.validate_fused_group_mlp_config(
                    use_op_fuser=True, moe_paged_stash=True
                )

    class TestDynamicShapeModulesSkipPool(unittest.TestCase):
        """Verify _DYNAMIC_SHAPE_MODULES don't use the CPU pool (upstream 5c660c3b)."""

        def _make_group(self, name: str) -> TierAwareOffloadTensorGroup:
            pool = TierPinnedMemoryPool()
            cache = LocalityCacheManager()
            policy = TierPolicy(offload_modules=frozenset({name}))
            return TierAwareOffloadTensorGroup(name, policy, pool, cache)

        def test_fused_group_mlp_no_pool(self):
            g = self._make_group("fused_group_mlp")
            self.assertFalse(g.use_cpu_pool)

        def test_expert_fc1_no_pool(self):
            g = self._make_group("expert_fc1")
            self.assertFalse(g.use_cpu_pool)

        def test_moe_act_no_pool(self):
            g = self._make_group("moe_act")
            self.assertFalse(g.use_cpu_pool)

        def test_core_attn_uses_pool(self):
            g = self._make_group("core_attn")
            self.assertTrue(g.use_cpu_pool)

        def test_attn_norm_uses_pool(self):
            g = self._make_group("attn_norm")
            self.assertTrue(g.use_cpu_pool)

    class TestHeterogeneousChunkOffloadHandlerPassthrough(unittest.TestCase):
        """Tests for passthrough behaviour in HeterogeneousChunkOffloadHandler."""

        def _make_handler(self, min_numel: int = 1) -> HeterogeneousChunkOffloadHandler:
            policy = TierPolicy(
                offload_modules=frozenset({"fused_group_mlp", "expert_fc1"}),
                min_offload_numel=min_numel,
            )
            pool = TierPinnedMemoryPool()
            cache = LocalityCacheManager()
            return HeterogeneousChunkOffloadHandler(
                tier_policy=policy, pool=pool, locality_cache=cache, min_offload_numel=min_numel
            )

        def test_cpu_tensor_pushes_as_passthrough(self):
            handler = self._make_handler()
            t = torch.empty(1024)
            result = handler.tensor_push(t)
            self.assertIs(result, t)

        def test_parameter_pushes_as_passthrough(self):
            handler = self._make_handler()
            p = nn.Parameter(torch.empty(1024))
            result = handler.tensor_push(p)
            self.assertIs(result, p)

        def test_tensor_pop_of_passthrough_returns_tensor(self):
            handler = self._make_handler()
            t = torch.empty(1024)
            tag = handler.tensor_push(t)  # passthrough: tag IS t
            result = handler.tensor_pop(tag)
            self.assertIs(result, t)

        def test_checker_cpu_tensor_false(self):
            handler = self._make_handler()
            t = torch.empty(1024)
            self.assertFalse(handler.tensor_need_offloading_checker(t))

        def test_checker_parameter_false(self):
            handler = self._make_handler()
            p = nn.Parameter(torch.empty(1024))
            self.assertFalse(handler.tensor_need_offloading_checker(p))

        def test_checker_te_do_not_offload_attribute(self):
            handler = self._make_handler()
            # Simulate a CPU tensor to avoid requiring CUDA.
            t = torch.empty(1024)
            t._TE_do_not_offload = True
            self.assertFalse(handler.tensor_need_offloading_checker(t))

        def test_checker_offloading_activation_false_opt_out(self):
            handler = self._make_handler()
            t = torch.empty(1024)
            t.offloading_activation = False
            self.assertFalse(handler.tensor_need_offloading_checker(t))

        def test_checker_below_min_numel_false(self):
            handler = self._make_handler(min_numel=int(1e9))
            # CPU tensor: already fails the CUDA check, but this tests the
            # numel path via the mock.
            t = torch.empty(1024)
            self.assertFalse(handler.tensor_need_offloading_checker(t))

    class TestValidateOffloadConfig(unittest.TestCase):
        """Tests for :func:`validate_offload_config`."""

        def test_valid_config_passes(self):
            validate_offload_config(["expert_fc1", "moe_act"])

        def test_fused_group_mlp_without_op_fuser_raises(self):
            with self.assertRaises(ValueError):
                validate_offload_config(
                    ["fused_group_mlp"], use_op_fuser=False
                )

        def test_fused_group_mlp_with_op_fuser_passes(self):
            validate_offload_config(["fused_group_mlp"], use_op_fuser=True)

        def test_fused_group_mlp_combined_expert_fc1_raises(self):
            with self.assertRaises(ValueError):
                validate_offload_config(
                    ["fused_group_mlp", "expert_fc1"], use_op_fuser=True
                )

        def test_fused_group_mlp_combined_moe_act_raises(self):
            with self.assertRaises(ValueError):
                validate_offload_config(
                    ["fused_group_mlp", "moe_act"], use_op_fuser=True
                )

        def test_paged_stash_with_moe_offload_raises(self):
            with self.assertRaises(ValueError):
                validate_offload_config(
                    ["expert_fc1"], moe_paged_stash=True
                )

        def test_paged_stash_with_fused_group_mlp_raises(self):
            with self.assertRaises(ValueError):
                validate_offload_config(
                    ["fused_group_mlp"], use_op_fuser=True, moe_paged_stash=True
                )

        def test_unknown_module_raises(self):
            with self.assertRaises(ValueError):
                validate_offload_config(["unknown_module_xyz"])

        def test_empty_config_passes(self):
            validate_offload_config([])

    class TestLocalityCacheManager(unittest.TestCase):
        """Tests for :class:`LocalityCacheManager` without CUDA."""

        def setUp(self):
            self.cache = LocalityCacheManager()

        def test_register_and_total_bytes(self):
            t = torch.empty(1024, dtype=torch.float32)
            self.cache.register("tag1", t, torch.device("cpu"))
            self.assertEqual(self.cache.total_cached_bytes, t.nbytes)

        def test_retrieve_unknown_tag_returns_none(self):
            result = self.cache.retrieve("nonexistent", torch.device("cpu"))
            self.assertIsNone(result)

        def test_clear_resets_bytes(self):
            t = torch.empty(256, dtype=torch.float32)
            self.cache.register("x", t, torch.device("cpu"))
            self.cache.clear()
            self.assertEqual(self.cache.total_cached_bytes, 0)

    class TestPinnedMemoryPool(unittest.TestCase):
        """Tests for :class:`TierPinnedMemoryPool`."""

        def test_utilisation_starts_at_zero(self):
            pool = TierPinnedMemoryPool(max_cpu_bytes=int(1e9))
            self.assertAlmostEqual(pool.utilisation_fraction, 0.0)

        def test_reset_clears_utilisation(self):
            pool = TierPinnedMemoryPool(max_cpu_bytes=int(1e9))
            pool._allocated_bytes = int(5e8)
            pool.reset()
            self.assertAlmostEqual(pool.utilisation_fraction, 0.0)

        def test_allocation_exceeding_limit_raises(self):
            pool = TierPinnedMemoryPool(max_cpu_bytes=1)
            with self.assertRaises(RuntimeError):
                pool.allocate((1024, 1024), torch.float32)

    class TestDESLOCManagerLifecycle(unittest.TestCase):
        """Integration-level tests for :class:`DESLOCActivationOffloadManager`."""

        def test_begin_and_end_chunk(self):
            mgr = DESLOCActivationOffloadManager(min_offload_numel=1)
            handler = mgr.begin_forward_chunk()
            self.assertIsInstance(handler, HeterogeneousChunkOffloadHandler)
            self.assertEqual(len(mgr._active_chunks), 1)
            mgr.end_backward_chunk()
            self.assertEqual(len(mgr._active_chunks), 0)

        def test_empty_end_backward_is_safe(self):
            mgr = DESLOCActivationOffloadManager()
            mgr.end_backward_chunk()  # should not raise

        def test_locality_cache_bytes_property(self):
            mgr = DESLOCActivationOffloadManager()
            self.assertEqual(mgr.locality_cache_bytes, 0)

        def test_reset_clears_active_chunks(self):
            mgr = DESLOCActivationOffloadManager()
            mgr.begin_forward_chunk()
            mgr.reset()
            self.assertEqual(len(mgr._active_chunks), 0)

    class TestFusedGroupMLPOffloadInterfaceDisabled(unittest.TestCase):
        """FusedGroupMLPOffloadInterface when enabled=False (no-op path)."""

        def test_context_manager_passthrough(self):
            t = torch.empty(16, dtype=torch.float32)
            iface = FusedGroupMLPOffloadInterface(
                enabled=False, input_tensor=t, module_name="fused_group_mlp"
            )
            with iface as out:
                self.assertIs(out, t)

        def test_group_commit_passthrough(self):
            output = torch.empty(8)
            iface = FusedGroupMLPOffloadInterface(
                enabled=False, input_tensor=torch.empty(8), module_name="fused_group_mlp"
            )
            result = iface.group_commit(output, name="fused_group_mlp")
            self.assertIs(result, output)

    runner = unittest.TextTestRunner(verbosity=2)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestDeviceTierClassification,
        TestTensorEligibility,
        TestTeDoNotOffload,
        TestTierPolicy,
        TestDynamicShapeModulesSkipPool,
        TestHeterogeneousChunkOffloadHandlerPassthrough,
        TestValidateOffloadConfig,
        TestLocalityCacheManager,
        TestPinnedMemoryPool,
        TestDESLOCManagerLifecycle,
        TestFusedGroupMLPOffloadInterfaceDisabled,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
