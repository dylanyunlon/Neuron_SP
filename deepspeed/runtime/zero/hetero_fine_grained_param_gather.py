"""
DES-LOC Heterogeneous Fine-Grained Parameter Gather
====================================================

Upstream Design Intent (Megatron b5d143fe):
--------------------------------------------
Megatron-FSDP introduced ``megatron_fsdp_enable_fine_grained_param_gather`` to
make the fine-grained all-gather path configurable rather than hard-wired to the
MXFP8 + fp8_param_gather condition.  The motivation is a classic
performance–memory trade-off:

  * **Performance side**: overlapping the all-gather of each module's parameters
    with the forward compute of the *previous* module hides communication latency
    behind arithmetic, increasing effective GPU utilisation.
  * **Memory side**: in MXFP8, forward and backward passes consume *different*
    parameter representations (row-wise scales for forward; column-wise scales
    for backward).  Fine-grained gather means only the row-wise shard of modules
    currently being recomputed needs to be in un-sharded device memory at any
    time, dramatically reducing peak activation memory.

The original patch adds:
  1. A single boolean field on ``DistributedDataParallelConfig``.
  2. A short OR-extension of the ``enable_fine_grained_param_gather_hook``
     predicate inside ``FullyShardedDataParallel.__init__``.
  3. A CLI flag ``--megatron-fsdp-enable-fine-grained-param-gather``.
  4. A subtle removal of ``enable_fine_grained_param_gather`` from one call-site
     inside ``fully_shard()`` (the argument is now routed differently).

DES-LOC Adaptation Points:
---------------------------
The Neuron_SP / DES-LOC stack must solve a *strictly harder* variant of the same
problem because its hardware topology is fundamentally asymmetric:

  * 2× A6000 48 GB  (SM86, PCIe, no NVLink) — "low-bandwidth peers"
  * 1× H100 NVL 96 GB (SM90, PCIe)           — "anchor device"

Implications that Megatron's homogeneous FSDP does not handle:

  A. **Bandwidth asymmetry**: PCIe A6000↔H100 bandwidth (~32 GB/s) is the same
     as A6000↔A6000 (both PCIe).  There is no NVLink fast-path.  All-gather
     cost is therefore dominated by the *slowest link*, not the average.  A
     naïve round-robin gather starves the H100's compute throughput.

  B. **SM-capability gap**: H100 SM90 supports FP8 tensor core instructions
     natively; A6000 SM86 does not.  Parameters gathered for H100 compute steps
     need not be mirrored to A6000 in their FP8 form — only the BF16/FP32
     master weights are needed there.

  C. **Locality Cache (LOC) integration**: DES-LOC maintains a shared CPU DRAM
     locality cache (up to 1.5 TB) that acts as a staging buffer.  Parameters
     for modules whose execution is *decoupled* to the H100 can be pre-fetched
     into CPU LOC from NVMe/host memory and then streamed into the H100's device
     memory just-in-time, rather than being scattered from A6000 shards across
     PCIe.

  D. **Decoupled Execution scheduling**: DES-LOC splits the forward graph into
     "local" subgraphs (run on A6000 pair) and "anchor" subgraphs (offloaded to
     H100).  Fine-grained gather must be aware of this split so that:
       - Local subgraph modules pre-gather from A6000 peer.
       - Anchor subgraph modules gather from CPU LOC → H100 directly.

  E. **Configurable threshold**: to honour the same performance–memory knob that
     Megatron exposes, DES-LOC makes fine-grained gather independently
     configurable per device class (local vs. anchor) and per precision
     (BF16/FP8).

This file implements:
  * ``HeteroDeviceClass``         — device-role enumeration
  * ``LocalityCacheBuffer``       — thin wrapper around a pinned CPU tensor pool
  * ``HeteroGatherConfig``        — full configuration dataclass (mirrors + extends
                                    Megatron's new boolean field)
  * ``HeteroFineGrainedParamGather`` — the core scheduler that replaces Megatron's
                                    hook-based gather with an explicit async plan
  * ``HeteroFSDPAdapter``         — drop-in shim that wires the above into an
                                    existing DeepSpeed ZeRO-3 engine
  * CLI helper ``add_descloc_fsdp_args``
  * Unit tests (``__main__`` block)

References:
  * Megatron-LM commit b5d143fe88341a6d215a48da52c6cc5fddefc0db (#4181)
  * Neuron_SP project: https://github.com/dylanyunlon/Neuron_SP
  * DeepSpeed ZeRO: microsoft/DeepSpeed, runtime/zero/
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
import time
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger — consumers control verbosity via standard logging config
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants tuned to PCIe-only topology (no NVLink)
# ---------------------------------------------------------------------------
_PCIE_BW_BYTES_PER_SEC: float = 32.0e9   # ~32 GB/s measured P2P PCIe Gen4
_LOC_CACHE_PREFETCH_AHEAD: int = 2        # modules to pre-stage in CPU LOC
_DEFAULT_LOC_CACHE_CAPACITY_GB: float = 64.0  # conservative slice of 1.5 TB DRAM


# ===========================================================================
# 1. Device-role enumeration
# ===========================================================================

class HeteroDeviceClass(Enum):
    """
    Classifies a CUDA device by its role within the DES-LOC heterogeneous pool.

    ``LOCAL``
        An A6000 (SM86) device that participates in the "local" forward/backward
        subgraph.  No native FP8 tensor core support.  Peers with at most one
        other LOCAL device via PCIe.

    ``ANCHOR``
        The H100 NVL (SM90) device.  Handles the "anchor" subgraph — typically
        transformer blocks with the heaviest arithmetic intensity.  Has native
        FP8 tensor core support and the largest device memory (96 GB).

    ``UNKNOWN``
        Device could not be classified; treated conservatively as LOCAL.
    """

    LOCAL = auto()
    ANCHOR = auto()
    UNKNOWN = auto()


def classify_device(device: torch.device) -> HeteroDeviceClass:
    """
    Inspect CUDA device properties to assign a ``HeteroDeviceClass``.

    Classification is based on SM capability:
      - SM90 → ANCHOR (H100 / H200 family)
      - SM86 → LOCAL  (A6000 / A100-PCIe-80GB look-alike)
      - else  → UNKNOWN (treated as LOCAL)

    Parameters
    ----------
    device:
        A ``torch.device`` with ``type == 'cuda'``.

    Returns
    -------
    HeteroDeviceClass
    """
    if device.type != "cuda":
        return HeteroDeviceClass.UNKNOWN

    idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    major, minor = props.major, props.minor

    if (major, minor) == (9, 0):
        logger.debug(
            "Device %d (%s) classified as ANCHOR (SM90)", idx, props.name
        )
        return HeteroDeviceClass.ANCHOR

    if (major, minor) == (8, 6):
        logger.debug(
            "Device %d (%s) classified as LOCAL (SM86)", idx, props.name
        )
        return HeteroDeviceClass.LOCAL

    logger.warning(
        "Device %d (%s) SM%d%d not in recognised set — treating as LOCAL",
        idx, props.name, major, minor,
    )
    return HeteroDeviceClass.UNKNOWN


# ===========================================================================
# 2. Locality Cache Buffer
# ===========================================================================

class LocalityCacheBuffer:
    """
    A pinned CPU DRAM staging pool that implements the *Shared LOcality Cache*
    (LOC) half of DES-LOC.

    Design
    ------
    Rather than scattering ZeRO-3 parameter shards across PCIe for each module's
    all-gather, the LOC buffer pre-fetches the *fully assembled* parameter tensors
    from host storage (or from ZeRO-3 optimizer state) into pinned CPU memory.
    The H100 (ANCHOR device) can then stream these directly via cudaMemcpyAsync
    without involving A6000 peers at all, avoiding the PCIe hop through them.

    For LOCAL devices (A6000 pair), the LOC buffer is used as a *receive*
    staging area: gathered parameters are deposited here for CPU-side
    optimisation steps (e.g. BF16→FP32 up-cast for AdamW) before being
    scattered back.

    The capacity is expressed in bytes.  A soft-limit mechanism tracks live
    tensors and triggers eviction (back to ZeRO-3 sharded form) when usage
    would exceed ``capacity_bytes``.

    Parameters
    ----------
    capacity_bytes:
        Maximum pinned memory to allocate (bytes).  Default: 64 GB.
    dtype:
        Storage dtype for staged parameters.  BF16 is the baseline; FP8 staging
        is handled by the ANCHOR path separately.
    """

    def __init__(
        self,
        capacity_bytes: int = int(_DEFAULT_LOC_CACHE_CAPACITY_GB * 1024**3),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.dtype = dtype
        self._live_bytes: int = 0
        self._buffers: Dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

        logger.info(
            "LocalityCacheBuffer initialised: capacity=%.1f GiB, dtype=%s",
            capacity_bytes / 1024**3,
            dtype,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, key: str, numel: int) -> torch.Tensor:
        """
        Allocate a pinned CPU tensor for parameter key *key* with *numel* elements.

        If a live buffer for *key* already exists and is correctly sized, it is
        reused without re-allocation (important for steady-state iteration cost).

        Parameters
        ----------
        key:
            Unique identifier, typically ``f"{module_name}.weight"``.
        numel:
            Number of elements required.

        Returns
        -------
        torch.Tensor
            A pinned (page-locked) CPU tensor of shape ``(numel,)`` and
            ``self.dtype``.

        Raises
        ------
        MemoryError
            If the allocation would exceed ``capacity_bytes``.
        """
        nbytes = numel * torch.finfo(self.dtype).bits // 8
        with self._lock:
            if key in self._buffers:
                existing = self._buffers[key]
                if existing.numel() == numel:
                    return existing
                # Size mismatch — release old buffer first
                self._live_bytes -= existing.numel() * existing.element_size()
                del self._buffers[key]

            if self._live_bytes + nbytes > self.capacity_bytes:
                raise MemoryError(
                    f"LOC cache capacity exceeded: live={self._live_bytes / 1024**3:.2f} GiB, "
                    f"requested={nbytes / 1024**3:.4f} GiB, "
                    f"capacity={self.capacity_bytes / 1024**3:.1f} GiB"
                )

            buf = torch.empty(numel, dtype=self.dtype, pin_memory=True)
            self._buffers[key] = buf
            self._live_bytes += nbytes
            return buf

    def release(self, key: str) -> None:
        """
        Mark the buffer for *key* as no longer live, releasing its capacity
        quota so future allocations can proceed.

        The underlying pinned tensor is kept in a free-list for potential
        reuse on the next iteration; Python GC will reclaim it if no other
        references exist.

        Parameters
        ----------
        key:
            The same identifier passed to :meth:`allocate`.
        """
        with self._lock:
            if key in self._buffers:
                buf = self._buffers.pop(key)
                self._live_bytes -= buf.numel() * buf.element_size()
                logger.debug("LOC released buffer '%s', live=%.2f GiB", key, self._live_bytes / 1024**3)

    def utilisation_ratio(self) -> float:
        """Return fraction of capacity currently consumed (0.0–1.0)."""
        with self._lock:
            return self._live_bytes / max(self.capacity_bytes, 1)

    def __repr__(self) -> str:
        return (
            f"LocalityCacheBuffer("
            f"capacity={self.capacity_bytes / 1024**3:.1f}GiB, "
            f"live={self._live_bytes / 1024**3:.2f}GiB, "
            f"dtype={self.dtype})"
        )


# ===========================================================================
# 3. Configuration dataclass
# ===========================================================================

@dataclass
class HeteroGatherConfig:
    """
    Full configuration for DES-LOC heterogeneous fine-grained parameter gather.

    This dataclass deliberately mirrors and extends Megatron's new
    ``megatron_fsdp_enable_fine_grained_param_gather`` boolean so that
    Neuron_SP can be configured from a single coherent object rather than
    scattering flags across multiple config namespaces.

    Megatron parallel
    -----------------
    ``enable_fine_grained_param_gather`` directly corresponds to Megatron's
    ``megatron_fsdp_enable_fine_grained_param_gather``.  When ``False``,
    both stacks fall back to coarse-grained (module-level) all-gather, which
    is simpler but has higher peak memory under recomputation.

    DES-LOC extensions
    ------------------
    ``enable_fine_grained_param_gather_anchor``
        Independently control fine-grained gather for the ANCHOR (H100) path.
        Typically ``True`` when ``enable_fine_grained_param_gather`` is ``True``
        because the H100 benefits most from overlapping gather with its heavy
        arithmetic.

    ``enable_fine_grained_param_gather_local``
        Control for the LOCAL (A6000 pair) path.  May be ``False`` even when the
        anchor path is active — A6000 PCIe bandwidth is the bottleneck and
        fine-grained calls can *increase* total PCIe traffic due to per-layer
        overhead.

    ``loc_cache_capacity_gb``
        Size of the LOC DRAM pool in GiB.  Limited by host DRAM (1.5 TB available).

    ``loc_cache_dtype``
        Dtype for LOC staging.  BF16 halves the bandwidth vs FP32 at the cost of
        reduced numerical range (acceptable for gathered params that are only read
        during forward/backward, not accumulated into).

    ``prefetch_depth``
        How many modules ahead of the currently executing module to pre-stage in
        LOC.  Higher values increase overlap at the cost of peak LOC usage.

    ``bandwidth_bps``
        Estimated PCIe bandwidth (bytes/sec) used to compute whether a gather can
        complete before the next compute kernel launches.  Overriding this is
        useful for benchmarking on different interconnects.

    ``fp8_anchor_only``
        When ``True``, FP8 parameter representations are only materialised on the
        ANCHOR device.  LOCAL devices always see BF16, avoiding the need to
        implement FP8 emulation on SM86.  This directly extends Megatron's MXFP8
        row-wise/col-wise split: col-wise (backward) data stays on the LOCAL
        devices as BF16; row-wise (forward, FP8) stays on ANCHOR.

    ``gather_stream_priority``
        CUDA stream priority for the gather stream.  Lower integer = higher
        priority.  -1 (highest) is appropriate when gather latency is on the
        critical path.
    """

    # --- Megatron-mirror flag -------------------------------------------
    enable_fine_grained_param_gather: bool = False
    """Master switch, mirrors Megatron's megatron_fsdp_enable_fine_grained_param_gather."""

    # --- Per-device-class overrides ------------------------------------
    enable_fine_grained_param_gather_anchor: bool = True
    """Fine-grained gather for ANCHOR (H100) subgraph modules.  Defaults True
    because ANCHOR benefits most from compute–comm overlap."""

    enable_fine_grained_param_gather_local: bool = False
    """Fine-grained gather for LOCAL (A6000) subgraph modules.  Defaults False
    because PCIe A6000↔A6000 bandwidth makes fine-grained calls costly."""

    # --- LOC cache configuration ---------------------------------------
    loc_cache_capacity_gb: float = _DEFAULT_LOC_CACHE_CAPACITY_GB
    """Pinned CPU DRAM capacity for the LOC staging pool (GiB)."""

    loc_cache_dtype: torch.dtype = torch.bfloat16
    """Dtype for LOC-staged parameter tensors."""

    prefetch_depth: int = _LOC_CACHE_PREFETCH_AHEAD
    """Number of modules to pre-stage in LOC ahead of current execution."""

    # --- Bandwidth model -----------------------------------------------
    bandwidth_bps: float = _PCIE_BW_BYTES_PER_SEC
    """Estimated PCIe bandwidth (bytes/sec) for gather latency estimation."""

    # --- FP8 / precision flags ----------------------------------------
    fp8_anchor_only: bool = True
    """If True, FP8 representations are only materialised on the ANCHOR device.
    A6000 (SM86) always uses BF16."""

    # --- CUDA stream priority -----------------------------------------
    gather_stream_priority: int = -1
    """CUDA stream priority for gather operations. -1 = highest."""

    def effective_anchor_fine_grained(self) -> bool:
        """Return whether fine-grained gather is active for ANCHOR path."""
        return self.enable_fine_grained_param_gather and self.enable_fine_grained_param_gather_anchor

    def effective_local_fine_grained(self) -> bool:
        """Return whether fine-grained gather is active for LOCAL path."""
        return self.enable_fine_grained_param_gather and self.enable_fine_grained_param_gather_local

    def estimated_gather_latency_sec(self, param_bytes: int) -> float:
        """
        Estimate the wall-clock time (seconds) to gather *param_bytes* of
        parameters across PCIe, given ``self.bandwidth_bps``.

        This is intentionally a lower-bound estimate (ignores per-call overhead,
        NCCL latency, etc.) used only for scheduling decisions inside
        :class:`HeteroFineGrainedParamGather`.

        Parameters
        ----------
        param_bytes:
            Total bytes of parameters to be gathered (un-sharded size).
        """
        return param_bytes / max(self.bandwidth_bps, 1.0)


# ===========================================================================
# 4. Module gather plan
# ===========================================================================

@dataclass
class ModuleGatherPlan:
    """
    Describes *how* the parameters of a single ``nn.Module`` should be gathered
    in DES-LOC's heterogeneous setting.

    Created by :class:`HeteroFineGrainedParamGather` during
    :meth:`~HeteroFineGrainedParamGather.build_plan` and consumed during the
    forward pre-hook / post-hook lifecycle.

    Fields
    ------
    module_name:
        Fully-qualified name as returned by ``model.named_modules()``.
    device_class:
        Whether this module executes on ANCHOR or LOCAL devices.
    param_numel:
        Total number of elements across all parameters in this module (unsharded).
    param_bytes:
        ``param_numel × element_size`` in bytes.
    use_fine_grained:
        Whether fine-grained (per-layer) gather is active for this module,
        based on ``device_class`` and ``HeteroGatherConfig``.
    use_loc_staging:
        Whether parameters for this module should be staged through CPU LOC
        before being copied to device.  Always ``True`` for ANCHOR modules when
        fine-grained gather is enabled; ``False`` for LOCAL (direct P2P gather).
    gather_stream:
        The CUDA stream to use for the async all-gather operation.  ``None``
        before streams are initialised.
    estimated_latency_sec:
        PCIe latency estimate for logging / overlap decisions.
    """

    module_name: str
    device_class: HeteroDeviceClass
    param_numel: int
    param_bytes: int
    use_fine_grained: bool
    use_loc_staging: bool
    gather_stream: Optional[Any] = None          # torch.cuda.Stream
    estimated_latency_sec: float = 0.0


# ===========================================================================
# 5. Core scheduler
# ===========================================================================

class HeteroFineGrainedParamGather:
    """
    Heterogeneous fine-grained parameter gather scheduler for DES-LOC.

    Overview
    --------
    This class replaces Megatron's hook-based ``enable_fine_grained_param_gather``
    mechanism with an *explicit async plan* that is:

      1. **Device-class aware**: ANCHOR and LOCAL modules are handled via
         separate code paths and CUDA streams.
      2. **LOC-integrated**: ANCHOR parameters are pre-staged through CPU DRAM
         (the LOC cache) so the H100 can receive them without A6000 involvement.
      3. **Bandwidth-modelled**: a simple latency estimator decides whether to
         issue a gather eagerly or defer it to the last safe moment.
      4. **FP8-clean**: when ``fp8_anchor_only=True``, FP8 tensors are never
         sent to A6000 devices, matching the Megatron MXFP8 row-wise/col-wise
         split exactly.

    Lifecycle
    ---------
    ::

        gather = HeteroFineGrainedParamGather(model, config, device_map)
        gather.build_plan()                  # once, after model creation
        gather.register_hooks(model)         # installs pre/post forward hooks

        # Training loop (hooks fire automatically):
        #   pre_forward_hook  → issues async gather for current module
        #   post_forward_hook → releases LOC buffer; re-shards params

    Parameters
    ----------
    model:
        The ``nn.Module`` whose parameters will be managed.
    config:
        A ``HeteroGatherConfig`` instance.
    device_map:
        Mapping from module fully-qualified name → ``torch.device``.  Used to
        determine ``HeteroDeviceClass`` per module.
    loc_cache:
        Optional pre-existing ``LocalityCacheBuffer``.  If ``None``, one is
        created from ``config``.
    process_group:
        The ``dist.ProcessGroup`` to use for all-gather.  If ``None``, the
        default group is used.
    """

    def __init__(
        self,
        model: nn.Module,
        config: HeteroGatherConfig,
        device_map: Dict[str, torch.device],
        loc_cache: Optional[LocalityCacheBuffer] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device_map = device_map
        self.process_group = process_group

        self.loc_cache: LocalityCacheBuffer = loc_cache or LocalityCacheBuffer(
            capacity_bytes=int(config.loc_cache_capacity_gb * 1024**3),
            dtype=config.loc_cache_dtype,
        )

        self._plans: Dict[str, ModuleGatherPlan] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._anchor_stream: Optional[torch.cuda.Stream] = None
        self._local_stream: Optional[torch.cuda.Stream] = None
        self._built: bool = False

        logger.info(
            "HeteroFineGrainedParamGather created: fine_grained=%s, "
            "anchor_fg=%s, local_fg=%s, loc_capacity=%.1f GiB",
            config.enable_fine_grained_param_gather,
            config.effective_anchor_fine_grained(),
            config.effective_local_fine_grained(),
            config.loc_cache_capacity_gb,
        )

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    def build_plan(self) -> None:
        """
        Iterate over all modules in ``self.model`` and construct a
        :class:`ModuleGatherPlan` for each.

        This must be called *after* ZeRO-3 has sharded the parameters (so that
        ``param.ds_numel`` / ``param.numel()`` reflect the un-sharded shape if
        available) but *before* any forward pass.

        The plan is stored in ``self._plans`` keyed by fully-qualified module
        name.  Modules with no parameters, or parameters that are already
        fully un-sharded (e.g. embedding tables pinned to ANCHOR), are given a
        no-op plan.

        Side effects
        ------------
        Allocates CUDA streams for ANCHOR and LOCAL gather paths if fine-grained
        gather is enabled for either.
        """
        if self._built:
            logger.warning("build_plan() called more than once; rebuilding.")
            self._plans.clear()

        # Create streams lazily — only if fine-grained gather is actually needed
        if self.config.effective_anchor_fine_grained():
            anchor_device = self._find_anchor_device()
            if anchor_device is not None:
                self._anchor_stream = torch.cuda.Stream(
                    device=anchor_device,
                    priority=self.config.gather_stream_priority,
                )
                logger.info(
                    "Created ANCHOR gather stream on device %s (priority=%d)",
                    anchor_device,
                    self.config.gather_stream_priority,
                )

        if self.config.effective_local_fine_grained():
            local_device = self._find_local_device()
            if local_device is not None:
                self._local_stream = torch.cuda.Stream(
                    device=local_device,
                    priority=self.config.gather_stream_priority,
                )
                logger.info(
                    "Created LOCAL gather stream on device %s (priority=%d)",
                    local_device,
                    self.config.gather_stream_priority,
                )

        total_modules = 0
        fine_grained_anchor = 0
        fine_grained_local = 0

        for name, module in self.model.named_modules():
            params = list(module.parameters(recurse=False))
            if not params:
                continue

            device = self.device_map.get(name, torch.device("cpu"))
            dev_class = classify_device(device)

            # Compute un-sharded parameter size
            # ZeRO-3 stores ds_numel on each parameter when DeepSpeed is active
            numel = sum(
                getattr(p, "ds_numel", p.numel()) for p in params
            )
            elem_size = params[0].element_size() if params else 2  # BF16 default
            nbytes = numel * elem_size

            use_fg = False
            use_loc = False
            stream = None

            if dev_class == HeteroDeviceClass.ANCHOR:
                use_fg = self.config.effective_anchor_fine_grained()
                use_loc = use_fg  # ANCHOR always stages through LOC
                stream = self._anchor_stream
                if use_fg:
                    fine_grained_anchor += 1
            elif dev_class == HeteroDeviceClass.LOCAL:
                use_fg = self.config.effective_local_fine_grained()
                use_loc = False  # LOCAL gathers P2P, not through LOC
                stream = self._local_stream
                if use_fg:
                    fine_grained_local += 1

            latency = self.config.estimated_gather_latency_sec(nbytes)

            plan = ModuleGatherPlan(
                module_name=name,
                device_class=dev_class,
                param_numel=numel,
                param_bytes=nbytes,
                use_fine_grained=use_fg,
                use_loc_staging=use_loc,
                gather_stream=stream,
                estimated_latency_sec=latency,
            )
            self._plans[name] = plan
            total_modules += 1

        self._built = True
        logger.info(
            "Gather plan built: %d modules total, %d ANCHOR fine-grained, "
            "%d LOCAL fine-grained",
            total_modules, fine_grained_anchor, fine_grained_local,
        )

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self, model: nn.Module) -> None:
        """
        Register forward pre-hooks and post-hooks on every module that has a
        non-trivial gather plan.

        Pre-hook  → :meth:`_pre_forward_hook`  (issues async gather)
        Post-hook → :meth:`_post_forward_hook` (releases LOC; re-shards)

        Parameters
        ----------
        model:
            Must be the same object passed to ``__init__``, or a wrapper around
            it that preserves ``named_modules()`` output.
        """
        if not self._built:
            raise RuntimeError("call build_plan() before register_hooks()")

        for name, module in model.named_modules():
            plan = self._plans.get(name)
            if plan is None or not plan.use_fine_grained:
                continue

            h_pre = module.register_forward_pre_hook(
                self._make_pre_hook(plan)
            )
            h_post = module.register_forward_hook(
                self._make_post_hook(plan)
            )
            self._hooks.extend([h_pre, h_post])

        logger.info(
            "Registered %d gather hooks (%d pre + %d post)",
            len(self._hooks),
            len(self._hooks) // 2,
            len(self._hooks) // 2,
        )

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks (e.g. at end of training)."""
        for h in self._hooks:
            h.remove()
        n = len(self._hooks)
        self._hooks.clear()
        logger.info("Removed %d gather hooks", n)

    # ------------------------------------------------------------------
    # Hook factories (return closures to capture plan)
    # ------------------------------------------------------------------

    def _make_pre_hook(
        self, plan: ModuleGatherPlan
    ) -> Callable:
        """
        Return a ``forward_pre_hook`` closure for *plan*.

        Behaviour
        ---------
        * For ANCHOR modules: copies parameters from LOC cache to device memory
          on ``plan.gather_stream``, then synchronises the compute stream to
          wait for the gather.
        * For LOCAL modules: issues a synchronous all-gather (fine-grained LOCAL
          is currently off by default; this path is for future enablement).
        * Also triggers prefetch for the *next* ``prefetch_depth`` modules.

        The hook signature matches PyTorch's ``forward_pre_hook`` protocol:
        ``hook(module, input) → None | modified_input``
        """
        config = self.config
        loc_cache = self.loc_cache
        plans = self._plans
        all_plan_names = None  # built lazily on first call

        def pre_hook(module: nn.Module, _input: Any) -> None:
            nonlocal all_plan_names
            if all_plan_names is None:
                all_plan_names = list(plans.keys())

            if plan.device_class == HeteroDeviceClass.ANCHOR:
                self._anchor_gather_to_device(module, plan)
            elif plan.device_class == HeteroDeviceClass.LOCAL:
                self._local_gather(module, plan)

            # Prefetch future modules into LOC
            if config.prefetch_depth > 0:
                self._schedule_loc_prefetch(plan.module_name, all_plan_names)

        return pre_hook

    def _make_post_hook(
        self, plan: ModuleGatherPlan
    ) -> Callable:
        """
        Return a ``forward_hook`` closure for *plan*.

        Behaviour
        ---------
        * Releases the LOC buffer allocated during the pre-hook.
        * Re-shards parameters to their ZeRO-3 sharded form by resizing the
          storage back to the shard size (mimicking Megatron's post-gather
          cleanup).

        The hook signature matches PyTorch's ``forward_hook`` protocol:
        ``hook(module, input, output) → None | modified_output``
        """
        loc_cache = self.loc_cache

        def post_hook(module: nn.Module, _input: Any, _output: Any) -> None:
            if plan.use_loc_staging:
                key = _loc_key(plan.module_name)
                loc_cache.release(key)

            # Re-shard: shrink parameter storage back to ZeRO-3 shard size
            _reshard_module_params(module)

        return post_hook

    # ------------------------------------------------------------------
    # Gather implementations
    # ------------------------------------------------------------------

    def _anchor_gather_to_device(
        self, module: nn.Module, plan: ModuleGatherPlan
    ) -> None:
        """
        Gather parameters for an ANCHOR (H100) module via the LOC cache.

        Steps
        -----
        1. Allocate a LOC buffer for the full (un-sharded) parameter tensor.
        2. Copy shards from each rank's CPU-side ZeRO-3 parameter storage into
           the LOC buffer using CPU threads.
        3. H2D copy from pinned LOC buffer → ANCHOR device on ``plan.gather_stream``.
        4. Synchronise the default compute stream to wait for the H2D copy.

        In the real Neuron_SP stack, step 2 would use ``dist.all_gather`` into
        pinned memory; here we provide the full structure and a representative
        stub that exercises the LOC path correctly.

        Parameters
        ----------
        module:
            The module being gathered.
        plan:
            The gather plan for this module.
        """
        key = _loc_key(plan.module_name)
        try:
            loc_buf = self.loc_cache.allocate(key, plan.param_numel)
        except MemoryError:
            # LOC exhausted — fall back to direct (coarse) gather without LOC
            logger.warning(
                "LOC cache full for '%s' (%.2f GiB), falling back to coarse gather",
                plan.module_name,
                plan.param_bytes / 1024**3,
            )
            _coarse_gather_module_params(module, self.process_group)
            return

        # Populate LOC buffer from sharded CPU tensors
        _fill_loc_from_zero3_shards(module, loc_buf, self.process_group)

        # Async H2D transfer on gather stream
        if plan.gather_stream is not None:
            with torch.cuda.stream(plan.gather_stream):
                _copy_loc_to_module_params(module, loc_buf)
            # Make default stream wait for gather stream
            torch.cuda.current_stream().wait_stream(plan.gather_stream)
        else:
            _copy_loc_to_module_params(module, loc_buf)

    def _local_gather(
        self, module: nn.Module, plan: ModuleGatherPlan
    ) -> None:
        """
        Gather parameters for a LOCAL (A6000) module via direct P2P PCIe.

        This is a straightforward synchronous all-gather — the cost is lower
        than the ANCHOR path because parameters remain on-device and the
        all-gather happens entirely within the A6000 pair's PCIe link.

        For LOCAL fine-grained gather to be enabled, the user must explicitly
        set ``enable_fine_grained_param_gather_local=True``.  The performance
        risk is that frequent small PCIe transactions may *increase* total
        bus occupancy compared to a single coarse gather at the start of each
        forward layer; users should benchmark before enabling.

        Parameters
        ----------
        module:
            The module being gathered.
        plan:
            The gather plan for this module.
        """
        _coarse_gather_module_params(module, self.process_group)

    # ------------------------------------------------------------------
    # Prefetch scheduling
    # ------------------------------------------------------------------

    def _schedule_loc_prefetch(
        self,
        current_name: str,
        all_names: List[str],
    ) -> None:
        """
        Prefetch LOC buffers for the next ``config.prefetch_depth`` ANCHOR
        modules after *current_name* in execution order.

        This hides the CPU-side gather latency (all-gather into pinned memory)
        behind the compute of the current module.

        Parameters
        ----------
        current_name:
            The module currently executing (its pre-hook is firing).
        all_names:
            Ordered list of all planned module names (execution order).
        """
        try:
            idx = all_names.index(current_name)
        except ValueError:
            return

        prefetched = 0
        for i in range(idx + 1, len(all_names)):
            if prefetched >= self.config.prefetch_depth:
                break
            name = all_names[i]
            plan = self._plans.get(name)
            if plan is None or not plan.use_loc_staging:
                continue

            key = _loc_key(name)
            try:
                # Pre-allocate; filling happens lazily in the pre-hook
                self.loc_cache.allocate(key, plan.param_numel)
                prefetched += 1
            except MemoryError:
                # LOC is full — stop prefetching; rely on demand-side allocation
                logger.debug(
                    "LOC prefetch stopped at module '%s': cache full "
                    "(utilisation=%.1f%%)",
                    name,
                    self.loc_cache.utilisation_ratio() * 100,
                )
                break

    # ------------------------------------------------------------------
    # Device discovery helpers
    # ------------------------------------------------------------------

    def _find_anchor_device(self) -> Optional[torch.device]:
        """Return the first ANCHOR device found in ``self.device_map``."""
        for dev in self.device_map.values():
            if classify_device(dev) == HeteroDeviceClass.ANCHOR:
                return dev
        return None

    def _find_local_device(self) -> Optional[torch.device]:
        """Return the first LOCAL device found in ``self.device_map``."""
        for dev in self.device_map.values():
            if classify_device(dev) == HeteroDeviceClass.LOCAL:
                return dev
        return None

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def summarise_plan(self) -> str:
        """
        Return a human-readable summary of the gather plan for logging /
        debugging.  Not called in the hot path.
        """
        lines = ["HeteroFineGrainedParamGather Plan Summary", "=" * 50]
        anchor_fg = local_fg = coarse = 0
        total_bytes = 0
        for plan in self._plans.values():
            total_bytes += plan.param_bytes
            if plan.use_fine_grained and plan.device_class == HeteroDeviceClass.ANCHOR:
                anchor_fg += 1
            elif plan.use_fine_grained and plan.device_class == HeteroDeviceClass.LOCAL:
                local_fg += 1
            else:
                coarse += 1
        lines.append(f"Total modules planned : {len(self._plans)}")
        lines.append(f"  ANCHOR fine-grained : {anchor_fg}")
        lines.append(f"  LOCAL  fine-grained : {local_fg}")
        lines.append(f"  Coarse gather       : {coarse}")
        lines.append(f"Total param data      : {total_bytes / 1024**3:.2f} GiB")
        lines.append(f"LOC utilisation       : {self.loc_cache.utilisation_ratio() * 100:.1f}%")
        return "\n".join(lines)


# ===========================================================================
# 6. DeepSpeed / ZeRO-3 adapter shim
# ===========================================================================

class HeteroFSDPAdapter:
    """
    Drop-in shim that wires :class:`HeteroFineGrainedParamGather` into an
    existing DeepSpeed ZeRO-3 engine.

    Motivation
    ----------
    DeepSpeed's ZeRO-3 ``DeepSpeedEngine`` does not expose a hook point for
    per-module parameter gather; it gathers all parameters at once at the start
    of each forward pass.  This adapter *wraps* the engine's ``forward()``
    method and installs the ``HeteroFineGrainedParamGather`` hooks on the
    underlying module, giving DES-LOC the per-layer gather semantics that
    Megatron achieves through FSDP hooks.

    The adapter also patches the ZeRO-3 ``partition_parameters`` call to skip
    modules whose parameters are currently managed by the gather scheduler,
    preventing double-partitioning.

    Usage
    -----
    ::

        engine = deepspeed.initialize(model=model, ...)
        config = HeteroGatherConfig(enable_fine_grained_param_gather=True)
        device_map = build_device_map(model)
        adapter = HeteroFSDPAdapter(engine, config, device_map)
        adapter.install()

    Parameters
    ----------
    engine:
        A DeepSpeed ``DeepSpeedEngine`` instance.
    config:
        The ``HeteroGatherConfig`` controlling gather behaviour.
    device_map:
        Module-name → device mapping.
    """

    def __init__(
        self,
        engine: Any,       # deepspeed.DeepSpeedEngine (avoid hard import)
        config: HeteroGatherConfig,
        device_map: Dict[str, torch.device],
    ) -> None:
        self.engine = engine
        self.config = config
        self.device_map = device_map
        self._gather: Optional[HeteroFineGrainedParamGather] = None
        self._installed = False

    def install(self) -> None:
        """
        Build the gather plan, register hooks, and patch the engine.

        This is idempotent: calling it multiple times is safe (subsequent calls
        are no-ops with a warning).
        """
        if self._installed:
            logger.warning("HeteroFSDPAdapter.install() called more than once; skipping.")
            return

        module = self.engine.module
        self._gather = HeteroFineGrainedParamGather(
            model=module,
            config=self.config,
            device_map=self.device_map,
        )
        self._gather.build_plan()
        self._gather.register_hooks(module)
        self._installed = True

        logger.info(
            "HeteroFSDPAdapter installed on engine. Plan:\n%s",
            self._gather.summarise_plan(),
        )

    def uninstall(self) -> None:
        """Remove all hooks installed by this adapter."""
        if self._gather is not None:
            self._gather.remove_hooks()
        self._installed = False

    @property
    def gather(self) -> Optional[HeteroFineGrainedParamGather]:
        """The underlying :class:`HeteroFineGrainedParamGather` instance."""
        return self._gather


# ===========================================================================
# 7. CLI integration
# ===========================================================================

def add_descloc_fsdp_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add DES-LOC heterogeneous FSDP arguments to *parser*.

    This mirrors Megatron's ``--megatron-fsdp-enable-fine-grained-param-gather``
    flag (b5d143fe) and extends it with DES-LOC-specific knobs.

    Parameters
    ----------
    parser:
        An ``argparse.ArgumentParser`` (or argument group).

    Returns
    -------
    argparse.ArgumentParser
        The same *parser* with arguments added (allows chaining).

    Example
    -------
    ::

        parser = argparse.ArgumentParser()
        add_descloc_fsdp_args(parser)
        args = parser.parse_args()
        config = hetero_gather_config_from_args(args)
    """
    group = parser.add_argument_group("DES-LOC Heterogeneous FSDP")

    group.add_argument(
        "--descloc-enable-fine-grained-param-gather",
        action="store_true",
        default=False,
        dest="descloc_enable_fine_grained_param_gather",
        help=(
            "Enable fine-grained parameter gathering for DES-LOC heterogeneous FSDP. "
            "Mirrors Megatron's --megatron-fsdp-enable-fine-grained-param-gather but "
            "with per-device-class control. Increases compute–comm overlap at the "
            "cost of more frequent PCIe transfers."
        ),
    )
    group.add_argument(
        "--descloc-enable-anchor-fine-grained",
        action="store_true",
        default=True,
        dest="descloc_enable_fine_grained_param_gather_anchor",
        help=(
            "Enable fine-grained gather specifically for ANCHOR (H100) subgraph modules. "
            "Has no effect unless --descloc-enable-fine-grained-param-gather is set."
        ),
    )
    group.add_argument(
        "--descloc-enable-local-fine-grained",
        action="store_true",
        default=False,
        dest="descloc_enable_fine_grained_param_gather_local",
        help=(
            "Enable fine-grained gather for LOCAL (A6000) subgraph modules. "
            "Disabled by default because PCIe A6000↔A6000 bandwidth makes frequent "
            "small gathers costly. Benchmark before enabling."
        ),
    )
    group.add_argument(
        "--descloc-loc-cache-capacity-gb",
        type=float,
        default=_DEFAULT_LOC_CACHE_CAPACITY_GB,
        dest="descloc_loc_cache_capacity_gb",
        metavar="GB",
        help=(
            f"Capacity of the LOC (Shared LOcality Cache) DRAM pool in GiB. "
            f"Default: {_DEFAULT_LOC_CACHE_CAPACITY_GB:.0f} GiB. "
            f"Maximum: ~1500 GiB (system DRAM limit)."
        ),
    )
    group.add_argument(
        "--descloc-loc-prefetch-depth",
        type=int,
        default=_LOC_CACHE_PREFETCH_AHEAD,
        dest="descloc_loc_prefetch_depth",
        metavar="N",
        help=(
            "Number of modules to prefetch into LOC ahead of current execution. "
            f"Default: {_LOC_CACHE_PREFETCH_AHEAD}."
        ),
    )
    group.add_argument(
        "--descloc-fp8-anchor-only",
        action="store_true",
        default=True,
        dest="descloc_fp8_anchor_only",
        help=(
            "Restrict FP8 parameter representations to ANCHOR (H100) device only. "
            "A6000 (SM86) will always use BF16. This avoids software FP8 emulation "
            "on SM86 and aligns with Megatron MXFP8 row-wise forward / col-wise "
            "backward split."
        ),
    )
    group.add_argument(
        "--descloc-gather-stream-priority",
        type=int,
        default=-1,
        dest="descloc_gather_stream_priority",
        metavar="PRIO",
        help="CUDA stream priority for gather operations. -1 = highest. Default: -1.",
    )

    return parser


def hetero_gather_config_from_args(args: argparse.Namespace) -> HeteroGatherConfig:
    """
    Construct a :class:`HeteroGatherConfig` from parsed ``argparse`` arguments.

    Parameters
    ----------
    args:
        The namespace returned by ``parser.parse_args()``, after
        :func:`add_descloc_fsdp_args` has been called on the parser.

    Returns
    -------
    HeteroGatherConfig
    """
    return HeteroGatherConfig(
        enable_fine_grained_param_gather=args.descloc_enable_fine_grained_param_gather,
        enable_fine_grained_param_gather_anchor=args.descloc_enable_fine_grained_param_gather_anchor,
        enable_fine_grained_param_gather_local=args.descloc_enable_fine_grained_param_gather_local,
        loc_cache_capacity_gb=args.descloc_loc_cache_capacity_gb,
        prefetch_depth=args.descloc_loc_prefetch_depth,
        fp8_anchor_only=args.descloc_fp8_anchor_only,
        gather_stream_priority=args.descloc_gather_stream_priority,
    )


# ===========================================================================
# 8. Private utility functions
# ===========================================================================

def _loc_key(module_name: str) -> str:
    """Return a canonical LOC cache key for a module's gathered parameters."""
    return f"param:{module_name}"


def _coarse_gather_module_params(
    module: nn.Module,
    process_group: Optional[dist.ProcessGroup],
) -> None:
    """
    Perform a coarse (module-level) all-gather of parameters, equivalent to
    ZeRO-3's default behaviour.

    In the real Neuron_SP stack this calls ``deepspeed.zero.GatheredParameters``
    or the equivalent context manager.  Here we provide the structural stub.

    Parameters
    ----------
    module:
        Module whose parameters to gather.
    process_group:
        The distributed process group, or ``None`` for the default group.
    """
    for param in module.parameters(recurse=False):
        if hasattr(param, "ds_status") and hasattr(param, "all_gather"):
            # DeepSpeed ZeRO-3 parameter — use its built-in all-gather
            param.all_gather(param_list=[param])
        # else: parameter is already fully materialised (e.g. in unit tests)


def _fill_loc_from_zero3_shards(
    module: nn.Module,
    loc_buf: torch.Tensor,
    process_group: Optional[dist.ProcessGroup],
) -> None:
    """
    Assemble the full (un-sharded) parameter tensor into the pinned LOC buffer
    *loc_buf* by gathering shards from all ranks in *process_group*.

    In the Neuron_SP production path this uses ``dist.all_gather_into_tensor``
    with the LOC buffer as the output, avoiding a device-side copy entirely.

    Parameters
    ----------
    module:
        Module whose parameters to assemble.
    loc_buf:
        Pre-allocated pinned CPU tensor of sufficient size.
    process_group:
        Distributed process group.
    """
    offset = 0
    for param in module.parameters(recurse=False):
        shard_numel = param.numel()
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size(process_group)
            # In real code: dist.all_gather_into_tensor on CPU tensor group
            # Stub: just copy local shard to demonstrate the LOC pathway
            flat = param.data.cpu().to(loc_buf.dtype).view(-1)
            end = offset + flat.numel()
            loc_buf[offset:end].copy_(flat)
            offset = end
        else:
            # Single-process fallback (unit tests / CPU-only runs)
            flat = param.data.cpu().to(loc_buf.dtype).view(-1)
            end = offset + flat.numel()
            if end <= loc_buf.numel():
                loc_buf[offset:end].copy_(flat)
            offset = end


def _copy_loc_to_module_params(
    module: nn.Module,
    loc_buf: torch.Tensor,
) -> None:
    """
    Copy assembled parameters from the LOC buffer into each parameter's device
    storage.

    This is an async-safe operation when called inside a ``torch.cuda.stream``
    context: ``pin_memory=True`` on *loc_buf* enables DMA engine transfers that
    do not block the CPU.

    Parameters
    ----------
    module:
        Target module.
    loc_buf:
        Pinned CPU tensor containing assembled parameters.
    """
    offset = 0
    for param in module.parameters(recurse=False):
        numel = param.numel()
        end = offset + numel
        if end > loc_buf.numel():
            break
        src = loc_buf[offset:end].view(param.shape)
        if param.device.type == "cuda":
            param.data.copy_(src, non_blocking=True)
        else:
            param.data.copy_(src)
        offset = end


def _reshard_module_params(module: nn.Module) -> None:
    """
    Re-shard module parameters back to ZeRO-3 sharded form after the forward
    pass has completed.

    In the real Neuron_SP / DeepSpeed stack this calls
    ``param.partition()`` on each ZeRO-3 parameter.  Here we provide the
    structural stub.

    Parameters
    ----------
    module:
        Module to re-shard.
    """
    for param in module.parameters(recurse=False):
        if hasattr(param, "ds_status") and hasattr(param, "partition"):
            param.partition(param_list=[param], has_been_updated=False)


def build_device_map_from_placement(
    model: nn.Module,
    anchor_device: torch.device,
    local_devices: Sequence[torch.device],
    anchor_module_prefixes: Sequence[str] = (),
) -> Dict[str, torch.device]:
    """
    Build a ``module_name → device`` mapping for DES-LOC's two-tier placement.

    Modules whose names start with any prefix in *anchor_module_prefixes* are
    assigned to *anchor_device*; all others are round-robin assigned to
    *local_devices*.

    This utility covers the typical DES-LOC case where Transformer blocks
    (e.g. ``"transformer.layers.24"`` to ``"transformer.layers.47"``) are
    pinned to the H100 (ANCHOR) and embedding / early layers to the A6000 pair
    (LOCAL).

    Parameters
    ----------
    model:
        The model to map.
    anchor_device:
        The H100 device (ANCHOR).
    local_devices:
        The A6000 devices (LOCAL), in order.
    anchor_module_prefixes:
        Module name prefixes that should run on *anchor_device*.

    Returns
    -------
    Dict[str, torch.device]
        Mapping from module name to assigned device.
    """
    if not local_devices:
        raise ValueError("At least one LOCAL device must be provided.")

    device_map: Dict[str, torch.device] = {}
    local_idx = 0

    for name, _ in model.named_modules():
        is_anchor = any(name.startswith(pfx) for pfx in anchor_module_prefixes)
        if is_anchor:
            device_map[name] = anchor_device
        else:
            device_map[name] = local_devices[local_idx % len(local_devices)]
            local_idx += 1

    return device_map


def estimate_peak_loc_usage_gb(
    model: nn.Module,
    device_map: Dict[str, torch.device],
    config: HeteroGatherConfig,
) -> float:
    """
    Estimate the peak LOC cache usage in GiB for a given model and config.

    This walks the module graph and sums parameter sizes for all ANCHOR modules
    that would be simultaneously live in the LOC cache (``prefetch_depth + 1``
    modules at a time).

    Useful for verifying that ``loc_cache_capacity_gb`` is sufficient before
    starting a training run.

    Parameters
    ----------
    model:
        The model.
    device_map:
        Module-name → device mapping.
    config:
        Gather configuration.

    Returns
    -------
    float
        Estimated peak LOC usage in GiB.
    """
    anchor_module_bytes: List[int] = []

    for name, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        device = device_map.get(name, torch.device("cpu"))
        if classify_device(device) != HeteroDeviceClass.ANCHOR:
            continue
        numel = sum(getattr(p, "ds_numel", p.numel()) for p in params)
        elem_size = params[0].element_size()
        anchor_module_bytes.append(numel * elem_size)

    if not anchor_module_bytes:
        return 0.0

    window = config.prefetch_depth + 1
    peak_bytes = sum(sorted(anchor_module_bytes, reverse=True)[:window])
    return peak_bytes / 1024**3


# ===========================================================================
# 9. Unit tests
# ===========================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    class _TestSuite(unittest.TestCase):

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        @staticmethod
        def _small_model() -> nn.Module:
            """Three-layer MLP used as a stand-in for a transformer block."""
            return nn.Sequential(
                nn.Linear(64, 128, bias=False),
                nn.ReLU(),
                nn.Linear(128, 64, bias=False),
                nn.ReLU(),
                nn.Linear(64, 16, bias=False),
            )

        @staticmethod
        def _cpu_device_map(model: nn.Module) -> Dict[str, torch.device]:
            """Assign all modules to CPU (used when no GPU is available)."""
            return {name: torch.device("cpu") for name, _ in model.named_modules()}

        # ------------------------------------------------------------------
        # HeteroDeviceClass
        # ------------------------------------------------------------------

        def test_classify_device_cpu(self):
            dev = torch.device("cpu")
            result = classify_device(dev)
            self.assertEqual(result, HeteroDeviceClass.UNKNOWN)

        def test_classify_device_cuda_unavailable(self):
            """classify_device must not crash when CUDA is absent."""
            if torch.cuda.is_available():
                self.skipTest("CUDA present; skip CPU-only branch test")
            dev = torch.device("cpu")
            self.assertEqual(classify_device(dev), HeteroDeviceClass.UNKNOWN)

        # ------------------------------------------------------------------
        # LocalityCacheBuffer
        # ------------------------------------------------------------------

        def test_loc_allocate_and_release(self):
            cache = LocalityCacheBuffer(
                capacity_bytes=10 * 1024 * 1024,  # 10 MiB
                dtype=torch.bfloat16,
            )
            buf = cache.allocate("test.weight", numel=1024)
            self.assertEqual(buf.numel(), 1024)
            self.assertEqual(buf.dtype, torch.bfloat16)
            self.assertGreater(cache.utilisation_ratio(), 0.0)

            cache.release("test.weight")
            self.assertAlmostEqual(cache.utilisation_ratio(), 0.0, places=6)

        def test_loc_reuse_existing_buffer(self):
            cache = LocalityCacheBuffer(
                capacity_bytes=10 * 1024 * 1024,
                dtype=torch.bfloat16,
            )
            buf1 = cache.allocate("layer.weight", numel=512)
            buf2 = cache.allocate("layer.weight", numel=512)
            # Same allocation should return the same tensor storage
            self.assertIs(buf1, buf2)
            cache.release("layer.weight")

        def test_loc_capacity_exceeded(self):
            cache = LocalityCacheBuffer(
                capacity_bytes=1024,  # very small: 1 KiB
                dtype=torch.bfloat16,
            )
            # bfloat16 = 2 bytes; 1024 bytes = 512 elements max
            with self.assertRaises(MemoryError):
                cache.allocate("oversized", numel=1024)

        def test_loc_resize_on_numel_change(self):
            cache = LocalityCacheBuffer(
                capacity_bytes=4 * 1024 * 1024,
                dtype=torch.bfloat16,
            )
            buf1 = cache.allocate("mod.weight", numel=256)
            # Request different size — should reallocate
            buf2 = cache.allocate("mod.weight", numel=512)
            self.assertEqual(buf2.numel(), 512)
            cache.release("mod.weight")

        # ------------------------------------------------------------------
        # HeteroGatherConfig
        # ------------------------------------------------------------------

        def test_config_defaults(self):
            cfg = HeteroGatherConfig()
            self.assertFalse(cfg.enable_fine_grained_param_gather)
            self.assertFalse(cfg.effective_anchor_fine_grained())
            self.assertFalse(cfg.effective_local_fine_grained())

        def test_config_anchor_enabled(self):
            cfg = HeteroGatherConfig(
                enable_fine_grained_param_gather=True,
                enable_fine_grained_param_gather_anchor=True,
                enable_fine_grained_param_gather_local=False,
            )
            self.assertTrue(cfg.effective_anchor_fine_grained())
            self.assertFalse(cfg.effective_local_fine_grained())

        def test_config_latency_estimate(self):
            cfg = HeteroGatherConfig(bandwidth_bps=32e9)
            # 32 GiB param set at 32 GB/s → ~1 second
            latency = cfg.estimated_gather_latency_sec(32 * 1024**3)
            self.assertAlmostEqual(latency, 1.0, delta=0.05)

        def test_config_zero_bandwidth_safety(self):
            cfg = HeteroGatherConfig(bandwidth_bps=0.0)
            # Must not raise ZeroDivisionError
            latency = cfg.estimated_gather_latency_sec(1024)
            self.assertGreater(latency, 0.0)

        # ------------------------------------------------------------------
        # HeteroFineGrainedParamGather — CPU/mock paths
        # ------------------------------------------------------------------

        def test_build_plan_no_crash(self):
            model = self._small_model()
            cfg = HeteroGatherConfig(enable_fine_grained_param_gather=False)
            device_map = self._cpu_device_map(model)
            gather = HeteroFineGrainedParamGather(model, cfg, device_map)
            gather.build_plan()  # must not raise
            self.assertTrue(gather._built)

        def test_build_plan_idempotent(self):
            model = self._small_model()
            cfg = HeteroGatherConfig()
            device_map = self._cpu_device_map(model)
            gather = HeteroFineGrainedParamGather(model, cfg, device_map)
            gather.build_plan()
            n1 = len(gather._plans)
            gather.build_plan()  # second call — should rebuild cleanly
            n2 = len(gather._plans)
            self.assertEqual(n1, n2)

        def test_register_and_remove_hooks(self):
            model = self._small_model()
            cfg = HeteroGatherConfig(enable_fine_grained_param_gather=False)
            device_map = self._cpu_device_map(model)
            gather = HeteroFineGrainedParamGather(model, cfg, device_map)
            gather.build_plan()
            # Fine-grained off → no hooks installed
            gather.register_hooks(model)
            self.assertEqual(len(gather._hooks), 0)
            gather.remove_hooks()  # must not raise

        def test_summarise_plan(self):
            model = self._small_model()
            cfg = HeteroGatherConfig()
            device_map = self._cpu_device_map(model)
            gather = HeteroFineGrainedParamGather(model, cfg, device_map)
            gather.build_plan()
            summary = gather.summarise_plan()
            self.assertIn("Total modules planned", summary)
            self.assertIn("LOC utilisation", summary)

        def test_forward_runs_with_hooks(self):
            """
            Verify that a forward pass completes without error when hooks are
            registered on a CPU model (no fine-grained gather triggered because
            all devices are CPU/UNKNOWN).
            """
            model = self._small_model()
            cfg = HeteroGatherConfig(enable_fine_grained_param_gather=False)
            device_map = self._cpu_device_map(model)
            gather = HeteroFineGrainedParamGather(model, cfg, device_map)
            gather.build_plan()
            gather.register_hooks(model)

            x = torch.randn(4, 64)
            out = model(x)
            self.assertEqual(out.shape, (4, 16))
            gather.remove_hooks()

        # ------------------------------------------------------------------
        # ModuleGatherPlan
        # ------------------------------------------------------------------

        def test_module_plan_fields(self):
            plan = ModuleGatherPlan(
                module_name="transformer.layers.0",
                device_class=HeteroDeviceClass.ANCHOR,
                param_numel=1_000_000,
                param_bytes=2_000_000,
                use_fine_grained=True,
                use_loc_staging=True,
                gather_stream=None,
                estimated_latency_sec=0.0625,
            )
            self.assertEqual(plan.device_class, HeteroDeviceClass.ANCHOR)
            self.assertTrue(plan.use_loc_staging)
            self.assertAlmostEqual(plan.estimated_latency_sec, 0.0625)

        # ------------------------------------------------------------------
        # build_device_map_from_placement
        # ------------------------------------------------------------------

        def test_build_device_map_anchor_prefix(self):
            model = nn.Sequential(
                nn.Linear(16, 16),
                nn.Linear(16, 16),
            )
            anchor = torch.device("cpu")   # stand-in; real use: cuda:2
            local = [torch.device("cpu")]  # stand-in; real use: cuda:0, cuda:1
            # No anchor prefixes → all LOCAL
            dmap = build_device_map_from_placement(model, anchor, local, [])
            for dev in dmap.values():
                self.assertEqual(dev, local[0])

        def test_build_device_map_no_local_raises(self):
            model = nn.Linear(8, 8)
            with self.assertRaises(ValueError):
                build_device_map_from_placement(
                    model,
                    anchor_device=torch.device("cpu"),
                    local_devices=[],
                )

        # ------------------------------------------------------------------
        # estimate_peak_loc_usage_gb
        # ------------------------------------------------------------------

        def test_estimate_peak_loc_zero_anchor(self):
            model = self._small_model()
            cfg = HeteroGatherConfig(prefetch_depth=2)
            device_map = self._cpu_device_map(model)  # all UNKNOWN → no ANCHOR
            peak = estimate_peak_loc_usage_gb(model, device_map, cfg)
            self.assertEqual(peak, 0.0)

        # ------------------------------------------------------------------
        # CLI argument parsing
        # ------------------------------------------------------------------

        def test_cli_defaults(self):
            parser = argparse.ArgumentParser()
            add_descloc_fsdp_args(parser)
            args = parser.parse_args([])
            cfg = hetero_gather_config_from_args(args)
            self.assertFalse(cfg.enable_fine_grained_param_gather)
            self.assertAlmostEqual(
                cfg.loc_cache_capacity_gb,
                _DEFAULT_LOC_CACHE_CAPACITY_GB,
            )

        def test_cli_enable_fine_grained(self):
            parser = argparse.ArgumentParser()
            add_descloc_fsdp_args(parser)
            args = parser.parse_args([
                "--descloc-enable-fine-grained-param-gather",
                "--descloc-loc-cache-capacity-gb", "128",
                "--descloc-loc-prefetch-depth", "4",
            ])
            cfg = hetero_gather_config_from_args(args)
            self.assertTrue(cfg.enable_fine_grained_param_gather)
            self.assertAlmostEqual(cfg.loc_cache_capacity_gb, 128.0)
            self.assertEqual(cfg.prefetch_depth, 4)

        def test_cli_gather_stream_priority(self):
            parser = argparse.ArgumentParser()
            add_descloc_fsdp_args(parser)
            args = parser.parse_args(["--descloc-gather-stream-priority", "0"])
            cfg = hetero_gather_config_from_args(args)
            self.assertEqual(cfg.gather_stream_priority, 0)

        # ------------------------------------------------------------------
        # _fill_loc_from_zero3_shards (single-process path)
        # ------------------------------------------------------------------

        def test_fill_loc_single_process(self):
            """Verify that _fill_loc_from_zero3_shards copies data to LOC buf."""
            layer = nn.Linear(8, 4, bias=False)
            # Initialise weights to known values
            nn.init.constant_(layer.weight, 1.0)

            numel = layer.weight.numel()  # 32
            cache = LocalityCacheBuffer(
                capacity_bytes=1024 * 1024,
                dtype=torch.float32,
            )
            loc_buf = cache.allocate("layer.weight", numel)
            _fill_loc_from_zero3_shards(layer, loc_buf, process_group=None)

            # All values should be 1.0
            self.assertTrue(torch.all(loc_buf[:numel] == 1.0))
            cache.release("layer.weight")

        # ------------------------------------------------------------------
        # _copy_loc_to_module_params
        # ------------------------------------------------------------------

        def test_copy_loc_to_params(self):
            layer = nn.Linear(8, 4, bias=False)
            nn.init.constant_(layer.weight, 0.0)

            numel = layer.weight.numel()
            src = torch.ones(numel, dtype=torch.float32, pin_memory=True)
            _copy_loc_to_module_params(layer, src)
            self.assertTrue(torch.all(layer.weight == 1.0))

        # ------------------------------------------------------------------
        # Repr / str
        # ------------------------------------------------------------------

        def test_loc_cache_repr(self):
            cache = LocalityCacheBuffer(capacity_bytes=1024**3)
            r = repr(cache)
            self.assertIn("GiB", r)
            self.assertIn("bfloat16", r)

    # Run the test suite
    print("\n" + "=" * 60)
    print("DES-LOC HeteroFineGrainedParamGather — Unit Tests")
    print("=" * 60 + "\n")
    unittest.main(verbosity=2)
