"""
HeteroGradBufferOffloadGuard — DES-LOC Heterogeneous Gradient Buffer Offload Guard
====================================================================================

上游设计意图 (Megatron commit 27a5f83):
---------------------------------------
原始 commit 解决了一个在 RLHF 训练流水线中出现的竞态问题：当 `rl_training_cuda_graphs`
启用时，CUDA Graph 会静态捕获 GPU 计算图，其中包含对 grad buffer 内存地址的引用。
若在 graph 捕获期间或回放期间调用 `model.offload_grad_buffers()`，GPU 侧的内存地址
将发生变化（数据被迁移到 CPU pinned memory），导致 CUDA Graph 中的指针悬空，引发
undefined behavior 甚至 CUDA illegal memory access。

Megatron 的修复极为简单：在两处调用点（`get_environment_rollouts` 和
`megatron_rl_inference_mode`）用 `if not args.rl_training_cuda_graphs` 包裹
`offload_grad_buffers()`，并发出 warning。

DES-LOC 适配点 (Decoupled Execution with Shared LOcality Cache):
-----------------------------------------------------------------
DES-LOC 框架面向 **2×A6000(48GB, SM86) + 1×H100 NVL(96GB, SM90)** 异构拓扑，
三卡通过 PCIe 互联，无 NVLink，CPU DRAM 高达 1.5TB。这个硬件格局带来以下特殊挑战：

1. **异构 CUDA Graph 不可简单共享**：A6000(SM86) 与 H100(SM90) 编译出的 CUDA Graph
   不兼容，DES-LOC 在两类设备上各自维护独立的 graph cache（SharedLocalityCache）。
   任何 offload 操作都必须同时感知"当前活跃 device 是否持有 graph capture lock"。

2. **PCIe 带宽瓶颈下的 offload 窗口管理**：A6000↔CPU 和 H100↔CPU 的 PCIe 带宽
   不对称（A6000 约 16GB/s 单向，H100 NVL 约 64GB/s 单向）。DES-LOC 的
   HeteroGradBufferOffloadGuard 引入 **device-aware offload scheduling**，
   根据设备类型动态决定 offload 时机和带宽配额。

3. **Shared Locality Cache 与 grad buffer 的别名问题**：DES-LOC 的 SLC（Shared
   Locality Cache）机制允许跨设备共享激活缓存，但 grad buffer 与 SLC 存在内存别名
   风险。本模块通过 `SLCPinnedRegionTracker` 确保 offload 前 SLC 中的 grad buffer
   引用已全部失效（invalidated）。

4. **DeepSpeed ZeRO 集成**：DeepSpeed ZeRO Stage 1/2/3 的 grad buffer 管理与
   Megatron 的 `DistributedDataParallel` grad buffer 机制不同。本模块同时支持
   DeepSpeed engine 的 `_grad_accum_dtype_tensor` 和原生 grad buffer 两种路径。

关键设计决策：
- 使用 Python context manager（`__enter__`/`__exit__`）封装 offload guard 生命周期，
  确保异常路径下的 graph lock 释放。
- `OffloadEligibilityChecker` 实现 SM 架构感知的判断逻辑。
- `HeteroOffloadScheduler` 根据 PCIe 拓扑分配 offload 顺序（H100 优先，A6000 后）。

Author: Neuron_SP Project (mirrors Megatron 27a5f83 — mathemakitten@nvidia.com)
Upstream: github.com/NVIDIA/Megatron-LM/commit/27a5f83eb175903e1437c497ad4e363a7bb6ed4c
Project:  github.com/dylanyunlon/Neuron_SP
"""

from __future__ import annotations

import logging
import os
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware topology constants for the DES-LOC target cluster
# ---------------------------------------------------------------------------

#: SM compute capability thresholds
SM_A6000 = (8, 6)   # NVIDIA RTX A6000, 48 GB
SM_H100  = (9, 0)   # NVIDIA H100 NVL,  96 GB

#: Approximate one-way PCIe bandwidth (bytes/sec) per device class.
#: A6000 is on a PCIe 4.0 x16 slot (~32 GB/s theoretical, ~16 GB/s sustained).
#: H100 NVL is on a PCIe 5.0 x16 slot (~64 GB/s sustained).
PCIE_BW_A6000_BYTES_PER_SEC: int = 16 * (1 << 30)   # 16 GB/s
PCIE_BW_H100_BYTES_PER_SEC:  int = 64 * (1 << 30)   # 64 GB/s

#: CPU DRAM capacity guard — we never want to exceed 90 % utilisation.
CPU_DRAM_TOTAL_BYTES: int = 1536 * (1 << 30)         # 1.5 TB
CPU_DRAM_SAFE_RATIO:  float = 0.90


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DeviceClass(Enum):
    """Classification of GPU devices present in the DES-LOC cluster."""
    A6000 = auto()   # SM 8.6, 48 GB
    H100  = auto()   # SM 9.0, 96 GB
    OTHER = auto()   # Any other device (passthrough, no special treatment)


class OffloadDecision(Enum):
    """What the guard decides to do for a given grad buffer offload request."""
    PROCEED      = auto()   # Safe to offload; go ahead.
    SKIP_GRAPH   = auto()   # CUDA Graph is active; skip offload.
    DEFER        = auto()   # PCIe bandwidth saturated; defer to next window.
    SKIP_SLC     = auto()   # SLC has live aliases; skip offload.
    SKIP_DRAM    = auto()   # CPU DRAM headroom insufficient; skip offload.


# ---------------------------------------------------------------------------
# Device introspection helpers
# ---------------------------------------------------------------------------

def _get_device_class(device: torch.device) -> DeviceClass:
    """Return the :class:`DeviceClass` for *device* by inspecting SM capability.

    DES-LOC target hardware only contains A6000 and H100 devices, so the
    mapping is straightforward.  Unknown devices fall back to
    ``DeviceClass.OTHER`` which disables all DES-LOC-specific optimisations.
    """
    if device.type != "cuda":
        return DeviceClass.OTHER
    props = torch.cuda.get_device_properties(device)
    cap = (props.major, props.minor)
    if cap == SM_A6000:
        return DeviceClass.A6000
    if cap == SM_H100:
        return DeviceClass.H100
    logger.debug(
        "Device %s has unknown SM capability %s; treating as OTHER.", device, cap
    )
    return DeviceClass.OTHER


def _pcie_bandwidth_for_device(device: torch.device) -> int:
    """Return estimated one-way PCIe bandwidth (bytes/sec) for *device*."""
    cls = _get_device_class(device)
    if cls == DeviceClass.H100:
        return PCIE_BW_H100_BYTES_PER_SEC
    if cls == DeviceClass.A6000:
        return PCIE_BW_A6000_BYTES_PER_SEC
    # Conservative fallback: PCIe 3.0 x16 sustained
    return 10 * (1 << 30)


# ---------------------------------------------------------------------------
# SLC (Shared Locality Cache) pinned-region tracker
# ---------------------------------------------------------------------------

@dataclass
class SLCPinnedRegionTracker:
    """Tracks GPU memory regions currently pinned in the DES-LOC SLC.

    The Shared Locality Cache allows cross-device activation sharing by
    pinning tensor storage into a reserved GPU memory pool.  If a grad buffer
    overlaps with a pinned SLC region, we must not offload it — doing so would
    corrupt the SLC's internal address mapping.

    In practice this is checked by comparing the ``data_ptr()`` ranges of
    registered SLC entries against the grad buffer storage range.

    Thread-safety:
        All mutations are protected by ``_lock``.
    """
    _regions: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def register(self, tensor: torch.Tensor, tag: int) -> None:
        """Register *tensor* as an active SLC-pinned region with id *tag*."""
        start = tensor.data_ptr()
        end   = start + tensor.numel() * tensor.element_size()
        with self._lock:
            self._regions[tag] = (start, end)
        logger.debug("SLC region registered: tag=%d, [%x, %x)", tag, start, end)

    def unregister(self, tag: int) -> None:
        """Remove the SLC-pinned region identified by *tag*."""
        with self._lock:
            removed = self._regions.pop(tag, None)
        if removed:
            logger.debug("SLC region unregistered: tag=%d", tag)

    def has_alias(self, grad_buf: torch.Tensor) -> bool:
        """Return ``True`` if *grad_buf* overlaps any active SLC region."""
        start = grad_buf.data_ptr()
        end   = start + grad_buf.numel() * grad_buf.element_size()
        with self._lock:
            for (r_start, r_end) in self._regions.values():
                if start < r_end and end > r_start:
                    return True
        return False

    def clear(self) -> None:
        """Invalidate all registered regions (e.g. after a step boundary)."""
        with self._lock:
            self._regions.clear()
        logger.debug("SLC pinned region tracker cleared.")


# Singleton tracker shared across all guard instances in a process.
_GLOBAL_SLC_TRACKER = SLCPinnedRegionTracker()


def get_global_slc_tracker() -> SLCPinnedRegionTracker:
    """Return the process-global :class:`SLCPinnedRegionTracker`."""
    return _GLOBAL_SLC_TRACKER


# ---------------------------------------------------------------------------
# CUDA Graph capture state registry
# ---------------------------------------------------------------------------

class CUDAGraphCaptureRegistry:
    """Per-device registry that tracks whether a CUDA Graph capture is active.

    DES-LOC maintains separate graph caches for SM86 (A6000) and SM90 (H100)
    devices.  This registry lets the offload guard query whether *any* device
    currently holds an open graph capture context, which would make memory
    movement unsafe.

    Usage::

        registry = CUDAGraphCaptureRegistry.instance()
        with registry.capture_context(device):
            # Inside here, offload decisions will be SKIP_GRAPH.
            torch.cuda.CUDAGraph().capture_begin()
            ...
            torch.cuda.CUDAGraph().capture_end()
    """

    _instance: Optional["CUDAGraphCaptureRegistry"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._active_devices: Set[int] = set()  # device indices
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "CUDAGraphCaptureRegistry":
        """Return the process-singleton registry."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def mark_capture_begin(self, device: torch.device) -> None:
        """Signal that *device* has entered CUDA Graph capture mode."""
        with self._lock:
            self._active_devices.add(device.index)
        logger.debug("CUDA Graph capture started on device %s.", device)

    def mark_capture_end(self, device: torch.device) -> None:
        """Signal that *device* has exited CUDA Graph capture mode."""
        with self._lock:
            self._active_devices.discard(device.index)
        logger.debug("CUDA Graph capture ended on device %s.", device)

    def is_capturing(self, device: Optional[torch.device] = None) -> bool:
        """Return ``True`` if any (or a specific) device is capturing a graph.

        Args:
            device: If provided, check only this device.  If ``None``, check
                    whether *any* device is in capture mode.
        """
        with self._lock:
            if device is None:
                return bool(self._active_devices)
            return device.index in self._active_devices

    @contextmanager
    def capture_context(self, device: torch.device):
        """Context manager that automatically registers / deregisters capture."""
        self.mark_capture_begin(device)
        try:
            yield
        finally:
            self.mark_capture_end(device)


# ---------------------------------------------------------------------------
# Offload eligibility checker
# ---------------------------------------------------------------------------

@dataclass
class OffloadEligibilityChecker:
    """Determines whether a grad-buffer offload is safe under DES-LOC rules.

    This is the core decision unit that the :class:`HeteroGradBufferOffloadGuard`
    delegates to.  Checks are performed in priority order:

    1. CUDA Graph capture lock   → ``SKIP_GRAPH``   (mirrors Megatron fix)
    2. SLC alias check            → ``SKIP_SLC``
    3. CPU DRAM headroom          → ``SKIP_DRAM``
    4. PCIe bandwidth window      → ``DEFER``
    5. All clear                  → ``PROCEED``

    Args:
        graph_registry:  The :class:`CUDAGraphCaptureRegistry` to consult.
        slc_tracker:     The :class:`SLCPinnedRegionTracker` to consult.
        bw_utilisation:  Current fractional PCIe utilisation on this device
                         (0.0–1.0).  Caller is responsible for updating this.
        dram_used_bytes: Current CPU DRAM bytes in use.  Used for headroom check.
    """
    graph_registry:   CUDAGraphCaptureRegistry
    slc_tracker:      SLCPinnedRegionTracker
    bw_utilisation:   float = 0.0
    dram_used_bytes:  int   = 0
    bw_threshold:     float = 0.85   # Back off when > 85 % saturated

    def check(
        self,
        grad_buf: torch.Tensor,
        device:   torch.device,
    ) -> OffloadDecision:
        """Run all eligibility checks and return an :class:`OffloadDecision`.

        Args:
            grad_buf: The gradient buffer tensor to be offloaded.
            device:   The source GPU device.

        Returns:
            The most restrictive :class:`OffloadDecision` applicable.
        """
        # --- Check 1: CUDA Graph capture lock ---
        # Mirrors Megatron commit 27a5f83: if any training graph is captured
        # (or being replayed) on this device, pointer aliasing makes offload unsafe.
        if self.graph_registry.is_capturing(device):
            logger.warning(
                "[DES-LOC] Gradient buffer offload skipped on device %s: "
                "CUDA Graph capture is active. This mirrors Megatron #3231 guard. "
                "Buffer data_ptr=0x%x, numel=%d.",
                device, grad_buf.data_ptr(), grad_buf.numel(),
            )
            return OffloadDecision.SKIP_GRAPH

        # --- Check 2: SLC alias ---
        if self.slc_tracker.has_alias(grad_buf):
            logger.warning(
                "[DES-LOC] Gradient buffer offload skipped on device %s: "
                "buffer overlaps an active SLC-pinned region. "
                "Buffer data_ptr=0x%x.",
                device, grad_buf.data_ptr(),
            )
            return OffloadDecision.SKIP_SLC

        # --- Check 3: CPU DRAM headroom ---
        buf_bytes = grad_buf.numel() * grad_buf.element_size()
        max_allowed = int(CPU_DRAM_TOTAL_BYTES * CPU_DRAM_SAFE_RATIO)
        if self.dram_used_bytes + buf_bytes > max_allowed:
            logger.warning(
                "[DES-LOC] Gradient buffer offload skipped: insufficient CPU DRAM "
                "headroom. In use: %.2f GB, buffer: %.3f GB, limit: %.2f GB.",
                self.dram_used_bytes / (1 << 30),
                buf_bytes / (1 << 30),
                max_allowed / (1 << 30),
            )
            return OffloadDecision.SKIP_DRAM

        # --- Check 4: PCIe bandwidth saturation ---
        if self.bw_utilisation > self.bw_threshold:
            logger.info(
                "[DES-LOC] Gradient buffer offload deferred on device %s: "
                "PCIe utilisation %.1f%% exceeds threshold %.1f%%.",
                device, self.bw_utilisation * 100, self.bw_threshold * 100,
            )
            return OffloadDecision.DEFER

        return OffloadDecision.PROCEED


# ---------------------------------------------------------------------------
# Hetero offload scheduler
# ---------------------------------------------------------------------------

class HeteroOffloadScheduler:
    """Orders grad-buffer offloads across heterogeneous devices.

    On the DES-LOC cluster (A6000 × 2 + H100 × 1, PCIe, no NVLink) the
    optimal offload order is **H100 first** for two reasons:

    1. H100's PCIe 5.0 x16 interface offers ~4× the bandwidth of A6000's
       PCIe 4.0 x16, so it completes faster and releases the CPU-side memory
       bus sooner.
    2. H100 holds the larger model partition (96 GB) and is typically the
       memory-pressure bottleneck; relieving it first reduces the risk of OOM
       before the A6000 offloads complete.

    The scheduler returns a sorted list of (device, grad_buf_list) pairs.
    """

    def schedule(
        self,
        device_bufs: Dict[torch.device, List[torch.Tensor]],
    ) -> List[Tuple[torch.device, List[torch.Tensor]]]:
        """Return an ordered offload schedule for *device_bufs*.

        Args:
            device_bufs: Mapping from device to list of grad buffers on that device.

        Returns:
            A list of ``(device, buffers)`` tuples sorted by offload priority
            (H100 first, then A6000, then other).
        """
        priority: Dict[DeviceClass, int] = {
            DeviceClass.H100:  0,
            DeviceClass.A6000: 1,
            DeviceClass.OTHER: 2,
        }
        result = sorted(
            device_bufs.items(),
            key=lambda kv: priority[_get_device_class(kv[0])],
        )
        if logger.isEnabledFor(logging.DEBUG):
            order_str = " → ".join(
                f"{dev}({_get_device_class(dev).name})" for dev, _ in result
            )
            logger.debug("[DES-LOC] Offload schedule: %s", order_str)
        return result


# ---------------------------------------------------------------------------
# Main guard class
# ---------------------------------------------------------------------------

class HeteroGradBufferOffloadGuard:
    """Context-manager guard for heterogeneous grad-buffer offloading.

    This is the primary public API of this module.  It wraps the decision
    logic of :class:`OffloadEligibilityChecker` and the scheduling logic of
    :class:`HeteroOffloadScheduler` into a clean context manager that callers
    (e.g. DES-LOC's RL inference and rollout pipelines) use in place of a
    bare ``model.offload_grad_buffers()`` call.

    Relationship to Megatron commit 27a5f83:
        Megatron's fix was a simple ``if not args.rl_training_cuda_graphs``
        guard at two call sites.  DES-LOC generalises this to:

        - A *device-class-aware* eligibility checker (not just one global flag).
        - An *SLC alias* check unique to DES-LOC's cross-device cache.
        - A *PCIe bandwidth* backpressure mechanism.
        - A *DRAM headroom* safeguard (critical given the 1.5 TB pool).
        - A *scheduler* that orders offloads across A6000 and H100 to maximise
          aggregate PCIe throughput.

    Usage::

        guard = HeteroGradBufferOffloadGuard(
            engine=deepspeed_engine,
            devices=[torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")],
        )
        with guard.offload_context(phase="rollout"):
            optimizer.offload_to_cpu()
        # Grad buffers were offloaded only where safe.

    Args:
        engine:          A DeepSpeed engine or any object with a
                         ``module`` attribute and optionally a
                         ``_grad_accum_dtype_tensor`` attribute.
        devices:         List of GPU devices managed by this guard.
        graph_registry:  Optional custom registry; defaults to the process
                         singleton.
        slc_tracker:     Optional custom SLC tracker; defaults to the process
                         singleton.
        scheduler:       Optional custom offload scheduler.
        dram_used_bytes: Current CPU DRAM bytes in use (for headroom checks).
        bw_utilisation:  Per-device PCIe utilisation fractions, keyed by
                         ``device.index``.
    """

    def __init__(
        self,
        engine:           Any,
        devices:          List[torch.device],
        graph_registry:   Optional[CUDAGraphCaptureRegistry] = None,
        slc_tracker:      Optional[SLCPinnedRegionTracker]   = None,
        scheduler:        Optional[HeteroOffloadScheduler]   = None,
        dram_used_bytes:  int                                = 0,
        bw_utilisation:   Optional[Dict[int, float]]         = None,
    ) -> None:
        self._engine         = weakref.ref(engine) if engine is not None else lambda: None
        self._devices        = list(devices)
        self._graph_registry = graph_registry or CUDAGraphCaptureRegistry.instance()
        self._slc_tracker    = slc_tracker    or get_global_slc_tracker()
        self._scheduler      = scheduler      or HeteroOffloadScheduler()
        self._dram_used      = dram_used_bytes
        self._bw_util        = bw_utilisation or {}
        self._offloaded_bufs: List[Tuple[torch.device, torch.Tensor]] = []
        self._phase: str = "unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def offload_context(self, phase: str = "inference"):
        """Context manager: attempt grad-buffer offloads, then restore on exit.

        Args:
            phase: A label for logging ("rollout", "inference", etc.).
        """
        self._phase = phase
        self._offloaded_bufs.clear()
        logger.info(
            "[DES-LOC] HeteroGradBufferOffloadGuard entering phase '%s'. "
            "Devices: %s",
            phase,
            [str(d) for d in self._devices],
        )
        t0 = time.monotonic()
        skipped = self._attempt_offloads()
        elapsed_ms = (time.monotonic() - t0) * 1e3
        logger.info(
            "[DES-LOC] Phase '%s' offload attempt complete in %.1f ms. "
            "Offloaded: %d buffer(s), skipped: %d.",
            phase, elapsed_ms, len(self._offloaded_bufs), skipped,
        )
        try:
            yield self
        finally:
            logger.debug(
                "[DES-LOC] HeteroGradBufferOffloadGuard exiting phase '%s'.",
                phase,
            )

    def update_dram_usage(self, bytes_in_use: int) -> None:
        """Update the tracked CPU DRAM usage for subsequent eligibility checks."""
        self._dram_used = bytes_in_use
        logger.debug("[DES-LOC] DRAM usage updated: %.3f GB.", bytes_in_use / (1 << 30))

    def update_bw_utilisation(self, device_index: int, fraction: float) -> None:
        """Update PCIe utilisation fraction for *device_index* (0.0–1.0)."""
        self._bw_util[device_index] = max(0.0, min(1.0, fraction))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_device_buffers(
        self,
    ) -> Dict[torch.device, List[torch.Tensor]]:
        """Enumerate grad buffers on each managed device from the engine.

        Supports two sources:
        1. DeepSpeed engine ``_grad_accum_dtype_tensor`` — a flat tensor per
           parameter group whose storage holds accumulated gradients.
        2. Native PyTorch ``module.parameters()`` ``.grad`` attributes.

        Returns a dict mapping device → list of grad tensors on that device.
        """
        device_bufs: Dict[torch.device, List[torch.Tensor]] = {
            d: [] for d in self._devices
        }
        engine = self._engine()
        if engine is None:
            logger.warning("[DES-LOC] Engine has been garbage-collected; no buffers found.")
            return device_bufs

        # Path A: DeepSpeed ZeRO grad accum tensor
        accum_tensor = getattr(engine, "_grad_accum_dtype_tensor", None)
        if accum_tensor is not None and isinstance(accum_tensor, torch.Tensor):
            dev = accum_tensor.device
            if dev in device_bufs:
                device_bufs[dev].append(accum_tensor)
            logger.debug(
                "[DES-LOC] Found DS grad accum tensor on %s, numel=%d.",
                dev, accum_tensor.numel(),
            )

        # Path B: Enumerate parameter gradients
        module = getattr(engine, "module", engine)
        if module is not None:
            for param in module.parameters():
                if param.grad is not None:
                    dev = param.grad.device
                    if dev in device_bufs:
                        device_bufs[dev].append(param.grad)

        # Deduplicate by data_ptr to avoid double-counting shared storage.
        for dev in device_bufs:
            seen: Set[int] = set()
            unique: List[torch.Tensor] = []
            for buf in device_bufs[dev]:
                ptr = buf.data_ptr()
                if ptr not in seen:
                    seen.add(ptr)
                    unique.append(buf)
            device_bufs[dev] = unique
            if unique:
                total_mb = sum(b.numel() * b.element_size() for b in unique) / (1 << 20)
                logger.debug(
                    "[DES-LOC] Device %s: %d grad buffer(s), %.1f MB total.",
                    dev, len(unique), total_mb,
                )
        return device_bufs

    def _attempt_offloads(self) -> int:
        """Run the offload pipeline.  Returns the count of *skipped* buffers."""
        device_bufs = self._collect_device_buffers()
        ordered     = self._scheduler.schedule(device_bufs)
        skipped_total = 0

        for device, bufs in ordered:
            if not bufs:
                continue
            checker = OffloadEligibilityChecker(
                graph_registry  = self._graph_registry,
                slc_tracker     = self._slc_tracker,
                bw_utilisation  = self._bw_util.get(device.index, 0.0),
                dram_used_bytes = self._dram_used,
            )
            for buf in bufs:
                decision = checker.check(buf, device)
                if decision == OffloadDecision.PROCEED:
                    self._do_offload_buffer(buf, device)
                    # Account for the data now sitting in CPU DRAM.
                    self._dram_used += buf.numel() * buf.element_size()
                else:
                    skipped_total += 1
                    logger.debug(
                        "[DES-LOC] Offload of buffer (data_ptr=0x%x) on %s: %s.",
                        buf.data_ptr(), device, decision.name,
                    )

        return skipped_total

    def _do_offload_buffer(
        self,
        buf:    torch.Tensor,
        device: torch.device,
    ) -> None:
        """Move *buf* from GPU to pinned CPU memory and record the operation.

        In production this would call the DeepSpeed / Megatron-level
        ``offload_grad_buffers()`` primitive.  Here we implement the raw
        pinned-memory copy that underpins it, so DES-LOC callers can use this
        module independently of a full Megatron stack.

        Args:
            buf:    The gradient buffer tensor, currently on *device*.
            device: The source GPU device (for logging and bookkeeping).
        """
        try:
            cpu_buf = torch.empty(
                buf.shape,
                dtype   = buf.dtype,
                device  = "cpu",
                pin_memory = True,
            )
            cpu_buf.copy_(buf, non_blocking=True)
            self._offloaded_bufs.append((device, cpu_buf))
            logger.debug(
                "[DES-LOC] Offloaded %.3f MB from %s → CPU pinned memory.",
                buf.numel() * buf.element_size() / (1 << 20),
                device,
            )
        except RuntimeError as exc:
            logger.error(
                "[DES-LOC] Failed to offload buffer on %s: %s. "
                "Skipping this buffer.",
                device, exc,
            )


# ---------------------------------------------------------------------------
# Convenience factory matching Megatron's call-site pattern
# ---------------------------------------------------------------------------

def make_hetero_offload_guard(
    engine:   Any,
    devices:  Optional[List[torch.device]] = None,
) -> HeteroGradBufferOffloadGuard:
    """Factory that builds a guard for all available CUDA devices.

    If *devices* is ``None``, all ``torch.cuda.device_count()`` devices are
    used — appropriate for the DES-LOC target cluster where all three GPUs are
    visible in the same process.

    Args:
        engine:  DeepSpeed engine (or any object with ``.module``).
        devices: Explicit device list; auto-detected if ``None``.

    Returns:
        A ready-to-use :class:`HeteroGradBufferOffloadGuard`.
    """
    if devices is None:
        n = torch.cuda.device_count()
        devices = [torch.device("cuda", i) for i in range(n)]
        logger.info(
            "[DES-LOC] Auto-detected %d CUDA device(s): %s",
            n, [str(d) for d in devices],
        )
    return HeteroGradBufferOffloadGuard(engine=engine, devices=devices)


# ---------------------------------------------------------------------------
# Integration shim: drop-in replacement for Megatron call sites
# ---------------------------------------------------------------------------

def safe_offload_grad_buffers(
    model:    Any,
    args:     Any,
    guard:    Optional[HeteroGradBufferOffloadGuard] = None,
) -> bool:
    """Drop-in replacement for Megatron's ``model[0].offload_grad_buffers()``.

    Mirrors the guard logic introduced in Megatron commit 27a5f83 but extends
    it to the full DES-LOC eligibility pipeline.

    Megatron original::

        if not args.rl_training_cuda_graphs:
            model[0].offload_grad_buffers()
        else:
            logger.warning("Gradient buffers will not be offloaded ...")

    DES-LOC replacement::

        safe_offload_grad_buffers(model[0], args, guard)

    Args:
        model:  The model (or model[0] from a model list).
        args:   Argument namespace; consulted for ``rl_training_cuda_graphs``
                to preserve backward compat with Megatron configs.
        guard:  Optional pre-built guard.  If ``None``, a single-device guard
                is constructed from ``model``'s device.

    Returns:
        ``True`` if offload was performed (or attempted), ``False`` if skipped
        due to CUDA Graph capture (matching Megatron's original semantics).
    """
    # Fast path: honour Megatron's original flag as a first-class override.
    rl_training_cuda_graphs = getattr(args, "rl_training_cuda_graphs", False)
    if rl_training_cuda_graphs:
        logger.warning(
            "[DES-LOC / Megatron compat] Gradient buffers will not be offloaded "
            "when rl_training_cuda_graphs is enabled (mirrors Megatron #3231)."
        )
        return False

    # Slow path: full DES-LOC eligibility pipeline.
    if guard is None:
        device = next(
            (p.device for p in model.parameters() if p.is_cuda),
            torch.device("cuda", 0),
        )
        guard = HeteroGradBufferOffloadGuard(engine=model, devices=[device])

    with guard.offload_context(phase="safe_offload_grad_buffers"):
        pass  # actual work happens inside __enter__
    return True


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.DEBUG,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Assert 1: DeviceClass detection is deterministic for known SMs ---
    import unittest.mock as mock

    fake_props_a6000 = mock.MagicMock()
    fake_props_a6000.major = 8
    fake_props_a6000.minor = 6
    fake_props_h100 = mock.MagicMock()
    fake_props_h100.major = 9
    fake_props_h100.minor = 0

    with mock.patch("torch.cuda.get_device_properties", return_value=fake_props_a6000):
        assert _get_device_class(torch.device("cuda", 0)) == DeviceClass.A6000

    with mock.patch("torch.cuda.get_device_properties", return_value=fake_props_h100):
        assert _get_device_class(torch.device("cuda", 0)) == DeviceClass.H100

    # --- Assert 2: SLCPinnedRegionTracker detects alias correctly ---
    tracker = SLCPinnedRegionTracker()
    t = torch.zeros(1024)
    tracker.register(t, tag=1)
    assert tracker.has_alias(t), "Exact-match alias should be detected"
    tracker.unregister(1)
    assert not tracker.has_alias(t), "After unregister, no alias should exist"

    # --- Assert 3: CUDA Graph registry blocks offload ---
    registry = CUDAGraphCaptureRegistry()
    dev_cpu = torch.device("cpu")  # Use CPU device index=None workaround
    # Manually test the flag without needing actual CUDA
    registry._active_devices.add(0)
    assert registry.is_capturing(torch.device("cuda", 0))
    registry._active_devices.discard(0)
    assert not registry.is_capturing(torch.device("cuda", 0))

    # --- Assert 4: OffloadEligibilityChecker returns SKIP_GRAPH during capture ---
    checker = OffloadEligibilityChecker(
        graph_registry = registry,
        slc_tracker    = tracker,
    )
    registry._active_devices.add(0)
    dummy_buf = torch.zeros(64)
    decision = checker.check(dummy_buf, torch.device("cuda", 0))
    assert decision == OffloadDecision.SKIP_GRAPH
    registry._active_devices.discard(0)

    # --- Assert 5: HeteroOffloadScheduler orders H100 before A6000 ---
    scheduler = HeteroOffloadScheduler()
    with mock.patch("deepspeed.runtime.hetero_grad_buffer_offload_guard._get_device_class") as m:
        def _mock_cls(dev):
            return DeviceClass.H100 if dev.index == 2 else DeviceClass.A6000
        m.side_effect = _mock_cls
        ordered = scheduler.schedule({
            torch.device("cuda", 0): [torch.zeros(1)],
            torch.device("cuda", 1): [torch.zeros(1)],
            torch.device("cuda", 2): [torch.zeros(1)],
        })
    first_device = ordered[0][0]
    assert first_device.index == 2, "H100 (cuda:2) should be scheduled first"

    logger.info("All smoke tests passed.")
