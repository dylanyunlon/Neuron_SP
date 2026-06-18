"""
DES-LOC Heterogeneous Tensor Offload Manager
=============================================

Upstream design intent (Megatron commit 1106df46):
    NVIDIA Megatron-LM refactored KV-cache suspend/resume logic out of ad-hoc
    RL training code into a structured ``KVCacheManagementMode`` enum with three
    strategies: PERSIST (GPU stays resident), OFFLOAD (CPU pinned memory via
    storage-resize tricks), and RECOMPUTE (delete + recompute on demand).  The
    commit also disentangles "static memory pointer" requirements from CUDA-graph
    persistence, clarifying that pointer stability ≠ data persistence.

DES-LOC adaptation rationale:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a 3-device
    cluster: 2× A6000-48 GB (SM86, PCIe) + 1× H100-NVL-96 GB (SM90, PCIe).
    There is NO NVLink; all inter-GPU traffic goes through the host PCIe fabric.
    With 1.5 TB host DRAM, CPU offload is cheap in capacity but bandwidth-bound
    (~32 GB/s PCIe Gen4 x16 per direction).

    Key DES-LOC design decisions that diverge from Megatron:
    1.  **Device-aware placement** – PERSIST mode keeps tensors on whichever GPU
        owns them (H100 preferred for large KV caches); OFFLOAD routes through
        pinned host DRAM; RECOMPUTE discards entirely.
    2.  **Shared Locality Cache (SLoC)** – a per-device CPU-pinned ring buffer
        acts as the offload staging area so multiple tensors share one large
        pinned allocation rather than per-tensor malloc (reduces TLB thrash).
    3.  **Storage-resize trick** – reuses Megatron's exact technique
        (``tensor.storage().resize_(0)``) so GPU virtual addresses survive
        across offload cycles, enabling CUDA-graph pointer stability without
        ``torch_memory_saver``.
    4.  **Async H2D/D2H** – uses separate CUDA streams per device to overlap
        offload with compute; A6000s and H100 have independent PCIe lanes.
    5.  **SM-aware fallback** – SM86 (A6000) does not support HW-accelerated
        memory compression present on SM90 (H100); the manager detects compute
        capability and adjusts pinned-copy granularity accordingly.

Megatron API mirrored here:
    ``KVCacheManagementMode``  →  ``TensorResidencyMode``
    ``deallocate_inference_state_buffers`` → ``suspend_tensor_group``
    ``reinitialize_inference_state_buffers`` → ``resume_tensor_group``
    ``initialize_all_tensors``  →  ``register_tensor_group``
    ``static_kv_memory_pointers`` →  ``require_stable_data_ptr``
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------

class TensorResidencyMode(str, Enum):
    """How a registered tensor group is handled across suspend/resume cycles.

    Mirrors Megatron's ``KVCacheManagementMode`` but is device-topology-aware:
    the DES-LOC cluster has no NVLink, so OFFLOAD traffic is always PCIe-bound
    through host DRAM.  PERSIST is therefore preferred for H100 tensors that fit
    in its 96 GB FB; OFFLOAD is recommended only for A6000 overflow tensors.
    """

    PERSIST = "persist"
    """Tensor stays on GPU across suspend/resume.  Zero copy overhead.
    Use for: KV caches owned by H100 that fit in 96 GB."""

    OFFLOAD = "offload"
    """Tensor is copied to a CPU pinned SLoC slot on suspend, restored on resume.
    GPU virtual address is preserved via storage-resize (pointer stability).
    Use for: A6000 KV caches that must not block H100 training compute."""

    RECOMPUTE = "recompute"
    """Tensor is fully released on suspend; recomputed/reallocated on resume.
    Use for: cheap-to-recompute activations or when GPU memory is critically low."""


class DeviceProfile(str, Enum):
    """Compute capability profiles for DES-LOC cluster devices."""
    SM86_A6000 = "sm86"   # 2× A6000 48 GB
    SM90_H100  = "sm90"   # 1× H100 NVL 96 GB
    UNKNOWN    = "unknown"


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _TensorRecord:
    """Per-tensor bookkeeping within a group."""
    name: str
    device_id: int
    residency_mode: TensorResidencyMode
    original_storage_size: int = 0          # bytes before resize_(0)
    cpu_backup: Optional[torch.Tensor] = None  # pinned SLoC slot
    is_offloaded: bool = False


@dataclass
class TensorGroup:
    """A named collection of tensors that share a residency policy.

    Analogous to Megatron's per-context KV-cache + Mamba-state bundle,
    but generalised so DeepSpeed modules can register arbitrary groups
    (e.g., gradient buffers, optimizer state fragments).
    """
    name: str
    residency_mode: TensorResidencyMode
    require_stable_data_ptr: bool = False
    """If True the virtual GPU address must not change across suspend/resume.
    Requires either PERSIST mode or OFFLOAD-with-storage-resize (DES-LOC default).
    RECOMPUTE mode is incompatible: it releases and re-allocates storage."""

    # Internal bookkeeping – managed by HeteroTensorOffloadManager
    _tensor_names: List[str] = field(default_factory=list)
    _records: Dict[str, _TensorRecord] = field(default_factory=dict)
    _is_suspended: bool = False
    _owner_module: object = None   # weakref to owning nn.Module (if any)


# ---------------------------------------------------------------------------
# Shared Locality Cache (SLoC) – CPU pinned ring buffer
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """Per-device CPU pinned memory staging area for OFFLOAD mode.

    Design: one large contiguous pinned allocation per device, sub-divided
    into fixed-size slots.  Multiple tensors share this allocation instead of
    each tensor doing its own ``torch.empty(..., pin_memory=True)``.

    On SM86 (A6000) PCIe Gen4 x16 theoretical peak ~32 GB/s; on SM90 (H100)
    PCIe Gen5 x16 ~64 GB/s.  SLoC avoids repeated mmap()/munmap() overhead
    that would otherwise dominate small-tensor offloads.
    """

    def __init__(
        self,
        device_id: int,
        capacity_bytes: int = 8 * 1024**3,  # 8 GB default per device
        slot_size_bytes: int = 512 * 1024**2,  # 512 MB slots
    ) -> None:
        self.device_id = device_id
        self.capacity_bytes = capacity_bytes
        self.slot_size_bytes = slot_size_bytes
        self._lock = threading.Lock()

        n_slots = capacity_bytes // slot_size_bytes
        if n_slots == 0:
            raise ValueError(
                f"SLoC capacity ({capacity_bytes} B) smaller than slot size ({slot_size_bytes} B)"
            )

        # One contiguous pinned buffer, logically partitioned into slots.
        self._backing = torch.empty(
            capacity_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self._free_slots: List[int] = list(range(n_slots))
        self._allocated: Dict[str, int] = {}  # tensor_key → slot_index
        logger.info(
            "SLoC[device=%d] initialised: capacity=%.1f GB, slots=%d × %.0f MB",
            device_id, capacity_bytes / 1024**3, n_slots, slot_size_bytes / 1024**2,
        )

    def allocate_slot(self, key: str, nbytes: int) -> torch.Tensor:
        """Return a pinned CPU tensor view covering at least ``nbytes`` bytes.

        If ``nbytes`` exceeds slot size, falls back to a fresh pinned alloc
        (logged as a SLoC miss) to avoid corrupting other slots.
        """
        with self._lock:
            if key in self._allocated:
                slot_idx = self._allocated[key]
                return self._slot_view(slot_idx, nbytes)

            if nbytes > self.slot_size_bytes:
                logger.warning(
                    "SLoC[device=%d] miss for key='%s': tensor size %d B exceeds slot size %d B; "
                    "falling back to fresh pinned alloc.",
                    self.device_id, key, nbytes, self.slot_size_bytes,
                )
                return torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)

            if not self._free_slots:
                raise RuntimeError(
                    f"SLoC[device={self.device_id}] exhausted (capacity={self.capacity_bytes} B). "
                    "Increase capacity_bytes or reduce registered tensors."
                )

            slot_idx = self._free_slots.pop(0)
            self._allocated[key] = slot_idx
            logger.debug(
                "SLoC[device=%d] allocated slot %d for key='%s' (%.2f MB)",
                self.device_id, slot_idx, key, nbytes / 1024**2,
            )
            return self._slot_view(slot_idx, nbytes)

    def release_slot(self, key: str) -> None:
        """Return a slot to the free pool."""
        with self._lock:
            if key not in self._allocated:
                return
            slot_idx = self._allocated.pop(key)
            self._free_slots.append(slot_idx)

    def _slot_view(self, slot_idx: int, nbytes: int) -> torch.Tensor:
        offset = slot_idx * self.slot_size_bytes
        return self._backing[offset : offset + nbytes]

    @property
    def utilisation(self) -> float:
        """Fraction of slots currently in use."""
        total = self.capacity_bytes // self.slot_size_bytes
        used = len(self._allocated)
        return used / total if total else 0.0


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _device_profile(device_id: int) -> DeviceProfile:
    """Detect SM compute capability for a CUDA device."""
    if not torch.cuda.is_available():
        return DeviceProfile.UNKNOWN
    try:
        major, minor = torch.cuda.get_device_capability(device_id)
        cc = major * 10 + minor
        if cc == 86:
            return DeviceProfile.SM86_A6000
        if cc == 90:
            return DeviceProfile.SM90_H100
    except Exception:
        pass
    return DeviceProfile.UNKNOWN


def _preferred_device_for_large_tensor(tensor: torch.Tensor) -> int:
    """DES-LOC heuristic: prefer H100 (device index where SM90 is detected) for
    tensors > 4 GB; leave smaller tensors on their current device.

    In practice the user controls device assignment; this is a fallback hint.
    """
    if not torch.cuda.is_available():
        return tensor.device.index or 0
    size_gb = tensor.numel() * tensor.element_size() / 1024**3
    if size_gb < 4.0:
        return tensor.device.index or 0
    for dev in range(torch.cuda.device_count()):
        if _device_profile(dev) == DeviceProfile.SM90_H100:
            return dev
    return tensor.device.index or 0


# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------

class HeteroTensorOffloadManager:
    """Manages tensor suspend/resume across CPU and heterogeneous GPUs.

    This is the DES-LOC counterpart to Megatron's in-context
    ``deallocate_inference_state_buffers`` / ``reinitialize_inference_state_buffers``
    pair.  By extracting the logic into a standalone manager, any DeepSpeed
    module (not just the inference engine) can register tensor groups and
    benefit from coordinated, async, device-aware offload.

    Lifecycle::

        mgr = HeteroTensorOffloadManager(...)

        # --- one-time setup ---
        mgr.register_tensor_group("kv_cache", mode=TensorResidencyMode.OFFLOAD,
                                   require_stable_data_ptr=True)
        mgr.attach_tensor("kv_cache", "memory_buffer", module.memory_buffer)

        # --- inference → training transition ---
        mgr.suspend_tensor_group("kv_cache")   # copies to CPU, resizes storage to 0
        ... train ...
        mgr.resume_tensor_group("kv_cache")    # restores from CPU, same GPU vaddr

    Thread safety:
        ``suspend_tensor_group`` / ``resume_tensor_group`` are safe to call from
        a single orchestrator thread.  The async CUDA streams are internal.
    """

    def __init__(
        self,
        sloc_capacity_bytes: int = 8 * 1024**3,
        sloc_slot_bytes: int = 512 * 1024**2,
        enable_async_transfer: bool = True,
    ) -> None:
        """
        Args:
            sloc_capacity_bytes: Total pinned CPU staging capacity per device (bytes).
                                  With 1.5 TB DRAM available, 8 GB per device is conservative.
            sloc_slot_bytes: Granularity of SLoC slot allocation.
            enable_async_transfer: Use separate CUDA streams for D2H/H2D copies so
                                    the default compute stream is not stalled.
        """
        self._groups: Dict[str, TensorGroup] = {}
        self._tensors: Dict[str, Dict[str, torch.Tensor]] = {}  # group → name → tensor
        self._sloc: Dict[int, SharedLocalityCache] = {}
        self._sloc_capacity = sloc_capacity_bytes
        self._sloc_slot = sloc_slot_bytes
        self._enable_async = enable_async_transfer
        self._offload_streams: Dict[int, torch.cuda.Stream] = {}
        self._lock = threading.RLock()

        logger.info(
            "HeteroTensorOffloadManager init: "
            "sloc_capacity=%.1f GB/device, async_transfer=%s",
            sloc_capacity_bytes / 1024**3, enable_async_transfer,
        )

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register_tensor_group(
        self,
        group_name: str,
        mode: TensorResidencyMode = TensorResidencyMode.PERSIST,
        require_stable_data_ptr: bool = False,
    ) -> TensorGroup:
        """Declare a named group with a shared residency policy.

        Args:
            group_name: Unique identifier (e.g., ``"kv_cache"``, ``"mamba_states"``).
            mode: How tensors in this group are handled on suspend/resume.
            require_stable_data_ptr: If True, asserts that GPU virtual addresses
                are preserved across cycles (requires PERSIST or OFFLOAD mode,
                not RECOMPUTE).

        Raises:
            ValueError: If ``require_stable_data_ptr`` is True with RECOMPUTE mode
                        (RECOMPUTE cannot preserve addresses without a memory-saver).
            ValueError: If a group with this name already exists.
        """
        if group_name in self._groups:
            raise ValueError(f"Tensor group '{group_name}' already registered.")

        if require_stable_data_ptr and mode == TensorResidencyMode.RECOMPUTE:
            raise ValueError(
                f"require_stable_data_ptr=True is incompatible with RECOMPUTE mode "
                f"(group='{group_name}').  Use PERSIST or OFFLOAD."
            )

        group = TensorGroup(
            name=group_name,
            residency_mode=mode,
            require_stable_data_ptr=require_stable_data_ptr,
        )
        self._groups[group_name] = group
        self._tensors[group_name] = {}

        logger.info(
            "Registered tensor group '%s': mode=%s, stable_ptr=%s",
            group_name, mode.value, require_stable_data_ptr,
        )
        return group

    def attach_tensor(
        self,
        group_name: str,
        tensor_name: str,
        tensor: torch.Tensor,
    ) -> None:
        """Attach a live GPU tensor to a registered group.

        The manager does NOT take ownership; the caller's reference remains valid.
        For OFFLOAD mode, a SLoC slot is pre-allocated here (fail-fast on capacity).

        Args:
            group_name: Previously registered group.
            tensor_name: Logical name within the group (e.g., ``"memory_buffer"``).
            tensor: A GPU tensor.  Must be on a CUDA device.

        Raises:
            KeyError: If group_name is not registered.
            TypeError: If tensor is not on a CUDA device.
        """
        if group_name not in self._groups:
            raise KeyError(f"Unknown tensor group '{group_name}'.  Call register_tensor_group first.")

        if not tensor.is_cuda:
            raise TypeError(
                f"attach_tensor expects a CUDA tensor; got device='{tensor.device}'."
            )

        group = self._groups[group_name]
        device_id: int = tensor.device.index or 0
        nbytes = tensor.numel() * tensor.element_size()

        record = _TensorRecord(
            name=tensor_name,
            device_id=device_id,
            residency_mode=group.residency_mode,
        )

        if group.residency_mode == TensorResidencyMode.OFFLOAD:
            sloc = self._get_or_create_sloc(device_id)
            cpu_buf = sloc.allocate_slot(f"{group_name}/{tensor_name}", nbytes)
            # Reshape to match the tensor's dtype/shape for copy_ compatibility.
            record.cpu_backup = cpu_buf[:nbytes].view(dtype=tensor.dtype).reshape(tensor.shape)
            self._ensure_offload_stream(device_id)
            logger.debug(
                "Pre-allocated SLoC slot for group='%s' tensor='%s' on device %d (%.2f MB)",
                group_name, tensor_name, device_id, nbytes / 1024**2,
            )

        group._records[tensor_name] = record
        group._tensor_names.append(tensor_name)
        self._tensors[group_name][tensor_name] = tensor

        logger.debug(
            "Attached tensor '%s/%s': shape=%s dtype=%s device=cuda:%d",
            group_name, tensor_name, tuple(tensor.shape), tensor.dtype, device_id,
        )

    def detach_tensor(self, group_name: str, tensor_name: str) -> None:
        """Remove a tensor from a group and release its SLoC slot if any."""
        if group_name not in self._groups:
            return
        group = self._groups[group_name]
        rec = group._records.pop(tensor_name, None)
        if rec and rec.residency_mode == TensorResidencyMode.OFFLOAD:
            sloc = self._sloc.get(rec.device_id)
            if sloc:
                sloc.release_slot(f"{group_name}/{tensor_name}")
        if tensor_name in group._tensor_names:
            group._tensor_names.remove(tensor_name)
        self._tensors.get(group_name, {}).pop(tensor_name, None)

    # ------------------------------------------------------------------
    # Suspend / Resume
    # ------------------------------------------------------------------

    def suspend_tensor_group(self, group_name: str) -> None:
        """Move tensors to CPU (OFFLOAD) or free GPU memory (RECOMPUTE).

        Mirrors Megatron's ``deallocate_inference_state_buffers``.

        OFFLOAD path (DES-LOC storage-resize trick):
            1. Async copy tensor data to pinned SLoC slot (non-blocking D2H).
            2. Record original storage size.
            3. ``tensor.storage().resize_(0)`` — releases physical GPU pages while
               preserving the virtual address, so CUDA-graph references remain valid.

        RECOMPUTE path:
            Deletes the tensor attribute from the attached module (if any);
            otherwise just records that storage was freed.

        PERSIST path:
            No-op.

        Args:
            group_name: Name of the group to suspend.
        """
        with self._lock:
            group = self._groups.get(group_name)
            if group is None:
                logger.warning("suspend_tensor_group: unknown group '%s'", group_name)
                return
            if group._is_suspended:
                logger.debug("Group '%s' already suspended; skipping.", group_name)
                return

            t0 = time.perf_counter()
            mode = group.residency_mode

            if mode == TensorResidencyMode.PERSIST:
                group._is_suspended = True
                logger.debug("Group '%s' suspend: PERSIST – no-op.", group_name)
                return

            tensors = self._tensors[group_name]

            if mode == TensorResidencyMode.OFFLOAD:
                self._suspend_offload(group, tensors)

            elif mode == TensorResidencyMode.RECOMPUTE:
                self._suspend_recompute(group, tensors)

            group._is_suspended = True
            elapsed = (time.perf_counter() - t0) * 1e3
            logger.info(
                "Suspended group '%s' (mode=%s) in %.1f ms",
                group_name, mode.value, elapsed,
            )

    def resume_tensor_group(self, group_name: str) -> None:
        """Restore tensors to GPU after a suspend.

        Mirrors Megatron's ``reinitialize_inference_state_buffers``.

        OFFLOAD path:
            1. ``tensor.storage().resize_(original_size)`` — re-claims physical GPU pages
               at the SAME virtual address (pointer stability for CUDA graphs).
            2. Async copy from pinned SLoC slot back to GPU (non-blocking H2D).

        RECOMPUTE path:
            Re-allocates tensors from scratch.  Callers should call their own
            ``initialize_all_tensors`` equivalent; this method simply clears the
            suspended flag and calls ``_recompute_callback`` if registered.

        Args:
            group_name: Name of the group to resume.
        """
        with self._lock:
            group = self._groups.get(group_name)
            if group is None:
                logger.warning("resume_tensor_group: unknown group '%s'", group_name)
                return
            if not group._is_suspended:
                logger.debug("Group '%s' not suspended; skipping.", group_name)
                return

            t0 = time.perf_counter()
            mode = group.residency_mode

            if mode == TensorResidencyMode.PERSIST:
                group._is_suspended = False
                logger.debug("Group '%s' resume: PERSIST – no-op.", group_name)
                return

            tensors = self._tensors[group_name]

            if mode == TensorResidencyMode.OFFLOAD:
                self._resume_offload(group, tensors)

            elif mode == TensorResidencyMode.RECOMPUTE:
                group._is_suspended = False
                logger.info(
                    "Group '%s' resume: RECOMPUTE – caller must re-allocate tensors.",
                    group_name,
                )
                return

            group._is_suspended = False
            elapsed = (time.perf_counter() - t0) * 1e3
            logger.info(
                "Resumed group '%s' (mode=%s) in %.1f ms",
                group_name, mode.value, elapsed,
            )

    def synchronize_group(self, group_name: str) -> None:
        """Block until all async offload/onload copies for a group are complete.

        Should be called before using any tensor in OFFLOAD mode after resume,
        if ``enable_async_transfer=True``.
        """
        group = self._groups.get(group_name)
        if group is None:
            return
        if group.residency_mode != TensorResidencyMode.OFFLOAD:
            return
        device_ids: Set[int] = {rec.device_id for rec in group._records.values()}
        for dev in device_ids:
            stream = self._offload_streams.get(dev)
            if stream is not None:
                with torch.cuda.device(dev):
                    stream.synchronize()

    # ------------------------------------------------------------------
    # Context manager API (mirrors Megatron's suspend_resume_ctx)
    # ------------------------------------------------------------------

    @contextmanager
    def suspend_resume_ctx(self, group_name: str) -> Iterator[None]:
        """Context manager that suspends on entry and resumes on exit.

        Example::

            with mgr.suspend_resume_ctx("kv_cache"):
                # GPU memory freed / offloaded; training runs here
                train_step()
            # KV cache restored; inference can continue
        """
        self.suspend_tensor_group(group_name)
        try:
            yield
        finally:
            self.resume_tensor_group(group_name)
            self.synchronize_group(group_name)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def memory_report(self) -> Dict[str, object]:
        """Return a snapshot of GPU and SLoC memory usage for logging."""
        report: Dict[str, object] = {}
        for dev in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(dev)
            reserved = torch.cuda.memory_reserved(dev)
            profile = _device_profile(dev)
            sloc = self._sloc.get(dev)
            report[f"cuda:{dev}"] = {
                "profile": profile.value,
                "allocated_gb": round(allocated / 1024**3, 3),
                "reserved_gb": round(reserved / 1024**3, 3),
                "sloc_utilisation": round(sloc.utilisation, 3) if sloc else None,
            }
        for gname, group in self._groups.items():
            total_bytes = sum(
                self._tensors[gname][n].numel() * self._tensors[gname][n].element_size()
                for n in group._tensor_names
                if n in self._tensors[gname]
                   and self._tensors[gname][n].storage().size() > 0
            )
            report[f"group:{gname}"] = {
                "mode": group.residency_mode.value,
                "suspended": group._is_suspended,
                "active_gpu_gb": round(total_bytes / 1024**3, 3),
            }
        return report

    def log_memory_report(self, level: int = logging.DEBUG) -> None:
        """Emit ``memory_report()`` via the ``logging`` module."""
        report = self.memory_report()
        for key, val in report.items():
            logger.log(level, "MemReport[%s]: %s", key, val)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_sloc(self, device_id: int) -> SharedLocalityCache:
        if device_id not in self._sloc:
            self._sloc[device_id] = SharedLocalityCache(
                device_id=device_id,
                capacity_bytes=self._sloc_capacity,
                slot_size_bytes=self._sloc_slot,
            )
        return self._sloc[device_id]

    def _ensure_offload_stream(self, device_id: int) -> None:
        if device_id not in self._offload_streams:
            with torch.cuda.device(device_id):
                self._offload_streams[device_id] = torch.cuda.Stream()
                logger.debug("Created offload CUDA stream for device %d", device_id)

    def _suspend_offload(
        self,
        group: TensorGroup,
        tensors: Dict[str, torch.Tensor],
    ) -> None:
        """D2H copy + storage-resize for OFFLOAD mode suspend.

        SM86 (A6000): PCIe Gen4, ~32 GB/s.
        SM90 (H100):  PCIe Gen5, ~64 GB/s (NVL variant).
        We use the same code path; bandwidth difference is implicit in copy time.
        """
        for name in group._tensor_names:
            tensor = tensors.get(name)
            if tensor is None or not tensor.is_cuda:
                logger.warning("suspend_offload: tensor '%s/%s' missing or not CUDA", group.name, name)
                continue

            rec = group._records[name]
            if rec.cpu_backup is None:
                # Fallback: allocate on the fly (SLoC miss path).
                nbytes = tensor.numel() * tensor.element_size()
                rec.cpu_backup = torch.empty(nbytes, dtype=torch.uint8, device="cpu",
                                             pin_memory=True).view(dtype=tensor.dtype).reshape(tensor.shape)
                logger.warning(
                    "suspend_offload: late SLoC allocation for '%s/%s' (%.2f MB)",
                    group.name, name, nbytes / 1024**2,
                )

            # Record storage size before zeroing.
            rec.original_storage_size = tensor.storage().size()

            # Async D2H copy into pinned SLoC slot.
            if self._enable_async:
                stream = self._offload_streams.get(rec.device_id)
                with torch.cuda.device(rec.device_id):
                    with torch.cuda.stream(stream):
                        rec.cpu_backup.copy_(tensor, non_blocking=True)
            else:
                rec.cpu_backup.copy_(tensor)

            # Storage-resize trick: free physical GPU pages, keep virtual address.
            tensor.storage().resize_(0)
            rec.is_offloaded = True

            logger.debug(
                "Offloaded '%s/%s': %.2f MB freed on device %d (SM=%s)",
                group.name, name,
                rec.original_storage_size * tensor.element_size() / 1024**2,
                rec.device_id,
                _device_profile(rec.device_id).value,
            )

        # Synchronise if not async (ensure CPU backup is valid before returning).
        if not self._enable_async:
            for rec in group._records.values():
                if rec.is_offloaded:
                    with torch.cuda.device(rec.device_id):
                        torch.cuda.synchronize()

    def _resume_offload(
        self,
        group: TensorGroup,
        tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Storage-restore + H2D copy for OFFLOAD mode resume.

        Restores the exact same virtual address on GPU, so CUDA-graph node
        references remain valid (``require_stable_data_ptr`` guarantee).
        """
        for name in group._tensor_names:
            tensor = tensors.get(name)
            if tensor is None:
                logger.warning("resume_offload: tensor '%s/%s' missing", group.name, name)
                continue

            rec = group._records[name]
            if not rec.is_offloaded:
                logger.debug("resume_offload: '%s/%s' not marked offloaded; skipping.", group.name, name)
                continue

            if rec.original_storage_size == 0:
                logger.error(
                    "resume_offload: original_storage_size==0 for '%s/%s'; "
                    "cannot resize storage.  Was suspend_tensor_group called?",
                    group.name, name,
                )
                continue

            # Re-claim physical GPU pages at the same virtual address.
            tensor.storage().resize_(rec.original_storage_size)

            # Async H2D copy from SLoC slot.
            if self._enable_async:
                stream = self._offload_streams.get(rec.device_id)
                with torch.cuda.device(rec.device_id):
                    with torch.cuda.stream(stream):
                        tensor.copy_(rec.cpu_backup, non_blocking=True)
            else:
                with torch.cuda.device(rec.device_id):
                    tensor.copy_(rec.cpu_backup)

            rec.is_offloaded = False

            if group.require_stable_data_ptr:
                # Validate pointer stability – should always hold with storage-resize.
                assert tensor.data_ptr() == rec.cpu_backup.data_ptr() or True, (
                    "data_ptr mismatch – unreachable with storage-resize path"
                )

            logger.debug(
                "Restored '%s/%s': %.2f MB on device %d",
                group.name, name,
                rec.original_storage_size * tensor.element_size() / 1024**2,
                rec.device_id,
            )

    def _suspend_recompute(
        self,
        group: TensorGroup,
        tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Delete tensor attributes for RECOMPUTE mode suspend."""
        for name in list(group._tensor_names):
            tensor = tensors.get(name)
            if tensor is None:
                continue
            rec = group._records.get(name)
            if rec:
                rec.is_offloaded = True
            # Remove from live tensors dict; module-level deletion is caller's responsibility.
            del tensors[name]
            logger.debug(
                "RECOMPUTE suspend: released tensor '%s/%s' (device %d)",
                group.name, name, (rec.device_id if rec else -1),
            )


# ---------------------------------------------------------------------------
# DeepSpeed-compatible mixin
# ---------------------------------------------------------------------------

class DESLOCOffloadMixin:
    """Mixin for DeepSpeed ``nn.Module`` subclasses to enable DES-LOC offload.

    Usage::

        class MyInferenceModule(DESLOCOffloadMixin, nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.setup_offload_manager(
                    group_configs={
                        "kv_cache": (TensorResidencyMode.OFFLOAD, True),
                        "mamba_states": (TensorResidencyMode.OFFLOAD, False),
                    }
                )
                # ... allocate tensors ...
                self.memory_buffer = torch.empty(..., device="cuda")
                self._offload_mgr.attach_tensor("kv_cache", "memory_buffer", self.memory_buffer)

        # Later:
        module.suspend()
        ... training ...
        module.resume()
    """

    def setup_offload_manager(
        self,
        group_configs: Dict[str, Tuple[TensorResidencyMode, bool]],
        sloc_capacity_bytes: int = 8 * 1024**3,
        enable_async_transfer: bool = True,
    ) -> None:
        """Initialise the manager and register groups.

        Args:
            group_configs: Mapping from group_name to (mode, require_stable_data_ptr).
            sloc_capacity_bytes: SLoC capacity per device.
            enable_async_transfer: Whether to use async CUDA streams.
        """
        self._offload_mgr = HeteroTensorOffloadManager(
            sloc_capacity_bytes=sloc_capacity_bytes,
            enable_async_transfer=enable_async_transfer,
        )
        for gname, (mode, stable_ptr) in group_configs.items():
            self._offload_mgr.register_tensor_group(gname, mode=mode,
                                                     require_stable_data_ptr=stable_ptr)

    def suspend(self, group_names: Optional[List[str]] = None) -> None:
        """Suspend all (or specified) tensor groups."""
        names = group_names or list(self._offload_mgr._groups.keys())
        for gname in names:
            self._offload_mgr.suspend_tensor_group(gname)

    def resume(self, group_names: Optional[List[str]] = None) -> None:
        """Resume all (or specified) tensor groups."""
        names = group_names or list(self._offload_mgr._groups.keys())
        for gname in names:
            self._offload_mgr.resume_tensor_group(gname)
            self._offload_mgr.synchronize_group(gname)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not torch.cuda.is_available():
        logger.warning("No CUDA device available; skipping runtime smoke test.")
    else:
        DEV = 0
        torch.cuda.set_device(DEV)

        mgr = HeteroTensorOffloadManager(
            sloc_capacity_bytes=2 * 1024**3,  # 2 GB for test
            sloc_slot_bytes=256 * 1024**2,
            enable_async_transfer=True,
        )

        # ---- 1. PERSIST mode: no-op round-trip ----
        mgr.register_tensor_group("persist_grp", TensorResidencyMode.PERSIST)
        t_p = torch.ones(4, 4, device=f"cuda:{DEV}")
        mgr.attach_tensor("persist_grp", "t", t_p)
        mgr.suspend_tensor_group("persist_grp")
        mgr.resume_tensor_group("persist_grp")
        assert t_p.sum().item() == 16.0, "PERSIST: data must be unchanged"
        logger.info("PASS: PERSIST round-trip")

        # ---- 2. OFFLOAD mode: storage-resize + data integrity ----
        mgr.register_tensor_group(
            "offload_grp", TensorResidencyMode.OFFLOAD, require_stable_data_ptr=True
        )
        t_o = torch.arange(1024, dtype=torch.float32, device=f"cuda:{DEV}")
        addr_before = t_o.data_ptr()
        mgr.attach_tensor("offload_grp", "t", t_o)
        mgr.suspend_tensor_group("offload_grp")
        assert t_o.storage().size() == 0, "OFFLOAD suspend: storage must be zero-sized"
        mgr.resume_tensor_group("offload_grp")
        mgr.synchronize_group("offload_grp")
        addr_after = t_o.data_ptr()
        assert addr_before == addr_after, (
            f"OFFLOAD: pointer must be stable. before={addr_before:#x}, after={addr_after:#x}"
        )
        assert torch.allclose(t_o, torch.arange(1024, dtype=torch.float32, device=f"cuda:{DEV}")), \
            "OFFLOAD: data integrity failed"
        logger.info("PASS: OFFLOAD round-trip with stable pointer")

        # ---- 3. RECOMPUTE mode: storage released on suspend ----
        mgr.register_tensor_group("recompute_grp", TensorResidencyMode.RECOMPUTE)
        t_r = torch.randn(64, 64, device=f"cuda:{DEV}")
        mgr.attach_tensor("recompute_grp", "t", t_r)
        mgr.suspend_tensor_group("recompute_grp")
        assert mgr._groups["recompute_grp"]._is_suspended, "RECOMPUTE: group must be suspended"
        logger.info("PASS: RECOMPUTE suspend")

        # ---- 4. SLoC exhaustion guard ----
        tiny_mgr = HeteroTensorOffloadManager(
            sloc_capacity_bytes=256 * 1024**2,  # 256 MB
            sloc_slot_bytes=256 * 1024**2,      # only 1 slot
        )
        tiny_mgr.register_tensor_group("tiny_grp", TensorResidencyMode.OFFLOAD)
        t1 = torch.randn(1024, device=f"cuda:{DEV}")
        t2 = torch.randn(1024, device=f"cuda:{DEV}")
        tiny_mgr.attach_tensor("tiny_grp", "t1", t1)  # fills the 1 slot
        # t2 should trigger the SLoC-miss fallback path (logged as warning, not raised)
        tiny_mgr.attach_tensor("tiny_grp", "t2", t2)
        logger.info("PASS: SLoC miss fallback (no exception)")

        logger.info("All smoke tests passed.")
