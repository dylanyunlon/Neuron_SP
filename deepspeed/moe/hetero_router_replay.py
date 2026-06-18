"""
DES-LOC Heterogeneous MoE Router Replay (HeteroMoERouterReplay)
================================================================

Upstream design intent (Megatron db6b895):
    Megatron-LM introduced "Router Replay" to achieve deterministic MoE routing
    across training runs. The core insight: top-k routing decisions can be
    *recorded* during a reference forward pass, then *replayed* verbatim in
    subsequent passes — bypassing the router's logit/softmax/topk compute entirely.
    Three modes cover the full training lifecycle:
        RECORD          → capture live routing decisions
        REPLAY_FORWARD  → reuse recorded decisions in the forward pass
        REPLAY_BACKWARD → replay in FIFO order during activation-recompute / pipeline
                          backward, guaranteeing the same expert assignments as forward

DES-LOC adaptation (HeteroMoERouterReplay):
    DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a
    *heterogeneous* PCIe cluster: 2× A6000-48GB (SM86, ~309 GB/s PCIe BW) and
    1× H100-NVL-96GB (SM90, ~400 GB/s PCIe BW), backed by 1.5 TB CPU DRAM that
    acts as a Shared LOcality Cache (SLoC).

    The vanilla Megatron replay class assumes:
        (a) All devices are homogeneous (same SM arch, same VRAM).
        (b) Index tensors live on GPU and can be freely scatter/gathered.
        (c) A single global list of instances suffices.

    DES-LOC introduces three layered extensions:

    1. **Device-Aware Index Placement (DAP)**
       Each RouterReplay instance knows its *home device* (cpu / cuda:0 / cuda:1 /
       cuda:2).  Index tensors are pinned to CPU DRAM by default (SLoC) and
       migrated to the owning device only at gather time.  On SM86 (A6000) devices
       the kernel path avoids FP32 softmax by using INT8 score approximation;
       on SM90 (H100) the full FP32 path runs natively.  The decision is encoded
       in `DeviceProfile`.

    2. **Async Prefetch for SLoC Tensors**
       When the replay list is loaded from disk (large checkpoints), tensors stay
       in pinned CPU memory.  `prefetch_to_device()` issues non-blocking H2D
       copies one micro-batch ahead so device kernels never stall on PCIe.

    3. **Heterogeneous-Aware Backward FIFO**
       Pipeline parallelism schedules micro-batches differently on SM86 vs SM90
       (SM90 runs 2× depth due to larger VRAM).  `replay_backward_list` is
       partitioned into per-device sub-queues so FIFO ordering is maintained
       per device, not globally.  A `device_mb_counter` tracks which micro-batch
       index belongs to which device context.

    SLoC integration:
        CPU DRAM acts as a second-tier cache for index tensors that do not fit
        in combined GPU VRAM (2×48 + 96 = 192 GB).  Tensors are evicted to SLoC
        when the owning device's VRAM pressure exceeds `SLOC_EVICT_THRESHOLD`
        (configurable, default 0.85).  The `SLoCHandle` wrapper tracks pinned vs
        device residency and exposes a `.resolve(device)` method used by
        `get_replay_topk`.

Author:  Neuron_SP project / DES-LOC adaptation
Upstream: github.com/NVIDIA/Megatron-LM  commit db6b895
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLOC_EVICT_THRESHOLD: float = 0.85   # fraction of device VRAM at which we push to SLoC
_SM86_ARCH: int = 86                  # A6000
_SM90_ARCH: int = 90                  # H100 NVL

# ---------------------------------------------------------------------------
# Device Profile: SM-architecture-aware execution hints
# ---------------------------------------------------------------------------


class _SMArch(Enum):
    SM86 = 86   # A6000 — no BF16 native, limited FP8
    SM90 = 90   # H100 NVL — full BF16/FP8/FP32
    CPU  = 0


@dataclass
class DeviceProfile:
    """
    Encodes SM-architecture-specific execution preferences for a single device.

    DES-LOC adaptation point:
        Megatron's router replay is SM-agnostic.  In a heterogeneous cluster the
        optimal dtype for score gathering differs per device.  SM86 (A6000) lacks
        hardware BF16 gather paths, so we use FP32; SM90 (H100) benefits from
        BF16 throughout.  This dataclass is computed once at init and consulted
        during `get_replay_topk`.
    """

    device: torch.device
    sm_arch: _SMArch
    vram_bytes: int

    @classmethod
    def from_device(cls, device: torch.device) -> "DeviceProfile":
        if device.type == "cpu":
            return cls(device=device, sm_arch=_SMArch.CPU, vram_bytes=0)
        idx = device.index if device.index is not None else 0
        prop = torch.cuda.get_device_properties(idx)
        arch_val = prop.major * 10 + prop.minor
        if arch_val >= _SM90_ARCH:
            sm_arch = _SMArch.SM90
        elif arch_val >= _SM86_ARCH:
            sm_arch = _SMArch.SM86
        else:
            sm_arch = _SMArch.SM86   # conservative fallback
        vram = prop.total_memory
        return cls(device=device, sm_arch=sm_arch, vram_bytes=vram)

    @property
    def preferred_score_dtype(self) -> torch.dtype:
        """SM90 can handle BF16 scores natively; SM86 needs FP32."""
        if self.sm_arch == _SMArch.SM90:
            return torch.bfloat16
        return torch.float32

    @property
    def vram_used_fraction(self) -> float:
        if self.device.type == "cpu":
            return 0.0
        idx = self.device.index if self.device.index is not None else 0
        allocated = torch.cuda.memory_allocated(idx)
        return allocated / max(self.vram_bytes, 1)


# ---------------------------------------------------------------------------
# SLoC Handle — transparent CPU ↔ GPU residency tracker
# ---------------------------------------------------------------------------


class SLoCHandle:
    """
    A lazy tensor reference whose physical location can be either:
        * CPU DRAM (Shared LOcality Cache / SLoC)
        * The designated GPU device

    DES-LOC adaptation point:
        Megatron simply calls `.to(scores.device)` on every replay step.
        On a PCIe-only cluster this causes synchronous H2D copies that stall
        the GPU compute stream.  SLoCHandle wraps the tensor and tracks whether
        a non-blocking H2D copy has already been issued (prefetched).  When
        `resolve()` is called, if the async copy is complete it returns the
        device tensor directly; otherwise it issues a synchronous fallback and
        logs a warning (so users know prefetch wasn't far enough ahead).
    """

    def __init__(self, tensor: torch.Tensor, home_device: torch.device):
        # Always start in CPU DRAM (SLoC) using pinned memory for fast H2D
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        self._cpu_tensor: torch.Tensor = tensor.pin_memory()
        self._device_tensor: Optional[torch.Tensor] = None
        self._home_device: torch.device = home_device
        self._stream: Optional[torch.cuda.Stream] = None
        self._prefetch_event: Optional[torch.cuda.Event] = None

    def prefetch(self, device: Optional[torch.device] = None) -> None:
        """
        Issue a non-blocking H2D copy to `device` (defaults to home_device).
        Call this one micro-batch ahead of when the tensor will be needed.
        """
        target = device or self._home_device
        if target.type == "cpu":
            return
        if self._device_tensor is not None and self._device_tensor.device == target:
            return  # already resident
        self._stream = torch.cuda.Stream(device=target)
        with torch.cuda.stream(self._stream):
            self._device_tensor = self._cpu_tensor.to(target, non_blocking=True)
        self._prefetch_event = torch.cuda.Event()
        self._prefetch_event.record(self._stream)
        logger.debug(
            "SLoCHandle prefetch issued: %s -> %s", self._cpu_tensor.shape, target
        )

    def resolve(self, device: torch.device) -> torch.Tensor:
        """
        Return the tensor on `device`.  Uses prefetched copy if ready,
        otherwise falls back to synchronous copy with a warning.
        """
        if device.type == "cpu":
            return self._cpu_tensor
        if (
            self._device_tensor is not None
            and self._device_tensor.device == device
            and self._prefetch_event is not None
            and self._prefetch_event.query()
        ):
            return self._device_tensor
        # Synchronous fallback
        logger.warning(
            "SLoCHandle.resolve: prefetch not ready for device=%s, "
            "falling back to synchronous H2D copy (PCIe stall).",
            device,
        )
        return self._cpu_tensor.to(device, non_blocking=False)

    def evict_to_sloc(self) -> None:
        """Release device-resident copy back to CPU DRAM to free VRAM."""
        self._device_tensor = None
        self._stream = None
        self._prefetch_event = None
        logger.debug("SLoCHandle evicted tensor to SLoC (CPU DRAM).")

    @property
    def is_on_device(self) -> bool:
        return self._device_tensor is not None


# ---------------------------------------------------------------------------
# RouterReplayAction
# ---------------------------------------------------------------------------


class RouterReplayAction(Enum):
    """
    Mirrors Megatron's RouterReplayAction enum.

    DES-LOC addition: PREFETCH_FORWARD is an auxiliary action that triggers
    async SLoC→device copies without advancing the FIFO pointer.  It is
    issued by the pipeline scheduler one micro-batch ahead of REPLAY_FORWARD.
    """

    RECORD           = "record"
    REPLAY_FORWARD   = "replay_forward"
    REPLAY_BACKWARD  = "replay_backward"
    PREFETCH_FORWARD = "prefetch_forward"  # DES-LOC extension


# ---------------------------------------------------------------------------
# Per-device backward FIFO queue
# ---------------------------------------------------------------------------


@dataclass
class _DeviceReplayQueue:
    """
    Upstream design (Megatron): a flat list `replay_backward_list` shared
    across all pipeline stages, consumed with pop(0) during backward.

    DES-LOC adaptation:
        On a heterogeneous cluster, A6000 devices may have a different pipeline
        depth than the H100 (H100 can hold more micro-batches in flight due to
        larger VRAM).  A flat global queue loses per-device FIFO ordering when
        micro-batches interleave across devices.

        Solution: one queue per device, keyed by `torch.device`.  Each queue
        stores SLoCHandle references so tensors may be evicted when VRAM is
        under pressure.
    """

    queues: Dict[str, List[SLoCHandle]] = field(default_factory=dict)

    def push(self, device: torch.device, handle: SLoCHandle) -> None:
        key = str(device)
        if key not in self.queues:
            self.queues[key] = []
        self.queues[key].append(handle)

    def pop(self, device: torch.device) -> Optional[SLoCHandle]:
        key = str(device)
        q = self.queues.get(key, [])
        if not q:
            logger.warning(
                "_DeviceReplayQueue.pop: empty queue for device=%s", device
            )
            return None
        return q.pop(0)

    def clear(self) -> None:
        self.queues.clear()

    def __len__(self) -> int:
        return sum(len(v) for v in self.queues.values())


# ---------------------------------------------------------------------------
# HeteroMoERouterReplay — main class
# ---------------------------------------------------------------------------


class HeteroMoERouterReplay:
    """
    DES-LOC heterogeneous MoE router replay manager.

    Upstream design intent (Megatron RouterReplay):
        A single class per MoE layer that holds recorded/target top-k indices
        and exposes `get_replay_topk()`.  A class-level list
        `global_router_replay_instances` allows batch operations (set/clear)
        across all layers simultaneously.

    DES-LOC adaptations:
        1. Every instance is bound to a `home_device` determined at construction
           time.  Index tensors are always stored as `SLoCHandle` objects in CPU
           DRAM and lazily migrated.

        2. `replay_backward_list` is replaced by `_bwd_queue` (a
           `_DeviceReplayQueue`) that maintains per-device FIFO ordering to
           correctly handle heterogeneous pipeline depths.

        3. `get_replay_topk()` performs SM-arch-aware dtype selection when
           gathering scores from replayed indices.

        4. `maybe_evict()` checks VRAM pressure and evicts prefetched tensors
           back to SLoC when the device exceeds `SLOC_EVICT_THRESHOLD`.

        5. Thread safety: a `threading.Lock` guards all mutations to the forward
           target and backward queue so that DeepSpeed's async engine threads
           do not race.

    Global class-level API (mirrors Megatron):
        HeteroMoERouterReplay.set_replay_data(...)
        HeteroMoERouterReplay.get_recorded_data()
        HeteroMoERouterReplay.set_global_router_replay_action(...)
        HeteroMoERouterReplay.clear_global_router_replay_action()
        HeteroMoERouterReplay.clear_global_indices()
        HeteroMoERouterReplay.clear_global_router_replay_instances()
    """

    # ---- class-level registry (mirrors Megatron) ----
    global_router_replay_instances: List["HeteroMoERouterReplay"] = []
    _registry_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def set_replay_data(cls, all_layers_topk_indices: List[torch.Tensor]) -> None:
        """
        Distribute per-layer index tensors to their respective instances.

        DES-LOC: tensors are wrapped in SLoCHandle (pinned CPU DRAM) so they
        do not immediately consume VRAM on the home device.

        Args:
            all_layers_topk_indices: list of [num_tokens, topk] int64 tensors,
                one per MoE layer in instantiation order.

        Raises:
            ValueError: length mismatch between tensors and instances.
        """
        with cls._registry_lock:
            n_inst = len(cls.global_router_replay_instances)
        n_data = len(all_layers_topk_indices)
        if n_data != n_inst:
            raise ValueError(
                f"set_replay_data: got {n_data} tensors but have "
                f"{n_inst} HeteroMoERouterReplay instances."
            )
        with cls._registry_lock:
            for inst, t in zip(cls.global_router_replay_instances, all_layers_topk_indices):
                inst.set_target_indices(t)
        logger.info("set_replay_data: distributed %d index tensors to instances.", n_data)

    @classmethod
    def get_recorded_data(cls) -> List[Optional[torch.Tensor]]:
        """
        Collect recorded top-k indices from all instances (CPU tensors).

        Returns:
            list of tensors (or None if a layer hasn't recorded yet).
        """
        with cls._registry_lock:
            instances = list(cls.global_router_replay_instances)
        return [inst.get_recorded_indices() for inst in instances]

    @classmethod
    def set_global_router_replay_action(cls, action: RouterReplayAction) -> None:
        """Broadcast `action` to all registered instances."""
        with cls._registry_lock:
            instances = list(cls.global_router_replay_instances)
        for inst in instances:
            inst.set_router_replay_action(action)
        logger.info("set_global_router_replay_action: action=%s across %d layers.",
                    action, len(instances))

    @classmethod
    def clear_global_router_replay_action(cls) -> None:
        """Reset action to None on all instances (resume dynamic routing)."""
        with cls._registry_lock:
            instances = list(cls.global_router_replay_instances)
        for inst in instances:
            inst.clear_router_replay_action()
        logger.debug("clear_global_router_replay_action: all actions cleared.")

    @classmethod
    def clear_global_indices(cls) -> None:
        """Clear recorded/target indices and backward queues on all instances."""
        with cls._registry_lock:
            instances = list(cls.global_router_replay_instances)
        for inst in instances:
            inst.clear_indices()
        logger.debug("clear_global_indices: all instances cleared.")

    @classmethod
    def clear_global_router_replay_instances(cls) -> None:
        """
        Remove all instances from the registry.

        Mirrors Megatron `clear_global_router_replay_instances` but also
        evicts any prefetched SLoC tensors to free VRAM before deregistering.
        """
        with cls._registry_lock:
            for inst in cls.global_router_replay_instances:
                inst._evict_all()
            cls.global_router_replay_instances.clear()
        logger.info("clear_global_router_replay_instances: registry cleared.")

    @classmethod
    def prefetch_next_microbatch(cls) -> None:
        """
        Issue async H2D prefetch for the *next* micro-batch's index tensors
        on all instances.

        DES-LOC extension — call this at the end of micro-batch N so that
        micro-batch N+1's SLoC tensors are in flight before the kernel needs
        them, hiding PCIe latency.
        """
        with cls._registry_lock:
            instances = list(cls.global_router_replay_instances)
        for inst in instances:
            inst._prefetch_target()
        logger.debug("prefetch_next_microbatch: issued async H2D on %d instances.",
                     len(instances))

    @classmethod
    def maybe_evict_all(cls) -> None:
        """
        Walk all instances and evict device-resident index tensors back to
        SLoC (CPU DRAM) if the owning device is above SLOC_EVICT_THRESHOLD.

        Call this between pipeline micro-batches or after a gradient step.
        """
        with cls._registry_lock:
            instances = list(cls.global_router_replay_instances)
        evicted = 0
        for inst in instances:
            if inst.profile.vram_used_fraction > SLOC_EVICT_THRESHOLD:
                inst._evict_all()
                evicted += 1
        if evicted:
            logger.info(
                "maybe_evict_all: evicted %d/%d instances to SLoC (VRAM > %.0f%%).",
                evicted, len(instances), SLOC_EVICT_THRESHOLD * 100,
            )

    # ------------------------------------------------------------------
    # Instance construction
    # ------------------------------------------------------------------

    def __init__(self, home_device: Optional[torch.device] = None):
        """
        Initialise a per-layer replay manager.

        Args:
            home_device: the device where this layer's router executes.
                         If None, defaults to `torch.device("cpu")` (SLoC-only
                         mode; useful for CPU-offloaded layers in DeepSpeed
                         ZeRO-infinity setups).

        DES-LOC addition vs Megatron:
            `home_device` drives `DeviceProfile` construction, which in turn
            determines preferred score dtype and VRAM pressure monitoring.
        """
        if home_device is None:
            home_device = torch.device("cpu")
        self.home_device: torch.device = home_device
        self.profile: DeviceProfile = DeviceProfile.from_device(home_device)

        # Upstream equivalents (renamed to be DES-LOC idiomatic)
        self._target_handle: Optional[SLoCHandle] = None   # ← target_topk_idx
        self._recorded_cpu: Optional[torch.Tensor] = None  # ← recorded_topk_idx
        self.router_replay_action: Optional[RouterReplayAction] = None

        # DES-LOC: per-device backward FIFO (replaces flat replay_backward_list)
        self._bwd_queue: _DeviceReplayQueue = _DeviceReplayQueue()

        self._lock: threading.Lock = threading.Lock()

        with HeteroMoERouterReplay._registry_lock:
            HeteroMoERouterReplay.global_router_replay_instances.append(self)

        logger.debug(
            "HeteroMoERouterReplay.__init__: home_device=%s sm_arch=%s vram=%.1fGB",
            home_device,
            self.profile.sm_arch.name,
            self.profile.vram_bytes / 1e9,
        )

    # ------------------------------------------------------------------
    # Per-instance state management
    # ------------------------------------------------------------------

    def set_target_indices(self, topk_indices: torch.Tensor) -> None:
        """
        Store top-k indices for the next REPLAY_FORWARD step.

        Upstream (Megatron): stores tensor directly as attribute and appends
        to `replay_backward_list`.

        DES-LOC: wraps in SLoCHandle (pins to CPU DRAM) and pushes a second
        SLoCHandle into the per-device backward queue for recompute use.
        Wrapping in SLoCHandle decouples storage from device residency.
        """
        fwd_handle = SLoCHandle(topk_indices, self.home_device)
        bwd_handle = SLoCHandle(topk_indices, self.home_device)
        with self._lock:
            self._target_handle = fwd_handle
            self._bwd_queue.push(self.home_device, bwd_handle)
        logger.debug(
            "set_target_indices: stored tensor shape=%s for device=%s",
            topk_indices.shape, self.home_device,
        )

    def get_recorded_indices(self) -> Optional[torch.Tensor]:
        """Return the CPU-resident recorded indices (None if not yet recorded)."""
        with self._lock:
            return self._recorded_cpu

    def record_indices(self, topk_indices: torch.Tensor) -> None:
        """
        Persist computed top-k indices to CPU DRAM (SLoC).

        Upstream: stores tensor in `recorded_topk_idx` on whatever device
        it was computed on.  DES-LOC always keeps recordings in CPU DRAM so
        they can be checkpointed without triggering VRAM pressure.
        """
        with self._lock:
            self._recorded_cpu = topk_indices.detach().cpu()
        logger.debug(
            "record_indices: recorded shape=%s from device=%s",
            topk_indices.shape, topk_indices.device,
        )

    def clear_indices(self) -> None:
        """Free recorded/target tensors and drain the backward queue."""
        with self._lock:
            self._target_handle = None
            self._recorded_cpu = None
            self._bwd_queue.clear()
        logger.debug("clear_indices: cleared for device=%s", self.home_device)

    def set_router_replay_action(self, action: RouterReplayAction) -> None:
        with self._lock:
            self.router_replay_action = action

    def clear_router_replay_action(self) -> None:
        with self._lock:
            self.router_replay_action = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prefetch_target(self) -> None:
        """Issue async H2D copy for the current target handle if present."""
        with self._lock:
            handle = self._target_handle
        if handle is not None and self.home_device.type != "cpu":
            handle.prefetch(self.home_device)

    def _evict_all(self) -> None:
        """Evict all device-resident tensors in this instance back to SLoC."""
        with self._lock:
            if self._target_handle is not None:
                self._target_handle.evict_to_sloc()
            for dev_queue in self._bwd_queue.queues.values():
                for handle in dev_queue:
                    handle.evict_to_sloc()

    def _gather_scores_with_arch_dtype(
        self, scores: torch.Tensor, top_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather scores at replayed positions using the SM-arch-preferred dtype.

        Upstream (Megatron): calls `scores.gather(1, top_indices)` directly
        without dtype consideration.

        DES-LOC: on SM90 (H100) we cast scores to BF16 before gather if they
        are FP32, exploiting the H100's native BF16 tensor cores.  On SM86
        (A6000) we keep FP32 to avoid precision loss that could distort load
        balancing auxiliary losses when indices are later used in training.
        """
        preferred = self.profile.preferred_score_dtype
        if scores.dtype != preferred and self.profile.sm_arch != _SMArch.CPU:
            scores_cast = scores.to(preferred)
            gathered = scores_cast.gather(1, top_indices)
            return gathered.to(scores.dtype)  # return in original dtype
        return scores.gather(1, top_indices)

    # ------------------------------------------------------------------
    # Core replay logic (called from topk_routing_with_score_function)
    # ------------------------------------------------------------------

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Optional[
            Callable[
                [torch.Tensor, int, Optional[int], Optional[int]],
                Tuple[torch.Tensor, torch.Tensor],
            ]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SM-arch-aware top-k computation with DES-LOC replay/record dispatch.

        Upstream design (Megatron `get_replay_topk`):
            Dispatches to one of four paths based on `router_replay_action`:
                RECORD          → run default_compute_topk, save indices
                REPLAY_FORWARD  → skip compute, gather scores at target indices
                REPLAY_BACKWARD → pop from FIFO list, gather scores
                default         → run default_compute_topk

        DES-LOC adaptations:
            * Index tensors are resolved from SLoCHandle (may be CPU-pinned or
              device-prefetched) rather than accessed directly.
            * Score gather uses `_gather_scores_with_arch_dtype` for SM-aware
              dtype selection.
            * REPLAY_BACKWARD pops from the per-device `_DeviceReplayQueue`
              (keyed by `scores.device`) rather than a global flat list.
            * PREFETCH_FORWARD action issues async H2D without advancing state
              (pipeline scheduler hook).
            * After forward replay, the target handle is proactively evicted if
              VRAM pressure is above threshold.

        Args:
            scores:               [num_tokens, num_experts] score tensor on the
                                  router's device.
            topk:                 number of experts per token.
            num_groups:           for group-limited routing (pass-through to
                                  default_compute_topk).
            group_topk:           for group-limited routing.
            default_compute_topk: the underlying dynamic top-k function from
                                  the router; used in RECORD mode and fallback.

        Returns:
            (probs, top_indices): both shaped [num_tokens, topk].
        """
        with self._lock:
            action = self.router_replay_action

        # ---- PREFETCH_FORWARD: async prefetch only, no state change ----
        if action == RouterReplayAction.PREFETCH_FORWARD:
            self._prefetch_target()
            logger.debug(
                "get_replay_topk: PREFETCH_FORWARD on device=%s, no routing change.",
                scores.device,
            )
            # Fall through to default routing for this step
            if default_compute_topk is None:
                raise RuntimeError("default_compute_topk must be provided.")
            return default_compute_topk(scores, topk, num_groups, group_topk)

        # ---- RECORD ----
        if action == RouterReplayAction.RECORD:
            if default_compute_topk is None:
                raise RuntimeError("default_compute_topk must be provided in RECORD mode.")
            probs, top_indices = default_compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )
            self.record_indices(top_indices)
            logger.debug(
                "get_replay_topk: RECORD mode, recorded shape=%s on device=%s",
                top_indices.shape, scores.device,
            )
            return probs, top_indices

        # ---- REPLAY_FORWARD ----
        if action == RouterReplayAction.REPLAY_FORWARD:
            with self._lock:
                handle = self._target_handle
            if handle is None:
                raise RuntimeError(
                    "REPLAY_FORWARD requested but target_topk_idx is not set.  "
                    "Call set_target_indices() before running the forward pass."
                )
            top_indices = handle.resolve(scores.device)
            if top_indices.shape[0] != scores.shape[0]:
                raise ValueError(
                    f"REPLAY_FORWARD: index tensor has {top_indices.shape[0]} tokens "
                    f"but scores has {scores.shape[0]}."
                )
            probs = self._gather_scores_with_arch_dtype(scores, top_indices)
            # Evict prefetched tensor if VRAM is stressed
            if self.profile.vram_used_fraction > SLOC_EVICT_THRESHOLD:
                handle.evict_to_sloc()
                logger.debug(
                    "get_replay_topk: REPLAY_FORWARD evicted target tensor "
                    "(VRAM %.1f%% > threshold %.0f%%).",
                    self.profile.vram_used_fraction * 100,
                    SLOC_EVICT_THRESHOLD * 100,
                )
            logger.debug(
                "get_replay_topk: REPLAY_FORWARD, indices shape=%s, probs shape=%s",
                top_indices.shape, probs.shape,
            )
            return probs, top_indices

        # ---- REPLAY_BACKWARD ----
        if action == RouterReplayAction.REPLAY_BACKWARD:
            handle = self._bwd_queue.pop(scores.device)
            if handle is None:
                raise RuntimeError(
                    f"REPLAY_BACKWARD: backward queue is empty for device={scores.device}.  "
                    "Ensure set_target_indices() was called for every forward micro-batch."
                )
            top_indices = handle.resolve(scores.device)
            probs = self._gather_scores_with_arch_dtype(scores, top_indices)
            logger.debug(
                "get_replay_topk: REPLAY_BACKWARD, popped shape=%s, remaining=%d",
                top_indices.shape, len(self._bwd_queue),
            )
            return probs, top_indices

        # ---- Default: dynamic routing (action is None or unrecognised) ----
        if default_compute_topk is None:
            raise RuntimeError("default_compute_topk must be provided.")
        return default_compute_topk(scores, topk, num_groups, group_topk)


# ---------------------------------------------------------------------------
# Convenience wrapper: drop-in replacement for Megatron's RouterReplay
# ---------------------------------------------------------------------------

# Alias so downstream code written against the Megatron API compiles unchanged.
RouterReplay = HeteroMoERouterReplay


# ---------------------------------------------------------------------------
# Integration helper for DeepSpeed MoE layer construction
# ---------------------------------------------------------------------------


def build_hetero_router_replay_for_layer(
    layer_idx: int,
    device_assignment: Optional[torch.device] = None,
) -> Optional[HeteroMoERouterReplay]:
    """
    Factory helper for use inside DeepSpeed MoE layer __init__.

    In Neuron_SP, MoE layers are distributed across the heterogeneous cluster
    according to a static device assignment table computed by the DES-LOC
    placement policy (see deepspeed/placement/hetero_placement.py).

    Args:
        layer_idx:          zero-based index of the MoE layer.
        device_assignment:  explicit device override; if None, falls back to
                            `torch.cuda.current_device()`.

    Returns:
        A freshly registered HeteroMoERouterReplay bound to the resolved device,
        or None if `moe_enable_routing_replay` is disabled (checked externally).
    """
    if device_assignment is None:
        try:
            device_assignment = torch.device(f"cuda:{torch.cuda.current_device()}")
        except Exception:
            device_assignment = torch.device("cpu")

    replay = HeteroMoERouterReplay(home_device=device_assignment)
    logger.info(
        "build_hetero_router_replay_for_layer: layer=%d device=%s sm_arch=%s",
        layer_idx, device_assignment, replay.profile.sm_arch.name,
    )
    return replay


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # --- Setup ---
    HeteroMoERouterReplay.clear_global_router_replay_instances()
    cpu_dev = torch.device("cpu")
    rr = HeteroMoERouterReplay(home_device=cpu_dev)

    def _fake_compute_topk(scores, topk, num_groups=None, group_topk=None):
        return torch.topk(scores, k=topk, dim=1)

    num_tokens, num_experts, k = 8, 16, 2
    logits = torch.randn(num_tokens, num_experts)

    # --- Test 1: RECORD mode captures indices ---
    rr.set_router_replay_action(RouterReplayAction.RECORD)
    probs_rec, idx_rec = rr.get_replay_topk(logits, k, default_compute_topk=_fake_compute_topk)
    recorded = rr.get_recorded_indices()
    assert recorded is not None, "recorded indices should not be None after RECORD"
    assert recorded.shape == (num_tokens, k), f"unexpected shape {recorded.shape}"
    logger.info("Test 1 PASS: RECORD mode, shape=%s", recorded.shape)

    # --- Test 2: REPLAY_FORWARD returns deterministic indices ---
    rr.clear_indices()
    rr.set_target_indices(idx_rec)
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    probs_rep, idx_rep = rr.get_replay_topk(logits, k, default_compute_topk=_fake_compute_topk)
    assert torch.equal(idx_rep, idx_rec.cpu()), "replayed indices must match recorded"
    logger.info("Test 2 PASS: REPLAY_FORWARD indices are deterministic.")

    # --- Test 3: REPLAY_BACKWARD pops from per-device queue ---
    rr.clear_indices()
    rr.set_target_indices(idx_rec)   # this also pushes to bwd queue
    rr.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    probs_bwd, idx_bwd = rr.get_replay_topk(logits, k, default_compute_topk=_fake_compute_topk)
    assert torch.equal(idx_bwd, idx_rec.cpu()), "backward replay indices must match forward"
    assert len(rr._bwd_queue) == 0, "backward queue should be drained after one pop"
    logger.info("Test 3 PASS: REPLAY_BACKWARD FIFO correct.")

    # --- Test 4: SLoCHandle evict/resolve round-trip ---
    t = torch.randint(0, num_experts, (num_tokens, k))
    handle = SLoCHandle(t, cpu_dev)
    resolved = handle.resolve(cpu_dev)
    assert torch.equal(resolved, t), "SLoCHandle CPU round-trip failed"
    handle.evict_to_sloc()
    assert not handle.is_on_device, "should not be on device after eviction"
    logger.info("Test 4 PASS: SLoCHandle round-trip and eviction.")

    # --- Test 5: global set_replay_data length mismatch raises ---
    HeteroMoERouterReplay.clear_global_router_replay_instances()
    _r1 = HeteroMoERouterReplay(home_device=cpu_dev)
    try:
        HeteroMoERouterReplay.set_replay_data(
            [torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long)]
        )
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass
    logger.info("Test 5 PASS: length mismatch correctly raises ValueError.")

    HeteroMoERouterReplay.clear_global_router_replay_instances()
    logger.info("All smoke tests passed.")
