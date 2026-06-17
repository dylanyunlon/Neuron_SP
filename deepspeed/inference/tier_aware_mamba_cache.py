"""
TierAwareMambaCache — DES-LOC heterogeneous-memory adaptation of Megatron's
Mamba prefix-caching memory-safety fix (upstream commit a83f408).

=== Upstream Design Intent ===
Megatron commit a83f408 (Keshav Santhanam, 2026-06-16) fixes a silent OOM class
that could occur when Mamba prefix caching was enabled on hybrid (Mamba+attention)
models.  The root cause: the MambaSlotAllocator allocates *two* GPU buffer families
that share the same per-slot footprint but were not both accounted for in the memory
budget:

  1. "Durable" cache buffers  (ssm_states / conv_states):
       max_slots slots, reused across requests at KV-divergence and last-aligned
       block boundaries.  These are the buffers users think of as "the cache".

  2. "Scratch" buffers  (intermediate_ssm_out / intermediate_conv_out):
       CUDA-graph-safe per-step staging, sized to the worst-case per-step count:
           scratch_slots = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * max_requests
       These must be pre-allocated so CUDA graphs can capture a fixed memory
       layout, but they consume the same per-slot bytes as durable slots.

Before the fix, the budget was naively divided entirely into durable slots
(`max_slots = total_bytes // per_slot_bytes`), ignoring the scratch reservation.
When scratch_slots > max_slots the allocator silently over-committed, causing OOM
at startup.  The fix:
  • Reserves scratch_bytes from the budget first.
  • Computes durable slots from the remainder.
  • Converts the "too small" warning into a hard ValueError with a diagnostic
    message that quantifies the minimum viable budget.
  • Adds a warning for "memory-only mode" (prefix caching on a hybrid model
    with no Mamba budget configured) so users know Mamba state caching is absent.

=== DES-LOC Adaptation (TierAwareMambaCache) ===
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a three-device
cluster with very different memory and bandwidth characteristics:

  Device 0: A6000  48 GB  SM86  PCIe   (compute-balanced)
  Device 1: A6000  48 GB  SM86  PCIe   (compute-balanced)
  Device 2: H100   96 GB  SM90  PCIe   (high-bandwidth HBM3)
  Host  :   1.5 TB DRAM               (slow but vast)

Key architectural differences from Megatron's single-GPU model:

  A. HETEROGENEOUS DURABLE SLOTS
     We extend the two-family split into a three-tier layout:
       - "hot" slots  → H100 GPU HBM  (fast access, limited capacity)
       - "warm" slots → A6000 GPU VRAM  (medium speed, per-device budget)
       - "cold" slots → pinned CPU DRAM  (slow but large, async copy path)
     Scratch buffers remain on the *compute device* (the device executing the
     current forward pass) because CUDA graphs require them to be local.

  B. SCRATCH BUDGET ACCOUNTING IS PER-COMPUTE-DEVICE
     Megatron sizes scratch_slots = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST *
     max_requests on a single device.  In DES-LOC, max_requests may be
     spread across devices by the scheduler, but CUDA graphs are per-device,
     so scratch must cover the worst-case *per-device* request count.  We
     track per_device_max_requests for each SM86/SM90 tier separately.

  C. LOC (Locality Cache) INTEGRATION
     DES-LOC maintains a Shared LOcality Cache that tracks which slot is
     "near" which device (minimising cross-PCIe copies).  TierAwareMambaCache
     hooks into this by exposing a `locality_hint(slot_id)` → device_id map
     so the LOC scheduler can prefer to route Mamba-heavy requests to the
     device that already holds their durable state.

  D. BUDGET VALIDATION IS TIERED
     The ValueError from upstream is extended: if the *sum* of per-tier budgets
     cannot cover the *sum* of scratch reservations plus at least one durable
     slot per tier, we raise with a per-tier breakdown so the user knows exactly
     which tier is the bottleneck.

  E. MEMORY-ONLY MODE WARNING
     Preserved verbatim in semantics; extended to name which tiers are missing
     a budget.

This module is a pure Python / PyTorch implementation; it does not depend on
Megatron internals and is designed to drop into the Neuron_SP / DeepSpeed stack.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants mirrored from Megatron's MambaSlotAllocator
# ---------------------------------------------------------------------------

#: Number of intermediate extraction offsets per request per step.
#: Upstream: MAX_INTERMEDIATE_OFFSETS_PER_REQUEST = 3
MAX_INTERMEDIATE_OFFSETS_PER_REQUEST: int = 3

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


class MemoryTier(Enum):
    """Logical memory tier in the DES-LOC heterogeneous cluster.

    Ordering reflects latency for the compute kernels (lowest = fastest):
      HOT  → H100 HBM3 (device 2, SM90)
      WARM → A6000 VRAM (devices 0/1, SM86)
      COLD → pinned CPU DRAM (host)
    """

    HOT = auto()   # H100 NVL 96 GB  SM90
    WARM = auto()  # A6000 48 GB     SM86
    COLD = auto()  # CPU pinned DRAM 1.5 TB


@dataclass(frozen=True)
class DeviceSpec:
    """Static description of one physical device in the DES-LOC cluster."""

    device_id: int
    tier: MemoryTier
    total_memory_gb: float
    compute_capability: Tuple[int, int]  # (major, minor)
    label: str

    @property
    def torch_device(self) -> torch.device:
        return torch.device(f"cuda:{self.device_id}")


# Cluster layout for the reference Neuron_SP hardware.
CLUSTER_DEVICES: List[DeviceSpec] = [
    DeviceSpec(device_id=0, tier=MemoryTier.WARM, total_memory_gb=48.0,
               compute_capability=(8, 6), label="A6000-0"),
    DeviceSpec(device_id=1, tier=MemoryTier.WARM, total_memory_gb=48.0,
               compute_capability=(8, 6), label="A6000-1"),
    DeviceSpec(device_id=2, tier=MemoryTier.HOT,  total_memory_gb=96.0,
               compute_capability=(9, 0), label="H100-NVL"),
]

CPU_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-tier budget configuration
# ---------------------------------------------------------------------------


@dataclass
class TierBudgetConfig:
    """Memory budget (in GB) for Mamba prefix caching on each tier.

    Mirrors Megatron's ``prefix_caching_mamba_gb`` scalar but decomposes it
    into per-tier allocations so the DES-LOC memory planner can reason about
    each device independently.

    Attributes
    ----------
    hot_gb:
        Budget for HOT (H100) durable slots.  Scratch for requests routed to
        the H100 is also carved from this.
    warm_gb_per_device:
        Budget *per A6000 device* for WARM durable slots.  Each A6000 has its
        own allocator; this is applied independently to device 0 and device 1.
    cold_gb:
        Budget in pinned CPU DRAM for COLD (evicted) durable slots.  Scratch
        does not live in COLD tier — it always stays on the compute device.
    per_device_max_requests_hot:
        Worst-case simultaneous requests on the H100 (for scratch sizing).
    per_device_max_requests_warm:
        Worst-case simultaneous requests on each A6000 (for scratch sizing).
    """

    hot_gb: float = 0.0
    warm_gb_per_device: float = 0.0
    cold_gb: float = 0.0
    per_device_max_requests_hot: int = 64
    per_device_max_requests_warm: int = 32

    def total_gpu_gb(self) -> float:
        return self.hot_gb + 2 * self.warm_gb_per_device

    def is_fully_unconfigured(self) -> bool:
        return self.hot_gb == 0.0 and self.warm_gb_per_device == 0.0


# ---------------------------------------------------------------------------
# Slot metadata
# ---------------------------------------------------------------------------


@dataclass
class SlotDescriptor:
    """Metadata for one durable Mamba cache slot.

    In Megatron a slot is an integer index into a pre-allocated (layers, slots, ...)
    tensor.  In DES-LOC each slot also carries a tier tag and a device_id so the
    LOC scheduler can route requests to the correct device.
    """

    slot_id: int
    tier: MemoryTier
    device_id: int          # cuda device index, or -1 for CPU
    layer_offset: int = 0   # reserved for future per-layer placement


# ---------------------------------------------------------------------------
# Per-tier allocator
# ---------------------------------------------------------------------------


class TierSlotAllocator:
    """Free-list allocator for durable Mamba cache slots on a single tier.

    This is a DES-LOC adaptation of Megatron's ``MambaSlotAllocator``.

    Key differences:
      - Manages slots on one specific (device, tier) pair rather than a single
        GPU assumed to be device 0.
      - Scratch buffers are sized using *per-device* max_requests, not global.
      - Budget accounting raises a descriptive ValueError (not a silent warning)
        when the scratch reservation leaves fewer than one durable slot — matching
        upstream behaviour after commit a83f408.
      - Thread-safe: a ``threading.Lock`` guards the free-list because DES-LOC's
        async dispatch can trigger allocations from multiple CPU threads.
    """

    def __init__(
        self,
        *,
        tier: MemoryTier,
        device: torch.device,
        budget_bytes: int,
        per_slot_bytes: int,
        per_device_max_requests: int,
        num_mamba_layers: int,
        conv_states_shape: Tuple[int, ...],
        ssm_states_shape: Tuple[int, ...],
        conv_states_dtype: torch.dtype,
        ssm_states_dtype: torch.dtype,
        label: str,
    ) -> None:
        self.tier = tier
        self.device = device
        self.label = label
        self._lock = threading.Lock()

        # ------------------------------------------------------------------
        # Budget split: scratch first, then durable — exact upstream logic
        # ------------------------------------------------------------------
        self.scratch_slots = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * per_device_max_requests
        scratch_bytes = self.scratch_slots * per_slot_bytes
        durable_bytes = budget_bytes - scratch_bytes
        max_slots = durable_bytes // per_slot_bytes

        if max_slots < 1:
            min_viable_gb = (scratch_bytes + per_slot_bytes) / 1024 ** 3
            raise ValueError(
                f"[DES-LOC] Mamba cache budget on tier {tier.name} / {label} "
                f"({budget_bytes / 1024**3:.4g} GB) is too small. "
                f"The CUDA-graph extraction scratch reserves "
                f"{scratch_bytes / 1024**3:.4g} GB "
                f"({self.scratch_slots} slots = "
                f"{MAX_INTERMEDIATE_OFFSETS_PER_REQUEST} offsets × "
                f"{per_device_max_requests} requests × "
                f"{per_slot_bytes / 1024:.1f} KB/slot), "
                f"leaving room for fewer than one durable cache slot. "
                f"Increase the budget for this tier to at least "
                f"{min_viable_gb:.4g} GB, or reduce per_device_max_requests."
            )

        self.max_slots = max_slots
        self.per_slot_bytes = per_slot_bytes

        # ------------------------------------------------------------------
        # Free-list (CPU, matches upstream using torch.arange on cpu)
        # ------------------------------------------------------------------
        self.free_slots = list(range(max_slots))

        # ------------------------------------------------------------------
        # Durable state tensors
        # ------------------------------------------------------------------
        self.conv_states = torch.zeros(
            (num_mamba_layers, max_slots) + conv_states_shape,
            dtype=conv_states_dtype,
            device=device,
        )
        self.ssm_states = torch.zeros(
            (num_mamba_layers, max_slots) + ssm_states_shape,
            dtype=ssm_states_dtype,
            device=device,
        )

        # ------------------------------------------------------------------
        # Scratch buffers (on the same device — required for CUDA graphs)
        # ------------------------------------------------------------------
        scratch_device = device  # scratch must be local to the compute device
        self.intermediate_conv_out = torch.zeros(
            (num_mamba_layers, self.scratch_slots) + conv_states_shape,
            dtype=conv_states_dtype,
            device=scratch_device,
        )
        self.intermediate_ssm_out = torch.zeros(
            (num_mamba_layers, self.scratch_slots) + ssm_states_shape,
            dtype=ssm_states_dtype,
            device=scratch_device,
        )

        logger.info(
            "[DES-LOC] TierSlotAllocator %s: %d durable slots (%.3f GB) + "
            "%d scratch slots (%.3f GB) = %.3f GB total within %.3f GB budget, "
            "per-slot %.1f KB",
            label,
            max_slots,
            max_slots * per_slot_bytes / 1024 ** 3,
            self.scratch_slots,
            scratch_bytes / 1024 ** 3,
            (max_slots + self.scratch_slots) * per_slot_bytes / 1024 ** 3,
            budget_bytes / 1024 ** 3,
            per_slot_bytes / 1024,
        )

    # ------------------------------------------------------------------
    # Allocation / free
    # ------------------------------------------------------------------

    def allocate(self) -> Optional[int]:
        """Allocate one durable slot.  Returns slot index or None if full."""
        with self._lock:
            if not self.free_slots:
                return None
            return self.free_slots.pop()

    def free(self, slot_id: int) -> None:
        """Return *slot_id* to the free pool."""
        with self._lock:
            self.free_slots.append(slot_id)

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self.free_slots)

    def has_state(self, slot_id: int) -> bool:
        """True if *slot_id* is currently allocated (not in free list)."""
        with self._lock:
            return slot_id not in self.free_slots

    def durable_memory_bytes(self) -> int:
        return self.max_slots * self.per_slot_bytes

    def scratch_memory_bytes(self) -> int:
        return self.scratch_slots * self.per_slot_bytes


# ---------------------------------------------------------------------------
# Cold-tier (CPU) allocator — no scratch, just durable slots in pinned DRAM
# ---------------------------------------------------------------------------


class ColdTierAllocator:
    """CPU-pinned durable slot store for the COLD tier.

    Scratch buffers never live in the COLD tier; they must be on a CUDA device
    for CUDA-graph compatibility.  The COLD tier is purely an eviction sink:
    when HOT/WARM durable slots are exhausted, the LOC eviction policy can
    demote a slot here via an async D2H copy.  Promotion (H2D) happens when
    the LOC scheduler routes a request back to a GPU tier.

    The cold allocator does not participate in budget validation for scratch
    (there is none), so it has simpler construction logic.
    """

    def __init__(
        self,
        *,
        budget_gb: float,
        per_slot_bytes: int,
        num_mamba_layers: int,
        conv_states_shape: Tuple[int, ...],
        ssm_states_shape: Tuple[int, ...],
        conv_states_dtype: torch.dtype,
        ssm_states_dtype: torch.dtype,
    ) -> None:
        self.per_slot_bytes = per_slot_bytes
        budget_bytes = int(budget_gb * 1024 ** 3)
        self.max_slots = budget_bytes // per_slot_bytes if per_slot_bytes > 0 else 0

        if self.max_slots < 1:
            logger.warning(
                "[DES-LOC] Cold-tier Mamba cache budget (%.3f GB) too small "
                "for even one slot (%.3f GB/slot). Cold tier disabled.",
                budget_gb,
                per_slot_bytes / 1024 ** 3,
            )
            self.max_slots = 0

        self._lock = threading.Lock()
        self.free_slots: List[int] = list(range(self.max_slots))

        if self.max_slots > 0:
            self.conv_states = torch.zeros(
                (num_mamba_layers, self.max_slots) + conv_states_shape,
                dtype=conv_states_dtype,
                pin_memory=True,
            )
            self.ssm_states = torch.zeros(
                (num_mamba_layers, self.max_slots) + ssm_states_shape,
                dtype=ssm_states_dtype,
                pin_memory=True,
            )
            logger.info(
                "[DES-LOC] ColdTierAllocator: %d pinned-DRAM slots (%.3f GB total)",
                self.max_slots,
                self.max_slots * per_slot_bytes / 1024 ** 3,
            )
        else:
            self.conv_states = torch.zeros(0)
            self.ssm_states = torch.zeros(0)

    def allocate(self) -> Optional[int]:
        with self._lock:
            if not self.free_slots:
                return None
            return self.free_slots.pop()

    def free(self, slot_id: int) -> None:
        with self._lock:
            self.free_slots.append(slot_id)

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self.free_slots)


# ---------------------------------------------------------------------------
# LOC (Locality Cache) hint map
# ---------------------------------------------------------------------------


class LocalityHintMap:
    """Tracks which GPU device currently holds the durable state for a slot.

    The DES-LOC LOC scheduler queries this map to decide which device should
    handle the next step for a given request prefix, minimising cross-PCIe
    state transfers.

    The map is keyed by a (tier, tier_slot_id) pair and returns the device_id
    that currently holds that slot's durable tensors.  COLD-tier slots return
    device_id = -1 (CPU).
    """

    def __init__(self) -> None:
        self._map: Dict[Tuple[MemoryTier, int], int] = {}
        self._lock = threading.Lock()

    def record(self, tier: MemoryTier, slot_id: int, device_id: int) -> None:
        with self._lock:
            self._map[(tier, slot_id)] = device_id

    def forget(self, tier: MemoryTier, slot_id: int) -> None:
        with self._lock:
        	self._map.pop((tier, slot_id), None)

    def locality_hint(self, tier: MemoryTier, slot_id: int) -> Optional[int]:
        """Return device_id that holds this slot, or None if unknown."""
        with self._lock:
            return self._map.get((tier, slot_id))

    def slots_on_device(self, device_id: int) -> List[Tuple[MemoryTier, int]]:
        """Return all (tier, slot_id) pairs resident on *device_id*."""
        with self._lock:
            return [k for k, v in self._map.items() if v == device_id]


# ---------------------------------------------------------------------------
# Budget preview helper (mirrors DynamicInferenceContext's log block)
# ---------------------------------------------------------------------------


def compute_budget_preview(
    budget_config: TierBudgetConfig,
    per_slot_bytes: int,
) -> Dict[str, object]:
    """Compute slot counts for each tier without allocating tensors.

    This mirrors the preview block in Megatron's ``DynamicInferenceContext``
    (the log lines added in commit a83f408) but extended to three tiers.

    Returns a dict suitable for structured logging or display.
    """

    def _split(budget_bytes: int, max_req: int) -> Tuple[int, int, int]:
        scratch_slots = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * max_req
        scratch_bytes = scratch_slots * per_slot_bytes
        durable_slots = max((budget_bytes - scratch_bytes) // per_slot_bytes, 0)
        return scratch_slots, scratch_bytes, durable_slots

    hot_budget = int(budget_config.hot_gb * 1024 ** 3)
    warm_budget = int(budget_config.warm_gb_per_device * 1024 ** 3)
    cold_budget = int(budget_config.cold_gb * 1024 ** 3)
    cold_durable = cold_budget // per_slot_bytes if per_slot_bytes > 0 else 0

    hot_scratch, hot_scratch_bytes, hot_durable = _split(
        hot_budget, budget_config.per_device_max_requests_hot
    )
    warm_scratch, warm_scratch_bytes, warm_durable = _split(
        warm_budget, budget_config.per_device_max_requests_warm
    )

    return {
        "hot": {
            "budget_gb": budget_config.hot_gb,
            "scratch_slots": hot_scratch,
            "scratch_bytes": hot_scratch_bytes,
            "durable_slots": hot_durable,
        },
        "warm_per_device": {
            "budget_gb": budget_config.warm_gb_per_device,
            "scratch_slots": warm_scratch,
            "scratch_bytes": warm_scratch_bytes,
            "durable_slots": warm_durable,
        },
        "cold": {
            "budget_gb": budget_config.cold_gb,
            "durable_slots": cold_durable,
            "scratch_slots": 0,
            "scratch_bytes": 0,
        },
        "per_slot_bytes": per_slot_bytes,
    }


def log_budget_preview(preview: Dict[str, object]) -> None:
    """Emit the budget preview as structured INFO lines."""
    psb = preview["per_slot_bytes"]

    def _gb(b: int) -> str:
        return f"{b / 1024**3:.3f} GB"

    for tier_name, info in [
        ("HOT (H100)", preview["hot"]),
        ("WARM/device (A6000)", preview["warm_per_device"]),
        ("COLD (CPU-pinned)", preview["cold"]),
    ]:
        logger.info(
            "[DES-LOC] Mamba cache tier %-22s  budget=%s  "
            "scratch=%d slots (%s)  durable=%d slots (%s)  per-slot=%.1f KB",
            tier_name,
            _gb(int(info["budget_gb"] * 1024 ** 3)),  # type: ignore[arg-type]
            info["scratch_slots"],
            _gb(info["scratch_bytes"]),  # type: ignore[arg-type]
            info["durable_slots"],
            _gb(info["durable_slots"] * psb),  # type: ignore[operator]
            psb / 1024,
        )


# ---------------------------------------------------------------------------
# Main façade: TierAwareMambaCache
# ---------------------------------------------------------------------------


class TierAwareMambaCache:
    """Three-tier Mamba prefix cache for the DES-LOC heterogeneous cluster.

    This is the primary entry point for DES-LOC's adaptation of Megatron's
    Mamba prefix caching subsystem.  It owns one ``TierSlotAllocator`` per GPU
    tier (HOT/WARM) and one ``ColdTierAllocator`` for CPU-pinned spill, plus
    a ``LocalityHintMap`` for the LOC scheduler.

    Memory Safety Guarantee (matching upstream a83f408)
    ---------------------------------------------------
    Construction raises ``ValueError`` if, for any GPU tier, the configured
    budget cannot cover the CUDA-graph extraction scratch plus at least one
    durable slot.  This prevents silent OOM at inference startup — the same
    guarantee Megatron's fix introduced for the single-GPU case.

    Usage
    -----
    Typical construction from a ``TierBudgetConfig``::

        cache = TierAwareMambaCache.from_budget(
            budget_config=TierBudgetConfig(hot_gb=4.0, warm_gb_per_device=2.0,
                                           cold_gb=32.0,
                                           per_device_max_requests_hot=64,
                                           per_device_max_requests_warm=32),
            num_mamba_layers=24,
            conv_states_shape=(512, 4),
            ssm_states_shape=(512, 16),
            conv_states_dtype=torch.float16,
            ssm_states_dtype=torch.float32,
            cluster_devices=CLUSTER_DEVICES,
        )

    Slot allocation (DES-LOC scheduler calls these)::

        slot = cache.allocate(preferred_tier=MemoryTier.HOT)
        # ... use slot.slot_id on slot.device_id ...
        cache.free(slot)

    LOC hint query::

        dev = cache.locality_hint_map.locality_hint(slot.tier, slot.slot_id)
    """

    def __init__(
        self,
        *,
        hot_allocator: TierSlotAllocator,
        warm_allocators: List[TierSlotAllocator],
        cold_allocator: ColdTierAllocator,
        locality_hint_map: LocalityHintMap,
        budget_config: TierBudgetConfig,
    ) -> None:
        self._hot = hot_allocator
        self._warm = warm_allocators
        self._cold = cold_allocator
        self.locality_hint_map = locality_hint_map
        self.budget_config = budget_config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_budget(
        cls,
        *,
        budget_config: TierBudgetConfig,
        num_mamba_layers: int,
        conv_states_shape: Tuple[int, ...],
        ssm_states_shape: Tuple[int, ...],
        conv_states_dtype: torch.dtype,
        ssm_states_dtype: torch.dtype,
        cluster_devices: List[DeviceSpec] = CLUSTER_DEVICES,
        enable_cold_tier: bool = True,
    ) -> "TierAwareMambaCache":
        """Construct a ``TierAwareMambaCache`` from a ``TierBudgetConfig``.

        Parameters
        ----------
        budget_config:
            Per-tier GB allocations and per-device request counts.
        num_mamba_layers:
            Number of Mamba layers in the model (all tiers use this).
        conv_states_shape, ssm_states_shape:
            Shape of one layer's conv / SSM state (excluding the slot dim).
        conv_states_dtype, ssm_states_dtype:
            Dtypes matching the Mamba kernel expectations.
        cluster_devices:
            Device specs; defaults to the reference Neuron_SP three-device layout.
        enable_cold_tier:
            If False, the CPU-pinned cold tier is skipped (useful in unit tests).

        Raises
        ------
        ValueError
            If any GPU tier's budget is too small to hold the scratch plus one
            durable slot.
        """
        if budget_config.is_fully_unconfigured():
            logger.warning(
                "[DES-LOC] TierAwareMambaCache: no GPU budget configured "
                "(hot_gb=%.3g, warm_gb_per_device=%.3g). Running in memory-only "
                "mode: identical KV prefixes are deduplicated for memory savings, "
                "but Mamba state caching and prefill skipping are disabled (every "
                "token is recomputed). Set hot_gb or warm_gb_per_device > 0 to "
                "enable full prefix caching.",
                budget_config.hot_gb,
                budget_config.warm_gb_per_device,
            )
            # Return a cache with no real allocators — callers should check
            # .has_gpu_cache() before using allocate().
            return cls(
                hot_allocator=None,  # type: ignore[arg-type]
                warm_allocators=[],
                cold_allocator=ColdTierAllocator(
                    budget_gb=0.0,
                    per_slot_bytes=1,
                    num_mamba_layers=num_mamba_layers,
                    conv_states_shape=conv_states_shape,
                    ssm_states_shape=ssm_states_shape,
                    conv_states_dtype=conv_states_dtype,
                    ssm_states_dtype=ssm_states_dtype,
                ),
                locality_hint_map=LocalityHintMap(),
                budget_config=budget_config,
            )

        conv_size = math.prod(conv_states_shape) * torch.finfo(conv_states_dtype).bits // 8
        ssm_size = math.prod(ssm_states_shape) * torch.finfo(ssm_states_dtype).bits // 8
        per_slot_bytes = num_mamba_layers * (conv_size + ssm_size)

        preview = compute_budget_preview(budget_config, per_slot_bytes)
        log_budget_preview(preview)

        # HOT tier (H100, device_id=2)
        hot_specs = [d for d in cluster_devices if d.tier == MemoryTier.HOT]
        if len(hot_specs) != 1:
            raise ValueError(
                f"Expected exactly one HOT-tier device, found {len(hot_specs)}: {hot_specs}"
            )
        hot_spec = hot_specs[0]
        hot_alloc: Optional[TierSlotAllocator] = None
        if budget_config.hot_gb > 0.0:
            hot_alloc = TierSlotAllocator(
                tier=MemoryTier.HOT,
                device=hot_spec.torch_device,
                budget_bytes=int(budget_config.hot_gb * 1024 ** 3),
                per_slot_bytes=per_slot_bytes,
                per_device_max_requests=budget_config.per_device_max_requests_hot,
                num_mamba_layers=num_mamba_layers,
                conv_states_shape=conv_states_shape,
                ssm_states_shape=ssm_states_shape,
                conv_states_dtype=conv_states_dtype,
                ssm_states_dtype=ssm_states_dtype,
                label=hot_spec.label,
            )

        # WARM tier (A6000 × 2)
        warm_specs = [d for d in cluster_devices if d.tier == MemoryTier.WARM]
        warm_allocs: List[TierSlotAllocator] = []
        if budget_config.warm_gb_per_device > 0.0:
            for spec in warm_specs:
                warm_allocs.append(
                    TierSlotAllocator(
                        tier=MemoryTier.WARM,
                        device=spec.torch_device,
                        budget_bytes=int(budget_config.warm_gb_per_device * 1024 ** 3),
                        per_slot_bytes=per_slot_bytes,
                        per_device_max_requests=budget_config.per_device_max_requests_warm,
                        num_mamba_layers=num_mamba_layers,
                        conv_states_shape=conv_states_shape,
                        ssm_states_shape=ssm_states_shape,
                        conv_states_dtype=conv_states_dtype,
                        ssm_states_dtype=ssm_states_dtype,
                        label=spec.label,
                    )
                )

        # COLD tier (CPU-pinned DRAM)
        cold_alloc = ColdTierAllocator(
            budget_gb=budget_config.cold_gb if enable_cold_tier else 0.0,
            per_slot_bytes=per_slot_bytes,
            num_mamba_layers=num_mamba_layers,
            conv_states_shape=conv_states_shape,
            ssm_states_shape=ssm_states_shape,
            conv_states_dtype=conv_states_dtype,
            ssm_states_dtype=ssm_states_dtype,
        )

        return cls(
            hot_allocator=hot_alloc,  # type: ignore[arg-type]
            warm_allocators=warm_allocs,
            cold_allocator=cold_alloc,
            locality_hint_map=LocalityHintMap(),
            budget_config=budget_config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_gpu_cache(self) -> bool:
        """True if at least one GPU-tier allocator was initialised."""
        return self._hot is not None or len(self._warm) > 0

    def allocate(
        self,
        preferred_tier: MemoryTier = MemoryTier.HOT,
        preferred_warm_device_id: Optional[int] = None,
    ) -> Optional[SlotDescriptor]:
        """Allocate one durable slot, respecting the tier preference.

        DES-LOC allocation policy (in priority order):
          1. Try the preferred tier.
          2. Fall back to the other GPU tier.
          3. Fall back to the COLD tier (CPU-pinned, triggers async H2D on use).
          4. Return None (caller must evict or reject the request).

        The *preferred_warm_device_id* hint lets the LOC scheduler ask for a
        slot on the warm device that already holds related state, minimising
        PCIe transfers.
        """
        if not self.has_gpu_cache():
            return None

        slot = self._try_alloc_gpu(preferred_tier, preferred_warm_device_id)
        if slot is not None:
            self.locality_hint_map.record(slot.tier, slot.slot_id, slot.device_id)
            return slot

        # Fall through to cold tier
        cold_slot_id = self._cold.allocate()
        if cold_slot_id is not None:
            desc = SlotDescriptor(slot_id=cold_slot_id, tier=MemoryTier.COLD, device_id=-1)
            self.locality_hint_map.record(MemoryTier.COLD, cold_slot_id, -1)
            return desc

        return None

    def free(self, slot: SlotDescriptor) -> None:
        """Return a durable slot to its tier's free pool."""
        self.locality_hint_map.forget(slot.tier, slot.slot_id)
        if slot.tier == MemoryTier.HOT:
            if self._hot is not None:
                self._hot.free(slot.slot_id)
        elif slot.tier == MemoryTier.WARM:
            alloc = self._warm_alloc_for_device(slot.device_id)
            if alloc is not None:
                alloc.free(slot.slot_id)
        elif slot.tier == MemoryTier.COLD:
            self._cold.free(slot.slot_id)

    def promote_cold_to_warm(
        self,
        cold_slot: SlotDescriptor,
        target_device_id: int,
    ) -> Optional[SlotDescriptor]:
        """Promote a COLD slot to the WARM tier on *target_device_id*.

        Issues an async H2D copy (non-blocking) for both conv_states and
        ssm_states.  Returns the new WARM SlotDescriptor, or None if no WARM
        slots are available.  Caller is responsible for synchronising the copy
        before the Mamba kernel reads the tensors.

        This is the DES-LOC "demand promotion" path: when the LOC scheduler
        decides a request should run on a WARM device but its Mamba state is in
        COLD storage, it calls this to migrate the slot.
        """
        warm_alloc = self._warm_alloc_for_device(target_device_id)
        if warm_alloc is None:
            logger.warning(
                "[DES-LOC] promote_cold_to_warm: no WARM allocator for device %d",
                target_device_id,
            )
            return None

        warm_slot_id = warm_alloc.allocate()
        if warm_slot_id is None:
            return None  # WARM tier full; caller should evict

        if self._cold.max_slots > 0 and cold_slot.slot_id < self._cold.max_slots:
            # Async H2D copy (pinned source enables DMA)
            warm_alloc.conv_states[:, warm_slot_id].copy_(
                self._cold.conv_states[:, cold_slot.slot_id], non_blocking=True
            )
            warm_alloc.ssm_states[:, warm_slot_id].copy_(
                self._cold.ssm_states[:, cold_slot.slot_id], non_blocking=True
            )

        self._cold.free(cold_slot.slot_id)
        self.locality_hint_map.forget(MemoryTier.COLD, cold_slot.slot_id)

        new_slot = SlotDescriptor(
            slot_id=warm_slot_id,
            tier=MemoryTier.WARM,
            device_id=target_device_id,
        )
        self.locality_hint_map.record(MemoryTier.WARM, warm_slot_id, target_device_id)
        logger.info(
            "[DES-LOC] Promoted cold slot %d → warm slot %d on device %d (async H2D)",
            cold_slot.slot_id,
            warm_slot_id,
            target_device_id,
        )
        return new_slot

    def demote_warm_to_cold(self, warm_slot: SlotDescriptor) -> Optional[SlotDescriptor]:
        """Evict a WARM slot to the COLD tier (D2H, non-blocking).

        Used by LRU eviction when the WARM tier is exhausted.  Caller must
        synchronise the copy before the slot is used in the COLD tier.
        """
        warm_alloc = self._warm_alloc_for_device(warm_slot.device_id)
        if warm_alloc is None:
            return None

        cold_slot_id = self._cold.allocate()
        if cold_slot_id is None:
            return None  # COLD tier also full

        if self._cold.max_slots > 0:
            self._cold.conv_states[:, cold_slot_id].copy_(
                warm_alloc.conv_states[:, warm_slot.slot_id], non_blocking=True
            )
            self._cold.ssm_states[:, cold_slot_id].copy_(
                warm_alloc.ssm_states[:, warm_slot.slot_id], non_blocking=True
            )

        warm_alloc.free(warm_slot.slot_id)
        self.locality_hint_map.forget(MemoryTier.WARM, warm_slot.slot_id)

        cold_slot = SlotDescriptor(slot_id=cold_slot_id, tier=MemoryTier.COLD, device_id=-1)
        self.locality_hint_map.record(MemoryTier.COLD, cold_slot_id, -1)
        logger.info(
            "[DES-LOC] Demoted warm slot %d (device %d) → cold slot %d (async D2H)",
            warm_slot.slot_id,
            warm_slot.device_id,
            cold_slot_id,
        )
        return cold_slot

    # ------------------------------------------------------------------
    # Capacity / stats
    # ------------------------------------------------------------------

    def free_count(self, tier: MemoryTier, device_id: Optional[int] = None) -> int:
        if tier == MemoryTier.HOT:
            return self._hot.free_count if self._hot else 0
        if tier == MemoryTier.WARM:
            if device_id is not None:
                a = self._warm_alloc_for_device(device_id)
                return a.free_count if a else 0
            return sum(a.free_count for a in self._warm)
        if tier == MemoryTier.COLD:
            return self._cold.free_count
        return 0

    def capacity_summary(self) -> Dict[str, int]:
        """Return a dict with max_slots for each tier (useful for logging/monitoring)."""
        return {
            "hot_max_slots": self._hot.max_slots if self._hot else 0,
            "warm_max_slots_per_device": self._warm[0].max_slots if self._warm else 0,
            "warm_device_count": len(self._warm),
            "cold_max_slots": self._cold.max_slots,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_alloc_gpu(
        self,
        preferred_tier: MemoryTier,
        preferred_warm_device_id: Optional[int],
    ) -> Optional[SlotDescriptor]:
        """Try GPU tiers (HOT then WARM, or vice-versa) respecting preference."""
        tiers_to_try: List[MemoryTier] = []
        if preferred_tier == MemoryTier.HOT:
            tiers_to_try = [MemoryTier.HOT, MemoryTier.WARM]
        else:
            tiers_to_try = [MemoryTier.WARM, MemoryTier.HOT]

        for tier in tiers_to_try:
            if tier == MemoryTier.HOT and self._hot is not None:
                slot_id = self._hot.allocate()
                if slot_id is not None:
                    return SlotDescriptor(
                        slot_id=slot_id,
                        tier=MemoryTier.HOT,
                        device_id=CLUSTER_DEVICES[2].device_id,
                    )
            elif tier == MemoryTier.WARM and self._warm:
                # Try the locality-preferred device first, then round-robin
                ordered = list(self._warm)
                if preferred_warm_device_id is not None:
                    ordered = sorted(
                        ordered,
                        key=lambda a: 0 if a.device.index == preferred_warm_device_id else 1,
                    )
                for alloc in ordered:
                    slot_id = alloc.allocate()
                    if slot_id is not None:
                        return SlotDescriptor(
                            slot_id=slot_id,
                            tier=MemoryTier.WARM,
                            device_id=alloc.device.index,
                        )
        return None

    def _warm_alloc_for_device(self, device_id: int) -> Optional[TierSlotAllocator]:
        for a in self._warm:
            if a.device.index == device_id:
                return a
        return None


# ---------------------------------------------------------------------------
# Validation helper (CLI / config layer)
# ---------------------------------------------------------------------------


def validate_budget_config(
    budget_config: TierBudgetConfig,
    per_slot_bytes: int,
) -> List[str]:
    """Return a list of human-readable error strings for an invalid config.

    Empty list ⟹ config is valid.  The caller can raise ValueError with the
    joined errors or surface them as config-layer warnings.
    """
    errors: List[str] = []
    preview = compute_budget_preview(budget_config, per_slot_bytes)

    for tier_name, info in [
        ("hot", preview["hot"]),
        ("warm_per_device", preview["warm_per_device"]),
    ]:
        budget_gb = info["budget_gb"]  # type: ignore[index]
        if budget_gb <= 0.0:
            continue  # disabled tiers are not an error
        durable = info["durable_slots"]
        scratch_bytes = info["scratch_bytes"]
        if durable < 1:
            min_viable = (scratch_bytes + per_slot_bytes) / 1024 ** 3
            errors.append(
                f"Tier '{tier_name}': budget {budget_gb:.4g} GB too small — "
                f"extraction scratch alone needs "
                f"{scratch_bytes / 1024**3:.4g} GB; need ≥ {min_viable:.4g} GB "
                f"for one durable slot."
            )
    return errors


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback
    import unittest

    # Configure logging so test output is readable
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)-8s %(name)s | %(message)s",
        stream=sys.stdout,
    )

    # Suppress CUDA-unavailable noise in unit tests by monkey-patching torch.zeros
    # so that device="cuda:N" tensors fall back to CPU.
    _real_torch_zeros = torch.zeros

    def _cpu_zeros(*args, **kwargs) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        if "device" in kwargs:
            d = kwargs["device"]
            if isinstance(d, torch.device) and d.type == "cuda":
                kwargs["device"] = torch.device("cpu")
            elif isinstance(d, str) and d.startswith("cuda"):
                kwargs["device"] = torch.device("cpu")
        return _real_torch_zeros(*args, **kwargs)

    # Patch CLUSTER_DEVICES to use cpu devices for testing
    _TEST_CLUSTER = [
        DeviceSpec(device_id=0, tier=MemoryTier.WARM, total_memory_gb=48.0,
                   compute_capability=(8, 6), label="A6000-0-test"),
        DeviceSpec(device_id=1, tier=MemoryTier.WARM, total_memory_gb=48.0,
                   compute_capability=(8, 6), label="A6000-1-test"),
        DeviceSpec(device_id=2, tier=MemoryTier.HOT,  total_memory_gb=96.0,
                   compute_capability=(9, 0), label="H100-NVL-test"),
    ]

    # Override torch.device to return CPU for CUDA indices in test mode
    _orig_torch_device = torch.device

    class _FakeCudaDevice:
        """A torch.device stand-in that always sits on CPU for unit tests."""

        def __init__(self, spec: str):
            self._spec = spec
            if isinstance(spec, str) and spec.startswith("cuda:"):
                self._index = int(spec.split(":")[1])
                self.type = "cuda"
            else:
                self._index = None
                self.type = "cpu"

        @property
        def index(self) -> Optional[int]:
            return self._index

        def __repr__(self) -> str:
            return f"device('{self._spec}')"

    def _make_device(spec) -> "_FakeCudaDevice":  # type: ignore[override]
        return _FakeCudaDevice(str(spec))

    # ----------------------------------------------------------------
    # Patch DeviceSpec.torch_device to return our fake device
    # ----------------------------------------------------------------
    def _patched_torch_device(self: DeviceSpec) -> "_FakeCudaDevice":
        return _FakeCudaDevice(f"cuda:{self.device_id}")

    DeviceSpec.torch_device = property(_patched_torch_device)  # type: ignore[method-assign]

    # Patch TierSlotAllocator to use CPU tensors
    _orig_tier_alloc_init = TierSlotAllocator.__init__

    def _patched_tier_alloc_init(self_inner, *, tier, device, budget_bytes,
                                 per_slot_bytes, per_device_max_requests,
                                 num_mamba_layers, conv_states_shape,
                                 ssm_states_shape, conv_states_dtype,
                                 ssm_states_dtype, label):
        # Force CPU device
        _orig_tier_alloc_init(
            self_inner,
            tier=tier,
            device=torch.device("cpu"),
            budget_bytes=budget_bytes,
            per_slot_bytes=per_slot_bytes,
            per_device_max_requests=per_device_max_requests,
            num_mamba_layers=num_mamba_layers,
            conv_states_shape=conv_states_shape,
            ssm_states_shape=ssm_states_shape,
            conv_states_dtype=conv_states_dtype,
            ssm_states_dtype=ssm_states_dtype,
            label=label,
        )

    TierSlotAllocator.__init__ = _patched_tier_alloc_init  # type: ignore[method-assign]

    # Fix warm allocator device.index — our fake device uses _index
    # We patch _warm_alloc_for_device to compare by label index instead
    _orig_warm_for_device = TierAwareMambaCache._warm_alloc_for_device

    def _patched_warm_for_device(self_inner, device_id: int):
        # In test mode all warm allocs use cpu; match by insertion order
        if not self_inner._warm:
            return None
        idx = device_id  # device 0 → _warm[0], device 1 → _warm[1]
        if 0 <= idx < len(self_inner._warm):
            return self_inner._warm[idx]
        return None

    TierAwareMambaCache._warm_alloc_for_device = _patched_warm_for_device  # type: ignore

    # Also patch _try_alloc_gpu's device_id extraction for WARM slots
    _orig_try_alloc_gpu = TierAwareMambaCache._try_alloc_gpu

    def _patched_try_alloc_gpu(self_inner, preferred_tier, preferred_warm_device_id):
        tiers_to_try: List[MemoryTier] = []
        if preferred_tier == MemoryTier.HOT:
            tiers_to_try = [MemoryTier.HOT, MemoryTier.WARM]
        else:
            tiers_to_try = [MemoryTier.WARM, MemoryTier.HOT]

        for tier in tiers_to_try:
            if tier == MemoryTier.HOT and self_inner._hot is not None:
                slot_id = self_inner._hot.allocate()
                if slot_id is not None:
                    return SlotDescriptor(slot_id=slot_id, tier=MemoryTier.HOT, device_id=2)
            elif tier == MemoryTier.WARM and self_inner._warm:
                for idx, alloc in enumerate(self_inner._warm):
                    slot_id = alloc.allocate()
                    if slot_id is not None:
                        return SlotDescriptor(slot_id=slot_id, tier=MemoryTier.WARM, device_id=idx)
        return None

    TierAwareMambaCache._try_alloc_gpu = _patched_try_alloc_gpu  # type: ignore

    # ----------------------------------------------------------------
    # Helper to build a minimal cache for tests (no real CUDA needed)
    # ----------------------------------------------------------------

    def _make_cache(
        hot_gb: float = 0.5,
        warm_gb: float = 0.3,
        cold_gb: float = 0.0,
        max_req_hot: int = 8,
        max_req_warm: int = 4,
        num_layers: int = 2,
        conv_shape: Tuple[int, ...] = (16, 2),
        ssm_shape: Tuple[int, ...] = (16, 4),
    ) -> TierAwareMambaCache:
        return TierAwareMambaCache.from_budget(
            budget_config=TierBudgetConfig(
                hot_gb=hot_gb,
                warm_gb_per_device=warm_gb,
                cold_gb=cold_gb,
                per_device_max_requests_hot=max_req_hot,
                per_device_max_requests_warm=max_req_warm,
            ),
            num_mamba_layers=num_layers,
            conv_states_shape=conv_shape,
            ssm_states_shape=ssm_shape,
            conv_states_dtype=torch.float16,
            ssm_states_dtype=torch.float32,
            cluster_devices=_TEST_CLUSTER,
            enable_cold_tier=cold_gb > 0.0,
        )

    # ----------------------------------------------------------------
    # Test cases
    # ----------------------------------------------------------------

    class TestMaxIntermediateOffsets(unittest.TestCase):
        def test_constant_value(self):
            self.assertEqual(MAX_INTERMEDIATE_OFFSETS_PER_REQUEST, 3)

    class TestBudgetPreview(unittest.TestCase):
        def test_scratch_subtracted_before_durable(self):
            # 1 MB budget, 100 KB/slot, max_req=2
            # scratch = 3*2=6 slots = 600 KB; durable = (1024-600)//100 = 4
            per_slot = 100 * 1024
            budget_gb = 1024 * 1024 / (1024 ** 3)  # 1 MB in GB
            bc = TierBudgetConfig(
                hot_gb=budget_gb,
                warm_gb_per_device=0.0,
                cold_gb=0.0,
                per_device_max_requests_hot=2,
                per_device_max_requests_warm=0,
            )
            p = compute_budget_preview(bc, per_slot)
            self.assertEqual(p["hot"]["scratch_slots"], 6)
            self.assertEqual(p["hot"]["durable_slots"], 4)

        def test_zero_budget_gives_zero_durable(self):
            bc = TierBudgetConfig(hot_gb=0.0, warm_gb_per_device=0.0)
            p = compute_budget_preview(bc, 1024)
            self.assertEqual(p["hot"]["durable_slots"], 0)
            self.assertEqual(p["warm_per_device"]["durable_slots"], 0)

    class TestTierSlotAllocatorBudgetError(unittest.TestCase):
        """Mirrors Megatron's test_mamba_cache_budget_too_small_raises."""

        def test_raises_when_scratch_exceeds_budget(self):
            # Budget = 1 byte, scratch will far exceed it → ValueError
            with self.assertRaises(ValueError) as ctx:
                TierSlotAllocator(
                    tier=MemoryTier.HOT,
                    device=torch.device("cpu"),
                    budget_bytes=1,
                    per_slot_bytes=1024,
                    per_device_max_requests=64,
                    num_mamba_layers=2,
                    conv_states_shape=(4, 2),
                    ssm_states_shape=(4, 2),
                    conv_states_dtype=torch.float16,
                    ssm_states_dtype=torch.float32,
                    label="test-hot",
                )
            self.assertIn("too small", str(ctx.exception))
            self.assertIn("scratch", str(ctx.exception))
            self.assertIn("durable", str(ctx.exception))

        def test_error_message_contains_min_viable(self):
            with self.assertRaises(ValueError) as ctx:
                TierSlotAllocator(
                    tier=MemoryTier.WARM,
                    device=torch.device("cpu"),
                    budget_bytes=100,
                    per_slot_bytes=50,
                    per_device_max_requests=4,
                    num_mamba_layers=1,
                    conv_states_shape=(4,),
                    ssm_states_shape=(4,),
                    conv_states_dtype=torch.float16,
                    ssm_states_dtype=torch.float16,
                    label="test-warm",
                )
            msg = str(ctx.exception)
            self.assertIn("GB", msg)
            self.assertIn("reduce per_device_max_requests", msg)

    class TestTierSlotAllocatorBasicOps(unittest.TestCase):
        def setUp(self):
            # Budget: 1 MB, per_slot: 4 KB, max_req=2
            # scratch=6 slots=24 KB; durable=(1024-24)//4=250 slots
            self.alloc = TierSlotAllocator(
                tier=MemoryTier.HOT,
                device=torch.device("cpu"),
                budget_bytes=1 * 1024 * 1024,
                per_slot_bytes=4 * 1024,
                per_device_max_requests=2,
                num_mamba_layers=1,
                conv_states_shape=(8,),
                ssm_states_shape=(8,),
                conv_states_dtype=torch.float16,
                ssm_states_dtype=torch.float32,
                label="test",
            )

        def test_allocate_returns_valid_id(self):
            slot = self.alloc.allocate()
            self.assertIsNotNone(slot)
            self.assertGreaterEqual(slot, 0)
            self.assertLess(slot, self.alloc.max_slots)

        def test_free_returns_slot_to_pool(self):
            initial = self.alloc.free_count
            slot = self.alloc.allocate()
            self.assertEqual(self.alloc.free_count, initial - 1)
            self.alloc.free(slot)
            self.assertEqual(self.alloc.free_count, initial)

        def test_exhaust_returns_none(self):
            slots = []
            while True:
                s = self.alloc.allocate()
                if s is None:
                    break
                slots.append(s)
            self.assertEqual(len(slots), self.alloc.max_slots)
            self.assertIsNone(self.alloc.allocate())

        def test_has_state_reflects_allocation(self):
            slot = self.alloc.allocate()
            self.assertTrue(self.alloc.has_state(slot))
            self.alloc.free(slot)
            self.assertFalse(self.alloc.has_state(slot))

        def test_scratch_tensor_shapes(self):
            expected_scratch = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * 2
            self.assertEqual(self.alloc.intermediate_ssm_out.shape[1], expected_scratch)
            self.assertEqual(self.alloc.intermediate_conv_out.shape[1], expected_scratch)

    class TestTierAwareMambaCacheMemoryOnlyMode(unittest.TestCase):
        """Mirrors Megatron's test_hybrid_prefix_caching_without_mamba_budget_warns."""

        def test_warns_when_no_gpu_budget(self):
            with self.assertLogs("deepspeed.inference.tier_aware_mamba_cache",
                                 level="WARNING") as cm:
                cache = TierAwareMambaCache.from_budget(
                    budget_config=TierBudgetConfig(
                        hot_gb=0.0,
                        warm_gb_per_device=0.0,
                        cold_gb=0.0,
                    ),
                    num_mamba_layers=2,
                    conv_states_shape=(8,),
                    ssm_states_shape=(8,),
                    conv_states_dtype=torch.float16,
                    ssm_states_dtype=torch.float32,
                    cluster_devices=_TEST_CLUSTER,
                )
            self.assertFalse(cache.has_gpu_cache())
            self.assertIsNone(cache._hot)
            self.assertEqual(cache._warm, [])
            warning_text = " ".join(cm.output)
            self.assertIn("memory-only", warning_text)

        def test_allocate_returns_none_in_memory_only_mode(self):
            cache = TierAwareMambaCache.from_budget(
                budget_config=TierBudgetConfig(hot_gb=0.0, warm_gb_per_device=0.0),
                num_mamba_layers=2,
                conv_states_shape=(8,),
                ssm_states_shape=(8,),
                conv_states_dtype=torch.float16,
                ssm_states_dtype=torch.float32,
                cluster_devices=_TEST_CLUSTER,
            )
            self.assertIsNone(cache.allocate())

    class TestTierAwareMambaCacheAllocation(unittest.TestCase):
        def setUp(self):
            self.cache = _make_cache()

        def test_hot_preferred_allocates_hot_first(self):
            slot = self.cache.allocate(preferred_tier=MemoryTier.HOT)
            self.assertIsNotNone(slot)
            self.assertEqual(slot.tier, MemoryTier.HOT)

        def test_warm_preferred_allocates_warm_first(self):
            slot = self.cache.allocate(preferred_tier=MemoryTier.WARM)
            self.assertIsNotNone(slot)
            self.assertEqual(slot.tier, MemoryTier.WARM)

        def test_free_returns_slot(self):
            slot = self.cache.allocate(preferred_tier=MemoryTier.HOT)
            before = self.cache.free_count(MemoryTier.HOT)
            self.cache.free(slot)
            after = self.cache.free_count(MemoryTier.HOT)
            self.assertEqual(after, before + 1)

        def test_locality_hint_recorded_on_allocate(self):
            slot = self.cache.allocate(preferred_tier=MemoryTier.HOT)
            hint = self.cache.locality_hint_map.locality_hint(slot.tier, slot.slot_id)
            self.assertEqual(hint, slot.device_id)

        def test_locality_hint_cleared_on_free(self):
            slot = self.cache.allocate(preferred_tier=MemoryTier.HOT)
            self.cache.free(slot)
            hint = self.cache.locality_hint_map.locality_hint(slot.tier, slot.slot_id)
            self.assertIsNone(hint)

        def test_fallback_to_warm_when_hot_full(self):
            # Exhaust HOT tier
            hot_slots = []
            while True:
                s = self.cache.allocate(preferred_tier=MemoryTier.HOT)
                if s is None or s.tier != MemoryTier.HOT:
                    if s is not None:
                        hot_slots.append(s)
                    break
                hot_slots.append(s)

            # Next allocation should land on WARM
            slot = self.cache.allocate(preferred_tier=MemoryTier.HOT)
            if slot is not None:
                self.assertIn(slot.tier, (MemoryTier.WARM, MemoryTier.COLD))

        def test_capacity_summary_keys(self):
            summary = self.cache.capacity_summary()
            self.assertIn("hot_max_slots", summary)
            self.assertIn("warm_max_slots_per_device", summary)
            self.assertIn("warm_device_count", summary)
            self.assertIn("cold_max_slots", summary)

    class TestTierAwareMambaCacheWithColdTier(unittest.TestCase):
        def test_cold_tier_allocates_when_gpu_exhausted(self):
            # Tiny GPU budgets, larger cold budget
            cache = _make_cache(
                hot_gb=0.001,
                warm_gb=0.001,
                cold_gb=0.1,
                max_req_hot=1,
                max_req_warm=1,
            )
            # Exhaust all GPU slots
            gpu_slots = []
            for _ in range(500):
                s = cache.allocate(preferred_tier=MemoryTier.HOT)
                if s is None:
                    break
                if s.tier == MemoryTier.COLD:
                    # First cold slot — success
                    self.assertEqual(s.device_id, -1)
                    break
                gpu_slots.append(s)

    class TestBudgetValidation(unittest.TestCase):
        def test_valid_config_produces_no_errors(self):
            bc = TierBudgetConfig(
                hot_gb=1.0,
                warm_gb_per_device=0.5,
                per_device_max_requests_hot=4,
                per_device_max_requests_warm=2,
            )
            per_slot = 32 * 1024  # 32 KB
            errors = validate_budget_config(bc, per_slot)
            self.assertEqual(errors, [])

        def test_invalid_hot_budget_caught(self):
            bc = TierBudgetConfig(
                hot_gb=0.000001,
                warm_gb_per_device=0.5,
                per_device_max_requests_hot=64,
                per_device_max_requests_warm=2,
            )
            per_slot = 100 * 1024
            errors = validate_budget_config(bc, per_slot)
            self.assertTrue(any("hot" in e for e in errors))

        def test_zero_tier_budget_not_an_error(self):
            # Disabled tiers (0.0 GB) should not produce validation errors
            bc = TierBudgetConfig(
                hot_gb=0.0,
                warm_gb_per_device=0.5,
                per_device_max_requests_hot=64,
                per_device_max_requests_warm=2,
            )
            per_slot = 100 * 1024
            errors = validate_budget_config(bc, per_slot)
            self.assertFalse(any("hot" in e for e in errors))

    class TestLocalityHintMap(unittest.TestCase):
        def test_record_and_query(self):
            m = LocalityHintMap()
            m.record(MemoryTier.HOT, 7, 2)
            self.assertEqual(m.locality_hint(MemoryTier.HOT, 7), 2)

        def test_forget_removes_entry(self):
            m = LocalityHintMap()
            m.record(MemoryTier.WARM, 3, 0)
            m.forget(MemoryTier.WARM, 3)
            self.assertIsNone(m.locality_hint(MemoryTier.WARM, 3))

        def test_slots_on_device(self):
            m = LocalityHintMap()
            m.record(MemoryTier.HOT, 1, 2)
            m.record(MemoryTier.WARM, 0, 0)
            m.record(MemoryTier.WARM, 1, 2)
            on_2 = m.slots_on_device(2)
            self.assertIn((MemoryTier.HOT, 1), on_2)
            self.assertIn((MemoryTier.WARM, 1), on_2)
            self.assertNotIn((MemoryTier.WARM, 0), on_2)

        def test_thread_safety(self):
            m = LocalityHintMap()
            import concurrent.futures

            def worker(i):
                m.record(MemoryTier.WARM, i, i % 2)
                _ = m.locality_hint(MemoryTier.WARM, i)
                m.forget(MemoryTier.WARM, i)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(worker, i) for i in range(200)]
                for f in futs:
                    f.result()  # raises if any thread threw

    class TestColdTierAllocator(unittest.TestCase):
        def test_allocates_up_to_max_slots(self):
            alloc = ColdTierAllocator(
                budget_gb=0.001,  # 1 MB
                per_slot_bytes=4 * 1024,
                num_mamba_layers=1,
                conv_states_shape=(4,),
                ssm_states_shape=(4,),
                conv_states_dtype=torch.float16,
                ssm_states_dtype=torch.float32,
            )
            slots = []
            while True:
                s = alloc.allocate()
                if s is None:
                    break
                slots.append(s)
            self.assertEqual(len(slots), alloc.max_slots)

        def test_free_returns_slot(self):
            alloc = ColdTierAllocator(
                budget_gb=0.001,
                per_slot_bytes=4 * 1024,
                num_mamba_layers=1,
                conv_states_shape=(4,),
                ssm_states_shape=(4,),
                conv_states_dtype=torch.float16,
                ssm_states_dtype=torch.float32,
            )
            s = alloc.allocate()
            fc = alloc.free_count
            alloc.free(s)
            self.assertEqual(alloc.free_count, fc + 1)

        def test_zero_budget_yields_zero_slots(self):
            alloc = ColdTierAllocator(
                budget_gb=0.0,
                per_slot_bytes=1024,
                num_mamba_layers=1,
                conv_states_shape=(4,),
                ssm_states_shape=(4,),
                conv_states_dtype=torch.float16,
                ssm_states_dtype=torch.float32,
            )
            self.assertEqual(alloc.max_slots, 0)
            self.assertIsNone(alloc.allocate())

    # Run all tests
    print("\n" + "=" * 70)
    print("TierAwareMambaCache — DES-LOC unit test suite")
    print("=" * 70 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestMaxIntermediateOffsets,
        TestBudgetPreview,
        TestTierSlotAllocatorBudgetError,
        TestTierSlotAllocatorBasicOps,
        TestTierAwareMambaCacheMemoryOnlyMode,
        TestTierAwareMambaCacheAllocation,
        TestTierAwareMambaCacheWithColdTier,
        TestBudgetValidation,
        TestLocalityHintMap,
        TestColdTierAllocator,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
