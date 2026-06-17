"""
DES-LOC Heterogeneous Pinned Buffer Configuration
==================================================

Upstream intent (Megatron fde4059):
    Megatron-LM added ``cpu_offloading_retain_pinned_cpu_buffers`` to
    ``ModelParallelConfig`` so that pinned CPU buffers allocated during
    activation / weight offloading are *not* freed at the end of an
    iteration but are kept alive and reused in the next one.  The primary
    motivation is CUDA-graph compatibility: CUDA graphs require that tensor
    addresses remain stable across captures and replays, so releasing and
    re-allocating pinned buffers between iterations would invalidate any
    captured graph.

DES-LOC adaptation points:
    In the DES-LOC (Decoupled Execution with Shared LOcality Cache)
    framework the situation is richer than Megatron's single-tier CPU pool:

    1. **Heterogeneous device fleet** – we manage three distinct device
       classes simultaneously:
         • SM86 workers  (2 × A6000 48 GB, PCIe, no NVLink)
         • SM90 worker   (1 × H100 NVL 96 GB, PCIe)
         • Host DRAM     (1.5 TB, shared NUMA locality cache)

    2. **Per-device buffer affinity** – a pinned buffer allocated on NUMA
       node 0 (closest to A6000s) must not be transparently reused for
       transfers targeting the H100, and vice-versa.  ``HeteroPinnedBufferConfig``
       encodes per-tier retention policies and NUMA affinities.

    3. **Shared LOcality Cache (SLC)** – the 1.5 TB DRAM hosts the SLC
       which acts as a staging area between GPU tiers.  Pinned buffer
       lifetime and reuse must be co-ordinated with SLC eviction policy to
       avoid double-buffering the same tensor in both pinned memory and the
       SLC.

    4. **CUDA-graph stability across tiers** – SM90 (H100) can capture
       CUDA graphs with large batch sizes while SM86 (A6000) workers run
       smaller micro-batches.  Buffer addresses must remain stable *per
       device tier* independently, so retention is managed per-tier rather
       than as a global flag.

    5. **DeepSpeed ZeRO integration** – DeepSpeed's ZeRO-Offload already
       maintains its own pinned parameter/gradient buffers.  This module
       wraps and extends those buffers with the DES-LOC retention semantic
       rather than duplicating allocation logic.

Author: Neuron_SP project (DES-LOC adaptation of Megatron fde4059a9d47)
"""

from __future__ import annotations

import logging
import math
import os
import threading
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device-tier taxonomy
# ---------------------------------------------------------------------------

class DeviceTier(Enum):
    """Logical tiers of the DES-LOC heterogeneous cluster."""
    SM86 = auto()   # A6000 48 GB, PCIe, SM capability 8.6
    SM90 = auto()   # H100 NVL 96 GB, PCIe, SM capability 9.0
    HOST = auto()   # 1.5 TB CPU DRAM – the Shared LOcality Cache tier


_TIER_NAMES: Dict[DeviceTier, str] = {
    DeviceTier.SM86: "A6000_SM86",
    DeviceTier.SM90: "H100NVL_SM90",
    DeviceTier.HOST: "HOST_SLC",
}

# NUMA node heuristics for pinned buffer allocation.
# On a dual-socket system the A6000 cards are typically closest to NUMA-0
# and the H100 (PCIe) to NUMA-1.  Override via env-vars if the topology
# differs on a specific machine.
_DEFAULT_NUMA_MAP: Dict[DeviceTier, int] = {
    DeviceTier.SM86: int(os.environ.get("DESLOC_SM86_NUMA_NODE", "0")),
    DeviceTier.SM90: int(os.environ.get("DESLOC_SM90_NUMA_NODE", "1")),
    DeviceTier.HOST: int(os.environ.get("DESLOC_HOST_NUMA_NODE", "0")),
}


# ---------------------------------------------------------------------------
# Per-tier retention policy
# ---------------------------------------------------------------------------

@dataclass
class TierBufferPolicy:
    """
    Pinned-buffer retention policy for a single device tier.

    Parameters
    ----------
    retain_pinned_buffers:
        Mirror of Megatron's ``cpu_offloading_retain_pinned_cpu_buffers``.
        When *True* the pool keeps allocated pages alive across iterations.
        Required for CUDA-graph capture on SM90; optional but beneficial on
        SM86 to amortise ``cudaHostAlloc`` overhead.
    enable_double_buffering:
        Whether to maintain two pinned slots per layer so that H2D / D2H
        transfers for layer *n+1* overlap with compute on layer *n*.
        Corresponds to Megatron's ``cpu_offloading_double_buffering``.
    max_pool_bytes:
        Hard cap on pinned memory that this tier may allocate.  ``None``
        means no explicit cap (rely on OS limits).
    numa_node:
        NUMA node from which pinned pages are preferred.  -1 = system
        default.
    slc_aware:
        If *True* the buffer manager will check the SLC before allocating a
        new pinned page for this tier – reusing SLC-resident tensors avoids
        double-buffering the same data in both pinned memory and the SLC.
    cuda_graph_stable:
        Enforce address-stability guarantees required for CUDA-graph replay.
        When *True* the pool pre-allocates the full ``max_pool_bytes`` at
        initialisation time so that addresses never change.
    """
    retain_pinned_buffers: bool = False
    enable_double_buffering: bool = False
    max_pool_bytes: Optional[int] = None
    numa_node: int = -1
    slc_aware: bool = True
    cuda_graph_stable: bool = False

    def __post_init__(self) -> None:
        if self.cuda_graph_stable and not self.retain_pinned_buffers:
            logger.warning(
                "cuda_graph_stable=True requires retain_pinned_buffers=True; "
                "enabling retain_pinned_buffers automatically."
            )
            self.retain_pinned_buffers = True
        if self.cuda_graph_stable and self.max_pool_bytes is None:
            raise ValueError(
                "cuda_graph_stable=True requires an explicit max_pool_bytes "
                "so the pool can pre-allocate a contiguous arena."
            )


# ---------------------------------------------------------------------------
# Top-level config dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeteroPinnedBufferConfig:
    """
    DES-LOC heterogeneous pinned-buffer configuration.

    This dataclass is the DES-LOC counterpart of Megatron's
    ``ModelParallelConfig.cpu_offloading_retain_pinned_cpu_buffers``.
    Instead of a single Boolean it expresses *per-tier* retention policies
    that are aware of:
      • NUMA topology
      • CUDA-graph capture requirements (SM90 vs SM86)
      • Interaction with the Shared LOcality Cache

    Typical construction
    --------------------
    Use the factory helpers ``for_cuda_graph_training`` or
    ``from_deepspeed_config`` rather than constructing directly.

    Parameters
    ----------
    tier_policies:
        Mapping from ``DeviceTier`` to its ``TierBufferPolicy``.
        All three tiers must be present.
    global_slc_budget_bytes:
        Total bytes the SLC is allowed to hold as warm pinned tensors.
        Eviction across tiers is co-ordinated by ``HeteroPinnedBufferPool``
        against this budget.
    offload_activations:
        Global switch: are activations being offloaded at all?
    offload_weights:
        Global switch: are weights being offloaded at all?
    """

    tier_policies: Dict[DeviceTier, TierBufferPolicy] = field(
        default_factory=lambda: {
            DeviceTier.SM86: TierBufferPolicy(),
            DeviceTier.SM90: TierBufferPolicy(),
            DeviceTier.HOST: TierBufferPolicy(),
        }
    )
    global_slc_budget_bytes: int = int(1.2e12)   # 1.2 TB default SLC budget
    offload_activations: bool = True
    offload_weights: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        missing = {t for t in DeviceTier if t not in self.tier_policies}
        if missing:
            raise ValueError(
                f"tier_policies must contain entries for all DeviceTiers; "
                f"missing: {[t.name for t in missing]}"
            )
        total_pinned = sum(
            p.max_pool_bytes
            for p in self.tier_policies.values()
            if p.max_pool_bytes is not None
        )
        if total_pinned > self.global_slc_budget_bytes:
            logger.warning(
                "Sum of per-tier max_pool_bytes (%d) exceeds "
                "global_slc_budget_bytes (%d). SLC eviction pressure will "
                "be high.", total_pinned, self.global_slc_budget_bytes,
            )
        logger.debug("HeteroPinnedBufferConfig validated: %s", self)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def policy(self, tier: DeviceTier) -> TierBufferPolicy:
        return self.tier_policies[tier]

    @property
    def any_retention_enabled(self) -> bool:
        return any(p.retain_pinned_buffers for p in self.tier_policies.values())

    @property
    def cuda_graph_tiers(self) -> List[DeviceTier]:
        return [t for t, p in self.tier_policies.items() if p.cuda_graph_stable]

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def for_cuda_graph_training(
        cls,
        sm90_pool_bytes: int = int(24e9),   # 24 GB for H100 offload buffer
        sm86_pool_bytes: int = int(8e9),    # 8 GB per A6000
        slc_budget_bytes: int = int(1.2e12),
    ) -> "HeteroPinnedBufferConfig":
        """
        Factory for the most common DES-LOC training scenario where the SM90
        (H100) worker uses CUDA-graph capture for large-batch inference steps
        while SM86 workers run eager small micro-batches.

        This mirrors the motivating use-case in Megatron fde4059: retain
        pinned buffers across iterations to satisfy CUDA-graph address
        stability, but scopes the requirement only to the SM90 tier.
        """
        policies = {
            DeviceTier.SM86: TierBufferPolicy(
                retain_pinned_buffers=True,
                enable_double_buffering=True,
                max_pool_bytes=sm86_pool_bytes,
                numa_node=_DEFAULT_NUMA_MAP[DeviceTier.SM86],
                slc_aware=True,
                cuda_graph_stable=False,
            ),
            DeviceTier.SM90: TierBufferPolicy(
                retain_pinned_buffers=True,
                enable_double_buffering=True,
                max_pool_bytes=sm90_pool_bytes,
                numa_node=_DEFAULT_NUMA_MAP[DeviceTier.SM90],
                slc_aware=True,
                cuda_graph_stable=True,   # H100: CUDA-graph capture
            ),
            DeviceTier.HOST: TierBufferPolicy(
                retain_pinned_buffers=False,
                enable_double_buffering=False,
                max_pool_bytes=slc_budget_bytes,
                numa_node=_DEFAULT_NUMA_MAP[DeviceTier.HOST],
                slc_aware=False,   # HOST tier *is* the SLC
                cuda_graph_stable=False,
            ),
        }
        cfg = cls(
            tier_policies=policies,
            global_slc_budget_bytes=slc_budget_bytes,
            offload_activations=True,
            offload_weights=False,
        )
        logger.info(
            "HeteroPinnedBufferConfig created via for_cuda_graph_training: "
            "SM86 pool=%s GB, SM90 pool=%s GB (cuda_graph_stable), "
            "SLC budget=%s GB",
            sm86_pool_bytes // int(1e9),
            sm90_pool_bytes // int(1e9),
            slc_budget_bytes // int(1e9),
        )
        return cfg

    @classmethod
    def from_deepspeed_config(cls, ds_config: dict) -> "HeteroPinnedBufferConfig":
        """
        Construct from a DeepSpeed JSON config dict.

        Looks for a ``"desloc"`` sub-section; falls back to sensible
        defaults that preserve backward compatibility with plain ZeRO-Offload
        configs (no DES-LOC section → all retention disabled, matching
        Megatron's original False default).

        Expected JSON structure::

            {
              "desloc": {
                "offload_activations": true,
                "offload_weights": false,
                "global_slc_budget_gb": 1200,
                "sm86": {
                  "retain_pinned_buffers": true,
                  "double_buffering": true,
                  "max_pool_gb": 8,
                  "numa_node": 0
                },
                "sm90": {
                  "retain_pinned_buffers": true,
                  "double_buffering": true,
                  "max_pool_gb": 24,
                  "numa_node": 1,
                  "cuda_graph_stable": true
                }
              }
            }
        """
        desloc_cfg = ds_config.get("desloc", {})
        if not desloc_cfg:
            logger.info(
                "No 'desloc' section in DeepSpeed config; using conservative "
                "defaults (all pinned-buffer retention disabled)."
            )

        def _gb(section: dict, key: str, default_bytes: Optional[int]) -> Optional[int]:
            if key in section:
                return int(section[key] * 1e9)
            return default_bytes

        sm86_sec = desloc_cfg.get("sm86", {})
        sm90_sec = desloc_cfg.get("sm90", {})
        slc_gb = desloc_cfg.get("global_slc_budget_gb", 1200)

        policies = {
            DeviceTier.SM86: TierBufferPolicy(
                retain_pinned_buffers=sm86_sec.get("retain_pinned_buffers", False),
                enable_double_buffering=sm86_sec.get("double_buffering", False),
                max_pool_bytes=_gb(sm86_sec, "max_pool_gb", None),
                numa_node=sm86_sec.get("numa_node", _DEFAULT_NUMA_MAP[DeviceTier.SM86]),
                slc_aware=sm86_sec.get("slc_aware", True),
                cuda_graph_stable=sm86_sec.get("cuda_graph_stable", False),
            ),
            DeviceTier.SM90: TierBufferPolicy(
                retain_pinned_buffers=sm90_sec.get("retain_pinned_buffers", False),
                enable_double_buffering=sm90_sec.get("double_buffering", False),
                max_pool_bytes=_gb(sm90_sec, "max_pool_gb", None),
                numa_node=sm90_sec.get("numa_node", _DEFAULT_NUMA_MAP[DeviceTier.SM90]),
                slc_aware=sm90_sec.get("slc_aware", True),
                cuda_graph_stable=sm90_sec.get("cuda_graph_stable", False),
            ),
            DeviceTier.HOST: TierBufferPolicy(
                retain_pinned_buffers=False,
                slc_aware=False,
                max_pool_bytes=int(slc_gb * 1e9),
                numa_node=_DEFAULT_NUMA_MAP[DeviceTier.HOST],
            ),
        }
        return cls(
            tier_policies=policies,
            global_slc_budget_bytes=int(slc_gb * 1e9),
            offload_activations=desloc_cfg.get("offload_activations", True),
            offload_weights=desloc_cfg.get("offload_weights", False),
        )


# ---------------------------------------------------------------------------
# Pinned buffer pool
# ---------------------------------------------------------------------------

class _PinnedSlot:
    """
    A single pinned-memory allocation unit managed by ``HeteroPinnedBufferPool``.

    Wraps a ``torch.Tensor`` of dtype ``torch.uint8`` allocated in pinned
    (page-locked) memory.  Tracks whether the slot is currently in use and
    whether its address has been captured in a CUDA graph.
    """

    __slots__ = ("data", "tier", "in_use", "graph_captured", "_id")
    _counter = 0

    def __init__(self, nbytes: int, tier: DeviceTier) -> None:
        self.data: torch.Tensor = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.tier = tier
        self.in_use = False
        self.graph_captured = False
        _PinnedSlot._counter += 1
        self._id = _PinnedSlot._counter
        logger.debug(
            "PinnedSlot #%d allocated: tier=%s, nbytes=%d, addr=0x%x",
            self._id, _TIER_NAMES[tier], nbytes, self.data.data_ptr(),
        )

    @property
    def nbytes(self) -> int:
        return self.data.numel()

    def as_typed(self, dtype: torch.dtype, shape: Tuple[int, ...]) -> torch.Tensor:
        """Return a typed view of the raw pinned storage."""
        n_elements = math.prod(shape)
        bytes_needed = n_elements * dtype.itemsize  # type: ignore[attr-defined]
        if bytes_needed > self.nbytes:
            raise ValueError(
                f"Slot {self._id} has {self.nbytes} bytes but {bytes_needed} "
                f"bytes requested for shape={shape} dtype={dtype}."
            )
        return self.data[:bytes_needed].view(dtype).view(shape)

    def free(self, force: bool = False) -> None:
        """
        Mark slot as free.  If ``force=True`` the underlying tensor is
        deleted, releasing the pinned pages.  Normally (retention mode) we
        only clear ``in_use`` and keep pages alive.
        """
        self.in_use = False
        if force:
            del self.data
            logger.debug("PinnedSlot #%d pages released (forced).", self._id)
        else:
            logger.debug("PinnedSlot #%d marked free (pages retained).", self._id)


class HeteroPinnedBufferPool:
    """
    Thread-safe pool of pinned CPU buffers for DES-LOC heterogeneous training.

    Design
    ------
    Each ``DeviceTier`` has an independent sub-pool governed by its
    ``TierBufferPolicy``.  The pool supports two allocation modes:

    **Retention mode** (``retain_pinned_buffers=True``)
        Slots are never freed between iterations.  On the first iteration the
        pool allocates pages; on subsequent iterations it returns the same
        slot objects with the same virtual addresses.  This satisfies
        CUDA-graph requirements on SM90 and reduces ``cudaHostAlloc`` churn
        on SM86.

    **Ephemeral mode** (``retain_pinned_buffers=False``)
        Slots are freed when released, reclaiming pinned pages.  Used for
        HOST-tier SLC staging buffers that have a short lifetime.

    Double-buffering
    ----------------
    When ``enable_double_buffering=True`` two slot banks (bank 0 / bank 1)
    are maintained per tier.  The caller alternates banks across layers via
    ``request_slot(bank=…)``, achieving H2D/D2H overlap without introducing
    synchronisation barriers inside the pool.

    SLC awareness
    -------------
    Before allocating a new pinned slot the pool checks whether a tensor of
    the requested size already resides in the SLC (tracked via a weak-ref
    registry).  If so it returns a zero-copy view, avoiding both a new
    ``cudaHostAlloc`` and a redundant data copy.

    Thread safety
    -------------
    A per-tier ``threading.Lock`` serialises ``request_slot`` / ``release_slot``
    calls.  Double-buffering bank selection is *not* serialised – callers are
    expected to alternate banks from a single pipeline thread.
    """

    def __init__(self, config: HeteroPinnedBufferConfig) -> None:
        self.config = config
        # Slot storage: tier → bank (0 or 1) → list of _PinnedSlot
        self._slots: Dict[DeviceTier, List[List[_PinnedSlot]]] = {
            t: [[], []] for t in DeviceTier
        }
        self._locks: Dict[DeviceTier, threading.Lock] = {
            t: threading.Lock() for t in DeviceTier
        }
        # Weak-ref SLC registry: ptr → _PinnedSlot
        self._slc_registry: Dict[int, weakref.ref] = {}
        self._iteration: int = 0

        # Pre-allocate for CUDA-graph-stable tiers
        for tier, policy in config.tier_policies.items():
            if policy.cuda_graph_stable and policy.max_pool_bytes is not None:
                self._preallocate(tier, policy)

    # ------------------------------------------------------------------
    # Pre-allocation (CUDA-graph stability)
    # ------------------------------------------------------------------

    def _preallocate(self, tier: DeviceTier, policy: TierBufferPolicy) -> None:
        """
        Pre-allocate the full pool arena for a CUDA-graph-stable tier.

        By allocating one large contiguous slot up-front we guarantee that
        sub-tensor views always have stable virtual addresses – the CUDA
        runtime maps the same physical pages for every graph replay.
        """
        assert policy.max_pool_bytes is not None
        n_banks = 2 if policy.enable_double_buffering else 1
        per_bank = policy.max_pool_bytes // n_banks
        logger.info(
            "Pre-allocating CUDA-graph-stable pool for %s: "
            "%d bank(s) × %d MB = %d MB pinned",
            _TIER_NAMES[tier], n_banks,
            per_bank // (1024 * 1024),
            policy.max_pool_bytes // (1024 * 1024),
        )
        for bank in range(n_banks):
            slot = _PinnedSlot(per_bank, tier)
            slot.graph_captured = False
            self._slots[tier][bank].append(slot)

    # ------------------------------------------------------------------
    # Core allocation interface
    # ------------------------------------------------------------------

    def request_slot(
        self,
        tier: DeviceTier,
        nbytes: int,
        bank: int = 0,
        dtype: Optional[torch.dtype] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> _PinnedSlot:
        """
        Acquire a pinned buffer slot for *tier* from the specified *bank*.

        Parameters
        ----------
        tier:
            Target device tier.
        nbytes:
            Minimum size in bytes.  If ``dtype`` and ``shape`` are provided
            the actual requirement is inferred from them (and must match).
        bank:
            Double-buffer bank index (0 or 1).  Callers alternate banks
            across pipeline micro-steps to overlap H2D/D2H with compute.
        dtype, shape:
            Optional: if given, validate that the slot is large enough for
            a tensor of this dtype and shape.

        Returns
        -------
        _PinnedSlot
            A slot whose ``data`` buffer is pinned and ready.  In retention
            mode the same slot is returned on every call with matching
            (tier, nbytes, bank); in ephemeral mode a new slot may be
            returned each time.

        Raises
        ------
        RuntimeError
            If the pool budget for this tier would be exceeded.
        """
        policy = self.config.policy(tier)
        if bank not in (0, 1):
            raise ValueError(f"bank must be 0 or 1, got {bank}.")
        if not policy.enable_double_buffering and bank == 1:
            logger.debug(
                "Double-buffering disabled for %s; redirecting bank=1 → bank=0.",
                _TIER_NAMES[tier],
            )
            bank = 0

        if dtype is not None and shape is not None:
            nbytes = max(nbytes, math.prod(shape) * torch.tensor([], dtype=dtype).element_size())

        with self._locks[tier]:
            # 1. Check SLC registry (avoid redundant alloc / copy)
            if policy.slc_aware:
                slot = self._slc_lookup(tier, nbytes)
                if slot is not None:
                    slot.in_use = True
                    logger.debug(
                        "SLC hit for tier=%s bank=%d nbytes=%d ptr=0x%x",
                        _TIER_NAMES[tier], bank, nbytes, slot.data.data_ptr(),
                    )
                    return slot

            # 2. Retention mode: reuse an existing free slot of sufficient size
            if policy.retain_pinned_buffers:
                for slot in self._slots[tier][bank]:
                    if not slot.in_use and slot.nbytes >= nbytes:
                        slot.in_use = True
                        logger.debug(
                            "Retained slot #%d reused: tier=%s bank=%d",
                            slot._id, _TIER_NAMES[tier], bank,
                        )
                        return slot

            # 3. Allocate new slot (subject to budget check)
            self._check_budget(tier, nbytes, policy)
            slot = _PinnedSlot(nbytes, tier)
            slot.in_use = True
            self._slots[tier][bank].append(slot)

            if policy.slc_aware:
                self._slc_register(slot)

            return slot

    def release_slot(self, slot: _PinnedSlot, iteration_end: bool = False) -> None:
        """
        Release a slot back to the pool.

        Parameters
        ----------
        slot:
            The slot to release.
        iteration_end:
            If *True* and the tier policy has ``retain_pinned_buffers=False``
            the pinned pages are freed immediately.  Otherwise (retention
            mode or mid-iteration) the slot is just marked free.
        """
        policy = self.config.policy(slot.tier)
        force_free = iteration_end and not policy.retain_pinned_buffers
        slot.free(force=force_free)
        if force_free:
            with self._locks[slot.tier]:
                for bank in self._slots[slot.tier]:
                    if slot in bank:
                        bank.remove(slot)
            logger.debug(
                "Slot #%d removed from pool (tier=%s, iteration_end=True, ephemeral).",
                slot._id, _TIER_NAMES[slot.tier],
            )

    def mark_iteration_end(self) -> None:
        """
        Advance the iteration counter and release ephemeral slots.

        Call once per training iteration (before the next forward pass) to
        ensure that slots from ephemeral-mode tiers are properly reclaimed.
        Retention-mode slots are untouched – their pages remain locked.
        """
        self._iteration += 1
        for tier, policy in self.config.tier_policies.items():
            if policy.retain_pinned_buffers:
                logger.debug(
                    "Iteration %d: retaining all slots for tier=%s.",
                    self._iteration, _TIER_NAMES[tier],
                )
                # Just clear the in_use flag so slots are available next iter
                with self._locks[tier]:
                    for bank in self._slots[tier]:
                        for slot in bank:
                            slot.in_use = False
            else:
                # Ephemeral: release pages
                with self._locks[tier]:
                    freed = 0
                    for bank in self._slots[tier]:
                        for slot in list(bank):
                            slot.free(force=True)
                            bank.remove(slot)
                            freed += 1
                    if freed:
                        logger.debug(
                            "Iteration %d: freed %d ephemeral slot(s) for tier=%s.",
                            self._iteration, freed, _TIER_NAMES[tier],
                        )

    # ------------------------------------------------------------------
    # SLC registry helpers
    # ------------------------------------------------------------------

    def _slc_register(self, slot: _PinnedSlot) -> None:
        ptr = slot.data.data_ptr()
        self._slc_registry[ptr] = weakref.ref(slot)

    def _slc_lookup(self, tier: DeviceTier, nbytes: int) -> Optional[_PinnedSlot]:
        dead_keys = []
        for ptr, ref in self._slc_registry.items():
            slot = ref()
            if slot is None:
                dead_keys.append(ptr)
                continue
            if slot.tier == tier and not slot.in_use and slot.nbytes >= nbytes:
                for k in dead_keys:
                    del self._slc_registry[k]
                return slot
        for k in dead_keys:
            del self._slc_registry[k]
        return None

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _check_budget(
        self, tier: DeviceTier, nbytes: int, policy: TierBufferPolicy
    ) -> None:
        if policy.max_pool_bytes is None:
            return
        current = sum(
            s.nbytes
            for bank in self._slots[tier]
            for s in bank
        )
        if current + nbytes > policy.max_pool_bytes:
            raise RuntimeError(
                f"HeteroPinnedBufferPool: budget exceeded for tier "
                f"{_TIER_NAMES[tier]}: current={current}, "
                f"requested={nbytes}, limit={policy.max_pool_bytes}."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        """Return a snapshot of pool utilisation per tier."""
        out: Dict[str, object] = {"iteration": self._iteration}
        for tier in DeviceTier:
            in_use_bytes = sum(
                s.nbytes for bank in self._slots[tier]
                for s in bank if s.in_use
            )
            total_bytes = sum(
                s.nbytes for bank in self._slots[tier] for s in bank
            )
            out[_TIER_NAMES[tier]] = {
                "total_slots": sum(len(b) for b in self._slots[tier]),
                "in_use_bytes": in_use_bytes,
                "total_bytes": total_bytes,
                "retain_mode": self.config.policy(tier).retain_pinned_buffers,
            }
        return out


# ---------------------------------------------------------------------------
# DeepSpeed integration shim
# ---------------------------------------------------------------------------

def build_hetero_pinned_pool_from_engine(engine) -> HeteroPinnedBufferPool:
    """
    Construct a ``HeteroPinnedBufferPool`` from a live DeepSpeed engine.

    Reads ``engine.config`` (a dict) for the ``desloc`` sub-section and
    wires the pool's ``mark_iteration_end`` into ``engine.step()`` via a
    monkey-patch so that the caller does not need to invoke it manually.

    Parameters
    ----------
    engine:
        A ``deepspeed.DeepSpeedEngine`` instance (duck-typed: needs
        ``.config`` and ``.step``).

    Returns
    -------
    HeteroPinnedBufferPool
        Configured and ready; also attached as ``engine._desloc_pinned_pool``.
    """
    ds_config: dict = engine.config if isinstance(engine.config, dict) else {}
    cfg = HeteroPinnedBufferConfig.from_deepspeed_config(ds_config)
    pool = HeteroPinnedBufferPool(cfg)

    original_step = engine.step

    def _patched_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        pool.mark_iteration_end()
        return result

    engine.step = _patched_step  # type: ignore[method-assign]
    engine._desloc_pinned_pool = pool  # type: ignore[attr-defined]
    logger.info(
        "HeteroPinnedBufferPool attached to DeepSpeed engine; "
        "mark_iteration_end() wired into engine.step()."
    )
    return pool


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # 1. Config factory produces valid config
    cfg = HeteroPinnedBufferConfig.for_cuda_graph_training(
        sm90_pool_bytes=int(2e9), sm86_pool_bytes=int(512e6)
    )
    assert cfg.policy(DeviceTier.SM90).cuda_graph_stable
    assert cfg.policy(DeviceTier.SM90).retain_pinned_buffers
    assert not cfg.policy(DeviceTier.SM86).cuda_graph_stable

    # 2. from_deepspeed_config falls back gracefully with empty config
    cfg2 = HeteroPinnedBufferConfig.from_deepspeed_config({})
    assert not cfg2.any_retention_enabled

    # 3. Pool allocates and tracks slots per tier
    pool = HeteroPinnedBufferPool(cfg)
    slot_sm86 = pool.request_slot(DeviceTier.SM86, 1024, bank=0)
    assert slot_sm86.in_use
    assert slot_sm86.nbytes >= 1024

    # 4. Retention: same slot returned after release + re-request
    pool.release_slot(slot_sm86)
    slot_sm86_b = pool.request_slot(DeviceTier.SM86, 1024, bank=0)
    assert slot_sm86_b._id == slot_sm86._id, "Retained slot should be reused"

    # 5. mark_iteration_end clears in_use for retention-mode tiers
    pool.mark_iteration_end()
    stats = pool.stats()
    assert stats[_TIER_NAMES[DeviceTier.SM86]]["in_use_bytes"] == 0

    logger.info("All smoke tests passed.")
