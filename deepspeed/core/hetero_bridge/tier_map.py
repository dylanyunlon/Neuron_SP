# SPDX-License-Identifier: Apache-2.0
"""tier_map.py — GPU tier discovery for the heterogeneous cluster.

Identifies each distributed rank's GPU tier (A6000 / H100 / Blackwell) via
torch.cuda device properties and cross-rank all_gather_object, then exposes
per-rank memory budgets for optimizer-state sizing.

Cluster topology (the contract from ARCHITECTURE.md):
  2 × A6000  (SM 8.6, 48 GB)
  1 × H100   (SM 9.0, 80–96 GB)
  2 × Blackwell (SM 12.0, 96+ GB)
  No NVLink — all inter-GPU comms over PCIe
"""
from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public enums / dataclasses  (signatures frozen per ARCHITECTURE.md)
# ---------------------------------------------------------------------------


class GPUTier(enum.Enum):
    A6000 = "a6000"          # SM8.6, ~48GB
    H100 = "h100"            # SM9.0, ~96GB
    BLACKWELL = "blackwell"  # SM12.0, ~96GB
    UNKNOWN = "unknown"


@dataclass
class TierInfo:
    rank: int
    tier: GPUTier
    total_vram_bytes: int
    numa_node: int
    peak_bf16_tflops: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# SM capability → GPUTier mapping
_SM_TO_TIER: Dict[tuple, GPUTier] = {
    (8, 6): GPUTier.A6000,   # RTX A6000, RTX 3090 (SM8.6)
    (8, 7): GPUTier.A6000,   # Jetson Orin (SM8.7, treat as A6000-class)
    (9, 0): GPUTier.H100,    # H100 SXM / PCIe
    (9, 4): GPUTier.H100,    # H100 NVL
    (12, 0): GPUTier.BLACKWELL,  # RTX PRO 6000 Blackwell, GB200
    (12, 1): GPUTier.BLACKWELL,
}

# Peak BF16 TFLOPS per tier (dense, no sparsity)
_TIER_TFLOPS: Dict[GPUTier, float] = {
    GPUTier.A6000:    309.7,
    GPUTier.H100:     989.0,
    GPUTier.BLACKWELL: 2250.0,  # RTX PRO 6000 Blackwell per-card BF16
    GPUTier.UNKNOWN:  100.0,
}

# Fraction of VRAM reserved for model weights + activations; the rest is
# available for optimizer state.  A6000 has a tighter budget.
_VRAM_RESERVE_FRACTION: Dict[GPUTier, float] = {
    GPUTier.A6000:    0.35,  # 35 % reserved → ~31 GB free on 48 GB card
    GPUTier.H100:     0.25,  # 25 % reserved → ~72 GB free on 96 GB card
    GPUTier.BLACKWELL: 0.20,  # 20 % reserved
    GPUTier.UNKNOWN:  0.40,
}

_BYTES_PER_GB = 1 << 30


def _detect_local_tier() -> TierInfo:
    """Detect the GPU tier of the *local* CUDA device.

    Falls back gracefully when CUDA is unavailable (CPU-only unit tests).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        prop = torch.cuda.get_device_properties(local_rank)
        sm = (prop.major, prop.minor)
        tier = _SM_TO_TIER.get(sm, GPUTier.UNKNOWN)
        total_vram = prop.total_memory  # bytes

        # NUMA node: try to read from sysfs; default to 0
        try:
            device_idx = torch.cuda.current_device()
            numa_path = f"/sys/bus/pci/devices/{device_idx}/numa_node"
            with open(numa_path) as f:
                numa_node = int(f.read().strip())
        except Exception:
            numa_node = 0

        global_rank = int(os.environ.get("RANK", "0"))
        return TierInfo(
            rank=global_rank,
            tier=tier,
            total_vram_bytes=total_vram,
            numa_node=max(0, numa_node),
            peak_bf16_tflops=_TIER_TFLOPS[tier],
        )

    except Exception as exc:
        logger.warning("TierMap: GPU detection failed (%s); using UNKNOWN tier", exc)
        global_rank = int(os.environ.get("RANK", "0"))
        return TierInfo(
            rank=global_rank,
            tier=GPUTier.UNKNOWN,
            total_vram_bytes=48 * _BYTES_PER_GB,  # conservative default
            numa_node=0,
            peak_bf16_tflops=_TIER_TFLOPS[GPUTier.UNKNOWN],
        )


def _gather_tier_infos(local_info: TierInfo, world_size: int) -> List[TierInfo]:
    """All-gather TierInfo objects from every rank via torch.distributed."""
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError("distributed not initialized")

        all_infos: List[Optional[TierInfo]] = [None] * world_size
        dist.all_gather_object(all_infos, local_info)
        # Type narrowing: after all_gather_object all slots are filled.
        return [info for info in all_infos if info is not None]  # type: ignore[misc]

    except Exception:
        # Single-rank or non-distributed test path: just return local info.
        logger.debug("TierMap: distributed gather unavailable; single-rank mode")
        return [local_info]


# ---------------------------------------------------------------------------
# TierMap  (frozen public API)
# ---------------------------------------------------------------------------


class TierMap:
    """Maps distributed rank → GPUTier + memory budget + NUMA affinity.

    Usage
    -----
    >>> tier_map = TierMap.discover()
    >>> tier_map.tier_of(0)        # GPUTier.A6000
    >>> tier_map.mem_budget(2)     # bytes available for optimizer state on H100
    >>> tier_map.ranks_of_tier(GPUTier.H100)  # [2]
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self._by_rank: Dict[int, TierInfo] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls) -> "TierMap":
        """Enumerate local+distributed ranks → TierInfo via torch.cuda + NVML.

        Each rank detects its own GPU properties and the results are
        exchanged via all_gather_object so every rank has the full map.
        """
        try:
            import torch.distributed as dist
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            world_size = 1

        local_info = _detect_local_tier()
        all_infos = _gather_tier_infos(local_info, world_size)

        tier_map = cls(world_size=world_size)
        for info in all_infos:
            tier_map._by_rank[info.rank] = info

        # Fill any gaps (ranks that didn't respond) with UNKNOWN.
        for r in range(world_size):
            if r not in tier_map._by_rank:
                tier_map._by_rank[r] = TierInfo(
                    rank=r,
                    tier=GPUTier.UNKNOWN,
                    total_vram_bytes=48 * _BYTES_PER_GB,
                    numa_node=0,
                    peak_bf16_tflops=_TIER_TFLOPS[GPUTier.UNKNOWN],
                )

        tiers_summary = {r: info.tier.value for r, info in tier_map._by_rank.items()}
        logger.info("TierMap.discover: %s", tiers_summary)
        return tier_map

    @classmethod
    def from_infos(cls, infos: List[TierInfo]) -> "TierMap":
        """Construct a TierMap directly from a list of TierInfo objects.

        Useful for unit tests and offline planning without a live cluster.

        Args:
            infos: One TierInfo per rank (length = world_size).

        Returns:
            Populated TierMap instance.
        """
        tm = cls(world_size=len(infos))
        for info in infos:
            tm._by_rank[info.rank] = info
        return tm

    # ------------------------------------------------------------------
    # Queries  (frozen API per ARCHITECTURE.md)
    # ------------------------------------------------------------------

    def tier_of(self, rank: int) -> GPUTier:
        """Return the GPU tier for *rank*.

        Args:
            rank: Global distributed rank (0-indexed).

        Returns:
            :class:`GPUTier` enum value.

        Raises:
            KeyError: If *rank* is not in the map.
        """
        return self._by_rank[rank].tier

    def mem_budget(self, rank: int) -> int:
        """Usable bytes for optimizer state after model + activations.

        Computed as ``total_vram * (1 - reserve_fraction)`` where the
        reserve fraction is tier-specific (A6000 reserves more because
        activations take proportionally more of a smaller VRAM pool).

        Args:
            rank: Global distributed rank.

        Returns:
            Integer byte count available for optimizer state.
        """
        info = self._by_rank[rank]
        reserve = _VRAM_RESERVE_FRACTION[info.tier]
        return int(info.total_vram_bytes * (1.0 - reserve))

    def ranks_of_tier(self, tier: GPUTier) -> List[int]:
        """Return all ranks belonging to *tier*, sorted ascending.

        Args:
            tier: :class:`GPUTier` to filter by.

        Returns:
            Sorted list of rank integers.
        """
        return sorted(r for r, info in self._by_rank.items() if info.tier == tier)

    # ------------------------------------------------------------------
    # Extra helpers (used by dist_opt_adapter)
    # ------------------------------------------------------------------

    def info(self, rank: int) -> TierInfo:
        """Return the full :class:`TierInfo` for *rank*."""
        return self._by_rank[rank]

    def is_low_vram(self, rank: int) -> bool:
        """Return True if *rank* is on A6000 (low VRAM → CPU optimizer path)."""
        return self._by_rank[rank].tier == GPUTier.A6000

    def is_high_vram(self, rank: int) -> bool:
        """Return True if *rank* is on H100 or Blackwell (fused AdamW path)."""
        return self._by_rank[rank].tier in (GPUTier.H100, GPUTier.BLACKWELL)

    def __repr__(self) -> str:  # pragma: no cover
        entries = ", ".join(
            f"rank{r}={info.tier.value}({info.total_vram_bytes // _BYTES_PER_GB}GB)"
            for r, info in sorted(self._by_rank.items())
        )
        return f"TierMap({entries})"
