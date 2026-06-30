# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""DES-LOC configuration — shared by all deepspeed/core/ modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch


class TierType(Enum):
    """GPU tier classification based on compute capability."""
    DATACENTER = auto()   # H100, A100, etc.
    PROFESSIONAL = auto() # RTX A6000, etc.
    BLACKWELL = auto()    # RTX PRO 6000 Blackwell (SM12.0)
    CONSUMER = auto()     # RTX 4090, 3090, etc.


@dataclass
class TierSpec:
    """Specification for a single GPU tier."""
    tier_type: TierType
    gpu_indices: List[int]
    sm_capability: Tuple[int, int]  # e.g. (9, 0) for SM9.0
    vram_gb: float
    bf16_tflops: float
    pcie_gen: int
    pcie_width: int
    numa_node: int


@dataclass
class DesLocConfig:
    """Configuration for DES-LOC decomposed synchronization.

    Controls how often parameters (Kx), first moments (Ku), and second
    moments (Kv) are synchronized across heterogeneous GPU tiers.
    """

    enabled: bool = True

    # Decomposed sync periods (in steps)
    kx: int = 8     # Parameter sync period
    ku: int = 32    # First moment sync period
    kv: int = 64    # Second moment sync period

    # ZeRO stage for params not pinned to a tier
    zero_stage: int = 3

    # Per-tier specs (populated at runtime by TierDiscovery)
    tiers: List[TierSpec] = field(default_factory=list)

    # Per-tier micro-batch sizes
    micro_batch_per_tier: Dict[int, int] = field(default_factory=dict)

    # LOC cache sizes per NUMA node (GB)
    loc_cache_gb: Dict[int, float] = field(default_factory=dict)

    # Broadcast every step (fixes Kx spike in ZeRO-3)
    broadcast_every_step: bool = True

    def is_kx_step(self, step: int) -> bool:
        return step % self.kx == 0

    def is_ku_step(self, step: int) -> bool:
        return step % self.ku == 0

    def is_kv_step(self, step: int) -> bool:
        return step % self.kv == 0

    def get_tier_for_gpu(self, gpu_index: int) -> Optional[TierSpec]:
        for tier in self.tiers:
            if gpu_index in tier.gpu_indices:
                return tier
        return None
