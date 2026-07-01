# SPDX-License-Identifier: Apache-2.0
"""autosp_hook.py — connect AutoSP/Ulysses SP group to optimizer + grad reduction.

Phase 1 skeleton. Signatures frozen. Bodies raise NotImplementedError.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tier_map import TierMap
    from .dist_opt_adapter import DistOptAdapter
    from .desloc_sync_policy import DesLocSyncPolicy


class AutoSPHook:
    def __init__(self, sp_group, tier_map: "TierMap") -> None:
        self.sp_group = sp_group
        self.tier_map = tier_map

    def wrap_grad_reduction(self, adapter: "DistOptAdapter") -> None:
        """Ensure SP-sharded params reduce correctly across the SP group."""
        raise NotImplementedError("AutoSPHook.wrap_grad_reduction")

    def sp_aware_sync(self, policy: "DesLocSyncPolicy") -> None:
        """Make DES-LOC sync respect the SP group boundaries."""
        raise NotImplementedError("AutoSPHook.sp_aware_sync")
