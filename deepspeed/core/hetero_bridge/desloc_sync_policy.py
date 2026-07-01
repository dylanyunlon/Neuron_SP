# SPDX-License-Identifier: Apache-2.0
"""desloc_sync_policy.py — DES-LOC decomposed-local-SGD per-parameter sync schedule.

Phase 1 skeleton. Signatures frozen. Bodies raise NotImplementedError.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class SyncPeriods:
    kx: int  # fast params (e.g. attention) sync period
    ku: int  # slow params
    kv: int  # very-slow params


class DesLocSyncPolicy:
    def __init__(self, periods: "SyncPeriods") -> None:
        self.periods = periods

    def classify(self, named_params: "list[tuple[str, torch.Tensor]]") -> dict[int, str]:
        """param_id -> {'x','u','v'} by gradient-variance heuristic."""
        raise NotImplementedError("DesLocSyncPolicy.classify")

    def should_sync(self, param_id: int, step: int) -> bool:
        raise NotImplementedError("DesLocSyncPolicy.should_sync")
