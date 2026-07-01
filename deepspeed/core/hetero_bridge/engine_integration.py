# SPDX-License-Identifier: Apache-2.0
"""engine_integration.py — single entrypoint that wires hetero_bridge into desloc_engine.

Phase 1 skeleton. Signature frozen. Body raises NotImplementedError.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from .desloc_sync_policy import SyncPeriods

if TYPE_CHECKING:
    pass


def install(engine, *, lr: float, betas: "tuple[float, float]" = (0.9, 0.95),
            weight_decay: float = 0.1,
            sync_periods: Optional["SyncPeriods"] = None) -> None:
    """Wire the full hetero_bridge stack onto `engine`.

    Order: TierMap.discover() -> HeteroShardPlanner.plan() -> DistOptAdapter.build()
    -> DesLocSyncPolicy -> PPScheduleAdapter -> AutoSPHook. After this call,
    engine.optimizer is the hetero DistributedOptimizer and the training loop uses
    adapter.reduce_scatter_grads / step / all_gather_params.
    """
    raise NotImplementedError("engine_integration.install")
