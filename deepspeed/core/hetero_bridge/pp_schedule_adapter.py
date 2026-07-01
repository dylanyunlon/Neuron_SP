# SPDX-License-Identifier: Apache-2.0
"""pp_schedule_adapter.py — bridge to core.pipeline_parallel.schedules for hetero layer split.

Phase 1 skeleton. Signatures frozen. Bodies raise NotImplementedError.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tier_map import TierMap


class PPScheduleAdapter:
    def __init__(self, tier_map: "TierMap", num_layers: int) -> None:
        self.tier_map = tier_map
        self.num_layers = num_layers

    def layer_split(self) -> list[int]:
        """VRAM-weighted layers-per-stage, e.g. [6,6,20] for 2xA6000+H100."""
        raise NotImplementedError("PPScheduleAdapter.layer_split")

    def forward_backward(self, *, data_iterator, model, num_microbatches: int,
                         seq_length: int, micro_batch_size: int) -> dict:
        """Delegate to schedules.forward_backward_pipelining_* with the hetero split."""
        raise NotImplementedError("PPScheduleAdapter.forward_backward")
