# SPDX-License-Identifier: Apache-2.0
"""pp_schedule_adapter.py — bridge to core.pipeline_parallel.schedules for hetero layer split.

Adapts ``core.pipeline_parallel.schedules`` to DES-LOC per-tier timing by:

  1. ``layer_split()`` — computes VRAM-proportional layer counts per PP stage,
     consistent with the ``pp_layer_split`` entry in configs/7b_5gpu.yaml.

  2. ``forward_backward()`` — dispatches to the correct schedule function:
       - PP == 1  → ``forward_backward_no_pipelining``
       - PP == 5 (heterogeneous) → ``forward_backward_pipelining_without_interleaving_pp5_heterogeneous``
       - PP > 1 (other) → ``forward_backward_pipelining_without_interleaving``

     The heterogeneous PP=5 path injects a ``HeterogeneousBubbleFiller`` so
     fast stages (H100, Blackwell) compute extra microbatches during the warmup
     bubble while slow stages (A6000) catch up.

Layer split algorithm
---------------------
Mirrors the VRAM-proportional calculation in configs/7b_5gpu.yaml:
  total_vram = sum(mem_budget(r) for r in range(world_size))
  layers_r   = round(num_layers * mem_budget(r) / total_vram)
Remainder is added to the H100/Blackwell stage with most VRAM.

Bubble filler
-------------
``HeterogeneousBubbleFiller`` (from schedules.py) is instantiated only when:
  - PP == world_size (every rank is exactly one stage), AND
  - the tier map contains at least one H100 or Blackwell rank (fast stage).
"""
from __future__ import annotations
import torch

import logging
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from .tier_map import TierMap

logger = logging.getLogger(__name__)


class PPScheduleAdapter:
    """Bridge from DES-LOC hetero cluster to core.pipeline_parallel.schedules.

    Args:
        tier_map:   Cluster GPU tier map; used for VRAM-weighted layer split
                    and fast-rank identification for bubble filling.
        num_layers: Total transformer layers across all PP stages.
    """

    def __init__(self, tier_map: "TierMap", num_layers: int) -> None:
        self.tier_map = tier_map
        self.num_layers = num_layers
        # Cached split computed by layer_split()
        self._cached_split: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Public API (frozen per ARCHITECTURE.md)
    # ------------------------------------------------------------------

    def layer_split(self) -> List[int]:
        """VRAM-proportional layers-per-stage list.

        Returns a list of length ``world_size`` where ``result[r]`` is the
        number of transformer layers assigned to pipeline stage ``r``.  The
        sum always equals ``self.num_layers``.

        Example (5-GPU cluster from configs/7b_5gpu.yaml):
          VRAM budgets: [31GB, 78GB, 72GB, 31GB, 78GB]
          num_layers=32 → [4, 8, 8, 4, 8]  (matches yaml pp_layer_split)

        Returns:
            List[int] of per-stage layer counts.
        """
        if self._cached_split is not None:
            return list(self._cached_split)

        world_size = self.tier_map.world_size
        if world_size <= 1:
            self._cached_split = [self.num_layers]
            return list(self._cached_split)

        # VRAM budgets (bytes) per rank from TierMap
        budgets: List[int] = [
            self.tier_map.mem_budget(r) for r in range(world_size)
        ]
        total_budget = sum(budgets)
        if total_budget <= 0:
            # Uniform fallback
            base = self.num_layers // world_size
            remainder = self.num_layers % world_size
            split = [base + (1 if i < remainder else 0) for i in range(world_size)]
            self._cached_split = split
            return list(split)

        # Proportional allocation, round down
        split: List[int] = []
        assigned = 0
        for r in range(world_size - 1):
            count = int(self.num_layers * budgets[r] / total_budget)
            count = max(1, count)  # every stage gets at least 1 layer
            split.append(count)
            assigned += count
        # Last stage absorbs remainder
        last = max(1, self.num_layers - assigned)
        split.append(last)

        # Adjust if we over-assigned (can happen with max(1, ...) clamping)
        overshoot = sum(split) - self.num_layers
        if overshoot > 0:
            # Remove extra layers from the stage with the most layers
            for _ in range(overshoot):
                max_idx = split.index(max(split))
                if split[max_idx] > 1:
                    split[max_idx] -= 1

        self._cached_split = split
        logger.info(
            "[PPScheduleAdapter] layer_split: num_layers=%d, world_size=%d, "
            "split=%s (budgets_GB=%s)",
            self.num_layers, world_size, split,
            [round(b / (1 << 30), 1) for b in budgets],
        )
        return list(self._cached_split)

    def forward_backward(
        self,
        *,
        data_iterator,
        model,
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        forward_step_func: Optional[Callable] = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        decoder_seq_length: Optional[int] = None,
        first_val_step: Optional[bool] = None,
        pg_collection=None,
    ) -> dict:
        """Dispatch to the appropriate pipeline schedule.

        Selects among:
          - ``forward_backward_no_pipelining`` when PP=1.
          - ``forward_backward_pipelining_without_interleaving_pp5_heterogeneous``
            when PP=5 and the tier map contains fast ranks.
          - ``forward_backward_pipelining_without_interleaving`` otherwise.

        Args:
            data_iterator:     Iterator yielding micro-batches for this stage.
            model:             Model module(s) for this pipeline stage.
            num_microbatches:  Number of micro-batches in this global batch.
            seq_length:        Token sequence length.
            micro_batch_size:  Batch size per micro-batch.
            forward_step_func: Callable(data_iter, model) → (output, loss_fn).
                               If None a default identity fn is used (for tests).
            forward_only:      If True, skip backward pass (inference).
            collect_non_loss_data: Collect extra tensors (e.g. logits).
            decoder_seq_length: Decoder sequence length for encoder-decoder.
            first_val_step:    First validation step flag (for KV-cache flush).
            pg_collection:     Process group collection for custom collectives.

        Returns:
            forward_data_store — list of loss tensors from the last stage.
        """
        try:
            from deepspeed.core.pipeline_parallel.schedules import (
                forward_backward_no_pipelining,
                forward_backward_pipelining_without_interleaving,
                forward_backward_pipelining_without_interleaving_pp5_heterogeneous,
            )
        except ImportError as exc:
            raise ImportError(
                "PPScheduleAdapter requires deepspeed.core.pipeline_parallel.schedules. "
                f"Import failed: {exc}"
            ) from exc

        world_size = self.tier_map.world_size

        # Default forward_step_func for tests / single-stage use.
        if forward_step_func is None:
            def forward_step_func(data_iter, m):
                batch = next(data_iter)
                output = m(batch)
                return output, lambda o: o.mean()

        # ── PP=1: no pipeline ─────────────────────────────────────
        if world_size <= 1:
            logger.debug("[PPScheduleAdapter] PP=1: forward_backward_no_pipelining")
            return forward_backward_no_pipelining(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model if isinstance(model, list) else [model],
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                forward_only=forward_only,
                collect_non_loss_data=collect_non_loss_data,
                decoder_seq_length=decoder_seq_length,
                first_val_step=first_val_step,
            )

        # ── PP=5 heterogeneous path ───────────────────────────────
        if world_size == 5 and self._has_fast_ranks():
            bubble_filler = self._build_bubble_filler()
            logger.debug(
                "[PPScheduleAdapter] PP=5 hetero: "
                "forward_backward_pipelining_without_interleaving_pp5_heterogeneous "
                "(bubble_filler=%s)",
                bubble_filler is not None,
            )
            return forward_backward_pipelining_without_interleaving_pp5_heterogeneous(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model if isinstance(model, list) else [model],
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                decoder_seq_length=decoder_seq_length,
                forward_only=forward_only,
                collect_non_loss_data=collect_non_loss_data,
                first_val_step=first_val_step,
                pg_collection=pg_collection,
                bubble_filler=bubble_filler,
            )

        # ── General PP>1 path ─────────────────────────────────────
        logger.debug(
            "[PPScheduleAdapter] PP=%d: "
            "forward_backward_pipelining_without_interleaving",
            world_size,
        )
        return forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model if isinstance(model, list) else [model],
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
            pg_collection=pg_collection,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


    def detect_layer_imbalance(
        self,
        stage_times: "list[float]",
        threshold: float = 0.20,
    ) -> "dict":
        """Detect load imbalance across pipeline stages and suggest rebalancing.

        Measures the coefficient of variation (CV = std/mean) of per-stage
        forward pass times.  If any stage is more than threshold faster or
        slower than the mean, suggests a layer count adjustment.

        Args:
            stage_times:  List of per-stage measured forward pass times (ms).
                          Length must equal world_size.
            threshold:    Fraction above mean that triggers rebalancing (0.20 = 20%).

        Returns:
            dict with keys:
              "imbalanced" (bool): True if CV > threshold.
              "cv" (float): Coefficient of variation.
              "slowest_stage" (int): Index of the slowest stage.
              "suggested_split" (list[int]): Adjusted layer counts (or None if balanced).
        """
        import math

        world_size = self.tier_map.world_size
        if len(stage_times) != world_size:
            raise ValueError(
                f"stage_times length {len(stage_times)} != world_size {world_size}"
            )

        mean_t = sum(stage_times) / world_size
        if mean_t <= 0:
            return {"imbalanced": False, "cv": 0.0, "slowest_stage": -1, "suggested_split": None}

        variance = sum((t - mean_t) ** 2 for t in stage_times) / world_size
        cv = math.sqrt(variance) / mean_t

        slowest = max(range(world_size), key=lambda i: stage_times[i])
        fastest = min(range(world_size), key=lambda i: stage_times[i])

        if cv <= threshold:
            return {"imbalanced": False, "cv": cv, "slowest_stage": slowest, "suggested_split": None}

        # Suggest adjustment: redistribute layers proportional to 1/time
        # (faster stages should hold more layers)
        inv_times = [1.0 / max(t, 1e-9) for t in stage_times]
        total_inv = sum(inv_times)
        current_split = self.layer_split()

        suggested: list = []
        assigned = 0
        for i in range(world_size - 1):
            count = int(self.num_layers * inv_times[i] / total_inv)
            count = max(1, count)
            suggested.append(count)
            assigned += count
        suggested.append(max(1, self.num_layers - assigned))

        # Correct overshoot
        overshoot = sum(suggested) - self.num_layers
        if overshoot > 0:
            for _ in range(overshoot):
                max_idx = suggested.index(max(suggested))
                if suggested[max_idx] > 1:
                    suggested[max_idx] -= 1

        logger.info(
            "[PPScheduleAdapter] detect_layer_imbalance: CV=%.3f > threshold=%.2f. "
            "Slowest stage=%d (%.1fms), fastest=%d (%.1fms). "
            "Current split=%s → suggested=%s",
            cv, threshold, slowest, stage_times[slowest],
            fastest, stage_times[fastest],
            current_split, suggested,
        )
        return {
            "imbalanced": True,
            "cv": cv,
            "slowest_stage": slowest,
            "suggested_split": suggested,
        }

    def rebalance_layers(self, suggested_split: "list[int]") -> None:
        """Apply a new layer split and invalidate the cached split.

        Validates the proposed split (sum == num_layers, all >= 1) then
        updates the cached value so subsequent layer_split() calls return
        the new assignment.

        Args:
            suggested_split: New per-stage layer counts from detect_layer_imbalance().

        Raises:
            ValueError: If the split is invalid.
        """
        if sum(suggested_split) != self.num_layers:
            raise ValueError(
                f"suggested_split sum {sum(suggested_split)} != num_layers {self.num_layers}"
            )
        if any(c < 1 for c in suggested_split):
            raise ValueError("All stages must have at least 1 layer.")
        if len(suggested_split) != self.tier_map.world_size:
            raise ValueError(
                f"suggested_split length {len(suggested_split)} != world_size "
                f"{self.tier_map.world_size}"
            )

        old_split = self._cached_split
        self._cached_split = list(suggested_split)
        logger.info(
            "[PPScheduleAdapter] rebalance_layers: %s → %s",
            old_split, suggested_split,
        )

    def estimate_bubble_fraction(self, num_microbatches: int) -> float:
        """Estimate the pipeline bubble fraction for the current schedule.

        For 1F1B without interleaving:
          bubble_fraction = (num_stages - 1) / num_microbatches

        For the heterogeneous PP=5 path with HeterogeneousBubbleFiller
        injecting extra_mb=2 forward passes per fast rank:
          effective_bubble ≈ max(0, (num_stages - 1 - 2) / num_microbatches)

        Args:
            num_microbatches: Number of micro-batches in the global batch.

        Returns:
            float: Estimated bubble fraction in [0, 1].
        """
        world_size = self.tier_map.world_size
        if world_size <= 1:
            return 0.0
        if num_microbatches <= 0:
            return 1.0

        # Heterogeneous PP=5: bubble filler provides 2 extra micro-batches
        # per bubble on fast stages, reducing effective bubble.
        if world_size == 5 and self._has_fast_ranks():
            extra_mb = 2  # matches _build_bubble_filler extra_mb=2
            effective_stages = max(1, world_size - 1 - extra_mb)
        else:
            effective_stages = world_size - 1

        bubble = effective_stages / max(num_microbatches, effective_stages)
        logger.debug(
            "[PPScheduleAdapter] estimate_bubble_fraction: world_size=%d, "
            "num_microbatches=%d, bubble=%.3f",
            world_size, num_microbatches, bubble,
        )
        return min(1.0, max(0.0, bubble))

    def _has_fast_ranks(self) -> bool:
        """Return True if any rank in the cluster is H100 or Blackwell."""
        from .tier_map import GPUTier
        for r in range(self.tier_map.world_size):
            if self.tier_map.tier_of(r) in (GPUTier.H100, GPUTier.BLACKWELL):
                return True
        return False

    def _build_bubble_filler(self):
        """Construct a HeterogeneousBubbleFiller for the 5-GPU cluster.

        Returns None if the class is unavailable (falls back to standard 1F1B).
        """
        try:
            from deepspeed.core.pipeline_parallel.schedules import (
                HeterogeneousBubbleFiller,
            )
        except ImportError:
            logger.debug(
                "[PPScheduleAdapter] HeterogeneousBubbleFiller not available; "
                "using standard 1F1B."
            )
            return None

        from .tier_map import GPUTier

        # Identify fast ranks (H100, Blackwell) and slow ranks (A6000).
        fast_ranks = set()
        slow_ranks = set()
        for r in range(self.tier_map.world_size):
            tier = self.tier_map.tier_of(r)
            if tier in (GPUTier.H100, GPUTier.BLACKWELL):
                fast_ranks.add(r)
            else:
                slow_ranks.add(r)

        try:
            filler = HeterogeneousBubbleFiller(
                fast_ranks=list(fast_ranks),
                slow_ranks=list(slow_ranks),
                # extra_mb=2: fast ranks fill 2 extra forward passes per bubble.
                # Derived from: bubble_fraction=4/M, fast_rank_headroom~2×.
                extra_mb=2,
            )
            logger.info(
                "[PPScheduleAdapter] HeterogeneousBubbleFiller: "
                "fast_ranks=%s, slow_ranks=%s, extra_mb=2",
                sorted(fast_ranks), sorted(slow_ranks),
            )
            return filler
        except Exception as exc:
            logger.warning(
                "[PPScheduleAdapter] HeterogeneousBubbleFiller init failed (%s); "
                "falling back to standard 1F1B.", exc
            )
            return None
