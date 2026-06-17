# Copyright (c) 2026 Neuron_SP Project Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DES-LOC Heterogeneous Gradient Norm Skip Controller
====================================================

Upstream Design Intent (Megatron commit 180131620a):
----------------------------------------------------
Megatron-LM introduced a mechanism to skip gradient updates entirely when the
global gradient norm exceeds a configurable threshold (``grad_norm_skip_threshold``).
The motivation is training stability: extremely large gradient norms (e.g., caused
by numerical instability, data spikes, or model divergence) can corrupt optimizer
state (Adam's first/second moment estimates) irreversibly if a step is taken.
Skipping the update preserves optimizer state integrity and allows training to
recover naturally in subsequent iterations.

The upstream implementation works within ``ChainedOptimizer.step()``:
  1. Compute ``grad_norm`` across all chained sub-optimizers.
  2. For each sub-optimizer, if ``grad_norm > optimizer.config.grad_norm_skip_threshold``,
     set ``should_skip_update = True``.
  3. Replace ``update_successful = self.step_with_ready_grads()`` with a conditional
     that short-circuits to ``False`` when the skip flag is set.
  4. The ``OptimizerConfig`` dataclass gains ``grad_norm_skip_threshold: float = float('inf')``
     so the feature is inert by default.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs heterogeneous hardware:
  - 2× A6000 48 GB (SM86, PCIe)   — named "anchor" devices
  - 1× H100 NVL 96 GB (SM90, PCIe) — named "compute" device

Key challenges absent from the homogeneous Megatron setting:

1. **Device-Asymmetric Norm Accumulation**: Gradient shards live on different devices
   with different FP precision capabilities (BF16 native on H100, FP16/FP32 on A6000).
   A single global norm computed naively may reflect precision artifacts rather than
   true gradient magnitude. We accumulate partial norms per device class, promote to
   FP64 on CPU (where 1.5 TB DRAM allows cheap accumulation), then combine.

2. **LOC-Cache Coherence on Skip**: The Shared LOcality Cache (LOC) holds gradient
   fragments that are in-flight between anchor↔compute PCIe transfers. On a skip
   event, in-flight fragments must be explicitly invalidated rather than allowed to
   accumulate stale state into the next step's norm. ``HeteroGradNormSkipController``
   exposes a ``invalidate_loc_cache()`` hook for this purpose.

3. **Threshold Heterogeneity**: Because anchor (A6000) and compute (H100) devices may
   accumulate different numeric ranges due to SM architecture differences (e.g., H100
   TF32 accumulation vs. A6000 FP32), we allow per-device-class skip thresholds rather
   than a single global scalar, while still exposing a unified ``effective_threshold``
   for logging and configuration.

4. **Skip Counting and Consecutive-Skip Detection**: In heterogeneous PCIe-connected
   training without NVLink, a skip event is more expensive than in NVLink clusters
   because the next step's gradient redistribution must re-pipeline from scratch.
   We track consecutive skips and emit a WARNING when a configurable run of consecutive
   skips is detected, suggesting the user should inspect loss curves or reduce LR.

5. **Grad Norm History for Adaptive Thresholding**: Optionally maintain an exponential
   moving average (EMA) of grad norms so that the effective threshold can be expressed
   as a multiple of the EMA, adapting to the training phase (warm-up vs. steady state).

Module Structure:
-----------------
  HeteroGradNormConfig        — dataclass mirroring OptimizerConfig additions
  DeviceClass                 — enum for anchor vs. compute device roles
  LOCCacheHandle              — minimal protocol for LOC cache invalidation
  PartialNormAccumulator      — per-device-class FP64 norm accumulator on CPU DRAM
  HeteroGradNormSkipController — main controller, drop-in around DeepSpeed step()
  integrate_with_deepspeed_engine() — convenience wiring into ds_engine

Usage:
------
  from deepspeed.runtime.hetero_grad_norm_skip import (
      HeteroGradNormConfig,
      HeteroGradNormSkipController,
      DeviceClass,
  )

  config = HeteroGradNormConfig(
      anchor_skip_threshold=50.0,
      compute_skip_threshold=80.0,
      consecutive_skip_warn_limit=3,
      use_adaptive_threshold=True,
      adaptive_ema_beta=0.98,
  )
  controller = HeteroGradNormSkipController(config, loc_cache_handle=my_loc_cache)

  # Inside your training loop:
  skip, info = controller.should_skip(anchor_grads, compute_grads)
  if not skip:
      engine.step()
  controller.record_step(skipped=skip, grad_norm=info["combined_norm"])
"""

from __future__ import annotations

import enum
import logging
import math
import time
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations and Protocols
# ---------------------------------------------------------------------------


class DeviceClass(enum.Enum):
    """Role classification for heterogeneous hardware in DES-LOC.

    ANCHOR corresponds to the 2× A6000 48 GB (SM86) nodes that hold parameter
    replicas and serve as the persistent gradient accumulation endpoints.
    COMPUTE corresponds to the 1× H100 NVL 96 GB (SM90) node that executes
    the bulk of forward/backward computation.
    """

    ANCHOR = "anchor"    # A6000 SM86
    COMPUTE = "compute"  # H100 SM90


class LOCCacheHandle(Protocol):
    """Protocol for the Shared LOcality Cache used in DES-LOC.

    The LOC cache buffers gradient fragments during PCIe transfers between
    anchor and compute devices.  On a grad-norm skip event the cache must be
    invalidated so stale fragments do not poison the next step's norm.
    """

    def invalidate(self, reason: str) -> None:
        """Flush all pending gradient fragments from the cache.

        Args:
            reason: Human-readable string explaining why the cache is being
                    invalidated; written to cache-internal telemetry if
                    available.
        """
        ...

    def fragment_count(self) -> int:
        """Return the number of gradient fragments currently resident."""
        ...


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------


@dataclass
class HeteroGradNormConfig:
    """Configuration for heterogeneous gradient norm skip behavior in DES-LOC.

    Mirrors the ``grad_norm_skip_threshold`` field added to Megatron's
    ``OptimizerConfig`` (commit 180131620a) but extends it to support
    per-device-class thresholds and adaptive EMA-based thresholding.

    Attributes:
        anchor_skip_threshold: Skip threshold applied to the gradient norm
            computed from ANCHOR (A6000) device shards.  Defaults to inf
            (disabled), matching Megatron upstream behaviour.
        compute_skip_threshold: Skip threshold applied to the gradient norm
            computed from COMPUTE (H100) device shards.  Defaults to inf.
        combined_skip_threshold: Skip threshold applied to the combined
            (cross-device) gradient norm.  A skip is triggered if ANY of the
            active thresholds are exceeded.  Defaults to inf.
        consecutive_skip_warn_limit: Emit a WARNING log after this many
            consecutive skip events.  Set to 0 to disable.  Defaults to 5.
        use_adaptive_threshold: If True, the effective thresholds are computed
            as ``base_threshold * adaptive_multiplier`` where ``base_threshold``
            is the EMA of historical norms.  The raw threshold fields then act
            as *multipliers* (e.g., ``anchor_skip_threshold=3.0`` means "skip
            if norm > 3× EMA").  Defaults to False to preserve standard
            behaviour.
        adaptive_ema_beta: EMA decay factor for norm history when
            ``use_adaptive_threshold=True``.  Defaults to 0.98 (slow-moving
            baseline suited to stable training phases).
        adaptive_warmup_steps: Number of steps before adaptive threshold
            becomes active; during warm-up the raw thresholds are used
            directly.  Defaults to 100.
        norm_type: p-norm exponent.  2 for L2, float('inf') for L-inf.
            Defaults to 2.
        log_per_device_norms: If True, log anchor and compute partial norms
            separately in addition to the combined norm.  Useful for debugging
            device-class imbalances.  Defaults to False.
    """

    anchor_skip_threshold: float = float("inf")
    compute_skip_threshold: float = float("inf")
    combined_skip_threshold: float = float("inf")
    consecutive_skip_warn_limit: int = 5
    use_adaptive_threshold: bool = False
    adaptive_ema_beta: float = 0.98
    adaptive_warmup_steps: int = 100
    norm_type: Union[int, float] = 2
    log_per_device_norms: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.adaptive_ema_beta < 1.0):
            raise ValueError(
                f"adaptive_ema_beta must be in (0, 1), got {self.adaptive_ema_beta}"
            )
        if self.adaptive_warmup_steps < 0:
            raise ValueError(
                f"adaptive_warmup_steps must be >= 0, got {self.adaptive_warmup_steps}"
            )
        if self.norm_type != float("inf") and self.norm_type <= 0:
            raise ValueError(f"norm_type must be positive or inf, got {self.norm_type}")

    @property
    def any_threshold_finite(self) -> bool:
        """Return True if at least one threshold is finite (feature is active)."""
        return any(
            math.isfinite(t)
            for t in (
                self.anchor_skip_threshold,
                self.compute_skip_threshold,
                self.combined_skip_threshold,
            )
        )


# ---------------------------------------------------------------------------
# Partial Norm Accumulator
# ---------------------------------------------------------------------------


class PartialNormAccumulator:
    """Accumulates gradient p-norms per device class in FP64 on CPU DRAM.

    Motivation (DES-LOC specific):
        Because anchor (A6000 SM86) and compute (H100 SM90) have different
        native precision behaviours (e.g., H100 uses TF32 accumulation in
        CUDA cores by default, while A6000 defaults to FP32), computing a
        single norm in device-native precision and then reducing can introduce
        bias.  By casting each device's partial norm to FP64 before summing
        on the host CPU (backed by 1.5 TB DRAM), we eliminate cross-device
        precision artifacts.

    The accumulator is reset at the start of each step via ``reset()`` and
    yields the final combined norm via ``finalize(norm_type)``.
    """

    def __init__(self) -> None:
        self._partials: Dict[DeviceClass, float] = {
            DeviceClass.ANCHOR: 0.0,
            DeviceClass.COMPUTE: 0.0,
        }
        self._counts: Dict[DeviceClass, int] = {
            DeviceClass.ANCHOR: 0,
            DeviceClass.COMPUTE: 0,
        }

    def reset(self) -> None:
        """Clear all accumulated partial norms for a new step."""
        for dc in DeviceClass:
            self._partials[dc] = 0.0
            self._counts[dc] = 0

    def accumulate(
        self,
        grads: Sequence[torch.Tensor],
        device_class: DeviceClass,
        norm_type: Union[int, float],
    ) -> float:
        """Compute the partial p-norm for a list of gradients and accumulate it.

        Gradients are cast to FP32 on their native device, the per-tensor
        norms are summed (appropriately for the p-norm type), then the result
        is transferred to CPU as FP64.

        Args:
            grads: Gradient tensors residing on the device class's hardware.
            device_class: Which device class these gradients belong to.
            norm_type: The p in the p-norm.

        Returns:
            The partial norm contribution added in this call (as a Python float).
        """
        if not grads:
            return 0.0

        partial: float
        if norm_type == float("inf"):
            # L-inf: max of absolute values across all tensors.
            max_val = max(
                g.detach().float().abs().max().item()
                for g in grads
                if g is not None
            )
            # For L-inf, combine by taking the running max.
            self._partials[device_class] = max(self._partials[device_class], max_val)
            partial = max_val
        else:
            # L-p: accumulate sum of ||g||_p^p, finalize later with 1/p power.
            sum_of_pow = sum(
                g.detach().float().norm(p=norm_type).double().item() ** norm_type
                for g in grads
                if g is not None
            )
            self._partials[device_class] += sum_of_pow
            partial = sum_of_pow

        self._counts[device_class] += len([g for g in grads if g is not None])
        return partial

    def partial_norm(self, device_class: DeviceClass, norm_type: Union[int, float]) -> float:
        """Return the finalized partial norm for a single device class.

        Args:
            device_class: Which device class to query.
            norm_type: The p in the p-norm (needed to apply the 1/p power for
                       L-p norms; L-inf is already finalized during accumulation).

        Returns:
            The finalized partial norm as a Python float.
        """
        raw = self._partials[device_class]
        if norm_type == float("inf"):
            return raw
        return raw ** (1.0 / norm_type)

    def finalize(self, norm_type: Union[int, float]) -> float:
        """Combine anchor and compute partial norms into a single global norm.

        For L-p norms, the combined norm is::

            combined = (anchor_partial^p + compute_partial^p)^(1/p)

        For L-inf, it is::

            combined = max(anchor_max, compute_max)

        Returns:
            The combined global gradient norm.
        """
        if norm_type == float("inf"):
            return max(self._partials[DeviceClass.ANCHOR], self._partials[DeviceClass.COMPUTE])
        combined_raw = (
            self._partials[DeviceClass.ANCHOR] + self._partials[DeviceClass.COMPUTE]
        )
        return combined_raw ** (1.0 / norm_type)

    def grad_counts(self) -> Dict[DeviceClass, int]:
        """Return the number of non-None gradients seen per device class."""
        return dict(self._counts)


# ---------------------------------------------------------------------------
# Adaptive Threshold Manager
# ---------------------------------------------------------------------------


class AdaptiveThresholdManager:
    """Maintains an EMA of gradient norms to support adaptive skip thresholds.

    When ``HeteroGradNormConfig.use_adaptive_threshold=True``, the effective
    threshold is ``base_norm_ema * multiplier`` where ``multiplier`` comes from
    the per-device-class threshold field.  This allows the skip threshold to
    track the training dynamics without manual tuning.

    The EMA is only updated on non-skipped steps to avoid the EMA being
    suppressed by a sequence of artificially low norm values that result from
    the skips themselves.
    """

    def __init__(self, beta: float, warmup_steps: int) -> None:
        self._beta = beta
        self._warmup_steps = warmup_steps
        self._ema: Optional[float] = None
        self._step_count: int = 0
        self._bias_correction_denom: float = 0.0

    def update(self, norm: float) -> None:
        """Update the EMA with a new (non-skipped) norm observation.

        Args:
            norm: The global combined gradient norm from a non-skipped step.
        """
        if self._ema is None:
            self._ema = norm
            self._bias_correction_denom = 1.0 - self._beta
        else:
            self._ema = self._beta * self._ema + (1.0 - self._beta) * norm
            self._bias_correction_denom = (
                self._bias_correction_denom * self._beta + (1.0 - self._beta)
            )
        self._step_count += 1

    @property
    def is_warmed_up(self) -> bool:
        """True once enough non-skip steps have elapsed for reliable EMA."""
        return self._step_count >= self._warmup_steps

    @property
    def bias_corrected_ema(self) -> Optional[float]:
        """Return the bias-corrected EMA, or None if not yet initialized."""
        if self._ema is None or self._bias_correction_denom == 0.0:
            return None
        return self._ema / self._bias_correction_denom

    def effective_threshold(self, multiplier: float) -> float:
        """Compute the effective threshold as ``EMA * multiplier``.

        Falls back to ``multiplier`` directly if the EMA is not yet available
        (warm-up phase), preserving the raw threshold semantics during start-up.

        Args:
            multiplier: The per-device-class threshold value from
                        ``HeteroGradNormConfig``, reinterpreted as a multiplier.

        Returns:
            The effective threshold to compare against the observed norm.
        """
        if not self.is_warmed_up or self.bias_corrected_ema is None:
            return multiplier  # raw threshold during warm-up
        return self.bias_corrected_ema * multiplier

    @property
    def step_count(self) -> int:
        return self._step_count


# ---------------------------------------------------------------------------
# Skip Decision Record
# ---------------------------------------------------------------------------


@dataclass
class GradNormSkipInfo:
    """Structured result from a single skip-decision evaluation.

    Attributes:
        skipped: Whether the gradient update should be skipped.
        combined_norm: The combined global gradient norm (anchor + compute).
        anchor_norm: Partial norm from ANCHOR device class.
        compute_norm: Partial norm from COMPUTE device class.
        triggered_by: Which device class (or "combined") triggered the skip,
                      or None if no skip occurred.
        effective_anchor_threshold: The threshold actually compared against
                                    anchor_norm.
        effective_compute_threshold: The threshold actually compared against
                                     compute_norm.
        effective_combined_threshold: The threshold actually compared against
                                      combined_norm.
        loc_cache_invalidated: Whether the LOC cache was invalidated as part
                               of this skip event.
        evaluation_time_us: Wall-clock time spent in norm computation (µs).
    """

    skipped: bool
    combined_norm: float
    anchor_norm: float
    compute_norm: float
    triggered_by: Optional[str]
    effective_anchor_threshold: float
    effective_compute_threshold: float
    effective_combined_threshold: float
    loc_cache_invalidated: bool = False
    evaluation_time_us: float = 0.0


# ---------------------------------------------------------------------------
# Main Controller
# ---------------------------------------------------------------------------


class HeteroGradNormSkipController:
    """Controls gradient update skipping for DES-LOC heterogeneous training.

    This class is the DES-LOC reinterpretation of the skip logic introduced in
    Megatron commit 180131620a.  It wraps the norm computation and threshold
    comparison logic that in Megatron lives inside ``ChainedOptimizer.step()``,
    lifting it into a standalone, device-aware controller suited for DeepSpeed
    engines running across heterogeneous PCIe-connected hardware.

    Key behavioral differences from upstream Megatron:

    1. Per-device-class thresholds (anchor vs. compute vs. combined).
    2. FP64 norm accumulation on CPU DRAM to avoid cross-device precision bias.
    3. LOC cache invalidation on skip events to prevent stale gradient fragments
       from corrupting subsequent steps.
    4. Consecutive-skip detection and WARNING emission.
    5. Optional adaptive thresholding via EMA of historical norms.
    6. Distributed-aware norm reduction: if a process group is provided, partial
       norms are all-reduced before the final decision.

    Thread Safety:
        Not thread-safe.  Assumes single-threaded step coordination, consistent
        with DeepSpeed's engine loop.

    Args:
        config: ``HeteroGradNormConfig`` instance controlling thresholds and
                adaptive behaviour.
        loc_cache_handle: Optional ``LOCCacheHandle`` instance for cache
                          invalidation on skip.  If None, cache invalidation
                          is skipped silently.
        process_group: Optional ``dist.ProcessGroup`` for all-reducing partial
                       norms across data-parallel ranks.  If None, no reduction
                       is performed (single-node or already-reduced gradients).
    """

    def __init__(
        self,
        config: HeteroGradNormConfig,
        loc_cache_handle: Optional[LOCCacheHandle] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self._config = config
        self._loc = loc_cache_handle
        self._pg = process_group
        self._accumulator = PartialNormAccumulator()
        self._adaptive_mgr = AdaptiveThresholdManager(
            beta=config.adaptive_ema_beta,
            warmup_steps=config.adaptive_warmup_steps,
        )
        self._total_steps: int = 0
        self._total_skips: int = 0
        self._consecutive_skips: int = 0
        self._last_skip_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_skip(
        self,
        anchor_grads: Sequence[Optional[torch.Tensor]],
        compute_grads: Sequence[Optional[torch.Tensor]],
    ) -> Tuple[bool, GradNormSkipInfo]:
        """Evaluate whether the current step's gradient update should be skipped.

        This is the primary entry point.  Call it after gradients are ready
        (i.e., after ``loss.backward()`` and any gradient accumulation) but
        before calling ``engine.step()`` or ``optimizer.step()``.

        The method:
          1. Accumulates partial norms per device class via ``PartialNormAccumulator``.
          2. Optionally all-reduces partial norms across data-parallel ranks.
          3. Computes effective thresholds (raw or adaptive).
          4. Determines whether any threshold is exceeded.
          5. If skipping, invalidates the LOC cache and increments skip counters.
          6. Emits appropriate log messages.

        Args:
            anchor_grads: Gradient tensors residing on ANCHOR (A6000) devices.
                          May contain None entries (for parameters without grads).
            compute_grads: Gradient tensors residing on COMPUTE (H100) device.
                           May contain None entries.

        Returns:
            A tuple ``(should_skip, info)`` where ``should_skip`` is a bool and
            ``info`` is a ``GradNormSkipInfo`` with full diagnostic detail.
        """
        t0 = time.perf_counter()

        self._accumulator.reset()

        # Filter None grads before accumulation.
        valid_anchor = [g for g in anchor_grads if g is not None]
        valid_compute = [g for g in compute_grads if g is not None]

        norm_type = self._config.norm_type

        self._accumulator.accumulate(valid_anchor, DeviceClass.ANCHOR, norm_type)
        self._accumulator.accumulate(valid_compute, DeviceClass.COMPUTE, norm_type)

        # All-reduce partial norm^p sums across data-parallel ranks if needed.
        self._maybe_allreduce_partials(norm_type)

        anchor_norm = self._accumulator.partial_norm(DeviceClass.ANCHOR, norm_type)
        compute_norm = self._accumulator.partial_norm(DeviceClass.COMPUTE, norm_type)
        combined_norm = self._accumulator.finalize(norm_type)

        if self._config.log_per_device_norms:
            logger.debug(
                "DES-LOC grad norms — anchor: %.6f  compute: %.6f  combined: %.6f",
                anchor_norm,
                compute_norm,
                combined_norm,
            )

        # Resolve effective thresholds.
        eff_anchor_thr = self._effective_threshold(self._config.anchor_skip_threshold)
        eff_compute_thr = self._effective_threshold(self._config.compute_skip_threshold)
        eff_combined_thr = self._effective_threshold(self._config.combined_skip_threshold)

        # Skip decision: any exceeded threshold triggers a skip.
        triggered_by: Optional[str] = None
        if math.isfinite(eff_anchor_thr) and anchor_norm > eff_anchor_thr:
            triggered_by = "anchor"
        elif math.isfinite(eff_compute_thr) and compute_norm > eff_compute_thr:
            triggered_by = "compute"
        elif math.isfinite(eff_combined_thr) and combined_norm > eff_combined_thr:
            triggered_by = "combined"

        skipped = triggered_by is not None
        loc_invalidated = False

        if skipped:
            loc_invalidated = self._handle_skip(
                triggered_by=triggered_by,  # type: ignore[arg-type]
                combined_norm=combined_norm,
                anchor_norm=anchor_norm,
                compute_norm=compute_norm,
                eff_anchor_thr=eff_anchor_thr,
                eff_compute_thr=eff_compute_thr,
                eff_combined_thr=eff_combined_thr,
            )

        elapsed_us = (time.perf_counter() - t0) * 1e6

        info = GradNormSkipInfo(
            skipped=skipped,
            combined_norm=combined_norm,
            anchor_norm=anchor_norm,
            compute_norm=compute_norm,
            triggered_by=triggered_by,
            effective_anchor_threshold=eff_anchor_thr,
            effective_compute_threshold=eff_compute_thr,
            effective_combined_threshold=eff_combined_thr,
            loc_cache_invalidated=loc_invalidated,
            evaluation_time_us=elapsed_us,
        )
        return skipped, info

    def record_step(self, *, skipped: bool, grad_norm: float) -> None:
        """Update internal counters after a training step decision.

        Must be called once per step regardless of whether the step was skipped.
        On non-skipped steps, updates the EMA baseline if adaptive thresholding
        is enabled.  On skipped steps, checks the consecutive-skip limit.

        Args:
            skipped: Whether the step was skipped.
            grad_norm: The combined gradient norm from ``should_skip``'s info.
        """
        self._total_steps += 1
        if skipped:
            self._total_skips += 1
            self._consecutive_skips += 1
            self._last_skip_step = self._total_steps

            limit = self._config.consecutive_skip_warn_limit
            if limit > 0 and self._consecutive_skips >= limit:
                logger.warning(
                    "DES-LOC: %d consecutive gradient update skips detected "
                    "(steps %d–%d). Combined grad norm at last skip: %.4f. "
                    "Consider inspecting loss curves, reducing learning rate, "
                    "or adjusting skip thresholds.",
                    self._consecutive_skips,
                    self._total_steps - self._consecutive_skips + 1,
                    self._total_steps,
                    grad_norm,
                )
        else:
            self._consecutive_skips = 0
            if self._config.use_adaptive_threshold:
                self._adaptive_mgr.update(grad_norm)

    @property
    def skip_rate(self) -> float:
        """Fraction of total steps that were skipped (0.0 if no steps yet)."""
        if self._total_steps == 0:
            return 0.0
        return self._total_skips / self._total_steps

    @property
    def statistics(self) -> Dict[str, object]:
        """Return a snapshot of controller statistics for monitoring/logging."""
        return {
            "total_steps": self._total_steps,
            "total_skips": self._total_skips,
            "consecutive_skips": self._consecutive_skips,
            "skip_rate": self.skip_rate,
            "last_skip_step": self._last_skip_step,
            "adaptive_ema": self._adaptive_mgr.bias_corrected_ema,
            "adaptive_warmed_up": self._adaptive_mgr.is_warmed_up,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_threshold(self, raw_threshold: float) -> float:
        """Resolve the effective comparison threshold.

        In raw mode (``use_adaptive_threshold=False``), returns ``raw_threshold``
        unchanged.  In adaptive mode, treats ``raw_threshold`` as a multiplier
        over the EMA baseline.

        Args:
            raw_threshold: The threshold value from ``HeteroGradNormConfig``.

        Returns:
            The threshold to compare against the observed norm.
        """
        if not self._config.use_adaptive_threshold:
            return raw_threshold
        return self._adaptive_mgr.effective_threshold(raw_threshold)

    def _handle_skip(
        self,
        *,
        triggered_by: str,
        combined_norm: float,
        anchor_norm: float,
        compute_norm: float,
        eff_anchor_thr: float,
        eff_compute_thr: float,
        eff_combined_thr: float,
    ) -> bool:
        """Execute skip-event side effects and return whether LOC was invalidated.

        Side effects:
          - Emits an INFO log with norm and threshold details.
          - Invalidates the LOC cache if a handle is configured and the cache
            has resident fragments.

        Args:
            triggered_by: "anchor", "compute", or "combined".
            combined_norm: The combined gradient norm value.
            anchor_norm: Anchor partial norm.
            compute_norm: Compute partial norm.
            eff_anchor_thr: Effective anchor threshold.
            eff_compute_thr: Effective compute threshold.
            eff_combined_thr: Effective combined threshold.

        Returns:
            True if the LOC cache was invalidated, False otherwise.
        """
        logger.info(
            "DES-LOC: Skipping gradient update — triggered by '%s' device class. "
            "combined_norm=%.6f (thr=%.6f), anchor_norm=%.6f (thr=%.6f), "
            "compute_norm=%.6f (thr=%.6f).",
            triggered_by,
            combined_norm,
            eff_combined_thr,
            anchor_norm,
            eff_anchor_thr,
            compute_norm,
            eff_compute_thr,
        )

        loc_invalidated = False
        if self._loc is not None:
            fragment_count = self._loc.fragment_count()
            if fragment_count > 0:
                reason = (
                    f"grad_norm_skip triggered_by={triggered_by} "
                    f"combined_norm={combined_norm:.4f}"
                )
                self._loc.invalidate(reason)
                loc_invalidated = True
                logger.info(
                    "DES-LOC: LOC cache invalidated (%d fragments flushed) "
                    "to prevent stale gradient fragments in next step.",
                    fragment_count,
                )

        return loc_invalidated

    def _maybe_allreduce_partials(self, norm_type: Union[int, float]) -> None:
        """All-reduce partial norm^p sums across data-parallel ranks.

        For L-p norms, we reduce the raw sum-of-powers (before taking the 1/p
        root) so that the final root is applied only once on the combined sum.
        For L-inf, we reduce the per-device-class maxima.

        This mirrors Megatron's ``get_grad_norm_fp32`` all-reduce pattern but
        operates on CPU-side FP64 tensors to avoid device synchronisation
        overhead on the PCIe-connected heterogeneous cluster.

        Args:
            norm_type: The p in the p-norm.
        """
        if self._pg is None or not dist.is_initialized():
            return

        for dc in DeviceClass:
            partial_val = self._accumulator._partials[dc]
            partial_tensor = torch.tensor([partial_val], dtype=torch.float64, device="cpu")
            if norm_type == float("inf"):
                dist.all_reduce(partial_tensor, op=dist.ReduceOp.MAX, group=self._pg)
            else:
                dist.all_reduce(partial_tensor, op=dist.ReduceOp.SUM, group=self._pg)
            self._accumulator._partials[dc] = partial_tensor.item()


# ---------------------------------------------------------------------------
# DeepSpeed Engine Integration Helpers
# ---------------------------------------------------------------------------


def integrate_with_deepspeed_engine(
    engine: object,
    config: HeteroGradNormConfig,
    anchor_param_names: Optional[List[str]] = None,
    compute_param_names: Optional[List[str]] = None,
    loc_cache_handle: Optional[LOCCacheHandle] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> HeteroGradNormSkipController:
    """Wire a ``HeteroGradNormSkipController`` into a DeepSpeed engine.

    This is a convenience function for users who want the minimal integration
    path.  It creates a controller and monkey-patches the engine's ``step``
    method to call ``should_skip`` before delegating to the original step.

    The patched step method signature is unchanged; it returns the original
    return value on non-skip, and ``None`` (matching DeepSpeed convention for
    a skipped update) on skip.

    Args:
        engine: A DeepSpeed ``DeepSpeedEngine`` instance (typed as ``object``
                to avoid a hard import dependency).
        config: The ``HeteroGradNormConfig`` to use.
        anchor_param_names: Parameter names (``module.named_parameters()`` keys)
                            whose gradients should be classified as ANCHOR.
                            If None, all parameters are treated as combined.
        compute_param_names: Parameter names classified as COMPUTE.
                             If None, all parameters not in ``anchor_param_names``
                             are treated as COMPUTE.
        loc_cache_handle: Optional LOC cache handle.
        process_group: Optional distributed process group for norm reduction.

    Returns:
        The ``HeteroGradNormSkipController`` instance (for later statistics
        queries or ``record_step`` calls from the training loop).
    """
    controller = HeteroGradNormSkipController(
        config=config,
        loc_cache_handle=loc_cache_handle,
        process_group=process_group,
    )
    original_step = engine.step  # type: ignore[attr-defined]

    def _patched_step(*args: object, **kwargs: object) -> object:
        # Gather gradients from the engine's module parameters.
        anchor_grads: List[Optional[torch.Tensor]] = []
        compute_grads: List[Optional[torch.Tensor]] = []

        module = engine.module  # type: ignore[attr-defined]
        for name, param in module.named_parameters():
            grad = param.grad
            if anchor_param_names is not None and name in anchor_param_names:
                anchor_grads.append(grad)
            elif compute_param_names is not None and name in compute_param_names:
                compute_grads.append(grad)
            else:
                # Default: treat everything as compute when no split is provided.
                compute_grads.append(grad)

        skip, info = controller.should_skip(anchor_grads, compute_grads)
        controller.record_step(skipped=skip, grad_norm=info.combined_norm)

        if skip:
            return None
        return original_step(*args, **kwargs)

    engine.step = _patched_step  # type: ignore[attr-defined]
    logger.info(
        "DES-LOC: HeteroGradNormSkipController integrated into DeepSpeed engine. "
        "anchor_thr=%.4f, compute_thr=%.4f, combined_thr=%.4f, adaptive=%s.",
        config.anchor_skip_threshold,
        config.compute_skip_threshold,
        config.combined_skip_threshold,
        config.use_adaptive_threshold,
    )
    return controller


# ---------------------------------------------------------------------------
# Standalone gradient norm utility (mirrors Megatron get_grad_norm_fp32)
# ---------------------------------------------------------------------------


def get_grad_norm_fp32_hetero(
    anchor_grads: Sequence[Optional[torch.Tensor]],
    compute_grads: Sequence[Optional[torch.Tensor]],
    norm_type: Union[int, float] = 2,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, float]:
    """Compute heterogeneous gradient norms in FP64 on CPU.

    This utility mirrors ``megatron.core.optimizer.clip_grads.get_grad_norm_fp32``
    but is adapted for DES-LOC's device-split gradient layout.  Unlike the
    upstream function which operates on a flat list of gradients and reduces
    within a process group, this function maintains per-device-class partial
    norms throughout and only combines them at the final step.

    The computation is performed in FP64 on CPU to avoid:
      - FP16 overflow on A6000 anchor devices during large gradient events.
      - TF32 accumulation bias on H100 compute devices.

    Args:
        anchor_grads: Gradient tensors on ANCHOR (A6000) devices.
        compute_grads: Gradient tensors on COMPUTE (H100) device.
        norm_type: The p in the p-norm; use ``float('inf')`` for L-inf norm.
        process_group: Optional process group for all-reducing the norms
                       across data-parallel ranks.

    Returns:
        A dict with keys "anchor_norm", "compute_norm", "combined_norm".
    """
    accumulator = PartialNormAccumulator()
    valid_anchor = [g for g in anchor_grads if g is not None]
    valid_compute = [g for g in compute_grads if g is not None]

    accumulator.accumulate(valid_anchor, DeviceClass.ANCHOR, norm_type)
    accumulator.accumulate(valid_compute, DeviceClass.COMPUTE, norm_type)

    if process_group is not None and dist.is_initialized():
        for dc in DeviceClass:
            partial_val = accumulator._partials[dc]
            partial_tensor = torch.tensor([partial_val], dtype=torch.float64, device="cpu")
            if norm_type == float("inf"):
                dist.all_reduce(partial_tensor, op=dist.ReduceOp.MAX, group=process_group)
            else:
                dist.all_reduce(partial_tensor, op=dist.ReduceOp.SUM, group=process_group)
            accumulator._partials[dc] = partial_tensor.item()

    return {
        "anchor_norm": accumulator.partial_norm(DeviceClass.ANCHOR, norm_type),
        "compute_norm": accumulator.partial_norm(DeviceClass.COMPUTE, norm_type),
        "combined_norm": accumulator.finalize(norm_type),
    }


def clip_grads_hetero(
    anchor_grads: List[Optional[torch.Tensor]],
    compute_grads: List[Optional[torch.Tensor]],
    max_norm: float,
    total_norm: float,
) -> None:
    """Clip gradients in-place using the provided total norm.

    DES-LOC adaptation of ``clip_grad_by_total_norm_fp32`` that operates on
    the split anchor/compute gradient lists.  The clip coefficient is computed
    once from ``total_norm`` (the combined heterogeneous norm) and applied to
    all tensors.

    Args:
        anchor_grads: Gradient tensors on ANCHOR devices (modified in-place).
        compute_grads: Gradient tensors on COMPUTE device (modified in-place).
        max_norm: Maximum permissible norm.
        total_norm: The combined gradient norm (from ``get_grad_norm_fp32_hetero``).
    """
    if total_norm == 0.0:
        return

    clip_coeff = max_norm / (total_norm + 1e-6)
    if clip_coeff < 1.0:
        for g in anchor_grads:
            if g is not None:
                g.detach().mul_(clip_coeff)
        for g in compute_grads:
            if g is not None:
                g.detach().mul_(clip_coeff)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    # Use unittest for structured test reporting.
    import unittest

    class TestHeteroGradNormConfig(unittest.TestCase):
        """Tests for HeteroGradNormConfig defaults and validation."""

        def test_default_thresholds_are_inf(self) -> None:
            """Mirrors Megatron test_grad_norm_skip_threshold_config: default is inf."""
            cfg = HeteroGradNormConfig()
            self.assertEqual(cfg.anchor_skip_threshold, float("inf"))
            self.assertEqual(cfg.compute_skip_threshold, float("inf"))
            self.assertEqual(cfg.combined_skip_threshold, float("inf"))

        def test_any_threshold_finite_false_by_default(self) -> None:
            cfg = HeteroGradNormConfig()
            self.assertFalse(cfg.any_threshold_finite)

        def test_any_threshold_finite_true_when_set(self) -> None:
            cfg = HeteroGradNormConfig(anchor_skip_threshold=10.0)
            self.assertTrue(cfg.any_threshold_finite)

        def test_invalid_ema_beta_raises(self) -> None:
            with self.assertRaises(ValueError):
                HeteroGradNormConfig(adaptive_ema_beta=1.5)
            with self.assertRaises(ValueError):
                HeteroGradNormConfig(adaptive_ema_beta=0.0)

        def test_invalid_norm_type_raises(self) -> None:
            with self.assertRaises(ValueError):
                HeteroGradNormConfig(norm_type=-1)

        def test_negative_warmup_steps_raises(self) -> None:
            with self.assertRaises(ValueError):
                HeteroGradNormConfig(adaptive_warmup_steps=-1)

    class TestPartialNormAccumulator(unittest.TestCase):
        """Tests for FP64 CPU norm accumulation across device classes."""

        def _make_grad(self, values: List[float], device: str = "cpu") -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32, device=device)

        def test_l2_norm_single_anchor(self) -> None:
            acc = PartialNormAccumulator()
            g = self._make_grad([3.0, 4.0])  # L2 norm = 5.0
            acc.accumulate([g], DeviceClass.ANCHOR, norm_type=2)
            result = acc.partial_norm(DeviceClass.ANCHOR, norm_type=2)
            self.assertAlmostEqual(result, 5.0, places=5)

        def test_l2_norm_combined(self) -> None:
            acc = PartialNormAccumulator()
            g_anchor = self._make_grad([3.0, 4.0])   # L2 = 5.0, sum_p2 = 25
            g_compute = self._make_grad([0.0, 12.0]) # L2 = 12.0, sum_p2 = 144
            acc.accumulate([g_anchor], DeviceClass.ANCHOR, norm_type=2)
            acc.accumulate([g_compute], DeviceClass.COMPUTE, norm_type=2)
            combined = acc.finalize(norm_type=2)
            # combined = sqrt(25 + 144) = sqrt(169) = 13.0
            self.assertAlmostEqual(combined, 13.0, places=5)

        def test_linf_norm_combined(self) -> None:
            acc = PartialNormAccumulator()
            g_anchor = self._make_grad([1.0, 7.0])
            g_compute = self._make_grad([5.0, 3.0])
            acc.accumulate([g_anchor], DeviceClass.ANCHOR, norm_type=float("inf"))
            acc.accumulate([g_compute], DeviceClass.COMPUTE, norm_type=float("inf"))
            combined = acc.finalize(norm_type=float("inf"))
            self.assertAlmostEqual(combined, 7.0, places=5)

        def test_reset_clears_state(self) -> None:
            acc = PartialNormAccumulator()
            g = self._make_grad([10.0])
            acc.accumulate([g], DeviceClass.ANCHOR, norm_type=2)
            acc.reset()
            result = acc.partial_norm(DeviceClass.ANCHOR, norm_type=2)
            self.assertAlmostEqual(result, 0.0, places=10)

        def test_none_grads_are_skipped(self) -> None:
            acc = PartialNormAccumulator()
            g = self._make_grad([3.0, 4.0])
            acc.accumulate([g, None, None], DeviceClass.COMPUTE, norm_type=2)
            result = acc.partial_norm(DeviceClass.COMPUTE, norm_type=2)
            self.assertAlmostEqual(result, 5.0, places=5)

        def test_grad_counts(self) -> None:
            acc = PartialNormAccumulator()
            grads = [self._make_grad([1.0]), None, self._make_grad([2.0])]
            acc.accumulate(grads, DeviceClass.ANCHOR, norm_type=2)
            counts = acc.grad_counts()
            self.assertEqual(counts[DeviceClass.ANCHOR], 2)

    class TestAdaptiveThresholdManager(unittest.TestCase):
        """Tests for EMA-based adaptive threshold computation."""

        def test_raw_threshold_during_warmup(self) -> None:
            mgr = AdaptiveThresholdManager(beta=0.9, warmup_steps=5)
            for _ in range(3):
                mgr.update(10.0)
            # Not yet warmed up (3 < 5 steps)
            self.assertFalse(mgr.is_warmed_up)
            # During warm-up, effective_threshold returns multiplier directly
            self.assertAlmostEqual(mgr.effective_threshold(2.5), 2.5, places=6)

        def test_adaptive_threshold_after_warmup(self) -> None:
            mgr = AdaptiveThresholdManager(beta=0.9, warmup_steps=3)
            for _ in range(5):
                mgr.update(10.0)  # EMA converges toward 10.0
            self.assertTrue(mgr.is_warmed_up)
            ema = mgr.bias_corrected_ema
            self.assertIsNotNone(ema)
            effective = mgr.effective_threshold(multiplier=3.0)
            # Should be approximately 10.0 * 3.0 = 30.0
            self.assertAlmostEqual(effective, ema * 3.0, places=5)

        def test_step_count_increments(self) -> None:
            mgr = AdaptiveThresholdManager(beta=0.9, warmup_steps=10)
            for _ in range(7):
                mgr.update(5.0)
            self.assertEqual(mgr.step_count, 7)

        def test_ema_none_before_first_update(self) -> None:
            mgr = AdaptiveThresholdManager(beta=0.9, warmup_steps=5)
            self.assertIsNone(mgr.bias_corrected_ema)

    class TestHeteroGradNormSkipController(unittest.TestCase):
        """Tests for the main skip controller — mirrors Megatron's
        test_chained_optimizer_reports_unsuccessful_when_grad_norm_skipped."""

        def _make_grads(
            self, values_list: List[List[float]]
        ) -> List[torch.Tensor]:
            return [torch.tensor(v, dtype=torch.float32) for v in values_list]

        def test_no_skip_when_thresholds_inf(self) -> None:
            """No skip when all thresholds are inf (default / feature disabled)."""
            config = HeteroGradNormConfig()
            ctrl = HeteroGradNormSkipController(config)
            anchor_grads = self._make_grads([[1.0, 2.0]])
            compute_grads = self._make_grads([[3.0, 4.0]])
            skip, info = ctrl.should_skip(anchor_grads, compute_grads)
            self.assertFalse(skip)
            self.assertIsNone(info.triggered_by)
            self.assertFalse(info.loc_cache_invalidated)

        def test_skip_when_combined_norm_exceeds_threshold(self) -> None:
            """Skip triggered by combined norm exceeding combined_skip_threshold."""
            config = HeteroGradNormConfig(combined_skip_threshold=5.0)
            ctrl = HeteroGradNormSkipController(config)
            # anchor: [3, 4] → L2=5, compute: [0, 12] → L2=12; combined=13
            anchor_grads = self._make_grads([[3.0, 4.0]])
            compute_grads = self._make_grads([[0.0, 12.0]])
            skip, info = ctrl.should_skip(anchor_grads, compute_grads)
            self.assertTrue(skip)
            self.assertEqual(info.triggered_by, "combined")
            self.assertAlmostEqual(info.combined_norm, 13.0, places=4)

        def test_skip_when_anchor_norm_exceeds_threshold(self) -> None:
            """Skip triggered by anchor norm exceeding anchor_skip_threshold."""
            config = HeteroGradNormConfig(anchor_skip_threshold=4.0)
            ctrl = HeteroGradNormSkipController(config)
            # anchor: [3, 4] → L2=5 > 4.0 → skip
            anchor_grads = self._make_grads([[3.0, 4.0]])
            compute_grads = self._make_grads([[1.0]])
            skip, info = ctrl.should_skip(anchor_grads, compute_grads)
            self.assertTrue(skip)
            self.assertEqual(info.triggered_by, "anchor")

        def test_skip_when_compute_norm_exceeds_threshold(self) -> None:
            """Skip triggered by compute norm exceeding compute_skip_threshold."""
            config = HeteroGradNormConfig(compute_skip_threshold=10.0)
            ctrl = HeteroGradNormSkipController(config)
            anchor_grads = self._make_grads([[1.0]])
            compute_grads = self._make_grads([[0.0, 12.0]])  # L2=12 > 10
            skip, info = ctrl.should_skip(anchor_grads, compute_grads)
            self.assertTrue(skip)
            self.assertEqual(info.triggered_by, "compute")

        def test_no_skip_just_below_threshold(self) -> None:
            """No skip when norm is strictly below the threshold."""
            config = HeteroGradNormConfig(combined_skip_threshold=14.0)
            ctrl = HeteroGradNormSkipController(config)
            anchor_grads = self._make_grads([[3.0, 4.0]])
            compute_grads = self._make_grads([[0.0, 12.0]])  # combined=13 < 14
            skip, info = ctrl.should_skip(anchor_grads, compute_grads)
            self.assertFalse(skip)

        def test_record_step_updates_consecutive_count(self) -> None:
            """Consecutive skip counter increments correctly."""
            config = HeteroGradNormConfig(combined_skip_threshold=1.0)
            ctrl = HeteroGradNormSkipController(config)
            grads = self._make_grads([[3.0, 4.0]])  # norm=5 > 1
            for _ in range(3):
                skip, info = ctrl.should_skip(grads, [])
                ctrl.record_step(skipped=skip, grad_norm=info.combined_norm)
            self.assertEqual(ctrl._consecutive_skips, 3)

        def test_record_step_resets_consecutive_on_non_skip(self) -> None:
            """Consecutive skip counter resets when a non-skip step occurs."""
            config = HeteroGradNormConfig(combined_skip_threshold=1.0)
            ctrl = HeteroGradNormSkipController(config)
            big_grads = self._make_grads([[3.0, 4.0]])  # norm=5 > 1
            small_grads = self._make_grads([[0.1]])      # norm=0.1 < 1
            skip, info = ctrl.should_skip(big_grads, [])
            ctrl.record_step(skipped=skip, grad_norm=info.combined_norm)
            self.assertEqual(ctrl._consecutive_skips, 1)
            skip, info = ctrl.should_skip(small_grads, [])
            ctrl.record_step(skipped=skip, grad_norm=info.combined_norm)
            self.assertEqual(ctrl._consecutive_skips, 0)

        def test_skip_rate_calculation(self) -> None:
            """skip_rate accurately reflects proportion of skipped steps."""
            config = HeteroGradNormConfig(combined_skip_threshold=3.0)
            ctrl = HeteroGradNormSkipController(config)
            big_grads = self._make_grads([[3.0, 4.0]])   # norm=5 > 3
            small_grads = self._make_grads([[1.0, 2.0]]) # norm≈2.24 < 3
            for grads in [big_grads, small_grads, big_grads, small_grads]:
                skip, info = ctrl.should_skip(grads, [])
                ctrl.record_step(skipped=skip, grad_norm=info.combined_norm)
            self.assertAlmostEqual(ctrl.skip_rate, 0.5, places=6)

        def test_empty_grad_lists_do_not_raise(self) -> None:
            """Empty gradient lists result in zero norm and no skip."""
            config = HeteroGradNormConfig(combined_skip_threshold=0.5)
            ctrl = HeteroGradNormSkipController(config)
            skip, info = ctrl.should_skip([], [])
            # combined norm = 0.0, which is NOT > 0.5
            self.assertFalse(skip)
            self.assertAlmostEqual(info.combined_norm, 0.0, places=10)

        def test_loc_cache_invalidated_on_skip(self) -> None:
            """LOC cache is invalidated when a skip event occurs."""

            class MockLOCCache:
                def __init__(self) -> None:
                    self.invalidate_called = False
                    self.invalidate_reason: Optional[str] = None
                    self._fragments = 5

                def invalidate(self, reason: str) -> None:
                    self.invalidate_called = True
                    self.invalidate_reason = reason
                    self._fragments = 0

                def fragment_count(self) -> int:
                    return self._fragments

            mock_loc = MockLOCCache()
            config = HeteroGradNormConfig(combined_skip_threshold=1.0)
            ctrl = HeteroGradNormSkipController(config, loc_cache_handle=mock_loc)
            skip, info = ctrl.should_skip(self._make_grads([[3.0, 4.0]]), [])
            self.assertTrue(skip)
            self.assertTrue(info.loc_cache_invalidated)
            self.assertTrue(mock_loc.invalidate_called)
            self.assertIn("grad_norm_skip", mock_loc.invalidate_reason or "")

        def test_loc_cache_not_invalidated_when_empty(self) -> None:
            """LOC cache invalidation is not called when cache has no fragments."""

            class EmptyLOCCache:
                def __init__(self) -> None:
                    self.invalidate_called = False

                def invalidate(self, reason: str) -> None:
                    self.invalidate_called = True

                def fragment_count(self) -> int:
                    return 0

            mock_loc = EmptyLOCCache()
            config = HeteroGradNormConfig(combined_skip_threshold=1.0)
            ctrl = HeteroGradNormSkipController(config, loc_cache_handle=mock_loc)
            skip, info = ctrl.should_skip(self._make_grads([[3.0, 4.0]]), [])
            self.assertTrue(skip)
            self.assertFalse(info.loc_cache_invalidated)
            self.assertFalse(mock_loc.invalidate_called)

        def test_statistics_dict_keys(self) -> None:
            """statistics property returns expected keys."""
            ctrl = HeteroGradNormSkipController(HeteroGradNormConfig())
            stats = ctrl.statistics
            for key in ("total_steps", "total_skips", "consecutive_skips",
                        "skip_rate", "last_skip_step", "adaptive_ema",
                        "adaptive_warmed_up"):
                self.assertIn(key, stats)

        def test_linf_norm_skip(self) -> None:
            """L-inf norm skip works correctly."""
            config = HeteroGradNormConfig(combined_skip_threshold=6.0, norm_type=float("inf"))
            ctrl = HeteroGradNormSkipController(config)
            anchor_grads = self._make_grads([[1.0, 7.0]])   # L-inf = 7.0
            compute_grads = self._make_grads([[2.0, 3.0]])  # L-inf = 3.0
            # combined L-inf = max(7, 3) = 7 > 6 → skip
            skip, info = ctrl.should_skip(anchor_grads, compute_grads)
            self.assertTrue(skip)
            self.assertAlmostEqual(info.combined_norm, 7.0, places=5)

    class TestGetGradNormFP32Hetero(unittest.TestCase):
        """Tests for the standalone norm utility function."""

        def test_returns_correct_keys(self) -> None:
            g = torch.tensor([3.0, 4.0])
            result = get_grad_norm_fp32_hetero([g], [], norm_type=2)
            self.assertIn("anchor_norm", result)
            self.assertIn("compute_norm", result)
            self.assertIn("combined_norm", result)

        def test_l2_combined_norm_value(self) -> None:
            anchor = [torch.tensor([3.0, 4.0])]   # L2=5
            compute = [torch.tensor([0.0, 12.0])] # L2=12
            result = get_grad_norm_fp32_hetero(anchor, compute, norm_type=2)
            self.assertAlmostEqual(result["combined_norm"], 13.0, places=4)

        def test_anchor_only_norm(self) -> None:
            anchor = [torch.tensor([3.0, 4.0])]
            result = get_grad_norm_fp32_hetero(anchor, [], norm_type=2)
            self.assertAlmostEqual(result["anchor_norm"], 5.0, places=5)
            self.assertAlmostEqual(result["compute_norm"], 0.0, places=10)

    class TestClipGradsHetero(unittest.TestCase):
        """Tests for the in-place gradient clipping utility."""

        def test_clips_when_norm_exceeds_max(self) -> None:
            g_anchor = torch.tensor([3.0, 4.0])   # contributes to norm
            g_compute = torch.tensor([0.0, 12.0])
            total_norm = 13.0
            max_norm = 6.5
            clip_grads_hetero([g_anchor], [g_compute], max_norm, total_norm)
            # clip_coeff = 6.5 / (13 + 1e-6) ≈ 0.5
            self.assertAlmostEqual(g_anchor[0].item(), 3.0 * (6.5 / 13.0), places=4)
            self.assertAlmostEqual(g_compute[1].item(), 12.0 * (6.5 / 13.0), places=4)

        def test_no_clip_when_norm_below_max(self) -> None:
            g = torch.tensor([1.0, 2.0])
            original = g.clone()
            clip_grads_hetero([g], [], max_norm=100.0, total_norm=2.236)
            torch.testing.assert_close(g, original)

        def test_zero_norm_does_not_raise(self) -> None:
            g = torch.tensor([0.0, 0.0])
            clip_grads_hetero([g], [], max_norm=1.0, total_norm=0.0)
            torch.testing.assert_close(g, torch.zeros(2))

    # Run all tests.
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in (
        TestHeteroGradNormConfig,
        TestPartialNormAccumulator,
        TestAdaptiveThresholdManager,
        TestHeteroGradNormSkipController,
        TestGetGradNormFP32Hetero,
        TestClipGradsHetero,
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
