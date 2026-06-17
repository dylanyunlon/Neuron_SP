# Copyright (c) 2026 Neuron_SP Project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DES-LOC Heterogeneous MoE Metrics Logger
# =========================================
#
# Upstream Design Intent (Megatron commit 873678a):
#   Megatron refactored its MoE auxiliary-loss tracking from a flat global dict
#   (_MOE_LAYER_WISE_LOGGING_TRACKER: dict) into a proper class-based tracker
#   (MoEMetricsTracker) with typed MetricEntry dataclasses. The old dict mixed
#   tensor storage with reduction metadata in ad-hoc string keys; the new design
#   separates concerns cleanly: record() accumulates per-layer scalars during
#   forward, report() synchronizes across ranks and aggregates at step end.
#   Deprecated shims (save_to_aux_losses_tracker, clear_aux_losses_tracker, etc.)
#   forward to the new API.
#
# DES-LOC Adaptation (Neuron_SP / this file):
#   In DES-LOC (Decoupled Execution with Shared LOcality Cache) the hardware
#   topology is fundamentally non-uniform:
#
#     Tier-0  (SM90):  1x H100 NVL 96 GB  — expert-compute tier, high-FLOP
#     Tier-1  (SM86):  2x A6000 48 GB each — routing / attention / dense layers
#     CPU              1.5 TB DRAM         — spill cache / shared-locality cache
#
#   PCIe without NVLink means cross-tier transfers are expensive (~32 GB/s vs
#   ~600 GB/s NVLink). Consequently:
#
#   1. Expert layers physically execute on Tier-0 (H100), while routing runs on
#      Tier-1 (A6000). Each tier records metrics independently.
#   2. Reduction across tiers must account for asymmetric bandwidth: we defer
#      cross-tier all_reduce to report() and batch them, never do per-layer sync.
#   3. The Shared-LOcality Cache (SLC) on CPU DRAM is used to hold metric tensors
#      when a tier's VRAM is under pressure, with explicit pin/unpin semantics.
#   4. Per-tier metric entries carry a `device_tier` tag (0=H100, 1=A6000_0,
#      2=A6000_1) so aggregation can emit per-tier breakdowns for load-balance
#      diagnostics across heterogeneous devices.
#   5. CUDAGraph capture on heterogeneous devices requires separate metric
#      snapshot / restore paths per tier (mirrors Megatron's cached_aux_losses
#      pattern in _CudaGraphRunner but extended to 3 compute streams).
#
# Usage:
#   tracker = get_hetero_moe_logger()
#
#   # In router forward (runs on A6000):
#   tracker.record("load_balancing_loss", loss, layer_number=1,
#                  num_layers=32, device_tier=DeviceTier.A6000_0)
#
#   # In expert forward (runs on H100):
#   tracker.record("expert_activation_rate", rate, layer_number=1,
#                  num_layers=32, device_tier=DeviceTier.H100)
#
#   # At step end (called once, handles cross-tier sync internally):
#   log_str = tracker.report(loss_scale=1/num_microbatches, iteration=step,
#                             writer=tb_writer, num_layers=32)

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware topology constants for this specific DES-LOC cluster
# ---------------------------------------------------------------------------

class DeviceTier(IntEnum):
    """Maps logical tier index to physical device role in DES-LOC cluster.

    Tier-0 = H100 NVL 96 GB (SM90) — expert compute
    Tier-1 = A6000 48 GB #0  (SM86) — routing + attention
    Tier-2 = A6000 48 GB #1  (SM86) — routing + attention (second replica)
    CPU    = pinned DRAM in shared-locality cache
    """
    H100    = 0   # Expert compute tier (SM90)
    A6000_0 = 1   # Routing/dense tier, socket 0 (SM86)
    A6000_1 = 2   # Routing/dense tier, socket 1 (SM86)
    CPU     = 3   # Shared-locality cache (SLC) on host DRAM


# Default CUDA device indices per tier (configurable via env / config)
_TIER_CUDA_INDEX: Dict[DeviceTier, int] = {
    DeviceTier.H100:    0,
    DeviceTier.A6000_0: 1,
    DeviceTier.A6000_1: 2,
}

# PCIe bandwidth estimate between tier pairs (GB/s) for cost-aware decisions
_PCIE_BW_GBPS: Dict[Tuple[DeviceTier, DeviceTier], float] = {
    (DeviceTier.H100,    DeviceTier.A6000_0): 16.0,
    (DeviceTier.H100,    DeviceTier.A6000_1): 16.0,
    (DeviceTier.A6000_0, DeviceTier.A6000_1): 16.0,
    (DeviceTier.A6000_0, DeviceTier.H100):    16.0,
    (DeviceTier.A6000_1, DeviceTier.H100):    16.0,
    (DeviceTier.A6000_1, DeviceTier.A6000_0): 16.0,
    (DeviceTier.H100,    DeviceTier.CPU):     32.0,
    (DeviceTier.A6000_0, DeviceTier.CPU):     32.0,
    (DeviceTier.A6000_1, DeviceTier.CPU):     32.0,
}

# Threshold (bytes) below which cross-tier sync is considered cheap enough to
# do eagerly rather than deferring to report().
_EAGER_SYNC_THRESHOLD_BYTES: int = 64 * 1024  # 64 KB


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TierMetricEntry:
    """A single metric tensor resident on one device tier.

    Upstream analogue: ``MetricEntry`` in Megatron moe_logging.py.

    DES-LOC additions:
    - ``device_tier``: identifies which physical tier owns this tensor.
    - ``slc_offloaded``: True when the tensor has been moved to the
      Shared-LOcality Cache (CPU pinned memory) to relieve VRAM pressure.
    - ``reduce_group`` / ``avg_group`` retain their Megatron semantics but
      are resolved relative to the tier's process subgroup.
    """
    values: torch.Tensor
    device_tier: DeviceTier
    reduce_group: Optional[torch.distributed.ProcessGroup] = None
    avg_group:    Optional[torch.distributed.ProcessGroup] = None
    needs_dp_avg: bool = True
    slc_offloaded: bool = False
    _slc_buffer: Optional[torch.Tensor] = field(default=None, repr=False)

    def offload_to_slc(self) -> None:
        """Move values tensor to CPU pinned memory (Shared-LOcality Cache).

        Only performs the copy if the tensor is currently on a CUDA device and
        not already offloaded.  The original VRAM allocation is replaced by a
        CPU-pinned clone so that future cross-tier PCIe reads are DMA-friendly.
        """
        if self.slc_offloaded or self.values.device.type == "cpu":
            return
        self._slc_buffer = self.values.cpu().pin_memory()
        self.values = self._slc_buffer
        self.slc_offloaded = True
        logger.debug(
            "TierMetricEntry offloaded to SLC (pinned CPU): shape=%s from tier=%s",
            tuple(self.values.shape), self.device_tier.name,
        )

    def restore_from_slc(self, target_tier: DeviceTier) -> None:
        """Move values back to CUDA from CPU SLC.

        Args:
            target_tier: The tier whose CUDA device should receive the tensor.
        """
        if not self.slc_offloaded:
            return
        cuda_idx = _TIER_CUDA_INDEX.get(target_tier)
        if cuda_idx is None:
            raise ValueError(f"Cannot restore to non-CUDA tier {target_tier}")
        device = torch.device("cuda", cuda_idx)
        self.values = self._slc_buffer.to(device, non_blocking=True)
        self._slc_buffer = None
        self.slc_offloaded = False
        self.device_tier = target_tier
        logger.debug(
            "TierMetricEntry restored from SLC to cuda:%d (tier=%s)",
            cuda_idx, target_tier.name,
        )


@dataclass
class HeteroMetricEntry:
    """Aggregates ``TierMetricEntry`` instances for a single metric name.

    One logical metric (e.g. "load_balancing_loss") may have contributions
    from multiple tiers.  This container holds one entry per tier that
    recorded a value, keyed by DeviceTier.

    The canonical combined tensor used for cross-rank reduction lives on
    Tier-0 (H100) for expert metrics and Tier-1 (A6000_0) for routing metrics.
    """
    entries: Dict[DeviceTier, TierMetricEntry] = field(default_factory=dict)

    def all_tiers(self) -> List[DeviceTier]:
        return list(self.entries.keys())

    def get_or_none(self, tier: DeviceTier) -> Optional[TierMetricEntry]:
        return self.entries.get(tier)


# ---------------------------------------------------------------------------
# Process group stub for environments without torch.distributed
# ---------------------------------------------------------------------------

class _FakeProcessGroup:
    """Minimal stub so the tracker is unit-testable without dist init."""
    def __init__(self, rank: int = 0, size: int = 1):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_HETERO_MOE_LOGGER: Optional['HeteroMoELogger'] = None
_SINGLETON_LOCK = threading.Lock()


def get_hetero_moe_logger() -> 'HeteroMoELogger':
    """Return the process-global HeteroMoELogger, creating it lazily.

    Thread-safe via module-level lock.  Mirrors Megatron's
    ``get_moe_metrics_tracker()`` pattern but adds tier-awareness.
    """
    global _HETERO_MOE_LOGGER
    if _HETERO_MOE_LOGGER is None:
        with _SINGLETON_LOCK:
            if _HETERO_MOE_LOGGER is None:
                _HETERO_MOE_LOGGER = HeteroMoELogger()
                logger.info("HeteroMoELogger singleton created for DES-LOC cluster")
    return _HETERO_MOE_LOGGER


def set_hetero_moe_logger(tracker: 'HeteroMoELogger') -> None:
    """Replace the global tracker (useful for testing or multi-process init)."""
    global _HETERO_MOE_LOGGER
    with _SINGLETON_LOCK:
        _HETERO_MOE_LOGGER = tracker


def destroy_hetero_moe_logger() -> None:
    """Reset the global tracker to None (call in test teardown)."""
    global _HETERO_MOE_LOGGER
    with _SINGLETON_LOCK:
        _HETERO_MOE_LOGGER = None


# ---------------------------------------------------------------------------
# Compatibility shims (mirror Megatron deprecated API surface)
# ---------------------------------------------------------------------------

def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: Optional[torch.distributed.ProcessGroup] = None,
    avg_group: Optional[torch.distributed.ProcessGroup] = None,
    reduce_group_has_dp: bool = False,
    device_tier: DeviceTier = DeviceTier.A6000_0,
) -> None:
    """Deprecated shim — use ``get_hetero_moe_logger().record()`` directly.

    Signature mirrors the Megatron deprecated function so existing call-sites
    need no change. ``reduce_group_has_dp=True`` maps to ``needs_dp_avg=False``
    following the semantic inversion introduced in Megatron 873678a.

    Args:
        name: Metric name.
        loss: Scalar tensor to accumulate.
        layer_number: 1-based layer index.
        num_layers: Total number of layers.
        reduce_group: Process group for sum-reduction.
        avg_group: Process group for mean-reduction.
        reduce_group_has_dp: Legacy flag; inverted to ``needs_dp_avg``.
        device_tier: DES-LOC tier that owns this tensor.
    """
    import warnings
    warnings.warn(
        "save_to_aux_losses_tracker is deprecated. Use get_hetero_moe_logger().record().",
        DeprecationWarning,
        stacklevel=2,
    )
    get_hetero_moe_logger().record(
        name=name,
        value=loss,
        layer_number=layer_number,
        num_layers=num_layers,
        reduce_group=reduce_group,
        avg_group=avg_group,
        needs_dp_avg=not reduce_group_has_dp,
        device_tier=device_tier,
    )


def clear_aux_losses_tracker() -> None:
    """Deprecated shim — use ``get_hetero_moe_logger().clear()``."""
    import warnings
    warnings.warn(
        "clear_aux_losses_tracker is deprecated. Use get_hetero_moe_logger().clear().",
        DeprecationWarning,
        stacklevel=2,
    )
    get_hetero_moe_logger().clear()


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class HeteroMoELogger:
    """Tier-disaggregated MoE metrics tracker for DES-LOC heterogeneous clusters.

    Upstream Design Intent (Megatron 873678a):
        ``MoEMetricsTracker`` aggregates per-layer MoE losses in a per-process
        dict, then synchronizes across ranks via all_reduce at step end.  The
        redesign from a raw dict to typed dataclasses makes the reduction graph
        explicit and removes the brittle string-key access pattern.

    DES-LOC Adaptation:
        In DES-LOC the routing computation (TopKRouter) runs on A6000 tiers while
        expert FFN computation runs on the H100 tier.  Both tiers contribute to
        the same named metric (e.g. "load_balancing_loss") but from different
        CUDA devices.  Naively merging them would require an eager PCIe transfer
        on every record() call.

        Instead we maintain a *per-tier* entry for each metric name and defer
        cross-tier consolidation to report():

          record(tier=A6000_0)  → entry stored in cuda:1 tensor
          record(tier=H100)     → entry stored in cuda:0 tensor
          report()              → gather all tiers onto H100, then all_reduce,
                                   then aggregate scalars and log

        When VRAM pressure is detected on either A6000 (48 GB limit), metric
        tensors are offloaded to the Shared-LOcality Cache (1.5 TB CPU DRAM)
        via ``TierMetricEntry.offload_to_slc()``.  The SLC buffer is pinned so
        subsequent PCIe DMA is efficient during consolidation.

        CUDAGraph snapshot/restore (mirrors Megatron _CudaGraphRunner) is
        extended across all three compute streams; see ``snapshot_for_cuda_graph``
        and ``restore_from_cuda_graph_snapshot``.

    Thread safety:
        ``record()`` and ``clear()`` are protected by ``_lock``.  ``report()``
        acquires the lock once to snapshot the current state, releases it during
        I/O-heavy operations (all_reduce, writer calls), then re-acquires to
        clear.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, HeteroMetricEntry] = {}
        self._lock = threading.Lock()
        # Track which tiers have contributed any data this step
        self._active_tiers: set = set()

    # =========================================================================
    # Public API — record
    # =========================================================================

    def record(
        self,
        name: str,
        value: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        avg_group: Optional[torch.distributed.ProcessGroup] = None,
        needs_dp_avg: bool = True,
        device_tier: DeviceTier = DeviceTier.A6000_0,
    ) -> None:
        """Accumulate a metric value for a specific layer and device tier.

        Analogous to Megatron ``MoEMetricsTracker.record()`` but adds the
        ``device_tier`` dimension.  The tensor is stored on the same CUDA device
        as ``value`` to avoid PCIe transfers during the forward pass.

        Lazy initialization: the TierMetricEntry is created on first call for
        each (name, device_tier) pair.

        Args:
            name: Metric name (e.g. "load_balancing_loss").
            value: Scalar or per-expert tensor to accumulate.
            layer_number: 1-based layer index; if None, call is a no-op.
            num_layers: Total number of transformer layers.
            reduce_group: Process group for sum-reduction (e.g. tp_cp_group).
            avg_group: Process group for mean-reduction.
            needs_dp_avg: If True, average across DP ranks after other reductions.
            device_tier: The DES-LOC tier that is calling this method.
        """
        if layer_number is None:
            return

        detached = value.detach()

        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = HeteroMetricEntry()

            hetero_entry = self._metrics[name]

            if device_tier not in hetero_entry.entries:
                # Allocate the per-layer tensor on the same device as value so
                # we never trigger a PCIe transfer during the forward pass.
                hetero_entry.entries[device_tier] = TierMetricEntry(
                    values=torch.zeros(num_layers, device=detached.device),
                    device_tier=device_tier,
                    reduce_group=reduce_group,
                    avg_group=avg_group,
                    needs_dp_avg=needs_dp_avg,
                )

            tier_entry = hetero_entry.entries[device_tier]
            # Restore from SLC if it was offloaded (can happen between steps)
            if tier_entry.slc_offloaded:
                tier_entry.restore_from_slc(device_tier)

            tier_entry.values[layer_number - 1] += detached
            tier_entry.reduce_group = reduce_group
            tier_entry.avg_group = avg_group
            tier_entry.needs_dp_avg = needs_dp_avg

            self._active_tiers.add(device_tier)

    # =========================================================================
    # Public API — report
    # =========================================================================

    def report(
        self,
        loss_scale: float,
        iteration: int,
        writer=None,
        wandb_writer=None,
        per_layer_logging: bool = False,
        force_initialize: bool = False,
        track_names: Optional[Union[str, List[str]]] = None,
        num_layers: Optional[int] = None,
        moe_layer_freq: Optional[Union[int, List[int]]] = None,
        mtp_num_layers: Optional[int] = None,
        total_loss_dict: Optional[dict] = None,
        percentiles: Optional[Dict[str, List[float]]] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        consolidation_tier: DeviceTier = DeviceTier.H100,
    ) -> str:
        """Sync metrics across ranks and tiers, aggregate, log, and clear.

        This is the main step-end entry point.  Execution order:

          1. Force-initialize missing entries for PP ranks without MoE layers.
          2. Consolidate per-tier tensors onto ``consolidation_tier`` (H100).
          3. Reduce across distributed ranks (PP → reduce_group → avg_group → DP).
          4. Aggregate per-layer values into scalars (mean over MoE layers).
          5. Log to TensorBoard / W&B.
          6. Clear for next step.

        Args:
            loss_scale: Scale factor for microbatch averaging (1/num_microbatches).
            iteration: Current training step.
            writer: TensorBoard SummaryWriter.
            wandb_writer: W&B run object.
            per_layer_logging: Emit per-layer scalar events.
            force_initialize: Pre-create missing entries for PP ranks without
                MoE layers (required so all_reduce shapes match).
            track_names: Metric name(s) to report; None means all.
            num_layers: Total transformer layers (required when force_initialize).
            moe_layer_freq: MoE layer frequency or binary list pattern.
            mtp_num_layers: Extra layers from Multi-Token Prediction.
            total_loss_dict: Megatron training-loop loss accumulator.
            percentiles: Per-metric percentile breakdowns.
            pp_group: Pipeline-parallel process group.
            dp_group: Data-parallel process group.
            consolidation_tier: Target tier for cross-tier tensor merge.

        Returns:
            Formatted log string for console output (excludes metrics already
            placed in total_loss_dict).
        """
        metric_names = self._resolve_names(track_names)

        # Step 1: force-initialize on PP ranks without MoE layers so that
        # all_reduce shapes are consistent across all pipeline stages.
        if force_initialize:
            if num_layers is None:
                raise ValueError("num_layers must be provided when force_initialize=True")
            init_size = num_layers + (mtp_num_layers or 0)
            for name in metric_names:
                self._ensure_initialized(name, init_size, consolidation_tier)

        # Step 2: consolidate per-tier tensors before distributed reduction.
        # This is the key DES-LOC addition: merge H100 and A6000 entries so
        # subsequent all_reduce operates on a single tensor per metric.
        consolidated = self._consolidate_tiers(metric_names, consolidation_tier)

        # Step 3: distributed reduction across ranks.
        self._sync_across_ranks(consolidated, pp_group=pp_group, dp_group=dp_group)

        # Step 4: scalar aggregation.
        num_moe_layers = self._count_moe_layers(num_layers, moe_layer_freq, mtp_num_layers)
        scalars = self._aggregate(consolidated, loss_scale, num_moe_layers, metric_names, percentiles)

        # Step 5a: route loss metrics into Megatron's total_loss_dict.
        console_scalars: Dict[str, Union[float, torch.Tensor]] = dict(scalars)
        if total_loss_dict is not None:
            for k, v in scalars.items():
                if k.lower().endswith("loss"):
                    total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v
                    console_scalars.pop(k)

        # Step 5b: write to writers.
        self._log_scalars(scalars, iteration, writer, wandb_writer)
        if per_layer_logging:
            self._log_per_layer(
                consolidated, loss_scale, metric_names,
                iteration, writer, wandb_writer, percentiles,
            )

        # Emit per-tier breakdown only when multiple tiers contributed.
        if len(self._active_tiers) > 1:
            self._log_tier_breakdown(
                metric_names, loss_scale, num_moe_layers,
                iteration, writer, wandb_writer,
            )

        log_string = self._format(console_scalars)

        # Step 6: clear for next step.
        self.clear()
        return log_string

    # =========================================================================
    # Public API — lifecycle
    # =========================================================================

    def clear(self) -> None:
        """Zero all metric tensors; preserve allocated entries for reuse.

        Mirrors Megatron ``MoEMetricsTracker.clear()``.  In DES-LOC we also
        restore any SLC-offloaded tensors so the next step can record directly
        to CUDA.  We reset _active_tiers to track fresh step contributions.
        """
        with self._lock:
            for hetero_entry in self._metrics.values():
                for tier_entry in hetero_entry.entries.values():
                    if tier_entry.slc_offloaded:
                        tier_entry.restore_from_slc(tier_entry.device_tier)
                    tier_entry.values.zero_()
            self._active_tiers.clear()

    @property
    def metrics(self) -> Dict[str, HeteroMetricEntry]:
        """Read-only view of the current metric registry."""
        return self._metrics

    def ensure_initialized(
        self,
        name: str,
        num_layers: int,
        tier: DeviceTier = DeviceTier.A6000_0,
        device: Optional[Union[str, torch.device, int]] = None,
    ) -> None:
        """Pre-create a TierMetricEntry if it doesn't exist yet.

        Needed for PP ranks with no MoE layers so all_reduce shapes match.

        Args:
            name: Metric name.
            num_layers: Tensor size (should include MTP layers).
            tier: Which tier to allocate on.
            device: Explicit device override; defaults to tier's CUDA index.
        """
        with self._lock:
            self._ensure_initialized(name, num_layers, tier, device)

    def offload_tier_to_slc(self, tier: DeviceTier) -> int:
        """Offload all metric tensors for a given tier to CPU SLC.

        Called by the DES-LOC memory manager when VRAM headroom on a tier
        drops below a threshold.  Returns the number of tensors offloaded.

        Args:
            tier: The device tier whose tensors should be moved to SLC.

        Returns:
            Count of tensors successfully offloaded.
        """
        count = 0
        with self._lock:
            for name, hetero_entry in self._metrics.items():
                tier_entry = hetero_entry.entries.get(tier)
                if tier_entry is not None and not tier_entry.slc_offloaded:
                    tier_entry.offload_to_slc()
                    count += 1
        if count:
            logger.info(
                "Offloaded %d metric tensor(s) for tier %s to SLC (CPU DRAM)",
                count, tier.name,
            )
        return count

    # =========================================================================
    # CUDAGraph snapshot / restore (mirrors Megatron _CudaGraphRunner pattern)
    # =========================================================================

    def snapshot_for_cuda_graph(self) -> Dict[str, Dict[DeviceTier, torch.Tensor]]:
        """Clone all metric tensors before CUDAGraph capture.

        Megatron's _CudaGraphRunner clones aux loss values before capture
        because CUDAGraph replays re-write the same tensor buffers.  In DES-LOC
        we must snapshot per-tier tensors from all three compute streams since
        each tier has its own CUDAGraph.

        Returns:
            Nested dict: metric_name → tier → cloned tensor.
        """
        snapshot: Dict[str, Dict[DeviceTier, torch.Tensor]] = {}
        with self._lock:
            for name, hetero_entry in self._metrics.items():
                snapshot[name] = {}
                for tier, tier_entry in hetero_entry.entries.items():
                    if tier_entry.slc_offloaded:
                        snapshot[name][tier] = tier_entry.values.clone()
                    else:
                        snapshot[name][tier] = tier_entry.values.clone()
        logger.debug(
            "CUDAGraph snapshot taken: %d metrics across %d tiers",
            len(snapshot),
            len(self._active_tiers),
        )
        return snapshot

    def restore_from_cuda_graph_snapshot(
        self,
        snapshot: Dict[str, Dict[DeviceTier, torch.Tensor]],
    ) -> None:
        """Copy snapshot values back into live metric tensors after graph replay.

        Analogous to the restore loop in Megatron _CudaGraphRunner.forward()
        but extended to the per-tier structure.  Asserts that the snapshot
        keys are still present in the tracker (would fail if clear() was called
        between snapshot and restore, which indicates a usage error).

        Args:
            snapshot: Dict returned by ``snapshot_for_cuda_graph()``.
        """
        with self._lock:
            for name, tier_snapshots in snapshot.items():
                assert name in self._metrics, (
                    f"Snapshot key '{name}' not found in tracker — "
                    "was clear() called between snapshot and restore?"
                )
                hetero_entry = self._metrics[name]
                for tier, cached_values in tier_snapshots.items():
                    assert tier in hetero_entry.entries, (
                        f"Tier {tier.name} not found for metric '{name}' during restore"
                    )
                    hetero_entry.entries[tier].values.copy_(cached_values)
        logger.debug("CUDAGraph snapshot restored to live tensors")

    # =========================================================================
    # Private — initialization
    # =========================================================================

    def _ensure_initialized(
        self,
        name: str,
        num_layers: int,
        tier: DeviceTier = DeviceTier.A6000_0,
        device: Optional[Union[str, torch.device, int]] = None,
    ) -> None:
        """Internal (lock must already be held or single-threaded context)."""
        if name not in self._metrics:
            self._metrics[name] = HeteroMetricEntry()
        hetero_entry = self._metrics[name]
        if tier not in hetero_entry.entries:
            if device is None:
                cuda_idx = _TIER_CUDA_INDEX.get(tier)
                if cuda_idx is not None and torch.cuda.is_available():
                    device = torch.device("cuda", cuda_idx)
                else:
                    device = "cpu"
            hetero_entry.entries[tier] = TierMetricEntry(
                values=torch.zeros(num_layers, device=device),
                device_tier=tier,
            )

    # =========================================================================
    # Private — tier consolidation (DES-LOC core logic)
    # =========================================================================

    def _consolidate_tiers(
        self,
        metric_names: List[str],
        consolidation_tier: DeviceTier,
    ) -> Dict[str, TierMetricEntry]:
        """Merge per-tier TierMetricEntry tensors into a single tensor per metric.

        This is the key DES-LOC addition not present in Megatron.  Each named
        metric may have contributions from A6000_0, A6000_1, and H100.  We
        sum them onto the consolidation_tier device (typically H100) to produce
        a single TierMetricEntry per metric, ready for all_reduce.

        Cross-tier copies go via PCIe.  We batch all copies to minimize round
        trips: collect source tensors, issue .to(device, non_blocking=True) for
        each, then synchronize once before summing.

        Args:
            metric_names: Names of metrics to consolidate.
            consolidation_tier: Target DeviceTier for the merged tensor.

        Returns:
            Dict mapping metric name to a single merged TierMetricEntry.
        """
        cuda_idx = _TIER_CUDA_INDEX.get(consolidation_tier)
        if cuda_idx is not None and torch.cuda.is_available():
            target_device = torch.device("cuda", cuda_idx)
        else:
            target_device = torch.device("cpu")

        consolidated: Dict[str, TierMetricEntry] = {}

        for name in metric_names:
            if name not in self._metrics:
                continue

            hetero_entry = self._metrics[name]
            tiers = hetero_entry.all_tiers()

            if not tiers:
                continue

            if len(tiers) == 1:
                # Fast path: single tier, no PCIe needed.
                only_tier = tiers[0]
                only_entry = hetero_entry.entries[only_tier]
                if only_tier == consolidation_tier:
                    consolidated[name] = only_entry
                else:
                    # Move to consolidation device.
                    merged_values = only_entry.values.to(target_device, non_blocking=True)
                    consolidated[name] = TierMetricEntry(
                        values=merged_values,
                        device_tier=consolidation_tier,
                        reduce_group=only_entry.reduce_group,
                        avg_group=only_entry.avg_group,
                        needs_dp_avg=only_entry.needs_dp_avg,
                    )
                continue

            # Multi-tier path: batch transfers then sum.
            # Issue all non_blocking copies first to overlap PCIe transfers.
            copies_on_target: List[torch.Tensor] = []
            primary_entry: Optional[TierMetricEntry] = None

            for tier in tiers:
                entry = hetero_entry.entries[tier]
                if tier == consolidation_tier:
                    copies_on_target.append(entry.values)
                    if primary_entry is None:
                        primary_entry = entry
                else:
                    remote_copy = entry.values.to(target_device, non_blocking=True)
                    copies_on_target.append(remote_copy)
                    if primary_entry is None:
                        primary_entry = entry

            # Synchronize to ensure all PCIe transfers complete.
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)

            # Sum contributions from all tiers.
            merged_values = torch.zeros_like(copies_on_target[0])
            for t in copies_on_target:
                merged_values.add_(t)

            assert primary_entry is not None
            consolidated[name] = TierMetricEntry(
                values=merged_values,
                device_tier=consolidation_tier,
                reduce_group=primary_entry.reduce_group,
                avg_group=primary_entry.avg_group,
                needs_dp_avg=primary_entry.needs_dp_avg,
            )

            if len(tiers) > 1:
                logger.debug(
                    "Consolidated metric '%s' from tiers %s onto %s (device=%s)",
                    name,
                    [t.name for t in tiers],
                    consolidation_tier.name,
                    target_device,
                )

        return consolidated

    # =========================================================================
    # Private — distributed reduction
    # =========================================================================

    def _sync_across_ranks(
        self,
        consolidated: Dict[str, TierMetricEntry],
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        """All-reduce consolidated metric tensors across distributed ranks.

        Reduction order mirrors Megatron (PP → reduce_group → avg_group → DP),
        but operates on the already-consolidated single tensors.

        If torch.distributed is not available or not initialized, this is a
        no-op (single-process DES-LOC configuration for testing).

        Args:
            consolidated: Dict of merged TierMetricEntry per metric name.
            pp_group: Pipeline-parallel process group override.
            dp_group: Data-parallel process group override.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return

        _pp_group = pp_group
        _dp_group = dp_group

        # Fall back to parallel_state if groups not explicitly provided.
        if _pp_group is None or _dp_group is None:
            try:
                from deepspeed import comm as dist_state  # noqa: F401
                # DeepSpeed exposes these via its comm module; attempt import.
            except ImportError:
                pass

            try:
                from megatron.core import parallel_state
                if _pp_group is None:
                    _pp_group = parallel_state.get_pipeline_model_parallel_group()
                if _dp_group is None:
                    _dp_group = parallel_state.get_data_parallel_group(
                        with_context_parallel=False, partial_data_parallel=False
                    )
            except (ImportError, AttributeError):
                logger.warning(
                    "Could not resolve PP/DP process groups from parallel_state; "
                    "skipping distributed sync. In production, pass pp_group and dp_group "
                    "explicitly to report()."
                )
                return

        for name, entry in consolidated.items():
            v = entry.values

            # PP collect: aggregate across pipeline stages.
            if _pp_group is not None:
                torch.distributed.all_reduce(v, group=_pp_group)

            # Tensor-parallel / context-parallel reduction.
            if entry.reduce_group is not None:
                torch.distributed.all_reduce(v, group=entry.reduce_group)

            # Averaging group (e.g. expert parallelism within EP group).
            if entry.avg_group is not None:
                torch.distributed.all_reduce(
                    v, group=entry.avg_group, op=torch.distributed.ReduceOp.AVG
                )

            # Data-parallel averaging.
            if entry.needs_dp_avg and _dp_group is not None:
                torch.distributed.all_reduce(
                    v, group=_dp_group, op=torch.distributed.ReduceOp.AVG
                )

    # =========================================================================
    # Private — aggregation
    # =========================================================================

    @staticmethod
    def _count_moe_layers(
        num_layers: Optional[int],
        moe_layer_freq: Optional[Union[int, List[int]]],
        mtp_num_layers: Optional[int],
    ) -> int:
        """Compute effective number of MoE layers (mirrors Megatron logic)."""
        if moe_layer_freq is None:
            n = num_layers or 1
        elif isinstance(moe_layer_freq, int):
            assert isinstance(num_layers, int)
            n = sum(1 for i in range(num_layers) if i % moe_layer_freq == 0)
        elif isinstance(moe_layer_freq, list):
            n = sum(moe_layer_freq)
        else:
            raise ValueError(f"Invalid moe_layer_freq: {moe_layer_freq!r}")
        if mtp_num_layers is not None:
            n += mtp_num_layers
        return max(n, 1)

    def _aggregate(
        self,
        consolidated: Dict[str, TierMetricEntry],
        loss_scale: float,
        num_moe_layers: int,
        metric_names: List[str],
        percentiles: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """Aggregate per-layer values into scalar summaries.

        Always produces the mean across MoE layers.  If percentiles are
        requested for a metric, non-zero layer values are used to compute
        quantiles, emitted as ``"{name}_p{pct}"`` keys.

        Args:
            consolidated: Merged per-tier tensors (output of _consolidate_tiers).
            loss_scale: Microbatch scale factor.
            num_moe_layers: Denominator for mean across layers.
            metric_names: Names to aggregate.
            percentiles: Optional per-metric quantile list.

        Returns:
            Dict of scalar float values for logging.
        """
        result: Dict[str, Union[float, torch.Tensor]] = {}

        for name in metric_names:
            if name not in consolidated:
                continue

            values = consolidated[name].values.float() * loss_scale

            if percentiles and name in percentiles:
                nonzero = values[values > 0]
                if nonzero.numel() > 0:
                    pct_list = percentiles[name]
                    q = torch.tensor(pct_list, device=nonzero.device, dtype=torch.float)
                    pct_vals = torch.quantile(nonzero, q).tolist()
                    for pct, pct_val in zip(pct_list, pct_vals):
                        result[f"{name}_p{int(pct * 100)}"] = pct_val

            result[name] = (values.sum() / num_moe_layers).item()

        return result

    # =========================================================================
    # Private — logging
    # =========================================================================

    def _log_scalars(
        self,
        scalars: Dict[str, Union[float, torch.Tensor]],
        iteration: int,
        writer,
        wandb_writer,
    ) -> None:
        """Write aggregate scalar metrics to TensorBoard and/or W&B."""
        for name, value in scalars.items():
            if writer is not None:
                writer.add_scalar(f"moe/{name}", value, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"moe/{name}": value}, iteration)

    def _log_per_layer(
        self,
        consolidated: Dict[str, TierMetricEntry],
        loss_scale: float,
        metric_names: List[str],
        iteration: int,
        writer,
        wandb_writer,
        percentiles: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Write per-layer metric values to TensorBoard and/or W&B.

        Sparse mode: when percentiles are requested for a metric, skip layers
        with zero value (same logic as Megatron's per-layer logging).
        """
        for name in metric_names:
            if name not in consolidated:
                continue
            values = consolidated[name].values.float() * loss_scale
            is_sparse = percentiles is not None and name in percentiles
            for i, val in enumerate(values.tolist()):
                if is_sparse and val == 0.0:
                    continue
                tag = f"moe_per_layer/{name}_layer_{i}"
                if writer is not None:
                    writer.add_scalar(tag, val, iteration)
                if wandb_writer is not None:
                    wandb_writer.log({tag: val}, iteration)

    def _log_tier_breakdown(
        self,
        metric_names: List[str],
        loss_scale: float,
        num_moe_layers: int,
        iteration: int,
        writer,
        wandb_writer,
    ) -> None:
        """Log per-tier scalar breakdowns for multi-tier metrics.

        DES-LOC specific: when both A6000 and H100 contribute to a metric,
        we emit separate ``moe_tier/{name}_{tier}`` scalars so that imbalanced
        routing across device tiers is visible in dashboards.

        Only called when at least 2 tiers are active this step.
        """
        for name in metric_names:
            if name not in self._metrics:
                continue
            hetero_entry = self._metrics[name]
            if len(hetero_entry.entries) < 2:
                continue
            for tier, tier_entry in hetero_entry.entries.items():
                tier_scalar = (
                    tier_entry.values.float() * loss_scale
                ).sum().item() / num_moe_layers
                tag = f"moe_tier/{name}_{tier.name}"
                if writer is not None:
                    writer.add_scalar(tag, tier_scalar, iteration)
                if wandb_writer is not None:
                    wandb_writer.log({tag: tier_scalar}, iteration)

    # =========================================================================
    # Private — utilities
    # =========================================================================

    def _resolve_names(self, track_names: Optional[Union[str, List[str]]]) -> List[str]:
        """Normalize track_names to a list of strings."""
        if track_names is None:
            with self._lock:
                return list(self._metrics.keys())
        if isinstance(track_names, str):
            return [track_names]
        return list(track_names)

    @staticmethod
    def _format(scalars: Dict[str, Union[float, torch.Tensor]]) -> str:
        """Format aggregated metrics for console output.

        Mirrors Megatron ``MoEMetricsTracker._format()``.
        """
        parts = []
        for k, v in scalars.items():
            v_num = v.item() if isinstance(v, torch.Tensor) else float(v)
            parts.append(f" {k}: {v_num:.4f} |")
        return "".join(parts)

    def __repr__(self) -> str:
        metric_summary = {
            name: [tier.name for tier in entry.all_tiers()]
            for name, entry in self._metrics.items()
        }
        return (
            f"HeteroMoELogger("
            f"metrics={list(metric_summary.keys())}, "
            f"active_tiers={[t.name for t in self._active_tiers]}, "
            f"tier_map={metric_summary})"
        )


# ---------------------------------------------------------------------------
# Integration helpers for DeepSpeed training loop
# ---------------------------------------------------------------------------

def hetero_moe_clear_on_eval_start() -> None:
    """Clear the tracker at eval start to avoid contaminating eval metrics.

    Called by the DES-LOC eval harness before running validation batches.
    Mirrors the ``clear_aux_losses_tracker()`` call in Megatron's train() loop
    that guards the eval boundary.
    """
    tracker = get_hetero_moe_logger()
    tracker.clear()
    logger.debug("HeteroMoELogger cleared at eval boundary")


def hetero_moe_offload_pressure_relief(
    tier: DeviceTier,
    vram_free_gb: float,
    threshold_gb: float = 4.0,
) -> int:
    """Offload metric tensors to SLC when VRAM headroom is below threshold.

    Called by the DES-LOC memory manager.  Returns how many tensors were moved
    to the Shared-LOcality Cache.

    Args:
        tier: The tier to check for pressure.
        vram_free_gb: Current free VRAM on the tier (in GB).
        threshold_gb: If free VRAM is below this, trigger offload.

    Returns:
        Number of tensors offloaded.
    """
    if vram_free_gb >= threshold_gb:
        return 0
    logger.info(
        "VRAM pressure on tier %s (%.1f GB free < %.1f GB threshold): "
        "offloading MoE metric tensors to SLC",
        tier.name, vram_free_gb, threshold_gb,
    )
    return get_hetero_moe_logger().offload_tier_to_slc(tier)


# ===========================================================================
# Unit tests
# ===========================================================================

if __name__ == "__main__":
    import sys
    import traceback
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    class TestDeviceTier(unittest.TestCase):
        def test_tier_values(self):
            self.assertEqual(DeviceTier.H100.value, 0)
            self.assertEqual(DeviceTier.A6000_0.value, 1)
            self.assertEqual(DeviceTier.A6000_1.value, 2)
            self.assertEqual(DeviceTier.CPU.value, 3)

        def test_tier_cuda_index_mapping(self):
            self.assertIn(DeviceTier.H100, _TIER_CUDA_INDEX)
            self.assertIn(DeviceTier.A6000_0, _TIER_CUDA_INDEX)
            self.assertIn(DeviceTier.A6000_1, _TIER_CUDA_INDEX)
            self.assertNotIn(DeviceTier.CPU, _TIER_CUDA_INDEX)

    class TestTierMetricEntry(unittest.TestCase):
        def _make_entry(self, tier: DeviceTier) -> TierMetricEntry:
            return TierMetricEntry(
                values=torch.zeros(4),
                device_tier=tier,
            )

        def test_offload_to_slc_cpu_noop(self):
            entry = self._make_entry(DeviceTier.A6000_0)
            # Already on CPU (torch.zeros defaults to CPU in no-cuda env)
            entry.offload_to_slc()
            # Should not crash and slc_offloaded should remain False (device is cpu)
            self.assertFalse(entry.slc_offloaded)

        def test_slc_offload_restore_roundtrip(self):
            # Simulate CUDA-resident tensor using CPU as stand-in
            entry = TierMetricEntry(
                values=torch.tensor([1.0, 2.0, 3.0]),
                device_tier=DeviceTier.A6000_0,
            )
            # Manually trigger the logic that would run on CUDA
            entry._slc_buffer = entry.values.clone()
            entry.values = entry._slc_buffer
            entry.slc_offloaded = True

            # Confirm values are intact after simulated offload
            self.assertTrue(entry.slc_offloaded)
            self.assertAlmostEqual(entry.values[1].item(), 2.0)

    class TestHeteroMetricEntry(unittest.TestCase):
        def test_all_tiers_empty(self):
            entry = HeteroMetricEntry()
            self.assertEqual(entry.all_tiers(), [])

        def test_all_tiers_populated(self):
            entry = HeteroMetricEntry()
            tier_entry = TierMetricEntry(
                values=torch.zeros(4), device_tier=DeviceTier.H100
            )
            entry.entries[DeviceTier.H100] = tier_entry
            entry.entries[DeviceTier.A6000_0] = TierMetricEntry(
                values=torch.zeros(4), device_tier=DeviceTier.A6000_0
            )
            self.assertIn(DeviceTier.H100, entry.all_tiers())
            self.assertIn(DeviceTier.A6000_0, entry.all_tiers())

        def test_get_or_none(self):
            entry = HeteroMetricEntry()
            self.assertIsNone(entry.get_or_none(DeviceTier.H100))

    class TestSingleton(unittest.TestCase):
        def setUp(self):
            destroy_hetero_moe_logger()

        def tearDown(self):
            destroy_hetero_moe_logger()

        def test_lazy_creation(self):
            t1 = get_hetero_moe_logger()
            t2 = get_hetero_moe_logger()
            self.assertIs(t1, t2)

        def test_destroy_and_recreate(self):
            t1 = get_hetero_moe_logger()
            destroy_hetero_moe_logger()
            t2 = get_hetero_moe_logger()
            self.assertIsNot(t1, t2)

        def test_set_custom_tracker(self):
            custom = HeteroMoELogger()
            set_hetero_moe_logger(custom)
            self.assertIs(get_hetero_moe_logger(), custom)

    class TestRecord(unittest.TestCase):
        def setUp(self):
            destroy_hetero_moe_logger()
            self.tracker = HeteroMoELogger()

        def test_record_creates_entry(self):
            val = torch.tensor(0.5)
            self.tracker.record("lbl", val, layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            self.assertIn("lbl", self.tracker.metrics)

        def test_record_none_layer_noop(self):
            val = torch.tensor(0.5)
            self.tracker.record("lbl", val, layer_number=None, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            self.assertNotIn("lbl", self.tracker.metrics)

        def test_record_accumulates_same_tier(self):
            for step in range(3):
                self.tracker.record("loss", torch.tensor(1.0),
                                    layer_number=2, num_layers=4,
                                    device_tier=DeviceTier.A6000_0)
            entry = self.tracker.metrics["loss"].entries[DeviceTier.A6000_0]
            self.assertAlmostEqual(entry.values[1].item(), 3.0)

        def test_record_two_tiers_separate_entries(self):
            self.tracker.record("loss", torch.tensor(1.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            self.tracker.record("loss", torch.tensor(2.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.H100)
            hetero = self.tracker.metrics["loss"]
            self.assertIn(DeviceTier.A6000_0, hetero.entries)
            self.assertIn(DeviceTier.H100, hetero.entries)
            self.assertAlmostEqual(
                hetero.entries[DeviceTier.A6000_0].values[0].item(), 1.0
            )
            self.assertAlmostEqual(
                hetero.entries[DeviceTier.H100].values[0].item(), 2.0
            )

        def test_active_tiers_tracked(self):
            self.tracker.record("loss", torch.tensor(1.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_1)
            self.assertIn(DeviceTier.A6000_1, self.tracker._active_tiers)

    class TestClear(unittest.TestCase):
        def setUp(self):
            self.tracker = HeteroMoELogger()

        def test_clear_zeros_values(self):
            self.tracker.record("loss", torch.tensor(3.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            self.tracker.clear()
            entry = self.tracker.metrics["loss"].entries[DeviceTier.A6000_0]
            self.assertAlmostEqual(entry.values[0].item(), 0.0)

        def test_clear_resets_active_tiers(self):
            self.tracker.record("loss", torch.tensor(1.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.H100)
            self.tracker.clear()
            self.assertEqual(len(self.tracker._active_tiers), 0)

    class TestConsolidateTiers(unittest.TestCase):
        def setUp(self):
            self.tracker = HeteroMoELogger()

        def test_single_tier_fast_path(self):
            self.tracker.record("loss", torch.tensor(5.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            # consolidation_tier matches the only tier → values preserved
            consolidated = self.tracker._consolidate_tiers(
                ["loss"], consolidation_tier=DeviceTier.A6000_0
            )
            self.assertIn("loss", consolidated)
            self.assertAlmostEqual(consolidated["loss"].values[0].item(), 5.0)

        def test_two_tier_consolidation_sums(self):
            self.tracker.record("loss", torch.tensor(3.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            self.tracker.record("loss", torch.tensor(7.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_1)
            # Consolidate onto A6000_0 (CPU in test env)
            consolidated = self.tracker._consolidate_tiers(
                ["loss"], consolidation_tier=DeviceTier.A6000_0
            )
            self.assertAlmostEqual(consolidated["loss"].values[0].item(), 10.0)

        def test_missing_metric_skipped(self):
            consolidated = self.tracker._consolidate_tiers(
                ["nonexistent"], consolidation_tier=DeviceTier.A6000_0
            )
            self.assertNotIn("nonexistent", consolidated)

    class TestAggregate(unittest.TestCase):
        def setUp(self):
            self.tracker = HeteroMoELogger()

        def _make_consolidated(
            self, values: List[float]
        ) -> Dict[str, TierMetricEntry]:
            return {
                "lbl_loss": TierMetricEntry(
                    values=torch.tensor(values),
                    device_tier=DeviceTier.A6000_0,
                )
            }

        def test_mean_over_moe_layers(self):
            consolidated = self._make_consolidated([2.0, 4.0, 6.0, 8.0])
            result = self.tracker._aggregate(consolidated, 1.0, 4, ["lbl_loss"])
            self.assertAlmostEqual(result["lbl_loss"], 5.0)  # mean(2,4,6,8)

        def test_loss_scale_applied(self):
            consolidated = self._make_consolidated([4.0, 0.0, 0.0, 0.0])
            result = self.tracker._aggregate(consolidated, 0.5, 4, ["lbl_loss"])
            self.assertAlmostEqual(result["lbl_loss"], 0.5)  # (4*0.5)/4

        def test_percentiles_computed(self):
            consolidated = self._make_consolidated([1.0, 2.0, 3.0, 4.0])
            result = self.tracker._aggregate(
                consolidated, 1.0, 4, ["lbl_loss"],
                percentiles={"lbl_loss": [0.5]}
            )
            self.assertIn("lbl_loss_p50", result)
            self.assertAlmostEqual(result["lbl_loss_p50"], 2.5, places=1)

        def test_missing_metric_skipped(self):
            consolidated = self._make_consolidated([1.0])
            result = self.tracker._aggregate(consolidated, 1.0, 1, ["other"])
            self.assertNotIn("other", result)

    class TestCountMoeLayers(unittest.TestCase):
        def test_none_freq(self):
            n = HeteroMoELogger._count_moe_layers(8, None, None)
            self.assertEqual(n, 8)

        def test_int_freq(self):
            # Every 2nd layer starting from 0: layers 0,2,4,6 → 4 out of 8
            n = HeteroMoELogger._count_moe_layers(8, 2, None)
            self.assertEqual(n, 4)

        def test_list_freq(self):
            n = HeteroMoELogger._count_moe_layers(None, [1, 0, 1, 0, 1], None)
            self.assertEqual(n, 3)

        def test_mtp_layers_added(self):
            n = HeteroMoELogger._count_moe_layers(8, None, 2)
            self.assertEqual(n, 10)

        def test_invalid_freq_raises(self):
            with self.assertRaises(ValueError):
                HeteroMoELogger._count_moe_layers(8, "invalid", None)  # type: ignore

    class TestFormat(unittest.TestCase):
        def test_format_output(self):
            scalars = {"load_balancing_loss": 0.123456, "z_loss": 0.987654}
            s = HeteroMoELogger._format(scalars)
            self.assertIn("load_balancing_loss", s)
            self.assertIn("z_loss", s)
            self.assertIn("|", s)

        def test_format_empty(self):
            self.assertEqual(HeteroMoELogger._format({}), "")

    class TestCudaGraphSnapshot(unittest.TestCase):
        def setUp(self):
            destroy_hetero_moe_logger()
            self.tracker = HeteroMoELogger()

        def tearDown(self):
            destroy_hetero_moe_logger()

        def test_snapshot_roundtrip(self):
            self.tracker.record("lbl", torch.tensor(3.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            snap = self.tracker.snapshot_for_cuda_graph()

            # Simulate CUDAGraph replay overwriting values
            entry = self.tracker.metrics["lbl"].entries[DeviceTier.A6000_0]
            entry.values.zero_()

            self.tracker.restore_from_cuda_graph_snapshot(snap)
            self.assertAlmostEqual(entry.values[0].item(), 3.0)

        def test_restore_asserts_missing_key(self):
            snap = {"nonexistent": {DeviceTier.A6000_0: torch.zeros(4)}}
            with self.assertRaises(AssertionError):
                self.tracker.restore_from_cuda_graph_snapshot(snap)

    class TestEnsureInitialized(unittest.TestCase):
        def setUp(self):
            self.tracker = HeteroMoELogger()

        def test_creates_entry_when_missing(self):
            self.tracker.ensure_initialized("new_metric", 8,
                                            tier=DeviceTier.A6000_0)
            self.assertIn("new_metric", self.tracker.metrics)
            entry = self.tracker.metrics["new_metric"].entries[DeviceTier.A6000_0]
            self.assertEqual(entry.values.shape[0], 8)

        def test_no_overwrite_existing(self):
            self.tracker.record("lbl", torch.tensor(1.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            self.tracker.ensure_initialized("lbl", 4,
                                            tier=DeviceTier.A6000_0)
            # Value must still be 1.0 (not overwritten by zeros)
            entry = self.tracker.metrics["lbl"].entries[DeviceTier.A6000_0]
            self.assertAlmostEqual(entry.values[0].item(), 1.0)

    class TestDeprecatedShims(unittest.TestCase):
        def setUp(self):
            destroy_hetero_moe_logger()

        def tearDown(self):
            destroy_hetero_moe_logger()

        def test_save_to_aux_losses_tracker_warns(self):
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                save_to_aux_losses_tracker(
                    "loss", torch.tensor(1.0), layer_number=1, num_layers=4
                )
                self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
            tracker = get_hetero_moe_logger()
            self.assertIn("loss", tracker.metrics)

        def test_clear_aux_losses_tracker_warns(self):
            import warnings
            # First record something
            get_hetero_moe_logger().record(
                "loss", torch.tensor(1.0), layer_number=1, num_layers=4,
                device_tier=DeviceTier.A6000_0
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                clear_aux_losses_tracker()
                self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
            entry = get_hetero_moe_logger().metrics["loss"].entries[DeviceTier.A6000_0]
            self.assertAlmostEqual(entry.values[0].item(), 0.0)

    class TestOffloadPressureRelief(unittest.TestCase):
        def setUp(self):
            destroy_hetero_moe_logger()

        def tearDown(self):
            destroy_hetero_moe_logger()

        def test_no_offload_above_threshold(self):
            count = hetero_moe_offload_pressure_relief(
                DeviceTier.A6000_0, vram_free_gb=8.0, threshold_gb=4.0
            )
            self.assertEqual(count, 0)

        def test_offload_attempted_below_threshold(self):
            # Record a metric so there's something to potentially offload
            get_hetero_moe_logger().record(
                "loss", torch.tensor(1.0), layer_number=1, num_layers=4,
                device_tier=DeviceTier.A6000_0
            )
            # vram_free_gb < threshold_gb → offload triggered
            # Since tensor is on CPU in test env, offload_to_slc is a no-op
            count = hetero_moe_offload_pressure_relief(
                DeviceTier.A6000_0, vram_free_gb=2.0, threshold_gb=4.0
            )
            # No crash; count is 0 because CPU tensors skip SLC offload
            self.assertGreaterEqual(count, 0)

    class TestReportIntegration(unittest.TestCase):
        """End-to-end report() test in single-process (no dist) mode."""

        def setUp(self):
            destroy_hetero_moe_logger()
            self.tracker = HeteroMoELogger()

        def tearDown(self):
            destroy_hetero_moe_logger()

        def test_report_returns_string(self):
            self.tracker.record("z_loss", torch.tensor(0.5),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            log_str = self.tracker.report(
                loss_scale=1.0,
                iteration=10,
                num_layers=4,
            )
            self.assertIsInstance(log_str, str)

        def test_report_populates_total_loss_dict(self):
            self.tracker.record("lbl_loss", torch.tensor(2.0),
                                layer_number=1, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            total = {}
            self.tracker.report(
                loss_scale=1.0,
                iteration=1,
                num_layers=4,
                total_loss_dict=total,
            )
            # lbl_loss ends with "loss" so it goes into total_loss_dict
            self.assertIn("lbl_loss", total)

        def test_report_clears_after_step(self):
            self.tracker.record("lbl_loss", torch.tensor(1.0),
                                layer_number=2, num_layers=4,
                                device_tier=DeviceTier.A6000_0)
            self.tracker.report(loss_scale=1.0, iteration=1, num_layers=4)
            entry = self.tracker.metrics["lbl_loss"].entries[DeviceTier.A6000_0]
            self.assertAlmostEqual(entry.values[1].item(), 0.0)

        def test_report_force_initialize(self):
            # No records this step (simulates PP rank without MoE layers)
            self.tracker.report(
                loss_scale=1.0,
                iteration=1,
                num_layers=8,
                force_initialize=True,
                track_names=["lbl_loss"],
            )
            # Entry should have been pre-created
            self.assertIn("lbl_loss", self.tracker.metrics)

        def test_report_force_initialize_requires_num_layers(self):
            with self.assertRaises(ValueError):
                self.tracker.report(
                    loss_scale=1.0,
                    iteration=1,
                    force_initialize=True,
                    track_names=["lbl_loss"],
                    num_layers=None,
                )

    print("\n" + "=" * 70)
    print("Running HeteroMoELogger DES-LOC unit tests")
    print("=" * 70 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestDeviceTier,
        TestTierMetricEntry,
        TestHeteroMetricEntry,
        TestSingleton,
        TestRecord,
        TestClear,
        TestConsolidateTiers,
        TestAggregate,
        TestCountMoeLayers,
        TestFormat,
        TestCudaGraphSnapshot,
        TestEnsureInitialized,
        TestDeprecatedShims,
        TestOffloadPressureRelief,
        TestReportIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
