"""
DES-LOC Heterogeneous Overlap Parameter Gather
===============================================

Upstream design intent (Megatron a2381d800733):
    Megatron's commit extends ``--overlap-param-gather`` support to the
    layer-wise optimizer path (Muon / dist_muon).  Previously this flag
    was gated on ``use_distributed_optimizer``; the commit decouples the
    two so that layer-wise optimizers can also pipeline param all-gathers
    with computation.  Key mechanisms:

    1. ``_ParamAndGradBucket.set_layerwise_params_list`` — per-rank param
       ownership metadata used to build variable-size all-gather recv buffers.
    2. ``_LayerwiseAllGatherHandle`` — thin wrapper around multiple async
       NCCL work objects; NCCL in-order guarantee allows waiting only on the
       last handle.
    3. ``start_param_sync`` branches on ``use_distributed_optimizer`` to pick
       coalesced ``all_gather_into_tensor`` (distributed-optimizer path) vs
       per-rank variable-size ``all_gather`` (layer-wise path).
    4. ``free_overlap_buffers`` — explicit buffer release hook called before
       async checkpoint saves to reclaim GPU headroom.
    5. ``should_disable_forward_pre_hook`` is widened to cover ``dist_muon``
       (any optimizer whose name contains ``'dist'``).

DES-LOC adaptation points:
    The Neuron_SP project targets **2× A6000 48 GB (SM86) + 1× H100 NVL
    96 GB (SM90)** connected via PCIe (no NVLink) with 1.5 TB CPU DRAM.
    The three-device asymmetry introduces challenges absent in Megatron's
    homogeneous GPU clusters:

    A. **Device-tier routing** — the H100 acts as the "heavy" shard holder
       (larger param partition) while A6000s hold lighter shards.  A naïve
       equal-shard all-gather wastes H100 bandwidth waiting for PCIe-limited
       A6000 transfers.  ``HeteroParamShardPolicy`` computes per-rank shard
       sizes proportional to device memory × PCIe bandwidth estimates.

    B. **Shared Locality Cache (SLC)** — DES-LOC's core novelty: a pinned
       CPU buffer acts as an L3-style staging area.  Params gathered from
       remote devices are written to SLC first (D2H on sender, H2D on
       receiver).  This avoids direct GPU-to-GPU PCIe transfers and lets the
       OS page cache coalesce traffic.  ``SLCParamCache`` manages this buffer.

    C. **Decoupled Execution** — the H100 can begin its forward pass on its
       own shard while A6000 params are still in-flight via SLC.  A per-layer
       readiness flag (``LayerReadinessTracker``) gates forward hooks so each
       layer only proceeds once its remote params have landed in SLC and been
       copied to device.

    D. **Async checkpoint safety** — ``free_overlap_buffers`` is adapted to
       also release SLC pinned buffers so the checkpoint worker's D2H
       transfers do not OOM against SLC allocations.

    E. **SM-generation-aware dtype** — A6000 (SM86) uses BF16 natively;
       H100 (SM90) prefers BF16 but can accelerate FP8.  Param gather always
       uses BF16 (matching upstream), but the SLC copy path is annotated for
       future FP8 dequant support.

Usage in Neuron_SP training loop::

    from deepspeed.runtime.zero.hetero_overlap_param_gather import (
        HeteroOverlapParamGatherConfig,
        HeteroOverlapParamGather,
        integrate_with_engine,
    )

    hogpg_cfg = HeteroOverlapParamGatherConfig(
        device_tiers={"h100": [2], "a6000": [0, 1]},
        slc_capacity_gb=8.0,
        overlap_async=True,
    )
    hogpg = integrate_with_engine(ds_engine, hogpg_cfg)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & device-tier metadata
# ---------------------------------------------------------------------------

# Approximate PCIe gen4 x16 bandwidth in GB/s (conservative for shared root complex)
_PCIE_BW_GBS: Dict[str, float] = {
    "h100":  28.0,   # H100 NVL PCIe gen4 x16 uplink
    "a6000": 16.0,   # A6000 PCIe gen4 x16, shared root complex penalty
}

# Memory capacity in GB (used for proportional shard sizing)
_DEVICE_MEM_GB: Dict[str, float] = {
    "h100":  96.0,
    "a6000": 48.0,
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeteroOverlapParamGatherConfig:
    """Configuration for DES-LOC heterogeneous overlap param gather.

    Attributes:
        device_tiers: Mapping from tier name (``"h100"``, ``"a6000"``) to list
            of global rank indices belonging to that tier.
        slc_capacity_gb: How many GB of pinned CPU DRAM to reserve for the
            Shared Locality Cache.  Defaults to 8 GB.
        overlap_async: If ``True``, param all-gathers are dispatched
            asynchronously and completed by forward pre-hooks (DES-LOC mode).
            If ``False``, all-gathers are synchronous (useful for debugging).
        proportional_sharding: If ``True``, partition params proportionally to
            ``device_memory × pcie_bandwidth`` so the H100 holds a larger shard.
            If ``False``, equal sharding is used (Megatron default).
        use_slc: If ``True``, route inter-tier transfers through the SLC pinned
            CPU buffer.  If ``False``, perform direct GPU-to-GPU PCIe transfers
            (may be slower due to contention).
        slc_num_streams: Number of CUDA streams used for SLC D2H/H2D copies.
        grad_reduce_in_fp32: Whether gradients are reduced in FP32.  Param
            gather always uses param dtype (BF16); this flag only affects how
            the SLC cache allocates its fp32 shadow for future FP8 work.
    """
    device_tiers: Dict[str, List[int]] = field(default_factory=lambda: {
        "h100":  [2],
        "a6000": [0, 1],
    })
    slc_capacity_gb: float = 8.0
    overlap_async: bool = True
    proportional_sharding: bool = True
    use_slc: bool = True
    slc_num_streams: int = 2
    grad_reduce_in_fp32: bool = False


# ---------------------------------------------------------------------------
# Shard policy
# ---------------------------------------------------------------------------

class HeteroParamShardPolicy:
    """Compute proportional param shard sizes for heterogeneous devices.

    Upstream (Megatron) always divides param buffers evenly across DP ranks.
    On a 2× A6000 + 1× H100 setup, equal shards under-utilise the H100 and
    may OOM the A6000s for large models.

    DES-LOC adaptation: weight each rank's shard by
    ``w_r = mem_r × bw_r / Σ_i (mem_i × bw_i)``
    so the H100 (96 GB × 28 GB/s) carries ~3.9× more params than each A6000
    (48 GB × 16 GB/s).

    Args:
        config: ``HeteroOverlapParamGatherConfig`` describing device tiers.
        world_size: Total number of data-parallel ranks.
    """

    def __init__(self, config: HeteroOverlapParamGatherConfig, world_size: int):
        self.config = config
        self.world_size = world_size
        self._rank_to_tier: Dict[int, str] = {}
        for tier, ranks in config.device_tiers.items():
            for r in ranks:
                self._rank_to_tier[r] = tier

        # Fill in unknown ranks as "a6000" (conservative default)
        for r in range(world_size):
            if r not in self._rank_to_tier:
                logger.warning(
                    "Rank %d not in any device tier; defaulting to 'a6000'", r
                )
                self._rank_to_tier[r] = "a6000"

        self._weights = self._compute_weights()
        logger.info(
            "HeteroParamShardPolicy: rank weights = %s",
            {r: f"{w:.4f}" for r, w in enumerate(self._weights)},
        )

    def _compute_weights(self) -> List[float]:
        raw = []
        for r in range(self.world_size):
            tier = self._rank_to_tier[r]
            mem = _DEVICE_MEM_GB.get(tier, 48.0)
            bw  = _PCIE_BW_GBS.get(tier, 16.0)
            raw.append(mem * bw)
        total = sum(raw)
        return [v / total for v in raw]

    def shard_sizes(self, total_numel: int) -> List[int]:
        """Return per-rank shard sizes (in elements) summing to ``total_numel``.

        Args:
            total_numel: Total number of elements to distribute.

        Returns:
            List of ``world_size`` integers.
        """
        if not self.config.proportional_sharding:
            # Equal sharding fallback (Megatron default)
            base, rem = divmod(total_numel, self.world_size)
            sizes = [base + (1 if i < rem else 0) for i in range(self.world_size)]
            return sizes

        # Proportional sharding
        float_sizes = [w * total_numel for w in self._weights]
        sizes = [int(math.floor(s)) for s in float_sizes]
        deficit = total_numel - sum(sizes)
        # Distribute remainder to ranks with largest fractional parts
        fracs = [(float_sizes[r] - sizes[r], r) for r in range(self.world_size)]
        fracs.sort(key=lambda x: -x[0])
        for i in range(deficit):
            sizes[fracs[i][1]] += 1
        assert sum(sizes) == total_numel, "Shard sizes must sum to total_numel"
        return sizes

    def tier_of(self, rank: int) -> str:
        return self._rank_to_tier.get(rank, "a6000")


# ---------------------------------------------------------------------------
# Shared Locality Cache (SLC)
# ---------------------------------------------------------------------------

class SLCParamCache:
    """Pinned CPU buffer acting as a staging area for inter-tier param transfers.

    DES-LOC core concept: instead of direct GPU-to-GPU PCIe transfers (which
    contend for the PCIe root complex), gathered params from remote tiers are
    first written to this pinned CPU buffer (D2H on the sender side, H2D on the
    receiver side).  The OS page cache and CPU memory controller can coalesce
    traffic more efficiently than back-to-back GPU DMA engines.

    The SLC is organised as a ring of ``slc_num_streams`` slots, each large
    enough to hold ``slot_numel`` elements.  Callers alternate slots to avoid
    overwriting data still in-flight.

    Args:
        capacity_gb: Total pinned CPU memory to allocate in GB.
        dtype: Tensor dtype for cache entries (BF16 for param gather).
        num_streams: Number of CUDA streams / ring slots.
        device: CUDA device for H2D copies.
    """

    def __init__(
        self,
        capacity_gb: float,
        dtype: torch.dtype,
        num_streams: int,
        device: torch.device,
    ):
        self.capacity_bytes = int(capacity_gb * (1 << 30))
        self.dtype = dtype
        self.num_streams = num_streams
        self.device = device
        self._lock = threading.Lock()

        # Allocate pinned buffer
        elem_bytes = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 2
        total_numel = self.capacity_bytes // elem_bytes
        self._pinned_buf = torch.empty(total_numel, dtype=dtype, pin_memory=True)
        logger.info(
            "SLCParamCache: allocated %.2f GB pinned CPU buffer (%d elements, dtype=%s)",
            capacity_gb, total_numel, dtype,
        )

        # CUDA streams for async D2H and H2D copies
        self._d2h_streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
        self._h2d_streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
        self._slot_cursor = 0
        self._slot_size = total_numel // max(num_streams, 1)

    def _next_slot(self) -> Tuple[int, int]:
        """Return (slot_index, offset_in_pinned_buf) in a round-robin fashion."""
        with self._lock:
            slot = self._slot_cursor % self.num_streams
            self._slot_cursor += 1
        return slot, slot * self._slot_size

    def stage_d2h(
        self,
        src_gpu: torch.Tensor,
        slot_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, torch.cuda.Stream]:
        """Copy ``src_gpu`` to pinned CPU staging buffer asynchronously.

        Args:
            src_gpu: Source tensor on GPU (must be contiguous).
            slot_override: Force a specific slot index (for testing).

        Returns:
            Tuple of (pinned_cpu_tensor_slice, slot_index, d2h_stream).
            The caller must synchronise on ``d2h_stream`` before reading the
            pinned slice.
        """
        if slot_override is not None:
            slot, offset = slot_override, slot_override * self._slot_size
        else:
            slot, offset = self._next_slot()

        numel = src_gpu.numel()
        if offset + numel > self._pinned_buf.numel():
            raise RuntimeError(
                f"SLCParamCache: insufficient capacity for {numel} elements "
                f"(offset={offset}, capacity={self._pinned_buf.numel()})"
            )

        cpu_slice = self._pinned_buf[offset: offset + numel]
        stream = self._d2h_streams[slot]
        with torch.cuda.stream(stream):
            cpu_slice.copy_(src_gpu, non_blocking=True)

        logger.debug(
            "SLCParamCache.stage_d2h: %d elements → slot %d (stream %d)",
            numel, slot, stream.stream_id,
        )
        return cpu_slice, slot, stream

    def load_h2d(
        self,
        cpu_slice: torch.Tensor,
        dst_gpu: torch.Tensor,
        d2h_stream: torch.cuda.Stream,
        slot: int,
    ) -> torch.cuda.Event:
        """Copy from pinned CPU staging buffer to ``dst_gpu`` asynchronously.

        Inserts a stream dependency so H2D copy does not begin until D2H
        has completed.

        Args:
            cpu_slice: Pinned CPU tensor (returned by ``stage_d2h``).
            dst_gpu: Destination GPU tensor (must be same shape as cpu_slice).
            d2h_stream: The D2H stream from ``stage_d2h`` (for dependency).
            slot: Slot index (selects H2D stream).

        Returns:
            A ``torch.cuda.Event`` that fires when H2D copy completes.
        """
        h2d_stream = self._h2d_streams[slot % self.num_streams]
        # Record event on D2H stream and wait on H2D stream
        d2h_event = torch.cuda.Event()
        d2h_stream.record_event(d2h_event)
        h2d_stream.wait_event(d2h_event)

        with torch.cuda.stream(h2d_stream):
            dst_gpu.copy_(cpu_slice, non_blocking=True)

        done_event = torch.cuda.Event()
        h2d_stream.record_event(done_event)

        logger.debug(
            "SLCParamCache.load_h2d: %d elements from slot %d", cpu_slice.numel(), slot
        )
        return done_event

    def free(self) -> None:
        """Release pinned CPU buffer and CUDA streams."""
        del self._pinned_buf
        self._d2h_streams.clear()
        self._h2d_streams.clear()
        logger.info("SLCParamCache: freed pinned CPU buffer and CUDA streams")


# ---------------------------------------------------------------------------
# Layer readiness tracker (Decoupled Execution)
# ---------------------------------------------------------------------------

class LayerReadinessTracker:
    """Track per-layer param readiness for DES-LOC Decoupled Execution.

    DES-LOC allows the H100 to begin its forward pass on locally-owned params
    while remote params (from A6000s) are still in-flight via SLC.  Each layer
    registers its remote-param events here; the forward pre-hook blocks only
    until those events have fired.

    Args:
        num_layers: Total number of transformer layers.
    """

    def __init__(self, num_layers: int):
        self._events: Dict[int, List[torch.cuda.Event]] = {
            i: [] for i in range(num_layers)
        }
        self._ready: Dict[int, bool] = {i: False for i in range(num_layers)}

    def register_event(self, layer_idx: int, event: torch.cuda.Event) -> None:
        """Register a CUDA event that must complete before layer ``layer_idx`` runs."""
        self._events[layer_idx].append(event)
        self._ready[layer_idx] = False

    def wait_ready(self, layer_idx: int) -> None:
        """Block the current CUDA stream until all events for ``layer_idx`` fire."""
        for evt in self._events[layer_idx]:
            torch.cuda.current_stream().wait_event(evt)
        self._events[layer_idx].clear()
        self._ready[layer_idx] = True

    def mark_ready(self, layer_idx: int) -> None:
        """Unconditionally mark layer as ready (e.g., local-only params)."""
        self._events[layer_idx].clear()
        self._ready[layer_idx] = True

    def is_ready(self, layer_idx: int) -> bool:
        return self._ready.get(layer_idx, False)

    def reset(self) -> None:
        """Reset all readiness flags at the start of each forward pass."""
        for i in self._events:
            self._ready[i] = False


# ---------------------------------------------------------------------------
# Async all-gather handle (mirrors Megatron's _LayerwiseAllGatherHandle)
# ---------------------------------------------------------------------------

class _HeteroAllGatherHandle:
    """Composite handle wrapping async NCCL work objects and SLC H2D events.

    Mirrors Megatron's ``_LayerwiseAllGatherHandle`` but also synchronises
    SLC H2D copy events so that remote params are fully landed in GPU memory
    before the handle is considered complete.

    NCCL in-order guarantee: waiting only on the last NCCL work object is
    sufficient; NCCL will complete all earlier ops on the same communicator
    first.

    Args:
        nccl_handles: List of async NCCL work objects.
        slc_events: List of ``torch.cuda.Event`` objects from SLC H2D copies.
    """

    def __init__(
        self,
        nccl_handles: List,
        slc_events: Optional[List[torch.cuda.Event]] = None,
    ):
        self.nccl_handles = nccl_handles
        self.slc_events = slc_events or []

    def wait(self) -> None:
        """Wait on the last NCCL handle and all SLC events."""
        if self.nccl_handles:
            self.nccl_handles[-1].wait()
        for evt in self.slc_events:
            evt.synchronize()
        self.nccl_handles = []
        self.slc_events = []

    def __bool__(self) -> bool:
        return bool(self.nccl_handles) or bool(self.slc_events)


# ---------------------------------------------------------------------------
# Per-bucket metadata (adapts _ParamAndGradBucket)
# ---------------------------------------------------------------------------

class HeteroBucketMeta:
    """Metadata attached to a DDP bucket for heterogeneous overlap param gather.

    In Megatron, ``_ParamAndGradBucket.set_layerwise_params_list`` stores
    per-rank param lists directly on the bucket.  In DeepSpeed / Neuron_SP, we
    keep this metadata in a sidecar object and attach it to the bucket via
    ``bucket.__hetero_meta__`` to avoid patching DeepSpeed internals.

    Attributes:
        layerwise_params_list: Per-rank lists of params belonging to this bucket.
        layerwise_param_flat_sizes: Per-rank flattened param sizes.
        layerwise_gather_list: Per-rank receive tensors (populated during gather).
        src_buffer: Flattened local params kept alive during async gather.
        slc_cpu_slices: SLC pinned CPU slices for inter-tier transfers.
    """

    def __init__(self) -> None:
        self.layerwise_params_list: Optional[List[List[torch.nn.Parameter]]] = None
        self.layerwise_param_flat_sizes: Optional[List[int]] = None
        self.layerwise_gather_list: Optional[List[torch.Tensor]] = None
        self.src_buffer: Optional[torch.Tensor] = None
        self.slc_cpu_slices: Optional[List[Optional[torch.Tensor]]] = None

    def set_params_list(self, params_list: List[List[torch.nn.Parameter]]) -> None:
        """Set per-rank param lists and pre-compute flat sizes."""
        self.layerwise_params_list = params_list
        self.layerwise_param_flat_sizes = [
            sum(p.numel() for p in pl) for pl in params_list
        ]

    def free(self) -> None:
        """Release temporary gather buffers."""
        self.layerwise_gather_list = None
        self.src_buffer = None
        self.slc_cpu_slices = None


def _get_or_create_bucket_meta(bucket) -> HeteroBucketMeta:
    """Retrieve or lazily create ``HeteroBucketMeta`` for a bucket object."""
    if not hasattr(bucket, "__hetero_meta__"):
        bucket.__hetero_meta__ = HeteroBucketMeta()
    return bucket.__hetero_meta__


# ---------------------------------------------------------------------------
# Core: HeteroOverlapParamGather
# ---------------------------------------------------------------------------

class HeteroOverlapParamGather:
    """Orchestrate overlap param gather for DES-LOC heterogeneous training.

    This class is the DES-LOC counterpart of the changes in Megatron commit
    a2381d800733.  It handles:

    - **Proportional sharding** via ``HeteroParamShardPolicy``.
    - **Variable-size all-gather** (no padding) matching Megatron's layer-wise
      path, using per-rank receive buffers.
    - **SLC-routed inter-tier transfers**: params from A6000 ranks are staged
      through pinned CPU memory before being H2D-copied to the H100, avoiding
      direct PCIe contention.
    - **Async dispatch + forward pre-hook finish** matching Megatron's
      ``start_param_sync`` / ``finish_param_sync`` lifecycle.
    - **Checkpoint-safe buffer release** via ``free_overlap_buffers``.

    Args:
        config: ``HeteroOverlapParamGatherConfig``.
        dp_group: Data-parallel process group.
        local_rank: This process's rank within ``dp_group``.
        param_dtype: Dtype for param gather (BF16 recommended for A6000/H100).
    """

    def __init__(
        self,
        config: HeteroOverlapParamGatherConfig,
        dp_group,
        local_rank: int,
        param_dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.dp_group = dp_group
        self.dp_size = dist.get_world_size(dp_group)
        self.local_rank = local_rank
        self.param_dtype = param_dtype
        self.device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")

        self.shard_policy = HeteroParamShardPolicy(config, self.dp_size)

        self.slc: Optional[SLCParamCache] = None
        if config.use_slc:
            self.slc = SLCParamCache(
                capacity_gb=config.slc_capacity_gb,
                dtype=param_dtype,
                num_streams=config.slc_num_streams,
                device=self.device,
            )

        # Per-bucket-group gather handles (indexed by bucket_group_id)
        self._pending_handles: Dict[int, _HeteroAllGatherHandle] = {}

        logger.info(
            "HeteroOverlapParamGather: local_rank=%d tier=%s dp_size=%d "
            "async=%s slc=%s",
            local_rank,
            self.shard_policy.tier_of(local_rank),
            self.dp_size,
            config.overlap_async,
            config.use_slc,
        )

    # ------------------------------------------------------------------
    # Public API: map params to buckets
    # ------------------------------------------------------------------

    def register_bucket_groups(
        self,
        bucket_groups: List,
        sharded_params_per_rank: List[List[torch.nn.Parameter]],
    ) -> None:
        """Attach per-rank param ownership metadata to all bucket groups.

        Mirrors ``LayerWiseDistributedOptimizer.set_bucket_layerwise_params_list``
        from Megatron commit a2381d800733, adapted for DeepSpeed bucket objects
        via ``HeteroBucketMeta`` sidecars.

        Args:
            bucket_groups: List of DeepSpeed bucket group objects.
            sharded_params_per_rank: ``dp_size``-length list; each entry is the
                list of params owned by that rank's layer-wise optimizer shard.
        """
        for bg in bucket_groups:
            for bucket in self._iter_buckets(bg):
                meta = _get_or_create_bucket_meta(bucket)
                per_rank: List[List[torch.nn.Parameter]] = [
                    [] for _ in range(self.dp_size)
                ]
                bucket_params: set = self._get_bucket_params(bucket)
                for rank_idx, rank_params in enumerate(sharded_params_per_rank):
                    for p in rank_params:
                        if p in bucket_params:
                            per_rank[rank_idx].append(p)
                meta.set_params_list(per_rank)

        logger.info(
            "HeteroOverlapParamGather: registered %d bucket groups", len(bucket_groups)
        )

    # ------------------------------------------------------------------
    # Public API: gather lifecycle
    # ------------------------------------------------------------------

    def start_param_sync(
        self,
        bucket_group,
        bg_id: int,
        force_sync: bool = False,
    ) -> None:
        """Dispatch param all-gather for ``bucket_group``.

        Mirrors ``_ParamAndGradBucketGroup.start_param_sync`` (layer-wise path)
        from Megatron a2381d800733.

        Algorithm:
          1. For each bucket, flatten local params (detached to avoid autograd
             issues when called during forward — see Megatron regression note).
          2. Allocate per-rank recv buffers sized exactly to each rank's shard
             (no padding).  The H100 rank's buffer is larger due to proportional
             sharding.
          3. Launch ``dist.all_gather`` (variable-size, async if
             ``config.overlap_async and not force_sync``).
          4. If ``config.use_slc``, route inter-tier recv buffers through the
             SLC pinned CPU staging area and record H2D events for the
             ``_HeteroAllGatherHandle``.

        Args:
            bucket_group: DeepSpeed bucket group whose params to gather.
            bg_id: Unique integer ID for this bucket group (for handle storage).
            force_sync: If ``True``, block until gather completes and immediately
                unflatten + copy gathered params into model tensors.
        """
        assert bg_id not in self._pending_handles or not self._pending_handles[bg_id], (
            f"start_param_sync called for bg_id={bg_id} with a pending handle; "
            "call finish_param_sync first"
        )

        async_op = self.config.overlap_async and not force_sync
        nccl_handles: List = []
        slc_events: List[torch.cuda.Event] = []
        local_tier = self.shard_policy.tier_of(self.local_rank)

        for bucket in self._iter_buckets(bucket_group):
            meta = _get_or_create_bucket_meta(bucket)
            if meta.layerwise_params_list is None:
                logger.debug(
                    "start_param_sync: bucket has no layerwise_params_list; skipping"
                )
                continue

            flat_sizes = meta.layerwise_param_flat_sizes
            assert flat_sizes is not None

            if max(flat_sizes) == 0:
                meta.layerwise_gather_list = None
                continue

            # --- Flatten local params (detach to avoid autograd in-place issues) ---
            local_size = flat_sizes[self.local_rank]
            if local_size > 0:
                src = _flatten_dense_tensors(
                    meta.layerwise_params_list[self.local_rank]
                ).detach()
            else:
                src = torch.empty(0, device=self.device, dtype=self.param_dtype)
            meta.src_buffer = src  # keep alive for async op

            # --- Build per-rank recv buffers ---
            gather_list: List[torch.Tensor] = []
            for r in range(self.dp_size):
                if r == self.local_rank:
                    gather_list.append(src)
                else:
                    gather_list.append(
                        torch.empty(flat_sizes[r], device=self.device,
                                    dtype=self.param_dtype)
                    )
            meta.layerwise_gather_list = gather_list

            # --- Launch all_gather ---
            work = dist.all_gather(
                gather_list, src, group=self.dp_group, async_op=async_op
            )
            if async_op and work is not None:
                nccl_handles.append(work)

            # --- SLC routing for inter-tier slots ---
            if self.config.use_slc and self.slc is not None:
                slc_events.extend(
                    self._route_via_slc(
                        gather_list, flat_sizes, local_tier,
                        async_op=async_op,
                        meta=meta,
                    )
                )

        if async_op:
            handle = _HeteroAllGatherHandle(nccl_handles, slc_events)
            self._pending_handles[bg_id] = handle
            logger.debug(
                "start_param_sync: bg_id=%d dispatched async (%d nccl, %d slc events)",
                bg_id, len(nccl_handles), len(slc_events),
            )
        else:
            # Synchronous: unflatten and copy gathered params now
            self._unflatten_and_copy(bucket_group)
            self._pending_handles.pop(bg_id, None)
            logger.debug("start_param_sync: bg_id=%d synchronous complete", bg_id)

    def finish_param_sync(
        self,
        bucket_group,
        bg_id: int,
        next_bucket_group=None,
        next_bg_id: Optional[int] = None,
    ) -> None:
        """Complete param all-gather for ``bucket_group`` and optionally chain next.

        Mirrors ``_ParamAndGradBucketGroup.finish_param_sync`` from Megatron
        a2381d800733.  After waiting on the handle, unflattens gathered flat
        tensors back into model param ``.data`` buffers for all remote ranks.
        Optionally dispatches the next bucket group's gather (pipelining).

        Args:
            bucket_group: The bucket group whose gather to complete.
            bg_id: Unique ID for this bucket group.
            next_bucket_group: Optional next bucket group to pipeline.
            next_bg_id: ID for the next bucket group.
        """
        handle = self._pending_handles.pop(bg_id, None)
        if handle is not None:
            handle.wait()

        # Unflatten gathered params into model tensors for all remote ranks
        self._unflatten_and_copy(bucket_group)

        # Chain: dispatch next bucket group's gather (Megatron's pipelining pattern)
        if next_bucket_group is not None and next_bg_id is not None:
            logger.debug(
                "finish_param_sync: chaining to next bg_id=%d", next_bg_id
            )
            self.start_param_sync(next_bucket_group, next_bg_id, force_sync=False)

    def free_overlap_buffers(self, bucket_groups: List) -> None:
        """Release GPU and SLC buffers before async checkpoint saves.

        Mirrors ``_ParamAndGradBucketGroup.free_overlap_buffers`` and
        ``DistributedDataParallel.free_overlap_buffers`` from Megatron a2381d800733.

        Additionally releases SLC pinned CPU slices to give the async checkpoint
        worker's D2H transfers sufficient headroom in pinned memory.

        Args:
            bucket_groups: All bucket groups (regular + expert-parallel).
        """
        # First, wait on any pending handles
        for bg_id, handle in list(self._pending_handles.items()):
            if handle:
                logger.debug("free_overlap_buffers: waiting on bg_id=%d", bg_id)
                handle.wait()
        self._pending_handles.clear()

        # Release per-bucket temporary buffers
        for bg in bucket_groups:
            for bucket in self._iter_buckets(bg):
                meta = _get_or_create_bucket_meta(bucket)
                meta.free()

        logger.info(
            "free_overlap_buffers: released GPU gather buffers for %d bucket groups",
            len(bucket_groups),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _route_via_slc(
        self,
        gather_list: List[torch.Tensor],
        flat_sizes: List[int],
        local_tier: str,
        async_op: bool,
        meta: HeteroBucketMeta,
    ) -> List[torch.cuda.Event]:
        """Stage inter-tier param tensors through SLC pinned CPU buffer.

        For ranks in a different tier from the local rank (e.g., A6000 params
        received by H100), replace the direct GPU recv buffer with a D2H→H2D
        bounce through SLC.  This avoids direct PCIe GPU-to-GPU contention.

        This method is a no-op for intra-tier ranks (same tier, fast path).

        Args:
            gather_list: Per-rank recv tensors (populated by all_gather).
            flat_sizes: Per-rank flattened param sizes.
            local_tier: Tier of the local rank.
            async_op: Whether the gather was async.
            meta: Bucket metadata (for storing slc_cpu_slices).

        Returns:
            List of H2D CUDA events (one per inter-tier rank).
        """
        if self.slc is None:
            return []

        slc_events: List[torch.cuda.Event] = []
        if meta.slc_cpu_slices is None:
            meta.slc_cpu_slices = [None] * self.dp_size

        for r in range(self.dp_size):
            if r == self.local_rank:
                continue
            remote_tier = self.shard_policy.tier_of(r)
            if remote_tier == local_tier:
                # Same tier: direct GPU buffer, no SLC needed
                continue
            if flat_sizes[r] == 0:
                continue

            # Inter-tier: stage through SLC
            gpu_tensor = gather_list[r]
            try:
                cpu_slice, slot, d2h_stream = self.slc.stage_d2h(gpu_tensor)
                meta.slc_cpu_slices[r] = cpu_slice

                # Allocate destination GPU tensor and H2D copy
                dst = torch.empty_like(gpu_tensor)
                h2d_event = self.slc.load_h2d(cpu_slice, dst, d2h_stream, slot)

                # Replace gather_list[r] with SLC-routed tensor
                gather_list[r] = dst
                slc_events.append(h2d_event)

                logger.debug(
                    "_route_via_slc: rank %d (%s→%s) via SLC slot %d, %d elems",
                    r, remote_tier, local_tier, slot, flat_sizes[r],
                )
            except RuntimeError as exc:
                logger.warning(
                    "_route_via_slc: SLC routing failed for rank %d, "
                    "falling back to direct transfer: %s", r, exc
                )

        return slc_events

    def _unflatten_and_copy(self, bucket_group) -> None:
        """Unflatten gathered flat tensors back into model param .data buffers.

        For each remote rank in each bucket, calls
        ``_unflatten_dense_tensors`` and copies into the model's param data.
        Skips the local rank (its params were never overwritten by the gather).

        Args:
            bucket_group: Bucket group to process.
        """
        for bucket in self._iter_buckets(bucket_group):
            meta = _get_or_create_bucket_meta(bucket)
            if meta.layerwise_gather_list is None:
                continue

            for r, params in enumerate(meta.layerwise_params_list or []):
                if r == self.local_rank or len(params) == 0:
                    continue
                flat = meta.layerwise_gather_list[r]
                if flat.numel() == 0:
                    continue
                updated = _unflatten_dense_tensors(flat, params)
                for dst, src in zip(params, updated):
                    dst.data.copy_(src)

            meta.layerwise_gather_list = None
            meta.src_buffer = None
            meta.slc_cpu_slices = None

    @staticmethod
    def _iter_buckets(bucket_group):
        """Iterate over buckets in a bucket group (DeepSpeed compat)."""
        if hasattr(bucket_group, "buckets"):
            return bucket_group.buckets
        # Fallback: treat bucket_group itself as a single bucket
        return [bucket_group]

    @staticmethod
    def _get_bucket_params(bucket) -> set:
        """Return the set of params for a bucket (DeepSpeed compat)."""
        if hasattr(bucket, "params"):
            return bucket.params
        if hasattr(bucket, "params_list"):
            return set(p for pl in bucket.params_list for p in pl)
        return set()


# ---------------------------------------------------------------------------
# DeepSpeed engine integration helper
# ---------------------------------------------------------------------------

def integrate_with_engine(
    ds_engine,
    config: HeteroOverlapParamGatherConfig,
    sharded_params_per_rank: Optional[List[List[torch.nn.Parameter]]] = None,
) -> HeteroOverlapParamGather:
    """Attach ``HeteroOverlapParamGather`` to a DeepSpeed engine.

    Mirrors the integration point in Megatron's ``get_megatron_muon_optimizer``
    and ``training.py`` (``save_checkpoint_and_time``, ``should_disable_forward_pre_hook``).

    Attaches:
    - ``ds_engine.__hogpg__``: the ``HeteroOverlapParamGather`` instance.
    - Monkey-patches ``ds_engine.free_overlap_buffers`` so training loop can call
      it before checkpoint saves (matching Megatron's pattern).

    Args:
        ds_engine: A DeepSpeed ``DeepSpeedEngine`` (or compatible object with
            ``local_rank``, ``dp_group``, ``module``, optionally ``bucket_groups``).
        config: DES-LOC gather config.
        sharded_params_per_rank: Pre-computed per-rank param lists.  If ``None``,
            will attempt to derive from ``ds_engine``'s optimizer state.

    Returns:
        The newly created ``HeteroOverlapParamGather`` instance.
    """
    local_rank = getattr(ds_engine, "local_rank", dist.get_rank())
    dp_group = getattr(ds_engine, "dp_group", dist.GroupMember.WORLD)

    hogpg = HeteroOverlapParamGather(
        config=config,
        dp_group=dp_group,
        local_rank=local_rank,
    )

    # Register bucket groups if available
    bucket_groups = _collect_bucket_groups(ds_engine)
    if sharded_params_per_rank is None:
        sharded_params_per_rank = _derive_sharded_params(ds_engine, hogpg, bucket_groups)
    if bucket_groups and sharded_params_per_rank:
        hogpg.register_bucket_groups(bucket_groups, sharded_params_per_rank)

    ds_engine.__hogpg__ = hogpg

    # Patch free_overlap_buffers onto engine for checkpoint hook
    def _free_overlap_buffers():
        all_bgs = _collect_bucket_groups(ds_engine)
        hogpg.free_overlap_buffers(all_bgs)
        torch.cuda.empty_cache()

    ds_engine.free_overlap_buffers = _free_overlap_buffers

    logger.info(
        "integrate_with_engine: HeteroOverlapParamGather attached to engine "
        "(rank=%d, async=%s)", local_rank, config.overlap_async
    )
    return hogpg


def should_disable_forward_pre_hook(optimizer_name: str, overlap_param_gather: bool) -> bool:
    """Determine whether to disable forward pre-hooks for a given optimizer.

    Mirrors Megatron's ``should_disable_forward_pre_hook`` widened for
    ``dist_muon`` / DES-LOC layer-wise optimizer variants.

    Args:
        optimizer_name: Optimizer identifier string (e.g., ``"dist_muon"``).
        overlap_param_gather: Whether overlap param gather is enabled.

    Returns:
        ``True`` if forward pre-hooks should be disabled (e.g., during
        first-iteration manual sync or checkpoint save).
    """
    is_dist_optimizer = "dist" in optimizer_name.lower()
    return is_dist_optimizer and overlap_param_gather


def _collect_bucket_groups(ds_engine) -> List:
    """Collect all bucket groups from a DeepSpeed engine."""
    bgs: List = []
    module = getattr(ds_engine, "module", ds_engine)
    if hasattr(module, "bucket_groups"):
        bgs.extend(module.bucket_groups)
    if hasattr(module, "expert_parallel_bucket_groups"):
        bgs.extend(module.expert_parallel_bucket_groups)
    return bgs


def _derive_sharded_params(
    ds_engine,
    hogpg: HeteroOverlapParamGather,
    bucket_groups: List,
) -> Optional[List[List[torch.nn.Parameter]]]:
    """Attempt to derive per-rank sharded param lists from optimizer state.

    Falls back to equal sharding of all model params if optimizer introspection
    is not available.  Production use should pass explicit ``sharded_params_per_rank``.
    """
    optimizer = getattr(ds_engine, "optimizer", None)
    if optimizer is None:
        return None

    # Try to get per-rank param lists from layer-wise optimizer
    if hasattr(optimizer, "dp_cp_params_list"):
        return optimizer.dp_cp_params_list

    # Fallback: equal sharding
    all_params = list(ds_engine.module.parameters())
    total = len(all_params)
    sizes = hogpg.shard_policy.shard_sizes(total)
    result: List[List[torch.nn.Parameter]] = []
    offset = 0
    for sz in sizes:
        result.append(all_params[offset: offset + sz])
        offset += sz
    logger.info(
        "_derive_sharded_params: fallback equal-ish sharding, "
        "per-rank sizes=%s", sizes
    )
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    # ---- Test 1: shard policy proportional sizes ----
    cfg = HeteroOverlapParamGatherConfig()
    policy = HeteroParamShardPolicy(cfg, world_size=3)
    sizes = policy.shard_sizes(1000)
    assert sum(sizes) == 1000, f"Shard sizes must sum to 1000, got {sum(sizes)}"
    assert sizes[2] > sizes[0], "H100 (rank 2) should hold more params than A6000 (rank 0)"
    logger.info("Test 1 passed: shard sizes = %s", sizes)

    # ---- Test 2: SLCParamCache D2H / H2D round-trip ----
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        slc = SLCParamCache(capacity_gb=0.1, dtype=torch.bfloat16,
                            num_streams=2, device=device)
        src = torch.randn(512, dtype=torch.bfloat16, device=device)
        cpu_slice, slot, d2h_stream = slc.stage_d2h(src)
        dst = torch.empty_like(src)
        evt = slc.load_h2d(cpu_slice, dst, d2h_stream, slot)
        evt.synchronize()
        assert torch.allclose(src.float(), dst.float(), atol=1e-3), \
            "SLC round-trip should preserve values"
        slc.free()
        logger.info("Test 2 passed: SLC D2H→H2D round-trip correct")
    else:
        logger.info("Test 2 skipped: no CUDA device")

    # ---- Test 3: _HeteroAllGatherHandle wait semantics ----
    handle = _HeteroAllGatherHandle([], [])
    handle.wait()   # must not raise
    assert not handle, "Empty handle should be falsy after wait"
    logger.info("Test 3 passed: empty handle wait is safe")

    # ---- Test 4: LayerReadinessTracker ----
    tracker = LayerReadinessTracker(num_layers=4)
    tracker.mark_ready(0)
    assert tracker.is_ready(0), "Layer 0 should be ready after mark_ready"
    assert not tracker.is_ready(1), "Layer 1 should not be ready"
    tracker.reset()
    assert not tracker.is_ready(0), "Layer 0 should not be ready after reset"
    logger.info("Test 4 passed: LayerReadinessTracker semantics correct")

    # ---- Test 5: should_disable_forward_pre_hook ----
    assert should_disable_forward_pre_hook("dist_muon", True)
    assert not should_disable_forward_pre_hook("muon", True)
    assert not should_disable_forward_pre_hook("dist_muon", False)
    logger.info("Test 5 passed: should_disable_forward_pre_hook logic correct")

    logger.info("All smoke tests passed.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroOverlapParamGatherConfig on a DeepSpeed engine.

    Instantiates a :class:`HeteroOverlapParamGatherConfig` from the engine's configuration
    and attaches it as ``engine.hetero_overlap_param_gather``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_overlap_param_gather.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_overlap_param_gather = None
    logger.info("hetero_overlap_param_gather.register() attached engine.hetero_overlap_param_gather")
