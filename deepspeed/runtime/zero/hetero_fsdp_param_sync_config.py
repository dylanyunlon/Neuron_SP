"""
hetero_fsdp_param_sync_config.py
=================================

DES-LOC Heterogeneous FSDP Parameter Synchronization Configuration
-------------------------------------------------------------------

Upstream Design Intent (Megatron fdde15e):
    Megatron-FSDP introduced ``fsdp_all_gather_in_start_param_sync`` to make the
    initial parameter all-gather at the start of each training step *optional*.
    When enabled, the first shard bucket is eagerly all-gathered *before* the
    forward pass begins, creating a pipeline bubble that computation can hide.
    When disabled, ``synchronize_param_gather()`` blocks until all shards are
    resident.  The flag was motivated by profiling on homogeneous NVLink clusters
    where the first all-gather latency (≈ constant) was predictably overlappable
    with the first layer's compute stream.

DES-LOC Adaptation (M3321-BF):
    In the Neuron_SP DES-LOC framework the hardware is deliberately *heterogeneous*:

        • 2× RTX A6000  48 GB  SM86  (PCIe Gen4 ×16, no NVLink)
        • 1× H100 NVL   96 GB  SM90  (PCIe Gen4 ×16, no NVLink)

    Because inter-device bandwidth is PCIe-only (≈ 32 GB/s bidirectional
    vs NVLink 600 GB/s), the flat "gather everything at once" assumption breaks:
    an A6000→H100 all-gather takes ~4× longer than an A6000→A6000 gather.
    Simply porting Megatron's flag would silently *hurt* throughput on the
    A6000 pair and still not saturate H100 compute.

    DES-LOC introduces a **Shared LOcality Cache (SLC)**: a pinned CPU DRAM
    region (from the 1.5 TB pool) that acts as a staging area.  Each device
    class gets its own SLC partition so that:

        1. A6000×2 shards are gathered via SLC when H100 is the parameter
           "owner" of a bucket, avoiding a slow PCIe P2P copy.
        2. The H100 can prefetch the *next* bucket into its L2/HBM while the
           A6000s are still computing on the current bucket.
        3. The ``all_gather_in_start_param_sync`` semantics are preserved but
           split into three device-class-aware modes:
               EAGER   – mirror Megatron default: gather first bucket before fwd
               LAZY    – gather on first use (good for memory-pressure on A6000)
               SLC     – stage via CPU DRAM, overlap PCIe DMA with compute

    This file owns:
        • ``DeviceClass``              – enum of physical device roles
        • ``HeteroDevicePlacement``    – maps rank → DeviceClass + bandwidth
        • ``HeteroFSDPParamSyncConfig``– drop-in replacement for Megatron's
                                         DistributedDataParallelConfig subset
        • ``HeteroParamSyncScheduler`` – decides *when* and *via which path*
                                         to fire each bucket's all-gather
        • ``SLCBuffer``                – thin wrapper around pinned CPU tensor
                                         with async copy helpers

References:
    Megatron commit fdde15e96cb9a778afe3758e7e350bf4bd620b6b
    DeepSpeed ZeRO stage-3 param fetch API (deepspeed/runtime/zero/stage3.py)
    Neuron_SP design doc: docs/des_loc_architecture.md
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants derived from lab hardware inventory
# ---------------------------------------------------------------------------

# PCIe Gen4 ×16 measured peak (bidirectional half-duplex observed in practice)
_PCIE_BW_GBPS: float = 28.0          # GB/s  – A6000 ↔ H100 measured
_A6000_TO_A6000_BW_GBPS: float = 24.0  # slightly less: share the same root complex
_SLC_PREFETCH_THRESHOLD_MB: float = 256.0  # buckets larger than this go via SLC


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DeviceClass(enum.Enum):
    """Physical device categories present in the DES-LOC cluster node.

    SM86 corresponds to Ampere (RTX A6000, A100-PCIe, etc.).
    SM90 corresponds to Hopper (H100 NVL, H100 SXM, etc.).
    CPU  is used for SLC staging buffers; it never *owns* a shard.
    """
    SM86 = "sm86"   # Ampere – RTX A6000 48 GB
    SM90 = "sm90"   # Hopper – H100 NVL 96 GB
    CPU  = "cpu"    # Pinned DRAM staging (SLC)


class ParamSyncMode(enum.Enum):
    """All-gather scheduling strategy for a shard bucket.

    EAGER:
        Mirrors Megatron's ``fsdp_all_gather_in_start_param_sync=True``.
        Issue all-gather for the first bucket *before* the forward pass.
        Appropriate when the source and destination share the same DeviceClass
        (A6000→A6000) so PCIe contention is minimal.

    LAZY:
        Mirrors Megatron's ``fsdp_all_gather_in_start_param_sync=False``.
        Block-gather just before the bucket's first consumer layer.
        Preferred on A6000 when H100 is the shard owner: avoids hogging PCIe
        bandwidth during the preceding layer's backward pass.

    SLC:
        DES-LOC-specific.  Stage the shard through a pinned CPU DRAM buffer
        (Shared LOcality Cache) to decouple the H100→CPU DMA from the
        CPU→A6000 DMA.  Allows double-buffering across PCIe lanes without
        P2P atomics.
    """
    EAGER = "eager"
    LAZY  = "lazy"
    SLC   = "slc"


# ---------------------------------------------------------------------------
# Hardware placement descriptor
# ---------------------------------------------------------------------------

@dataclass
class HeteroDevicePlacement:
    """Maps a distributed rank to its physical device and measured bandwidth.

    Attributes
    ----------
    rank:
        Global rank index.
    device_class:
        Which DeviceClass this rank's GPU belongs to.
    local_rank:
        Local rank within the node (0-based).
    device_bw_gbps:
        Measured host↔device PCIe bandwidth in GB/s.  Used by the scheduler
        to choose SLC vs direct P2P transfer.
    slc_partition_bytes:
        How many bytes of the 1.5 TB SLC pool are reserved for this rank.
        Defaults to 16 GiB (comfortable headroom for 48 GB A6000 shards).
    """
    rank: int
    device_class: DeviceClass
    local_rank: int
    device_bw_gbps: float = _PCIE_BW_GBPS
    slc_partition_bytes: int = 16 * 1024 ** 3  # 16 GiB default

    @property
    def is_high_memory(self) -> bool:
        """Return True for the H100 NVL (96 GB) which can act as shard owner."""
        return self.device_class == DeviceClass.SM90

    def estimated_transfer_latency_ms(self, num_bytes: int) -> float:
        """Rough PCIe transfer time in milliseconds for *num_bytes* bytes."""
        bw_bytes_per_ms = self.device_bw_gbps * 1e9 / 1e3
        return num_bytes / bw_bytes_per_ms


def build_default_placement_map(world_size: int = 3) -> Dict[int, HeteroDevicePlacement]:
    """Construct the default 2×A6000 + 1×H100 placement used in Neuron_SP.

    Layout convention:
        rank 0 → A6000 #0  (local_rank=0)
        rank 1 → A6000 #1  (local_rank=1)
        rank 2 → H100  NVL (local_rank=2)

    Parameters
    ----------
    world_size:
        Must be 3 for the reference hardware.  Provided as a parameter for
        unit-test flexibility.
    """
    if world_size != 3:
        logger.warning(
            "build_default_placement_map: world_size=%d, expected 3 for "
            "the A6000×2+H100 DES-LOC node.  Falling back to all-SM86.",
            world_size,
        )
        return {
            r: HeteroDevicePlacement(r, DeviceClass.SM86, r)
            for r in range(world_size)
        }

    return {
        0: HeteroDevicePlacement(
            rank=0,
            device_class=DeviceClass.SM86,
            local_rank=0,
            device_bw_gbps=_A6000_TO_A6000_BW_GBPS,
            slc_partition_bytes=16 * 1024 ** 3,
        ),
        1: HeteroDevicePlacement(
            rank=1,
            device_class=DeviceClass.SM86,
            local_rank=1,
            device_bw_gbps=_A6000_TO_A6000_BW_GBPS,
            slc_partition_bytes=16 * 1024 ** 3,
        ),
        2: HeteroDevicePlacement(
            rank=2,
            device_class=DeviceClass.SM90,
            local_rank=2,
            device_bw_gbps=_PCIE_BW_GBPS,
            slc_partition_bytes=48 * 1024 ** 3,  # H100 can act as owner
        ),
    }


# ---------------------------------------------------------------------------
# Shared LOcality Cache buffer
# ---------------------------------------------------------------------------

class SLCBuffer:
    """Pinned CPU DRAM staging buffer for DES-LOC cross-device all-gathers.

    The SLC buffer acts as a relay for H100→A6000 parameter copies that would
    otherwise require slow PCIe P2P atomics.  The protocol is:

        1. H100 DMA-copies its shard to this buffer (non-blocking, CUDA stream).
        2. A6000 DMA-copies from this buffer to its local GPU memory.
        3. Steps 1 and 2 are serialised only by a CPU-side event; PCIe lanes
           to H100 and A6000 are *different* root-complex ports, so the two
           DMAs do not contend.

    Parameters
    ----------
    capacity_bytes:
        Total pinned allocation.  Should equal the largest shard bucket
        expected, with a 2× headroom for double-buffering.
    dtype:
        Element dtype; defaults to bfloat16 to match typical LLM weights.
    """

    def __init__(
        self,
        capacity_bytes: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._capacity = capacity_bytes
        self._dtype = dtype
        elem_size = torch.finfo(dtype).bits // 8
        n_elems = capacity_bytes // elem_size
        logger.info(
            "SLCBuffer: allocating %.2f GiB pinned CPU DRAM (dtype=%s, n_elems=%d)",
            capacity_bytes / 1024 ** 3,
            dtype,
            n_elems,
        )
        self._buf: torch.Tensor = torch.empty(
            n_elems, dtype=dtype, device="cpu", pin_memory=True
        )
        self._lock = threading.Lock()
        self._in_use: bool = False

    @property
    def capacity(self) -> int:
        return self._capacity

    def acquire_view(self, num_elements: int) -> torch.Tensor:
        """Return a view into the pinned buffer for *num_elements* elements.

        Thread-safe; blocks until the buffer is free.  In production the
        scheduler ensures only one in-flight transfer at a time per SLC
        partition, so contention should be rare.
        """
        elem_size = torch.finfo(self._dtype).bits // 8
        needed = num_elements * elem_size
        if needed > self._capacity:
            raise RuntimeError(
                f"SLCBuffer: requested {needed} bytes > capacity {self._capacity} bytes. "
                "Increase slc_partition_bytes in HeteroDevicePlacement."
            )
        with self._lock:
            self._in_use = True
            view = self._buf[:num_elements]
        return view

    def release(self) -> None:
        """Signal that the staging transfer is complete."""
        with self._lock:
            self._in_use = False

    def async_copy_to_device(
        self,
        src_tensor: torch.Tensor,
        dst_device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.cuda.Event]:
        """Stage *src_tensor* through the SLC buffer onto *dst_device*.

        Returns
        -------
        dst_tensor:
            A new tensor on *dst_device* backed by the async copy.
        done_event:
            A CUDA event recorded on *stream* after the copy completes.
            Callers should call ``done_event.synchronize()`` before reading
            *dst_tensor*.
        """
        stream = stream or torch.cuda.current_stream(dst_device)
        n_elems = src_tensor.numel()
        stage = self.acquire_view(n_elems).reshape_as(src_tensor)

        # Phase-1: GPU(src) → CPU pinned
        stage.copy_(src_tensor, non_blocking=True)

        # Phase-2: CPU pinned → GPU(dst)
        dst_tensor = torch.empty_like(src_tensor, device=dst_device)
        with torch.cuda.stream(stream):
            dst_tensor.copy_(stage, non_blocking=True)

        done_event = torch.cuda.Event()
        done_event.record(stream)
        logger.debug(
            "SLCBuffer.async_copy_to_device: %d elements, src=%s → dst=%s",
            n_elems,
            src_tensor.device,
            dst_device,
        )
        return dst_tensor, done_event


# ---------------------------------------------------------------------------
# Main configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeteroFSDPParamSyncConfig:
    """Configuration for DES-LOC heterogeneous FSDP parameter synchronization.

    This dataclass is the DES-LOC analogue of Megatron's
    ``DistributedDataParallelConfig.fsdp_all_gather_in_start_param_sync``.
    Instead of a single boolean, it exposes per-device-class scheduling knobs
    that the ``HeteroParamSyncScheduler`` interprets at runtime.

    Upstream semantics preserved
    ----------------------------
    Setting ``sm86_sync_mode = ParamSyncMode.EAGER`` and
    ``sm90_sync_mode = ParamSyncMode.EAGER`` replicates the Megatron default
    (``fsdp_all_gather_in_start_param_sync=True``).  Setting both to
    ``ParamSyncMode.LAZY`` replicates the disabled path.

    DES-LOC extensions
    ------------------
    ``sm86_sync_mode = ParamSyncMode.SLC`` activates CPU-DRAM staging for
    A6000 ranks gathering shards owned by the H100.  The scheduler will
    automatically fall back to EAGER if the bucket is small enough that
    direct PCIe copy is faster than the SLC round-trip overhead (≈ 40 µs
    pinned-alloc + 2× DMA setup).

    Attributes
    ----------
    sm86_sync_mode:
        Sync strategy for Ampere (A6000) ranks.
    sm90_sync_mode:
        Sync strategy for Hopper (H100) rank.  Typically EAGER because H100
        is the shard owner and issues the all-gather to peers.
    overlap_param_gather:
        Mirror of Megatron's flag; enables the prefetch pipeline.
    slc_fallback_threshold_bytes:
        Bucket byte threshold below which SLC mode falls back to EAGER.
        Avoids SLC overhead for tiny embedding shards.
    max_prefetch_buckets:
        How many buckets ahead the H100 may prefetch into SLC.
    placement_map:
        rank → HeteroDevicePlacement.  Built automatically if None.
    enable_bandwidth_profiling:
        If True, record per-transfer timing and log bandwidth stats every
        ``bw_profile_log_interval`` steps.
    bw_profile_log_interval:
        Steps between bandwidth profile log lines.
    """

    sm86_sync_mode: ParamSyncMode = ParamSyncMode.SLC
    sm90_sync_mode: ParamSyncMode = ParamSyncMode.EAGER
    overlap_param_gather: bool = True
    slc_fallback_threshold_bytes: int = int(
        _SLC_PREFETCH_THRESHOLD_MB * 1024 ** 2
    )
    max_prefetch_buckets: int = 2
    placement_map: Optional[Dict[int, HeteroDevicePlacement]] = field(
        default=None, repr=False
    )
    enable_bandwidth_profiling: bool = False
    bw_profile_log_interval: int = 100

    def __post_init__(self) -> None:
        if self.placement_map is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 3
            self.placement_map = build_default_placement_map(world_size)
            logger.info(
                "HeteroFSDPParamSyncConfig: auto-built placement map for "
                "world_size=%d",
                world_size,
            )

    def sync_mode_for_rank(self, rank: int) -> ParamSyncMode:
        """Return the configured sync mode for *rank*'s device class."""
        placement = self.placement_map.get(rank)
        if placement is None:
            logger.warning(
                "sync_mode_for_rank: rank %d not in placement_map, defaulting LAZY",
                rank,
            )
            return ParamSyncMode.LAZY
        if placement.device_class == DeviceClass.SM90:
            return self.sm90_sync_mode
        return self.sm86_sync_mode

    def effective_mode_for_bucket(
        self, rank: int, bucket_bytes: int
    ) -> ParamSyncMode:
        """Resolve the *effective* mode after applying the SLC fallback rule.

        If the configured mode is SLC but the bucket is below the fallback
        threshold, EAGER is returned instead (direct PCIe copy is cheaper).

        Parameters
        ----------
        rank:
            The rank that will receive this bucket.
        bucket_bytes:
            Size of the all-gather bucket in bytes.
        """
        mode = self.sync_mode_for_rank(rank)
        if mode == ParamSyncMode.SLC and bucket_bytes < self.slc_fallback_threshold_bytes:
            logger.debug(
                "effective_mode_for_bucket: rank=%d bucket=%d bytes < threshold=%d, "
                "falling back SLC→EAGER",
                rank,
                bucket_bytes,
                self.slc_fallback_threshold_bytes,
            )
            return ParamSyncMode.EAGER
        return mode


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class _BucketTransferRecord:
    """Internal bookkeeping for an in-flight bucket transfer."""
    bucket_id: int
    mode: ParamSyncMode
    issued_at: float           # time.monotonic()
    done_event: Optional[torch.cuda.Event] = None
    slc_buffer: Optional[SLCBuffer] = None


class HeteroParamSyncScheduler:
    """Decides *when* and *via which path* each FSDP bucket's all-gather fires.

    This class is the execution engine behind ``HeteroFSDPParamSyncConfig``.
    It is called by the DES-LOC engine at two hook points:

        ``start_param_sync(force_sync)``
            Called at the very beginning of ``forward()``.  Mirrors the
            Megatron hook that was gated by ``fsdp_all_gather_in_start_param_sync``.

        ``maybe_prefetch_next(bucket_id)``
            Called after each bucket's forward compute completes to trigger
            look-ahead prefetch of ``bucket_id + 1 … + max_prefetch_buckets``.

    Parameters
    ----------
    config:
        The ``HeteroFSDPParamSyncConfig`` governing this scheduler instance.
    num_buckets:
        Total number of all-gather buckets for this FSDP module.
    slc_buffers:
        Per-rank SLC staging buffers.  Keyed by rank index.
    local_rank:
        The rank this process owns.
    """

    def __init__(
        self,
        config: HeteroFSDPParamSyncConfig,
        num_buckets: int,
        slc_buffers: Dict[int, SLCBuffer],
        local_rank: int,
    ) -> None:
        self._cfg = config
        self._num_buckets = num_buckets
        self._slc_buffers = slc_buffers
        self._local_rank = local_rank
        self._in_flight: Dict[int, _BucketTransferRecord] = {}
        self._step_count: int = 0
        self._bw_stats: List[Tuple[int, float, int]] = []  # (step, bw_gbps, bytes)

        placement = config.placement_map.get(local_rank)
        self._device_class = (
            placement.device_class if placement else DeviceClass.SM86
        )
        logger.info(
            "HeteroParamSyncScheduler: rank=%d device_class=%s num_buckets=%d",
            local_rank,
            self._device_class,
            num_buckets,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_param_sync(
        self,
        params: List[torch.nn.Parameter],
        force_sync: bool = False,
    ) -> None:
        """Entry-point called at the start of each forward pass.

        Replicates and extends the Megatron logic guarded by
        ``fsdp_all_gather_in_start_param_sync``:

            if not force_sync and overlap_param_gather:
                if fsdp_all_gather_in_start_param_sync:
                    gather first bucket (non-blocking prefetch)
            else:
                synchronize_param_gather()
                gather all buckets (blocking)

        DES-LOC extension: the first bucket may be dispatched via SLC if
        ``effective_mode_for_bucket`` returns ``ParamSyncMode.SLC``.

        Parameters
        ----------
        params:
            Ordered list of parameters (first element = first bucket's anchor).
        force_sync:
            If True, block until all buckets are gathered (same as Megatron's
            else-branch).
        """
        self._step_count += 1

        if force_sync or not self._cfg.overlap_param_gather:
            logger.debug(
                "start_param_sync: rank=%d step=%d FORCE_SYNC path",
                self._local_rank,
                self._step_count,
            )
            self._synchronize_all_buckets(params)
            return

        # Non-blocking path: kick off first bucket, let compute overlap it.
        if not params:
            logger.warning("start_param_sync: params list is empty, nothing to prefetch")
            return

        first_param = params[0]
        bucket_bytes = first_param.data.nbytes
        mode = self._cfg.effective_mode_for_bucket(self._local_rank, bucket_bytes)

        logger.debug(
            "start_param_sync: rank=%d step=%d bucket_bytes=%d mode=%s",
            self._local_rank,
            self._step_count,
            bucket_bytes,
            mode.value,
        )

        self._dispatch_bucket(bucket_id=0, param=first_param, mode=mode, prefetch=True)

    def maybe_prefetch_next(
        self,
        current_bucket_id: int,
        params_by_bucket: Dict[int, List[torch.nn.Parameter]],
    ) -> None:
        """Prefetch up to ``max_prefetch_buckets`` ahead of *current_bucket_id*.

        Called after bucket *current_bucket_id*'s forward compute slice
        completes.  Issues non-blocking all-gathers for upcoming buckets so
        their data arrives before they are needed.

        Parameters
        ----------
        current_bucket_id:
            The bucket whose compute just finished.
        params_by_bucket:
            Mapping bucket_id → list of parameters in that bucket.
        """
        for offset in range(1, self._cfg.max_prefetch_buckets + 1):
            bid = current_bucket_id + offset
            if bid >= self._num_buckets:
                break
            if bid in self._in_flight:
                continue  # already issued

            bucket_params = params_by_bucket.get(bid, [])
            if not bucket_params:
                continue

            first_param = bucket_params[0]
            bucket_bytes = sum(p.data.nbytes for p in bucket_params)
            mode = self._cfg.effective_mode_for_bucket(self._local_rank, bucket_bytes)

            logger.debug(
                "maybe_prefetch_next: rank=%d prefetching bucket=%d mode=%s",
                self._local_rank,
                bid,
                mode.value,
            )
            self._dispatch_bucket(bucket_id=bid, param=first_param, mode=mode, prefetch=True)

    def wait_bucket_ready(self, bucket_id: int) -> None:
        """Block until bucket *bucket_id*'s all-gather is complete.

        If the bucket was dispatched via SLC, this also synchronises the
        CPU→GPU DMA event and releases the SLC staging buffer.
        """
        record = self._in_flight.pop(bucket_id, None)
        if record is None:
            logger.debug(
                "wait_bucket_ready: bucket=%d not in-flight (already synced or never issued)",
                bucket_id,
            )
            return

        if record.done_event is not None:
            t0 = time.monotonic()
            record.done_event.synchronize()
            elapsed = time.monotonic() - t0
            if elapsed > 0.005:  # 5 ms – log unexpectedly long waits
                logger.warning(
                    "wait_bucket_ready: rank=%d bucket=%d waited %.3f ms for done_event",
                    self._local_rank,
                    bucket_id,
                    elapsed * 1e3,
                )

        if record.slc_buffer is not None:
            record.slc_buffer.release()
            logger.debug(
                "wait_bucket_ready: released SLC buffer for bucket=%d", bucket_id
            )

        if self._cfg.enable_bandwidth_profiling and record.done_event is not None:
            self._record_bw_stat(record)

    def log_bandwidth_profile(self) -> None:
        """Emit a bandwidth summary log line (called every bw_profile_log_interval steps)."""
        if not self._bw_stats:
            return
        avg_bw = sum(s[1] for s in self._bw_stats) / len(self._bw_stats)
        total_bytes = sum(s[2] for s in self._bw_stats)
        logger.info(
            "BW profile: rank=%d step=%d avg_transfer_bw=%.2f GB/s total_bytes=%.2f MB n_transfers=%d",
            self._local_rank,
            self._step_count,
            avg_bw,
            total_bytes / 1024 ** 2,
            len(self._bw_stats),
        )
        self._bw_stats.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_bucket(
        self,
        bucket_id: int,
        param: torch.nn.Parameter,
        mode: ParamSyncMode,
        prefetch: bool,
    ) -> None:
        """Issue the all-gather for *bucket_id* according to *mode*.

        Parameters
        ----------
        bucket_id:
            Index of the shard bucket being gathered.
        param:
            Representative parameter from the bucket (used for size / device).
        mode:
            How to transfer the bucket.
        prefetch:
            If True, do not wait for completion (non-blocking).
        """
        record = _BucketTransferRecord(
            bucket_id=bucket_id,
            mode=mode,
            issued_at=time.monotonic(),
        )

        if mode == ParamSyncMode.EAGER:
            self._dispatch_eager(record, param, prefetch)
        elif mode == ParamSyncMode.LAZY:
            self._dispatch_lazy(record, param)
        elif mode == ParamSyncMode.SLC:
            self._dispatch_slc(record, param)
        else:
            raise ValueError(f"Unknown ParamSyncMode: {mode}")

        self._in_flight[bucket_id] = record

    def _dispatch_eager(
        self,
        record: _BucketTransferRecord,
        param: torch.nn.Parameter,
        prefetch: bool,
    ) -> None:
        """Issue a direct all-gather (mirrors Megatron's default path).

        In a real DeepSpeed integration this would call
        ``deepspeed.runtime.zero.stage3.PartitionedParameterCoordinator``
        APIs.  Here we record a CUDA event that the test harness can verify.
        """
        if param.device.type == "cuda":
            stream = torch.cuda.current_stream(param.device)
            evt = torch.cuda.Event()
            # Production: call coordinator.fetch_sub_module(module, forward=True)
            # Placeholder: record current-stream event as "gather done" marker.
            evt.record(stream)
            record.done_event = evt
        logger.debug(
            "_dispatch_eager: bucket=%d device=%s prefetch=%s",
            record.bucket_id,
            param.device,
            prefetch,
        )

    def _dispatch_lazy(
        self,
        record: _BucketTransferRecord,
        param: torch.nn.Parameter,
    ) -> None:
        """Register a lazy callback; actual gather fires at first access.

        This path stores the record without a done_event; ``wait_bucket_ready``
        will detect the absence and return immediately (the gather will have
        been forced inline by the parameter access hook).
        """
        logger.debug(
            "_dispatch_lazy: bucket=%d queued for on-demand gather", record.bucket_id
        )

    def _dispatch_slc(
        self,
        record: _BucketTransferRecord,
        param: torch.nn.Parameter,
    ) -> None:
        """Stage the all-gather through the CPU DRAM SLC buffer.

        Protocol (see SLCBuffer docstring):
            1. Identify the shard-owner rank (H100 for large shards by default).
            2. Issue async GPU→CPU copy on owner's stream (H100 → pinned SLC).
            3. Issue async CPU→GPU copy on local stream (SLC → A6000).
            4. Record a CUDA event after step 3.

        In this reference implementation the source tensor is the param's
        existing .data (which DeepSpeed stage-3 keeps as a view into the
        gathered buffer).  Production code would fetch the remote shard handle.
        """
        slc = self._slc_buffers.get(self._local_rank)
        if slc is None:
            logger.warning(
                "_dispatch_slc: no SLC buffer for rank=%d, falling back to EAGER",
                self._local_rank,
            )
            self._dispatch_eager(record, param, prefetch=True)
            return

        if param.device.type != "cuda":
            logger.debug(
                "_dispatch_slc: param on CPU, skipping SLC staging for bucket=%d",
                record.bucket_id,
            )
            return

        dst_device = param.device
        try:
            _, done_event = slc.async_copy_to_device(
                src_tensor=param.data,
                dst_device=dst_device,
            )
            record.done_event = done_event
            record.slc_buffer = slc
        except RuntimeError as exc:
            logger.error(
                "_dispatch_slc: SLC copy failed for bucket=%d: %s. "
                "Falling back to EAGER.",
                record.bucket_id,
                exc,
            )
            self._dispatch_eager(record, param, prefetch=True)

        logger.debug(
            "_dispatch_slc: bucket=%d staged via SLC (%.2f MB)",
            record.bucket_id,
            param.data.nbytes / 1024 ** 2,
        )

    def _synchronize_all_buckets(self, params: List[torch.nn.Parameter]) -> None:
        """Blocking gather of all buckets (force_sync path)."""
        for bucket_id, param in enumerate(params):
            bucket_bytes = param.data.nbytes
            mode = self._cfg.effective_mode_for_bucket(self._local_rank, bucket_bytes)
            self._dispatch_bucket(bucket_id=bucket_id, param=param, mode=mode, prefetch=False)
            self.wait_bucket_ready(bucket_id)
        logger.debug(
            "_synchronize_all_buckets: rank=%d synced %d buckets",
            self._local_rank,
            len(params),
        )

    def _record_bw_stat(self, record: _BucketTransferRecord) -> None:
        """Approximate bandwidth from transfer duration (best-effort)."""
        elapsed = time.monotonic() - record.issued_at
        if elapsed <= 0:
            return
        # We don't have exact bytes here without the param ref; use 0 as sentinel.
        bw_gbps = 0.0  # production: param.data.nbytes / elapsed / 1e9
        self._bw_stats.append((self._step_count, bw_gbps, 0))


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_hetero_param_sync(
    config: Optional[HeteroFSDPParamSyncConfig] = None,
    num_buckets: int = 8,
    local_rank: Optional[int] = None,
) -> HeteroParamSyncScheduler:
    """Convenience factory for creating a fully wired scheduler.

    Parameters
    ----------
    config:
        If None, a default ``HeteroFSDPParamSyncConfig`` is constructed.
    num_buckets:
        Number of FSDP all-gather buckets.
    local_rank:
        If None, inferred from ``torch.distributed.get_rank()`` or 0.
    """
    if config is None:
        config = HeteroFSDPParamSyncConfig()

    if local_rank is None:
        local_rank = dist.get_rank() if dist.is_initialized() else 0

    placement = config.placement_map.get(local_rank)
    slc_bytes = placement.slc_partition_bytes if placement else 16 * 1024 ** 3

    slc_buffers: Dict[int, SLCBuffer] = {
        local_rank: SLCBuffer(
            capacity_bytes=slc_bytes,
            dtype=torch.bfloat16,
        )
    }

    return HeteroParamSyncScheduler(
        config=config,
        num_buckets=num_buckets,
        slc_buffers=slc_buffers,
        local_rank=local_rank,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    # --- Test 1: default placement map has correct device classes ---
    pmap = build_default_placement_map(3)
    assert pmap[0].device_class == DeviceClass.SM86
    assert pmap[1].device_class == DeviceClass.SM86
    assert pmap[2].device_class == DeviceClass.SM90
    logger.info("Test 1 PASSED: placement map device classes correct")

    # --- Test 2: SLC fallback for small buckets ---
    cfg = HeteroFSDPParamSyncConfig()
    small_bytes = 1024  # well below 256 MB threshold
    assert cfg.effective_mode_for_bucket(0, small_bytes) == ParamSyncMode.EAGER, \
        "Small bucket on SM86 should fall back SLC→EAGER"
    large_bytes = 512 * 1024 ** 2  # 512 MB
    assert cfg.effective_mode_for_bucket(0, large_bytes) == ParamSyncMode.SLC, \
        "Large bucket on SM86 should use SLC"
    logger.info("Test 2 PASSED: SLC fallback threshold logic correct")

    # --- Test 3: H100 rank always uses EAGER regardless of size ---
    assert cfg.effective_mode_for_bucket(2, large_bytes) == ParamSyncMode.EAGER, \
        "H100 rank should always be EAGER"
    logger.info("Test 3 PASSED: H100 rank mode correct")

    # --- Test 4: scheduler dispatches first bucket in start_param_sync ---
    scheduler = make_hetero_param_sync(config=cfg, num_buckets=4, local_rank=0)
    dummy_param = torch.nn.Parameter(torch.randn(128, 128))  # CPU param (no CUDA needed)
    scheduler.start_param_sync(params=[dummy_param], force_sync=False)
    assert 0 in scheduler._in_flight, "Bucket 0 should be in-flight after start_param_sync"
    logger.info("Test 4 PASSED: bucket 0 in-flight after start_param_sync")

    # --- Test 5: force_sync path clears in-flight after sync ---
    scheduler2 = make_hetero_param_sync(config=cfg, num_buckets=2, local_rank=0)
    p1 = torch.nn.Parameter(torch.randn(64))
    p2 = torch.nn.Parameter(torch.randn(64))
    scheduler2.start_param_sync(params=[p1, p2], force_sync=True)
    assert len(scheduler2._in_flight) == 0, "force_sync should drain in_flight"
    logger.info("Test 5 PASSED: force_sync drains in-flight map")

    logger.info("All smoke tests PASSED")
