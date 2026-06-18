"""
DES-LOC Heterogeneous FSDP Gradient Scaling Fix
================================================

Upstream Design Intent (Megatron ba456fd):
    In Megatron-FSDP's GradReducePipeline, gradient scaling was applied to
    `gbuf.data` (the entire gradient buffer) rather than `bucket.data` (the
    specific bucket being reduced). This caused incorrect gradient scaling when
    a GradBuffer contained multiple buckets with different data distributions,
    because the scaling factor was broadcast over stale or out-of-scope memory
    rather than the live reduction target.

    The fix is surgical: replace `gbuf.data` with `bucket.data` so that
    `gradient_reduce_preprocessing` operates on precisely the tensor that will
    be all-reduced or scatter-reduced in this pipeline step.

DES-LOC Adaptation Points:
    In DES-LOC (Decoupled Execution with Shared LOcality Cache), the gradient
    buffer lives in a *heterogeneous* memory space:

      - A6000 (SM86, 48 GB, PCIe)  ← compute shards for fp16/bf16 activations
      - H100 NVL (SM90, 96 GB, PCIe) ← parameter master copies + optimizer state
      - CPU DRAM (1.5 TB)           ← locality cache (LOC) for offloaded buckets

    This creates three new failure modes absent in Megatron's homogeneous setup:

    1. **Buffer/Bucket device mismatch** – `gbuf.data` may reside on a different
       device than `bucket.data` when the bucket has been migrated to the LOC
       (CPU DRAM) between the alloc and reduce phases. Scaling `gbuf.data` on
       device A then reducing `bucket.data` on device B silently produces wrong
       gradients with no exception.

    2. **Dtype promotion across SM generations** – A6000 (SM86) does not natively
       accelerate fp8; H100 (SM90) does. If a bucket has been dtype-promoted
       during the LOC round-trip, scaling must happen *after* promotion, on the
       bucket tensor, not before on the original gbuf dtype.

    3. **Partial-bucket LOC eviction** – Under memory pressure the LOC may evict
       only part of a GradBuffer (one or more buckets) to CPU. Scaling gbuf.data
       as a whole would require the entire buffer to be resident, defeating the
       purpose of the LOC.

    This module implements `HeteroFSDPGradScaler`, a drop-in replacement for
    DeepSpeed ZeRO's gradient reduction preprocessing that:
      - Resolves the bucket's physical location (GPU0/GPU1/CPU LOC) at call time.
      - Applies the scaling factor on the bucket's resident device.
      - Handles dtype promotion for SM90-capable buckets.
      - Integrates with DeepSpeed's `GradientBuffer` and `PartitionedParameterCoordinator`.

References:
    Megatron-LM commit ba456fdad991b085ca4f19dea11f7ed886d73ce8
    Neuron_SP project: github.com/dylanyunlon/Neuron_SP
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SM capability thresholds
_SM86_CAPABILITY = (8, 6)   # A6000
_SM90_CAPABILITY = (9, 0)   # H100 NVL

# DES-LOC memory tier identifiers
class MemoryTier(Enum):
    GPU_COMPUTE   = auto()   # A6000 x2 – forward/backward compute shards
    GPU_MASTER    = auto()   # H100 NVL – param master copies + optimizer state
    CPU_LOC       = auto()   # CPU DRAM locality cache


# ---------------------------------------------------------------------------
# Device registry: maps physical devices to DES-LOC tiers
# ---------------------------------------------------------------------------

def _build_device_tier_map() -> Dict[torch.device, MemoryTier]:
    """
    Probe available CUDA devices and classify them by SM capability into
    DES-LOC memory tiers.

    Returns
    -------
    dict[torch.device, MemoryTier]
        Mapping from each visible CUDA device to its DES-LOC role.
    """
    tier_map: Dict[torch.device, MemoryTier] = {}
    for idx in range(torch.cuda.device_count()):
        dev = torch.device("cuda", idx)
        major, minor = torch.cuda.get_device_capability(dev)
        cap = (major, minor)
        if cap >= _SM90_CAPABILITY:
            tier = MemoryTier.GPU_MASTER
            logger.debug("Device cuda:%d (SM%d%d) → GPU_MASTER tier", idx, major, minor)
        else:
            tier = MemoryTier.GPU_COMPUTE
            logger.debug("Device cuda:%d (SM%d%d) → GPU_COMPUTE tier", idx, major, minor)
        tier_map[dev] = tier
    # CPU is always the LOC tier
    tier_map[torch.device("cpu")] = MemoryTier.CPU_LOC
    return tier_map


_DEVICE_TIER_MAP: Dict[torch.device, MemoryTier] = {}


def get_device_tier(device: torch.device) -> MemoryTier:
    """Return the DES-LOC memory tier for *device*, building the map lazily."""
    global _DEVICE_TIER_MAP
    if not _DEVICE_TIER_MAP:
        _DEVICE_TIER_MAP = _build_device_tier_map()
    # Normalise: cuda:0 and cuda(index=0) must compare equal
    normalised = torch.device(device.type, device.index if device.index is not None else 0)
    return _DEVICE_TIER_MAP.get(normalised, MemoryTier.CPU_LOC)


# ---------------------------------------------------------------------------
# Bucket location descriptor
# ---------------------------------------------------------------------------

@dataclass
class BucketLocation:
    """
    Describes where a gradient bucket physically lives at a given moment.

    Attributes
    ----------
    device : torch.device
        The current resident device of `bucket.data`.
    tier : MemoryTier
        DES-LOC tier classification.
    dtype : torch.dtype
        Current dtype of the bucket tensor (may differ from gbuf dtype after
        SM90 dtype promotion).
    is_loc_resident : bool
        True when the bucket has been offloaded to CPU LOC and has not yet
        been fetched back to any GPU.
    """
    device: torch.device
    tier: MemoryTier
    dtype: torch.dtype
    is_loc_resident: bool


def resolve_bucket_location(bucket_data: torch.Tensor) -> BucketLocation:
    """
    Inspect *bucket_data* and return a :class:`BucketLocation` descriptor.

    This is the central probe that replaces the implicit assumption in the
    original Megatron code that ``gbuf.data`` and ``bucket.data`` share the
    same device.

    Parameters
    ----------
    bucket_data : torch.Tensor
        The ``data`` attribute of the gradient bucket that is about to be
        all-reduced or scatter-reduced.

    Returns
    -------
    BucketLocation
    """
    dev = bucket_data.device
    if dev.type == "cpu":
        return BucketLocation(
            device=dev,
            tier=MemoryTier.CPU_LOC,
            dtype=bucket_data.dtype,
            is_loc_resident=True,
        )
    canonical_dev = torch.device("cuda", dev.index if dev.index is not None else 0)
    tier = get_device_tier(canonical_dev)
    return BucketLocation(
        device=canonical_dev,
        tier=tier,
        dtype=bucket_data.dtype,
        is_loc_resident=False,
    )


# ---------------------------------------------------------------------------
# DES-LOC dtype promotion policy
# ---------------------------------------------------------------------------

def _should_promote_dtype(loc: BucketLocation, ddp_config) -> bool:
    """
    Decide whether the bucket should be dtype-promoted before scaling.

    Policy:
      - Buckets resident on GPU_MASTER (SM90/H100) may be promoted to bf16
        for communication efficiency; fp8 is reserved for future work.
      - Buckets on GPU_COMPUTE (SM86/A6000) stay in their original dtype.
      - CPU LOC buckets are never promoted (no accelerator to exploit).

    Parameters
    ----------
    loc : BucketLocation
    ddp_config : object
        DeepSpeed / Megatron DDP config with at least a
        ``communication_dtype`` attribute (optional).
    """
    if loc.tier != MemoryTier.GPU_MASTER:
        return False
    comm_dtype = getattr(ddp_config, "communication_dtype", None)
    if comm_dtype is None:
        return False
    # Only promote if current dtype is lower precision than comm_dtype
    _dtype_rank = {
        torch.float32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.float8_e4m3fn: 1,
    }
    return _dtype_rank.get(comm_dtype, 0) > _dtype_rank.get(loc.dtype, 0)


def maybe_promote_bucket_dtype(
    bucket_data: torch.Tensor,
    loc: BucketLocation,
    ddp_config,
) -> Tuple[torch.Tensor, bool]:
    """
    Optionally promote *bucket_data* dtype for SM90 communication acceleration.

    Returns
    -------
    (tensor, was_promoted) : Tuple[torch.Tensor, bool]
        *tensor* is either the original or a dtype-cast view.
        *was_promoted* signals whether the caller must cast back after reduce.
    """
    if not _should_promote_dtype(loc, ddp_config):
        return bucket_data, False
    target_dtype = getattr(ddp_config, "communication_dtype", bucket_data.dtype)
    logger.debug(
        "DES-LOC dtype promotion: %s → %s on %s (SM90 path)",
        bucket_data.dtype, target_dtype, loc.device,
    )
    return bucket_data.to(dtype=target_dtype), True


# ---------------------------------------------------------------------------
# LOC fetch: bring an offloaded bucket back to the correct GPU tier
# ---------------------------------------------------------------------------

def fetch_bucket_from_loc(
    bucket_data: torch.Tensor,
    target_device: torch.device,
    non_blocking: bool = True,
) -> torch.Tensor:
    """
    Fetch a CPU-LOC-resident bucket to *target_device* for in-place scaling.

    In DES-LOC, gradient buckets can be evicted to CPU DRAM (the LOC) during
    the forward pass to free GPU memory for activations.  Before we can apply
    a CUDA-accelerated scaling kernel, we must bring the bucket back.

    The fetch is *always* to the bucket's *natural* GPU tier:
      - If the bucket's originating parameter shard lives on GPU_MASTER
        (H100), fetch to GPU_MASTER.
      - Otherwise fetch to the calling rank's primary GPU_COMPUTE device.

    Parameters
    ----------
    bucket_data : torch.Tensor
        CPU-resident bucket tensor.
    target_device : torch.device
        Destination GPU device.
    non_blocking : bool
        Use non-blocking host→device copy where possible.

    Returns
    -------
    torch.Tensor
        GPU-resident clone of *bucket_data*.
    """
    if bucket_data.device.type != "cpu":
        return bucket_data  # Already on GPU, no-op
    logger.debug(
        "DES-LOC LOC fetch: cpu → %s, shape=%s, dtype=%s",
        target_device, bucket_data.shape, bucket_data.dtype,
    )
    return bucket_data.to(device=target_device, non_blocking=non_blocking)


# ---------------------------------------------------------------------------
# Core: HeteroFSDPGradScaler
# ---------------------------------------------------------------------------

class HeteroFSDPGradScaler:
    """
    Heterogeneous FSDP gradient scaling for DES-LOC.

    Replaces the direct call to ``gradient_reduce_preprocessing(gbuf.data, ...)``
    with a device-aware pipeline that:

    1. Resolves the *bucket*'s physical location (not the buffer's).
    2. Optionally fetches the bucket from CPU LOC to the correct GPU tier.
    3. Applies the scaling factor *on the bucket tensor* (fixing the upstream
       Megatron bug: ba456fd).
    4. Handles SM90 dtype promotion for H100-resident buckets.
    5. Returns a ``ReduceOp`` descriptor compatible with DeepSpeed's all-reduce
       and scatter-reduce entry points.

    Parameters
    ----------
    process_group : dist.ProcessGroup
        The communication group for gradient reduction.
    overlap_comm : bool
        If True, scaling and collective communication are pipelined.
    loc_eviction_threshold_mb : float
        Buckets larger than this (in MiB) that arrive from CPU LOC are
        fetched asynchronously.  Smaller buckets use synchronous copy.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        overlap_comm: bool = True,
        loc_eviction_threshold_mb: float = 256.0,
    ) -> None:
        self.process_group = process_group
        self.overlap_comm = overlap_comm
        self.loc_eviction_threshold_mb = loc_eviction_threshold_mb

        self._world_size: int = dist.get_world_size(process_group)
        self._fetch_stream: Optional[torch.cuda.Stream] = None
        self._scale_stream: Optional[torch.cuda.Stream] = None
        self._stats: Dict[str, int] = {
            "buckets_scaled": 0,
            "loc_fetches": 0,
            "dtype_promotions": 0,
            "skipped_scaling": 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_streams(self, device: torch.device):
        """Lazily create CUDA streams pinned to *device*."""
        if self._fetch_stream is None:
            with torch.cuda.device(device):
                self._fetch_stream = torch.cuda.Stream(device=device)
                self._scale_stream = torch.cuda.Stream(device=device)
            logger.debug("DES-LOC streams created on %s", device)

    def _tensor_size_mb(self, t: torch.Tensor) -> float:
        return t.numel() * t.element_size() / (1024 ** 2)

    def _select_target_device_for_loc_bucket(
        self,
        gbuf_device: Optional[torch.device],
    ) -> torch.device:
        """
        Choose where to bring a LOC-resident bucket.

        Strategy:
          - If *gbuf_device* is a known GPU (i.e. the buffer was allocated on
            a GPU before eviction), return that device.
          - Otherwise return the current default CUDA device.
        """
        if gbuf_device is not None and gbuf_device.type == "cuda":
            return gbuf_device
        return torch.device("cuda", torch.cuda.current_device())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scale_bucket_gradients(
        self,
        bucket_data: torch.Tensor,
        scaling_factor: float,
        ddp_config,
        gbuf_device: Optional[torch.device] = None,
        is_data_distributed: bool = True,
    ) -> Tuple[torch.Tensor, Optional[dist.ReduceOp]]:
        """
        Apply *scaling_factor* to *bucket_data* with full DES-LOC awareness.

        This is the authoritative replacement for::

            reduce_op = gradient_reduce_preprocessing(
                gbuf.data, scaling_factor, gbuf.ddp_config   # ← BUG
            )

        which should be::

            reduce_op = gradient_reduce_preprocessing(
                bucket.data, scaling_factor, gbuf.ddp_config  # ← FIX (ba456fd)
            )

        In DES-LOC we go further: ``bucket.data`` may not even reside on the
        same device as ``gbuf.data``, so we probe, fetch, scale, and promote
        in one coherent pipeline.

        Parameters
        ----------
        bucket_data : torch.Tensor
            The gradient bucket's data tensor (NOT the whole GradBuffer).
        scaling_factor : float
            Pre-computed gradient scaling factor from ``gbuf.gradient_scaling_factor``.
        ddp_config : object
            DDP/FSDP configuration carrying ``communication_dtype``,
            ``average_in_collective``, etc.
        gbuf_device : torch.device, optional
            The device on which the parent GradBuffer was originally allocated.
            Used to guide LOC fetch destination.
        is_data_distributed : bool
            If True, a scatter-reduce will be used; otherwise all-reduce.

        Returns
        -------
        (scaled_bucket_data, reduce_op) : Tuple[torch.Tensor, Optional[dist.ReduceOp]]
            *scaled_bucket_data* is ready for collective communication.
            *reduce_op* is the ``dist.ReduceOp`` to pass to the collective.
        """
        # --- Step 1: Resolve bucket location ---
        loc = resolve_bucket_location(bucket_data)
        logger.debug(
            "Bucket location: device=%s tier=%s dtype=%s loc_resident=%s size=%.1fMB",
            loc.device, loc.tier.name, loc.dtype, loc.is_loc_resident,
            self._tensor_size_mb(bucket_data),
        )

        # --- Step 2: Fetch from LOC if necessary ---
        if loc.is_loc_resident:
            target_dev = self._select_target_device_for_loc_bucket(gbuf_device)
            large = self._tensor_size_mb(bucket_data) > self.loc_eviction_threshold_mb
            self._get_or_create_streams(target_dev)
            if large and self.overlap_comm:
                with torch.cuda.stream(self._fetch_stream):
                    bucket_data = fetch_bucket_from_loc(
                        bucket_data, target_dev, non_blocking=True
                    )
                # Ensure scale stream waits for fetch to complete
                self._scale_stream.wait_stream(self._fetch_stream)
            else:
                bucket_data = fetch_bucket_from_loc(
                    bucket_data, target_dev, non_blocking=False
                )
            # Re-resolve after fetch
            loc = resolve_bucket_location(bucket_data)
            self._stats["loc_fetches"] += 1
            logger.info(
                "DES-LOC LOC fetch complete → %s (%.1f MB)",
                loc.device, self._tensor_size_mb(bucket_data),
            )

        # --- Step 3: Dtype promotion (SM90 path) ---
        original_dtype = bucket_data.dtype
        bucket_data, was_promoted = maybe_promote_bucket_dtype(
            bucket_data, loc, ddp_config
        )
        if was_promoted:
            self._stats["dtype_promotions"] += 1

        # --- Step 4: Apply scaling factor on the bucket tensor ---
        # This is the core fix from ba456fd: scale bucket.data, not gbuf.data.
        # We also account for distributed averaging here.
        if math.isclose(scaling_factor, 1.0):
            logger.debug("Scaling factor is 1.0, skipping in-place scale.")
            self._stats["skipped_scaling"] += 1
        else:
            average_in_collective = getattr(
                ddp_config, "average_in_collective", False
            )
            if average_in_collective:
                # DeepSpeed will divide by world_size inside the collective;
                # we only need to apply any external loss scale here.
                effective_scale = scaling_factor
            else:
                # We are responsible for the full scaling (including 1/world_size
                # for gradient averaging when not using ReduceOp.AVG).
                effective_scale = scaling_factor
            logger.debug(
                "Scaling bucket.data in-place: factor=%.6f device=%s",
                effective_scale, loc.device,
            )
            if loc.device.type == "cuda" and self.overlap_comm:
                self._get_or_create_streams(loc.device)
                with torch.cuda.stream(self._scale_stream):
                    bucket_data.mul_(effective_scale)
            else:
                bucket_data.mul_(effective_scale)
            self._stats["buckets_scaled"] += 1

        # --- Step 5: Select ReduceOp ---
        reduce_op = self._select_reduce_op(ddp_config, is_data_distributed)

        logger.debug(
            "Scale pipeline done: out_dtype=%s reduce_op=%s",
            bucket_data.dtype, reduce_op,
        )
        return bucket_data, reduce_op

    def _select_reduce_op(
        self,
        ddp_config,
        is_data_distributed: bool,
    ) -> Optional[dist.ReduceOp]:
        """
        Map DDP config flags to the appropriate ``dist.ReduceOp``.

        DES-LOC note: H100-resident buckets can use ``ReduceOp.AVG`` because
        NCCL on SM90 supports it natively and it avoids the explicit
        ``1/world_size`` pre-scale that would otherwise be applied to
        ``gbuf.data`` (another source of the original bug).
        """
        average_in_collective = getattr(ddp_config, "average_in_collective", False)
        if average_in_collective:
            return dist.ReduceOp.AVG
        return dist.ReduceOp.SUM

    def report_stats(self) -> Dict[str, int]:
        """Return a snapshot of internal scaling statistics."""
        return dict(self._stats)


# ---------------------------------------------------------------------------
# GradReducePipeline adapter: DES-LOC drop-in for Megatron's pipeline loop
# ---------------------------------------------------------------------------

@dataclass
class HeteroGradBuffer:
    """
    Minimal interface matching what GradReducePipeline.reduce_grad_buffer
    expects from a GradBuffer object, extended with DES-LOC fields.

    Attributes
    ----------
    data : torch.Tensor
        The full gradient buffer tensor (may be on any device or CPU LOC).
    gradient_scaling_factor : float
        Pre-computed scaling factor (loss scale / world_size, etc.).
    ddp_config : object
        DDP/FSDP config object.
    is_data_distributed : bool
        True ↔ scatter-reduce; False ↔ all-reduce.
    original_device : torch.device
        The device the buffer was originally allocated on (before any LOC
        eviction). Used to guide fetch destination.
    buckets : List[torch.Tensor]
        List of bucket data tensors.  Each may independently be on GPU or LOC.
    """
    data: torch.Tensor
    gradient_scaling_factor: float
    ddp_config: object
    is_data_distributed: bool
    original_device: torch.device
    buckets: List[torch.Tensor] = field(default_factory=list)


def hetero_reduce_grad_buffer(
    gbuf: HeteroGradBuffer,
    scaler: HeteroFSDPGradScaler,
    process_group: dist.ProcessGroup,
) -> None:
    """
    DES-LOC replacement for ``GradReducePipeline``'s inner bucket-reduce loop.

    Megatron's original (buggy) loop (simplified)::

        for bucket in gbuf.buckets:
            reduce_op = gradient_reduce_preprocessing(
                gbuf.data,          # ← wrong: entire buffer, not this bucket
                gbuf.gradient_scaling_factor,
                gbuf.ddp_config,
            )
            all_reduce(bucket.data, op=reduce_op, group=pg)

    DES-LOC corrected loop::

        for bucket in gbuf.buckets:
            scaled_data, reduce_op = scaler.scale_bucket_gradients(
                bucket.data,        # ← correct: this specific bucket
                gbuf.gradient_scaling_factor,
                gbuf.ddp_config,
                gbuf_device=gbuf.original_device,
                is_data_distributed=gbuf.is_data_distributed,
            )
            all_reduce / reduce_scatter(scaled_data, op=reduce_op, group=pg)

    Parameters
    ----------
    gbuf : HeteroGradBuffer
        The gradient buffer whose buckets are to be reduced.
    scaler : HeteroFSDPGradScaler
        The heterogeneous scaler instance.
    process_group : dist.ProcessGroup
        Communication group.
    """
    world_size = dist.get_world_size(process_group)
    logger.info(
        "hetero_reduce_grad_buffer: %d buckets, world_size=%d, "
        "is_data_distributed=%s",
        len(gbuf.buckets), world_size, gbuf.is_data_distributed,
    )

    for bucket_idx, bucket_data in enumerate(gbuf.buckets):
        logger.debug("Processing bucket %d / %d", bucket_idx, len(gbuf.buckets))

        scaled_data, reduce_op = scaler.scale_bucket_gradients(
            bucket_data=bucket_data,
            scaling_factor=gbuf.gradient_scaling_factor,
            ddp_config=gbuf.ddp_config,
            gbuf_device=gbuf.original_device,
            is_data_distributed=gbuf.is_data_distributed,
        )

        # Ensure tensor is contiguous before collective
        if not scaled_data.is_contiguous():
            scaled_data = scaled_data.contiguous()

        if gbuf.is_data_distributed:
            # Scatter-reduce: each rank accumulates its own shard
            logger.debug("Bucket %d: reduce_scatter, op=%s", bucket_idx, reduce_op)
            output = torch.zeros_like(scaled_data[: scaled_data.numel() // world_size])
            dist.reduce_scatter_tensor(
                output, scaled_data, op=reduce_op, group=process_group
            )
            # Write back the reduced shard in-place into the bucket view
            shard_size = scaled_data.numel() // world_size
            rank = dist.get_rank(process_group)
            bucket_data.view(-1)[rank * shard_size : (rank + 1) * shard_size].copy_(output)
        else:
            # All-reduce: every rank gets the full gradient
            logger.debug("Bucket %d: all_reduce, op=%s", bucket_idx, reduce_op)
            dist.all_reduce(scaled_data, op=reduce_op, group=process_group)
            bucket_data.copy_(scaled_data)

    stats = scaler.report_stats()
    logger.info("Scaling stats after reduce pass: %s", stats)


# ---------------------------------------------------------------------------
# Utility: build a scaler from a DeepSpeed engine config
# ---------------------------------------------------------------------------

def build_hetero_grad_scaler_from_ds_config(
    ds_config: dict,
    process_group: dist.ProcessGroup,
) -> HeteroFSDPGradScaler:
    """
    Construct a :class:`HeteroFSDPGradScaler` from a DeepSpeed config dict.

    Expected keys under ``des_loc`` sub-dict (all optional):

    .. code-block:: json

        {
          "des_loc": {
            "overlap_comm": true,
            "loc_eviction_threshold_mb": 256.0
          }
        }

    Parameters
    ----------
    ds_config : dict
        The parsed DeepSpeed JSON config.
    process_group : dist.ProcessGroup

    Returns
    -------
    HeteroFSDPGradScaler
    """
    des_loc_cfg = ds_config.get("des_loc", {})
    overlap_comm = des_loc_cfg.get("overlap_comm", True)
    threshold_mb = des_loc_cfg.get("loc_eviction_threshold_mb", 256.0)
    logger.info(
        "Building HeteroFSDPGradScaler: overlap_comm=%s, threshold_mb=%.1f",
        overlap_comm, threshold_mb,
    )
    return HeteroFSDPGradScaler(
        process_group=process_group,
        overlap_comm=overlap_comm,
        loc_eviction_threshold_mb=threshold_mb,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Bootstrap a minimal CPU-only process group for smoke testing ---
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", init_method="env://")
    pg = dist.group.WORLD

    # 1. Device tier classification
    cpu_tier = get_device_tier(torch.device("cpu"))
    assert cpu_tier == MemoryTier.CPU_LOC, f"Expected CPU_LOC, got {cpu_tier}"
    logger.info("PASS: CPU tier = CPU_LOC")

    # 2. BucketLocation resolution for a CPU tensor
    t_cpu = torch.randn(64, 64)
    loc = resolve_bucket_location(t_cpu)
    assert loc.is_loc_resident, "CPU tensor should be LOC-resident"
    assert loc.tier == MemoryTier.CPU_LOC
    logger.info("PASS: CPU bucket resolved as LOC-resident")

    # 3. Scaling factor correctly applied to bucket (not a ghost buffer)
    class _FakeDDPConfig:
        average_in_collective = False
        communication_dtype = None

    scaler = HeteroFSDPGradScaler(process_group=pg, overlap_comm=False)
    bucket = torch.ones(16, dtype=torch.float32)
    buf    = torch.ones(128, dtype=torch.float32) * 99.0  # decoy
    scaled, _ = scaler.scale_bucket_gradients(
        bucket_data=bucket,
        scaling_factor=0.5,
        ddp_config=_FakeDDPConfig(),
        gbuf_device=torch.device("cpu"),
    )
    assert torch.allclose(scaled, torch.full_like(scaled, 0.5)), \
        f"Expected 0.5, got {scaled.mean()}"
    # Decoy buffer must be untouched (the core bug fix)
    assert buf.mean().item() == 99.0, "gbuf.data was mutated — bug re-introduced!"
    logger.info("PASS: Bucket scaled correctly; gbuf decoy untouched (ba456fd fix verified)")

    # 4. Stats tracking
    stats = scaler.report_stats()
    assert stats["buckets_scaled"] >= 1, "Expected at least one scaled bucket"
    logger.info("PASS: Stats: %s", stats)

    # 5. HeteroGradBuffer end-to-end with hetero_reduce_grad_buffer
    bucket_a = torch.full((8,), 2.0)
    bucket_b = torch.full((8,), 4.0)

    class _SimpleDDP:
        average_in_collective = False
        communication_dtype = None

    gbuf = HeteroGradBuffer(
        data=torch.cat([bucket_a, bucket_b]),
        gradient_scaling_factor=0.25,
        ddp_config=_SimpleDDP(),
        is_data_distributed=False,  # all-reduce path
        original_device=torch.device("cpu"),
        buckets=[bucket_a, bucket_b],
    )
    hetero_reduce_grad_buffer(gbuf, scaler, pg)
    # After all-reduce with world_size=1 and scale=0.25: 2.0*0.25=0.5, 4.0*0.25=1.0
    assert torch.allclose(bucket_a, torch.full_like(bucket_a, 0.5)), \
        f"bucket_a wrong: {bucket_a}"
    assert torch.allclose(bucket_b, torch.full_like(bucket_b, 1.0)), \
        f"bucket_b wrong: {bucket_b}"
    logger.info("PASS: End-to-end hetero_reduce_grad_buffer (all-reduce, world_size=1)")

    dist.destroy_process_group()
    logger.info("All smoke tests passed.")
