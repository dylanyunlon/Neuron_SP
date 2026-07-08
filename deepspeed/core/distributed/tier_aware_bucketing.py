# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""tier_aware_bucketing.py — Tier-aware gradient bucketing and C++ kernel dispatch.

Issue #35: core/distributed — DDP + gradient sync + tier-aware bucketing
addresses #35

Design overview
---------------
Heterogeneous GPU clusters in Neuron_SP (A6000 × 2 + H100-NVL × 1 + Blackwell × 2)
present two separate problems for gradient synchronization:

  1. **Bucket sizing**: each tier has a different optimal bucket size because
     PCIe bandwidth (A6000: ~16 GB/s, Blackwell via PCIe: ~64 GB/s) is much
     lower than NVLink (H100 SXM5: ~900 GB/s).  The existing
     ``compute_tier_bucket_sizes`` in param_and_grad_buffer.py handles the
     *sizing* problem by applying per-tier multipliers and syncing the minimum
     across the DP group.

  2. **Collective dispatch**: once gradients are ready, the all-reduce or
     reduce-scatter collective should use the fused INT8-compressed ring
     kernel (``fused_gradient_allreduce.cu``) on PCIe hops, which cuts PCIe
     traffic 2× vs BF16 and 4× vs FP32.  The existing Python-level
     ``_try_hetero_allreduce`` in finalize_model_grads.py only covers the
     non-DDP fallback path.  This module wires the C++ kernel into the
     primary ``ParamAndGradBucketGroup.start_grad_sync`` hot path.

This module provides:
  - ``TierAwareBucketingConfig`` — parameters governing tier-aware behaviour.
  - ``get_local_sm_version()`` — query the SM version of the local CUDA device.
  - ``get_local_tier_name()`` — map SM version to tier string (a6000/h100/…).
  - ``tier_bucket_multiplier(tier)`` — per-tier bucket-size multiplier table.
  - ``fused_allreduce_bucket(grad_data, group, sm_version, …)`` — call
    ``HeteroReduceOp.fused_gradient_allreduce`` C++ kernel on a flat BF16
    gradient tensor; falls back to ``torch.distributed.all_reduce`` if the
    kernel is not compiled or the tensor does not meet the preconditions.
  - ``fused_reduce_scatter_bucket(grad_data, output_shard, group, …)`` — call
    ``HeteroReduceOp.hetero_reduce_scatter`` for the ZeRO/distributed-optimizer
    reduce-scatter path.
  - ``TierAwareGradSyncMixin`` — mixin class with
    ``tier_aware_start_grad_sync()`` that ``ParamAndGradBucketGroup`` can call
    instead of the plain torch.distributed path when tier-aware mode is active.

Wiring into ParamAndGradBucketGroup
------------------------------------
``ParamAndGradBucketGroup.start_grad_sync`` already has the dispatch logic.
This module is imported there as an optional fast-path:

    from deepspeed.core.distributed.tier_aware_bucketing import (
        fused_allreduce_bucket,
        fused_reduce_scatter_bucket,
        get_local_sm_version,
    )

When the C++ kernel is unavailable (not compiled, non-BF16 tensor, tensor too
small) the functions return ``False`` and the caller falls back to the existing
torch.distributed path transparently.

Wiring into finalize_model_grads
----------------------------------
``_try_hetero_allreduce`` in finalize_model_grads.py already calls
``HeteroReduceOp.fused_gradient_allreduce``; this module provides the same
functionality via ``fused_allreduce_bucket`` so both code paths share one
implementation.

Public API
----------
  TierAwareBucketingConfig
  get_local_sm_version() → int
  get_local_tier_name() → str
  tier_bucket_multiplier(tier_name: str) → float
  fused_allreduce_bucket(grad_data, group, sm_version, min_elems) → bool
  fused_reduce_scatter_bucket(grad_data, output_shard, shard_offset,
                               shard_count, group, sm_version) → bool
  TierAwareGradSyncMixin
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier name → SM version mapping (used for reverse lookup)
# ---------------------------------------------------------------------------
_TIER_TO_SM: Dict[str, int] = {
    "blackwell": 120,   # B200 / GB200, NVLink 5.0
    "h100":      90,    # H100 SXM5 / H100 NVL
    "a6000":     86,    # A6000 / A6000 Ada (PCIe)
    "consumer":  86,    # RTX 4090 / RTX 3090 (PCIe)
    "unknown":   86,    # fallback
}

# Tier bucket-size multipliers (mirrors _TIER_BUCKET_MULTIPLIERS in
# param_and_grad_buffer.py; kept here for direct import by callers that
# want per-tier bucket info without importing the full buffer module).
_TIER_MULTIPLIERS: Dict[str, float] = {
    "blackwell": 2.0,
    "h100":      1.5,
    "a6000":     0.5,
    "consumer":  0.4,
    "unknown":   1.0,
}

# Minimum gradient tensor size in BF16 elements before the fused C++ kernel
# is applied.  Below this threshold the NCCL collective startup overhead
# dominates and the kernel provides no bandwidth benefit.
_DEFAULT_FUSED_AR_MIN_ELEMS: int = 65_536   # 128 KB of BF16


# ---------------------------------------------------------------------------
# TierAwareBucketingConfig
# ---------------------------------------------------------------------------

@dataclass
class TierAwareBucketingConfig:
    """Configuration governing tier-aware gradient bucketing behaviour.

    Attributes
    ----------
    enabled
        Master switch.  When False, all tier-aware behaviour is skipped and
        the module is a no-op.
    use_fused_gradient_allreduce
        When True, use the C++ INT8-compressed ring allreduce kernel
        (``fused_gradient_allreduce.cu``) instead of plain
        ``torch.distributed.all_reduce`` for BF16 gradient buckets that are
        large enough (≥ ``fused_ar_min_elems`` elements).
    use_fused_reduce_scatter
        When True, use the C++ heterogeneous reduce-scatter kernel
        (``hetero_reduce_scatter``) in the distributed-optimizer path.
    fused_ar_min_elems
        Minimum number of BF16 gradient elements for the fused kernel to be
        applied.  Below this threshold the standard torch.distributed path is
        used to avoid kernel-launch overhead exceeding the bandwidth saving.
        Default: 65 536 (128 KB of BF16).
    int8_compression_threshold_elems
        Minimum gradient tensor size for INT8 block-wise compression to be
        applied before the ring allreduce.  For very small tensors the
        compression overhead (block-norm compute + scale write) exceeds the
        bandwidth saving.  Default: 262 144 (512 KB of BF16).
    sm_version_override
        If non-zero, override the automatically-detected SM version of the
        local GPU.  Useful for testing on CPU or in environments where CUDA is
        not available.
    log_kernel_fallbacks
        Emit a DEBUG log entry each time the fused kernel falls back to the
        standard torch.distributed path.  Useful for diagnosing kernel
        availability issues without polluting production logs.
    """
    enabled: bool = True
    use_fused_gradient_allreduce: bool = True
    use_fused_reduce_scatter: bool = True
    fused_ar_min_elems: int = _DEFAULT_FUSED_AR_MIN_ELEMS
    int8_compression_threshold_elems: int = 262_144
    sm_version_override: int = 0
    log_kernel_fallbacks: bool = False


# ---------------------------------------------------------------------------
# Lazy-loaded C++ kernel module
# ---------------------------------------------------------------------------

_hetero_reduce_module = None
_hetero_reduce_available: Optional[bool] = None   # None = not yet checked


def _try_load_hetero_reduce() -> bool:
    """Attempt to load the hetero_reduce CUDA extension.

    Returns True on success, False if the extension has not been compiled.
    Result is cached after the first call.
    """
    global _hetero_reduce_module, _hetero_reduce_available

    if _hetero_reduce_available is not None:
        return _hetero_reduce_available

    try:
        from deepspeed.ops.hetero_reduce import HeteroReduceOp  # noqa: F401
        from deepspeed.ops.op_builder import HeteroReduceBuilder
        _hetero_reduce_module = HeteroReduceBuilder().load()
        # Verify the two functions needed for Issue #35 are present.
        if not (
            hasattr(_hetero_reduce_module, "gradient_compress")
            and hasattr(_hetero_reduce_module, "gradient_decompress")
            and hasattr(_hetero_reduce_module, "int8_ring_reduce_step")
            and hasattr(_hetero_reduce_module, "hetero_reduce_scatter")
        ):
            logger.debug(
                "[tier_aware_bucketing] hetero_reduce loaded but missing "
                "required functions; disabling fused kernel path."
            )
            _hetero_reduce_module = None
            _hetero_reduce_available = False
            return False
        _hetero_reduce_available = True
        logger.info(
            "[tier_aware_bucketing] hetero_reduce C++ extension loaded OK; "
            "fused gradient allreduce enabled."
        )
        return True
    except Exception as exc:
        logger.debug(
            "[tier_aware_bucketing] hetero_reduce extension not available (%s); "
            "fused kernel path disabled.",
            exc,
        )
        _hetero_reduce_available = False
        return False


def _hetero_op():
    """Return the loaded hetero_reduce module or None."""
    _try_load_hetero_reduce()
    return _hetero_reduce_module


# ---------------------------------------------------------------------------
# SM version / tier detection
# ---------------------------------------------------------------------------

def get_local_sm_version(override: int = 0) -> int:
    """Return the SM version of the current CUDA device.

    Args:
        override: If non-zero, return this value instead of querying the device.
                  Useful for testing and environments without a GPU.

    Returns:
        Integer SM version, e.g. 86, 90, 120.  Returns 86 as a safe default
        when CUDA is not available.
    """
    if override > 0:
        return override
    try:
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except Exception:
        return 86   # safe default: A6000 / consumer GPU


def get_local_tier_name(sm_version: Optional[int] = None) -> str:
    """Map a CUDA SM version to a DES-LOC tier name.

    Args:
        sm_version: Integer SM version.  If None, queries the current device.

    Returns:
        One of: "blackwell", "h100", "a6000", "consumer", "unknown".
    """
    if sm_version is None:
        sm_version = get_local_sm_version()

    if sm_version >= 120:
        return "blackwell"
    if sm_version >= 90:
        return "h100"
    if sm_version >= 86:
        # Distinguish A6000 (datacenter) from consumer GPUs by VRAM capacity.
        # RTX 4090 has 24 GB; A6000 has 48 GB; A6000 Ada has 48 GB.
        try:
            vram_bytes = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            if vram_gb >= 40:
                return "a6000"
            return "consumer"
        except Exception:
            return "a6000"   # assume datacenter GPU if we can't check
    return "unknown"


def tier_bucket_multiplier(tier_name: str) -> float:
    """Return the bucket-size multiplier for the given tier.

    Args:
        tier_name: Tier string from ``get_local_tier_name()``.

    Returns:
        Float multiplier to apply to the base bucket size.  Values < 1
        shrink the bucket (slow PCIe tiers); values > 1 enlarge it
        (fast NVLink tiers).
    """
    return _TIER_MULTIPLIERS.get(tier_name, 1.0)


# ---------------------------------------------------------------------------
# Fused allreduce — wires fused_gradient_allreduce.cu into DDP path
# ---------------------------------------------------------------------------

def fused_allreduce_bucket(
    grad_data: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    sm_version: int = 86,
    min_elems: int = _DEFAULT_FUSED_AR_MIN_ELEMS,
    config: Optional[TierAwareBucketingConfig] = None,
) -> bool:
    """Apply INT8-compressed ring allreduce to a flat BF16 gradient bucket.

    This is the primary C++ kernel integration point for Issue #35.  It calls
    the three-phase pipeline from ``fused_gradient_allreduce.cu``:

      Phase 1 — Compress: BF16 → INT8 with per-block FP32 scale
        (``gradient_compress`` / ``fused_compress_kernel``).

      Phase 2 — Ring reduce: ``world_size - 1`` steps of
        ``int8_ring_reduce_step`` (fused dequant + sum + requant) using
        ``torch.distributed.{isend,recv}`` for the peer transfers.

      Phase 3 — Finalise + decompress: scale by 1/world_size, then
        ``gradient_decompress`` (INT8 + scale → BF16).

    The function modifies ``grad_data`` **in-place** to hold the reduced
    gradient.

    Falls back to ``False`` (caller should use standard path) when:
      - The C++ extension is not compiled.
      - ``grad_data`` is not BF16 or not on CUDA.
      - ``grad_data.numel() < min_elems`` (too small for bandwidth benefit).
      - Any other exception during kernel invocation.

    Args:
        grad_data:  Flat BF16 CUDA tensor — the gradient bucket to reduce.
        group:      Data-parallel process group for peer communication.
        sm_version: SM version of the local GPU (86, 90, 120).
        min_elems:  Minimum element count to engage the fused path.
        config:     Optional TierAwareBucketingConfig; overrides min_elems if
                    ``config.fused_ar_min_elems`` is set.

    Returns:
        True if the fused kernel ran successfully, False to fall through to
        the standard torch.distributed.all_reduce path.
    """
    if config is not None:
        if not config.enabled or not config.use_fused_gradient_allreduce:
            return False
        min_elems = config.fused_ar_min_elems

    # Precondition checks — fast-path exits that skip the extension load.
    if not grad_data.is_cuda:
        return False
    if grad_data.dtype != torch.bfloat16:
        return False
    if not grad_data.is_contiguous():
        return False

    n_elems = grad_data.numel()
    if n_elems < min_elems:
        if config is not None and config.log_kernel_fallbacks:
            logger.debug(
                "[fused_allreduce_bucket] n_elems=%d < min_elems=%d; skipping",
                n_elems, min_elems,
            )
        return False

    world_size = dist.get_world_size(group=group)
    if world_size <= 1:
        return False   # no-op; caller needn't check world size separately

    op = _hetero_op()
    if op is None:
        if config is not None and config.log_kernel_fallbacks:
            logger.debug(
                "[fused_allreduce_bucket] hetero_reduce extension unavailable; "
                "falling back to standard allreduce."
            )
        return False

    try:
        # Allocate staging buffers for INT8 compression.
        # gradient_compress_bytes(n) = n bytes (one INT8 per BF16 element).
        # gradient_scale_bytes(n) = ceil(n / 256) * sizeof(float).
        n_scale_blocks = (n_elems + 255) // 256
        int8_self  = torch.empty(n_elems, dtype=torch.int8, device=grad_data.device)
        scale_self = torch.empty(n_scale_blocks, dtype=torch.float32, device=grad_data.device)
        int8_recv  = torch.empty(n_elems, dtype=torch.int8, device=grad_data.device)
        scale_recv = torch.empty(n_scale_blocks, dtype=torch.float32, device=grad_data.device)

        rank = dist.get_rank(group=group)

        # Phase 1: compress local gradient BF16 → INT8 + per-block scales.
        op.gradient_compress(int8_self, scale_self, grad_data, sm_version)

        # Phase 2: ring reduce-scatter.
        # Each step: send our current int8_self to the next rank, receive
        # from the previous rank, then fuse the two INT8 chunks in-place.
        # After (world_size - 1) steps every rank holds the fully-reduced
        # INT8 gradient with per-block scales.
        for step in range(world_size - 1):
            send_to   = (rank + 1) % world_size
            recv_from = (rank - 1 + world_size) % world_size

            # Overlap send and receive (isend + blocking recv pattern).
            req_data  = dist.isend(int8_self,  dst=send_to,  group=group, tag=step * 2)
            req_scale = dist.isend(scale_self, dst=send_to,  group=group, tag=step * 2 + 1)
            dist.recv(int8_recv,  src=recv_from, group=group, tag=step * 2)
            dist.recv(scale_recv, src=recv_from, group=group, tag=step * 2 + 1)
            req_data.wait()
            req_scale.wait()

            # Fused INT8 ring-reduce: dequant + sum + requant (in-place on self).
            op.int8_ring_reduce_step(
                int8_self, scale_self, int8_recv, scale_recv, sm_version
            )

        # Phase 3: finalise (divide scales by world_size) then decompress.
        op.gradient_allreduce_finalise(scale_self, n_elems, world_size)
        op.gradient_decompress(grad_data, int8_self, scale_self, sm_version)

        logger.debug(
            "[fused_allreduce_bucket] INT8-compressed ring allreduce done: "
            "n_elems=%d, world_size=%d, sm_version=%d",
            n_elems, world_size, sm_version,
        )
        return True

    except Exception as exc:
        logger.warning(
            "[fused_allreduce_bucket] kernel invocation failed (%s); "
            "caller will fall back to standard allreduce.",
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Fused reduce-scatter — wires hetero_reduce_scatter.cu into DistOpt path
# ---------------------------------------------------------------------------

def fused_reduce_scatter_bucket(
    grad_data: torch.Tensor,
    output_shard: torch.Tensor,
    shard_offset: int,
    shard_count: int,
    group: torch.distributed.ProcessGroup,
    sm_version: int = 86,
    config: Optional[TierAwareBucketingConfig] = None,
) -> bool:
    """Apply heterogeneous reduce-scatter to a flat BF16 gradient bucket.

    Calls ``hetero_reduce_scatter`` from the C++ extension, which:
      - Reduces all inputs across ranks in BF16 with FP32 accumulation
        (``launch_hetero_reduce_scatter``).
      - Writes only the local shard [shard_offset, shard_offset + shard_count)
        to ``output_shard``.

    This replaces ``torch.distributed.reduce_scatter_tensor`` on the
    distributed-optimizer path for BF16 gradient buckets.

    Falls back to False when the C++ extension is not available or preconditions
    are not met.

    Args:
        grad_data:    Full BF16 gradient tensor [total_elements].
        output_shard: Output buffer for the local shard [shard_count].
        shard_offset: Starting element index in the full gradient tensor.
        shard_count:  Number of elements this rank receives.
        group:        Data-parallel process group.
        sm_version:   SM version of the local GPU.
        config:       Optional TierAwareBucketingConfig.

    Returns:
        True if the reduce-scatter was applied by the fused kernel, False
        to fall through to the standard path.
    """
    if config is not None:
        if not config.enabled or not config.use_fused_reduce_scatter:
            return False

    if not grad_data.is_cuda:
        return False
    if grad_data.dtype != torch.bfloat16:
        return False
    if not grad_data.is_contiguous():
        return False
    if not output_shard.is_contiguous():
        return False
    if shard_count <= 0 or shard_count % 8 != 0:
        return False

    world_size = dist.get_world_size(group=group)
    if world_size <= 1:
        # Single-rank: copy the shard directly, no collective needed.
        output_shard.copy_(grad_data[shard_offset: shard_offset + shard_count])
        return True

    op = _hetero_op()
    if op is None:
        return False

    try:
        # Gather all ranks' gradient tensors via all-gather first, then
        # apply hetero_reduce_scatter to write only the local shard.
        #
        # Implementation: use all_gather_into_tensor to collect all shards
        # on each rank, then apply the fused reduce-scatter kernel that reads
        # all inputs and writes only the local shard.  This trades extra
        # memory (holding all shards) for the fused kernel's better compute
        # efficiency on heterogeneous hardware.
        #
        # For the simpler single-buffer path (already used by Megatron):
        # call all_reduce then slice — the hetero_reduce_scatter kernel gives
        # identical results with the heterogeneous-shard-weights optimisation
        # (larger shard to faster GPUs).
        #
        # NOTE: The canonical path gathers all world_size copies of grad_data
        # and passes them as a list[Tensor] to hetero_reduce_scatter.  In
        # practice the peer grad_data buffers live on different devices;
        # cross-device pointer passing requires cudaMemcpyPeer staging.  We
        # instead use the simpler all_gather+slice pattern here and let the
        # C++ kernel handle the local reduction.
        #
        # Gather: allocate a [world_size * numel] buffer, all-gather.
        n = grad_data.numel()
        gathered = torch.empty(
            world_size * n, dtype=torch.bfloat16, device=grad_data.device
        )
        dist.all_gather_into_tensor(gathered, grad_data, group=group)

        # Reshape into a list of tensors for the C++ kernel.
        inputs: List[torch.Tensor] = [
            gathered[i * n : (i + 1) * n] for i in range(world_size)
        ]

        op.hetero_reduce_scatter(
            output_shard, inputs, shard_offset, shard_count, sm_version
        )
        logger.debug(
            "[fused_reduce_scatter_bucket] n=%d, shard=[%d, %d), sm=%d",
            n, shard_offset, shard_offset + shard_count, sm_version,
        )
        return True

    except Exception as exc:
        logger.warning(
            "[fused_reduce_scatter_bucket] kernel invocation failed (%s); "
            "caller will fall back to standard reduce_scatter_tensor.",
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# TierAwareGradSyncMixin
# ---------------------------------------------------------------------------

class TierAwareGradSyncMixin:
    """Mixin for ``ParamAndGradBucketGroup`` to enable tier-aware grad sync.

    When mixed into ``ParamAndGradBucketGroup`` this mixin exposes
    ``tier_aware_start_grad_sync()``, which tries the fused C++ kernel path
    before falling back to the existing torch.distributed dispatch in
    ``start_grad_sync()``.

    Usage in ParamAndGradBucketGroup
    ----------------------------------
    The existing class structure requires no inheritance change; instead the
    bucket group calls these helpers directly:

        from deepspeed.core.distributed.tier_aware_bucketing import (
            fused_allreduce_bucket,
            fused_reduce_scatter_bucket,
            get_local_sm_version,
        )

        # In start_grad_sync, BEFORE the standard torch.distributed path:
        if _tier_aware_cfg is not None and not force_all_reduce:
            sm = get_local_sm_version(_tier_aware_cfg.sm_version_override)
            if use_dist_opt:
                # reduce-scatter path (ZeRO)
                for idx, bucket in enumerate(self.buckets):
                    local_view = self.cached_grad_buffer_shard_list[idx][rank]
                    applied = fused_reduce_scatter_bucket(
                        bucket.grad_data, local_view,
                        shard_offset=bucket.offset + rank * shard_size,
                        shard_count=shard_size,
                        group=communication_group,
                        sm_version=sm,
                        config=_tier_aware_cfg,
                    )
                    if not applied:
                        # fall back to dist_reduce_scatter_func
                        …
            else:
                for idx, bucket in enumerate(self.buckets):
                    applied = fused_allreduce_bucket(
                        bucket.grad_data,
                        group=communication_group,
                        sm_version=sm,
                        config=_tier_aware_cfg,
                    )
                    if not applied:
                        torch.distributed.all_reduce(bucket.grad_data, …)

    The ``tier_aware_start_grad_sync()`` method below encapsulates this
    pattern for convenience.

    Attributes
    ----------
    _tier_aware_config : Optional[TierAwareBucketingConfig]
        Configuration governing tier-aware behaviour.  Set by DDP init when
        the ``TierMap`` is available from ``parallel_state``.
    _local_sm_version : int
        Cached SM version of the local GPU.
    """

    # Set by DistributedDataParallel.__init__ via set_tier_aware_config().
    _tier_aware_config: Optional[TierAwareBucketingConfig] = None
    _local_sm_version: int = 86

    def set_tier_aware_config(
        self,
        config: TierAwareBucketingConfig,
        sm_version_override: int = 0,
    ) -> None:
        """Attach a TierAwareBucketingConfig and cache the SM version.

        Args:
            config: Tier-aware bucketing configuration.
            sm_version_override: If non-zero, use this SM version instead of
                                 querying the device.
        """
        self._tier_aware_config = config
        self._local_sm_version = get_local_sm_version(
            config.sm_version_override or sm_version_override
        )

    def tier_aware_start_grad_sync(
        self,
        buckets,
        communication_group: torch.distributed.ProcessGroup,
        use_dist_opt: bool,
        force_all_reduce: bool,
        reduce_op: torch.distributed.ReduceOp,
        gradient_scaling_factor: float,
        cached_grad_buffer_shard_list: list,
        intra_distributed_optimizer_instance_rank: int,
        dist_reduce_scatter_func,
        async_op: bool,
    ) -> bool:
        """Try the fused C++ kernel path for gradient synchronization.

        Applies per-bucket gradient scaling BEFORE calling the fused kernel
        (matching the scaling applied by the standard path in
        ``ParamAndGradBucketGroup.start_grad_sync``).

        Args:
            buckets:                         List of ParamAndGradBucket.
            communication_group:             DP / intra-DistOpt process group.
            use_dist_opt:                    True for reduce-scatter path.
            force_all_reduce:                True to bypass reduce-scatter.
            reduce_op:                       SUM or AVG (ignored for fused kernel
                                             which always averages).
            gradient_scaling_factor:         1/dp_size or MoE-adjusted factor.
            cached_grad_buffer_shard_list:   Per-bucket shard cache.
            intra_distributed_optimizer_instance_rank: Local rank within the
                                             intra-DistOpt group.
            dist_reduce_scatter_func:        Standard reduce-scatter function
                                             (fallback).
            async_op:                        Whether to dispatch async.

        Returns:
            True if the fused kernel handled the collective for ALL buckets.
            False if any bucket fell back — in which case the caller should
            proceed with the standard dispatch for all buckets.
        """
        cfg = self._tier_aware_config
        if cfg is None or not cfg.enabled:
            return False

        # Fused path is synchronous-only for now (the INT8 ring needs
        # blocking send/recv to maintain buffer ownership between steps).
        # When async_op=True, defer to the standard path which properly
        # manages the communication handle.
        if async_op:
            return False

        sm = self._local_sm_version
        all_applied = True

        for idx, bucket in enumerate(buckets):
            # Apply gradient scaling before the collective.
            scaled_grad = bucket.grad_data
            if gradient_scaling_factor != 1.0:
                scaled_grad = bucket.grad_data.clone()
                scaled_grad.mul_(gradient_scaling_factor)

            applied = False

            if use_dist_opt and not force_all_reduce:
                # Distributed-optimizer reduce-scatter path.
                # Determine the shard layout from the shard cache.
                if cached_grad_buffer_shard_list[idx] is None:
                    from deepspeed.core.distributed.param_and_grad_buffer import shard_buffer
                    dp_size = dist.get_world_size(group=communication_group)
                    cached_grad_buffer_shard_list[idx] = shard_buffer(
                        bucket.grad_data, dp_size
                    )
                shards = cached_grad_buffer_shard_list[idx]
                rank   = intra_distributed_optimizer_instance_rank
                local_shard = shards[rank]
                shard_count  = local_shard.numel()
                shard_offset = bucket.offset + rank * shard_count

                # Ensure shard_count is aligned to 8 for the kernel.
                if shard_count > 0 and shard_count % 8 == 0:
                    applied = fused_reduce_scatter_bucket(
                        scaled_grad,
                        local_shard,
                        shard_offset=shard_offset,
                        shard_count=shard_count,
                        group=communication_group,
                        sm_version=sm,
                        config=cfg,
                    )
                    if applied and gradient_scaling_factor != 1.0:
                        # The kernel already applied the scaled_grad; copy back.
                        bucket.grad_data.copy_(scaled_grad)
            else:
                # Standard all-reduce path.
                if gradient_scaling_factor != 1.0:
                    bucket.grad_data.mul_(gradient_scaling_factor)
                applied = fused_allreduce_bucket(
                    bucket.grad_data,
                    group=communication_group,
                    sm_version=sm,
                    config=cfg,
                )

            if not applied:
                all_applied = False
                # Undo the scaling we may have applied in-place above.
                if not use_dist_opt and gradient_scaling_factor != 1.0:
                    # Restore original so caller can apply its own scaling.
                    bucket.grad_data.div_(gradient_scaling_factor)

        return all_applied


# ---------------------------------------------------------------------------
# DDP-level integration helper
# ---------------------------------------------------------------------------

def attach_tier_aware_config_to_bucket_groups(
    bucket_groups: list,
    config: TierAwareBucketingConfig,
    sm_version_override: int = 0,
) -> None:
    """Attach a TierAwareBucketingConfig to all bucket groups in a DDP model.

    Call this once from ``DistributedDataParallel.__init__`` after the bucket
    groups have been created:

        attach_tier_aware_config_to_bucket_groups(
            self.bucket_groups + self.expert_parallel_bucket_groups,
            tier_cfg,
        )

    This requires each bucket group to either inherit ``TierAwareGradSyncMixin``
    or have been extended by ``extend_bucket_group_with_tier_aware_sync()``.

    Args:
        bucket_groups:      List of ParamAndGradBucketGroup instances.
        config:             Tier-aware bucketing configuration.
        sm_version_override: Override for SM version detection.
    """
    for bg in bucket_groups:
        if hasattr(bg, "set_tier_aware_config"):
            bg.set_tier_aware_config(config, sm_version_override)
        else:
            # Dynamically attach the mixin methods without full class rewrite.
            bg._tier_aware_config = config
            bg._local_sm_version = get_local_sm_version(
                config.sm_version_override or sm_version_override
            )
            # Bind the mixin method to the instance.
            import types
            bg.tier_aware_start_grad_sync = types.MethodType(
                TierAwareGradSyncMixin.tier_aware_start_grad_sync, bg
            )


def build_tier_aware_config_from_ddp_config(
    ddp_config,
) -> Optional[TierAwareBucketingConfig]:
    """Construct a TierAwareBucketingConfig from a DistributedDataParallelConfig.

    Returns None when tier-aware features are disabled (use_pcie_aware_overlap
    is False and no explicit tier configuration is set).

    This is the recommended construction path for DDP initialization:

        tier_cfg = build_tier_aware_config_from_ddp_config(ddp_config)
        if tier_cfg is not None:
            attach_tier_aware_config_to_bucket_groups(
                self.bucket_groups, tier_cfg
            )

    Args:
        ddp_config: DistributedDataParallelConfig instance.

    Returns:
        TierAwareBucketingConfig or None.
    """
    use_pcie = getattr(ddp_config, "use_pcie_aware_overlap", False)
    allow_skip = getattr(ddp_config, "allow_skip_grad_sync", True)

    # Enable tier-aware fused kernels when PCIe-aware overlap is on or when
    # the cluster is heterogeneous (TierMap available in parallel_state).
    try:
        import deepspeed.core.parallel_state as ps
        has_tier_map = ps.is_initialized() and ps.get_tier_map() is not None
    except Exception:
        has_tier_map = False

    if not use_pcie and not has_tier_map:
        return None

    cfg = TierAwareBucketingConfig(
        enabled=True,
        use_fused_gradient_allreduce=True,
        use_fused_reduce_scatter=has_tier_map,
        fused_ar_min_elems=_DEFAULT_FUSED_AR_MIN_ELEMS,
    )
    logger.info(
        "[tier_aware_bucketing] TierAwareBucketingConfig created: "
        "use_pcie=%s, has_tier_map=%s, fused_ar=%s, fused_rs=%s",
        use_pcie, has_tier_map,
        cfg.use_fused_gradient_allreduce,
        cfg.use_fused_reduce_scatter,
    )
    return cfg


# ---------------------------------------------------------------------------
# Bucket-size recommendation helpers (supplement param_and_grad_buffer.py)
# ---------------------------------------------------------------------------

def recommend_bucket_size_for_tier(
    base_bucket_size: int,
    tier_name: Optional[str] = None,
    sm_version: Optional[int] = None,
) -> int:
    """Return a tier-adjusted bucket size for the local GPU.

    Convenience wrapper around ``tier_bucket_multiplier`` for callers that
    prefer not to import ``param_and_grad_buffer`` directly.

    Args:
        base_bucket_size: NVLink-tuned default bucket size in elements.
        tier_name:        Override tier name.  If None, detected from SM.
        sm_version:       Override SM version.  If None, detected from device.

    Returns:
        Adjusted bucket size (int), always ≥ 1.
    """
    if tier_name is None:
        tier_name = get_local_tier_name(sm_version)
    mult = tier_bucket_multiplier(tier_name)
    return max(1, int(base_bucket_size * mult))


def dp_group_min_bucket_size(
    local_bucket_size: int,
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> int:
    """Synchronise the per-rank bucket size via all-reduce MIN across the DP group.

    All ranks must call this function.  Returns the globally-agreed minimum
    so that bucket boundaries are consistent for overlap_grad_reduce correctness.

    This is a thin Python wrapper around the same logic already implemented in
    ``compute_tier_bucket_sizes`` in ``param_and_grad_buffer.py``; duplicated
    here so this module is self-contained for Issue #35 unit testing.

    Args:
        local_bucket_size: This rank's tier-adjusted bucket size.
        dp_group:          Data-parallel process group.  None = world group.

    Returns:
        The minimum bucket size across all DP ranks.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_bucket_size

    try:
        size_tensor = torch.tensor(
            [local_bucket_size],
            dtype=torch.int64,
            device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
        )
        dist.all_reduce(size_tensor, op=dist.ReduceOp.MIN, group=dp_group)
        return int(size_tensor.item())
    except Exception as exc:
        logger.warning(
            "[tier_aware_bucketing] dp_group_min_bucket_size all-reduce MIN "
            "failed (%s); using local value %d.",
            exc, local_bucket_size,
        )
        return local_bucket_size
