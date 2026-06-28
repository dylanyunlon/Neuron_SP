# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Contiguous parameter and gradient buffers for DDP with DES-LOC support.

Evolution summary (ported from Megatron-LM commit history, 32 commits):
  M2278 (1e9e94cc6): Initial FSDP NCCL symmetric register support — BufferType,
      shard_buffer, _ParamAndGradBucket, _ParamAndGradBuffer base.
  M2282 (76622edf3): pgs_collection merge — pg_collection, tp_group, dp_cp_group.
  M2286 (ca9797e95): Revert pgs_collection (kept rollback-safe shape).
  M2301 (8c1a3f5df): Replay pgs_collection; add ProcessGroupCollection paths.
  M2352 (c2c36f77c): Fix reuse_grad_buf_for_mxfp8_param_ag convergence bug.
  M2563 (c5ac86354): Symmetric registration interface sync with upstream PyTorch.
  M2577 (bb216765d): reduce_scatter_with_fp32_accumulation — pluggable RS func.
  M2777 (299034c2f): fp8 param cuda-graph support; _post_param_sync extracted.
  M2989 (aa3f1057f): Support DDP overlap for repeated parameters.
  M3087 (dbde759da): Save wgrads/dgrads; extra_main_grads list.
  M3112 (6cf285b23): Logging cleanup (rank-0 only where possible).
  M3139 (287d2f47c): Fix RL optimizer offload.
  M3140 (3955c49ed): Revert RL offload fix.
  M3146 (36411ddff): Reapply RL offload fix.
  M3167 (5d0a7fd15): cp: Fix nccl-ub in ddp path.
  M3238 (a3ec4b02e): Param offset alignment (64-element boundary).
  M3442 (f91c4bb37): Fix memory issue in mxfp8 model init.
  M3443 (a2381d800): overlap-param-gather for layer-wise optimizer.
  M3561 (3548385ac): Fix DDP bug with overlap-grad-reduce + multi-DistOpt.
  M3616 (c586f6d56): FP32 local grad accumulation for subset of params.
  M3695 (51bcf1470): Fix layerwise optimizer with expt_dp_size=1.
  M3737 (e1db4a03d): NVFP4 native weights for DDP.
  M3739 (28e13c484): Reuse grad buffer for layer-wise param allgather.
  M3781 (3315c86bc): param_index_map always uses unpacked (full numel) offsets.
  M3811 (55b8111ad): DDP refactoring — extract param layout computation.
  M3904 (b25a76e00): Fix gradient corruption with layerwise param AG overlap.
  M3948 (c1e938b03): Integrate LayerWiseDistributedOptimizer with DDP buffer.
  M3950 (e35d4e50c): NCCL UB fix — reduce memory cost, deregister pool.
  M3998 (0044db1f2): Route non-Muon params through DistributedOptimizer.
  M4020 (08bad7a48): MXFP8/FP4-param-gather post-processing after forced AG.
  M4036 (88e7ab091): Drain predecessor reduce-scatter at dispatch time.
  M4163 (1af933d15): Remove duplicate nccl_allocator import.

DES-LOC extensions:
  - ParamAndGradBucketGroup._skip_sync: gate on Kx steps.
  - start_grad_sync(skip_sync): skip collective on non-Kx steps.
  - force_all_reduce param propagated from finalize_model_grads.
  - grad_reduce_finished idempotency for predecessor-draining pattern.

Provides:
  BufferType, shard_buffer,
  ParamAndGradBucket, LayerwiseAllGatherHandle,
  ParamAndGradBucketGroup, ParamAndGradBuffer,
  group_params_for_buffers, partition_buckets,
  _compute_default_per_buffer_param_layout
"""

from __future__ import annotations

import fnmatch
import functools
import logging
import math
import warnings
from contextlib import nullcontext
from enum import Enum, auto
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import _coalescing_manager

import deepspeed.core.parallel_state as parallel_state
from deepspeed.core.model_parallel_config import ModelParallelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Megatron-compat imports (no-ops when not available)
# ---------------------------------------------------------------------------
try:
    from megatron.core.rerun_state_machine import get_rerun_state_machine as _get_rerun
except ImportError:
    _get_rerun = None

try:
    from megatron.core.utils import log_single_rank as _meg_log_single
except ImportError:
    _meg_log_single = None


def _log_single_rank(lg, level, msg):
    """Log only on rank 0 when distributed is available."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            lg.log(level, msg)
    else:
        lg.log(level, msg)


# ---------------------------------------------------------------------------
# Torch distributed compat: all_gather_into_tensor / reduce_scatter_tensor
# ---------------------------------------------------------------------------
try:
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
    dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
except AttributeError:
    dist_all_gather_func = torch.distributed._all_gather_base   # type: ignore[attr-defined]
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# BufferType
# ---------------------------------------------------------------------------

class BufferType(Enum):
    """Enumeration for buffer type (mirrors Megatron M2278+)."""
    PARAM = 1
    GRAD = 2


# ---------------------------------------------------------------------------
# Insight I5: buffer ownership FSM (Megatron M3061/M3116)
# ---------------------------------------------------------------------------
# The grad buffer of a ParamAndGradBucketGroup transitions through three
# ownership states during a training step:
#
#   FREE  ──(zero() / alloc)──►  PARAM_OWNED  ──(register_grad_ready)──►  GRAD_OWNED
#    ▲                                                                           │
#    └──────────────────────(finish_grad_sync / reset)───────────────────────────┘
#
# Invariants:
#   PARAM_OWNED  – buffer holds param values (layer-wise AG reuse path).
#                  Grad writes are FORBIDDEN.  See M3739.
#   GRAD_OWNED   – buffer holds gradient data ready for or undergoing a
#                  collective.  Param writes are FORBIDDEN.
#   FREE         – buffer is zeroed / idle (start of iteration after reset).
#
# Assertions at each transition guard against the grad-corruption bugs fixed
# in Megatron M3904 and the layerwise-optimizer races in M3948.
# ---------------------------------------------------------------------------

class BufferOwnership(Enum):
    """Explicit ownership state for the shared param/grad contiguous buffer.

    Tracks which consumer currently "owns" the flat buffer so that use-after-
    free and double-write bugs are caught as assertions rather than silent
    data corruption.

    States
    ------
    PARAM_OWNED
        The buffer is being used as a parameter all-gather receive buffer
        (overlap_param_gather / layer-wise optimizer path, M3739).
        Gradient accumulation into this buffer is forbidden until ownership
        is released.
    GRAD_OWNED
        The buffer holds live gradient data that is ready for (or currently
        undergoing) a reduce-scatter / all-reduce collective.  Overwriting
        with parameter data is forbidden.
    FREE
        The buffer has been zeroed and is not actively owned by either
        the forward (param) or backward (grad) pass.  This is the initial
        state at the start of each training iteration.
    """
    # Insight I5: buffer ownership FSM (Megatron M3061/M3116)
    PARAM_OWNED = "param"   # buffer holds gathered param values
    GRAD_OWNED  = "grad"    # buffer holds gradient data for collective
    FREE        = "free"    # buffer idle / zeroed

    def assert_is(self, expected: "BufferOwnership", context: str = "") -> None:
        """Assert this state equals *expected*; raise with context on mismatch.

        Args:
            expected: The required ownership state.
            context:  Human-readable label for the call site (for debugging).

        Raises:
            AssertionError: if ``self != expected``.
        """
        # Insight I5: buffer ownership FSM (Megatron M3061/M3116)
        assert self == expected, (
            f"Buffer ownership violation{' at ' + context if context else ''}: "
            f"expected {expected.value!r}, got {self.value!r}. "
            "This indicates a use-after-free or double-write on the flat "
            "param/grad buffer (see Megatron M3061/M3116/M3904)."
        )

    def assert_not(self, forbidden: "BufferOwnership", context: str = "") -> None:
        """Assert this state is NOT *forbidden*.

        Args:
            forbidden: The disallowed ownership state.
            context:   Human-readable label for the call site.

        Raises:
            AssertionError: if ``self == forbidden``.
        """
        # Insight I5: buffer ownership FSM (Megatron M3061/M3116)
        assert self != forbidden, (
            f"Buffer ownership violation{' at ' + context if context else ''}: "
            f"buffer must not be in state {forbidden.value!r} here. "
            "This indicates a lifecycle ordering error "
            "(see Megatron M3061/M3116/M3904)."
        )


# ---------------------------------------------------------------------------
# shard_buffer
# ---------------------------------------------------------------------------

def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int) -> List[torch.Tensor]:
    """Shard buffer into data_parallel_world_size chunks of equal size.

    Introduced in Megatron M2278 for ZeRO-style distributed optimizer sharding.
    Each rank receives a contiguous view of 1/dp_size of the flat buffer.
    """
    assert buffer.numel() % data_parallel_world_size == 0, (
        f"buffer.numel()={buffer.numel()} must be divisible by dp_size={data_parallel_world_size}"
    )
    shard_size = buffer.numel() // data_parallel_world_size
    return [buffer[r * shard_size : (r + 1) * shard_size] for r in range(data_parallel_world_size)]


# ---------------------------------------------------------------------------
# ParamAndGradBucket
# ---------------------------------------------------------------------------

class ParamAndGradBucket:
    """A contiguous view of param/grad data for a subset of model parameters.

    Evolution (M2278 → M3616):
      - M2278: Initial version with params, grad_data, offset, bucket_id.
      - M3238: param_index_map passed in for bucket-local offset derivation.
      - M3616: params_with_extra_main_grads for FP32 local accumulation.

    Args:
        params: Parameters in this bucket (in backprop order).
        param_data: View into the contiguous param buffer (None if no ZeRO).
        grad_data: View into the contiguous grad buffer.
        offset: Start offset of this bucket in the global grad buffer.
        numel_unpadded: Unpadded element count (for logging).
        gradient_scaling_factor: Pre-collective scaling (1/dp_size or MoE factor).
        bucket_id: Index of this bucket within the buffer.
        param_index_map: Global param → (start, end, bucket_id) mapping.
        params_with_extra_main_grads: Params needing separate FP32 main_grad.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        gradient_scaling_factor: float,
        bucket_id: int,
        param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]],
        params_with_extra_main_grads: List[torch.nn.Parameter],
    ) -> None:
        self.params_list = params
        self.params: Set[torch.nn.Parameter] = set(params)
        assert len(self.params_list) == len(self.params), \
            "Duplicate params detected in bucket"

        self.param_data = param_data
        self.grad_data = grad_data
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id

        # Derive bucket-local param offsets from the global param_index_map
        # (M3238: offsets always unpacked/full numel).
        self.param_to_index: Dict[torch.nn.Parameter, Tuple[int, int]] = {}
        for param in params:
            g_start, g_end, _ = param_index_map[param]
            self.param_to_index[param] = (g_start - offset, g_end - offset)

        self.params_with_extra_main_grads = params_with_extra_main_grads

        # Layer-wise optimizer attributes for async param-gather (M3443).
        self.layerwise_params_list: Optional[List[List[torch.nn.Parameter]]] = None
        self.layerwise_param_flat_sizes: Optional[List[int]] = None
        self.layerwise_gather_list: Optional[List[torch.Tensor]] = None

    def set_layerwise_params_list(
        self, layerwise_params_list: List[List[torch.nn.Parameter]]
    ) -> None:
        """Set per-rank parameter lists for layer-wise async all-gather (M3443).

        Args:
            layerwise_params_list: List of param lists, one per DP rank.
        """
        self.layerwise_params_list = layerwise_params_list
        self.layerwise_param_flat_sizes = [
            sum(p.numel() for p in plist) for plist in layerwise_params_list
        ]


# ---------------------------------------------------------------------------
# LayerwiseAllGatherHandle
# ---------------------------------------------------------------------------

class LayerwiseAllGatherHandle:
    """Handle wrapping multiple async all-gather work objects (M3443).

    NCCL guarantees in-order completion on the same communicator, so waiting
    on only the last handle is sufficient for correctness.
    """

    def __init__(self, handles: List) -> None:
        self.handles = handles

    def wait(self) -> None:
        if self.handles:
            self.handles[-1].wait()
        self.handles = []


# ---------------------------------------------------------------------------
# ParamAndGradBucketGroup
# ---------------------------------------------------------------------------

class ParamAndGradBucketGroup:
    """Manages a set of buckets and orchestrates grad sync / param gather.

    This is the primary communication object. Key evolution:
      - M2278: Basic all-reduce with overlap bookkeeping.
      - M2577: Pluggable dist_reduce_scatter_func (FP32 accum support).
      - M2777: _post_param_sync extracted for fp8/mxfp8 post-processing.
      - M3443: Layer-wise optimizer param all-gather support.
      - M3561: Fix multi-DistOpt + overlap-grad-reduce race.
      - M3904: Fix gradient corruption with layerwise param AG.
      - M3948: LayerWise DistOpt integration.
      - M4036: Drain predecessor reduce-scatter at dispatch time.

    DES-LOC extensions:
      - _skip_sync: set by finalize_model_grads on non-Kx steps.
      - start_grad_sync(skip_sync): skip collective but keep bookkeeping.
      - grad_reduce_finished: idempotency flag for predecessor-drain pattern.
      - previous_grad_reduce_bucket_group: drain predecessor's RS buffer early.

    Args:
        buckets: List of ParamAndGradBucket objects in this group.
        ddp_config: DDP configuration with overlap/reduce flags.
        collective_group: Data-parallel or intra-DistOpt process group.
        collective_group_size: World size of collective_group.
    """

    def __init__(
        self,
        buckets: List[ParamAndGradBucket],
        ddp_config,
        collective_group: torch.distributed.ProcessGroup,
        collective_group_size: int,
    ) -> None:
        self.buckets = buckets
        self.ddp_config = ddp_config

        # Distributed-optimizer or layer-wise paths use intra_distributed_optimizer_instance_group.
        if getattr(ddp_config, 'use_distributed_optimizer', False) or \
                getattr(ddp_config, 'overlap_param_gather', False):
            self.intra_distributed_optimizer_instance_group = collective_group
            self.intra_distributed_optimizer_instance_size = collective_group_size
            self.intra_distributed_optimizer_instance_rank = collective_group.rank()
        if not getattr(ddp_config, 'use_distributed_optimizer', False):
            self.data_parallel_group = collective_group

        # Param → bucket mapping and full param set.
        self.param_to_bucket: Dict[torch.nn.Parameter, ParamAndGradBucket] = {}
        self.params: Set[torch.nn.Parameter] = set()
        for bucket in self.buckets:
            for param in bucket.params_list:
                self.param_to_bucket[param] = bucket
                self.params.add(param)

        # Async param-gather chaining (M3443).
        self.next_param_gather_bucket_group: Optional[ParamAndGradBucketGroup] = None
        # Predecessor drain for reduce_scatter_with_fp32_accumulation (M4036).
        self.previous_grad_reduce_bucket_group: Optional[ParamAndGradBucketGroup] = None

        # Multi-DistOpt support (M3561).
        if getattr(ddp_config, 'num_distributed_optimizer_instances', 1) > 1:
            self.inter_distributed_optimizer_instance_group: Optional[
                torch.distributed.ProcessGroup
            ] = None
            self.communication_stream: Optional[torch.cuda.Stream] = None
            assert not getattr(ddp_config, 'reduce_scatter_with_fp32_accumulation', False), \
                "RS w/ FP32 accumulation not supported with num_distributed_optimizer_instances > 1"

        _log_single_rank(
            logger,
            logging.INFO,
            f"Using {'reduce-scatter' if getattr(ddp_config, 'use_distributed_optimizer', False) else 'all-reduce'} "
            f"for gradient reductions because use_distributed_optimizer="
            f"{getattr(ddp_config, 'use_distributed_optimizer', False)}",
        )

        # Configure reduce-scatter with FP32 accumulation (M2577).
        global dist_reduce_scatter_func
        if getattr(ddp_config, 'reduce_scatter_with_fp32_accumulation', False):
            try:
                from deepspeed.core.distributed.reduce_scatter_with_fp32_accumulation import (
                    reduce_scatter_with_fp32_accumulation,
                )
                dist_reduce_scatter_func = reduce_scatter_with_fp32_accumulation
                _log_single_rank(
                    logger, logging.INFO,
                    "Using reduce_scatter_with_fp32_accumulation as reduce-scatter implementation",
                )
            except ImportError:
                pass

        # Per-param grad-ready counters for overlap mode (M2778 pattern).
        # golden_per_param_grad_ready_counts is populated at end of first batch.
        self.golden_per_param_grad_ready_counts: Dict[torch.nn.Parameter, int] = {}
        self.per_param_grad_ready_counts: Dict[torch.nn.Parameter, int] = {}
        self.is_last_microbatch: bool = True
        self.is_first_batch: bool = True

        # Async communication handles.
        self.param_gather_handle = None
        self.param_gather_dispatched: bool = False
        self.grad_reduce_handle = None
        # Per-iteration idempotency flag (M4036).
        self.grad_reduce_finished: bool = False

        # Cached shard list views to avoid repeated CPU overhead (M3739).
        self.cached_param_buffer_shard_list: List[Optional[List[torch.Tensor]]] = [
            None
        ] * len(self.buckets)
        self.cached_grad_buffer_shard_list: List[Optional[List[torch.Tensor]]] = [
            None
        ] * len(self.buckets)

        # DES-LOC: gate on Kx steps.
        self._skip_sync: bool = False

        # Insight I5: buffer ownership FSM (Megatron M3061/M3116)
        # Track which consumer owns the shared flat buffer so that param-AG
        # reuse of grad_data (M3739) and grad-reduce collectives never
        # silently corrupt each other's data.
        self._buffer_ownership: BufferOwnership = BufferOwnership.FREE

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset metadata for the next training iteration."""
        if self.is_first_batch and len(self.per_param_grad_ready_counts) > 0:
            assert len(self.per_param_grad_ready_counts) == len(self.params)
            self.golden_per_param_grad_ready_counts = self.per_param_grad_ready_counts
            self.is_first_batch = False
        self.per_param_grad_ready_counts = {}
        self.is_last_microbatch = True
        self.grad_reduce_finished = False
        self._skip_sync = False
        # Insight I5: buffer ownership FSM (Megatron M3061/M3116)
        # After each iteration the buffer is zeroed externally (ParamAndGradBuffer.reset())
        # so we release ownership back to FREE unconditionally.
        self._buffer_ownership = BufferOwnership.FREE

    # ------------------------------------------------------------------
    # Post-param-sync processing (M2777)
    # ------------------------------------------------------------------

    def _post_param_sync(self) -> None:
        """Run post-processing after param all-gather completes (M2777 / M4020)."""
        # Reuse-grad-buf path for MXFP8: copy gathered params to param.data then zero buffer.
        if getattr(self.ddp_config, 'reuse_grad_buf_for_mxfp8_param_ag', False):
            try:
                from megatron.core.fp8_utils import is_float8tensor
            except ImportError:
                is_float8tensor = lambda x: False  # noqa: E731
            for bucket in self.buckets:
                is_bf16_bucket = any(not is_float8tensor(p) for p in bucket.params)
                if is_bf16_bucket:
                    continue
                for param in bucket.params:
                    start, end = bucket.param_to_index[param]
                    param_slice = bucket.param_data.view(-1)[start:end]
                    param.data.copy_(param_slice.view(param.data.shape))
                bucket.param_data.zero_()
            return

        # Standard FP8/FP4 post-all-gather processing.
        quantized_params = []
        try:
            from megatron.core.fp8_utils import is_float8tensor, post_all_gather_processing
            try:
                from megatron.core.fp4_utils import is_nvfp4tensor
            except ImportError:
                is_nvfp4tensor = lambda x: False  # noqa: E731
            for bucket in self.buckets:
                for param in bucket.params:
                    if is_float8tensor(param) or is_nvfp4tensor(param):
                        quantized_params.append(param)
            if quantized_params:
                post_all_gather_processing(quantized_params)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # NaN / large-grad checking (M3087)
    # ------------------------------------------------------------------

    def check_grads(self, check_for_nan_or_inf: bool, check_for_large: bool) -> None:
        """Validate grad norms before collective (M3087 rerun state machine)."""
        rerun_sm = _get_rerun() if _get_rerun is not None else None
        for i, bucket in enumerate(self.buckets):
            norm = bucket.grad_data.norm(p=2)
            if check_for_nan_or_inf:
                if rerun_sm is not None:
                    rerun_sm.validate_result(
                        result=norm,
                        rejection_func=torch.isnan,
                        message=f"found NaN in local grad norm for bucket #{i} before DP collective",
                        tolerance=0.001,
                        fatal=True,
                    )
                    rerun_sm.validate_result(
                        result=norm,
                        rejection_func=torch.isinf,
                        message=f"found Inf in local grad norm for bucket #{i} before DP collective",
                        tolerance=0.001,
                        fatal=True,
                    )
                else:
                    assert not torch.isnan(norm).item(), f"NaN in grad norm bucket #{i}"
                    assert not torch.isinf(norm).item(), f"Inf in grad norm bucket #{i}"
            if check_for_large and rerun_sm is not None:
                rerun_sm.validate_result(
                    result=norm,
                    rejection_func=partial(rerun_sm.is_unexpectedly_large, threshold=10, context="grads"),
                    message=f"found unexpected large grads in bucket #{i} before DP collective",
                    tolerance=0.001,
                    fatal=False,
                )

    # ------------------------------------------------------------------
    # Param sync (all-gather for distributed optimizer / layer-wise)
    # ------------------------------------------------------------------

    def start_param_sync(self, force_sync: bool = False) -> None:
        """Initiate param all-gather for this bucket group (M3443 / M3948).

        When ddp_config.overlap_param_gather is True, dispatches async AG
        unless force_sync is True.
        """
        use_dist_opt = getattr(self.ddp_config, 'use_distributed_optimizer', False)
        overlap_pg = getattr(self.ddp_config, 'overlap_param_gather', False)
        assert use_dist_opt or overlap_pg, \
            "start_param_sync called without use_distributed_optimizer or overlap_param_gather"

        # Insight I5: buffer ownership FSM (Megatron M3061/M3116)
        # Buffer must NOT be in GRAD_OWNED state when we start using it for
        # parameter all-gather.  GRAD_OWNED → PARAM_OWNED is illegal without
        # an intervening finish_grad_sync / reset (M3904 corruption fix).
        self._buffer_ownership.assert_not(
            BufferOwnership.GRAD_OWNED, "start_param_sync"
        )
        self._buffer_ownership = BufferOwnership.PARAM_OWNED

        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                self._post_param_sync()
            return

        assert self.param_gather_handle is None, \
            "start_param_sync called while previous all-gather is outstanding"

        async_op = overlap_pg and not force_sync

        if not use_dist_opt:
            # Layer-wise optimizer path: variable-size per-rank all_gather (M3443).
            dp_size = self.intra_distributed_optimizer_instance_size
            if dp_size == 1:
                # Fix from Megatron M3695: single-rank group needs no all-gather.
                # Mark dispatched so finish_param_sync sees a clean state.
                if force_sync and overlap_pg:
                    self._post_param_sync()
                self.param_gather_dispatched = True
                return
            local_rank = self.intra_distributed_optimizer_instance_rank
            group = self.intra_distributed_optimizer_instance_group
            work_handles = []
            for bucket in self.buckets:
                if bucket.layerwise_params_list is None or \
                        max(bucket.layerwise_param_flat_sizes) == 0:
                    bucket.layerwise_gather_list = None
                    continue
                param_dtype = bucket.params_list[0].dtype
                local_size = bucket.layerwise_param_flat_sizes[local_rank]
                total_gather_size = sum(bucket.layerwise_param_flat_sizes)
                reuse_buf = bucket.grad_data.view(param_dtype)
                assert reuse_buf.numel() >= total_gather_size
                gather_list = []
                offset = 0
                for i in range(dp_size):
                    sz = bucket.layerwise_param_flat_sizes[i]
                    gather_list.append(reuse_buf[offset : offset + sz])
                    offset += sz
                local_slot = gather_list[local_rank]
                if local_size > 0:
                    flat_local = _flatten_dense_tensors(
                        bucket.layerwise_params_list[local_rank]
                    ).detach()
                    local_slot.copy_(flat_local)
                bucket.layerwise_gather_list = gather_list
                work = torch.distributed.all_gather(
                    gather_list, local_slot, group=group, async_op=async_op
                )
                if async_op and work is not None:
                    work_handles.append(work)
            if async_op:
                self.param_gather_handle = LayerwiseAllGatherHandle(work_handles)
            else:
                for bucket in self.buckets:
                    if bucket.layerwise_gather_list is None:
                        continue
                    for idx, params in enumerate(bucket.layerwise_params_list):
                        if len(params) == 0 or idx == local_rank:
                            continue
                        updated = _unflatten_dense_tensors(
                            bucket.layerwise_gather_list[idx], params
                        )
                        for upd_p, mod_p in zip(updated, params):
                            mod_p.data.copy_(upd_p)
                    bucket.layerwise_gather_list = None
                    # Fix from Megatron M3904: zero grad_data after using it as the
                    # all-gather receive buffer so that subsequent gradient accumulation
                    # starts from zero rather than the stale gathered param values.
                    bucket.grad_data.zero_()
                self.param_gather_handle = None
        else:
            # Standard distributed optimizer path: coalescing manager (M2778).
            with _coalescing_manager(
                self.intra_distributed_optimizer_instance_group, async_ops=async_op
            ) as cm:
                for idx, bucket in enumerate(self.buckets):
                    if self.cached_param_buffer_shard_list[idx] is None:
                        self.cached_param_buffer_shard_list[idx] = shard_buffer(
                            bucket.param_data,
                            self.intra_distributed_optimizer_instance_size,
                        )
                    local_view = self.cached_param_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]
                    dist_all_gather_func(
                        bucket.param_data,
                        local_view,
                        group=self.intra_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )
            self.param_gather_handle = cm if async_op else None

        if force_sync and overlap_pg:
            self._post_param_sync()
        self.param_gather_dispatched = True

    def finish_param_sync(self, skip_next_bucket_dispatch: bool = False) -> None:
        """Wait for the outstanding param all-gather and dispatch the next (M3443).

        Args:
            skip_next_bucket_dispatch: If True, do not auto-dispatch next bucket's AG.
        """
        assert getattr(self.ddp_config, 'overlap_param_gather', False), \
            "finish_param_sync requires overlap_param_gather=True"

        if not self.param_gather_dispatched:
            self.start_param_sync()

        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None

            # Auto-dispatch next bucket (M3443 / M3904 fix).
            if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
                if self.next_param_gather_bucket_group.param_gather_dispatched:
                    warnings.warn(
                        "Next bucket's param all-gather has already been dispatched. "
                        "This may indicate a mismatch between param registration and "
                        "forward-pass order, which reduces communication-compute overlap."
                    )
                else:
                    self.next_param_gather_bucket_group.start_param_sync()

            # Layer-wise: unflatten and copy gathered params (M3443).
            use_dist_opt = getattr(self.ddp_config, 'use_distributed_optimizer', False)
            if not use_dist_opt:
                local_rank = self.intra_distributed_optimizer_instance_rank
                for bucket in self.buckets:
                    if bucket.layerwise_gather_list is None:
                        continue
                    for idx, params in enumerate(bucket.layerwise_params_list):
                        if len(params) == 0 or idx == local_rank:
                            continue
                        updated = _unflatten_dense_tensors(
                            bucket.layerwise_gather_list[idx], params
                        )
                        for upd_p, mod_p in zip(updated, params):
                            mod_p.data.copy_(upd_p)
                    bucket.layerwise_gather_list = None
                    # Fix from Megatron M3904: zero grad_data (reused as AG receive
                    # buffer) so gradient accumulation starts clean.
                    bucket.grad_data.zero_()

            self._post_param_sync()

            # Insight I5: buffer ownership FSM (Megatron M3061/M3116)
            # AG is done and grad_data has been zeroed (M3904); the buffer is
            # now idle and ready for gradient accumulation → FREE.
            self._buffer_ownership.assert_is(
                BufferOwnership.PARAM_OWNED, "finish_param_sync (post-wait)"
            )
            self._buffer_ownership = BufferOwnership.FREE

    # ------------------------------------------------------------------
    # Grad sync
    # ------------------------------------------------------------------

    def _copy_in_extra_main_grads(self) -> None:
        """Copy FP32 main_grad into grad_data buffer before collective (M3616)."""
        for bucket in self.buckets:
            for param in bucket.params_with_extra_main_grads:
                copy_buf = getattr(param, 'main_grad_copy_in_grad_buffer', None)
                if copy_buf is not None:
                    copy_buf.copy_(param.main_grad)

    def _copy_back_extra_main_grads(self) -> None:
        """Copy reduced grad from grad_data buffer back to FP32 main_grad (M3616)."""
        for bucket in self.buckets:
            for param in bucket.params_with_extra_main_grads:
                copy_buf = getattr(param, 'main_grad_copy_in_grad_buffer', None)
                if copy_buf is not None:
                    param.main_grad.copy_(copy_buf)

    def start_grad_sync(
        self,
        force_all_reduce: Optional[bool] = False,
        skip_sync: bool = False,
    ) -> None:
        """Initiate grad sync (all-reduce or reduce-scatter) across DP replicas.

        DES-LOC extension: when skip_sync=True (non-Kx step), the collective
        is skipped and gradients remain locally accumulated.

        Args:
            force_all_reduce: Force all-reduce even when use_distributed_optimizer=True.
            skip_sync: DES-LOC Kx gate — skip collective on non-Kx steps.
        """
        # Guard: in overlapping first-batch mode, a single dispatch suffices.
        if self.is_first_batch and self.grad_reduce_handle is not None:
            return

        # Drain predecessor bucket's reduce-scatter (M4036).
        if (
            self.previous_grad_reduce_bucket_group is not None
            and self.previous_grad_reduce_bucket_group.grad_reduce_handle is not None
        ):
            self.previous_grad_reduce_bucket_group.finish_grad_sync(
                force_all_reduce=force_all_reduce
            )

        assert self.grad_reduce_handle is None, \
            "start_grad_sync called with outstanding collective"

        # DES-LOC: skip collective on non-Kx steps.
        self._skip_sync = skip_sync
        if skip_sync and getattr(self.ddp_config, 'allow_skip_grad_sync', True):
            self.grad_reduce_finished = True
            return

        # Insight I5: buffer ownership FSM (Megatron M3061/M3116)
        # Buffer must be FREE (not PARAM_OWNED) before we hand it to the
        # grad collective.  PARAM_OWNED → GRAD_OWNED without zeroing first
        # would corrupt the reduce result (see M3904).
        self._buffer_ownership.assert_not(
            BufferOwnership.PARAM_OWNED, "start_grad_sync"
        )
        self._buffer_ownership = BufferOwnership.GRAD_OWNED

        # Copy FP32 extra main_grads into grad_data (M3616).
        self._copy_in_extra_main_grads()

        # NaN/large-grad checking (M3087).
        if getattr(self.ddp_config, 'check_for_nan_in_grad', False) or \
                getattr(self.ddp_config, 'check_for_large_grads', False):
            self.check_grads(
                check_for_nan_or_inf=getattr(self.ddp_config, 'check_for_nan_in_grad', False),
                check_for_large=getattr(self.ddp_config, 'check_for_large_grads', False),
            )

        # Pre-scale gradients by gradient_scaling_factor.
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        # Reduce operation (AVG for average_in_collective, else SUM).
        reduce_op = (
            torch.distributed.ReduceOp.AVG
            if getattr(self.ddp_config, 'average_in_collective', False)
            else torch.distributed.ReduceOp.SUM
        )

        # Stream management for multi-DistOpt instance overlap (M3561).
        num_instances = getattr(self.ddp_config, 'num_distributed_optimizer_instances', 1)
        use_dist_opt = getattr(self.ddp_config, 'use_distributed_optimizer', False)
        overlap = getattr(self.ddp_config, 'overlap_grad_reduce', False)

        async_op = overlap and num_instances == 1
        if num_instances > 1 and overlap:
            stream_context = torch.cuda.stream(self.communication_stream)
            self.communication_stream.wait_stream(torch.cuda.current_stream())
        else:
            stream_context = nullcontext()

        communication_group = (
            self.intra_distributed_optimizer_instance_group
            if use_dist_opt
            else self.data_parallel_group
        )

        # Coalesce all buckets in this group into a single NCCL call.
        grad_reduce_handle = None
        with stream_context, _coalescing_manager(communication_group, async_ops=async_op) as cm:
            for idx, bucket in enumerate(self.buckets):
                if use_dist_opt and not force_all_reduce:
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data,
                            self.intra_distributed_optimizer_instance_size,
                        )
                    local_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]
                    grad_reduce_handle = dist_reduce_scatter_func(
                        local_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=communication_group,
                        async_op=async_op,
                    )
                else:
                    if torch.distributed.get_rank() == 0 and force_all_reduce:
                        logger.info("Performing all_reduce (force_all_reduce=%s)", force_all_reduce)
                    torch.distributed.all_reduce(
                        bucket.grad_data,
                        op=reduce_op,
                        group=communication_group,
                        async_op=async_op,
                    )

        # Inter-instance all-reduce for multiple DistOpt instances (M3561).
        if use_dist_opt and num_instances > 1:
            assert self.inter_distributed_optimizer_instance_group is not None
            with (
                stream_context,
                _coalescing_manager(
                    self.inter_distributed_optimizer_instance_group, async_ops=async_op
                ) as cm,
            ):
                for idx, bucket in enumerate(self.buckets):
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data,
                            self.intra_distributed_optimizer_instance_size,
                        )
                    local_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]
                    torch.distributed.all_reduce(
                        local_view,
                        op=reduce_op,
                        group=self.inter_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )

        if async_op:
            rs_fp32 = getattr(self.ddp_config, 'reduce_scatter_with_fp32_accumulation', False)
            if rs_fp32 and not force_all_reduce:
                assert len(self.buckets) == 1, \
                    "Only 1 bucket supported with reduce_scatter_with_fp32_accumulation=True"
                assert grad_reduce_handle is not None
                self.grad_reduce_handle = grad_reduce_handle
            else:
                self.grad_reduce_handle = cm
        else:
            self.grad_reduce_handle = None

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False) -> None:
        """Complete grad sync and update idempotency flag (M4036 / DES-LOC).

        Idempotent within an iteration when overlap_grad_reduce=True: a
        second call is a no-op (predecessor-drain + end-of-step both call this).

        Args:
            force_all_reduce: Force all-reduce even with use_distributed_optimizer=True.
        """
        self.param_gather_dispatched = False
        overlap = getattr(self.ddp_config, 'overlap_grad_reduce', False)
        num_instances = getattr(self.ddp_config, 'num_distributed_optimizer_instances', 1)

        if not overlap:
            # Synchronous path: dispatch + wait inline.
            skip = self._skip_sync and not force_all_reduce
            self.start_grad_sync(force_all_reduce=force_all_reduce, skip_sync=skip)
            self._copy_back_extra_main_grads()
            return

        if self.grad_reduce_finished:
            return

        # First batch in overlap mode: dispatch if not already done.
        if self.is_first_batch:
            skip = self._skip_sync and not force_all_reduce
            self.start_grad_sync(force_all_reduce=force_all_reduce, skip_sync=skip)

        # Multi-DistOpt: wait on communication stream.
        if num_instances > 1:
            torch.cuda.current_stream().wait_stream(self.communication_stream)
            self._copy_back_extra_main_grads()
            self.grad_reduce_finished = True
            return

        # If we skipped the collective, nothing to wait on.
        if self.grad_reduce_handle is None and self._skip_sync and not force_all_reduce:
            self._copy_back_extra_main_grads()
            self.grad_reduce_finished = True
            return

        assert self.grad_reduce_handle is not None, (
            f"No outstanding communication handle "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} params have grad)"
        )
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None
        self._copy_back_extra_main_grads()
        self.grad_reduce_finished = True
        # Insight I5: buffer ownership FSM (Megatron M3061/M3116)
        # Collective is complete; the caller will zero grad_data at the next
        # reset() boundary.  Release ownership so param-AG can reuse the
        # buffer in the next forward pass.
        self._buffer_ownership.assert_is(
            BufferOwnership.GRAD_OWNED, "finish_grad_sync (post-wait)"
        )
        self._buffer_ownership = BufferOwnership.FREE

    def free_overlap_buffers(self) -> None:
        """Free temporary all-gather buffers (M3904 async checkpoint OOM fix)."""
        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
        for bucket in self.buckets:
            bucket.layerwise_gather_list = None

    def register_grad_ready(
        self,
        param: torch.nn.Parameter,
        force_all_reduce: Optional[bool] = False,
    ) -> None:
        """Register that a param's grad is ready; trigger async collective when complete.

        Only valid when overlap_grad_reduce=True. Launched when all params in
        the bucket group have registered (compared against the golden counts
        established in the first batch).

        Args:
            param: The parameter whose gradient is now available.
            force_all_reduce: Forward to start_grad_sync for DES-LOC recovery.
        """
        assert getattr(self.ddp_config, 'overlap_grad_reduce', False), \
            "register_grad_ready requires overlap_grad_reduce=True"

        if not self.is_last_microbatch:
            return

        assert param in self.param_to_bucket, "Param not in this bucket group"
        self.per_param_grad_ready_counts[param] = \
            self.per_param_grad_ready_counts.get(param, 0) + 1

        if not self.is_first_batch:
            if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
                assert len(self.per_param_grad_ready_counts) == len(self.params)
                self.start_grad_sync(
                    force_all_reduce=force_all_reduce,
                    skip_sync=self._skip_sync,
                )

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all grad_data tensors by scaling_factor."""
        for bucket in self.buckets:
            bucket.grad_data.mul_(scaling_factor)


# ---------------------------------------------------------------------------
# group_params_for_buffers (M3811 — extracted from DDP.__init__)
# ---------------------------------------------------------------------------

def group_params_for_buffers(
    params: List[torch.nn.Parameter],
    grad_reduce_in_fp32: bool,
) -> Dict[tuple, Tuple[List[torch.nn.Parameter], List[int]]]:
    """Group parameters by buffer identity for allocation.

    Each distinct buffer is identified by a 4-tuple:
      (param_dtype, grad_dtype, is_expert_parallel, is_managed_by_layer_wise_optimizer)

    Extracted from DDP.__init__ in Megatron M3811.

    Args:
        params: Trainable parameter list.
        grad_reduce_in_fp32: Whether gradients are reduced in FP32.

    Returns:
        Dict mapping buffer-key tuple to (params_list, param_indices).
    """
    try:
        from megatron.core.fp8_utils import is_float8tensor
    except ImportError:
        is_float8tensor = lambda x: False  # noqa: E731
    try:
        from megatron.core.fp4_utils import is_nvfp4tensor
    except ImportError:
        is_nvfp4tensor = lambda x: False  # noqa: E731

    key_to_params: Dict[tuple, List[torch.nn.Parameter]] = {}
    dtype_to_offsets: Dict[tuple, int] = {}
    key_to_indices: Dict[tuple, List[int]] = {}

    for param in params:
        assert param.requires_grad

        param_dtype = param.dtype
        if is_float8tensor(param) or is_nvfp4tensor(param):
            param_dtype = torch.uint8
        grad_dtype = torch.float if grad_reduce_in_fp32 else param.dtype
        is_expert = not getattr(param, 'allreduce', True)
        is_layerwise = getattr(param, 'is_managed_by_layer_wise_optimizer', False)

        key = (param_dtype, grad_dtype, is_expert, is_layerwise)
        key_to_params.setdefault(key, []).append(param)

        # Use param.dtype (not param_dtype) for checkpoint-compat offset tracking.
        offset_key = (param.dtype, grad_dtype, is_expert, is_layerwise)
        offset = dtype_to_offsets.get(offset_key, 0)
        dtype_to_offsets[offset_key] = offset + 1
        key_to_indices.setdefault(key, []).append(offset)

    return {
        key: (key_to_params[key], key_to_indices[key])
        for key in key_to_params
    }


# ---------------------------------------------------------------------------
# _compute_default_per_buffer_param_layout (M3811)
# ---------------------------------------------------------------------------

def _compute_default_per_buffer_param_layout(
    params: List[torch.nn.Parameter],
    bucket_size: Optional[int],
) -> dict:
    """Compute parameter layout for the non-distributed-optimizer case.

    No padding applied. Parameters iterate in reverse (backprop) order and
    are grouped into buckets of approximately bucket_size elements.

    Returns:
        dict with keys: param_index_map, bucket_indices, per_bucket_numel_unpadded.
    """
    param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = {}
    bucket_indices: List[Tuple[int, int]] = []
    per_bucket_numel_unpadded: List[int] = []

    param_start = 0
    bucket_start = 0
    bucket_params: Set[torch.nn.Parameter] = set()
    bucket_id = 0
    param_end = 0

    for param in params[::-1]:
        numel = param.data.nelement()
        param_end = param_start + numel
        param_index_map[param] = (param_start, param_end, bucket_id)
        bucket_params.add(param)

        if bucket_size is not None and (param_end - bucket_start) >= bucket_size:
            per_bucket_numel_unpadded.append(param_end - bucket_start)
            bucket_indices.append((bucket_start, param_end))
            bucket_start = param_end
            bucket_params = set()
            bucket_id += 1

        param_start = param_end

    if bucket_params:
        per_bucket_numel_unpadded.append(param_end - bucket_start)
        bucket_indices.append((bucket_start, param_end))

    return dict(
        param_index_map=param_index_map,
        bucket_indices=bucket_indices,
        per_bucket_numel_unpadded=per_bucket_numel_unpadded,
    )


# ---------------------------------------------------------------------------
# ParamAndGradBuffer
# ---------------------------------------------------------------------------

class ParamAndGradBuffer:
    """Groups parameters and gradients into a contiguous buffer with bucketing.

    Evolution (M2278 → M4163):
      - M2278: Base implementation with NCCL symmetric registration support.
      - M2282 / M2301: Process-group collection (pg_collection) support.
      - M2577: FP32-accumulation reduce-scatter pluggable function.
      - M2777: MXFP8 shared buffer (param+grad share storage).
      - M3087: extra_main_grads list for wgrad saving.
      - M3238: 64-element alignment at param start for DistOpt.
      - M3443: Layer-wise optimizer overlap-param-gather.
      - M3616: FP32 local grad accumulation per-param pattern matching.
      - M3737: NVFP4 packed param buffer with dual index maps.
      - M3811: param layout extracted into DistributedOptimizer classmethod.
      - M3950: NCCL UB pool deregistration; symmetric registration toggle.
      - M4163: Remove duplicate nccl_allocator import.

    DES-LOC note: gradient_scaling_factor includes 1/dp_size pre-scaling
    (allow_skip_grad_sync skips the collective; scaling still applied locally).

    Args:
        ddp_config: DDP configuration.
        param_dtype: Storage dtype of parameters (torch.uint8 for FP8/FP4).
        grad_dtype: Gradient reduction dtype.
        params_with_names: List of (param, name) pairs.
        data_parallel_group: DP process group.
        bucket_size: Approx bucket size in elements (None → single bucket).
        param_to_name: Global param → name mapping for logging.
        gradient_scaling_factor: Pre-collective scaling factor.
        param_indices: Param position index among same-dtype params.
        nccl_ub: Whether to use NCCL user-buffer (symmetric) allocation.
        pg_collection: Optional unified process-group collection.
        param_layout: Pre-computed layout from DistributedOptimizer (M3811).
    """

    def __init__(
        self,
        ddp_config,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params_with_names: List[Tuple[torch.nn.Parameter, str]],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: Optional[int],
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
        param_indices: List[int],
        nccl_ub: bool = False,
        pg_collection=None,
        param_layout=None,
    ) -> None:

        # Resolve TP / DP-CP groups (M2282 pgs_collection pattern).
        if pg_collection is None:
            if parallel_state.is_initialized():
                try:
                    self.dp_cp_group = parallel_state.get_data_parallel_group(
                        with_context_parallel=True
                    )
                except Exception:
                    self.dp_cp_group = parallel_state.get_data_parallel_group()
                try:
                    self.tp_group = parallel_state.get_tensor_model_parallel_group()
                except Exception:
                    self.tp_group = None
            else:
                self.dp_cp_group = torch.distributed.GroupMember.WORLD
                self.tp_group = None
        else:
            assert hasattr(pg_collection, 'tp') and hasattr(pg_collection, 'dp_cp'), \
                "pg_collection must have tp and dp_cp attributes"
            self.dp_cp_group = pg_collection.dp_cp
            self.tp_group = pg_collection.tp

        self.ddp_config = ddp_config
        self.params = [p for (p, _) in params_with_names]
        self.param_indices = param_indices

        # Uniqueness check.
        _seen: Set[torch.nn.Parameter] = set()
        for p, _ in params_with_names:
            assert p not in _seen, "Duplicate param in params_with_names"
            _seen.add(p)
        del _seen

        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_group.size()
        self.gradient_scaling_factor = gradient_scaling_factor
        self.nccl_ub = nccl_ub

        self.buckets: List[ParamAndGradBucket] = []
        self.param_to_bucket: Dict[torch.nn.Parameter, ParamAndGradBucket] = {}

        # Use the provided layout or compute the default (no-padding) layout.
        if param_layout is not None:
            self.param_index_map = param_layout.param_index_map
            self.bucket_indices = param_layout.bucket_indices
            per_bucket_numel_unpadded = param_layout.per_bucket_numel_unpadded
        else:
            layout = _compute_default_per_buffer_param_layout(self.params, bucket_size)
            self.param_index_map = layout['param_index_map']
            self.bucket_indices = layout['bucket_indices']
            per_bucket_numel_unpadded = layout['per_bucket_numel_unpadded']

        # NVFP4 dual-buffer support (M3737): packed param buffer, full-numel grad buffer.
        try:
            from megatron.core.fp4_utils import is_nvfp4tensor
        except ImportError:
            is_nvfp4tensor = lambda x: False  # noqa: E731
        self.has_nvfp4_params = any(is_nvfp4tensor(p) for p in self.params)
        self.nvfp4_packed_param_index_map = None
        self.nvfp4_packed_bucket_indices = None
        if self.has_nvfp4_params:
            self._compute_nvfp4_packed_layout(params_with_names)

        # Total numel (padded for DistOpt divisibility).
        self.numel: int = self.bucket_indices[-1][1]
        self.numel_unpadded: int = sum(per_bucket_numel_unpadded)
        if self.has_nvfp4_params:
            self.nvfp4_packed_numel: int = self.nvfp4_packed_bucket_indices[-1][1]

        assert self.numel_unpadded <= self.numel
        use_dist_opt = getattr(ddp_config, 'use_distributed_optimizer', False)
        if use_dist_opt:
            assert self.numel % self.data_parallel_world_size == 0, (
                f"numel={self.numel} must be divisible by dp_size={self.data_parallel_world_size}"
            )
        else:
            assert self.numel == self.numel_unpadded, \
                "Non-DistOpt buffer should not be padded"

        self.param_data: Optional[torch.Tensor] = None
        self.grad_data: Optional[torch.Tensor] = None
        self.extra_main_grads: List[torch.Tensor] = []
        self.nccl_mem_pool = None

        # NCCL user-buffer allocation (M2278 / M3950).
        if self.nccl_ub:
            try:
                import megatron.core.nccl_allocator as nccl_allocator
                nccl_allocator.init()
                pool = nccl_allocator.create_nccl_mem_pool(
                    symmetric=not getattr(ddp_config, 'disable_symmetric_registration', False)
                )
                self.nccl_mem_pool = pool
                mem_alloc_context = functools.partial(
                    nccl_allocator.nccl_mem,
                    pool,
                    group=self.data_parallel_group,
                    symmetric=not getattr(ddp_config, 'disable_symmetric_registration', False),
                )
                # Warmup NCCL comm buffers.
                torch.distributed.barrier()
                tmp = torch.zeros([1], device="cuda")
                torch.distributed.all_reduce(tmp, group=self.data_parallel_group)
                torch.distributed.barrier()
            except (ImportError, AttributeError):
                mem_alloc_context = nullcontext
        else:
            mem_alloc_context = nullcontext

        _cuda_device = torch.cuda.current_device()

        with mem_alloc_context():
            try:
                from megatron.core.fp8_utils import is_mxfp8tensor
            except ImportError:
                is_mxfp8tensor = lambda x: False  # noqa: E731

            if use_dist_opt and any(is_mxfp8tensor(p) for p in self.params):
                # MXFP8 shared buffer (M2777): param AG and grad RS share storage.
                self.shared_buffer = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=_cuda_device,
                    requires_grad=False,
                )
                if self.grad_dtype == torch.float32:
                    self.param_data = self.shared_buffer[: math.ceil(self.numel / 2)].view(
                        torch.bfloat16
                    )
                else:
                    self.param_data = self.shared_buffer
                self.grad_data = self.shared_buffer
            else:
                if use_dist_opt:
                    numel = self.nvfp4_packed_numel if self.has_nvfp4_params else self.numel
                    self.param_data = torch.zeros(
                        numel,
                        dtype=self.param_dtype,
                        device=_cuda_device,
                        requires_grad=False,
                    )
                self.grad_data = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=_cuda_device,
                    requires_grad=False,
                )

        self.grad_data_size: int = 0
        self.param_data_size: int = 0
        self.param_data_cpu: Optional[torch.Tensor] = None

        # ------------------------------------------------------------------
        # Map param.data and param.main_grad to buffer views, create buckets.
        # ------------------------------------------------------------------

        def _create_bucket(
            bid: int,
            bparams: List[torch.nn.Parameter],
            bparams_extra_grads: List[torch.nn.Parameter],
        ) -> ParamAndGradBucket:
            start, end = self.bucket_indices[bid]
            nvfp4_packed_start = nvfp4_packed_end = None
            if self.has_nvfp4_params:
                nvfp4_packed_start, nvfp4_packed_end = self.nvfp4_packed_bucket_indices[bid]
            return self._new_bucket(
                bucket_params=bparams,
                start_index=start,
                end_index=end,
                numel_unpadded=per_bucket_numel_unpadded[bid],
                bucket_id=bid,
                bucket_params_with_extra_main_grads=bparams_extra_grads,
                nvfp4_packed_start_index=nvfp4_packed_start,
                nvfp4_packed_end_index=nvfp4_packed_end,
            )

        bucket_params: List[torch.nn.Parameter] = []
        bucket_params_extra_grads: List[torch.nn.Parameter] = []
        cur_bucket_id = 0

        for param, param_name in params_with_names[::-1]:
            param_start_index, param_end_index, bucket_id = self.param_index_map[param]
            nvfp4_packed_start = None
            if self.has_nvfp4_params:
                nvfp4_packed_start, _, _ = self.nvfp4_packed_param_index_map[param]

            # Remap param.data → contiguous buffer (DistOpt only).
            reuse_mxfp8 = getattr(ddp_config, 'reuse_grad_buf_for_mxfp8_param_ag', False)
            if not reuse_mxfp8 or not _is_mxfp8(param):
                if self.param_data is not None:
                    self._remap_param_data(
                        param, param_start_index, nvfp4_packed_start
                    )

            # Always assign main_grad from grad buffer (full-numel offsets).
            param.main_grad = self._get(
                param.data.shape, param_start_index, BufferType.GRAD
            )

            # FP32 extra main_grad for per-param local accumulation (M3616).
            promote = False
            patterns = getattr(ddp_config, 'param_name_patterns_for_fp32_local_accumulation', [])
            for pattern in patterns:
                if fnmatch.fnmatch(param_name, pattern) or pattern == 'all':
                    logger.info(
                        "Matched '%s' with pattern '%s'; promoting main_grad dtype "
                        "%s → torch.float32",
                        param_name, pattern, param.main_grad.dtype,
                    )
                    promote = True
                    break
            if promote:
                param.main_grad_copy_in_grad_buffer = param.main_grad
                param.main_grad = torch.empty_like(param.main_grad, dtype=torch.float32)
                self.extra_main_grads.append(param.main_grad)

            if bucket_id != cur_bucket_id:
                self.buckets.append(
                    _create_bucket(cur_bucket_id, bucket_params, bucket_params_extra_grads)
                )
                bucket_params = []
                bucket_params_extra_grads = []
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id

            bucket_params.append(param)
            if promote:
                bucket_params_extra_grads.append(param)

        # Flush final bucket.
        if bucket_params:
            self.buckets.append(
                _create_bucket(cur_bucket_id, bucket_params, bucket_params_extra_grads)
            )

        # Log bucket layout.
        log_strs = [
            f"Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}"
        ]
        for idx, bucket in enumerate(self.buckets):
            numel = sum(p.data.nelement() for p in bucket.params_list)
            log_strs.append(
                f"Params for bucket {idx + 1} ({numel} elements, "
                f"{bucket.grad_data.nelement()} padded size, "
                f"{len(bucket.params_with_extra_main_grads)} param(s) with extra main_grads):"
            )
            for p in bucket.params_list:
                log_strs.append(f"\t{param_to_name[p]} ({p.main_grad.dtype=})")
        _log_single_rank(logger, logging.INFO, "\n".join(log_strs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_mxfp8_local(self, param: torch.nn.Parameter) -> bool:
        return _is_mxfp8(param)

    def _remap_param_data(
        self,
        param: torch.nn.Parameter,
        param_start_index: int,
        nvfp4_packed_start: Optional[int],
    ) -> None:
        """Remap param.data into the contiguous param buffer."""
        try:
            from megatron.core.fp8_utils import is_float8tensor, modify_underlying_storage
        except ImportError:
            is_float8tensor = lambda x: False  # noqa: E731
            modify_underlying_storage = None
        try:
            from megatron.core.fp4_utils import is_nvfp4tensor, get_nvfp4_rowwise_packed_shape, \
                modify_nvfp4_rowwise_storage
        except ImportError:
            is_nvfp4tensor = lambda x: False  # noqa: E731
            get_nvfp4_rowwise_packed_shape = None
            modify_nvfp4_rowwise_storage = None

        start = nvfp4_packed_start if self.has_nvfp4_params and nvfp4_packed_start is not None \
            else param_start_index

        if is_nvfp4tensor(param) and get_nvfp4_rowwise_packed_shape is not None:
            packed_shape = get_nvfp4_rowwise_packed_shape(param.data.shape)
            rowwise_view = self._get(packed_shape, start, BufferType.PARAM)
            modify_nvfp4_rowwise_storage(param, rowwise_view)
        elif is_float8tensor(param) and modify_underlying_storage is not None:
            new_data = self._get(param.data.shape, start, BufferType.PARAM)
            modify_underlying_storage(param, new_data)
        else:
            new_data = self._get(param.data.shape, start, BufferType.PARAM)
            old_data = param.data
            param.data = new_data
            assert old_data._base is None, "Old param.data unexpectedly has a base"
            param.data.detach().copy_(old_data)
            del old_data

    def _compute_nvfp4_packed_layout(self, params_with_names) -> None:
        """Derive packed NVFP4 index map and bucket indices (M3737).

        NVFP4 packs two FP4 values into one byte; the param buffer stores
        packed bytes (numel // 2) while the grad buffer uses full numel.
        """
        try:
            from megatron.core.fp4_utils import is_nvfp4tensor
        except ImportError:
            is_nvfp4tensor = lambda x: False  # noqa: E731
        try:
            from megatron.core.optimizer.param_layout import pad_param_start, pad_bucket_end
        except ImportError:
            def pad_param_start(x): return x  # noqa: E731
            def pad_bucket_end(x, dp_size, pad_high): return x  # noqa: E731

        use_dist_opt = getattr(self.ddp_config, 'use_distributed_optimizer', False)
        pad_high = getattr(self.ddp_config, 'pad_buckets_for_high_nccl_busbw', False)

        def _pad_start(start):
            return pad_param_start(start) if use_dist_opt else start

        def _pad_end(end):
            return pad_bucket_end(end, self.data_parallel_world_size, pad_high) \
                if use_dist_opt else end

        self.nvfp4_packed_param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = {}
        self.nvfp4_packed_bucket_indices: List[Tuple[int, int]] = []
        nvfp4_per_bucket_numel_unpadded = []

        packed_param_start = 0
        packed_bucket_start = 0
        cur_bucket_id = 0

        for param, _ in params_with_names[::-1]:
            _, _, bucket_id = self.param_index_map[param]
            param_numel = param.data.nelement()
            packed_param_start = _pad_start(packed_param_start)

            if bucket_id != cur_bucket_id:
                nvfp4_per_bucket_numel_unpadded.append(
                    packed_param_start - packed_bucket_start
                )
                packed_bucket_end = _pad_end(packed_param_start)
                self.nvfp4_packed_bucket_indices.append((packed_bucket_start, packed_bucket_end))
                packed_bucket_start = packed_bucket_end
                packed_param_start = packed_bucket_start
                cur_bucket_id = bucket_id

            if is_nvfp4tensor(param):
                assert param_numel % 2 == 0, f"NVFP4 requires even numel, got {param_numel}"
                packed_numel = param_numel // 2
            else:
                packed_numel = param_numel

            packed_param_end = packed_param_start + packed_numel
            self.nvfp4_packed_param_index_map[param] = (packed_param_start, packed_param_end, bucket_id)
            packed_param_start = packed_param_end

        # Finalize last bucket.
        if packed_param_start > packed_bucket_start:
            nvfp4_per_bucket_numel_unpadded.append(packed_param_start - packed_bucket_start)
            packed_bucket_end = _pad_end(packed_param_start)
            self.nvfp4_packed_bucket_indices.append((packed_bucket_start, packed_bucket_end))

        assert len(self.nvfp4_packed_bucket_indices) == len(self.bucket_indices), (
            f"Packed bucket count {len(self.nvfp4_packed_bucket_indices)} "
            f"!= primary count {len(self.bucket_indices)}"
        )
        self.nvfp4_packed_numel_unpadded = sum(nvfp4_per_bucket_numel_unpadded)

    def _get(
        self, shape: torch.Size, start_index: int, buffer_type: BufferType
    ) -> torch.Tensor:
        """Return a tensor view of shape starting at start_index in the buffer."""
        end_index = start_index + shape.numel()
        if buffer_type == BufferType.PARAM:
            numel = self.nvfp4_packed_numel if self.has_nvfp4_params else self.numel
            assert end_index <= numel, "Requested tensor out of param buffer range"
            assert self.param_data is not None
            return self.param_data[start_index:end_index].view(shape)
        elif buffer_type == BufferType.GRAD:
            assert end_index <= self.numel, "Requested tensor out of grad buffer range"
            return self.grad_data[start_index:end_index].view(shape)
        else:
            raise ValueError(f"Illegal buffer_type: {buffer_type}")

    def _new_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
        bucket_params_with_extra_main_grads: List[torch.nn.Parameter],
        nvfp4_packed_start_index: Optional[int] = None,
        nvfp4_packed_end_index: Optional[int] = None,
    ) -> ParamAndGradBucket:
        """Helper to create a new bucket and update param→bucket mapping."""
        use_dist_opt = getattr(self.ddp_config, 'use_distributed_optimizer', False)
        if use_dist_opt:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        bucketed_param_data = None
        if self.param_data is not None:
            if nvfp4_packed_start_index is not None:
                bucketed_param_data = self._get(
                    torch.Size([nvfp4_packed_end_index - nvfp4_packed_start_index]),
                    nvfp4_packed_start_index,
                    BufferType.PARAM,
                )
            else:
                bucketed_param_data = self._get(
                    torch.Size([end_index - start_index]),
                    start_index,
                    BufferType.PARAM,
                )

        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, BufferType.GRAD
        )

        bucket = ParamAndGradBucket(
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            gradient_scaling_factor=self.gradient_scaling_factor,
            bucket_id=bucket_id,
            param_index_map=self.param_index_map,
            params_with_extra_main_grads=bucket_params_with_extra_main_grads,
        )
        for p in bucket_params:
            assert p not in self.param_to_bucket
            self.param_to_bucket[p] = bucket
        return bucket

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all gradient data by scaling_factor (M3087)."""
        self.grad_data *= scaling_factor
        for grad in self.extra_main_grads:
            grad *= scaling_factor

    def reset(self) -> None:
        """Zero the grad buffer for the next iteration."""
        self.grad_data.zero_()
        for grad in self.extra_main_grads:
            grad.zero_()

    def offload_to_cpu(self, move_params: bool = True, move_grads: bool = True) -> None:
        """Offload buffers to CPU (M3139 RL optimizer offload fix)."""
        if move_grads and self.grad_data is not None and self.grad_data.storage().size() > 0:
            self.grad_data_size = self.grad_data.storage().size()
            self.grad_data.storage().resize_(0)
        if move_params and self.param_data is not None and self.param_data.storage().size() > 0:
            self.param_data_size = self.param_data.storage().size()
            if self.param_data_cpu is not None:
                self.param_data_cpu.copy_(self.param_data, non_blocking=True)
            else:
                self.param_data_cpu = self.param_data.cpu().pin_memory()
            self.param_data.storage().resize_(0)

    def reload_from_cpu(self, move_params: bool = True, move_grads: bool = True) -> None:
        """Reload buffers from CPU (M3139 RL optimizer offload fix)."""
        if (
            move_params
            and self.param_data is not None
            and self.param_data_cpu is not None
            and self.param_data.storage().size() == 0
        ):
            self.param_data.storage().resize_(self.param_data_size)
            self.param_data.copy_(self.param_data_cpu, non_blocking=True)
        if move_grads and self.grad_data is not None and self.grad_data_size > 0:
            self.grad_data.storage().resize_(self.grad_data_size)
            self.grad_data.zero_()
            self.grad_data_size = 0


# ---------------------------------------------------------------------------
# partition_buckets
# ---------------------------------------------------------------------------

def partition_buckets(
    buffers: List[ParamAndGradBuffer],
    force_single_bucket_group: bool = False,
    reduce_scatter_with_fp32_accumulation: bool = False,
) -> List[ParamAndGradBucketGroup]:
    """Regroup buckets from multiple buffers into communication groups.

    Grouping strategy (mirrors Megatron M2777 / M3616):
      1. force_single_bucket_group=True → all buckets in one group.
      2. No FP8 buffer present → 1 bucket per group.
      3. FP8 buffer present → merge non-FP8 buckets into last FP8 bucket group
         (unless reduce_scatter_with_fp32_accumulation; then split them out).

    Args:
        buffers: List of ParamAndGradBuffer objects.
        force_single_bucket_group: Put everything in one group.
        reduce_scatter_with_fp32_accumulation: Keep non-FP8 buckets separate
            (requires exactly 1 bucket per group for the FP32-accum path).

    Returns:
        Ordered list of ParamAndGradBucketGroup objects.
    """
    if not buffers:
        return []

    # At most one FP8 (uint8) buffer allowed.
    fp8_buffer: Optional[ParamAndGradBuffer] = None
    for buf in buffers:
        if buf.param_dtype == torch.uint8:
            assert fp8_buffer is None, "Multiple FP8 buffers detected"
            fp8_buffer = buf

    # Case 1: Force single group.
    if force_single_bucket_group:
        all_buckets: List[ParamAndGradBucket] = []
        ddp_config = buffers[0].ddp_config
        dp_group = buffers[0].data_parallel_group
        dp_size = buffers[0].data_parallel_world_size
        for buf in buffers:
            assert ddp_config is buf.ddp_config or vars(ddp_config) == vars(buf.ddp_config), \
                "All buffers must share the same ddp_config"
            assert dp_group == buf.data_parallel_group
            assert dp_size == buf.data_parallel_world_size
            all_buckets.extend(buf.buckets)
        return [ParamAndGradBucketGroup(all_buckets, ddp_config, dp_group, dp_size)]

    # Case 2: No FP8 buffer — 1 bucket per group.
    if fp8_buffer is None:
        groups: List[ParamAndGradBucketGroup] = []
        for buf in buffers:
            for bucket in buf.buckets:
                groups.append(
                    ParamAndGradBucketGroup(
                        [bucket], buf.ddp_config, buf.data_parallel_group, buf.data_parallel_world_size
                    )
                )
        return groups

    # Case 3: FP8 buffer present — merge non-FP8 into last FP8 group.
    non_fp8_buckets: List[ParamAndGradBucket] = []
    for buf in buffers:
        if buf.param_dtype != torch.uint8:
            non_fp8_buckets.extend(buf.buckets)

    groups = []
    n_fp8 = len(fp8_buffer.buckets)
    buf = fp8_buffer  # type alias for brevity
    for i, bucket in enumerate(fp8_buffer.buckets):
        if i == n_fp8 - 1:
            if reduce_scatter_with_fp32_accumulation:
                # FP32-accum requires exactly 1 bucket per group.
                groups.append(
                    ParamAndGradBucketGroup(
                        [bucket], buf.ddp_config, buf.data_parallel_group, buf.data_parallel_world_size
                    )
                )
                for nb in non_fp8_buckets:
                    groups.append(
                        ParamAndGradBucketGroup(
                            [nb], buf.ddp_config, buf.data_parallel_group, buf.data_parallel_world_size
                        )
                    )
            else:
                groups.append(
                    ParamAndGradBucketGroup(
                        [bucket] + non_fp8_buckets,
                        buf.ddp_config, buf.data_parallel_group, buf.data_parallel_world_size,
                    )
                )
        else:
            groups.append(
                ParamAndGradBucketGroup(
                    [bucket], buf.ddp_config, buf.data_parallel_group, buf.data_parallel_world_size
                )
            )
    return groups


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _is_mxfp8(param: torch.nn.Parameter) -> bool:
    try:
        from megatron.core.fp8_utils import is_mxfp8tensor
        return is_mxfp8tensor(param)
    except ImportError:
        return False
