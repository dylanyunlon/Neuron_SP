# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Contiguous parameter and gradient buffers for DDP with DES-LOC support.

Extracted from the monolithic __init__.py; provides:
  - BufferType
  - ParamAndGradBucket
  - ParamAndGradBucketGroup
  - ParamAndGradBuffer
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Dict, List, Optional

import torch

from deepspeed.core.model_parallel_config import ModelParallelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BufferType
# ---------------------------------------------------------------------------

class BufferType(Enum):
    PARAM = auto()
    GRAD = auto()


# ---------------------------------------------------------------------------
# ParamAndGradBucket
# ---------------------------------------------------------------------------

class ParamAndGradBucket:
    """A contiguous buffer holding a subset of model params and their grads."""

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        gradient_scaling_factor: float,
        communication_dtype: torch.dtype,
    ) -> None:
        self.params_list = params
        self.params = set(params)
        assert len(self.params_list) == len(self.params), \
            "Duplicate params detected in bucket"

        self.param_data = param_data
        self.grad_data = grad_data
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.gradient_scaling_factor = gradient_scaling_factor
        self.communication_dtype = communication_dtype

        # Map each param to its (bucket-local) start/end offsets in grad_data
        self.param_to_index: Dict[torch.nn.Parameter, tuple] = {}


# ---------------------------------------------------------------------------
# ParamAndGradBucketGroup
# ---------------------------------------------------------------------------

class ParamAndGradBucketGroup:
    """Manages multiple buckets and orchestrates grad sync / param gather.

    This is the core communication abstraction. DES-LOC hooks into
    ``start_grad_sync`` and ``finish_grad_sync`` to gate on Kx steps.
    """

    def __init__(
        self,
        buckets: List[ParamAndGradBucket],
        ddp_config,  # DistributedDataParallelConfig — imported by caller
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
    ) -> None:
        self.buckets = buckets
        self.ddp_config = ddp_config
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size

        # Build param -> bucket mapping and full param set
        self.param_to_bucket: Dict[torch.nn.Parameter, ParamAndGradBucket] = {}
        self.params: set = set()
        for bucket in self.buckets:
            for param in bucket.params_list:
                self.param_to_bucket[param] = bucket
                self.params.add(param)

        # Overlapping grad-reduce bookkeeping
        self.golden_per_param_grad_ready_counts: Dict[torch.nn.Parameter, int] = {}
        self.per_param_grad_ready_counts: Dict[torch.nn.Parameter, int] = {}
        self.is_last_microbatch: bool = True
        self.is_first_batch: bool = True

        # Async handles
        self.grad_reduce_handle = None
        self.grad_reduce_finished: bool = False
        self.param_gather_handle = None
        self.param_gather_dispatched: bool = False

        # DES-LOC: track whether the current step should skip the all-reduce
        self._skip_sync: bool = False

        # Chaining for overlap: filled by DDP __init__
        self.next_param_gather_bucket_group: Optional[ParamAndGradBucketGroup] = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset metadata in bucket group for the next training iteration."""
        if self.is_first_batch and len(self.per_param_grad_ready_counts) > 0:
            assert len(self.per_param_grad_ready_counts) == len(self.params)
            self.golden_per_param_grad_ready_counts = self.per_param_grad_ready_counts
            self.is_first_batch = False
        self.per_param_grad_ready_counts = {}
        self.is_last_microbatch = True
        self.grad_reduce_finished = False
        self._skip_sync = False

    # ------------------------------------------------------------------
    # Param sync (all-gather for distributed optimizer)
    # ------------------------------------------------------------------

    def start_param_sync(self, force_sync: bool = False) -> None:
        """Initiate param all-gather for this bucket group.

        For the basic (non-distributed-optimizer) DDP case used in Neuron_SP,
        parameters live in their original locations and no all-gather is
        needed.  We keep the method present for API compatibility with the
        optimizer and pipeline code that may call it.
        """
        # No-op in the base (non-ZeRO-3) case.
        self.param_gather_dispatched = True

    def finish_param_sync(self) -> None:
        """Wait for any outstanding param all-gather."""
        # No-op in the base case.
        self.param_gather_dispatched = False

    # ------------------------------------------------------------------
    # Grad sync
    # ------------------------------------------------------------------

    def _scale_grads(self) -> None:
        """Apply gradient_scaling_factor to all buckets before collective."""
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data.mul_(bucket.gradient_scaling_factor)

    def _nan_check(self) -> None:
        """Abort (via assertion) if any grad bucket contains NaN/Inf."""
        for i, bucket in enumerate(self.buckets):
            norm = bucket.grad_data.norm(p=2)
            assert not torch.isnan(norm).item(), \
                f"NaN in grad norm for bucket #{i} before all-reduce"
            assert not torch.isinf(norm).item(), \
                f"Inf in grad norm for bucket #{i} before all-reduce"

    def start_grad_sync(self, skip_sync: bool = False) -> None:
        """Start gradient synchronization.

        Args:
            skip_sync: If True (non-Kx step in DES-LOC), skip the all-reduce
                       but still accumulate locally.
        """
        # Guard: in overlapped mode, first-batch calls after one dispatch are no-ops.
        if self.is_first_batch and self.grad_reduce_handle is not None:
            return

        assert self.grad_reduce_handle is None, \
            "start_grad_sync called while a previous collective is outstanding"

        # Record the skip flag for finish_grad_sync to honour.
        self._skip_sync = skip_sync

        # When DES-LOC requests a skip, do not launch any collective.
        if skip_sync and self.ddp_config.allow_skip_grad_sync:
            # Gradients remain locally accumulated; mark as "finished"
            # so finish_grad_sync does not try to wait on a None handle.
            self.grad_reduce_finished = True
            return

        # Optionally abort on NaN/Inf.
        if self.ddp_config.check_for_nan_in_grad:
            self._nan_check()

        # Pre-scale gradients.
        self._scale_grads()

        reduce_op = torch.distributed.ReduceOp.SUM
        async_op = self.ddp_config.overlap_grad_reduce

        with torch.distributed._coalescing_manager(
            self.data_parallel_group, async_ops=async_op
        ) as cm:
            for bucket in self.buckets:
                torch.distributed.all_reduce(
                    bucket.grad_data,
                    op=reduce_op,
                    group=self.data_parallel_group,
                    async_op=async_op,
                )

        if async_op:
            self.grad_reduce_handle = cm
        else:
            self.grad_reduce_handle = None

    def finish_grad_sync(self, force_all_reduce: bool = False) -> None:
        """Complete any outstanding grad-sync collective.

        Args:
            force_all_reduce: If True, force an all-reduce even if skip_sync
                              was set (used by DES-LOC Kx step recovery).
        """
        self.param_gather_dispatched = False

        if not self.ddp_config.overlap_grad_reduce:
            # Synchronous path: dispatch + wait inline.
            skip = self._skip_sync and not force_all_reduce
            self.start_grad_sync(skip_sync=skip)
            return

        if self.grad_reduce_finished:
            return

        # First batch in overlapping mode: dispatch now if not already done.
        if self.is_first_batch:
            skip = self._skip_sync and not force_all_reduce
            self.start_grad_sync(skip_sync=skip)

        # If we skipped, there is nothing to wait on.
        if self.grad_reduce_handle is None and self._skip_sync and \
                not force_all_reduce:
            self.grad_reduce_finished = True
            return

        assert self.grad_reduce_handle is not None, (
            f"No outstanding communication handle "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
            "params have grad available)"
        )
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None
        self.grad_reduce_finished = True

    def register_grad_ready(self, param: torch.nn.Parameter) -> None:
        """Register that a param's grad is ready; trigger async collective when
        all grads in the bucket group are ready (overlap mode only)."""
        assert self.ddp_config.overlap_grad_reduce, \
            "register_grad_ready should only be called when overlap_grad_reduce is True"

        if not self.is_last_microbatch:
            return

        assert param in self.param_to_bucket, "Param not in this bucket group"
        self.per_param_grad_ready_counts[param] = \
            self.per_param_grad_ready_counts.get(param, 0) + 1

        if not self.is_first_batch:
            if self.per_param_grad_ready_counts == \
                    self.golden_per_param_grad_ready_counts:
                assert len(self.per_param_grad_ready_counts) == len(self.params)
                self.start_grad_sync(skip_sync=self._skip_sync)

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all grad_data tensors in this bucket group."""
        for bucket in self.buckets:
            bucket.grad_data.mul_(scaling_factor)


# ---------------------------------------------------------------------------
# ParamAndGradBuffer
# ---------------------------------------------------------------------------

class ParamAndGradBuffer:
    """Top-level buffer manager. One per model chunk, holds all bucket groups."""

    def __init__(
        self,
        config: ModelParallelConfig,
        ddp_config,  # DistributedDataParallelConfig
        param_to_name: Dict[torch.nn.Parameter, str],
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: Optional[int],
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.ddp_config = ddp_config
        self.param_to_name = param_to_name
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = torch.distributed.get_world_size(
            group=data_parallel_group
        )

        # Validate inputs
        assert len(params) > 0, "ParamAndGradBuffer requires at least one param"
        for p in params:
            assert p.requires_grad, "All params must require grad"

        # ------------------------------------------------------------------
        # Layout: iterate params in REVERSE order (back-prop order) and pack
        # into buckets of ~bucket_size elements.
        # ------------------------------------------------------------------
        param_index_map: Dict[torch.nn.Parameter, tuple] = {}
        bucket_indices: List[tuple] = []
        per_bucket_numel_unpadded: List[int] = []

        param_start = 0
        bucket_start = 0
        bucket_params: List[torch.nn.Parameter] = []
        bucket_id = 0
        eff_bsize = bucket_size  # None → one giant bucket

        for param in params[::-1]:
            numel = param.data.numel()
            param_end = param_start + numel
            param_index_map[param] = (param_start, param_end, bucket_id)
            bucket_params.append(param)

            if eff_bsize is not None and (param_end - bucket_start) >= eff_bsize:
                per_bucket_numel_unpadded.append(param_end - bucket_start)
                bucket_indices.append((bucket_start, param_end))
                bucket_start = param_end
                bucket_params = []
                bucket_id += 1

            param_start = param_end

        # Last (possibly only) bucket
        if bucket_params:
            per_bucket_numel_unpadded.append(param_end - bucket_start)
            bucket_indices.append((bucket_start, param_end))

        self.param_index_map = param_index_map
        self.bucket_indices = bucket_indices

        # ------------------------------------------------------------------
        # Allocate the contiguous grad buffer.
        # ------------------------------------------------------------------
        total_numel = bucket_indices[-1][1]
        _device = params[0].device if params else torch.device("cpu")
        try:
            _cuda_dev = torch.cuda.current_device()
            _device = torch.device("cuda", _cuda_dev)
        except (AssertionError, RuntimeError):
            pass  # No CUDA; stay on params[0].device

        self.grad_data = torch.zeros(
            total_numel,
            dtype=grad_dtype,
            device=_device,
            requires_grad=False,
        )
        self.param_data = None  # No param remapping in non-ZeRO mode

        # ------------------------------------------------------------------
        # Build bucket objects.
        # ------------------------------------------------------------------
        gradient_scaling_factor = 1.0 / self.data_parallel_world_size

        self.buckets: List[ParamAndGradBucket] = []
        self.param_to_bucket: Dict[torch.nn.Parameter, ParamAndGradBucket] = {}

        cur_bucket_id = -1
        _b_params: List[torch.nn.Parameter] = []

        for param in params[::-1]:
            ps, pe, bid = param_index_map[param]
            # Assign main_grad view into contiguous buffer
            param.main_grad = self.grad_data[ps:pe].view(param.data.shape)

            if bid != cur_bucket_id:
                if cur_bucket_id >= 0:
                    self.buckets.append(self._make_bucket(
                        cur_bucket_id, _b_params, bucket_indices,
                        per_bucket_numel_unpadded, gradient_scaling_factor, grad_dtype
                    ))
                _b_params = []
                cur_bucket_id = bid

            _b_params.append(param)

        # Flush last bucket
        if _b_params:
            self.buckets.append(self._make_bucket(
                cur_bucket_id, _b_params, bucket_indices,
                per_bucket_numel_unpadded, gradient_scaling_factor, grad_dtype
            ))

        for bucket in self.buckets:
            for p in bucket.params_list:
                self.param_to_bucket[p] = bucket

        # Build one bucket group per bucket (simple 1:1 mapping)
        self.bucket_groups: List[ParamAndGradBucketGroup] = [
            ParamAndGradBucketGroup(
                buckets=[bucket],
                ddp_config=ddp_config,
                data_parallel_group=data_parallel_group,
                data_parallel_world_size=self.data_parallel_world_size,
            )
            for bucket in self.buckets
        ]

        # Log bucket layout
        logger.info(
            "ParamAndGradBuffer: %d bucket(s), total numel=%d",
            len(self.buckets), total_numel,
        )
        for idx, bucket in enumerate(self.buckets):
            n = sum(p.data.numel() for p in bucket.params_list)
            logger.debug(
                "  Bucket %d: %d params, %d elements",
                idx, len(bucket.params_list), n,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_bucket(
        self,
        bucket_id: int,
        bucket_params: List[torch.nn.Parameter],
        bucket_indices: List[tuple],
        per_bucket_numel_unpadded: List[int],
        gradient_scaling_factor: float,
        grad_dtype: torch.dtype,
    ) -> ParamAndGradBucket:
        start, end = bucket_indices[bucket_id]
        grad_view = self.grad_data[start:end]
        bucket = ParamAndGradBucket(
            params=list(bucket_params),
            param_data=None,
            grad_data=grad_view,
            offset=start,
            numel_unpadded=per_bucket_numel_unpadded[bucket_id],
            gradient_scaling_factor=gradient_scaling_factor,
            communication_dtype=grad_dtype,
        )
        return bucket

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all gradient data by scaling_factor."""
        self.grad_data.mul_(scaling_factor)

    def reset(self) -> None:
        """Zero the grad buffer and reset per-bucket-group state."""
        self.grad_data.zero_()
        for bg in self.bucket_groups:
            bg.reset()

    def zero_grad_buffer(self) -> None:
        """Zero only the grad data (alias used by DDP)."""
        self.grad_data.zero_()
