# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed data parallelism with DES-LOC support."""

from __future__ import annotations

import contextlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.desloc_config import DesLocConfig
import deepspeed.core.parallel_state as parallel_state

logger = logging.getLogger(__name__)

# ---- deepspeed/core/distributed/__init__.py ----
# Re-exports all public names from submodules.


# ===========================================================================
# distributed_data_parallel_config.py
# ===========================================================================

@dataclass
class DistributedDataParallelConfig:
    """Configuration for DDP wrapper.

    Mirrors Megatron's DistributedDataParallelConfig with DES-LOC extensions.
    """

    grad_reduce_in_fp32: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    align_param_gather: bool = False
    use_distributed_optimizer: bool = False
    num_distributed_optimizer_instances: int = 1
    check_for_nan_in_grad: bool = False
    bucket_size: Optional[int] = None

    # DES-LOC: allow skipping grad sync on non-Kx steps
    allow_skip_grad_sync: bool = False


# ===========================================================================
# param_and_grad_buffer.py
# ===========================================================================

class BufferType(Enum):
    PARAM = auto()
    GRAD = auto()


class ParamAndGradBucket:
    """A contiguous buffer holding a subset of model params and their grads."""

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: torch.Tensor,
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


class ParamAndGradBucketGroup:
    """Manages multiple buckets and orchestrates grad sync / param gather.

    This is the core communication abstraction. DES-LOC hooks into
    `start_grad_sync` and `finish_grad_sync` to gate on Kx steps.
    """

    def __init__(
        self,
        buckets: List[ParamAndGradBucket],
        ddp_config: DistributedDataParallelConfig,
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
        # golden_per_param_grad_ready_counts is populated at the end of the
        # first batch and remains fixed thereafter.
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
            # Honour force_all_reduce by temporarily clearing skip_sync.
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


class ParamAndGradBuffer:
    """Top-level buffer manager. One per model chunk, holds all bucket groups."""

    def __init__(
        self,
        config: ModelParallelConfig,
        ddp_config: DistributedDataParallelConfig,
        param_to_name: Dict[torch.nn.Parameter, str],
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
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
        bucket_indices: List[tuple] = []          # (start, end) per bucket
        per_bucket_numel_unpadded: List[int] = []

        param_start = 0
        bucket_start = 0
        bucket_params: List[torch.nn.Parameter] = []
        bucket_id = 0

        # effective_bucket_size: None means one giant bucket
        eff_bsize = bucket_size  # may be None

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
        # Allocate the contiguous grad buffer (param buffer not needed here
        # since we are not using distributed optimizer in base case).
        # Use the device of the first param so that CPU-only environments
        # (tests, non-GPU servers) work correctly without CUDA.
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
        # Map param.main_grad -> contiguous grad_data slice.
        # Build bucket objects.
        # ------------------------------------------------------------------
        gradient_scaling_factor = 1.0 / self.data_parallel_world_size

        self.buckets: List[ParamAndGradBucket] = []
        self.param_to_bucket: Dict[torch.nn.Parameter, ParamAndGradBucket] = {}

        # We need to build buckets in order; iterate params in reverse again
        cur_bucket_id = -1
        _b_params: List[torch.nn.Parameter] = []

        for param in params[::-1]:
            ps, pe, bid = param_index_map[param]
            # Assign main_grad view into contiguous buffer
            param.main_grad = self.grad_data[ps:pe].view(param.data.shape)

            if bid != cur_bucket_id:
                if cur_bucket_id >= 0:
                    # Flush previous bucket
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


# ===========================================================================
# distributed_data_parallel.py
# ===========================================================================

class DistributedDataParallel(nn.Module):
    """DDP wrapper with DES-LOC Kx-gated gradient synchronization.

    Unlike torch.nn.parallel.DistributedDataParallel, this implementation:
    - Supports skipping gradient all-reduce on non-Kx steps
    - Supports per-shard param broadcast after optimizer step
    - Manages param_and_grad_buffers directly
    - Integrates with Megatron-style pipeline parallelism
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        ddp_config: DistributedDataParallelConfig,
        module: nn.Module,
        data_parallel_group: torch.distributed.ProcessGroup,
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        disable_bucketing: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.ddp_config = ddp_config
        self._module = module
        self.data_parallel_group = data_parallel_group
        self.expert_data_parallel_group = expert_data_parallel_group or data_parallel_group

        data_parallel_world_size = torch.distributed.get_world_size(
            group=data_parallel_group
        )

        # ------------------------------------------------------------------
        # Determine bucket size
        # ------------------------------------------------------------------
        if ddp_config.bucket_size is None:
            # Default: 40 M elements, scaled with DP size
            bucket_size: Optional[int] = max(40_000_000, 1_000_000 * data_parallel_world_size)
        else:
            bucket_size = ddp_config.bucket_size

        # No bucketing when overlap is disabled or explicitly requested
        if not ddp_config.overlap_grad_reduce or disable_bucketing:
            bucket_size = None

        # ------------------------------------------------------------------
        # Collect trainable parameters
        # ------------------------------------------------------------------
        self.param_to_name: Dict[torch.nn.Parameter, str] = {}
        self.params_with_grad: List[torch.nn.Parameter] = []
        regular_params: List[torch.nn.Parameter] = []
        expert_params: List[torch.nn.Parameter] = []

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            param.grad_added_to_main_grad = False
            self.param_to_name[param] = name
            self.params_with_grad.append(param)

            is_expert = not getattr(param, 'allreduce', True)
            if is_expert:
                expert_params.append(param)
            else:
                regular_params.append(param)

        # ------------------------------------------------------------------
        # Dtype selection
        # ------------------------------------------------------------------
        param_dtype = config.params_dtype
        grad_dtype = torch.float32 if ddp_config.grad_reduce_in_fp32 else param_dtype

        # ------------------------------------------------------------------
        # Allocate buffers
        # ------------------------------------------------------------------
        self.buffers: List[ParamAndGradBuffer] = []
        self.expert_parallel_buffers: List[ParamAndGradBuffer] = []

        if regular_params:
            buf = ParamAndGradBuffer(
                config=config,
                ddp_config=ddp_config,
                param_to_name=self.param_to_name,
                params=regular_params,
                data_parallel_group=data_parallel_group,
                bucket_size=bucket_size,
                param_dtype=param_dtype,
                grad_dtype=grad_dtype,
            )
            self.buffers.append(buf)

        if expert_params:
            expert_buf = ParamAndGradBuffer(
                config=config,
                ddp_config=ddp_config,
                param_to_name=self.param_to_name,
                params=expert_params,
                data_parallel_group=self.expert_data_parallel_group,
                bucket_size=bucket_size,
                param_dtype=param_dtype,
                grad_dtype=grad_dtype,
            )
            self.expert_parallel_buffers.append(expert_buf)

        # ------------------------------------------------------------------
        # Collect all bucket groups from all buffers
        # ------------------------------------------------------------------
        self.bucket_groups: List[ParamAndGradBucketGroup] = []
        self.expert_parallel_bucket_groups: List[ParamAndGradBucketGroup] = []

        for buf in self.buffers:
            self.bucket_groups.extend(buf.bucket_groups)
        for buf in self.expert_parallel_buffers:
            self.expert_parallel_bucket_groups.extend(buf.bucket_groups)

        # ------------------------------------------------------------------
        # param -> bucket_group index for backward hook
        # ------------------------------------------------------------------
        self.param_to_bucket_group: Dict[
            torch.nn.Parameter, ParamAndGradBucketGroup
        ] = {}
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            for bucket in bg.buckets:
                for param in bucket.params_list:
                    self.param_to_bucket_group[param] = bg

        # ------------------------------------------------------------------
        # Chain bucket groups for async param-gather overlap
        # ------------------------------------------------------------------
        if ddp_config.overlap_param_gather:
            for all_bgs in [self.bucket_groups, self.expert_parallel_bucket_groups]:
                n = len(all_bgs)
                for i in range(1, n):
                    all_bgs[n - i].next_param_gather_bucket_group = all_bgs[n - i - 1]

        # ------------------------------------------------------------------
        # Register backward hooks
        # ------------------------------------------------------------------
        self.grad_accs = []
        for param in module.parameters():
            if not param.requires_grad:
                continue
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(self._make_backward_post_hook(param))
            self.grad_accs.append(grad_acc)

    # ------------------------------------------------------------------
    # Properties / forward
    # ------------------------------------------------------------------

    @property
    def module(self) -> nn.Module:
        return self._module

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    # ------------------------------------------------------------------
    # Backward hook
    # ------------------------------------------------------------------

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        def hook(*unused):
            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert param.grad is not None, \
                        "param.grad is None but overlap_grad_reduce is True"
                if param.grad is not None and not param.grad_added_to_main_grad:
                    param.main_grad.add_(param.grad.data)
                param.grad = None
                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(param)
        return hook

    # ------------------------------------------------------------------
    # Grad sync
    # ------------------------------------------------------------------

    def start_grad_sync(self, skip_sync: bool = False) -> None:
        """Start gradient synchronization.

        Args:
            skip_sync: On non-Kx DES-LOC steps, skip the collective but
                       keep local gradient accumulation.
        """
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            bg.start_grad_sync(skip_sync=skip_sync)

    def finish_grad_sync(self, force_all_reduce: bool = False) -> None:
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            bg.finish_grad_sync(force_all_reduce=force_all_reduce)

    # ------------------------------------------------------------------
    # Param sync
    # ------------------------------------------------------------------

    def start_param_sync(self, force_sync: bool = False) -> None:
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            bg.start_param_sync(force_sync=force_sync)

    def broadcast_params(self) -> None:
        """Broadcast all parameters from rank 0 of each shard to all ranks.

        Called every step in DES-LOC to prevent the Kx spike bug
        (ZeRO-3 shard inconsistency).
        """
        for param in self._module.parameters():
            is_expert = not getattr(param, 'allreduce', True)
            dp_group = self.expert_data_parallel_group if is_expert \
                else self.data_parallel_group
            src_rank = torch.distributed.get_global_rank(dp_group, 0)
            torch.distributed.broadcast(
                param.data,
                src=src_rank,
                group=dp_group,
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def zero_grad_buffer(self) -> None:
        for param in self.params_with_grad:
            param.grad_added_to_main_grad = False
        for buf in self.buffers + self.expert_parallel_buffers:
            buf.reset()

    def scale_gradients(self, scaling_factor: float) -> None:
        for buf in self.buffers + self.expert_parallel_buffers:
            buf.scale_gradients(scaling_factor)

    @contextmanager
    def no_sync(self):
        """Context manager to disable gradient sync (used in gradient accumulation)."""
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            bg.is_last_microbatch = False
        try:
            yield
        finally:
            for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
                bg.is_last_microbatch = True


# ===========================================================================
# finalize_model_grads.py
# ===========================================================================

def _get_main_grad_attr(param: torch.nn.Parameter) -> str:
    """Return 'main_grad' if present, else 'grad'."""
    if hasattr(param, 'main_grad'):
        return 'main_grad'
    return 'grad'


def _allreduce_word_embedding_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
) -> None:
    """All-reduce word-embedding gradients across first and last PP stages.

    When the embedding table is shared between the input embedding and the
    output linear (lm_head), its gradient must be summed across the first
    and last pipeline stages so that both copies stay in sync.
    """
    if not parallel_state.is_initialized():
        return

    pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pp_world_size <= 1:
        return

    is_first = parallel_state.is_pipeline_first_stage()
    is_last = parallel_state.is_pipeline_last_stage()

    if not (is_first or is_last):
        return

    # Collect embedding params that are shared across PP stages
    # (marked by having attribute `pipeline_parallel=True` OR by being a
    # word-embedding weight on the first/last stage).
    grads: List[torch.Tensor] = []
    for model_chunk in model:
        # Try to find the shared embedding weight through common attribute names
        shared_emb_weight = None
        for attr in ('word_embeddings', 'shared_embedding', 'embed_tokens'):
            module = getattr(model_chunk, attr, None)
            if module is None:
                # Try nested: model_chunk.model.embed_tokens etc.
                try:
                    for sub in model_chunk.modules():
                        candidate = getattr(sub, attr, None)
                        if candidate is not None and hasattr(candidate, 'weight'):
                            module = candidate
                            break
                except Exception:
                    pass
            if module is not None and hasattr(module, 'weight'):
                shared_emb_weight = module.weight
                break

        if shared_emb_weight is None:
            continue
        if not shared_emb_weight.requires_grad:
            continue
        grad_attr = _get_main_grad_attr(shared_emb_weight)
        grad = getattr(shared_emb_weight, grad_attr, None)
        if grad is not None:
            grads.append(grad)

    if not grads:
        return

    pp_group = parallel_state.get_pipeline_model_parallel_group()
    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=pp_group)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


def _allreduce_sequence_parallel_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
) -> None:
    """All-reduce grads for params that use sequence parallelism across TP ranks.

    Params with ``param.sequence_parallel = True`` receive partial gradients
    from each TP rank (because activations are sharded across the sequence
    dimension).  We must sum them across the TP group before the DP all-reduce
    so that every replica sees the correct full-sequence gradient.
    """
    if not parallel_state.is_initialized():
        return
    if not config.sequence_parallel:
        return

    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_world_size <= 1:
        return

    tp_group = parallel_state.get_tensor_model_parallel_group()
    grads: List[torch.Tensor] = []

    for model_chunk in model:
        inner = getattr(model_chunk, '_module', model_chunk)
        for param in inner.parameters():
            if not param.requires_grad:
                continue
            if not getattr(param, 'sequence_parallel', False):
                continue
            grad_attr = _get_main_grad_attr(param)
            grad = getattr(param, grad_attr, None)
            if grad is not None:
                grads.append(grad.data)

    if not grads:
        return

    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=tp_group)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


def finalize_model_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
    num_tokens: Optional[torch.Tensor] = None,
    skip_grad_sync: bool = False,
) -> None:
    """Finalize gradients before optimizer step.

    Handles:
    - All-reduce of embedding grads across PP stages
    - All-reduce of sequence-parallel grads across TP ranks
    - Scaling by number of tokens (per-token loss)
    - DES-LOC: skip grad all-reduce on non-Kx steps when skip_grad_sync=True

    Args:
        model: List of model chunks (for virtual pipeline parallelism).
        config: Model parallel config.
        num_tokens: Token count for per-token loss scaling.
        skip_grad_sync: Skip the main gradient all-reduce (DES-LOC non-Kx step).
    """
    # ------------------------------------------------------------------
    # 1. Main DP all-reduce / reduce-scatter across data-parallel ranks.
    #    On non-Kx steps (skip_grad_sync=True) we skip the collective but
    #    must still call finish_grad_sync so that each DDP module can clean
    #    up its bookkeeping state.
    # ------------------------------------------------------------------
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )

    for model_chunk in model:
        if isinstance(model_chunk, DistributedDataParallel):
            if skip_grad_sync:
                # Tell each bucket group to skip the collective.
                for bg in (model_chunk.bucket_groups +
                           model_chunk.expert_parallel_bucket_groups):
                    bg._skip_sync = True
            # finish_grad_sync respects _skip_sync internally.
            model_chunk.finish_grad_sync(force_all_reduce=False)
        else:
            # Raw nn.Module or wrapped model without DDP — fall back to
            # direct parameter iteration (no-op when skip requested).
            if not skip_grad_sync:
                _direct_allreduce_grads(model_chunk, config)

    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # ------------------------------------------------------------------
    # 2. All-reduce embedding grads across PP stages.
    #    This is always required regardless of skip_grad_sync because the
    #    embedding params on the first and last PP stages must stay in sync.
    # ------------------------------------------------------------------
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )
    _allreduce_word_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    # ------------------------------------------------------------------
    # 3. All-reduce sequence-parallel grads across TP ranks.
    # ------------------------------------------------------------------
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )
    _allreduce_sequence_parallel_grads(model, config)
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce').stop()

    # ------------------------------------------------------------------
    # 4. Per-token loss normalization: scale gradients by 1/num_tokens.
    #    num_tokens lives on the last PP stage; broadcast to all stages,
    #    then all-reduce across DP ranks so every rank uses the global count.
    # ------------------------------------------------------------------
    if num_tokens is not None:
        if parallel_state.is_initialized():
            pp_group = parallel_state.get_pipeline_model_parallel_group()
            last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
            torch.distributed.broadcast(num_tokens, src=last_rank, group=pp_group)

            # Prefer DP-with-CP group; fall back to plain DP when CP is not configured.
            try:
                dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
            except AssertionError:
                dp_group = parallel_state.get_data_parallel_group(with_context_parallel=False)
            torch.distributed.all_reduce(num_tokens, group=dp_group)

        safe_num_tokens = torch.clamp(num_tokens, min=1)
        scaling = 1.0 / safe_num_tokens

        for model_chunk in model:
            if isinstance(model_chunk, DistributedDataParallel):
                model_chunk.scale_gradients(scaling)
            else:
                for param in model_chunk.parameters():
                    if not param.requires_grad:
                        continue
                    grad_attr = _get_main_grad_attr(param)
                    grad = getattr(param, grad_attr, None)
                    if grad is not None:
                        grad.mul_(scaling)


def _direct_allreduce_grads(
    model_chunk: nn.Module,
    config: ModelParallelConfig,
) -> None:
    """Fallback: directly all-reduce grads on a raw nn.Module (no DDP buffer)."""
    if not parallel_state.is_initialized():
        return
    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    if torch.distributed.get_world_size(group=dp_group) <= 1:
        return

    inner = getattr(model_chunk, '_module', model_chunk)
    grads: List[torch.Tensor] = []
    for param in inner.parameters():
        if not param.requires_grad:
            continue
        grad_attr = _get_main_grad_attr(param)
        grad = getattr(param, grad_attr, None)
        if grad is not None:
            grads.append(grad.data)

    if not grads:
        return

    dp_world_size = torch.distributed.get_world_size(group=dp_group)
    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=dp_group)
    coalesced.div_(dp_world_size)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)
