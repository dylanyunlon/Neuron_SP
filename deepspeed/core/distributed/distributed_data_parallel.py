# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""DistributedDataParallel wrapper with DES-LOC Kx-gated gradient synchronization.

Provides:
  - DistributedDataParallelConfig
  - DistributedDataParallel
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.distributed.param_and_grad_buffer import (
    ParamAndGradBuffer,
    ParamAndGradBucketGroup,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DistributedDataParallelConfig
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DistributedDataParallel
# ---------------------------------------------------------------------------

class DistributedDataParallel(nn.Module):
    """DDP wrapper with DES-LOC Kx-gated gradient synchronization.

    Unlike ``torch.nn.parallel.DistributedDataParallel``, this implementation:

    - Supports skipping gradient all-reduce on non-Kx steps (DES-LOC Algorithm 1).
    - Supports per-shard param broadcast after optimizer step.
    - Manages ``param_and_grad_buffers`` directly for fine-grained control.
    - Integrates with Megatron-style pipeline parallelism.

    Args:
        config: Model parallel configuration.
        ddp_config: DDP-specific configuration (bucketing, overlap, etc.).
        module: The model to wrap.
        data_parallel_group: Process group for all-reduce across DP replicas.
        expert_data_parallel_group: Optional separate group for MoE expert params.
        disable_bucketing: If True, put all params in one bucket (no overlap).
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
            bucket_size: Optional[int] = max(
                40_000_000, 1_000_000 * data_parallel_world_size
            )
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
