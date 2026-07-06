# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed data parallelism with DES-LOC support.

Re-exports all public names from submodules so that callers can use:

    from deepspeed.core.distributed import DistributedDataParallel, finalize_model_grads

without knowing the internal file layout.
"""

from deepspeed.core.distributed.param_and_grad_buffer import (
    BufferType,
    BufferOwnership,
    shard_buffer,
    ParamAndGradBucket,
    LayerwiseAllGatherHandle,
    ParamAndGradBucketGroup,
    ParamAndGradBuffer,
    group_params_for_buffers,
    partition_buckets,
    _compute_default_per_buffer_param_layout,
    get_tier_bucket_size,
    compute_tier_bucket_sizes,
)

from deepspeed.core.distributed.reduce_scatter_with_fp32_accumulation import (
    _ReduceScatterWithFP32AccumulationWorkHandle,
    reduce_scatter_with_fp32_accumulation,
)

from deepspeed.core.distributed.finalize_model_grads import (
    finalize_model_grads,
    # PCIe-friendly grad-reduction fusion helper (M4149 pattern / DES-LOC).
    fuse_grad_reductions,
    # Fused word + position embedding AllReduce for PCIe (M4149 / DES-LOC).
    _allreduce_all_embedding_grads,
    # Private helpers re-exported for test introspection
    _get_main_grad_attr,
    _allreduce_word_embedding_grads,
    _allreduce_position_embedding_grads,
    _allreduce_conditional_embedding_grads,
    # M-rename: was _allreduce_sequence_parallel_grads in early ports;
    # renamed to _allreduce_non_tensor_model_parallel_grads to match
    # the broader scope (SUM + AVG TP-domain grads, not just SP).
    _allreduce_non_tensor_model_parallel_grads,
    _direct_allreduce_grads,
    _desloc_should_sync_grads,
    _desloc_sync_optimizer_moments,
    # M3981: MoE expert-bias grad finalization with explicit tp_dp_cp group.
    _update_router_expert_bias,
    reset_model_temporary_tensors,
)

from deepspeed.core.distributed.distributed_data_parallel import (
    DistributedDataParallelConfig,
    DistributedDataParallel,
    # M2853: checkpoint param-sync helper.
    force_param_sync,
)

__all__ = [
    # param_and_grad_buffer
    "BufferType",
    "BufferOwnership",
    "shard_buffer",
    "ParamAndGradBucket",
    "LayerwiseAllGatherHandle",
    "ParamAndGradBucketGroup",
    "ParamAndGradBuffer",
    "group_params_for_buffers",
    "partition_buckets",
    "_compute_default_per_buffer_param_layout",
    "get_tier_bucket_size",
    "compute_tier_bucket_sizes",
    # reduce_scatter_with_fp32_accumulation
    "_ReduceScatterWithFP32AccumulationWorkHandle",
    "reduce_scatter_with_fp32_accumulation",
    # distributed_data_parallel
    "DistributedDataParallelConfig",
    "DistributedDataParallel",
    "force_param_sync",
    # finalize_model_grads
    "finalize_model_grads",
    "fuse_grad_reductions",
    "_allreduce_all_embedding_grads",
    "_get_main_grad_attr",
    "_allreduce_word_embedding_grads",
    "_allreduce_position_embedding_grads",
    "_allreduce_conditional_embedding_grads",
    "_allreduce_non_tensor_model_parallel_grads",   # was _allreduce_sequence_parallel_grads
    "_direct_allreduce_grads",
    "_desloc_should_sync_grads",
    "_desloc_sync_optimizer_moments",
    "_update_router_expert_bias",                   # M3981
    "reset_model_temporary_tensors",
]
