# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed data parallelism with DES-LOC support.

Re-exports all public names from submodules so that callers can use:

    from deepspeed.core.distributed import DistributedDataParallel, finalize_model_grads
    from deepspeed.core.distributed import save_checkpoint, load_checkpoint, sharded_state_dict

without knowing the internal file layout.
"""

from deepspeed.core.distributed.grad_buffer import (
    GradBuffer,
    GradBufferRegistry,
    build_grad_buffer_registry,
)

from deepspeed.core.distributed.bucket_aware_grad_sync import (
    pcie_overlap_trigger_elems,
    pcie_bucket_size,
    should_use_async_op,
    BucketAwareGradSync,
    bucket_aware_grad_sync_context,
    log_bucket_sync_plan,
)

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
    # M3871: Flextron router grad sync across PP stages.
    _allreduce_router_grads,
    # M-rename: was _allreduce_sequence_parallel_grads in early ports;
    # renamed to _allreduce_non_tensor_model_parallel_grads to match
    # the broader scope (SUM + AVG TP-domain grads, not just SP).
    _allreduce_non_tensor_model_parallel_grads,
    # Legacy alias maintained for unit tests (mcore 0.14 removal target).
    _allreduce_layernorm_grads,
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

# Issue #121: ShardedTensor-based save / load for DDP models.
from deepspeed.core.distributed.sharded_checkpoint import (
    sharded_state_dict,
    save_checkpoint,
    load_checkpoint,
)

# Issue #35: tier-aware gradient bucketing with C++ fused kernel dispatch.
from deepspeed.core.distributed.tier_aware_bucketing import (
    TierAwareBucketingConfig,
    get_local_sm_version,
    get_local_tier_name,
    tier_bucket_multiplier,
    fused_allreduce_bucket,
    fused_reduce_scatter_bucket,
    TierAwareGradSyncMixin,
    attach_tier_aware_config_to_bucket_groups,
    build_tier_aware_config_from_ddp_config,
    recommend_bucket_size_for_tier,
    dp_group_min_bucket_size,
)

# Overlapped gradient reduce-scatter for pipeline-parallel stages.
from deepspeed.core.distributed.overlap_grad_reduce import (
    OverlapGradReduceManager,
    PPStageGradReduceScheduler,
    recommend_pp_stage_bucket_size,
)

# Extended overlapped grad-reduce: VPP multi-chunk + dedicated CUDA streams.
from deepspeed.core.distributed.overlapped_grad_reduce import (
    StreamPool,
    VPPOverlapGradReduceManager,
    OverlappedGradReduceContext,
    build_overlapped_grad_reduce_manager,
)

# Async parameter all-gather synchronization (overlap_param_gather / LayerWise opt).
from deepspeed.core.distributed.async_param_sync import (
    ParamSyncScheduler,
    AsyncParamSyncManager,
    async_param_sync_context,
    build_vpp_param_sync_manager,
)

__all__ = [
    # grad_buffer
    "GradBuffer",
    "GradBufferRegistry",
    "build_grad_buffer_registry",
    # bucket_aware_grad_sync
    "pcie_overlap_trigger_elems",
    "pcie_bucket_size",
    "should_use_async_op",
    "BucketAwareGradSync",
    "bucket_aware_grad_sync_context",
    "log_bucket_sync_plan",
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
    # sharded_checkpoint (issue #121)
    "sharded_state_dict",
    "save_checkpoint",
    "load_checkpoint",
    # finalize_model_grads
    "finalize_model_grads",
    "fuse_grad_reductions",
    "_allreduce_all_embedding_grads",
    "_get_main_grad_attr",
    "_allreduce_word_embedding_grads",
    "_allreduce_position_embedding_grads",
    "_allreduce_conditional_embedding_grads",
    "_allreduce_router_grads",                      # M3871 Flextron
    "_allreduce_non_tensor_model_parallel_grads",   # was _allreduce_sequence_parallel_grads
    "_allreduce_layernorm_grads",                   # legacy alias
    "_direct_allreduce_grads",
    "_desloc_should_sync_grads",
    "_desloc_sync_optimizer_moments",
    "_update_router_expert_bias",                   # M3981
    "reset_model_temporary_tensors",
    # tier_aware_bucketing (Issue #35: DDP + gradient sync + tier-aware bucketing)
    "TierAwareBucketingConfig",
    "get_local_sm_version",
    "get_local_tier_name",
    "tier_bucket_multiplier",
    "fused_allreduce_bucket",
    "fused_reduce_scatter_bucket",
    "TierAwareGradSyncMixin",
    "attach_tier_aware_config_to_bucket_groups",
    "build_tier_aware_config_from_ddp_config",
    "recommend_bucket_size_for_tier",
    "dp_group_min_bucket_size",
    # overlap_grad_reduce (pipeline-parallel overlapped reduce-scatter)
    "OverlapGradReduceManager",
    "PPStageGradReduceScheduler",
    "recommend_pp_stage_bucket_size",
    # overlapped_grad_reduce (VPP multi-chunk + dedicated CUDA streams)
    "StreamPool",
    "VPPOverlapGradReduceManager",
    "OverlappedGradReduceContext",
    "build_overlapped_grad_reduce_manager",
    # async_param_sync (overlap_param_gather / LayerWise optimizer)
    "ParamSyncScheduler",
    "AsyncParamSyncManager",
    "async_param_sync_context",
    "build_vpp_param_sync_manager",
]
