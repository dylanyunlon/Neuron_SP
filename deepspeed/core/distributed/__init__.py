# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed data parallelism with DES-LOC support.

Re-exports all public names from submodules so that callers can use:

    from deepspeed.core.distributed import DistributedDataParallel, finalize_model_grads

without knowing the internal file layout.
"""

from deepspeed.core.distributed.param_and_grad_buffer import (
    BufferType,
    ParamAndGradBucket,
    ParamAndGradBucketGroup,
    ParamAndGradBuffer,
)

from deepspeed.core.distributed.distributed_data_parallel import (
    DistributedDataParallelConfig,
    DistributedDataParallel,
)

from deepspeed.core.distributed.finalize_model_grads import (
    finalize_model_grads,
    # Private helpers re-exported for test introspection
    _get_main_grad_attr,
    _allreduce_word_embedding_grads,
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

__all__ = [
    # param_and_grad_buffer
    "BufferType",
    "ParamAndGradBucket",
    "ParamAndGradBucketGroup",
    "ParamAndGradBuffer",
    # distributed_data_parallel
    "DistributedDataParallelConfig",
    "DistributedDataParallel",
    # finalize_model_grads
    "finalize_model_grads",
    "_get_main_grad_attr",
    "_allreduce_word_embedding_grads",
    "_allreduce_non_tensor_model_parallel_grads",   # was _allreduce_sequence_parallel_grads
    "_direct_allreduce_grads",
    "_desloc_should_sync_grads",
    "_desloc_sync_optimizer_moments",
    "_update_router_expert_bias",                   # M3981
    "reset_model_temporary_tensors",
]
