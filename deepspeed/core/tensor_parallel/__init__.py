# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tensor parallelism package.

This package mirrors the structure of Megatron-LM's
``megatron/core/tensor_parallel/`` and re-exports all public symbols from
the three sub-modules:

* ``layers.py``   — VocabParallelEmbedding, ColumnParallelLinear,
                    RowParallelLinear, TP attribute helpers
* ``random.py``   — CudaRNGStatesTracker, checkpoint, model_parallel_cuda_manual_seed
* ``mappings.py`` — scatter/gather/reduce region functions,
                    split_tensor_into_1d_equal_chunks, gather_split_1d_tensor
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# layers.py
# ---------------------------------------------------------------------------
from deepspeed.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    copy_tensor_model_parallel_attributes,
    linear_with_grad_accumulation_and_async_allreduce,
    param_is_not_tensor_parallel_duplicate,
    set_defaults_if_not_set_tensor_model_parallel_attributes,
    set_tensor_model_parallel_attributes,
)

# ---------------------------------------------------------------------------
# random.py
# ---------------------------------------------------------------------------
from deepspeed.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CudaRNGStatesTracker,
    checkpoint,
    convert_cuda_rng_state,
    get_all_rng_states,
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
    initialize_rng_tracker,
    is_checkpointing,
    is_graph_safe_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)

# ---------------------------------------------------------------------------
# mappings.py
# ---------------------------------------------------------------------------
from deepspeed.core.tensor_parallel.mappings import (
    all_gather_last_dim_from_tensor_parallel_region,
    all_to_all,
    all_to_all_hp2sp,
    all_to_all_sp2hp,
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    gather_split_1d_tensor,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_last_dim_to_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
    split_tensor_into_1d_equal_chunks,
)

__all__ = [
    # layers.py
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "copy_tensor_model_parallel_attributes",
    "linear_with_grad_accumulation_and_async_allreduce",
    "param_is_not_tensor_parallel_duplicate",
    "set_defaults_if_not_set_tensor_model_parallel_attributes",
    "set_tensor_model_parallel_attributes",
    # random.py
    "CheckpointWithoutOutput",
    "CudaRNGStatesTracker",
    "checkpoint",
    "convert_cuda_rng_state",
    "get_all_rng_states",
    "get_cuda_rng_tracker",
    "get_data_parallel_rng_tracker_name",
    "get_expert_parallel_rng_tracker_name",
    "initialize_rng_tracker",
    "is_checkpointing",
    "is_graph_safe_cuda_rng_tracker",
    "model_parallel_cuda_manual_seed",
    # mappings.py
    "all_gather_last_dim_from_tensor_parallel_region",
    "all_to_all",
    "all_to_all_hp2sp",
    "all_to_all_sp2hp",
    "copy_to_tensor_model_parallel_region",
    "gather_from_sequence_parallel_region",
    "gather_from_tensor_model_parallel_region",
    "gather_split_1d_tensor",
    "reduce_from_tensor_model_parallel_region",
    "reduce_scatter_last_dim_to_tensor_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
    "scatter_to_sequence_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    "split_tensor_into_1d_equal_chunks",
]
