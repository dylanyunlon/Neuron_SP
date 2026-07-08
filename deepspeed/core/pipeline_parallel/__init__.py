# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Pipeline parallelism public API.

This package is split into three sub-modules:

``p2p_communication.py``
    Point-to-point send/recv for activations and gradients between adjacent
    pipeline stages.  Mirrors Megatron's P2PCommunicator with DES-LOC
    extensions for heterogeneous PCIe-only clusters.

``schedules.py``
    1F1B and interleaved 1F1B (VPP) pipeline schedules.  Includes the
    DES-LOC ``HeterogeneousBubbleFiller`` that keeps fast stages (H100)
    busy during the bubbles imposed by slow stages (A6000).

``__init__.py`` (this file)
    Re-exports the complete public API so that callers can do either::

        from deepspeed.core.pipeline_parallel import (
            forward_backward_pipelining_without_interleaving,
        )

    or access the sub-modules directly::

        from deepspeed.core.pipeline_parallel.schedules import (
            forward_backward_pipelining_without_interleaving,
        )

Guaranteed export (required by task spec):
    from deepspeed.core.pipeline_parallel import (
        forward_backward_pipelining_without_interleaving
    )
"""

# ---------------------------------------------------------------------------
# p2p_communication re-exports
# ---------------------------------------------------------------------------
from deepspeed.core.pipeline_parallel.p2p_communication import (
    P2PCommunicator,
    is_single_shape,
    is_cross_numa_transfer,
    get_numa_node_for_rank,
    _batched_p2p_ops,
    _p2p_ops,
)

# ---------------------------------------------------------------------------
# schedules re-exports
# ---------------------------------------------------------------------------
from deepspeed.core.pipeline_parallel.schedules import (
    # Schedule selector
    get_forward_backward_func,
    # Step functions
    forward_step,
    backward_step,
    # Schedule functions
    forward_backward_no_pipelining,
    forward_backward_pipelining_without_interleaving,   # 1F1B — required export
    forward_backward_pipelining_with_interleaving,
    # DES-LOC heterogeneous schedules
    forward_backward_hetero_1f1b,
    forward_backward_pipelining_without_interleaving_pp5_heterogeneous,
    forward_backward_hetero_1f1b,
    # DES-LOC bubble filling (full implementation, replaces stub)
    StageClock,
    AsymmetricClockScheduler,
    HeterogeneousBubbleFiller,
    # DES-LOC PCIe-aware P2P bandwidth manager
    HeterogeneousP2PManager,
    # DES-LOC PP=5 layout constants and factory helpers
    PP5_DESLOC_FAST_RANKS,
    PP5_DESLOC_SLOW_RANKS,
    make_pp5_bubble_filler,
    make_pp5_p2p_manager,
    # Utilities
    get_tensor_shapes,
    deallocate_output_tensor,
    custom_backward,
    get_num_microbatches,
    get_pipeline_model_parallel_rank_for_layer,
    set_pipeline_layer_split,
)

# ---------------------------------------------------------------------------
# utils re-exports
# ---------------------------------------------------------------------------
from deepspeed.core.pipeline_parallel.utils import (
    # PP stage predicates
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
    # Rank lookup
    get_pp_first_rank,
    get_pp_last_rank,
    get_pp_next_rank,
    get_pp_prev_rank,
    # Tensor utilities
    make_viewless,
    # Fine-grained scheduling
    NoopScheduleNode,
    ScheduleNode,
    AbstractSchedulePlan,
    # Stream management
    set_streams,
    get_comp_stream,
    get_comm_stream,
    # DES-LOC tier helpers
    get_tier_for_rank,
    tier_priority_stream,
    # DES-LOC stage assignment + micro-batch sizing
    get_pp_stage_compute_factor,
    is_fast_stage,
    is_slow_stage,
    optimal_pp_stage_assignment,
    get_pp_stage_micro_batch_size,
)

# ---------------------------------------------------------------------------
# bridge_communicator re-exports
# ---------------------------------------------------------------------------
from deepspeed.core.pipeline_parallel.bridge_communicator import (
    BridgeCommunicator,
    CommRole,
    RankCommInfo,
)

# ---------------------------------------------------------------------------
# multimodule_communicator re-exports
# ---------------------------------------------------------------------------
from deepspeed.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
    RankModuleInfo,
)

__all__ = [
    # P2P communication
    "P2PCommunicator",
    "is_single_shape",
    "is_cross_numa_transfer",
    "get_numa_node_for_rank",
    "_batched_p2p_ops",
    "_p2p_ops",
    # Schedule selector
    "get_forward_backward_func",
    # Step functions
    "forward_step",
    "backward_step",
    # Standard schedules
    "forward_backward_no_pipelining",
    "forward_backward_pipelining_without_interleaving",
    "forward_backward_pipelining_with_interleaving",
    # DES-LOC heterogeneous schedules
    "forward_backward_hetero_1f1b",
    "forward_backward_pipelining_without_interleaving_pp5_heterogeneous",
    "forward_backward_hetero_1f1b",
    # DES-LOC bubble filling (full implementation)
    "StageClock",
    "AsymmetricClockScheduler",
    "HeterogeneousBubbleFiller",
    # DES-LOC PCIe-aware P2P manager
    "HeterogeneousP2PManager",
    # DES-LOC PP=5 layout constants and factories
    "PP5_DESLOC_FAST_RANKS",
    "PP5_DESLOC_SLOW_RANKS",
    "make_pp5_bubble_filler",
    "make_pp5_p2p_manager",
    # Utilities (from schedules.py)
    "get_tensor_shapes",
    "deallocate_output_tensor",
    "custom_backward",
    "get_num_microbatches",
    "get_pipeline_model_parallel_rank_for_layer",
    "set_pipeline_layer_split",
    # Utilities (from utils.py)
    "is_pp_first_stage",
    "is_pp_last_stage",
    "is_vp_first_stage",
    "is_vp_last_stage",
    "get_pp_first_rank",
    "get_pp_last_rank",
    "get_pp_next_rank",
    "get_pp_prev_rank",
    "make_viewless",
    "NoopScheduleNode",
    "ScheduleNode",
    "AbstractSchedulePlan",
    "set_streams",
    "get_comp_stream",
    "get_comm_stream",
    "get_tier_for_rank",
    "tier_priority_stream",
    "get_pp_stage_compute_factor",
    "is_fast_stage",
    "is_slow_stage",
    "optimal_pp_stage_assignment",
    "get_pp_stage_micro_batch_size",
    # Bridge communicator
    "BridgeCommunicator",
    "CommRole",
    "RankCommInfo",
    # Multi-module communicator
    "MultiModulePipelineCommunicator",
    "RankModuleInfo",
]
