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
    # DES-LOC
    HeterogeneousBubbleFiller,
    # Utilities
    get_tensor_shapes,
    deallocate_output_tensor,
    custom_backward,
    get_num_microbatches,
    get_pipeline_model_parallel_rank_for_layer,
    set_pipeline_layer_split,
)

__all__ = [
    # P2P communication
    "P2PCommunicator",
    "is_single_shape",
    "_batched_p2p_ops",
    "_p2p_ops",
    # Schedule selector
    "get_forward_backward_func",
    # Step functions
    "forward_step",
    "backward_step",
    # Schedules
    "forward_backward_no_pipelining",
    "forward_backward_pipelining_without_interleaving",
    "forward_backward_pipelining_with_interleaving",
    # DES-LOC bubble filling
    "HeterogeneousBubbleFiller",
    # Utilities
    "get_tensor_shapes",
    "deallocate_output_tensor",
    "custom_backward",
    "get_num_microbatches",
    "get_pipeline_model_parallel_rank_for_layer",
    "set_pipeline_layer_split",
]
