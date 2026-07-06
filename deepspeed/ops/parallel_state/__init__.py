# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""deepspeed.ops.parallel_state
================================

C++ NCCL bootstrap extension for bulk communicator creation (issue #139).

Usage::

    from deepspeed.ops.parallel_state import NcclBootstrapOp

    op   = NcclBootstrapOp()                   # lazy-load on first call
    uid  = op.get_unique_id()                  # rank 0 only; broadcast to all
    # ... dist.broadcast(uid, src=0) ...
    handles = op.bulk_create_comms(
        rank_lists=[[0,1,2,3], [0,1], [2,3]],  # TP, DP, PP groups
        this_rank=dist.get_rank(),
        nccl_id_data=uid,
    )
    tp_handle, dp_handle, pp_handle = handles
"""

from .nccl_bootstrap_op import NcclBootstrapOp

__all__ = ["NcclBootstrapOp"]
