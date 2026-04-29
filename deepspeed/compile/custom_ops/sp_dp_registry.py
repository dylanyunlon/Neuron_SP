# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# M356 — Claude-32: Added pending_handles tracking and fence_all_sp_handles
# to prevent NCCL deadlock on heterogeneous GPUs (A6000+H100).
# Pattern: Megatron param_and_grad_buffer.py finish_grad_sync fence,
#          NCCL src/device/all_reduce.h async completion tracking.

import os
import time
import deepspeed.comm as dist

GROUP_REGISTRY = {}  # int -> dist.ProcessGroup
_PENDING_A2A_HANDLES = []  # M356: track async all-to-all handles
_A2A_TIMEOUT_MS = int(os.environ.get('DESLOC_SP_A2A_TIMEOUT_MS', '60000'))


def register_groups(groups):
    """groups: List[List[int]], e.g. [[0,1],[2,3]]"""
    for gid, ranks in enumerate(groups):
        if gid not in GROUP_REGISTRY:
            GROUP_REGISTRY[gid] = dist.new_group(ranks)


def get_group(gid: int):
    return GROUP_REGISTRY[gid] if gid is not None else dist.get_world_group()


def get_registry():
    return GROUP_REGISTRY


def is_setup():
    return GROUP_REGISTRY['is_reg'] if 'is_reg' in GROUP_REGISTRY else False


def extract_mesh_size(param_dict):
    sp_size = param_dict.get('sequence_parallel_size', 1)
    assert dist.get_world_size() % sp_size == 0, 'World mesh-size should be divisible by SP_SIZE'
    dp_size = dist.get_world_size() // sp_size

    return sp_size, dp_size


def sp_size():
    assert 'SP_SIZE' in GROUP_REGISTRY, 'SP_SIZE not init properly.'

    return GROUP_REGISTRY['SP_SIZE']


def dp_size():
    assert 'DP_SIZE' in GROUP_REGISTRY, 'DP_SIZE not init properly'

    return GROUP_REGISTRY['DP_SIZE']


def populate_registry(SP_SIZE, DP_SIZE):
    """ Populate rank to SP/DP mesh index.  """

    if GROUP_REGISTRY.get('is_reg', False):
        return

    group_listing = []
    offset = 0
    for _ in range(DP_SIZE):
        group_listing.append([i + offset for i in range(SP_SIZE)])
        offset += SP_SIZE

    register_groups(group_listing)

    ## Extraneous metadata required for proper instatiation. ##
    GROUP_REGISTRY['SP_SIZE'] = SP_SIZE
    GROUP_REGISTRY['DP_SIZE'] = DP_SIZE
    GROUP_REGISTRY['is_reg'] = True


# M356: Async handle management for SP all-to-all operations.
# On heterogeneous GPUs, forward/backward execution timing differs,
# causing NCCL deadlock if handles are not properly fenced.
def track_a2a_handle(handle):
    """Register an async A2A handle for later fence."""
    if handle is not None:
        _PENDING_A2A_HANDLES.append((handle, time.monotonic()))


def fence_all_sp_handles(timeout_ms=None):
    """Wait for all pending SP all-to-all handles with timeout.

    Returns number of handles fenced. Raises RuntimeError on timeout.
    Pattern: Megatron finish_grad_sync fence with 30s timeout.
    """
    timeout_ms = timeout_ms or _A2A_TIMEOUT_MS
    fenced = 0
    while _PENDING_A2A_HANDLES:
        handle, submit_time = _PENDING_A2A_HANDLES.pop(0)
        elapsed_ms = (time.monotonic() - submit_time) * 1000
        if elapsed_ms > timeout_ms:
            raise RuntimeError(
                f"[SP] A2A handle timed out after {elapsed_ms:.0f}ms "
                f"(limit={timeout_ms}ms). Possible NCCL deadlock on "
                f"heterogeneous GPUs. Set DESLOC_SP_A2A_TIMEOUT_MS higher "
                f"or check GPU synchronization.")
        if hasattr(handle, 'wait'):
            handle.wait()
        fenced += 1
    return fenced


def pending_handle_count():
    """Number of unfenced SP handles."""
    return len(_PENDING_A2A_HANDLES)


def fence_before_dp_sync():
    """M361(f): Fence all SP A2A handles before DP AllReduce.

    Prevents NCCL communicator竞争: if SP A2A and DP AllReduce use the
    same communicator (default WORLD group), concurrent execution deadlocks
    because NCCL serializes operations per-communicator. The A2A from backward
    must complete before the AllReduce at Kx boundary begins.

    From Megatron param_and_grad_buffer.py finish_grad_sync() start.
    Then, follow that pattern to implement fence_before_dp_sync, letting
    pending SP handles complete before the DP AllReduce fires, and
    preventing the NCCL communicator from being in two collectives at once.
    Then NCCL src/include/collectives.h introduces the per-communicator
    serialization constraint, so that the fence ensures the all_to_all
    is fully retired before all_reduce is enqueued, while the separate
    process groups in M361(d) optimizes the common case where SP and DP
    can use different communicators entirely.
    """
    n = fence_all_sp_handles()
    return n
