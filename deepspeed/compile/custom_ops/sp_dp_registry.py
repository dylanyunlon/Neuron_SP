import os
import time
import logging
import deepspeed.comm as dist

logger = logging.getLogger(__name__)

_PROCESS_GROUPS = {}

_MESH_META = {
    "sp_size": 0,
    "dp_size": 0,
    "is_registered": False,
}

_PENDING_A2A_HANDLES = []
_A2A_TIMEOUT_MS = int(os.environ.get('DESLOC_SP_A2A_TIMEOUT_MS', '60000'))
_A2A_HANDLE_HIGH_WATER = 64


def register_groups(groups):
    for gid, ranks in enumerate(groups):
        if gid not in _PROCESS_GROUPS:
            _PROCESS_GROUPS[gid] = dist.new_group(ranks)


def get_group(gid):
    if gid is None:
        return dist.get_world_group()
    if gid not in _PROCESS_GROUPS:
        raise KeyError(
            f"No process group registered for DP-group index {gid}. "
            f"Available: {list(_PROCESS_GROUPS.keys())}")
    return _PROCESS_GROUPS[gid]


def is_setup():
    return _MESH_META["is_registered"]


def extract_mesh_size(param_dict):
    sp = param_dict.get('sequence_parallel_size', 1)
    assert dist.get_world_size() % sp == 0, 'World mesh-size should be divisible by SP_SIZE'
    dp = dist.get_world_size() // sp
    return sp, dp


def sp_size():
    assert _MESH_META["is_registered"], 'SP mesh not initialised. Call populate_registry() first.'
    return _MESH_META["sp_size"]


def dp_size():
    assert _MESH_META["is_registered"], 'DP mesh not initialised. Call populate_registry() first.'
    return _MESH_META["dp_size"]


def populate_registry(SP_SIZE, DP_SIZE):
    if _MESH_META["is_registered"]:
        return

    group_listing = []
    offset = 0
    for _ in range(DP_SIZE):
        group_listing.append([i + offset for i in range(SP_SIZE)])
        offset += SP_SIZE

    register_groups(group_listing)

    _MESH_META["sp_size"] = SP_SIZE
    _MESH_META["dp_size"] = DP_SIZE
    _MESH_META["is_registered"] = True

    _r = dist.get_rank()
    _ws = dist.get_world_size()
    _gid = _r // SP_SIZE
    _sp_local = _r % SP_SIZE
    logger.info(
        f"[SP-REG] rank={_r}/{_ws} SP={SP_SIZE} DP={DP_SIZE} "
        f"sp_group_id={_gid} sp_rank_in_group={_sp_local} "
        f"groups={[list(range(i*SP_SIZE,(i+1)*SP_SIZE)) for i in range(DP_SIZE)]}")


def track_a2a_handle(handle):
    if handle is not None:
        _PENDING_A2A_HANDLES.append((handle, time.monotonic()))
    if len(_PENDING_A2A_HANDLES) > _A2A_HANDLE_HIGH_WATER:
        _enforce_high_water()


def _enforce_high_water():
    fenced = 0
    while len(_PENDING_A2A_HANDLES) > _A2A_HANDLE_HIGH_WATER // 2:
        handle, _ = _PENDING_A2A_HANDLES.pop(0)
        if hasattr(handle, 'wait'):
            handle.wait()
        fenced += 1
    if fenced > 0:
        logger.debug(f"[SP] Force-fenced {fenced} A2A handles (high-water={_A2A_HANDLE_HIGH_WATER})")


def fence_all_sp_handles(timeout_ms=None):
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
    return len(_PENDING_A2A_HANDLES)


def fence_before_dp_sync():
    return fence_all_sp_handles()


def finalize_a2a_pass(is_last_pass, counter_update_fn):
    fence_all_sp_handles()
    counter_update_fn()
    if not is_last_pass:
        _PENDING_A2A_HANDLES.clear()


def cleanup_sp_groups():
    global _PROCESS_GROUPS, _PENDING_A2A_HANDLES
    try:
        fence_all_sp_handles(timeout_ms=5000)
    except RuntimeError:
        logger.warning("[SP cleanup] Timeout fencing pending A2A handles")

    try:
        from .double_buffer_a2a import get_buffer_pool
        get_buffer_pool().free_all()
    except ImportError:
        pass

    try:
        from .sp_histogram import get_histogram_kernel
        get_histogram_kernel().reset()
    except ImportError:
        pass

    for gid, group in list(_PROCESS_GROUPS.items()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            pass

    _PROCESS_GROUPS.clear()
    _PENDING_A2A_HANDLES.clear()
    _MESH_META["sp_size"] = 0
    _MESH_META["dp_size"] = 0
    _MESH_META["is_registered"] = False


_BUFFER_LIFECYCLE = {
    "created": 0,
    "swapped": 0,
    "freed": 0,
}


def track_buffer_event(event):
    if event in _BUFFER_LIFECYCLE:
        _BUFFER_LIFECYCLE[event] += 1


def get_buffer_lifecycle_stats():
    return dict(_BUFFER_LIFECYCLE)
