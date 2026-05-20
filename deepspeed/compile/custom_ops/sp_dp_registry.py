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
    "is_heterogeneous": False,
    "loc_enabled": False,
    "loc_sp_group_ids": [],
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


def is_heterogeneous():
    return _MESH_META.get("is_heterogeneous", False)


def mark_heterogeneous(flag=True):
    _MESH_META["is_heterogeneous"] = flag


def is_loc_enabled():
    return _MESH_META.get("loc_enabled", False)


def enable_loc(sp_group_ids=None):
    _MESH_META["loc_enabled"] = True
    if sp_group_ids is not None:
        _MESH_META["loc_sp_group_ids"] = list(sp_group_ids)


def get_loc_sp_group_ids():
    return _MESH_META.get("loc_sp_group_ids", [])


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


def populate_registry_with_loc(SP_SIZE, DP_SIZE, loc_peer_ranks=None):
    populate_registry(SP_SIZE, DP_SIZE)

    if loc_peer_ranks is not None and len(loc_peer_ranks) > 0:
        loc_gid = max(_PROCESS_GROUPS.keys(), default=-1) + 1
        _PROCESS_GROUPS[loc_gid] = dist.new_group(loc_peer_ranks)
        enable_loc(sp_group_ids=[loc_gid])
        logger.info(
            f"[SP-REG-LOC] rank={dist.get_rank()} LOC peer group "
            f"gid={loc_gid} ranks={loc_peer_ranks}")


def _drain_threshold():
    if _MESH_META.get("is_heterogeneous", False):
        return _A2A_HANDLE_HIGH_WATER // 4
    if _MESH_META.get("loc_enabled", False):
        return _A2A_HANDLE_HIGH_WATER // 2
    return _A2A_HANDLE_HIGH_WATER


def effective_timeout_ms():
    base = _A2A_TIMEOUT_MS
    if _MESH_META.get("is_heterogeneous", False):
        base = int(base * 1.5)
    if _MESH_META.get("loc_enabled", False):
        base = int(base * 2.0)
    return base


def track_a2a_handle(handle):
    if handle is not None:
        _PENDING_A2A_HANDLES.append((handle, time.monotonic()))
    if len(_PENDING_A2A_HANDLES) > _drain_threshold():
        _enforce_high_water()


def _drain_handles(stop_when, on_stale):
    fenced = 0
    while _PENDING_A2A_HANDLES and not stop_when():
        handle, submit_time = _PENDING_A2A_HANDLES.pop(0)
        elapsed_ms = (time.monotonic() - submit_time) * 1000
        if elapsed_ms > _A2A_TIMEOUT_MS:
            on_stale(elapsed_ms)
        if hasattr(handle, 'wait'):
            handle.wait()
        fenced += 1
    return fenced


def _enforce_high_water():
    target = _drain_threshold() // 2
    fenced = _drain_handles(
        stop_when=lambda: len(_PENDING_A2A_HANDLES) <= target,
        on_stale=lambda ms: logger.error(
            f"[SP] A2A handle stale for {ms:.0f}ms "
            f"(limit={_A2A_TIMEOUT_MS}ms), peer may have left"))
    if fenced > 0:
        logger.debug(f"[SP] Force-fenced {fenced} A2A handles (threshold={_drain_threshold()})")


def fence_all_sp_handles(timeout_ms=None):
    effective_timeout = timeout_ms or effective_timeout_ms()

    def on_stale(ms):
        raise RuntimeError(
            f"[SP] A2A handle timed out after {ms:.0f}ms "
            f"(limit={effective_timeout}ms). Possible NCCL deadlock on "
            f"heterogeneous GPUs. Set DESLOC_SP_A2A_TIMEOUT_MS higher "
            f"or check GPU synchronization.")

    return _drain_handles(
        stop_when=lambda: False,
        on_stale=on_stale)


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

    try:
        from ..passes.long_context_checkpointing import restore_default_checkpointing
        restore_default_checkpointing()
    except ImportError:
        pass

    for gid, group in list(_PROCESS_GROUPS.items()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            pass

    try:
        from .hetero_mesh import reset_hetero_plan
        reset_hetero_plan()
    except ImportError:
        pass

    _PROCESS_GROUPS.clear()
    _PENDING_A2A_HANDLES.clear()
    _MESH_META["sp_size"] = 0
    _MESH_META["dp_size"] = 0
    _MESH_META["is_registered"] = False
    _MESH_META["is_heterogeneous"] = False
    _MESH_META["loc_enabled"] = False
    _MESH_META["loc_sp_group_ids"] = []


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
