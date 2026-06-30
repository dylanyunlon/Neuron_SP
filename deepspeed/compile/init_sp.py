import logging

import torch
from torch.fx import GraphModule
from .passes.sp_compile import apply_autosp
from .custom_ops.sp_dp_registry import extract_mesh_size
from .custom_ops.sp_compat import _check_autosp_compatibility, preflight_memory_check
from .custom_ops import all_to_all as _force_register_a2a
from .passes.long_context_checkpointing import register_long_context_checkpointing

logger = logging.getLogger(__name__)


def _resolve_head_counts(param_dict):
    n_heads = param_dict.get('n_heads', param_dict.get('num_attention_heads', 0))
    n_kv_heads = param_dict.get('num_key_value_heads',
                                 param_dict.get('n_kv_heads', n_heads))
    min_heads = min(n_heads, n_kv_heads) if n_kv_heads > 0 else n_heads
    return n_heads, n_kv_heads, min_heads


def _auto_downgrade_sp_size(sp_size, min_heads, world_size):
    for cand in range(sp_size - 1, 0, -1):
        if min_heads % cand == 0 and world_size % cand == 0:
            return cand, world_size // cand
    return 1, world_size


def _parse_desloc_config(param_dict):
    cfg = param_dict.get('desloc', {})
    return {
        'enabled': cfg.get('enabled', False),
        'Kx': cfg.get('Kx', 1),
    }


def _parse_hetero_config(param_dict):
    cfg = param_dict.get('hetero_mesh', {})
    return {
        'strategy': cfg.get('strategy', 'contiguous'),
    }


def _parse_histogram_config(param_dict):
    cfg = param_dict.get('sp_histogram', {})
    return {
        'enabled': cfg.get('enabled', False),
        'num_bins': cfg.get('num_bins', 256),
    }


def _parse_loc_config(param_dict):
    cfg = param_dict.get('loc', {})
    return {
        'enabled': cfg.get('enabled', False),
        'peer_ranks': cfg.get('peer_ranks', []),
        'model_size': cfg.get('model_size', '7B'),
        'dht_prefix': cfg.get('dht_prefix', 'neuron_sp'),
    }


def _ensure_clean_state():
    from .custom_ops.sp_dp_registry import cleanup_sp_groups, is_setup, pending_handle_count
    if not is_setup():
        return
    pending = pending_handle_count()
    if pending > 0:
        logger.warning(
            f"[AutoSP] Reinitializing with {pending} pending A2A handles. "
            f"Fencing before cleanup to prevent NCCL errors.")
    cleanup_sp_groups()


def _resolve_sp_dp(config):
    import torch.distributed as dist
    sp_size, dp_size = extract_mesh_size(config._param_dict)
    n_heads, n_kv_heads, min_heads = _resolve_head_counts(config._param_dict)

    if min_heads > 0 and sp_size > 1 and min_heads % sp_size != 0:
        old_sp = sp_size
        sp_size, dp_size = _auto_downgrade_sp_size(sp_size, min_heads, dist.get_world_size())
        logger.warning(
            f"[AutoSP] n_heads={n_heads}, n_kv_heads={n_kv_heads} "
            f"(min={min_heads}) not divisible by sp_size={old_sp}. "
            f"Reduced to sp_size={sp_size}, dp_size={dp_size}.")
        if sp_size <= 1:
            raise RuntimeError(
                f"[AutoSP] Cannot find valid sp_size for n_kv_heads={n_kv_heads}. "
                f"All candidates down to 2 fail divisibility against "
                f"min(n_heads,n_kv_heads)={min_heads} and world_size={dist.get_world_size()}. "
                f"Set sequence_parallel_size=1 explicitly to disable SP.")

    config._param_dict['_effective_sp_size'] = sp_size
    config._param_dict['_effective_dp_size'] = dp_size
    config._param_dict['sequence_parallel_size'] = sp_size
    return sp_size, dp_size


def _estimate_param_billions(param_dict):
    model_size_str = param_dict.get('model_size', '')
    _SIZE_MAP = {
        '125M': 0.125, '350M': 0.35, '700M': 0.7,
        '1.3B': 1.3, '1.7B': 1.7, '3B': 3.0,
        '7B': 7.0, '13B': 12.6,
    }
    return _SIZE_MAP.get(model_size_str, 0.0)


def init_autosp(config):
    _check_autosp_compatibility()
    _ensure_clean_state()

    sp_size, dp_size = _resolve_sp_dp(config)

    n_params_b = _estimate_param_billions(config._param_dict)
    loc_cfg = _parse_loc_config(config._param_dict)
    cpu_offload = config._param_dict.get('cpu_offload', False)
    if n_params_b > 0:
        preflight_memory_check(n_params_b, sp_size,
                               dist.get_world_size() if dist.is_initialized() else 1,
                               cpu_offload=cpu_offload)

    register_long_context_checkpointing()

    desloc_cfg = _parse_desloc_config(config._param_dict)
    hetero_cfg = _parse_hetero_config(config._param_dict)
    histogram_cfg = _parse_histogram_config(config._param_dict)
    loc_cfg = _parse_loc_config(config._param_dict)

    if loc_cfg['enabled']:
        from .custom_ops.sp_dp_registry import populate_registry_with_loc
        populate_registry_with_loc(
            sp_size, dp_size,
            loc_peer_ranks=loc_cfg['peer_ranks'] or None)
    elif hetero_cfg['strategy'] != 'contiguous':
        try:
            from .custom_ops.hetero_mesh import populate_hetero_registry
            populate_hetero_registry(sp_size, dp_size, strategy=hetero_cfg['strategy'])
            from .custom_ops.sp_dp_registry import mark_heterogeneous
            mark_heterogeneous(True)
        except ImportError:
            pass

    if histogram_cfg['enabled']:
        from .custom_ops.sp_histogram import get_histogram_kernel
        get_histogram_kernel(num_bins=histogram_cfg['num_bins'])

    logger.info(
        f"[AutoSP] sp={sp_size} dp={dp_size} desloc={desloc_cfg['enabled']} "
        f"Kx={desloc_cfg['Kx']} mesh_strategy={hetero_cfg['strategy']} "
        f"histogram={histogram_cfg['enabled']} "
        f"loc={loc_cfg['enabled']} loc_model={loc_cfg['model_size']}")

    def backend_fn(gm: GraphModule, real_inputs):
        apply_autosp(gm, real_inputs, debug=False, sp_size=sp_size, dp_size=dp_size)
        return torch._inductor.compile(gm, real_inputs)

    return backend_fn
