import logging

import torch
from torch.fx import GraphModule
from .passes.sp_compile import apply_autosp
from .custom_ops.sp_dp_registry import extract_mesh_size
from .custom_ops.sp_compat import _check_autosp_compatibility
from .custom_ops import all_to_all as _force_register_a2a  # noqa: F401
from .passes.long_context_checkpointing import register_long_context_checkpointing

logger = logging.getLogger(__name__)


def init_autosp(config):
    _check_autosp_compatibility()
    from .custom_ops.sp_dp_registry import cleanup_sp_groups, is_setup
    if is_setup():
        cleanup_sp_groups()
    sp_size, dp_size = extract_mesh_size(config._param_dict)
    register_long_context_checkpointing()

    import deepspeed.comm as dist
    _n_heads = config._param_dict.get('n_heads', config._param_dict.get('num_attention_heads', 0))
    _n_kv_heads = config._param_dict.get('num_key_value_heads',
                                          config._param_dict.get('n_kv_heads', _n_heads))
    _min_heads = min(_n_heads, _n_kv_heads) if _n_kv_heads > 0 else _n_heads
    if _min_heads > 0 and sp_size > 1 and _min_heads % sp_size != 0:
        old_sp = sp_size
        for cand in range(sp_size - 1, 0, -1):
            if _min_heads % cand == 0 and dist.get_world_size() % cand == 0:
                sp_size = cand
                dp_size = dist.get_world_size() // cand
                break
        logger.warning(
            f"[AutoSP] n_heads={_n_heads}, n_kv_heads={_n_kv_heads} "
            f"(min={_min_heads}) not divisible by sp_size={old_sp}. "
            f"Reduced to sp_size={sp_size}, dp_size={dp_size}.")

    config._param_dict['_effective_sp_size'] = sp_size
    config._param_dict['_effective_dp_size'] = dp_size

    _desloc_cfg = config._param_dict.get('desloc', {})
    _desloc_enabled = _desloc_cfg.get('enabled', False)
    _desloc_Kx = _desloc_cfg.get('Kx', 1)

    _hetero_cfg = config._param_dict.get('hetero_mesh', {})
    _mesh_strategy = _hetero_cfg.get('strategy', 'contiguous')

    _histogram_cfg = config._param_dict.get('sp_histogram', {})
    _histogram_enabled = _histogram_cfg.get('enabled', False)
    _histogram_bins = _histogram_cfg.get('num_bins', 256)

    if _mesh_strategy != 'contiguous':
        from .custom_ops.hetero_mesh import populate_hetero_registry
        populate_hetero_registry(sp_size, dp_size, strategy=_mesh_strategy)

    if _histogram_enabled:
        from .custom_ops.sp_histogram import get_histogram_kernel
        get_histogram_kernel(num_bins=_histogram_bins)

    logger.info(
        f"[AutoSP] sp={sp_size} dp={dp_size} desloc={_desloc_enabled} "
        f"Kx={_desloc_Kx} mesh_strategy={_mesh_strategy} "
        f"histogram={_histogram_enabled}")

    def backend_fn(gm: GraphModule, real_inputs):
        apply_autosp(gm, real_inputs, debug=False, sp_size=sp_size, dp_size=dp_size)
        return torch._inductor.compile(gm, real_inputs)

    return backend_fn
