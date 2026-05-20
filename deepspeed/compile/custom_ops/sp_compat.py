import logging

import torch
from packaging.version import Version

logger = logging.getLogger(__name__)

_TRANSFORMERS_VALIDATED_CEILING = "4.50.3"
_TORCH_MIN = "2.9"
_TORCH_MAX = "2.10"


def _check_autosp_compatibility():
    torch_version = Version(torch.__version__.split("+")[0])
    if torch_version < Version(_TORCH_MIN):
        raise RuntimeError("AutoSP requires PyTorch >= 2.9, found "
                           f"{torch.__version__}.")

    if torch_version >= Version(_TORCH_MAX):
        logger.warning(
            f"AutoSP validated on PyTorch <{_TORCH_MAX}, found {torch.__version__}. "
            "FX graph structure may differ; run unit tests before production use.")

    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] < 7:
            raise RuntimeError(
                f"AutoSP A2A requires compute capability >= 7.0 (Volta), "
                f"found {cc[0]}.{cc[1]} on {torch.cuda.get_device_name()}.")

    try:
        import transformers
        if Version(transformers.__version__) > Version(_TRANSFORMERS_VALIDATED_CEILING):
            logger.warning(
                f"AutoSP was validated with transformers <= {_TRANSFORMERS_VALIDATED_CEILING}, "
                f"but found {transformers.__version__}. AutoSP may still work; "
                "if you encounter graph breaks or numerical issues, please "
                f"downgrade to transformers=={_TRANSFORMERS_VALIDATED_CEILING}.")
    except ImportError:
        pass


def get_torch_version_tuple():
    raw = torch.__version__.split("+")[0]
    parts = raw.split(".")
    return tuple(int(p) for p in parts[:2])


def preflight_memory_check(n_params_billions, sp_size, world_size, cpu_offload=False):
    if not torch.cuda.is_available():
        return
    dev = torch.cuda.current_device()
    mem_gb = torch.cuda.get_device_properties(dev).total_mem / (1024 ** 3)
    param_bytes_bf16 = n_params_billions * 1e9 * 2
    grad_bytes_bf16 = param_bytes_bf16
    model_per_gpu_gb = (param_bytes_bf16 + grad_bytes_bf16) / (1024 ** 3)
    if cpu_offload:
        model_per_gpu_gb *= 0.5
    if sp_size > 1:
        model_per_gpu_gb *= 0.8
    headroom_gb = mem_gb * 0.85
    if model_per_gpu_gb > headroom_gb:
        logger.warning(
            f"[AutoSP-preflight] {n_params_billions:.1f}B model needs "
            f"~{model_per_gpu_gb:.1f}GB/GPU but {torch.cuda.get_device_name(dev)} "
            f"has {mem_gb:.1f}GB. Enable --cpu_offload and --use_ac.")
