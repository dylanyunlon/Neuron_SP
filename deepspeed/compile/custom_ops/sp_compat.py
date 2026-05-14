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
