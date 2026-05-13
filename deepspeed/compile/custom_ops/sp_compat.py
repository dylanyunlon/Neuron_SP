# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging

import torch
from packaging.version import Version

logger = logging.getLogger(__name__)

_TRANSFORMERS_VALIDATED_CEILING = "4.50.3"


def _check_autosp_compatibility():
    torch_version = Version(torch.__version__.split("+")[0])
    if torch_version < Version("2.9"):
        raise RuntimeError("AutoSP requires PyTorch >= 2.9, found "
                           f"{torch.__version__}.")

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
