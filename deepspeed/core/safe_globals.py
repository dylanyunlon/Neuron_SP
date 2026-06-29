# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Centralised registry of types safe to unpickle via torch.load.

From Megatron M2620: mirrors megatron/core/safe_globals.py. As PyTorch tightens
weights_only=True semantics, keeping all safe types here makes auditing easy.

From Megatron M2275 (708f56559): ckpt loading safe strategy — expanded the
allow-list with argparse.Namespace, pathlib.PosixPath, numpy core types, and
types.SimpleNamespace so that checkpoints saved with weights_only=False can be
reloaded with weights_only=True after PyTorch 2.6.  In DES-LOC this is
essential because A6000 and H100 nodes may run different PyTorch versions;
the safe-globals list ensures checkpoint portability across the tier boundary.

Call register_safe_globals() once at process startup, or import this module
(it auto-registers on import).
"""
from __future__ import annotations

import io
import logging
from argparse import Namespace
from pathlib import PosixPath
from types import SimpleNamespace
from typing import Any, List, Type

import numpy as np
from numpy import dtype as np_dtype
from numpy import ndarray as np_ndarray
from numpy.core.multiarray import _reconstruct as np_reconstruct

logger = logging.getLogger(__name__)

# Add types here when torch.load with weights_only=True raises:
# "Unsupported global: GLOBAL io.BytesIO was not an allowed global"
# Only add provably safe types (stdlib / numpy / torch builtins).
# From Megatron M2275: extended with numpy + argparse + pathlib types.
SAFE_GLOBALS: List[Type[Any]] = [
    io.BytesIO,
    # From Megatron M2275 (708f56559): checkpoint safe strategy additions.
    Namespace,           # argparse.Namespace — appears in optimizer state dicts
    SimpleNamespace,     # types.SimpleNamespace — used in some config checkpoints
    PosixPath,           # pathlib.PosixPath — checkpoint file paths in state dicts
    np_reconstruct,      # numpy.core.multiarray._reconstruct — numpy array rebuild
    np_ndarray,          # numpy.ndarray — data arrays in checkpoints
    np_dtype,            # numpy.dtype — dtype descriptors
]


def register_safe_globals() -> None:
    """Register SAFE_GLOBALS with torch.serialization.

    Safe to call multiple times. No-op on PyTorch < 2.4.
    """
    try:
        import torch
        torch.serialization.add_safe_globals(SAFE_GLOBALS)
        logger.debug(
            "register_safe_globals: registered %d types: %s",
            len(SAFE_GLOBALS), [t.__name__ for t in SAFE_GLOBALS],
        )
    except AttributeError:
        logger.debug("register_safe_globals: add_safe_globals not available (torch < 2.4)")
    except Exception as exc:
        logger.warning("register_safe_globals: %s", exc)


# Auto-register on import.
register_safe_globals()

