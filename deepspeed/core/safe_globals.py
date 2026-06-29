# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Centralised registry of types safe to unpickle via torch.load.

From Megatron M2620: mirrors megatron/core/safe_globals.py. As PyTorch tightens
weights_only=True semantics, keeping all safe types here makes auditing easy.

Call register_safe_globals() once at process startup, or import this module
(it auto-registers on import).
"""
from __future__ import annotations

import io
import logging
from typing import Any, List, Type

logger = logging.getLogger(__name__)

# Add types here when torch.load with weights_only=True raises:
# "Unsupported global: GLOBAL io.BytesIO was not an allowed global"
# Only add provably safe types (stdlib / numpy / torch builtins).
SAFE_GLOBALS: List[Type[Any]] = [
    io.BytesIO,
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
