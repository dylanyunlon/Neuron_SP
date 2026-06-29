"""FP8 training utilities for heterogeneous GPU clusters.

Adapted from Megatron megatron/core/fp8_utils.py.

Key heterogeneous constraint:
- A6000 (SM86, compute_cap 8.6): NO FP8 hardware → auto-fallback to BF16
- H100 NVL (SM90, compute_cap 9.0): FP8 supported
- Blackwell PRO 6000 (SM120, compute_cap 12.0): FP8 supported

This module provides a unified API that transparently handles the mixed case:
FP8 on capable GPUs, BF16 on others, within the same training run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# TE availability check
try:
    import transformer_engine as te  # noqa: F401
    from transformer_engine.pytorch import fp8_autocast

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class FP8Precision(Enum):
    """FP8 sub-formats."""
    E4M3 = "e4m3"  # range [-448, 448], 8-bit
    E5M2 = "e5m2"  # range [-57344, 57344], wider range for gradients


@dataclass
class FP8Config:
    """Configuration for FP8 training.

    Designed for heterogeneous clusters where some GPUs support FP8 and others don't.
    """
    enabled: bool = False
    """Master switch for FP8 training."""

    forward_precision: FP8Precision = FP8Precision.E4M3
    """FP8 format for forward pass (E4M3 = more precision)."""

    backward_precision: FP8Precision = FP8Precision.E5M2
    """FP8 format for backward pass (E5M2 = wider range for gradients)."""

    margin: int = 0
    """Margin for amax history scaling."""

    amax_history_len: int = 1024
    """Length of amax history window for delayed scaling."""

    amax_compute_algo: str = "max"
    """Algorithm for computing amax: 'max' or 'most_recent'."""

    min_compute_capability: float = 9.0
    """Minimum GPU compute capability for FP8. GPUs below this use BF16."""

    def is_fp8_available(self, device: Optional[torch.device] = None) -> bool:
        """Check if FP8 is available on the given device.

        Returns False for A6000 (SM86), True for H100 (SM90+).
        """
        if not self.enabled:
            return False
        if not HAVE_TE:
            return False
        if device is None:
            device = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(device)
        compute_cap = cap[0] + cap[1] / 10.0
        return compute_cap >= self.min_compute_capability


def get_device_fp8_support() -> dict:
    """Query all visible GPUs for FP8 support.

    Returns:
        Dict mapping device index to (name, compute_cap, fp8_supported).
    """
    result = {}
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        compute_cap = cap[0] + cap[1] / 10.0
        fp8_ok = compute_cap >= 9.0 and HAVE_TE
        result[i] = {
            "name": name,
            "compute_capability": compute_cap,
            "fp8_supported": fp8_ok,
        }
    return result


class FP8LinearWrapper(nn.Module):
    """Wrapper that runs nn.Linear in FP8 on capable GPUs, BF16 otherwise.

    Used by TransformerLayer to transparently handle mixed-precision
    in heterogeneous clusters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fp8_config: Optional[FP8Config] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.fp8_config = fp8_config or FP8Config()
        self._device = device

        if self.fp8_config.is_fp8_available(device) and HAVE_TE:
            # Use TE's FP8-aware Linear
            from transformer_engine.pytorch import Linear as TELinear
            self.linear = TELinear(in_features, out_features, bias=bias)
            self._use_fp8 = True
            logger.debug("FP8LinearWrapper: using TE FP8 on device %s", device)
        else:
            # Fallback to standard BF16 Linear
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self._use_fp8 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_fp8 and HAVE_TE:
            with fp8_autocast(enabled=True):
                return self.linear(x)
        else:
            return self.linear(x)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.linear, "bias", None)


def log_fp8_status():
    """Log FP8 support status for all GPUs."""
    info = get_device_fp8_support()
    for idx, details in info.items():
        status = "FP8" if details["fp8_supported"] else "BF16 (no FP8)"
        logger.info(
            "GPU %d: %s (SM%.1f) → %s",
            idx, details["name"], details["compute_capability"], status,
        )
