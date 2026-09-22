# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""mem_probe.py — runtime VRAM measurement and budget validation.

Centralises the cudaMemGetInfo probe logic introduced in issue #593.
Other modules should call ``probe_free_vram()`` instead of touching
``torch.cuda.mem_get_info`` directly, so that:

  1. The safety-margin constant lives in one place.
  2. Non-CUDA environments (CPU unit tests) get a clean fallback.
  3. Budget sanity checks (negative free, free > total) are caught early.

AST call chain (4 nodes):
  TierMap.discover()
    -> _detect_local_tier()
      -> probe_free_vram(local_rank)
        -> torch.cuda.mem_get_info(local_rank)
"""
from __future__ import annotations

import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

# Uniform safety margin applied to measured free VRAM.
# Covers NCCL workspace, torch compile cache, and dynamic peak
# allocations that are not yet visible at discover() time.
VRAM_SAFETY_MARGIN: float = 0.15


def probe_free_vram(device_index: int = 0) -> Tuple[int, int]:
    """Measure free and total VRAM on *device_index* via cudaMemGetInfo.

    Returns:
        ``(free_bytes, total_bytes)``  — both ≥ 0.
        When CUDA is unavailable, returns ``(0, 0)``.

    The caller is responsible for applying ``VRAM_SAFETY_MARGIN`` to
    ``free_bytes`` when computing a budget.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return (0, 0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        # Sanity: clamp impossible values.
        if free_bytes < 0:
            logger.warning(
                "mem_probe: negative free_bytes (%d) on device %d; clamping to 0",
                free_bytes, device_index,
            )
            free_bytes = 0
        if free_bytes > total_bytes:
            logger.warning(
                "mem_probe: free_bytes (%d) > total_bytes (%d) on device %d; "
                "clamping free to total",
                free_bytes, total_bytes, device_index,
            )
            free_bytes = total_bytes
        return (free_bytes, total_bytes)
    except Exception as exc:
        logger.debug("mem_probe: CUDA probe failed (%s); returning (0, 0)", exc)
        return (0, 0)


def budget_from_probe(free_bytes: int, total_bytes: int, tier_reserve: float) -> int:
    """Compute optimizer-state budget from probe results.

    When ``free_bytes > 0``, applies the uniform safety margin.
    Otherwise falls back to ``total_bytes * (1 - tier_reserve)``
    (the legacy path for offline / test environments).

    Args:
        free_bytes:   Measured free VRAM (from ``probe_free_vram``).
        total_bytes:  Total VRAM reported by the device.
        tier_reserve: Legacy per-tier reserve fraction (0.0 – 1.0).

    Returns:
        Integer byte count available for optimizer state.  Always ≥ 0.
    """
    if free_bytes > 0:
        return max(0, int(free_bytes * (1.0 - VRAM_SAFETY_MARGIN)))
    return max(0, int(total_bytes * (1.0 - tier_reserve)))


def validate_tier_info_vram(
    free_vram_bytes: int,
    total_vram_bytes: int,
    rank: int,
) -> int:
    """Validate and sanitise free_vram_bytes for a TierInfo.

    Clamps impossible values and logs warnings.  Called during
    TierMap.discover() and TierMap.from_infos() construction.

    Returns:
        Sanitised ``free_vram_bytes`` (0 ≤ result ≤ total_vram_bytes).
    """
    if free_vram_bytes < 0:
        logger.warning(
            "validate_tier_info_vram: rank %d has negative free_vram (%d); "
            "clamping to 0", rank, free_vram_bytes,
        )
        return 0
    if total_vram_bytes > 0 and free_vram_bytes > total_vram_bytes:
        logger.warning(
            "validate_tier_info_vram: rank %d has free (%d) > total (%d); "
            "clamping to total", rank, free_vram_bytes, total_vram_bytes,
        )
        return total_vram_bytes
    return free_vram_bytes
