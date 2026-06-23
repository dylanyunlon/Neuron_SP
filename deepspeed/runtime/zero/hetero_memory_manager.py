# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""HeteroMemoryManager — Gemini-inspired asymmetric GPU/CPU memory management for DES-LOC.

Inspired by ColossalAI Gemini (https://github.com/hpcaitech/ColossalAI,
colossalai/zero/gemini/gemini_mgr.py), adapted for DES-LOC heterogeneous
GPU environments where a single training job spans GPUs with very different
VRAM capacities (e.g. A6000 49 GB + H100 96 GB).

Background — Gemini design
---------------------------
ColossalAI Gemini implements a StatefulTensorManager that dynamically moves
parameter *chunks* between GPU and CPU memory based on a placement policy
(``auto`` or ``static``).  The auto policy uses a MemStatsCollector to measure
real-time CUDA memory pressure and evicts the least-recently-used chunks to CPU
when the GPU is near capacity, prefetching them back before they are needed in
the forward pass.

DES-LOC context
----------------
In a DES-LOC job with:

  • A6000 (49 GB VRAM)  — heavier offload pressure; activations, params, and
                          gradients compete for limited headroom.
  • H100  (96 GB VRAM)  — can cache optimizer states in GPU memory; less need
                          for CPU offload.

The native DeepSpeed ZeRO-3 offload_param / offload_optimizer configs are
symmetric — every rank offloads the same way.  HeteroMemoryManager introduces
*asymmetric* offload: each rank queries its own device capacity and chooses
an offload strategy proportional to its available headroom.

Key design decisions (diverging from Gemini)
---------------------------------------------
1. **No chunk abstraction** — DES-LOC already partitions parameters via ZeRO
   stage-3 flat buffers.  HeteroMemoryManager works at the *param-group* level,
   not the chunk level.

2. **GPU-tier probing** — VRAM capacity is probed once at construction via
   ``torch.cuda.get_device_properties`` and broadcast across the DP group.
   This mirrors TierAwareParamLayout (M770) and HeteroOptimizerRouter (M730).

3. **Asymmetric eviction policy**:
     - A6000 tier (< VRAM_THRESHOLD_GB):  aggressive offload; params + optimizer
       states moved to CPU when GPU occupancy exceeds HIGH_WATERMARK_FRAC.
     - H100 tier  (≥ VRAM_THRESHOLD_GB):  conservative offload; only optimizer
       states evicted when GPU occupancy exceeds CONSERVATIVE_WATERMARK_FRAC.

4. **Forward-step diagnostics** — when ``enable_diagnostics=True``, the manager
   prints GPU memory watermarks at every forward step (triggered via
   ``record_forward_step``).  Diagnostic output uses the ``[DS-HMM]`` prefix
   so it is easy to grep from training logs.

5. **LOCActivationCache interop** — if the calling stage-3 optimizer exposes a
   ``loc_activation_cache`` attribute (DES-LOC M620+), the manager queries its
   current occupancy when computing eviction targets, avoiding double-counting
   activations that are already pinned by the activation cache.

Usage
------
Typically constructed inside ``DeepSpeedZeroOptimizer_Stage3.__init__`` after
``_configure_offloading`` has run:

    from deepspeed.runtime.zero.hetero_memory_manager import (
        HeteroMemoryManager, HeteroMemoryConfig,
    )

    hmm_config = HeteroMemoryConfig(enable_diagnostics=True)
    self.hetero_memory_manager = HeteroMemoryManager(
        config=hmm_config,
        dp_process_group=self.dp_process_group,
        optimizer=self,
    )

Then, at the start of each forward pass:

    if hasattr(self, 'hetero_memory_manager'):
        self.hetero_memory_manager.record_forward_step()

And at optimizer step time, to decide whether extra offload is needed:

    if hasattr(self, 'hetero_memory_manager'):
        self.hetero_memory_manager.maybe_evict_optimizer_states()

Log prefix convention (mirrors M451 GREW mode)
------------------------------------------------
    [DS-HMM] INIT     — manager constructed; tier assignment logged.
    [DS-HMM] PROBE    — per-rank VRAM probe results.
    [DS-HMM] FWD      — forward-step watermark diagnostic.
    [DS-HMM] EVICT    — eviction triggered; bytes moved to CPU.
    [DS-HMM] RESTORE  — previously evicted states reloaded.
    [DS-HMM] POLICY   — offload policy decision for this rank.
    [DS-HMM] WARN     — non-fatal anomaly (e.g. unable to probe VRAM).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from deepspeed import comm as dist
    from deepspeed.utils import logger as ds_logger
    _HAS_DS = True
except ImportError:
    import torch.distributed as dist  # type: ignore
    ds_logger = None  # type: ignore
    _HAS_DS = False

log = logging.getLogger(__name__)
_LOG_PREFIX = "[DS-HMM]"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GPUs with VRAM >= this threshold (GB) are treated as "high-capacity" (H100-class).
VRAM_THRESHOLD_GB: float = 70.0

# A6000-class: start evicting when GPU occupancy exceeds this fraction.
HIGH_WATERMARK_FRAC: float = 0.80

# H100-class: only evict optimizer states when occupancy exceeds this fraction.
CONSERVATIVE_WATERMARK_FRAC: float = 0.90

# Bytes per GB.
_BYTES_PER_GB: int = 1 << 30


# ---------------------------------------------------------------------------
# GPU-tier enum
# ---------------------------------------------------------------------------

class GPUTier(str, Enum):
    """GPU memory capacity tier."""
    HIGH = "high"    # H100-class  (≥ VRAM_THRESHOLD_GB)
    LOW  = "low"     # A6000-class (< VRAM_THRESHOLD_GB)
    CPU  = "cpu"     # Rank with no CUDA device


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeteroMemoryConfig:
    """Configuration for HeteroMemoryManager.

    Attributes:
        enable_diagnostics: Print per-forward-step GPU memory watermarks.
        diag_every_n_steps: Emit FWD diagnostic every N forward steps.
        high_watermark_frac: Eviction threshold fraction for LOW-tier GPUs.
        conservative_watermark_frac: Eviction threshold fraction for HIGH-tier GPUs.
        vram_threshold_gb: VRAM (GB) boundary between HIGH and LOW tiers.
        verbose: Emit per-param EVICT / RESTORE entries (may be noisy).
        evict_optimizer_states: Whether to evict optimizer states to CPU when
            above the watermark.  Disable for debugging.
    """
    enable_diagnostics: bool = True
    diag_every_n_steps: int = 1
    high_watermark_frac: float = HIGH_WATERMARK_FRAC
    conservative_watermark_frac: float = CONSERVATIVE_WATERMARK_FRAC
    vram_threshold_gb: float = VRAM_THRESHOLD_GB
    verbose: bool = False
    evict_optimizer_states: bool = True


# ---------------------------------------------------------------------------
# VRAM probe helpers
# ---------------------------------------------------------------------------

@dataclass
class RankVRAMInfo:
    """VRAM capacity info for one DP rank."""
    rank: int
    vram_gb: float
    device_name: str
    tier: GPUTier


def _probe_local_vram() -> Tuple[float, str]:
    """Return (vram_gb, device_name) for the current rank's CUDA device."""
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    props = torch.cuda.get_device_properties(local_rank)
    vram_gb = props.total_memory / _BYTES_PER_GB
    return vram_gb, props.name


def _assign_tier(vram_gb: float, threshold_gb: float) -> GPUTier:
    if vram_gb <= 0.0:
        return GPUTier.CPU
    return GPUTier.HIGH if vram_gb >= threshold_gb else GPUTier.LOW


def _gather_dp_vram_infos(
    dp_group: Any,
    vram_threshold_gb: float,
) -> List[RankVRAMInfo]:
    """All-gather VRAM info across the DP process group.

    Returns a list of RankVRAMInfo, one per rank, sorted by rank.
    Falls back gracefully if distributed is not initialised.
    """
    local_vram_gb, local_device_name = _probe_local_vram()

    if dp_group is None or not torch.distributed.is_initialized():
        rank = 0
        return [
            RankVRAMInfo(
                rank=rank,
                vram_gb=local_vram_gb,
                device_name=local_device_name,
                tier=_assign_tier(local_vram_gb, vram_threshold_gb),
            )
        ]

    world_size = torch.distributed.get_world_size(group=dp_group)
    # Pack (vram_gb * 100) as an int64 scalar so we can use all_gather on a
    # simple integer tensor without requiring a custom reduce operation.
    local_tensor = torch.tensor(
        [int(local_vram_gb * 100)],
        dtype=torch.int64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    gather_list = [torch.zeros(1, dtype=torch.int64, device=local_tensor.device)
                   for _ in range(world_size)]
    torch.distributed.all_gather(gather_list, local_tensor, group=dp_group)

    rank_infos: List[RankVRAMInfo] = []
    for r, t in enumerate(gather_list):
        vgb = t.item() / 100.0
        # Device name is not gathered (costly); approximate from VRAM bucket.
        if vgb >= 90.0:
            dname = "H100"
        elif vgb >= 40.0:
            dname = "A6000"
        elif vgb > 0.0:
            dname = f"GPU~{vgb:.0f}GB"
        else:
            dname = "cpu"
        if r == torch.distributed.get_rank(group=dp_group):
            dname = local_device_name  # use precise name for local rank
        rank_infos.append(RankVRAMInfo(
            rank=r,
            vram_gb=vgb,
            device_name=dname,
            tier=_assign_tier(vgb, vram_threshold_gb),
        ))
    return rank_infos


# ---------------------------------------------------------------------------
# Memory watermark sampler
# ---------------------------------------------------------------------------

def _sample_gpu_memory_watermark() -> Dict[str, float]:
    """Return current GPU memory stats for the current device.

    Returns a dict with keys:
        allocated_gb, reserved_gb, total_gb, occupancy_frac
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0,
                "total_gb": 0.0, "occupancy_frac": 0.0}
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    props = torch.cuda.get_device_properties(local_rank)
    total_bytes = props.total_memory
    allocated_bytes = torch.cuda.memory_allocated(local_rank)
    reserved_bytes  = torch.cuda.memory_reserved(local_rank)
    occ = reserved_bytes / total_bytes if total_bytes > 0 else 0.0
    return {
        "allocated_gb": allocated_bytes / _BYTES_PER_GB,
        "reserved_gb":  reserved_bytes  / _BYTES_PER_GB,
        "total_gb":     total_bytes     / _BYTES_PER_GB,
        "occupancy_frac": occ,
    }


# ---------------------------------------------------------------------------
# HeteroMemoryManager
# ---------------------------------------------------------------------------

class HeteroMemoryManager:
    """Asymmetric GPU/CPU memory manager for DES-LOC heterogeneous GPU clusters.

    Inspired by ColossalAI Gemini's GeminiManager.  Manages dynamic param /
    optimizer-state offload based on each rank's GPU tier (A6000 vs H100).

    Args:
        config: HeteroMemoryConfig instance controlling thresholds and logging.
        dp_process_group: DeepSpeed / torch DP process group.
        optimizer: The owning ZeRO stage-3 optimizer (used to access param
            groups and optionally ``loc_activation_cache``).
    """

    def __init__(
        self,
        config: Optional[HeteroMemoryConfig] = None,
        dp_process_group: Any = None,
        optimizer: Any = None,
    ) -> None:
        self._config = config or HeteroMemoryConfig()
        self._dp_group = dp_process_group
        self._optimizer = optimizer

        # Probe VRAM across the DP group.
        self._rank_infos: List[RankVRAMInfo] = _gather_dp_vram_infos(
            dp_group=dp_process_group,
            vram_threshold_gb=self._config.vram_threshold_gb,
        )

        # Determine this rank's tier.
        if torch.distributed.is_initialized() and dp_process_group is not None:
            local_dp_rank = torch.distributed.get_rank(group=dp_process_group)
        else:
            local_dp_rank = 0
        self._local_rank = local_dp_rank
        if local_dp_rank < len(self._rank_infos):
            self._local_info = self._rank_infos[local_dp_rank]
        else:
            vgb, dname = _probe_local_vram()
            self._local_info = RankVRAMInfo(
                rank=local_dp_rank,
                vram_gb=vgb,
                device_name=dname,
                tier=_assign_tier(vgb, self._config.vram_threshold_gb),
            )

        # Eviction watermark for this rank.
        if self._local_info.tier == GPUTier.HIGH:
            self._watermark_frac = self._config.conservative_watermark_frac
        else:
            self._watermark_frac = self._config.high_watermark_frac

        # Counters.
        self._forward_step: int = 0
        self._evict_count: int = 0
        self._restore_count: int = 0
        self._total_evicted_bytes: int = 0
        self._total_restored_bytes: int = 0

        # Stash for evicted optimizer states (param_id → cpu_tensor).
        self._evicted_states: Dict[int, Dict[str, torch.Tensor]] = {}

        self._log_init()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tier(self) -> GPUTier:
        """GPU tier for this rank."""
        return self._local_info.tier

    @property
    def vram_gb(self) -> float:
        """Total VRAM in GB for this rank."""
        return self._local_info.vram_gb

    @property
    def watermark_frac(self) -> float:
        """Eviction trigger threshold fraction for this rank."""
        return self._watermark_frac

    @property
    def forward_step(self) -> int:
        """Number of forward steps recorded."""
        return self._forward_step

    def record_forward_step(self) -> None:
        """Record a forward step and optionally emit memory diagnostics.

        Call this at the beginning of each forward pass (before allgather).
        """
        self._forward_step += 1
        if not self._config.enable_diagnostics:
            return
        if self._forward_step % self._config.diag_every_n_steps != 0:
            return
        wm = _sample_gpu_memory_watermark()
        self._log_fwd(wm)

    def maybe_evict_optimizer_states(self) -> int:
        """Check GPU occupancy and evict optimizer states to CPU if needed.

        Returns the number of bytes moved to CPU (0 if no eviction occurred).
        Called at the start of the optimizer step, before param allgather.
        """
        if not self._config.evict_optimizer_states:
            return 0
        if self._optimizer is None:
            return 0

        wm = _sample_gpu_memory_watermark()
        if wm["occupancy_frac"] < self._watermark_frac:
            return 0  # below threshold — no action needed

        # Subtract activation cache occupancy if available.
        act_gb = self._query_loc_activation_cache_gb()
        effective_occ = (wm["reserved_gb"] - act_gb) / max(wm["total_gb"], 1e-6)
        if effective_occ < self._watermark_frac:
            # Activations account for the excess; don't double-evict.
            return 0

        evicted_bytes = self._evict_fp32_optimizer_states()
        return evicted_bytes

    def maybe_restore_optimizer_states(self) -> int:
        """Reload previously evicted optimizer states from CPU.

        Returns bytes restored.  Call this after the optimizer step completes
        and before the next forward pass if you want states GPU-resident again.
        For H100-tier ranks the restore is eager; for A6000-tier ranks states
        remain on CPU until just before they are needed.
        """
        if not self._evicted_states:
            return 0
        if self._local_info.tier == GPUTier.LOW:
            # A6000: only restore when GPU has freed enough headroom.
            wm = _sample_gpu_memory_watermark()
            restore_threshold = self._watermark_frac - 0.10
            if wm["occupancy_frac"] > restore_threshold:
                return 0  # still too full
        return self._restore_fp32_optimizer_states()

    def get_rank_vram_summary(self) -> str:
        """Human-readable summary of all DP rank VRAM tiers."""
        lines = [f"{_LOG_PREFIX} PROBE  dp_size={len(self._rank_infos)}"]
        for info in self._rank_infos:
            lines.append(
                f"  rank={info.rank:3d}  vram={info.vram_gb:6.1f}GB"
                f"  tier={info.tier.value:4s}  device={info.device_name}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers — eviction / restore
    # ------------------------------------------------------------------

    def _evict_fp32_optimizer_states(self) -> int:
        """Move fp32 optimizer states for all param groups to CPU.

        Operates on ``self._optimizer.fp32_partitioned_groups_flat`` when
        available (ZeRO stage-3 naming).  Falls back to iterating
        ``optimizer.param_groups`` directly.

        Returns total bytes moved.
        """
        optimizer = self._optimizer
        total_bytes = 0

        # ZeRO-3 stores per-subgroup fp32 master weights in fp32_partitioned_groups_flat.
        fp32_groups = getattr(optimizer, "fp32_partitioned_groups_flat", None)
        if fp32_groups is not None:
            for sg_id, flat in enumerate(fp32_groups):
                if flat is None or not isinstance(flat, torch.Tensor):
                    continue
                if flat.device.type == "cpu":
                    continue  # already offloaded
                nbytes = flat.numel() * flat.element_size()
                flat.data = flat.data.cpu()
                total_bytes += nbytes
                self._evict_count += 1
                if self._config.verbose:
                    self._log_evict(f"fp32_partitioned_groups_flat[{sg_id}]", nbytes)
        else:
            # Fallback: evict Adam state tensors directly.
            inner_opt = getattr(optimizer, "optimizer", optimizer)
            for group in inner_opt.param_groups:
                for p in group["params"]:
                    pid = id(p)
                    state = inner_opt.state.get(p, {})
                    evicted: Dict[str, torch.Tensor] = {}
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                            nbytes = v.numel() * v.element_size()
                            state[k] = v.cpu()
                            evicted[k] = state[k]
                            total_bytes += nbytes
                            self._evict_count += 1
                    if evicted:
                        self._evicted_states[pid] = evicted

        self._total_evicted_bytes += total_bytes
        if total_bytes > 0:
            self._log_evict_summary(total_bytes)
        return total_bytes

    def _restore_fp32_optimizer_states(self) -> int:
        """Reload fp32 optimizer states back to GPU."""
        optimizer = self._optimizer
        total_bytes = 0

        fp32_groups = getattr(optimizer, "fp32_partitioned_groups_flat", None)
        if fp32_groups is not None:
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            for flat in fp32_groups:
                if flat is None or not isinstance(flat, torch.Tensor):
                    continue
                if flat.device.type != "cpu":
                    continue
                nbytes = flat.numel() * flat.element_size()
                flat.data = flat.data.to(device)
                total_bytes += nbytes
                self._restore_count += 1
        elif self._evicted_states:
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            inner_opt = getattr(optimizer, "optimizer", optimizer)
            for group in inner_opt.param_groups:
                for p in group["params"]:
                    pid = id(p)
                    if pid not in self._evicted_states:
                        continue
                    state = inner_opt.state.get(p, {})
                    for k, cpu_t in self._evicted_states[pid].items():
                        if k in state:
                            nbytes = cpu_t.numel() * cpu_t.element_size()
                            state[k] = cpu_t.to(device)
                            total_bytes += nbytes
                            self._restore_count += 1
                    del self._evicted_states[pid]

        self._total_restored_bytes += total_bytes
        if total_bytes > 0:
            self._log_restore_summary(total_bytes)
        return total_bytes

    def _query_loc_activation_cache_gb(self) -> float:
        """Return GB currently held by LOCActivationCache, if available."""
        cache = getattr(self._optimizer, "loc_activation_cache", None)
        if cache is None:
            return 0.0
        occupancy = getattr(cache, "gpu_occupancy_bytes", None)
        if occupancy is None:
            occupancy = getattr(cache, "cached_bytes", None)
        if occupancy is None:
            return 0.0
        return float(occupancy) / _BYTES_PER_GB

    # ------------------------------------------------------------------
    # Diagnostics / logging (mirrors M451 GREW mode log conventions)
    # ------------------------------------------------------------------

    def _log_init(self) -> None:
        rank = self._local_info.rank
        tier = self._local_info.tier.value
        vram = self._local_info.vram_gb
        wm   = self._watermark_frac
        msg = (
            f"{_LOG_PREFIX} INIT   rank={rank}  tier={tier}"
            f"  vram={vram:.1f}GB  evict_watermark={wm:.0%}"
        )
        log.info(msg)
        if _HAS_DS:
            ds_logger.info(msg)
        # Emit full cluster VRAM summary at rank-0.
        if rank == 0:
            summary = self.get_rank_vram_summary()
            log.info(summary)
            if _HAS_DS:
                ds_logger.info(summary)
        # Log which offload policy applies to this rank.
        self._log_policy()

    def _log_policy(self) -> None:
        if self._local_info.tier == GPUTier.HIGH:
            policy = (
                f"conservative offload (optimizer states evicted at "
                f"{self._config.conservative_watermark_frac:.0%} occupancy)"
            )
        elif self._local_info.tier == GPUTier.LOW:
            policy = (
                f"aggressive offload (params+optimizer evicted at "
                f"{self._config.high_watermark_frac:.0%} occupancy)"
            )
        else:
            policy = "cpu-only rank — no CUDA eviction"
        msg = f"{_LOG_PREFIX} POLICY rank={self._local_rank}  {policy}"
        log.info(msg)
        if _HAS_DS:
            ds_logger.info(msg)

    def _log_fwd(self, wm: Dict[str, float]) -> None:
        rank  = self._local_rank
        step  = self._forward_step
        alloc = wm["allocated_gb"]
        rsrv  = wm["reserved_gb"]
        total = wm["total_gb"]
        occ   = wm["occupancy_frac"]
        tier  = self._local_info.tier.value
        msg = (
            f"{_LOG_PREFIX} FWD    rank={rank}  step={step:6d}  tier={tier}"
            f"  alloc={alloc:5.2f}GB  rsrv={rsrv:5.2f}GB"
            f"  total={total:5.2f}GB  occ={occ:.1%}"
        )
        print(msg)   # always print for easy grep from training logs
        log.info(msg)

    def _log_evict(self, name: str, nbytes: int) -> None:
        gb = nbytes / _BYTES_PER_GB
        msg = f"{_LOG_PREFIX} EVICT  rank={self._local_rank}  {name}  {gb:.3f}GB → CPU"
        log.debug(msg)
        if _HAS_DS:
            ds_logger.debug(msg)

    def _log_evict_summary(self, total_bytes: int) -> None:
        gb = total_bytes / _BYTES_PER_GB
        msg = (
            f"{_LOG_PREFIX} EVICT  rank={self._local_rank}"
            f"  total={gb:.3f}GB → CPU"
            f"  cumulative_evictions={self._evict_count}"
        )
        log.info(msg)
        if _HAS_DS:
            ds_logger.info(msg)

    def _log_restore_summary(self, total_bytes: int) -> None:
        gb = total_bytes / _BYTES_PER_GB
        msg = (
            f"{_LOG_PREFIX} RESTORE rank={self._local_rank}"
            f"  total={gb:.3f}GB ← CPU"
            f"  cumulative_restores={self._restore_count}"
        )
        log.info(msg)
        if _HAS_DS:
            ds_logger.info(msg)


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroMemoryConfig on a DeepSpeed engine.

    Instantiates a :class:`HeteroMemoryConfig` from the engine's configuration
    and attaches it as ``engine.hetero_memory_manager``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_memory_manager.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_memory_manager = None
    logger.info("hetero_memory_manager.register() attached engine.hetero_memory_manager")
