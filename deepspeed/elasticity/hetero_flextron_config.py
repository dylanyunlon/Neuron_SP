# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""HeteroFlextronConfig — heterogeneous Flextron subnet selection for DES-LOC.

Mirrors Megatron 2d862fe0c (Flextron commit) FlextronConfig + MemoryConfig,
reinterpreted for asymmetric GPU clusters (e.g. A6000 48 GB + H100 80 GB)
where a single fixed budget is wasteful: weak GPUs need a narrow subnet to
fit in VRAM and meet compute SLAs, while strong GPUs should run the full
model width to maximise utilisation.

Design intent (upstream 2d862fe0c)
------------------------------------
Megatron Flextron trains a *family* of sub-networks parameterised by a
``budget`` scalar in [0, 1]: budget=1.0 is the full-width model, smaller
budgets progressively narrow FFN hidden-size (``mlp_int_list``), Mamba
heads (``mamba_int_list``), embedding dimension (``emb_int_list``), and
MoE expert count (``moe_expert_int_list``).  At inference time, a
FlextronRouter picks one budget from ``budget_list`` per forward pass given
the current hardware target.

DES-LOC adaptation
-------------------
In a heterogeneous cluster every GPU rank has a *different* optimal budget:
  - A6000 (48 GB, ~310 TFLOP BF16):  fit a small subnet, leave headroom for
    KV cache at long sequence lengths.
  - H100 (80 GB, ~1979 TFLOP BF16):  run full-width (budget=1.0) and keep
    the router in evaluation mode so it doesn't straggle on the budget-
    selection overhead.

``select_subnet_for_device`` implements this mapping:
1. Query available VRAM and an optional compute-capacity weight.
2. Iterate ``budget_list`` (assumed descending by sort_budget_list_descending)
   and pick the *largest* budget whose estimated memory footprint fits within
   ``memory_headroom_fraction`` of available VRAM.
3. Emit a structured diagnostic (rank-0 or per-rank, controlled by
   ``verbose``) that mirrors the M451 scale-grow event pattern: one log line
   at the decision boundary, not per-iteration noise.

``DeslocWSDKxAligned``-style config replay
------------------------------------------
``HeteroFlextronConfig.replay_budget_selection`` re-runs the VRAM probe at
checkpoint-resume time using the *current* device (which may differ from the
device at initial training time) and overwrites ``selected_budget``.  This
mirrors M451's ``load_state_dict`` replay: LR / budget derived from the live
formula rather than trusting a serialized float that may be stale.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[DS-HFlextron]"

# ---------------------------------------------------------------------------
# GPU capability catalogue (compute TFLOP BF16, typical VRAM GB)
# ---------------------------------------------------------------------------

# Recognised GPU product names (substring match, case-insensitive).
# Values: (compute_tflop_bf16, typical_vram_gb).  Used when torch.cuda
# VRAM query is unavailable or for unit tests.
_GPU_CATALOGUE: Dict[str, Tuple[float, float]] = {
    "a6000":  (310.0,  48.0),
    "a100":   (312.0,  80.0),
    "h100":   (1979.0, 80.0),
    "h200":   (1979.0, 141.0),
    "a40":    (149.7,  48.0),
    "l40":    (362.0,  48.0),
    "rtx 4090": (165.2, 24.0),
    "v100":   (125.0,  32.0),
}


def _probe_device_vram_gb(device_index: int = 0) -> float:
    """Return available VRAM in GB for *device_index* via torch.cuda."""
    try:
        import torch
        props = torch.cuda.get_device_properties(device_index)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def _probe_device_name(device_index: int = 0) -> str:
    """Return torch.cuda device name or empty string."""
    try:
        import torch
        return torch.cuda.get_device_properties(device_index).name
    except Exception:
        return ""


def _catalogue_lookup(device_name: str) -> Optional[Tuple[float, float]]:
    """Find (compute_tflop, vram_gb) from the catalogue by substring match."""
    lower = device_name.lower()
    for key, vals in _GPU_CATALOGUE.items():
        if key in lower:
            return vals
    return None


# ---------------------------------------------------------------------------
# Simple memory estimator (mirrors flex_budget_utils.get_memory_footprint)
# ---------------------------------------------------------------------------

def _estimate_subnet_memory_gb(
    *,
    budget: float,
    full_vram_gb: float,
    bpe_params: float = 2.0,
    bpe_kv_cache: float = 2.0,
    bpe_ssm_cache: float = 2.0,
    kv_cache_fraction: float = 0.30,
    ssm_cache_fraction: float = 0.05,
) -> float:
    """Estimate subnet memory in GB given a budget fraction and full-model VRAM.

    Upstream ``get_memory_footprint`` computes bytes from first principles
    (hybrid_pattern, mamba heads, FFN sizes, etc.).  That requires a fully-
    materialised model config which is unavailable at scheduler init time in
    DS.  We use a linear approximation instead:

        param_mem ≈ budget * full_param_mem
        kv_cache  ≈ budget * kv_cache_fraction * full_vram      (scales with heads)
        ssm_cache ≈ budget * ssm_cache_fraction * full_vram     (scales with mamba heads)

    The bpe_* scalars mirror MemoryConfig so downstream code can pass the
    same quantisation profile (bf16, fp8_kv, int8) that was used at
    Megatron training time.

    Parameters
    ----------
    budget : float
        Sub-network budget in (0, 1].
    full_vram_gb : float
        Total device VRAM in GB (used to partition between param / KV cache
        / SSM cache).
    bpe_params / bpe_kv_cache / bpe_ssm_cache : float
        Bytes-per-element for each memory component; mirrors MemoryConfig.
        Default 2.0 = BF16 everywhere.
    kv_cache_fraction : float
        Fraction of full_vram allocated to KV cache when budget=1.0.
    ssm_cache_fraction : float
        Fraction of full_vram allocated to SSM state cache when budget=1.0.
    """
    # Param memory scales linearly with budget.
    # Use BF16 as the reference; apply bpe ratio relative to BF16 (=2.0).
    bpe_ratio_params = bpe_params / 2.0
    bpe_ratio_kv = bpe_kv_cache / 2.0
    bpe_ratio_ssm = bpe_ssm_cache / 2.0

    param_mem  = budget * (1.0 - kv_cache_fraction - ssm_cache_fraction) * full_vram_gb * bpe_ratio_params
    kv_mem     = budget * kv_cache_fraction  * full_vram_gb * bpe_ratio_kv
    ssm_mem    = budget * ssm_cache_fraction * full_vram_gb * bpe_ratio_ssm
    return param_mem + kv_mem + ssm_mem


# ---------------------------------------------------------------------------
# HeteroFlextronConfig
# ---------------------------------------------------------------------------

@dataclass
class HeteroFlextronConfig:
    """Per-GPU Flextron subnet configuration for DES-LOC heterogeneous clusters.

    Mirrors FlextronConfig (Megatron 2d862fe0c) but selects a *per-rank*
    budget rather than a globally shared one.  Designed to be constructed
    once per training process (before the first forward pass) and replayed
    from checkpoint.

    Fields
    ------
    budget_list : list of float
        Available budget fractions, *descending* (largest first).  Must
        match the list used during Megatron Flextron training.
    budget_probs : list of float or None
        Training-time sampling probabilities per budget.  Not used for
        selection but stored so the config can be round-tripped through
        checkpoints without data loss.
    bpe_params / bpe_kv_cache / bpe_ssm_cache : float
        Bytes-per-element for memory estimation; mirrors MemoryConfig.
    memory_headroom_fraction : float
        Reserve this fraction of device VRAM as headroom (for activations,
        OS, NCCL buffers).  Selection picks the largest budget whose
        estimated footprint fits within
        (1 - memory_headroom_fraction) * available_vram_gb.
    selected_budget : float or None
        Set by ``select_subnet_for_device``; None before first selection.
    device_vram_gb : float
        Probed or overridden device VRAM used for the last selection.
    device_name : str
        Probed GPU product name used for the last selection.
    """

    budget_list: List[float] = field(default_factory=lambda: [1.0])
    budget_probs: Optional[List[float]] = None

    # Memory quantisation profile (mirrors MemoryConfig fields)
    bpe_params:    float = 2.0
    bpe_kv_cache:  float = 2.0
    bpe_ssm_cache: float = 2.0

    # KV / SSM cache fraction of total VRAM when budget=1.0
    kv_cache_fraction:  float = 0.30
    ssm_cache_fraction: float = 0.05

    # How much headroom to keep free (activations, NCCL, OS)
    memory_headroom_fraction: float = 0.15

    # Filled in by select_subnet_for_device / replay_budget_selection
    selected_budget: Optional[float] = None
    device_vram_gb:  float = 0.0
    device_name:     str   = ""

    def __post_init__(self):
        if len(self.budget_list) == 0:
            raise ValueError("budget_list must contain at least one entry.")
        # Enforce descending order (mirrors sort_budget_list_descending)
        if self.budget_list != sorted(self.budget_list, reverse=True):
            self.budget_list = sorted(self.budget_list, reverse=True)
            logger.warning(
                f"{_LOG_PREFIX} budget_list was not descending; re-sorted to "
                f"{self.budget_list}"
            )
        if self.budget_probs is not None:
            if len(self.budget_probs) != len(self.budget_list):
                raise ValueError(
                    f"budget_probs length {len(self.budget_probs)} != "
                    f"budget_list length {len(self.budget_list)}"
                )

    # ------------------------------------------------------------------
    # Core selection logic
    # ------------------------------------------------------------------

    def select_subnet_for_device(
        self,
        device_index: int = 0,
        *,
        override_vram_gb: Optional[float] = None,
        verbose: bool = True,
        rank: int = 0,
    ) -> float:
        """Pick the largest budget fitting within available device VRAM.

        Algorithm
        ---------
        1. Probe device VRAM (or use ``override_vram_gb`` for tests).
        2. Compute usable VRAM = total_vram * (1 - memory_headroom_fraction).
        3. For each budget in budget_list (descending), estimate subnet
           memory via ``_estimate_subnet_memory_gb``.
        4. Return the first budget whose estimate ≤ usable_vram.
        5. Fall back to the smallest budget if nothing fits.
        6. Emit a structured diagnostic at the decision boundary (rank-0
           or all ranks when verbose=True and rank=0).

        Parameters
        ----------
        device_index : int
            CUDA device index to probe.
        override_vram_gb : float or None
            Override the VRAM probe (useful in unit tests or when NUMA
            topology prevents torch.cuda from reporting accurately).
        verbose : bool
            Emit ``[DS-HFlextron] SELECT`` diagnostic at rank 0.
        rank : int
            Caller's distributed rank; diagnostic only fires at rank 0.

        Returns
        -------
        float
            The selected budget fraction.  Also stored in
            ``self.selected_budget``.
        """
        vram_gb = override_vram_gb if override_vram_gb is not None \
                  else _probe_device_vram_gb(device_index)
        device_name = _probe_device_name(device_index)

        self.device_vram_gb = vram_gb
        self.device_name = device_name

        usable_vram_gb = vram_gb * (1.0 - self.memory_headroom_fraction)

        selected = self.budget_list[-1]  # fallback: smallest budget
        estimated_gb_at_selected = None
        for budget in self.budget_list:  # descending
            est_gb = _estimate_subnet_memory_gb(
                budget=budget,
                full_vram_gb=vram_gb,
                bpe_params=self.bpe_params,
                bpe_kv_cache=self.bpe_kv_cache,
                bpe_ssm_cache=self.bpe_ssm_cache,
                kv_cache_fraction=self.kv_cache_fraction,
                ssm_cache_fraction=self.ssm_cache_fraction,
            )
            if est_gb <= usable_vram_gb:
                selected = budget
                estimated_gb_at_selected = est_gb
                break

        self.selected_budget = selected

        # M451-style: one structured event at the decision boundary
        if verbose and rank == 0:
            est_str = (
                f"{estimated_gb_at_selected:.2f} GB"
                if estimated_gb_at_selected is not None
                else "N/A (fallback)"
            )
            msg = (
                f"{_LOG_PREFIX} SELECT device={device_name!r} "
                f"vram={vram_gb:.1f} GB  usable={usable_vram_gb:.1f} GB  "
                f"-> budget={selected:.4f}  est_subnet_mem={est_str}  "
                f"budget_list={self.budget_list}  "
                f"headroom={self.memory_headroom_fraction:.0%}"
            )
            print(msg, flush=True)
            logger.info(msg)

        return selected

    # ------------------------------------------------------------------
    # Checkpoint replay (mirrors M451 DeslocWSDKxAligned.load_state_dict)
    # ------------------------------------------------------------------

    def replay_budget_selection(
        self,
        device_index: int = 0,
        *,
        override_vram_gb: Optional[float] = None,
        verbose: bool = True,
        rank: int = 0,
    ) -> float:
        """Re-run subnet selection at checkpoint-resume time.

        Upstream Megatron AnnealingLR.__init__ calls self.step(self.num_iters)
        to snap LR to the formula rather than trusting a serialized float.
        M451 applied the same discipline to DeslocWSDKxAligned.load_state_dict:
        replay from step 0 so the value is always formula-derived.

        Here we replay the VRAM probe instead of trusting the
        ``selected_budget`` serialized into the checkpoint, because:
          - The checkpoint may have been created on a different GPU tier.
          - The cluster topology may have changed between runs.
          - Memory headroom may differ (driver version, OS, other processes).

        The previous ``selected_budget`` is logged alongside the new one so
        mismatches are visible without requiring a diff of checkpoint files.

        Returns
        -------
        float
            The newly selected budget (also stored in self.selected_budget).
        """
        prev_budget = self.selected_budget
        new_budget = self.select_subnet_for_device(
            device_index=device_index,
            override_vram_gb=override_vram_gb,
            verbose=False,   # suppress per-call log; emit our own below
            rank=rank,
        )

        if verbose and rank == 0:
            mismatch_tag = "CHANGED" if prev_budget != new_budget else "unchanged"
            msg = (
                f"{_LOG_PREFIX} REPLAY device={self.device_name!r} "
                f"vram={self.device_vram_gb:.1f} GB  "
                f"serialized_budget={prev_budget}  "
                f"replayed_budget={new_budget}  [{mismatch_tag}]"
            )
            print(msg, flush=True)
            logger.info(msg)

            if prev_budget is not None and prev_budget != new_budget:
                # Emit an additional WARNING so the change surfaces in logs
                # even when INFO is suppressed — mirrors the scale-grew print
                # in M451 loss_scaler which also writes at WARNING level.
                warn_msg = (
                    f"{_LOG_PREFIX} BUDGET_SHIFT: replayed budget "
                    f"{new_budget:.4f} != serialized {prev_budget:.4f}  "
                    f"(device VRAM changed or cluster re-topology); "
                    f"subnet width will differ from checkpoint run"
                )
                print(warn_msg, flush=True)
                logger.warning(warn_msg)

        return new_budget

    # ------------------------------------------------------------------
    # State dict (for DeepSpeed checkpoint integration)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise to a plain dict for inclusion in a DeepSpeed checkpoint."""
        return {
            "budget_list":               self.budget_list,
            "budget_probs":              self.budget_probs,
            "bpe_params":                self.bpe_params,
            "bpe_kv_cache":              self.bpe_kv_cache,
            "bpe_ssm_cache":             self.bpe_ssm_cache,
            "kv_cache_fraction":         self.kv_cache_fraction,
            "ssm_cache_fraction":        self.ssm_cache_fraction,
            "memory_headroom_fraction":  self.memory_headroom_fraction,
            "selected_budget":           self.selected_budget,
            "device_vram_gb":            self.device_vram_gb,
            "device_name":               self.device_name,
        }

    def load_state_dict(
        self,
        d: dict,
        device_index: int = 0,
        *,
        override_vram_gb: Optional[float] = None,
        rank: int = 0,
    ) -> None:
        """Load from a checkpoint dict, then replay selection on the live device.

        Follows M451 replay discipline: structural hyperparams (budget_list,
        bpe_*) are restored from the checkpoint; selected_budget is *not*
        trusted but re-derived from the current device via
        ``replay_budget_selection``.
        """
        self.budget_list               = d.get("budget_list",              self.budget_list)
        self.budget_probs              = d.get("budget_probs",             self.budget_probs)
        self.bpe_params                = d.get("bpe_params",               self.bpe_params)
        self.bpe_kv_cache              = d.get("bpe_kv_cache",             self.bpe_kv_cache)
        self.bpe_ssm_cache             = d.get("bpe_ssm_cache",            self.bpe_ssm_cache)
        self.kv_cache_fraction         = d.get("kv_cache_fraction",        self.kv_cache_fraction)
        self.ssm_cache_fraction        = d.get("ssm_cache_fraction",       self.ssm_cache_fraction)
        self.memory_headroom_fraction  = d.get("memory_headroom_fraction", self.memory_headroom_fraction)
        # Restore serialized budget so replay can log the delta
        self.selected_budget           = d.get("selected_budget",          None)
        self.device_vram_gb            = d.get("device_vram_gb",           0.0)
        self.device_name               = d.get("device_name",              "")

        # Re-derive from live device (ignores serialized selected_budget)
        self.replay_budget_selection(
            device_index=device_index,
            override_vram_gb=override_vram_gb,
            rank=rank,
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_hetero_flextron_config(
    budget_list: List[float],
    *,
    budget_probs: Optional[List[float]] = None,
    memory_profile: str = "bf16",
    memory_headroom_fraction: float = 0.15,
    kv_cache_fraction: float = 0.30,
    ssm_cache_fraction: float = 0.05,
) -> HeteroFlextronConfig:
    """Construct a HeteroFlextronConfig from a named memory profile.

    Mirrors ``load_memory_config(args)`` from Megatron's memory_config.py:
    resolves bpe_* from a named preset so callers don't need to hard-code
    bytes-per-element values.

    Parameters
    ----------
    budget_list : list of float
        Budget fractions to consider; will be sorted descending.
    budget_probs : list of float or None
        Optional per-budget training sampling probabilities.
    memory_profile : str
        One of 'bf16', 'fp8_kv', 'fp8_kv_ssm', 'fp8_all', 'int8', 'fp4'.
    memory_headroom_fraction : float
        Fraction of VRAM to reserve as headroom.
    kv_cache_fraction : float
        Fraction of full-model VRAM used by KV cache at budget=1.0.
    ssm_cache_fraction : float
        Fraction of full-model VRAM used by SSM state cache at budget=1.0.

    Returns
    -------
    HeteroFlextronConfig
        Ready to call ``select_subnet_for_device`` on.
    """
    _PROFILE_BPE = {
        "bf16":       (2.0, 2.0, 2.0),
        "fp8_kv":     (2.0, 1.0, 2.0),
        "fp8_kv_ssm": (2.0, 1.0, 1.0),
        "fp8_all":    (1.0, 1.0, 1.0),
        "int8":       (1.0, 1.0, 1.0),
        "fp4":        (0.5625, 0.5625, 1.0),
    }
    if memory_profile not in _PROFILE_BPE:
        raise ValueError(
            f"Unknown memory_profile '{memory_profile}'. "
            f"Available: {list(_PROFILE_BPE.keys())}"
        )
    bpe_params, bpe_kv_cache, bpe_ssm_cache = _PROFILE_BPE[memory_profile]

    try:
        import torch.distributed as tdist
        _rank = tdist.get_rank() if tdist.is_initialized() else 0
    except Exception:
        _rank = 0
    if _rank == 0:
        msg = (
            f"{_LOG_PREFIX} BUILD profile={memory_profile!r}  "
            f"bpe_params={bpe_params}  bpe_kv_cache={bpe_kv_cache}  "
            f"bpe_ssm_cache={bpe_ssm_cache}  "
            f"budget_list={sorted(budget_list, reverse=True)}"
        )
        print(msg, flush=True)
        logger.info(msg)

    return HeteroFlextronConfig(
        budget_list=budget_list,
        budget_probs=budget_probs,
        bpe_params=bpe_params,
        bpe_kv_cache=bpe_kv_cache,
        bpe_ssm_cache=bpe_ssm_cache,
        kv_cache_fraction=kv_cache_fraction,
        ssm_cache_fraction=ssm_cache_fraction,
        memory_headroom_fraction=memory_headroom_fraction,
    )
