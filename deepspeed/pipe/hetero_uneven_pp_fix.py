"""
deepspeed/pipe/hetero_uneven_pp_fix.py
=======================================

DES-LOC Heterogeneous Uneven Pipeline Parallelism Layer Assignment
-------------------------------------------------------------------

Upstream Design Intent (Megatron d10eb6fc):
    Megatron-LM's ``MambaStack._select_layers_for_pipeline_parallel`` originally
    used a naive equal-split:

        offset = pp_rank * (num_layers // pp_size)

    This breaks when ``num_layers_in_first_pipeline_stage`` or
    ``num_layers_in_last_pipeline_stage`` are configured (Nemotron3-nano / hybrid
    Mamba-Transformer models).  The fix mirrors the logic already present in
    ``get_transformer_layer_offset`` so that MambaStack and TransformerLayer agree
    on which global layer indices belong to each pipeline stage.

DES-LOC Adaptation (HeteroUnevenPPFix):
    In the Neuron_SP DES-LOC framework the three physical devices
        • 2× A6000 48 GB  (SM86, PCIe)
        • 1× H100 NVL 96 GB (SM90, PCIe)
    are treated as *heterogeneous pipeline stages*.  Equal-split layer assignment
    is doubly wrong here:
        1. Layer counts differ between first/last and middle stages (upstream bug).
        2. Compute capacity differs between GPU generations, so even a "correct"
           equal split wastes the H100's headroom or starves the A6000s.

    DES-LOC introduces a **Shared LOcality Cache** (SLC) in CPU DRAM (1.5 TB).
    Each GPU's ``HeteroStageDescriptor`` carries a ``slc_budget_bytes`` field that
    governs how many layer activations may be spilled to / prefetched from the SLC.
    The layer-assignment algorithm must therefore produce not just (offset, count)
    but also a DES-LOC ``HeteroStageDescriptor`` that downstream execution
    engines use for SLC scheduling.

    Key divergences from upstream:
        • ``HeteroUnevenPPSolver`` replaces the hard-coded first/last special-case
          with a capacity-weighted rebalancing pass so the H100 (when placed at
          any stage) can absorb more layers without violating the uneven-PP
          contract.
        • ``SLCLayerCache`` wraps the layer_type_list slice and tracks which
          activations are resident in CPU DRAM vs GPU HBM, enabling async
          prefetch on PCIe.
        • ``HeteroStageDescriptor`` is the wire format passed between DeepSpeed
          pipeline engine and the DES-LOC scheduler.

Hardware topology assumed:
    rank 0  → A6000 #0  (48 GB, SM86)   — "first stage"  by default
    rank 1  → A6000 #1  (48 GB, SM86)   — "middle stage"
    rank 2  → H100 NVL  (96 GB, SM90)   — "last stage"   by default

    PCIe bandwidth (~32 GB/s per slot) is modelled in ``_PCIe_BW_GBps`` and used
    to estimate SLC prefetch latency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware capability constants for the DES-LOC cluster
# ---------------------------------------------------------------------------

# SM version → (HBM GB, relative FLOPS weight)
_DEVICE_PROFILES: Dict[int, Tuple[float, float]] = {
    86: (48.0, 1.0),   # A6000 — baseline
    90: (96.0, 2.7),   # H100 NVL — ~2.7× A6000 on bf16 GEMM
}

_PCIe_BW_GBps: float = 32.0          # conservative single-slot PCIe 4.0 ×16
_SLC_DRAM_TOTAL_GB: float = 1_500.0  # 1.5 TB CPU DRAM budget for the SLC


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeviceCapability:
    """
    Runtime-detected capability of a single GPU pipeline stage.

    Attributes
    ----------
    pp_rank : int
        Pipeline-parallel rank.
    sm_version : int
        CUDA SM version (86 for A6000, 90 for H100 NVL).
    hbm_gb : float
        Available HBM in GB.
    flops_weight : float
        Relative FLOPS weight used for capacity-proportional layer assignment.
    slc_budget_bytes : int
        Number of bytes in the SLC this rank may use (derived from total SLC
        budget divided proportionally by ``flops_weight``).
    """
    pp_rank: int
    sm_version: int
    hbm_gb: float
    flops_weight: float
    slc_budget_bytes: int = 0

    @classmethod
    def from_sm(cls, pp_rank: int, sm_version: int) -> "DeviceCapability":
        hbm, weight = _DEVICE_PROFILES.get(sm_version, (16.0, 0.5))
        return cls(pp_rank=pp_rank, sm_version=sm_version,
                   hbm_gb=hbm, flops_weight=weight)


@dataclass
class HeteroStageDescriptor:
    """
    Per-stage descriptor passed to the DES-LOC pipeline execution engine.

    This is the output contract of ``HeteroUnevenPPSolver``.  Every field is
    consumed by the downstream SLC scheduler and activation-memory manager.

    Attributes
    ----------
    pp_rank : int
        Pipeline-parallel rank this descriptor belongs to.
    layer_offset : int
        Index of the first global layer assigned to this stage.
    num_layers : int
        Number of layers assigned to this stage.
    layer_types : list
        Slice of the full ``layer_type_list`` for this stage.
    slc_budget_bytes : int
        SLC quota for this stage (bytes in CPU DRAM).
    device_cap : DeviceCapability
        Hardware capability of the GPU at this stage.
    is_first_stage : bool
        Whether this is the first PP stage (receives micro-batch input).
    is_last_stage : bool
        Whether this is the last PP stage (produces loss).
    slc_cache : Optional[SLCLayerCache]
        Lazily populated SLC cache object; None until ``attach_slc_cache`` is
        called.
    """
    pp_rank: int
    layer_offset: int
    num_layers: int
    layer_types: List
    slc_budget_bytes: int
    device_cap: DeviceCapability
    is_first_stage: bool = False
    is_last_stage: bool = False
    slc_cache: Optional["SLCLayerCache"] = field(default=None, repr=False)

    def attach_slc_cache(self, cache: "SLCLayerCache") -> None:
        """Bind a SLC cache object to this stage descriptor."""
        self.slc_cache = cache
        logger.debug(
            "rank=%d  SLC cache attached: budget=%.2f GB  layers=%d–%d",
            self.pp_rank,
            self.slc_budget_bytes / (1 << 30),
            self.layer_offset,
            self.layer_offset + self.num_layers - 1,
        )

    def prefetch_hint(self, next_micro_batch_id: int) -> int:
        """
        Return approximate number of bytes to prefetch from SLC for the next
        micro-batch.  Used by the DES-LOC async-prefetch thread.

        Heuristic: prefetch one layer's worth of activations if PCIe transfer
        time < one forward-pass compute time on this device.
        """
        if self.slc_cache is None or self.num_layers == 0:
            return 0
        bytes_per_layer = self.slc_budget_bytes // max(self.num_layers, 1)
        transfer_ns = (bytes_per_layer / (_PCIe_BW_GBps * 1e9)) * 1e9
        # rough compute budget per layer in ns (placeholder; real impl hooks
        # into DeepSpeed's timing infrastructure)
        compute_ns = 1e6 * (1.0 / self.device_cap.flops_weight)
        if transfer_ns < compute_ns:
            return bytes_per_layer
        return 0


# ---------------------------------------------------------------------------
# SLC Layer Cache
# ---------------------------------------------------------------------------

class SLCLayerCache:
    """
    Shared LOcality Cache backed by CPU DRAM for a single pipeline stage.

    In DES-LOC each pipeline stage can spill intermediate activations to the
    SLC and prefetch them asynchronously over PCIe.  This class provides the
    bookkeeping interface; actual tensor movement is handled by the DeepSpeed
    activation-offload engine (``deepspeed.runtime.zero.offload_engine``).

    The SLC is *not* a transparent cache: layers must explicitly ``pin`` an
    activation tensor to reserve space and ``release`` it when no longer
    needed.  This avoids the coherency overhead of a fully transparent scheme
    on PCIe-only topologies (no NVLink / NVSwitch).

    Parameters
    ----------
    budget_bytes : int
        Maximum bytes this cache instance may occupy in CPU DRAM.
    pp_rank : int
        Owning pipeline rank (for logging).
    pin_memory : bool
        Whether to allocate pinned CPU memory (recommended for PCIe DMA).
    """

    def __init__(self, budget_bytes: int, pp_rank: int,
                 pin_memory: bool = True) -> None:
        self.budget_bytes = budget_bytes
        self.pp_rank = pp_rank
        self.pin_memory = pin_memory
        self._resident: Dict[str, torch.Tensor] = {}
        self._used_bytes: int = 0
        logger.info(
            "SLCLayerCache init: rank=%d  budget=%.2f GB  pin=%s",
            pp_rank, budget_bytes / (1 << 30), pin_memory,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def free_bytes(self) -> int:
        return max(0, self.budget_bytes - self._used_bytes)

    def pin(self, key: str, tensor: torch.Tensor) -> bool:
        """
        Pin ``tensor`` into the SLC under ``key``.

        The tensor is moved to CPU pinned memory if it is currently on GPU.
        Returns True on success, False if there is insufficient SLC budget.

        DES-LOC note: this is a *synchronous* fallback path; the hot path
        uses DeepSpeed's async offload pipeline and only calls ``pin`` when
        the async queue is full.
        """
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.free_bytes:
            logger.warning(
                "SLC rank=%d: cannot pin '%s' (%.2f MB): only %.2f MB free",
                self.pp_rank, key, nbytes / 1e6, self.free_bytes / 1e6,
            )
            return False
        cpu_tensor = tensor.detach().cpu()
        if self.pin_memory:
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except RuntimeError:
                logger.debug("SLC rank=%d: pin_memory unavailable, using pageable",
                             self.pp_rank)
        self._resident[key] = cpu_tensor
        self._used_bytes += nbytes
        return True

    def fetch(self, key: str, device: torch.device,
              non_blocking: bool = True) -> Optional[torch.Tensor]:
        """
        Fetch a pinned tensor back to ``device``.  Returns None if not cached.
        """
        t = self._resident.get(key)
        if t is None:
            return None
        return t.to(device, non_blocking=non_blocking)

    def release(self, key: str) -> bool:
        """Free the SLC entry for ``key``.  Returns True if it was present."""
        t = self._resident.pop(key, None)
        if t is None:
            return False
        self._used_bytes -= t.numel() * t.element_size()
        return True

    def evict_lru(self, target_free_bytes: int) -> int:
        """
        Evict entries (oldest first) until ``target_free_bytes`` is available.
        Returns the number of bytes freed.
        """
        freed = 0
        for key in list(self._resident.keys()):
            if self.free_bytes >= target_free_bytes:
                break
            nbytes = (self._resident[key].numel()
                      * self._resident[key].element_size())
            self.release(key)
            freed += nbytes
            logger.debug("SLC rank=%d: evicted '%s' (%.2f MB)",
                         self.pp_rank, key, nbytes / 1e6)
        return freed

    def stats(self) -> Dict[str, float]:
        return {
            "used_gb": self._used_bytes / (1 << 30),
            "free_gb": self.free_bytes / (1 << 30),
            "budget_gb": self.budget_bytes / (1 << 30),
            "num_entries": len(self._resident),
        }


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

class HeteroUnevenPPSolver:
    """
    Heterogeneous Uneven Pipeline Parallelism Solver for DES-LOC.

    Replaces Megatron's ``_select_layers_for_pipeline_parallel`` with a
    two-phase algorithm that:

    Phase 1 — Uneven PP contract (mirrors Megatron d10eb6fc):
        Respect ``num_layers_in_first_pipeline_stage`` and
        ``num_layers_in_last_pipeline_stage`` exactly, distributing the
        remainder evenly among middle stages.  This matches Megatron's
        ``get_transformer_layer_offset`` so MambaStack and TransformerLayer
        layers are co-located.

    Phase 2 — Capacity-weighted rebalancing (DES-LOC extension):
        If ``enable_hetero_rebalance=True``, redistribute *middle* stage
        layers proportionally to ``flops_weight`` so the H100 absorbs more
        work.  First and last stage counts are *never* changed (they are
        architectural constraints for Nemotron-style models).

    SLC budget allocation:
        Each stage's SLC quota is proportional to its ``flops_weight`` so
        fast GPUs can maintain hotter caches.  The total SLC budget is
        ``_SLC_DRAM_TOTAL_GB`` minus a small OS reservation.

    Parameters
    ----------
    num_layers : int
        Total number of model layers.
    pp_size : int
        Number of pipeline-parallel stages.
    num_layers_in_first : Optional[int]
        Override for first-stage layer count (``None`` = use equal split).
    num_layers_in_last : Optional[int]
        Override for last-stage layer count (``None`` = use equal split).
    device_caps : Optional[List[DeviceCapability]]
        Per-rank device capabilities.  If None, all ranks assumed equal.
    enable_hetero_rebalance : bool
        Enable Phase 2 capacity-weighted rebalancing of middle stages.
    slc_os_reserve_gb : float
        GB of CPU DRAM reserved for the OS / non-SLC usage.
    """

    def __init__(
        self,
        num_layers: int,
        pp_size: int,
        num_layers_in_first: Optional[int] = None,
        num_layers_in_last: Optional[int] = None,
        device_caps: Optional[List[DeviceCapability]] = None,
        enable_hetero_rebalance: bool = True,
        slc_os_reserve_gb: float = 100.0,
    ) -> None:
        self.num_layers = num_layers
        self.pp_size = pp_size
        self.num_layers_in_first = num_layers_in_first
        self.num_layers_in_last = num_layers_in_last
        self.enable_hetero_rebalance = enable_hetero_rebalance

        self._slc_total_bytes = int(
            (_SLC_DRAM_TOTAL_GB - slc_os_reserve_gb) * (1 << 30)
        )

        # Fall back to uniform capability if not provided
        if device_caps is None:
            device_caps = [
                DeviceCapability(r, 86, 48.0, 1.0) for r in range(pp_size)
            ]
        if len(device_caps) != pp_size:
            raise ValueError(
                f"device_caps length {len(device_caps)} != pp_size {pp_size}"
            )
        self.device_caps: List[DeviceCapability] = device_caps

        # Allocate SLC budget proportional to flops_weight
        total_weight = sum(d.flops_weight for d in self.device_caps)
        for dc in self.device_caps:
            dc.slc_budget_bytes = int(
                self._slc_total_bytes * (dc.flops_weight / total_weight)
            )

        logger.info(
            "HeteroUnevenPPSolver init: num_layers=%d  pp_size=%d  "
            "first=%s  last=%s  rebalance=%s  slc_total=%.1f GB",
            num_layers, pp_size,
            num_layers_in_first, num_layers_in_last,
            enable_hetero_rebalance,
            self._slc_total_bytes / (1 << 30),
        )

    # ------------------------------------------------------------------
    # Phase 1: Uneven PP layer counts (mirrors Megatron d10eb6fc)
    # ------------------------------------------------------------------

    def _compute_uneven_pp_assignment(self) -> List[Tuple[int, int]]:
        """
        Compute (offset, num_layers) for each pp_rank using the same logic
        as Megatron's ``get_transformer_layer_offset`` + the bugfix in
        d10eb6fc.

        Returns
        -------
        List of (offset, num_layers) tuples, one per pp_rank.
        """
        pp_size = self.pp_size
        num_layers = self.num_layers
        first = self.num_layers_in_first
        last = self.num_layers_in_last

        if first is None and last is None:
            # Simple equal split — no uneven-PP config
            layers_per = num_layers // pp_size
            return [(r * layers_per, layers_per) for r in range(pp_size)]

        f = 0 if first is None else first
        l = 0 if last is None else last
        middle_total = num_layers - f - l

        middle_stages = pp_size - sum(
            1 for x in (first, last) if x is not None
        )
        if middle_stages < 0:
            raise ValueError(
                "pp_size too small for the requested first/last stage config"
            )

        layers_per_middle = middle_total // middle_stages if middle_stages > 0 else 0

        # Remainder layers that cannot be divided evenly go to the last
        # middle stage (consistent with Megatron's choice).
        remainder = (middle_total - layers_per_middle * middle_stages
                     if middle_stages > 0 else 0)

        assignment: List[Tuple[int, int]] = []
        middle_rank_idx = 0
        for pp_rank in range(pp_size):
            is_first = (first is not None and pp_rank == 0)
            is_last  = (last  is not None and pp_rank == pp_size - 1)

            if is_first:
                offset = 0
                n = f
            elif is_last:
                offset = num_layers - l
                n = l
            else:
                extra = remainder if (middle_rank_idx == middle_stages - 1) else 0
                offset = middle_rank_idx * layers_per_middle + f
                n = layers_per_middle + extra
                middle_rank_idx += 1

            assignment.append((offset, n))

        # Sanity check
        total_assigned = sum(n for _, n in assignment)
        if total_assigned != num_layers:
            raise RuntimeError(
                f"Layer assignment mismatch: assigned {total_assigned} "
                f"!= num_layers {num_layers}"
            )

        return assignment

    # ------------------------------------------------------------------
    # Phase 2: Capacity-weighted rebalancing of middle stages
    # ------------------------------------------------------------------

    def _rebalance_middle_stages(
        self, assignment: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Redistribute middle-stage layer counts proportionally to
        ``flops_weight`` while keeping first/last stage counts fixed.

        The rebalancing is *integer-safe*: rounding errors are absorbed by
        the middle stage with the highest weight (typically the H100).

        Parameters
        ----------
        assignment : list of (offset, num_layers) — from Phase 1.

        Returns
        -------
        Updated assignment with rebalanced middle stages.
        """
        first = self.num_layers_in_first
        last  = self.num_layers_in_last
        pp_size = self.pp_size

        # Identify middle ranks
        middle_ranks = [
            r for r in range(pp_size)
            if not (first is not None and r == 0)
            and not (last  is not None and r == pp_size - 1)
        ]
        if not middle_ranks:
            return assignment  # nothing to rebalance

        total_middle_layers = sum(assignment[r][1] for r in middle_ranks)
        total_weight = sum(self.device_caps[r].flops_weight for r in middle_ranks)

        new_counts: Dict[int, int] = {}
        allocated = 0
        sorted_middle = sorted(
            middle_ranks, key=lambda r: self.device_caps[r].flops_weight, reverse=True
        )
        for i, r in enumerate(sorted_middle):
            if i == len(sorted_middle) - 1:
                # absorb rounding remainder
                n = total_middle_layers - allocated
            else:
                n = math.floor(
                    total_middle_layers
                    * self.device_caps[r].flops_weight / total_weight
                )
            new_counts[r] = max(1, n)   # every middle rank gets ≥ 1 layer
            allocated += new_counts[r]

        # Fix up if rounding pushed us over/under
        diff = total_middle_layers - sum(new_counts.values())
        if diff != 0:
            # Add/remove from the rank with most capacity
            anchor = sorted_middle[0]
            new_counts[anchor] = max(1, new_counts[anchor] + diff)

        # Rebuild assignment with new middle counts; recalculate offsets
        rebalanced: List[Tuple[int, int]] = list(assignment)

        # Compute contiguous offsets
        f = 0 if first is None else (first or 0)
        running_offset = f
        for r in range(pp_size):
            is_first_r = (first is not None and r == 0)
            is_last_r  = (last  is not None and r == pp_size - 1)

            if is_first_r:
                running_offset = assignment[r][1]  # = f
                continue
            if is_last_r:
                # last stage offset is fixed
                continue

            n = new_counts[r]
            rebalanced[r] = (running_offset, n)
            running_offset += n
            logger.debug(
                "Rebalance: rank=%d  old_n=%d → new_n=%d  "
                "weight=%.2f  offset=%d",
                r, assignment[r][1], n,
                self.device_caps[r].flops_weight,
                running_offset - n,
            )

        return rebalanced

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(
        self, layer_type_list: Sequence
    ) -> List[HeteroStageDescriptor]:
        """
        Full two-phase solve: uneven-PP assignment + optional hetero
        rebalance, returning a ``HeteroStageDescriptor`` for every rank.

        Parameters
        ----------
        layer_type_list : sequence
            Ordered sequence of layer-type tokens (e.g. strings like
            ``'mamba'``, ``'attention'``, ``'mlp'``).  Length must equal
            ``self.num_layers``.

        Returns
        -------
        List of ``HeteroStageDescriptor`` (one per pp_rank).
        """
        if len(layer_type_list) != self.num_layers:
            raise ValueError(
                f"layer_type_list length {len(layer_type_list)} "
                f"!= num_layers {self.num_layers}"
            )

        # Phase 1
        assignment = self._compute_uneven_pp_assignment()
        logger.info("Phase-1 assignment: %s", assignment)

        # Phase 2 (optional)
        if self.enable_hetero_rebalance:
            assignment = self._rebalance_middle_stages(assignment)
            logger.info("Phase-2 rebalanced: %s", assignment)

        # Build descriptors
        descriptors: List[HeteroStageDescriptor] = []
        pp_size = self.pp_size
        for pp_rank in range(pp_size):
            offset, n = assignment[pp_rank]
            layer_slice = list(layer_type_list[offset: offset + n])
            dc = self.device_caps[pp_rank]

            desc = HeteroStageDescriptor(
                pp_rank=pp_rank,
                layer_offset=offset,
                num_layers=n,
                layer_types=layer_slice,
                slc_budget_bytes=dc.slc_budget_bytes,
                device_cap=dc,
                is_first_stage=(self.num_layers_in_first is not None
                                and pp_rank == 0),
                is_last_stage=(self.num_layers_in_last is not None
                               and pp_rank == pp_size - 1),
            )

            # Attach SLC cache for this stage
            slc = SLCLayerCache(
                budget_bytes=dc.slc_budget_bytes,
                pp_rank=pp_rank,
                pin_memory=True,
            )
            desc.attach_slc_cache(slc)
            descriptors.append(desc)

            logger.info(
                "Stage pp_rank=%d  SM%d  layers=[%d, %d)  "
                "slc_budget=%.2f GB  layer_types=%s",
                pp_rank, dc.sm_version,
                offset, offset + n,
                dc.slc_budget_bytes / (1 << 30),
                layer_slice[:4],   # first 4 for brevity
            )

        return descriptors

    # ------------------------------------------------------------------
    # Compat shim: drop-in for Megatron's _select_layers_for_pipeline_parallel
    # ------------------------------------------------------------------

    def select_layers_for_rank(
        self, pp_rank: int, layer_type_list: Sequence
    ) -> Tuple[int, List]:
        """
        Return ``(offset, selected_list)`` for a single ``pp_rank``.

        This is a compatibility shim so existing DeepSpeed pipe engine code
        that calls the Megatron-style two-return-value API can migrate
        incrementally to ``HeteroStageDescriptor``.
        """
        descriptors = self.solve(layer_type_list)
        desc = descriptors[pp_rank]
        return desc.layer_offset, desc.layer_types


# ---------------------------------------------------------------------------
# Convenience factory for the Neuron_SP default cluster
# ---------------------------------------------------------------------------

def build_default_des_loc_solver(
    num_layers: int,
    num_layers_in_first: Optional[int] = None,
    num_layers_in_last: Optional[int] = None,
    enable_hetero_rebalance: bool = True,
) -> HeteroUnevenPPSolver:
    """
    Instantiate ``HeteroUnevenPPSolver`` pre-configured for the Neuron_SP
    cluster:
        rank 0 → A6000 48 GB (SM86)
        rank 1 → A6000 48 GB (SM86)
        rank 2 → H100 NVL 96 GB (SM90)

    Parameters
    ----------
    num_layers : int
        Total model layer count.
    num_layers_in_first : Optional[int]
        Uneven-PP first-stage override.
    num_layers_in_last : Optional[int]
        Uneven-PP last-stage override.
    enable_hetero_rebalance : bool
        If True, Phase 2 capacity-weighted rebalance is applied to middle
        stages.

    Returns
    -------
    Configured ``HeteroUnevenPPSolver``.
    """
    device_caps = [
        DeviceCapability.from_sm(pp_rank=0, sm_version=86),   # A6000 #0
        DeviceCapability.from_sm(pp_rank=1, sm_version=86),   # A6000 #1
        DeviceCapability.from_sm(pp_rank=2, sm_version=90),   # H100 NVL
    ]
    return HeteroUnevenPPSolver(
        num_layers=num_layers,
        pp_size=3,
        num_layers_in_first=num_layers_in_first,
        num_layers_in_last=num_layers_in_last,
        device_caps=device_caps,
        enable_hetero_rebalance=enable_hetero_rebalance,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )

    # --- Test 1: equal split (no uneven-PP config) ---
    solver = build_default_des_loc_solver(num_layers=24)
    layers = ["mamba"] * 16 + ["attention"] * 8
    descs = solver.solve(layers)
    total = sum(d.num_layers for d in descs)
    assert total == 24, f"equal-split total mismatch: {total}"
    logger.info("Test 1 passed: equal split  %s",
                [(d.pp_rank, d.layer_offset, d.num_layers) for d in descs])

    # --- Test 2: uneven PP (first=6, last=8, middle=10) ---
    solver2 = build_default_des_loc_solver(
        num_layers=24, num_layers_in_first=6, num_layers_in_last=8
    )
    descs2 = solver2.solve(layers)
    assert descs2[0].num_layers == 6,  f"first stage: {descs2[0].num_layers}"
    assert descs2[2].num_layers == 8,  f"last  stage: {descs2[2].num_layers}"
    assert sum(d.num_layers for d in descs2) == 24
    logger.info("Test 2 passed: uneven-PP %s",
                [(d.pp_rank, d.layer_offset, d.num_layers) for d in descs2])

    # --- Test 3: H100 last stage absorbs more middle layers when rebalanced ---
    solver3 = build_default_des_loc_solver(
        num_layers=30, num_layers_in_first=6, num_layers_in_last=None,
        enable_hetero_rebalance=True,
    )
    layers3 = ["mamba"] * 30
    descs3 = solver3.solve(layers3)
    # rank 2 is H100 (SM90, weight 2.7) → should have more layers than rank 1 (SM86)
    assert descs3[2].num_layers >= descs3[1].num_layers, (
        f"H100 should have ≥ layers than A6000: "
        f"H100={descs3[2].num_layers}  A6000={descs3[1].num_layers}"
    )
    assert sum(d.num_layers for d in descs3) == 30
    logger.info("Test 3 passed: hetero rebalance  %s",
                [(d.pp_rank, d.device_cap.sm_version,
                  d.num_layers, f"{d.slc_budget_bytes/(1<<30):.2f}GB-SLC")
                 for d in descs3])

    # --- Test 4: SLC cache pin / fetch / release ---
    slc = SLCLayerCache(budget_bytes=4 * (1 << 30), pp_rank=0)
    t = torch.randn(1024, 1024)
    ok = slc.pin("act_layer0_mb0", t)
    assert ok, "SLC pin failed"
    fetched = slc.fetch("act_layer0_mb0", device=torch.device("cpu"))
    assert fetched is not None and fetched.shape == t.shape
    slc.release("act_layer0_mb0")
    assert slc.used_bytes == 0
    logger.info("Test 4 passed: SLC pin/fetch/release  stats=%s", slc.stats())

    # --- Test 5: compat shim returns Megatron-style (offset, list) ---
    offset, sel = solver2.select_layers_for_rank(0, layers)
    assert offset == 0 and len(sel) == 6
    logger.info("Test 5 passed: compat shim  offset=%d  len=%d", offset, len(sel))

    print("All smoke tests passed.")
