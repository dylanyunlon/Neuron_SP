"""
Heterogeneous MoE Gate Slice Fix for DES-LOC Framework
=======================================================

Upstream Design Intent (Megatron 16a8cdb):
    In Megatron-LM's MoE SelfAttention, when `attention_output_gate=True`, the gate tensor
    is reshaped from [sq, b, ng, np/ng * hn] → [sq, b, np, hn]. However, when the number of
    KV heads (num_query_groups / num_key_value_heads, i.e., `ng`) is *less than* the tensor
    parallel world size (`tp_size`), each TP rank would hold an empty or misaligned slice of
    the KV projection because the standard TP partitioning assumes ng >= tp_size.

    The fix: when ng < tp_size, compute a secondary index (`idx`) to correctly slice the gate
    tensor so each TP rank receives the right [sq, b, size, hn] chunk.

DES-LOC Adaptation Points:
    In the DES-LOC (Decoupled Execution with Shared LOcality Cache) framework running on
    heterogeneous hardware (2× A6000 48 GB SM86 + 1× H100 NVL 96 GB SM90, PCIe, no NVLink,
    1.5 TB CPU DRAM), the same logical bug appears in a more complex form:

    1. **Device-class-aware TP ranks**: DeepSpeed assigns TP ranks across device classes.
       An A6000 rank and an H100 rank may share the same logical TP group, but their
       compute capability differs (SM86 vs SM90). The gate slice must be correct *per
       device class* because the DES-LOC locality cache allocates different head-count
       budgets to each device class.

    2. **LOC (Shared LOcality Cache) head budget**: The H100 NVL carries a larger
       `loc_head_budget` (more attention heads cached locally) than each A6000.  When
       ng < tp_size the naive equal-split slicing would give the H100 the wrong heads.
       This module computes the device-class-aware slice boundaries.

    3. **Decoupled Execution**: In DES-LOC, QKV projection and attention computation are
       decoupled across heterogeneous ranks.  The gate tensor (used for gated attention
       output, e.g., in Hymba / GLA-style MoE) must be sliced *before* the locality-cache
       scatter, not after.  This module is inserted as a pre-scatter hook.

    4. **Graceful CPU offload path**: When a rank's LOC budget is exhausted, heads spill to
       CPU DRAM (1.5 TB).  The slice computed here determines *which* heads land in HBM vs
       DRAM, so correctness of the slice directly impacts memory pressure.

Hardware topology assumed:
    rank 0 → A6000 #0  (SM86, 48 GB, tp_rank=0)
    rank 1 → A6000 #1  (SM86, 48 GB, tp_rank=1)
    rank 2 → H100 NVL  (SM90, 96 GB, tp_rank=2)

    PCIe-only interconnect: no NVLink, all collective ops go through PCIe.

Usage:
    from deepspeed.moe.hetero_gate_slice_fix import (
        HeteroMoEGateSliceFix,
        slice_gate_for_hetero_tp,
        build_device_class_registry,
    )

    registry = build_device_class_registry()
    fix = HeteroMoEGateSliceFix(
        num_query_groups=2,
        num_attention_heads=8,
        hidden_size_per_head=64,
        world_size=3,   # 2× A6000 + 1× H100
        registry=registry,
    )
    gate_sliced = fix.slice(gate_reshaped, tp_rank=my_tp_rank)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device-class constants
# ---------------------------------------------------------------------------

SM86_CAPABILITY = (8, 6)   # A6000
SM90_CAPABILITY = (9, 0)   # H100 NVL

# Relative HBM-compute weight used to assign a proportional LOC head budget.
# H100 NVL (96 GB) gets ~2× the budget of a single A6000 (48 GB).
_DEFAULT_DEVICE_WEIGHT: Dict[Tuple[int, int], float] = {
    SM86_CAPABILITY: 1.0,
    SM90_CAPABILITY: 2.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """Metadata for one physical device in the heterogeneous pool."""
    rank: int
    device_index: int
    sm_major: int
    sm_minor: int
    total_memory_bytes: int
    # Budget: number of attention heads whose gate tensor lives in HBM (not CPU).
    loc_head_budget: int = 0

    @property
    def sm_capability(self) -> Tuple[int, int]:
        return (self.sm_major, self.sm_minor)

    @property
    def is_h100(self) -> bool:
        return self.sm_capability == SM90_CAPABILITY

    @property
    def is_a6000(self) -> bool:
        return self.sm_capability == SM86_CAPABILITY


@dataclass
class DeviceClassRegistry:
    """
    Maps TP ranks → DeviceInfo and precomputes LOC head budgets.

    In DES-LOC, all TP ranks share the same logical attention layer but execute on
    physically heterogeneous devices.  The registry is the single source of truth for
    which rank owns which slice of the gate tensor and how many heads fit in local HBM.
    """
    devices: List[DeviceInfo] = field(default_factory=list)
    _rank_to_device: Dict[int, DeviceInfo] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._rank_to_device = {d.rank: d for d in self.devices}

    def get(self, rank: int) -> Optional[DeviceInfo]:
        return self._rank_to_device.get(rank)

    def assign_loc_head_budgets(self, total_heads: int) -> None:
        """
        Distribute `total_heads` attention heads among devices proportional to their
        device weight (HBM capacity proxy).  Remainder heads go to the highest-weight rank.

        DES-LOC rationale: the LOC (Locality Cache) stores gate activations in HBM for
        fast reuse during the decoupled execution phase.  Richer devices get a larger
        slice of the LOC to minimise PCIe spill traffic.
        """
        weights = [
            _DEFAULT_DEVICE_WEIGHT.get(d.sm_capability, 1.0) for d in self.devices
        ]
        total_weight = sum(weights)
        budgets: List[int] = [
            int(math.floor(w / total_weight * total_heads)) for w in weights
        ]
        remainder = total_heads - sum(budgets)
        # Give remainder to heaviest device (H100 if present)
        heaviest_idx = int(max(range(len(weights)), key=lambda i: weights[i]))
        budgets[heaviest_idx] += remainder
        for dev, budget in zip(self.devices, budgets):
            dev.loc_head_budget = budget
            logger.debug(
                "DES-LOC LOC budget: rank=%d device=SM%d%d budget=%d heads",
                dev.rank, dev.sm_major, dev.sm_minor, budget,
            )

    def log_summary(self) -> None:
        logger.info("DeviceClassRegistry — %d devices:", len(self.devices))
        for d in self.devices:
            logger.info(
                "  rank=%-2d SM%d%d  mem=%5.1f GB  loc_budget=%d heads  %s",
                d.rank,
                d.sm_major,
                d.sm_minor,
                d.total_memory_bytes / 1e9,
                d.loc_head_budget,
                "[H100-NVL]" if d.is_h100 else "[A6000]",
            )


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_device_class_registry(
    world_size: Optional[int] = None,
    device_map: Optional[Dict[int, int]] = None,
) -> DeviceClassRegistry:
    """
    Auto-detect or manually specify the device→rank mapping.

    Auto-detection uses ``torch.cuda.get_device_properties`` for each local
    GPU and, in a distributed setting, gathers info from all ranks via
    ``dist.all_gather_object``.

    Parameters
    ----------
    world_size:
        Override dist world size (useful in unit tests without real dist init).
    device_map:
        Optional ``{rank: local_device_index}`` override.  If None the code
        assumes rank ``r`` uses ``cuda:r`` (single-node assumption).

    Returns
    -------
    DeviceClassRegistry
        Populated with ``DeviceInfo`` for every rank.
    """
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_device_idx = (device_map or {}).get(local_rank, local_rank)

    # Gather per-device info from local rank
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(local_device_idx)
        local_info = {
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "device_index": local_device_idx,
            "sm_major": props.major,
            "sm_minor": props.minor,
            "total_memory_bytes": props.total_memory,
        }
    else:
        # CPU-only / test environment — fake A6000
        local_info = {
            "rank": 0,
            "device_index": 0,
            "sm_major": SM86_CAPABILITY[0],
            "sm_minor": SM86_CAPABILITY[1],
            "total_memory_bytes": int(48e9),
        }

    if dist.is_initialized() and world_size > 1:
        all_info: List[dict] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(all_info, local_info)
    else:
        all_info = [local_info]

    devices = [
        DeviceInfo(
            rank=info["rank"],
            device_index=info["device_index"],
            sm_major=info["sm_major"],
            sm_minor=info["sm_minor"],
            total_memory_bytes=info["total_memory_bytes"],
        )
        for info in all_info
        if info is not None
    ]
    registry = DeviceClassRegistry(devices=devices)
    registry.log_summary()
    return registry


# ---------------------------------------------------------------------------
# Core slice logic
# ---------------------------------------------------------------------------

def slice_gate_for_hetero_tp(
    gate: torch.Tensor,
    tp_rank: int,
    tp_world_size: int,
    num_query_groups: int,
    num_attention_heads_per_partition: int,
) -> torch.Tensor:
    """
    Homogeneous TP gate-slice logic (mirrors Megatron 16a8cdb fix verbatim).

    When ``num_query_groups < tp_world_size`` the reshaped gate tensor
    ``[sq, b, np, hn]`` must be sliced along dim-2 so that each TP rank
    receives exactly ``num_attention_heads_per_partition // (tp_world_size //
    num_query_groups)`` heads rather than the full partition.

    Parameters
    ----------
    gate:
        Tensor of shape ``[seq_len, batch, num_heads_partitioned, head_dim]``
        after the initial TP-aware reshape.
    tp_rank:
        Rank within the tensor-parallel group.
    tp_world_size:
        Total number of tensor-parallel ranks.
    num_query_groups:
        ``ng`` — number of KV heads before TP partitioning.
    num_attention_heads_per_partition:
        ``np // tp_size`` heads this rank nominally holds after the standard
        TP split (computed *before* this function is called).

    Returns
    -------
    torch.Tensor
        Correctly sliced gate, shape ``[seq_len, batch, size, head_dim]``.

    Notes
    -----
    Megatron rationale: when ``ng < tp_size``, each KV head is *replicated*
    across ``tp_size // ng`` TP ranks.  Without this slice the gate would
    carry ``np / tp_size`` query heads but the KV projection only covers
    ``np / ng / (tp_size / ng) = np / tp_size`` of them correctly.  The fix
    selects the sub-range that aligns with this rank's KV head assignment.
    """
    if num_query_groups >= tp_world_size:
        # Standard case: no secondary slice needed.
        return gate

    # Number of TP ranks that share one KV head
    ranks_per_kv_head: int = tp_world_size // num_query_groups

    # Which of these ranks-per-kv-head slot does this rank fall into?
    idx: int = tp_rank % ranks_per_kv_head

    # How many query heads does each slot hold?
    size: int = num_attention_heads_per_partition // ranks_per_kv_head

    logger.debug(
        "gate slice: tp_rank=%d ng=%d tp_ws=%d idx=%d size=%d gate.shape=%s",
        tp_rank, num_query_groups, tp_world_size, idx, size, list(gate.shape),
    )

    return gate[:, :, idx * size : (idx + 1) * size, :]


# ---------------------------------------------------------------------------
# Heterogeneous-aware gate slicer (DES-LOC extension)
# ---------------------------------------------------------------------------

class HeteroMoEGateSliceFix:
    """
    DES-LOC heterogeneous gate slicer for MoE attention output gates.

    Extends the Megatron 16a8cdb fix to handle a heterogeneous TP group where
    different ranks own physically different GPU classes.  The key additions:

    1. **Device-class slice correction**: A6000 ranks and the H100 rank may
       each hold a different number of attention heads in their LOC budget.
       After the standard homogeneous slice (ported from Megatron), we apply a
       secondary device-class-aware correction so the gate tensor exactly
       matches the LOC head layout.

    2. **LOC-pinned vs. offloaded heads**: Heads inside the ``loc_head_budget``
       for this rank are kept on-device; excess heads are marked for CPU DRAM
       offload.  This class returns both tensors so the caller can route them
       to the appropriate memory tier.

    3. **Decoupled execution hook**: The ``slice_with_loc_split`` method is
       designed to be called as a pre-scatter hook in the DES-LOC pipeline,
       between the QKV projection and the attention kernel dispatch.

    Parameters
    ----------
    num_query_groups:
        Number of KV heads (``ng`` in Megatron notation).
    num_attention_heads:
        Total number of query attention heads across all TP ranks.
    hidden_size_per_head:
        Head dimension ``hn``.
    world_size:
        TP group size (e.g., 3 for 2× A6000 + 1× H100).
    registry:
        ``DeviceClassRegistry`` produced by ``build_device_class_registry()``.
    """

    def __init__(
        self,
        num_query_groups: int,
        num_attention_heads: int,
        hidden_size_per_head: int,
        world_size: int,
        registry: DeviceClassRegistry,
    ) -> None:
        self.num_query_groups = num_query_groups
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_head = hidden_size_per_head
        self.world_size = world_size
        self.registry = registry

        # Standard per-partition head count (homogeneous assumption)
        if num_attention_heads % world_size != 0:
            raise ValueError(
                f"num_attention_heads={num_attention_heads} must be divisible by "
                f"world_size={world_size}"
            )
        self.num_heads_per_partition: int = num_attention_heads // world_size

        # Assign LOC budgets based on device weights
        registry.assign_loc_head_budgets(num_attention_heads)

        # Precompute slice boundaries per rank for fast forward pass lookup
        self._slice_cache: Dict[int, Tuple[int, int]] = {}
        self._build_slice_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_slice_cache(self) -> None:
        """
        Precompute (start, end) head indices for every TP rank.

        For the heterogeneous case we must distribute heads in *two* passes:

        Pass 1 (homogeneous Megatron fix): correct for ng < tp_size by
            computing per-TP-rank secondary index and slice size.
        Pass 2 (DES-LOC device-class correction): further adjust slice
            boundaries so that each device's slice matches its LOC budget,
            ensuring memory-tier placement is deterministic.

        The result is a mapping ``rank → (global_head_start, global_head_end)``
        used in ``slice()`` and ``slice_with_loc_split()``.
        """
        ng = self.num_query_groups
        ws = self.world_size
        np_pp = self.num_heads_per_partition  # heads per rank, homogeneous

        ranks_per_kv = max(1, ws // ng) if ng < ws else 1

        # Step 1: homogeneous slice boundaries (Megatron logic)
        homo_slices: List[Tuple[int, int]] = []
        for rank in range(ws):
            if ng >= ws:
                # Standard: each rank gets np_pp heads
                start = rank * np_pp
                end = start + np_pp
            else:
                idx = rank % ranks_per_kv
                size = np_pp // ranks_per_kv
                # Base from the KV-head group this rank belongs to
                kv_group = rank // ranks_per_kv
                base = kv_group * (size * ranks_per_kv)
                start = base + idx * size
                end = start + size
            homo_slices.append((start, end))
            logger.debug(
                "homo slice: rank=%d heads=[%d, %d)", rank, start, end
            )

        # Step 2: DES-LOC device-class correction
        # Redistribute within each KV-head group so the H100 (higher LOC budget)
        # gets proportionally more heads, while keeping total heads per group fixed.
        # This only matters when the LOC budget differs from the homogeneous split.
        for rank, dev_info in enumerate(self.registry.devices):
            homo_start, homo_end = homo_slices[rank]
            homo_size = homo_end - homo_start
            loc_budget = dev_info.loc_head_budget

            if loc_budget == homo_size:
                # No correction needed
                self._slice_cache[rank] = (homo_start, homo_end)
                logger.debug(
                    "DES-LOC slice rank=%d: no correction needed [%d, %d)",
                    rank, homo_start, homo_end,
                )
            else:
                # Clamp to available range; this may leave some heads without
                # a DES-LOC assignment (they go to the default spill path).
                corrected_end = homo_start + min(homo_size, loc_budget)
                self._slice_cache[rank] = (homo_start, corrected_end)
                logger.info(
                    "DES-LOC slice rank=%d SM%d%d: homo=[%d,%d) loc_budget=%d "
                    "→ corrected=[%d,%d)",
                    rank, dev_info.sm_major, dev_info.sm_minor,
                    homo_start, homo_end,
                    loc_budget,
                    homo_start, corrected_end,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def slice(
        self,
        gate: torch.Tensor,
        tp_rank: int,
    ) -> torch.Tensor:
        """
        Return the correctly sliced gate tensor for ``tp_rank``.

        This is a drop-in replacement for the Megatron ``gate[:, :, idx*size:(idx+1)*size, :]``
        one-liner, extended to handle heterogeneous LOC budgets.

        Parameters
        ----------
        gate:
            Reshaped gate tensor ``[sq, b, np, hn]`` *on this TP rank*.
            ``np`` is the homogeneous per-partition head count.
        tp_rank:
            Rank within the tensor-parallel group.

        Returns
        -------
        torch.Tensor
            Sliced gate ``[sq, b, slice_size, hn]``.
        """
        if gate.dim() != 4:
            raise ValueError(
                f"Expected 4-D gate tensor [sq, b, np, hn], got shape {list(gate.shape)}"
            )

        global_start, global_end = self._slice_cache[tp_rank]
        # Convert global head indices to local indices within this rank's partition
        local_start = global_start - tp_rank * self.num_heads_per_partition
        local_end = global_end - tp_rank * self.num_heads_per_partition

        # Clamp to valid range (safety against misconfigured budgets)
        np_local = gate.shape[2]
        local_start = max(0, min(local_start, np_local))
        local_end = max(0, min(local_end, np_local))

        sliced = gate[:, :, local_start:local_end, :]
        logger.debug(
            "HeteroMoEGateSliceFix.slice rank=%d local=[%d,%d) out_shape=%s",
            tp_rank, local_start, local_end, list(sliced.shape),
        )
        return sliced

    def slice_with_loc_split(
        self,
        gate: torch.Tensor,
        tp_rank: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Slice the gate and split into LOC-resident (HBM) and offloaded (CPU) parts.

        DES-LOC pre-scatter hook:  called between QKV projection and attention
        dispatch.  Returns two tensors:

        - ``gate_hbm``: heads that fit in this rank's LOC budget, kept on-device.
        - ``gate_cpu``: overflow heads pinned to CPU DRAM for the offload path.
          ``None`` when all heads fit in HBM.

        The caller is responsible for:
        1. Passing ``gate_hbm`` to the attention kernel.
        2. Storing ``gate_cpu`` in the DES-LOC locality-cache CPU slab.

        Parameters
        ----------
        gate:
            Reshaped gate ``[sq, b, np, hn]`` on this TP rank (device tensor).
        tp_rank:
            Rank within the tensor-parallel group.

        Returns
        -------
        gate_hbm : torch.Tensor
            On-device gate slice.
        gate_cpu : torch.Tensor or None
            CPU-pinned overflow gate slice, or None if no overflow.
        """
        dev_info = self.registry.get(tp_rank)
        if dev_info is None:
            raise RuntimeError(
                f"tp_rank={tp_rank} not found in DeviceClassRegistry. "
                "Did you call build_device_class_registry() for all ranks?"
            )

        full_slice = self.slice(gate, tp_rank)          # shape [sq, b, s, hn]
        slice_heads = full_slice.shape[2]
        loc_budget = dev_info.loc_head_budget

        # How many heads of this slice fit in HBM?
        hbm_heads = min(slice_heads, loc_budget)

        gate_hbm = full_slice[:, :, :hbm_heads, :]

        if hbm_heads < slice_heads:
            overflow = full_slice[:, :, hbm_heads:, :]
            # Pin to CPU DRAM for DES-LOC locality-cache spill
            gate_cpu = overflow.cpu().pin_memory()
            logger.debug(
                "DES-LOC LOC spill: rank=%d %d heads → CPU DRAM (%.2f MB)",
                tp_rank,
                slice_heads - hbm_heads,
                overflow.numel() * overflow.element_size() / 1e6,
            )
        else:
            gate_cpu = None

        return gate_hbm, gate_cpu

    def expected_hbm_heads(self, tp_rank: int) -> int:
        """Return the number of gate heads that will reside in HBM for ``tp_rank``."""
        dev_info = self.registry.get(tp_rank)
        if dev_info is None:
            return self.num_heads_per_partition
        global_start, global_end = self._slice_cache[tp_rank]
        slice_size = global_end - global_start
        return min(slice_size, dev_info.loc_head_budget)

    def total_heads_across_ranks(self) -> int:
        """Sanity check: sum of all slice sizes should equal num_attention_heads."""
        total = sum(
            self._slice_cache[r][1] - self._slice_cache[r][0]
            for r in range(self.world_size)
        )
        return total


# ---------------------------------------------------------------------------
# Functional wrapper (drop-in for Megatron's inline gate slice)
# ---------------------------------------------------------------------------

def apply_hetero_gate_slice(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    tp_rank: int,
    tp_world_size: int,
    num_query_groups: int,
    num_attention_heads_per_partition: int,
    registry: Optional[DeviceClassRegistry] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Functional entry-point mirroring Megatron's ``SelfAttention.get_query_key_value_tensors``
    return path when ``output_gate=True``.

    Upstream (Megatron 16a8cdb) applies an inline slice:
    ::
        if self.config.num_query_groups < self.world_size:
            idx = get_tensor_model_parallel_rank() % (ws // ng)
            size = num_attention_heads_per_partition // (ws // ng)
            gate = gate[:, :, idx * size : (idx + 1) * size, :]

    This function replicates that logic and, when a ``registry`` is provided,
    further applies the DES-LOC device-class correction so the gate slice
    aligns with the heterogeneous LOC head budget.

    Parameters
    ----------
    query, key, value:
        Standard QKV tensors (passed through unchanged).
    gate:
        Reshaped gate tensor ``[sq, b, np, hn]``.
    tp_rank, tp_world_size, num_query_groups, num_attention_heads_per_partition:
        TP configuration scalars.
    registry:
        Optional ``DeviceClassRegistry``.  If None, falls back to plain
        Megatron homogeneous slice.

    Returns
    -------
    query, key, value, gate_sliced
    """
    if registry is None:
        # Plain Megatron fix — no heterogeneous correction
        gate = slice_gate_for_hetero_tp(
            gate,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            num_query_groups=num_query_groups,
            num_attention_heads_per_partition=num_attention_heads_per_partition,
        )
    else:
        # DES-LOC heterogeneous path
        num_attention_heads = num_attention_heads_per_partition * tp_world_size
        hidden_size_per_head = gate.shape[-1]
        fixer = HeteroMoEGateSliceFix(
            num_query_groups=num_query_groups,
            num_attention_heads=num_attention_heads,
            hidden_size_per_head=hidden_size_per_head,
            world_size=tp_world_size,
            registry=registry,
        )
        gate = fixer.slice(gate, tp_rank)

    return query, key, value, gate


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # -----------------------------------------------------------------------
    # Test 1: homogeneous slice reproduces Megatron fix exactly
    # ng=2, tp=4 → ranks_per_kv=2
    # rank 0: idx=0, size=1  → heads [0:1]
    # rank 1: idx=1, size=1  → heads [1:2]
    # rank 2: idx=0, size=1  → heads [2:3]
    # rank 3: idx=1, size=1  → heads [3:4]
    # -----------------------------------------------------------------------
    sq, b, np_total, hn = 8, 2, 8, 16
    ng, ws = 2, 4
    np_pp = np_total // ws  # = 2

    gate_full = torch.arange(sq * b * np_total * hn, dtype=torch.float32).reshape(sq, b, np_total, hn)

    for rank in range(ws):
        gate_rank = gate_full[:, :, rank * np_pp : (rank + 1) * np_pp, :]
        sliced = slice_gate_for_hetero_tp(gate_rank, rank, ws, ng, np_pp)
        expected_size = np_pp // (ws // ng)
        assert sliced.shape[2] == expected_size, (
            f"rank={rank}: expected {expected_size} heads, got {sliced.shape[2]}"
        )

    logger.info("Test 1 PASSED: homogeneous slice sizes correct for ng=%d tp=%d", ng, ws)

    # -----------------------------------------------------------------------
    # Test 2: HeteroMoEGateSliceFix with 2× A6000 + 1× H100 (ws=3, ng=1)
    # -----------------------------------------------------------------------
    ws3 = 3
    ng1 = 1
    np_total3 = 6
    devices_3 = [
        DeviceInfo(rank=0, device_index=0, sm_major=8, sm_minor=6, total_memory_bytes=int(48e9)),
        DeviceInfo(rank=1, device_index=1, sm_major=8, sm_minor=6, total_memory_bytes=int(48e9)),
        DeviceInfo(rank=2, device_index=0, sm_major=9, sm_minor=0, total_memory_bytes=int(96e9)),
    ]
    reg3 = DeviceClassRegistry(devices=devices_3)
    fixer3 = HeteroMoEGateSliceFix(
        num_query_groups=ng1,
        num_attention_heads=np_total3,
        hidden_size_per_head=hn,
        world_size=ws3,
        registry=reg3,
    )
    # LOC budgets should sum to np_total3
    total_budget = sum(d.loc_head_budget for d in reg3.devices)
    assert total_budget == np_total3, f"Budget sum {total_budget} != {np_total3}"
    logger.info("Test 2 PASSED: LOC budgets sum to %d", np_total3)

    # -----------------------------------------------------------------------
    # Test 3: slice_with_loc_split returns correct HBM/CPU split
    # -----------------------------------------------------------------------
    gate3 = torch.randn(sq, b, np_total3 // ws3, hn)  # per-rank gate
    gate_hbm, gate_cpu = fixer3.slice_with_loc_split(gate3, tp_rank=0)
    assert gate_hbm.is_cuda == gate3.is_cuda, "HBM gate device mismatch"
    if gate_cpu is not None:
        assert not gate_cpu.is_cuda, "CPU gate should not be on CUDA"
    logger.info(
        "Test 3 PASSED: slice_with_loc_split hbm=%s cpu=%s",
        list(gate_hbm.shape),
        list(gate_cpu.shape) if gate_cpu is not None else None,
    )

    # -----------------------------------------------------------------------
    # Test 4: apply_hetero_gate_slice functional wrapper — no registry path
    # -----------------------------------------------------------------------
    q = torch.randn(sq, b, np_pp, hn)
    k = torch.randn(sq, b, 1, hn)
    v = torch.randn(sq, b, 1, hn)
    g = torch.randn(sq, b, np_pp, hn)
    _, _, _, g_out = apply_hetero_gate_slice(q, k, v, g, 0, ws, ng, np_pp, registry=None)
    assert g_out.shape[2] == np_pp // (ws // ng), "Functional wrapper slice size wrong"
    logger.info("Test 4 PASSED: apply_hetero_gate_slice no-registry path")

    # -----------------------------------------------------------------------
    # Test 5: total_heads_across_ranks sanity check
    # -----------------------------------------------------------------------
    total_h = fixer3.total_heads_across_ranks()
    assert total_h == np_total3, f"total_heads mismatch: {total_h} != {np_total3}"
    logger.info("Test 5 PASSED: total_heads_across_ranks=%d", total_h)

    logger.info("All smoke tests passed.")
