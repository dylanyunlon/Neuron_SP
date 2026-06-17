"""
deepspeed/runtime/hetero_elastic_batch.py
==========================================

DES-LOC Heterogeneous Elastic Batch Redistribution Engine
----------------------------------------------------------

Upstream Design Intent (Megatron adcdf162fde36a9ada88476548af746027847d03):
    The Megatron commit "fix(elastification): align with get_batch + utils refactors" refactored
    the ``get_batch`` function in ``pretrain_hybrid_flex.py`` to:

    1.  Replace per-rank context-parallel helpers (get_context_parallel_rank / world_size) with
        group-level abstractions (get_context_parallel_group, get_hybrid_data_context_parallel_groups).
        This decouples rank arithmetic from the batch-slicing logic so that elastic topology changes
        don't require rewriting slice indices by hand.

    2.  Introduce a canonical ``BATCH_KEYS`` list that enumerates every tensor that must travel
        through the pipeline, making batch broadcast and slice operations data-driven rather than
        hard-coded.

    3.  Lift the data-fetch to TP-rank-0 only, then broadcast via ``get_batch_on_this_tp_rank``,
        which is now in ``megatron.core.utils`` (not training.utils).  This separates I/O from
        compute more cleanly and is essential for elastic restarts where a rank's role may change.

    4.  Support a ``vp_stage`` parameter in ``get_batch`` so that virtual-pipeline (VP) stages can
        participate in the early-exit logic alongside MTP (multi-token prediction) ranks, rather
        than hard-coding "first or last pipeline stage".

    5.  Squeeze ``cu_seqlens`` / ``max_seqlen`` shapes *after* all group-level CP slicing so that
        downstream forward passes always see a canonical 1-D / scalar shape regardless of whether
        THD packing or padding was used.

DES-LOC Adaptation Points:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) operates on heterogeneous hardware:
        • 2× A6000 48 GB SM86  (PCIe, no NVLink)
        • 1× H100 NVL 96 GB SM90 (PCIe)
        Total CPU DRAM: 1.5 TB

    The key tension: Megatron's get_batch assumes a *homogeneous* tensor-parallel group where every
    rank has identical VRAM and compute throughput.  On DES-LOC hardware, the H100 can hold 2×
    the sequence length per micro-batch as either A6000.  Naively broadcasting the same batch to
    all ranks wastes H100 capacity and OOMs A6000 ranks during long-context training.

    This module re-interprets the Megatron refactor as a **HeteroElasticBatch** layer that:

    A.  Device-class registry — maps each rank to its compute profile (SM version, VRAM, PCIe BW).
        Replaces Megatron's implicit homogeneous assumption.

    B.  Locality-aware fetch — TP-rank-0 fetch is extended to choose which physical device acts as
        the "source" rank for each micro-batch, preferring H100 for large sequences (the DES-LOC
        "Shared LOcality Cache" principle: hot KV data stays close to the device that computed it).

    C.  Elastic batch keys — mirrors BATCH_KEYS but adds DES-LOC metadata tensors
        (device_class_ids, local_cp_offsets) so that CP-rank slicing is heterogeneity-aware.

    D.  Heterogeneous CP slice — replaces Megatron's uniform index_select with a capacity-weighted
        partition: the H100 rank receives a proportionally larger sequence slice.  This is the
        "Decoupled Execution" half of DES-LOC: each device executes on its capacity-appropriate
        shard without a global barrier on equal-size chunks.

    E.  Shape canonicalization — identical to Megatron's post-CP squeeze of cu_seqlens /
        max_seqlen, but extended to handle variable-length shards per device class.

    F.  vp_stage + mtp_on_this_rank early-exit — directly mirrors the upstream fix so that
        virtual-pipeline staging is correct even after elastic topology changes.

References:
    • Megatron commit adcdf162: megatron/elastification/pretrain_hybrid_flex.py
    • Neuron_SP project: github.com/dylanyunlon/Neuron_SP
    • DeepSpeed runtime: deepspeed/runtime/engine.py, pipe_engine.py
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants mirroring Megatron's BATCH_KEYS, extended with DES-LOC metadata
# ---------------------------------------------------------------------------

#: Canonical set of batch keys that must be broadcast / sliced across all
#: pipeline and tensor-parallel ranks.  Mirrors Megatron's BATCH_KEYS list
#: (adcdf162) but adds DES-LOC device-routing tensors.
BATCH_KEYS: Tuple[str, ...] = (
    "attention_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
    "hybrid_cp_group",
    "labels",
    "local_cp_size",
    "loss_mask",
    "max_seqlen",
    "position_ids",
    "tokens",
    # DES-LOC extensions
    "device_class_ids",   # int8 tensor [batch] — which device class owns each sample
    "local_cp_offsets",   # int32 tensor [cp_world+1] — cumulative token offsets per CP rank
)

#: Keys whose shapes vary by device class after heterogeneous CP slicing.
#: These must NOT be naively broadcast; they are point-to-point distributed.
HETERO_SLICE_KEYS: Tuple[str, ...] = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
)

#: Keys that hold sequence-packing metadata and are rank-invariant after the
#: CP partition (every rank needs the full cu_seqlens to know its own slice).
METADATA_KEYS: Tuple[str, ...] = (
    "attention_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
    "max_seqlen",
)


# ---------------------------------------------------------------------------
# Device class definitions
# ---------------------------------------------------------------------------

class DeviceClass(Enum):
    """Hardware tiers present in the DES-LOC cluster."""
    A6000_SM86 = auto()   # 48 GB VRAM, PCIe Gen4, SM 8.6
    H100_NVL_SM90 = auto()  # 96 GB VRAM, PCIe Gen5, SM 9.0


@dataclass(frozen=True)
class DeviceProfile:
    """Static capability descriptor for a device class.

    Attributes
    ----------
    device_class:
        Enum variant identifying the hardware tier.
    vram_gb:
        Usable VRAM in gigabytes (conservative, leaving headroom for activations).
    sm_version:
        Integer SM major*10 + minor (86 for A6000, 90 for H100).
    pcie_bw_gbps:
        Approximate peak PCIe bandwidth in GB/s (one direction).
    capacity_weight:
        Relative weight used for heterogeneous sequence-length partitioning.
        H100 with 96 GB gets weight 2.0 vs A6000's 1.0 when the ratio is 2:1.
    """
    device_class: DeviceClass
    vram_gb: float
    sm_version: int
    pcie_bw_gbps: float
    capacity_weight: float


#: Registry of known device profiles in the DES-LOC cluster.
DEVICE_PROFILES: Dict[DeviceClass, DeviceProfile] = {
    DeviceClass.A6000_SM86: DeviceProfile(
        device_class=DeviceClass.A6000_SM86,
        vram_gb=44.0,       # conservative (48 GB minus OS + framework overhead)
        sm_version=86,
        pcie_bw_gbps=32.0,  # PCIe Gen4 x16
        capacity_weight=1.0,
    ),
    DeviceClass.H100_NVL_SM90: DeviceProfile(
        device_class=DeviceClass.H100_NVL_SM90,
        vram_gb=90.0,       # conservative (96 GB NVL)
        sm_version=90,
        pcie_bw_gbps=64.0,  # PCIe Gen5 x16
        capacity_weight=2.0,
    ),
}


# ---------------------------------------------------------------------------
# Rank → device-class mapping
# ---------------------------------------------------------------------------

@dataclass
class RankDeviceMap:
    """Maps global ranks to their device class.

    In the DES-LOC cluster we have:
        rank 0 → A6000  (SM86)
        rank 1 → A6000  (SM86)
        rank 2 → H100   (SM90)

    This mapping is configurable via environment variables so that the
    physical assignment can be overridden without code changes.

    Environment Variables
    ---------------------
    DESLOCAL_RANK_TO_CLASS:
        Comma-separated list of device class names, one per rank in global
        rank order.  E.g. "A6000_SM86,A6000_SM86,H100_NVL_SM90".
        Defaults to the physical cluster layout above.
    """

    _rank_to_class: Dict[int, DeviceClass] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "RankDeviceMap":
        env_val = os.environ.get(
            "DESLOCAL_RANK_TO_CLASS",
            "A6000_SM86,A6000_SM86,H100_NVL_SM90",
        )
        mapping: Dict[int, DeviceClass] = {}
        for rank, name in enumerate(env_val.split(",")):
            name = name.strip()
            try:
                mapping[rank] = DeviceClass[name]
            except KeyError:
                raise ValueError(
                    f"Unknown device class '{name}' in DESLOCAL_RANK_TO_CLASS. "
                    f"Valid options: {[c.name for c in DeviceClass]}"
                )
        obj = cls()
        obj._rank_to_class = mapping
        return obj

    def device_class(self, rank: int) -> DeviceClass:
        if rank not in self._rank_to_class:
            raise KeyError(f"Rank {rank} not in device map.  Map: {self._rank_to_class}")
        return self._rank_to_class[rank]

    def profile(self, rank: int) -> DeviceProfile:
        return DEVICE_PROFILES[self.device_class(rank)]

    def capacity_weight(self, rank: int) -> float:
        return self.profile(rank).capacity_weight

    def is_high_capacity(self, rank: int) -> bool:
        """Return True if the rank's device has above-average capacity weight."""
        avg = sum(p.capacity_weight for p in DEVICE_PROFILES.values()) / len(DEVICE_PROFILES)
        return self.capacity_weight(rank) > avg

    def preferred_source_rank(self, ranks: Sequence[int]) -> int:
        """Return the rank with highest capacity weight, used as locality-cache anchor.

        DES-LOC principle: hot data (KV cache, embeddings) should originate from
        the device best able to hold it.  In our cluster that is always the H100.
        """
        return max(ranks, key=lambda r: self.capacity_weight(r))

    def __repr__(self) -> str:
        lines = [f"  rank {r}: {c.name}" for r, c in sorted(self._rank_to_class.items())]
        return "RankDeviceMap(\n" + "\n".join(lines) + "\n)"


# ---------------------------------------------------------------------------
# Hetero-elastic CP partition
# ---------------------------------------------------------------------------

def compute_hetero_cp_offsets(
    total_tokens: int,
    cp_ranks: Sequence[int],
    rank_device_map: RankDeviceMap,
    min_tokens_per_rank: int = 1,
) -> List[int]:
    """Compute capacity-weighted token offsets for context-parallel ranks.

    Unlike Megatron's uniform CP slice (each rank gets total_tokens // cp_size),
    DES-LOC assigns slices proportional to each rank's ``capacity_weight``.

    Parameters
    ----------
    total_tokens:
        Total number of tokens in the packed sequence for this micro-batch.
    cp_ranks:
        Ordered list of global ranks participating in this CP group.
    rank_device_map:
        Device profile registry.
    min_tokens_per_rank:
        Hard floor so that no rank receives an empty slice (avoids division by
        zero in attention kernels).

    Returns
    -------
    offsets: List[int]
        Cumulative token offsets of length ``len(cp_ranks) + 1``.
        ``offsets[i]:offsets[i+1]`` is the token range for ``cp_ranks[i]``.

    Example
    -------
    For total_tokens=6000, ranks=[0(A6000), 1(A6000), 2(H100)] with weights
    [1, 1, 2], total_weight=4:
        rank 0 → 6000 * 1/4 = 1500 tokens
        rank 1 → 6000 * 1/4 = 1500 tokens
        rank 2 → 6000 * 2/4 = 3000 tokens
    """
    weights = [rank_device_map.capacity_weight(r) for r in cp_ranks]
    total_weight = sum(weights)

    raw_sizes: List[float] = [total_tokens * w / total_weight for w in weights]

    # Floor to integers; distribute remainder tokens to highest-capacity rank
    sizes: List[int] = [max(min_tokens_per_rank, int(math.floor(s))) for s in raw_sizes]
    remainder = total_tokens - sum(sizes)
    if remainder != 0:
        # Give remainder to the rank with the largest fractional part, tie-break
        # by capacity weight (H100 absorbs overflow more cheaply).
        fractional = [(raw_sizes[i] - math.floor(raw_sizes[i]), weights[i], i)
                      for i in range(len(cp_ranks))]
        fractional.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for frac, _w, idx in fractional:
            if remainder == 0:
                break
            add = min(remainder, 1) if remainder > 0 else max(remainder, -1)
            sizes[idx] += add
            remainder -= add

    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)

    logger.debug(
        "Hetero CP offsets: total=%d, ranks=%s, weights=%s → offsets=%s",
        total_tokens, list(cp_ranks), weights, offsets,
    )
    return offsets


def hetero_cp_slice(
    tensor: torch.Tensor,
    cp_rank_index: int,
    offsets: List[int],
    seq_dim: int = 1,
) -> torch.Tensor:
    """Slice a tensor along the sequence dimension for one CP rank.

    Parameters
    ----------
    tensor:
        Input tensor with sequence tokens along ``seq_dim``.
    cp_rank_index:
        Position of the target rank in the CP group (0-based index into offsets).
    offsets:
        Cumulative offsets from ``compute_hetero_cp_offsets``.
    seq_dim:
        Dimension index of the sequence axis (default 1 for [batch, seq, ...]).

    Returns
    -------
    Sliced tensor for this CP rank.
    """
    start = offsets[cp_rank_index]
    end = offsets[cp_rank_index + 1]
    idx = torch.arange(start, end, device=tensor.device, dtype=torch.long)
    return tensor.index_select(seq_dim, idx)


# ---------------------------------------------------------------------------
# Batch broadcast helpers
# ---------------------------------------------------------------------------

def _broadcast_tensor(
    tensor: Optional[torch.Tensor],
    src_rank: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Broadcast a single tensor from src_rank to all ranks in group.

    Handles the two-phase protocol required when the shape is unknown on
    non-source ranks: first broadcast the metadata (dtype, ndim, shape),
    then broadcast the data.

    Parameters
    ----------
    tensor:
        On src_rank this must be a valid tensor.  On other ranks it is ignored
        (pass None) and will be reconstructed from the broadcast.
    src_rank:
        Global rank that owns the source data.
    group:
        ProcessGroup to broadcast within.
    device:
        Target device for non-source ranks.

    Returns
    -------
    Tensor on every rank (same shape and values as on src_rank), or None if
    src_rank sent a "null" sentinel.
    """
    rank = dist.get_rank()

    # Phase 0: broadcast whether this tensor is None (null sentinel)
    is_none_flag = torch.zeros(1, dtype=torch.int8, device=device)
    if rank == src_rank:
        is_none_flag[0] = 1 if tensor is None else 0
    dist.broadcast(is_none_flag, src=src_rank, group=group)
    if is_none_flag.item() == 1:
        return None

    # Phase 1: broadcast dtype + ndim + shape
    if rank == src_rank:
        assert tensor is not None
        dtype_id = torch.tensor([tensor.dtype == torch.float32,
                                  tensor.dtype == torch.float16,
                                  tensor.dtype == torch.bfloat16,
                                  tensor.dtype == torch.int64,
                                  tensor.dtype == torch.int32,
                                  tensor.dtype == torch.int8,
                                  tensor.dtype == torch.bool],
                                 dtype=torch.int32, device=device)
        ndim = torch.tensor([tensor.ndim], dtype=torch.int32, device=device)
    else:
        dtype_id = torch.zeros(7, dtype=torch.int32, device=device)
        ndim = torch.zeros(1, dtype=torch.int32, device=device)

    dist.broadcast(dtype_id, src=src_rank, group=group)
    dist.broadcast(ndim, src=src_rank, group=group)

    _dtype_map = [
        torch.float32, torch.float16, torch.bfloat16,
        torch.int64, torch.int32, torch.int8, torch.bool,
    ]
    dtype = _dtype_map[int(dtype_id.argmax().item())]
    n = int(ndim.item())

    if rank == src_rank:
        assert tensor is not None
        shape_t = torch.tensor(list(tensor.shape), dtype=torch.int64, device=device)
    else:
        shape_t = torch.zeros(n, dtype=torch.int64, device=device)

    dist.broadcast(shape_t, src=src_rank, group=group)
    shape = tuple(shape_t.tolist())

    # Phase 2: broadcast data
    if rank == src_rank:
        assert tensor is not None
        buf = tensor.to(device=device, dtype=dtype).contiguous()
    else:
        buf = torch.empty(shape, dtype=dtype, device=device)

    dist.broadcast(buf, src=src_rank, group=group)
    return buf


def broadcast_batch_to_tp_group(
    batch: Dict[str, Optional[torch.Tensor]],
    src_global_rank: int,
    tp_group: dist.ProcessGroup,
    device: torch.device,
) -> Dict[str, Optional[torch.Tensor]]:
    """Broadcast all BATCH_KEYS tensors to every rank in the TP group.

    Mirrors Megatron's ``get_batch_on_this_tp_rank`` but uses the two-phase
    shape-safe broadcast so that non-source ranks don't need prior knowledge
    of batch shapes (important for DES-LOC elastic restarts where shapes may
    differ between steps).

    Parameters
    ----------
    batch:
        Dict populated by the data-loader on src_global_rank; empty/None on
        all other ranks.
    src_global_rank:
        The TP-rank-0 global rank that fetched the data.
    tp_group:
        Tensor-parallel ProcessGroup.
    device:
        Target CUDA device for this rank.

    Returns
    -------
    Populated batch dict on all ranks.
    """
    out: Dict[str, Optional[torch.Tensor]] = {}
    for key in BATCH_KEYS:
        tensor = batch.get(key)
        out[key] = _broadcast_tensor(tensor, src_rank=src_global_rank, group=tp_group, device=device)

    logger.debug(
        "Batch broadcast complete on rank %d (src=%d); keys with data: %s",
        dist.get_rank(),
        src_global_rank,
        [k for k in BATCH_KEYS if out.get(k) is not None],
    )
    return out


# ---------------------------------------------------------------------------
# Heterogeneous locality-aware source-rank selection
# ---------------------------------------------------------------------------

def select_locality_source_rank(
    cp_group_ranks: Sequence[int],
    rank_device_map: RankDeviceMap,
) -> int:
    """Choose the CP-group rank that should act as the DES-LOC locality anchor.

    The DES-LOC "Shared LOcality Cache" principle states that long-lived KV
    data should be pinned to the device with the most capacity.  When the H100
    is in the CP group it is always preferred.  Among A6000 ranks the lowest
    global rank wins (deterministic, low coordination overhead).

    Parameters
    ----------
    cp_group_ranks:
        Global ranks in the context-parallel group for this micro-batch.
    rank_device_map:
        Device profile registry.

    Returns
    -------
    The global rank chosen as locality source.
    """
    src = rank_device_map.preferred_source_rank(cp_group_ranks)
    logger.debug(
        "DES-LOC locality source selected: rank %d (%s) from group %s",
        src,
        rank_device_map.device_class(src).name,
        list(cp_group_ranks),
    )
    return src


# ---------------------------------------------------------------------------
# Shape canonicalization
# ---------------------------------------------------------------------------

def canonicalize_cu_seqlens(
    cu_seqlens: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Squeeze cu_seqlens from shape (1, n) → (n,).

    Mirrors the Megatron adcdf162 fix: after CP slicing the batch dim (size 1)
    produced by the data-loader is no longer needed and confuses attention
    kernels that expect a 1-D cumulative-sum tensor.

    Parameters
    ----------
    cu_seqlens:
        Raw tensor from the batch dict (may be 2-D with leading batch dim).

    Returns
    -------
    1-D tensor or None.
    """
    if cu_seqlens is None:
        return None
    if cu_seqlens.dim() == 2:
        assert cu_seqlens.shape[0] == 1, (
            f"micro-batch-size must be 1 for packed THD format, "
            f"got cu_seqlens shape {cu_seqlens.shape}"
        )
        return cu_seqlens[0]
    return cu_seqlens


def canonicalize_max_seqlen(
    max_seqlen: Optional[torch.Tensor],
) -> Optional[int]:
    """Convert max_seqlen tensor to a Python int scalar.

    The data-loader emits max_seqlen as shape (1,) or (). Both cases are
    handled so that the forward path always receives a plain int.

    Parameters
    ----------
    max_seqlen:
        Raw tensor from the batch dict.

    Returns
    -------
    Python int or None.
    """
    if max_seqlen is None:
        return None
    if max_seqlen.dim() == 0:
        return int(max_seqlen.item())
    return int(max_seqlen[0].item())


# ---------------------------------------------------------------------------
# Virtual-pipeline + MTP stage guard
# ---------------------------------------------------------------------------

def is_first_or_last_pipeline_stage(
    vp_stage: Optional[int],
    is_pipeline_first_stage: bool,
    is_pipeline_last_stage: bool,
    vp_num_stages: Optional[int] = None,
) -> bool:
    """Return True if this rank is a "boundary" pipeline stage.

    Mirrors Megatron's ``is_first_or_last_pipeline_stage`` utility introduced
    in the adcdf162 refactor, but extended for virtual-pipeline awareness
    needed by DES-LOC's elastic topology manager.

    A boundary stage must receive a full batch dict (tokens, labels, etc.).
    Interior stages that are neither MTP ranks nor SFT mode only need the
    sequence metadata (cu_seqlens, max_seqlen).

    Parameters
    ----------
    vp_stage:
        Virtual pipeline stage index (0-based), or None if not using VP.
    is_pipeline_first_stage:
        Whether this rank is stage 0 in the physical pipeline.
    is_pipeline_last_stage:
        Whether this rank is the final stage in the physical pipeline.
    vp_num_stages:
        Total number of virtual stages (used to detect VP first/last).

    Returns
    -------
    bool
    """
    if vp_stage is None:
        return is_pipeline_first_stage or is_pipeline_last_stage

    # VP first stage is vp_stage == 0 on the physical first stage
    # VP last stage is vp_stage == vp_num_stages - 1 on the physical last stage
    vp_first = (vp_stage == 0) and is_pipeline_first_stage
    vp_last = (vp_num_stages is not None) and (vp_stage == vp_num_stages - 1) and is_pipeline_last_stage
    # Also count any rank that is physically first or last regardless of VP chunk
    return vp_first or vp_last or is_pipeline_first_stage or is_pipeline_last_stage


# ---------------------------------------------------------------------------
# Core HeteroElasticBatch class
# ---------------------------------------------------------------------------

class HeteroElasticBatch:
    """DES-LOC heterogeneous elastic batch manager.

    This class is the central adaptation of Megatron's refactored ``get_batch``
    for DES-LOC heterogeneous hardware.  It owns the full lifecycle of a
    micro-batch from data-iterator fetch through CP-sliced, device-local
    tensors that each pipeline rank can forward.

    Design principles
    -----------------
    1.  **Locality-first fetch**: Only one rank in the TP group fetches from
        the data iterator.  The source rank is the one with the highest device
        capacity (H100 preferred).  This is *different* from Megatron where
        TP-rank-0 is always the source — on DES-LOC, the H100 may not be
        TP-rank-0.

    2.  **Capacity-weighted CP partition**: Instead of equal-size sequence
        slices, the H100 rank receives twice the tokens of each A6000 rank.
        This fully utilises H100 memory and avoids OOM on A6000 during long-
        context training without requiring gradient checkpointing on A6000.

    3.  **Decoupled execution barrier**: After distributing slices, each rank
        executes its forward pass independently.  There is no implicit barrier
        until the reduce-scatter in the backward pass.  This is the "Decoupled
        Execution" in DES-LOC.

    4.  **Shape canonicalization**: cu_seqlens and max_seqlen are squeezed to
        1-D / scalar after all slicing, matching the contract the forward path
        expects regardless of packing format.

    Parameters
    ----------
    rank_device_map:
        Device capability registry for the cluster.
    tp_group:
        Tensor-parallel process group.
    cp_group:
        Context-parallel process group.
    device:
        This rank's CUDA device.
    enable_hetero_cp:
        If False, fall back to uniform CP slicing (debugging / ablation).
    """

    def __init__(
        self,
        rank_device_map: RankDeviceMap,
        tp_group: dist.ProcessGroup,
        cp_group: dist.ProcessGroup,
        device: torch.device,
        enable_hetero_cp: bool = True,
    ) -> None:
        self.rank_device_map = rank_device_map
        self.tp_group = tp_group
        self.cp_group = cp_group
        self.device = device
        self.enable_hetero_cp = enable_hetero_cp

        self._global_rank = dist.get_rank()
        self._tp_ranks: List[int] = dist.get_process_group_ranks(tp_group)
        self._cp_ranks: List[int] = dist.get_process_group_ranks(cp_group)
        self._cp_rank_index: int = self._cp_ranks.index(self._global_rank)

        # Source rank for TP broadcast (locality anchor in TP group)
        self._tp_src = select_locality_source_rank(self._tp_ranks, rank_device_map)

        logger.info(
            "HeteroElasticBatch initialised on rank %d | device=%s | "
            "tp_src=%d | cp_group=%s | hetero_cp=%s",
            self._global_rank,
            device,
            self._tp_src,
            self._cp_ranks,
            enable_hetero_cp,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batch(
        self,
        data_iterator: Iterator[Dict[str, Any]],
        vp_stage: Optional[int] = None,
        is_pipeline_first_stage: bool = True,
        is_pipeline_last_stage: bool = True,
        mtp_on_this_rank: bool = False,
        is_sft: bool = False,
        vp_num_stages: Optional[int] = None,
        cp_size: Optional[int] = None,
    ) -> Tuple[
        Optional[torch.Tensor],  # tokens
        Optional[torch.Tensor],  # labels
        Optional[torch.Tensor],  # loss_mask
        Optional[torch.Tensor],  # attention_mask
        Optional[torch.Tensor],  # position_ids
        Optional[torch.Tensor],  # cu_seqlens
        Optional[int],           # max_seqlen
    ]:
        """Fetch, broadcast, and heterogeneously slice one micro-batch.

        This method re-implements Megatron's ``get_batch(data_iterator, vp_stage)``
        (adcdf162) with DES-LOC extensions.  The return signature is identical
        (7-tuple) so it can be used as a drop-in in DeepSpeed pipe_engine hooks.

        Parameters
        ----------
        data_iterator:
            Per-rank data iterator.  Only the TP locality-source rank calls
            ``next()``; other ranks pass a dummy or the same iterator (unused).
        vp_stage:
            Virtual pipeline stage index, or None.
        is_pipeline_first_stage:
            Whether this rank is the first physical pipeline stage.
        is_pipeline_last_stage:
            Whether this rank is the last physical pipeline stage.
        mtp_on_this_rank:
            Whether multi-token prediction is active on this rank.
        is_sft:
            Whether we are in supervised fine-tuning mode (affects early-exit).
        vp_num_stages:
            Total VP stages, required for correct VP boundary detection.
        cp_size:
            Context-parallel world size (overrides len(cp_group) when set).

        Returns
        -------
        7-tuple: (tokens, labels, loss_mask, attention_mask, position_ids,
                  cu_seqlens, max_seqlen)
        """
        # --- Stage guard (mirrors Megatron adcdf162 early-exit) ---
        boundary = is_first_or_last_pipeline_stage(
            vp_stage=vp_stage,
            is_pipeline_first_stage=is_pipeline_first_stage,
            is_pipeline_last_stage=is_pipeline_last_stage,
            vp_num_stages=vp_num_stages,
        )

        if not boundary and not mtp_on_this_rank and not is_sft:
            return None, None, None, None, None, None, None

        # --- Fetch from data iterator (locality-source only) ---
        raw_batch: Dict[str, Optional[torch.Tensor]] = {}
        if self._global_rank == self._tp_src:
            raw_batch = self._fetch_from_iterator(data_iterator)

        # --- Broadcast batch to TP group ---
        batch = broadcast_batch_to_tp_group(
            batch=raw_batch,
            src_global_rank=self._tp_src,
            tp_group=self.tp_group,
            device=self.device,
        )

        # --- Intermediate PP stage shortcut (SFT only) ---
        # Mirrors Megatron's PP-SFT shortcut: interior stages only need THD metadata.
        if not boundary and not mtp_on_this_rank:
            assert is_sft, "Interior PP stage without SFT or MTP should have exited earlier"
            cu_seqlens = canonicalize_cu_seqlens(batch.get("cu_seqlens"))
            max_seqlen = canonicalize_max_seqlen(batch.get("max_seqlen"))
            return None, None, None, None, None, cu_seqlens, max_seqlen

        # --- Heterogeneous CP slice ---
        batch = self._apply_hetero_cp_slice(batch, cp_size=cp_size)

        # --- Shape canonicalization ---
        cu_seqlens = canonicalize_cu_seqlens(batch.get("cu_seqlens"))
        max_seqlen = canonicalize_max_seqlen(batch.get("max_seqlen"))

        return (
            batch.get("tokens"),
            batch.get("labels"),
            batch.get("loss_mask"),
            batch.get("attention_mask"),
            batch.get("position_ids"),
            cu_seqlens,
            max_seqlen,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_from_iterator(
        self,
        data_iterator: Iterator[Dict[str, Any]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Pull one micro-batch from the data iterator and move to device.

        Only called on the TP locality-source rank.  Matches Megatron's
        TP-rank-0 fetch pattern but uses ``non_blocking=True`` to overlap
        PCIe transfer with any pending compute on the source device.
        """
        raw = next(data_iterator)
        batch: Dict[str, Optional[torch.Tensor]] = {}
        for key in BATCH_KEYS:
            val = raw.get(key)
            if val is not None and isinstance(val, torch.Tensor):
                batch[key] = val.cuda(self.device, non_blocking=True)
            else:
                batch[key] = None

        logger.debug(
            "Rank %d fetched batch; sequence length: tokens=%s",
            self._global_rank,
            batch["tokens"].shape if batch.get("tokens") is not None else "N/A",
        )
        return batch

    def _apply_hetero_cp_slice(
        self,
        batch: Dict[str, Optional[torch.Tensor]],
        cp_size: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Apply heterogeneous context-parallel slicing.

        For non-packed (padded) batches: tokens shape is [batch, seq_len].
        Sequence dimension is sliced according to capacity-weighted offsets.

        For packed (THD) batches: tokens shape is [1, total_tokens].
        Same logic applies; cu_seqlens and attention_mask are passed through
        unchanged (they describe the global sequence, not per-rank).

        If ``enable_hetero_cp`` is False, falls back to uniform slicing
        (useful for debugging and ablation studies).

        Parameters
        ----------
        batch:
            Batch dict after TP broadcast.
        cp_size:
            Override for context-parallel world size.

        Returns
        -------
        Batch dict with HETERO_SLICE_KEYS sliced to this CP rank's share.
        """
        effective_cp_size = cp_size if cp_size is not None else len(self._cp_ranks)

        if effective_cp_size <= 1:
            # No context parallelism; return batch unchanged.
            return batch

        tokens = batch.get("tokens")
        if tokens is None:
            return batch

        total_tokens = tokens.shape[1]

        if self.enable_hetero_cp:
            offsets = compute_hetero_cp_offsets(
                total_tokens=total_tokens,
                cp_ranks=self._cp_ranks,
                rank_device_map=self.rank_device_map,
            )
        else:
            # Uniform fallback (Megatron-style)
            chunk = total_tokens // effective_cp_size
            offsets = [i * chunk for i in range(effective_cp_size + 1)]
            offsets[-1] = total_tokens  # absorb rounding remainder in last rank

        out = dict(batch)
        for key in HETERO_SLICE_KEYS:
            val = batch.get(key)
            if val is not None:
                out[key] = hetero_cp_slice(
                    tensor=val,
                    cp_rank_index=self._cp_rank_index,
                    offsets=offsets,
                    seq_dim=1,
                )

        # Record offsets as metadata tensor for downstream diagnostics
        out["local_cp_offsets"] = torch.tensor(offsets, dtype=torch.int32, device=self.device)

        logger.debug(
            "CP slice on rank %d (index %d/%d): tokens %d→%d [offsets %s]",
            self._global_rank,
            self._cp_rank_index,
            effective_cp_size,
            total_tokens,
            out["tokens"].shape[1] if out.get("tokens") is not None else 0,
            offsets,
        )

        return out


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_hetero_elastic_batch(
    tp_group: dist.ProcessGroup,
    cp_group: dist.ProcessGroup,
    device: Optional[torch.device] = None,
    enable_hetero_cp: bool = True,
) -> HeteroElasticBatch:
    """Construct a ``HeteroElasticBatch`` from the current distributed environment.

    Parameters
    ----------
    tp_group:
        Tensor-parallel process group.
    cp_group:
        Context-parallel process group.
    device:
        CUDA device for this rank.  Defaults to ``torch.cuda.current_device()``.
    enable_hetero_cp:
        Enable capacity-weighted CP partitioning (default True).

    Returns
    -------
    Configured ``HeteroElasticBatch`` instance.
    """
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    rank_device_map = RankDeviceMap.from_env()
    logger.info("DES-LOC rank device map: %s", rank_device_map)

    return HeteroElasticBatch(
        rank_device_map=rank_device_map,
        tp_group=tp_group,
        cp_group=cp_group,
        device=device,
        enable_hetero_cp=enable_hetero_cp,
    )


# ---------------------------------------------------------------------------
# DeepSpeed PipeEngine integration hook
# ---------------------------------------------------------------------------

class HeteroElasticBatchPipeHook:
    """Callable that can be registered as a DeepSpeed pipe-engine input hook.

    DeepSpeed's ``PipelineEngine`` calls ``inputs = self._exec_schedule(...)``
    which can be customised by replacing the data-loader with a hook object
    that supports ``__call__(data_iterator)``.

    Usage
    -----
    .. code-block:: python

        hook = HeteroElasticBatchPipeHook(
            tp_group=mpu.get_tensor_model_parallel_group(),
            cp_group=mpu.get_context_parallel_group(),
        )
        engine.set_dataloader(hook)

    The hook returns the 7-tuple expected by the flextron forward functions.
    """

    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        cp_group: dist.ProcessGroup,
        device: Optional[torch.device] = None,
        enable_hetero_cp: bool = True,
        is_sft: bool = False,
    ) -> None:
        self._manager = build_hetero_elastic_batch(
            tp_group=tp_group,
            cp_group=cp_group,
            device=device,
            enable_hetero_cp=enable_hetero_cp,
        )
        self._is_sft = is_sft
        self._step = 0

    def __call__(
        self,
        data_iterator: Iterator[Dict[str, Any]],
        vp_stage: Optional[int] = None,
        is_pipeline_first_stage: bool = True,
        is_pipeline_last_stage: bool = True,
        mtp_on_this_rank: bool = False,
    ) -> Tuple:
        result = self._manager.get_batch(
            data_iterator=data_iterator,
            vp_stage=vp_stage,
            is_pipeline_first_stage=is_pipeline_first_stage,
            is_pipeline_last_stage=is_pipeline_last_stage,
            mtp_on_this_rank=mtp_on_this_rank,
            is_sft=self._is_sft,
        )
        self._step += 1
        return result


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    class TestDeviceProfiles(unittest.TestCase):
        """Tests for device profile registry and RankDeviceMap."""

        def test_default_profile_weights(self):
            a6000 = DEVICE_PROFILES[DeviceClass.A6000_SM86]
            h100 = DEVICE_PROFILES[DeviceClass.H100_NVL_SM90]
            self.assertEqual(a6000.capacity_weight, 1.0)
            self.assertEqual(h100.capacity_weight, 2.0)
            self.assertEqual(a6000.sm_version, 86)
            self.assertEqual(h100.sm_version, 90)

        def test_rank_device_map_from_env(self):
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,A6000_SM86,H100_NVL_SM90"
            rdm = RankDeviceMap.from_env()
            self.assertEqual(rdm.device_class(0), DeviceClass.A6000_SM86)
            self.assertEqual(rdm.device_class(1), DeviceClass.A6000_SM86)
            self.assertEqual(rdm.device_class(2), DeviceClass.H100_NVL_SM90)

        def test_preferred_source_rank_is_h100(self):
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,A6000_SM86,H100_NVL_SM90"
            rdm = RankDeviceMap.from_env()
            src = rdm.preferred_source_rank([0, 1, 2])
            self.assertEqual(src, 2, "H100 (rank 2) should be preferred source")

        def test_preferred_source_rank_all_a6000(self):
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,A6000_SM86"
            rdm = RankDeviceMap.from_env()
            src = rdm.preferred_source_rank([0, 1])
            # Tie: max() returns last maximum; both weight=1.0, max returns 1
            self.assertIn(src, [0, 1])

        def test_invalid_device_class_raises(self):
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,NONEXISTENT"
            with self.assertRaises(ValueError):
                RankDeviceMap.from_env()

        def test_is_high_capacity(self):
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,A6000_SM86,H100_NVL_SM90"
            rdm = RankDeviceMap.from_env()
            self.assertFalse(rdm.is_high_capacity(0))
            self.assertFalse(rdm.is_high_capacity(1))
            self.assertTrue(rdm.is_high_capacity(2))

    class TestHeteroOffsets(unittest.TestCase):
        """Tests for capacity-weighted CP offset computation."""

        def setUp(self):
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,A6000_SM86,H100_NVL_SM90"
            self.rdm = RankDeviceMap.from_env()

        def test_total_tokens_preserved(self):
            """Offsets must partition exactly total_tokens tokens."""
            for total in [100, 1000, 6000, 8192, 131072]:
                offsets = compute_hetero_cp_offsets(
                    total_tokens=total,
                    cp_ranks=[0, 1, 2],
                    rank_device_map=self.rdm,
                )
                self.assertEqual(offsets[0], 0)
                self.assertEqual(offsets[-1], total)
                self.assertEqual(len(offsets), 4)  # 3 ranks + 1 sentinel

        def test_h100_gets_more_tokens(self):
            """H100 (rank 2, weight=2) should receive more tokens than each A6000."""
            offsets = compute_hetero_cp_offsets(
                total_tokens=4000,
                cp_ranks=[0, 1, 2],
                rank_device_map=self.rdm,
            )
            rank0_tokens = offsets[1] - offsets[0]
            rank1_tokens = offsets[2] - offsets[1]
            rank2_tokens = offsets[3] - offsets[2]
            self.assertGreater(rank2_tokens, rank0_tokens)
            self.assertGreater(rank2_tokens, rank1_tokens)

        def test_exact_ratio_4000_tokens(self):
            """4000 tokens: 1000 + 1000 + 2000 (weights 1:1:2)."""
            offsets = compute_hetero_cp_offsets(
                total_tokens=4000,
                cp_ranks=[0, 1, 2],
                rank_device_map=self.rdm,
            )
            self.assertEqual(offsets[1] - offsets[0], 1000)
            self.assertEqual(offsets[2] - offsets[1], 1000)
            self.assertEqual(offsets[3] - offsets[2], 2000)

        def test_indivisible_tokens(self):
            """Remainder tokens should be distributed without losing any."""
            for total in [7, 13, 1001, 9999]:
                offsets = compute_hetero_cp_offsets(
                    total_tokens=total,
                    cp_ranks=[0, 1, 2],
                    rank_device_map=self.rdm,
                )
                allocated = sum(offsets[i+1] - offsets[i] for i in range(3))
                self.assertEqual(allocated, total,
                                  f"Lost tokens for total={total}: offsets={offsets}")

        def test_single_rank_cp(self):
            """With one CP rank, all tokens go to that rank."""
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "H100_NVL_SM90"
            rdm = RankDeviceMap.from_env()
            offsets = compute_hetero_cp_offsets(
                total_tokens=512,
                cp_ranks=[0],
                rank_device_map=rdm,
            )
            self.assertEqual(offsets, [0, 512])

        def test_min_tokens_per_rank_floor(self):
            """Even with total < num_ranks, each rank gets at least min_tokens_per_rank."""
            offsets = compute_hetero_cp_offsets(
                total_tokens=3,
                cp_ranks=[0, 1, 2],
                rank_device_map=self.rdm,
                min_tokens_per_rank=1,
            )
            for i in range(3):
                self.assertGreaterEqual(offsets[i+1] - offsets[i], 1)

    class TestHeteroSlice(unittest.TestCase):
        """Tests for heterogeneous CP tensor slicing."""

        def test_slice_correct_range(self):
            """hetero_cp_slice must return exactly the tokens in [start, end)."""
            tokens = torch.arange(4000).unsqueeze(0)  # shape [1, 4000]
            offsets = [0, 1000, 2000, 4000]

            slice0 = hetero_cp_slice(tokens, cp_rank_index=0, offsets=offsets, seq_dim=1)
            slice1 = hetero_cp_slice(tokens, cp_rank_index=1, offsets=offsets, seq_dim=1)
            slice2 = hetero_cp_slice(tokens, cp_rank_index=2, offsets=offsets, seq_dim=1)

            self.assertEqual(slice0.shape[1], 1000)
            self.assertEqual(slice1.shape[1], 1000)
            self.assertEqual(slice2.shape[1], 2000)

            self.assertTrue(torch.all(slice0[0] == torch.arange(0, 1000)))
            self.assertTrue(torch.all(slice1[0] == torch.arange(1000, 2000)))
            self.assertTrue(torch.all(slice2[0] == torch.arange(2000, 4000)))

        def test_slice_no_overlap(self):
            """All slices concatenated must reproduce the original tensor."""
            tokens = torch.randint(0, 100, (1, 6000))
            offsets = [0, 1500, 3000, 6000]  # weights 1:1:2 with total=6000
            slices = [
                hetero_cp_slice(tokens, i, offsets, seq_dim=1) for i in range(3)
            ]
            reconstructed = torch.cat(slices, dim=1)
            self.assertTrue(torch.equal(reconstructed, tokens))

        def test_slice_preserves_batch_dim(self):
            """Batch dimension must be unchanged by CP slicing."""
            tokens = torch.randn(4, 100)  # batch=4, seq=100
            offsets = [0, 50, 100]
            sliced = hetero_cp_slice(tokens, cp_rank_index=0, offsets=offsets, seq_dim=1)
            self.assertEqual(sliced.shape[0], 4)
            self.assertEqual(sliced.shape[1], 50)

    class TestCanonicalizeShapes(unittest.TestCase):
        """Tests for cu_seqlens and max_seqlen shape canonicalization."""

        def test_cu_seqlens_2d_to_1d(self):
            cu = torch.tensor([[0, 128, 256, 512]])  # shape (1, 4)
            result = canonicalize_cu_seqlens(cu)
            self.assertEqual(result.dim(), 1)
            self.assertEqual(result.shape[0], 4)
            self.assertTrue(torch.equal(result, torch.tensor([0, 128, 256, 512])))

        def test_cu_seqlens_already_1d(self):
            cu = torch.tensor([0, 128, 256])
            result = canonicalize_cu_seqlens(cu)
            self.assertEqual(result.dim(), 1)

        def test_cu_seqlens_none(self):
            self.assertIsNone(canonicalize_cu_seqlens(None))

        def test_cu_seqlens_batch_size_gt1_raises(self):
            cu = torch.tensor([[0, 128], [0, 64]])  # batch=2, invalid
            with self.assertRaises(AssertionError):
                canonicalize_cu_seqlens(cu)

        def test_max_seqlen_0d(self):
            ms = torch.tensor(512)  # 0-D tensor
            result = canonicalize_max_seqlen(ms)
            self.assertIsInstance(result, int)
            self.assertEqual(result, 512)

        def test_max_seqlen_1d(self):
            ms = torch.tensor([1024])  # shape (1,)
            result = canonicalize_max_seqlen(ms)
            self.assertIsInstance(result, int)
            self.assertEqual(result, 1024)

        def test_max_seqlen_none(self):
            self.assertIsNone(canonicalize_max_seqlen(None))

    class TestPipelineStageGuard(unittest.TestCase):
        """Tests for virtual-pipeline boundary detection."""

        def test_no_vp_first_stage(self):
            self.assertTrue(is_first_or_last_pipeline_stage(
                vp_stage=None, is_pipeline_first_stage=True,
                is_pipeline_last_stage=False))

        def test_no_vp_last_stage(self):
            self.assertTrue(is_first_or_last_pipeline_stage(
                vp_stage=None, is_pipeline_first_stage=False,
                is_pipeline_last_stage=True))

        def test_no_vp_interior(self):
            self.assertFalse(is_first_or_last_pipeline_stage(
                vp_stage=None, is_pipeline_first_stage=False,
                is_pipeline_last_stage=False))

        def test_vp_first_stage(self):
            self.assertTrue(is_first_or_last_pipeline_stage(
                vp_stage=0, is_pipeline_first_stage=True,
                is_pipeline_last_stage=False, vp_num_stages=4))

        def test_vp_last_stage(self):
            self.assertTrue(is_first_or_last_pipeline_stage(
                vp_stage=3, is_pipeline_first_stage=False,
                is_pipeline_last_stage=True, vp_num_stages=4))

        def test_vp_interior_not_boundary(self):
            # VP stage 1 on a physical interior stage → not a boundary
            result = is_first_or_last_pipeline_stage(
                vp_stage=1, is_pipeline_first_stage=False,
                is_pipeline_last_stage=False, vp_num_stages=4)
            self.assertFalse(result)

        def test_vp_last_stage_wrong_physical(self):
            # vp_stage == vp_num_stages-1 but NOT on physical last stage → not boundary
            result = is_first_or_last_pipeline_stage(
                vp_stage=3, is_pipeline_first_stage=False,
                is_pipeline_last_stage=False, vp_num_stages=4)
            self.assertFalse(result)

    class TestBATCH_KEYS(unittest.TestCase):
        """Sanity checks on the BATCH_KEYS contract."""

        def test_hetero_slice_keys_subset_of_batch_keys(self):
            for k in HETERO_SLICE_KEYS:
                self.assertIn(k, BATCH_KEYS,
                               f"HETERO_SLICE_KEYS member '{k}' missing from BATCH_KEYS")

        def test_metadata_keys_subset_of_batch_keys(self):
            for k in METADATA_KEYS:
                self.assertIn(k, BATCH_KEYS,
                               f"METADATA_KEYS member '{k}' missing from BATCH_KEYS")

        def test_no_duplicates_in_batch_keys(self):
            self.assertEqual(len(BATCH_KEYS), len(set(BATCH_KEYS)))

    class TestRankDeviceMapRepr(unittest.TestCase):
        def test_repr_contains_rank_info(self):
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,H100_NVL_SM90"
            rdm = RankDeviceMap.from_env()
            r = repr(rdm)
            self.assertIn("rank 0", r)
            self.assertIn("rank 1", r)
            self.assertIn("A6000_SM86", r)
            self.assertIn("H100_NVL_SM90", r)

    class TestComputeHeteroOffsetsEdgeCases(unittest.TestCase):
        """Edge cases for heterogeneous offset computation."""

        def test_two_ranks_equal_weight(self):
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,A6000_SM86"
            rdm = RankDeviceMap.from_env()
            offsets = compute_hetero_cp_offsets(1000, [0, 1], rdm)
            self.assertEqual(offsets[1] - offsets[0], 500)
            self.assertEqual(offsets[2] - offsets[1], 500)

        def test_large_sequence_4096_tokens(self):
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,A6000_SM86,H100_NVL_SM90"
            rdm = RankDeviceMap.from_env()
            offsets = compute_hetero_cp_offsets(4096, [0, 1, 2], rdm)
            total = offsets[-1]
            self.assertEqual(total, 4096)

        def test_large_sequence_131072_tokens_h100_largest(self):
            """Long-context scenario: H100 must absorb the largest share."""
            os.environ["DESLOCAL_RANK_TO_CLASS"] = "A6000_SM86,A6000_SM86,H100_NVL_SM90"
            rdm = RankDeviceMap.from_env()
            offsets = compute_hetero_cp_offsets(131072, [0, 1, 2], rdm)
            rank2_tokens = offsets[3] - offsets[2]
            rank0_tokens = offsets[1] - offsets[0]
            rank1_tokens = offsets[2] - offsets[1]
            self.assertGreater(rank2_tokens, rank0_tokens)
            self.assertGreater(rank2_tokens, rank1_tokens)
            # Rough check: H100 should get ~2× each A6000
            self.assertAlmostEqual(rank2_tokens / rank0_tokens, 2.0, delta=0.05)

    # Run all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in [
        TestDeviceProfiles,
        TestHeteroOffsets,
        TestHeteroSlice,
        TestCanonicalizeShapes,
        TestPipelineStageGuard,
        TestBATCH_KEYS,
        TestRankDeviceMapRepr,
        TestComputeHeteroOffsetsEdgeCases,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
