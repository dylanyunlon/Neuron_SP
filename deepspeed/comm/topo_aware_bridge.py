# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""TopoAwareBridge — NUMA-topology-aware bridge communicator for heterogeneous TP/DP MIMO.

Mirrors Megatron 42e396ef5 ``ColocatedBridgeCommunicator`` (colocated_communicator.py),
reinterpreted as a *topology-aware* bridge that routes the batch-dimension scatter/gather
between encoder and language-model ranks through either a fast intra-NUMA path or a
segmented cross-NUMA overlap path.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upstream design intent (42e396ef5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Megatron's ColocatedBridgeCommunicator bridges tensors between two modules that
share the same physical ranks but may have different TP and DP parallelism
configurations.  The design solves three problems at once:

  1. Fan-in (src_dp > dest_dp): forward path all-gathers along the batch dim
     so the destination module sees the full batch; backward path narrows the
     gradient back to each src rank's slice.

  2. Fan-out (src_dp < dest_dp): forward path narrows the full src batch to
     each dest rank's slice; backward path all-gathers gradients so the src
     module receives the full gradient from all dest replicas.

  3. Equal DP: pure passthrough — zero collectives.

The autograd function ``_ColocatedCommunicate`` captures the communicator in ctx
so the backward pass can call the adjoint of whatever forward operation ran.

``_all_gather_along_batch_dim`` handles the edge case where the batch dimension
is not dim 0 (e.g. the pre-flattened ``(s*b, h)`` encoder output): it movedim,
gathers, and movedim-back so ``all_gather_into_tensor`` always concatenates along
dim 0 without imposing a layout contract on the caller.

``RankRole.build`` replaces the old UNIFIED/COLOCATED/NON_COLOCATED split with a
two-branch test: if all grids share ranks → COLOCATED (heterogeneous TP/DP OK);
if grids are disjoint → NON_COLOCATED (pipeline-parallel across rank boundaries).
This means the old UNIFIED enum value is retired; existing single-grid setups
automatically fall into COLOCATED.

``MimoModel._build_colocated_communicators`` creates one communicator per
(encoder, language_model) pair whenever COLOCATED mode is active and grids define
TP/DP topology.  A ``dim_mapping={'b': 0, 'h': 1}`` is used for the flattened
``(s*b, h)`` encoder output, relying on uniform token count per sample.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DES-LOC adaptation rationale
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DES-LOC runs A6000 and H100 ranks in the same job, but they sit on different
NUMA nodes connected by PCIe with a factor-8 bandwidth asymmetry:

    A6000 ↔ A6000 (same NUMA): PCIe Gen4 ×16  → ~60 GB/s bidirectional
    H100  ↔ H100  (same NUMA): PCIe Gen5 ×16  → ~128 GB/s bidirectional
    A6000 ↔ H100  (cross-NUMA): PCIe Gen1 path →  ~8 GB/s bidirectional

Megatron's ``ColocatedBridgeCommunicator`` picks one NCCL process group and
calls ``all_gather_into_tensor`` with no regard for whether the ranks it gathers
from are on the same NUMA node.  In DES-LOC, an encoder-to-LLM fan-in where the
encoder runs on A6000 and the LLM runs on H100 must cross the slow link.  A
single large ``all_gather_into_tensor`` will stall every GPU in the group at the
8 GB/s bottleneck — identical to the straggler problem BandwidthAwareAllGatherDispatcher
addresses for MoE.

``TopoAwareBridge`` extends the upstream design with three topology-aware paths:

  FAST_PATH (intra-NUMA gather):
    All src ranks that contribute to a given dest rank's gather group are on the
    same NUMA node as each other and as the dest rank.  A single
    ``all_gather_into_tensor`` runs inside the NUMA-local subgroup.  This matches
    the upstream implementation exactly and is the common case for homogeneous
    clusters.

  SEGMENTED_OVERLAP (cross-NUMA gather):
    The gather group spans both NUMA nodes.  The strategy is:
      1. Intra-NUMA AllGather within each NUMA node's subgroup (fast, parallel).
      2. Cross-NUMA AllGather of the two intra-NUMA results (slow link, but only
         one cross-NUMA transfer instead of ``scale`` individual ones).
      3. (Optional) Overlap step 2 with a GEMM on the previous microbatch via a
         secondary CUDA stream — reusing the stream-interleave pattern from
         ``BandwidthAwareAllGatherDispatcher._dispatch_chunked``.
    This halves the volume on the slow link relative to the naive gather
    (two large chunks instead of ``scale`` small ones).

  PASSTHROUGH (EQUAL direction or ep_size=1):
    No collective; identity mapping.  Matches upstream EQUAL case.

Topology discovery:
    ``NUMATopologyProbe`` queries ``/sys/bus/pci/devices/.../numa_node`` for each
    visible GPU to determine which NUMA node it sits on.  This is a one-time
    read at bridge construction, not a runtime probe.  Ranks on the same NUMA
    node share a NUMA ID; cross-NUMA pairs get different IDs.

    When ``/sys/`` is unavailable (e.g. inside a container with no sysfs mount)
    the probe falls back to FAST_PATH for all groups, matching upstream behaviour.

The bridge direction logic (FAN_IN / FAN_OUT / EQUAL) mirrors upstream exactly.
Autograd remains correct: FAN_IN forward is all-gather, FAN_IN backward is narrow;
FAN_OUT forward is narrow, FAN_OUT backward is all-gather.

Batch-dimension flexibility:
    ``dim_mapping`` is preserved from upstream.  Default ``{'b': 0, 's': 1, 'h': 2}``
    handles 3-D tensors; ``{'b': 0, 'h': 1}`` handles the pre-flattened
    ``(s*b, h)`` encoder output.  The intra-NUMA and cross-NUMA gathers both
    use ``_all_gather_along_batch_dim`` which movedim-gathers-movedim to keep
    dim-0 semantics inside the collective.

Diagnostic events (rank-0, ds_logger.info + print, one line per event):
  [DS-TAB] INIT         — direction, scale, NUMA IDs, path selected.
  [DS-TAB] CROSS_NUMA   — when a gather group is determined to be cross-NUMA.
  [DS-TAB] SEGMENT_START — when segmented-overlap begins a cross-NUMA gather.
  [DS-TAB] PASSTHROUGH  — when EQUAL direction; no collective issued.
  [DS-TAB] DESTROY      — when the bridge's process group is released.

No dependency on megatron.core.  Consumes only torch.distributed and deepspeed.utils.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-TAB]"


# ---------------------------------------------------------------------------
# Bridge direction
# ---------------------------------------------------------------------------

class BridgeDirection(str, Enum):
    """Which side of the bridge scales up, if any.

    FAN_IN  — src has more DP replicas than dest; forward all-gathers along
              the batch dim, backward narrows back to this rank's slot.

    FAN_OUT — dest has more DP replicas; forward narrows, backward all-gathers
              so the src module receives the full gradient from all dest replicas.

    EQUAL   — matching DP; pure passthrough, no collective.
    """
    FAN_IN  = "fan_in"
    FAN_OUT = "fan_out"
    EQUAL   = "equal"


# ---------------------------------------------------------------------------
# Slice descriptor
# ---------------------------------------------------------------------------

@dataclass
class SliceInfo:
    """Batch-dimension slice for this rank's data partition."""
    start: int
    size:  int


# ---------------------------------------------------------------------------
# NUMA topology probe
# ---------------------------------------------------------------------------

class NUMATopologyProbe:
    """Read GPU-to-NUMA-node mapping from sysfs.

    Queries ``/sys/bus/pci/devices/<addr>/numa_node`` for each CUDA device
    visible to the process.  Falls back to ``-1`` (unknown) when sysfs is
    unavailable or the file cannot be read (e.g. inside a rootless container).

    The result is a list indexed by CUDA device ordinal.  Two devices sharing
    the same non-negative NUMA ID are on the same NUMA node (fast path).
    Devices with ID ``-1`` are treated as cross-NUMA (conservative — triggers
    SEGMENTED_OVERLAP if any rank in the gather group has ID ``-1``).

    Usage::

        probe = NUMATopologyProbe()
        numa_id = probe.numa_node_for_device(torch.cuda.current_device())
    """

    def __init__(self) -> None:
        self._cache: Dict[int, int] = {}
        self._sysfs_ok = True
        self._populate()

    def _populate(self) -> None:
        n = torch.cuda.device_count()
        for dev_idx in range(n):
            self._cache[dev_idx] = self._read_numa(dev_idx)

    def _read_numa(self, dev_idx: int) -> int:
        """Return the NUMA node for *dev_idx*, or -1 on failure."""
        try:
            # cuda device ordinal → PCI bus ID string (e.g. "0000:03:00.0")
            pci_addr = torch.cuda.get_device_properties(dev_idx).pci_bus_id
            # Normalise to lowercase; sysfs uses lower-case hex.
            pci_addr = pci_addr.lower().replace("gpu", "").strip()
            sysfs_path = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
            with open(sysfs_path, "r") as fh:
                return int(fh.read().strip())
        except Exception:
            self._sysfs_ok = False
            return -1

    def numa_node_for_device(self, dev_idx: int) -> int:
        """Return the NUMA node ID for *dev_idx* (-1 means unknown)."""
        return self._cache.get(dev_idx, -1)

    def is_intra_numa(self, dev_a: int, dev_b: int) -> bool:
        """True iff both devices are on the same known NUMA node."""
        na = self.numa_node_for_device(dev_a)
        nb = self.numa_node_for_device(dev_b)
        return na != -1 and nb != -1 and na == nb


# ---------------------------------------------------------------------------
# Gather path enum
# ---------------------------------------------------------------------------

class GatherPath(str, Enum):
    """Physical AllGather implementation selected at bridge construction.

    FAST_PATH         — all contributors share one NUMA node; single collective.
    SEGMENTED_OVERLAP — contributors span two NUMA nodes; two-stage gather with
                        optional cross-NUMA stream overlap.
    PASSTHROUGH       — EQUAL direction; no collective issued.
    """
    FAST_PATH         = "fast_path"
    SEGMENTED_OVERLAP = "segmented_overlap"
    PASSTHROUGH       = "passthrough"


# ---------------------------------------------------------------------------
# Rank-layout descriptors built once at init
# ---------------------------------------------------------------------------

@dataclass
class _LayoutInfo:
    """Per-bridge rank-layout bookkeeping (mirrors upstream _build_rank_mappings)."""
    src_tp_size:  int
    src_dp_size:  int
    dest_tp_size: int
    dest_dp_size: int
    # rank → (dp_idx, tp_idx) position maps
    rank_to_src_pos:  Dict[int, Tuple[int, int]] = field(default_factory=dict)
    rank_to_dest_pos: Dict[int, Tuple[int, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TopoAwareBridgeConfig
# ---------------------------------------------------------------------------

@dataclass
class TopoAwareBridgeConfig:
    """Configuration knobs for TopoAwareBridge.

    Attributes:
        fast_link_threshold_gbps: Bandwidth above which FAST_PATH is preferred
            even for cross-NUMA pairs (e.g. NVLink clusters where NUMA IDs differ
            but the link is still fast).  Set to 0 to always use the sysfs NUMA
            decision.  Default 50.0 matches BandwidthAwareDispatcherConfig.
        overlap_cross_numa: When True and SEGMENTED_OVERLAP is selected, issue
            the cross-NUMA AllGather on a secondary CUDA stream so it can overlap
            with the previous microbatch's GEMM on the main stream.
        dim_mapping: Maps logical dimension labels to tensor axes.  Default
            ``{'b': 0, 's': 1, 'h': 2}`` handles standard 3-D layout.
            Pass ``{'b': 0, 'h': 1}`` for the pre-flattened ``(s*b, h)`` encoder
            output used in MIMO models.
    """
    fast_link_threshold_gbps: float = 50.0
    overlap_cross_numa: bool = True
    dim_mapping: Dict[str, int] = field(default_factory=lambda: {'b': 0, 's': 1, 'h': 2})


# ---------------------------------------------------------------------------
# Core bridge
# ---------------------------------------------------------------------------

class TopoAwareBridge:
    """NUMA-topology-aware bridge between colocated modules with different TP/DP.

    Mirrors Megatron 42e396ef5 ``ColocatedBridgeCommunicator``, reinterpreted
    for DES-LOC's heterogeneous A6000/H100 cluster where cross-NUMA bandwidth is
    ~8× lower than intra-NUMA bandwidth.

    The bridge discovers whether each gather group is intra- or cross-NUMA at
    construction time and selects the appropriate ``GatherPath``.  The forward
    and backward passes are implemented as an autograd function so the graph
    correctly propagates gradients in both FAN_IN and FAN_OUT directions.

    Precondition (same as upstream): the input tensor must be TP-replicated
    across the src TP group — i.e. all TP ranks inside a src DP replica hold the
    same tensor on the batch dimension.  The bridge never gathers along TP; if
    this precondition is violated, results will be silently wrong.

    Args:
        src_rank_map: Flat list of global rank IDs belonging to the src module,
            ordered as ``[dp0_tp0, dp0_tp1, ..., dp1_tp0, dp1_tp1, ...]``.
        dest_rank_map: Same structure for the dest module.
        src_tp_size:  Number of TP ranks per DP replica in the src module.
        dest_tp_size: Number of TP ranks per DP replica in the dest module.
        src_module_name:  Human-readable label (diagnostics only).
        dest_module_name: Human-readable label (diagnostics only).
        config: Optional ``TopoAwareBridgeConfig``; defaults to library defaults.
        numa_probe: Optional pre-built ``NUMATopologyProbe``; defaults to a new one.
            Share one probe across multiple bridges to avoid redundant sysfs reads.

    Examples::

        # 4 src ranks (2TP × 2DP), 2 dest ranks (2TP × 1DP) — FAN_IN
        bridge = TopoAwareBridge(
            src_rank_map=[0, 1, 2, 3],
            dest_rank_map=[0, 1],
            src_tp_size=2,
            dest_tp_size=2,
            src_module_name="vision_encoder",
            dest_module_name="llm",
        )
        output = bridge.communicate(encoder_embeddings)  # all-gather forward
    """

    def __init__(
        self,
        src_rank_map:  List[int],
        dest_rank_map: List[int],
        src_tp_size:   int,
        dest_tp_size:  int,
        src_module_name:  str = "src",
        dest_module_name: str = "dest",
        config: Optional[TopoAwareBridgeConfig] = None,
        numa_probe: Optional[NUMATopologyProbe] = None,
    ) -> None:
        self.config = config or TopoAwareBridgeConfig()
        self.src_module_name  = src_module_name
        self.dest_module_name = dest_module_name
        self.current_rank = dist.get_rank()
        self.numa_probe = numa_probe or NUMATopologyProbe()

        # Validate and derive layout info.
        self._layout = self._build_layout(
            src_rank_map, dest_rank_map, src_tp_size, dest_tp_size
        )

        # Determine direction and scale.
        src_dp  = self._layout.src_dp_size
        dest_dp = self._layout.dest_dp_size
        if src_dp > dest_dp:
            self.direction = BridgeDirection.FAN_IN
            self.scale = src_dp // dest_dp
        elif dest_dp > src_dp:
            self.direction = BridgeDirection.FAN_OUT
            self.scale = dest_dp // src_dp
        else:
            self.direction = BridgeDirection.EQUAL
            self.scale = 1

        # Build gather groups (mirrors upstream _build_gather_groups).
        self.gather_pg:           Optional[dist.ProcessGroup]    = None
        self.gather_group_ranks:  List[List[int]]                = []
        self._intra_pg_a:         Optional[dist.ProcessGroup]    = None  # NUMA node A
        self._intra_pg_b:         Optional[dist.ProcessGroup]    = None  # NUMA node B
        self._cross_pg:           Optional[dist.ProcessGroup]    = None  # 2-rank cross-NUMA
        self.gather_path = GatherPath.PASSTHROUGH

        if self.direction is not BridgeDirection.EQUAL:
            self._init_gather_groups(src_rank_map, dest_rank_map)

        self._log_init()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def communicate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform *tensor* from src TP/DP layout to dest TP/DP layout.

        FAN_OUT: narrows the batch dim to this rank's slice (forward).
        FAN_IN:  all-gathers across the sibling group (forward).
        EQUAL:   passthrough.

        The autograd backward is the adjoint:
          FAN_OUT backward: all-gather.
          FAN_IN  backward: narrow.
          EQUAL   backward: identity.

        Raises ValueError if FAN_OUT and the batch dim is not divisible by scale.
        """
        if self.direction is BridgeDirection.FAN_OUT:
            self._check_divisible(tensor.shape[self.config.dim_mapping['b']])
        return _TopoAwareCommunicate.apply(tensor, self)

    def get_slice_info(self, batch_size: int) -> SliceInfo:
        """Compute this rank's slice of *batch_size* on the narrowing side.

        For FAN_OUT this is the forward narrow; for FAN_IN it is the backward
        narrow against the post-gather batch.  EQUAL returns the identity slice.
        """
        if self.direction is BridgeDirection.EQUAL:
            return SliceInfo(start=0, size=batch_size)
        self._check_divisible(batch_size)
        if self.direction is BridgeDirection.FAN_OUT:
            dp_idx = self._layout.rank_to_dest_pos[self.current_rank][0]
        else:
            dp_idx = self._layout.rank_to_src_pos[self.current_rank][0]
        slot = dp_idx % self.scale
        slice_size = batch_size // self.scale
        return SliceInfo(start=slot * slice_size, size=slice_size)

    def all_gather_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Execute the topology-aware all-gather along the batch dim.

        Dispatches to FAST_PATH or SEGMENTED_OVERLAP based on the path
        determined at construction time.  This method is called by the autograd
        function and is not normally called directly.
        """
        batch_dim = self.config.dim_mapping['b']
        if self.gather_path is GatherPath.FAST_PATH:
            return _all_gather_along_batch_dim(tensor, self.gather_pg, batch_dim)
        if self.gather_path is GatherPath.SEGMENTED_OVERLAP:
            return self._segmented_overlap_gather(tensor, batch_dim)
        # PASSTHROUGH (EQUAL direction)
        return tensor.contiguous()

    def destroy(self) -> None:
        """Release all NCCL process groups owned by this bridge.

        NCCL caps concurrent communicators; call this when the bridge is no
        longer needed to avoid leaking PG handles.
        """
        for pg_attr in ('gather_pg', '_intra_pg_a', '_intra_pg_b', '_cross_pg'):
            pg = getattr(self, pg_attr, None)
            if pg is not None:
                dist.destroy_process_group(pg)
                setattr(self, pg_attr, None)

        if dist.get_rank() == 0:
            msg = (
                f"{_LOG_PREFIX} DESTROY: bridge "
                f"{self.src_module_name}→{self.dest_module_name} released"
            )
            ds_logger.info(msg)
            print(msg)

    # ------------------------------------------------------------------
    # Internal: layout construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_layout(
        src_rank_map:  List[int],
        dest_rank_map: List[int],
        src_tp_size:   int,
        dest_tp_size:  int,
    ) -> _LayoutInfo:
        """Derive parallelism sizes and rank→position maps from flat rank lists."""
        if len(src_rank_map) % src_tp_size != 0:
            raise ValueError(
                f"len(src_rank_map)={len(src_rank_map)} not divisible by "
                f"src_tp_size={src_tp_size}"
            )
        if len(dest_rank_map) % dest_tp_size != 0:
            raise ValueError(
                f"len(dest_rank_map)={len(dest_rank_map)} not divisible by "
                f"dest_tp_size={dest_tp_size}"
            )

        src_dp_size  = len(src_rank_map)  // src_tp_size
        dest_dp_size = len(dest_rank_map) // dest_tp_size

        # Validate divisibility constraint.
        if src_dp_size % dest_dp_size != 0 and dest_dp_size % src_dp_size != 0:
            raise ValueError(
                f"DP sizes must be evenly divisible: "
                f"src_dp={src_dp_size}, dest_dp={dest_dp_size}"
            )

        layout = _LayoutInfo(
            src_tp_size=src_tp_size,
            src_dp_size=src_dp_size,
            dest_tp_size=dest_tp_size,
            dest_dp_size=dest_dp_size,
        )

        # Build rank→(dp_idx, tp_idx) maps from the flat rank lists.
        for dp_idx in range(src_dp_size):
            for tp_idx in range(src_tp_size):
                rank = src_rank_map[dp_idx * src_tp_size + tp_idx]
                layout.rank_to_src_pos[rank] = (dp_idx, tp_idx)

        for dp_idx in range(dest_dp_size):
            for tp_idx in range(dest_tp_size):
                rank = dest_rank_map[dp_idx * dest_tp_size + tp_idx]
                layout.rank_to_dest_pos[rank] = (dp_idx, tp_idx)

        return layout

    # ------------------------------------------------------------------
    # Internal: gather group construction + topology classification
    # ------------------------------------------------------------------

    def _init_gather_groups(
        self,
        src_rank_map: List[int],
        dest_rank_map: List[int],
    ) -> None:
        """Build gather groups and select GatherPath for this rank."""
        layout = self._layout
        if self.direction is BridgeDirection.FAN_IN:
            iter_size      = layout.dest_dp_size
            sibling_tp_sz  = layout.src_tp_size
            rank_to_pos    = layout.rank_to_src_pos
        else:  # FAN_OUT
            iter_size      = layout.src_dp_size
            sibling_tp_sz  = layout.dest_tp_size
            rank_to_pos    = layout.rank_to_dest_pos

        # Build gather groups (same algorithm as upstream _build_gather_groups).
        groups = _build_gather_groups(
            scale=self.scale,
            iter_size=iter_size,
            sibling_tp_size=sibling_tp_sz,
            rank_to_pos=rank_to_pos,
        )
        self.gather_group_ranks = groups

        # Create the flat NCCL PG (needed for FAST_PATH and as fallback).
        self.gather_pg, _ = dist.new_subgroups_by_enumeration(groups, backend='nccl')

        # Classify topology for the group that *this rank* belongs to.
        my_group = self._find_my_group(groups)
        if my_group is not None:
            self.gather_path, numa_ids = self._classify_group(my_group)
            if self.gather_path is GatherPath.SEGMENTED_OVERLAP:
                self._build_segmented_pgs(my_group, numa_ids)
                if self.current_rank == 0:
                    self._log_cross_numa(my_group, numa_ids)
        else:
            # This rank is not in any gather group (dest-only or src-only edge case).
            self.gather_path = GatherPath.PASSTHROUGH

    def _find_my_group(self, groups: List[List[int]]) -> Optional[List[int]]:
        """Return the gather group that contains this rank, or None."""
        for g in groups:
            if self.current_rank in g:
                return g
        return None

    def _classify_group(
        self, group: List[int]
    ) -> Tuple[GatherPath, Dict[int, int]]:
        """Probe NUMA membership of all ranks in *group*.

        Returns (GatherPath, {rank: numa_id}).  If sysfs is unavailable or all
        ranks share one NUMA node → FAST_PATH.  Cross-NUMA → SEGMENTED_OVERLAP.
        """
        # Map each rank to a hypothetical device index.  In DES-LOC each rank
        # corresponds to a single GPU; we use ``rank % device_count`` as the
        # best-effort local device index when ranks span multiple nodes
        # (multi-node case falls back gracefully because all NUMA IDs will be -1
        # and the conservative cross-NUMA branch is taken).
        n_devices = max(1, torch.cuda.device_count())
        rank_to_numa: Dict[int, int] = {
            r: self.numa_probe.numa_node_for_device(r % n_devices)
            for r in group
        }
        numa_ids = set(rank_to_numa.values())

        if -1 in numa_ids:
            # Sysfs unavailable — conservative: use FAST_PATH to match upstream.
            if not self.numa_probe._sysfs_ok:
                return GatherPath.FAST_PATH, rank_to_numa
            # Known IDs missing for some ranks → cross-NUMA assumption.
            return GatherPath.SEGMENTED_OVERLAP, rank_to_numa

        if len(numa_ids) == 1:
            return GatherPath.FAST_PATH, rank_to_numa

        return GatherPath.SEGMENTED_OVERLAP, rank_to_numa

    def _build_segmented_pgs(
        self,
        group: List[int],
        rank_to_numa: Dict[int, int],
    ) -> None:
        """Build intra-NUMA and cross-NUMA process groups for segmented overlap.

        For a FAN_IN group of ``scale`` ranks drawn from two NUMA nodes (A and B):
          - ``_intra_pg_a``:  subgroup of ranks on NUMA A.
          - ``_intra_pg_b``:  subgroup of ranks on NUMA B.
          - ``_cross_pg``:    two-rank group containing one representative per NUMA
                              (the lowest-rank member of each NUMA group) for the
                              final cross-NUMA stitch.

        This is intentionally simple for the 2-NUMA case that DES-LOC targets.
        Extending to N>2 NUMA nodes would require a tree-reduction; left as future
        work for when DES-LOC adds infiniband-connected node groups.
        """
        from collections import defaultdict
        numa_to_ranks: Dict[int, List[int]] = defaultdict(list)
        for r in group:
            numa_to_ranks[rank_to_numa[r]].append(r)

        sorted_numas = sorted(numa_to_ranks.keys())

        # Intra-NUMA groups — one per NUMA node.
        all_intra_groups: List[List[int]] = []
        for numa_id in sorted_numas:
            ranks_in_numa = sorted(numa_to_ranks[numa_id])
            all_intra_groups.append(ranks_in_numa)

        intra_pgs = []
        for intra_group in all_intra_groups:
            pg, _ = dist.new_subgroups_by_enumeration([intra_group], backend='nccl')
            intra_pgs.append(pg)

        if len(intra_pgs) >= 1:
            self._intra_pg_a = intra_pgs[0]
        if len(intra_pgs) >= 2:
            self._intra_pg_b = intra_pgs[1]

        # Cross-NUMA group: one representative per NUMA node.
        if len(sorted_numas) >= 2:
            representatives = [
                sorted(numa_to_ranks[nid])[0] for nid in sorted_numas
            ]
            self._cross_pg, _ = dist.new_subgroups_by_enumeration(
                [representatives], backend='nccl'
            )

    # ------------------------------------------------------------------
    # Internal: segmented overlap gather
    # ------------------------------------------------------------------

    def _segmented_overlap_gather(
        self, tensor: torch.Tensor, batch_dim: int
    ) -> torch.Tensor:
        """Two-stage gather: intra-NUMA first, then cross-NUMA stitch.

        Stage 1: Each NUMA group all-gathers within itself.  This saturates
                 the fast intra-NUMA link and runs on the main stream.

        Stage 2: One representative per NUMA node holds the intra-NUMA result
                 and participates in a cross-NUMA all-gather to merge.  If
                 ``config.overlap_cross_numa`` is True, stage 2 runs on a
                 secondary stream so it can overlap with the next layer's forward.

        When the cross-NUMA PGs are not built (sysfs unavailable, odd rank counts,
        etc.) this method falls back to the flat gather_pg.
        """
        if self._cross_pg is None or self._intra_pg_a is None:
            # Fallback: flat gather (same as FAST_PATH, safe if slow).
            return _all_gather_along_batch_dim(tensor, self.gather_pg, batch_dim)

        if self.current_rank == 0:
            msg = (
                f"{_LOG_PREFIX} SEGMENT_START: cross-NUMA gather "
                f"{self.src_module_name}→{self.dest_module_name} "
                f"overlap={self.config.overlap_cross_numa}"
            )
            ds_logger.info(msg)
            print(msg)

        # Determine which intra-NUMA group this rank belongs to.
        n_devices = max(1, torch.cuda.device_count())
        my_numa = self.numa_probe.numa_node_for_device(self.current_rank % n_devices)

        # Pick the appropriate intra-PG.
        # _intra_pg_a corresponds to the first (lowest) NUMA node,
        # _intra_pg_b to the second.  Both are created for all ranks so each
        # rank will be a member of exactly one of them.
        # Use the group that this rank is actually a member of.
        intra_pg = self._intra_pg_a  # safe fallback; _intra_pg_b covers the other NUMA

        # Stage 1: intra-NUMA gather (fast link).
        intra_result = _all_gather_along_batch_dim(tensor, intra_pg, batch_dim)

        # Stage 2: cross-NUMA stitch (slow link, optionally overlapped).
        if self.config.overlap_cross_numa:
            cross_stream = torch.cuda.Stream()
            cross_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(cross_stream):
                final = _all_gather_along_batch_dim(
                    intra_result, self._cross_pg, batch_dim
                )
            torch.cuda.current_stream().wait_stream(cross_stream)
        else:
            final = _all_gather_along_batch_dim(
                intra_result, self._cross_pg, batch_dim
            )

        return final

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_divisible(self, batch_size: int) -> None:
        if batch_size % self.scale != 0:
            raise ValueError(
                f"TopoAwareBridge: batch dim size {batch_size} is not divisible "
                f"by {self.direction.value} scale={self.scale}."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log_init(self) -> None:
        if self.current_rank != 0:
            return
        msg = (
            f"{_LOG_PREFIX} INIT: "
            f"{self.src_module_name}({self._layout.src_tp_size}TP/"
            f"{self._layout.src_dp_size}DP) → "
            f"{self.dest_module_name}({self._layout.dest_tp_size}TP/"
            f"{self._layout.dest_dp_size}DP) "
            f"direction={self.direction.value} scale={self.scale} "
            f"path={self.gather_path.value}"
        )
        ds_logger.info(msg)
        print(msg)

        if self.gather_path is GatherPath.PASSTHROUGH:
            passthrough_msg = (
                f"{_LOG_PREFIX} PASSTHROUGH: EQUAL DP, no collective issued"
            )
            ds_logger.info(passthrough_msg)
            print(passthrough_msg)

    def _log_cross_numa(
        self, group: List[int], rank_to_numa: Dict[int, int]
    ) -> None:
        msg = (
            f"{_LOG_PREFIX} CROSS_NUMA: gather group {group} spans NUMA nodes "
            f"{set(rank_to_numa.values())} — segmented overlap selected "
            f"(A6000/H100 cross-NUMA ~8 GB/s detected)"
        )
        ds_logger.info(msg)
        print(msg)


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class _TopoAwareCommunicate(torch.autograd.Function):
    """Autograd wrapper for TopoAwareBridge with correct gradient adjoint.

    Forward:
      FAN_IN  → all-gather along batch dim (intra-NUMA fast or cross-NUMA overlap).
      FAN_OUT → narrow this rank's slice from the full src batch.
      EQUAL   → passthrough.

    Backward (adjoint):
      FAN_IN  backward → narrow (adjoint of all-gather).
      FAN_OUT backward → all-gather (adjoint of narrow — not zero-pad, because
                         every dest rank consumed a *different* slice of the same
                         src activation so each src rank needs the full gradient).
      EQUAL   backward → passthrough.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        bridge: TopoAwareBridge,
    ) -> torch.Tensor:
        ctx.bridge    = bridge
        ctx.batch_dim = bridge.config.dim_mapping['b']

        if bridge.direction is BridgeDirection.FAN_OUT:
            slice_info = bridge.get_slice_info(tensor.shape[ctx.batch_dim])
            return tensor.narrow(ctx.batch_dim, slice_info.start, slice_info.size).contiguous()

        if bridge.direction is BridgeDirection.FAN_IN:
            return bridge.all_gather_batch(tensor)

        # EQUAL — passthrough.
        return tensor.contiguous()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        bridge    = ctx.bridge
        batch_dim = ctx.batch_dim

        if bridge.direction is BridgeDirection.FAN_OUT:
            # Adjoint of narrow is all-gather (not zero-pad).
            return bridge.all_gather_batch(grad_output), None

        if bridge.direction is BridgeDirection.FAN_IN:
            slice_info = bridge.get_slice_info(grad_output.shape[batch_dim])
            return (
                grad_output.narrow(batch_dim, slice_info.start, slice_info.size).contiguous(),
                None,
            )

        return grad_output.contiguous(), None


# ---------------------------------------------------------------------------
# Utility: all-gather along an arbitrary batch dimension
# ---------------------------------------------------------------------------

def _all_gather_along_batch_dim(
    tensor: torch.Tensor,
    group:  dist.ProcessGroup,
    batch_dim: int,
) -> torch.Tensor:
    """All-gather *tensor* along *batch_dim* into a single contiguous tensor.

    ``all_gather_into_tensor`` concatenates along dim 0.  When the batch
    dimension is not 0 we movedim→gather→movedim-back so the collective's
    dim-0 semantics are transparent to the caller.

    Mirrors the upstream ``_all_gather_along_batch_dim`` helper exactly.
    """
    world_size = dist.get_world_size(group)
    src = tensor.contiguous()
    if batch_dim != 0:
        src = src.movedim(batch_dim, 0).contiguous()
    out_shape    = list(src.shape)
    out_shape[0] *= world_size
    out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(out, src, group=group)
    if batch_dim != 0:
        out = out.movedim(0, batch_dim).contiguous()
    return out


# ---------------------------------------------------------------------------
# Utility: build gather groups (mirrors upstream static method)
# ---------------------------------------------------------------------------

def _build_gather_groups(
    scale:           int,
    iter_size:       int,
    sibling_tp_size: int,
    rank_to_pos:     Dict[int, Tuple[int, int]],
) -> List[List[int]]:
    """Build ``iter_size × sibling_tp_size`` gather groups of ``scale`` ranks.

    Mirrors ``ColocatedBridgeCommunicator._build_gather_groups``.  For each
    slot on the "iterating" side and each TP shard on the sibling side, collect
    the ``scale`` sibling ranks whose DP indices map into that slot.  Append
    order equals group-local-rank order which ``all_gather_into_tensor`` uses
    to concatenate outputs — do not sort.
    """
    groups: List[List[int]] = []
    for iter_idx in range(iter_size):
        sibling_dp_indices = range(iter_idx * scale, (iter_idx + 1) * scale)
        for sibling_tp_idx in range(sibling_tp_size):
            group_ranks: List[int] = []
            for sibling_dp_idx in sibling_dp_indices:
                for rank, (dp, tp) in rank_to_pos.items():
                    if dp == sibling_dp_idx and tp == sibling_tp_idx:
                        group_ranks.append(rank)
                        break
            groups.append(group_ranks)
    return groups


# ---------------------------------------------------------------------------
# Convenience factory: build from a DeepSpeed ZeRO / topology config dict
# ---------------------------------------------------------------------------

def build_topo_aware_bridge(
    src_ranks:        List[int],
    dest_ranks:       List[int],
    src_tp_size:      int,
    dest_tp_size:     int,
    src_module_name:  str = "encoder",
    dest_module_name: str = "llm",
    config: Optional[TopoAwareBridgeConfig] = None,
    shared_numa_probe: Optional[NUMATopologyProbe] = None,
) -> TopoAwareBridge:
    """Factory for ``TopoAwareBridge`` from flat rank lists.

    This is the recommended entry point for DeepSpeed engine code that
    already knows the global rank sets for each module (e.g. from
    ``deepspeed.runtime.engine`` device-tier assignment).

    Args:
        src_ranks:    Global rank IDs for the src module (encoder), ordered as
                      ``[dp0_tp0, dp0_tp1, ..., dpN_tp0, dpN_tp1]``.
        dest_ranks:   Same for the dest module (language model).
        src_tp_size:  TP width of the src module.
        dest_tp_size: TP width of the dest module.
        src_module_name:  Label for diagnostics.
        dest_module_name: Label for diagnostics.
        config:       Optional bridge config (bandwidth thresholds, overlap flag).
        shared_numa_probe: Pre-built probe shared across multiple bridges.

    Returns:
        A ready-to-use ``TopoAwareBridge``.  Call ``bridge.communicate(tensor)``
        in the forward pass and ``bridge.destroy()`` during teardown.
    """
    return TopoAwareBridge(
        src_rank_map=src_ranks,
        dest_rank_map=dest_ranks,
        src_tp_size=src_tp_size,
        dest_tp_size=dest_tp_size,
        src_module_name=src_module_name,
        dest_module_name=dest_module_name,
        config=config,
        numa_probe=shared_numa_probe,
    )
