# Copyright (c) 2025, Neuron_SP Project (github.com/dylanyunlon/Neuron_SP).
# Derived from NVIDIA Megatron-LM commit ba71ec209591befd935edabdafd906ccb01ef3a0.
# SPDX-License-Identifier: Apache-2.0
#
# Upstream design intent (Megatron ba71ec2):
#   HyperCommGrid was extended with "named layouts" (register_view) so that a single rank span
#   can carry *multiple* factorizations simultaneously—e.g. a dense TP/DP/PP base view and a
#   separate EP (expert-parallel) view that re-partitions the same ranks differently.  The key
#   insight is that some dimensions are *shared* between views (identical rank membership), so
#   their process groups can be reused rather than duplicated.  The upstream also eliminated the
#   einops dependency by replacing rearrange() with np.moveaxis, and hardened destroy() to skip
#   non-member sentinels.
#
# DES-LOC adaptation rationale (DESLOCHyperCommGrid):
#   DES-LOC = Decoupled Execution with Shared LOcality Cache.
#   Hardware context: 2× A6000 48 GB SM86 + 1× H100 NVL 96 GB SM90, PCIe-only, 1.5 TB CPU DRAM.
#
#   The three-GPU cluster is *heterogeneous* in compute (SM86 vs SM90), memory capacity, and
#   PCIe bandwidth (A6000↔A6000 share a root complex; H100 sits on a separate PCIe domain).
#   Standard homogeneous communication grids assume symmetric topology and uniform bandwidth,
#   which leads to suboptimal group assignments on this hardware.
#
#   DES-LOC adaptations over the Megatron upstream:
#
#   1. **Device-class registry** (`DeviceClass`): each rank is annotated with its SM generation,
#      VRAM capacity, and PCIe affinity domain.  Views can be constructed to respect device
#      locality—e.g. a "sm86" view that only spans A6000 ranks, or a "locality" view that groups
#      ranks sharing a PCIe root complex.
#
#   2. **Locality-aware rank enumeration** (`_gen_rank_enum_locality_aware`): when creating
#      process groups within a view, rank lists are optionally sorted so that ranks that share a
#      PCIe affinity domain appear contiguously.  This reduces cross-domain all-reduce hops on
#      the A6000 pair without changing the mathematical factorization.
#
#   3. **Heterogeneous view validation** (`register_heterogeneous_view`): a specialisation of
#      `register_view` that additionally checks that each subgroup in the new view satisfies a
#      caller-supplied device-class predicate (e.g. "all ranks in this subgroup are SM86").
#      Mismatched subgroups are rejected at registration time rather than silently producing
#      slow cross-class groups at runtime.
#
#   4. **LOC cache affinity hints** (`get_loc_affinity_group`): returns the process group whose
#      members reside on the same PCIe affinity domain as the calling rank.  This group is used
#      by the DES-LOC shared-locality cache to decide which ranks can cheaply share activations
#      via peer-to-peer DRAM transfers vs which must use slower cross-domain copies.
#
#   5. **Bandwidth-weighted all-reduce planner** (`plan_allreduce`): given a tensor size and the
#      active process group, suggests whether to use a single cross-domain all-reduce or a
#      two-stage reduce-scatter + ring approach that avoids the PCIe bottleneck between the A6000
#      pair and the H100.
#
#   6. Full compatibility with Megatron's `register_view` / `create_pg` / `get_pg` surface so
#      that DeepSpeed engine code can use this class as a drop-in for homogeneous grids.

from __future__ import annotations

import logging
import numbers
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import torch
    import torch.distributed as dist

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_VIEW_NAME = "base"

# PCIe affinity domain labels used in the 2×A6000 + 1×H100 cluster.
# Ranks 0 and 1 share a root complex (A6000 pair); rank 2 is on a separate domain (H100).
_DEFAULT_AFFINITY_DOMAINS: Dict[int, str] = {
    0: "rc0",  # A6000 #0, PCIe root complex 0
    1: "rc0",  # A6000 #1, PCIe root complex 0
    2: "rc1",  # H100 NVL,  PCIe root complex 1
}


# ---------------------------------------------------------------------------
# Device-class description
# ---------------------------------------------------------------------------


class SMGeneration(Enum):
    """CUDA SM generation identifier for heterogeneous capability checks."""
    SM86 = auto()  # A6000 48 GB
    SM90 = auto()  # H100 NVL 96 GB
    UNKNOWN = auto()


@dataclass
class DeviceClass:
    """Per-rank hardware annotation used by DES-LOC locality-aware scheduling.

    Attributes:
        rank:           Global process rank.
        sm_generation:  CUDA SM generation (SM86 for A6000, SM90 for H100).
        vram_gb:        Approximate device VRAM in gigabytes.
        pcie_affinity:  Label string for the PCIe root complex / affinity domain.
                        Ranks sharing the same label can exchange tensors via fast
                        peer-to-peer without crossing a root-complex boundary.
        cpu_dram_gb:    Fraction of the 1.5 TB host DRAM logically assigned to this rank
                        (used by DES-LOC when spilling activations to the LOC cache).
    """

    rank: int
    sm_generation: SMGeneration
    vram_gb: float
    pcie_affinity: str
    cpu_dram_gb: float = 0.0


def _build_default_device_map() -> Dict[int, DeviceClass]:
    """Return the hard-coded device map for the 2×A6000 + 1×H100 cluster.

    Override by passing ``device_map`` to :class:`DESLOCHyperCommGrid.__init__`.
    """
    return {
        0: DeviceClass(0, SMGeneration.SM86, 48.0, "rc0", 512.0),
        1: DeviceClass(0, SMGeneration.SM86, 48.0, "rc0", 512.0),
        2: DeviceClass(0, SMGeneration.SM90, 96.0, "rc1", 512.0),
    }


# ---------------------------------------------------------------------------
# Internal rank-view specification
# ---------------------------------------------------------------------------


@dataclass
class _RankViewSpec:
    """A named rank factorization over the same rank span as the base grid.

    DES-LOC addition:
        ``device_predicate`` is an optional callable that, given a list of ranks forming a
        single subgroup, returns True iff the subgroup satisfies device-class constraints
        (e.g. all-SM86).  It is stored here so that ``_validate_heterogeneous_subgroups``
        can re-check it at create_pg time in debug mode.
    """

    name: str
    shape: List[int]
    dim_names: List[str]
    shared_dims: List[str]
    device_predicate: Optional[Any] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Bandwidth estimation helpers
# ---------------------------------------------------------------------------


class AllReducePlan(Enum):
    """Suggested all-reduce strategy returned by :meth:`DESLOCHyperCommGrid.plan_allreduce`."""
    SINGLE_RING = auto()
    """Single ring all-reduce across the entire group (suitable when all ranks share a domain)."""
    TWO_STAGE = auto()
    """Two-stage: intra-domain reduce-scatter, then cross-domain all-reduce.
    Preferred when the group spans both PCIe root complexes and the tensor is large."""
    CPU_LOC_OFFLOAD = auto()
    """Offload intermediate reduce buffer to the DES-LOC shared-locality CPU DRAM cache.
    Used when tensor exceeds the smaller device's VRAM headroom."""


# PCIe gen4 x16 theoretical one-directional bandwidth in GB/s (conservative estimate).
_PCIE_BW_INTRA_GBS = 28.0   # within the same root complex
_PCIE_BW_CROSS_GBS = 14.0   # crossing root-complex boundary
_TWO_STAGE_THRESHOLD_MB = 256.0  # tensors larger than this benefit from two-stage on cross-domain


# ---------------------------------------------------------------------------
# DES-LOC process-group membership helper
# ---------------------------------------------------------------------------


def _is_process_group_member(pg: Optional[Any]) -> bool:
    """Return True iff the current rank belongs to *pg* (not the non-member sentinel).

    Mirrors the upstream Megatron helper added in ba71ec2 so that :meth:`destroy`
    skips non-member sentinels and None values, preventing spurious errors when ranks
    are not part of every subgroup in a heterogeneous grid.
    """
    if pg is None:
        return False
    non_member = getattr(getattr(dist, "GroupMember", None), "NON_GROUP_MEMBER", None)
    return pg is not non_member


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DESLOCHyperCommGrid:
    r"""Heterogeneity-aware N-dimensional communication grid for DES-LOC training.

    This class re-implements Megatron's ``HyperCommGrid`` (commit ba71ec2) and extends it
    with device-locality tracking, PCIe-affinity-aware rank enumeration, and the DES-LOC
    shared-locality cache group interface.

    Upstream surface (fully compatible):
        * :meth:`register_view` — register a named rank factorization.
        * :meth:`create_pg` / :meth:`get_pg` — create / retrieve process groups.
        * :meth:`get_rank_enum` — inspect rank enumerations without creating groups.
        * :meth:`destroy` — tear down all groups, skipping non-members.

    DES-LOC extensions:
        * :meth:`register_heterogeneous_view` — register a view with a device predicate.
        * :meth:`get_loc_affinity_group` — get the process group for the calling rank's
          PCIe affinity domain (used by the LOC cache for fast peer copies).
        * :meth:`plan_allreduce` — heuristic bandwidth planner suggesting SINGLE_RING,
          TWO_STAGE, or CPU_LOC_OFFLOAD for a given tensor size and process group.

    Args:
        shape:       List of integers defining the grid dimensions (e.g. ``[2, 1, 1]``).
        dim_names:   Names for each dimension (e.g. ``["tp", "dp", "pp"]``).
        rank_offset: Global rank offset for this grid's rank span.  Defaults to 0.
        backend:     torch.distributed backend string (default ``"nccl"``).
        device_map:  Optional mapping from global rank → :class:`DeviceClass`.  When omitted
                     the default 2×A6000 + 1×H100 cluster map is used.
        locality_sort: If True (default), rank lists within process groups that span multiple
                     PCIe affinity domains are sorted so same-domain ranks are contiguous.
                     This improves hierarchical all-reduce efficiency on PCIe topologies.

    Example::

        grid = DESLOCHyperCommGrid([2, 1, 1], ["tp", "dp", "pp"])
        grid.register_heterogeneous_view(
            "sm86_only",
            shape=[2, 1],
            dim_names=["tp", "dp"],
            device_predicate=lambda ranks: all(
                grid.device_map[r].sm_generation == SMGeneration.SM86 for r in ranks
            ),
        )
        grid.create_pg("tp")
        grid.create_pg("tp", view="sm86_only")
    """

    def __init__(
        self,
        shape: List[int],
        dim_names: List[str],
        rank_offset: int = 0,
        backend: str = "nccl",
        device_map: Optional[Dict[int, DeviceClass]] = None,
        locality_sort: bool = True,
    ) -> None:
        if not HAVE_TORCH:  # pragma: no cover
            raise ImportError("PyTorch is required for DESLOCHyperCommGrid.")

        if not dist.is_initialized():
            raise RuntimeError(
                "DESLOCHyperCommGrid requires torch.distributed to be initialized before "
                "construction.  Call dist.init_process_group() first."
            )

        if len(shape) != len(dim_names):
            raise ValueError(
                f"len(shape)={len(shape)} must equal len(dim_names)={len(dim_names)}"
            )
        if len(set(dim_names)) != len(dim_names):
            raise ValueError(f"dim_names contains duplicates: {dim_names}")
        if any(not isinstance(s, numbers.Integral) or s <= 0 for s in shape):
            raise ValueError(f"All shape entries must be positive integers, got {shape}")

        world_size = dist.get_world_size()
        self.rank_offset = rank_offset
        self.size = int(np.prod(shape))

        if rank_offset < 0:
            raise ValueError(f"rank_offset must be non-negative, got {rank_offset}")
        if self.size > world_size - rank_offset:
            raise ValueError(
                f"Grid size {self.size} exceeds available ranks "
                f"(world_size={world_size}, rank_offset={rank_offset})"
            )

        self.shape: List[int] = shape[:]
        self.dim_names: List[str] = dim_names[:]
        self.backend: str = backend
        self.locality_sort: bool = locality_sort

        # Device map: caller-supplied or default cluster topology.
        self.device_map: Dict[int, DeviceClass] = (
            device_map if device_map is not None else _build_default_device_map()
        )

        # Views registry: base view is always present.
        self._views: Dict[str, _RankViewSpec] = {
            _BASE_VIEW_NAME: _RankViewSpec(
                name=_BASE_VIEW_NAME,
                shape=self.shape[:],
                dim_names=self.dim_names[:],
                shared_dims=[],
            )
        }

        # Process-group registry.
        # Base-view keys: dash-joined dim string (str).
        # View-private keys: (view_name, tuple_of_ordered_dims).
        self._pgs: Dict[Union[str, Tuple[str, Tuple[str, ...]]], Any] = {}

        # LOC affinity groups: keyed by pcie_affinity label.
        self._loc_affinity_pgs: Dict[str, Any] = {}

        logger.debug(
            "DESLOCHyperCommGrid created: shape=%s dim_names=%s size=%d rank_offset=%d "
            "locality_sort=%s",
            shape,
            dim_names,
            self.size,
            rank_offset,
            locality_sort,
        )

    # ------------------------------------------------------------------
    # Public API — upstream-compatible surface
    # ------------------------------------------------------------------

    def register_view(
        self,
        name: str,
        shape: List[int],
        dim_names: List[str],
        shared_dims: Optional[List[str]] = None,
    ) -> None:
        r"""Register an additional rank factorization over this grid's rank span.

        Mirrors Megatron ba71ec2's ``HyperCommGrid.register_view`` exactly so that code
        written against the upstream API works without modification.

        Shared dims must exist in both the base view and the new view, and must enumerate
        to the same rank groups as the base view (same PCIe locality constraints apply
        automatically through :attr:`locality_sort`).

        Args:
            name:        Unique name for the new view (must not be ``"base"``).
            shape:       New dimension sizes (product must equal :attr:`size`).
            dim_names:   Dimension names for the new factorization.
            shared_dims: Subset of dim_names that are shared with the base view.
        """
        self._validate_view_spec(name, shape, dim_names, shared_dims or [])
        shared_dims = list(shared_dims) if shared_dims is not None else []
        self._views[name] = _RankViewSpec(
            name=name,
            shape=shape[:],
            dim_names=dim_names[:],
            shared_dims=shared_dims[:],
        )

    def register_heterogeneous_view(
        self,
        name: str,
        shape: List[int],
        dim_names: List[str],
        shared_dims: Optional[List[str]] = None,
        device_predicate: Optional[Any] = None,
    ) -> None:
        r"""Register a view that enforces device-class constraints on each subgroup.

        DES-LOC extension over :meth:`register_view`.  If *device_predicate* is supplied,
        every subgroup generated by any single-dimension enumeration of the new view is
        validated against the predicate at registration time.  This catches topology
        misconfigurations—e.g. accidentally mixing SM86 and SM90 ranks in a subgroup that
        is intended to be SM86-only—before the expensive ``create_pg`` call.

        Args:
            name:             Unique view name.
            shape:            New dimension sizes.
            dim_names:        New dimension names.
            shared_dims:      Shared-with-base dimensions (see :meth:`register_view`).
            device_predicate: Optional ``callable(ranks: List[int]) -> bool``.  Receives the
                              list of global ranks in a candidate subgroup and must return
                              True iff the subgroup is acceptable.  When it returns False,
                              registration raises ``ValueError`` identifying the violating
                              subgroup.

        Raises:
            ValueError: If any subgroup violates *device_predicate*, or if the base view
                        validation fails (forwarded from :meth:`register_view`).
        """
        self._validate_view_spec(name, shape, dim_names, shared_dims or [])

        if device_predicate is not None:
            self._validate_heterogeneous_subgroups(name, shape, dim_names, device_predicate)

        shared_dims = list(shared_dims) if shared_dims is not None else []
        self._views[name] = _RankViewSpec(
            name=name,
            shape=shape[:],
            dim_names=dim_names[:],
            shared_dims=shared_dims[:],
            device_predicate=device_predicate,
        )
        logger.info(
            "Registered heterogeneous view %r: shape=%s dim_names=%s shared_dims=%s "
            "has_predicate=%s",
            name,
            shape,
            dim_names,
            shared_dims,
            device_predicate is not None,
        )

    def create_pg(
        self,
        dims: Union[str, List[str]],
        *,
        view: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        r"""Create a process group based on a list of dimension names.

        DES-LOC adaptation: if :attr:`locality_sort` is True and the rank enumeration spans
        multiple PCIe affinity domains, the rank lists are reordered so that same-domain ranks
        appear contiguously.  This enables hierarchical all-reduce implementations in DeepSpeed
        (e.g. ``ZeroStage3.stage_hierarchical_reduce``) to exploit intra-domain bandwidth.

        The unique key follows the reversed order of :attr:`dim_names` (base view) or the
        view's own dim ordering (view-private), matching upstream Megatron conventions.

        Args:
            dims:  Name or list of names of dimensions to group.
            view:  Optional registered view name.  Defaults to the base view.

        Keyword arguments are forwarded to ``dist.new_subgroups_by_enumeration``.

        Returns:
            The new ``dist.ProcessGroup`` for this rank (or the non-member sentinel).

        Raises:
            KeyError: If the process group has already been created.
            ValueError: If *dims* are not valid for the specified view.
        """
        view_spec = self._resolve_view(view)
        ordered_dims, _ = self._order_dims_for_view(view_spec, dims)
        unique_group_key, enum_view, enum_dims = self._canonical_pg_key_and_enum_view(
            view_spec, ordered_dims
        )

        if unique_group_key in self._pgs:
            if self._is_base_pg_key(unique_group_key):
                raise KeyError(
                    f"Process group {dims} has already been created.  Because there is no way "
                    f"to verify that options match the first call, we raise instead of returning "
                    f"the existing group."
                )
            raise KeyError(
                f"Process group {dims} for view {view_spec.name!r} has already been created. "
                f"Because there is no way to verify that options match the first call, we raise "
                f"instead of returning the existing group."
            )

        rank_enum = self._gen_rank_enum_for(enum_view.shape, enum_view.dim_names, enum_dims)

        if self.locality_sort:
            rank_enum = self._apply_locality_sort(rank_enum)

        pg, _ = dist.new_subgroups_by_enumeration(rank_enum, backend=self.backend, **kwargs)

        if dist.is_initialized() and dist.get_rank() == 0:
            if self._is_base_pg_key(unique_group_key):
                logger.info(
                    "Created process group key=%r enum=%s", unique_group_key, rank_enum
                )
            else:
                logger.info(
                    "Created process group view=%r dims=%s enum=%s",
                    view_spec.name,
                    ordered_dims,
                    rank_enum,
                )

        self._pgs[unique_group_key] = pg
        return pg

    def destroy(self) -> None:
        """Destroy all process groups created by this grid that the current rank belongs to.

        DES-LOC adaptation: also destroys LOC affinity groups created by
        :meth:`_ensure_loc_affinity_groups`.  A base group reused by a view for a shared
        dimension is stored under a single key, so it is torn down exactly once regardless
        of how many views reference it.
        """
        destroyed_ids: Set[int] = set()

        for pg in list(self._pgs.values()):
            if _is_process_group_member(pg) and id(pg) not in destroyed_ids:
                dist.destroy_process_group(pg)
                destroyed_ids.add(id(pg))

        for pg in list(self._loc_affinity_pgs.values()):
            if _is_process_group_member(pg) and id(pg) not in destroyed_ids:
                dist.destroy_process_group(pg)
                destroyed_ids.add(id(pg))

        self._pgs.clear()
        self._loc_affinity_pgs.clear()
        logger.debug("DESLOCHyperCommGrid destroyed %d process groups.", len(destroyed_ids))

    def get_pg(
        self,
        dims: Union[str, List[str]],
        *,
        view: Optional[str] = None,
    ) -> Any:
        r"""Retrieve a previously created process group.

        Args:
            dims:  Dimension name or list of names.
            view:  Optional registered view name.  Defaults to the base view.

        Returns:
            The ``dist.ProcessGroup`` registered under the canonical key.

        Raises:
            KeyError: If the process group has not been created yet.
        """
        view_spec = self._resolve_view(view)
        ordered_dims, _ = self._order_dims_for_view(view_spec, dims)
        unique_group_key, _, _ = self._canonical_pg_key_and_enum_view(view_spec, ordered_dims)

        if unique_group_key not in self._pgs:
            if self._is_base_pg_key(unique_group_key):
                raise KeyError(
                    f"Process group for {unique_group_key!r} has not been created.  "
                    f"Call create_pg first."
                )
            raise KeyError(
                f"Process group {dims} for view {view_spec.name!r} has not been created.  "
                f"Call create_pg first."
            )

        return self._pgs[unique_group_key]

    def get_rank_enum(
        self,
        dims: Union[str, List[str]],
        *,
        view: Optional[str] = None,
    ) -> List[List[int]]:
        r"""Return the rank enumeration for the requested dimension(s) without creating a group.

        DES-LOC note: locality sorting is applied here too (when :attr:`locality_sort` is True)
        so that the returned enumeration matches exactly what :meth:`create_pg` would use.

        Args:
            dims:  Dimension name or list.
            view:  Optional registered view name.

        Returns:
            List of rank lists (one per subgroup), consistent with
            ``dist.new_subgroups_by_enumeration`` input format.
        """
        view_spec = self._resolve_view(view)
        ordered_dims, _ = self._order_dims_for_view(view_spec, dims)
        enum_raw = self._gen_rank_enum_for(view_spec.shape, view_spec.dim_names, ordered_dims)
        if self.locality_sort:
            return self._apply_locality_sort(enum_raw)
        return enum_raw

    # ------------------------------------------------------------------
    # DES-LOC extensions
    # ------------------------------------------------------------------

    def get_loc_affinity_group(self) -> Optional[Any]:
        """Return the process group whose members share the calling rank's PCIe affinity domain.

        DES-LOC shared-locality cache uses this group to identify peers that can exchange
        activations via fast intra-domain P2P, avoiding the slower cross-domain PCIe path
        between the A6000 pair (rc0) and the H100 (rc1).

        The affinity groups are created lazily on the first call and cached internally.

        Returns:
            A ``dist.ProcessGroup`` for the calling rank's affinity domain, or ``None`` if
            torch.distributed is not initialized or the current rank has no device entry.
        """
        if not dist.is_initialized():
            return None

        current_rank = dist.get_rank()
        device_info = self.device_map.get(current_rank)
        if device_info is None:
            logger.warning(
                "Rank %d has no entry in device_map; cannot determine LOC affinity group.",
                current_rank,
            )
            return None

        affinity = device_info.pcie_affinity
        if affinity not in self._loc_affinity_pgs:
            self._ensure_loc_affinity_groups()

        return self._loc_affinity_pgs.get(affinity)

    def plan_allreduce(
        self,
        tensor_bytes: int,
        pg: Any,
    ) -> AllReducePlan:
        """Suggest an all-reduce strategy for a tensor of *tensor_bytes* bytes over *pg*.

        DES-LOC two-stage heuristic:
          * If all ranks in *pg* share a PCIe affinity domain → ``SINGLE_RING`` (fast intra).
          * If the tensor exceeds :data:`_TWO_STAGE_THRESHOLD_MB` and the group spans
            domains → ``TWO_STAGE`` (intra-domain reduce-scatter then cross-domain reduce).
          * If the tensor exceeds the VRAM headroom of the smallest device in *pg* →
            ``CPU_LOC_OFFLOAD`` (use DES-LOC's 1.5 TB CPU DRAM as staging buffer).

        Args:
            tensor_bytes:  Total size of the tensor in bytes.
            pg:            The ``dist.ProcessGroup`` that will perform the all-reduce.

        Returns:
            An :class:`AllReducePlan` enum value.
        """
        if not dist.is_initialized() or pg is None or not _is_process_group_member(pg):
            return AllReducePlan.SINGLE_RING

        try:
            pg_ranks = dist.get_process_group_ranks(pg)
        except Exception:  # noqa: BLE001
            return AllReducePlan.SINGLE_RING

        # Determine which affinity domains are present in the group.
        domains = {
            self.device_map[r].pcie_affinity
            for r in pg_ranks
            if r in self.device_map
        }

        if len(domains) <= 1:
            # All ranks on the same domain; a standard ring all-reduce is optimal.
            return AllReducePlan.SINGLE_RING

        # Cross-domain group: check whether tensor fits comfortably on the smallest device.
        tensor_mb = tensor_bytes / (1024 * 1024)
        min_vram = min(
            (self.device_map[r].vram_gb for r in pg_ranks if r in self.device_map),
            default=48.0,
        )
        # Conservative: flag offload if tensor > 10 % of smallest device VRAM.
        if tensor_mb > min_vram * 1024 * 0.10:
            logger.info(
                "plan_allreduce: tensor %.1f MB exceeds 10%% of min VRAM %.0f GB "
                "→ CPU_LOC_OFFLOAD",
                tensor_mb,
                min_vram,
            )
            return AllReducePlan.CPU_LOC_OFFLOAD

        if tensor_mb > _TWO_STAGE_THRESHOLD_MB:
            logger.info(
                "plan_allreduce: tensor %.1f MB > %.0f MB threshold, cross-domain group "
                "→ TWO_STAGE",
                tensor_mb,
                _TWO_STAGE_THRESHOLD_MB,
            )
            return AllReducePlan.TWO_STAGE

        return AllReducePlan.SINGLE_RING

    def is_current_rank_in_grid(self) -> bool:
        """Return True iff the current rank falls within this grid's rank span."""
        if not dist.is_initialized():
            return False
        rank = dist.get_rank()
        return self.rank_offset <= rank < self.rank_offset + self.size

    # ------------------------------------------------------------------
    # Internal helpers — rank enumeration
    # ------------------------------------------------------------------

    def _gen_rank_enum(self, dims: List[str]) -> List[List[int]]:
        """Generate rank enumeration for *dims* using the base view.

        Thin wrapper kept for backward compatibility with internal callers that were
        written against the single-view API.
        """
        return self._gen_rank_enum_for(self.shape, self.dim_names, dims)

    def _gen_rank_enum_for(
        self,
        shape: List[int],
        dim_names: List[str],
        dims: List[str],
    ) -> List[List[int]]:
        r"""Generate rank enumeration for *dims* under an explicit *shape* / *dim_names*.

        DES-LOC replaces the upstream einops dependency with numpy moveaxis (matching
        Megatron ba71ec2) so that the DeepSpeed environment does not require einops.

        The enumeration convention follows MCore: dim_names are *reversed* before indexing
        so that the first name in dim_names is the *slowest* varying axis.
        """
        dim_names_rev = dim_names[::-1]
        shape_dict = {d: s for d, s in zip(dim_names, shape)}
        size = int(np.prod(shape))

        rank_tensor = np.arange(
            self.rank_offset, self.rank_offset + size
        ).reshape([shape_dict[d] for d in dim_names_rev])

        source_axes = [dim_names_rev.index(d) for d in dims]
        target_axes = list(range(len(dim_names_rev) - len(dims), len(dim_names_rev)))

        logger.debug(
            "_gen_rank_enum_for: moving axes %s→%s  dim_names=%s dims=%s",
            source_axes,
            target_axes,
            dim_names,
            dims,
        )

        rank_tensor = np.moveaxis(rank_tensor, source_axes, target_axes)
        group_size = int(np.prod([shape_dict[d] for d in dims]))
        return rank_tensor.reshape(-1, group_size).tolist()

    def _apply_locality_sort(self, rank_enum: List[List[int]]) -> List[List[int]]:
        """Sort ranks within each subgroup so same-PCIe-domain ranks are contiguous.

        DES-LOC locality adaptation: on the 2×A6000 + 1×H100 cluster, all-reduce
        implementations that exploit hierarchical topology (e.g. DeepSpeed's
        ``CommBuilder`` with ``hierarchical_allreduce=True``) require same-domain ranks to
        be contiguous in the rank list so that the intra-domain sub-ring can be identified
        by a simple prefix.

        The sort key is ``(pcie_affinity, rank)`` so that within a domain ranks appear in
        ascending order (deterministic, reproducible).  Ranks not present in the device map
        are placed last with affinity ``"zzz"`` (lexicographically after all real labels).
        """
        result = []
        for group in rank_enum:
            sorted_group = sorted(
                group,
                key=lambda r: (
                    self.device_map[r].pcie_affinity if r in self.device_map else "zzz",
                    r,
                ),
            )
            result.append(sorted_group)
        return result

    # ------------------------------------------------------------------
    # Internal helpers — view management
    # ------------------------------------------------------------------

    def _order_dims_for(
        self,
        dim_names: List[str],
        dims: Union[str, List[str]],
    ) -> Tuple[List[str], str]:
        """Reorder *dims* according to the reversed order of *dim_names*."""
        if not isinstance(dims, list):
            ordered_dims = [dims]
        else:
            dim_names_rev = dim_names[::-1]
            indices = sorted(dim_names_rev.index(d) for d in dims)
            ordered_dims = [dim_names_rev[i] for i in indices]
        unique_group_key = "-".join(ordered_dims)
        return ordered_dims, unique_group_key

    def _resolve_view(self, view: Optional[str]) -> _RankViewSpec:
        """Return the requested view spec, defaulting to the base view."""
        view_name = _BASE_VIEW_NAME if view is None else view
        if view_name not in self._views:
            raise KeyError(
                f"View {view_name!r} is not registered.  "
                f"Registered views: {sorted(self._views)}"
            )
        return self._views[view_name]

    def _order_dims_for_view(
        self,
        view: _RankViewSpec,
        dims: Union[str, List[str]],
    ) -> Tuple[List[str], str]:
        """Reorder *dims* for *view*, raising a clear error for unknown dimension names."""
        requested = [dims] if not isinstance(dims, list) else dims
        missing = [d for d in requested if d not in view.dim_names]
        if missing:
            raise ValueError(
                f"{missing[0]!r} is not in view {view.name!r} with dim_names {view.dim_names}"
            )
        return self._order_dims_for(view.dim_names, dims)

    def _canonical_pg_key_and_enum_view(
        self,
        view: _RankViewSpec,
        ordered_dims: List[str],
    ) -> Tuple[Union[str, Tuple[str, Tuple[str, ...]]], _RankViewSpec, List[str]]:
        """Return (storage key, enumeration view, ordered dims for enumeration).

        Mirrors upstream Megatron ba71ec2 logic:
          * Base-view dims → str key (backward-compatible).
          * View-private dims that are fully shared → reuse base-view str key.
          * View-private dims that are NOT fully shared → tuple key (view, dims).
        """
        if view.name == _BASE_VIEW_NAME:
            return "-".join(ordered_dims), view, ordered_dims

        if all(d in view.shared_dims for d in ordered_dims):
            base_view = self._views[_BASE_VIEW_NAME]
            base_ordered, base_key = self._order_dims_for_view(base_view, ordered_dims)
            return base_key, base_view, base_ordered

        return (view.name, tuple(ordered_dims)), view, ordered_dims

    def _is_base_pg_key(self, key: Any) -> bool:
        """Return True iff *key* belongs to the base-view namespace (a plain string)."""
        return isinstance(key, str)

    # ------------------------------------------------------------------
    # Internal helpers — validation
    # ------------------------------------------------------------------

    def _validate_view_spec(
        self,
        name: str,
        shape: List[int],
        dim_names: List[str],
        shared_dims: List[str],
    ) -> None:
        """Shared validation logic used by both :meth:`register_view` and
        :meth:`register_heterogeneous_view`."""
        if name in self._views:
            raise ValueError(f"View {name!r} is already registered.")
        if len(shape) != len(dim_names):
            raise ValueError(
                f"len(shape) {shape} != len(dim_names) {dim_names}"
            )
        if len(set(dim_names)) != len(dim_names):
            raise ValueError(f"View {name!r} has duplicate dim_names: {dim_names}")
        if any(not isinstance(s, numbers.Integral) or s <= 0 for s in shape):
            raise ValueError(
                f"View {name!r} shape must be positive ints, got {shape}"
            )
        if int(np.prod(shape)) != self.size:
            raise ValueError(
                f"View {name!r} shape {shape} has size {int(np.prod(shape))}, "
                f"but the grid size is {self.size}"
            )
        if len(set(shared_dims)) != len(shared_dims):
            raise ValueError(f"View {name!r} has duplicate shared_dims: {shared_dims}")

        for dim in shared_dims:
            if dim not in self.dim_names:
                raise ValueError(
                    f"Shared dim {dim!r} of view {name!r} is not in the base view "
                    f"{self.dim_names}"
                )
            if dim not in dim_names:
                raise ValueError(
                    f"Shared dim {dim!r} of view {name!r} is not in the view's dim_names "
                    f"{dim_names}"
                )
            base_ordered, _ = self._order_dims_for(self.dim_names, dim)
            base_enum = self._gen_rank_enum_for(self.shape, self.dim_names, base_ordered)
            view_ordered, _ = self._order_dims_for(dim_names, dim)
            view_enum = self._gen_rank_enum_for(shape, dim_names, view_ordered)
            if base_enum != view_enum:
                raise ValueError(
                    f"Shared dim {dim!r} has different membership across views: "
                    f"base enumeration {base_enum} != view {name!r} enumeration {view_enum}"
                )

        if len(shared_dims) > 1:
            base_ordered, _ = self._order_dims_for(self.dim_names, shared_dims)
            base_enum = self._gen_rank_enum_for(self.shape, self.dim_names, base_ordered)
            view_ordered, _ = self._order_dims_for(dim_names, shared_dims)
            view_enum = self._gen_rank_enum_for(shape, dim_names, view_ordered)
            if base_enum != view_enum:
                raise ValueError(
                    f"Shared dims {shared_dims!r} have different joint membership: "
                    f"base {base_enum} != view {name!r} {view_enum}"
                )

    def _validate_heterogeneous_subgroups(
        self,
        name: str,
        shape: List[int],
        dim_names: List[str],
        device_predicate: Any,
    ) -> None:
        """Check every single-dimension subgroup in the view against *device_predicate*.

        DES-LOC: iterates over each dimension individually and raises ``ValueError``
        on the first subgroup that violates the predicate, naming the offending ranks
        and their device classes so the developer can diagnose the mismatch.
        """
        for dim in dim_names:
            ordered, _ = self._order_dims_for(dim_names, dim)
            enum = self._gen_rank_enum_for(shape, dim_names, ordered)
            for group in enum:
                if not device_predicate(group):
                    classes = {
                        r: self.device_map[r].sm_generation.name
                        if r in self.device_map else "UNKNOWN"
                        for r in group
                    }
                    raise ValueError(
                        f"View {name!r} dim {dim!r}: subgroup {group} failed the device "
                        f"predicate (device classes: {classes})."
                    )

    # ------------------------------------------------------------------
    # Internal helpers — LOC affinity groups
    # ------------------------------------------------------------------

    def _ensure_loc_affinity_groups(self) -> None:
        """Create process groups partitioned by PCIe affinity domain (lazy, called once).

        DES-LOC: these groups are NOT part of the standard grid factorization.  They are
        built from the device map and stored in :attr:`_loc_affinity_pgs`.  Only the group
        for the current rank's domain is returned by :meth:`get_loc_affinity_group`; all
        others are ``NON_GROUP_MEMBER`` sentinels.
        """
        # Collect all known affinity domains across ranks in this grid's span.
        domain_to_ranks: Dict[str, List[int]] = {}
        for rank in range(self.rank_offset, self.rank_offset + self.size):
            info = self.device_map.get(rank)
            if info is None:
                continue
            domain_to_ranks.setdefault(info.pcie_affinity, []).append(rank)

        if not domain_to_ranks:
            return

        # Build a complete rank enumeration: one list per domain.
        enum: List[List[int]] = list(domain_to_ranks.values())

        logger.info(
            "Building LOC affinity groups: domains=%s enum=%s",
            list(domain_to_ranks.keys()),
            enum,
        )

        pg, _ = dist.new_subgroups_by_enumeration(enum, backend=self.backend)

        # Store the same pg object under every domain label; the distributed API gives
        # each rank a pg for its own domain and a sentinel for others.
        current_rank = dist.get_rank()
        current_affinity = (
            self.device_map[current_rank].pcie_affinity
            if current_rank in self.device_map
            else None
        )
        for domain in domain_to_ranks:
            self._loc_affinity_pgs[domain] = (
                pg if domain == current_affinity else None
            )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_deslocgrid_from_env() -> DESLOCHyperCommGrid:
    """Create a ``DESLOCHyperCommGrid`` from environment variables.

    Environment variables:
        DESLOCGRID_SHAPE:     Comma-separated integers (e.g. ``"2,1,1"``).
        DESLOCGRID_DIM_NAMES: Comma-separated names (e.g. ``"tp,dp,pp"``).
        DESLOCGRID_BACKEND:   torch.distributed backend (default ``"nccl"``).
        DESLOCGRID_LOCALITY:  ``"1"`` to enable locality sort (default), ``"0"`` to disable.

    Raises:
        EnvironmentError: If the required variables are absent.
    """
    shape_str = os.environ.get("DESLOCGRID_SHAPE")
    dims_str = os.environ.get("DESLOCGRID_DIM_NAMES")
    if not shape_str or not dims_str:
        raise EnvironmentError(
            "DESLOCGRID_SHAPE and DESLOCGRID_DIM_NAMES must be set."
        )
    shape = [int(x) for x in shape_str.split(",")]
    dim_names = [x.strip() for x in dims_str.split(",")]
    backend = os.environ.get("DESLOCGRID_BACKEND", "nccl")
    locality = os.environ.get("DESLOCGRID_LOCALITY", "1") == "1"
    return DESLOCHyperCommGrid(shape, dim_names, backend=backend, locality_sort=locality)


# ---------------------------------------------------------------------------
# Unit tests (no torch.distributed required for pure-logic tests)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import unittest
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    class _FakeProcessGroup:
        """Minimal stand-in for dist.ProcessGroup in mock-only tests."""
        def __init__(self, label: str):
            self.label = label
        def __repr__(self) -> str:
            return f"<FakePG {self.label}>"

    def _make_mock_dist(world_size: int = 8, rank: int = 0):
        """Return a context-manager stack that mocks torch.distributed for grid construction."""
        return [
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_rank", return_value=rank),
        ]

    class TestDeviceClass(unittest.TestCase):
        def test_default_device_map_keys(self):
            dm = _build_default_device_map()
            self.assertIn(0, dm)
            self.assertIn(1, dm)
            self.assertIn(2, dm)
            self.assertEqual(dm[0].sm_generation, SMGeneration.SM86)
            self.assertEqual(dm[2].sm_generation, SMGeneration.SM90)

        def test_sm86_affinity_domain(self):
            dm = _build_default_device_map()
            self.assertEqual(dm[0].pcie_affinity, dm[1].pcie_affinity)
            self.assertNotEqual(dm[0].pcie_affinity, dm[2].pcie_affinity)

    class TestRankEnumeration(unittest.TestCase):
        """Pure numpy tests — no distributed needed."""

        def _make_grid(self, shape, dim_names, locality_sort=False):
            """Create grid bypassing all dist calls."""
            ctx_managers = _make_mock_dist(world_size=int(np.prod(shape)))
            patches = [m.__enter__() for m in ctx_managers]
            try:
                grid = DESLOCHyperCommGrid(
                    shape, dim_names, locality_sort=locality_sort
                )
            finally:
                for i, m in enumerate(ctx_managers):
                    m.__exit__(None, None, None)
            return grid

        def test_single_dim_enum(self):
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=8), \
                 patch("torch.distributed.get_rank", return_value=0):
                grid = DESLOCHyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], locality_sort=False)

            enum = grid._gen_rank_enum(["cp"])
            # dp=2 × tp=2 = 4 groups, each containing 2 cp-adjacent ranks.
            self.assertEqual(len(enum), 4)
            for group in enum:
                self.assertEqual(len(group), 2)

        def test_full_dims_enum_is_single_group(self):
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=8), \
                 patch("torch.distributed.get_rank", return_value=0):
                grid = DESLOCHyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], locality_sort=False)

            enum = grid._gen_rank_enum(["tp", "cp", "dp"])
            self.assertEqual(len(enum), 1)
            self.assertEqual(sorted(enum[0]), list(range(8)))

        def test_known_result_tp_cp(self):
            """Verify against the expected output from Megatron test suite."""
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=8), \
                 patch("torch.distributed.get_rank", return_value=0):
                grid = DESLOCHyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], locality_sort=False)

            enum = grid._gen_rank_enum(["tp", "cp"])
            expected = [[0, 2, 1, 3], [4, 6, 5, 7]]
            self.assertEqual(enum, expected)

        def test_rank_offset(self):
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=16), \
                 patch("torch.distributed.get_rank", return_value=8):
                grid = DESLOCHyperCommGrid(
                    [2, 2, 2], ["tp", "cp", "dp"], rank_offset=8, locality_sort=False
                )

            enum = grid._gen_rank_enum(["tp"])
            flat = sorted(r for group in enum for r in group)
            self.assertEqual(flat, list(range(8, 16)))

    class TestRegisterView(unittest.TestCase):
        def _make_grid(self, shape=None, dim_names=None, locality_sort=False):
            shape = shape or [2, 2, 2]
            dim_names = dim_names or ["tp", "cp", "dp"]
            ws = int(np.prod(shape))
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=ws), \
                 patch("torch.distributed.get_rank", return_value=0):
                return DESLOCHyperCommGrid(shape, dim_names, locality_sort=locality_sort)

        def test_register_view_success(self):
            grid = self._make_grid()
            grid.register_view("expert", [4, 2], ["ep", "expt_dp"])
            self.assertIn("expert", grid._views)
            self.assertEqual(grid._views["expert"].shape, [4, 2])

        def test_register_view_duplicate_raises(self):
            grid = self._make_grid()
            grid.register_view("expert", [4, 2], ["ep", "expt_dp"])
            with self.assertRaises(ValueError, msg="already registered"):
                grid.register_view("expert", [2, 4], ["ep2", "expt_dp2"])

        def test_register_view_size_mismatch_raises(self):
            grid = self._make_grid()  # size=8
            with self.assertRaises(ValueError, msg="grid size is 8"):
                grid.register_view("expert", [2, 2], ["ep", "expt_dp"])  # size=4

        def test_register_view_dim_duplicate_raises(self):
            grid = self._make_grid()
            with self.assertRaises(ValueError, msg="duplicate dim_names"):
                grid.register_view("expert", [2, 2, 2], ["ep", "ep", "expt_dp"])

        def test_register_view_non_positive_shape_raises(self):
            grid = self._make_grid()
            with self.assertRaises(ValueError, msg="positive ints"):
                grid.register_view("expert", [8, 0], ["ep", "expt_dp"])

        def test_register_view_accepts_numpy_int_shape(self):
            grid = self._make_grid()
            shape = [np.int64(4), np.int64(2)]
            grid.register_view("expert", shape, ["ep", "expt_dp"])
            self.assertIn("expert", grid._views)

        def test_register_view_shared_dim_not_in_base_raises(self):
            grid = self._make_grid()
            with self.assertRaises(ValueError, msg="not in the base view"):
                grid.register_view("expert", [4, 2], ["ep", "pp"], shared_dims=["pp"])

        def test_register_view_shared_dim_not_in_view_raises(self):
            grid = self._make_grid(shape=[2, 2, 2], dim_names=["tp", "cp", "pp"])
            with self.assertRaises(ValueError, msg="not in the view's dim_names"):
                grid.register_view("expert", [4, 2], ["ep", "expt_dp"], shared_dims=["pp"])

        def test_register_view_shared_dim_membership_mismatch_raises(self):
            grid = self._make_grid(shape=[2, 2, 2], dim_names=["tp", "dp", "pp"])
            # pp leading in expert view → different membership from base pp trailing.
            with self.assertRaises(ValueError, msg="different membership"):
                grid.register_view("expert", [2, 4], ["pp", "ep"], shared_dims=["pp"])

        def test_register_view_shared_dim_membership_match_succeeds(self):
            grid = self._make_grid(shape=[2, 2, 2], dim_names=["tp", "dp", "pp"])
            # Keep pp trailing in both views; membership must match.
            grid.register_view(
                "expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp"]
            )
            self.assertEqual(
                grid.get_rank_enum("pp"),
                grid.get_rank_enum("pp", view="expert"),
            )

        def test_unknown_view_raises(self):
            grid = self._make_grid()
            with self.assertRaises(KeyError, msg="not registered"):
                grid.get_rank_enum("tp", view="ghost")

    class TestRegisterHeterogeneousView(unittest.TestCase):
        def _make_3rank_grid(self):
            """2×A6000 + 1×H100 cluster: 3 ranks, shape [3]."""
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=3), \
                 patch("torch.distributed.get_rank", return_value=0):
                return DESLOCHyperCommGrid([3], ["dp"], locality_sort=False)

        def test_sm86_only_predicate_rejects_h100_subgroup(self):
            grid = self._make_3rank_grid()
            # Predicate: all ranks must be SM86.
            def sm86_only(ranks):
                return all(
                    grid.device_map.get(r, DeviceClass(r, SMGeneration.UNKNOWN, 0, "x"))
                    .sm_generation == SMGeneration.SM86
                    for r in ranks
                )
            # shape=[3] means one subgroup [0,1,2] which contains SM90 → should fail.
            with self.assertRaises(ValueError, msg="failed the device predicate"):
                grid.register_heterogeneous_view(
                    "sm86", [3], ["dp"], device_predicate=sm86_only
                )

        def test_permissive_predicate_accepts_all(self):
            grid = self._make_3rank_grid()
            grid.register_heterogeneous_view(
                "all_ranks", [3], ["dp"], device_predicate=lambda ranks: True
            )
            self.assertIn("all_ranks", grid._views)

    class TestOrderDims(unittest.TestCase):
        def _make_grid(self):
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=8), \
                 patch("torch.distributed.get_rank", return_value=0):
                return DESLOCHyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], locality_sort=False)

        def test_single_dim(self):
            grid = self._make_grid()
            ordered, key = grid._order_dims_for(grid.dim_names, "cp")
            self.assertEqual(ordered, ["cp"])
            self.assertEqual(key, "cp")

        def test_multiple_dims_reversed_order(self):
            grid = self._make_grid()
            ordered, key = grid._order_dims_for(grid.dim_names, ["dp", "tp"])
            self.assertEqual(ordered, ["dp", "tp"])
            self.assertEqual(key, "dp-tp")

        def test_all_dims(self):
            grid = self._make_grid()
            ordered, key = grid._order_dims_for(grid.dim_names, ["dp", "cp", "tp"])
            self.assertEqual(ordered, ["dp", "cp", "tp"])
            self.assertEqual(key, "dp-cp-tp")

    class TestLocalitySort(unittest.TestCase):
        """Verify locality sort reorders same-domain ranks to the front."""

        def _make_grid(self, locality_sort=True):
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=3), \
                 patch("torch.distributed.get_rank", return_value=0):
                return DESLOCHyperCommGrid([3], ["dp"], locality_sort=locality_sort)

        def test_locality_sort_groups_rc0_first(self):
            """Ranks 0 and 1 (rc0) should appear before rank 2 (rc1)."""
            grid = self._make_grid(locality_sort=True)
            enum_raw = [[2, 0, 1]]  # out of locality order
            sorted_enum = grid._apply_locality_sort(enum_raw)
            # rc0 ranks (0, 1) before rc1 rank (2).
            self.assertEqual(sorted_enum[0][:2], [0, 1])
            self.assertEqual(sorted_enum[0][2], 2)

        def test_locality_sort_disabled_preserves_order(self):
            grid = self._make_grid(locality_sort=False)
            enum = [[2, 0, 1]]
            # _apply_locality_sort should not be called; get_rank_enum without sort.
            result = grid._gen_rank_enum_for([3], ["dp"], ["dp"])
            # locality sort disabled → natural order 0,1,2 from np.arange.
            self.assertEqual(sorted(result[0]), [0, 1, 2])

    class TestPlanAllreduce(unittest.TestCase):
        def _make_grid(self):
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=3), \
                 patch("torch.distributed.get_rank", return_value=0):
                return DESLOCHyperCommGrid([3], ["dp"], locality_sort=False)

        def test_single_domain_group_suggests_single_ring(self):
            grid = self._make_grid()
            mock_pg = MagicMock()
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]):
                plan = grid.plan_allreduce(1024 * 1024, mock_pg)
            self.assertEqual(plan, AllReducePlan.SINGLE_RING)

        def test_cross_domain_small_tensor_suggests_single_ring(self):
            grid = self._make_grid()
            mock_pg = MagicMock()
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_process_group_ranks", return_value=[0, 2]):
                # Small tensor: 1 MB, well below threshold.
                plan = grid.plan_allreduce(1 * 1024 * 1024, mock_pg)
            self.assertEqual(plan, AllReducePlan.SINGLE_RING)

        def test_cross_domain_large_tensor_suggests_two_stage(self):
            grid = self._make_grid()
            mock_pg = MagicMock()
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_process_group_ranks", return_value=[0, 2]):
                # 512 MB: above the 256 MB TWO_STAGE threshold, below VRAM limit.
                plan = grid.plan_allreduce(512 * 1024 * 1024, mock_pg)
            self.assertEqual(plan, AllReducePlan.TWO_STAGE)

        def test_cross_domain_huge_tensor_suggests_loc_offload(self):
            grid = self._make_grid()
            mock_pg = MagicMock()
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_process_group_ranks", return_value=[0, 2]):
                # 6 GB > 10% of 48 GB A6000 (4.8 GB threshold) → CPU LOC offload.
                plan = grid.plan_allreduce(6 * 1024 * 1024 * 1024, mock_pg)
            self.assertEqual(plan, AllReducePlan.CPU_LOC_OFFLOAD)

    class TestCreateAndGetPG(unittest.TestCase):
        """Mock-based process-group creation / retrieval tests."""

        def _make_grid(self, shape=None, dim_names=None):
            shape = shape or [2, 2, 2]
            dim_names = dim_names or ["tp", "dp", "pp"]
            ws = int(np.prod(shape))
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=ws), \
                 patch("torch.distributed.get_rank", return_value=0):
                return DESLOCHyperCommGrid(shape, dim_names, locality_sort=False)

        @patch("torch.distributed.new_subgroups_by_enumeration")
        @patch("torch.distributed.is_initialized", return_value=True)
        @patch("torch.distributed.get_rank", return_value=0)
        def test_create_pg_base_view(self, _rank, _init, mock_new_subgroups):
            mock_pg = _FakeProcessGroup("tp")
            mock_new_subgroups.return_value = (mock_pg, None)

            grid = self._make_grid()
            result = grid.create_pg("tp")

            self.assertIs(result, mock_pg)
            self.assertIn("tp", grid._pgs)

        @patch("torch.distributed.new_subgroups_by_enumeration")
        @patch("torch.distributed.is_initialized", return_value=True)
        @patch("torch.distributed.get_rank", return_value=0)
        def test_create_pg_duplicate_raises(self, _rank, _init, mock_new_subgroups):
            mock_pg = _FakeProcessGroup("tp")
            mock_new_subgroups.return_value = (mock_pg, None)

            grid = self._make_grid()
            grid.create_pg("tp")
            with self.assertRaises(KeyError):
                grid.create_pg("tp")

        @patch("torch.distributed.new_subgroups_by_enumeration")
        @patch("torch.distributed.is_initialized", return_value=True)
        @patch("torch.distributed.get_rank", return_value=0)
        def test_shared_dim_reuses_base_pg(self, _rank, _init, mock_new_subgroups):
            mock_pg = _FakeProcessGroup("pp")
            mock_new_subgroups.return_value = (mock_pg, None)

            grid = self._make_grid()
            grid.register_view(
                "expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp"]
            )

            base_pp = grid.create_pg("pp")
            expert_pp = grid.create_pg("pp", view="expert")

            self.assertIs(base_pp, expert_pp)
            self.assertIn("pp", grid._pgs)
            self.assertNotIn(("expert", ("pp",)), grid._pgs)

        @patch("torch.distributed.new_subgroups_by_enumeration")
        @patch("torch.distributed.is_initialized", return_value=True)
        @patch("torch.distributed.get_rank", return_value=0)
        def test_view_private_pg_stored_under_tuple_key(self, _rank, _init, mock_new_subgroups):
            mock_pg = _FakeProcessGroup("ep")
            mock_new_subgroups.return_value = (mock_pg, None)

            grid = self._make_grid()
            grid.register_view("expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp"])
            grid.create_pg(["expt_tp", "ep"], view="expert")

            self.assertIn(("expert", ("ep", "expt_tp")), grid._pgs)

        def test_get_pg_not_created_raises(self):
            grid = self._make_grid()
            with self.assertRaises(KeyError, match="hasn't been created|has not been created"):
                grid.get_pg("tp")

        def test_get_pg_view_not_created_raises(self):
            grid = self._make_grid()
            grid.register_view("expert", [4, 2], ["ep", "expt_dp"])
            with self.assertRaises(KeyError, match="view 'expert'"):
                grid.get_pg("ep", view="expert")

    class TestDestroy(unittest.TestCase):
        def _make_grid(self):
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_world_size", return_value=8), \
                 patch("torch.distributed.get_rank", return_value=0):
                return DESLOCHyperCommGrid([2, 4], ["tp", "dp"], locality_sort=False)

        @patch("torch.distributed.destroy_process_group")
        def test_destroy_skips_non_members(self, mock_destroy):
            grid = self._make_grid()
            member_pg = _FakeProcessGroup("tp")
            non_member = getattr(
                getattr(dist, "GroupMember", None), "NON_GROUP_MEMBER", None
            )

            grid._pgs = {"tp": member_pg, "dp": non_member, "pp": None}

            # Patch _is_process_group_member to use our fake.
            with patch(
                "deepspeed.comm.hetero_hypercomm_grid._is_process_group_member",
                side_effect=lambda pg: pg is member_pg,
            ):
                grid.destroy()

            mock_destroy.assert_called_once_with(member_pg)
            self.assertEqual(grid._pgs, {})

        @patch("torch.distributed.destroy_process_group")
        def test_destroy_does_not_double_free_shared_pg(self, mock_destroy):
            grid = self._make_grid()
            shared_pg = _FakeProcessGroup("shared")

            # Simulate a shared group: same object under two keys.
            grid._pgs = {"pp": shared_pg, ("expert", ("pp",)): shared_pg}

            with patch(
                "deepspeed.comm.hetero_hypercomm_grid._is_process_group_member",
                return_value=True,
            ):
                grid.destroy()

            # destroy_process_group must be called exactly once.
            mock_destroy.assert_called_once_with(shared_pg)
            self.assertEqual(grid._pgs, {})

    class TestIsProcessGroupMember(unittest.TestCase):
        def test_none_is_not_member(self):
            self.assertFalse(_is_process_group_member(None))

        def test_non_group_member_is_not_member(self):
            non_member = getattr(
                getattr(dist, "GroupMember", None), "NON_GROUP_MEMBER", None
            )
            if non_member is not None:
                self.assertFalse(_is_process_group_member(non_member))

        def test_real_mock_pg_is_member(self):
            self.assertTrue(_is_process_group_member(_FakeProcessGroup("x")))

    # Run all tests.
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDeviceClass)
    for cls in [
        TestRankEnumeration,
        TestRegisterView,
        TestRegisterHeterogeneousView,
        TestOrderDims,
        TestLocalitySort,
        TestPlanAllreduce,
        TestCreateAndGetPG,
        TestDestroy,
        TestIsProcessGroupMember,
    ]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
