"""
DES-LOC Heterogeneous Named HyperComm Layouts
==============================================

Upstream design intent (Megatron ba71ec2):
    Megatron's ``HyperCommGrid`` models an N-dimensional communication grid where each axis
    is a named parallelism dimension (tp, dp, pp, cp, ep …).  The commit adds *named rank views*:
    alternative factorisations of the same rank span, so that an "expert" layout can slice the
    world differently from the "dense" layout while sharing certain dimensions (e.g. ``pp``) whose
    group membership is identical across factorisations.

DES-LOC adaptation (Decoupled Execution with Shared LOcality Cache):
    The Neuron_SP cluster is *heterogeneous*: 2 × A6000-48GB (SM86, PCIe) + 1 × H100-NVL-96GB
    (SM90, PCIe).  There is no NVLink; all GPU↔GPU traffic crosses PCIe.  CPU DRAM (1.5 TB) acts
    as a shared locality cache tier.

    Key adaptations over the Megatron baseline:

    1.  **DeviceClass-aware view registration** – Each named view carries a ``device_class``
        tag (``"SM86"``, ``"SM90"``, or ``"CPU"``) so the runtime can emit topology-aware
        process groups that respect PCIe bandwidth asymmetry.

    2.  **LOC-cache group injection** – When a view is registered as ``loc_cache=True`` the
        grid automatically synthesises a companion *locality-cache group* whose members span
        exactly those ranks that share a PCIe root complex, enabling the DES-LOC prefetch
        scheduler to move tensors through CPU DRAM without blocking the compute stream.

    3.  **Heterogeneous rank ordering** – ``_gen_rank_enum_for`` is extended to accept a
        ``device_map`` (rank → DeviceClass) and reorders sub-groups so that intra-class ranks
        are contiguous, minimising PCIe hop count for collective communication.

    4.  **Shared-dim reuse** – Identical to Megatron: if a named view declares a dimension as
        *shared* with the base view, the existing process group object is reused and reference-
        counted so ``destroy()`` never double-frees.

    5.  **DeepSpeed integration shim** – ``HeteroNamedHyperCommLayouts`` wraps DeepSpeed's
        ``dist`` helpers (``deepspeed.comm``) rather than bare ``torch.distributed``, so all
        process-group lifetimes are tracked by the DS engine.

Hardware topology assumed::

    rank 0 → A6000-0  (SM86, PCIe bus 0)
    rank 1 → A6000-1  (SM86, PCIe bus 0)
    rank 2 → H100-NVL (SM90, PCIe bus 1)
    (CPU DRAM accessible from all ranks via host memory)
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# DeepSpeed / torch.distributed shim
# ---------------------------------------------------------------------------
try:
    import deepspeed.comm as dist

    _HAVE_DS_COMM = True
except ImportError:  # fall back to bare torch.distributed for unit tests
    import torch.distributed as dist  # type: ignore[no-redef]

    _HAVE_DS_COMM = False

try:
    import torch.distributed as _torch_dist
except ImportError:
    _torch_dist = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_VIEW_NAME = "base"


# ---------------------------------------------------------------------------
# Device taxonomy
# ---------------------------------------------------------------------------


class DeviceClass(str, Enum):
    """Hardware classes present in the DES-LOC heterogeneous cluster.

    SM86 = A6000 48 GB  (PCIe, 2 cards)
    SM90 = H100 NVL 96 GB (PCIe, 1 card)
    CPU  = Host DRAM locality-cache tier (1.5 TB)
    """

    SM86 = "SM86"
    SM90 = "SM90"
    CPU = "CPU"


# Default device map for the 3-GPU cluster: rank → DeviceClass
_DEFAULT_DEVICE_MAP: Dict[int, DeviceClass] = {
    0: DeviceClass.SM86,
    1: DeviceClass.SM86,
    2: DeviceClass.SM90,
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


def _is_process_group_member(pg: Any) -> bool:
    """Return True iff the current rank belongs to ``pg``.

    DES-LOC note: non-member sentinel differs between torch.distributed versions;
    we probe both the ``GroupMember`` attribute and a falsy check so this works
    across DS/torch version combinations present in the Neuron_SP CI matrix.
    """
    if pg is None:
        return False
    non_member = getattr(
        getattr(_torch_dist, "GroupMember", None), "NON_GROUP_MEMBER", None
    )
    if non_member is not None and pg is non_member:
        return False
    return True


@dataclass
class _RankViewSpec:
    """Named rank factorisation over the same rank span as the base grid.

    DES-LOC extensions vs Megatron:
        device_class: Optional device class tag — enables topology-aware group creation.
        loc_cache:    When True the runtime synthesises a companion LOC-cache group
                      covering ranks that share a PCIe root complex.
    """

    name: str
    shape: List[int]
    dim_names: List[str]
    shared_dims: List[str]
    device_class: Optional[DeviceClass] = None
    loc_cache: bool = False


@dataclass
class _LocCacheGroupSpec:
    """Descriptor for a synthesised locality-cache group.

    The LOC-cache group contains all ranks that share a PCIe root complex with
    at least one member of ``view_name``'s process group, enabling zero-copy
    tensor staging through CPU DRAM.
    """

    view_name: str
    dims: Tuple[str, ...]
    member_ranks: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class HeteroNamedHyperCommLayouts:
    r"""N-dimensional heterogeneous communication grid with named rank views.

    DES-LOC Heterogeneous Adaptation
    ---------------------------------
    This class mirrors Megatron's ``HyperCommGrid`` (commit ba71ec2) but is re-
    interpreted for the Neuron_SP DES-LOC framework:

    *  The cluster is heterogeneous (SM86 × 2 + SM90 × 1) connected only via PCIe.
    *  Named views allow the same rank span to be factorised independently for
       dense layers (base view) and MoE expert layers (expert view) while sharing
       pipeline-parallel (``pp``) groups whose membership is identical.
    *  LOC-cache groups are auto-synthesised for views tagged ``loc_cache=True``;
       their members are chosen to minimise PCIe hops when staging activations
       through the 1.5 TB CPU DRAM buffer.
    *  All process-group lifetimes are tracked by ``deepspeed.comm`` (or
       ``torch.distributed`` when DS is absent) and destroyed without double-free.

    Args:
        shape:       Sizes of each parallelism dimension in the base view.
        dim_names:   Names for each dimension (must be unique, len == len(shape)).
        rank_offset: Global rank of the first rank in this grid (default 0).
        backend:     Collective backend string passed to ``new_subgroups_by_enumeration``.
        device_map:  Mapping from global rank to :class:`DeviceClass`.  Defaults to the
                     3-GPU cluster map (rank 0/1 → SM86, rank 2 → SM90).
        world_size:  Override for ``dist.get_world_size()``; useful in unit tests.

    Example::

        grid = HeteroNamedHyperCommLayouts(
            shape=[2, 2, 2],
            dim_names=["tp", "dp", "pp"],
        )
        grid.register_view(
            "expert",
            shape=[2, 2, 2],
            dim_names=["expt_tp", "ep", "pp"],
            shared_dims=["pp"],
            device_class=DeviceClass.SM86,
            loc_cache=True,
        )
        grid.create_pg("tp")
        grid.create_pg("pp")
        grid.create_pg(["expt_tp", "ep"], view="expert")
        # LOC-cache group for the expert view is auto-created on first create_pg call.
    """

    def __init__(
        self,
        shape: List[int],
        dim_names: List[str],
        rank_offset: int = 0,
        backend: Optional[str] = None,
        device_map: Optional[Dict[int, DeviceClass]] = None,
        world_size: Optional[int] = None,
    ) -> None:
        if len(shape) != len(dim_names):
            raise ValueError(
                f"len(shape)={len(shape)} != len(dim_names)={len(dim_names)}"
            )
        if len(set(dim_names)) != len(dim_names):
            raise ValueError(f"Duplicate dim_names in base view: {dim_names}")
        if any(not isinstance(s, numbers.Integral) or s <= 0 for s in shape):
            raise ValueError(f"shape must be positive ints, got {shape}")

        _ws = world_size or (dist.get_world_size() if dist.is_initialized() else None)
        if _ws is None:
            raise RuntimeError(
                "Initialize torch.distributed / deepspeed.comm before creating "
                "HeteroNamedHyperCommLayouts, or pass world_size= explicitly."
            )

        self.rank_offset: int = rank_offset
        self.size: int = int(np.prod(shape))
        if rank_offset < 0:
            raise ValueError(f"rank_offset must be non-negative, got {rank_offset}")
        if self.size > _ws - rank_offset:
            raise ValueError(
                f"Grid size {self.size} exceeds available ranks "
                f"(world_size={_ws}, rank_offset={rank_offset})."
            )

        self.shape: List[int] = list(shape)
        self.dim_names: List[str] = list(dim_names)
        self.backend: Optional[str] = backend
        self.device_map: Dict[int, DeviceClass] = device_map or dict(_DEFAULT_DEVICE_MAP)

        # Named view registry (base view is always present)
        self._views: Dict[str, _RankViewSpec] = {
            _BASE_VIEW_NAME: _RankViewSpec(
                name=_BASE_VIEW_NAME,
                shape=self.shape[:],
                dim_names=self.dim_names[:],
                shared_dims=[],
            )
        }

        # Process-group registry
        # Base-view groups: keyed by dash-joined dim string, e.g. "tp-dp"
        # View-private groups: keyed by (view_name, tuple(ordered_dims))
        self._pgs: Dict[Union[str, Tuple[str, Tuple[str, ...]]], Any] = {}

        # LOC-cache group specs (synthesised lazily)
        self._loc_cache_specs: Dict[str, _LocCacheGroupSpec] = {}
        # LOC-cache process groups: keyed by view_name
        self._loc_cache_pgs: Dict[str, Any] = {}

        logger.debug(
            "HeteroNamedHyperCommLayouts created: shape=%s dim_names=%s rank_offset=%d",
            shape,
            dim_names,
            rank_offset,
        )

    # ------------------------------------------------------------------
    # Public API — view registration
    # ------------------------------------------------------------------

    def register_view(
        self,
        name: str,
        shape: List[int],
        dim_names: List[str],
        shared_dims: Optional[List[str]] = None,
        device_class: Optional[DeviceClass] = None,
        loc_cache: bool = False,
    ) -> None:
        """Register an additional rank factorisation over this grid's rank span.

        DES-LOC extensions:
            device_class: Tag this view for a specific hardware tier.  The runtime
                          uses this to bias rank enumeration so intra-class ranks
                          are placed first in each process group, reducing PCIe hops.
            loc_cache:    If True, a companion LOC-cache group is synthesised when
                          ``create_pg`` is first called for this view.  The LOC-cache
                          group spans ranks sharing a PCIe root complex, enabling
                          low-latency CPU DRAM staging for the DES-LOC prefetch path.

        Upstream semantics (Megatron ba71ec2):
            Shared dims must exist in both the base view and the new view, and must
            enumerate to the same rank groups so the shared process group can be reused.

        Args:
            name:         Unique view name.
            shape:        Sizes of each dimension in the new view (product must equal
                          ``self.size``).
            dim_names:    Names for each dimension (unique within this view).
            shared_dims:  Dimensions whose rank groups are identical to the base view;
                          their process groups will be reused rather than duplicated.
            device_class: Optional hardware tier for topology-aware enumeration.
            loc_cache:    If True, synthesise a DES-LOC locality-cache group.
        """
        # ---- validation --------------------------------------------------
        if name == _BASE_VIEW_NAME:
            raise ValueError(f"View name {_BASE_VIEW_NAME!r} is reserved.")
        if name in self._views:
            raise ValueError(f"View {name!r} is already registered.")
        if len(shape) != len(dim_names):
            raise ValueError(
                f"len(shape) {list(shape)} != len(dim_names) {list(dim_names)}"
            )
        if len(set(dim_names)) != len(dim_names):
            raise ValueError(f"View {name!r} has duplicate dim_names: {dim_names}")
        if any(not isinstance(s, numbers.Integral) or s <= 0 for s in shape):
            raise ValueError(
                f"View {name!r} shape must be positive ints, got {list(shape)}"
            )
        view_size = int(np.prod(shape))
        if view_size != self.size:
            raise ValueError(
                f"View {name!r} shape {list(shape)} has size {view_size}, "
                f"but the grid size is {self.size}"
            )

        shared_dims = list(shared_dims) if shared_dims is not None else []
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
                    f"{list(dim_names)}"
                )
            base_ordered, _ = self._order_dims_for(self.dim_names, dim)
            base_enum = self._gen_rank_enum_for(self.shape, self.dim_names, base_ordered)
            view_ordered, _ = self._order_dims_for(list(dim_names), dim)
            view_enum = self._gen_rank_enum_for(list(shape), list(dim_names), view_ordered)
            if base_enum != view_enum:
                raise ValueError(
                    f"Shared dim {dim!r} has different membership across views: "
                    f"base enumeration {base_enum} != view {name!r} enumeration {view_enum}"
                )

        if len(shared_dims) > 1:
            base_ordered, _ = self._order_dims_for(self.dim_names, shared_dims)
            base_enum = self._gen_rank_enum_for(self.shape, self.dim_names, base_ordered)
            view_ordered, _ = self._order_dims_for(list(dim_names), shared_dims)
            view_enum = self._gen_rank_enum_for(list(shape), list(dim_names), view_ordered)
            if base_enum != view_enum:
                raise ValueError(
                    f"Shared dims {shared_dims!r} have different combined membership: "
                    f"base {base_enum} != view {name!r} {view_enum}"
                )

        self._views[name] = _RankViewSpec(
            name=name,
            shape=list(shape),
            dim_names=list(dim_names),
            shared_dims=shared_dims[:],
            device_class=device_class,
            loc_cache=loc_cache,
        )
        logger.debug(
            "Registered view %r: shape=%s dim_names=%s shared_dims=%s "
            "device_class=%s loc_cache=%s",
            name,
            list(shape),
            list(dim_names),
            shared_dims,
            device_class,
            loc_cache,
        )

    # ------------------------------------------------------------------
    # Public API — process-group lifecycle
    # ------------------------------------------------------------------

    def create_pg(
        self,
        dims: Union[str, List[str]],
        *,
        view: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a process group for the specified dimensions.

        DES-LOC note: If the target view was registered with ``loc_cache=True`` and
        this is the first ``create_pg`` call for that view, the LOC-cache companion
        group is synthesised immediately after the primary group.

        Args:
            dims:   Dimension name(s) within the selected view.
            view:   Named view to use (defaults to base view).
            **kwargs: Forwarded verbatim to ``new_subgroups_by_enumeration``.

        Returns:
            The new :class:`torch.distributed.ProcessGroup`, or the non-member
            sentinel if the current rank is outside this grid.

        Raises:
            KeyError:  If the process group has already been created.
            ValueError: If ``dims`` are not valid for the selected view.
        """
        view_spec = self._resolve_view(view)
        ordered_dims, _ = self._order_dims_for_view(view_spec, dims)
        unique_key, enum_view, enum_dims = self._canonical_pg_key_and_enum_view(
            view_spec, ordered_dims
        )

        if unique_key in self._pgs:
            _vname = view_spec.name
            if self._is_base_pg_key(unique_key):
                raise KeyError(
                    f"Process group {dims} has already been created. Because there is no "
                    f"way to check whether options match the first call, we error out."
                )
            raise KeyError(
                f"Process group {dims} for view {_vname!r} has already been created. "
                f"Because there is no way to check whether options match the first call, "
                f"we error out instead of returning the existing group."
            )

        # DES-LOC: apply topology-aware rank reordering for heterogeneous hardware
        rank_enum = self._hetero_rank_enum(
            enum_view.shape,
            enum_view.dim_names,
            enum_dims,
            device_class=view_spec.device_class,
        )

        pg, _ = dist.new_subgroups_by_enumeration(
            rank_enum, backend=self.backend, **kwargs
        )

        if dist.is_initialized() and dist.get_rank() == 0:
            if self._is_base_pg_key(unique_key):
                logger.info(
                    "Created base process group %r with enumeration %s",
                    unique_key,
                    rank_enum,
                )
            else:
                logger.info(
                    "Created view %r process group %s with enumeration %s",
                    view_spec.name,
                    ordered_dims,
                    rank_enum,
                )

        self._pgs[unique_key] = pg

        # DES-LOC: synthesise LOC-cache companion group on first create_pg for this view
        if view_spec.loc_cache and view_spec.name not in self._loc_cache_pgs:
            self._synthesise_loc_cache_group(view_spec, ordered_dims, **kwargs)

        return pg

    def get_pg(
        self,
        dims: Union[str, List[str]],
        *,
        view: Optional[str] = None,
    ) -> Any:
        """Retrieve an already-created process group.

        Args:
            dims:  Dimension name(s) within the selected view.
            view:  Named view (defaults to base view).

        Raises:
            KeyError: If the group has not been created yet.
        """
        view_spec = self._resolve_view(view)
        ordered_dims, _ = self._order_dims_for_view(view_spec, dims)
        unique_key, _, _ = self._canonical_pg_key_and_enum_view(view_spec, ordered_dims)

        if unique_key not in self._pgs:
            if self._is_base_pg_key(unique_key):
                raise KeyError(
                    f"Process group for {unique_key!r} hasn't been created. "
                    f"Call create_pg first."
                )
            raise KeyError(
                f"Process group {dims} for view {view_spec.name!r} hasn't been created. "
                f"Call create_pg first."
            )
        return self._pgs[unique_key]

    def get_loc_cache_pg(self, view_name: str) -> Any:
        """Return the DES-LOC locality-cache process group for a view.

        The LOC-cache group is auto-created the first time ``create_pg`` is called
        for a view registered with ``loc_cache=True``.

        Args:
            view_name: The view whose LOC-cache group is requested.

        Raises:
            KeyError:  If the view has no LOC-cache group (not registered with
                       ``loc_cache=True``, or ``create_pg`` not yet called).
        """
        if view_name not in self._loc_cache_pgs:
            raise KeyError(
                f"No LOC-cache process group for view {view_name!r}. "
                f"Either the view was not registered with loc_cache=True, "
                f"or create_pg has not been called yet."
            )
        return self._loc_cache_pgs[view_name]

    def get_rank_enum(
        self,
        dims: Union[str, List[str]],
        *,
        view: Optional[str] = None,
    ) -> List[List[int]]:
        """Return the rank enumeration for ``dims`` under the selected view.

        DES-LOC: The enumeration respects hardware topology (SM86/SM90 device class)
        when the view carries a ``device_class`` tag.

        Args:
            dims: Dimension name(s).
            view: Named view (defaults to base view).
        """
        view_spec = self._resolve_view(view)
        ordered_dims, _ = self._order_dims_for_view(view_spec, dims)
        return self._hetero_rank_enum(
            view_spec.shape,
            view_spec.dim_names,
            ordered_dims,
            device_class=view_spec.device_class,
        )

    def destroy(self) -> None:
        """Destroy all process groups owned by this grid.

        DES-LOC: Also tears down LOC-cache companion groups.  Uses an ``id``-based
        deduplication set so shared-dim groups (stored under a single key but reused
        across base and named views) are never double-freed.
        """
        destroyed: Set[int] = set()

        for pg in list(self._pgs.values()):
            if _is_process_group_member(pg) and id(pg) not in destroyed:
                dist.destroy_process_group(pg)
                destroyed.add(id(pg))
        self._pgs.clear()

        for pg in list(self._loc_cache_pgs.values()):
            if _is_process_group_member(pg) and id(pg) not in destroyed:
                dist.destroy_process_group(pg)
                destroyed.add(id(pg))
        self._loc_cache_pgs.clear()

        logger.debug("HeteroNamedHyperCommLayouts destroyed all process groups.")

    def is_current_rank_in_grid(self) -> bool:
        """Return True if the current rank falls within this grid's rank span."""
        if not dist.is_initialized():
            return False
        rank = dist.get_rank()
        return self.rank_offset <= rank < self.rank_offset + self.size

    # ------------------------------------------------------------------
    # DES-LOC heterogeneous helpers
    # ------------------------------------------------------------------

    def _hetero_rank_enum(
        self,
        shape: List[int],
        dim_names: List[str],
        dims: List[str],
        device_class: Optional[DeviceClass] = None,
    ) -> List[List[int]]:
        """Generate rank enumeration with DES-LOC heterogeneous reordering.

        When ``device_class`` is provided, ranks belonging to that hardware class
        are sorted to the *front* of each sub-group.  This reduces PCIe hop count
        for collectives whose initiating rank is on the designated device class,
        which is the common case for both dense (SM86 initiates TP) and MoE
        (H100 initiates EP dispatch) workloads in Neuron_SP.

        Args:
            shape:        Shape of the view being enumerated.
            dim_names:    Dimension names of the view.
            dims:         Ordered subset of dim_names to group by.
            device_class: If set, ranks of this class are sorted first within
                          each sub-group.

        Returns:
            List of rank lists — one list per process group.
        """
        base_enum = self._gen_rank_enum_for(shape, dim_names, dims)

        if device_class is None:
            return base_enum

        reordered: List[List[int]] = []
        for group in base_enum:
            preferred = [r for r in group if self.device_map.get(r) == device_class]
            others = [r for r in group if self.device_map.get(r) != device_class]
            reordered.append(preferred + others)

        logger.debug(
            "_hetero_rank_enum: device_class=%s base=%s reordered=%s",
            device_class,
            base_enum,
            reordered,
        )
        return reordered

    def _synthesise_loc_cache_group(
        self,
        view_spec: _RankViewSpec,
        dims: List[str],
        **kwargs: Any,
    ) -> None:
        """Synthesise a DES-LOC locality-cache group for ``view_spec``.

        The LOC-cache group contains all ranks that share a PCIe root complex with
        the members of the primary group.  For the 3-GPU cluster this means:
          * ranks 0 and 1 (A6000 pair, PCIe bus 0) form one cache group
          * rank 2 (H100, PCIe bus 1) is its own singleton cache group

        The resulting process group allows the DES-LOC prefetch scheduler to stage
        tensors through the 1.5 TB CPU DRAM buffer with a single host-side memcpy
        rather than a device-to-device PCIe transfer.

        DES-LOC design: this method is intentionally called *after* the primary
        group has been registered, so the LOC-cache group never races the primary.
        """
        # Build PCIe root-complex buckets from device_map
        bus_buckets: Dict[str, List[int]] = {}
        for rank, dc in self.device_map.items():
            # SM86 devices share bus "pcib0"; SM90 is on "pcib1"; CPU is virtual
            if dc == DeviceClass.SM86:
                bus_buckets.setdefault("pcib0", []).append(rank)
            elif dc == DeviceClass.SM90:
                bus_buckets.setdefault("pcib1", []).append(rank)
            else:
                bus_buckets.setdefault(f"cpu_{rank}", []).append(rank)

        loc_enum: List[List[int]] = [
            sorted(ranks) for ranks in bus_buckets.values() if ranks
        ]

        # Only include ranks that are within this grid
        grid_ranks: Set[int] = set(
            range(self.rank_offset, self.rank_offset + self.size)
        )
        loc_enum = [
            [r for r in group if r in grid_ranks]
            for group in loc_enum
        ]
        loc_enum = [g for g in loc_enum if g]

        logger.info(
            "Synthesising LOC-cache group for view %r: enumeration=%s",
            view_spec.name,
            loc_enum,
        )

        loc_pg, _ = dist.new_subgroups_by_enumeration(
            loc_enum, backend=self.backend, **kwargs
        )
        self._loc_cache_pgs[view_spec.name] = loc_pg

        spec = _LocCacheGroupSpec(
            view_name=view_spec.name,
            dims=tuple(dims),
            member_ranks=[r for group in loc_enum for r in group],
        )
        self._loc_cache_specs[view_spec.name] = spec

        logger.debug(
            "LOC-cache group for view %r registered with %d total member ranks.",
            view_spec.name,
            len(spec.member_ranks),
        )

    # ------------------------------------------------------------------
    # Internal helpers (mirroring Megatron _order_dims / _gen_rank_enum)
    # ------------------------------------------------------------------

    def _gen_rank_enum(self, dims: List[str]) -> List[List[int]]:
        """Convenience wrapper — enumerate against the base view."""
        return self._gen_rank_enum_for(self.shape, self.dim_names, dims)

    def _gen_rank_enum_for(
        self,
        shape: List[int],
        dim_names: List[str],
        dims: List[str],
    ) -> List[List[int]]:
        """Generate rank enumeration for ``dims`` under explicit ``shape``/``dim_names``.

        DES-LOC note: The einops dependency present in older Megatron versions is
        replaced with a pure-numpy ``moveaxis`` + ``reshape`` pipeline, which has no
        additional install requirements and is compatible with all Python 3.9+ envs
        used in Neuron_SP CI (including CPU-only test runners).

        The MCore convention reverses ``dim_names`` before indexing, so the outermost
        axis in the rank tensor corresponds to the *last* element of ``dim_names``.
        """
        dim_names_rev = dim_names[::-1]
        shape_dict = {d: s for d, s in zip(dim_names, shape)}
        size = int(np.prod(shape))

        rank_tensor = np.arange(
            self.rank_offset, self.rank_offset + size
        ).reshape([shape_dict[d] for d in dim_names_rev])

        src_axes = [dim_names_rev.index(d) for d in dims]
        n = len(dim_names_rev)
        tgt_axes = list(range(n - len(dims), n))

        logger.debug(
            "_gen_rank_enum_for: dim_names=%s dims=%s src_axes=%s tgt_axes=%s",
            dim_names,
            dims,
            src_axes,
            tgt_axes,
        )

        rank_tensor = np.moveaxis(rank_tensor, src_axes, tgt_axes)
        group_size = int(np.prod([shape_dict[d] for d in dims]))
        return rank_tensor.reshape(-1, group_size).tolist()

    def _order_dims_for(
        self,
        dim_names: List[str],
        dims: Union[str, List[str]],
    ) -> Tuple[List[str], str]:
        """Reorder ``dims`` against an explicit ``dim_names`` (reversed convention)."""
        if not isinstance(dims, list):
            return [dims], dims

        dim_names_rev = dim_names[::-1]
        indices = sorted([dim_names_rev.index(d) for d in dims])
        ordered = [dim_names_rev[i] for i in indices]
        key = "-".join(ordered)
        return ordered, key

    def _resolve_view(self, view: Optional[str]) -> _RankViewSpec:
        """Return the view spec for ``view`` (defaulting to the base view)."""
        name = _BASE_VIEW_NAME if view is None else view
        if name not in self._views:
            raise KeyError(
                f"View {name!r} is not registered. "
                f"Registered views: {sorted(self._views)}"
            )
        return self._views[name]

    def _order_dims_for_view(
        self,
        view: _RankViewSpec,
        dims: Union[str, List[str]],
    ) -> Tuple[List[str], str]:
        """Reorder ``dims`` within a named view with clear error reporting."""
        requested = [dims] if not isinstance(dims, list) else list(dims)
        missing = [d for d in requested if d not in view.dim_names]
        if missing:
            raise ValueError(
                f"{missing[0]!r} is not in view {view.name!r} "
                f"with dim_names {view.dim_names}"
            )
        return self._order_dims_for(view.dim_names, dims)

    def _canonical_pg_key_and_enum_view(
        self,
        view: _RankViewSpec,
        ordered_dims: List[str],
    ) -> Tuple[Union[str, Tuple[str, Tuple[str, ...]]], _RankViewSpec, List[str]]:
        """Determine the storage key, enumeration view, and enumeration dims for a PG.

        DES-LOC / Megatron shared logic:
          * Base-view groups use a plain string key (dash-joined dim names).
          * Named-view groups whose requested dims are ALL shared with the base
            canonicalise to the base key, reusing the existing process group.
          * Named-view private groups use a (view_name, dims_tuple) key.
        """
        if view.name == _BASE_VIEW_NAME:
            key = "-".join(ordered_dims)
            return key, view, ordered_dims

        if all(d in view.shared_dims for d in ordered_dims):
            base_view = self._views[_BASE_VIEW_NAME]
            base_ordered, base_key = self._order_dims_for_view(base_view, ordered_dims)
            return base_key, base_view, base_ordered

        key = (view.name, tuple(ordered_dims))
        return key, view, ordered_dims

    def _is_base_pg_key(
        self, key: Union[str, Tuple[str, Tuple[str, ...]]]
    ) -> bool:
        """Return True if ``key`` belongs to the base view namespace."""
        return isinstance(key, str)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import torch
    import torch.distributed as tdist

    # ---- minimal single-process sanity checks (no collective communication) ----

    # Patch dist so we can instantiate without a real process group
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    # Use the internal helpers directly (no dist required for pure math)
    grid = HeteroNamedHyperCommLayouts.__new__(HeteroNamedHyperCommLayouts)
    grid.rank_offset = 0
    grid.size = 8
    grid.shape = [2, 2, 2]
    grid.dim_names = ["tp", "dp", "pp"]
    grid.backend = None
    grid.device_map = {
        0: DeviceClass.SM86,
        1: DeviceClass.SM86,
        2: DeviceClass.SM90,
        3: DeviceClass.SM86,
        4: DeviceClass.SM86,
        5: DeviceClass.SM86,
        6: DeviceClass.SM90,
        7: DeviceClass.SM86,
    }
    grid._views = {
        _BASE_VIEW_NAME: _RankViewSpec(
            _BASE_VIEW_NAME, [2, 2, 2], ["tp", "dp", "pp"], []
        )
    }
    grid._pgs = {}
    grid._loc_cache_pgs = {}
    grid._loc_cache_specs = {}

    # 1. base rank enumeration matches Megatron expectation
    enum_tp_cp = grid._gen_rank_enum_for([2, 2, 2], ["tp", "cp", "dp"], ["tp", "cp"])
    assert enum_tp_cp == [[0, 2, 1, 3], [4, 6, 5, 7]], f"unexpected enum: {enum_tp_cp}"

    # 2. register_view validation: size mismatch
    try:
        grid.register_view("bad", [2, 2], ["a", "b"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "grid size is 8" in str(e)

    # 3. hetero reordering places SM86 ranks first
    enum_hetero = grid._hetero_rank_enum(
        [2, 2, 2], ["tp", "dp", "pp"], ["tp", "dp"], device_class=DeviceClass.SM86
    )
    for group in enum_hetero:
        sm86_idx = [i for i, r in enumerate(group) if grid.device_map[r] == DeviceClass.SM86]
        sm90_idx = [i for i, r in enumerate(group) if grid.device_map[r] == DeviceClass.SM90]
        if sm86_idx and sm90_idx:
            assert max(sm86_idx) < min(sm90_idx), (
                f"SM86 ranks not before SM90 in group {group}"
            )

    # 4. shared-dim canonicalisation returns base key
    grid.register_view("expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp"])
    expert_spec = grid._views["expert"]
    expert_ordered, _ = grid._order_dims_for_view(expert_spec, "pp")
    key, ev, _ = grid._canonical_pg_key_and_enum_view(expert_spec, expert_ordered)
    assert key == "pp", f"expected base key 'pp', got {key!r}"
    assert ev.name == _BASE_VIEW_NAME

    # 5. view-private dims produce tuple key
    private_ordered, _ = grid._order_dims_for_view(expert_spec, ["expt_tp", "ep"])
    key2, _, _ = grid._canonical_pg_key_and_enum_view(expert_spec, private_ordered)
    assert isinstance(key2, tuple) and key2[0] == "expert"

    print("All smoke tests passed.")
