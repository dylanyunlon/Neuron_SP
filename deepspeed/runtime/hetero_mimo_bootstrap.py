"""
DES-LOC Heterogeneous MIMO Bootstrap
=====================================

Upstream design intent (Megatron aa1057124):
    Megatron's MIMO hetero topology commit introduces two orthogonal concerns:
    1. ``distributed.py``: Bring up ``torch.distributed`` + a global memory buffer
       *without* initialising the full MPU (Model Parallel Unit).  This is needed
       because hetero MIMO routes different modality encoders to different rank
       subsets, so the monolithic MPU initialisation order breaks down.
    2. ``topology.py``: A ``ModuleGridSpec`` / ``HyperCommGrid`` / ``ProcessGroupCollection``
       stack that factorises the world into per-module grids, validates that grids
       either fully share ranks (colocated) or partition the world (non-colocated),
       and builds embedding process groups only for the language module.

DES-LOC adaptation points:
    DES-LOC = Decoupled Execution with Shared LOcality Cache.

    Hardware context:
        • Rank 0 – A6000 48 GB SM86  (PCIe, no NVLink)
        • Rank 1 – A6000 48 GB SM86  (PCIe, no NVLink)
        • Rank 2 – H100 NVL  96 GB SM90  (PCIe to the A6000s)
        • Host   – 1.5 TB CPU DRAM shared across all ranks via ``mmap`` pinned buffers

    Key divergences from Megatron's upstream:
    A. **No MPU** – DeepSpeed does not have Megatron's ``parallel_state``.  All
       process-group bookkeeping is done with raw ``torch.distributed`` primitives
       and a lightweight ``DESlOCParallelState`` dataclass.
    B. **Locality-aware grid placement** – ``ModuleGridSpec`` carries a
       ``device_tier`` (FAST/SLOW) so the bootstrap can route compute-heavy
       layers to SM90 and memory-heavy activations to A6000 + CPU DRAM.
    C. **Shared LOcality Cache (SLC)** – each rank registers a ``SLCBuffer`` backed
       by pinned CPU DRAM.  The SLC acts as a staging area for cross-rank KV-cache
       sharing without NVLink; inter-rank copies flow through CPU DRAM instead of
       peer P2P, which is bandwidth-efficient on PCIe-only topologies.
    D. **Expert view** – Megatron registers a second HyperCommGrid *view* for MoE;
       here we replicate that semantics by building a separate ``ExpertGridSpec``
       that shares the PP dimension with the base dense grid.
    E. **Embedding groups** – Only the language module gets word/position embedding
       process groups.  Modality encoders (images, audio …) get ``None`` sentinels,
       mirroring Megatron's ``is_language`` flag.
    F. **Destroy / cleanup** – ``HeteroTopology.destroy()`` iterates every owned PG
       and the SLC buffers; idempotent via an ``id``-based ``destroyed`` set.

Usage::

    from deepspeed.runtime.hetero_mimo_bootstrap import (
        initialize_distributed,
        create_topology,
        ModuleGridSpec,
        DeviceTier,
        shutdown_distributed,
    )

    initialize_distributed()          # bring up dist + SLC
    topo = create_topology([
        ModuleGridSpec("images",    num_ranks=2, tp=2,
                       rank_offset=0, device_tier=DeviceTier.SLOW),
        ModuleGridSpec("language",  num_ranks=1, tp=1,
                       rank_offset=2, device_tier=DeviceTier.FAST),
    ])
    # ... training loop ...
    topo.destroy()
    shutdown_distributed()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

__all__ = [
    "DeviceTier",
    "ModuleGridSpec",
    "ExpertGridSpec",
    "SLCBuffer",
    "ProcessGroupCollection",
    "HeteroTopology",
    "DESlOCParallelState",
    "initialize_distributed",
    "shutdown_distributed",
    "create_topology",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGE_MODULE_KEY: str = "language"
"""Canonical name for the language module; must appear exactly once in every topology."""

_EXPERT_SUFFIX: str = "__expert"
"""Appended to a module name when building the expert sub-grid."""

_SLC_TAG_BASE: int = 8800
"""Base tag for SLC staging transfers; offset by rank to avoid collisions."""

_PINNED_ALLOC_WARN_BYTES: int = 4 * 1024 ** 3  # 4 GiB
"""Warn when a single SLC buffer exceeds this threshold."""


# ---------------------------------------------------------------------------
# Device tier – locality annotation
# ---------------------------------------------------------------------------

class DeviceTier(Enum):
    """Coarse locality tier for a module's rank set.

    FAST  → SM90 H100 NVL (higher compute throughput, larger VRAM).
    SLOW  → SM86 A6000 (good memory bandwidth, PCIe-only cross-rank).
    CPU   → Offloaded to pinned CPU DRAM via DES-LOC SLC.
    """
    FAST = auto()
    SLOW = auto()
    CPU  = auto()


# ---------------------------------------------------------------------------
# Grid specifications
# ---------------------------------------------------------------------------

@dataclass
class ExpertGridSpec:
    """Expert-parallel factorisation that *shares* the PP dimension with the dense grid.

    Mirrors Megatron's ``register_view(_EXPERT_VIEW, …, shared_dims=['pp'])`` but
    expressed as a plain dataclass so we don't depend on HyperCommGrid.

    Fields
    ------
    expt_tp   : expert tensor parallelism (default 1, set explicitly for MoE)
    ep        : expert parallelism (number of expert shards)
    expt_dp   : derived as num_ranks // (expt_tp * ep * pp); validated in __post_init__
    """
    num_ranks: int
    pp: int = 1
    expt_tp: int = 1
    ep: int = 1
    expt_dp: int = field(init=False)

    def __post_init__(self) -> None:
        denom = self.expt_tp * self.ep * self.pp
        if self.num_ranks % denom != 0:
            raise ValueError(
                f"ExpertGridSpec: num_ranks ({self.num_ranks}) must be divisible by "
                f"expt_tp*ep*pp = {self.expt_tp}*{self.ep}*{self.pp} = {denom}"
            )
        self.expt_dp = self.num_ranks // denom
        logger.debug(
            "ExpertGridSpec resolved: expt_tp=%d ep=%d pp=%d expt_dp=%d",
            self.expt_tp, self.ep, self.pp, self.expt_dp,
        )


@dataclass
class ModuleGridSpec:
    """One MIMO module's dense grid factorisation + DES-LOC placement metadata.

    Upstream (Megatron) fields
    --------------------------
    name        : module identifier; use ``LANGUAGE_MODULE_KEY`` for the LLM
    num_ranks   : total ranks assigned to this module
    tp          : tensor parallelism
    cp          : context parallelism
    pp          : pipeline parallelism
    ep          : expert parallelism (dense grid; see ExpertGridSpec for full MoE)
    rank_offset : first global rank index in this module's contiguous slice

    DES-LOC additions
    -----------------
    device_tier : DeviceTier annotation for locality-aware scheduling
    expert      : optional ExpertGridSpec; if None, a default (expt_tp=1, ep=1) is
                  synthesised in ``__post_init__`` so the expert PG path always exists
    slc_size_bytes : bytes to pre-allocate in pinned CPU DRAM for this module's SLC
                     staging buffer (0 = disabled)

    Derived fields (set in __post_init__)
    --------------------------------------
    dp          : data parallelism = num_ranks // (tp*cp*pp)
    """

    name: str
    num_ranks: int
    tp: int = 1
    cp: int = 1
    pp: int = 1
    rank_offset: int = 0
    device_tier: DeviceTier = DeviceTier.SLOW
    expert: Optional[ExpertGridSpec] = None
    slc_size_bytes: int = 0
    dp: int = field(init=False)

    def __post_init__(self) -> None:
        dense = self.tp * self.cp * self.pp
        if self.num_ranks % dense != 0:
            raise ValueError(
                f"ModuleGridSpec '{self.name}': num_ranks ({self.num_ranks}) must be "
                f"divisible by tp*cp*pp = {self.tp}*{self.cp}*{self.pp} = {dense}"
            )
        self.dp = self.num_ranks // dense

        # Synthesise default expert spec if not provided (ep=1, expt_tp=1)
        if self.expert is None:
            self.expert = ExpertGridSpec(num_ranks=self.num_ranks, pp=self.pp)

        # Warn on large SLC allocations
        if self.slc_size_bytes > _PINNED_ALLOC_WARN_BYTES:
            logger.warning(
                "ModuleGridSpec '%s': SLC buffer is %.1f GiB – ensure host has enough "
                "pinned memory (system has 1.5 TB, but other modules compete for it).",
                self.name, self.slc_size_bytes / 1024 ** 3,
            )
        logger.debug(
            "ModuleGridSpec '%s': tp=%d cp=%d pp=%d dp=%d tier=%s offset=%d",
            self.name, self.tp, self.cp, self.pp, self.dp,
            self.device_tier.name, self.rank_offset,
        )

    @property
    def rank_end(self) -> int:
        """Exclusive upper bound of this module's rank slice."""
        return self.rank_offset + self.num_ranks

    def contains_rank(self, rank: int) -> bool:
        return self.rank_offset <= rank < self.rank_end


# ---------------------------------------------------------------------------
# Shared LOcality Cache buffer
# ---------------------------------------------------------------------------

class SLCBuffer:
    """Pinned CPU DRAM staging buffer for cross-rank KV-cache sharing.

    DES-LOC rationale
    -----------------
    Without NVLink, P2P GPU copies between A6000s and the H100 must transit the
    CPU.  Rather than ad-hoc ``cudaMemcpy`` calls, each rank pre-allocates a
    pinned buffer here.  Activations / KV-cache slices are staged through this
    buffer before being scattered/gathered over the process group.

    The buffer is intentionally *not* a CUDA tensor – it lives in CPU DRAM and is
    accessed via ``torch.frombuffer`` at call time to avoid pinning VRAM.
    """

    def __init__(self, module_name: str, size_bytes: int) -> None:
        self.module_name = module_name
        self.size_bytes = size_bytes
        self._storage: Optional[torch.Tensor] = None
        if size_bytes > 0:
            self._storage = torch.empty(
                size_bytes, dtype=torch.uint8, pin_memory=True
            )
            logger.info(
                "SLCBuffer '%s': allocated %.2f MiB pinned CPU DRAM",
                module_name, size_bytes / 1024 ** 2,
            )
        else:
            logger.debug("SLCBuffer '%s': size_bytes=0, buffer disabled", module_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._storage is not None

    def as_tensor(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Return a typed view of the pinned buffer (zero-copy reinterpretation)."""
        if self._storage is None:
            raise RuntimeError(
                f"SLCBuffer '{self.module_name}' is disabled (size_bytes=0)"
            )
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = self.size_bytes // elem_size
        return self._storage[:numel * elem_size].view(dtype=dtype)

    def stage_send(
        self,
        tensor: torch.Tensor,
        dst: int,
        pg: dist.ProcessGroup,
    ) -> None:
        """Copy *tensor* to the pinned buffer then issue an isend.

        This is the DES-LOC cross-tier transfer primitive.  The copy from VRAM to
        pinned CPU DRAM is synchronous (``cuda.synchronize`` scoped to the tensor's
        stream); the ``dist.isend`` is non-blocking so the caller can overlap.
        """
        if self._storage is None:
            raise RuntimeError(f"SLCBuffer '{self.module_name}': cannot stage_send, buffer disabled")
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.size_bytes:
            raise ValueError(
                f"SLCBuffer '{self.module_name}': tensor ({nbytes} B) exceeds "
                f"buffer capacity ({self.size_bytes} B)"
            )
        cpu_view = self._storage[:nbytes].view(dtype=tensor.dtype).reshape(tensor.shape)
        # Synchronous D2H copy into pinned memory
        cpu_view.copy_(tensor, non_blocking=False)
        logger.debug(
            "SLCBuffer '%s': staged %d bytes to pinned, sending to rank %d",
            self.module_name, nbytes, dst,
        )
        dist.isend(cpu_view, dst=dst, group=pg, tag=_SLC_TAG_BASE + dist.get_rank())

    def stage_recv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        pg: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Blocking recv into pinned buffer; returns a *CPU* tensor (caller moves to GPU).

        Separating the recv from the H2D copy lets the caller pipeline the copy
        with subsequent compute – a key DES-LOC scheduling trick on PCIe topologies.
        """
        if self._storage is None:
            raise RuntimeError(f"SLCBuffer '{self.module_name}': cannot stage_recv, buffer disabled")
        nbytes = 1
        for s in shape:
            nbytes *= s
        nbytes *= torch.tensor([], dtype=dtype).element_size()
        cpu_view = self._storage[:nbytes].view(dtype=dtype).reshape(shape)
        dist.recv(cpu_view, src=src, group=pg, tag=_SLC_TAG_BASE + src)
        logger.debug(
            "SLCBuffer '%s': received %d bytes from rank %d into pinned",
            self.module_name, nbytes, src,
        )
        return cpu_view

    def destroy(self) -> None:
        """Release the pinned allocation (sets storage to None; GC handles dealloc)."""
        if self._storage is not None:
            logger.debug("SLCBuffer '%s': releasing pinned buffer", self.module_name)
            del self._storage
            self._storage = None


# ---------------------------------------------------------------------------
# Process group collection
# ---------------------------------------------------------------------------

@dataclass
class ProcessGroupCollection:
    """All process groups for one MIMO module.

    Fields mirror Megatron's ``ProcessGroupCollection`` but are plain attributes
    (no dependency on megatron.core).  Groups are ``None`` when not applicable
    (e.g. embedding groups for non-language modules).
    """
    tp:            Optional[dist.ProcessGroup] = None  # tensor parallel
    cp:            Optional[dist.ProcessGroup] = None  # context parallel
    pp:            Optional[dist.ProcessGroup] = None  # pipeline parallel
    dp:            Optional[dist.ProcessGroup] = None  # data parallel
    dp_cp:         Optional[dist.ProcessGroup] = None  # dp × cp joint group
    tp_cp:         Optional[dist.ProcessGroup] = None  # tp × cp joint group
    mp:            Optional[dist.ProcessGroup] = None  # model parallel (tp × pp)
    ep:            Optional[dist.ProcessGroup] = None  # expert parallel
    expt_tp:       Optional[dist.ProcessGroup] = None  # expert tensor parallel
    expt_dp:       Optional[dist.ProcessGroup] = None  # expert data parallel
    tp_ep:         Optional[dist.ProcessGroup] = None  # tp × ep
    tp_ep_pp:      Optional[dist.ProcessGroup] = None  # tp × ep × pp
    embd:          Optional[dist.ProcessGroup] = None  # word embedding
    pos_embd:      Optional[dist.ProcessGroup] = None  # position embedding

    def owned_groups(self) -> List[dist.ProcessGroup]:
        """Return non-None groups in a stable order."""
        return [
            g for g in (
                self.tp, self.cp, self.pp, self.dp,
                self.dp_cp, self.tp_cp, self.mp,
                self.ep, self.expt_tp, self.expt_dp,
                self.tp_ep, self.tp_ep_pp,
                self.embd, self.pos_embd,
            )
            if g is not None
        ]


# ---------------------------------------------------------------------------
# DES-LOC parallel state (replaces Megatron's parallel_state globals)
# ---------------------------------------------------------------------------

@dataclass
class DESlOCParallelState:
    """Lightweight parallel-state container for DES-LOC.

    Megatron stores process groups in module-level globals inside
    ``megatron.core.parallel_state``.  DeepSpeed doesn't have that module, so we
    aggregate the same information here and pass it explicitly.

    slc_buffers : per-module SLC staging buffers keyed by module name
    local_rank  : CUDA device index for this process
    """
    module_pgs:  Dict[str, ProcessGroupCollection] = field(default_factory=dict)
    slc_buffers: Dict[str, SLCBuffer]              = field(default_factory=dict)
    local_rank:  int = 0

    def get_pg(self, module_name: str, attr: str) -> Optional[dist.ProcessGroup]:
        pgc = self.module_pgs.get(module_name)
        return getattr(pgc, attr, None) if pgc else None


# ---------------------------------------------------------------------------
# Hetero topology container
# ---------------------------------------------------------------------------

@dataclass
class HeteroTopology:
    """Process groups, grids metadata, SLC buffers and rank topology for one DES-LOC run.

    Mirrors Megatron's ``HeteroTopology`` but replaces ``HyperCommGrid`` objects
    with plain ``ModuleGridSpec`` (grids are implicit in the rank layout) and adds
    DES-LOC-specific state.
    """
    specs:                 List[ModuleGridSpec]
    module_pgs:            Dict[str, ProcessGroupCollection]
    parallel_state:        DESlOCParallelState
    language_module_name:  Optional[str]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def pgc(self, module_name: str) -> Optional[ProcessGroupCollection]:
        return self.module_pgs.get(module_name)

    def slc(self, module_name: str) -> Optional[SLCBuffer]:
        return self.parallel_state.slc_buffers.get(module_name)

    def spec_for_rank(self, rank: Optional[int] = None) -> Optional[ModuleGridSpec]:
        """Return the ModuleGridSpec that owns *rank* (default: current rank)."""
        if rank is None:
            rank = dist.get_rank()
        for spec in self.specs:
            if spec.contains_rank(rank):
                return spec
        return None

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Destroy every owned process group and SLC buffer (idempotent)."""
        destroyed: Set[int] = set()
        for pgc in self.module_pgs.values():
            for pg in pgc.owned_groups():
                if id(pg) in destroyed:
                    continue
                try:
                    if _is_process_group_member(pg):
                        dist.destroy_process_group(pg)
                except Exception as exc:
                    logger.warning("destroy_process_group failed (ignored): %s", exc)
                destroyed.add(id(pg))
        for buf in self.parallel_state.slc_buffers.values():
            buf.destroy()
        logger.info("HeteroTopology destroyed (%d PGs, %d SLC buffers)",
                    len(destroyed), len(self.parallel_state.slc_buffers))


# ---------------------------------------------------------------------------
# Distributed bootstrap (replaces Megatron's examples/mimo/training/distributed.py)
# ---------------------------------------------------------------------------

def initialize_distributed() -> None:
    """Bring up ``torch.distributed`` without initialising DeepSpeed's engine.

    DES-LOC bootstrap sequence
    --------------------------
    1. Resolve ``LOCAL_RANK`` from the environment (torchrun sets this).
    2. Bind CUDA device so that NCCL can pick the right NIC/PCIe path.
    3. Initialise the default process group (NCCL backend).
    4. Log the hardware tier of this rank (SM86 vs SM90).
    5. ``dist.barrier()`` so all ranks are synchronised before topology build.

    Unlike Megatron's version we do *not* touch ``parallel_state`` because
    DeepSpeed doesn't have it.  The ``DESlOCParallelState`` is populated later
    in ``create_topology``.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        backend = os.environ.get("DESCLOC_DIST_BACKEND", "nccl")
        logger.info(
            "initialize_distributed: local_rank=%d backend=%s", local_rank, backend
        )
        dist.init_process_group(
            backend=backend,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        logger.debug("initialize_distributed: process group already initialised, skipping")

    _log_device_capability(local_rank)
    dist.barrier()
    logger.info(
        "initialize_distributed: world_size=%d rank=%d ready",
        dist.get_world_size(), dist.get_rank(),
    )


def shutdown_distributed() -> None:
    """Tear down ``torch.distributed`` (thin wrapper; mirrors Megatron's shutdown_distributed)."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        logger.info("shutdown_distributed: process group destroyed")
    else:
        logger.debug("shutdown_distributed: already destroyed, no-op")


def _log_device_capability(local_rank: int) -> None:
    """Emit an INFO line indicating SM version – helps identify A6000 vs H100 ranks."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(local_rank)
        name = torch.cuda.get_device_name(local_rank)
        tier = "FAST(SM90)" if major >= 9 else "SLOW(SM86)"
        logger.info("Rank %d device: %s  SM%d%d  DES-LOC tier: %s",
                    dist.get_rank() if dist.is_initialized() else local_rank,
                    name, major, minor, tier)


# ---------------------------------------------------------------------------
# Topology validation helpers
# ---------------------------------------------------------------------------

def _validate_grid_layout(specs: List[ModuleGridSpec]) -> None:
    """Assert that specs either fully share ranks (colocated) or partition the world.

    Mirrors Megatron's ``_validate_grid_layout`` but operates on ``ModuleGridSpec``
    objects instead of ``HyperCommGrid`` instances.

    Rules (identical semantics to upstream)
    ----------------------------------------
    1. Spans must be either all-equal (colocated) or pairwise-disjoint.
    2. Disjoint spans must tile ``[0, world_size)`` with no gaps.
    """
    spans: Dict[str, Tuple[int, int]] = {
        s.name: (s.rank_offset, s.rank_end) for s in specs
    }
    names = list(spans)

    all_same = all(spans[n] == spans[names[0]] for n in names)
    pairwise_disjoint = all(
        spans[a][1] <= spans[b][0] or spans[b][1] <= spans[a][0]
        for i, a in enumerate(names)
        for b in names[i + 1:]
    )

    if not (all_same or pairwise_disjoint):
        raise ValueError(
            "DES-LOC: Module grids must either fully share ranks (colocated) or be "
            f"pairwise disjoint, got: {spans}"
        )

    if pairwise_disjoint and not all_same:
        world_size = dist.get_world_size()
        covered: Set[int] = set()
        for start, end in spans.values():
            covered.update(range(start, end))
        if covered != set(range(world_size)):
            raise ValueError(
                f"DES-LOC: Module grids must partition [0, {world_size}) with no gaps, "
                f"got: {spans}"
            )

    logger.debug("_validate_grid_layout: %s – layout valid", "colocated" if all_same else "disjoint")


def _assert_language_module_unique(specs: List[ModuleGridSpec]) -> None:
    """Exactly one spec must carry the ``LANGUAGE_MODULE_KEY`` name."""
    lang_specs = [s for s in specs if s.name == LANGUAGE_MODULE_KEY]
    if len(lang_specs) != 1:
        raise ValueError(
            f"create_topology requires exactly one spec named {LANGUAGE_MODULE_KEY!r}, "
            f"got {len(lang_specs)}: {[s.name for s in lang_specs]}"
        )


# ---------------------------------------------------------------------------
# Process group construction
# ---------------------------------------------------------------------------

def _rank_groups_for_dim(
    spec: ModuleGridSpec,
    outer_size: int,
    inner_size: int,
) -> List[List[int]]:
    """Enumerate contiguous rank sublists of ``inner_size`` within this module's span.

    This replicates what ``HyperCommGrid.get_rank_enum`` does for a single dim.
    We stride over the module's rank slice in steps of ``inner_size``.
    """
    groups: List[List[int]] = []
    stride = inner_size
    count = outer_size
    base = spec.rank_offset
    for outer in range(count):
        group = list(range(base + outer * stride, base + outer * stride + inner_size))
        groups.append(group)
    return groups


def _new_pg_if_member(
    rank_list: List[int],
    current_rank: int,
    backend: str = "nccl",
) -> Optional[dist.ProcessGroup]:
    """Collective new_group; return the group if *current_rank* is a member, else None.

    This mirrors Megatron's pattern: every rank participates in the collective
    ``new_group`` call (required by NCCL), but only members receive a non-None handle.
    """
    pg = dist.new_group(ranks=rank_list, backend=backend)
    if current_rank in rank_list:
        return pg
    return None


def _build_dense_pgs(
    spec: ModuleGridSpec,
    current_rank: int,
) -> ProcessGroupCollection:
    """Build TP / CP / PP / DP and joint groups for *spec*.

    Group enumeration strategy (mirrors Megatron _build_grid)
    ----------------------------------------------------------
    We treat the rank layout as a 4-D tensor of shape [tp, cp, dp, pp] laid out
    in row-major order within the module's rank slice.  All processes in the world
    participate in each ``new_group`` call (NCCL collective requirement).
    """
    pgc = ProcessGroupCollection()
    world = dist.get_world_size()
    base  = spec.rank_offset

    tp, cp, dp, pp = spec.tp, spec.cp, spec.dp, spec.pp

    # Helpers to linearise a 4-D coordinate (tp_i, cp_i, dp_i, pp_i) → global rank
    def rank_of(ti: int, ci: int, di: int, pi: int) -> int:
        return base + ti * (cp * dp * pp) + ci * (dp * pp) + di * pp + pi

    # ------------------------------------------------------------------
    # TP groups  (vary ti; fix ci, di, pi)
    # ------------------------------------------------------------------
    for ci in range(cp):
        for di in range(dp):
            for pi in range(pp):
                rlist = [rank_of(ti, ci, di, pi) for ti in range(tp)]
                pg = _new_pg_if_member(rlist, current_rank)
                if pg is not None and pgc.tp is None:
                    pgc.tp = pg

    # ------------------------------------------------------------------
    # CP groups  (vary ci; fix ti, di, pi)
    # ------------------------------------------------------------------
    for ti in range(tp):
        for di in range(dp):
            for pi in range(pp):
                rlist = [rank_of(ti, ci, di, pi) for ci in range(cp)]
                pg = _new_pg_if_member(rlist, current_rank)
                if pg is not None and pgc.cp is None:
                    pgc.cp = pg

    # ------------------------------------------------------------------
    # PP groups  (vary pi; fix ti, ci, di)
    # ------------------------------------------------------------------
    for ti in range(tp):
        for ci in range(cp):
            for di in range(dp):
                rlist = [rank_of(ti, ci, di, pi) for pi in range(pp)]
                pg = _new_pg_if_member(rlist, current_rank)
                if pg is not None and pgc.pp is None:
                    pgc.pp = pg

    # ------------------------------------------------------------------
    # DP groups  (vary di; fix ti, ci, pi)
    # ------------------------------------------------------------------
    for ti in range(tp):
        for ci in range(cp):
            for pi in range(pp):
                rlist = [rank_of(ti, ci, di, pi) for di in range(dp)]
                pg = _new_pg_if_member(rlist, current_rank)
                if pg is not None and pgc.dp is None:
                    pgc.dp = pg

    # ------------------------------------------------------------------
    # DP×CP joint groups  (vary di, ci; fix ti, pi)
    # ------------------------------------------------------------------
    for ti in range(tp):
        for pi in range(pp):
            rlist = [
                rank_of(ti, ci, di, pi)
                for di in range(dp) for ci in range(cp)
            ]
            pg = _new_pg_if_member(rlist, current_rank)
            if pg is not None and pgc.dp_cp is None:
                pgc.dp_cp = pg

    # ------------------------------------------------------------------
    # TP×CP joint groups  (vary ti, ci; fix di, pi)
    # ------------------------------------------------------------------
    for di in range(dp):
        for pi in range(pp):
            rlist = [
                rank_of(ti, ci, di, pi)
                for ti in range(tp) for ci in range(cp)
            ]
            pg = _new_pg_if_member(rlist, current_rank)
            if pg is not None and pgc.tp_cp is None:
                pgc.tp_cp = pg

    # ------------------------------------------------------------------
    # MP = TP×PP  (vary ti, pi; fix ci, di)
    # ------------------------------------------------------------------
    for ci in range(cp):
        for di in range(dp):
            rlist = [
                rank_of(ti, ci, di, pi)
                for ti in range(tp) for pi in range(pp)
            ]
            pg = _new_pg_if_member(rlist, current_rank)
            if pg is not None and pgc.mp is None:
                pgc.mp = pg

    logger.debug(
        "_build_dense_pgs '%s': rank %d in tp=%s cp=%s pp=%s dp=%s",
        spec.name, current_rank,
        pgc.tp is not None, pgc.cp is not None,
        pgc.pp is not None, pgc.dp is not None,
    )
    return pgc


def _build_expert_pgs(
    spec: ModuleGridSpec,
    pgc: ProcessGroupCollection,
    current_rank: int,
) -> None:
    """Build expert-parallel groups (EP / expt_tp / expt_dp and joints) in-place.

    The expert grid shares PP with the dense grid (``shared_dims=['pp']`` in
    Megatron) but uses a different TP factorisation.  We reuse the PP group from
    the dense grid and only build new groups for the expert-specific dimensions.
    """
    espec = spec.expert
    if espec is None:
        return

    base = spec.rank_offset
    etp, ep, edp, pp = espec.expt_tp, espec.ep, espec.expt_dp, espec.pp

    def erank_of(eti: int, ei: int, edi: int, pi: int) -> int:
        return base + eti * (ep * edp * pp) + ei * (edp * pp) + edi * pp + pi

    # EP groups
    for eti in range(etp):
        for edi in range(edp):
            for pi in range(pp):
                rlist = [erank_of(eti, ei, edi, pi) for ei in range(ep)]
                pg = _new_pg_if_member(rlist, current_rank)
                if pg is not None and pgc.ep is None:
                    pgc.ep = pg

    # expt_tp groups
    for ei in range(ep):
        for edi in range(edp):
            for pi in range(pp):
                rlist = [erank_of(eti, ei, edi, pi) for eti in range(etp)]
                pg = _new_pg_if_member(rlist, current_rank)
                if pg is not None and pgc.expt_tp is None:
                    pgc.expt_tp = pg

    # expt_dp groups
    for eti in range(etp):
        for ei in range(ep):
            for pi in range(pp):
                rlist = [erank_of(eti, ei, edi, pi) for edi in range(edp)]
                pg = _new_pg_if_member(rlist, current_rank)
                if pg is not None and pgc.expt_dp is None:
                    pgc.expt_dp = pg

    # TP×EP joint
    for edi in range(edp):
        for pi in range(pp):
            rlist = [
                erank_of(eti, ei, edi, pi)
                for eti in range(etp) for ei in range(ep)
            ]
            pg = _new_pg_if_member(rlist, current_rank)
            if pg is not None and pgc.tp_ep is None:
                pgc.tp_ep = pg

    # TP×EP×PP joint
    for edi in range(edp):
        rlist = [
            erank_of(eti, ei, edi, pi)
            for eti in range(etp) for ei in range(ep) for pi in range(pp)
        ]
        pg = _new_pg_if_member(rlist, current_rank)
        if pg is not None and pgc.tp_ep_pp is None:
            pgc.tp_ep_pp = pg

    logger.debug(
        "_build_expert_pgs '%s': rank %d in ep=%s expt_tp=%s expt_dp=%s",
        spec.name, current_rank,
        pgc.ep is not None, pgc.expt_tp is not None, pgc.expt_dp is not None,
    )


def _default_embedding_ranks(pp_ranks: List[int]) -> List[int]:
    """Return word-embedding ranks: first + last pipeline stage if pp > 1, else just first.

    Mirrors Megatron's ``default_embedding_ranks`` without the megatron import.
    """
    if len(pp_ranks) <= 1:
        return pp_ranks
    return [pp_ranks[0], pp_ranks[-1]]


def _default_position_embedding_ranks(pp_ranks: List[int]) -> List[int]:
    """Return position-embedding ranks: only the first pipeline stage."""
    return [pp_ranks[0]] if pp_ranks else []


def _build_language_embedding_pgs(
    spec: ModuleGridSpec,
    pgc: ProcessGroupCollection,
    current_rank: int,
) -> None:
    """Build word/position embedding groups for the language module (in-place).

    Creation is collective: every rank in the world calls new_group for each PP
    tuple, mirroring Megatron's ``_build_language_embedding_groups``.
    """
    base = spec.rank_offset
    tp, cp, dp, pp_size = spec.tp, spec.cp, spec.dp, spec.pp

    def rank_of(ti: int, ci: int, di: int, pi: int) -> int:
        return base + ti * (cp * dp * pp_size) + ci * (dp * pp_size) + di * pp_size + pi

    # Enumerate all PP rank lists (one per (ti, ci, di) combination)
    for ti in range(tp):
        for ci in range(cp):
            for di in range(dp):
                pp_ranks = [rank_of(ti, ci, di, pi) for pi in range(pp_size)]
                emb_ranks = _default_embedding_ranks(pp_ranks)
                pos_ranks = _default_position_embedding_ranks(pp_ranks)

                emb_pg = dist.new_group(ranks=emb_ranks)
                pos_pg = dist.new_group(ranks=pos_ranks)

                if current_rank in emb_ranks and pgc.embd is None:
                    pgc.embd = emb_pg
                if current_rank in pos_ranks and pgc.pos_embd is None:
                    pgc.pos_embd = pos_pg

    logger.debug(
        "_build_language_embedding_pgs '%s': rank %d embd=%s pos_embd=%s",
        spec.name, current_rank,
        pgc.embd is not None, pgc.pos_embd is not None,
    )


# ---------------------------------------------------------------------------
# Public topology factory
# ---------------------------------------------------------------------------

def create_topology(specs: List[ModuleGridSpec]) -> HeteroTopology:
    """Build the full DES-LOC heterogeneous topology for a MIMO run.

    Steps
    -----
    1. Validate inputs (≥1 spec, exactly one language module).
    2. Validate grid layout (colocated xor disjoint, no gaps).
    3. Build dense PGs for every module (all ranks participate collectively).
    4. Build expert PGs for every module.
    5. Build embedding PGs for the language module only.
    6. Allocate SLC staging buffers.
    7. Construct and return a ``HeteroTopology``.

    On any exception during steps 3-7 all already-created groups are destroyed
    before re-raising, preventing resource leaks.

    Parameters
    ----------
    specs : list of ModuleGridSpec
        One entry per MIMO module; must include exactly one named
        ``LANGUAGE_MODULE_KEY``.

    Returns
    -------
    HeteroTopology
        Fully initialised topology; caller must call ``.destroy()`` when done.
    """
    if not specs:
        raise ValueError("create_topology requires at least one ModuleGridSpec")

    _assert_language_module_unique(specs)
    _validate_grid_layout(specs)

    current_rank = dist.get_rank()
    language_name: Optional[str] = None
    module_pgs: Dict[str, ProcessGroupCollection] = {}
    slc_buffers: Dict[str, SLCBuffer] = {}

    try:
        for spec in specs:
            logger.info(
                "create_topology: building PGs for module '%s' "
                "(ranks %d–%d, tier=%s)",
                spec.name, spec.rank_offset, spec.rank_end - 1,
                spec.device_tier.name,
            )
            pgc = _build_dense_pgs(spec, current_rank)
            _build_expert_pgs(spec, pgc, current_rank)

            if spec.name == LANGUAGE_MODULE_KEY:
                _build_language_embedding_pgs(spec, pgc, current_rank)
                language_name = spec.name

            module_pgs[spec.name] = pgc

            # SLC buffer – only for ranks that belong to this module
            if spec.contains_rank(current_rank) and spec.slc_size_bytes > 0:
                slc_buffers[spec.name] = SLCBuffer(spec.name, spec.slc_size_bytes)
            else:
                slc_buffers[spec.name] = SLCBuffer(spec.name, 0)

    except Exception:
        # Cleanup partial state before propagating
        _partial_cleanup(module_pgs, slc_buffers)
        raise

    ps = DESlOCParallelState(
        module_pgs=module_pgs,
        slc_buffers=slc_buffers,
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )

    topo = HeteroTopology(
        specs=specs,
        module_pgs=module_pgs,
        parallel_state=ps,
        language_module_name=language_name,
    )
    logger.info(
        "create_topology: topology ready – %d modules, language='%s', rank=%d",
        len(specs), language_name, current_rank,
    )
    return topo


def _partial_cleanup(
    module_pgs: Dict[str, ProcessGroupCollection],
    slc_buffers: Dict[str, SLCBuffer],
) -> None:
    """Best-effort cleanup used when ``create_topology`` aborts mid-way."""
    destroyed: Set[int] = set()
    for pgc in module_pgs.values():
        for pg in pgc.owned_groups():
            if id(pg) not in destroyed:
                try:
                    if _is_process_group_member(pg):
                        dist.destroy_process_group(pg)
                except Exception as exc:
                    logger.warning("_partial_cleanup: destroy failed: %s", exc)
                destroyed.add(id(pg))
    for buf in slc_buffers.values():
        buf.destroy()
    logger.debug("_partial_cleanup: freed %d PGs and %d SLC buffers",
                 len(destroyed), len(slc_buffers))


def _is_process_group_member(pg: Optional[dist.ProcessGroup]) -> bool:
    """Return True iff the current rank belongs to *pg*.

    ``dist.get_rank(group=pg)`` returns -1 for non-members without raising.
    """
    return pg is not None and dist.get_rank(group=pg) >= 0


# ---------------------------------------------------------------------------
# Smoke test  (single-process, no GPU required for structural checks)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # --- Unit-level structural tests (no distributed init needed) ---

    # 1. ModuleGridSpec dp derivation
    s = ModuleGridSpec("images", num_ranks=4, tp=2, rank_offset=0, device_tier=DeviceTier.SLOW)
    assert s.dp == 2, f"expected dp=2, got {s.dp}"

    # 2. Expert spec defaults synthesised
    assert s.expert is not None
    assert s.expert.ep == 1 and s.expert.expt_tp == 1

    # 3. Bad divisibility raises
    try:
        ModuleGridSpec("bad", num_ranks=5, tp=2)
        assert False, "should have raised ValueError"
    except ValueError:
        pass

    # 4. ExpertGridSpec indivisible raises
    try:
        ExpertGridSpec(num_ranks=4, pp=1, expt_tp=3, ep=2)
        assert False, "should have raised ValueError"
    except ValueError:
        pass

    # 5. SLCBuffer disabled when size=0
    buf = SLCBuffer("test", 0)
    assert not buf.enabled

    logger.info("All smoke tests passed.")
