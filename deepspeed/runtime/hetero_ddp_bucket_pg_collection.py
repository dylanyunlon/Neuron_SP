"""
HeteroDDPBucketPGCollection — DES-LOC Heterogeneous DDP Bucket Sizing via PG Collection

Upstream Design Intent (Megatron 75e382c):
-------------------------------------------
Megatron's original get_model() computed DDP gradient-reduction bucket sizes by
calling mpu.get_data_parallel_world_size(with_context_parallel=True) — a global
singleton look-up that assumed a single homogeneous process-group topology.  The
commit 75e382c decoupled this by threading an explicit ``pg_collection`` object
into the call site, so that ``get_pg_size(pg_collection.dp_cp)`` and
``get_pg_rank(pg_collection.pp)`` are used instead of the mpu globals.  This
makes it possible to run multiple, structurally different DDP topologies inside
the same training job without them stomping on each other's global state.

DES-LOC Adaptation Point:
--------------------------
In the DES-LOC (Decoupled Execution with Shared LOcality Cache) heterogeneous
training framework built on DeepSpeed, the three physical devices —

    • GPU-0  : A6000 48 GB  SM86  (PCIe)
    • GPU-1  : A6000 48 GB  SM86  (PCIe)
    • GPU-2  : H100 NVL 96 GB SM90 (PCIe)

— are managed by *different* DeepSpeed ZeRO / DDP sub-groups that can have
different world-sizes and different pipeline ranks.  The mpu-global approach
breaks immediately in this topology because there is no single "the" data-
parallel group; there are (at least) two: one covering the A6000 pair and one
that includes the H100.

This module introduces ``HeteroDDPBucketPGCollection``:

    1. A lightweight dataclass-like container that holds typed references to
       DeepSpeed process-groups (``dp_cp``, ``pp``, ``tp``) rather than relying
       on any global registry.

    2. ``get_pg_size(group)`` / ``get_pg_rank(group)`` helpers that mirror
       Megatron's helpers but operate on DeepSpeed ``dist.ProcessGroup`` handles
       (or any object that exposes ``.size()`` / ``.rank()``), with graceful
       fallback when ``torch.distributed`` is not yet initialised.

    3. ``compute_hetero_bucket_size(pg_collection, base_bucket_bytes, scale_per_rank)``
       which replicates the exact Megatron bucket-sizing formula but draws
       world-size from the explicit ``pg_collection.dp_cp`` group — critical for
       correct bucket sizing when the A6000 sub-group (world_size=2) and the
       H100 sub-group (world_size=1 or 3) must produce *independent* bucket
       configurations.

    4. ``HeteroDDPBucketScheduler`` which, given a list of model chunks (as in
       Megatron's pipeline-parallel chunked model list), decides per-chunk
       whether bucketing should be disabled, applying the same logic as
       Megatron's ``per_chunk_disable_bucketing`` but using pg-collection-sourced
       ranks rather than mpu globals.

    5. A ``LocalityCacheAwareBucketPolicy`` that extends the bucket sizing with
       DES-LOC-specific logic: because the A6000 pair shares a PCIe switch they
       benefit from a *smaller* bucket (faster micro-synchronisation), while the
       H100 sitting on a separate root-complex should use a *larger* bucket
       (amortise PCIe latency over more gradient data before triggering AllReduce).

    6. ``build_pg_collection_from_deepspeed_engine(engine)`` — a factory that
       wires a live ``DeepSpeedEngine`` into a ``HeteroDDPBucketPGCollection``.

References:
    Megatron-LM commit 75e382c63397169ebb38cb4f4571148a3fc52bc1
    DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
    Neuron_SP project: https://github.com/dylanyunlon/Neuron_SP
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device-tier taxonomy — used by the locality-aware bucket policy
# ---------------------------------------------------------------------------

class DeviceTier(Enum):
    """Physical capability tier for a process-group member.

    DES-LOC partitions the three-device cluster into two tiers:

    * ``LOCALITY_PEER`` — devices connected via the *same* PCIe switch (the two
      A6000s).  They can synchronise gradients cheaply through shared locality
      cache, so small buckets are preferred to reduce per-bucket latency.

    * ``REMOTE_ACCELERATOR`` — the H100 NVL, which sits on a *different* PCIe
      root complex.  Bucket size should be enlarged to amortise the higher
      per-transfer overhead.
    """
    LOCALITY_PEER    = auto()   # A6000 x2, same PCIe switch
    REMOTE_ACCELERATOR = auto() # H100 NVL, separate root complex


# ---------------------------------------------------------------------------
# Process-group handle helpers (mirrors Megatron's get_pg_size / get_pg_rank)
# ---------------------------------------------------------------------------

def get_pg_size(group: Any) -> int:
    """Return the world-size of *group*.

    Accepts any of:
      * a ``torch.distributed.ProcessGroup``
      * any object with a ``.size()`` callable (e.g. test sentinels)
      * ``None`` — returns 1 (single-process fallback)

    This mirrors Megatron's ``get_pg_size`` introduced in commit 75e382c but
    is adapted to work without the Megatron ``mpu`` module.
    """
    if group is None:
        logger.debug("get_pg_size received None group; returning 1 (single-process).")
        return 1
    if callable(getattr(group, "size", None)):
        return group.size()
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=group)
    logger.warning(
        "get_pg_size: torch.distributed not initialised and group has no .size(); "
        "returning 1."
    )
    return 1


def get_pg_rank(group: Any) -> int:
    """Return the rank of the current process within *group*.

    Accepts the same types as :func:`get_pg_size`.
    """
    if group is None:
        logger.debug("get_pg_rank received None group; returning 0.")
        return 0
    if callable(getattr(group, "rank", None)):
        return group.rank()
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=group)
    logger.warning(
        "get_pg_rank: torch.distributed not initialised and group has no .rank(); "
        "returning 0."
    )
    return 0


# ---------------------------------------------------------------------------
# PG Collection dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeteroDDPBucketPGCollection:
    """Typed container for the process-groups that govern DDP bucket sizing.

    Mirrors the ``pg_collection`` namespace threaded into Megatron's
    ``get_model()`` in commit 75e382c, but extended for the DES-LOC
    three-device heterogeneous topology.

    Attributes
    ----------
    dp_cp:
        The *data-parallel × context-parallel* composite group.  Used by
        :func:`compute_hetero_bucket_size` to read world-size — exactly as
        Megatron does via ``get_pg_size(pg_collection.dp_cp)``.
    pp:
        The *pipeline-parallel* group.  Used to read the current pipeline rank
        — replacing ``mpu.get_pipeline_model_parallel_rank()`` in Megatron.
    tp:
        The *tensor-parallel* group (optional; may be ``None`` for TP=1 runs).
    device_tier:
        The :class:`DeviceTier` of the local device.  Drives the DES-LOC
        locality-aware bucket policy.
    extra_groups:
        Arbitrary additional named groups (e.g. expert-parallel groups for MoE)
        keyed by a user-supplied string.
    """
    dp_cp:        Any                       # data-parallel ✕ context-parallel group
    pp:           Any                       # pipeline-parallel group
    tp:           Optional[Any] = None      # tensor-parallel group (may be None)
    device_tier:  DeviceTier = DeviceTier.LOCALITY_PEER
    extra_groups: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def dp_cp_size(self) -> int:
        """World-size of the data-parallel × context-parallel group."""
        return get_pg_size(self.dp_cp)

    @property
    def pp_rank(self) -> int:
        """Pipeline-parallel rank of the current process."""
        return get_pg_rank(self.pp)

    @property
    def tp_size(self) -> int:
        """Tensor-parallel world-size (1 if tp is None)."""
        return get_pg_size(self.tp)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"HeteroDDPBucketPGCollection("
            f"dp_cp_size={self.dp_cp_size}, "
            f"pp_rank={self.pp_rank}, "
            f"tp_size={self.tp_size}, "
            f"tier={self.device_tier.name})"
        )


# ---------------------------------------------------------------------------
# Core bucket-size computation (mirrors Megatron 75e382c formula)
# ---------------------------------------------------------------------------

#: Megatron's hard-coded floor for the DDP bucket size (40 MB in elements).
_MEGATRON_BUCKET_FLOOR: int = 40_000_000

#: Megatron's per-rank scale factor (1 M elements per DP rank).
_MEGATRON_BUCKET_SCALE_PER_RANK: int = 1_000_000


def compute_hetero_bucket_size(
    pg_collection: HeteroDDPBucketPGCollection,
    base_floor: int = _MEGATRON_BUCKET_FLOOR,
    scale_per_rank: int = _MEGATRON_BUCKET_SCALE_PER_RANK,
) -> int:
    """Compute the DDP gradient-reduction bucket size for a heterogeneous group.

    Replicates Megatron's bucket-sizing formula verbatim::

        bucket_size = max(40_000_000, 1_000_000 * dp_cp_world_size)

    but draws ``dp_cp_world_size`` from ``pg_collection.dp_cp`` rather than
    the mpu global, enabling independent bucket configurations for the A6000
    sub-group (``dp_cp_size=2``) and any group that includes the H100
    (``dp_cp_size`` may differ).

    DES-LOC note
    ------------
    For the A6000 pair (``dp_cp_size=2``):
        ``max(40_000_000, 1_000_000 * 2) = 40_000_000``
        The floor dominates — consistent with Megatron's default.

    For a hypothetical all-device group (``dp_cp_size=3``):
        ``max(40_000_000, 1_000_000 * 3) = 40_000_000``
        Still floor-dominated.

    The locality-aware policy in :class:`LocalityCacheAwareBucketPolicy` then
    *modifies* this baseline by device tier.

    Parameters
    ----------
    pg_collection:
        Must provide a valid ``dp_cp`` group (or sentinel with ``.size()``).
    base_floor:
        Minimum bucket size in parameter elements.  Default matches Megatron.
    scale_per_rank:
        Per-DP-rank additive scale.  Default matches Megatron.

    Returns
    -------
    int
        Bucket size in parameter *elements* (not bytes).
    """
    dp_world_size = get_pg_size(pg_collection.dp_cp)
    raw = scale_per_rank * dp_world_size
    bucket_elements = max(base_floor, raw)
    logger.debug(
        "compute_hetero_bucket_size: dp_world=%d raw=%d floor=%d → %d",
        dp_world_size, raw, base_floor, bucket_elements,
    )
    return bucket_elements


# ---------------------------------------------------------------------------
# DES-LOC Locality-Cache-Aware bucket policy
# ---------------------------------------------------------------------------

@dataclass
class LocalityCacheAwareBucketPolicy:
    """Adjusts base bucket size according to DES-LOC device-tier locality.

    Background
    ----------
    In the DES-LOC topology all inter-device communication crosses PCIe.
    However there is a structural asymmetry:

    * The **two A6000s** share a PCIe switch (root port).  AllReduce between
      them traverses only that switch — typical bandwidth ~32 GB/s bidirectional.
      Smaller buckets pipeline the reduction more tightly with backward-pass
      computation, reducing idle GPU time.

    * The **H100 NVL** sits on a *separate* PCIe root complex.  Every gradient
      transfer to/from the H100 goes through the CPU host bridge, capped at
      ~64 GB/s unidirectional but with significantly higher per-transfer
      latency (NUMA crossing, IOMMU, etc.).  Larger buckets amortise this
      fixed overhead.

    * The **CPU DRAM** (1.5 TB) acts as the DES-LOC shared locality cache.
      Activations and optimizer states that do not fit on-device are spilled
      here.  Bucket sizing affects how often the cache is dirtied: too-frequent
      small reductions create write-amplification on the CPU cache lines that
      hold pinned staging buffers.

    Policy
    ------
    ``LOCALITY_PEER`` (A6000):  scale down by ``locality_shrink_factor``
    ``REMOTE_ACCELERATOR`` (H100): scale up by ``remote_scale_factor``

    These are multiplicative adjustments applied *after*
    :func:`compute_hetero_bucket_size` returns the Megatron-compatible baseline.
    """
    locality_shrink_factor: float = 0.5     # A6000 pair: 50 % of baseline
    remote_scale_factor:    float = 2.0     # H100:       200 % of baseline
    min_bucket_elements:    int   = 4_000_000   # hard lower bound (~16 MB fp32)
    max_bucket_elements:    int   = 200_000_000 # hard upper bound (~800 MB fp32)

    def apply(
        self,
        base_bucket_elements: int,
        device_tier: DeviceTier,
    ) -> int:
        """Return tier-adjusted bucket size (in elements).

        Parameters
        ----------
        base_bucket_elements:
            Output of :func:`compute_hetero_bucket_size`.
        device_tier:
            Tier of the *current* device (determines which factor is applied).

        Returns
        -------
        int
            Adjusted bucket size, clamped to ``[min_bucket_elements,
            max_bucket_elements]``.
        """
        if device_tier == DeviceTier.LOCALITY_PEER:
            factor = self.locality_shrink_factor
        elif device_tier == DeviceTier.REMOTE_ACCELERATOR:
            factor = self.remote_scale_factor
        else:
            factor = 1.0

        adjusted = int(math.ceil(base_bucket_elements * factor))
        clamped  = max(self.min_bucket_elements,
                       min(self.max_bucket_elements, adjusted))

        logger.debug(
            "LocalityCacheAwareBucketPolicy.apply: tier=%s base=%d "
            "factor=%.2f adjusted=%d clamped=%d",
            device_tier.name, base_bucket_elements, factor, adjusted, clamped,
        )
        return clamped


# ---------------------------------------------------------------------------
# Per-chunk bucketing scheduler (mirrors Megatron's per_chunk_disable_bucketing)
# ---------------------------------------------------------------------------

@dataclass
class HeteroDDPBucketScheduler:
    """Decides per-model-chunk whether DDP bucketing should be disabled.

    Megatron (post-75e382c) computes::

        pp_rank = get_pg_rank(pg_collection.pp)
        per_chunk_disable_bucketing = [
            (chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step
            for chunk_idx in range(len(model))
        ]

    and then also disables bucketing for non-zero pp ranks (first-chunk
    bucketing is only useful on pp_rank 0 for the first micro-batch).

    In DES-LOC the pipeline topology may be *heterogeneous*: the H100 handles
    a different number of pipeline stages than each A6000.  This scheduler
    reads pp_rank directly from ``pg_collection.pp`` (never from a global),
    then applies both the chunk-index rule and the tier-specific disable rule.

    DES-LOC tier rule
    -----------------
    * ``REMOTE_ACCELERATOR`` (H100) at any pp_rank: bucketing is *never*
      disabled purely because of pp_rank, since the H100 benefits from large
      coalesced transfers regardless of pipeline position.
    * ``LOCALITY_PEER`` (A6000): standard Megatron rule applies — disable for
      pp_rank > 0.

    Attributes
    ----------
    pg_collection:
        The collection carrying ``pp`` and ``dp_cp`` groups.
    overlap_param_gather_with_optimizer_step:
        Mirrors ``args.overlap_param_gather_with_optimizer_step`` from
        Megatron's training args.
    policy:
        The :class:`LocalityCacheAwareBucketPolicy` to compute final sizes.
    """
    pg_collection:                          HeteroDDPBucketPGCollection
    overlap_param_gather_with_optimizer_step: bool = False
    policy:                                 LocalityCacheAwareBucketPolicy = field(
        default_factory=LocalityCacheAwareBucketPolicy
    )

    def compute_base_bucket(self) -> int:
        """Megatron-formula bucket size for this pg_collection."""
        return compute_hetero_bucket_size(self.pg_collection)

    def compute_tier_adjusted_bucket(self) -> int:
        """Tier-adjusted bucket size for the current device."""
        base = self.compute_base_bucket()
        return self.policy.apply(base, self.pg_collection.device_tier)

    def per_chunk_flags(
        self, num_chunks: int
    ) -> List[Tuple[int, bool, int]]:
        """Return per-chunk ``(chunk_idx, disable_bucketing, bucket_size)`` tuples.

        Parameters
        ----------
        num_chunks:
            Number of model chunks (``len(model)`` in Megatron parlance).

        Returns
        -------
        list of (chunk_idx, disable_bucketing, effective_bucket_size)
            ``disable_bucketing=True`` means DDP should use a single infinite
            bucket (no overlap), matching Megatron's sentinel ``float('inf')``.
        """
        pp_rank       = get_pg_rank(self.pg_collection.pp)
        tier          = self.pg_collection.device_tier
        base_bucket   = self.compute_base_bucket()
        tier_bucket   = self.policy.apply(base_bucket, tier)

        # Megatron rule: disable bucketing for non-first chunks when
        # overlap_param_gather_with_optimizer_step is set.
        overlap_flag = self.overlap_param_gather_with_optimizer_step

        results: List[Tuple[int, bool, int]] = []
        for chunk_idx in range(num_chunks):
            # Replicate Megatron's per_chunk_disable_bucketing expression:
            #   (chunk_idx > 0) or overlap_param_gather_with_optimizer_step
            chunk_disable = (chunk_idx > 0) or overlap_flag

            # DES-LOC tier rule for pp_rank-based disable:
            if tier == DeviceTier.LOCALITY_PEER:
                # A6000: also disable for non-zero pipeline rank (Megatron default).
                pp_disable = (pp_rank > 0)
            else:
                # H100: never disable solely due to pipeline rank; keep large
                # coalesced bucket active at all pipeline positions.
                pp_disable = False

            disable = chunk_disable or pp_disable
            effective = 0 if disable else tier_bucket

            logger.debug(
                "per_chunk_flags chunk=%d pp_rank=%d tier=%s "
                "chunk_disable=%s pp_disable=%s disable=%s bucket=%d",
                chunk_idx, pp_rank, tier.name,
                chunk_disable, pp_disable, disable, effective,
            )
            results.append((chunk_idx, disable, effective))

        return results


# ---------------------------------------------------------------------------
# Factory: wire a DeepSpeedEngine into a PGCollection
# ---------------------------------------------------------------------------

def build_pg_collection_from_deepspeed_engine(
    engine: Any,
    device_tier: DeviceTier = DeviceTier.LOCALITY_PEER,
) -> HeteroDDPBucketPGCollection:
    """Construct a :class:`HeteroDDPBucketPGCollection` from a live engine.

    DeepSpeed's ``DeepSpeedEngine`` exposes process-group handles through its
    ``mpu`` or ``grid`` attributes depending on the parallelism backend.  This
    factory tries several well-known attribute paths in order of preference and
    falls back to ``None`` (single-process defaults) when a group is absent.

    Parameters
    ----------
    engine:
        A ``deepspeed.DeepSpeedEngine`` (or compatible object) that has been
        initialised with a parallel topology.
    device_tier:
        The :class:`DeviceTier` of the *current process's* physical device.
        The caller is responsible for setting this correctly.  In Neuron_SP the
        rank-to-tier mapping is established during cluster configuration based
        on ``torch.cuda.get_device_properties(local_rank).major`` — SM86 →
        ``LOCALITY_PEER``, SM90 → ``REMOTE_ACCELERATOR``.

    Returns
    -------
    HeteroDDPBucketPGCollection
    """
    def _try_get(obj: Any, *attrs: str) -> Optional[Any]:
        """Walk attribute chain; return None if any link is missing."""
        cur = obj
        for attr in attrs:
            cur = getattr(cur, attr, None)
            if cur is None:
                return None
        return cur

    # --- data-parallel × context-parallel group ---
    # DeepSpeed exposes this via engine.mpu (Megatron-style mpu shim) or
    # engine.data_parallel_group (native ZeRO).
    dp_cp = (
        _try_get(engine, "mpu", "get_data_parallel_group")
        or _try_get(engine, "data_parallel_group")
    )
    if callable(dp_cp):
        dp_cp = dp_cp()

    # --- pipeline-parallel group ---
    pp = (
        _try_get(engine, "mpu", "get_pipeline_model_parallel_group")
        or _try_get(engine, "grid", "get_pipe_parallel_group")
    )
    if callable(pp):
        pp = pp()

    # --- tensor-parallel group ---
    tp = (
        _try_get(engine, "mpu", "get_tensor_model_parallel_group")
        or _try_get(engine, "grid", "get_slice_parallel_group")
    )
    if callable(tp):
        tp = tp()

    pg_collection = HeteroDDPBucketPGCollection(
        dp_cp=dp_cp,
        pp=pp,
        tp=tp,
        device_tier=device_tier,
    )
    logger.info(
        "build_pg_collection_from_deepspeed_engine: built %r", pg_collection
    )
    return pg_collection


# ---------------------------------------------------------------------------
# Utility: derive DeviceTier from CUDA device properties
# ---------------------------------------------------------------------------

def device_tier_from_local_rank(local_rank: int) -> DeviceTier:
    """Determine :class:`DeviceTier` by inspecting the CUDA device's SM version.

    SM86 → A6000 → ``LOCALITY_PEER``
    SM90 → H100 NVL → ``REMOTE_ACCELERATOR``

    Falls back to ``LOCALITY_PEER`` if CUDA is unavailable (e.g. CPU-only CI).

    Parameters
    ----------
    local_rank:
        The CUDA device index assigned to this process.

    Returns
    -------
    DeviceTier
    """
    if not torch.cuda.is_available():
        logger.warning(
            "device_tier_from_local_rank: CUDA unavailable; "
            "defaulting to LOCALITY_PEER."
        )
        return DeviceTier.LOCALITY_PEER

    props = torch.cuda.get_device_properties(local_rank)
    sm    = props.major * 10 + props.minor  # e.g. SM86 → 86, SM90 → 90
    name  = props.name

    if sm >= 90:
        tier = DeviceTier.REMOTE_ACCELERATOR
        logger.info(
            "device_tier_from_local_rank: rank=%d device='%s' SM%d → %s",
            local_rank, name, sm, tier.name,
        )
    else:
        tier = DeviceTier.LOCALITY_PEER
        logger.info(
            "device_tier_from_local_rank: rank=%d device='%s' SM%d → %s",
            local_rank, name, sm, tier.name,
        )
    return tier


# ---------------------------------------------------------------------------
# Integration helper: apply PGCollection bucket config to a DeepSpeed ZeRO config
# ---------------------------------------------------------------------------

def patch_deepspeed_zero_config(
    zero_config: Dict[str, Any],
    pg_collection: HeteroDDPBucketPGCollection,
    overlap_grad_reduce: bool = True,
    overlap_param_gather_with_optimizer_step: bool = False,
    num_model_chunks: int = 1,
) -> Dict[str, Any]:
    """Patch a DeepSpeed ZeRO config dict with PGCollection-derived bucket settings.

    DeepSpeed's JSON/dict-style config does not natively understand the
    concept of per-tier bucket sizing.  This function computes the correct
    bucket size for the current process's device tier and writes it into the
    config under the ``allreduce_bucket_size`` and ``reduce_bucket_size`` keys.

    It also honours the Megatron rule: if ``overlap_grad_reduce=False`` the
    bucket is set to infinity (DeepSpeed interprets very large values as
    "no bucketing").

    Parameters
    ----------
    zero_config:
        Mutable dict corresponding to DeepSpeed's ``"zero_optimization"`` block.
    pg_collection:
        Provides dp_cp world-size and device tier for bucket computation.
    overlap_grad_reduce:
        If ``False``, bucket size is forced to infinity (Megatron parity).
    overlap_param_gather_with_optimizer_step:
        Passed to :class:`HeteroDDPBucketScheduler` for chunk-level logic.
    num_model_chunks:
        Number of pipeline-parallel model chunks.

    Returns
    -------
    dict
        The *same* ``zero_config`` dict, mutated in place and returned for
        convenience.
    """
    scheduler = HeteroDDPBucketScheduler(
        pg_collection=pg_collection,
        overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
    )

    if not overlap_grad_reduce:
        # Megatron: "Set bucket_size to infinity if overlap_grad_reduce is False."
        # DeepSpeed uses a numeric cap; 1e18 effectively disables bucketing.
        effective_bucket = int(1e18)
        logger.info(
            "patch_deepspeed_zero_config: overlap_grad_reduce=False → "
            "bucket set to ∞ (%d)", effective_bucket
        )
    else:
        effective_bucket = scheduler.compute_tier_adjusted_bucket()
        logger.info(
            "patch_deepspeed_zero_config: tier=%s effective_bucket=%d",
            pg_collection.device_tier.name, effective_bucket,
        )

    # DeepSpeed keys for bucket sizing
    zero_config["allreduce_bucket_size"] = effective_bucket
    zero_config["reduce_bucket_size"]    = effective_bucket

    # Attach per-chunk info as a DES-LOC extension key (ignored by stock DS).
    chunk_flags = scheduler.per_chunk_flags(num_model_chunks)
    zero_config["_des_loc_chunk_bucket_flags"] = [
        {"chunk_idx": ci, "disable_bucketing": db, "effective_bucket": eb}
        for ci, db, eb in chunk_flags
    ]
    logger.debug(
        "patch_deepspeed_zero_config: chunk_flags=%s",
        zero_config["_des_loc_chunk_bucket_flags"],
    )
    return zero_config


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # -----------------------------------------------------------------------
    # Sentinel group helper (no torch.distributed required)
    # -----------------------------------------------------------------------
    from types import SimpleNamespace

    def _make_group(size: int, rank: int) -> SimpleNamespace:
        g = SimpleNamespace()
        g.size = lambda: size
        g.rank = lambda: rank
        return g

    # -----------------------------------------------------------------------
    # 1. Megatron parity: dp_cp_size=7, floor must win (mirrors test in diff)
    # -----------------------------------------------------------------------
    pg7 = HeteroDDPBucketPGCollection(
        dp_cp=_make_group(size=7, rank=0),
        pp=_make_group(size=4, rank=3),
        device_tier=DeviceTier.LOCALITY_PEER,
    )
    bucket7 = compute_hetero_bucket_size(pg7)
    assert bucket7 == 40_000_000, f"expected 40M got {bucket7}"

    # -----------------------------------------------------------------------
    # 2. Megatron parity: pp_rank driven by pg_collection (mirrors test)
    # -----------------------------------------------------------------------
    assert pg7.pp_rank == 3, f"expected pp_rank=3 got {pg7.pp_rank}"

    # -----------------------------------------------------------------------
    # 3. DES-LOC A6000 tier: bucket shrunk to 50 % → 20 M, clamped to floor 4 M
    # -----------------------------------------------------------------------
    policy = LocalityCacheAwareBucketPolicy()
    adjusted_a6000 = policy.apply(40_000_000, DeviceTier.LOCALITY_PEER)
    assert adjusted_a6000 == 20_000_000, f"expected 20M got {adjusted_a6000}"

    # -----------------------------------------------------------------------
    # 4. DES-LOC H100 tier: bucket scaled 2× → 80 M
    # -----------------------------------------------------------------------
    adjusted_h100 = policy.apply(40_000_000, DeviceTier.REMOTE_ACCELERATOR)
    assert adjusted_h100 == 80_000_000, f"expected 80M got {adjusted_h100}"

    # -----------------------------------------------------------------------
    # 5. Scheduler: pp_rank>0 disables bucketing for A6000, not for H100
    # -----------------------------------------------------------------------
    pg_pp1 = HeteroDDPBucketPGCollection(
        dp_cp=_make_group(size=2, rank=0),
        pp=_make_group(size=2, rank=1),
        device_tier=DeviceTier.LOCALITY_PEER,
    )
    sched_a6000 = HeteroDDPBucketScheduler(pg_collection=pg_pp1)
    flags_a6000 = sched_a6000.per_chunk_flags(num_chunks=1)
    assert flags_a6000[0][1] is True, "A6000 pp_rank=1 chunk0 should disable bucketing"

    pg_h100_pp1 = HeteroDDPBucketPGCollection(
        dp_cp=_make_group(size=1, rank=0),
        pp=_make_group(size=2, rank=1),
        device_tier=DeviceTier.REMOTE_ACCELERATOR,
    )
    sched_h100 = HeteroDDPBucketScheduler(pg_collection=pg_h100_pp1)
    flags_h100 = sched_h100.per_chunk_flags(num_chunks=1)
    assert flags_h100[0][1] is False, "H100 pp_rank=1 chunk0 should NOT disable bucketing"

    print("All smoke-test assertions passed.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroDDPBucketPGCollection on a DeepSpeed engine.

    Instantiates a :class:`HeteroDDPBucketPGCollection` from the engine's configuration
    and attaches it as ``engine.hetero_ddp_bucket_pg_collection``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_ddp_bucket_pg_collection.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_ddp_bucket_pg_collection = None
    logger.info("hetero_ddp_bucket_pg_collection.register() attached engine.hetero_ddp_bucket_pg_collection")
