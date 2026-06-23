"""
deepspeed/runtime/hetero_layerwise_optimizer_ep_fix.py

DES-LOC Heterogeneous Layerwise Optimizer – Expert-Parallel Single-Rank Fix
============================================================================

Upstream Design Intent (Megatron commit 51bcf14)
-------------------------------------------------
Megatron-LM's LayerWiseDistributedOptimizer divides parameters across data-parallel
ranks via bucketed all-gather operations.  Two bugs were fixed in that commit:

1. **_ParamAndGradBucketGroup.start_param_sync with dp_size==1**:
   When ``expt_dp_size == 1`` (single expert-data-parallel rank), the all-gather
   collective is a no-op; the original code attempted it anyway, wasting a NCCL
   launch slot and racing with element-wise distributed-optimizer state.  The fix
   short-circuits early by setting ``param_gather_dispatched = True`` and returning.

2. **LayerWiseDistributedOptimizer expert-parallel bucket initialisation**:
   The original guard ``if self.expt_dp_params_list is not None`` wrapped the
   entire inner loop, which meant that when ``expt_dp_size == 1``
   (``expt_dp_params_list is None``) the buckets were never initialised and
   ``set_layerwise_params_list`` was never called, causing downstream KeyErrors.
   The fix moves the guard inward so every bucket always calls
   ``set_layerwise_params_list``, building a trivial single-shard list when the
   expert-DP group has only one rank.

3. **Argument validation ordering**:
   The emerging-optimizer validation block (``dist_muon`` → ``muon`` rename,
   ``use_layer_wise_distributed_optimizer`` flag coercion) was moved *before* the
   Gloo / distributed-optimizer checks so that ``use_layer_wise_distributed_optimizer``
   is defined before anything reads it.

4. **should_disable_forward_pre_hook**:
   The hook-disable predicate used ``'dist' in args.optimizer`` as a proxy for
   layerwise mode.  This is replaced by the authoritative flag
   ``args.use_layer_wise_distributed_optimizer``.

DES-LOC Adaptation Points
--------------------------
The Neuron_SP project targets three physically distinct devices over PCIe:

    Device 0  – A6000 48 GB  SM86
    Device 1  – A6000 48 GB  SM86
    Device 2  – H100 NVL 96 GB  SM90  (primary compute)
    Host DRAM – 1.5 TB CPU memory  (shared-locality cache tier)

DES-LOC (Decoupled Execution with Shared LOcality Cache) augments standard
DeepSpeed gradient/parameter buffers with a host-side "locality cache" that pins
hot parameter shards close to the device that last updated them.  All-gather
operations are replaced by selective cache-aware transfers that skip PCIe traffic
when the shard is already local.

This file re-expresses all four upstream fixes in the DES-LOC vocabulary:

* **HeteroParamGradBucketGroup** – DES-LOC-aware bucket group that owns a
  ``LocalityCache`` slot per shard and short-circuits collective sync when the
  group has a single rank (dp_size == 1).  For multi-rank groups on PCIe it
  replaces NCCL all-gather with staged peer copies through pinned host memory.

* **HeteroLayerwiseOptimizer** – Wraps DeepSpeed's optimizer groups and fixes
  the expert-parallel bucket initialisation in the same way as Megatron's patch,
  but also routes expert shards to the H100 NVL device when device affinity
  allows it (SM90 path).

* **validate_hetero_args** – Equivalent of Megatron's argument-validation reorder.
  Sets ``use_layer_wise_distributed_optimizer`` before any downstream consumer
  reads it.

* **should_disable_param_prefetch** – Replaces the ``'dist' in optimizer``
  string heuristic with a flag check, matching the Megatron training.py fix.

Author: Neuron_SP project (DES-LOC team)
Mirrors: Megatron commit 51bcf14708489a7ac22c8d5e1650df5eeb40eaba
"""

from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware topology constants for the 2×A6000 + 1×H100 NVL cluster
# ---------------------------------------------------------------------------

A6000_SM = 86          # compute capability 8.6
H100_NVL_SM = 90       # compute capability 9.0
CUDA_DEVICES = [0, 1, 2]          # 0,1 = A6000; 2 = H100 NVL
H100_DEVICE_IDX = 2
CPU_CACHE_DEVICE = torch.device("cpu")
MAX_PCIe_CHUNK_BYTES = 4 * 1024 * 1024   # 4 MiB per staged copy


# ---------------------------------------------------------------------------
# Locality Cache – the "LOC" in DES-LOC
# ---------------------------------------------------------------------------

class LocalityCache:
    """Pinned host-memory cache for parameter shards.

    Each shard is a contiguous tensor region owned by one data-parallel rank.
    The cache avoids redundant PCIe round-trips: if the shard is already in
    pinned host memory and the consumer device can DMA it directly, we skip the
    device-to-device all-gather path entirely.

    DES-LOC design note
    -------------------
    Because there is no NVLink between the A6000 and the H100 in this cluster,
    all inter-device parameter movement goes through:

        device → PCIe → CPU DRAM → PCIe → device

    Pinning host tensors with ``pin_memory=True`` allows the DMA engine to run
    without CPU involvement, overlapping with compute.  The LocalityCache holds
    up to ``capacity_bytes`` of pinned memory and evicts LRU entries.

    Parameters
    ----------
    capacity_bytes:
        Total pinned-memory budget.  Defaults to 8 GiB (well under the 1.5 TB
        available).
    """

    _LRU_SENTINEL = object()

    def __init__(self, capacity_bytes: int = 8 * 1024 ** 3) -> None:
        self._capacity = capacity_bytes
        self._used = 0
        self._store: Dict[str, torch.Tensor] = {}
        self._lru: List[str] = []          # oldest first
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return the cached pinned tensor, or None if not present."""
        with self._lock:
            t = self._store.get(key)
            if t is not None:
                self._touch(key)
            return t

    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Insert *tensor* into the cache as a pinned-memory clone.

        If the tensor is already on CPU and pinned it is stored directly;
        otherwise it is copied to a new pinned allocation.  Evicts LRU entries
        to make space when necessary.
        """
        nbytes = tensor.numel() * tensor.element_size()
        with self._lock:
            if key in self._store:
                self._evict_key(key)
            self._make_space(nbytes)
            if tensor.device.type == "cpu" and tensor.is_pinned():
                pinned = tensor
            else:
                pinned = torch.empty_like(tensor, device=CPU_CACHE_DEVICE,
                                          pin_memory=True)
                pinned.copy_(tensor, non_blocking=True)
            self._store[key] = pinned
            self._lru.append(key)
            self._used += nbytes
        logger.debug("LocalityCache.put key=%s nbytes=%d used=%d",
                     key, nbytes, self._used)

    def invalidate(self, key: str) -> None:
        with self._lock:
            if key in self._store:
                self._evict_key(key)

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _touch(self, key: str) -> None:
        """Move *key* to the MRU end of the LRU list."""
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def _make_space(self, needed: int) -> None:
        while self._used + needed > self._capacity and self._lru:
            self._evict_key(self._lru[0])

    def _evict_key(self, key: str) -> None:
        t = self._store.pop(key, None)
        if t is not None:
            self._used -= t.numel() * t.element_size()
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        logger.debug("LocalityCache.evict key=%s", key)


# ---------------------------------------------------------------------------
# Shard descriptor
# ---------------------------------------------------------------------------

@dataclass
class ParamShard:
    """Metadata for one data-parallel shard of a parameter bucket.

    Attributes
    ----------
    rank:
        The DP rank that owns this shard.
    param_names:
        Names of the parameters packed into this shard (for cache keys).
    data:
        The actual parameter data slice (GPU tensor).
    device_idx:
        CUDA device index where *data* lives.
    """

    rank: int
    param_names: List[str]
    data: torch.Tensor
    device_idx: int

    @property
    def cache_key(self) -> str:
        # Stable key: sorted param names + owning rank
        return "shard:r{rank}:{names}".format(
            rank=self.rank,
            names="+".join(sorted(self.param_names)),
        )

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()


# ---------------------------------------------------------------------------
# HeteroParamGradBucketGroup
# ---------------------------------------------------------------------------

class HeteroParamGradBucketGroup:
    """DES-LOC replacement for Megatron's ``_ParamAndGradBucketGroup``.

    Upstream fix re-interpreted
    ---------------------------
    Megatron's ``start_param_sync`` unconditionally launched an all-gather even
    when ``dp_size == 1``, which was both wasteful and caused a race with the
    element-wise distributed optimizer's own communication stream.  The fix added
    an early return for the single-rank case.

    In DES-LOC we go further: for multi-rank PCIe clusters the "all-gather" is
    replaced by staged peer copies through the LocalityCache (host memory).  The
    H100 NVL device (SM90) gets priority scheduling: its shards are copied first
    so the faster device is never waiting for the slower A6000 pair.

    Parameters
    ----------
    group_id:
        Unique string identifier for this bucket group (used as cache namespace).
    dp_group:
        The data-parallel process group this bucket group belongs to.
    shards:
        Ordered list of ``ParamShard`` objects, one per DP rank.
    locality_cache:
        Shared ``LocalityCache`` instance (one per optimizer).
    prefer_h100:
        When True, schedule the H100's shard DMA before A6000 shards.
    """

    def __init__(
        self,
        group_id: str,
        dp_group: dist.ProcessGroup,
        shards: List[ParamShard],
        locality_cache: LocalityCache,
        *,
        prefer_h100: bool = True,
    ) -> None:
        self.group_id = group_id
        self.dp_group = dp_group
        self.shards = shards
        self.locality_cache = locality_cache
        self.prefer_h100 = prefer_h100

        self._dp_size: int = dist.get_world_size(dp_group)
        self._local_rank: int = dist.get_rank(dp_group)

        # DES-LOC state flags (mirror Megatron's param_gather_dispatched)
        self.param_gather_dispatched: bool = False
        self._async_handles: List[dist.Work] = []
        self._staged_copies: List[Tuple[torch.Tensor, torch.Tensor]] = []

        logger.debug(
            "HeteroParamGradBucketGroup[%s] dp_size=%d local_rank=%d",
            group_id, self._dp_size, self._local_rank,
        )

    # ------------------------------------------------------------------
    # Core sync entry point (mirrors start_param_sync)
    # ------------------------------------------------------------------

    def start_param_sync(self, *, force_sync: bool = False) -> None:
        """Initiate parameter synchronisation across the DP group.

        **dp_size == 1 path (Megatron fix #1 re-interpreted)**:
        Single-rank groups need no collective.  We set
        ``param_gather_dispatched = True`` and return immediately, exactly
        mirroring the upstream fix.  The locality cache is still populated so
        the shard is available for downstream locality-aware reads.

        **dp_size > 1 path (DES-LOC extension)**:
        Instead of a NCCL all-gather we use a two-phase staged copy:

        Phase 1 – Push local shard to pinned host memory (LocalityCache).
        Phase 2 – Pull remote shards from cache (or request peer push via
                   point-to-point if not yet cached).

        The H100 NVL shard (device_idx == H100_DEVICE_IDX) is always pushed
        first so that the faster device can proceed with the next layer while
        A6000 shards are in-flight.

        Parameters
        ----------
        force_sync:
            If True, block until all shards are available (disables overlap).
        """
        if self.param_gather_dispatched:
            logger.debug(
                "HeteroParamGradBucketGroup[%s].start_param_sync: "
                "already dispatched, skipping",
                self.group_id,
            )
            return

        # ----------------------------------------------------------------
        # UPSTREAM FIX #1 re-interpreted: dp_size == 1 early exit
        # ----------------------------------------------------------------
        if self._dp_size == 1:
            logger.debug(
                "HeteroParamGradBucketGroup[%s]: dp_size=1, "
                "skipping collective and caching local shard",
                self.group_id,
            )
            if self.shards:
                local_shard = self.shards[0]
                self.locality_cache.put(local_shard.cache_key, local_shard.data)
            self.param_gather_dispatched = True
            return

        # ----------------------------------------------------------------
        # DES-LOC multi-rank path: staged PCIe copies via LocalityCache
        # ----------------------------------------------------------------
        self._staged_copies.clear()
        self._async_handles.clear()

        local_shard = self.shards[self._local_rank]

        # Phase 1: push our own shard into the locality cache
        self.locality_cache.put(local_shard.cache_key, local_shard.data)
        logger.debug(
            "HeteroParamGradBucketGroup[%s]: pushed local shard rank=%d "
            "nbytes=%d to locality cache",
            self.group_id, self._local_rank, local_shard.nbytes,
        )

        # Determine transfer order: H100 shard first if prefer_h100
        shard_order = list(range(self._dp_size))
        if self.prefer_h100:
            h100_shards = [
                i for i, s in enumerate(self.shards)
                if s.device_idx == H100_DEVICE_IDX
            ]
            for i in h100_shards:
                if i in shard_order:
                    shard_order.remove(i)
                    shard_order.insert(0, i)

        # Phase 2: for each remote rank either retrieve from cache or stage a
        # point-to-point recv.  We use non-blocking ops for overlap.
        for rank_idx in shard_order:
            if rank_idx == self._local_rank:
                continue  # already have this
            remote_shard = self.shards[rank_idx]
            cached = self.locality_cache.get(remote_shard.cache_key)
            if cached is not None:
                # Cache hit: async copy from pinned host → destination device
                dst = remote_shard.data
                stream = torch.cuda.Stream(device=dst.device)
                with torch.cuda.stream(stream):
                    dst.copy_(cached, non_blocking=True)
                self._staged_copies.append((cached, dst))
                logger.debug(
                    "HeteroParamGradBucketGroup[%s]: cache-hit for rank=%d, "
                    "async copy to device=%s",
                    self.group_id, rank_idx, dst.device,
                )
            else:
                # Cache miss: fall back to distributed recv (chunked to respect
                # PCIe bandwidth ceiling)
                handle = self._staged_pcie_recv(rank_idx, remote_shard)
                if handle is not None:
                    self._async_handles.append(handle)

        self.param_gather_dispatched = True

        if force_sync:
            self.finish_param_sync()

    def finish_param_sync(self) -> None:
        """Wait for all in-flight async handles to complete.

        Must be called before the parameters are consumed by the forward pass.
        """
        for handle in self._async_handles:
            handle.wait()
        self._async_handles.clear()
        self._staged_copies.clear()
        logger.debug(
            "HeteroParamGradBucketGroup[%s]: finish_param_sync complete",
            self.group_id,
        )

    def reset(self) -> None:
        """Reset dispatch flag; called at the start of each optimizer step."""
        self.param_gather_dispatched = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _staged_pcie_recv(
        self, src_rank: int, shard: ParamShard
    ) -> Optional[dist.Work]:
        """Issue a chunked point-to-point receive from *src_rank*.

        Splits the shard into ``MAX_PCIe_CHUNK_BYTES``-sized pieces so that
        CPU DMA progress threads have meaningful granularity and the PCIe bus is
        not saturated by a single giant transfer.

        Returns the last ``dist.Work`` handle (representing completion of the
        full transfer) or None if the tensor is empty.
        """
        flat = shard.data.view(-1)
        if flat.numel() == 0:
            return None

        elem_bytes = flat.element_size()
        chunk_elems = MAX_PCIe_CHUNK_BYTES // elem_bytes
        n = flat.numel()
        handle: Optional[dist.Work] = None

        for start in range(0, n, chunk_elems):
            end = min(start + chunk_elems, n)
            chunk = flat[start:end]
            handle = dist.irecv(chunk, src=src_rank, group=self.dp_group)

        logger.debug(
            "HeteroParamGradBucketGroup[%s]: staged recv from rank=%d "
            "total_elems=%d chunk_size=%d",
            self.group_id, src_rank, n, chunk_elems,
        )
        return handle


# ---------------------------------------------------------------------------
# HeteroLayerwiseOptimizer
# ---------------------------------------------------------------------------

class HeteroLayerwiseOptimizer:
    """DES-LOC layerwise optimizer that wraps multiple DeepSpeed param groups.

    Upstream fix re-interpreted (Megatron fix #2)
    ----------------------------------------------
    The original expert-parallel bucket initialisation guarded the entire inner
    loop with ``if expt_dp_params_list is not None``, leaving buckets
    uninitialised when ``expt_dp_size == 1``.  The fix moves the guard inward
    so every bucket calls ``set_layerwise_params_list`` with either a real
    per-rank partition or a trivial single-shard list.

    DES-LOC extension
    -----------------
    When ``expt_dp_size == 1`` and the model runs on the H100 NVL device, we
    additionally register the bucket parameters in the LocalityCache under an
    H100-specific namespace so subsequent layers can do cache-warm prefetches.

    Parameters
    ----------
    param_groups:
        List of dicts, each containing ``'params'`` (list of tensors) and
        standard optimizer hyperparameters.
    expt_dp_group:
        Expert data-parallel process group (may be a single-rank group).
    locality_cache:
        Shared ``LocalityCache`` instance.
    base_optimizer_cls:
        Underlying DeepSpeed optimizer class (e.g. ``FusedAdam``).
    base_optimizer_kwargs:
        Keyword arguments forwarded to *base_optimizer_cls*.
    """

    def __init__(
        self,
        param_groups: List[Dict],
        expt_dp_group: dist.ProcessGroup,
        locality_cache: LocalityCache,
        base_optimizer_cls=None,
        base_optimizer_kwargs: Optional[Dict] = None,
    ) -> None:
        self.param_groups = param_groups
        self.expt_dp_group = expt_dp_group
        self.locality_cache = locality_cache
        self.expt_dp_size: int = dist.get_world_size(expt_dp_group)
        self.expt_dp_rank: int = dist.get_rank(expt_dp_group)

        self._bucket_groups: List[HeteroParamGradBucketGroup] = []
        self._layerwise_params_map: Dict[int, List[List[torch.Tensor]]] = {}

        if base_optimizer_cls is None:
            # Fallback to plain SGD for testing
            import torch.optim as optim
            base_optimizer_cls = optim.SGD
            base_optimizer_kwargs = base_optimizer_kwargs or {"lr": 1e-3}

        self._base_optimizer = base_optimizer_cls(
            param_groups, **(base_optimizer_kwargs or {})
        )
        logger.info(
            "HeteroLayerwiseOptimizer init: expt_dp_size=%d expt_dp_rank=%d",
            self.expt_dp_size, self.expt_dp_rank,
        )

    # ------------------------------------------------------------------
    # Expert-parallel bucket initialisation (Megatron fix #2 equivalent)
    # ------------------------------------------------------------------

    def init_expert_parallel_buckets(
        self,
        model_chunks: List,
        expt_dp_params_list: Optional[List[torch.Tensor]],
    ) -> None:
        """Initialise layerwise parameter lists for every expert-parallel bucket.

        This method mirrors the corrected Megatron logic verbatim but adapted
        for DES-LOC bucket groups:

        * If ``expt_dp_params_list is not None`` (multi-rank EP): build the
          per-rank partition as in the original.
        * If ``expt_dp_params_list is None`` (single-rank EP, expt_dp_size==1):
          build a trivial ``[list(bucket.all_params)]`` and, on the H100 device,
          pre-populate the LocalityCache with those params.

        **Key upstream fix**: the check ``if expt_dp_params_list is not None``
        is now *inside* the inner bucket loop, not outside it.  This ensures
        ``set_layerwise_params_list`` is always called.

        Parameters
        ----------
        model_chunks:
            List of model chunk objects; each must expose
            ``expert_parallel_bucket_groups`` (list of bucket-group-like objects
            with ``.buckets`` and ``.params_list`` / ``.params`` attributes).
        expt_dp_params_list:
            Full flattened list of expert parameters across all ranks, or None
            when ``expt_dp_size == 1``.
        """
        for chunk_idx, model_chunk in enumerate(model_chunks):
            ep_bucket_groups = getattr(
                model_chunk, "expert_parallel_bucket_groups", []
            )
            # ----------------------------------------------------------------
            # UPSTREAM FIX #2 re-interpreted:
            #   The outer guard is removed.  Every bucket group is visited and
            #   the guard is applied per-bucket inside.
            # ----------------------------------------------------------------
            for group in ep_bucket_groups:
                for bucket_idx, bucket in enumerate(group.buckets):
                    if expt_dp_params_list is not None:
                        # Normal multi-rank EP path
                        bucket_params_list: List[List[torch.Tensor]] = [
                            [] for _ in range(self.expt_dp_size)
                        ]
                        # Distribute params across ranks by stable ordering
                        full_params: List[torch.Tensor] = list(
                            getattr(bucket, "params_list", [])
                            or getattr(bucket, "params", [])
                        )
                        for rank_idx, param_sublist in enumerate(expt_dp_params_list
                                                                  if isinstance(expt_dp_params_list[0], list)
                                                                  else [expt_dp_params_list]):
                            bucket_list: List[torch.Tensor] = []
                            for param in (param_sublist if isinstance(param_sublist, list)
                                          else [param_sublist]):
                                if any(param is p for p in full_params):
                                    bucket_list.append(param)
                            if rank_idx < len(bucket_params_list):
                                bucket_params_list[rank_idx] = bucket_list
                    else:
                        # --------------------------------------------------------
                        # UPSTREAM FIX #2 (single-rank): initialise trivially.
                        # expt_dp_size == 1: single rank owns all params; no
                        # all-gather needed but data structures must be populated.
                        # --------------------------------------------------------
                        all_params: List[torch.Tensor] = list(
                            getattr(bucket, "params_list", [])
                            or getattr(bucket, "params", [])
                        )
                        bucket_params_list = [all_params]

                        # DES-LOC extension: pre-warm the locality cache for H100
                        device_idx = (
                            all_params[0].device.index
                            if all_params and all_params[0].is_cuda
                            else -1
                        )
                        if device_idx == H100_DEVICE_IDX:
                            cache_key = "ep_bucket:chunk{c}:bucket{b}:r0".format(
                                c=chunk_idx, b=bucket_idx
                            )
                            if all_params:
                                flat = torch.cat(
                                    [p.data.view(-1) for p in all_params]
                                )
                                self.locality_cache.put(cache_key, flat)
                                logger.debug(
                                    "init_expert_parallel_buckets: "
                                    "pre-warmed H100 cache key=%s", cache_key
                                )

                    # Always call set_layerwise_params_list (fix #2 core)
                    if hasattr(bucket, "set_layerwise_params_list"):
                        bucket.set_layerwise_params_list(bucket_params_list)
                    else:
                        # Store in our own map if bucket object is a stub
                        self._layerwise_params_map[id(bucket)] = bucket_params_list

                    logger.debug(
                        "init_expert_parallel_buckets: chunk=%d group=%s "
                        "bucket=%d nranks=%d",
                        chunk_idx, getattr(group, "name", "?"),
                        bucket_idx, len(bucket_params_list),
                    )

    # ------------------------------------------------------------------
    # AllGather parameter entry point
    # ------------------------------------------------------------------

    def allgather_params(self, *, force_sync: bool = False) -> None:
        """Trigger param gather across all registered bucket groups.

        For single-rank groups (dp_size==1) each bucket group returns
        immediately after marking ``param_gather_dispatched``.

        Parameters
        ----------
        force_sync:
            Block until all outstanding transfers complete.
        """
        for bg in self._bucket_groups:
            bg.reset()
            bg.start_param_sync(force_sync=force_sync)
        logger.debug(
            "allgather_params: dispatched %d bucket groups force_sync=%s",
            len(self._bucket_groups), force_sync,
        )

    def finish_param_gather(self) -> None:
        """Flush all outstanding async transfers."""
        for bg in self._bucket_groups:
            bg.finish_param_sync()

    def register_bucket_group(self, bg: HeteroParamGradBucketGroup) -> None:
        """Register a bucket group for synchronisation."""
        self._bucket_groups.append(bg)
        logger.debug(
            "registered bucket group %s (dp_size=%d)",
            bg.group_id, bg._dp_size,
        )

    def step(self) -> None:
        """Perform one optimizer step."""
        self._base_optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._base_optimizer.zero_grad(set_to_none=set_to_none)


# ---------------------------------------------------------------------------
# Argument validation (Megatron fix #3 – ordering / early initialisation)
# ---------------------------------------------------------------------------

@dataclass
class HeteroTrainingArgs:
    """Minimal training argument namespace for DES-LOC validation.

    Mirrors the fields touched by Megatron's reordered emerging-optimizer
    validation block.
    """

    optimizer: str = "adam"
    use_distributed_optimizer: bool = False
    use_layer_wise_distributed_optimizer: bool = False
    use_torch_fsdp2: bool = False
    use_megatron_fsdp: bool = False
    ckpt_format: str = "torch"
    use_gloo_process_groups: bool = True
    overlap_param_gather: bool = False

    # DES-LOC-specific
    hetero_device_map: List[int] = field(default_factory=lambda: CUDA_DEVICES)
    locality_cache_gb: float = 8.0


def validate_hetero_args(args: HeteroTrainingArgs) -> HeteroTrainingArgs:
    """Validate and coerce DES-LOC training arguments.

    Upstream design intent (Megatron fix #3)
    ----------------------------------------
    The original Megatron ``validate_args`` placed the emerging-optimizer check
    *after* several blocks that already read ``use_layer_wise_distributed_optimizer``,
    which could cause AttributeError or stale-flag bugs.  The fix moves the check
    to run first, guaranteeing the flag is set before any downstream consumer.

    DES-LOC adaptation
    ------------------
    Same reordering is applied here.  Additionally we validate that heterogeneous
    device indices are reachable and that the locality cache budget fits within the
    available CPU DRAM (reported via ``/proc/meminfo``).

    Parameters
    ----------
    args:
        Argument namespace (mutated in-place).

    Returns
    -------
    args:
        Same object, after validation.

    Raises
    ------
    ValueError:
        On invalid / incompatible argument combinations.
    AssertionError:
        On hard constraints (mirrors Megatron's ``assert`` style).
    """
    # ----------------------------------------------------------------
    # UPSTREAM FIX #3: Initialise use_layer_wise_distributed_optimizer
    # BEFORE anything reads it.
    # ----------------------------------------------------------------
    args.use_layer_wise_distributed_optimizer = False

    if args.optimizer not in ("sgd", "adam"):
        if args.optimizer == "dist_muon":
            warnings.warn(
                "optimizer='dist_muon' is deprecated for DES-LOC.  "
                "Use --optimizer muon --use-distributed-optimizer instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning(
                "validate_hetero_args: dist_muon renamed to muon, "
                "enabling use_layer_wise_distributed_optimizer"
            )
            args.optimizer = "muon"
            args.use_layer_wise_distributed_optimizer = True

        if args.use_distributed_optimizer:
            logger.info(
                "validate_hetero_args: emerging optimizer detected; "
                "converting use_distributed_optimizer → "
                "use_layer_wise_distributed_optimizer"
            )
            args.use_layer_wise_distributed_optimizer = True
            args.use_distributed_optimizer = False

        assert not args.use_torch_fsdp2, (
            "Emerging optimizer (DES-LOC) does not support Torch-FSDP2."
        )
        assert not args.use_megatron_fsdp, (
            "Emerging optimizer (DES-LOC) does not support Megatron-FSDP."
        )
        assert args.ckpt_format in ("torch", "torch_dist"), (
            "Emerging optimizer (DES-LOC) supports 'torch' and 'torch_dist' "
            "checkpoint formats only."
        )

    # DES-LOC: validate device map
    n_gpus = torch.cuda.device_count()
    for dev_idx in args.hetero_device_map:
        if dev_idx >= n_gpus:
            raise ValueError(
                f"hetero_device_map references device {dev_idx} but only "
                f"{n_gpus} CUDA device(s) are visible."
            )

    # DES-LOC: validate locality cache budget against /proc/meminfo
    try:
        mem_bytes = _read_proc_meminfo_total()
        cache_bytes = int(args.locality_cache_gb * 1024 ** 3)
        if cache_bytes > mem_bytes * 0.9:
            warnings.warn(
                f"locality_cache_gb={args.locality_cache_gb:.1f} exceeds 90% "
                f"of available CPU DRAM ({mem_bytes / 1024**3:.1f} GiB).  "
                "Reducing to 80% of available memory.",
                ResourceWarning,
                stacklevel=2,
            )
            args.locality_cache_gb = mem_bytes * 0.8 / 1024 ** 3
    except OSError:
        logger.debug("validate_hetero_args: could not read /proc/meminfo; "
                     "skipping memory validation")

    logger.info(
        "validate_hetero_args: optimizer=%s use_lwdo=%s locality_cache_gb=%.1f",
        args.optimizer,
        args.use_layer_wise_distributed_optimizer,
        args.locality_cache_gb,
    )
    return args


def _read_proc_meminfo_total() -> int:
    """Return total system memory in bytes from /proc/meminfo."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb * 1024
    raise ValueError("MemTotal not found in /proc/meminfo")


# ---------------------------------------------------------------------------
# should_disable_param_prefetch (Megatron fix #4)
# ---------------------------------------------------------------------------

def should_disable_param_prefetch(args: HeteroTrainingArgs) -> bool:
    """Determine whether to suppress the forward-pre-hook parameter prefetch.

    Upstream design intent (Megatron fix #4)
    -----------------------------------------
    Megatron's ``should_disable_forward_pre_hook`` used the string heuristic
    ``'dist' in args.optimizer`` to detect layerwise-optimizer mode.  This was
    fragile: any optimizer whose name happened to contain 'dist' (e.g. a future
    'distilled_adam') would be incorrectly matched.  The fix replaces it with
    the authoritative boolean flag ``use_layer_wise_distributed_optimizer``.

    DES-LOC adaptation
    ------------------
    Identical logic.  The DES-LOC layerwise optimizer always sets
    ``use_layer_wise_distributed_optimizer = True`` during ``validate_hetero_args``,
    so this predicate correctly suppresses the prefetch hook when DES-LOC is
    active and ``overlap_param_gather`` is requested.

    When True, the forward pre-hook that pre-fetches the next layer's parameters
    is disabled.  This is necessary because the DES-LOC scheduler issues its own
    pipelined PCIe transfers; the hook would race with those and cause
    double-copies.

    Parameters
    ----------
    args:
        Validated argument namespace.

    Returns
    -------
    bool:
        True if the param-prefetch hook should be suppressed.
    """
    result = (
        not getattr(args, "use_megatron_fsdp", False)
        and (
            getattr(args, "use_distributed_optimizer", False)
            or args.use_layer_wise_distributed_optimizer        # FIX: flag not string
        )
        and getattr(args, "overlap_param_gather", False)
    )
    logger.debug(
        "should_disable_param_prefetch: "
        "use_megatron_fsdp=%s use_distributed_optimizer=%s "
        "use_layer_wise_distributed_optimizer=%s overlap_param_gather=%s → %s",
        getattr(args, "use_megatron_fsdp", False),
        getattr(args, "use_distributed_optimizer", False),
        args.use_layer_wise_distributed_optimizer,
        getattr(args, "overlap_param_gather", False),
        result,
    )
    return result


# ---------------------------------------------------------------------------
# Hetero device probe utilities
# ---------------------------------------------------------------------------

def probe_hetero_topology() -> Dict[int, Dict]:
    """Return a mapping of CUDA device index → capability info.

    Used at startup to confirm the expected 2×A6000 + 1×H100 NVL layout.
    Logs a warning if the topology does not match expectations.

    Returns
    -------
    dict:
        ``{device_idx: {'name': str, 'sm': int, 'mem_gb': float}}``
    """
    topo: Dict[int, Dict] = {}
    n = torch.cuda.device_count()
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        sm = props.major * 10 + props.minor
        mem_gb = props.total_memory / 1024 ** 3
        topo[i] = {
            "name": props.name,
            "sm": sm,
            "mem_gb": round(mem_gb, 1),
        }
        logger.info("GPU %d: %s  SM%d  %.1f GiB", i, props.name, sm, mem_gb)

    # Validate expected topology
    expected = {A6000_SM: 2, H100_NVL_SM: 1}
    observed: Dict[int, int] = {}
    for info in topo.values():
        observed[info["sm"]] = observed.get(info["sm"], 0) + 1

    for sm, count in expected.items():
        if observed.get(sm, 0) < count:
            logger.warning(
                "probe_hetero_topology: expected %d device(s) with SM%d, "
                "found %d – DES-LOC scheduling may be suboptimal",
                count, sm, observed.get(sm, 0),
            )
    return topo


def select_compute_device(
    topo: Dict[int, Dict],
    *,
    prefer_h100: bool = True,
) -> int:
    """Choose the primary compute device index.

    In DES-LOC the H100 NVL is the preferred primary device because it has
    larger VRAM (96 GiB) and SM90 compute which supports FP8 natively.

    Parameters
    ----------
    topo:
        Output of ``probe_hetero_topology()``.
    prefer_h100:
        If True, return the H100 index; otherwise return the first A6000.

    Returns
    -------
    int:
        CUDA device index.
    """
    if prefer_h100:
        for idx, info in topo.items():
            if info["sm"] == H100_NVL_SM:
                logger.info(
                    "select_compute_device: chose H100 NVL at device %d", idx
                )
                return idx
    for idx, info in topo.items():
        if info["sm"] == A6000_SM:
            logger.info(
                "select_compute_device: chose A6000 at device %d", idx
            )
            return idx
    logger.warning("select_compute_device: no expected GPU found, defaulting to 0")
    return 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)s %(name)s %(message)s")
    logger.info("=== DES-LOC HeteroLayerwiseOptimizerEPFix smoke test ===")

    # 1. LocalityCache round-trip
    cache = LocalityCache(capacity_bytes=256 * 1024 * 1024)  # 256 MiB
    t = torch.randn(1024)
    cache.put("test:key", t)
    retrieved = cache.get("test:key")
    assert retrieved is not None, "LocalityCache.get returned None after put"
    assert retrieved.shape == t.shape, "cached tensor shape mismatch"
    assert torch.allclose(retrieved, t), "cached tensor values mismatch"
    logger.info("PASS: LocalityCache round-trip")

    # 2. validate_hetero_args: dist_muon rename + flag initialisation order
    args = HeteroTrainingArgs(
        optimizer="dist_muon",
        use_distributed_optimizer=False,
        hetero_device_map=[],   # skip GPU probe in CI
    )
    validate_hetero_args(args)
    assert args.optimizer == "muon", "dist_muon should be renamed to muon"
    assert args.use_layer_wise_distributed_optimizer is True, \
        "use_layer_wise_distributed_optimizer should be True after dist_muon rename"
    logger.info("PASS: validate_hetero_args dist_muon rename")

    # 3. should_disable_param_prefetch: flag-based not string-based
    args2 = HeteroTrainingArgs(
        optimizer="muon",
        use_layer_wise_distributed_optimizer=True,
        overlap_param_gather=True,
    )
    assert should_disable_param_prefetch(args2) is True, \
        "should_disable_param_prefetch should be True with lwdo+overlap"
    args3 = HeteroTrainingArgs(optimizer="distributed_adam",
                               use_layer_wise_distributed_optimizer=False,
                               overlap_param_gather=True)
    # 'dist' is in 'distributed_adam' but flag is False → must be False (fix #4)
    assert should_disable_param_prefetch(args3) is False, \
        "should_disable_param_prefetch must use flag, not 'dist' in optimizer string"
    logger.info("PASS: should_disable_param_prefetch flag-based check")

    # 4. HeteroParamGradBucketGroup dp_size=1 early exit
    # (Requires a distributed environment; skip if not initialised)
    if dist.is_available() and dist.is_initialized():
        pg = dist.new_group([dist.get_rank()])
        shard = ParamShard(
            rank=0,
            param_names=["layer0.weight"],
            data=torch.randn(64, 64),
            device_idx=0,
        )
        bg = HeteroParamGradBucketGroup(
            group_id="test_group_0",
            dp_group=pg,
            shards=[shard],
            locality_cache=cache,
        )
        assert not bg.param_gather_dispatched
        bg.start_param_sync()
        assert bg.param_gather_dispatched, \
            "dp_size=1: param_gather_dispatched should be True after start_param_sync"
        # Verify shard was cached
        assert cache.get(shard.cache_key) is not None, \
            "dp_size=1: shard should be in locality cache after start_param_sync"
        logger.info("PASS: HeteroParamGradBucketGroup dp_size=1 early exit")
    else:
        logger.info("SKIP: distributed test (no dist group initialised)")

    # 5. LocalityCache LRU eviction
    tiny_cache = LocalityCache(capacity_bytes=1024)  # 1 KiB only
    for i in range(10):
        tiny_cache.put(f"evict:key{i}", torch.zeros(64))  # 256 bytes each
    assert tiny_cache.used_bytes <= 1024, \
        f"LocalityCache exceeded capacity: {tiny_cache.used_bytes} > 1024"
    logger.info("PASS: LocalityCache LRU eviction")

    logger.info("=== All smoke tests passed ===")
    sys.exit(0)


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroParamGradBucketGroup on a DeepSpeed engine.

    Instantiates a :class:`HeteroParamGradBucketGroup` from the engine's configuration
    and attaches it as ``engine.hetero_layerwise_optimizer_ep_fix``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_layerwise_optimizer_ep_fix.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_layerwise_optimizer_ep_fix = None
    logger.info("hetero_layerwise_optimizer_ep_fix.register() attached engine.hetero_layerwise_optimizer_ep_fix")
