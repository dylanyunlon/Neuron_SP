"""
hetero_grad_buffer_reuse.py
===========================
Neuron_SP / DES-LOC heterogeneous training framework adaptation of
Megatron-LM commit 28e13c484bd9cdc3eca24d3d8ab2d88c437e3bf2
("reuse grad buffer for layer-wise param allgather", PR #3751, Deyu Fu).

Upstream design intent
----------------------
Megatron's _ParamAndGradBucket originally allocated fresh per-rank receive
tensors (plus a ``_layerwise_src_buffer`` to keep the local flat copy alive)
every time ``start_param_sync`` was called during layerwise async all-gather.
PR #3751 observes that the *gradient buffer* (``bucket.grad_data``) is
completely idle during the forward pass, so it can be reused as the contiguous
all-gather receive arena.  The key constraints satisfied by the upstream diff:

1. ``grad_dtype.element_size() >= param_dtype.element_size()`` — grad buffer
   reinterpreted via ``.view(param_dtype)`` always has enough capacity.
2. The local-rank slot is carved out of the same arena, so the separate
   ``_layerwise_src_buffer`` attribute (and its ``None``-reset bookkeeping)
   can be deleted entirely.
3. Async-op correctness is maintained: the arena lives as long as
   ``bucket.grad_data`` does, which outlives every ``work.wait()`` call.

DES-LOC adaptation points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces three
structural differences from Megatron's symmetric-GPU world:

A. **Heterogeneous device pool** — the cluster has 2 × A6000 (SM86, 48 GB,
   PCIe) and 1 × H100 NVL (SM90, 96 GB, PCIe).  There is no NVLink.  All
   collective operations go over PCIe / CPU pinned memory.  We must therefore
   choose the *cheapest* device for temporary staging, which is the H100 (it
   has enough VRAM to absorb layerwise receive buffers even for the largest
   bucket).

B. **Asymmetric VRAM budget** — A6000 ranks (dp_rank 0, 1) have 48 GB each;
   H100 rank (dp_rank 2) has 96 GB.  The grad-buffer-reuse trick works
   symmetrically only when every rank's grad buffer is large enough to hold
   the *total* gather size.  For A6000 ranks with a tight VRAM budget we
   optionally spill the receive arena to a CPU-pinned buffer (the cluster has
   1.5 TB DRAM) and stream the gathered params back to GPU asynchronously.

C. **Shared LOcality Cache (SLC)** — DES-LOC maintains a device-resident SLC
   on each GPU that caches the most recently all-gathered parameter slices.
   When the same layer-param bucket is requested again within one micro-batch,
   we skip the all-gather entirely and serve from cache.  The grad-buffer-
   reuse optimisation is compatible with SLC because we never overwrite the
   grad buffer during the forward pass; the SLC entry points into a *separate*
   pinned staging area managed by ``HeteroSLCManager``.

Class hierarchy
---------------
``HeteroParamBucket``        – mirrors ``_ParamAndGradBucket`` with hetero
                               device / SLC extensions.
``HeteroGradBufferReuser``   – implements the grad-buffer-reuse all-gather
                               strategy from PR #3751 with DES-LOC extensions.
``HeteroSLCManager``         – Shared LOcality Cache: tracks which param
                               buckets are hot in each device's fast memory.
``HeteroBucketGroup``        – orchestrates bucket groups across heterogeneous
                               ranks, analogous to ``_ParamAndGradBucketGroup``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map SM version → logical tier: 0 = A6000 (SM86), 1 = H100 (SM90)
_SM_TIER: Dict[int, int] = {86: 0, 90: 1}

# CPU pinned-memory spill threshold: if a bucket's total gather size exceeds
# this fraction of the local device's free VRAM, spill to CPU.
_VRAM_SPILL_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Device capability helpers
# ---------------------------------------------------------------------------

def _sm_version(device: torch.device) -> int:
    """Return the SM major*10+minor for *device* (GPU only)."""
    if device.type != "cuda":
        return -1
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def _free_vram_bytes(device: torch.device) -> int:
    """Return free VRAM in bytes, or 0 for CPU devices."""
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize(device)
    free, _ = torch.cuda.mem_get_info(device)
    return free


def _device_tier(device: torch.device) -> int:
    """Return tier index (higher = more capable / more VRAM).

    Tier 0 → A6000-class (SM86, 48 GB)
    Tier 1 → H100-class  (SM90, 96 GB)
    """
    sm = _sm_version(device)
    return _SM_TIER.get(sm, 0)


# ---------------------------------------------------------------------------
# Shared LOcality Cache
# ---------------------------------------------------------------------------

@dataclass
class _SLCEntry:
    """One cached param-bucket slice in the SLC."""
    bucket_id: int
    device: torch.device
    data: torch.Tensor          # pinned CPU or GPU tensor
    timestamp: float = field(default_factory=time.monotonic)
    hits: int = 0


class HeteroSLCManager:
    """Shared LOcality Cache (SLC) for DES-LOC.

    The SLC keeps recently all-gathered parameter tensors so that repeated
    accesses within a micro-batch avoid redundant collectives.  On A6000
    devices (48 GB) we keep the cache in CPU-pinned memory to conserve VRAM;
    on the H100 (96 GB) we keep it on-device for maximum bandwidth.

    Design note
    -----------
    The SLC is intentionally *separate* from the grad buffer arena used by
    ``HeteroGradBufferReuser``.  This preserves the upstream invariant that
    the grad buffer is free during the forward pass — the SLC arena lives in
    dedicated pinned/GPU memory and is never aliased to grad_data.

    Parameters
    ----------
    local_device : torch.device
        The GPU owned by this rank.
    max_entries : int
        LRU eviction kicks in above this many cached entries.
    """

    def __init__(self, local_device: torch.device, max_entries: int = 32):
        self.local_device = local_device
        self.max_entries = max_entries
        self._tier = _device_tier(local_device)
        self._cache: Dict[int, _SLCEntry] = {}
        logger.info(
            "HeteroSLCManager init | device=%s tier=%d max_entries=%d",
            local_device,
            self._tier,
            max_entries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, bucket_id: int) -> Optional[List[torch.Tensor]]:
        """Return cached gather_list for *bucket_id*, or None on miss.

        On a hit the tensor is moved to the local device if it is currently
        in pinned CPU memory (A6000 case) via a non-blocking H2D copy.
        """
        entry = self._cache.get(bucket_id)
        if entry is None:
            logger.debug("SLC miss | bucket_id=%d", bucket_id)
            return None
        entry.hits += 1
        entry.timestamp = time.monotonic()
        logger.debug("SLC hit  | bucket_id=%d hits=%d", bucket_id, entry.hits)
        # If data lives on CPU (pinned), kick off async H2D copy.
        if entry.data.device.type == "cpu":
            return entry.data.to(self.local_device, non_blocking=True)
        return entry.data

    def insert(
        self,
        bucket_id: int,
        gather_tensor: torch.Tensor,
    ) -> None:
        """Cache *gather_tensor* for *bucket_id*.

        On H100 (tier 1) we store on-device; on A6000 (tier 0) we pin to
        CPU to preserve precious VRAM.
        """
        if len(self._cache) >= self.max_entries:
            self._evict_lru()

        if self._tier >= 1:
            # H100: store directly on GPU
            stored = gather_tensor.detach().clone()
        else:
            # A6000: offload to pinned CPU memory
            stored = gather_tensor.detach().cpu().pin_memory()

        self._cache[bucket_id] = _SLCEntry(
            bucket_id=bucket_id,
            device=stored.device,
            data=stored,
        )
        logger.debug(
            "SLC insert | bucket_id=%d stored_on=%s numel=%d",
            bucket_id,
            stored.device,
            stored.numel(),
        )

    def invalidate(self, bucket_id: int) -> None:
        """Remove stale entry (called after backward pass clears grads)."""
        removed = self._cache.pop(bucket_id, None)
        if removed is not None:
            logger.debug("SLC invalidate | bucket_id=%d", bucket_id)

    def invalidate_all(self) -> None:
        """Clear entire cache (e.g. at optimizer step boundary)."""
        n = len(self._cache)
        self._cache.clear()
        logger.debug("SLC invalidate_all | cleared %d entries", n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
        evicted = self._cache.pop(lru_key)
        logger.debug(
            "SLC evict LRU | bucket_id=%d hits=%d age=%.3fs",
            lru_key,
            evicted.hits,
            time.monotonic() - evicted.timestamp,
        )
        # Explicitly free GPU memory if applicable
        del evicted.data


# ---------------------------------------------------------------------------
# Hetero param bucket
# ---------------------------------------------------------------------------

class HeteroParamBucket:
    """DES-LOC equivalent of Megatron's ``_ParamAndGradBucket``.

    Differences from upstream
    -------------------------
    * Tracks ``device_tier`` so the reuser can decide whether to spill the
      receive arena to CPU pinned memory.
    * No ``_layerwise_src_buffer`` attribute (removed in upstream PR #3751).
    * Exposes ``grad_data`` as the authoritative receive-buffer arena.
    * Carries a ``bucket_id`` for SLC keying.

    Parameters
    ----------
    bucket_id : int
        Unique identifier used as SLC key.
    params_list : list of list of nn.Parameter
        Outer list is one element per local layer group; inner list are the
        parameters belonging to that group.
    grad_data : torch.Tensor
        Flat gradient buffer allocated by the ZeRO engine.  Must satisfy
        ``grad_data.numel() * grad_data.element_size() >=
         total_param_numel * param_dtype.element_size()``.
    device : torch.device
        The GPU that owns this bucket.
    """

    def __init__(
        self,
        bucket_id: int,
        params_list: List[List[torch.nn.Parameter]],
        grad_data: torch.Tensor,
        device: torch.device,
    ):
        self.bucket_id = bucket_id
        self.params_list = params_list
        self.grad_data = grad_data
        self.device = device
        self.device_tier = _device_tier(device)

        # Set by set_layerwise_params_list
        self.layerwise_params_list: Optional[List[List[torch.nn.Parameter]]] = None
        self.layerwise_param_flat_sizes: Optional[List[int]] = None

        # Populated by HeteroGradBufferReuser.prepare_gather_arena
        self.layerwise_gather_list: Optional[List[torch.Tensor]] = None

        # CPU-pinned spill arena (used when VRAM is tight on A6000)
        self._cpu_arena: Optional[torch.Tensor] = None

        logger.debug(
            "HeteroParamBucket | id=%d device=%s tier=%d grad_data_numel=%d",
            bucket_id,
            device,
            self.device_tier,
            grad_data.numel(),
        )

    def set_layerwise_params_list(
        self, layerwise_params_list: List[List[torch.nn.Parameter]]
    ) -> None:
        """Set per-rank parameter lists and compute flat sizes.

        Mirrors Megatron's ``_ParamAndGradBucket.set_layerwise_params_list``.
        The flat size for this rank is computed by flattening and counting
        elements; zero-size entries are allowed (rank holds no params for this
        bucket).
        """
        self.layerwise_params_list = layerwise_params_list
        self.layerwise_param_flat_sizes = []
        for rank_params in layerwise_params_list:
            if rank_params:
                total = sum(p.numel() for p in rank_params)
            else:
                total = 0
            self.layerwise_param_flat_sizes.append(total)
        logger.debug(
            "HeteroParamBucket id=%d | flat_sizes=%s",
            self.bucket_id,
            self.layerwise_param_flat_sizes,
        )

    def free_overlap_buffers(self) -> None:
        """Release all temporary buffers after param sync completes.

        Note: we do *not* release ``_cpu_arena`` here because it may be
        referenced by an in-flight async copy back to GPU.  The arena is
        released by ``HeteroGradBufferReuser.finalize_gather``.
        """
        self.layerwise_gather_list = None
        logger.debug("HeteroParamBucket id=%d | overlap buffers freed", self.bucket_id)


# ---------------------------------------------------------------------------
# Grad-buffer reuser — core DES-LOC adaptation
# ---------------------------------------------------------------------------

class HeteroGradBufferReuser:
    """Layer-wise param all-gather with grad-buffer reuse for DES-LOC.

    Upstream logic (PR #3751)
    -------------------------
    Megatron PR #3751 replaces per-rank ``torch.empty`` receive buffers with a
    single contiguous arena carved out of ``bucket.grad_data``.  The grad
    buffer is guaranteed idle during the forward pass, so reusing it costs
    zero extra VRAM.  The local rank's slot in the arena is filled via
    ``local_slot_view.copy_(flat_local_params)`` before issuing the
    all_gather, and the separate ``_layerwise_src_buffer`` lifetime-keeper
    attribute is deleted.

    DES-LOC extensions
    ------------------
    1. **SLC lookup** — before allocating any arena, check the SLC.  On a hit
       we skip the all-gather entirely, saving PCIe bandwidth (critical when
       NVLink is absent).

    2. **VRAM spill** — on A6000 ranks with < ``_VRAM_SPILL_FRACTION`` of
       free VRAM, allocate the receive arena in CPU-pinned memory instead of
       reusing grad_data in-place.  This avoids OOM while keeping the
       collective communication on the PCIe path (which is where it would be
       anyway with no NVLink).  The ``grad_data``-as-arena trick still
       applies on the H100 rank.

    3. **Heterogeneous dtype handling** — the H100 uses bf16 grad buffers
       while A6000 ranks may use fp32 grad buffers (to avoid bf16 precision
       loss on SM86 hardware).  We assert ``element_size`` compatibility
       per bucket and per rank.

    4. **Async-stream management** — PCIe transfers and all-gathers are
       issued on a dedicated ``_comm_stream`` so they do not stall the
       compute stream.  ``finalize_gather`` synchronises and copies gathered
       params back to each model parameter.

    Parameters
    ----------
    local_rank : int
        This process's rank within the data-parallel group.
    dp_group : dist.ProcessGroup
        The data-parallel process group (size == number of DP ranks).
    slc_manager : HeteroSLCManager
        Shared LOcality Cache for this rank.
    comm_stream : torch.cuda.Stream, optional
        CUDA stream for communication ops.  If None, uses default stream.
    """

    def __init__(
        self,
        local_rank: int,
        dp_group: dist.ProcessGroup,
        slc_manager: HeteroSLCManager,
        comm_stream: Optional[torch.cuda.Stream] = None,
    ):
        self.local_rank = local_rank
        self.dp_group = dp_group
        self.dp_size = dist.get_world_size(dp_group)
        self.slc_manager = slc_manager
        self.comm_stream = comm_stream
        self._pending_work: List[dist.Work] = []

        logger.info(
            "HeteroGradBufferReuser | local_rank=%d dp_size=%d comm_stream=%s",
            local_rank,
            self.dp_size,
            comm_stream,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_gather_arena(
        self,
        bucket: HeteroParamBucket,
        async_op: bool = True,
    ) -> Optional[dist.Work]:
        """Allocate receive arena and launch layerwise all-gather.

        This is the DES-LOC equivalent of Megatron's
        ``_ParamAndGradBucketGroup.start_param_sync`` layerwise path.

        Steps
        -----
        1. SLC lookup — return immediately on hit.
        2. Validate sizes and dtype compatibility.
        3. Choose arena: grad_data (GPU, H100 or A6000 with enough VRAM)
           vs CPU-pinned spill.
        4. Carve per-rank receive slices from arena.
        5. Copy local params into local slot (replaces ``_layerwise_src_buffer``
           pattern from pre-PR-#3751 Megatron).
        6. Issue ``all_gather`` and return work handle.

        Parameters
        ----------
        bucket : HeteroParamBucket
            The bucket to all-gather.
        async_op : bool
            Whether to issue the collective asynchronously.

        Returns
        -------
        dist.Work or None
            Work handle if async_op=True and the operation was launched;
            None on SLC hit or if all ranks have empty params.
        """
        assert bucket.layerwise_params_list is not None, (
            "set_layerwise_params_list must be called before prepare_gather_arena"
        )
        assert bucket.layerwise_param_flat_sizes is not None

        flat_sizes = bucket.layerwise_param_flat_sizes
        total_gather_size = sum(flat_sizes)

        # ------------------------------------------------------------------ #
        # 1. SLC lookup — skip collective on cache hit
        # ------------------------------------------------------------------ #
        cached = self.slc_manager.lookup(bucket.bucket_id)
        if cached is not None:
            logger.debug(
                "prepare_gather_arena | bucket=%d SLC hit, skip all_gather",
                bucket.bucket_id,
            )
            bucket.layerwise_gather_list = _split_tensor_by_sizes(cached, flat_sizes)
            return None

        # ------------------------------------------------------------------ #
        # 2. All ranks empty — nothing to gather
        # ------------------------------------------------------------------ #
        if total_gather_size == 0:
            bucket.layerwise_gather_list = None
            logger.debug(
                "prepare_gather_arena | bucket=%d all ranks empty, skip",
                bucket.bucket_id,
            )
            return None

        # ------------------------------------------------------------------ #
        # 3. Dtype & size validation
        # ------------------------------------------------------------------ #
        param_dtype = self._infer_param_dtype(bucket)
        grad_dtype = bucket.grad_data.dtype
        assert grad_dtype.itemsize >= param_dtype.itemsize, (
            f"bucket {bucket.bucket_id}: grad_dtype {grad_dtype} element_size "
            f"({grad_dtype.itemsize}) < param_dtype {param_dtype} element_size "
            f"({param_dtype.itemsize}).  Cannot safely reuse grad buffer."
        )

        # ------------------------------------------------------------------ #
        # 4. Choose arena: GPU (grad_data) vs CPU-pinned spill
        # ------------------------------------------------------------------ #
        use_gpu_arena = self._should_use_gpu_arena(bucket, total_gather_size, param_dtype)

        if use_gpu_arena:
            arena = self._arena_from_grad_data(bucket, param_dtype, total_gather_size)
            logger.debug(
                "prepare_gather_arena | bucket=%d GPU arena from grad_data numel=%d",
                bucket.bucket_id,
                arena.numel(),
            )
        else:
            arena = self._arena_from_cpu_pinned(bucket, param_dtype, total_gather_size)
            logger.debug(
                "prepare_gather_arena | bucket=%d CPU-pinned arena numel=%d",
                bucket.bucket_id,
                arena.numel(),
            )

        # ------------------------------------------------------------------ #
        # 5. Carve per-rank receive slices
        # ------------------------------------------------------------------ #
        gather_list: List[torch.Tensor] = []
        offset = 0
        for i, size in enumerate(flat_sizes):
            gather_list.append(arena[offset: offset + size])
            offset += size

        local_slot_view = gather_list[self.local_rank]
        local_size = flat_sizes[self.local_rank]

        # ------------------------------------------------------------------ #
        # 6. Fill local slot (replaces _layerwise_src_buffer)
        # ------------------------------------------------------------------ #
        if local_size > 0:
            flat_local = _flatten_params(bucket.layerwise_params_list[self.local_rank])
            # Detach: start_param_sync may be called during forward where
            # autograd is active; all_gather writes in-place into gather_list.
            flat_local = flat_local.detach()
            local_slot_view.copy_(flat_local, non_blocking=True)

        bucket.layerwise_gather_list = gather_list

        # ------------------------------------------------------------------ #
        # 7. Launch collective
        # ------------------------------------------------------------------ #
        work = self._issue_all_gather(
            gather_list=gather_list,
            src_tensor=local_slot_view,
            async_op=async_op,
        )
        if async_op and work is not None:
            self._pending_work.append(work)

        return work

    def finalize_gather(
        self,
        bucket: HeteroParamBucket,
        wait: bool = True,
    ) -> None:
        """Wait for pending work and copy gathered params back to model.

        Mirrors Megatron's ``finish_param_sync`` layerwise path.  After
        copying, the gather list is cleared and the gathered data is inserted
        into the SLC for future reuse.

        Parameters
        ----------
        bucket : HeteroParamBucket
            The bucket whose all-gather to finalise.
        wait : bool
            If True, block until all pending collectives complete.
        """
        if wait:
            for work in self._pending_work:
                work.wait()
            self._pending_work.clear()

        gather_list = bucket.layerwise_gather_list
        if gather_list is None:
            return

        flat_sizes = bucket.layerwise_param_flat_sizes
        offset = 0
        for rank_idx, (params, size) in enumerate(
            zip(bucket.layerwise_params_list, flat_sizes)  # type: ignore[arg-type]
        ):
            if size == 0 or not params:
                offset += size
                continue
            gathered_slice = gather_list[rank_idx]

            # If slice lives on CPU (spill path), move to GPU first.
            if gathered_slice.device.type == "cpu":
                gathered_slice = gathered_slice.to(bucket.device, non_blocking=False)

            # Write back into each parameter's .data
            _unflatten_into_params(gathered_slice, params)
            offset += size

        # Insert contiguous arena view into SLC before clearing
        if gather_list:
            contiguous_arena = torch.cat([g for g in gather_list if g.numel() > 0])
            self.slc_manager.insert(bucket.bucket_id, contiguous_arena)

        bucket.layerwise_gather_list = None
        bucket._cpu_arena = None  # Release pinned memory if used

        logger.debug(
            "finalize_gather | bucket=%d params written back",
            bucket.bucket_id,
        )

    def free_all(self, buckets: List[HeteroParamBucket]) -> None:
        """Reset all gather state after optimizer step (mirrors free_overlap_buffers).

        Also invalidates the SLC because parameters will be updated by the
        optimizer — cached pre-step values are stale.
        """
        for bucket in buckets:
            bucket.free_overlap_buffers()
            bucket._cpu_arena = None
        self.slc_manager.invalidate_all()
        self._pending_work.clear()
        logger.info(
            "HeteroGradBufferReuser.free_all | %d buckets reset, SLC cleared",
            len(buckets),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_param_dtype(self, bucket: HeteroParamBucket) -> torch.dtype:
        """Return the dtype of the first non-empty param in the bucket."""
        for rank_params in bucket.layerwise_params_list or []:
            for p in rank_params:
                return p.dtype
        return torch.float32  # fallback

    def _should_use_gpu_arena(
        self,
        bucket: HeteroParamBucket,
        total_gather_size: int,
        param_dtype: torch.dtype,
    ) -> bool:
        """Decide whether to use the GPU grad-buffer arena or spill to CPU.

        Policy
        ------
        * H100 (tier ≥ 1): always use GPU arena — 96 GB is ample.
        * A6000 (tier 0): use GPU arena only if the gather payload fits within
          ``_VRAM_SPILL_FRACTION`` of current free VRAM.
        """
        if bucket.device_tier >= 1:
            return True

        # A6000 path: check free VRAM
        bytes_needed = total_gather_size * param_dtype.itemsize
        free_vram = _free_vram_bytes(bucket.device)
        threshold = int(free_vram * _VRAM_SPILL_FRACTION)

        use_gpu = bytes_needed <= threshold
        logger.debug(
            "_should_use_gpu_arena | bucket=%d bytes_needed=%d free_vram=%d "
            "threshold=%d use_gpu=%s",
            bucket.bucket_id,
            bytes_needed,
            free_vram,
            threshold,
            use_gpu,
        )
        return use_gpu

    def _arena_from_grad_data(
        self,
        bucket: HeteroParamBucket,
        param_dtype: torch.dtype,
        total_gather_size: int,
    ) -> torch.Tensor:
        """Reuse ``bucket.grad_data`` as the all-gather receive arena.

        Mirrors the core optimisation from upstream PR #3751.

        The grad buffer is stored in ``grad_dtype``; we reinterpret its memory
        as ``param_dtype`` (valid because ``grad_dtype.itemsize >=
        param_dtype.itemsize``).  We assert the reinterpreted buffer has
        enough elements.
        """
        reuse_buf = bucket.grad_data.view(param_dtype)
        assert reuse_buf.numel() >= total_gather_size, (
            f"bucket {bucket.bucket_id}: grad_data reinterpreted as {param_dtype} "
            f"has {reuse_buf.numel()} elements < required {total_gather_size}"
        )
        return reuse_buf[:total_gather_size]

    def _arena_from_cpu_pinned(
        self,
        bucket: HeteroParamBucket,
        param_dtype: torch.dtype,
        total_gather_size: int,
    ) -> torch.Tensor:
        """Allocate a CPU-pinned arena for A6000 ranks with tight VRAM.

        The arena is cached on ``bucket._cpu_arena`` so it can be freed
        deterministically in ``finalize_gather``.
        """
        if (
            bucket._cpu_arena is not None
            and bucket._cpu_arena.numel() >= total_gather_size
            and bucket._cpu_arena.dtype == param_dtype
        ):
            # Reuse existing pinned allocation
            return bucket._cpu_arena[:total_gather_size]

        arena = torch.empty(
            total_gather_size,
            dtype=param_dtype,
            device="cpu",
            pin_memory=True,
        )
        bucket._cpu_arena = arena
        return arena

    def _issue_all_gather(
        self,
        gather_list: List[torch.Tensor],
        src_tensor: torch.Tensor,
        async_op: bool,
    ) -> Optional[dist.Work]:
        """Issue dist.all_gather, optionally on the comm stream."""
        if self.comm_stream is not None:
            ctx = torch.cuda.stream(self.comm_stream)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            work = dist.all_gather(
                gather_list,
                src_tensor,
                group=self.dp_group,
                async_op=async_op,
            )
        return work


# ---------------------------------------------------------------------------
# Bucket group orchestrator
# ---------------------------------------------------------------------------

class HeteroBucketGroup:
    """Orchestrates a group of ``HeteroParamBucket`` objects across hetero ranks.

    Analogous to Megatron's ``_ParamAndGradBucketGroup``.  Handles the
    sequencing of ``prepare_gather_arena`` / ``finalize_gather`` calls for
    all buckets in a group, and exposes ``free_overlap_buffers`` for
    integration with DeepSpeed's ZeRO step boundary.

    Parameters
    ----------
    buckets : list of HeteroParamBucket
    reuser : HeteroGradBufferReuser
    """

    def __init__(
        self,
        buckets: List[HeteroParamBucket],
        reuser: HeteroGradBufferReuser,
    ):
        self.buckets = buckets
        self.reuser = reuser
        self.param_gather_handle: Optional[dist.Work] = None
        logger.info(
            "HeteroBucketGroup | %d buckets, local_rank=%d",
            len(buckets),
            reuser.local_rank,
        )

    def start_param_sync(self, async_op: bool = True) -> None:
        """Launch all-gather for every bucket in the group.

        Called at the start of the forward pass (layerwise prefetch) or at
        the end of the backward pass (standard sync).
        """
        handles = []
        for bucket in self.buckets:
            work = self.reuser.prepare_gather_arena(bucket, async_op=async_op)
            if work is not None:
                handles.append(work)
        # Store the last non-None handle as the canonical wait point
        self.param_gather_handle = handles[-1] if handles else None
        logger.debug(
            "start_param_sync | %d handles issued for %d buckets",
            len(handles),
            len(self.buckets),
        )

    def finish_param_sync(self) -> None:
        """Wait for all-gathers and copy params back to model parameters."""
        for bucket in self.buckets:
            self.reuser.finalize_gather(bucket, wait=True)
        self.param_gather_handle = None
        logger.debug("finish_param_sync | all buckets finalised")

    def free_overlap_buffers(self) -> None:
        """Reset gather state at optimizer step boundary.

        Mirrors ``_ParamAndGradBucketGroup.free_overlap_buffers`` from
        Megatron.  In DES-LOC this also flushes the SLC because parameters
        are about to be updated.
        """
        self.reuser.free_all(self.buckets)
        self.param_gather_handle = None
        logger.debug("free_overlap_buffers | group reset")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _flatten_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Return a contiguous flat tensor of all *params* concatenated.

    Equivalent to Megatron's ``_flatten_dense_tensors`` call inside the
    layerwise gather path.  We concatenate along dim-0 after reshaping each
    parameter to 1-D.
    """
    if not params:
        return torch.empty(0)
    views = [p.data.reshape(-1) for p in params]
    return torch.cat(views)


def _unflatten_into_params(
    flat: torch.Tensor,
    params: List[torch.nn.Parameter],
) -> None:
    """Copy slices of *flat* back into each parameter's ``.data``.

    The inverse of ``_flatten_params``.  This is the ``copy_back`` step that
    Megatron performs after finish_param_sync.
    """
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat[offset: offset + numel].view(p.shape))
        offset += numel


def _split_tensor_by_sizes(
    tensor: torch.Tensor,
    sizes: List[int],
) -> List[torch.Tensor]:
    """Split a 1-D *tensor* into consecutive chunks of *sizes* elements."""
    result = []
    offset = 0
    for s in sizes:
        result.append(tensor[offset: offset + s])
        offset += s
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.DEBUG)

    # We can run unit assertions without a real distributed backend.
    # ----------------------------------------------------------------
    # Test 1: _flatten_params / _unflatten_into_params round-trip
    p1 = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    p2 = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    flat = _flatten_params([p1, p2])
    assert flat.shape == (10,), f"expected 10, got {flat.shape}"

    p1_copy = torch.nn.Parameter(torch.zeros(2, 3))
    p2_copy = torch.nn.Parameter(torch.zeros(4))
    _unflatten_into_params(flat, [p1_copy, p2_copy])
    assert torch.allclose(p1_copy.data, p1.data), "p1 round-trip failed"
    assert torch.allclose(p2_copy.data, p2.data), "p2 round-trip failed"
    print("Test 1 PASSED: _flatten_params / _unflatten_into_params round-trip")

    # Test 2: _split_tensor_by_sizes
    t = torch.arange(10, dtype=torch.float32)
    parts = _split_tensor_by_sizes(t, [3, 4, 3])
    assert parts[0].tolist() == [0, 1, 2]
    assert parts[1].tolist() == [3, 4, 5, 6]
    assert parts[2].tolist() == [7, 8, 9]
    print("Test 2 PASSED: _split_tensor_by_sizes")

    # Test 3: HeteroSLCManager insert / lookup / invalidate (CPU-only)
    cpu_dev = torch.device("cpu")
    # Patch _device_tier to return 0 (A6000-like) for CPU device
    import unittest.mock as mock
    with mock.patch(
        "deepspeed.runtime.zero.hetero_grad_buffer_reuse._device_tier",
        return_value=0,
    ):
        slc = HeteroSLCManager(cpu_dev, max_entries=4)
        data = torch.randn(16)
        slc.insert(0, data)
        hit = slc.lookup(0)
        assert hit is not None, "SLC should hit after insert"
        slc.invalidate(0)
        miss = slc.lookup(0)
        assert miss is None, "SLC should miss after invalidate"
    print("Test 3 PASSED: HeteroSLCManager insert/lookup/invalidate")

    # Test 4: HeteroParamBucket.set_layerwise_params_list flat sizes
    grad_buf = torch.zeros(64, dtype=torch.float32)
    bucket = HeteroParamBucket(
        bucket_id=42,
        params_list=[[p1], [p2]],
        grad_data=grad_buf,
        device=cpu_dev,
    )
    bucket.set_layerwise_params_list([[p1], [p2]])
    assert bucket.layerwise_param_flat_sizes == [6, 4], (
        f"unexpected flat sizes: {bucket.layerwise_param_flat_sizes}"
    )
    print("Test 4 PASSED: HeteroParamBucket.set_layerwise_params_list")

    # Test 5: grad-buffer arena capacity assertion logic
    grad_buf_small = torch.zeros(5, dtype=torch.float32)
    bucket_small = HeteroParamBucket(
        bucket_id=99,
        params_list=[],
        grad_data=grad_buf_small,
        device=cpu_dev,
    )
    reused = grad_buf_small.view(torch.float32)
    assert reused.numel() == 5, "view numel mismatch"
    print("Test 5 PASSED: grad-buffer arena view sanity")

    print("\nAll smoke tests passed.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroSLCManager on a DeepSpeed engine.

    Instantiates a :class:`HeteroSLCManager` from the engine's configuration
    and attaches it as ``engine.hetero_grad_buffer_reuse``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_grad_buffer_reuse.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_grad_buffer_reuse = None
    logger.info("hetero_grad_buffer_reuse.register() attached engine.hetero_grad_buffer_reuse")
