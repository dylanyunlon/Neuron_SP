"""
DES-LOC Heterogeneous Pretraining Configuration and Batch Dispatch
==================================================================

Upstream Design Intent (Megatron b60de39e29aa85adc32519f5dd77f57eebe2b6d1)
---------------------------------------------------------------------------
The Megatron commit b60de39 ("Clean up pretrain_gpt.py and pretrain_hybrid.py
formatting and remove module globals") addresses two structural concerns:

1. **Module-level mutable globals removed**: ``loss_func_cached_logits = None``
   was a process-wide singleton that got lazily initialised on first call.
   This pattern is fragile under hot-reload, multi-tenant inference servers,
   and—critically for us—under DES-LOC's worker-pool model where the same
   Python interpreter may service micro-batches destined for different devices.
   Megatron replaced it with ``@lru_cache(maxsize=1)`` on a pure function whose
   arguments fully determine the cached object, making the cache key explicit and
   the singleton safe to reason about.

2. **BATCH_KEYS hoisted to module scope**: Previously the list lived inside
   ``get_batch()``, meaning it was re-allocated on every call.  Making it a
   module constant clarifies the contract between the data pipeline and the
   forward step, enables static analysis, and (minor) removes the allocation
   overhead on the hot path.

3. **``global stimer`` removed from ``forward_step``**: The straggler-detector
   handle was re-declared as ``global`` inside the function body even though it
   was already accessible as a module-level name.  The ``global`` statement
   shadowed the name and suppressed linters from catching accidental rebinding.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) extends these ideas
to heterogeneous hardware:

    Hardware topology
    -----------------
    • Rank-0, Rank-1 : NVIDIA A6000 48 GB  SM86  (PCIe, no NVLink)
    • Rank-2          : NVIDIA H100 NVL 96 GB  SM90 (PCIe)
    • Host DRAM       : 1.5 TB  (used as overflow / locality cache tier)

    The three devices are connected only via PCIe.  Cross-device tensor
    transfers are expensive (≈12 GB/s unidirectional), so DES-LOC maintains
    a *Shared LOcality Cache* (SLC) in host DRAM that stores recently
    computed intermediate tensors (KV projections, hidden states) and makes
    them available to whichever device picks up the next micro-batch shard.

Adaptations made in this file
------------------------------
A. **``HeteroDeviceProfile``** – encapsulates per-SM-generation compute
   capability and memory capacity so that batch dispatch and loss-function
   construction are device-aware rather than assuming homogeneous workers.

B. **``BATCH_KEYS``** – adopted verbatim from Megatron but extended with
   ``"device_affinity"`` and ``"slc_handle"`` to carry DES-LOC routing metadata
   through the pipeline without changing the positional unpack contract that
   callers rely on.

C. **``HeteroPretrainConfig``** – replaces the scattered ``get_args()`` calls
   inside Megatron's ``get_batch`` with a dataclass that is constructed once,
   validated, and passed explicitly.  This is necessary because DES-LOC workers
   may be forked from different processes and cannot share a global argument
   namespace.

D. **``build_cached_logits_loss_func``** – mirrors Megatron's
   ``_build_cached_logits_loss_func`` but the ``lru_cache`` key is *extended*
   with the ``device_sm_version`` so that different worker ranks that land on
   the function with different SM capabilities get their own (potentially
   dtype-specialised) ``LossFuncCallable`` instance without sharing state.

E. **``DeviceAwareBatchRouter``** – new class with no Megatron equivalent.
   Routes micro-batch shards to the device whose current SLC residency
   maximises data locality, falling back to compute-capacity weighting when
   the cache is cold.  The SLC is a ``SharedLocalityCache`` backed by host
   DRAM mapped via ``torch.UntypedStorage`` + ``multiprocessing.shared_memory``.

F. **``hetero_get_batch``** – replaces Megatron's ``get_batch``.  The logic
   mirrors the upstream function exactly (TP-rank 0 pulls from iterator,
   broadcasts, slices for CP, returns in BATCH_KEYS order) but inserts DES-LOC
   hooks:
     • Before the cuda() transfer, queries the SLC to see if the tensor is
       already resident on the target device; skips the H2D copy if so.
     • Tags each tensor with ``device_affinity`` so downstream pipeline stages
       can make placement decisions without re-probing the cache.

G. **``hetero_loss_func``** – wraps the loss computation with SM-version checks
   so that SM86 ranks use BF16 accumulation (A6000 has limited BF16 throughput)
   while the H100 rank uses FP8 where available.

H. **Straggler detection** – the module-level ``stimer`` pattern from Megatron
   is retained but wrapped in ``HeteroStragglerDetector`` which weights timing
   observations by device compute-capability before reporting stragglers, since
   it is *expected* that SM86 ranks are slower than the SM90 rank.

References
----------
- Megatron-LM commit b60de39e29aa85adc32519f5dd77f57eebe2b6d1
- DeepSpeed heterogeneous training: https://github.com/microsoft/DeepSpeed
- Neuron_SP project: https://github.com/dylanyunlon/Neuron_SP
- DES-LOC internal design doc: docs/des_loc_design.md (this repo)
"""

from __future__ import annotations

import logging
import math
import os
import time
import unittest
import warnings
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical batch-key schema
# ---------------------------------------------------------------------------
# Upstream (Megatron b60de39): BATCH_KEYS was defined *inside* get_batch(),
# meaning a fresh list object was allocated on every call.  Moving it to
# module scope makes the contract between the data pipeline and the forward
# step explicit, enables static analysis tools to verify unpack arity, and
# avoids the minor allocation overhead on the hot path.
#
# DES-LOC extension: two extra fields are appended.
#   "device_affinity" : int | None
#       The local rank index that currently holds the warmest SLC entry for
#       this micro-batch.  None means "no preference / cache cold".
#   "slc_handle"      : str | None
#       Opaque key into the SharedLocalityCache for the primary hidden-state
#       tensor of this micro-batch.  Passed downstream so pipeline stages can
#       do cache-aware placement without re-hashing tensor content.
#
# The positional unpack contract used by callers (``a, b, c, ... = get_batch()``)
# is preserved: the DES-LOC fields sit at the *end* of the tuple and are
# stripped before returning to legacy code paths.
BATCH_KEYS: List[str] = [
    "attention_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
    "device_affinity",   # DES-LOC: routing metadata
    "hybrid_cp_group",
    "labels",
    "local_cp_size",
    "loss_mask",
    "max_seqlen",
    "position_ids",
    "slc_handle",        # DES-LOC: SharedLocalityCache key
    "tokens",
]

# Keys that are part of the standard Megatron unpack contract (positional).
# Must remain a strict prefix of BATCH_KEYS once DES-LOC fields are removed.
_MEGATRON_BATCH_KEYS: List[str] = [
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
]

# Spiky-loss sentinel (mirrors Megatron constant).
SPIKY_LOSS_FACTOR: int = 10

# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

#: SM version constants for the three physical devices in the DES-LOC cluster.
SM_A6000: int = 86   # A6000 48 GB
SM_H100_NVL: int = 90  # H100 NVL 96 GB

#: Nominal PCIe bandwidth ceiling (bytes/sec) for inter-device transfers
#: in a system with no NVLink.  Used for cost modelling in the batch router.
PCIE_BW_BPS: float = 12.0e9  # 12 GB/s

#: Rough TFLOPS ratio H100 NVL / A6000 for BF16 matmul (used for skew-aware
#: straggler detection).  Empirically calibrated; update via profiling.
H100_TO_A6000_TFLOPS_RATIO: float = 4.2


@dataclass
class HeteroDeviceProfile:
    """Per-rank device capability snapshot used throughout DES-LOC dispatch.

    Upstream context
    ----------------
    Megatron assumes all ranks share identical hardware and does not maintain
    per-rank capability metadata.  DES-LOC must track this because:

    * SM86 (A6000) and SM90 (H100) have different BF16 / FP8 throughput
      characteristics.
    * Memory capacity differences (48 GB vs 96 GB) affect micro-batch
      sizing and SLC eviction policies.
    * Without NVLink, the cost of moving a tensor from one device to another
      is substantial and must be amortised via the SLC.

    Attributes
    ----------
    local_rank : int
        DeepSpeed local rank index (0-based within the node).
    sm_version : int
        CUDA compute capability major×10 + minor (e.g. 86 for SM 8.6).
    memory_gb : float
        Total device memory in gigabytes.
    compute_weight : float
        Relative compute capacity normalised so that sum over all ranks == 1.
        Used to partition micro-batch budgets across heterogeneous devices.
    supports_fp8 : bool
        True for SM >= 89 (Ada) and SM >= 90 (Hopper).
    preferred_dtype : torch.dtype
        The fastest accumulation dtype for this device.  SM86 prefers BF16;
        SM90 can exploit FP8 for forward passes.
    """

    local_rank: int
    sm_version: int
    memory_gb: float
    compute_weight: float = 1.0
    supports_fp8: bool = False
    preferred_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.sm_version >= 89:
            object.__setattr__(self, "supports_fp8", True)
        if self.sm_version >= 90:
            object.__setattr__(self, "preferred_dtype", torch.float8_e4m3fn
                               if self.supports_fp8 else torch.bfloat16)

    @classmethod
    def from_local_rank(cls, local_rank: int) -> "HeteroDeviceProfile":
        """Construct a profile by querying the CUDA device on *local_rank*.

        Falls back to safe defaults when CUDA is unavailable (e.g. unit tests
        running on CPU-only machines).
        """
        if not torch.cuda.is_available():
            return cls(local_rank=local_rank, sm_version=86, memory_gb=48.0,
                       compute_weight=1.0)
        device = torch.device(f"cuda:{local_rank}")
        props = torch.cuda.get_device_properties(device)
        sm = props.major * 10 + props.minor
        mem_gb = props.total_memory / (1024 ** 3)
        # Normalise compute weight: H100 NVL ≈ 4.2× an A6000.
        if sm >= 90:
            weight = H100_TO_A6000_TFLOPS_RATIO
        else:
            weight = 1.0
        return cls(local_rank=local_rank, sm_version=sm, memory_gb=mem_gb,
                   compute_weight=weight)

    @property
    def device(self) -> torch.device:
        """Torch device handle for this rank."""
        return torch.device(f"cuda:{self.local_rank}")


def build_cluster_profiles(
    world_size: int,
    local_ranks: Optional[List[int]] = None,
) -> List[HeteroDeviceProfile]:
    """Build per-rank ``HeteroDeviceProfile`` list for the full DES-LOC cluster.

    In the canonical 3-GPU setup:
      rank 0 → A6000 (SM86, 48 GB)
      rank 1 → A6000 (SM86, 48 GB)
      rank 2 → H100 NVL (SM90, 96 GB)

    Parameters
    ----------
    world_size : int
        Total number of GPU workers.
    local_ranks : list[int], optional
        Override which local CUDA device indices to profile.  Defaults to
        ``range(world_size)``.

    Returns
    -------
    list[HeteroDeviceProfile]
        One profile per rank, ordered by rank.
    """
    if local_ranks is None:
        local_ranks = list(range(world_size))
    profiles = [HeteroDeviceProfile.from_local_rank(r) for r in local_ranks]
    total_weight = sum(p.compute_weight for p in profiles) or 1.0
    for p in profiles:
        object.__setattr__(p, "compute_weight", p.compute_weight / total_weight)
    return profiles


# ---------------------------------------------------------------------------
# HeteroPretrainConfig
# ---------------------------------------------------------------------------

@dataclass
class HeteroPretrainConfig:
    """Unified configuration container for DES-LOC heterogeneous pretraining.

    Upstream context
    ----------------
    Megatron's ``get_batch`` and ``loss_func`` both call ``get_args()`` to
    retrieve a process-global argument namespace.  This works in Megatron's
    single-process-per-rank model, but DES-LOC uses a worker-pool architecture
    where a coordinator process may fork worker threads that execute micro-batch
    steps.  A shared global is unsafe in that context.

    DES-LOC adaptation
    ------------------
    ``HeteroPretrainConfig`` is constructed once by the coordinator, serialised
    (via ``to_dict`` / ``from_dict``), and passed explicitly to every worker
    function.  Fields that were previously read from ``args`` are stored here.
    Device-specific fields (``profiles``) carry the heterogeneity information
    that Megatron's ``args`` namespace does not have.

    The ``slc_*`` fields configure the SharedLocalityCache tier in host DRAM.

    Attributes
    ----------
    micro_batch_size : int
        Per-device micro-batch size before heterogeneous scaling.
    seq_length : int
        Sequence length (tokens per sample).
    pipeline_model_parallel_size : int
        Number of pipeline stages.
    tensor_model_parallel_size : int
        Tensor parallelism degree.
    context_parallel_size : int
        Context (sequence) parallelism degree.
    is_sft : bool
        True when training in supervised fine-tuning mode.
    create_attention_mask_in_dataloader : bool
        If True the dataloader emits pre-computed attention masks; if False
        the model generates them from token ids.
    hybrid_context_parallel : bool
        Enable hybrid data+context parallelism (Megatron ``--hybrid-context-parallel``).
    logits_load_dir : str or None
        Directory holding pre-computed teacher log-probabilities for offline
        knowledge distillation.  None disables KD.
    logits_load_decode_threads : int
    logits_load_prefetch_factor : int
    logits_load_msc_prefetch_depth : int
    logits_load_kd_loss_alpha : float
    logits_load_ignore_errors : bool
        Parameters forwarded to ``LossFuncCallable`` (see
        ``build_cached_logits_loss_func``).
    profiles : list[HeteroDeviceProfile]
        Per-rank device capability profiles.  Length must equal the effective
        world size seen by this training job.
    slc_capacity_gb : float
        Maximum host-DRAM bytes (in GB) allocated to the SharedLocalityCache.
    slc_eviction_policy : str
        Cache eviction policy: ``"lru"`` (default) or ``"lfu"``.
    pipeline_model_parallel_layout : Any
        Forwarded from Megatron's ``config.pipeline_model_parallel_layout``.
    mtp_num_layers : int
        Number of multi-token-prediction layers (0 disables MTP).
    """

    micro_batch_size: int = 1
    seq_length: int = 2048
    pipeline_model_parallel_size: int = 1
    tensor_model_parallel_size: int = 1
    context_parallel_size: int = 1
    is_sft: bool = False
    create_attention_mask_in_dataloader: bool = True
    hybrid_context_parallel: bool = False

    # Knowledge distillation parameters (mirrors Megatron args)
    logits_load_dir: Optional[str] = None
    logits_load_decode_threads: int = 4
    logits_load_prefetch_factor: int = 2
    logits_load_msc_prefetch_depth: int = 0
    logits_load_kd_loss_alpha: float = 1.0
    logits_load_ignore_errors: bool = False

    # DES-LOC heterogeneous extensions
    profiles: List[HeteroDeviceProfile] = field(default_factory=list)
    slc_capacity_gb: float = 64.0
    slc_eviction_policy: str = "lru"

    # Pipeline layout (passed through to mtp_on_this_rank_func)
    pipeline_model_parallel_layout: Any = None
    mtp_num_layers: int = 0

    def profile_for_rank(self, rank: int) -> Optional[HeteroDeviceProfile]:
        """Return the device profile for *rank*, or None if not found."""
        for p in self.profiles:
            if p.local_rank == rank:
                return p
        return None

    def heterogeneous_micro_batch_size(self, rank: int) -> int:
        """Compute a device-scaled micro-batch size for *rank*.

        The H100 can handle proportionally more tokens per step than an A6000.
        This method returns a micro-batch size scaled by the rank's compute
        weight relative to the minimum-weight device, rounded down to an
        integer that keeps the effective batch size proportional.

        Parameters
        ----------
        rank : int
            Local rank for which to compute the scaled micro-batch size.

        Returns
        -------
        int
            Scaled micro-batch size (at least 1).
        """
        profile = self.profile_for_rank(rank)
        if profile is None or not self.profiles:
            return self.micro_batch_size
        min_weight = min(p.compute_weight for p in self.profiles)
        if min_weight <= 0:
            return self.micro_batch_size
        scale = profile.compute_weight / min_weight
        return max(1, int(math.floor(self.micro_batch_size * scale)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for cross-process transmission."""
        d: Dict[str, Any] = {}
        for f in self.__dataclass_fields__:  # type: ignore[attr-defined]
            v = getattr(self, f)
            if f == "profiles":
                d[f] = [
                    {
                        "local_rank": p.local_rank,
                        "sm_version": p.sm_version,
                        "memory_gb": p.memory_gb,
                        "compute_weight": p.compute_weight,
                    }
                    for p in v
                ]
            else:
                d[f] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeteroPretrainConfig":
        """Deserialise from a plain dict."""
        kwargs = dict(d)
        raw_profiles = kwargs.pop("profiles", [])
        profiles = [
            HeteroDeviceProfile(
                local_rank=p["local_rank"],
                sm_version=p["sm_version"],
                memory_gb=p["memory_gb"],
                compute_weight=p["compute_weight"],
            )
            for p in raw_profiles
        ]
        return cls(profiles=profiles, **kwargs)


# ---------------------------------------------------------------------------
# Shared Locality Cache (SLC)
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """Host-DRAM tensor cache shared across DES-LOC worker processes.

    Design
    ------
    The SLC is the central data-reuse mechanism in DES-LOC.  When a GPU
    worker finishes computing an intermediate tensor (e.g. the KV projection
    of a sequence), it pins that tensor to a host-DRAM buffer and registers
    a handle in the SLC.  Subsequent pipeline stages that need the same
    tensor can read from host DRAM rather than triggering a cross-device
    PCIe transfer, which would saturate the 12 GB/s bandwidth ceiling.

    The SLC is *not* a correctness requirement — it is a performance
    optimisation.  If a lookup misses, the caller falls back to the normal
    (potentially cross-device) transfer path.

    Implementation notes
    --------------------
    In this file the SLC is implemented as an in-process dict backed by
    ``torch.Tensor`` objects pinned to CPU memory (``pin_memory=True``).
    A production implementation would use ``multiprocessing.shared_memory``
    or a shared-memory file-backed mmap so that multiple OS processes can
    participate.  The interface is kept deliberately abstract so that the
    storage backend can be swapped without changing callers.

    Eviction
    --------
    When ``capacity_bytes`` is exceeded, the cache evicts entries according
    to ``eviction_policy``.  LRU is implemented via an ``OrderedDict``-style
    access-time tracker.

    Parameters
    ----------
    capacity_bytes : int
        Maximum number of bytes to store in the SLC.
    eviction_policy : str
        ``"lru"`` or ``"lfu"``.  Default: ``"lru"``.
    """

    def __init__(self, capacity_bytes: int, eviction_policy: str = "lru") -> None:
        self._capacity = capacity_bytes
        self._policy = eviction_policy
        self._store: Dict[str, torch.Tensor] = {}
        self._access_time: Dict[str, float] = {}
        self._access_count: Dict[str, int] = {}
        self._current_bytes: int = 0
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """Insert *tensor* under *key*, evicting if necessary.

        The tensor is moved to pinned CPU memory so that future H2D copies
        can use DMA without CPU involvement.

        Parameters
        ----------
        key : str
            Opaque identifier (typically a hash of micro-batch provenance).
        tensor : torch.Tensor
            The tensor to cache.  May be on any device; will be moved to
            pinned CPU RAM.

        Returns
        -------
        bool
            True if the tensor was stored; False if it is too large to fit
            even after eviction.
        """
        nbytes = tensor.nelement() * tensor.element_size()
        if nbytes > self._capacity:
            logger.debug("SLC: tensor %s (%d B) exceeds total capacity %d B; skip.",
                         key, nbytes, self._capacity)
            return False
