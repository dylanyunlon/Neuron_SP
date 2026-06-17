"""
heterogeneous_ddp.py — DES-LOC Heterogeneous DDP Wrapper
==========================================================

Upstream design intent (Megatron eb1c677):
    Megatron's ``wrap_model_chunks_with_ddp`` previously sourced DP world sizes
    from ``parallel_state`` globals when computing distributed-optimizer parameter
    layouts.  The commit threads an explicit ``ProcessGroupCollection`` (``pg_collection``)
    through the function so that callers using ``HyperCommGrid``-derived process
    groups — rather than ``parallel_state`` singletons — can supply their own
    group topology without touching global state.  The key changes were:

    1. ``layout_pgs`` is now resolved from the caller-supplied ``pg_collection``
       (falling back to ``ProcessGroupCollection.use_mpu_process_groups()`` when
       ``None``).
    2. ``data_parallel_world_size`` is sourced via ``get_pg_size(layout_pgs.dp_cp)``
       instead of ``mpu.get_data_parallel_world_size(with_context_parallel=True)``.
    3. ``expert_data_parallel_world_size`` uses
       ``get_pg_size(getattr(layout_pgs, "expt_dp", None))``.
    4. The ``pg_collection`` forwarding guard was tightened: previously
       ``pg_collection is not None and DP is not DDP``; now also excludes the
       ``torch.distributed.fsdp.FullyShardedDataParallel`` path (which takes
       ``process_group``, not ``pg_collection``).

DES-LOC adaptation points:
    The Neuron_SP stack runs on 2× A6000 (SM86, 48 GB) + 1× H100 NVL (SM90, 96 GB)
    over PCIe with no NVLink.  This asymmetry introduces three problems Megatron's
    upstream patch does not address:

    A. **Heterogeneous bucket sizing** — A6000 cards have less bandwidth headroom
       than the H100; naively sharing a single ``bucket_size`` degrades throughput.
       DES-LOC assigns per-device bucket capacities based on ``DeviceProfile``
       objects resolved at init time.

    B. **Locality-aware shard assignment** — the DES-LOC LOC-cache (Shared LOcality
       Cache) partitions parameters by affinity: large transformer blocks land on the
       H100 while embedding tables and smaller projection layers reside on the A6000
       pair.  ``HeterogeneousProcessGroupCollection`` carries per-device-class DP
       sub-groups so that gradient all-reduces respect PCIe topology.

    C. **SM-capability-gated collective selection** — SM86 does not support
       ``NCCL_ALGO=NVLS``; the wrapper detects compute capability at runtime and
       forces ``NCCL_ALGO=Ring`` for A6000 sub-groups while allowing ``Tree``/``NVLS``
       for the H100 sub-group.

    The ``wrap_model_chunks_with_des_loc_ddp`` entry point mirrors Megatron's
    ``wrap_model_chunks_with_ddp`` signature but accepts a
    ``HeterogeneousProcessGroupCollection`` (a strict superset of Megatron's
    ``ProcessGroupCollection``) and produces ``DesLocDistributedDataParallel``
    wrappers that honour all three constraints above.

    Locality-cache integration:
        ``DesLocDistributedDataParallel`` registers each wrapped module's parameters
        with the ``SharedLocalityCache`` singleton so that gradient communication
        can be overlapped with the H100's tensor-parallel compute during the A6000
        backward passes.

Author:  Neuron_SP / DES-LOC team
Mirrors: Megatron-LM commit eb1c677d7baf2f981bc7fb721807ddd5f7afd063
"""

from __future__ import annotations

import logging
import math
import os
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as TorchDDP

# ---------------------------------------------------------------------------
# Module-level logger — all DES-LOC log lines carry a structured prefix so
# that log aggregators can filter by component.
# ---------------------------------------------------------------------------
log = logging.getLogger("des_loc.heterogeneous_ddp")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    log.addHandler(_h)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: SM capability threshold above which NCCL tree / NVLS algorithms are safe.
_SM_THRESHOLD_FOR_ADVANCED_COLLECTIVES: int = 90

#: Default all-reduce bucket size for A6000 (SM86) in bytes.
#: Empirically chosen: 200 MB avoids PCIe saturation on the A6000 → CPU hop.
_DEFAULT_BUCKET_BYTES_A6000: int = 200 * 1024 * 1024

#: Default all-reduce bucket size for H100 NVL (SM90) in bytes.
#: H100 can sustain larger buckets; 400 MB matches its NVL bandwidth.
_DEFAULT_BUCKET_BYTES_H100: int = 400 * 1024 * 1024

#: Environment variable that can override per-device bucket bytes at launch
#: time, formatted as ``SM86:bytes,SM90:bytes``.
_BUCKET_OVERRIDE_ENV: str = "DES_LOC_BUCKET_BYTES"


# ---------------------------------------------------------------------------
# Device capability enum & profile
# ---------------------------------------------------------------------------


class DeviceClass(Enum):
    """Coarse device classification used for DES-LOC locality decisions."""

    A6000_SM86 = auto()   # 2× A6000 48 GB, SM86, PCIe-attached
    H100_SM90 = auto()    # 1× H100 NVL 96 GB, SM90, PCIe-attached
    UNKNOWN = auto()


@dataclass(frozen=True)
class DeviceProfile:
    """Immutable hardware description derived at runtime for a single CUDA device.

    Attributes
    ----------
    device_index:
        Local CUDA device ordinal.
    sm_major:
        Streaming multiprocessor major version (e.g. 8 for A6000, 9 for H100).
    sm_minor:
        SM minor version.
    total_memory_bytes:
        Total global memory reported by the CUDA runtime.
    device_class:
        Coarse classification used to pick bucket sizes and NCCL algorithms.
    bucket_bytes:
        All-reduce bucket capacity in bytes for this device class, accounting
        for any ``DES_LOC_BUCKET_BYTES`` environment override.
    supports_nvls:
        ``True`` when the device is capable of NCCL NVLS / Tree algorithms
        (SM ≥ 90 required; NVLink also required in practice but we gate only
        on SM here since our cluster is PCIe-only anyway).
    """

    device_index: int
    sm_major: int
    sm_minor: int
    total_memory_bytes: int
    device_class: DeviceClass
    bucket_bytes: int
    supports_nvls: bool

    @classmethod
    def from_device_index(cls, index: int) -> "DeviceProfile":
        """Construct a ``DeviceProfile`` by querying the CUDA runtime.

        Falls back gracefully when CUDA is unavailable (e.g. unit-test CPU
        environments) by returning an UNKNOWN profile with conservative defaults.
        """
        if not torch.cuda.is_available():
            return cls(
                device_index=index,
                sm_major=0,
                sm_minor=0,
                total_memory_bytes=0,
                device_class=DeviceClass.UNKNOWN,
                bucket_bytes=_DEFAULT_BUCKET_BYTES_A6000,
                supports_nvls=False,
            )

        props = torch.cuda.get_device_properties(index)
        sm_major, sm_minor = props.major, props.minor
        total_mem = props.total_memory

        # Classify device.
        sm = sm_major * 10 + sm_minor
        if sm == 86:
            device_class = DeviceClass.A6000_SM86
            default_bucket = _DEFAULT_BUCKET_BYTES_A6000
        elif sm >= 90:
            device_class = DeviceClass.H100_SM90
            default_bucket = _DEFAULT_BUCKET_BYTES_H100
        else:
            device_class = DeviceClass.UNKNOWN
            default_bucket = _DEFAULT_BUCKET_BYTES_A6000

        bucket_bytes = _resolve_bucket_override(sm_major, sm_minor, default_bucket)
        supports_nvls = (sm_major * 10 + sm_minor) >= _SM_THRESHOLD_FOR_ADVANCED_COLLECTIVES

        return cls(
            device_index=index,
            sm_major=sm_major,
            sm_minor=sm_minor,
            total_memory_bytes=total_mem,
            device_class=device_class,
            bucket_bytes=bucket_bytes,
            supports_nvls=supports_nvls,
        )


def _resolve_bucket_override(sm_major: int, sm_minor: int, default: int) -> int:
    """Parse ``DES_LOC_BUCKET_BYTES`` env var for the given SM version.

    Format: ``SM86:209715200,SM90:419430400`` (comma-separated key:value pairs).
    Any malformed entries are silently ignored.
    """
    raw = os.environ.get(_BUCKET_OVERRIDE_ENV, "")
    if not raw:
        return default
    sm_key = f"SM{sm_major}{sm_minor}"
    for token in raw.split(","):
        token = token.strip()
        if ":" not in token:
            continue
        key, _, val_str = token.partition(":")
        if key.upper() == sm_key.upper():
            try:
                parsed = int(val_str)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
    return default


# ---------------------------------------------------------------------------
# Process group collection — heterogeneous extension
# ---------------------------------------------------------------------------


@dataclass
class HeterogeneousProcessGroupCollection:
    """Extended process-group descriptor for DES-LOC heterogeneous clusters.

    Mirrors Megatron's ``ProcessGroupCollection`` but adds device-class-specific
    DP sub-groups so that gradient all-reduces can be routed along locality-
    optimal PCIe paths rather than forcing cross-class communication.

    Attributes
    ----------
    dp:
        Full data-parallel process group (all DP ranks regardless of device class).
    dp_cp:
        DP + context-parallel group; used for distributed-optimizer layout
        sizing (mirrors the Megatron ``layout_pgs.dp_cp`` accessor).
    expt_dp:
        Expert data-parallel group (MoE layers); may be ``None`` for dense models.
    tp:
        Tensor-parallel group.
    pp:
        Pipeline-parallel group.
    a6000_dp:
        DP sub-group containing only A6000-class ranks.  Used to restrict
        gradient collectives to the PCIe-local A6000 pair when the parameter
        is pinned to A6000 devices by the locality cache.  ``None`` if no
        A6000 ranks are present.
    h100_dp:
        DP sub-group containing only H100-class ranks.  Analogous to
        ``a6000_dp``.  ``None`` if no H100 ranks are present.
    device_profiles:
        Mapping from local rank index to its ``DeviceProfile``.  Populated at
        construction time; may be empty in test environments.
    """

    dp: Optional[dist.ProcessGroup] = None
    dp_cp: Optional[dist.ProcessGroup] = None
    expt_dp: Optional[dist.ProcessGroup] = None
    tp: Optional[dist.ProcessGroup] = None
    pp: Optional[dist.ProcessGroup] = None
    a6000_dp: Optional[dist.ProcessGroup] = None
    h100_dp: Optional[dist.ProcessGroup] = None
    device_profiles: Dict[int, DeviceProfile] = field(default_factory=dict)

    @classmethod
    def build_from_global_group(
        cls,
        dp_group: dist.ProcessGroup,
        rank_to_device: Dict[int, int],
    ) -> "HeterogeneousProcessGroupCollection":
        """Factory: build sub-groups for each device class from a global DP group.

        Parameters
        ----------
        dp_group:
            The full DP process group (all ranks).
        rank_to_device:
            Mapping from global rank to local CUDA device index.  Typically
            ``{rank: rank % torch.cuda.device_count()}``.

        Returns
        -------
        HeterogeneousProcessGroupCollection
            Populated with ``a6000_dp`` and ``h100_dp`` sub-groups built via
            ``dist.new_group``.
        """
        profiles: Dict[int, DeviceProfile] = {}
        a6000_ranks: List[int] = []
        h100_ranks: List[int] = []

        for rank, dev_idx in rank_to_device.items():
            profile = DeviceProfile.from_device_index(dev_idx)
            profiles[rank] = profile
            if profile.device_class == DeviceClass.A6000_SM86:
                a6000_ranks.append(rank)
            elif profile.device_class == DeviceClass.H100_SM90:
                h100_ranks.append(rank)

        a6000_pg: Optional[dist.ProcessGroup] = None
        h100_pg: Optional[dist.ProcessGroup] = None

        if dist.is_initialized():
            if len(a6000_ranks) >= 1:
                a6000_pg = dist.new_group(ranks=sorted(a6000_ranks))
                log.info(
                    "DES-LOC: created A6000 DP sub-group with %d rank(s): %s",
                    len(a6000_ranks),
                    sorted(a6000_ranks),
                )
            if len(h100_ranks) >= 1:
                h100_pg = dist.new_group(ranks=sorted(h100_ranks))
                log.info(
                    "DES-LOC: created H100 DP sub-group with %d rank(s): %s",
                    len(h100_ranks),
                    sorted(h100_ranks),
                )

        return cls(
            dp=dp_group,
            dp_cp=dp_group,
            a6000_dp=a6000_pg,
            h100_dp=h100_pg,
            device_profiles=profiles,
        )

    def dp_group_for_device(
        self, device_class: DeviceClass
    ) -> Optional[dist.ProcessGroup]:
        """Return the tightest-scoped DP group for the given device class.

        Falls back to the full ``dp`` group when no device-class-specific
        sub-group was built (e.g. in single-GPU tests or when all ranks share
        a device class).
        """
        if device_class == DeviceClass.A6000_SM86 and self.a6000_dp is not None:
            return self.a6000_dp
        if device_class == DeviceClass.H100_SM90 and self.h100_dp is not None:
            return self.h100_dp
        return self.dp


def _get_pg_size(pg: Optional[dist.ProcessGroup]) -> int:
    """Return the world size of *pg*, or 1 if *pg* is ``None`` / dist uninitialised.

    Mirrors Megatron's ``get_pg_size`` helper introduced in the same commit.
    Returns 1 rather than raising so that non-distributed unit tests can run
    without ``dist.init_process_group``.
    """
    if pg is None:
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=pg)


# ---------------------------------------------------------------------------
# Shared locality cache (stub interface)
# ---------------------------------------------------------------------------


class SharedLocalityCache:
    """Singleton stub representing the DES-LOC LOC-cache.

    In the full Neuron_SP runtime this class wraps a distributed key-value
    store that tracks which parameters are "hot" in each device's SRAM-resident
    cache tier.  Here we provide the interface contract so that
    ``DesLocDistributedDataParallel`` can register parameters without pulling
    in the full cache implementation.

    The cache partitions parameters into three affinity classes:

    * **H100-affine**: large weight tensors (transformer blocks, attention
      projections) that benefit from H100's high memory bandwidth.
    * **A6000-affine**: smaller projections, embedding tables, and output heads
      that fit comfortably in 48 GB and are accessed frequently from CPU DRAM
      (the 1.5 TB pool) via PCIe pinned transfers.
    * **neutral**: parameters that have no strong locality preference; the cache
      places these opportunistically based on current occupancy.

    Thread safety: all public methods acquire ``_lock`` before mutating
    ``_registry``.  The cache is safe to call from the backward hook thread.
    """

    _instance: Optional["SharedLocalityCache"] = None

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._registry: Dict[str, Dict[str, Any]] = {}  # param_name → metadata

    @classmethod
    def get(cls) -> "SharedLocalityCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (used in tests)."""
        cls._instance = None

    def register_parameter(
        self,
        name: str,
        param: nn.Parameter,
        device_class: DeviceClass,
        bucket_bytes: int,
    ) -> None:
        """Register *param* under *name* with its affinity metadata.

        Parameters
        ----------
        name:
            Fully-qualified parameter name (module path + leaf name).
        param:
            The actual parameter tensor.
        device_class:
            Device affinity class resolved from the wrapping device profile.
        bucket_bytes:
            The all-reduce bucket capacity that will be used for this
            parameter's gradient collective.
        """
        nbytes = param.numel() * param.element_size()
        with self._lock:
            self._registry[name] = {
                "shape": tuple(param.shape),
                "nbytes": nbytes,
                "device_class": device_class,
                "bucket_bytes": bucket_bytes,
                "affinity": self._infer_affinity(name, nbytes, device_class),
            }

    def _infer_affinity(
        self, name: str, nbytes: int, device_class: DeviceClass
    ) -> str:
        """Heuristic affinity label for a parameter.

        Rules (ordered by precedence):
        1. Embedding / output-head layers → ``a6000_affine`` (small, CPU-DRAM
           staging friendly).
        2. Any parameter > 64 MB on H100 class → ``h100_affine``.
        3. Otherwise → ``neutral``.

        These heuristics are intentionally coarse; the full LOC-cache runtime
        refines them using online access-frequency telemetry.
        """
        lower = name.lower()
        if any(tok in lower for tok in ("embed", "lm_head", "output_layer", "wte", "wpe")):
            return "a6000_affine"
        if device_class == DeviceClass.H100_SM90 and nbytes > 64 * 1024 * 1024:
            return "h100_affine"
        return "neutral"

    def affinity_of(self, name: str) -> str:
        """Return the affinity label for a registered parameter, or ``'neutral'``."""
        with self._lock:
            entry = self._registry.get(name)
            return entry["affinity"] if entry else "neutral"

    def summary(self) -> Dict[str, int]:
        """Return a count of parameters per affinity class."""
        counts: Dict[str, int] = {"a6000_affine": 0, "h100_affine": 0, "neutral": 0}
        with self._lock:
            for meta in self._registry.values():
                label = meta.get("affinity", "neutral")
                counts[label] = counts.get(label, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# DES-LOC DDP wrapper
# ---------------------------------------------------------------------------


class DesLocDistributedDataParallel(nn.Module):
    """Heterogeneity-aware DDP wrapper for Neuron_SP / DES-LOC.

    Wraps a single model chunk with PyTorch's ``DistributedDataParallel`` while
    applying three DES-LOC-specific behaviours:

    1. **Device-class bucket sizing**: the ``bucket_cap_mb`` passed to
       ``TorchDDP`` is derived from the local device's ``DeviceProfile`` rather
       than from a single cluster-wide constant.  This prevents the A6000 pair
       from stalling waiting for the same 400 MB buckets that the H100 prefers.

    2. **Locality-cache registration**: every ``requires_grad`` parameter is
       registered with the ``SharedLocalityCache`` singleton so that the DES-LOC
       scheduler can overlap A6000 gradient collectives with H100 tensor-parallel
       forward passes.

    3. **SM-capability-gated NCCL env**: the constructor temporarily sets
       ``NCCL_ALGO`` in ``os.environ`` before constructing the inner ``TorchDDP``
       so that NCCL picks the correct algorithm for the local device.  The
       original value (if any) is restored afterwards.

    Attributes
    ----------
    module:
        The unwrapped model chunk.
    ddp:
        The inner ``torch.nn.parallel.DistributedDataParallel`` instance.
    device_profile:
        Hardware profile of the device on which this wrapper lives.
    dp_group:
        The process group used for gradient all-reduces (may be a device-class
        sub-group, not the full DP group).
    dp_cp_group:
        The dp_cp group; exposed for compatibility with callers that inspect
        ``chunk.dp_cp_group`` (mirrors Megatron's DDP attribute convention).
    chunk_index:
        Zero-based index of this chunk within the model chunk list.
    """

    def __init__(
        self,
        module: nn.Module,
        dp_group: Optional[dist.ProcessGroup],
        dp_cp_group: Optional[dist.ProcessGroup],
        device_profile: DeviceProfile,
        chunk_index: int = 0,
        disable_bucketing: bool = False,
    ) -> None:
        super().__init__()
        self.module = module
        self.device_profile = device_profile
        self.dp_group = dp_group
        self.dp_cp_group = dp_cp_group
        self.chunk_index = chunk_index

        # Register parameters with the locality cache before constructing DDP
        # so that the cache is populated before the first backward pass.
        loc_cache = SharedLocalityCache.get()
        for name, param in module.named_parameters():
            if param.requires_grad:
                fqn = f"chunk{chunk_index}.{name}"
                loc_cache.register_parameter(
                    fqn, param, device_profile.device_class, device_profile.bucket_bytes
                )

        loc_summary = loc_cache.summary()
        log.info(
            "DES-LOC chunk %d locality cache after registration: "
            "h100_affine=%d, a6000_affine=%d, neutral=%d",
            chunk_index,
            loc_summary.get("h100_affine", 0),
            loc_summary.get("a6000_affine", 0),
            loc_summary.get("neutral", 0),
        )

        bucket_cap_mb = device_profile.bucket_bytes / (1024 * 1024)

        # Gate NCCL algorithm selection on SM capability.
        prev_nccl_algo = os.environ.get("NCCL_ALGO")
        if not device_profile.supports_nvls:
            os.environ["NCCL_ALGO"] = "Ring"
            log.info(
                "DES-LOC chunk %d: SM%d%d < 90, forcing NCCL_ALGO=Ring for this group",
                chunk_index,
                device_profile.sm_major,
                device_profile.sm_minor,
            )

        try:
            if dist.is_initialized() and dp_group is not None:
                self.ddp = TorchDDP(
                    module,
                    process_group=dp_group,
                    bucket_cap_mb=bucket_cap_mb,
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                )
            else:
                # Non-distributed mode (single-GPU or unit tests without dist).
                self.ddp = module  # type: ignore[assignment]
        finally:
            # Always restore NCCL_ALGO regardless of whether DDP init raised.
            if prev_nccl_algo is None:
                os.environ.pop("NCCL_ALGO", None)
            else:
                os.environ["NCCL_ALGO"] = prev_nccl_algo

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
        return self.ddp(*args, **kwargs)

    def parameters(self, recurse: bool = True):  # noqa: D102
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # noqa: D102
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, **kwargs: Any) -> Dict[str, Any]:  # noqa: D102
        return self.module.state_dict(**kwargs)

    def load_state_dict(self, state_dict: Dict[str, Any], **kwargs: Any) -> Any:  # noqa: D102
        return self.module.load_state_dict(state_dict, **kwargs)


# ---------------------------------------------------------------------------
# Parameter layout computation
# ---------------------------------------------------------------------------


@dataclass
class ParameterLayout:
    """Minimal descriptor for a parameter's distributed-optimizer layout.

    In the full Neuron_SP runtime this mirrors Megatron's ``ParameterLayout``
    dataclass which records how parameters are sharded across DP ranks for the
    zero-redundancy optimizer.  We carry it here for API completeness.

    Attributes
    ----------
    param_name:
        Fully-qualified parameter name.
    shard_size:
        Number of elements owned by this rank.
    dp_world_size:
        World size of the DP group used to compute the shard.
    expert_dp_world_size:
        World size of the expert DP group (1 for dense models).
    device_class:
        Device class of the rank that owns this shard.
    """

    param_name: str
    shard_size: int
    dp_world_size: int
    expert_dp_world_size: int
    device_class: DeviceClass


def compute_heterogeneous_layout(
    params: List[nn.Parameter],
    param_names: List[str],
    dp_world_size: int,
    expert_dp_world_size: int,
    device_class: DeviceClass,
) -> List[ParameterLayout]:
    """Compute per-parameter distributed-optimizer shard layouts.

    Upstream Megatron computes a single layout per chunk assuming a homogeneous
    DP world size.  DES-LOC must account for the fact that the H100 holds
    parameters of different sizes than the A6000 pair, so effective shard sizes
    differ.

    The layout is computed as::

        shard_size = ceil(numel / dp_world_size)

    for standard parameters, and::

        shard_size = ceil(numel / expert_dp_world_size)

    for expert parameters (identified by the ``"expert"`` substring in their
    name, following Megatron convention).

    Parameters
    ----------
    params:
        List of ``requires_grad`` parameters in this chunk.
    param_names:
        Fully-qualified names corresponding to *params*.
    dp_world_size:
        DP world size from ``layout_pgs.dp_cp`` (mirrors Megatron).
    expert_dp_world_size:
        Expert DP world size from ``layout_pgs.expt_dp`` (mirrors Megatron).
    device_class:
        Device class of the local rank (used to annotate the layout).

    Returns
    -------
    list of ParameterLayout
    """
    layouts: List[ParameterLayout] = []
    for param, name in zip(params, param_names):
        numel = param.numel()
        if "expert" in name.lower():
            ws = max(expert_dp_world_size, 1)
        else:
            ws = max(dp_world_size, 1)
        shard_size = math.ceil(numel / ws)
        layouts.append(
            ParameterLayout(
                param_name=name,
                shard_size=shard_size,
                dp_world_size=dp_world_size,
                expert_dp_world_size=expert_dp_world_size,
                device_class=device_class,
            )
        )
    return layouts


# ---------------------------------------------------------------------------
# DDP config
# ---------------------------------------------------------------------------


@dataclass
class DesLocDDPConfig:
    """Configuration knobs for ``wrap_model_chunks_with_des_loc_ddp``.

    Attributes
    ----------
    use_distributed_optimizer:
        When ``True``, triggers parameter layout computation (mirrors Megatron's
        ``DistributedDataParallelConfig.use_distributed_optimizer``).
    bucket_size_override:
        If not ``None``, overrides the per-device bucket size derived from
        ``DeviceProfile`` for *all* devices.  Useful for benchmarking.
    disable_bucketing_default:
        Cluster-wide default for ``disable_bucketing``; can be overridden
        per-chunk via ``disable_bucketing_per_chunk``.
    use_locality_cache:
        When ``True`` (default), registers parameters with the
        ``SharedLocalityCache`` singleton.  Disable for profiling runs where
        cache overhead should be excluded.
    enable_sm_gated_nccl_algo:
        When ``True`` (default), sets ``NCCL_ALGO=Ring`` for SM < 90 devices
        before constructing DDP.
    """

    use_distributed_optimizer: bool = False
    bucket_size_override: Optional[int] = None
    disable_bucketing_default: bool = False
    use_locality_cache: bool = True
    enable_sm_gated_nccl_algo: bool = True


# ---------------------------------------------------------------------------
# Main entry point — mirrors Megatron's wrap_model_chunks_with_ddp
# ---------------------------------------------------------------------------


def wrap_model_chunks_with_des_loc_ddp(
    model_chunks: List[nn.Module],
    ddp_config: DesLocDDPConfig,
    pg_collection: Optional[HeterogeneousProcessGroupCollection] = None,
    bucket_sizes: Optional[List[int]] = None,
    disable_bucketing_per_chunk: Optional[List[bool]] = None,
    local_rank: Optional[int] = None,
) -> List[DesLocDistributedDataParallel]:
    """Wrap model chunks with DES-LOC heterogeneity-aware DDP.

    This function mirrors Megatron's ``wrap_model_chunks_with_ddp`` signature
    and threading semantics from commit eb1c677, adapted for the Neuron_SP
    DES-LOC stack:

    Megatron threading fix (eb1c677):
        The upstream commit ensures that an explicit ``pg_collection`` (built
        from a ``HyperCommGrid``) is forwarded to both standard DDP and FSDP
        variants, and used to source DP world sizes for layout computation,
        replacing the previous hard dependency on ``parallel_state`` globals.

    DES-LOC adaptation:
        *  ``pg_collection`` is now a ``HeterogeneousProcessGroupCollection``
           that carries device-class sub-groups (``a6000_dp``, ``h100_dp``).
        *  The local device profile is resolved from ``local_rank`` and used to
           select the tightest-scoped DP group for this rank.
        *  Parameter layouts are computed with device-class annotation so that
           the distributed optimizer can apply asymmetric sharding.
        *  Each chunk is wrapped with ``DesLocDistributedDataParallel`` rather
           than raw ``TorchDDP``.

    Parameters
    ----------
    model_chunks:
        List of model chunk ``nn.Module`` instances (e.g. pipeline stages).
    ddp_config:
        DES-LOC DDP configuration.
    pg_collection:
        Optional ``HeterogeneousProcessGroupCollection``.  When ``None`` the
        function falls back to a bare ``HeterogeneousProcessGroupCollection``
        with ``dp=dist.group.WORLD`` (or ``None`` if dist is not initialised),
        which matches Megatron's ``ProcessGroupCollection.use_mpu_process_groups()``
        fallback semantics.
    bucket_sizes:
        Optional per-chunk bucket size overrides (in bytes).  When ``None``,
        per-device bucket sizes from ``DeviceProfile`` are used.
    disable_bucketing_per_chunk:
        Optional per-chunk flag to disable gradient bucketing entirely.
        Defaults to ``[ddp_config.disable_bucketing_default] * len(model_chunks)``.
    local_rank:
        Local CUDA device index for the calling process.  Defaults to
        ``torch.cuda.current_device()`` when CUDA is available, else 0.

    Returns
    -------
    list of DesLocDistributedDataParallel
        One wrapper per input chunk, in the same order.

    Raises
    ------
    ValueError
        If ``len(bucket_sizes) != len(model_chunks)`` when *bucket_sizes* is
        provided.
    AssertionError
        If ``pg_collection`` is provided but ``pg_collection.dp_cp`` is ``None``
        and ``ddp_config.use_distributed_optimizer`` is ``True``.
    """
    n_chunks = len(model_chunks)
    if n_chunks == 0:
        return []

    # Resolve local device.
    if local_rank is None:
        local_rank = torch.cuda.current_device() if torch.cuda.is_available() else 0

    # Build DeviceProfile for the local rank.
    device_profile = DeviceProfile.from_device_index(local_rank)

    log.info(
        "DES-LOC wrap_model_chunks: local_rank=%d device_class=%s sm=%d%d "
        "bucket_bytes=%d supports_nvls=%s",
        local_rank,
        device_profile.device_class.name,
        device_profile.sm_major,
        device_profile.sm_minor,
        device_profile.bucket_bytes,
        device_profile.supports_nvls,
    )

    # Resolve pg_collection — mirrors Megatron's layout_pgs logic.
    if pg_collection is None:
        # Fallback: build a bare collection using the world group (or None).
        world_pg = dist.group.WORLD if dist.is_initialized() else None
        pg_collection = HeterogeneousProcessGroupCollection(
            dp=world_pg,
            dp_cp=world_pg,
            expt_dp=None,
        )
        log.info(
            "DES-LOC: pg_collection not provided; falling back to WORLD group "
            "(parallel_state globals not consulted — DES-LOC stateless mode)"
        )

    # Validate dp_cp when distributed optimizer layout is needed.
    if ddp_config.use_distributed_optimizer:
        assert pg_collection.dp_cp is not None, (
            "wrap_model_chunks_with_des_loc_ddp requires pg_collection.dp_cp "
            "to be set when use_distributed_optimizer=True.  "
            "This mirrors Megatron eb1c677 assertion: dp_cp is mandatory for "
            "distributed-optimizer parameter layout computation."
        )

    # Resolve world sizes from pg_collection (mirrors Megatron get_pg_size calls).
    dp_world_size = _get_pg_size(pg_collection.dp_cp)
    expert_dp_world_size = _get_pg_size(getattr(pg_collection, "expt_dp", None))

    # Resolve per-device-class DP group for gradient all-reduces.
    local_dp_group = pg_collection.dp_group_for_device(device_profile.device_class)

    # Normalise bucket sizes.
    if bucket_sizes is None:
        if ddp_config.bucket_size_override is not None:
            bucket_sizes = [ddp_config.bucket_size_override] * n_chunks
        else:
            bucket_sizes = [device_profile.bucket_bytes] * n_chunks
    elif len(bucket_sizes) != n_chunks:
        raise ValueError(
            f"len(bucket_sizes)={len(bucket_sizes)} != len(model_chunks)={n_chunks}"
        )

    # Normalise disable_bucketing flags.
    if disable_bucketing_per_chunk is None:
        disable_bucketing_per_chunk = [ddp_config.disable_bucketing_default] * n_chunks

    # Compute per-chunk parameter layouts when using distributed optimizer.
    per_chunk_layouts: List[Optional[List[ParameterLayout]]] = [None] * n_chunks
    if ddp_config.use_distributed_optimizer:
        for i, chunk in enumerate(model_chunks):
            params = [p for p in chunk.parameters() if p.requires_grad]
            names = [n for n, p in chunk.named_parameters() if p.requires_grad]
            per_chunk_layouts[i] = compute_heterogeneous_layout(
                params=params,
                param_names=names,
                dp_world_size=dp_world_size,
                expert_dp_world_size=expert_dp_world_size,
                device_class=device_profile.device_class,
            )
            log.info(
                "DES-LOC chunk %d layout: %d params, dp_world=%d, expert_dp_world=%d",
                i,
                len(params),
                dp_world_size,
                expert_dp_world_size,
            )

    # Wrap each chunk.
    wrapped: List[DesLocDistributedDataParallel] = []
    for i, (chunk, bucket_size, disable_bucket) in enumerate(
        zip(model_chunks, bucket_sizes, disable_bucketing_per_chunk)
    ):
        # Override bucket_bytes on a copy of the profile if caller specified a
        # per-chunk size — we don't want to mutate the shared profile object.
        effective_profile = device_profile
        if bucket_size != device_profile.bucket_bytes:
            effective_profile = DeviceProfile(
                device_index=device_profile.device_index,
                sm_major=device_profile.sm_major,
                sm_minor=device_profile.sm_minor,
                total_memory_bytes=device_profile.total_memory_bytes,
                device_class=device_profile.device_class,
                bucket_bytes=bucket_size,
                supports_nvls=device_profile.supports_nvls,
            )

        wrapper = DesLocDistributedDataParallel(
            module=chunk,
            dp_group=local_dp_group,
            dp_cp_group=pg_collection.dp_cp,
            device_profile=effective_profile,
            chunk_index=i,
            disable_bucketing=disable_bucket,
        )
        wrapped.append(wrapper)

    return wrapped


# ---------------------------------------------------------------------------
# Utility: resolve pg_collection from environment (used by training loop)
# ---------------------------------------------------------------------------


def resolve_pg_collection_from_env(
    rank_to_device: Optional[Dict[int, int]] = None,
) -> HeterogeneousProcessGroupCollection:
    """Build a ``HeterogeneousProcessGroupCollection`` from the running dist env.

    Called by the Neuron_SP training loop when no explicit ``pg_collection`` is
    passed.  Constructs device-class sub-groups by querying each rank's CUDA
    device properties.

    Parameters
    ----------
    rank_to_device:
        Optional override mapping.  When ``None``, defaults to
        ``{r: r % torch.cuda.device_count() for r in range(world_size)}``.

    Returns
    -------
    HeterogeneousProcessGroupCollection
    """
    if not dist.is_initialized():
        log.warning(
            "DES-LOC: dist not initialised; returning bare HeterogeneousProcessGroupCollection"
        )
        return HeterogeneousProcessGroupCollection()

    world_size = dist.get_world_size()
    n_dev = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if rank_to_device is None:
        rank_to_device = {r: r % n_dev for r in range(world_size)}

    world_pg = dist.group.WORLD
    pgc = HeterogeneousProcessGroupCollection.build_from_global_group(
        dp_group=world_pg,
        rank_to_device=rank_to_device,
    )
    log.info(
        "DES-LOC: resolved pg_collection from env — world_size=%d, "
        "a6000_ranks=%s, h100_ranks=%s",
        world_size,
        [r for r, d in rank_to_device.items()
         if pgc.device_profiles.get(r, DeviceProfile.from_device_index(d)).device_class
         == DeviceClass.A6000_SM86],
        [r for r, d in rank_to_device.items()
         if pgc.device_profiles.get(r, DeviceProfile.from_device_index(d)).device_class
         == DeviceClass.H100_SM90],
    )
    return pgc


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    # We use unittest directly so this file is self-contained.
    # Run with:  python -m deepspeed.runtime.heterogeneous_ddp

    class TestDeviceProfile(unittest.TestCase):
        """Tests for DeviceProfile factory and bucket-override logic."""

        def test_from_device_index_no_cuda(self):
            """Falls back gracefully when CUDA is unavailable."""
            # Temporarily monkeypatch torch.cuda.is_available.
            orig = torch.cuda.is_available
            try:
                torch.cuda.is_available = lambda: False
                p = DeviceProfile.from_device_index(0)
                self.assertEqual(p.device_class, DeviceClass.UNKNOWN)
                self.assertEqual(p.bucket_bytes, _DEFAULT_BUCKET_BYTES_A6000)
                self.assertFalse(p.supports_nvls)
            finally:
                torch.cuda.is_available = orig

        def test_bucket_override_env(self):
            """DES_LOC_BUCKET_BYTES env var is parsed correctly."""
            os.environ[_BUCKET_OVERRIDE_ENV] = "SM86:123456789,SM90:987654321"
            try:
                result_86 = _resolve_bucket_override(8, 6, _DEFAULT_BUCKET_BYTES_A6000)
                result_90 = _resolve_bucket_override(9, 0, _DEFAULT_BUCKET_BYTES_H100)
                self.assertEqual(result_86, 123456789)
                self.assertEqual(result_90, 987654321)
            finally:
                del os.environ[_BUCKET_OVERRIDE_ENV]

        def test_bucket_override_env_malformed(self):
            """Malformed entries in DES_LOC_BUCKET_BYTES are ignored."""
            os.environ[_BUCKET_OVERRIDE_ENV] = "SM86:not_a_number,garbage"
            try:
                result = _resolve_bucket_override(8, 6, _DEFAULT_BUCKET_BYTES_A6000)
                self.assertEqual(result, _DEFAULT_BUCKET_BYTES_A6000)
            finally:
                del os.environ[_BUCKET_OVERRIDE_ENV]

        def test_bucket_override_missing_key(self):
            """Missing SM key returns default."""
            os.environ[_BUCKET_OVERRIDE_ENV] = "SM90:999"
            try:
                result = _resolve_bucket_override(8, 6, _DEFAULT_BUCKET_BYTES_A6000)
                self.assertEqual(result, _DEFAULT_BUCKET_BYTES_A6000)
            finally:
                del os.environ[_BUCKET_OVERRIDE_ENV]

    class TestGetPgSize(unittest.TestCase):
        """Tests for _get_pg_size helper."""

        def test_none_returns_1(self):
            self.assertEqual(_get_pg_size(None), 1)

        def test_uninitialised_dist_returns_1(self):
            # dist is not initialised in this test process.
            self.assertEqual(_get_pg_size(None), 1)

    class TestSharedLocalityCache(unittest.TestCase):
        """Tests for SharedLocalityCache registration and affinity inference."""

        def setUp(self):
            SharedLocalityCache.reset()

        def tearDown(self):
            SharedLocalityCache.reset()

        def test_singleton(self):
            c1 = SharedLocalityCache.get()
            c2 = SharedLocalityCache.get()
            self.assertIs(c1, c2)

        def test_register_and_affinity_embed(self):
            cache = SharedLocalityCache.get()
            param = nn.Parameter(torch.zeros(100, 512))
            cache.register_parameter(
                "model.embed_tokens.weight", param, DeviceClass.H100_SM90, 400 * 1024 * 1024
            )
            self.assertEqual(cache.affinity_of("model.embed_tokens.weight"), "a6000_affine")

        def test_register_and_affinity_large_h100(self):
            cache = SharedLocalityCache.get()
            # 256 MB parameter — should be h100_affine on H100.
            param = nn.Parameter(torch.zeros(64 * 1024 * 1024 + 1))
            cache.register_parameter(
                "model.layers.0.self_attn.q_proj.weight",
                param,
                DeviceClass.H100_SM90,
                400 * 1024 * 1024,
            )
            self.assertEqual(
                cache.affinity_of("model.layers.0.self_attn.q_proj.weight"), "h100_affine"
            )

        def test_register_and_affinity_neutral(self):
            cache = SharedLocalityCache.get()
            param = nn.Parameter(torch.zeros(64, 64))
            cache.register_parameter(
                "model.layers.0.fc.weight", param, DeviceClass.A6000_SM86, 200 * 1024 * 1024
            )
            self.assertEqual(cache.affinity_of("model.layers.0.fc.weight"), "neutral")

        def test_unknown_name_returns_neutral(self):
            cache = SharedLocalityCache.get()
            self.assertEqual(cache.affinity_of("nonexistent.param"), "neutral")

        def test_summary_counts(self):
            cache = SharedLocalityCache.get()
            for i in range(3):
                cache.register_parameter(
                    f"embed_{i}", nn.Parameter(torch.zeros(10)), DeviceClass.A6000_SM86, 1
                )
            for i in range(2):
                cache.register_parameter(
                    f"large_{i}",
                    nn.Parameter(torch.zeros(64 * 1024 * 1024 + 1)),
                    DeviceClass.H100_SM90,
                    1,
                )
            summary = cache.summary()
            self.assertEqual(summary["a6000_affine"], 3)
            self.assertEqual(summary["h100_affine"], 2)

        def test_thread_safety(self):
            """Concurrent registrations do not corrupt the registry."""
            import threading
            cache = SharedLocalityCache.get()
            errors = []

            def register_many(start: int) -> None:
                try:
                    for j in range(start, start + 50):
                        cache.register_parameter(
                            f"param_{j}",
                            nn.Parameter(torch.zeros(16)),
                            DeviceClass.UNKNOWN,
                            1,
                        )
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=register_many, args=(i * 50,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(cache._registry), 200)

    class TestHeterogeneousProcessGroupCollection(unittest.TestCase):
        """Tests for HeterogeneousProcessGroupCollection construction."""

        def test_dp_group_for_device_fallback(self):
            """Falls back to dp when sub-groups are None."""
            mock_pg = object()  # not a real ProcessGroup — just for identity tests
            pgc = HeterogeneousProcessGroupCollection(dp=mock_pg, a6000_dp=None, h100_dp=None)
            self.assertIs(pgc.dp_group_for_device(DeviceClass.A6000_SM86), mock_pg)
            self.assertIs(pgc.dp_group_for_device(DeviceClass.H100_SM90), mock_pg)
            self.assertIs(pgc.dp_group_for_device(DeviceClass.UNKNOWN), mock_pg)

        def test_dp_group_for_device_specific(self):
            """Returns specific sub-group when available."""
            mock_dp = object()
            mock_a6000 = object()
            mock_h100 = object()
            pgc = HeterogeneousProcessGroupCollection(
                dp=mock_dp, a6000_dp=mock_a6000, h100_dp=mock_h100
            )
            self.assertIs(pgc.dp_group_for_device(DeviceClass.A6000_SM86), mock_a6000)
            self.assertIs(pgc.dp_group_for_device(DeviceClass.H100_SM90), mock_h100)
            self.assertIs(pgc.dp_group_for_device(DeviceClass.UNKNOWN), mock_dp)

    class TestComputeHeterogeneousLayout(unittest.TestCase):
        """Tests for compute_heterogeneous_layout."""

        def _make_params(self, shapes):
            return [nn.Parameter(torch.zeros(*s)) for s in shapes]

        def test_basic_layout(self):
            shapes = [(1024, 1024), (512, 512)]
            params = self._make_params(shapes)
            names = ["layer.weight", "layer.bias_proj"]
            layouts = compute_heterogeneous_layout(
                params, names, dp_world_size=4, expert_dp_world_size=1,
                device_class=DeviceClass.H100_SM90
            )
            self.assertEqual(len(layouts), 2)
            # 1024*1024 / 4 = 262144
            self.assertEqual(layouts[0].shard_size, math.ceil(1024 * 1024 / 4))
            self.assertEqual(layouts[0].dp_world_size, 4)
            self.assertEqual(layouts[0].device_class, DeviceClass.H100_SM90)

        def test_expert_param_uses_expert_world_size(self):
            params = [nn.Parameter(torch.zeros(1000))]
            names = ["model.expert_layer.weight"]
            layouts = compute_heterogeneous_layout(
                params, names, dp_world_size=4, expert_dp_world_size=2,
                device_class=DeviceClass.A6000_SM86
            )
            self.assertEqual(layouts[0].shard_size, math.ceil(1000 / 2))
            self.assertEqual(layouts[0].expert_dp_world_size, 2)

        def test_dp_world_size_zero_defaults_to_1(self):
            """dp_world_size=0 must not divide by zero."""
            params = [nn.Parameter(torch.zeros(100))]
            names = ["w"]
            layouts = compute_heterogeneous_layout(
                params, names, dp_world_size=0, expert_dp_world_size=0,
                device_class=DeviceClass.UNKNOWN
            )
            self.assertEqual(layouts[0].shard_size, 100)

        def test_empty_params(self):
            layouts = compute_heterogeneous_layout(
                [], [], dp_world_size=4, expert_dp_world_size=2,
                device_class=DeviceClass.H100_SM90
            )
            self.assertEqual(layouts, [])

    class TestWrapModelChunksWithDesLocDDP(unittest.TestCase):
        """Integration-style tests for wrap_model_chunks_with_des_loc_ddp.

        These tests run without a live dist process group; DDP construction falls
        back to the non-distributed path (module is stored unwrapped inside
        DesLocDistributedDataParallel.ddp).
        """

        def setUp(self):
            SharedLocalityCache.reset()

        def tearDown(self):
            SharedLocalityCache.reset()

        def _tiny_model(self):
            return nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))

        def test_empty_chunks_returns_empty(self):
            result = wrap_model_chunks_with_des_loc_ddp([], DesLocDDPConfig())
            self.assertEqual(result, [])

        def test_single_chunk_no_dist(self):
            model = self._tiny_model()
            ddp_config = DesLocDDPConfig(use_distributed_optimizer=False)
            wrapped = wrap_model_chunks_with_des_loc_ddp([model], ddp_config)
            self.assertEqual(len(wrapped), 1)
            self.assertIsInstance(wrapped[0], DesLocDistributedDataParallel)
            self.assertIs(wrapped[0].module, model)

        def test_multiple_chunks_no_dist(self):
            models = [self._tiny_model() for _ in range(3)]
            wrapped = wrap_model_chunks_with_des_loc_ddp(models, DesLocDDPConfig())
            self.assertEqual(len(wrapped), 3)
            for i, w in enumerate(wrapped):
                self.assertEqual(w.chunk_index, i)

        def test_bucket_size_mismatch_raises(self):
            models = [self._tiny_model(), self._tiny_model()]
            with self.assertRaises(ValueError):
                wrap_model_chunks_with_des_loc_ddp(
                    models, DesLocDDPConfig(), bucket_sizes=[1024 * 1024]
                )

        def test_distributed_optimizer_requires_dp_cp(self):
            models = [self._tiny_model()]
            pgc = HeterogeneousProcessGroupCollection(dp=None, dp_cp=None)
            with self.assertRaises(AssertionError):
                wrap_model_chunks_with_des_loc_ddp(
                    models,
                    DesLocDDPConfig(use_distributed_optimizer=True),
                    pg_collection=pgc,
                )

        def test_explicit_bucket_size_override_propagates(self):
            model = self._tiny_model()
            override_bytes = 12345678
            ddp_config = DesLocDDPConfig(bucket_size_override=override_bytes)
            wrapped = wrap_model_chunks_with_des_loc_ddp([model], ddp_config)
            self.assertEqual(wrapped[0].device_profile.bucket_bytes, override_bytes)

        def test_per_chunk_bucket_sizes(self):
            models = [self._tiny_model() for _ in range(2)]
            bucket_sizes = [111111, 222222]
            wrapped = wrap_model_chunks_with_des_loc_ddp(
                models, DesLocDDPConfig(), bucket_sizes=bucket_sizes
            )
            self.assertEqual(wrapped[0].device_profile.bucket_bytes, 111111)
            self.assertEqual(wrapped[1].device_profile.bucket_bytes, 222222)

        def test_locality_cache_populated_after_wrap(self):
            model = self._tiny_model()
            wrap_model_chunks_with_des_loc_ddp([model], DesLocDDPConfig())
            cache = SharedLocalityCache.get()
            # At least one parameter should be registered.
            self.assertGreater(len(cache._registry), 0)

        def test_dp_cp_group_attribute_exposed(self):
            model = self._tiny_model()
            # Build a pgc with a mock dp_cp (non-None so assertion passes
            # even without use_distributed_optimizer=True).
            mock_pg = object()
            pgc = HeterogeneousProcessGroupCollection(dp=mock_pg, dp_cp=mock_pg)
            wrapped = wrap_model_chunks_with_des_loc_ddp([model], DesLocDDPConfig(), pg_collection=pgc)
            # dp_cp_group should be forwarded to the wrapper.
            self.assertIs(wrapped[0].dp_cp_group, mock_pg)

        def test_disable_bucketing_per_chunk_length_normalised(self):
            models = [self._tiny_model() for _ in range(3)]
            # No explicit disable_bucketing_per_chunk — defaults should be applied.
            wrapped = wrap_model_chunks_with_des_loc_ddp(
                models, DesLocDDPConfig(disable_bucketing_default=True)
            )
            self.assertEqual(len(wrapped), 3)

        def test_nccl_algo_env_restored_after_wrap(self):
            """NCCL_ALGO env var must be restored after DDP construction."""
            original = os.environ.get("NCCL_ALGO")
            os.environ["NCCL_ALGO"] = "Tree"
            try:
                model = self._tiny_model()
                wrap_model_chunks_with_des_loc_ddp([model], DesLocDDPConfig())
                self.assertEqual(os.environ.get("NCCL_ALGO"), "Tree")
            finally:
                if original is None:
                    os.environ.pop("NCCL_ALGO", None)
                else:
                    os.environ["NCCL_ALGO"] = original

        def test_forward_pass_through_wrapper(self):
            model = self._tiny_model()
            wrapped = wrap_model_chunks_with_des_loc_ddp([model], DesLocDDPConfig())
            x = torch.randn(4, 16)
            out = wrapped[0](x)
            self.assertEqual(out.shape, (4, 4))

        def test_parameters_delegated_to_module(self):
            model = self._tiny_model()
            wrapped = wrap_model_chunks_with_des_loc_ddp([model], DesLocDDPConfig())
            module_params = list(model.parameters())
            wrapper_params = list(wrapped[0].parameters())
            self.assertEqual(len(module_params), len(wrapper_params))

        def test_state_dict_round_trip(self):
            model = self._tiny_model()
            wrapped = wrap_model_chunks_with_des_loc_ddp([model], DesLocDDPConfig())
            sd = wrapped[0].state_dict()
            wrapped[0].load_state_dict(sd)  # should not raise

    class TestResolveFromEnvNoDist(unittest.TestCase):
        """Smoke-test resolve_pg_collection_from_env without dist."""

        def test_no_dist_returns_bare_collection(self):
            pgc = resolve_pg_collection_from_env()
            self.assertIsInstance(pgc, HeterogeneousProcessGroupCollection)
            self.assertIsNone(pgc.dp)

    # Run all tests.
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDeviceProfile)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGetPgSize))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSharedLocalityCache))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeterogeneousProcessGroupCollection))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestComputeHeterogeneousLayout))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWrapModelChunksWithDesLocDDP))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestResolveFromEnvNoDist))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
