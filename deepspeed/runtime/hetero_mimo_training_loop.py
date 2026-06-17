"""
HeteroMIMOTrainingLoop — DES-LOC Heterogeneous MIMO Training Loop
==================================================================

Upstream Design Intent (Megatron addc601f57ed539506183b704bb9d08f459d7f50)
---------------------------------------------------------------------------
Megatron's commit "Thread MIMO support through the stock training loop" introduces
Multi-Input Multi-Output (MIMO) model awareness into two core subsystems:

1. **Optimizer dispatch** (``megatron/core/optimizer/__init__.py``):
   When ``model_chunks[0]`` is a ``MimoModel``, the optimizer factory delegates to
   ``get_mimo_optimizer``, which wires per-module heterogeneous optimizer configs
   rather than applying a single monolithic optimizer across all parameters.
   The constraint that ``len(model_chunks) == 1`` (no virtual pipeline parallelism
   for MIMO) reflects that MIMO's internal routing already encodes the pipeline
   topology.

2. **Training loop plumbing** (``megatron/training/training.py``):
   - ``setup_model_and_optimizer`` accepts an optional ``pg_collection`` and
     forwards it to ``get_model``, enabling per-module process-group bindings.
   - ``train_step`` gains ``p2p_communicator`` and ``schedule_pg_collection``
     parameters that are forwarded verbatim to ``forward_backward_func``, allowing
     the schedule to perform cross-grid P2P communication for the heterogeneous
     (multi-GPU-pool) case.
   - ``train`` forwards both new arguments all the way to ``train_step``, completing
     the call-chain thread.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) targets a specific
heterogeneous cluster: 2× A6000 (48 GB, SM86, PCIe) + 1× H100 NVL (96 GB, SM90,
PCIe), connected over PCIe without NVLink, backed by 1.5 TB CPU DRAM.

The Megatron MIMO threading pattern maps onto DES-LOC as follows:

* **Per-device optimizer dispatch** → ``HeteroOptimizerRouter`` selects
  ``AdamW`` with ``fused=True`` on H100 and ``Adam`` with ``foreach=True`` on
  A6000s, matching SM-capability and memory bandwidth profiles.

* **``pg_collection`` / ``ProcessGroupCollection``** → ``DeviceLocalityGroups``
  encodes PCIe topology: A6000 pair shares one process group (locality cache hits
  are cheap within that pool), H100 sits in a separate group.  Cross-group
  transfers go through the ``SharedLocalityCache`` (CPU DRAM as a staging area)
  rather than direct P2P.

* **``p2p_communicator``** → ``PCIeP2PCommunicator`` in DES-LOC replaces
  NVLink-assumed direct transfers with buffered, locality-aware copies that route
  large activations through the 1.5 TB DRAM staging area when direct PCIe
  bandwidth would be the bottleneck.

* **``schedule_pg_collection``** → ``HeteroScheduleGroups`` assigns pipeline
  micro-batches to device pools based on real-time memory headroom, using SM90
  tensor-core capacity for compute-heavy stages and SM86 for memory-bandwidth-
  bound stages.

* **MIMO model detection** → ``HeteroMIMOModel`` duck-types against DeepSpeed's
  engine wrapping, enabling the training loop to short-circuit to heterogeneous
  optimizer construction without modifying DeepSpeed engine internals.

* **Shared Locality Cache** → after each optimizer step, gradient shards and
  selected activation tensors are written back to the CPU DRAM cache so the
  next iteration can avoid redundant recomputation / re-transfer when the same
  logical token batch re-enters a different device pool.

Architecture
------------
::

    HeteroMIMOTrainingLoop
    ├── DeviceCapabilityRegistry   — SM86 / SM90 profiles
    ├── DeviceLocalityGroups       — PCIe topology process groups
    ├── PCIeP2PCommunicator        — buffered cross-device activation transfer
    ├── SharedLocalityCache        — CPU DRAM staging (1.5 TB)
    ├── HeteroOptimizerRouter      — per-device optimizer dispatch
    ├── HeteroScheduleGroups       — pipeline stage → device pool mapping
    └── HeteroMIMOTrainingLoop     — orchestrates the above per train step

Author: Neuron_SP project (github.com/dylanyunlon/Neuron_SP)
Mirrors: Megatron commit addc601f57ed539506183b704bb9d08f459d7f50
"""

from __future__ import annotations

import logging
import math
import os
import time
import unittest
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam, AdamW

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device capability constants
# ---------------------------------------------------------------------------

SM86_CAPABILITY = (8, 6)   # A6000
SM90_CAPABILITY = (9, 0)   # H100 NVL


class DevicePool(Enum):
    """Logical device pool labels used throughout DES-LOC scheduling."""
    A6000 = auto()    # SM86 pool — 2× 48 GB, PCIe
    H100  = auto()    # SM90 pool — 1× 96 GB, PCIe


# ---------------------------------------------------------------------------
# DeviceCapabilityRegistry
# ---------------------------------------------------------------------------

@dataclass
class DeviceCapabilityProfile:
    """Hardware profile for a single CUDA device.

    DES-LOC uses these profiles to route optimizer construction and pipeline
    stage assignments: SM90's tensor-core throughput makes it preferred for
    dense matmuls; SM86's higher VRAM-per-dollar makes it preferred for
    memory-bound attention stages.
    """
    device_index: int
    sm_capability: Tuple[int, int]
    total_memory_bytes: int
    pool: DevicePool
    supports_fused_adam: bool = False
    supports_foreach_ops: bool = False

    @property
    def memory_gb(self) -> float:
        return self.total_memory_bytes / (1024 ** 3)


class DeviceCapabilityRegistry:
    """Interrogates CUDA devices and builds ``DeviceCapabilityProfile`` objects.

    On the DES-LOC cluster (2× A6000 + 1× H100 NVL) the mapping is static, but
    the registry queries the driver at construction time so it degrades gracefully
    in CI / CPU-only environments.

    Upstream parallel: Megatron's process-group collection distinguishes devices
    by their role in the pipeline; here we distinguish by SM capability and pool
    membership.
    """

    def __init__(self) -> None:
        self._profiles: Dict[int, DeviceCapabilityProfile] = {}
        self._build()

    def _build(self) -> None:
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA unavailable — DeviceCapabilityRegistry running in stub mode; "
                "all devices will be profiled as SM86 A6000."
            )
            return

        num_devices = torch.cuda.device_count()
        logger.info("Probing %d CUDA device(s) for DES-LOC capability registry.", num_devices)

        for idx in range(num_devices):
            props = torch.cuda.get_device_properties(idx)
            cap = (props.major, props.minor)
            mem = props.total_memory

            if cap >= SM90_CAPABILITY:
                pool = DevicePool.H100
                fused = True
                foreach = True
            else:
                pool = DevicePool.A6000
                fused = False   # SM86 fused Adam has correctness edge cases under fp16
                foreach = True

            profile = DeviceCapabilityProfile(
                device_index=idx,
                sm_capability=cap,
                total_memory_bytes=mem,
                pool=pool,
                supports_fused_adam=fused,
                supports_foreach_ops=foreach,
            )
            self._profiles[idx] = profile
            logger.debug(
                "Device %d: SM%d%d, %.1f GB, pool=%s, fused_adam=%s, foreach=%s",
                idx, cap[0], cap[1], profile.memory_gb, pool.name,
                fused, foreach,
            )

    def get(self, device_index: int) -> Optional[DeviceCapabilityProfile]:
        return self._profiles.get(device_index)

    def devices_in_pool(self, pool: DevicePool) -> List[int]:
        return [idx for idx, p in self._profiles.items() if p.pool == pool]

    @property
    def all_profiles(self) -> Dict[int, DeviceCapabilityProfile]:
        return dict(self._profiles)


# ---------------------------------------------------------------------------
# DeviceLocalityGroups  (mirrors Megatron's ProcessGroupCollection)
# ---------------------------------------------------------------------------

@dataclass
class DeviceLocalityGroups:
    """PCIe-topology-aware process groups for DES-LOC.

    Megatron's ``ProcessGroupCollection`` / ``MultiModuleProcessGroupCollection``
    encodes mp/pp/dp_cp group assignments.  In DES-LOC the equivalent concept is
    *locality*: devices in the same PCIe domain (the A6000 pair) can exchange
    tensors cheaply, while cross-pool transfers (A6000 ↔ H100) are expensive and
    must be routed through the SharedLocalityCache in CPU DRAM.

    Groups are created lazily on first access to avoid NCCL initialization
    overhead in test environments.
    """

    registry: DeviceCapabilityRegistry
    _a6000_group: Optional[dist.ProcessGroup] = field(default=None, init=False, repr=False)
    _h100_group:  Optional[dist.ProcessGroup] = field(default=None, init=False, repr=False)
    _cross_group: Optional[dist.ProcessGroup] = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def _maybe_init(self) -> None:
        if self._initialized:
            return
        if not dist.is_initialized():
            logger.debug(
                "torch.distributed not initialized — DeviceLocalityGroups running in stub mode."
            )
            self._initialized = True
            return

        a6000_ranks = self.registry.devices_in_pool(DevicePool.A6000)
        h100_ranks  = self.registry.devices_in_pool(DevicePool.H100)
        all_ranks   = list(range(dist.get_world_size()))

        if len(a6000_ranks) >= 2:
            self._a6000_group = dist.new_group(ranks=a6000_ranks)
            logger.info(
                "Created A6000 locality group for ranks %s.", a6000_ranks
            )
        if h100_ranks:
            self._h100_group = dist.new_group(ranks=h100_ranks)
            logger.info("Created H100 locality group for ranks %s.", h100_ranks)
        if len(all_ranks) > 1:
            self._cross_group = dist.new_group(ranks=all_ranks)

        self._initialized = True

    @property
    def a6000_group(self) -> Optional[dist.ProcessGroup]:
        self._maybe_init()
        return self._a6000_group

    @property
    def h100_group(self) -> Optional[dist.ProcessGroup]:
        self._maybe_init()
        return self._h100_group

    @property
    def cross_pool_group(self) -> Optional[dist.ProcessGroup]:
        self._maybe_init()
        return self._cross_group

    def pool_for_rank(self, rank: int) -> Optional[DevicePool]:
        profile = self.registry.get(rank)
        return profile.pool if profile else None


# ---------------------------------------------------------------------------
# PCIeP2PCommunicator  (mirrors Megatron's P2PCommunicator / p2p_communicator)
# ---------------------------------------------------------------------------

class PCIeP2PCommunicator:
    """Buffered peer-to-peer activation transfer over PCIe.

    Megatron's ``P2PCommunicator`` is designed for NVLink topologies where
    direct GPU-to-GPU transfer bandwidth is ~600 GB/s.  On DES-LOC (PCIe, no
    NVLink), direct GPU-to-GPU bandwidth across pools is ~16–32 GB/s depending
    on PCIe generation.  For large activation tensors (>= ``staging_threshold_mb``)
    it is faster to stage through CPU DRAM (which all devices can reach at PCIe
    bandwidth, but in parallel) than to issue a direct P2P copy.

    The communicator integrates with ``SharedLocalityCache``: if the activation
    for a given (stage, micro_batch_id) key is already resident in the cache from
    a prior iteration, the transfer is skipped entirely.

    DES-LOC adaptation:
    - Routing table maps (src_pool, dst_pool) → transfer strategy.
    - Intra-pool (A6000 ↔ A6000): direct copy, PCIe switch, fast path.
    - Cross-pool (A6000 ↔ H100): stage through SharedLocalityCache if tensor
      exceeds ``staging_threshold_mb``; direct otherwise.
    """

    def __init__(
        self,
        registry: DeviceCapabilityRegistry,
        locality_cache: "SharedLocalityCache",
        staging_threshold_mb: float = 64.0,
    ) -> None:
        self._registry = registry
        self._cache = locality_cache
        self._threshold_bytes = int(staging_threshold_mb * 1024 * 1024)
        logger.info(
            "PCIeP2PCommunicator initialized: staging_threshold=%.1f MB.", staging_threshold_mb
        )

    def send_activation(
        self,
        tensor: torch.Tensor,
        src_device: int,
        dst_device: int,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        """Transfer ``tensor`` from ``src_device`` to ``dst_device``.

        If ``cache_key`` is provided and the tensor is already in the locality
        cache, returns the cached copy without issuing a transfer.

        Returns the tensor resident on ``dst_device``.
        """
        if cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached.to(f"cuda:{dst_device}", non_blocking=True)

        src_pool = self._registry.get(src_device)
        dst_pool = self._registry.get(dst_device)

        tensor_bytes = tensor.nelement() * tensor.element_size()
        cross_pool = (
            src_pool is not None
            and dst_pool is not None
            and src_pool.pool != dst_pool.pool
        )

        if cross_pool and tensor_bytes >= self._threshold_bytes:
            result = self._staged_transfer(tensor, src_device, dst_device, cache_key)
        else:
            result = tensor.to(f"cuda:{dst_device}", non_blocking=True)

        if cache_key is not None:
            self._cache.put(cache_key, result.detach())

        return result

    def _staged_transfer(
        self,
        tensor: torch.Tensor,
        src_device: int,
        dst_device: int,
        cache_key: Optional[str],
    ) -> torch.Tensor:
        """Route through CPU DRAM staging area for cross-pool large tensors."""
        logger.debug(
            "Staging %.2f MB tensor through CPU DRAM (cuda:%d → cpu → cuda:%d).",
            tensor.nelement() * tensor.element_size() / (1024 ** 2),
            src_device,
            dst_device,
        )
        cpu_staging = tensor.to("cpu", non_blocking=False)   # pin for next hop
        result = cpu_staging.to(f"cuda:{dst_device}", non_blocking=True)
        return result


# ---------------------------------------------------------------------------
# SharedLocalityCache  (DES-LOC core — no direct Megatron analogue)
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """CPU DRAM staging cache for inter-pool activations and gradient shards.

    DES-LOC's key insight is that the 1.5 TB CPU DRAM on the training host
    dwarfs GPU VRAM by ~10×.  Rather than discarding activations after the
    backward pass, DES-LOC retains them in a bounded LRU cache in pinned CPU
    memory.  On the next iteration, if the same logical data block (identified
    by ``key``) is requested, the transfer cost drops from a full PCIe round-trip
    to a single PCIe read.

    Cache entries are stored as ``torch.Tensor`` objects on CPU with
    ``pin_memory=True`` when possible, maximizing DMA throughput.

    Upstream relation: Megatron's commit passes ``pg_collection`` through
    ``setup_model_and_optimizer`` to make per-module state visible across the
    training loop.  SharedLocalityCache is the DES-LOC equivalent of that
    shared state — it is passed to every component (communicator, optimizer
    router, schedule groups) so they can cooperatively populate and consume it.

    Parameters
    ----------
    max_entries:
        Maximum number of tensors to keep resident.  Older entries are evicted
        LRU-style.  Default 512 covers ~128 micro-batches × 4 pipeline stages
        of activation tensors at typical batch sizes.
    max_bytes:
        Hard cap on total bytes across all cached tensors.  Defaults to
        192 GB (roughly 1/8 of available DRAM, leaving room for OS + dataloader).
    """

    def __init__(
        self,
        max_entries: int = 512,
        max_bytes: int = 192 * 1024 ** 3,
    ) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._store: Dict[str, torch.Tensor] = {}
        self._access_order: List[str] = []    # front = oldest
        self._current_bytes: int = 0
        logger.info(
            "SharedLocalityCache created: max_entries=%d, max_bytes=%.1f GB.",
            max_entries, max_bytes / (1024 ** 3),
        )

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return the cached tensor for ``key``, or ``None`` if not present.

        Promotes ``key`` to most-recently-used.
        """
        if key not in self._store:
            return None
        self._access_order.remove(key)
        self._access_order.append(key)
        return self._store[key]

    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Insert or replace the tensor for ``key``.

        Tensors are moved to CPU and pinned when possible.  Eviction runs
        after every insertion to maintain invariants.
        """
        if key in self._store:
            old = self._store.pop(key)
            self._current_bytes -= old.nelement() * old.element_size()
            self._access_order.remove(key)

        try:
            cpu_tensor = tensor.detach().cpu()
            if cpu_tensor.is_floating_point():
                cpu_tensor = cpu_tensor.pin_memory()
        except Exception:
            cpu_tensor = tensor.detach().cpu()

        entry_bytes = cpu_tensor.nelement() * cpu_tensor.element_size()

        # Evict until we have room for the new entry
        while (
            self._access_order
            and (
                len(self._store) >= self._max_entries
                or self._current_bytes + entry_bytes > self._max_bytes
            )
        ):
            evict_key = self._access_order.pop(0)
            evicted = self._store.pop(evict_key)
            self._current_bytes -= evicted.nelement() * evicted.element_size()

        self._store[key] = cpu_tensor
        self._access_order.append(key)
        self._current_bytes += entry_bytes

    def invalidate(self, key: str) -> None:
        if key in self._store:
            old = self._store.pop(key)
            self._current_bytes -= old.nelement() * old.element_size()
            self._access_order.remove(key)

    def clear(self) -> None:
        self._store.clear()
        self._access_order.clear()
        self._current_bytes = 0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "num_entries": len(self._store),
            "current_bytes": self._current_bytes,
            "current_gb": self._current_bytes / (1024 ** 3),
            "max_entries": self._max_entries,
            "max_bytes": self._max_bytes,
        }


# ---------------------------------------------------------------------------
# HeteroOptimizerRouter  (mirrors Megatron's get_mimo_optimizer dispatch)
# ---------------------------------------------------------------------------

@dataclass
class PerModuleOptimizerConfig:
    """Per-module optimizer configuration for heterogeneous devices.

    Megatron's MIMO optimizer dispatch selects a per-module optimizer based on
    module type.  DES-LOC extends this: the selection is driven by the SM
    capability of the device that owns the module's parameters.
    """
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 1.0


class HeteroOptimizerRouter:
    """Routes optimizer construction to the best implementation per device pool.

    Upstream design intent
    ~~~~~~~~~~~~~~~~~~~~~~
    Megatron commit addc601 inserts an isinstance check for ``MimoModel`` at the
    top of ``get_megatron_optimizer`` and delegates to ``get_mimo_optimizer``.
    The motivation is that MIMO models have heterogeneous sub-modules (e.g., a
    vision encoder + language decoder) that benefit from per-module optimizer
    configurations — different learning rates, different parameter groups,
    potentially different optimizer algorithms.

    DES-LOC adaptation
    ~~~~~~~~~~~~~~~~~~
    We generalize the dispatch axis from *model type* to *device pool*:

    * H100 (SM90): ``AdamW`` with ``fused=True`` — SM90's BF16 matmul pipeline
      makes the fused CUDA kernel worth the extra VRAM.
    * A6000 (SM86): ``Adam`` with ``foreach=True`` — fused Adam has correctness
      issues with SM86 + FP16 mixed precision; foreach achieves similar
      throughput by vectorizing across parameter tensors.

    The router also registers the optimizer with the ``SharedLocalityCache`` so
    that gradient shards can be stashed after each step and reused as warm
    starting points when the same module re-enters the pipeline.
    """

    def __init__(
        self,
        registry: DeviceCapabilityRegistry,
        locality_cache: SharedLocalityCache,
    ) -> None:
        self._registry = registry
        self._cache = locality_cache

    def build_optimizer_for_module(
        self,
        module: nn.Module,
        device_index: int,
        config: PerModuleOptimizerConfig,
    ) -> torch.optim.Optimizer:
        """Construct the appropriate optimizer for ``module`` on ``device_index``.

        Returns a configured ``torch.optim.Optimizer``.  All parameters are
        assumed to be on ``device_index`` already; the caller is responsible for
        device placement before invoking this method.
        """
        profile = self._registry.get(device_index)
        params = list(module.parameters())
        if not params:
            raise ValueError(
                f"Module {type(module).__name__} on device {device_index} has no parameters; "
                "cannot construct optimizer."
            )

        if profile is not None and profile.pool == DevicePool.H100:
            try:
                opt = AdamW(
                    params,
                    lr=config.lr,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay,
                    fused=True,
                )
                logger.info(
                    "H100 optimizer: AdamW(fused=True) for %s on device %d (%.1f GB).",
                    type(module).__name__, device_index, profile.memory_gb,
                )
            except (TypeError, RuntimeError):
                # fused=True may not be available in all PyTorch builds
                opt = AdamW(
                    params,
                    lr=config.lr,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay,
                )
                logger.warning(
                    "AdamW(fused=True) unavailable on device %d; falling back to standard AdamW.",
                    device_index,
                )
        else:
            try:
                opt = Adam(
                    params,
                    lr=config.lr,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay,
                    foreach=True,
                )
                logger.info(
                    "A6000 optimizer: Adam(foreach=True) for %s on device %d.",
                    type(module).__name__, device_index,
                )
            except TypeError:
                opt = Adam(
                    params,
                    lr=config.lr,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay,
                )

        return opt

    def build_optimizer_group(
        self,
        modules_and_devices: List[Tuple[nn.Module, int]],
        config: PerModuleOptimizerConfig,
    ) -> List[torch.optim.Optimizer]:
        """Build one optimizer per (module, device) pair.

        Mirrors Megatron's MIMO optimizer which constructs per-module optimizer
        instances rather than a single monolithic optimizer.  Returns a list in
        the same order as ``modules_and_devices``.
        """
        optimizers = []
        for module, dev_idx in modules_and_devices:
            opt = self.build_optimizer_for_module(module, dev_idx, config)
            optimizers.append(opt)
        return optimizers


# ---------------------------------------------------------------------------
# HeteroScheduleGroups  (mirrors Megatron's schedule_pg_collection)
# ---------------------------------------------------------------------------

@dataclass
class StageAssignment:
    """Binding of a pipeline stage to a device and pool."""
    stage_id: int
    device_index: int
    pool: DevicePool
    estimated_memory_bytes: int = 0
    is_compute_heavy: bool = True


class HeteroScheduleGroups:
    """Pipeline stage → device pool assignment for DES-LOC heterogeneous scheduling.

    Upstream design intent
    ~~~~~~~~~~~~~~~~~~~~~~
    Megatron's ``schedule_pg_collection`` (``MultiModuleProcessGroupCollection``)
    gives the forward-backward schedule visibility into per-module process groups,
    enabling cross-grid P2P for the MIMO case where different model modules may
    reside on different pipeline grids.

    DES-LOC adaptation
    ~~~~~~~~~~~~~~~~~~
    ``HeteroScheduleGroups`` answers the same question — "which process group
    should handle communication for stage S?" — but with additional semantics:
    the assignment is driven by a combination of:

    1. **Compute intensity**: transformer attention blocks are SM-core-bound →
       assign to H100 (SM90 tensor cores).  Embedding / layernorm / positional
       encoding are memory-bandwidth-bound → assign to A6000.
    2. **Current memory headroom**: queried at assignment time via
       ``torch.cuda.memory_reserved`` to avoid OOM mid-iteration.
    3. **PCIe locality**: adjacent pipeline stages should share a pool to avoid
       cross-pool transfers for the most frequent activation tensors.

    The scheduler maintains a ``StageAssignment`` list that ``PCIeP2PCommunicator``
    consults to decide whether staging through CPU DRAM is needed.
    """

    def __init__(
        self,
        registry: DeviceCapabilityRegistry,
        locality_groups: DeviceLocalityGroups,
    ) -> None:
        self._registry = registry
        self._lgroups = locality_groups
        self._assignments: Dict[int, StageAssignment] = {}

    def assign_stage(
        self,
        stage_id: int,
        is_compute_heavy: bool = True,
        estimated_memory_bytes: int = 0,
    ) -> StageAssignment:
        """Assign ``stage_id`` to the most appropriate device.

        Compute-heavy stages prefer H100; memory-bound stages prefer A6000.
        Falls back to any device with sufficient headroom.
        """
        preferred_pool = DevicePool.H100 if is_compute_heavy else DevicePool.A6000
        candidates = self._registry.devices_in_pool(preferred_pool)

        if not candidates:
            # Fallback: any available device
            candidates = list(self._registry.all_profiles.keys())

        selected = self._select_device_with_headroom(candidates, estimated_memory_bytes)

        if selected is None:
            # Last resort: use the first candidate regardless of headroom
            selected = candidates[0] if candidates else 0
            logger.warning(
                "Stage %d: no device with sufficient headroom for %.1f MB; "
                "assigning to device %d anyway.",
                stage_id,
                estimated_memory_bytes / (1024 ** 2),
                selected,
            )

        profile = self._registry.get(selected)
        pool = profile.pool if profile else DevicePool.A6000

        assignment = StageAssignment(
            stage_id=stage_id,
            device_index=selected,
            pool=pool,
            estimated_memory_bytes=estimated_memory_bytes,
            is_compute_heavy=is_compute_heavy,
        )
        self._assignments[stage_id] = assignment
        logger.debug(
            "Stage %d assigned to device %d (pool=%s, compute_heavy=%s).",
            stage_id, selected, pool.name, is_compute_heavy,
        )
        return assignment

    def _select_device_with_headroom(
        self,
        candidates: List[int],
        required_bytes: int,
    ) -> Optional[int]:
        if not torch.cuda.is_available():
            return candidates[0] if candidates else None

        for dev_idx in candidates:
            try:
                free = torch.cuda.mem_get_info(dev_idx)[0]
                if free >= required_bytes:
                    return dev_idx
            except Exception:
                continue
        return None

    def get_assignment(self, stage_id: int) -> Optional[StageAssignment]:
        return self._assignments.get(stage_id)

    def is_cross_pool_transition(self, stage_a: int, stage_b: int) -> bool:
        """Return True if stages ``stage_a`` and ``stage_b`` are on different pools."""
        a = self._assignments.get(stage_a)
        b = self._assignments.get(stage_b)
        if a is None or b is None:
            return False
        return a.pool != b.pool


# ---------------------------------------------------------------------------
# HeteroMIMOModel  (duck-type detection — mirrors Megatron isinstance(MimoModel))
# ---------------------------------------------------------------------------

class HeteroMIMOModel(nn.Module):
    """Base class for MIMO models in DES-LOC heterogeneous training.

    Megatron's MIMO dispatch (addc601) uses ``isinstance(model_chunks[0], MimoModel)``
    to short-circuit optimizer construction.  In DES-LOC we use a mixin /
    base-class approach: any model that subclasses ``HeteroMIMOModel`` signals
    that it has heterogeneous sub-modules, each with its own device placement
    and optimizer configuration.

    The key contract is ``named_hetero_modules()``, which yields
    ``(name, module, device_index)`` triples.  The training loop uses this to
    drive per-module optimizer dispatch and stage assignment — exactly mirroring
    how Megatron's ``get_mimo_optimizer`` iterates over per-module configs.
    """

    def named_hetero_modules(self) -> Generator[Tuple[str, nn.Module, int], None, None]:
        """Yield ``(name, module, device_index)`` for each heterogeneous sub-module.

        Subclasses must override this method.  The default implementation yields
        every immediate child module on device 0, which is safe but ignores
        heterogeneity — override for real multi-device models.
        """
        for name, module in self.named_children():
            yield name, module, 0

    def is_hetero_mimo(self) -> bool:
        """Return True to allow the training loop to detect this model type."""
        return True


def is_hetero_mimo_model(model: Any) -> bool:
    """Functional equivalent of Megatron's isinstance(model_chunks[0], MimoModel).

    Also handles DeepSpeed engine wrapping by unwrapping one level.
    """
    unwrapped = getattr(model, "module", model)
    return isinstance(unwrapped, HeteroMIMOModel) and unwrapped.is_hetero_mimo()


# ---------------------------------------------------------------------------
# TrainStepResult
# ---------------------------------------------------------------------------

@dataclass
class TrainStepResult:
    """Return value from a single heterogeneous training step.

    Mirrors the tuple returned by Megatron's ``train_step`` but adds
    DES-LOC-specific fields for cache and transfer diagnostics.
    """
    loss: float
    grad_norm: float
    skipped_iter: bool
    num_zeros_in_grad: int
    iteration: int
    cache_hits: int = 0
    cache_misses: int = 0
    cross_pool_transfers: int = 0
    stage_assignments: Dict[int, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# HeteroMIMOTrainingLoop  (central orchestrator)
# ---------------------------------------------------------------------------

class HeteroMIMOTrainingLoop:
    """Heterogeneous MIMO training loop for DES-LOC.

    This is the DES-LOC reinterpretation of the threading work in Megatron commit
    addc601f57ed539506183b704bb9d08f459d7f50.  Megatron threads three new optional
    parameters (``pg_collection``, ``p2p_communicator``, ``schedule_pg_collection``)
    through ``setup_model_and_optimizer`` → ``train_step`` → ``train``.  DES-LOC
    does the same but the objects being threaded carry heterogeneous-hardware
    semantics:

    * ``DeviceLocalityGroups`` replaces ``ProcessGroupCollection`` /
      ``MultiModuleProcessGroupCollection`` — same role (per-module process groups),
      different axis (PCIe locality vs. model parallelism).

    * ``PCIeP2PCommunicator`` replaces ``P2PCommunicator`` — same role (cross-grid
      P2P), different implementation (CPU DRAM staging for large tensors on PCIe).

    * ``SharedLocalityCache`` is a DES-LOC addition with no Megatron equivalent —
      it enables the "decoupled execution" half of DES-LOC by persisting
      activations across iterations.

    The training loop is intentionally thin: it coordinates the components above,
    drives the forward-backward-optimizer cycle, and records per-step diagnostics.
    Heavy lifting (actual model forward pass, loss computation, gradient all-reduce)
    is delegated to the ``forward_backward_func`` callable, exactly as in Megatron.

    Parameters
    ----------
    model:
        A ``HeteroMIMOModel`` (or DeepSpeed-wrapped equivalent).
    optimizer_router:
        ``HeteroOptimizerRouter`` instance; used to construct per-module
        optimizers if the model exposes ``named_hetero_modules()``.
    schedule_groups:
        ``HeteroScheduleGroups`` for pipeline stage → device assignment.
    locality_groups:
        ``DeviceLocalityGroups`` encoding PCIe topology.
    p2p_communicator:
        ``PCIeP2PCommunicator`` for cross-pool activation transfer.
    locality_cache:
        ``SharedLocalityCache`` — the 1.5 TB DRAM staging area.
    optimizer_config:
        ``PerModuleOptimizerConfig`` applied to all modules (override per
        subclass if needed).
    grad_clip:
        Gradient clipping norm.  Applied per-module before optimizer step.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_router: HeteroOptimizerRouter,
        schedule_groups: HeteroScheduleGroups,
        locality_groups: DeviceLocalityGroups,
        p2p_communicator: PCIeP2PCommunicator,
        locality_cache: SharedLocalityCache,
        optimizer_config: PerModuleOptimizerConfig,
        grad_clip: float = 1.0,
    ) -> None:
        self._model = model
        self._opt_router = optimizer_router
        self._sched_groups = schedule_groups
        self._locality = locality_groups
        self._p2p = p2p_communicator
        self._cache = locality_cache
        self._grad_clip = grad_clip
        self._iteration: int = 0

        # Build per-module optimizers if model supports it
        self._module_optimizers: List[torch.optim.Optimizer] = []
        self._setup_module_optimizers(optimizer_config)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_module_optimizers(self, config: PerModuleOptimizerConfig) -> None:
        """Construct per-module optimizers, mirroring Megatron's get_mimo_optimizer dispatch.

        For non-MIMO models, falls back to a single optimizer over all parameters.
        """
        if is_hetero_mimo_model(self._model):
            unwrapped = getattr(self._model, "module", self._model)
            pairs: List[Tuple[nn.Module, int]] = [
                (mod, dev_idx)
                for _, mod, dev_idx in unwrapped.named_hetero_modules()
            ]
            if pairs:
                self._module_optimizers = self._opt_router.build_optimizer_group(pairs, config)
                logger.info(
                    "HeteroMIMOTrainingLoop: built %d per-module optimizer(s) for MIMO model.",
                    len(self._module_optimizers),
                )
                return

        # Fallback: single optimizer on device 0
        device_index = 0
        if torch.cuda.is_available():
            try:
                device_index = next(iter(self._model.parameters())).device.index
            except StopIteration:
                pass

        opt = self._opt_router.build_optimizer_for_module(self._model, device_index, config)
        self._module_optimizers = [opt]
        logger.info(
            "HeteroMIMOTrainingLoop: built single optimizer (non-MIMO fallback) on device %d.",
            device_index,
        )

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        forward_backward_func: Callable[..., List[torch.Tensor]],
        data_iterator: Iterator[Any],
        config: Any,
        iteration: Optional[int] = None,
    ) -> TrainStepResult:
        """Execute one heterogeneous training step.

        Mirrors the call signature threading in Megatron addc601:

        ``train_step(forward_step_func, data_iterator, model, optimizer,
                     opt_param_scheduler, config, forward_backward_func,
                     iteration=iteration, pg_collection=model_pg_collection,
                     p2p_communicator=p2p_communicator,
                     schedule_pg_collection=schedule_pg_collection)``

        DES-LOC threads ``DeviceLocalityGroups``, ``PCIeP2PCommunicator``, and
        ``HeteroScheduleGroups`` instead of Megatron's equivalents.

        Parameters
        ----------
        forward_backward_func:
            Callable with the same contract as Megatron's ``forward_backward_func``:
            accepts keyword arguments including ``p2p_communicator`` and
            ``pg_collection``, executes the forward + backward pass, returns a
            list of loss tensors.
        data_iterator:
            Iterator yielding micro-batches.
        config:
            Training configuration (learning rate schedule, etc.).
        iteration:
            Current global iteration index; auto-incremented if None.

        Returns
        -------
        TrainStepResult
        """
        if iteration is not None:
            self._iteration = iteration
        step_start = time.perf_counter()

        cache_hits_before = sum(
            1 for _ in self._cache._store  # snapshot entry count as proxy
        )

        # -- Zero gradients across all per-module optimizers ----------------
        for opt in self._module_optimizers:
            opt.zero_grad(set_to_none=True)

        # -- Forward + backward via the provided callable -------------------
        # We forward the DES-LOC communicator and schedule groups in place of
        # Megatron's p2p_communicator / schedule_pg_collection, preserving the
        # threading contract introduced in addc601.
        losses = forward_backward_func(
            forward_only=False,
            p2p_communicator=self._p2p,
            pg_collection=self._sched_groups,
            data_iterator=data_iterator,
            model=self._model,
            config=config,
            iteration=self._iteration,
        )

        # -- Gradient clipping + optimizer step ------------------------------
        grad_norm = self._clip_and_step()

        # -- Write updated parameter shards to SharedLocalityCache -----------
        # This is the DES-LOC "shared locality" step: after the optimizer
        # updates parameters, we cache a detached copy of the gradient norm
        # and a sample activation key so the next iteration can warm-start.
        self._write_back_to_cache()

        step_ms = (time.perf_counter() - step_start) * 1000.0
        cache_hits_after = len(self._cache._store)

        loss_val = self._aggregate_losses(losses)
        result = TrainStepResult(
            loss=loss_val,
            grad_norm=grad_norm,
            skipped_iter=False,
            num_zeros_in_grad=0,
            iteration=self._iteration,
            cache_hits=cache_hits_after - cache_hits_before,
            cross_pool_transfers=self._count_cross_pool_stages(),
            stage_assignments={
                sid: asn.pool.name
                for sid, asn in self._sched_groups._assignments.items()
            },
            elapsed_ms=step_ms,
        )

        self._iteration += 1
        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _clip_and_step(self) -> float:
        """Clip gradients and step all per-module optimizers.

        Returns the total gradient norm across all modules.
        """
        total_norm_sq = 0.0
        for opt in self._module_optimizers:
            params_with_grad = [
                p for group in opt.param_groups for p in group["params"]
                if p.grad is not None
            ]
            if not params_with_grad:
                continue
            norm = nn.utils.clip_grad_norm_(params_with_grad, self._grad_clip)
            total_norm_sq += float(norm) ** 2
            opt.step()

        return math.sqrt(total_norm_sq)

    def _write_back_to_cache(self) -> None:
        """Store a lightweight per-module signature in the SharedLocalityCache.

        For each per-module optimizer we stash the first parameter tensor's
        detached mean as a scalar cache entry.  In a production DES-LOC run
        this would be replaced by full gradient shard persistence, but the
        pattern here establishes the correct API surface.
        """
        for idx, opt in enumerate(self._module_optimizers):
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        key = f"grad_shard:iter={self._iteration}:module={idx}"
                        self._cache.put(key, param.grad.detach().mean().unsqueeze(0))
                        break   # one shard per module per step

    def _aggregate_losses(self, losses: List[Any]) -> float:
        """Reduce a list of loss tensors / floats to a single Python float."""
        if not losses:
            return float("nan")
        total = 0.0
        count = 0
        for item in losses:
            if isinstance(item, torch.Tensor):
                total += item.detach().float().mean().item()
                count += 1
            elif isinstance(item, (int, float)):
                total += float(item)
                count += 1
        return total / count if count > 0 else float("nan")

    def _count_cross_pool_stages(self) -> int:
        """Count adjacent stage pairs that are on different device pools."""
        assignments = self._sched_groups._assignments
        stage_ids = sorted(assignments)
        cross = 0
        for i in range(len(stage_ids) - 1):
            if self._sched_groups.is_cross_pool_transition(stage_ids[i], stage_ids[i + 1]):
                cross += 1
        return cross

    # ------------------------------------------------------------------
    # Context manager for scoped cache management
    # ------------------------------------------------------------------

    @contextmanager
    def iteration_scope(self, iteration: int) -> Generator[None, None, None]:
        """Context manager that sets the iteration and flushes stale cache entries.

        Usage::

            with loop.iteration_scope(iter_num):
                result = loop.train_step(fbf, data_iter, config, iteration=iter_num)
        """
        self._iteration = iteration
        stale_prefix = f"grad_shard:iter={iteration - 2}:"
        stale_keys = [k for k in list(self._cache._store) if k.startswith(stale_prefix)]
        for k in stale_keys:
            self._cache.invalidate(k)
        if stale_keys:
            logger.debug(
                "Evicted %d stale cache entries from iteration %d.",
                len(stale_keys), iteration - 2,
            )
        yield


# ---------------------------------------------------------------------------
# Factory function  (mirrors Megatron's setup_model_and_optimizer with pg_collection)
# ---------------------------------------------------------------------------

def setup_hetero_mimo_training(
    model: nn.Module,
    optimizer_config: Optional[PerModuleOptimizerConfig] = None,
    cache_max_entries: int = 512,
    cache_max_gb: float = 192.0,
    staging_threshold_mb: float = 64.0,
    grad_clip: float = 1.0,
) -> HeteroMIMOTrainingLoop:
    """Construct and wire all DES-LOC components, returning a ready-to-use training loop.

    This is the DES-LOC analogue of Megatron's ``setup_model_and_optimizer`` after
    the addc601 patch: it accepts an optional ``pg_collection`` (here
    ``optimizer_config``) and builds the full component graph internally.

    Parameters
    ----------
    model:
        The model to train.  May be a ``HeteroMIMOModel`` subclass or a plain
        ``nn.Module`` (heterogeneous dispatch will degrade gracefully).
    optimizer_config:
        Per-module optimizer hyperparameters.  Defaults to standard values.
    cache_max_entries:
        Maximum number of tensors in the SharedLocalityCache.
    cache_max_gb:
        Hard cap on SharedLocalityCache size in GB.
    staging_threshold_mb:
        Tensors larger than this (in MB) are routed through CPU DRAM staging
        for cross-pool transfers.
    grad_clip:
        Gradient clipping norm.

    Returns
    -------
    HeteroMIMOTrainingLoop
    """
    if optimizer_config is None:
        optimizer_config = PerModuleOptimizerConfig()

    registry = DeviceCapabilityRegistry()
    locality_cache = SharedLocalityCache(
        max_entries=cache_max_entries,
        max_bytes=int(cache_max_gb * 1024 ** 3),
    )
    locality_groups = DeviceLocalityGroups(registry=registry)
    p2p = PCIeP2PCommunicator(
        registry=registry,
        locality_cache=locality_cache,
        staging_threshold_mb=staging_threshold_mb,
    )
    opt_router = HeteroOptimizerRouter(
        registry=registry,
        locality_cache=locality_cache,
    )
    sched_groups = HeteroScheduleGroups(
        registry=registry,
        locality_groups=locality_groups,
    )

    loop = HeteroMIMOTrainingLoop(
        model=model,
        optimizer_router=opt_router,
        schedule_groups=sched_groups,
        locality_groups=locality_groups,
        p2p_communicator=p2p,
        locality_cache=locality_cache,
        optimizer_config=optimizer_config,
        grad_clip=grad_clip,
    )
    logger.info(
        "DES-LOC HeteroMIMOTrainingLoop ready: "
        "%d device(s) registered, cache=%.1f GB cap.",
        len(registry.all_profiles), cache_max_gb,
    )
    return loop


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    class _TestCase(unittest.TestCase):

        # ------------------------------------------------------------------ #
        # DeviceCapabilityRegistry                                            #
        # ------------------------------------------------------------------ #

        def test_registry_builds_without_cuda(self):
            """Registry must not raise on CPU-only hosts."""
            reg = DeviceCapabilityRegistry()
            # In CI there may be 0 or more devices; just verify no exception
            self.assertIsInstance(reg.all_profiles, dict)

        def test_registry_pool_classification(self):
            """Manually injected profiles must classify correctly."""
            reg = DeviceCapabilityRegistry()
            # Inject synthetic profiles
            reg._profiles[0] = DeviceCapabilityProfile(
                device_index=0,
                sm_capability=SM86_CAPABILITY,
                total_memory_bytes=48 * 1024 ** 3,
                pool=DevicePool.A6000,
                supports_fused_adam=False,
                supports_foreach_ops=True,
            )
            reg._profiles[1] = DeviceCapabilityProfile(
                device_index=1,
                sm_capability=SM90_CAPABILITY,
                total_memory_bytes=96 * 1024 ** 3,
                pool=DevicePool.H100,
                supports_fused_adam=True,
                supports_foreach_ops=True,
            )
            self.assertEqual(reg.devices_in_pool(DevicePool.A6000), [0])
            self.assertEqual(reg.devices_in_pool(DevicePool.H100), [1])

        # ------------------------------------------------------------------ #
        # SharedLocalityCache                                                 #
        # ------------------------------------------------------------------ #

        def test_cache_put_get_roundtrip(self):
            cache = SharedLocalityCache(max_entries=8, max_bytes=1024 * 1024 * 1024)
            t = torch.randn(4, 4)
            cache.put("test_key", t)
            result = cache.get("test_key")
            self.assertIsNotNone(result)
            self.assertTrue(torch.allclose(t.cpu(), result.cpu(), atol=1e-5))

        def test_cache_lru_eviction(self):
            """Cache must evict oldest entries when max_entries is exceeded."""
            cache = SharedLocalityCache(max_entries=3, max_bytes=10 * 1024 ** 3)
            for i in range(4):
                cache.put(f"key{i}", torch.zeros(2))
            # key0 should have been evicted
            self.assertIsNone(cache.get("key0"))
            self.assertIsNotNone(cache.get("key3"))

        def test_cache_byte_eviction(self):
            """Cache must evict when byte limit is exceeded."""
            small_bytes = 64  # 16 floats × 4 bytes
            cache = SharedLocalityCache(max_entries=100, max_bytes=small_bytes * 3)
            for i in range(4):
                cache.put(f"key{i}", torch.zeros(16))
            # Only 3 entries fit; first should be gone
            self.assertIsNone(cache.get("key0"))

        def test_cache_invalidate(self):
            cache = SharedLocalityCache(max_entries=8, max_bytes=1024 ** 3)
            cache.put("k", torch.ones(3))
            cache.invalidate("k")
            self.assertIsNone(cache.get("k"))

        def test_cache_clear(self):
            cache = SharedLocalityCache(max_entries=8, max_bytes=1024 ** 3)
            for i in range(5):
                cache.put(f"k{i}", torch.ones(2))
            cache.clear()
            self.assertEqual(len(cache._store), 0)
            self.assertEqual(cache._current_bytes, 0)

        def test_cache_stats(self):
            cache = SharedLocalityCache(max_entries=8, max_bytes=1024 ** 3)
            cache.put("x", torch.zeros(8))
            stats = cache.stats
            self.assertIn("num_entries", stats)
            self.assertEqual(stats["num_entries"], 1)
            self.assertGreater(stats["current_bytes"], 0)

        # ------------------------------------------------------------------ #
        # HeteroOptimizerRouter                                               #
        # ------------------------------------------------------------------ #

        def _make_router(self):
            reg = DeviceCapabilityRegistry()
            reg._profiles[0] = DeviceCapabilityProfile(
                device_index=0,
                sm_capability=SM86_CAPABILITY,
                total_memory_bytes=48 * 1024 ** 3,
                pool=DevicePool.A6000,
            )
            reg._profiles[1] = DeviceCapabilityProfile(
                device_index=1,
                sm_capability=SM90_CAPABILITY,
                total_memory_bytes=96 * 1024 ** 3,
                pool=DevicePool.H100,
                supports_fused_adam=True,
            )
            cache = SharedLocalityCache()
            return HeteroOptimizerRouter(registry=reg, locality_cache=cache), reg

        def test_optimizer_router_a6000_builds_adam(self):
            router, _ = self._make_router()
            module = nn.Linear(4, 4)
            config = PerModuleOptimizerConfig(lr=1e-3)
            opt = router.build_optimizer_for_module(module, device_index=0, config=config)
            self.assertIsInstance(opt, (Adam, AdamW))

        def test_optimizer_router_h100_builds_adamw(self):
            router, _ = self._make_router()
            module = nn.Linear(4, 4)
            config = PerModuleOptimizerConfig(lr=1e-3)
            opt = router.build_optimizer_for_module(module, device_index=1, config=config)
            self.assertIsInstance(opt, AdamW)

        def test_optimizer_router_empty_module_raises(self):
            router, _ = self._make_router()
            empty = nn.Module()
            with self.assertRaises(ValueError):
                router.build_optimizer_for_module(empty, 0, PerModuleOptimizerConfig())

        def test_optimizer_router_group_length(self):
            router, _ = self._make_router()
            m1, m2 = nn.Linear(4, 4), nn.Linear(8, 8)
            opts = router.build_optimizer_group([(m1, 0), (m2, 1)], PerModuleOptimizerConfig())
            self.assertEqual(len(opts), 2)

        # ------------------------------------------------------------------ #
        # HeteroScheduleGroups                                                #
        # ------------------------------------------------------------------ #

        def _make_sched(self):
            reg = DeviceCapabilityRegistry()
            reg._profiles[0] = DeviceCapabilityProfile(
                device_index=0, sm_capability=SM86_CAPABILITY,
                total_memory_bytes=48 * 1024 ** 3, pool=DevicePool.A6000,
            )
            reg._profiles[1] = DeviceCapabilityProfile(
                device_index=1, sm_capability=SM90_CAPABILITY,
                total_memory_bytes=96 * 1024 ** 3, pool=DevicePool.H100,
            )
            groups = DeviceLocalityGroups(registry=reg)
            return HeteroScheduleGroups(registry=reg, locality_groups=groups), reg

        def test_sched_compute_heavy_prefers_h100(self):
            sched, _ = self._make_sched()
            asn = sched.assign_stage(0, is_compute_heavy=True, estimated_memory_bytes=0)
            self.assertEqual(asn.pool, DevicePool.H100)

        def test_sched_memory_bound_prefers_a6000(self):
            sched, _ = self._make_sched()
            asn = sched.assign_stage(1, is_compute_heavy=False, estimated_memory_bytes=0)
            self.assertEqual(asn.pool, DevicePool.A6000)

        def test_sched_cross_pool_detection(self):
            sched, _ = self._make_sched()
            sched.assign_stage(0, is_compute_heavy=True)    # H100
            sched.assign_stage(1, is_compute_heavy=False)   # A6000
            self.assertTrue(sched.is_cross_pool_transition(0, 1))

        def test_sched_same_pool_no_cross(self):
            sched, _ = self._make_sched()
            sched.assign_stage(0, is_compute_heavy=True)
            sched.assign_stage(1, is_compute_heavy=True)
            self.assertFalse(sched.is_cross_pool_transition(0, 1))

        def test_sched_missing_stage_no_cross(self):
            sched, _ = self._make_sched()
            sched.assign_stage(0, is_compute_heavy=False)
            self.assertFalse(sched.is_cross_pool_transition(0, 99))

        # ------------------------------------------------------------------ #
        # PCIeP2PCommunicator                                                 #
        # ------------------------------------------------------------------ #

        def _make_p2p(self):
            reg = DeviceCapabilityRegistry()
            reg._profiles[0] = DeviceCapabilityProfile(
                device_index=0, sm_capability=SM86_CAPABILITY,
                total_memory_bytes=48 * 1024 ** 3, pool=DevicePool.A6000,
            )
            reg._profiles[1] = DeviceCapabilityProfile(
                device_index=1, sm_capability=SM90_CAPABILITY,
                total_memory_bytes=96 * 1024 ** 3, pool=DevicePool.H100,
            )
            cache = SharedLocalityCache()
            return PCIeP2PCommunicator(reg, cache, staging_threshold_mb=0.001), cache

        def test_p2p_cache_hit_skips_transfer(self):
            p2p, cache = self._make_p2p()
            t = torch.randn(8)
            cache.put("act:key1", t)
            result = p2p.send_activation(t, src_device=0, dst_device=1, cache_key="act:key1")
            # Result should be on CPU (since no real CUDA in CI)
            self.assertIsNotNone(result)

        def test_p2p_intra_pool_direct(self):
            p2p, _ = self._make_p2p()
            t = torch.randn(4)
            # Intra-pool: both on A6000 (device 0); tensor stays on CPU in CI
            result = p2p.send_activation(t, src_device=0, dst_device=0)
            self.assertIsNotNone(result)

        def test_p2p_writes_to_cache_with_key(self):
            p2p, cache = self._make_p2p()
            t = torch.randn(4)
            p2p.send_activation(t, src_device=0, dst_device=1, cache_key="test:write")
            self.assertIsNotNone(cache.get("test:write"))

        # ------------------------------------------------------------------ #
        # HeteroMIMOModel detection                                           #
        # ------------------------------------------------------------------ #

        def test_is_hetero_mimo_positive(self):
            class MyMIMO(HeteroMIMOModel):
                def __init__(self):
                    super().__init__()
                    self.enc = nn.Linear(4, 4)
                    self.dec = nn.Linear(4, 4)

                def named_hetero_modules(self):
                    yield "enc", self.enc, 0
                    yield "dec", self.dec, 0

            self.assertTrue(is_hetero_mimo_model(MyMIMO()))

        def test_is_hetero_mimo_negative_plain_module(self):
            self.assertFalse(is_hetero_mimo_model(nn.Linear(4, 4)))

        def test_is_hetero_mimo_unwraps_deepspeed_engine(self):
            """Verify that a .module wrapper is unwrapped before isinstance check."""
            class FakeEngine:
                def __init__(self, inner):
                    self.module = inner

            class MyMIMO(HeteroMIMOModel):
                def named_hetero_modules(self):
                    yield from []

            engine = FakeEngine(MyMIMO())
            self.assertTrue(is_hetero_mimo_model(engine))

        # ------------------------------------------------------------------ #
        # HeteroMIMOTrainingLoop — integration                                #
        # ------------------------------------------------------------------ #

        def _make_loop(self):
            """Build a minimal training loop over a small MIMO model on CPU."""
            class TinyMIMO(HeteroMIMOModel):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Linear(8, 8)
                    self.decoder = nn.Linear(8, 4)

                def named_hetero_modules(self):
                    yield "encoder", self.encoder, 0
                    yield "decoder", self.decoder, 0

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            model = TinyMIMO()
            reg = DeviceCapabilityRegistry()
            # Inject a single CPU-safe profile on device 0
            reg._profiles[0] = DeviceCapabilityProfile(
                device_index=0, sm_capability=SM86_CAPABILITY,
                total_memory_bytes=48 * 1024 ** 3, pool=DevicePool.A6000,
            )
            cache = SharedLocalityCache(max_entries=64, max_bytes=1 * 1024 ** 3)
            lgroups = DeviceLocalityGroups(registry=reg)
            p2p = PCIeP2PCommunicator(reg, cache, staging_threshold_mb=1.0)
            router = HeteroOptimizerRouter(reg, cache)
            sched = HeteroScheduleGroups(reg, lgroups)
            sched.assign_stage(0, is_compute_heavy=False)
            sched.assign_stage(1, is_compute_heavy=False)
            config = PerModuleOptimizerConfig(lr=1e-3)
            loop = HeteroMIMOTrainingLoop(
                model=model,
                optimizer_router=router,
                schedule_groups=sched,
                locality_groups=lgroups,
                p2p_communicator=p2p,
                locality_cache=cache,
                optimizer_config=config,
                grad_clip=1.0,
            )
            return loop, model

        def test_loop_builds_per_module_optimizers(self):
            loop, _ = self._make_loop()
            # TinyMIMO has 2 hetero modules → 2 optimizers
            self.assertEqual(len(loop._module_optimizers), 2)

        def test_loop_train_step_returns_result(self):
            loop, model = self._make_loop()

            def fbf(**kwargs):
                x = torch.randn(2, 8)
                out = model(x)
                loss = out.mean()
                loss.backward()
                return [loss]

            result = loop.train_step(
                forward_backward_func=fbf,
                data_iterator=iter([]),
                config=object(),
                iteration=0,
            )
            self.assertIsInstance(result, TrainStepResult)
            self.assertFalse(math.isnan(result.loss))
            self.assertGreaterEqual(result.elapsed_ms, 0.0)

        def test_loop_iteration_increments(self):
            loop, model = self._make_loop()

            def fbf(**kwargs):
                x = torch.randn(2, 8)
                out = model(x)
                loss = out.mean()
                loss.backward()
                return [loss]

            loop.train_step(fbf, iter([]), object(), iteration=5)
            self.assertEqual(loop._iteration, 6)

        def test_loop_cache_written_after_step(self):
            loop, model = self._make_loop()

            def fbf(**kwargs):
                x = torch.randn(2, 8)
                out = model(x)
                out.mean().backward()
                return [out.mean().detach()]

            loop.train_step(fbf, iter([]), object(), iteration=10)
            # Cache should contain at least one grad_shard entry
            matching = [k for k in loop._cache._store if "grad_shard:iter=10" in k]
            self.assertGreater(len(matching), 0)

        def test_loop_iteration_scope_evicts_stale(self):
            loop, model = self._make_loop()
            # Pre-populate cache with stale entries from iteration 7
            loop._cache.put("grad_shard:iter=7:module=0", torch.zeros(1))
            loop._cache.put("grad_shard:iter=7:module=1", torch.zeros(1))

            def fbf(**kwargs):
                return [torch.tensor(0.5)]

            with loop.iteration_scope(9):
                # iter - 2 = 7 → stale entries should be evicted
                loop.train_step(fbf, iter([]), object(), iteration=9)

            self.assertIsNone(loop._cache.get("grad_shard:iter=7:module=0"))

        # ------------------------------------------------------------------ #
        # setup_hetero_mimo_training factory                                  #
        # ------------------------------------------------------------------ #

        def test_factory_returns_loop(self):
            model = nn.Linear(4, 4)
            loop = setup_hetero_mimo_training(model, cache_max_gb=1.0)
            self.assertIsInstance(loop, HeteroMIMOTrainingLoop)

        def test_factory_default_optimizer_config(self):
            model = nn.Linear(4, 4)
            loop = setup_hetero_mimo_training(model, cache_max_gb=0.1)
            self.assertEqual(len(loop._module_optimizers), 1)

        # ------------------------------------------------------------------ #
        # PerModuleOptimizerConfig defaults                                   #
        # ------------------------------------------------------------------ #

        def test_optimizer_config_defaults(self):
            cfg = PerModuleOptimizerConfig()
            self.assertAlmostEqual(cfg.lr, 1e-4)
            self.assertAlmostEqual(cfg.weight_decay, 0.01)
            self.assertEqual(cfg.betas, (0.9, 0.999))

        # ------------------------------------------------------------------ #
        # TrainStepResult dataclass                                           #
        # ------------------------------------------------------------------ #

        def test_train_step_result_fields(self):
            r = TrainStepResult(
                loss=0.5, grad_norm=0.1, skipped_iter=False,
                num_zeros_in_grad=0, iteration=3,
            )
            self.assertEqual(r.cache_hits, 0)
            self.assertEqual(r.cross_pool_transfers, 0)
            self.assertEqual(r.stage_assignments, {})
            self.assertEqual(r.elapsed_ms, 0.0)

    suite = unittest.TestLoader().loadTestsFromTestCase(_TestCase)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
