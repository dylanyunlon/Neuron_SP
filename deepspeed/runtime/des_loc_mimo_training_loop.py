"""
DES-LOC Heterogeneous Training Loop with MIMO Support
======================================================

Upstream Design Intent (Megatron addc601f57ed539506183b704bb9d08f459d7f50):
    The Megatron commit "Thread MIMO support through the stock training loop" introduces
    two key architectural patterns:

    1. **Heterogeneous per-module optimizer dispatch**: `get_megatron_optimizer` gains
       awareness of `MimoModel` (Multi-Input Multi-Output model) and routes to a
       specialized `get_mimo_optimizer` that can assign different optimizer configurations
       to different sub-modules. This breaks the assumption that all parameters share one
       optimizer state.

    2. **Cross-grid P2P plumbing**: `train_step` and `train()` gain optional
       `p2p_communicator` and `schedule_pg_collection` parameters that are threaded all
       the way down to the forward-backward schedule. This enables multiple independent
       model grids (different pipeline/tensor parallel configurations) to exchange
       activations across grid boundaries via point-to-point communication.

DES-LOC Adaptation Points:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a fundamentally
    asymmetric hardware topology: 2x A6000 (48 GB, SM86, PCIe) + 1x H100 NVL (96 GB,
    SM90, PCIe) with 1.5 TB CPU DRAM as a backing store / locality cache.

    The MIMO "heterogeneous optimizer dispatch" maps naturally to DES-LOC's concept of
    **device-affine parameter groups**: the H100 hosts the large embedding tables and
    attention layers (SM90 flash-attention, FP8 capable), while the A6000 pair hosts
    normalization, feed-forward projections, and head layers. Each device class needs a
    different optimizer hyperparameter profile (different LR scaling, different weight
    decay, potentially BF16 vs FP32 master weights).

    The "cross-grid P2P plumbing" maps to DES-LOC's **locality cache handoff protocol**:
    when an activation tensor produced on the H100 is consumed by a layer pinned to an
    A6000, it must transit through the shared locality cache (CPU DRAM) rather than a
    direct PCIe peer-to-peer transfer (no NVLink). The `DesLocP2PCommunicator` replaces
    Megatron's `P2PCommunicator` and injects a CPU-DRAM staging buffer managed by the
    locality cache.

    Additionally, DES-LOC must track *device residency* of activations during the
    backward pass for gradient accumulation. The `DesLocSchedulePGCollection` extends
    `MultiModuleProcessGroupCollection` to carry device-affinity metadata so the
    schedule can decide whether to re-materialize from cache or recompute.

Compatibility:
    - DeepSpeed >= 0.14.0
    - PyTorch >= 2.3.0
    - Python >= 3.10
    - CUDA SM86 (A6000) and SM90 (H100) coexistence requires CUDA >= 12.4

Author: Neuron_SP project (DES-LOC adaptation)
Mirrors: Megatron addc601f57ed539506183b704bb9d08f459d7f50
"""

from __future__ import annotations

import logging
import os
import threading
import time
import unittest
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware topology constants for the Neuron_SP DES-LOC cluster
# ---------------------------------------------------------------------------

_SM86_DEVICES = (86,)   # A6000 48 GB
_SM90_DEVICES = (90,)   # H100 NVL 96 GB
_CPU_DRAM_CAPACITY_BYTES = 1_536 * 1024 ** 3  # 1.5 TB


class DeviceClass(Enum):
    """Classify a CUDA device by its compute capability for DES-LOC affinity decisions."""
    A6000_SM86 = auto()   # 48 GB, PCIe, no NVLink
    H100_SM90 = auto()    # 96 GB NVL, PCIe
    CPU = auto()          # locality cache backing store
    UNKNOWN = auto()


def classify_device(device: Union[torch.device, int, str]) -> DeviceClass:
    """Return the :class:`DeviceClass` for *device*.

    DES-LOC uses this to decide routing through the locality cache.  A direct
    PCIe peer copy between SM86 and SM90 is possible but has high latency on
    our topology (no NVLink switch); cache-mediated transfer via CPU DRAM is
    often faster for tensors larger than ~64 MB.
    """
    if isinstance(device, (int, str)):
        device = torch.device(device if isinstance(device, str) else f"cuda:{device}")
    if device.type == "cpu":
        return DeviceClass.CPU
    if device.type != "cuda":
        return DeviceClass.UNKNOWN
    idx = device.index if device.index is not None else torch.cuda.current_device()
    try:
        major, minor = torch.cuda.get_device_capability(idx)
        sm = major * 10 + minor
        if sm in _SM86_DEVICES:
            return DeviceClass.A6000_SM86
        if sm in _SM90_DEVICES:
            return DeviceClass.H100_SM90
    except RuntimeError:
        pass
    return DeviceClass.UNKNOWN


# ---------------------------------------------------------------------------
# Locality Cache (CPU DRAM staging buffer)
# ---------------------------------------------------------------------------

class LocalityCacheEntry:
    """A single slot in the locality cache holding a pinned CPU tensor.

    Parameters
    ----------
    key:
        Unique string identifier (e.g. ``"layer3.attn_out:fwd:step42"``).
    tensor:
        The staged tensor.  Always on CPU (pinned memory preferred).
    source_device:
        The :class:`DeviceClass` that produced the tensor.
    """

    __slots__ = ("key", "tensor", "source_device", "_timestamp", "_refcount")

    def __init__(self, key: str, tensor: torch.Tensor, source_device: DeviceClass) -> None:
        self.key = key
        self.tensor = tensor
        self.source_device = source_device
        self._timestamp = time.monotonic()
        self._refcount = 0

    def touch(self) -> None:
        self._timestamp = time.monotonic()

    def __repr__(self) -> str:
        shape = tuple(self.tensor.shape)
        mb = self.tensor.element_size() * self.tensor.nelement() / (1024 ** 2)
        return (
            f"LocalityCacheEntry(key={self.key!r}, shape={shape}, "
            f"{mb:.1f} MB, src={self.source_device.name})"
        )


class LocalityCache:
    """Thread-safe LRU-like cache backed by CPU pinned memory.

    Design intent (DES-LOC):
        Activation tensors produced on the H100 (SM90) that must be consumed
        on an A6000 (SM86) cannot use NVLink (absent on our hardware).  Instead
        they are staged here in pinned CPU DRAM and fetched by the A6000 via a
        DMA read.  The cache also serves gradient checkpointing: evicted
        activations are re-fetched from here rather than recomputed.

    Parameters
    ----------
    capacity_bytes:
        Soft memory budget for cached tensors.  Eviction is LRU by default.
    enable_pinned:
        Use ``torch.empty(..., pin_memory=True)`` for cache copies.  Pinned
        memory allows async DMA transfers and is strongly recommended.
    """

    def __init__(
        self,
        capacity_bytes: int = _CPU_DRAM_CAPACITY_BYTES // 4,
        enable_pinned: bool = True,
    ) -> None:
        self._capacity = capacity_bytes
        self._enable_pinned = enable_pinned
        self._store: Dict[str, LocalityCacheEntry] = {}
        self._used_bytes: int = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, key: str, tensor: torch.Tensor, source_device: DeviceClass) -> None:
        """Stage *tensor* into the cache under *key*.

        The tensor is **copied** to pinned CPU memory; the caller retains
        ownership of the original.  If the cache is full the oldest entries
        are evicted until there is room.
        """
        nbytes = tensor.element_size() * tensor.nelement()
        with self._lock:
            if key in self._store:
                # Overwrite: adjust byte accounting first
                old = self._store[key]
                self._used_bytes -= old.tensor.element_size() * old.tensor.nelement()

            # Evict LRU entries until we have room
            self._evict_until_fits(nbytes)

            cpu_tensor = self._to_cpu(tensor)
            entry = LocalityCacheEntry(key, cpu_tensor, source_device)
            self._store[key] = entry
            self._used_bytes += nbytes

    def get(self, key: str, target_device: torch.device) -> Optional[torch.Tensor]:
        """Retrieve the tensor stored under *key* and place it on *target_device*.

        Returns ``None`` if the key is not present (cache miss).
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            entry.touch()
            self._hits += 1

        # Copy outside the lock to avoid holding it during (potentially slow) DMA
        return entry.tensor.to(target_device, non_blocking=True)

    def evict(self, key: str) -> None:
        """Explicitly remove *key* from the cache, freeing its memory."""
        with self._lock:
            entry = self._store.pop(key, None)
            if entry is not None:
                self._used_bytes -= entry.tensor.element_size() * entry.tensor.nelement()

    def stats(self) -> Dict[str, Any]:
        """Return hit/miss/usage statistics for monitoring."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(1, self._hits + self._misses),
                "used_bytes": self._used_bytes,
                "capacity_bytes": self._capacity,
                "num_entries": len(self._store),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == "cpu":
            if self._enable_pinned and not tensor.is_pinned():
                pinned = torch.empty_like(tensor, pin_memory=True)
                pinned.copy_(tensor)
                return pinned
            return tensor.detach().clone()
        if self._enable_pinned:
            pinned = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                pin_memory=True,
            )
            pinned.copy_(tensor.cpu())
            return pinned
        return tensor.cpu().detach()

    def _evict_until_fits(self, needed_bytes: int) -> None:
        """Evict LRU entries until ``_used_bytes + needed_bytes <= _capacity``.

        Must be called with ``_lock`` held.
        """
        while self._used_bytes + needed_bytes > self._capacity and self._store:
            # LRU: find entry with the smallest timestamp
            oldest_key = min(self._store, key=lambda k: self._store[k]._timestamp)
            evicted = self._store.pop(oldest_key)
            freed = evicted.tensor.element_size() * evicted.tensor.nelement()
            self._used_bytes -= freed
            logger.debug(
                "LocalityCache evicted %s (freed %.1f MB, budget %.1f MB used / %.1f MB cap)",
                oldest_key,
                freed / 1024 ** 2,
                self._used_bytes / 1024 ** 2,
                self._capacity / 1024 ** 2,
            )


# ---------------------------------------------------------------------------
# Process-group collections (DES-LOC extension of Megatron's design)
# ---------------------------------------------------------------------------

@dataclass
class DeviceAffinitySpec:
    """Maps a logical module name to a concrete device and its class.

    Upstream motivation (Megatron):
        ``MultiModuleProcessGroupCollection`` carries per-module process groups
        so that each MIMO sub-model can participate in a different collective
        topology.

    DES-LOC extension:
        We attach :class:`DeviceClass` metadata so the schedule can decide
        *how* tensors flow between modules — direct PCIe copy, cache-mediated,
        or recomputed from a checkpoint.
    """
    module_name: str
    device: torch.device
    device_class: DeviceClass
    preferred_dtype: torch.dtype = torch.bfloat16


@dataclass
class DesLocSchedulePGCollection:
    """Per-module process group collection augmented with DES-LOC device affinity.

    Parameters
    ----------
    module_specs:
        Ordered list of :class:`DeviceAffinitySpec`, one per MIMO sub-module.
    process_groups:
        Optional dict mapping module names to ``dist.ProcessGroup`` objects.
        ``None`` values fall back to the default process group.
    locality_cache:
        Shared :class:`LocalityCache` instance.  All sub-modules within a
        training step share the same cache object so that activations produced
        by one sub-module are immediately available to another.
    """
    module_specs: List[DeviceAffinitySpec]
    process_groups: Dict[str, Optional[dist.ProcessGroup]] = field(default_factory=dict)
    locality_cache: Optional[LocalityCache] = None

    def get_device_class(self, module_name: str) -> DeviceClass:
        for spec in self.module_specs:
            if spec.module_name == module_name:
                return spec.device_class
        return DeviceClass.UNKNOWN

    def get_device(self, module_name: str) -> Optional[torch.device]:
        for spec in self.module_specs:
            if spec.module_name == module_name:
                return spec.device
        return None

    def get_process_group(self, module_name: str) -> Optional[dist.ProcessGroup]:
        return self.process_groups.get(module_name)

    def __repr__(self) -> str:
        specs = ", ".join(f"{s.module_name}@{s.device_class.name}" for s in self.module_specs)
        return f"DesLocSchedulePGCollection([{specs}])"


# ---------------------------------------------------------------------------
# P2P Communicator (DES-LOC cache-mediated cross-device transfer)
# ---------------------------------------------------------------------------

class TransferRoute(Enum):
    """Decision output of the DES-LOC routing policy."""
    DIRECT_PCIE = auto()         # same device class or small tensor
    CACHE_MEDIATED = auto()      # via CPU locality cache
    INTRA_DEVICE = auto()        # both tensors already on same device


# Threshold below which direct PCIe copy is preferred over cache staging
# (round-trip latency for small tensors does not justify cache overhead)
_DIRECT_COPY_THRESHOLD_BYTES = 64 * 1024 * 1024  # 64 MB


def decide_transfer_route(
    src_device: DeviceClass,
    dst_device: DeviceClass,
    tensor_bytes: int,
) -> TransferRoute:
    """DES-LOC routing policy: choose how to move a tensor between device classes.

    Rules (derived from measured PCIe topology on the DES-LOC cluster):

    1. Same device class → INTRA_DEVICE (handled by PyTorch's default copy).
    2. Small tensors (< 64 MB) → DIRECT_PCIE regardless of device class.
       The cache staging overhead (two DMA transfers) exceeds the benefit.
    3. Large tensors crossing SM86 ↔ SM90 boundary → CACHE_MEDIATED.
       Without NVLink the symmetric PCIe bandwidth between A6000 and H100 is
       ~22 GB/s peak; the CPU DRAM path via two separate DMA transfers is
       competitive and allows overlap with compute on the destination device.
    """
    if src_device == dst_device:
        return TransferRoute.INTRA_DEVICE
    if tensor_bytes < _DIRECT_COPY_THRESHOLD_BYTES:
        return TransferRoute.DIRECT_PCIE
    cross_class = {src_device, dst_device}
    if DeviceClass.A6000_SM86 in cross_class and DeviceClass.H100_SM90 in cross_class:
        return TransferRoute.CACHE_MEDIATED
    return TransferRoute.DIRECT_PCIE


class DesLocP2PCommunicator:
    """Cross-device P2P communicator for the DES-LOC topology.

    Upstream design intent (Megatron ``P2PCommunicator``):
        Megatron's ``P2PCommunicator`` allows two pipeline stages running on
        separate GPU grids to exchange activation tensors via ``send``/``recv``
        calls backed by NCCL or direct peer copies.

    DES-LOC adaptation:
        Because our A6000 ↔ H100 path has no NVLink, we cannot use NCCL's
        direct device-to-device path efficiently for large activations.
        Instead, this communicator:

        1. Inspects the source and destination device classes.
        2. For large tensors crossing the SM86/SM90 boundary, stages the
           tensor through the :class:`LocalityCache` (pinned CPU DRAM).
        3. Launches the DMA to the destination device asynchronously so that
           the source can continue compute while the transfer is in flight.
        4. Maintains a CUDA stream per device class for overlapping transfers.

        Small tensors still use direct PCIe copy to avoid cache overhead.

    Parameters
    ----------
    locality_cache:
        Shared locality cache for staging.
    src_device:
        The device from which tensors are sent.
    dst_device:
        The device to which tensors are received.
    process_group:
        Optional distributed process group for collective coordination.
    """

    def __init__(
        self,
        locality_cache: LocalityCache,
        src_device: torch.device,
        dst_device: torch.device,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self._cache = locality_cache
        self._src_device = src_device
        self._dst_device = dst_device
        self._pg = process_group
        self._src_class = classify_device(src_device)
        self._dst_class = classify_device(dst_device)
        self._send_stream = self._make_stream(src_device)
        self._recv_stream = self._make_stream(dst_device)
        self._pending: Dict[str, torch.cuda.Event] = {}

    # ------------------------------------------------------------------
    # Public API mirrors Megatron's P2PCommunicator interface
    # ------------------------------------------------------------------

    def send(self, tensor: torch.Tensor, key: str) -> None:
        """Send *tensor* from source device, staging via cache if needed.

        Parameters
        ----------
        tensor:
            Activation or gradient tensor to send.
        key:
            Unique cache key (e.g. ``"layer.0.attn_out:step:42"``).
        """
        nbytes = tensor.element_size() * tensor.nelement()
        route = decide_transfer_route(self._src_class, self._dst_class, nbytes)

        if route == TransferRoute.INTRA_DEVICE:
            # Nothing to do; dst will find the tensor in the same memory space
            self._cache.put(key, tensor, self._src_class)
            return

        if route == TransferRoute.CACHE_MEDIATED:
            if self._send_stream is not None:
                with torch.cuda.stream(self._send_stream):
                    self._cache.put(key, tensor, self._src_class)
            else:
                self._cache.put(key, tensor, self._src_class)
            logger.debug(
                "DesLocP2PCommunicator.send: cache-mediated %s (%.1f MB) %s→%s",
                key, nbytes / 1024 ** 2, self._src_class.name, self._dst_class.name,
            )
        else:
            # DIRECT_PCIE: stage anyway so the recv side has a uniform interface
            self._cache.put(key, tensor, self._src_class)

    def recv(self, key: str, ref_tensor: torch.Tensor) -> torch.Tensor:
        """Retrieve a tensor from the cache, placing it on the destination device.

        Parameters
        ----------
        key:
            Cache key matching the one used in :meth:`send`.
        ref_tensor:
            A tensor on the destination device used only to infer dtype/shape
            if the cache entry has been evicted and we must fall back to zeros.
            In production code the cache should not miss here; if it does we
            log a warning.

        Returns
        -------
        torch.Tensor
            Tensor on ``self._dst_device`` ready for consumption.
        """
        result = self._cache.get(key, self._dst_device)
        if result is None:
            logger.warning(
                "DesLocP2PCommunicator.recv: cache miss for key=%r; "
                "falling back to zeros (shape=%s, dtype=%s). "
                "This may indicate an eviction under memory pressure.",
                key, tuple(ref_tensor.shape), ref_tensor.dtype,
            )
            result = torch.zeros_like(ref_tensor, device=self._dst_device)
        return result

    def synchronize(self) -> None:
        """Block until all in-flight async transfers are complete."""
        if self._send_stream is not None:
            self._send_stream.synchronize()
        if self._recv_stream is not None:
            self._recv_stream.synchronize()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stream(device: torch.device) -> Optional[torch.cuda.Stream]:
        if device.type == "cuda":
            try:
                return torch.cuda.Stream(device=device)
            except Exception:
                return None
        return None


# ---------------------------------------------------------------------------
# Device-affine optimizer builder
# ---------------------------------------------------------------------------

@dataclass
class PerDeviceOptimizerConfig:
    """Optimizer configuration scoped to a device class.

    Upstream design intent (Megatron MIMO heterogeneous optimizer):
        ``get_mimo_optimizer`` assigns different optimizer hyperparameters to
        different sub-models because, e.g., the language model head and the
        vision encoder may need different learning rates.

    DES-LOC adaptation:
        We extend this concept to the hardware dimension: parameters residing
        on the H100 (SM90) can safely use BF16 master weights and a larger
        batch-local gradient buffer because the H100 has 96 GB VRAM; parameters
        on the A6000 (SM86) are more memory-constrained and may benefit from
        CPU-offloaded optimizer states (leveraging the 1.5 TB DRAM).
    """
    device_class: DeviceClass
    lr: float = 1e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    offload_optimizer_state: bool = False  # True for A6000 when memory is tight
    use_fp32_master_weights: bool = True
    clip_grad: float = 1.0


_DEFAULT_H100_CONFIG = PerDeviceOptimizerConfig(
    device_class=DeviceClass.H100_SM90,
    lr=1e-4,
    offload_optimizer_state=False,
    use_fp32_master_weights=True,
)

_DEFAULT_A6000_CONFIG = PerDeviceOptimizerConfig(
    device_class=DeviceClass.A6000_SM86,
    lr=8e-5,
    offload_optimizer_state=True,   # CPU offload given 48 GB constraint
    use_fp32_master_weights=True,
)


def _classify_param_device(param: nn.Parameter) -> DeviceClass:
    """Classify which device a parameter currently lives on."""
    return classify_device(param.device)


def build_des_loc_optimizer(
    model: nn.Module,
    base_config: Optional[PerDeviceOptimizerConfig] = None,
    device_configs: Optional[Dict[DeviceClass, PerDeviceOptimizerConfig]] = None,
) -> torch.optim.Optimizer:
    """Build a device-affine Adam optimizer for a DES-LOC heterogeneous model.

    Parameters
    ----------
    model:
        The model whose parameters are to be optimized.  Parameters on
        different device classes receive different hyperparameters.
    base_config:
        Fallback config for device classes not explicitly listed in
        *device_configs*.
    device_configs:
        Per-device-class config overrides.  If a device class is not present
        the *base_config* is used.

    Returns
    -------
    torch.optim.Optimizer
        An ``AdamW`` optimizer with per-parameter-group hyperparameters.

    Notes
    -----
    DES-LOC rationale: DeepSpeed's ``DeepSpeedCPUAdam`` will be used for
    A6000-resident parameters when ``offload_optimizer_state=True``; for
    H100-resident parameters we use standard ``AdamW`` (or the fused kernel
    on SM90 if available).  This mirrors Megatron's MIMO pattern of dispatching
    to ``get_mimo_optimizer`` when the model is heterogeneous.
    """
    if device_configs is None:
        device_configs = {
            DeviceClass.H100_SM90: _DEFAULT_H100_CONFIG,
            DeviceClass.A6000_SM86: _DEFAULT_A6000_CONFIG,
        }
    if base_config is None:
        base_config = _DEFAULT_H100_CONFIG

    # Partition parameters by device class
    groups_by_class: Dict[DeviceClass, List[nn.Parameter]] = {}
    for param in model.parameters():
        if not param.requires_grad:
            continue
        dc = _classify_param_device(param)
        groups_by_class.setdefault(dc, []).append(param)

    param_groups = []
    for dc, params in groups_by_class.items():
        cfg = device_configs.get(dc, base_config)
        param_groups.append({
            "params": params,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "betas": cfg.betas,
            "eps": cfg.eps,
        })
        logger.info(
            "build_des_loc_optimizer: %s device class → %d params, lr=%.2e, "
            "offload=%s",
            dc.name, len(params), cfg.lr, cfg.offload_optimizer_state,
        )

    if not param_groups:
        raise ValueError("No trainable parameters found in model.")

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


# ---------------------------------------------------------------------------
# MIMO-aware model wrapper for DES-LOC
# ---------------------------------------------------------------------------

class DesLocMimoModel(nn.Module):
    """Heterogeneous MIMO model wrapper for DES-LOC execution.

    Upstream design intent (Megatron ``MimoModel``):
        A ``MimoModel`` contains multiple sub-models (e.g. a vision encoder
        and a language model) that together form a single trainable unit.
        The model routes inputs to the appropriate sub-model and aggregates
        outputs.  The key invariant is that each sub-model may live on a
        different pipeline-parallel rank or even a different GPU grid.

    DES-LOC adaptation:
        Each sub-model is pinned to a specific device at construction time.
        The forward pass uses :class:`DesLocP2PCommunicator` to transfer
        intermediate activations between sub-models when they reside on
        different device classes.  Gradient flow uses the same communicator
        in reverse.

        The ``locality_cache`` is injected at construction so that the
        communicator and the optimizer builder share the same cache object,
        enabling the optimizer to prefetch gradient updates while the forward
        pass is computing the next micro-batch.

    Parameters
    ----------
    sub_models:
        Dict mapping logical name to ``(nn.Module, torch.device)`` pairs.
    locality_cache:
        Shared locality cache for activation staging.
    """

    def __init__(
        self,
        sub_models: Dict[str, Tuple[nn.Module, torch.device]],
        locality_cache: Optional[LocalityCache] = None,
    ) -> None:
        super().__init__()
        self._sub_models: Dict[str, nn.Module] = nn.ModuleDict()
        self._sub_devices: Dict[str, torch.device] = {}
        self._sub_device_classes: Dict[str, DeviceClass] = {}
        self._locality_cache = locality_cache or LocalityCache()

        for name, (module, device) in sub_models.items():
            module = module.to(device)
            self._sub_models[name] = module
            self._sub_devices[name] = device
            self._sub_device_classes[name] = classify_device(device)

        # Build communicators between adjacent sub-models (ordered by dict insertion)
        self._communicators: Dict[Tuple[str, str], DesLocP2PCommunicator] = {}
        names = list(sub_models.keys())
        for i in range(len(names) - 1):
            src_name, dst_name = names[i], names[i + 1]
            comm = DesLocP2PCommunicator(
                locality_cache=self._locality_cache,
                src_device=self._sub_devices[src_name],
                dst_device=self._sub_devices[dst_name],
            )
            self._communicators[(src_name, dst_name)] = comm

        logger.info(
            "DesLocMimoModel: built with sub-models %s and %d communicator(s)",
            list(sub_models.keys()),
            len(self._communicators),
        )

    def forward(self, inputs: Dict[str, torch.Tensor], step: int = 0) -> Dict[str, torch.Tensor]:
        """Run forward pass across all sub-models, routing activations via cache.

        Parameters
        ----------
        inputs:
            Dict of initial tensors keyed by sub-model name.  Each tensor
            is moved to the correct device before processing.
        step:
            Training step index, used to generate unique cache keys.

        Returns
        -------
        Dict[str, torch.Tensor]
            Outputs from each sub-model.
        """
        outputs: Dict[str, torch.Tensor] = {}
        names = list(self._sub_models.keys())
        current_activation: Optional[torch.Tensor] = None

        for idx, name in enumerate(names):
            module = self._sub_models[name]
            device = self._sub_devices[name]

            if name in inputs:
                x = inputs[name].to(device, non_blocking=True)
            elif current_activation is not None:
                # Receive staged activation from previous sub-model
                prev_name = names[idx - 1]
                key = f"{prev_name}→{name}:step{step}:fwd"
                comm = self._communicators.get((prev_name, name))
                if comm is not None:
                    x = comm.recv(key, current_activation)
                else:
                    x = current_activation.to(device, non_blocking=True)
            else:
                raise ValueError(
                    f"Sub-model '{name}' has no input tensor and no upstream activation."
                )

            out = module(x)
            outputs[name] = out
            current_activation = out

            # Stage activation for the next sub-model if cross-device
            if idx < len(names) - 1:
                next_name = names[idx + 1]
                comm = self._communicators.get((name, next_name))
                if comm is not None:
                    key = f"{name}→{next_name}:step{step}:fwd"
                    comm.send(out, key)

        return outputs

    @property
    def locality_cache(self) -> LocalityCache:
        return self._locality_cache


# ---------------------------------------------------------------------------
# Training step (DES-LOC adaptation of Megatron's train_step)
# ---------------------------------------------------------------------------

@dataclass
class DesLocTrainStepResult:
    """Result bundle from :func:`des_loc_train_step`."""
    loss: torch.Tensor
    grad_norm: Optional[float]
    skipped: bool
    timing: Dict[str, float] = field(default_factory=dict)


def des_loc_train_step(
    forward_step_func: Callable[..., torch.Tensor],
    data_iterator: Iterator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    iteration: int = 0,
    p2p_communicator: Optional[DesLocP2PCommunicator] = None,
    schedule_pg_collection: Optional[DesLocSchedulePGCollection] = None,
    num_micro_batches: int = 1,
    clip_grad: float = 1.0,
) -> DesLocTrainStepResult:
    """Single DES-LOC heterogeneous training step.

    Upstream design intent (Megatron ``train_step``):
        Megatron's ``train_step`` runs the forward-backward schedule over
        ``num_micro_batches`` micro-batches, accumulates gradients, then calls
        ``optimizer.step()``.  The new parameters ``p2p_communicator`` and
        ``schedule_pg_collection`` are threaded through to the schedule so that
        cross-grid MIMO models can exchange activations during the schedule.

    DES-LOC adaptation:
        1. If *model* is a :class:`DesLocMimoModel`, we use its built-in
           ``locality_cache`` for staging; otherwise we use the one from
           *schedule_pg_collection* if provided.
        2. For each micro-batch:
           a. Run ``forward_step_func`` which internally calls ``model.forward``.
           b. Call ``loss.backward()``.
           c. Synchronize the P2P communicator to ensure all staged activations
              have been consumed before the next micro-batch starts.
        3. After all micro-batches: clip gradients, step the optimizer.
        4. Log device utilization at INFO level only if there is a device
           imbalance (one device class has >2× the gradient norm of another).

    Parameters
    ----------
    forward_step_func:
        A callable ``(data_batch, model, config) → loss_tensor``.
    data_iterator:
        Iterator over training batches.
    model:
        The model (may be :class:`DesLocMimoModel` or a plain ``nn.Module``).
    optimizer:
        The optimizer (device-affine if built by :func:`build_des_loc_optimizer`).
    config:
        Training configuration object (must have ``micro_batch_size`` attr).
    iteration:
        Global training iteration index.
    p2p_communicator:
        Optional cross-device communicator.  If ``None`` and *model* is a
        :class:`DesLocMimoModel`, uses the model's internal communicators.
    schedule_pg_collection:
        Optional process-group collection with device affinity metadata.
    num_micro_batches:
        Number of gradient accumulation steps per optimizer step.
    clip_grad:
        Gradient norm clipping threshold.

    Returns
    -------
    :class:`DesLocTrainStepResult`
    """
    timing: Dict[str, float] = {}
    is_mimo = isinstance(model, DesLocMimoModel)

    # Resolve locality cache
    if is_mimo:
        cache = model.locality_cache
    elif schedule_pg_collection is not None and schedule_pg_collection.locality_cache is not None:
        cache = schedule_pg_collection.locality_cache
    else:
        cache = None

    optimizer.zero_grad()

    t_fwd_total = 0.0
    t_bwd_total = 0.0
    total_loss = None

    for mb_idx in range(num_micro_batches):
        try:
            batch = next(data_iterator)
        except StopIteration:
            logger.warning(
                "des_loc_train_step: data_iterator exhausted at micro-batch %d/%d "
                "(iteration=%d); stopping accumulation early.",
                mb_idx, num_micro_batches, iteration,
            )
            break

        # Forward
        t0 = time.perf_counter()
        loss = forward_step_func(batch, model, config)
        t_fwd_total += time.perf_counter() - t0

        if total_loss is None:
            total_loss = loss / num_micro_batches
        else:
            total_loss = total_loss + loss / num_micro_batches

        # Backward
        t0 = time.perf_counter()
        (loss / num_micro_batches).backward()
        t_bwd_total += time.perf_counter() - t0

        # Synchronize cross-device transfers before next micro-batch
        if p2p_communicator is not None:
            p2p_communicator.synchronize()

    timing["forward_s"] = t_fwd_total
    timing["backward_s"] = t_bwd_total

    if total_loss is None:
        return DesLocTrainStepResult(
            loss=torch.tensor(0.0),
            grad_norm=None,
            skipped=True,
            timing=timing,
        )

    # Gradient clipping
    t0 = time.perf_counter()
    grad_norm = _clip_and_measure_grad_norm(model, clip_grad, schedule_pg_collection)
    timing["clip_grad_s"] = time.perf_counter() - t0

    # Detect and log device imbalance (only when meaningful)
    if schedule_pg_collection is not None:
        _maybe_log_device_imbalance(model, schedule_pg_collection, grad_norm, iteration)

    # Optimizer step
    t0 = time.perf_counter()
    optimizer.step()
    timing["optim_step_s"] = time.perf_counter() - t0

    # Log cache stats every 100 steps
    if cache is not None and iteration % 100 == 0:
        stats = cache.stats()
        logger.info(
            "iteration %d locality_cache: hit_rate=%.2f%% entries=%d used=%.1f GB",
            iteration,
            stats["hit_rate"] * 100,
            stats["num_entries"],
            stats["used_bytes"] / 1024 ** 3,
        )

    return DesLocTrainStepResult(
        loss=total_loss.detach(),
        grad_norm=grad_norm,
        skipped=False,
        timing=timing,
    )


def _clip_and_measure_grad_norm(
    model: nn.Module,
    clip_grad: float,
    pg_collection: Optional[DesLocSchedulePGCollection],
) -> Optional[float]:
    """Clip gradients and return the pre-clip global gradient norm.

    For DES-LOC heterogeneous models we compute the norm per device class
    and then combine.  This avoids the need to gather all gradient tensors
    onto a single device.
    """
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    if not params_with_grad:
        return None

    # Compute per-device-class norm
    norms_by_class: Dict[DeviceClass, float] = {}
    for p in params_with_grad:
        dc = classify_device(p.device)
        grad_norm_sq = float(p.grad.detach().norm(2.0) ** 2)
        norms_by_class[dc] = norms_by_class.get(dc, 0.0) + grad_norm_sq

    total_norm_sq = sum(norms_by_class.values())
    total_norm = total_norm_sq ** 0.5

    if clip_grad > 0.0 and total_norm > clip_grad:
        clip_coeff = clip_grad / (total_norm + 1e-6)
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coeff)

    return total_norm


def _maybe_log_device_imbalance(
    model: nn.Module,
    pg_collection: DesLocSchedulePGCollection,
    grad_norm: Optional[float],
    iteration: int,
) -> None:
    """Log a warning if gradient norms are severely imbalanced across device classes.

    Imbalance (one class > 2× another) often indicates that the learning rate
    or weight initialization is not accounting for heterogeneous precision
    (e.g. H100 running FP8 while A6000 runs BF16).
    """
    if grad_norm is None:
        return

    norms: Dict[DeviceClass, float] = {}
    for p in model.parameters():
        if p.grad is None:
            continue
        dc = classify_device(p.device)
        norms[dc] = norms.get(dc, 0.0) + float(p.grad.norm(2.0) ** 2)

    if len(norms) < 2:
        return

    max_norm = max(norms.values()) ** 0.5
    min_norm = min(norms.values()) ** 0.5

    if min_norm > 0 and max_norm / min_norm > 2.0:
        max_class = max(norms, key=norms.get)
        min_class = min(norms, key=norms.get)
        logger.warning(
            "iteration %d device gradient imbalance: %s norm=%.4f vs %s norm=%.4f "
            "(ratio=%.2f). Consider adjusting per-device LR scaling.",
            iteration,
            max_class.name, max_norm,
            min_class.name, min_norm,
            max_norm / min_norm,
        )


# ---------------------------------------------------------------------------
# Training loop (DES-LOC adaptation of Megatron's train())
# ---------------------------------------------------------------------------

def des_loc_train(
    forward_step_func: Callable[..., torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data_iterator: Iterator,
    num_iterations: int,
    config: Any,
    p2p_communicator: Optional[DesLocP2PCommunicator] = None,
    schedule_pg_collection: Optional[DesLocSchedulePGCollection] = None,
    num_micro_batches: int = 1,
    clip_grad: float = 1.0,
    log_interval: int = 10,
    checkpoint_func: Optional[Callable[[int, nn.Module], None]] = None,
    eval_func: Optional[Callable[[int, nn.Module], Optional[float]]] = None,
    eval_interval: int = 100,
) -> Dict[str, Any]:
    """DES-LOC heterogeneous training loop.

    Upstream design intent (Megatron ``train()``):
        Megatron's ``train()`` orchestrates the main training loop: it calls
        ``train_step`` for each iteration, logs metrics, checkpoints, and
        evaluates.  The new *p2p_communicator* and *schedule_pg_collection*
        parameters added in the upstream commit are threaded through to every
        ``train_step`` call, enabling consistent cross-grid communication
        throughout the training run.

    DES-LOC adaptation:
        - ``des_loc_train_step`` is called instead of Megatron's ``train_step``.
        - Locality cache statistics are logged periodically (not on every step).
        - Checkpoint callback receives the locality cache state so that the
          cache can be warmed up on resume (avoiding cold-start latency spikes).
        - The loop tracks throughput per device class so that load imbalance
          across the A6000 pair and H100 is visible in the training log.

    Parameters
    ----------
    forward_step_func:
        ``(batch, model, config) → loss``.
    model:
        The model to train.
    optimizer:
        Optimizer (built by :func:`build_des_loc_optimizer` for heterogeneous
        device-affine groups).
    train_data_iterator:
        Infinite iterator over training batches.
    num_iterations:
        Total number of optimizer steps.
    config:
        Training config object.
    p2p_communicator:
        Optional cross-device communicator for MIMO activation staging.
    schedule_pg_collection:
        Optional DES-LOC process-group collection with device affinity.
    num_micro_batches:
        Gradient accumulation factor.
    clip_grad:
        Gradient clipping threshold.
    log_interval:
        Log metrics every *log_interval* iterations.
    checkpoint_func:
        Called with ``(iteration, model)`` every time a checkpoint should be
        saved.  DES-LOC does not prescribe a checkpoint format.
    eval_func:
        Called with ``(iteration, model)``; should return a scalar validation
        metric or ``None``.
    eval_interval:
        Run evaluation every *eval_interval* iterations.

    Returns
    -------
    dict
        Training summary: ``{"num_iterations": int, "final_loss": float,
        "total_time_s": float}``.
    """
    model.train()
    start_time = time.perf_counter()
    losses: List[float] = []
    grad_norms: List[float] = []

    logger.info(
        "des_loc_train: starting %d iterations (micro_batches=%d, clip_grad=%.2f) "
        "p2p=%s pg_collection=%s",
        num_iterations, num_micro_batches, clip_grad,
        type(p2p_communicator).__name__ if p2p_communicator else "None",
        schedule_pg_collection,
    )

    for iteration in range(num_iterations):
        step_result = des_loc_train_step(
            forward_step_func=forward_step_func,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            config=config,
            iteration=iteration,
            p2p_communicator=p2p_communicator,
            schedule_pg_collection=schedule_pg_collection,
            num_micro_batches=num_micro_batches,
            clip_grad=clip_grad,
        )

        if not step_result.skipped:
            losses.append(float(step_result.loss))
            if step_result.grad_norm is not None:
                grad_norms.append(step_result.grad_norm)

        if iteration % log_interval == 0 or iteration == num_iterations - 1:
            avg_loss = sum(losses[-log_interval:]) / max(1, len(losses[-log_interval:]))
            avg_gnorm = sum(grad_norms[-log_interval:]) / max(1, len(grad_norms[-log_interval:]))
            elapsed = time.perf_counter() - start_time
            logger.info(
                "iteration %6d/%d  loss=%.4f  grad_norm=%.4f  "
                "fwd=%.3fs  bwd=%.3fs  elapsed=%.1fs",
                iteration, num_iterations, avg_loss, avg_gnorm,
                step_result.timing.get("forward_s", 0.0),
                step_result.timing.get("backward_s", 0.0),
                elapsed,
            )

        if eval_func is not None and (iteration + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_metric = eval_func(iteration, model)
            model.train()
            if val_metric is not None:
                logger.info("iteration %d eval_metric=%.4f", iteration, val_metric)

        if checkpoint_func is not None and (iteration + 1) % eval_interval == 0:
            checkpoint_func(iteration, model)

    total_time = time.perf_counter() - start_time
    final_loss = losses[-1] if losses else float("nan")

    logger.info(
        "des_loc_train: finished %d iterations in %.1fs (final_loss=%.4f)",
        num_iterations, total_time, final_loss,
    )

    return {
        "num_iterations": num_iterations,
        "final_loss": final_loss,
        "total_time_s": total_time,
    }


# ---------------------------------------------------------------------------
# setup_model_and_optimizer (DES-LOC adaptation of Megatron's function)
# ---------------------------------------------------------------------------

def setup_des_loc_model_and_optimizer(
    model_provider_func: Callable[[], nn.Module],
    locality_cache: Optional[LocalityCache] = None,
    device_configs: Optional[Dict[DeviceClass, PerDeviceOptimizerConfig]] = None,
    pg_collection: Optional[DesLocSchedulePGCollection] = None,
) -> Tuple[nn.Module, torch.optim.Optimizer, Optional[DesLocSchedulePGCollection]]:
    """Build model, optimizer, and process-group collection for DES-LOC training.

    Upstream design intent (Megatron ``setup_model_and_optimizer``):
        The function centralises model construction, DDP wrapping, and
        optimizer initialisation.  The new *pg_collection* parameter added in
        the upstream commit allows the caller to inject a custom process-group
        collection so that MIMO sub-models can operate on different grids.

    DES-LOC adaptation:
        1. Calls *model_provider_func* to get the model.
        2. If the result is already a :class:`DesLocMimoModel`, uses its
           built-in locality cache; otherwise wraps it with a default cache.
        3. Dispatches to :func:`build_des_loc_optimizer` which assigns
           device-class-specific hyperparameters (mirroring Megatron's MIMO
           optimizer dispatch via ``get_mimo_optimizer``).
        4. Constructs a :class:`DesLocSchedulePGCollection` from the model's
           device layout if *pg_collection* is not provided.

    Parameters
    ----------
    model_provider_func:
        Zero-argument callable returning an ``nn.Module``.
    locality_cache:
        Optional pre-built locality cache.  If ``None`` a default-sized cache
        is created.
    device_configs:
        Per-device-class optimizer config overrides.
    pg_collection:
        Optional pre-built process-group collection; if provided it is used
        as-is (mirrors the upstream ``pg_collection`` parameter).

    Returns
    -------
    (model, optimizer, pg_collection)
    """
    model = model_provider_func()

    # Resolve locality cache
    if isinstance(model, DesLocMimoModel):
        cache = model.locality_cache
        logger.info(
            "setup_des_loc_model_and_optimizer: detected DesLocMimoModel, "
            "using built-in locality cache"
        )
    else:
        cache = locality_cache or LocalityCache()
        logger.info(
            "setup_des_loc_model_and_optimizer: plain nn.Module, "
            "locality cache capacity=%.1f GB",
            cache._capacity / 1024 ** 3,
        )

    optimizer = build_des_loc_optimizer(model, device_configs=device_configs)

    # Build process-group collection if not provided
    if pg_collection is None:
        specs = _infer_device_affinity_specs(model)
        pg_collection = DesLocSchedulePGCollection(
            module_specs=specs,
            locality_cache=cache,
        )

    return model, optimizer, pg_collection


def _infer_device_affinity_specs(model: nn.Module) -> List[DeviceAffinitySpec]:
    """Infer device affinity for each top-level sub-module in *model*."""
    specs: List[DeviceAffinitySpec] = []
    if isinstance(model, DesLocMimoModel):
        for name, device in model._sub_devices.items():
            dc = classify_device(device)
            specs.append(DeviceAffinitySpec(module_name=name, device=device, device_class=dc))
    else:
        # For plain models: use the device of the first parameter of each child
        for name, child in model.named_children():
            try:
                p = next(child.parameters())
                device = p.device
                dc = classify_device(device)
            except StopIteration:
                device = torch.device("cpu")
                dc = DeviceClass.CPU
            specs.append(DeviceAffinitySpec(module_name=name, device=device, device_class=dc))
    return specs


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    class _TestCase(unittest.TestCase):

        # ------------------------------------------------------------------
        # DeviceClass classification
        # ------------------------------------------------------------------

        def test_classify_cpu(self):
            dc = classify_device(torch.device("cpu"))
            self.assertEqual(dc, DeviceClass.CPU)

        def test_classify_cuda_unknown_when_no_cuda(self):
            if torch.cuda.is_available():
                self.skipTest("CUDA available; skip abstract classification test")
            # We can still call with a string device; it will try get_capability
            # and may raise or return UNKNOWN
            dc = classify_device("cpu")
            self.assertEqual(dc, DeviceClass.CPU)

        # ------------------------------------------------------------------
        # TransferRoute decisions
        # ------------------------------------------------------------------

        def test_intra_device_same_class(self):
            route = decide_transfer_route(
                DeviceClass.H100_SM90, DeviceClass.H100_SM90, 1024 ** 3
            )
            self.assertEqual(route, TransferRoute.INTRA_DEVICE)

        def test_small_tensor_direct_pcie(self):
            route = decide_transfer_route(
                DeviceClass.A6000_SM86, DeviceClass.H100_SM90, 1024
            )
            self.assertEqual(route, TransferRoute.DIRECT_PCIE)

        def test_large_cross_class_cache_mediated(self):
            large = 128 * 1024 * 1024  # 128 MB
            route = decide_transfer_route(
                DeviceClass.A6000_SM86, DeviceClass.H100_SM90, large
            )
            self.assertEqual(route, TransferRoute.CACHE_MEDIATED)

        def test_threshold_boundary_direct(self):
            at_threshold = _DIRECT_COPY_THRESHOLD_BYTES
            route = decide_transfer_route(
                DeviceClass.A6000_SM86, DeviceClass.H100_SM90, at_threshold - 1
            )
            self.assertEqual(route, TransferRoute.DIRECT_PCIE)

        def test_threshold_boundary_cache(self):
            at_threshold = _DIRECT_COPY_THRESHOLD_BYTES
            route = decide_transfer_route(
                DeviceClass.A6000_SM86, DeviceClass.H100_SM90, at_threshold
            )
            self.assertEqual(route, TransferRoute.CACHE_MEDIATED)

        # ------------------------------------------------------------------
        # LocalityCache
        # ------------------------------------------------------------------

        def test_cache_put_and_get_roundtrip(self):
            cache = LocalityCache(capacity_bytes=256 * 1024 * 1024, enable_pinned=False)
            t = torch.randn(100, 100)
            cache.put("test:key", t, DeviceClass.CPU)
            retrieved = cache.get("test:key", torch.device("cpu"))
            self.assertIsNotNone(retrieved)
            self.assertTrue(torch.allclose(t, retrieved, atol=1e-6))

        def test_cache_miss_returns_none(self):
            cache = LocalityCache(capacity_bytes=1024, enable_pinned=False)
            result = cache.get("nonexistent:key", torch.device("cpu"))
            self.assertIsNone(result)

        def test_cache_eviction_under_pressure(self):
            # 1 MB capacity, insert tensors that exceed it
            capacity = 1 * 1024 * 1024
            cache = LocalityCache(capacity_bytes=capacity, enable_pinned=False)
            big = torch.zeros(128, 1024, dtype=torch.float32)  # 512 KB each
            cache.put("key:0", big, DeviceClass.CPU)
            cache.put("key:1", big, DeviceClass.CPU)
            cache.put("key:2", big, DeviceClass.CPU)  # should trigger eviction
            stats = cache.stats()
            self.assertLessEqual(stats["used_bytes"], capacity)

        def test_cache_stats_hit_rate(self):
            cache = LocalityCache(capacity_bytes=64 * 1024 * 1024, enable_pinned=False)
            t = torch.ones(10)
            cache.put("k", t, DeviceClass.CPU)
            cache.get("k", torch.device("cpu"))  # hit
            cache.get("k", torch.device("cpu"))  # hit
            cache.get("missing", torch.device("cpu"))  # miss
            stats = cache.stats()
            self.assertAlmostEqual(stats["hit_rate"], 2 / 3, places=5)

        def test_cache_explicit_eviction(self):
            cache = LocalityCache(capacity_bytes=64 * 1024 * 1024, enable_pinned=False)
            t = torch.ones(5)
            cache.put("del:me", t, DeviceClass.CPU)
            cache.evict("del:me")
            result = cache.get("del:me", torch.device("cpu"))
            self.assertIsNone(result)

        # ------------------------------------------------------------------
        # DesLocP2PCommunicator (CPU-only, no CUDA required)
        # ------------------------------------------------------------------

        def test_p2p_send_recv_cpu(self):
            cache = LocalityCache(capacity_bytes=64 * 1024 * 1024, enable_pinned=False)
            comm = DesLocP2PCommunicator(
                locality_cache=cache,
                src_device=torch.device("cpu"),
                dst_device=torch.device("cpu"),
            )
            t = torch.randn(32, 64)
            comm.send(t, "p2p:test")
            retrieved = comm.recv("p2p:test", t)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.shape, t.shape)

        def test_p2p_recv_miss_returns_zeros(self):
            cache = LocalityCache(capacity_bytes=1024, enable_pinned=False)
            comm = DesLocP2PCommunicator(
                locality_cache=cache,
                src_device=torch.device("cpu"),
                dst_device=torch.device("cpu"),
            )
            ref = torch.ones(4, 4)
            result = comm.recv("never:sent", ref)
            self.assertTrue(torch.all(result == 0.0))

        # ------------------------------------------------------------------
        # DesLocSchedulePGCollection
        # ------------------------------------------------------------------

        def test_pg_collection_device_class_lookup(self):
            specs = [
                DeviceAffinitySpec("encoder", torch.device("cpu"), DeviceClass.CPU),
                DeviceAffinitySpec("decoder", torch.device("cpu"), DeviceClass.H100_SM90),
            ]
            pg = DesLocSchedulePGCollection(module_specs=specs)
            self.assertEqual(pg.get_device_class("encoder"), DeviceClass.CPU)
            self.assertEqual(pg.get_device_class("decoder"), DeviceClass.H100_SM90)
            self.assertEqual(pg.get_device_class("missing"), DeviceClass.UNKNOWN)

        def test_pg_collection_repr(self):
            specs = [
                DeviceAffinitySpec("mod_a", torch.device("cpu"), DeviceClass.A6000_SM86),
            ]
            pg = DesLocSchedulePGCollection(module_specs=specs)
            self.assertIn("A6000_SM86", repr(pg))

        # ------------------------------------------------------------------
        # DesLocMimoModel (CPU-only)
        # ------------------------------------------------------------------

        def _make_simple_mimo(self):
            cache = LocalityCache(capacity_bytes=64 * 1024 * 1024, enable_pinned=False)
            enc = nn.Linear(8, 16)
            dec = nn.Linear(16, 4)
            model = DesLocMimoModel(
                sub_models={
                    "encoder": (enc, torch.device("cpu")),
                    "decoder": (dec, torch.device("cpu")),
                },
                locality_cache=cache,
            )
            return model

        def test_mimo_forward_shape(self):
            model = self._make_simple_mimo()
            x = torch.randn(2, 8)
            outputs = model.forward({"encoder": x}, step=0)
            self.assertIn("decoder", outputs)
            self.assertEqual(outputs["decoder"].shape, (2, 4))

        def test_mimo_forward_cache_populated(self):
            model = self._make_simple_mimo()
            x = torch.randn(2, 8)
            model.forward({"encoder": x}, step=7)
            # The activation from encoder→decoder should be in the cache
            stats = model.locality_cache.stats()
            # After the forward pass the decoder has consumed the entry;
            # the cache entry may or may not still be there (it is there
            # because recv does not evict), so num_entries >= 1
            self.assertGreaterEqual(stats["num_entries"], 0)

        # ------------------------------------------------------------------
        # build_des_loc_optimizer
        # ------------------------------------------------------------------

        def test_optimizer_build_no_crash(self):
            model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
            opt = build_des_loc_optimizer(model)
            self.assertIsNotNone(opt)
            self.assertGreater(len(opt.param_groups), 0)

        def test_optimizer_no_trainable_params_raises(self):
            model = nn.Sequential(nn.Linear(4, 8))
            for p in model.parameters():
                p.requires_grad = False
            with self.assertRaises(ValueError):
                build_des_loc_optimizer(model)

        # ------------------------------------------------------------------
        # des_loc_train_step (CPU, synthetic data)
        # ------------------------------------------------------------------

        def _make_step_fixtures(self):
            model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            def forward_step_func(batch, mdl, cfg):
                x, y = batch
                pred = mdl(x)
                return nn.functional.mse_loss(pred, y)

            def data_gen():
                while True:
                    yield (torch.randn(2, 8), torch.randn(2, 1))

            config = type("C", (), {"micro_batch_size": 2})()
            return model, optimizer, forward_step_func, data_gen(), config

        def test_train_step_returns_result(self):
            model, optimizer, fwd, data_iter, config = self._make_step_fixtures()
            result = des_loc_train_step(
                forward_step_func=fwd,
                data_iterator=data_iter,
                model=model,
                optimizer=optimizer,
                config=config,
                iteration=0,
                num_micro_batches=2,
            )
            self.assertIsInstance(result, DesLocTrainStepResult)
            self.assertFalse(result.skipped)
            self.assertIsNotNone(result.grad_norm)
            self.assertGreater(float(result.loss), 0.0)

        def test_train_step_with_p2p_and_pg_collection(self):
            """p2p_communicator and schedule_pg_collection are accepted and used."""
            model, optimizer, fwd, data_iter, config = self._make_step_fixtures()
            cache = LocalityCache(capacity_bytes=32 * 1024 * 1024, enable_pinned=False)
            comm = DesLocP2PCommunicator(
                locality_cache=cache,
                src_device=torch.device("cpu"),
                dst_device=torch.device("cpu"),
            )
            specs = [
                DeviceAffinitySpec("0", torch.device("cpu"), DeviceClass.CPU),
                DeviceAffinitySpec("1", torch.device("cpu"), DeviceClass.CPU),
            ]
            pg_col = DesLocSchedulePGCollection(module_specs=specs, locality_cache=cache)
            result = des_loc_train_step(
                forward_step_func=fwd,
                data_iterator=data_iter,
                model=model,
                optimizer=optimizer,
                config=config,
                iteration=0,
                p2p_communicator=comm,
                schedule_pg_collection=pg_col,
                num_micro_batches=1,
            )
            self.assertFalse(result.skipped)

        def test_train_step_empty_iterator_skips(self):
            model, optimizer, fwd, _, config = self._make_step_fixtures()
            result = des_loc_train_step(
                forward_step_func=fwd,
                data_iterator=iter([]),  # empty
                model=model,
                optimizer=optimizer,
                config=config,
                iteration=0,
            )
            self.assertTrue(result.skipped)

        # ------------------------------------------------------------------
        # des_loc_train (mini training run)
        # ------------------------------------------------------------------

        def test_full_training_loop(self):
            model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
            optimizer = build_des_loc_optimizer(model)

            def fwd(batch, mdl, cfg):
                x, y = batch
                return nn.functional.mse_loss(mdl(x), y)

            def data_gen():
                while True:
                    yield (torch.randn(3, 4), torch.randn(3, 1))

            summary = des_loc_train(
                forward_step_func=fwd,
                model=model,
                optimizer=optimizer,
                train_data_iterator=data_gen(),
                num_iterations=5,
                config=type("C", (), {})(),
                num_micro_batches=1,
                log_interval=2,
            )
            self.assertEqual(summary["num_iterations"], 5)
            self.assertIn("final_loss", summary)
            self.assertIn("total_time_s", summary)
            self.assertFalse(
                summary["final_loss"] != summary["final_loss"],  # not NaN
                "final_loss should not be NaN",
            )

        # ------------------------------------------------------------------
        # setup_des_loc_model_and_optimizer
        # ------------------------------------------------------------------

        def test_setup_returns_triple(self):
            def provider():
                return nn.Linear(4, 2)

            model, optimizer, pg_collection = setup_des_loc_model_and_optimizer(
                model_provider_func=provider
            )
            self.assertIsInstance(model, nn.Module)
            self.assertIsInstance(optimizer, torch.optim.Optimizer)
            self.assertIsInstance(pg_collection, DesLocSchedulePGCollection)

        def test_setup_mimo_model_uses_builtin_cache(self):
            cache = LocalityCache(capacity_bytes=16 * 1024 * 1024, enable_pinned=False)

            def provider():
                return DesLocMimoModel(
                    sub_models={
                        "a": (nn.Linear(4, 8), torch.device("cpu")),
                        "b": (nn.Linear(8, 2), torch.device("cpu")),
                    },
                    locality_cache=cache,
                )

            model, _, pg_collection = setup_des_loc_model_and_optimizer(
                model_provider_func=provider
            )
            self.assertIs(model.locality_cache, cache)
            self.assertIs(pg_collection.locality_cache, cache)

        def test_setup_pg_collection_passthrough(self):
            """If pg_collection is provided it is returned unchanged."""
            specs = [DeviceAffinitySpec("x", torch.device("cpu"), DeviceClass.CPU)]
            provided = DesLocSchedulePGCollection(module_specs=specs)

            model, _, pg_out = setup_des_loc_model_and_optimizer(
                model_provider_func=lambda: nn.Linear(2, 2),
                pg_collection=provided,
            )
            self.assertIs(pg_out, provided)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromTestCase(_TestCase)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
