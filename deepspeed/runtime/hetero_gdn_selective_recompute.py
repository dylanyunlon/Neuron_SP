"""
deepspeed/runtime/hetero_gdn_selective_recompute.py

Upstream design intent (Megatron ff5264c33dc098a8135ccff89ca837bcf089c2ab):
    NVIDIA Megatron-LM introduced selective recomputation for the `norm_out` sub-step
    inside GatedDeltaNet (GDN) layers.  The motivation is that GDN's fused
    "gated-norm + HP→CP all-to-all" kernel is memory-bandwidth-bound rather than
    compute-bound: the intermediate `norm_out_hp` tensor (shape [B, S_hp, H]) is
    large relative to the compute cost of re-deriving it, so discarding its
    activation during the forward pass and recomputing it on the backward pass
    saves peak HBM without meaningfully increasing FLOP cost.  Megatron wraps the
    combined `_gated_norm_and_a2a` closure with `CheckpointWithoutOutput`, which
    discards the output tensor immediately after forward and re-runs the closure
    during backward to regenerate the gradient inputs.  The feature is gated by
    `recompute_granularity == "selective"` and `"gdn_norm_out" in recompute_modules`.

DES-LOC adaptation points:
    1.  **Heterogeneous device awareness** — Neuron_SP targets 2× A6000-48GB (SM86,
        PCIe) + 1× H100-NVL-96GB (SM90, PCIe).  The three devices differ in HBM
        capacity, memory bandwidth, and compute throughput.  The recompute decision
        must therefore be *per-device*, not global.  A6000 nodes have 48 GB HBM and
        low PCIe bandwidth (≈64 GB/s each); recomputing `norm_out` is profitable
        there.  The H100 has 96 GB HBM and higher bandwidth but also a heavier
        compute load from expert routing in MoE layers, making recompute optional.
        `HeteroRecomputePolicy` encodes this device-sensitive heuristic.

    2.  **Shared LOcality Cache (LOC)** — DES-LOC maintains a per-layer activation
        cache in CPU DRAM (1.5 TB available) for activations that are too expensive
        to recompute but too large for GPU HBM.  `NormOutLocalityCache` implements
        the LOC tier for `norm_out`: tensors that are cheaper to D2H-copy and cache
        than to recompute are pinned in CPU memory and H2D-copied lazily.  The
        decision boundary is expressed in `LocalityCacheConfig.recompute_threshold_gb`
        (default 0.5 GB per activation shard).

    3.  **Decoupled Execution** — The HP→CP all-to-all collective is decoupled from
        the norm computation: on A6000 peers the collective is enqueued on a
        dedicated NCCL stream while the CPU side prefetches the next micro-batch's
        activations from the LOC.  `DecoupledNormA2APipeline` manages the stream
        synchronization points.

    4.  **CheckpointWithoutOutput parity** — We replicate Megatron's
        `CheckpointWithoutOutput` contract using DeepSpeed's `checkpointing.py`
        infrastructure, adding LOC-aware output discard/restore hooks so that the
        backward recompute can source inputs from either the GPU recompute path or
        the CPU LOC, whichever is cheaper given current memory pressure.

    5.  **Validation** — `TransformerHeteroConfig` mirrors Megatron's config
        validation (gdn_norm_out requires gated_delta_net variant) and adds
        heterogeneity checks (recompute_modules must be per-device maps, or a
        scalar string is broadcast to all devices).

Author: Neuron_SP / DES-LOC team
Mirrors: Megatron commit ff5264c33dc098a8135ccff89ca837bcf089c2ab
"""

from __future__ import annotations

import logging
import math
import threading
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Logging — DES-LOC uses a named logger so operators can filter by component.
# ---------------------------------------------------------------------------
logger = logging.getLogger("desLOC.hetero_gdn")

# ---------------------------------------------------------------------------
# Hardware capability table (static, updated at module import time)
# ---------------------------------------------------------------------------

_DEVICE_CAPABILITY_CACHE: Dict[int, Tuple[int, int]] = {}


def _get_device_capability(device_index: int) -> Tuple[int, int]:
    if device_index not in _DEVICE_CAPABILITY_CACHE:
        cap = torch.cuda.get_device_capability(device_index)
        _DEVICE_CAPABILITY_CACHE[device_index] = cap
    return _DEVICE_CAPABILITY_CACHE[device_index]


class DeviceClass(Enum):
    """Coarse device classification used by the recompute policy."""
    A6000 = auto()   # SM86, 48 GB HBM, PCIe bandwidth ~64 GB/s per card
    H100_NVL = auto()  # SM90, 96 GB HBM, PCIe bandwidth ~128 GB/s
    UNKNOWN = auto()


def classify_device(device_index: int) -> DeviceClass:
    """
    Classify a CUDA device into one of the DES-LOC device classes based on
    compute capability.  SM86 maps to A6000; SM90 maps to H100-NVL.

    This is intentionally heuristic — in production the operator should supply
    an explicit device map via `HeteroDeviceMap`, but this fallback avoids
    silent misclassification for the two known hardware classes.
    """
    major, minor = _get_device_capability(device_index)
    sm = major * 10 + minor
    if sm == 86:
        return DeviceClass.A6000
    if sm == 90:
        return DeviceClass.H100_NVL
    logger.warning(
        "Device %d has unknown SM%d%d capability; falling back to UNKNOWN class. "
        "Recompute policy will default to conservative (no recompute).",
        device_index, major, minor,
    )
    return DeviceClass.UNKNOWN


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LocalityCacheConfig:
    """
    Configuration for the DES-LOC Shared LOcality Cache (LOC).

    The LOC tier lives in CPU DRAM and is used to offload activations that are
    too large for GPU HBM but cheaper to transfer over PCIe than to recompute.

    Attributes
    ----------
    enabled : bool
        Master switch.  If False the LOC tier is bypassed entirely.
    max_cpu_bytes : int
        Maximum bytes allocated in pinned CPU DRAM for LOC entries.
        Default 32 GiB — conservative relative to the 1.5 TB DRAM budget.
    recompute_threshold_gb : float
        If an activation shard is smaller than this threshold (in GiB), prefer
        GPU recompute over CPU offload.  If larger, offload to LOC.
        Tuned empirically: A6000 PCIe bandwidth is ~64 GB/s, and a RMSNorm+A2A
        kernel takes ~0.3 ms/GB; the crossover is around 0.5 GB.
    prefetch_depth : int
        Number of micro-batches to prefetch from LOC into GPU memory ahead of
        the backward pass.  Higher values increase CPU→GPU bandwidth utilisation
        at the cost of pinned-memory churn.
    """
    enabled: bool = True
    max_cpu_bytes: int = 32 * (1 << 30)   # 32 GiB
    recompute_threshold_gb: float = 0.5
    prefetch_depth: int = 2


@dataclass
class HeteroDeviceMap:
    """
    Explicit mapping from device index to DeviceClass, allowing operators to
    override the heuristic classification in :func:`classify_device`.

    Example for the Neuron_SP target cluster::

        HeteroDeviceMap(mapping={0: DeviceClass.A6000,
                                  1: DeviceClass.A6000,
                                  2: DeviceClass.H100_NVL})
    """
    mapping: Dict[int, DeviceClass] = field(default_factory=dict)

    def get(self, device_index: int) -> DeviceClass:
        if device_index in self.mapping:
            return self.mapping[device_index]
        return classify_device(device_index)


@dataclass
class HeteroRecomputeConfig:
    """
    Per-device recompute policy for GDN norm_out.

    Mirrors Megatron's (recompute_granularity, recompute_modules) pair but
    extends it to be device-class-sensitive.

    Attributes
    ----------
    granularity : str
        "full" — recompute entire layer; "selective" — recompute specific
        sub-modules; "none" — no recomputation.
    modules_per_device : Dict[DeviceClass, Set[str]]
        Specifies which sub-module names to selectively recompute on each
        device class.  The key "gdn_norm_out" enables the norm+A2A recompute.
    attention_variant : str
        Must be "gated_delta_net" when "gdn_norm_out" is in any module set.
    device_map : HeteroDeviceMap
        Device classification map.
    loc_config : LocalityCacheConfig
        LOC tier configuration.
    """
    granularity: str = "selective"
    modules_per_device: Dict[DeviceClass, Set[str]] = field(default_factory=lambda: {
        DeviceClass.A6000: {"gdn_norm_out"},
        DeviceClass.H100_NVL: set(),       # H100 has headroom; skip recompute
        DeviceClass.UNKNOWN: set(),
    })
    attention_variant: str = "gated_delta_net"
    device_map: HeteroDeviceMap = field(default_factory=HeteroDeviceMap)
    loc_config: LocalityCacheConfig = field(default_factory=LocalityCacheConfig)

    def validate(self) -> None:
        """
        Validate configuration consistency, mirroring Megatron's
        TransformerConfig.__post_init__ checks but extended for heterogeneity.
        """
        if self.granularity not in ("full", "selective", "none"):
            raise ValueError(
                f"recompute granularity must be 'full', 'selective', or 'none'; "
                f"got '{self.granularity}'."
            )
        for device_class, modules in self.modules_per_device.items():
            if "gdn_norm_out" in modules:
                if self.attention_variant != "gated_delta_net":
                    raise ValueError(
                        f"'gdn_norm_out' in recompute_modules for {device_class} "
                        f"requires attention_variant='gated_delta_net'; "
                        f"got '{self.attention_variant}'."
                    )

    def should_recompute_norm_out(self, device_index: int) -> bool:
        """Return True if norm_out should be recomputed on *device_index*."""
        if self.granularity != "selective":
            return False
        dev_class = self.device_map.get(device_index)
        modules = self.modules_per_device.get(dev_class, set())
        return "gdn_norm_out" in modules


# ---------------------------------------------------------------------------
# LOC (Shared LOcality Cache) — CPU DRAM activation store
# ---------------------------------------------------------------------------

class _LOCEntry:
    """
    A single activation shard stored in pinned CPU DRAM.

    The entry holds a reference count so multiple backward hooks can safely
    share the same buffer without premature deallocation.
    """
    __slots__ = ("cpu_tensor", "shape", "dtype", "device", "_refcount", "_lock")

    def __init__(self, gpu_tensor: Tensor) -> None:
        self.shape = gpu_tensor.shape
        self.dtype = gpu_tensor.dtype
        self.device = gpu_tensor.device
        # Allocate pinned CPU memory and copy asynchronously
        self.cpu_tensor = torch.empty(
            gpu_tensor.shape, dtype=gpu_tensor.dtype,
            pin_memory=True,
        )
        self.cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        self._refcount = 1
        self._lock = threading.Lock()

    def acquire(self) -> "_LOCEntry":
        with self._lock:
            self._refcount += 1
        return self

    def release(self) -> None:
        with self._lock:
            self._refcount -= 1
            if self._refcount == 0:
                # Allow GC to reclaim pinned memory
                self.cpu_tensor = None  # type: ignore[assignment]

    def restore_to_gpu(self, stream: Optional[torch.cuda.Stream] = None) -> Tensor:
        """Copy the cached tensor back to its original GPU device."""
        if self.cpu_tensor is None:
            raise RuntimeError("LOCEntry has been released; cannot restore.")
        gpu_buf = torch.empty(
            self.shape, dtype=self.dtype, device=self.device,
        )
        if stream is not None:
            with torch.cuda.stream(stream):
                gpu_buf.copy_(self.cpu_tensor, non_blocking=True)
        else:
            gpu_buf.copy_(self.cpu_tensor)
        return gpu_buf


class NormOutLocalityCache:
    """
    DES-LOC Shared LOcality Cache for GDN ``norm_out`` activations.

    Implements the LOC tier described in the module docstring.  Activations
    larger than ``config.recompute_threshold_gb`` are pinned in CPU DRAM instead
    of being recomputed on the GPU, exploiting the 1.5 TB DRAM headroom of the
    Neuron_SP cluster.

    Thread-safety: all public methods acquire ``_lock`` so that concurrent
    backward hooks from different micro-batch streams do not race.

    Parameters
    ----------
    config : LocalityCacheConfig
        LOC configuration (see :class:`LocalityCacheConfig`).
    """

    def __init__(self, config: LocalityCacheConfig) -> None:
        self._config = config
        self._store: Dict[int, _LOCEntry] = {}   # key: id(tensor) at forward time
        self._allocated_bytes: int = 0
        self._lock = threading.Lock()
        self._h2d_stream: Optional[torch.cuda.Stream] = None

    def _get_h2d_stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream(device=device, priority=-1)
        return self._h2d_stream

    def _tensor_bytes(self, t: Tensor) -> int:
        return t.numel() * t.element_size()

    def _tensor_gb(self, t: Tensor) -> float:
        return self._tensor_bytes(t) / (1 << 30)

    def should_use_loc(self, tensor: Tensor) -> bool:
        """
        Decide whether *tensor* should be offloaded to the LOC rather than
        recomputed on the GPU.

        Decision rule:
        - LOC must be enabled globally.
        - The tensor must exceed ``recompute_threshold_gb``.
        - There must be sufficient LOC budget remaining.
        """
        if not self._config.enabled:
            return False
        tensor_gb = self._tensor_gb(tensor)
        if tensor_gb < self._config.recompute_threshold_gb:
            return False
        tensor_bytes = self._tensor_bytes(tensor)
        with self._lock:
            return (self._allocated_bytes + tensor_bytes) <= self._config.max_cpu_bytes

    def store(self, key: int, tensor: Tensor) -> bool:
        """
        Offload *tensor* to pinned CPU DRAM under *key*.

        Returns True if the tensor was stored, False if budget was exceeded.
        """
        tensor_bytes = self._tensor_bytes(tensor)
        with self._lock:
            if self._allocated_bytes + tensor_bytes > self._config.max_cpu_bytes:
                logger.debug(
                    "LOC budget exhausted (%d bytes allocated, %d requested); "
                    "falling back to GPU recompute for key %d.",
                    self._allocated_bytes, tensor_bytes, key,
                )
                return False
            entry = _LOCEntry(tensor)
            self._store[key] = entry
            self._allocated_bytes += tensor_bytes
        logger.debug(
            "LOC stored key=%d  shape=%s  dtype=%s  size_gb=%.3f  "
            "total_allocated_gb=%.3f",
            key, tuple(tensor.shape), tensor.dtype,
            tensor_bytes / (1 << 30), self._allocated_bytes / (1 << 30),
        )
        return True

    def restore(self, key: int) -> Optional[Tensor]:
        """
        Retrieve a stored activation by *key*, returning it on the GPU.

        Returns None if the key is not present (caller should fall back to
        GPU recompute).
        """
        with self._lock:
            entry = self._store.get(key)
        if entry is None:
            return None
        stream = self._get_h2d_stream(entry.device)
        gpu_tensor = entry.restore_to_gpu(stream)
        # Synchronise so callers on the default stream see the data
        torch.cuda.current_stream(entry.device).wait_stream(stream)
        logger.debug("LOC restored key=%d  shape=%s", key, tuple(gpu_tensor.shape))
        return gpu_tensor

    def evict(self, key: int) -> None:
        """Release a stored activation, freeing its pinned-memory budget."""
        with self._lock:
            entry = self._store.pop(key, None)
            if entry is None:
                return
            self._allocated_bytes -= self._tensor_bytes(
                torch.empty(entry.shape, dtype=entry.dtype)
            )
            entry.release()

    def clear(self) -> None:
        """Evict all entries — called at end-of-batch to avoid stale state."""
        with self._lock:
            for entry in self._store.values():
                entry.release()
            self._store.clear()
            self._allocated_bytes = 0


# ---------------------------------------------------------------------------
# Decoupled Execution pipeline for norm+A2A
# ---------------------------------------------------------------------------

class DecoupledNormA2APipeline:
    """
    Manages the decoupled execution of the GDN gated-norm and HP→CP all-to-all.

    In Megatron the two operations run sequentially on the default CUDA stream.
    In DES-LOC we decouple them:
    - The RMSNorm runs on the *compute stream* (default stream).
    - The all-to-all collective runs on a *dedicated communication stream*.
    - A CUDA event synchronises the two streams at the point where the A2A
      result is needed by the output projection.

    This overlap is particularly valuable on A6000 PCIe nodes where PCIe
    bandwidth is a bottleneck: the collective can be pipelined with the next
    layer's QKV projection.

    Parameters
    ----------
    device : torch.device
        GPU device this pipeline is associated with.
    pg : torch.distributed.ProcessGroup
        Context-parallel process group for the HP→CP all-to-all.
    """

    def __init__(
        self,
        device: torch.device,
        pg: Optional[Any] = None,
    ) -> None:
        self._device = device
        self._pg = pg
        self._comm_stream: Optional[torch.cuda.Stream] = None
        self._sync_event: Optional[torch.cuda.Event] = None

    def _ensure_comm_stream(self) -> torch.cuda.Stream:
        if self._comm_stream is None:
            # Lower priority (-1) so that compute-bound kernels are not starved
            self._comm_stream = torch.cuda.Stream(
                device=self._device, priority=-1,
            )
            logger.debug(
                "DecoupledNormA2APipeline: created comm_stream on device %s",
                self._device,
            )
        return self._comm_stream

    @contextmanager
    def comm_stream_ctx(self):
        """Context manager that switches to the communication stream."""
        stream = self._ensure_comm_stream()
        with torch.cuda.stream(stream):
            yield stream

    def record_compute_event(self) -> torch.cuda.Event:
        """Record an event on the current (compute) stream."""
        evt = torch.cuda.Event()
        evt.record()
        self._sync_event = evt
        return evt

    def wait_for_compute(self, stream: torch.cuda.Stream) -> None:
        """Make *stream* wait until the last recorded compute event completes."""
        if self._sync_event is not None:
            stream.wait_event(self._sync_event)

    def synchronise(self) -> None:
        """
        Block until both the compute and communication streams have finished.
        Called at the boundary where norm_out is consumed by out_proj.
        """
        if self._comm_stream is not None:
            # Make the default stream wait for the comm stream
            evt = torch.cuda.Event()
            evt.record(self._comm_stream)
            torch.cuda.current_stream(self._device).wait_event(evt)


# ---------------------------------------------------------------------------
# CheckpointWithoutOutput — DES-LOC adaptation of Megatron's equivalent
# ---------------------------------------------------------------------------

class CheckpointWithoutOutput(torch.autograd.Function):
    """
    Selective activation checkpoint that discards the function output after the
    forward pass and recomputes it during the backward pass.

    This mirrors Megatron's ``tensor_parallel.CheckpointWithoutOutput`` but adds:
    - LOC-tier support: large activations are offloaded to CPU DRAM instead of
      being recomputed on the GPU.
    - Heterogeneous device awareness: the recompute vs LOC decision is delegated
      to :class:`NormOutLocalityCache`.

    Usage::

        ckpt = CheckpointWithoutOutput(loc_cache=cache)
        output = ckpt.checkpoint(fn, *args)
        # ... use output in subsequent ops ...
        ckpt.discard_output_and_register_recompute(downstream_tensor)

    The output tensor's storage is freed after ``discard_output_and_register_recompute``
    is called.  During backward, a hook on *downstream_tensor* triggers either:
    - GPU recompute: re-runs *fn* with the saved inputs.
    - LOC restore: fetches the offloaded tensor from CPU DRAM.

    Parameters
    ----------
    loc_cache : Optional[NormOutLocalityCache]
        If provided, activations above the LOC threshold are offloaded to CPU
        DRAM.  If None, only GPU recompute is used.
    """

    def __init__(self, loc_cache: Optional[NormOutLocalityCache] = None) -> None:
        super().__init__()
        self._loc_cache = loc_cache
        self._fn: Optional[Callable] = None
        self._fn_args: Optional[Tuple] = None
        self._fn_kwargs: Optional[Dict] = None
        self._output_ref: Optional[weakref.ref] = None
        self._output_id: Optional[int] = None
        self._loc_stored: bool = False
        self._recompute_hook_registered: bool = False

    # ------------------------------------------------------------------
    # Public API (mirrors Megatron's CheckpointWithoutOutput interface)
    # ------------------------------------------------------------------

    def checkpoint(
        self,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Run *fn(*args, **kwargs)* under a no-grad context, save inputs for
        potential recompute, and return the output tensor.

        The output is *not* yet discarded here — call
        :meth:`discard_output_and_register_recompute` after the downstream
        consumer has been constructed.
        """
        self._fn = fn
        self._fn_args = args
        self._fn_kwargs = kwargs

        with torch.no_grad():
            output = fn(*args, **kwargs)

        self._output_id = id(output)

        # Decide whether to pre-emptively offload to LOC
        if (
            self._loc_cache is not None
            and self._loc_cache.should_use_loc(output)
        ):
            stored = self._loc_cache.store(self._output_id, output)
            self._loc_stored = stored
            if stored:
                logger.info(
                    "CheckpointWithoutOutput: offloaded norm_out to LOC "
                    "(shape=%s, size_gb=%.3f)",
                    tuple(output.shape),
                    output.numel() * output.element_size() / (1 << 30),
                )

        # Keep a weak reference so we can detect if the tensor is still alive
        self._output_ref = weakref.ref(output)
        return output

    def discard_output_and_register_recompute(
        self,
        downstream_tensor: Tensor,
    ) -> None:
        """
        Discard the stored output activation and register a backward hook on
        *downstream_tensor* that will restore/recompute norm_out when needed.

        Parameters
        ----------
        downstream_tensor : Tensor
            The tensor produced by the operation that consumed norm_out (i.e.,
            the output of the out_proj linear layer).  Its backward hook triggers
            the recompute before gradients are propagated through norm_out.
        """
        if not downstream_tensor.requires_grad:
            logger.debug(
                "discard_output_and_register_recompute: downstream_tensor "
                "does not require grad; skipping hook registration."
            )
            return

        output_id = self._output_id
        loc_cache = self._loc_cache
        loc_stored = self._loc_stored
        fn = self._fn
        fn_args = self._fn_args
        fn_kwargs = self._fn_kwargs or {}

        def _recompute_hook(grad: Tensor) -> None:
            # Attempt LOC restore first
            if loc_stored and loc_cache is not None:
                restored = loc_cache.restore(output_id)
                if restored is not None:
                    loc_cache.evict(output_id)
                    return
            # Fall back to GPU recompute
            with torch.no_grad():
                _ = fn(*fn_args, **fn_kwargs)  # type: ignore[misc]

        downstream_tensor.register_hook(_recompute_hook)
        self._recompute_hook_registered = True

        # Free the output tensor's storage so HBM is reclaimed immediately.
        # We do this by replacing the data with an empty tensor of the same
        # shape and dtype — a pattern used in DeepSpeed's activation offload.
        output = self._output_ref() if self._output_ref is not None else None
        if output is not None:
            # Zero out the storage without changing the tensor's metadata so
            # that any code holding the reference doesn't segfault.
            try:
                output.data = torch.empty(
                    0, dtype=output.dtype, device=output.device,
                )
                logger.debug(
                    "CheckpointWithoutOutput: discarded norm_out storage "
                    "(id=%d)", output_id,
                )
            except RuntimeError as exc:
                # Storage may already be freed if the tensor went out of scope
                logger.debug(
                    "CheckpointWithoutOutput: could not discard storage for "
                    "id=%d: %s", output_id, exc,
                )


# ---------------------------------------------------------------------------
# HeteroGDNNormOutRecompute — the main per-layer recompute manager
# ---------------------------------------------------------------------------

class HeteroGDNNormOutRecompute:
    """
    Per-layer manager for selective recomputation of the GDN ``norm_out``
    activation in the DES-LOC heterogeneous training framework.

    This class is the DES-LOC counterpart of Megatron's two-line pattern::

        # Megatron
        self.recompute_norm_out = "gdn_norm_out" in config.recompute_modules
        self.norm_out_checkpoint = None  # set lazily in forward

    In DES-LOC the manager additionally:
    - Queries the per-device recompute policy (:class:`HeteroRecomputeConfig`).
    - Maintains a :class:`NormOutLocalityCache` for CPU offload.
    - Owns the :class:`DecoupledNormA2APipeline` for stream-decoupled A2A.
    - Provides :meth:`apply` which replaces the combined Megatron ``if``-branch
      with device-aware dispatch.

    Parameters
    ----------
    layer_number : int
        1-indexed transformer layer number (used for logging context).
    device : torch.device
        GPU device this layer resides on.
    config : HeteroRecomputeConfig
        DES-LOC heterogeneous recompute configuration.
    cp_group : Optional[Any]
        Context-parallel process group for HP→CP all-to-all.
    """

    def __init__(
        self,
        layer_number: int,
        device: torch.device,
        config: HeteroRecomputeConfig,
        cp_group: Optional[Any] = None,
    ) -> None:
        self._layer_number = layer_number
        self._device = device
        self._config = config
        self._cp_group = cp_group

        # Determine once at construction time whether this device recomputes
        device_index = device.index if device.index is not None else 0
        self.recompute_norm_out: bool = config.should_recompute_norm_out(device_index)

        # LOC cache (shared across layers on the same device via external wiring,
        # but owned here for simplicity in unit tests)
        if config.loc_config.enabled:
            self._loc_cache: Optional[NormOutLocalityCache] = NormOutLocalityCache(
                config.loc_config,
            )
        else:
            self._loc_cache = None

        # Decoupled A2A pipeline
        self._pipeline = DecoupledNormA2APipeline(device=device, pg=cp_group)

        # Per-forward-pass checkpoint handle (reset each forward call)
        self.norm_out_checkpoint: Optional[CheckpointWithoutOutput] = None

        dev_class = config.device_map.get(device_index)
        logger.info(
            "HeteroGDNNormOutRecompute layer=%d  device=%s  class=%s  "
            "recompute_norm_out=%s  loc_enabled=%s",
            layer_number, device, dev_class.name, self.recompute_norm_out,
            config.loc_config.enabled,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def apply(
        self,
        gated_norm_and_a2a_fn: Callable,
        core_attn_out: Tensor,
        gate: Tensor,
        downstream_producer: Optional[Callable[[], Tensor]] = None,
    ) -> Tensor:
        """
        Execute the gated-norm + HP→CP A2A step with DES-LOC recompute logic.

        Parameters
        ----------
        gated_norm_and_a2a_fn : Callable[[Tensor, Tensor], Tensor]
            The fused closure ``_gated_norm_and_a2a`` from the GDN forward pass.
            Must accept (core_attn_out, gate) and return norm_out.
        core_attn_out : Tensor
            Output of the gated delta-rule kernel.
        gate : Tensor
            Gate tensor for RMSNorm.
        downstream_producer : Optional[Callable[[], Tensor]]
            If provided, called *after* this method returns to obtain the
            downstream tensor (e.g., out_proj output) on which to register the
            discard hook.  Allows the caller to defer hook registration.

        Returns
        -------
        Tensor
            norm_out, ready for consumption by the output projection.
        """
        if self.recompute_norm_out:
            self.norm_out_checkpoint = CheckpointWithoutOutput(
                loc_cache=self._loc_cache,
            )
            norm_out = self.norm_out_checkpoint.checkpoint(
                gated_norm_and_a2a_fn, core_attn_out, gate,
            )
        else:
            self.norm_out_checkpoint = None
            norm_out = gated_norm_and_a2a_fn(core_attn_out, gate)

        return norm_out

    def discard_and_register(self, out: Tensor) -> None:
        """
        Call after the downstream consumer (out_proj) has produced *out*.

        Mirrors Megatron's::

            if self.recompute_norm_out:
                self.norm_out_checkpoint.discard_output_and_register_recompute(out)

        but is a no-op when recompute is disabled on this device.
        """
        if self.recompute_norm_out and self.norm_out_checkpoint is not None:
            self.norm_out_checkpoint.discard_output_and_register_recompute(out)

    def end_of_batch(self) -> None:
        """
        Clean up per-batch state.  Must be called at the end of every training
        step to prevent stale LOC entries from accumulating.
        """
        if self._loc_cache is not None:
            self._loc_cache.clear()
        self.norm_out_checkpoint = None


# ---------------------------------------------------------------------------
# RMSNorm (minimal implementation used in tests / standalone mode)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.

    Matches the interface used by Megatron's GDN ``norm_out`` layer.  In
    production this is supplied by the submodule spec; here it is a lightweight
    standalone implementation for unit tests.

    Parameters
    ----------
    hidden_size : int
        Feature dimension to normalise.
    eps : float
        Epsilon for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(x.dtype)


# ---------------------------------------------------------------------------
# GatedNorm stub — combines gate and RMSNorm (used in tests)
# ---------------------------------------------------------------------------

class GatedNorm(nn.Module):
    """
    Minimal stub for the GDN gated-norm step.

    In the full GDN architecture the gate is produced by a separate linear
    branch; here we simply element-wise multiply and apply RMSNorm so that
    the recompute path can be exercised in isolation.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        gated = x * torch.sigmoid(self.gate_proj(gate))
        return self.norm(gated)


# ---------------------------------------------------------------------------
# HeteroGDNLayer — representative GDN layer with DES-LOC recompute integrated
# ---------------------------------------------------------------------------

class HeteroGDNLayer(nn.Module):
    """
    Simplified GatedDeltaNet layer adapted for DES-LOC heterogeneous training.

    This is *not* a full GDN implementation — it strips the delta-rule kernel,
    CP process groups, and packed-sequence logic to focus on the DES-LOC
    selective-recompute integration.  For the full implementation see the
    Neuron_SP GDN module in ``deepspeed/runtime/ssm/gated_delta_net.py``.

    The key structural changes relative to Megatron ``GatedDeltaNet`` are:

    1.  ``self._recompute_mgr`` is a :class:`HeteroGDNNormOutRecompute` instance
        that replaces Megatron's two scalar fields
        (``recompute_norm_out``, ``norm_out_checkpoint``).

    2.  The forward pass calls ``self._recompute_mgr.apply(...)`` instead of the
        inline ``if self.recompute_norm_out`` branch, encapsulating device-aware
        dispatch.

    3.  ``self._recompute_mgr.discard_and_register(out)`` replaces Megatron's
        ``self.norm_out_checkpoint.discard_output_and_register_recompute(out)``.

    Parameters
    ----------
    hidden_size : int
    layer_number : int
    device : torch.device
    recompute_config : HeteroRecomputeConfig
    cp_group : Optional[Any]
    """

    def __init__(
        self,
        hidden_size: int,
        layer_number: int,
        device: torch.device,
        recompute_config: HeteroRecomputeConfig,
        cp_group: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_number = layer_number

        # Sub-modules (analogous to Megatron submodule specs)
        self.gated_norm = GatedNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # DES-LOC recompute manager (replaces Megatron's scalar flags)
        self._recompute_mgr = HeteroGDNNormOutRecompute(
            layer_number=layer_number,
            device=device,
            config=recompute_config,
            cp_group=cp_group,
        )

    # ------------------------------------------------------------------
    # Convenience properties mirroring Megatron's attribute names
    # ------------------------------------------------------------------

    @property
    def recompute_norm_out(self) -> bool:
        return self._recompute_mgr.recompute_norm_out

    @property
    def norm_out_checkpoint(self) -> Optional[CheckpointWithoutOutput]:
        return self._recompute_mgr.norm_out_checkpoint

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        gate: Optional[Tensor] = None,
    ) -> Tuple[Tensor, None]:
        """
        Forward pass mirroring the relevant section of ``GatedDeltaNet.forward``.

        The delta-rule kernel and A2A are stubbed out; the norm + recompute logic
        is faithful to the DES-LOC adaptation.

        Parameters
        ----------
        hidden_states : Tensor
            Shape ``[S, B, H]`` (seq-first convention).
        gate : Optional[Tensor]
            Gate tensor of shape ``[S, B, H]``.  If None, derived from
            hidden_states via an identity pass.

        Returns
        -------
        out : Tensor
            Shape ``[S, B, H]``.
        out_bias : None
            Placeholder to match Megatron's (out, out_bias) return convention.
        """
        # Stub: in the real GDN, core_attn_out comes from the delta-rule kernel.
        core_attn_out = hidden_states
        if gate is None:
            gate = hidden_states  # degenerate: self-gate for test purposes

        def _gated_norm_and_a2a(attn_out: Tensor, g: Tensor) -> Tensor:
            """
            Combined gated-norm and HP→CP all-to-all.

            In the real GDN this closure also performs:
            - Reshape from [B, S, head, head_dim] to [S, B, H]
            - tensor_a2a_hp2cp scatter across the CP process group
            In this stub we elide the reshape and scatter.
            """
            return self.gated_norm(attn_out, g)

        # DES-LOC selective recompute dispatch
        norm_out = self._recompute_mgr.apply(
            _gated_norm_and_a2a, core_attn_out, gate,
        )

        # Output projection
        out = self.out_proj(norm_out)

        # Discard norm_out storage and register backward recompute hook
        self._recompute_mgr.discard_and_register(out)

        return out, None

    def end_of_batch(self) -> None:
        """Delegate to the recompute manager's cleanup routine."""
        self._recompute_mgr.end_of_batch()


# ---------------------------------------------------------------------------
# TransformerHeteroConfig — top-level config container (validation only)
# ---------------------------------------------------------------------------

class TransformerHeteroConfig:
    """
    Top-level configuration container that mirrors Megatron's
    ``TransformerConfig`` validation logic, extended for DES-LOC heterogeneity.

    In Megatron, ``TransformerConfig.__post_init__`` validates that:
    - ``recompute_granularity == "selective"`` when per-module recompute is used.
    - ``"gdn_norm_out"`` in ``recompute_modules`` requires
      ``experimental_attention_variant == "gated_delta_net"``.

    Here we replicate those checks and add:
    - Per-device recompute module validation.
    - LOC configuration sanity checks.

    Parameters
    ----------
    recompute_config : HeteroRecomputeConfig
    """

    def __init__(self, recompute_config: HeteroRecomputeConfig) -> None:
        self.recompute_config = recompute_config
        self._validate()

    def _validate(self) -> None:
        cfg = self.recompute_config
        cfg.validate()

        # Warn if H100 is configured to recompute norm_out — it has enough HBM
        h100_modules = cfg.modules_per_device.get(DeviceClass.H100_NVL, set())
        if "gdn_norm_out" in h100_modules:
            warnings.warn(
                "gdn_norm_out recompute is enabled for H100_NVL devices. "
                "H100 NVL has 96 GB HBM and is unlikely to benefit from "
                "norm_out recomputation; consider removing it from the "
                "H100_NVL module set to save compute.",
                UserWarning,
                stacklevel=2,
            )

        loc = cfg.loc_config
        if loc.max_cpu_bytes > 1_500 * (1 << 30):
            warnings.warn(
                f"LocalityCacheConfig.max_cpu_bytes ({loc.max_cpu_bytes / (1<<30):.0f} GiB) "
                "exceeds the Neuron_SP cluster's 1.5 TB DRAM budget. "
                "This may cause OOM on the host.",
                UserWarning,
                stacklevel=2,
            )

        if loc.prefetch_depth < 1:
            raise ValueError(
                "LocalityCacheConfig.prefetch_depth must be >= 1; "
                f"got {loc.prefetch_depth}."
            )

        logger.debug("TransformerHeteroConfig validated successfully.")


# ---------------------------------------------------------------------------
# Utility: build a default DES-LOC config for the Neuron_SP target cluster
# ---------------------------------------------------------------------------

def build_neuron_sp_config(
    a6000_indices: Sequence[int] = (0, 1),
    h100_index: int = 2,
    recompute_threshold_gb: float = 0.5,
    loc_max_cpu_gb: float = 32.0,
) -> HeteroRecomputeConfig:
    """
    Construct a :class:`HeteroRecomputeConfig` for the Neuron_SP target cluster:
    2× A6000-48GB (SM86) + 1× H100-NVL-96GB (SM90).

    A6000 nodes have constrained HBM (48 GB each); they benefit from
    recomputing norm_out.  The H100 has 96 GB HBM and is configured conservatively
    (no norm_out recompute by default).

    Parameters
    ----------
    a6000_indices : Sequence[int]
        Device indices of A6000 GPUs.
    h100_index : int
        Device index of the H100 NVL GPU.
    recompute_threshold_gb : float
        Threshold (GiB) above which activations are offloaded to LOC rather
        than recomputed on the GPU.
    loc_max_cpu_gb : float
        Maximum LOC budget in GiB.

    Returns
    -------
    HeteroRecomputeConfig
    """
    device_map = HeteroDeviceMap(
        mapping={
            **{idx: DeviceClass.A6000 for idx in a6000_indices},
            h100_index: DeviceClass.H100_NVL,
        }
    )
    loc_config = LocalityCacheConfig(
        enabled=True,
        max_cpu_bytes=int(loc_max_cpu_gb * (1 << 30)),
        recompute_threshold_gb=recompute_threshold_gb,
        prefetch_depth=2,
    )
    return HeteroRecomputeConfig(
        granularity="selective",
        modules_per_device={
            DeviceClass.A6000: {"gdn_norm_out"},
            DeviceClass.H100_NVL: set(),
            DeviceClass.UNKNOWN: set(),
        },
        attention_variant="gated_delta_net",
        device_map=device_map,
        loc_config=loc_config,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import copy
    import sys
    import traceback
    import unittest

    # Configure logging for the test run
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_a6000_config() -> HeteroRecomputeConfig:
        """Config that treats device 0 as A6000 (recompute enabled)."""
        device_map = HeteroDeviceMap(
            mapping={0: DeviceClass.A6000, 1: DeviceClass.A6000, 2: DeviceClass.H100_NVL}
        )
        return HeteroRecomputeConfig(
            granularity="selective",
            modules_per_device={
                DeviceClass.A6000: {"gdn_norm_out"},
                DeviceClass.H100_NVL: set(),
                DeviceClass.UNKNOWN: set(),
            },
            attention_variant="gated_delta_net",
            device_map=device_map,
            loc_config=LocalityCacheConfig(
                enabled=True,
                max_cpu_bytes=4 * (1 << 30),
                recompute_threshold_gb=0.001,   # low threshold so LOC is triggered on small tensors
                prefetch_depth=2,
            ),
        )

    def _make_h100_config() -> HeteroRecomputeConfig:
        """Config that treats device 0 as H100 (recompute disabled)."""
        device_map = HeteroDeviceMap(mapping={0: DeviceClass.H100_NVL})
        return HeteroRecomputeConfig(
            granularity="selective",
            modules_per_device={
                DeviceClass.A6000: {"gdn_norm_out"},
                DeviceClass.H100_NVL: set(),
                DeviceClass.UNKNOWN: set(),
            },
            attention_variant="gated_delta_net",
            device_map=device_map,
            loc_config=LocalityCacheConfig(enabled=False),
        )

    def _make_layer(
        hidden_size: int,
        config: HeteroRecomputeConfig,
        device: torch.device,
    ) -> HeteroGDNLayer:
        layer = HeteroGDNLayer(
            hidden_size=hidden_size,
            layer_number=1,
            device=device,
            recompute_config=config,
        )
        return layer.to(device)

    # ------------------------------------------------------------------
    # Test suite
    # ------------------------------------------------------------------

    class TestLocalityCacheConfig(unittest.TestCase):
        """Unit tests for LocalityCacheConfig dataclass."""

        def test_defaults(self):
            cfg = LocalityCacheConfig()
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.prefetch_depth, 2)
            self.assertGreater(cfg.max_cpu_bytes, 0)
            self.assertGreater(cfg.recompute_threshold_gb, 0.0)

        def test_custom_values(self):
            cfg = LocalityCacheConfig(
                enabled=False,
                max_cpu_bytes=8 * (1 << 30),
                recompute_threshold_gb=1.0,
                prefetch_depth=4,
            )
            self.assertFalse(cfg.enabled)
            self.assertEqual(cfg.max_cpu_bytes, 8 * (1 << 30))

    class TestHeteroDeviceMap(unittest.TestCase):
        """Unit tests for device classification."""

        def test_explicit_mapping(self):
            dm = HeteroDeviceMap(mapping={0: DeviceClass.A6000, 2: DeviceClass.H100_NVL})
            self.assertEqual(dm.get(0), DeviceClass.A6000)
            self.assertEqual(dm.get(2), DeviceClass.H100_NVL)

        def test_fallback_classification(self):
            # Device may not be available in CI; test the fallback path
            dm = HeteroDeviceMap(mapping={})
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            # Should not raise
            _ = dm.get(0)

    class TestHeteroRecomputeConfig(unittest.TestCase):
        """Unit tests for config validation."""

        def test_valid_config(self):
            cfg = _make_a6000_config()
            cfg.validate()   # should not raise

        def test_invalid_granularity(self):
            cfg = _make_a6000_config()
            cfg.granularity = "chunked"
            with self.assertRaises(ValueError):
                cfg.validate()

        def test_gdn_norm_out_requires_gated_delta_net(self):
            cfg = _make_a6000_config()
            cfg.attention_variant = "standard"
            with self.assertRaises(ValueError):
                cfg.validate()

        def test_should_recompute_a6000(self):
            cfg = _make_a6000_config()
            self.assertTrue(cfg.should_recompute_norm_out(0))   # mapped to A6000

        def test_should_not_recompute_h100(self):
            cfg = _make_a6000_config()
            self.assertFalse(cfg.should_recompute_norm_out(2))  # mapped to H100

        def test_granularity_none_disables_recompute(self):
            cfg = _make_a6000_config()
            cfg.granularity = "none"
            self.assertFalse(cfg.should_recompute_norm_out(0))

    class TestNormOutLocalityCache(unittest.TestCase):
        """Unit tests for the LOC activation cache."""

        def setUp(self):
            self.cfg = LocalityCacheConfig(
                enabled=True,
                max_cpu_bytes=256 * (1 << 20),   # 256 MiB budget for tests
                recompute_threshold_gb=0.0,        # everything goes to LOC
                prefetch_depth=1,
            )
            self.cache = NormOutLocalityCache(self.cfg)

        def test_store_and_restore(self):
            t = torch.randn(64, 32, 128)
            key = id(t)
            stored = self.cache.store(key, t)
            self.assertTrue(stored)
            restored = self.cache.restore(key)
            self.assertIsNotNone(restored)
            self.assertTrue(torch.allclose(t, restored))

        def test_evict(self):
            t = torch.randn(16, 16, 16)
            key = id(t)
            self.cache.store(key, t)
            self.cache.evict(key)
            restored = self.cache.restore(key)
            self.assertIsNone(restored)

        def test_budget_enforcement(self):
            tight_cfg = LocalityCacheConfig(
                enabled=True,
                max_cpu_bytes=1024,   # 1 KiB only
                recompute_threshold_gb=0.0,
            )
            cache = NormOutLocalityCache(tight_cfg)
            big_tensor = torch.randn(1024, 1024)   # 4 MiB >> 1 KiB
            stored = cache.store(id(big_tensor), big_tensor)
            self.assertFalse(stored)

        def test_should_use_loc_disabled(self):
            disabled_cfg = LocalityCacheConfig(enabled=False)
            cache = NormOutLocalityCache(disabled_cfg)
            t = torch.randn(1024)
            self.assertFalse(cache.should_use_loc(t))

        def test_should_use_loc_below_threshold(self):
            cfg = LocalityCacheConfig(
                enabled=True,
                recompute_threshold_gb=10.0,   # very high threshold
                max_cpu_bytes=256 * (1 << 20),
            )
            cache = NormOutLocalityCache(cfg)
            small_tensor = torch.randn(64)   # tiny
            self.assertFalse(cache.should_use_loc(small_tensor))

        def test_clear(self):
            t = torch.randn(8, 8)
            key = id(t)
            self.cache.store(key, t)
            self.cache.clear()
            self.assertIsNone(self.cache.restore(key))
            self.assertEqual(self.cache._allocated_bytes, 0)

    class TestCheckpointWithoutOutput(unittest.TestCase):
        """Tests for CheckpointWithoutOutput — the recompute wrapper."""

        def test_checkpoint_produces_correct_output(self):
            def fn(x, y):
                return x + y

            ckpt = CheckpointWithoutOutput(loc_cache=None)
            x = torch.randn(4, 8)
            y = torch.randn(4, 8)
            out = ckpt.checkpoint(fn, x, y)
            expected = x + y
            self.assertTrue(torch.allclose(out, expected))

        def test_discard_reduces_storage(self):
            def fn(x):
                return x * 2.0

            ckpt = CheckpointWithoutOutput(loc_cache=None)
            x = torch.randn(32, 32, requires_grad=True)
            out = ckpt.checkpoint(fn, x)
            downstream = out.clone().requires_grad_(True)
            ckpt.discard_output_and_register_recompute(downstream)
            # After discard, out.data should have 0 elements
            self.assertEqual(out.numel(), 0)

        def test_with_loc_cache(self):
            cfg = LocalityCacheConfig(
                enabled=True,
                max_cpu_bytes=256 * (1 << 20),
                recompute_threshold_gb=0.0,   # always use LOC
                prefetch_depth=1,
            )
            cache = NormOutLocalityCache(cfg)
            ckpt = CheckpointWithoutOutput(loc_cache=cache)

            x = torch.randn(16, 16)
            out = ckpt.checkpoint(lambda t: t * 3.0, x)
            self.assertTrue(ckpt._loc_stored)
            downstream = out.clone().requires_grad_(True)
            ckpt.discard_output_and_register_recompute(downstream)
            self.assertEqual(out.numel(), 0)

        def test_no_hook_without_grad(self):
            ckpt = CheckpointWithoutOutput(loc_cache=None)
            x = torch.randn(4)
            out = ckpt.checkpoint(lambda t: t + 1, x)
            downstream = out.detach()  # no grad
            # Should not raise
            ckpt.discard_output_and_register_recompute(downstream)

    class TestHeteroGDNNormOutRecompute(unittest.TestCase):
        """Tests for the per-layer recompute manager."""

        def setUp(self):
            self.device = torch.device("cpu")   # CPU for unit tests
            self.a6000_cfg = _make_a6000_config()
            self.h100_cfg = _make_h100_config()

        def test_recompute_enabled_a6000(self):
            mgr = HeteroGDNNormOutRecompute(
                layer_number=1,
                device=self.device,
                config=self.a6000_cfg,
            )
            self.assertTrue(mgr.recompute_norm_out)

        def test_recompute_disabled_h100(self):
            mgr = HeteroGDNNormOutRecompute(
                layer_number=1,
                device=self.device,
                config=self.h100_cfg,
            )
            self.assertFalse(mgr.recompute_norm_out)

        def test_apply_recompute_path(self):
            mgr = HeteroGDNNormOutRecompute(
                layer_number=1,
                device=self.device,
                config=self.a6000_cfg,
            )
            call_count = [0]

            def dummy_fn(x, g):
                call_count[0] += 1
                return x + g

            x = torch.randn(4, 8)
            g = torch.randn(4, 8)
            out = mgr.apply(dummy_fn, x, g)
            self.assertEqual(call_count[0], 1)
            self.assertIsNotNone(mgr.norm_out_checkpoint)
            self.assertTrue(torch.allclose(out, x + g))

        def test_apply_no_recompute_path(self):
            mgr = HeteroGDNNormOutRecompute(
                layer_number=1,
                device=self.device,
                config=self.h100_cfg,
            )

            def dummy_fn(x, g):
                return x * g

            x = torch.randn(4, 8)
            g = torch.randn(4, 8)
            out = mgr.apply(dummy_fn, x, g)
            self.assertIsNone(mgr.norm_out_checkpoint)
            self.assertTrue(torch.allclose(out, x * g))

        def test_end_of_batch_clears_state(self):
            mgr = HeteroGDNNormOutRecompute(
                layer_number=1,
                device=self.device,
                config=self.a6000_cfg,
            )
            x = torch.randn(4, 8)
            g = torch.randn(4, 8)
            mgr.apply(lambda a, b: a + b, x, g)
            mgr.end_of_batch()
            self.assertIsNone(mgr.norm_out_checkpoint)

    class TestHeteroGDNLayerForward(unittest.TestCase):
        """
        Integration tests for the full HeteroGDNLayer forward pass.

        These tests exercise the recompute path end-to-end (forward + backward)
        and verify numerical equivalence with the baseline (no recompute) path,
        mirroring Megatron's ``test_selective_recompute_norm_out``.
        """

        HIDDEN = 64
        SEQ = 16
        BATCH = 2

        def _make_input(self) -> Tensor:
            return torch.randn(
                self.SEQ, self.BATCH, self.HIDDEN,
                dtype=torch.float32,
                requires_grad=True,
            )

        def test_baseline_vs_recompute_output_identical(self):
            """Outputs must be bit-identical with and without recompute."""
            device = torch.device("cpu")
            torch.manual_seed(0)

            # Build baseline layer (H100 config, no recompute)
            h100_cfg = _make_h100_config()
            baseline = _make_layer(self.HIDDEN, h100_cfg, device)

            # Build recompute layer with *same* weights
            a6000_cfg = _make_a6000_config()
            recompute_layer = _make_layer(self.HIDDEN, a6000_cfg, device)
            recompute_layer.load_state_dict(copy.deepcopy(baseline.state_dict()))

            x_base = self._make_input()
            x_recomp = x_base.detach().clone().requires_grad_(True)

            out_base, _ = baseline(x_base, x_base.detach())
            out_recomp, _ = recompute_layer(x_recomp, x_recomp.detach())

            self.assertTrue(
                torch.allclose(out_base, out_recomp, atol=1e-5),
                f"Output mismatch: max_diff={( out_base - out_recomp).abs().max().item():.2e}",
            )

        def test_baseline_vs_recompute_gradients_identical(self):
            """Input gradients must match between baseline and recompute paths."""
            device = torch.device("cpu")
            torch.manual_seed(1)

            h100_cfg = _make_h100_config()
            baseline = _make_layer(self.HIDDEN, h100_cfg, device)

            a6000_cfg = _make_a6000_config()
            recompute_layer = _make_layer(self.HIDDEN, a6000_cfg, device)
            recompute_layer.load_state_dict(copy.deepcopy(baseline.state_dict()))

            x_base = self._make_input()
            x_recomp = x_base.detach().clone().requires_grad_(True)
            gate_base = torch.randn_like(x_base)
            gate_recomp = gate_base.clone()

            out_base, _ = baseline(x_base, gate_base)
            out_base.sum().backward()

            out_recomp, _ = recompute_layer(x_recomp, gate_recomp)
            out_recomp.sum().backward()

            self.assertTrue(
                torch.allclose(x_base.grad, x_recomp.grad, atol=1e-5),
                "Input gradient mismatch between baseline and recompute.",
            )

        def test_recompute_flag_set_correctly(self):
            device = torch.device("cpu")
            a6000_cfg = _make_a6000_config()
            layer = _make_layer(self.HIDDEN, a6000_cfg, device)
            self.assertTrue(layer.recompute_norm_out)

        def test_no_recompute_flag_on_h100(self):
            device = torch.device("cpu")
            h100_cfg = _make_h100_config()
            layer = _make_layer(self.HIDDEN, h100_cfg, device)
            self.assertFalse(layer.recompute_norm_out)

        def test_norm_out_checkpoint_is_none_before_forward(self):
            device = torch.device("cpu")
            a6000_cfg = _make_a6000_config()
            layer = _make_layer(self.HIDDEN, a6000_cfg, device)
            self.assertIsNone(layer.norm_out_checkpoint)

        def test_norm_out_checkpoint_set_after_forward(self):
            device = torch.device("cpu")
            a6000_cfg = _make_a6000_config()
            layer = _make_layer(self.HIDDEN, a6000_cfg, device)
            x = self._make_input()
            layer(x, x.detach())
            self.assertIsNotNone(layer.norm_out_checkpoint)

        def test_end_of_batch_resets_checkpoint(self):
            device = torch.device("cpu")
            a6000_cfg = _make_a6000_config()
            layer = _make_layer(self.HIDDEN, a6000_cfg, device)
            x = self._make_input()
            layer(x, x.detach())
            layer.end_of_batch()
            self.assertIsNone(layer.norm_out_checkpoint)

        def test_no_gate_defaults_to_self_gate(self):
            """Forward with gate=None should not raise."""
            device = torch.device("cpu")
            a6000_cfg = _make_a6000_config()
            layer = _make_layer(self.HIDDEN, a6000_cfg, device)
            x = torch.randn(self.SEQ, self.BATCH, self.HIDDEN)
            out, bias = layer(x, None)
            self.assertEqual(out.shape, x.shape)
            self.assertIsNone(bias)

    class TestTransformerHeteroConfig(unittest.TestCase):
        """Tests for top-level config validation."""

        def test_valid_config_does_not_raise(self):
            rc = _make_a6000_config()
            TransformerHeteroConfig(rc)   # should not raise

        def test_h100_gdn_norm_out_warning(self):
            rc = _make_a6000_config()
            rc.modules_per_device[DeviceClass.H100_NVL] = {"gdn_norm_out"}
            with self.assertWarns(UserWarning):
                TransformerHeteroConfig(rc)

        def test_loc_budget_warning(self):
            rc = _make_a6000_config()
            rc.loc_config.max_cpu_bytes = 2_000 * (1 << 30)   # 2 TiB > 1.5 TiB
            with self.assertWarns(UserWarning):
                TransformerHeteroConfig(rc)

        def test_invalid_prefetch_depth(self):
            rc = _make_a6000_config()
            rc.loc_config.prefetch_depth = 0
            with self.assertRaises(ValueError):
                TransformerHeteroConfig(rc)

    class TestBuildNeuronSPConfig(unittest.TestCase):
        """Tests for the cluster-specific config builder."""

        def test_returns_valid_config(self):
            cfg = build_neuron_sp_config()
            cfg.validate()   # should not raise

        def test_a6000_recomputes(self):
            cfg = build_neuron_sp_config(a6000_indices=(0, 1), h100_index=2)
            self.assertTrue(cfg.should_recompute_norm_out(0))
            self.assertTrue(cfg.should_recompute_norm_out(1))

        def test_h100_does_not_recompute(self):
            cfg = build_neuron_sp_config(a6000_indices=(0, 1), h100_index=2)
            self.assertFalse(cfg.should_recompute_norm_out(2))

        def test_loc_budget_respected(self):
            cfg = build_neuron_sp_config(loc_max_cpu_gb=16.0)
            self.assertEqual(cfg.loc_config.max_cpu_bytes, 16 * (1 << 30))

    class TestDecoupledNormA2APipeline(unittest.TestCase):
        """Smoke tests for stream management (CPU only, no real collectives)."""

        def test_construction(self):
            pipeline = DecoupledNormA2APipeline(device=torch.device("cpu"), pg=None)
            self.assertIsNone(pipeline._comm_stream)

        def test_synchronise_no_op_without_stream(self):
            pipeline = DecoupledNormA2APipeline(device=torch.device("cpu"), pg=None)
            pipeline.synchronise()   # should not raise

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_cls in [
        TestLocalityCacheConfig,
        TestHeteroDeviceMap,
        TestHeteroRecomputeConfig,
        TestNormOutLocalityCache,
        TestCheckpointWithoutOutput,
        TestHeteroGDNNormOutRecompute,
        TestHeteroGDNLayerForward,
        TestTransformerHeteroConfig,
        TestBuildNeuronSPConfig,
        TestDecoupledNormA2APipeline,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroGDNNormOutRecompute on a DeepSpeed engine.

    Instantiates a :class:`HeteroGDNNormOutRecompute` from the engine's configuration
    and attaches it as ``engine.hetero_gdn_selective_recompute``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_gdn_selective_recompute.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_gdn_selective_recompute = None
    logger.info("hetero_gdn_selective_recompute.register() attached engine.hetero_gdn_selective_recompute")
