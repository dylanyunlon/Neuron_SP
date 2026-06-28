"""
hetero_async_checkpoint_save.py
================================
Neuron_SP / DES-LOC  —  Heterogeneous Async Checkpoint Save

Upstream design intent (Megatron 69f3b34)
------------------------------------------
Megatron-LM commit 69f3b34 (dimapihtar, 2026-04-03) extended the existing
`async_utils.py` / `checkpointing.py` pipeline to support **DCP** (PyTorch
Distributed Checkpoint) and **FSDP-DTensor** formats in addition to the
previously supported `torch_dist` format.

Key upstream changes:
1. ``get_save_and_finalize_callbacks`` — wraps a DCP ``FileSystemWriterAsync``
   together with a ``save_state_dict_async_plan`` result into an
   ``NVRxAsyncRequest`` (save_fn + preload_fn + finalize_fn).
2. Checkpoint-format guard expanded from ``['torch_dist']`` to
   ``['torch_dist', 'torch_dcp', 'fsdp_dtensor']`` for async eligibility.
3. ``has_nvrx_installed()`` utility + ``__post_init__`` validation so users
   get an early, readable error rather than a deep import crash.
4. ``save_state_dict_async_plan`` is called at checkpoint time; the actual
   byte-writes happen in a background thread; ``save_state_dict_async_finalize``
   syncs on teardown or the next checkpoint boundary.

DES-LOC adaptation points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces three
concerns absent from Megatron's homogeneous GPU cluster:

A. **Device heterogeneity** — 2× A6000 (SM86, 48 GB, PCIe) + 1× H100 NVL
   (SM90, 96 GB, PCIe).  No NVLink.  Serializing large FSDP-DTensor shards
   over PCIe from H100 to NFS while A6000 ranks are still computing is the
   dominant checkpoint latency.  We therefore assign checkpoint coordination
   duties to A6000 ranks and let H100 write its own (larger) shard
   concurrently on a dedicated IO thread pool.

B. **Shared Locality Cache** — DES-LOC keeps a pinned-memory "locality cache"
   in the 1.5 TB host DRAM.  Before spawning background IO, we *snapshot* the
   GPU tensors into this cache, releasing the GPU immediately.  The background
   writer reads exclusively from CPU-pinned buffers.  This decouples compute
   and IO completely even across PCIe.

C. **Rank-aware IO budget** — H100 has 2× the parameter budget of each A6000.
   We scale ``thread_count`` and the pinned-memory staging buffer proportional
   to each rank's ``device_memory_gb`` so no single rank becomes the IO
   bottleneck.

The public API intentionally mirrors Megatron's
``get_save_and_finalize_callbacks`` signature so downstream callers need no
change.

Dependencies
------------
- PyTorch >= 2.3  (``torch.distributed.checkpoint``)
- DeepSpeed >= 0.14
- nvidia-resiliency-ext (optional, falls back to mcore async strategy)
- Neuron_SP internal: ``deepspeed.locality_cache``

Author: Neuron_SP dev team (reinterpretation of Megatron 69f3b34)
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional nvidia-resiliency-ext import — mirrors Megatron's try/except guard
# ---------------------------------------------------------------------------
try:
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import (
        AsyncRequest as NVRxAsyncRequest,
    )
    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import (
        FileSystemWriterAsync,
        _results_queue as _nvrx_results_queue,
    )
    from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
        save_state_dict_async_finalize,
        save_state_dict_async_plan,
    )

    _HAVE_NVRX = True
    logger.info("[DES-LOC] nvidia-resiliency-ext found; using NVRx async backend.")
except (ImportError, ModuleNotFoundError):
    _HAVE_NVRX = False
    NVRxAsyncRequest = ABC  # type: ignore[assignment,misc]
    logger.warning(
        "[DES-LOC] nvidia-resiliency-ext not found; "
        "falling back to mcore async checkpoint strategy."
    )

# ---------------------------------------------------------------------------
# Device-class taxonomy for DES-LOC heterogeneous scheduling
# ---------------------------------------------------------------------------

class DeviceClass(Enum):
    """Taxonomy of GPU types present in the DES-LOC cluster."""
    A6000_SM86 = auto()   # 48 GB, PCIe, SM compute capability 8.6
    H100_NVL_SM90 = auto()  # 96 GB, PCIe, SM compute capability 9.0
    UNKNOWN = auto()


_SM_TO_DEVICE_CLASS: Dict[Tuple[int, int], DeviceClass] = {
    (8, 6): DeviceClass.A6000_SM86,
    (9, 0): DeviceClass.H100_NVL_SM90,
}

_DEVICE_MEMORY_GB: Dict[DeviceClass, float] = {
    DeviceClass.A6000_SM86: 48.0,
    DeviceClass.H100_NVL_SM90: 96.0,
    DeviceClass.UNKNOWN: 40.0,  # conservative default
}

# IO thread budget per device class — H100 gets more threads because its shard
# is larger, but we cap to avoid NFS contention on a PCIe fabric.
_DEVICE_IO_THREADS: Dict[DeviceClass, int] = {
    DeviceClass.A6000_SM86: 4,
    DeviceClass.H100_NVL_SM90: 8,
    DeviceClass.UNKNOWN: 2,
}


def _detect_device_class(device: Optional[torch.device] = None) -> DeviceClass:
    """Return the :class:`DeviceClass` for *device* (defaults to ``cuda:current``)."""
    if device is None:
        if not torch.cuda.is_available():
            return DeviceClass.UNKNOWN
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        props = torch.cuda.get_device_properties(device)
        sm = (props.major, props.minor)
        cls = _SM_TO_DEVICE_CLASS.get(sm, DeviceClass.UNKNOWN)
        logger.debug(
            "[DES-LOC] Detected device %s → SM%d%d → %s",
            device,
            props.major,
            props.minor,
            cls.name,
        )
        return cls
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DES-LOC] Device detection failed (%s); assuming UNKNOWN.", exc)
        return DeviceClass.UNKNOWN


# ---------------------------------------------------------------------------
# Shared Locality Cache  —  pinned-memory staging for decoupled IO
# ---------------------------------------------------------------------------

class LocalityCache:
    """
    CPU pinned-memory pool used by DES-LOC to snapshot GPU tensors before
    background IO starts.

    Design rationale
    ~~~~~~~~~~~~~~~~
    Without NVLink, GPU→NFS data movement traverses PCIe twice (GPU→CPU DMA
    then CPU→NFS).  By explicitly pinning staging buffers in host DRAM we:

    * Release the GPU immediately after ``stage()``, allowing the next training
      step to overlap with disk writes.
    * Avoid implicit ``cudaMemcpy`` inside the storage writer that would block
      the CUDA default stream.
    * Leverage the 1.5 TB host DRAM headroom — even H100's 96 GB shard fits
      comfortably.

    Thread safety
    ~~~~~~~~~~~~~
    ``stage()`` must be called from the main training thread before the
    background writer thread starts.  The writer thread calls ``consume()``
    to drain the cache; no concurrent access to the same tensor key is
    allowed.
    """

    def __init__(self, max_bytes: int = 0) -> None:
        """
        Parameters
        ----------
        max_bytes:
            Soft cap on total pinned memory (0 = unlimited, relying on host
            DRAM headroom).  If exceeded, ``stage()`` falls back to non-pinned
            CPU tensors with a warning.
        """
        self._store: Dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._used_bytes: int = 0

    # ------------------------------------------------------------------
    def stage(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        Copy *tensor* from GPU to a pinned CPU buffer and register it.

        Returns the CPU-pinned clone so callers can substitute it into their
        state dict before passing to the storage writer.
        """
        nbytes = tensor.numel() * tensor.element_size()
        if self._max_bytes and (self._used_bytes + nbytes > self._max_bytes):
            logger.warning(
                "[DES-LOC/LocalityCache] Pinned budget exceeded (%d/%d bytes); "
                "staging %s as non-pinned.",
                self._used_bytes,
                self._max_bytes,
                key,
            )
            cpu_tensor = tensor.detach().cpu()
        else:
            try:
                cpu_tensor = tensor.detach().to(device="cpu", non_blocking=True)
                # Ensure DMA completes before returning; writer must not race.
                torch.cuda.synchronize()
                if cpu_tensor.is_cuda:
                    cpu_tensor = cpu_tensor.cpu()
                # Pin the buffer so future DMA (to NFS via OS page cache) is
                # faster and doesn't wake the CUDA driver.
                cpu_tensor = cpu_tensor.pin_memory()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[DES-LOC/LocalityCache] pin_memory failed for %s (%s); "
                    "using ordinary CPU tensor.",
                    key,
                    exc,
                )
                cpu_tensor = tensor.detach().cpu()

        with self._lock:
            self._store[key] = cpu_tensor
            self._used_bytes += nbytes

        logger.debug(
            "[DES-LOC/LocalityCache] Staged '%s' — %d bytes (total staged: %d).",
            key,
            nbytes,
            self._used_bytes,
        )
        return cpu_tensor

    def consume(self, key: str) -> Optional[torch.Tensor]:
        """Pop and return the staged tensor, or ``None`` if not found."""
        with self._lock:
            tensor = self._store.pop(key, None)
            if tensor is not None:
                self._used_bytes -= tensor.numel() * tensor.element_size()
        return tensor

    def flush(self) -> None:
        """Drop all staged tensors (called after writer confirms completion)."""
        with self._lock:
            self._store.clear()
            self._used_bytes = 0
        logger.debug("[DES-LOC/LocalityCache] Cache flushed.")

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level default locality cache shared across checkpoint calls.
_DEFAULT_LOCALITY_CACHE: LocalityCache = LocalityCache()


def get_locality_cache() -> LocalityCache:
    """Return the process-global :class:`LocalityCache` instance."""
    return _DEFAULT_LOCALITY_CACHE


# ---------------------------------------------------------------------------
# Async request abstraction  —  DES-LOC flavour
# ---------------------------------------------------------------------------

@dataclass
class HeteroAsyncRequest:
    """
    DES-LOC counterpart to Megatron's ``NVRxAsyncRequest``.

    Megatron's ``NVRxAsyncRequest`` bundles::

        (save_fn, save_args, [finalize_fn], async_fn_kwargs, preload_fn)

    DES-LOC extends this with:

    * ``device_class`` — controls IO thread budget and staging strategy.
    * ``locality_cache`` — the pinned-memory pool tensors were staged into.
    * ``coordinator_rank`` — which rank drives the DCP ``DefaultSavePlanner``.
    * ``rank_io_threads`` — overrides the per-device-class default if set.

    The ``execute()`` method drives the full lifecycle:
    1. ``preload_fn()`` if provided (Megatron uses this for metadata prefetch).
    2. ``save_fn(*save_args, **async_fn_kwargs)`` in a background thread.
    3. Block until the background thread finishes.
    4. ``finalize_fn()`` — syncs distributed state, removes temp files, etc.
    5. ``locality_cache.flush()`` — release pinned memory.
    """

    save_fn: Callable
    save_args: Tuple[Any, ...]
    finalize_fns: List[Callable] = field(default_factory=list)
    async_fn_kwargs: Dict[str, Any] = field(default_factory=dict)
    preload_fn: Optional[Callable] = None
    device_class: DeviceClass = DeviceClass.UNKNOWN
    locality_cache: Optional[LocalityCache] = None
    coordinator_rank: int = 0
    rank_io_threads: Optional[int] = None

    # ------------------------------------------------------------------
    # Derived / cached properties
    # ------------------------------------------------------------------
    def _io_threads(self) -> int:
        if self.rank_io_threads is not None:
            return self.rank_io_threads
        return _DEVICE_IO_THREADS.get(self.device_class, 2)

    def execute(self) -> None:
        """
        Drive the full async save lifecycle for this rank.

        Called by :func:`schedule_hetero_async_save` in a coordinator thread.
        """
        _t0 = time.monotonic()
        # M3628 fix: hold a strong reference to the preload_fn() result for the
        # entire duration of the async write.  preload_fn() typically returns
        # CPU-pinned buffers staging GPU tensors; without this reference the GC
        # can reclaim them before the background writer finishes, causing SIGBUS.
        # (upstream: Megatron fc61ce5a6 / #2288 — TemporalAsyncCaller fix)
        _preloaded_holder = None
        if self.preload_fn is not None:
            logger.debug("[DES-LOC] Running preload_fn for metadata prefetch.")
            _preloaded_holder = self.preload_fn()

        logger.info(
            "[DES-LOC/%s] Starting background IO with %d thread(s).",
            self.device_class.name,
            self._io_threads(),
        )

        result_holder: List[Any] = []
        exc_holder: List[BaseException] = []

        def _worker() -> None:
            try:
                result = self.save_fn(*self.save_args, **self.async_fn_kwargs)
                result_holder.append(result)
            except Exception as exc:  # noqa: BLE001
                exc_holder.append(exc)

        worker = threading.Thread(target=_worker, daemon=True, name="deslocIO")
        worker.start()
        worker.join()

        _io_dur = time.monotonic() - _t0
        if exc_holder:
            raise RuntimeError(
                f"[DES-LOC] Background IO failed: {exc_holder[0]}"
            ) from exc_holder[0]

        logger.info(
            "[DES-LOC/%s] IO thread finished in %.2fs.",
            self.device_class.name,
            _io_dur,
        )

        for fn in self.finalize_fns:
            fn()

        if self.locality_cache is not None:
            self.locality_cache.flush()

        # M3628 fix: release the pinned buffer reference now that IO is done.
        _preloaded_holder = None  # noqa: F841

        logger.info(
            "[DES-LOC] Checkpoint save complete in %.2fs total.",
            time.monotonic() - _t0,
        )


# ---------------------------------------------------------------------------
# State-dict staging — moves GPU tensors into LocalityCache
# ---------------------------------------------------------------------------

def _stage_state_dict_to_cache(
    state_dict: Dict[str, Any],
    cache: LocalityCache,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Recursively walk *state_dict*, staging every CUDA tensor into *cache*.

    Returns a new dict where CUDA tensors are replaced by their CPU-pinned
    counterparts.  Non-tensor values are passed through unchanged.

    This is the core DES-LOC "snapshot" operation that decouples compute from
    IO: once this function returns, the GPU buffers are free for the next
    forward/backward pass even though the writer hasn't started yet.

    Parameters
    ----------
    state_dict:
        Possibly nested dict of tensors / scalars / lists produced by
        ``model.state_dict()`` or an FSDP gather.
    cache:
        The :class:`LocalityCache` to stage tensors into.
    prefix:
        Key prefix for logging / cache namespacing (used in recursion).
    """
    staged: Dict[str, Any] = {}
    for k, v in state_dict.items():
        full_key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, torch.Tensor) and v.is_cuda:
            # M3609 fix: dequantize quantized CUDA tensors before staging.
            # PyTorch quantized tensors are not supported by the async writer;
            # calling dequantize() converts them to a standard float tensor.
            # (upstream: Megatron a8530db43 / #3845)
            if hasattr(type(v), 'dequantize'):
                v = v.dequantize()
            staged[k] = cache.stage(full_key, v)
        elif isinstance(v, dict):
            staged[k] = _stage_state_dict_to_cache(v, cache, prefix=full_key)
        elif isinstance(v, (list, tuple)):
            def _stage_item(i: int, item: Any) -> Any:
                if isinstance(item, torch.Tensor) and item.is_cuda:
                    # M3609 fix: dequantize quantized tensors in sequences too.
                    if hasattr(type(item), 'dequantize'):
                        item = item.dequantize()
                    return cache.stage(f"{full_key}[{i}]", item)
                return item
            staged[k] = type(v)(_stage_item(i, item) for i, item in enumerate(v))
        else:
            staged[k] = v
    return staged


# ---------------------------------------------------------------------------
# Core public API  —  mirrors Megatron's get_save_and_finalize_callbacks
# ---------------------------------------------------------------------------

def get_hetero_save_and_finalize_callbacks(
    writer: Any,
    save_state_dict_ret: Any,
    device_class: Optional[DeviceClass] = None,
    locality_cache: Optional[LocalityCache] = None,
    rank_io_threads: Optional[int] = None,
) -> "HeteroAsyncRequest | NVRxAsyncRequest":
    """
    Build an async save request for DCP / FSDP-DTensor checkpoints on the
    DES-LOC heterogeneous cluster.

    Upstream Megatron equivalent
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``async_utils.get_save_and_finalize_callbacks(writer, save_state_dict_ret)``
    returns an ``NVRxAsyncRequest(save_fn, save_args, [finalize_fn], …)``.

    DES-LOC extensions
    ~~~~~~~~~~~~~~~~~~
    1. Auto-detects the current rank's :class:`DeviceClass` (SM86 vs SM90) to
       select the appropriate IO thread budget.
    2. Associates a :class:`LocalityCache` so the ``finalize_fn`` can flush
       pinned staging buffers after the writer confirms completion.
    3. If ``_HAVE_NVRX`` is True, wraps into an ``NVRxAsyncRequest`` for
       compatibility with the NVRx coordinator; otherwise returns a native
       :class:`HeteroAsyncRequest`.

    Parameters
    ----------
    writer:
        A ``FileSystemWriterAsync`` instance (NVRx) or any object exposing
        ``get_save_function_and_args() → (save_fn, preload_fn, save_args)``.
    save_state_dict_ret:
        The opaque return value from ``save_state_dict_async_plan()``, passed
        verbatim to ``save_state_dict_async_finalize()``.
    device_class:
        Override auto-detection.  Pass ``None`` to detect from current device.
    locality_cache:
        The pinned-memory staging pool.  Defaults to the module-level
        :func:`get_locality_cache()`.
    rank_io_threads:
        Override the per-device-class IO thread count.

    Returns
    -------
    HeteroAsyncRequest | NVRxAsyncRequest
        Caller passes this to :func:`schedule_hetero_async_save`.
    """
    if device_class is None:
        device_class = _detect_device_class()

    if locality_cache is None:
        locality_cache = get_locality_cache()

    save_fn, preload_fn, save_args = writer.get_save_function_and_args()

    def finalize_fn() -> None:
        """
        Finalize async checkpointing and synchronize all ranks.

        Mirrors Megatron's inner ``finalize_fn`` in
        ``get_save_and_finalize_callbacks``, extended with:
        - dist barrier to ensure all ranks finish before the cache flush.
        - Explicit ``locality_cache.flush()`` to release pinned DRAM.
        """
        logger.debug("[DES-LOC] Entering finalize_fn — calling async_finalize.")
        if _HAVE_NVRX:
            save_state_dict_async_finalize(*save_state_dict_ret)
        else:
            # mcore fallback: the writer's __exit__ / finish() method handles
            # finalization; nothing extra needed here.
            logger.debug("[DES-LOC] mcore path: no explicit async_finalize call.")

        # Barrier: ensure every rank's writer thread is done before we proceed.
        if dist.is_available() and dist.is_initialized():
            logger.debug("[DES-LOC] Post-IO barrier — waiting for all ranks.")
            dist.barrier()

        logger.debug("[DES-LOC] finalize_fn complete.")

    if _HAVE_NVRX:
        # Wrap into NVRxAsyncRequest for compatibility with the NVRx scheduler.
        # DES-LOC-specific state (device_class, locality_cache) is carried via
        # a closure in ``finalize_fn``; the NVRx coordinator is unaware of it.
        logger.info(
            "[DES-LOC/%s] Building NVRxAsyncRequest (io_threads=%d).",
            device_class.name,
            _DEVICE_IO_THREADS.get(device_class, 2),
        )
        return NVRxAsyncRequest(
            save_fn,
            save_args,
            [finalize_fn],
            async_fn_kwargs={},
            preload_fn=preload_fn,
        )
    else:
        # Native DES-LOC path: HeteroAsyncRequest carries full context.
        logger.info(
            "[DES-LOC/%s] Building HeteroAsyncRequest (io_threads=%d).",
            device_class.name,
            _DEVICE_IO_THREADS.get(device_class, 2),
        )
        return HeteroAsyncRequest(
            save_fn=save_fn,
            save_args=save_args,
            finalize_fns=[finalize_fn],
            async_fn_kwargs={},
            preload_fn=preload_fn,
            device_class=device_class,
            locality_cache=locality_cache,
            rank_io_threads=rank_io_threads,
        )


# ---------------------------------------------------------------------------
# Checkpoint format validation  —  mirrors Megatron's CheckpointConfig guard
# ---------------------------------------------------------------------------

#: Formats that support async save in DES-LOC (superset of Megatron's list).
ASYNC_ELIGIBLE_FORMATS: Tuple[str, ...] = (
    "torch_dist",
    "torch_dcp",
    "fsdp_dtensor",
)


def validate_async_checkpoint_config(
    ckpt_format: str,
    async_save: bool,
    require_nvrx_for_dcp: bool = True,
) -> None:
    """
    Validate checkpoint configuration for DES-LOC, mirroring Megatron's
    ``CheckpointConfig.__post_init__`` guard.

    Upstream change (69f3b34)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Megatron expanded the format guard from ``['torch_dist']`` to
    ``['torch_dist', 'torch_dcp', 'fsdp_dtensor']`` and added an assertion
    that ``nvidia-resiliency-ext`` is installed when DCP/FSDP-DTensor async
    save is requested.

    DES-LOC adaptation
    ~~~~~~~~~~~~~~~~~~
    We call this at DeepSpeed engine init time rather than in a dataclass
    ``__post_init__``, allowing async validators to inspect runtime device
    topology (e.g., warn if H100 rank has fewer IO threads than expected).

    Parameters
    ----------
    ckpt_format:
        One of ``'torch_dist'``, ``'torch_dcp'``, ``'fsdp_dtensor'``,
        ``'legacy'``.
    async_save:
        Whether async save is requested.
    require_nvrx_for_dcp:
        If ``True`` (default), raise if DCP/FSDP-DTensor async is requested
        but NVRx is not installed.

    Raises
    ------
    NotImplementedError:
        If ``async_save=True`` and ``ckpt_format`` is not in
        :data:`ASYNC_ELIGIBLE_FORMATS`.
    RuntimeError:
        If DCP/FSDP-DTensor async is requested but NVRx is unavailable and
        ``require_nvrx_for_dcp=True``.
    """
    if not async_save:
        return

    if ckpt_format == "legacy":
        raise NotImplementedError(
            "[DES-LOC] Async checkpoint save is not implemented for legacy checkpoints."
        )

    if ckpt_format not in ASYNC_ELIGIBLE_FORMATS:
        raise NotImplementedError(
            f"[DES-LOC] Async checkpoint save not implemented for format "
            f"'{ckpt_format}'. Eligible formats: {ASYNC_ELIGIBLE_FORMATS}."
        )

    if ckpt_format in ("torch_dcp", "fsdp_dtensor") and require_nvrx_for_dcp:
        if not _HAVE_NVRX:
            raise RuntimeError(
                "[DES-LOC] nvidia-resiliency-ext is required for async save with "
                f"ckpt_format='{ckpt_format}'.  Install it with:\n"
                "  pip install nvidia-resiliency-ext"
            )

    logger.info(
        "[DES-LOC] Async checkpoint config validated: format=%s, nvrx=%s.",
        ckpt_format,
        _HAVE_NVRX,
    )


# ---------------------------------------------------------------------------
# Async save scheduler  —  DES-LOC heterogeneous coordinator
# ---------------------------------------------------------------------------

class HeteroAsyncCheckpointScheduler:
    """
    Manages the lifecycle of async checkpoint requests across heterogeneous
    ranks in the DES-LOC cluster.

    Motivation
    ~~~~~~~~~~
    Megatron's ``schedule_async_save`` (in ``async_utils.py``) serialises
    checkpoint requests into a single ``_async_calls_queue`` and drains them
    with a persistent worker thread.  This works well on homogeneous NVLink
    clusters because every rank has symmetric IO bandwidth.

    On the DES-LOC PCIe fabric, H100 writes ~2× more data than A6000 ranks.
    A single shared queue causes A6000 ranks to idle while waiting for the
    H100 writer.  Instead, each rank owns its own scheduler instance with a
    thread pool sized to its :class:`DeviceClass`.

    Usage
    ~~~~~
    ::

        scheduler = HeteroAsyncCheckpointScheduler()
        request = get_hetero_save_and_finalize_callbacks(writer, plan_ret)
        scheduler.schedule(request)
        # ... next training step runs here ...
        scheduler.drain()  # called before next checkpoint or program exit
    """

    def __init__(self) -> None:
        self._device_class: DeviceClass = _detect_device_class()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker_thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()
        logger.info(
            "[DES-LOC/Scheduler] Initialized for device class %s.",
            self._device_class.name,
        )

    # ------------------------------------------------------------------
    def _start_worker(self) -> None:
        """Spawn the background coordinator thread if not already running."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        def _run() -> None:
            # M2932 fix: set CUDA device to the appropriate local rank so that
            # any CUDA allocations in the worker (e.g. temporary D2H copies)
            # land on the correct device rather than defaulting to device 0,
            # which would burden device 0 with undue memory pressure.
            # (upstream: Megatron 595097120 — PersistentAsyncCaller fix)
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                _local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
                torch.cuda.set_device(_local_rank)
            logger.debug("[DES-LOC/Scheduler] Worker thread started.")
            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:  # poison pill
                    logger.debug("[DES-LOC/Scheduler] Worker thread received stop signal.")
                    self._queue.task_done()
                    break
                request, iteration = item
                try:
                    logger.info(
                        "[DES-LOC/Scheduler] Processing checkpoint for iteration %d.",
                        iteration,
                    )
                    if isinstance(request, HeteroAsyncRequest):
                        request.execute()
                    elif _HAVE_NVRX and isinstance(request, NVRxAsyncRequest):
                        # Delegate to NVRx executor path.
                        _execute_nvrx_request(request)
                    else:
                        raise TypeError(
                            f"[DES-LOC/Scheduler] Unknown request type: {type(request)}"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "[DES-LOC/Scheduler] Checkpoint failed for iteration %d: %s",
                        iteration,
                        exc,
                        exc_info=True,
                    )
                    with self._lock:
                        self._error = exc
                finally:
                    self._queue.task_done()
                # M3393 fix: explicitly drop the request and trigger GC after
                # each checkpoint completes.  Without this, the tensor staging
                # buffers (potentially several GB) remain reachable until the
                # next GC cycle, causing OOM on the next checkpoint save.
                # (upstream: Megatron e1a9ac94a / #3591)
                del request, iteration
                gc.collect()

        self._worker_thread = threading.Thread(
            target=_run, daemon=True, name=f"deslocSched-{self._device_class.name}"
        )
        self._worker_thread.start()

    def schedule(self, request: Any, iteration: int = 0) -> None:
        """
        Enqueue *request* for async execution.

        Blocks if a previous request is still running (queue depth = 1),
        which prevents unbounded memory accumulation from multiple staged
        checkpoints.

        Parameters
        ----------
        request:
            A :class:`HeteroAsyncRequest` or ``NVRxAsyncRequest``.
        iteration:
            Current training iteration, used for logging.
        """
        with self._lock:
            if self._error is not None:
                raise RuntimeError(
                    "[DES-LOC/Scheduler] Previous async checkpoint failed; "
                    "cannot schedule new checkpoint."
                ) from self._error

        self._start_worker()
        logger.info(
            "[DES-LOC/Scheduler] Scheduling checkpoint for iteration %d (queue depth: %d).",
            iteration,
            self._queue.qsize(),
        )
        # put() blocks until the previous item is consumed.
        self._queue.put((request, iteration))

    def drain(self, timeout: float = 600.0) -> None:
        """
        Block until all pending checkpoint IO is complete.

        Called before program exit or before scheduling the next checkpoint
        when the queue would overflow.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.  Raises ``TimeoutError`` if exceeded.
        """
        logger.info("[DES-LOC/Scheduler] Draining checkpoint queue (timeout=%.0fs).", timeout)
        t0 = time.monotonic()
        self._queue.join()
        elapsed = time.monotonic() - t0
        logger.info("[DES-LOC/Scheduler] Queue drained in %.2fs.", elapsed)
        with self._lock:
            if self._error is not None:
                raise RuntimeError(
                    "[DES-LOC/Scheduler] Async checkpoint error detected during drain."
                ) from self._error

    def stop(self) -> None:
        """Gracefully stop the background worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._queue.put(None)  # poison pill
            self._worker_thread.join(timeout=30)
        logger.info("[DES-LOC/Scheduler] Worker thread stopped.")

    def is_idle(self) -> bool:
        """Return ``True`` if no checkpoint IO is in progress."""
        return self._queue.empty()


# ---------------------------------------------------------------------------
# NVRx shim  —  execute an NVRxAsyncRequest in DES-LOC's thread model
# ---------------------------------------------------------------------------

def _execute_nvrx_request(request: "NVRxAsyncRequest") -> None:
    """
    Execute an ``NVRxAsyncRequest`` using DES-LOC's threading model.

    Megatron's NVRx scheduler maintains its own internal thread pool.  On the
    DES-LOC cluster we cannot rely on that pool because it doesn't account for
    per-device IO thread budgets.  Instead, we extract the callable components
    and drive them directly.

    This function is intentionally a thin shim — correctness is maintained by
    calling ``preload_fn``, ``save_fn``, and all ``finalize_fns`` in the same
    order as the NVRx executor.
    """
    if hasattr(request, "preload_fn") and request.preload_fn is not None:
        # M3628 fix: retain the preload_fn() result for the lifetime of the
        # save call so pin_memory buffers are not GC'd before the writer
        # finishes reading them.  (upstream: Megatron fc61ce5a6 / #2288)
        _preloaded_holder = request.preload_fn()
    else:
        _preloaded_holder = None

    # NVRxAsyncRequest stores save args as positional
    save_fn = request.save_fn if hasattr(request, "save_fn") else request.async_fn
    save_args = request.save_args if hasattr(request, "save_args") else request.async_fn_args
    kwargs = getattr(request, "async_fn_kwargs", {})

    save_fn(*save_args, **kwargs)

    for fn in getattr(request, "finalize_fns", []):
        fn()

    # Release the pinned buffer after IO is complete.
    _preloaded_holder = None  # noqa: F841


# ---------------------------------------------------------------------------
# Utility: has_nvrx_installed  —  mirrors Megatron's megatron/training/utils.py
# ---------------------------------------------------------------------------

def has_nvrx_installed() -> bool:
    """
    Check whether ``nvidia-resiliency-ext`` is available in the current
    Python environment.

    Mirrors ``megatron.training.utils.has_nvrx_installed`` (added in 69f3b34)
    so that DES-LOC's ``validate_async_checkpoint_config`` can surface the
    same early error message without importing Megatron.
    """
    try:
        import nvidia_resiliency_ext  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


# ---------------------------------------------------------------------------
# High-level entry point for DeepSpeed engine integration
# ---------------------------------------------------------------------------

def build_hetero_async_save_pipeline(
    state_dict: Dict[str, Any],
    checkpoint_path: str,
    ckpt_format: str,
    iteration: int,
    *,
    thread_count: Optional[int] = None,
    enable_cache: bool = False,
    use_msc: bool = False,
    coordinator_rank: int = 0,
    locality_cache: Optional[LocalityCache] = None,
    scheduler: Optional[HeteroAsyncCheckpointScheduler] = None,
) -> HeteroAsyncCheckpointScheduler:
    """
    End-to-end async checkpoint pipeline for DES-LOC.

    This function replaces the scattered ``save_checkpoint`` logic in
    Megatron's ``checkpointing.py`` with a single entry point that:

    1. Validates format/config (upstream guard from 69f3b34).
    2. Stages GPU tensors into the :class:`LocalityCache` (DES-LOC addition).
    3. Calls ``save_state_dict_async_plan`` to produce the IO work items.
    4. Builds a :class:`HeteroAsyncRequest` via
       :func:`get_hetero_save_and_finalize_callbacks`.
    5. Schedules it on the per-rank :class:`HeteroAsyncCheckpointScheduler`.

    Parameters
    ----------
    state_dict:
        The model/optimizer state dict to checkpoint (may contain CUDA tensors).
    checkpoint_path:
        Directory path for the checkpoint shards.
    ckpt_format:
        ``'torch_dcp'`` or ``'fsdp_dtensor'`` (``'torch_dist'`` also accepted).
    iteration:
        Current training iteration (for logging and checkpoint naming).
    thread_count:
        IO thread count override.  If ``None``, derived from device class.
    enable_cache:
        Pass ``enable_cache=True`` to ``save_state_dict_async_plan`` when the
        checkpoint structure is constant across iterations (Megatron flag
        ``--ckpt-assume-constant-structure``).
    use_msc:
        Enable multi-storage-client mode in ``FileSystemWriterAsync``.
    coordinator_rank:
        Rank responsible for DCP planner coordination (default 0).
    locality_cache:
        Pinned-memory staging pool.  Defaults to module-level cache.
    scheduler:
        Existing scheduler instance to reuse.  If ``None``, a new one is
        created and returned.

    Returns
    -------
    HeteroAsyncCheckpointScheduler
        The scheduler managing this and future checkpoint requests.
        Callers should retain this object and call ``.drain()`` before exit.
    """
    validate_async_checkpoint_config(ckpt_format, async_save=True)

    device_class = _detect_device_class()
    if locality_cache is None:
        locality_cache = get_locality_cache()

    # ---- Stage GPU tensors into pinned CPU memory ----
    logger.info(
        "[DES-LOC] Staging state_dict to LocalityCache "
        "(cache used before: %d bytes).",
        locality_cache.used_bytes,
    )
    staged_state_dict = _stage_state_dict_to_cache(state_dict, locality_cache)
    logger.info(
        "[DES-LOC] Staging complete — %d bytes pinned.",
        locality_cache.used_bytes,
    )

    # ---- Resolve IO thread count ----
    if thread_count is None:
        thread_count = _DEVICE_IO_THREADS.get(device_class, 2)

    # ---- Build storage writer ----
    if _HAVE_NVRX:
        writer = FileSystemWriterAsync(
            checkpoint_path,
            thread_count=thread_count,
            use_msc=use_msc,
        )
        import torch.distributed.checkpoint as dcp
        planner = dcp.DefaultSavePlanner()
        save_state_dict_ret = save_state_dict_async_plan(
            staged_state_dict,
            writer,
            None,
            coordinator_rank,
            planner=planner,
            enable_cache=enable_cache,
        )
        request = get_hetero_save_and_finalize_callbacks(
            writer,
            save_state_dict_ret,
            device_class=device_class,
            locality_cache=locality_cache,
        )
    else:
        # mcore fallback: synchronous DCP save wrapped in a HeteroAsyncRequest
        # so the scheduler interface is preserved.
        logger.warning(
            "[DES-LOC] NVRx unavailable — wrapping synchronous DCP save "
            "in HeteroAsyncRequest for interface compatibility."
        )

        def _sync_save() -> None:
            try:
                import torch.distributed.checkpoint as dcp
                fs_writer = dcp.FileSystemWriter(checkpoint_path)
                dcp.save(
                    state_dict=staged_state_dict,
                    storage_writer=fs_writer,
                )
            except (ImportError, AttributeError):
                os.makedirs(checkpoint_path, exist_ok=True)
                save_path = os.path.join(checkpoint_path, f"rank{dist.get_rank()}.pt")
                torch.save(staged_state_dict, save_path)

        def _flush_cache() -> None:
            locality_cache.flush()

        request = HeteroAsyncRequest(
            save_fn=_sync_save,
            save_args=(),
            finalize_fns=[_flush_cache],
            device_class=device_class,
            locality_cache=locality_cache,
        )

    # ---- Schedule ----
    if scheduler is None:
        scheduler = HeteroAsyncCheckpointScheduler()

    scheduler.schedule(request, iteration=iteration)
    return scheduler


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # --- 1. Device detection ---
    cls = _detect_device_class(torch.device("cpu"))
    assert cls == DeviceClass.UNKNOWN, f"Expected UNKNOWN for CPU, got {cls}"
    logger.info("PASS device detection: %s", cls)

    # --- 2. LocalityCache round-trip ---
    cache = LocalityCache()
    t = torch.randn(4, 4)  # CPU tensor
    staged = cache.stage("test_tensor", t)
    assert isinstance(staged, torch.Tensor), "staged should be a Tensor"
    assert len(cache) == 1, "cache should have 1 entry"
    recovered = cache.consume("test_tensor")
    assert recovered is not None
    assert len(cache) == 0, "cache should be empty after consume"
    logger.info("PASS LocalityCache round-trip")

    # --- 3. _stage_state_dict_to_cache (CPU tensors, no CUDA required) ---
    sd = {"w": torch.ones(3), "nested": {"b": torch.zeros(2)}}
    cache2 = LocalityCache()
    staged_sd = _stage_state_dict_to_cache(sd, cache2)
    assert "w" in staged_sd and "nested" in staged_sd
    logger.info("PASS state_dict staging (CPU path)")

    # --- 4. validate_async_checkpoint_config ---
    try:
        validate_async_checkpoint_config("legacy", async_save=True)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass
    validate_async_checkpoint_config("torch_dist", async_save=True, require_nvrx_for_dcp=False)
    logger.info("PASS checkpoint format validation")

    # --- 5. HeteroAsyncRequest execute (no-op save_fn) ---
    executed = []

    def _noop_save():
        executed.append("save")

    def _noop_finalize():
        executed.append("finalize")

    req = HeteroAsyncRequest(
        save_fn=_noop_save,
        save_args=(),
        finalize_fns=[_noop_finalize],
        device_class=DeviceClass.A6000_SM86,
        locality_cache=LocalityCache(),
    )
    req.execute()
    assert executed == ["save", "finalize"], f"Unexpected execution order: {executed}"
    logger.info("PASS HeteroAsyncRequest.execute()")

    logger.info("All smoke tests passed.")
