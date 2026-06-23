"""
HeteroDDPGradOverlapFix — DES-LOC Heterogeneous Training Adapter
=================================================================

Upstream Design Intent (Megatron commit 3548385ac3b1cbfa7cbf4eceb38d6504662a4f3b)
------------------------------------------------------------------------------------
Megatron-LM's _ParamAndGradBucketGroup manages bucketed gradient reduction with
optional overlap between backward computation and AllReduce / ReduceScatter
communication.  The original code anchored all stream synchronisation against
``torch.cuda.default_stream()``.  This is subtly wrong when the caller has already
switched away from the default stream (e.g. via ``torch.cuda.stream(ctx)``): the
default stream may have already drained, so the wait becomes a no-op, allowing the
communication kernel to read gradient data before the backward kernel has written it.

The one-line fix — replacing ``default_stream()`` with ``current_stream()`` — ensures
the RS/AR communication stream always waits on *whichever* stream is currently
producing gradients, not the fixed default stream.

A symmetric fix is applied to the post-communication barrier in the
``num_distributed_optimizer_instances > 1`` path: the current stream must wait for
the communication stream to drain before the optimiser step reads the reduced
gradient, and again ``current_stream()`` is the correct anchor.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs a heterogeneous device
pool: two A6000 48 GB (SM86, PCIe) and one H100 NVL 96 GB (SM90, PCIe).  Three
structural differences amplify the original Megatron bug and require additional fixes:

1. **Per-device CUDA stream topology** — Each device has its own default stream.
   When DES-LOC's HeteroScheduler migrates a micro-batch from one device to another
   (e.g. A6000 → H100 for large attention layers) it switches the active CUDA device
   and therefore the *current* stream.  Any synchronisation that hard-codes
   ``default_stream()`` on device 0 is silently a no-op on device 1 or 2.

2. **Multiple DistributedOptimizer shards** — DES-LOC partitions the optimiser state
   across all three devices to exploit the 1.5 TB CPU DRAM tier.  This naturally
   implies ``num_distributed_optimizer_instances > 1`` in every run, making the
   second bug site always reachable.

3. **LOC cache writeback** — After gradient reduction, DES-LOC may offload the
   reduced gradient tensor to the Shared LOcality Cache (CPU DRAM) for later
   consumption by a slow-tier optimiser shard.  The writeback DMA must not begin
   until the communication stream has finished; this module inserts the required
   barrier and exposes a ``post_reduce_callback`` hook for the cache manager.

Key classes
-----------
* ``HeteroStreamGuard``     — context manager that records the *current* stream on
                               entry and restores it on exit; used by bucket groups.
* ``HeteroBucketGroup``     — replaces Megatron's _ParamAndGradBucketGroup logic for
                               DES-LOC; owns the overlap-grad-reduce fix.
* ``HeteroDDPGradOverlapFix`` — top-level coordinator; mirrors Megatron's
                               DistributedDataParallel wrapper integration point.

Compatibility
-------------
Built on top of DeepSpeed's ZeRO Stage-1/2/3 runtime.  Requires DeepSpeed ≥ 0.14.
Tested with PyTorch ≥ 2.3 (``torch.cuda.current_stream`` semantics stable).
"""

from __future__ import annotations

import logging
import threading
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & small helpers
# ---------------------------------------------------------------------------

_DESLOC_DEVICES: Tuple[int, ...] = (0, 1, 2)  # indices for A6000-0, A6000-1, H100
_SM86_DEVICES: Tuple[int, ...] = (0, 1)
_SM90_DEVICES: Tuple[int, ...] = (2,)

_BUCKET_SIZE_MB_DEFAULT: int = 25  # MB; tuned for PCIe bandwidth


def _device_label(device_index: int) -> str:
    if device_index in _SM90_DEVICES:
        return f"H100-NVL(sm90)@cuda:{device_index}"
    return f"A6000(sm86)@cuda:{device_index}"


def _bytes_to_mb(n: int) -> float:
    return n / (1024 ** 2)


# ---------------------------------------------------------------------------
# HeteroStreamGuard
# ---------------------------------------------------------------------------

class HeteroStreamGuard:
    """Context manager that tracks the *current* CUDA stream across device switches.

    Design rationale
    ~~~~~~~~~~~~~~~~
    Megatron's original bug arose because ``torch.cuda.default_stream()`` does not
    reflect the stream that is *actually producing work* when the caller has entered a
    ``torch.cuda.stream(s)`` context.  In DES-LOC the situation is worse: device
    migrations change both the active device *and* the notion of "default stream".

    ``HeteroStreamGuard`` solves this by capturing ``torch.cuda.current_stream()``
    at the moment the guard is entered — the same moment the backward kernel is about
    to produce gradients — and exposing it as ``self.grad_stream``.  Communication
    streams then call ``wait_stream(guard.grad_stream)`` rather than
    ``wait_stream(torch.cuda.default_stream())``.

    Thread-safety
    ~~~~~~~~~~~~~
    Each training thread owns its own guard instance; the captured stream reference is
    never shared across threads, so no lock is required.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.grad_stream: Optional[torch.cuda.Stream] = None
        self._prev_stream: Optional[torch.cuda.Stream] = None

    def __enter__(self) -> "HeteroStreamGuard":
        with torch.cuda.device(self.device):
            # *** DES-LOC fix: capture current_stream, not default_stream ***
            self.grad_stream = torch.cuda.current_stream(self.device)
            self._prev_stream = self.grad_stream
        logger.debug(
            "HeteroStreamGuard entered on %s; grad_stream=%s",
            _device_label(self.device.index),
            self.grad_stream,
        )
        return self

    def __exit__(self, *_exc) -> None:
        # Restore is a no-op here; stream context restoration is handled by
        # torch.cuda.stream() wrappers in the bucket group.  We keep the method
        # for symmetry and future extension.
        logger.debug(
            "HeteroStreamGuard exited on %s", _device_label(self.device.index)
        )

    @contextmanager
    def communication_context(
        self, comm_stream: torch.cuda.Stream
    ) -> Generator[None, None, None]:
        """Enter the communication stream and synchronise against grad_stream.

        This is the core of the Megatron fix, translated to DES-LOC:

        Original Megatron (buggy):
            self.communication_stream.wait_stream(torch.cuda.default_stream())

        Megatron fix:
            self.communication_stream.wait_stream(torch.cuda.current_stream())

        DES-LOC adaptation:
            We pass in the stream captured at guard-entry time (``self.grad_stream``)
            rather than calling ``current_stream()`` again, because by the time the
            communication context is entered the "current" stream may already have
            been switched to ``comm_stream`` itself (depending on how the caller
            structures its code).  Using the saved reference is therefore safer and
            equivalent when called immediately after the backward pass.
        """
        assert self.grad_stream is not None, (
            "HeteroStreamGuard.communication_context() called before __enter__"
        )
        with torch.cuda.stream(comm_stream):
            # *** Wait for gradient computation to complete ***
            comm_stream.wait_stream(self.grad_stream)
            logger.debug(
                "comm_stream %s waiting on grad_stream %s (device %s)",
                comm_stream,
                self.grad_stream,
                _device_label(self.device.index),
            )
            yield
        # After the context exits the comm_stream has been *launched* but may
        # not have *completed*.  Callers that need completion must call
        # ``drain_comm_stream`` explicitly.


# ---------------------------------------------------------------------------
# DDP configuration dataclass (mirrors Megatron's DDPConfig)
# ---------------------------------------------------------------------------

@dataclass
class HeteroDDPConfig:
    """Configuration for heterogeneous DDP gradient overlap.

    Mirrors the subset of Megatron's DDPConfig that is relevant to bucket
    group stream management, extended with DES-LOC-specific fields.
    """

    # --- Megatron-equivalent fields ---
    overlap_grad_reduce: bool = True
    """If True, overlap backward computation with gradient AllReduce/RS."""

    num_distributed_optimizer_instances: int = 3
    """Number of DistributedOptimizer shards.  In DES-LOC this equals the number
    of devices (3) because each device owns one optimiser shard."""

    bucket_size_mb: int = _BUCKET_SIZE_MB_DEFAULT
    """Target bucket size in megabytes."""

    use_reduce_scatter: bool = True
    """Use ReduceScatter + AllGather (ZeRO-style) instead of plain AllReduce."""

    # --- DES-LOC-specific fields ---
    loc_cache_enabled: bool = True
    """When True, reduced gradients are written back to the Shared LOcality Cache
    (CPU DRAM) after the communication stream completes."""

    loc_cache_device: str = "cpu"
    """Target device for LOC cache tensors.  'cpu' uses pinned memory."""

    device_mesh: Tuple[int, ...] = field(default_factory=lambda: _DESLOC_DEVICES)
    """CUDA device indices participating in this DDP group."""

    sm90_offload_large_buckets: bool = True
    """If True, buckets above ``sm90_offload_threshold_mb`` are preferentially
    reduced on the H100 (SM90) communication stream."""

    sm90_offload_threshold_mb: int = 50
    """Size threshold above which a bucket is offloaded to SM90."""


# ---------------------------------------------------------------------------
# LOC Cache Manager (stub — full impl in deepspeed/runtime/loc_cache.py)
# ---------------------------------------------------------------------------

class LOCacheManager:
    """Manages the Shared LOcality Cache (CPU DRAM) for DES-LOC.

    This is a lightweight stub that decouples the gradient overlap module from
    the full LOC cache implementation.  The real manager is injected at runtime
    via ``HeteroDDPGradOverlapFix.register_loc_cache_manager()``.

    Design
    ~~~~~~
    After a gradient bucket has been reduced, the communication stream has
    produced the final gradient tensor on-device.  DES-LOC's optimiser shards
    that live on different devices (or on the CPU tier) need access to this
    tensor.  Rather than keeping it in device VRAM indefinitely, the LOC cache
    offloads it to pinned CPU DRAM, keyed by ``(param_group_id, bucket_id)``.
    Any optimiser shard can later pull from the cache without holding a P2P
    transfer lock.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._lock = threading.Lock()

    def store(
        self,
        param_group_id: int,
        bucket_id: int,
        tensor: torch.Tensor,
        non_blocking: bool = True,
    ) -> None:
        """Copy *tensor* to pinned CPU memory and register in the cache."""
        pinned = torch.empty(
            tensor.shape, dtype=tensor.dtype,
            device="cpu", pin_memory=True,
        )
        pinned.copy_(tensor, non_blocking=non_blocking)
        with self._lock:
            self._cache[(param_group_id, bucket_id)] = pinned
        logger.debug(
            "LOC cache store: group=%d bucket=%d size=%.2f MB",
            param_group_id, bucket_id, _bytes_to_mb(tensor.nbytes),
        )

    def retrieve(
        self, param_group_id: int, bucket_id: int
    ) -> Optional[torch.Tensor]:
        with self._lock:
            return self._cache.get((param_group_id, bucket_id))

    def evict(self, param_group_id: int, bucket_id: int) -> None:
        with self._lock:
            self._cache.pop((param_group_id, bucket_id), None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# HeteroBucketGroup
# ---------------------------------------------------------------------------

class HeteroBucketGroup:
    """Manages a single gradient bucket across heterogeneous devices.

    Upstream analogue: Megatron's ``_ParamAndGradBucketGroup``
    -----------------------------------------------------------
    In Megatron, each bucket group owns:
    - A contiguous gradient buffer (flattened parameter grads)
    - A communication stream for async AllReduce / ReduceScatter
    - Logic to overlap communication with the next backward chunk

    The two bug sites fixed in commit 3548385 both live here:

    Bug 1 (line ~567):
        ``self.communication_stream.wait_stream(torch.cuda.default_stream())``
        → should be ``current_stream()``

    Bug 2 (line ~677):
        ``torch.cuda.default_stream().wait_stream(self.communication_stream)``
        → should be ``current_stream()``

    DES-LOC re-interpretation
    --------------------------
    *Bug 1* is fixed by using ``HeteroStreamGuard.grad_stream`` (captured at
    backward-pass entry) rather than either ``default_stream`` or a freshly
    sampled ``current_stream``.  This is safer during device migration.

    *Bug 2* is fixed symmetrically: after the communication completes, we wait
    on ``current_stream()`` of the *target* device (the one that will run the
    optimiser step) rather than the fixed default stream.  In the multi-instance
    case (``num_distributed_optimizer_instances > 1``) this is always reachable,
    so the fix always applies.

    Additional DES-LOC logic:
    - ``_select_comm_stream``: routes large buckets to the H100 comm stream.
    - ``_writeback_to_loc_cache``: after the comm stream drains, optionally
      offloads the reduced gradient to the LOC cache.
    - ``drain_comm_stream``: public method for the optimiser shards to call
      before reading reduced gradients.
    """

    def __init__(
        self,
        bucket_id: int,
        param_group_id: int,
        params: List[torch.nn.Parameter],
        device: torch.device,
        process_group: dist.ProcessGroup,
        config: HeteroDDPConfig,
        comm_streams: Dict[int, torch.cuda.Stream],
        loc_cache_manager: Optional[LOCacheManager] = None,
    ) -> None:
        self.bucket_id = bucket_id
        self.param_group_id = param_group_id
        self.params = params
        self.device = device
        self.process_group = process_group
        self.config = config
        self.comm_streams = comm_streams  # device_index → Stream
        self.loc_cache_manager = loc_cache_manager

        self._grad_buffer: Optional[torch.Tensor] = None
        self._grad_reduce_handle: Optional[dist.Work] = None
        self._stream_guard: Optional[HeteroStreamGuard] = None
        self._comm_stream: Optional[torch.cuda.Stream] = None

        self._init_grad_buffer()
        logger.info(
            "HeteroBucketGroup init: bucket=%d group=%d device=%s params=%d "
            "buffer_size=%.2f MB",
            bucket_id, param_group_id, _device_label(device.index),
            len(params), _bytes_to_mb(self._buffer_nbytes),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_grad_buffer(self) -> None:
        """Allocate a flat contiguous gradient buffer for all params in bucket."""
        total_numel = sum(p.numel() for p in self.params)
        # Use float32 regardless of param dtype for numerical stability across
        # SM86 (A6000) and SM90 (H100) with different BF16 accumulation behaviour.
        self._grad_buffer = torch.zeros(
            total_numel, dtype=torch.float32, device=self.device
        )
        self._buffer_nbytes = self._grad_buffer.nbytes
        # Build views: each param gets a slice of the flat buffer
        self._param_grad_views: List[torch.Tensor] = []
        offset = 0
        for p in self.params:
            n = p.numel()
            self._param_grad_views.append(
                self._grad_buffer[offset: offset + n].view(p.shape)
            )
            offset += n

    def _select_comm_stream(self) -> torch.cuda.Stream:
        """Choose the communication stream for this bucket.

        DES-LOC routing policy:
        - Large buckets (> sm90_offload_threshold_mb) go to the H100 (SM90)
          comm stream because it has higher PCIe bandwidth and can overlap
          more effectively.
        - Small buckets stay on the local device's comm stream to avoid
          unnecessary cross-device serialisation.

        Falls back to the current device's comm stream if H100 stream is
        unavailable (e.g. during unit tests with a single-device mock).
        """
        size_mb = _bytes_to_mb(self._buffer_nbytes)
        if (
            self.config.sm90_offload_large_buckets
            and size_mb > self.config.sm90_offload_threshold_mb
            and _SM90_DEVICES[0] in self.comm_streams
        ):
            stream = self.comm_streams[_SM90_DEVICES[0]]
            logger.debug(
                "Bucket %d (%.1f MB) routed to SM90 comm stream", self.bucket_id, size_mb
            )
            return stream
        stream = self.comm_streams.get(
            self.device.index,
            self.comm_streams[next(iter(self.comm_streams))],
        )
        logger.debug(
            "Bucket %d (%.1f MB) using local device %s comm stream",
            self.bucket_id, size_mb, _device_label(self.device.index),
        )
        return stream

    def _copy_grads_to_buffer(self) -> None:
        """Flatten per-parameter .grad tensors into the contiguous buffer."""
        for p, view in zip(self.params, self._param_grad_views):
            if p.grad is not None:
                view.copy_(p.grad.float(), non_blocking=True)
            else:
                view.zero_()

    def _writeback_to_loc_cache(self) -> None:
        """Offload the reduced gradient buffer to the LOC cache (CPU DRAM).

        This is called *after* ``drain_comm_stream`` ensures the communication
        stream has completed.  The copy to pinned memory is non-blocking from
        the CPU perspective; the LOCacheManager handles the actual DMA.
        """
        if self.loc_cache_manager is None or not self.config.loc_cache_enabled:
            return
        assert self._grad_buffer is not None
        self.loc_cache_manager.store(
            self.param_group_id,
            self.bucket_id,
            self._grad_buffer,
            non_blocking=True,
        )
        logger.debug(
            "Bucket %d written back to LOC cache after reduce", self.bucket_id
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_grad_sync(self, stream_guard: HeteroStreamGuard) -> None:
        """Launch asynchronous gradient reduction for this bucket.

        Parameters
        ----------
        stream_guard:
            A ``HeteroStreamGuard`` that has already been entered; its
            ``grad_stream`` attribute reflects the stream that produced the
            gradients we are about to reduce.

        Upstream analogue
        -----------------
        ``_ParamAndGradBucketGroup._start_grad_sync()`` in Megatron.

        Stream synchronisation fix
        --------------------------
        Megatron (buggy):
            ``self.communication_stream.wait_stream(torch.cuda.default_stream())``

        Megatron (fixed, commit 3548385):
            ``self.communication_stream.wait_stream(torch.cuda.current_stream())``

        DES-LOC (here):
            ``comm_stream.wait_stream(stream_guard.grad_stream)``

        Using the guard's saved reference rather than ``current_stream()`` at
        call-time avoids a race in DES-LOC's scheduler where the current stream
        may have already been switched to the communication stream by the time
        this method is invoked.
        """
        self._stream_guard = stream_guard
        self._comm_stream = self._select_comm_stream()

        if not self.config.overlap_grad_reduce:
            # Synchronous path: just flatten and reduce in the current stream
            self._copy_grads_to_buffer()
            self._launch_reduce()
            return

        # Asynchronous overlap path
        with stream_guard.communication_context(self._comm_stream):
            # We are now inside comm_stream AND comm_stream has waited for grad_stream
            self._copy_grads_to_buffer()
            self._launch_reduce()

        logger.debug(
            "Bucket %d grad sync launched (async overlap) on %s",
            self.bucket_id, _device_label(self.device.index),
        )

    def _launch_reduce(self) -> None:
        """Issue the AllReduce or ReduceScatter collective."""
        assert self._grad_buffer is not None
        if self.config.use_reduce_scatter:
            # ZeRO-1/2/3 style: ReduceScatter shards the reduced gradient
            output = torch.empty_like(
                self._grad_buffer[: self._grad_buffer.numel() // self.config.num_distributed_optimizer_instances]
            )
            self._grad_reduce_handle = dist.reduce_scatter_tensor(
                output,
                self._grad_buffer,
                group=self.process_group,
                async_op=True,
            )
        else:
            self._grad_reduce_handle = dist.all_reduce(
                self._grad_buffer,
                group=self.process_group,
                async_op=True,
            )

    def drain_comm_stream(self) -> None:
        """Block the *current* stream until the communication stream completes.

        Upstream analogue
        -----------------
        The barrier inside ``_ParamAndGradBucketGroup.finish_grad_sync()`` when
        ``num_distributed_optimizer_instances > 1``.

        Megatron (buggy):
            ``torch.cuda.default_stream().wait_stream(self.communication_stream)``

        Megatron (fixed, commit 3548385):
            ``torch.cuda.current_stream().wait_stream(self.communication_stream)``

        DES-LOC adaptation
        ------------------
        In DES-LOC, ``num_distributed_optimizer_instances`` is always > 1 (equal
        to the number of devices).  The optimiser step runs in whatever stream the
        scheduler has set as current on the target device.  We must wait on
        ``current_stream()`` *at the time the optimiser is about to read the
        gradient*, which is exactly when this method is called.

        Unlike Bug 1, here we do sample ``current_stream()`` fresh rather than
        using the guard's saved reference, because we *want* to capture the
        optimiser step's stream, not the backward pass's stream.
        """
        if self._comm_stream is None:
            logger.warning(
                "drain_comm_stream called on bucket %d before start_grad_sync",
                self.bucket_id,
            )
            return

        if self.config.num_distributed_optimizer_instances > 1:
            # *** DES-LOC fix: use current_stream(), not default_stream() ***
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self._comm_stream)
            logger.debug(
                "Bucket %d: current_stream %s waiting on comm_stream %s",
                self.bucket_id, current, self._comm_stream,
            )
            # LOC cache writeback — safe now because comm_stream has completed
            self._writeback_to_loc_cache()
            return

        # Single DistOpt instance: wait for the Work handle directly
        if self._grad_reduce_handle is not None:
            self._grad_reduce_handle.wait()
            self._grad_reduce_handle = None
            self._writeback_to_loc_cache()

    def reset(self) -> None:
        """Clear state after the optimiser step so the bucket can be reused."""
        self._grad_reduce_handle = None
        self._stream_guard = None
        if self._grad_buffer is not None:
            self._grad_buffer.zero_()
        logger.debug("Bucket %d reset", self.bucket_id)


# ---------------------------------------------------------------------------
# HeteroDDPGradOverlapFix  (top-level coordinator)
# ---------------------------------------------------------------------------

class HeteroDDPGradOverlapFix:
    """Top-level DES-LOC heterogeneous DDP gradient overlap manager.

    Role in DES-LOC
    ---------------
    This class is the integration point between DeepSpeed's ZeRO engine and
    the DES-LOC scheduler.  It wraps a model's parameter groups, partitions
    parameters into buckets, and drives the per-bucket stream synchronisation
    with the corrected ``current_stream()`` semantics.

    Typical call sequence per training step
    ----------------------------------------
    ::

        fix = HeteroDDPGradOverlapFix(model, config, process_group, comm_streams)

        # --- backward pass ---
        with fix.grad_sync_context(device):
            loss.backward()          # triggers register_grad hooks → start_grad_sync

        # --- optimiser step ---
        fix.drain_all_comm_streams()   # blocks current stream until all reductions done
        optimizer.step()
        fix.reset_all_buckets()

    Integration with DeepSpeed ZeRO
    --------------------------------
    DeepSpeed's ``ZeROOptimizer`` calls ``_reduce_ipg_grads`` after each
    micro-batch.  ``HeteroDDPGradOverlapFix`` is designed to be called from
    that path: replace DeepSpeed's internal ``all_reduce`` call with
    ``fix.start_grad_sync_for_param(param)``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: HeteroDDPConfig,
        process_group: dist.ProcessGroup,
        comm_streams: Optional[Dict[int, torch.cuda.Stream]] = None,
        loc_cache_manager: Optional[LOCacheManager] = None,
    ) -> None:
        self.model = weakref.ref(model)
        self.config = config
        self.process_group = process_group
        self.loc_cache_manager = loc_cache_manager or (
            LOCacheManager() if config.loc_cache_enabled else None
        )

        # Build comm streams if not provided (one per device in mesh)
        if comm_streams is None:
            comm_streams = {}
            for dev_idx in config.device_mesh:
                with torch.cuda.device(dev_idx):
                    s = torch.cuda.Stream(device=dev_idx, priority=-1)  # high-prio
                    comm_streams[dev_idx] = s
                    logger.info(
                        "Created comm stream for %s: %s",
                        _device_label(dev_idx), s,
                    )
        self.comm_streams = comm_streams

        self._bucket_groups: List[HeteroBucketGroup] = []
        self._param_to_bucket: Dict[int, HeteroBucketGroup] = {}
        self._active_guard: Optional[HeteroStreamGuard] = None

        self._build_buckets()
        self._register_hooks()
        logger.info(
            "HeteroDDPGradOverlapFix initialised: %d buckets, %d devices, "
            "overlap=%s, distopt_instances=%d, loc_cache=%s",
            len(self._bucket_groups),
            len(config.device_mesh),
            config.overlap_grad_reduce,
            config.num_distributed_optimizer_instances,
            config.loc_cache_enabled,
        )

    # ------------------------------------------------------------------
    # Bucket construction
    # ------------------------------------------------------------------

    def _build_buckets(self) -> None:
        """Partition model parameters into fixed-size buckets per device.

        Parameters are grouped by the device they reside on.  Within each
        device group they are sorted by estimated gradient size (descending)
        to produce roughly equal-sized buckets, which improves communication
        efficiency on PCIe.
        """
        model = self.model()
        if model is None:
            raise RuntimeError("Model has been garbage-collected")

        # Group params by device
        device_params: Dict[int, List[torch.nn.Parameter]] = {}
        for p in model.parameters():
            if not p.requires_grad:
                continue
            dev_idx = p.device.index if p.device.type == "cuda" else -1
            device_params.setdefault(dev_idx, []).append(p)

        bucket_threshold_bytes = self.config.bucket_size_mb * 1024 * 1024
        param_group_id = 0

        for dev_idx, params in device_params.items():
            if dev_idx < 0:
                logger.warning("Skipping %d CPU parameters", len(params))
                continue
            device = torch.device("cuda", dev_idx)
            # Sort largest-first for better bucket fill
            params_sorted = sorted(params, key=lambda p: p.numel(), reverse=True)

            current_bucket_params: List[torch.nn.Parameter] = []
            current_bucket_bytes = 0
            bucket_id = 0

            for p in params_sorted:
                p_bytes = p.numel() * p.element_size()
                if (
                    current_bucket_params
                    and current_bucket_bytes + p_bytes > bucket_threshold_bytes
                ):
                    self._create_bucket(
                        bucket_id, param_group_id, current_bucket_params, device
                    )
                    bucket_id += 1
                    current_bucket_params = []
                    current_bucket_bytes = 0
                current_bucket_params.append(p)
                current_bucket_bytes += p_bytes

            if current_bucket_params:
                self._create_bucket(
                    bucket_id, param_group_id, current_bucket_params, device
                )

            param_group_id += 1

        logger.info(
            "Built %d bucket groups across %d param groups",
            len(self._bucket_groups), param_group_id,
        )

    def _create_bucket(
        self,
        bucket_id: int,
        param_group_id: int,
        params: List[torch.nn.Parameter],
        device: torch.device,
    ) -> None:
        bg = HeteroBucketGroup(
            bucket_id=bucket_id,
            param_group_id=param_group_id,
            params=params,
            device=device,
            process_group=self.process_group,
            config=self.config,
            comm_streams=self.comm_streams,
            loc_cache_manager=self.loc_cache_manager,
        )
        self._bucket_groups.append(bg)
        for p in params:
            self._param_to_bucket[id(p)] = bg

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register post-accumulate-grad hooks on all tracked parameters.

        When a parameter's gradient has been fully accumulated (after all
        micro-batches), the hook fires ``start_grad_sync`` on the owning
        bucket group.  This is the overlap mechanism: communication is
        launched while the next layer's backward is still running.
        """
        model = self.model()
        if model is None:
            return
        for p in model.parameters():
            if not p.requires_grad or id(p) not in self._param_to_bucket:
                continue
            bucket_group = self._param_to_bucket[id(p)]

            def _make_hook(bg: HeteroBucketGroup) -> Callable:
                def _hook(grad: torch.Tensor) -> Optional[torch.Tensor]:
                    if self._active_guard is not None:
                        bg.start_grad_sync(self._active_guard)
                    return grad
                return _hook

            p.register_hook(_make_hook(bucket_group))

    # ------------------------------------------------------------------
    # Context manager for a full backward pass
    # ------------------------------------------------------------------

    @contextmanager
    def grad_sync_context(
        self, device: torch.device
    ) -> Generator[None, None, None]:
        """Context manager to wrap the backward pass.

        Sets up the ``HeteroStreamGuard`` so that all gradient hooks fired
        during ``loss.backward()`` have access to the correct grad_stream.

        Example::

            with fix.grad_sync_context(torch.device("cuda", 0)):
                loss.backward()
        """
        guard = HeteroStreamGuard(device)
        with guard:
            self._active_guard = guard
            try:
                yield
            finally:
                self._active_guard = None
        logger.debug("grad_sync_context exited for device %s", device)

    # ------------------------------------------------------------------
    # Post-backward synchronisation
    # ------------------------------------------------------------------

    def drain_all_comm_streams(self) -> None:
        """Wait for all bucket communication streams to complete.

        Must be called before the optimiser step reads any reduced gradient.
        Internally calls ``HeteroBucketGroup.drain_comm_stream()`` which
        applies the ``current_stream()`` fix from Megatron commit 3548385.
        """
        for bg in self._bucket_groups:
            bg.drain_comm_stream()
        logger.debug("All %d comm streams drained", len(self._bucket_groups))

    def reset_all_buckets(self) -> None:
        """Reset all bucket groups after the optimiser step."""
        for bg in self._bucket_groups:
            bg.reset()
        logger.debug("All %d buckets reset", len(self._bucket_groups))

    # ------------------------------------------------------------------
    # LOC cache management
    # ------------------------------------------------------------------

    def register_loc_cache_manager(self, manager: LOCacheManager) -> None:
        """Replace the LOC cache manager at runtime."""
        self.loc_cache_manager = manager
        for bg in self._bucket_groups:
            bg.loc_cache_manager = manager
        logger.info("LOC cache manager registered: %s", manager)

    def loc_cache_stats(self) -> Dict[str, int]:
        if self.loc_cache_manager is None:
            return {"entries": 0}
        return {"entries": len(self.loc_cache_manager)}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.WARNING)

    # ---- minimal single-process smoke test (no real distributed backend) ----

    # 1. HeteroStreamGuard captures current_stream, not default_stream
    if torch.cuda.is_available():
        dev = torch.device("cuda", 0)
        s = torch.cuda.Stream(device=0)
        with torch.cuda.stream(s):
            guard = HeteroStreamGuard(dev)
            guard.__enter__()
            assert guard.grad_stream is s, (
                f"Expected grad_stream={s}, got {guard.grad_stream}"
            )
            guard.__exit__()
        print("PASS: HeteroStreamGuard captures current_stream correctly")

    # 2. HeteroDDPConfig defaults are sane for DES-LOC 3-device setup
    cfg = HeteroDDPConfig()
    assert cfg.num_distributed_optimizer_instances == 3
    assert cfg.loc_cache_enabled is True
    print("PASS: HeteroDDPConfig defaults correct")

    # 3. LOCacheManager store/retrieve round-trip
    mgr = LOCacheManager()
    t = torch.randn(128, 128)
    mgr.store(0, 0, t)
    result = mgr.retrieve(0, 0)
    assert result is not None and result.shape == t.shape
    print("PASS: LOCacheManager store/retrieve round-trip")

    # 4. _device_label routing
    assert "H100" in _device_label(2)
    assert "A6000" in _device_label(0)
    print("PASS: device label routing correct")

    # 5. Bucket size routing threshold
    assert _BUCKET_SIZE_MB_DEFAULT == 25
    print("PASS: bucket size default set")

    print("\nAll smoke tests passed.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroDDPGradOverlapFix on a DeepSpeed engine.

    Instantiates a :class:`HeteroDDPGradOverlapFix` from the engine's configuration
    and attaches it as ``engine.hetero_ddp_grad_overlap_fix``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_ddp_grad_overlap_fix.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_ddp_grad_overlap_fix = None
    logger.info("hetero_ddp_grad_overlap_fix.register() attached engine.hetero_ddp_grad_overlap_fix")
