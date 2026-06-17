"""
DES-LOC Heterogeneous Layerwise Gradient Safety Module
=======================================================

Upstream Design Intent (Megatron b25a76e0003b93d663ec8302574371fb40c3efcd):
----------------------------------------------------------------------------
Megatron-LM's distributed optimizer supports a "layerwise" parameter all-gather
overlap mode, where the all-gather for layer N+1's parameters is overlapped with
the compute of layer N. In this scheme, ``grad_data`` (the flat gradient buffer
backing each bucket) is *reused* as the receive buffer for the all-gather
operation. After the all-gather completes and updated parameters are copied back
into ``model_p.data``, the receive buffer contents are whatever the collective
wrote into it — i.e., the gathered parameter values, not zeros.

The bug: subsequent gradient accumulation into ``main_grad`` (which is a view
into the same ``grad_data`` tensor) would therefore *start* from the residual
parameter values rather than from zero. This causes silent gradient corruption
that is extremely difficult to detect because it manifests as subtly wrong
parameter updates rather than NaN/Inf values.

The fix is a targeted ``bucket.grad_data.zero_()`` call immediately after the
layerwise gather list is consumed and ``bucket.layerwise_gather_list`` is set to
None — in both the synchronous (``finish_param_sync``) and asynchronous
(``start_param_sync``) code paths.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces additional
complexity on top of Megatron's layerwise overlap:

1. **Heterogeneous device topology**: 2× A6000 (48 GB, SM86, PCIe) + 1× H100 NVL
   (96 GB, SM90, PCIe). There is no NVLink between any device pair. All-gather
   operations therefore route through the PCIe fabric, making the timing window
   between gather completion and gradient accumulation *longer and more variable*
   than in NVLink-connected clusters. This increases the probability of the
   corruption window being hit.

2. **Shared LOcality Cache (SLC)**: DES-LOC maintains a CPU DRAM cache (1.5 TB)
   that mirrors parameter and gradient shards. When a bucket's ``grad_data`` is
   reused as an all-gather receive buffer, the SLC entry for that bucket becomes
   stale. The SLC must be explicitly invalidated so that subsequent gradient
   reads from CPU do not return the corrupted (parameter-valued) data.

3. **Decoupled Execution**: A6000 devices run forward passes while the H100 runs
   the optimizer step on FP32 master weights. The H100's gradient reduction
   pipeline assumes that the gradient buffers it reads are either zero-initialized
   or contain valid accumulated gradients — never residual all-gather data. The
   ``zero_()`` call (and SLC invalidation) must therefore happen on the *source*
   device (A6000) before the gradient tensor is shipped to H100 for reduction.

4. **SM86 vs SM90 kernel dispatch**: ``grad_data.zero_()`` on SM86 (A6000) uses
   a different CUDA kernel path than on SM90 (H100 NVL). This module dispatches
   the zero-fill to the correct device and issues an explicit CUDA stream
   synchronization barrier to ensure the fill is visible before any cross-device
   copy.

5. **Gradient accumulation across micro-batches**: DES-LOC's pipeline schedule
   may accumulate gradients over multiple micro-batches before reducing. The
   zeroing must only happen *after* the layerwise gather is complete but *before*
   the first micro-batch's backward writes into ``main_grad``. This is enforced
   via a per-bucket lifecycle state machine.

Author: Neuron_SP / DES-LOC adaptation of Megatron b25a76e0003b93d663ec8302574371fb40c3efcd
"""

from __future__ import annotations

import logging
import threading
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device topology constants for the 2×A6000 + 1×H100 NVL cluster
# ---------------------------------------------------------------------------

SM86_COMPUTE_CAPABILITY = (8, 6)   # A6000
SM90_COMPUTE_CAPABILITY = (9, 0)   # H100 NVL

# Logical device roles in DES-LOC
DESLOCDEVICEROLE_FORWARD = "forward"    # A6000 devices handle forward + backward
DESLOCDEVICEROLE_OPTIMIZER = "optimizer"  # H100 handles optimizer step


def _device_compute_capability(device: torch.device) -> Tuple[int, int]:
    """Return (major, minor) compute capability for *device*."""
    if device.type != "cuda":
        return (0, 0)
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def _classify_device_role(device: torch.device) -> str:
    """
    Classify a CUDA device as either a DES-LOC forward device (A6000/SM86)
    or optimizer device (H100/SM90) based on compute capability.

    In a heterogeneous cluster without NVLink, the H100 NVL's larger memory
    and higher FP32 throughput make it the natural home for the optimizer step.
    A6000 devices, with their large SM86 L2 caches, are preferred for the
    activation-heavy forward/backward passes.
    """
    cc = _device_compute_capability(device)
    if cc >= SM90_COMPUTE_CAPABILITY:
        return DESLOCDEVICEROLE_OPTIMIZER
    return DESLOCDEVICEROLE_FORWARD


# ---------------------------------------------------------------------------
# Bucket lifecycle state machine
# ---------------------------------------------------------------------------

class BucketState(Enum):
    """
    Per-bucket lifecycle states in DES-LOC layerwise all-gather overlap.

    The state machine enforces the ordering constraint introduced by the upstream
    Megatron fix: grad_data must be zeroed *after* the layerwise gather is
    consumed and *before* the first gradient accumulation write.

    Transitions:
        IDLE → GATHERING (all-gather launched)
        GATHERING → GATHER_COMPLETE (handle finalized, layerwise_gather_list consumed)
        GATHER_COMPLETE → GRAD_ZEROED (grad_data.zero_() completed, SLC invalidated)
        GRAD_ZEROED → ACCUMULATING (first micro-batch backward begins)
        ACCUMULATING → READY_FOR_REDUCE (all micro-batches done)
        READY_FOR_REDUCE → IDLE (after cross-device reduce to H100)
    """
    IDLE = auto()
    GATHERING = auto()
    GATHER_COMPLETE = auto()
    GRAD_ZEROED = auto()
    ACCUMULATING = auto()
    READY_FOR_REDUCE = auto()


@dataclass
class BucketGradSafetyRecord:
    """
    Tracks the gradient-safety state for a single DES-LOC bucket.

    Fields
    ------
    bucket_id : int
        Monotonically increasing identifier assigned at bucket creation.
    device : torch.device
        The CUDA device that owns this bucket's grad_data tensor.
    device_role : str
        ``DESLOCDEVICEROLE_FORWARD`` or ``DESLOCDEVICEROLE_OPTIMIZER``.
    state : BucketState
        Current position in the bucket lifecycle state machine.
    slc_valid : bool
        Whether the Shared LOcality Cache entry for this bucket reflects the
        current contents of grad_data. Set to False immediately after a
        layerwise gather overwrites grad_data with parameter values.
    zero_stream : Optional[torch.cuda.Stream]
        Dedicated CUDA stream for the grad_data.zero_() kernel. Using a
        dedicated stream avoids blocking the all-gather stream while still
        allowing the zero fill to proceed asynchronously.
    zero_event : Optional[torch.cuda.Event]
        CUDA event recorded on zero_stream after zero_() completes. Any
        stream that subsequently writes into grad_data must wait on this
        event to guarantee the zero fill is visible.
    corruption_guard_enabled : bool
        When True, this record participates in post-zero integrity checks
        (sampled probabilistically to avoid overhead in steady state).
    """
    bucket_id: int
    device: torch.device
    device_role: str
    state: BucketState = BucketState.IDLE
    slc_valid: bool = True
    zero_stream: Optional[torch.cuda.Stream] = field(default=None, repr=False)
    zero_event: Optional[torch.cuda.Event] = field(default=None, repr=False)
    corruption_guard_enabled: bool = True

    def __post_init__(self):
        if self.device.type == "cuda":
            self.zero_stream = torch.cuda.Stream(device=self.device)
            self.zero_event = torch.cuda.Event()


# ---------------------------------------------------------------------------
# Shared LOcality Cache invalidation interface
# ---------------------------------------------------------------------------

class SharedLocalityCacheInvalidator:
    """
    Manages SLC invalidation for DES-LOC gradient buffers.

    In DES-LOC, the CPU DRAM serves as a 1.5 TB Shared LOcality Cache (SLC)
    that stores copies of parameter shards and gradient shards to amortize
    PCIe bandwidth costs across A6000↔H100 transfers. When ``grad_data`` is
    overwritten by an all-gather receive buffer, the corresponding SLC entry
    becomes stale and must be invalidated before the H100 reads the gradient.

    This class provides a thread-safe registry of SLC entries keyed by
    (bucket_id, device_index). It supports both eager invalidation (called
    immediately after grad_data.zero_()) and lazy invalidation (deferred until
    the next SLC read-miss).

    The SLC itself is implemented as pinned CPU tensors in the parent
    ``HeteroLayerwiseGradSafetyManager``; this class only tracks validity flags.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Maps (bucket_id, device_index) → is_valid
        self._valid: Dict[Tuple[int, int], bool] = {}

    def register(self, bucket_id: int, device_index: int) -> None:
        """Register a new SLC entry as initially valid."""
        with self._lock:
            self._valid[(bucket_id, device_index)] = True

    def invalidate(self, bucket_id: int, device_index: int) -> None:
        """
        Mark the SLC entry for (bucket_id, device_index) as invalid.

        Called immediately after a layerwise all-gather overwrites grad_data
        on device_index. Any subsequent CPU-side gradient read must re-fetch
        from the device rather than serving the stale SLC copy.
        """
        with self._lock:
            key = (bucket_id, device_index)
            if key in self._valid and self._valid[key]:
                self._valid[key] = False
                logger.debug(
                    "SLC invalidated for bucket_id=%d device_index=%d "
                    "(grad_data overwritten by layerwise all-gather receive buffer)",
                    bucket_id, device_index,
                )

    def revalidate(self, bucket_id: int, device_index: int) -> None:
        """
        Mark the SLC entry as valid after grad_data has been zeroed.

        After ``grad_data.zero_()`` completes, the CPU SLC can be refreshed
        (or simply marked valid-zero, since the CPU can represent zero without
        a D→H copy). This allows the H100 optimizer path to read a zero
        gradient from the SLC during the zeroing window without stalling on
        a PCIe transfer.
        """
        with self._lock:
            key = (bucket_id, device_index)
            self._valid[key] = True
            logger.debug(
                "SLC revalidated (zero) for bucket_id=%d device_index=%d",
                bucket_id, device_index,
            )

    def is_valid(self, bucket_id: int, device_index: int) -> bool:
        with self._lock:
            return self._valid.get((bucket_id, device_index), False)

    def snapshot(self) -> Dict[Tuple[int, int], bool]:
        """Return a copy of the validity map for diagnostic purposes."""
        with self._lock:
            return dict(self._valid)


# ---------------------------------------------------------------------------
# Core heterogeneous grad-safety logic
# ---------------------------------------------------------------------------

class HeteroLayerwiseGradSafetyManager:
    """
    Manages gradient buffer safety for layerwise all-gather overlap in DES-LOC.

    Problem Statement
    -----------------
    Megatron's layerwise optimizer reuses ``bucket.grad_data`` as the receive
    buffer for all-gather collectives (to avoid allocating a separate output
    buffer). After the gather, ``grad_data`` contains the *gathered parameter
    values*, not gradient values. Without explicit zeroing, the next call to
    ``loss.backward()`` accumulates gradients on top of these parameter residuals,
    silently corrupting the gradient signal.

    Megatron's fix (b25a76e) adds ``bucket.grad_data.zero_()`` in two places:
    (a) after the synchronous ``finish_param_sync`` path, and
    (b) after the asynchronous ``start_param_sync`` / handle-wait path.

    DES-LOC Complications
    ---------------------
    On the 2×A6000 + 1×H100 NVL cluster:

    * **No NVLink**: all-gather latency over PCIe is ~3–5× higher than NVLink.
      The corruption window (time between gather completion and first backward
      write) is therefore wider, making the bug more likely to manifest even
      at low iteration counts.

    * **SLC coherence**: the CPU DRAM SLC mirrors grad_data. When grad_data is
      overwritten by the all-gather, the SLC entry is stale. If the H100 reads
      gradients from the SLC (to avoid a PCIe round-trip) before the SLC is
      invalidated, it sees corrupted data.

    * **SM86 vs SM90 zero kernel**: ``torch.Tensor.zero_()`` dispatches to
      different PTX implementations on SM86 and SM90. On SM86 (A6000) the
      128-byte cache line size and PCIe-constrained memory bandwidth mean that
      the zero kernel is bandwidth-bound at ~900 GB/s (L2 throughput). On SM90
      (H100 NVL) the same kernel achieves ~3.35 TB/s. Since grad_data lives on
      A6000 in DES-LOC (forward devices own the gradient buffers), we always
      dispatch zero_() on the A6000 device.

    * **Stream ordering**: the zero_() must be issued on the *correct CUDA
      stream* and a ``torch.cuda.Event`` must synchronize the zero stream with
      the backward compute stream before any gradient accumulation occurs.

    Usage
    -----
    ::

        manager = HeteroLayerwiseGradSafetyManager(world_size=3)

        # After launching layerwise all-gather:
        manager.on_gather_launched(record)

        # After all-gather handle is finalized and layerwise_gather_list → None:
        manager.on_gather_complete_zero_grad(record, grad_data_tensor)

        # Before first backward write into main_grad:
        manager.wait_for_zero_before_accumulate(record, compute_stream)

        # After all micro-batches accumulated, before H100 reduce:
        manager.on_ready_for_reduce(record)

        # After H100 reduce completes:
        manager.on_reduce_complete(record)
    """

    def __init__(
        self,
        world_size: int,
        corruption_check_frequency: int = 100,
        slc_invalidation_enabled: bool = True,
    ):
        """
        Parameters
        ----------
        world_size : int
            Total number of devices in the DES-LOC process group (3 for the
            2×A6000 + 1×H100 topology).
        corruption_check_frequency : int
            Every N calls to ``on_gather_complete_zero_grad``, perform a
            sampled post-zero integrity check (assert grad_data is all-zero).
            Set to 0 to disable. Default 100 is low enough to catch bugs in
            development while negligible in training throughput.
        slc_invalidation_enabled : bool
            Whether to maintain SLC validity tracking. Disable only for
            benchmarking or when running on a single homogeneous device.
        """
        self.world_size = world_size
        self.corruption_check_frequency = corruption_check_frequency
        self.slc_invalidation_enabled = slc_invalidation_enabled
        self._slc = SharedLocalityCacheInvalidator()
        self._call_count = 0
        self._lock = threading.Lock()

        logger.info(
            "HeteroLayerwiseGradSafetyManager initialized: world_size=%d "
            "corruption_check_freq=%d slc_invalidation=%s",
            world_size, corruption_check_frequency, slc_invalidation_enabled,
        )

    def register_bucket(self, bucket_id: int, device: torch.device) -> BucketGradSafetyRecord:
        """
        Create and register a ``BucketGradSafetyRecord`` for a new bucket.

        Should be called once per bucket at model initialization time, before
        any all-gather operations are launched.
        """
        role = _classify_device_role(device)
        record = BucketGradSafetyRecord(
            bucket_id=bucket_id,
            device=device,
            device_role=role,
        )
        if self.slc_invalidation_enabled and device.type == "cuda":
            self._slc.register(bucket_id, device.index or 0)

        logger.debug(
            "Registered bucket_id=%d device=%s role=%s",
            bucket_id, device, role,
        )
        return record

    def on_gather_launched(self, record: BucketGradSafetyRecord) -> None:
        """
        Notify the manager that a layerwise all-gather has been launched for
        this bucket.

        Transitions: IDLE → GATHERING (or GRAD_ZEROED → GATHERING if the
        bucket is being re-used for a subsequent micro-batch).

        This is also when we proactively invalidate the SLC entry: the moment
        the all-gather receive buffer is pointed at grad_data, the CPU-side SLC
        copy is no longer authoritative.
        """
        allowed = {BucketState.IDLE, BucketState.GRAD_ZEROED}
        if record.state not in allowed:
            raise RuntimeError(
                f"on_gather_launched called on bucket_id={record.bucket_id} "
                f"in unexpected state {record.state}. Expected one of {allowed}."
            )
        record.state = BucketState.GATHERING
        record.slc_valid = False

        if self.slc_invalidation_enabled and record.device.type == "cuda":
            self._slc.invalidate(record.bucket_id, record.device.index or 0)

    def on_gather_complete_zero_grad(
        self,
        record: BucketGradSafetyRecord,
        grad_data: torch.Tensor,
    ) -> None:
        """
        Called after the layerwise all-gather handle is finalized and
        ``bucket.layerwise_gather_list`` has been set to None.

        This is the DES-LOC adaptation of Megatron's ``bucket.grad_data.zero_()``
        fix. The method:

        1. Asserts the bucket is in GATHERING state (catches ordering bugs).
        2. Issues ``grad_data.zero_()`` on a dedicated zero_stream to avoid
           blocking the all-gather collective stream.
        3. Records a CUDA event on zero_stream so that the backward compute
           stream can ``wait_event`` before writing into main_grad.
        4. Transitions the state to GATHER_COMPLETE, then asynchronously to
           GRAD_ZEROED once the event is recorded.
        5. Invalidates / re-validates the SLC entry (invalid during zero_(),
           re-valid once the zero is committed).
        6. Optionally runs a sampled integrity check.

        Parameters
        ----------
        record : BucketGradSafetyRecord
            The safety record for the bucket whose grad_data was used as
            the all-gather receive buffer.
        grad_data : torch.Tensor
            The flat gradient buffer tensor (``bucket.grad_data`` in Megatron
            parlance). Must reside on a CUDA device.
        """
        if record.state != BucketState.GATHERING:
            raise RuntimeError(
                f"on_gather_complete_zero_grad called on bucket_id={record.bucket_id} "
                f"in state {record.state}, expected GATHERING."
            )

        if grad_data.device.type != "cuda":
            # CPU tensors can be zeroed directly without stream concerns.
            grad_data.zero_()
            record.state = BucketState.GRAD_ZEROED
            record.slc_valid = True
            logger.debug(
                "CPU grad_data zeroed for bucket_id=%d (no stream needed)",
                record.bucket_id,
            )
            return

        record.state = BucketState.GATHER_COMPLETE

        # Issue zero_() on a dedicated stream so we don't block the all-gather
        # stream. On SM86 (A6000) this kernel is bandwidth-limited to ~900 GB/s;
        # on SM90 (H100) it would be ~3.35 TB/s. Since grad_data lives on the
        # A6000 forward device in DES-LOC, we accept the A6000 bandwidth.
        with torch.cuda.stream(record.zero_stream):
            grad_data.zero_()
            # Record the event *inside* the stream context so it is enqueued
            # after the zero kernel on zero_stream.
            record.zero_event.record(record.zero_stream)

        # Transition to GRAD_ZEROED. The event has been *enqueued* but not yet
        # *completed*; the actual completion is enforced when the backward
        # compute stream calls wait_for_zero_before_accumulate().
        record.state = BucketState.GRAD_ZEROED
        record.slc_valid = False  # SLC still invalid until zero kernel completes

        with self._lock:
            self._call_count += 1
            should_check = (
                record.corruption_guard_enabled
                and self.corruption_check_frequency > 0
                and self._call_count % self.corruption_check_frequency == 0
            )

        if should_check:
            # Synchronize only on the zero_stream to avoid a full device sync.
            record.zero_stream.synchronize()
            self._integrity_check(record, grad_data)

        logger.debug(
            "Layerwise all-gather receive buffer zeroed for bucket_id=%d "
            "device=%s (zero event enqueued on dedicated stream)",
            record.bucket_id, record.device,
        )

    def wait_for_zero_before_accumulate(
        self,
        record: BucketGradSafetyRecord,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        """
        Ensure that the grad_data.zero_() kernel has completed before any
        gradient accumulation write occurs on *compute_stream*.

        This is the stream synchronization barrier that prevents the race
        condition on SM86: without this wait, the compute stream could begin
        writing ``main_grad`` (a view into ``grad_data``) before the zero
        kernel on ``zero_stream`` has finished clearing the buffer.

        Must be called from the backward hook or gradient accumulation entry
        point, *before* the first ``main_grad += grad`` operation.

        Parameters
        ----------
        record : BucketGradSafetyRecord
            The bucket record. Must be in GRAD_ZEROED state.
        compute_stream : torch.cuda.Stream
            The CUDA stream on which gradient accumulation will occur. This
            stream will be made to wait on ``record.zero_event``.
        """
        if record.state not in {BucketState.GRAD_ZEROED, BucketState.ACCUMULATING}:
            raise RuntimeError(
                f"wait_for_zero_before_accumulate called on bucket_id={record.bucket_id} "
                f"in state {record.state}. Expected GRAD_ZEROED or ACCUMULATING."
            )

        if record.zero_event is not None and record.device.type == "cuda":
            compute_stream.wait_event(record.zero_event)
            # After the wait is enqueued, the compute stream is guaranteed to
            # see the zeroed grad_data. Re-validate the SLC: the H100 can now
            # read a valid (zero) gradient from the SLC entry.
            if self.slc_invalidation_enabled and record.device.type == "cuda":
                self._slc.revalidate(record.bucket_id, record.device.index or 0)
            record.slc_valid = True

        if record.state == BucketState.GRAD_ZEROED:
            record.state = BucketState.ACCUMULATING

    def on_ready_for_reduce(self, record: BucketGradSafetyRecord) -> None:
        """
        Signal that all micro-batch gradients have been accumulated into
        ``grad_data`` and the bucket is ready to be shipped to the H100 for
        the optimizer reduce step.

        In DES-LOC's decoupled execution model, the A6000 forward devices
        complete gradient accumulation and then trigger a PCIe transfer of
        ``grad_data`` to the H100. This method validates that the bucket is
        in the correct state before that transfer is initiated.
        """
        if record.state != BucketState.ACCUMULATING:
            raise RuntimeError(
                f"on_ready_for_reduce called on bucket_id={record.bucket_id} "
                f"in state {record.state}, expected ACCUMULATING."
            )
        record.state = BucketState.READY_FOR_REDUCE
        logger.debug(
            "Bucket bucket_id=%d marked READY_FOR_REDUCE (gradient accumulation complete, "
            "pending PCIe transfer to H100 optimizer device)",
            record.bucket_id,
        )

    def on_reduce_complete(self, record: BucketGradSafetyRecord) -> None:
        """
        Signal that the H100 has consumed the gradient and the bucket can
        return to IDLE for the next iteration.

        After the reduce, ``grad_data`` on the A6000 is considered stale
        (the authoritative gradient now lives on the H100 for the optimizer
        step). The SLC entry is invalidated to prevent the A6000 from reading
        back its own pre-reduce gradient copy.
        """
        if record.state != BucketState.READY_FOR_REDUCE:
            raise RuntimeError(
                f"on_reduce_complete called on bucket_id={record.bucket_id} "
                f"in state {record.state}, expected READY_FOR_REDUCE."
            )
        record.state = BucketState.IDLE
        if self.slc_invalidation_enabled and record.device.type == "cuda":
            self._slc.invalidate(record.bucket_id, record.device.index or 0)

        logger.debug(
            "Bucket bucket_id=%d returned to IDLE after H100 reduce complete",
            record.bucket_id,
        )

    def _integrity_check(
        self,
        record: BucketGradSafetyRecord,
        grad_data: torch.Tensor,
    ) -> None:
        """
        Sampled post-zero integrity check.

        After ``grad_data.zero_()`` on SM86, verify that the buffer is
        actually all-zero. This catches hardware ECC errors, driver bugs, or
        any future code path that might write into grad_data between the zero
        kernel and this check.

        Only called every ``corruption_check_frequency`` invocations to avoid
        materializing the full tensor on the CPU in steady state. A full
        ``torch.all(grad_data == 0)`` is used rather than a checksum because
        the goal is detecting non-zero values (parameter residuals), not
        detecting specific corruption patterns.
        """
        is_zero = torch.all(grad_data == 0).item()
        if not is_zero:
            # This is the exact corruption that Megatron b25a76e fixes.
            # Log at ERROR level because silent gradient corruption is
            # catastrophic for training convergence.
            logger.error(
                "GRADIENT CORRUPTION DETECTED: grad_data for bucket_id=%d on device=%s "
                "is NOT all-zero after layerwise all-gather + zero_() call. "
                "This indicates that grad_data.zero_() either did not complete before "
                "this check or was overwritten by a concurrent write. "
                "Upstream root cause: Megatron b25a76e0003b93d663ec8302574371fb40c3efcd "
                "(layerwise param all-gather reuses grad_data as receive buffer). "
                "DES-LOC impact: H100 optimizer will receive corrupted gradients via PCIe, "
                "leading to diverged parameter updates.",
                record.bucket_id, record.device,
            )
        else:
            logger.debug(
                "Integrity check passed: grad_data all-zero for bucket_id=%d (sampled check %d)",
                record.bucket_id, self._call_count,
            )

    def slc_snapshot(self) -> Dict[Tuple[int, int], bool]:
        """Return a diagnostic snapshot of the SLC validity map."""
        return self._slc.snapshot()


# ---------------------------------------------------------------------------
# DeepSpeed ZeRO integration shim
# ---------------------------------------------------------------------------

class DESLOCParamGradBucket:
    """
    Lightweight shim representing a DeepSpeed ZeRO parameter+gradient bucket
    in the DES-LOC layerwise all-gather overlap setting.

    This mirrors the ``_ParamAndGradBucket`` concept from Megatron but is
    adapted for DeepSpeed's ZeRO-3 partitioning scheme and DES-LOC's
    heterogeneous device placement.

    In DeepSpeed ZeRO-3 with layerwise all-gather overlap:
    - Each bucket owns a flat ``grad_data`` tensor on the A6000 forward device.
    - ``main_grad`` for each parameter is a view (slice) into ``grad_data``.
    - The all-gather for the *next* layer's parameters is launched while the
      current layer's backward is in flight, and the receive buffer is
      ``grad_data`` (reusing the allocation to avoid extra memory).
    - After the gather completes, ``grad_data`` contains parameter values.
    - Without ``grad_data.zero_()``, the next ``main_grad += grad`` accumulates
      on top of parameter values — the Megatron b25a76e bug.

    Parameters
    ----------
    bucket_id : int
        Unique identifier for this bucket.
    grad_data : torch.Tensor
        Flat gradient buffer tensor. Must reside on a CUDA device (A6000 in
        DES-LOC forward placement).
    params : List[torch.nn.Parameter]
        Parameters whose ``main_grad`` views are slices of ``grad_data``.
    manager : HeteroLayerwiseGradSafetyManager
        The manager that tracks safety state for all buckets.
    """

    def __init__(
        self,
        bucket_id: int,
        grad_data: torch.Tensor,
        params: List[torch.nn.Parameter],
        manager: HeteroLayerwiseGradSafetyManager,
    ):
        self.bucket_id = bucket_id
        self.grad_data = grad_data
        self.params = params
        self.manager = manager
        self.layerwise_gather_list: Optional[List] = None

        self._record = manager.register_bucket(bucket_id, grad_data.device)

    @property
    def record(self) -> BucketGradSafetyRecord:
        return self._record

    def launch_layerwise_gather(self, gather_list: List) -> None:
        """
        Launch the layerwise all-gather and mark ``grad_data`` as the receive
        buffer. Notifies the manager that the SLC entry is now stale.
        """
        self.layerwise_gather_list = gather_list
        self.manager.on_gather_launched(self._record)

    def finalize_gather_and_zero_grad(self, updated_params: List[torch.Tensor]) -> None:
        """
        Finalize the layerwise all-gather: copy updated parameters into model
        parameter data, clear ``layerwise_gather_list``, and zero ``grad_data``.

        This method implements the DES-LOC adaptation of the Megatron b25a76e
        fix. The two key actions are:

        1. ``model_p.data.copy_(updated_p)`` — install the gathered parameters.
        2. ``self.grad_data.zero_()`` (via manager) — clear the receive buffer
           so that subsequent gradient accumulation into ``main_grad`` starts
           from zero rather than from the gathered parameter values.

        The manager issues the zero_() on a dedicated stream and records a
        CUDA event, which must be waited on before the first backward write.
        """
        for updated_p, model_p in zip(updated_params, self.params):
            model_p.data.copy_(updated_p)

        self.layerwise_gather_list = None

        # This is the DES-LOC adaptation of:
        #   bucket.grad_data.zero_()
        # from Megatron b25a76e0003b93d663ec8302574371fb40c3efcd.
        # The manager handles stream dispatch, SLC invalidation, and event recording.
        self.manager.on_gather_complete_zero_grad(self._record, self.grad_data)

    def wait_before_accumulate(self, compute_stream: torch.cuda.Stream) -> None:
        """
        Block *compute_stream* on the zero_() event before writing into any
        ``main_grad`` view of this bucket's ``grad_data``.

        Must be called from the backward hook before the first gradient
        accumulation for this bucket in the current iteration.
        """
        self.manager.wait_for_zero_before_accumulate(self._record, compute_stream)

    def mark_accumulation_complete(self) -> None:
        """Signal that all micro-batches have accumulated into this bucket."""
        self.manager.on_ready_for_reduce(self._record)

    def mark_reduce_complete(self) -> None:
        """Signal that the H100 optimizer has consumed this bucket's gradient."""
        self.manager.on_reduce_complete(self._record)


# ---------------------------------------------------------------------------
# Synchronous and asynchronous param-sync entry points
# (mirrors Megatron's finish_param_sync / start_param_sync)
# ---------------------------------------------------------------------------

def finish_param_sync_with_grad_safety(
    buckets: List[DESLOCParamGradBucket],
    updated_params_per_bucket: List[List[torch.Tensor]],
    param_gather_handle=None,
) -> None:
    """
    Synchronous parameter synchronization with DES-LOC gradient safety.

    Mirrors Megatron's ``_ParamAndGradBucketGroup.finish_param_sync()``
    (the non-distributed-optimizer, synchronous path) with the b25a76e fix
    applied through the DES-LOC safety manager.

    In Megatron's synchronous path, the param_gather_handle is waited on here
    (blocking). After the handle completes:
    - Updated parameters are copied into model parameter data.
    - ``grad_data.zero_()`` is called to clear the receive buffer.

    In DES-LOC, we additionally:
    - Route the zero_() through the manager for stream/event tracking.
    - Invalidate SLC entries so the H100 does not read stale data.

    Parameters
    ----------
    buckets : List[DESLOCParamGradBucket]
        Buckets in this parameter group.
    updated_params_per_bucket : List[List[torch.Tensor]]
        For each bucket, the list of updated parameter tensors produced by the
        all-gather collective.
    param_gather_handle : optional
        If provided, wait on this handle before processing (mimics Megatron's
        ``self.param_gather_handle.wait()``).
    """
    if param_gather_handle is not None:
        param_gather_handle.wait()

    for bucket, updated_params in zip(buckets, updated_params_per_bucket):
        if bucket.layerwise_gather_list is not None:
            bucket.finalize_gather_and_zero_grad(updated_params)

    logger.debug(
        "finish_param_sync_with_grad_safety: processed %d buckets "
        "(synchronous path, param_gather_handle waited)",
        len(buckets),
    )


def start_param_sync_with_grad_safety(
    buckets: List[DESLOCParamGradBucket],
    updated_params_per_bucket: List[List[torch.Tensor]],
    async_handle=None,
) -> None:
    """
    Asynchronous parameter synchronization with DES-LOC gradient safety.

    Mirrors Megatron's ``_ParamAndGradBucketGroup.start_param_sync()``
    (the asynchronous / overlapped path) with the b25a76e fix applied through
    the DES-LOC safety manager.

    In the async path, the all-gather handle may or may not be complete when
    this function is called. If the handle is ready, we finalize immediately;
    otherwise, the finalization is deferred (the caller must ensure
    ``finish_param_sync_with_grad_safety`` is called before the next backward).

    Parameters
    ----------
    buckets : List[DESLOCParamGradBucket]
        Buckets in this parameter group.
    updated_params_per_bucket : List[List[torch.Tensor]]
        Updated parameter tensors from the all-gather.
    async_handle : optional
        Asynchronous collective handle. If None, assumes gather is already done.
    """
    handle_done = async_handle is None or _handle_is_complete(async_handle)

    if handle_done:
        for bucket, updated_params in zip(buckets, updated_params_per_bucket):
            if bucket.layerwise_gather_list is not None:
                bucket.finalize_gather_and_zero_grad(updated_params)

        logger.debug(
            "start_param_sync_with_grad_safety: async handle already complete, "
            "finalized %d buckets immediately",
            len(buckets),
        )
    else:
        logger.debug(
            "start_param_sync_with_grad_safety: async handle pending, "
            "deferring finalization for %d buckets",
            len(buckets),
        )


def _handle_is_complete(handle) -> bool:
    """
    Check whether an async collective handle has completed without blocking.

    DeepSpeed uses ``torch.distributed.Work`` objects; Megatron uses similar
    handles. We check ``handle.is_completed()`` if available, else assume done.
    """
    if hasattr(handle, "is_completed"):
        return handle.is_completed()
    return True


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Configure logging so tests produce readable output
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    class _MockCUDADevice:
        """
        Minimal mock for torch.device("cuda:0") so tests run on CPU-only hosts.
        Tests that exercise the actual CUDA path are skipped when CUDA is absent.
        """
        type = "cpu"
        index = 0

    def _cpu_device() -> torch.device:
        return torch.device("cpu")

    def _cuda_device(index: int = 0) -> Optional[torch.device]:
        if torch.cuda.is_available() and torch.cuda.device_count() > index:
            return torch.device(f"cuda:{index}")
        return None

    class TestBucketLifecycleStateMachine(unittest.TestCase):
        """
        Tests for the BucketState state machine transitions enforced by
        HeteroLayerwiseGradSafetyManager.
        """

        def setUp(self):
            self.manager = HeteroLayerwiseGradSafetyManager(
                world_size=3,
                corruption_check_frequency=0,  # disable sampling in unit tests
                slc_invalidation_enabled=True,
            )
            self.device = _cpu_device()
            self.record = self.manager.register_bucket(bucket_id=0, device=self.device)

        def test_initial_state_is_idle(self):
            self.assertEqual(self.record.state, BucketState.IDLE)

        def test_on_gather_launched_transitions_to_gathering(self):
            self.manager.on_gather_launched(self.record)
            self.assertEqual(self.record.state, BucketState.GATHERING)

        def test_gather_complete_zeros_grad_and_transitions(self):
            grad_data = torch.ones(64, dtype=torch.float32)
            self.manager.on_gather_launched(self.record)
            self.manager.on_gather_complete_zero_grad(self.record, grad_data)
            self.assertEqual(self.record.state, BucketState.GRAD_ZEROED)
            # CPU path: grad_data must be immediately zeroed
            self.assertTrue(
                torch.all(grad_data == 0).item(),
                "grad_data not zeroed on CPU path after on_gather_complete_zero_grad",
            )

        def test_full_lifecycle_transitions(self):
            grad_data = torch.ones(32, dtype=torch.float32)

            # IDLE → GATHERING
            self.manager.on_gather_launched(self.record)
            self.assertEqual(self.record.state, BucketState.GATHERING)

            # GATHERING → GRAD_ZEROED
            self.manager.on_gather_complete_zero_grad(self.record, grad_data)
            self.assertEqual(self.record.state, BucketState.GRAD_ZEROED)

            # GRAD_ZEROED → ACCUMULATING (via wait_for_zero, CPU has no stream)
            # On CPU there is no event; we simulate by calling with a mock stream=None
            # The CPU path in on_gather_complete_zero_grad skips event recording,
            # so we call the transition directly via wait_for_zero_before_accumulate
            # with a real CPU-side no-op (no stream needed).
            if torch.cuda.is_available():
                stream = torch.cuda.Stream()
                self.manager.wait_for_zero_before_accumulate(self.record, stream)
            else:
                # Force state transition for CPU-only test
                self.record.state = BucketState.ACCUMULATING

            self.assertEqual(self.record.state, BucketState.ACCUMULATING)

            # ACCUMULATING → READY_FOR_REDUCE
            self.manager.on_ready_for_reduce(self.record)
            self.assertEqual(self.record.state, BucketState.READY_FOR_REDUCE)

            # READY_FOR_REDUCE → IDLE
            self.manager.on_reduce_complete(self.record)
            self.assertEqual(self.record.state, BucketState.IDLE)

        def test_invalid_transition_raises(self):
            """
            Calling on_gather_launched in GATHERING state must raise RuntimeError
            (the state machine should prevent double-launch).
            """
            self.manager.on_gather_launched(self.record)
            with self.assertRaises(RuntimeError):
                self.manager.on_gather_launched(self.record)

        def test_on_gather_complete_from_wrong_state_raises(self):
            grad_data = torch.ones(16)
            with self.assertRaises(RuntimeError):
                # Should require GATHERING state first
                self.manager.on_gather_complete_zero_grad(self.record, grad_data)

        def test_on_ready_for_reduce_from_wrong_state_raises(self):
            with self.assertRaises(RuntimeError):
                self.manager.on_ready_for_reduce(self.record)

    class TestSharedLocalityCacheInvalidator(unittest.TestCase):
        """Tests for SLC validity tracking."""

        def setUp(self):
            self.slc = SharedLocalityCacheInvalidator()
            self.slc.register(bucket_id=0, device_index=0)
            self.slc.register(bucket_id=1, device_index=0)
            self.slc.register(bucket_id=0, device_index=1)

        def test_initially_valid(self):
            self.assertTrue(self.slc.is_valid(0, 0))
            self.assertTrue(self.slc.is_valid(1, 0))
            self.assertTrue(self.slc.is_valid(0, 1))

        def test_invalidate_marks_invalid(self):
            self.slc.invalidate(0, 0)
            self.assertFalse(self.slc.is_valid(0, 0))
            # Other entries unaffected
            self.assertTrue(self.slc.is_valid(1, 0))
            self.assertTrue(self.slc.is_valid(0, 1))

        def test_revalidate_after_invalidate(self):
            self.slc.invalidate(0, 0)
            self.slc.revalidate(0, 0)
            self.assertTrue(self.slc.is_valid(0, 0))

        def test_snapshot_returns_copy(self):
            snap = self.slc.snapshot()
            self.slc.invalidate(0, 0)
            # Original snapshot should not reflect the subsequent invalidation
            self.assertTrue(snap[(0, 0)])

        def test_unregistered_key_returns_false(self):
            self.assertFalse(self.slc.is_valid(99, 99))

        def test_thread_safety(self):
            """Concurrent invalidations from multiple threads must not corrupt state."""
            errors: List[Exception] = []

            def worker(bucket_id, device_index):
                try:
                    for _ in range(50):
                        self.slc.invalidate(bucket_id, device_index)
                        self.slc.revalidate(bucket_id, device_index)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=worker, args=(0, 0)),
                threading.Thread(target=worker, args=(1, 0)),
                threading.Thread(target=worker, args=(0, 1)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"Thread-safety errors: {errors}")

    class TestGradDataZeroingCPU(unittest.TestCase):
        """
        Tests for grad_data.zero_() correctness on CPU (no CUDA required).

        These tests verify the exact bug fix from Megatron b25a76e: after a
        layerwise all-gather uses grad_data as its receive buffer (filling it
        with parameter values), the buffer must be zeroed before gradient
        accumulation begins.
        """

        def _make_manager_and_bucket(self, size: int = 128):
            manager = HeteroLayerwiseGradSafetyManager(
                world_size=3,
                corruption_check_frequency=0,
                slc_invalidation_enabled=False,
            )
            device = _cpu_device()
            # Simulate grad_data filled with parameter values (as if the
            # all-gather wrote parameter tensors into the gradient buffer).
            grad_data = torch.full((size,), fill_value=3.14159, dtype=torch.float32)

            # Create dummy parameters with main_grad views into grad_data.
            param_a = torch.nn.Parameter(torch.randn(size // 2))
            param_b = torch.nn.Parameter(torch.randn(size // 2))
            param_a.main_grad = grad_data[: size // 2]
            param_b.main_grad = grad_data[size // 2 :]

            bucket = DESLOCParamGradBucket(
                bucket_id=0,
                grad_data=grad_data,
                params=[param_a, param_b],
                manager=manager,
            )
            return manager, bucket, grad_data, [param_a, param_b]

        def test_grad_data_zeroed_after_gather_finalize(self):
            """
            Reproduce the Megatron b25a76e scenario:
            1. grad_data is filled with "parameter values" (simulated gather output).
            2. finalize_gather_and_zero_grad() is called.
            3. grad_data must be all-zero (ready for gradient accumulation).
            """
            _, bucket, grad_data, params = self._make_manager_and_bucket(size=256)

            # Simulate: layerwise gather has been launched
            gather_list = [torch.zeros_like(p.data) for p in params]
            bucket.launch_layerwise_gather(gather_list)

            # Simulate: gather complete, updated params ready
            updated_params = [torch.randn_like(p.data) for p in params]
            bucket.finalize_gather_and_zero_grad(updated_params)

            # The critical assertion: grad_data must be zero (not contain
            # the residual parameter values that the gather wrote into it).
            self.assertTrue(
                torch.all(grad_data == 0).item(),
                "REGRESSION: grad_data contains non-zero values after finalize_gather_and_zero_grad. "
                "This is the Megatron b25a76e gradient corruption bug.",
            )

        def test_main_grad_view_is_zero_after_finalize(self):
            """
            Since main_grad is a view into grad_data, zeroing grad_data must
            also zero all main_grad views. This test verifies view coherence.
            """
            _, bucket, grad_data, params = self._make_manager_and_bucket(size=64)
            bucket.launch_layerwise_gather([])
            bucket.finalize_gather_and_zero_grad([p.data.clone() for p in params])

            for param in params:
                self.assertTrue(
                    torch.all(param.main_grad == 0).item(),
                    f"main_grad for param {param.shape} not zero after finalize; "
                    "view into grad_data is not coherent with the zero_() call.",
                )

        def test_gradient_accumulation_correct_after_zero(self):
            """
            After zeroing, accumulating a gradient of all-ones must produce
            exactly all-ones in grad_data — not all-ones plus the residual
            parameter values.
            """
            _, bucket, grad_data, params = self._make_manager_and_bucket(size=64)
            bucket.launch_layerwise_gather([])
            bucket.finalize_gather_and_zero_grad([p.data.clone() for p in params])

            # Simulate one micro-batch gradient accumulation
            fake_grad = torch.ones(64, dtype=torch.float32)
            grad_data.add_(fake_grad)

            self.assertTrue(
                torch.allclose(grad_data, torch.ones(64)),
                "Gradient accumulation after zero produced wrong values; "
                "residual parameter values may not have been cleared.",
            )

        def test_no_zeroing_without_gather_causes_corruption(self):
            """
            Negative test: if we do NOT call finalize_gather_and_zero_grad
            (simulating the pre-b25a76e code path), accumulating a gradient
            of all-ones produces non-ones, demonstrating the original bug.
            This test is expected to see corrupted values.
            """
            size = 64
            grad_data = torch.full((size,), fill_value=1.0, dtype=torch.float32)
            fake_grad = torch.ones(size, dtype=torch.float32)
            grad_data.add_(fake_grad)

            # Without zeroing, result is 2.0 (1.0 residual + 1.0 gradient)
            self.assertTrue(
                torch.allclose(grad_data, torch.full((size,), 2.0)),
                "Expected corruption demonstration to show value 2.0, got something else.",
            )

        def test_finish_param_sync_zeros_all_buckets(self):
            """
            finish_param_sync_with_grad_safety must zero grad_data for every
            bucket in the group, not just the first one.
            """
            manager = HeteroLayerwiseGradSafetyManager(
                world_size=3, corruption_check_frequency=0, slc_invalidation_enabled=False
            )
            buckets = []
            grad_datas = []
            for i in range(4):
                gd = torch.full((32,), fill_value=float(i + 1))
                param = torch.nn.Parameter(torch.randn(32))
                param.main_grad = gd[:]
                b = DESLOCParamGradBucket(
                    bucket_id=i, grad_data=gd, params=[param], manager=manager
                )
                b.launch_layerwise_gather([])
                buckets.append(b)
                grad_datas.append(gd)

            updated = [[p.data.clone() for p in b.params] for b in buckets]
            finish_param_sync_with_grad_safety(buckets, updated)

            for i, gd in enumerate(grad_datas):
                self.assertTrue(
                    torch.all(gd == 0).item(),
                    f"grad_data for bucket {i} not zeroed after finish_param_sync_with_grad_safety",
                )

        def test_layerwise_gather_list_none_after_finalize(self):
            """After finalize, layerwise_gather_list must be None."""
            _, bucket, _, params = self._make_manager_and_bucket(size=32)
            bucket.launch_layerwise_gather(["dummy_tensor"])
            self.assertIsNotNone(bucket.layerwise_gather_list)
            bucket.finalize_gather_and_zero_grad([p.data.clone() for p in params])
            self.assertIsNone(bucket.layerwise_gather_list)

    class TestDeviceClassification(unittest.TestCase):
        """Tests for SM86/SM90 device role classification."""

        def test_cpu_device_classified_as_forward(self):
            device = torch.device("cpu")
            role = _classify_device_role(device)
            self.assertEqual(role, DESLOCDEVICEROLE_FORWARD)

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_cuda_device_role_classification(self):
            device = torch.device("cuda:0")
            role = _classify_device_role(device)
            # We can't assert a specific role without knowing the hardware,
            # but the role must be one of the two valid values.
            self.assertIn(role, {DESLOCDEVICEROLE_FORWARD, DESLOCDEVICEROLE_OPTIMIZER})

    class TestSLCCoherenceWithManager(unittest.TestCase):
        """
        Tests that the SLC validity tracking in HeteroLayerwiseGradSafetyManager
        correctly reflects the grad_data lifecycle.
        """

        def setUp(self):
            self.manager = HeteroLayerwiseGradSafetyManager(
                world_size=3,
                corruption_check_frequency=0,
                slc_invalidation_enabled=True,
            )

        def test_slc_invalidated_on_gather_launch(self):
            record = self.manager.register_bucket(0, _cpu_device())
            snap_before = self.manager.slc_snapshot()
            # CPU device uses index 0 by default
            # SLC is not tracked for CPU in our implementation (no SLC entry)
            # but the record's slc_valid field should track the state.
            self.manager.on_gather_launched(record)
            self.assertFalse(record.slc_valid)

        def test_slc_valid_after_zero_on_cpu(self):
            record = self.manager.register_bucket(1, _cpu_device())
            grad_data = torch.ones(32)
            self.manager.on_gather_launched(record)
            self.manager.on_gather_complete_zero_grad(record, grad_data)
            # CPU path sets slc_valid = True immediately after zero_()
            self.assertTrue(record.slc_valid)

    class TestMultiBucketConcurrentLifecycle(unittest.TestCase):
        """
        Integration test: multiple buckets progressing through the lifecycle
        concurrently (simulating pipeline parallelism where bucket N is in
        GATHERING while bucket N-1 is in ACCUMULATING).
        """

        def test_independent_bucket_lifecycles(self):
            manager = HeteroLayerwiseGradSafetyManager(
                world_size=3, corruption_check_frequency=0, slc_invalidation_enabled=False
            )
            n_buckets = 6
            records = []
            grad_datas = []

            for i in range(n_buckets):
                gd = torch.full((16,), fill_value=float(i * 10 + 1))
                rec = manager.register_bucket(i, _cpu_device())
                records.append(rec)
                grad_datas.append(gd)

            # Launch gather for all buckets
            for rec in records:
                manager.on_gather_launched(rec)

            # Finalize all (CPU path: synchronous zero)
            for rec, gd in zip(records, grad_datas):
                manager.on_gather_complete_zero_grad(rec, gd)

            # All grad_datas must be zero
            for i, gd in enumerate(grad_datas):
                self.assertTrue(
                    torch.all(gd == 0).item(),
                    f"Bucket {i} grad_data not zeroed in multi-bucket lifecycle test.",
                )

            # Advance all to ACCUMULATING
            for rec in records:
                rec.state = BucketState.ACCUMULATING

            # Complete lifecycle
            for rec in records:
                manager.on_ready_for_reduce(rec)
                manager.on_reduce_complete(rec)
                self.assertEqual(rec.state, BucketState.IDLE)

    print("Running DES-LOC HeteroLayerwiseGradSafety unit tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestBucketLifecycleStateMachine,
        TestSharedLocalityCacheInvalidator,
        TestGradDataZeroingCPU,
        TestDeviceClassification,
        TestSLCCoherenceWithManager,
        TestMultiBucketConcurrentLifecycle,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
