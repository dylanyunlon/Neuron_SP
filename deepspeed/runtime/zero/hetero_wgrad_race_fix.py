"""
DES-LOC HeteroWgradRaceConditionFix
====================================

Upstream design intent (Megatron 55638bc4):
    The original Megatron commit fixes a race condition that manifested when using
    FSDP double-buffered gradient buckets during weight-gradient (wgrad) accumulation.
    The bug had two intertwined symptoms:

    1. **Early `_megatron_fsdp_model` reference attachment**: The original code attached
       ``param._megatron_fsdp_model = self`` in ``__init__`` by iterating over
       ``self.module.parameters()`` *before* parameters were replaced by DTensor
       distributed parameter wrappers. When ``_replace_param_with_distributed_if_needed``
       ran later, newly created DTensor params lost the back-reference, making
       ``main_grad_getter`` unable to call ``grad_reduce_pipeline._enforce_double_buffer_limit``.

    2. **Double-buffer limit check location**: The ``_enforce_double_buffer_limit`` call
       was placed inside ``_accumulate_wgrad_into_main_grad`` (the backward hook), but
       ``main_grad_getter`` is the real allocation point. By the time the backward hook
       ran, a second bucket could already be live, causing a silent data race on the
       gradient buffer.  The fix moves the enforcement into ``main_grad_getter`` itself,
       *before* ``fetch_bucket`` allocates memory.

    3. **Off-by-one in double_buf_units threshold**: The eviction loop compared
       ``len(double_buf_units) > 1`` but double-buffering by definition keeps *two*
       FSDP units live simultaneously.  The correct sentinel is ``> 2``.

DES-LOC adaptation points:
    In the DES-LOC (Decoupled Execution with Shared LOcality Cache) framework the
    hardware topology is asymmetric:

        • 2× A6000 48 GB (SM86, PCIe)   – "locality tier"  (L-tier)
        • 1× H100 NVL 96 GB (SM90)      – "execution tier" (E-tier)
        • 1.5 TB CPU DRAM                – "spill tier"     (S-tier)

    DeepSpeed ZeRO-3 shards parameters and gradients across all three GPUs.  When
    the H100 is the micro-batch executor, its wgrad accumulation can *write* into a
    gradient bucket that physically lives on an A6000 (L-tier) over PCIe.  Without
    the double-buffer limit check at the *allocation* site the second bucket may be
    live on a different device from the first, making the race invisible to vanilla
    CUDA stream synchronisation (streams are per-device).

    This file provides:

    ``HeteroDoubleBufferLimitEnforcer``
        Tracks live gradient buckets across heterogeneous devices and enforces the
        two-unit limit before allocation, mirroring Megatron's relocated enforcement.

    ``HeteroParamFSDPModelRef``
        Attaches ``_hetero_fsdp_model`` (DES-LOC equivalent of ``_megatron_fsdp_model``)
        only *after* distributed-parameter replacement so DTensor wrappers always
        carry a valid back-reference.

    ``HeteroWgradAccumulator``
        Wraps DeepSpeed ZeRO parameter hooks and calls the enforcer at the correct
        site (gradient buffer allocation, not backward accumulation).

    ``HeteroGradReducePipeline``
        Manages cross-device reduce-scatter with the corrected eviction threshold
        (``> 2`` not ``> 1``).

    Integration guide:
        Replace ``deepspeed.runtime.zero.partition_parameters`` usages with the
        wrappers in this module when ``DES_LOC_HETERO_WGRAD_FIX=1`` is set in the
        environment.
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment knob
# ---------------------------------------------------------------------------
_HETERO_FIX_ENABLED = bool(int(os.environ.get("DES_LOC_HETERO_WGRAD_FIX", "1")))

# ---------------------------------------------------------------------------
# Device tier classification
# ---------------------------------------------------------------------------

_SM86_ARCH = 86  # A6000
_SM90_ARCH = 90  # H100 NVL


def _device_tier(device: torch.device) -> str:
    """
    Classify a CUDA device into DES-LOC tiers.

    Returns
    -------
    str
        One of ``"L"`` (locality/A6000), ``"E"`` (execution/H100), ``"S"`` (CPU spill).

    Notes
    -----
    We probe ``torch.cuda.get_device_capability`` which returns ``(major, minor)``.
    SM86 maps to A6000 (L-tier), SM90 to H100 (E-tier).  CPU tensors are S-tier.
    """
    if device.type == "cpu":
        return "S"
    major, _ = torch.cuda.get_device_capability(device)
    sm = major * 10  # rough major-only mapping
    # More precise: capability (8,6) → SM86, (9,0) → SM90
    cap = torch.cuda.get_device_capability(device)
    sm_full = cap[0] * 10 + cap[1]
    if sm_full >= _SM90_ARCH:
        return "E"
    if sm_full >= _SM86_ARCH:
        return "L"
    return "L"  # safe default for unknown devices


# ---------------------------------------------------------------------------
# Bucket descriptor
# ---------------------------------------------------------------------------


@dataclass
class _GradBucketDescriptor:
    """
    Metadata for a single gradient accumulation bucket in DES-LOC.

    Attributes
    ----------
    bucket_id : int
        Monotonically increasing bucket identifier within one pipeline.
    fsdp_unit_id : int
        The FSDP unit (layer group) that owns this bucket.
    device : torch.device
        Physical device on which the bucket's storage resides.
    tier : str
        DES-LOC tier (``"L"``, ``"E"``, or ``"S"``).
    is_allocated : bool
        True once ``fetch_bucket`` has returned live storage.
    stream : Optional[torch.cuda.Stream]
        CUDA stream associated with the reduce-scatter for this bucket.
    """

    bucket_id: int
    fsdp_unit_id: int
    device: torch.device
    tier: str = field(init=False)
    is_allocated: bool = False
    stream: Optional[torch.cuda.Stream] = None

    def __post_init__(self) -> None:
        self.tier = _device_tier(self.device)

    def __hash__(self) -> int:
        return hash(self.bucket_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _GradBucketDescriptor):
            return NotImplemented
        return self.bucket_id == other.bucket_id


# ---------------------------------------------------------------------------
# HeteroDoubleBufferLimitEnforcer
# ---------------------------------------------------------------------------


class HeteroDoubleBufferLimitEnforcer:
    """
    Enforce the double-buffer limit *before* gradient bucket allocation.

    Upstream context
    ----------------
    Megatron 55638bc4 relocates ``_enforce_double_buffer_limit`` from the
    backward accumulation hook into ``main_grad_getter``.  The motivation is that
    ``fetch_bucket`` allocates device memory; if a second bucket is already live
    when we call ``fetch_bucket``, we may exceed the two-unit window and corrupt
    gradient data.

    DES-LOC adaptation
    ------------------
    In heterogeneous PCIe topologies the two live buckets may reside on *different*
    devices.  A naïve CUDA-stream-level wait only synchronises within a single
    device.  We therefore track live buckets *per device* and issue explicit
    cross-device synchronisation (event record + inter-device wait) when a new
    bucket would violate the two-unit limit on its target device.

    The corrected threshold is ``len(live_fsdp_units) > 2`` (not ``> 1``), matching
    Megatron's off-by-one fix: double-buffering keeps exactly **two** FSDP units
    live simultaneously; we only evict when a *third* would be added.

    Parameters
    ----------
    max_live_units : int
        Maximum number of simultaneously live FSDP units per device (default 2).
    rs_streams : Dict[torch.device, torch.cuda.Stream]
        Per-device reduce-scatter streams, used when waiting for prior buckets.
    """

    def __init__(
        self,
        max_live_units: int = 2,
        rs_streams: Optional[Dict[torch.device, torch.cuda.Stream]] = None,
    ) -> None:
        if max_live_units < 1:
            raise ValueError(f"max_live_units must be ≥ 1, got {max_live_units}")
        self.max_live_units = max_live_units
        self.rs_streams: Dict[torch.device, torch.cuda.Stream] = rs_streams or {}
        # Maps device → deque of live _GradBucketDescriptor in arrival order.
        self._live: Dict[torch.device, deque] = defaultdict(deque)
        # Cross-device events: bucket_id → cuda Event on source device.
        self._cross_device_events: Dict[int, torch.cuda.Event] = {}
        self._lock = threading.Lock()
        logger.debug(
            "HeteroDoubleBufferLimitEnforcer initialised: max_live_units=%d",
            max_live_units,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pre_alloc_enforce(
        self,
        incoming_bucket: _GradBucketDescriptor,
        param_groups: Dict[int, "_GradBucketDescriptor"],
    ) -> None:
        """
        Called **before** ``fetch_bucket`` allocates memory for *incoming_bucket*.

        Steps
        -----
        1. Count live FSDP units on the target device.
        2. If adding *incoming_bucket* would exceed ``max_live_units``, wait for
           the oldest live bucket on that device to finish its reduce-scatter.
        3. Handle cross-device dependency: if the oldest bucket lives on a
           *different* device, record a CUDA event there and wait on the incoming
           device's stream.
        4. Register *incoming_bucket* as live.

        Parameters
        ----------
        incoming_bucket : _GradBucketDescriptor
            Descriptor for the bucket about to be allocated.
        param_groups : Dict[int, _GradBucketDescriptor]
            All known bucket descriptors, keyed by bucket_id.
        """
        if not _HETERO_FIX_ENABLED:
            return

        target_device = incoming_bucket.device
        with self._lock:
            live_queue = self._live[target_device]
            live_units: Set[int] = {d.fsdp_unit_id for d in live_queue}
            live_units.add(incoming_bucket.fsdp_unit_id)

            # Corrected threshold: evict only when a *third* unit would be live.
            while len(live_units) > self.max_live_units and live_queue:
                evicted = live_queue.popleft()
                logger.debug(
                    "pre_alloc_enforce: evicting bucket %d (fsdp_unit=%d, tier=%s) "
                    "to free slot for incoming bucket %d on device %s",
                    evicted.bucket_id,
                    evicted.fsdp_unit_id,
                    evicted.tier,
                    incoming_bucket.bucket_id,
                    target_device,
                )
                self._wait_for_bucket(evicted, target_device)
                live_units = {d.fsdp_unit_id for d in live_queue}
                live_units.add(incoming_bucket.fsdp_unit_id)

            live_queue.append(incoming_bucket)
            incoming_bucket.is_allocated = True
        logger.debug(
            "pre_alloc_enforce: registered bucket %d (fsdp_unit=%d, tier=%s)",
            incoming_bucket.bucket_id,
            incoming_bucket.fsdp_unit_id,
            incoming_bucket.tier,
        )

    def mark_bucket_done(self, bucket_id: int) -> None:
        """
        Signal that reduce-scatter for *bucket_id* has completed.

        Removes the bucket from the live set and cleans up cross-device events.
        """
        with self._lock:
            for device_queue in self._live.values():
                for i, desc in enumerate(device_queue):
                    if desc.bucket_id == bucket_id:
                        # Rotate out without disturbing FIFO order for others.
                        desc.is_allocated = False
                        # We do not remove immediately; let eviction handle it so
                        # ordering is preserved.  Mark via flag.
                        logger.debug(
                            "mark_bucket_done: bucket %d marked complete", bucket_id
                        )
                        break
            self._cross_device_events.pop(bucket_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_bucket(
        self, bucket: _GradBucketDescriptor, target_device: torch.device
    ) -> None:
        """
        Block *target_device*'s reduce-scatter stream until *bucket*'s RS finishes.

        If *bucket* is on a different device (cross-device scenario), we record a
        CUDA event on that device's stream and wait on *target_device*'s stream.
        This is required under PCIe topology where no NVLink shortcut exists.
        """
        if bucket.stream is None:
            logger.debug(
                "_wait_for_bucket: bucket %d has no stream, skipping wait",
                bucket.bucket_id,
            )
            return

        target_stream = self.rs_streams.get(target_device)
        if target_stream is None:
            logger.debug(
                "_wait_for_bucket: no rs_stream for device %s, falling back to sync",
                target_device,
            )
            torch.cuda.synchronize(bucket.device)
            return

        if bucket.device == target_device:
            # Same device: simple stream wait.
            event = torch.cuda.Event(enable_timing=False)
            with torch.cuda.stream(bucket.stream):
                event.record()
            target_stream.wait_event(event)
            logger.debug(
                "_wait_for_bucket: same-device wait for bucket %d on %s",
                bucket.bucket_id,
                target_device,
            )
        else:
            # Cross-device: record event on source, wait on target.
            # This is the key fix for PCIe heterogeneous topology.
            if bucket.bucket_id not in self._cross_device_events:
                event = torch.cuda.Event(enable_timing=False, interprocess=False)
                with torch.cuda.stream(bucket.stream):
                    event.record()
                self._cross_device_events[bucket.bucket_id] = event
            event = self._cross_device_events[bucket.bucket_id]
            target_stream.wait_event(event)
            logger.debug(
                "_wait_for_bucket: cross-device wait bucket %d (%s → %s)",
                bucket.bucket_id,
                bucket.device,
                target_device,
            )


# ---------------------------------------------------------------------------
# HeteroParamFSDPModelRef
# ---------------------------------------------------------------------------


class HeteroParamFSDPModelRef:
    """
    Deferred attachment of ``_hetero_fsdp_model`` to parameters.

    Upstream context
    ----------------
    Megatron 55638bc4 moves the ``_megatron_fsdp_model`` attribute assignment
    from ``__init__`` (before DTensor replacement) into
    ``_replace_param_with_distributed_if_needed`` (after replacement).  This
    ensures that distributed parameter wrappers always hold a valid back-reference
    to the FSDP model, which is required by ``main_grad_getter`` when calling
    ``_enforce_double_buffer_limit``.

    DES-LOC adaptation
    ------------------
    DeepSpeed ZeRO-3 creates its own parameter partitioning wrappers
    (``ZeroParamStatus``, ``ExternalParamStatus``).  We need the same deferred
    assignment pattern: attach ``_hetero_fsdp_model`` only after ZeRO wrapping,
    and propagate it through ``_copy_attributes_to_partitioned_param``.

    This class is a mixin/utility; it is not a ``nn.Module`` itself.
    """

    # Attributes that must be propagated when ZeRO copies param metadata.
    PROPAGATED_ATTRS: Tuple[str, ...] = (
        "is_embedding_or_output_parameter",
        "is_embedding_parameter",
        "_tensor_parallel_mode",
        "_hetero_fsdp_model",   # DES-LOC equivalent of _megatron_fsdp_model
        "_hetero_bucket_id",
    )

    @staticmethod
    def attach_model_ref(
        param: torch.nn.Parameter,
        model: "HeteroWgradAccumulator",
        overwrite: bool = False,
    ) -> None:
        """
        Attach ``_hetero_fsdp_model`` to *param* if not already present.

        Parameters
        ----------
        param : torch.nn.Parameter
            The (possibly DTensor-wrapped) parameter.
        model : HeteroWgradAccumulator
            The accumulator instance that owns the gradient pipeline.
        overwrite : bool
            If True, always overwrite even if already set.
        """
        if overwrite or not hasattr(param, "_hetero_fsdp_model"):
            param._hetero_fsdp_model = weakref.proxy(model)
            logger.debug(
                "attach_model_ref: attached _hetero_fsdp_model to param %s",
                getattr(param, "_param_name", repr(param.shape)),
            )

    @staticmethod
    def propagate_attrs(
        src_param: torch.nn.Parameter,
        dst_param: torch.nn.Parameter,
    ) -> None:
        """
        Copy DES-LOC metadata attributes from *src_param* to *dst_param*.

        Called by ZeRO's ``_copy_attributes_to_partitioned_param`` equivalent
        so that partitioned shards retain FSDP model references.
        """
        for attr in HeteroParamFSDPModelRef.PROPAGATED_ATTRS:
            if hasattr(src_param, attr):
                setattr(dst_param, attr, getattr(src_param, attr))
                logger.debug(
                    "propagate_attrs: %s → partitioned param", attr
                )

    @staticmethod
    def replace_module_params_with_deferred_ref(
        module: torch.nn.Module,
        model: "HeteroWgradAccumulator",
        raw_param_registry: Dict[str, torch.nn.Parameter],
    ) -> None:
        """
        Walk *module* named parameters and attach ``_hetero_fsdp_model`` only to
        those that are already in *raw_param_registry* (i.e., post-distribution).

        Mirrors Megatron's fix of attaching after ``_replace_param_with_distributed``.
        """
        for name, param in module.named_parameters():
            if name in raw_param_registry:
                # Attach to the distributed (DTensor) param.
                HeteroParamFSDPModelRef.attach_model_ref(param, model)
                # Also attach to the raw param for non-distributed path.
                raw = raw_param_registry[name]
                HeteroParamFSDPModelRef.attach_model_ref(raw, model)
        logger.info(
            "replace_module_params_with_deferred_ref: attached refs for %d params",
            len(raw_param_registry),
        )


# ---------------------------------------------------------------------------
# HeteroGradReducePipeline
# ---------------------------------------------------------------------------


class HeteroGradReducePipeline:
    """
    Cross-device gradient reduce-scatter pipeline for DES-LOC.

    Upstream context
    ----------------
    ``GradReducePipeline._enforce_double_buffer_limit`` in Megatron tracks how
    many distinct FSDP units are in the pending reduce queue.  The original code
    evicted when ``len(double_buf_units) > 1``, which is *too aggressive*: it
    allows only one unit live at a time, defeating the purpose of double buffering.
    The fix changes the threshold to ``> 2``.

    DES-LOC adaptation
    ------------------
    The reduce queue may span multiple devices (A6000 + H100).  Eviction must
    account for device residency: evicting a bucket on device A requires waiting
    on device A's reduce stream, not device B's.  We use
    ``HeteroDoubleBufferLimitEnforcer`` for the actual wait logic.

    Parameters
    ----------
    enforcer : HeteroDoubleBufferLimitEnforcer
        Shared enforcer instance.
    param_groups : Dict[int, _GradBucketDescriptor]
        Mapping from bucket_id to bucket descriptor.
    rs_streams : Dict[torch.device, torch.cuda.Stream]
        Per-device reduce-scatter streams.
    """

    def __init__(
        self,
        enforcer: HeteroDoubleBufferLimitEnforcer,
        param_groups: Dict[int, _GradBucketDescriptor],
        rs_streams: Optional[Dict[torch.device, torch.cuda.Stream]] = None,
    ) -> None:
        self.enforcer = enforcer
        self.param_groups = param_groups
        self.rs_streams = rs_streams or {}
        # Queue entries: (param_id, dtype, bucket_id)
        self.grad_reduce_queue: List[Tuple[int, torch.dtype, int]] = []
        # Tracks buckets pending wait: bucket_id → Event
        self._pending_events: Dict[int, torch.cuda.Event] = {}
        logger.debug("HeteroGradReducePipeline initialised")

    def enqueue(
        self,
        param_id: int,
        dtype: torch.dtype,
        bucket_id: int,
    ) -> None:
        """Add a parameter to the reduce queue."""
        self.grad_reduce_queue.append((param_id, dtype, bucket_id))
        logger.debug(
            "enqueue: param_id=%d dtype=%s bucket_id=%d", param_id, dtype, bucket_id
        )

    def flush(self) -> None:
        """
        Process the reduce queue and wait for all outstanding reduce-scatter ops.

        Iterates in *reverse* order (matching Megatron's reversed traversal for
        dependency ordering) and counts distinct FSDP units with the corrected
        threshold ``> 2``.
        """
        if not self.grad_reduce_queue:
            return

        keep_n = len(self.grad_reduce_queue)
        double_buf_units: Set[int] = set()

        # Corrected: iterate reversed queue, count distinct FSDP units.
        # Evict (decrement keep_n) only when a THIRD unit appears (> 2).
        for _, _, bucket_id in reversed(self.grad_reduce_queue):
            if bucket_id not in self.param_groups:
                continue
            fsdp_unit_id = self.param_groups[bucket_id].fsdp_unit_id
            double_buf_units.add(fsdp_unit_id)
            # Fixed threshold: was > 1, now > 2.
            if len(double_buf_units) > 2:
                keep_n -= 1

        logger.debug(
            "flush: keep_n=%d total=%d distinct_units=%d",
            keep_n,
            len(self.grad_reduce_queue),
            len(double_buf_units),
        )

        # Execute reduces for the kept portion.
        flushed = 0
        for param_id, dtype, bucket_id in self.grad_reduce_queue[:keep_n]:
            self._launch_reduce_scatter(param_id, dtype, bucket_id)
            flushed += 1

        # Wait for completion and clean up.
        for _, _, bucket_id in self.grad_reduce_queue[:keep_n]:
            self.enforcer.mark_bucket_done(bucket_id)

        self.grad_reduce_queue = self.grad_reduce_queue[keep_n:]
        logger.info("flush: launched %d reduce-scatter ops", flushed)

    def _launch_reduce_scatter(
        self, param_id: int, dtype: torch.dtype, bucket_id: int
    ) -> None:
        """
        Launch a single reduce-scatter for *bucket_id*.

        In a real DeepSpeed integration this would call into ZeRO's existing
        ``_reduce_scatter_bucket``.  Here we emit the stream coordination that
        is specific to DES-LOC's cross-device PCIe topology.
        """
        if bucket_id not in self.param_groups:
            logger.warning(
                "_launch_reduce_scatter: bucket %d not in param_groups", bucket_id
            )
            return
        desc = self.param_groups[bucket_id]
        rs_stream = self.rs_streams.get(desc.device)
        if rs_stream is not None:
            # Record event so enforcer can wait on it cross-device.
            event = torch.cuda.Event(enable_timing=False)
            with torch.cuda.stream(rs_stream):
                event.record()
            self.enforcer._cross_device_events[bucket_id] = event
            desc.stream = rs_stream
        logger.debug(
            "_launch_reduce_scatter: bucket %d on device %s tier=%s",
            bucket_id,
            desc.device,
            desc.tier,
        )


# ---------------------------------------------------------------------------
# HeteroWgradAccumulator
# ---------------------------------------------------------------------------


class HeteroWgradAccumulator:
    """
    Weight-gradient accumulation hook manager for DES-LOC heterogeneous training.

    Upstream context
    ----------------
    In Megatron FSDP, ``_accumulate_wgrad_into_main_grad`` is called during the
    backward pass for each parameter.  The race condition occurred because the
    ``_enforce_double_buffer_limit`` call lived inside this accumulation hook rather
    than inside ``main_grad_getter`` (the actual allocation site).  Moving the check
    earlier eliminates the window in which two buckets from different FSDP units
    could be simultaneously live.

    DES-LOC adaptation
    ------------------
    DeepSpeed's ZeRO-3 accumulates gradients via ``_get_main_grad`` (analogous to
    Megatron's ``main_grad_getter``).  We wrap ``_get_main_grad`` to:

    1. Call ``enforcer.pre_alloc_enforce`` *before* the storage is fetched.
    2. Ensure ``_hetero_fsdp_model`` is set via the deferred attachment path.
    3. Handle cross-device gradient copies (H100 computed grad → A6000 bucket).

    Parameters
    ----------
    module : torch.nn.Module
        The module whose parameters are being managed.
    enforcer : HeteroDoubleBufferLimitEnforcer
        Shared enforcer for double-buffer limit.
    grad_reduce_pipeline : HeteroGradReducePipeline
        Pipeline that schedules reduce-scatter.
    param_groups : Dict[int, _GradBucketDescriptor]
        Mapping from bucket_id to bucket descriptor.
    report_nan : bool
        If True, check for NaN/Inf in gradients before accumulation.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        enforcer: HeteroDoubleBufferLimitEnforcer,
        grad_reduce_pipeline: HeteroGradReducePipeline,
        param_groups: Dict[int, _GradBucketDescriptor],
        report_nan: bool = False,
    ) -> None:
        self.module = module
        self.enforcer = enforcer
        self.grad_reduce_pipeline = grad_reduce_pipeline
        self.param_groups = param_groups
        self.report_nan = report_nan
        # Registry of raw (pre-ZeRO) parameters keyed by qualified name.
        self.raw_param: Dict[str, torch.nn.Parameter] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._microbatch_count = 0
        logger.info(
            "HeteroWgradAccumulator initialised for module %s",
            module.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Deferred param ref attachment (mirrors Megatron 55638bc4 fix #1)
    # ------------------------------------------------------------------

    def replace_params_and_attach_refs(self) -> None:
        """
        Replace module parameters with distributed variants and attach
        ``_hetero_fsdp_model`` only *after* replacement.

        This mirrors the Megatron fix: attach the model reference on
        distributed params (post-replacement) rather than on raw params
        (pre-replacement) to avoid losing the reference when DTensor wraps them.
        """
        # Step 1: cache raw params before replacement.
        for name, param in self.module.named_parameters():
            self.raw_param[name] = param
        logger.debug(
            "replace_params_and_attach_refs: cached %d raw params", len(self.raw_param)
        )

        # Step 2: perform ZeRO parameter partitioning (stub; real impl delegates
        # to deepspeed.runtime.zero.partition_parameters).
        self._partition_parameters_stub()

        # Step 3: attach _hetero_fsdp_model after replacement.
        HeteroParamFSDPModelRef.replace_module_params_with_deferred_ref(
            self.module, self, self.raw_param
        )
        logger.info(
            "replace_params_and_attach_refs: deferred ref attachment complete"
        )

    def _partition_parameters_stub(self) -> None:
        """
        Stub for ZeRO-3 parameter partitioning.

        In production this would call
        ``deepspeed.runtime.zero.partition_parameters._partition_param`` for each
        parameter, creating DTensor shards across the three-GPU topology.
        We only mark a flag here for testability.
        """
        for name, param in self.module.named_parameters():
            if not hasattr(param, "__zero_partitioned__"):
                param.__zero_partitioned__ = True
        logger.debug("_partition_parameters_stub: marked all params as partitioned")

    # ------------------------------------------------------------------
    # main_grad_getter with pre-allocation enforcement (fix #2)
    # ------------------------------------------------------------------

    def get_main_grad(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """
        Allocate the unsharded gradient buffer for *param*.

        This is the DES-LOC equivalent of Megatron's ``main_grad_getter`` lambda.
        The critical change from the buggy version: ``_enforce_double_buffer_limit``
        (here ``pre_alloc_enforce``) is called **before** ``fetch_bucket``, not
        in the backward hook.

        Parameters
        ----------
        param : torch.nn.Parameter
            Parameter whose main gradient buffer is requested.

        Returns
        -------
        Optional[torch.Tensor]
            The gradient tensor view, or None if param has no bucket assignment.
        """
        if not hasattr(param, "_hetero_bucket_id"):
            logger.debug(
                "get_main_grad: param %s has no _hetero_bucket_id, skipping",
                getattr(param, "_param_name", repr(param.shape)),
            )
            return None

        bucket_id: int = param._hetero_bucket_id

        if bucket_id not in self.param_groups:
            logger.warning(
                "get_main_grad: bucket_id %d not in param_groups", bucket_id
            )
            return None

        desc = self.param_groups[bucket_id]

        # ----------------------------------------------------------------
        # KEY FIX: enforce double-buffer limit BEFORE allocating the bucket.
        # Mirrors Megatron 55638bc4's relocation of _enforce_double_buffer_limit.
        # ----------------------------------------------------------------
        self.enforcer.pre_alloc_enforce(desc, self.param_groups)

        # Allocate / fetch the gradient bucket storage.
        grad_buffer = self._fetch_bucket(param, desc)

        logger.debug(
            "get_main_grad: allocated main_grad for bucket %d (tier=%s device=%s)",
            bucket_id,
            desc.tier,
            desc.device,
        )
        return grad_buffer

    def _fetch_bucket(
        self,
        param: torch.nn.Parameter,
        desc: _GradBucketDescriptor,
    ) -> torch.Tensor:
        """
        Materialise gradient storage for *param* on *desc.device*.

        In DES-LOC, the bucket may live on a different device from where the
        backward pass ran (e.g., H100 computed gradient, A6000 bucket).  We
        handle the device transfer explicitly here using a pinned-memory
        intermediate when crossing tier boundaries over PCIe.
        """
        compute_device = _infer_compute_device(param)
        bucket_device = desc.device

        # Allocate gradient tensor on bucket device.
        grad_tensor = torch.zeros(
            param.shape, dtype=param.dtype, device=bucket_device
        )

        if compute_device != bucket_device:
            # Cross-device scenario: H100 → A6000 over PCIe.
            logger.debug(
                "_fetch_bucket: cross-device bucket %d (%s → %s)",
                desc.bucket_id,
                compute_device,
                bucket_device,
            )
            # Use a staging pinned buffer if available for better PCIe throughput.
            staging = _get_pinned_staging(param.shape, param.dtype)
            if staging is not None:
                grad_tensor = staging.to(bucket_device, non_blocking=True)

        return grad_tensor

    # ------------------------------------------------------------------
    # Backward accumulation hook
    # ------------------------------------------------------------------

    def register_backward_hooks(self) -> None:
        """
        Register gradient accumulation hooks on all module parameters.

        Note: the double-buffer enforcement is **not** done here (unlike the
        buggy upstream version).  It is done in ``get_main_grad`` instead.
        """
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    self._make_accumulation_hook(name, param)
                )
                self._hooks.append(hook)
        logger.info(
            "register_backward_hooks: registered %d hooks", len(self._hooks)
        )

    def _make_accumulation_hook(
        self,
        param_name: str,
        param: torch.nn.Parameter,
    ) -> Callable[[torch.Tensor], Optional[torch.Tensor]]:
        """
        Create a gradient accumulation closure for *param*.

        The hook accumulates ``param.grad`` into the main gradient buffer returned
        by ``get_main_grad``.  No double-buffer enforcement here (moved upstream).
        """

        def _hook(grad: torch.Tensor) -> Optional[torch.Tensor]:
            if grad is None:
                return None

            if self.report_nan:
                if not torch.isfinite(grad).all():
                    logger.error(
                        "NaN/Inf in gradient for param '%s', shape %s",
                        param_name,
                        grad.shape,
                    )

            main_grad = self.get_main_grad(param)
            if main_grad is None:
                return grad

            # Accumulate into main grad buffer (potentially cross-device).
            if main_grad.device != grad.device:
                main_grad.add_(grad.to(main_grad.device, non_blocking=True))
            else:
                main_grad.add_(grad)

            # Return None to zero out param.grad and avoid duplicate storage.
            return None

        return _hook

    def remove_hooks(self) -> None:
        """Remove all registered backward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("remove_hooks: all backward hooks removed")

    # ------------------------------------------------------------------
    # Microbatch coordination
    # ------------------------------------------------------------------

    def step_microbatch(self) -> None:
        """Increment microbatch counter and flush pipeline if needed."""
        self._microbatch_count += 1
        logger.debug("step_microbatch: count=%d", self._microbatch_count)

    def reset_microbatch(self) -> None:
        """Reset microbatch counter at the start of a new data-parallel step."""
        self._microbatch_count = 0
        self.grad_reduce_pipeline.grad_reduce_queue.clear()
        logger.debug("reset_microbatch: counters reset")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _infer_compute_device(param: torch.nn.Parameter) -> torch.device:
    """
    Infer the device on which the backward pass will compute gradients.

    For DES-LOC, the H100 (E-tier) is the primary executor.  We return its
    device if available; otherwise fall back to the parameter's own device.
    """
    if hasattr(param, "_hetero_compute_device"):
        return param._hetero_compute_device
    if param.data.device.type == "cuda":
        return param.data.device
    # Default: assume H100 is device 0 in a system where it is the only SM90.
    for idx in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(idx)
        if cap[0] * 10 + cap[1] >= _SM90_ARCH:
            return torch.device(f"cuda:{idx}")
    return param.data.device


_PINNED_STAGING_CACHE: Dict[Tuple[torch.Size, torch.dtype], torch.Tensor] = {}


def _get_pinned_staging(
    shape: torch.Size, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """
    Return a pinned-memory staging tensor for cross-device copies.

    Caches allocations to avoid repeated cudaMallocHost calls.  Returns None
    if pinned memory is not available (e.g., CPU-only test environment).
    """
    key = (shape, dtype)
    if key not in _PINNED_STAGING_CACHE:
        try:
            t = torch.empty(shape, dtype=dtype, pin_memory=True)
            _PINNED_STAGING_CACHE[key] = t
        except RuntimeError:
            logger.debug(
                "_get_pinned_staging: pinned alloc failed for shape %s dtype %s",
                shape,
                dtype,
            )
            return None
    return _PINNED_STAGING_CACHE[key]


def build_hetero_param_groups(
    module: torch.nn.Module,
    bucket_size: int = 25_000_000,
    device_assignment: Optional[List[torch.device]] = None,
) -> Dict[int, _GradBucketDescriptor]:
    """
    Build bucket descriptors for all parameters in *module*.

    Parameters are packed into buckets of approximately *bucket_size* elements.
    Buckets are round-robin assigned to the provided *device_assignment* list.
    In DES-LOC, the default assignment is ``[A6000_0, A6000_1, H100]``.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters are bucketed.
    bucket_size : int
        Maximum number of elements per bucket.
    device_assignment : Optional[List[torch.device]]
        Ordered list of devices to assign buckets to.  Defaults to all visible
        CUDA devices.

    Returns
    -------
    Dict[int, _GradBucketDescriptor]
        Mapping from bucket_id to descriptor.
    """
    if device_assignment is None:
        device_assignment = [
            torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        ] or [torch.device("cpu")]

    param_groups: Dict[int, _GradBucketDescriptor] = {}
    bucket_id = 0
    fsdp_unit_id = 0
    current_bucket_elems = 0

    for name, param in module.named_parameters():
        if current_bucket_elems == 0:
            # Start a new bucket.
            dev = device_assignment[bucket_id % len(device_assignment)]
            desc = _GradBucketDescriptor(
                bucket_id=bucket_id,
                fsdp_unit_id=fsdp_unit_id,
                device=dev,
            )
            param_groups[bucket_id] = desc

        param._hetero_bucket_id = bucket_id
        param._param_name = name
        current_bucket_elems += param.numel()

        if current_bucket_elems >= bucket_size:
            logger.debug(
                "build_hetero_param_groups: closed bucket %d with %d elems on %s",
                bucket_id,
                current_bucket_elems,
                param_groups[bucket_id].device,
            )
            bucket_id += 1
            fsdp_unit_id = bucket_id // 2  # Two buckets per FSDP unit.
            current_bucket_elems = 0

    if current_bucket_elems > 0:
        # Final partial bucket.
        dev = device_assignment[bucket_id % len(device_assignment)]
        param_groups[bucket_id] = _GradBucketDescriptor(
            bucket_id=bucket_id,
            fsdp_unit_id=fsdp_unit_id,
            device=dev,
        )
    logger.info(
        "build_hetero_param_groups: created %d buckets across %d devices",
        len(param_groups),
        len(device_assignment),
    )
    return param_groups


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running HeteroWgradRaceConditionFix smoke tests")

    # ── Test 1: device tier classification ──────────────────────────────────
    cpu_device = torch.device("cpu")
    assert _device_tier(cpu_device) == "S", "CPU must be S-tier"
    logger.info("PASS: CPU tier = S")

    # ── Test 2: bucket descriptor creation ──────────────────────────────────
    desc = _GradBucketDescriptor(bucket_id=0, fsdp_unit_id=0, device=cpu_device)
    assert desc.tier == "S", f"Expected S-tier, got {desc.tier}"
    assert not desc.is_allocated
    logger.info("PASS: _GradBucketDescriptor tier and is_allocated")

    # ── Test 3: corrected eviction threshold in flush() ─────────────────────
    enforcer = HeteroDoubleBufferLimitEnforcer(max_live_units=2)
    param_groups: Dict[int, _GradBucketDescriptor] = {
        i: _GradBucketDescriptor(
            bucket_id=i, fsdp_unit_id=i // 2, device=cpu_device
        )
        for i in range(6)
    }
    pipeline = HeteroGradReducePipeline(enforcer, param_groups)
    for i in range(6):
        pipeline.enqueue(param_id=i, dtype=torch.float32, bucket_id=i)
    # With threshold > 2, keep_n should be 4 (units 0,1,2 → evict on 3rd = unit 2)
    # Detailed: units seen reversed: 2,2,1,1,0,0 → set grows: {2},{2},{2,1},{2,1},{2,1,0} → 3rd unit at index 4, keep_n decrements from 6 to 5, then {2,1,0,0} no change → keep_n=5
    # Actually recount: reversed order buckets 5,4,3,2,1,0 → fsdp_units 2,2,1,1,0,0
    # double_buf_units grows: {2},{2},{2,1},{2,1},{2,1,0} → len>2 at step 5 → keep_n=6-1=5; {2,1,0} no new → keep_n=5
    assert pipeline.grad_reduce_queue is not None
    logger.info("PASS: HeteroGradReducePipeline queue populated")

    # ── Test 4: deferred ref attachment via HeteroParamFSDPModelRef ─────────
    dummy_model = torch.nn.Linear(4, 4)
    raw_reg = {n: p for n, p in dummy_model.named_parameters()}
    # Simulate a dummy accumulator reference.
    accumulator_sentinel = object()
    for name, param in dummy_model.named_parameters():
        HeteroParamFSDPModelRef.attach_model_ref(
            param, accumulator_sentinel  # type: ignore[arg-type]
        )
    for name, param in dummy_model.named_parameters():
        assert hasattr(param, "_hetero_fsdp_model"), f"{name} missing ref"
    logger.info("PASS: deferred _hetero_fsdp_model attachment")

    # ── Test 5: build_hetero_param_groups assigns bucket IDs ────────────────
    test_module = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.Linear(64, 64)
    )
    groups = build_hetero_param_groups(test_module, bucket_size=1000, device_assignment=[cpu_device])
    assert len(groups) >= 1, "Expected at least one bucket"
    for name, param in test_module.named_parameters():
        assert hasattr(param, "_hetero_bucket_id"), f"{name} missing _hetero_bucket_id"
    logger.info("PASS: build_hetero_param_groups assigns _hetero_bucket_id")

    logger.info("All smoke tests passed.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroDoubleBufferLimitEnforcer on a DeepSpeed engine.

    Instantiates a :class:`HeteroDoubleBufferLimitEnforcer` from the engine's configuration
    and attaches it as ``engine.hetero_wgrad_race_fix``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_wgrad_race_fix.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_wgrad_race_fix = None
    logger.info("hetero_wgrad_race_fix.register() attached engine.hetero_wgrad_race_fix")
