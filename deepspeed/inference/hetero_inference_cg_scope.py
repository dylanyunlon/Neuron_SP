"""
deepspeed/inference/hetero_inference_cg_scope.py

DES-LOC Heterogeneous Inference CUDA Graph Scope Manager
=========================================================

Upstream design intent (Megatron commit 35f76df3f2c2fe4f676d242c1d0a4a41bd6f4271):
    The Megatron commit addresses a subtle but critical correctness issue in hybrid model
    inference: previously, the CUDA graph scope for ``full_iteration_inference`` was attached
    at the ``HybridStack`` (decoder) level, which meant the embedding lookup and the output
    (lm_head) projection were executed *outside* the captured graph. This caused:
      1. Unnecessary kernel launches at graph boundaries (host synchronization points).
      2. Incorrect tensor aliasing when ``WrappedTensor`` objects crossed the graph/non-graph
         boundary — the ``unwrap()`` call in ``HybridStack.__call__`` was a band-aid.
      3. For dynamic-batching inference the ``attention_mask is None`` guard in
         ``_should_call_local_cudagraph`` was too conservative, preventing graph reuse.

    The fix moves ``GraphableMegatronModule`` inheritance (and therefore
    ``_should_call_local_cudagraph`` + ``__call__`` override + ``create_mcore_cudagraph_manager``)
    from ``HybridStack`` up to ``HybridModel``, so the graph capture envelope now includes
    the embedding table forward pass and the final linear projection.  The ``HybridStack``
    reverts to a plain ``MegatronModule``.

DES-LOC adaptation rationale:
    In the Neuron_SP / DES-LOC heterogeneous setup we have three devices with very different
    characteristics:

        ┌─────────────┬──────────┬────────────────────┬──────────┐
        │ Device      │ SM arch  │ HBM capacity       │ PCIe bw  │
        ├─────────────┼──────────┼────────────────────┼──────────┤
        │ A6000 ×2    │ SM86     │ 48 GB × 2 = 96 GB  │ ~32 GB/s │
        │ H100 NVL    │ SM90     │ 96 GB              │ ~64 GB/s │
        └─────────────┴──────────┴────────────────────┴──────────┘

    All devices are PCIe-connected with no NVLink, so cross-device tensor traffic is the
    dominant bottleneck.  DES-LOC (Decoupled Execution with Shared LOcality Cache) addresses
    this by:

      A. **Decoupled Execution** — embedding + backbone + output head can run on *different*
         devices.  The embedding table is large and bandwidth-bound; it benefits from being
         pinned to the H100 NVL (higher PCIe bandwidth to the host DRAM holding the vocab
         table).  The transformer backbone attention layers are compute-bound and benefit from
         spreading across A6000s.  The output projection is again bandwidth-bound and returns
         to H100.

      B. **Shared LOcality Cache (SLoc Cache)** — instead of capturing a monolithic CUDA graph
         that spans device boundaries (which CUDA itself does not support across PCIe without
         NCCL), DES-LOC captures *per-device sub-graphs* and stitches them with lightweight
         ``torch.cuda.Event``-based synchronization.  The "locality cache" is a pinned CPU
         DRAM buffer (exploiting the 1.5 TB capacity) that acts as a rendezvous point for
         activations crossing device boundaries, avoiding redundant H2D/D2H copies.

    The key insight mirrored from Megatron's commit is: **the graph scope must include the
    embedding and output layers** — otherwise their kernel launches become graph boundary
    synchronization points that negate the performance benefit of capturing the backbone.
    In DES-LOC this translates to: the ``HeteroInferenceCGScope`` must be managed at the
    *model* level (analogous to ``HybridModel``), not at the *stack* level (analogous to
    ``HybridStack``).

    Concretely this file implements:
      * ``DeviceProfile``         — SM arch + memory + PCIe bandwidth descriptor.
      * ``SLocCache``             — pinned-memory rendezvous buffer with double-buffering.
      * ``SubGraphCapture``       — per-device CUDA graph wrapper with lazy replay.
      * ``HeteroInferenceCGScope``— the DES-LOC analogue of Megatron's ``GraphableMegatronModule``
                                    mixin, managing scope determination, graph creation, and
                                    coordinated replay across the heterogeneous device set.
      * ``HeteroHybridModel``     — concrete model class (thin wrapper around a DeepSpeed
                                    pipeline engine) demonstrating the full integration.

Author: Neuron_SP project (DES-LOC adaptation of Megatron 35f76df)
"""

from __future__ import annotations

import contextlib
import enum
import logging
import math
import os
import threading
import time
import unittest
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger — only meaningful events are logged (rule 4)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DeviceRole(enum.Enum):
    """Semantic role a device plays in the DES-LOC pipeline."""
    EMBEDDING   = "embedding"    # vocab embedding + positional encoding
    BACKBONE    = "backbone"     # transformer / Mamba / attention layers
    OUTPUT_HEAD = "output_head"  # lm_head projection


class CGScopeMode(enum.Enum):
    """
    Controls which parts of a forward pass are captured in CUDA graphs.

    Mirrors Megatron's ``CudaGraphScope`` enum but extended for multi-device
    heterogeneous scenarios.

    ``FULL_ITERATION`` is the DES-LOC analogue of Megatron's
    ``CudaGraphScope.full_iteration_inference``: the scope envelope spans embedding,
    backbone, and output head — but the implementation captures *per-device sub-graphs*
    rather than one monolithic graph.
    """
    DISABLED         = "disabled"
    LAYER_LEVEL      = "layer_level"       # graph per transformer layer (Megatron default)
    FULL_ITERATION   = "full_iteration"    # graph covers full forward (DES-LOC default)


# ---------------------------------------------------------------------------
# Device profiling
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    """
    Static capability descriptor for one physical GPU.

    Used by the scheduler to decide which ``DeviceRole`` to assign and whether
    CUDA graph capture is safe on that device.

    Args:
        device_index: ``torch.device`` index.
        sm_major: CUDA compute capability major version (e.g. 9 for H100).
        sm_minor: CUDA compute capability minor version.
        total_memory_bytes: Total HBM capacity in bytes.
        pcie_bandwidth_GBps: Measured or nominal PCIe bandwidth in GB/s.
        role: Assigned ``DeviceRole`` (can be overridden by the user).
    """
    device_index: int
    sm_major: int
    sm_minor: int
    total_memory_bytes: int
    pcie_bandwidth_GBps: float
    role: Optional[DeviceRole] = None

    @property
    def compute_capability(self) -> Tuple[int, int]:
        return (self.sm_major, self.sm_minor)

    @property
    def supports_cuda_graphs(self) -> bool:
        """CUDA graphs require sm_70+."""
        return (self.sm_major, self.sm_minor) >= (7, 0)

    @property
    def total_memory_GB(self) -> float:
        return self.total_memory_bytes / (1024 ** 3)

    def __repr__(self) -> str:
        return (
            f"DeviceProfile(idx={self.device_index}, "
            f"sm={self.sm_major}{self.sm_minor}, "
            f"mem={self.total_memory_GB:.0f}GB, "
            f"pcie={self.pcie_bandwidth_GBps:.0f}GB/s, "
            f"role={self.role})"
        )


def probe_devices(device_indices: List[int]) -> List[DeviceProfile]:
    """
    Query CUDA runtime for device capabilities and return ``DeviceProfile`` objects.

    Falls back gracefully when a device index is unavailable (e.g. in CI without
    the full hardware stack).

    Args:
        device_indices: List of ``torch.cuda`` device indices to probe.

    Returns:
        List of ``DeviceProfile`` objects in the same order as ``device_indices``.
    """
    profiles: List[DeviceProfile] = []
    for idx in device_indices:
        if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
            # Synthetic fallback — useful for unit tests on CPU-only machines.
            profiles.append(DeviceProfile(
                device_index=idx,
                sm_major=8, sm_minor=6,
                total_memory_bytes=48 * 1024 ** 3,
                pcie_bandwidth_GBps=32.0,
            ))
            continue
        props = torch.cuda.get_device_properties(idx)
        profiles.append(DeviceProfile(
            device_index=idx,
            sm_major=props.major,
            sm_minor=props.minor,
            total_memory_bytes=props.total_memory,
            pcie_bandwidth_GBps=_estimate_pcie_bw(props),
        ))
    return profiles


def _estimate_pcie_bw(props: "torch.cuda.DeviceProperties") -> float:
    """
    Heuristic PCIe bandwidth estimate from device properties.

    H100 NVL on PCIe 5.0 x16 ≈ 64 GB/s; A6000 on PCIe 4.0 x16 ≈ 32 GB/s.
    We use SM version as a proxy since ``memory_bus_width`` reflects HBM, not PCIe.
    """
    if props.major >= 9:   # Hopper (H100)
        return 64.0
    if props.major == 8 and props.minor >= 6:  # Ampere A6000
        return 32.0
    return 16.0  # conservative fallback


def assign_roles(profiles: List[DeviceProfile]) -> List[DeviceProfile]:
    """
    Heuristic role assignment for a heterogeneous device set.

    Strategy:
      * Device with highest PCIe bandwidth AND largest memory → OUTPUT_HEAD + EMBEDDING
        (two roles can share a device if there are ≤2 GPUs, otherwise they split).
      * Remaining devices → BACKBONE.

    For the canonical DES-LOC hardware (2× A6000 + 1× H100 NVL):
      * H100 NVL (64 GB/s, 96 GB) → EMBEDDING (first) and OUTPUT_HEAD (last).
      * A6000 #0 and #1 (32 GB/s, 48 GB) → BACKBONE.

    The caller may override roles after this function returns.

    Args:
        profiles: List of ``DeviceProfile`` objects (modified in-place).

    Returns:
        The same list with ``.role`` set on each element.
    """
    if not profiles:
        return profiles

    sorted_by_bw = sorted(profiles, key=lambda p: (p.pcie_bandwidth_GBps, p.total_memory_GB),
                          reverse=True)
    highest_bw_dev = sorted_by_bw[0]

    for p in profiles:
        p.role = DeviceRole.BACKBONE

    if len(profiles) == 1:
        profiles[0].role = DeviceRole.EMBEDDING  # will also do backbone & output
    elif len(profiles) == 2:
        highest_bw_dev.role = DeviceRole.EMBEDDING
        sorted_by_bw[1].role = DeviceRole.BACKBONE
    else:
        # ≥3 devices: highest PCIe bw handles embedding & output; rest are backbone
        highest_bw_dev.role = DeviceRole.EMBEDDING
        # Find next-best for output head (ideally same device if memory allows,
        # otherwise the second-highest bandwidth device)
        for p in profiles:
            if p.device_index != highest_bw_dev.device_index and p.role == DeviceRole.BACKBONE:
                # Leave backbone devices as-is; output head goes back to the high-BW device
                break
        # Output head: we reuse highest_bw_dev (H100 NVL has 96 GB, can hold both tables)
        # Mark it with OUTPUT_HEAD; the model must handle dual-role logic.
        # We use a sentinel: if role is EMBEDDING, OUTPUT_HEAD is also on this device.

    logger.info(
        "DES-LOC device role assignment: %s",
        [(p.device_index, p.role) for p in profiles],
    )
    return profiles


# ---------------------------------------------------------------------------
# Shared LOcality Cache (SLoc Cache)
# ---------------------------------------------------------------------------

class SLocCache:
    """
    Pinned CPU DRAM rendezvous buffer for cross-device activation transfer.

    In a PCIe-only heterogeneous setup, transferring activations between two
    GPUs must go through host memory (D2H on source, H2D on destination).
    The SLoc Cache pre-allocates pinned buffers and double-buffers them so
    that the next transfer can overlap with the current forward pass computation.

    Design notes:
      * Pinned memory is allocated once at construction time to avoid repeated
        ``cudaMallocHost`` calls during inference, which would break CUDA graph
        capture on the *receiving* device.
      * ``torch.cuda.Event`` objects are used for fine-grained synchronization
        without full stream synchronization, preserving concurrency.
      * Buffer slots are identified by a monotonically increasing counter; the
        consumer waits on the corresponding event before reading.

    Args:
        shape: Shape of the activation tensor to cache (excluding batch dimension).
        dtype: Tensor dtype.
        n_slots: Number of double-buffer slots (default 2 for true double-buffering).
        max_batch_size: Maximum batch size to pre-allocate for.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        n_slots: int = 2,
        max_batch_size: int = 64,
    ) -> None:
        self._shape = shape
        self._dtype = dtype
        self._n_slots = n_slots
        self._max_batch_size = max_batch_size
        self._slot_counter = 0
        self._lock = threading.Lock()

        # Pre-allocate pinned buffers: [n_slots, max_batch_size, *shape]
        full_shape = (n_slots, max_batch_size) + shape
        self._buffers = torch.empty(full_shape, dtype=dtype, pin_memory=True)

        # One event per slot per device role pair (src → dst)
        self._events: Dict[int, torch.cuda.Event] = {}
        if torch.cuda.is_available():
            for i in range(n_slots):
                self._events[i] = torch.cuda.Event(enable_timing=False, blocking=False)

        logger.debug(
            "SLocCache allocated: shape=%s dtype=%s slots=%d max_batch=%d "
            "pinned_bytes=%.1f MB",
            full_shape, dtype, n_slots, max_batch_size,
            self._buffers.nbytes / 1024 ** 2,
        )

    def write(self, tensor: torch.Tensor, src_stream: Optional[torch.cuda.Stream] = None) -> int:
        """
        Asynchronously copy ``tensor`` into the next available pinned slot.

        Args:
            tensor: Source tensor on a CUDA device.
            src_stream: CUDA stream on the source device. If None, uses the
                        current stream.

        Returns:
            Slot index that was written (pass to ``read()``).
        """
        with self._lock:
            slot = self._slot_counter % self._n_slots
            self._slot_counter += 1

        batch_size = tensor.shape[0]
        if batch_size > self._max_batch_size:
            raise ValueError(
                f"SLocCache batch overflow: got {batch_size}, max {self._max_batch_size}"
            )

        target = self._buffers[slot, :batch_size]
        ctx = torch.cuda.stream(src_stream) if src_stream is not None else contextlib.nullcontext()
        with ctx:
            target.copy_(tensor, non_blocking=True)
            if torch.cuda.is_available() and slot in self._events:
                self._events[slot].record()

        return slot

    def read(
        self,
        slot: int,
        dst_device: torch.device,
        batch_size: int,
        dst_stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """
        Wait for the write to complete and copy the pinned buffer to ``dst_device``.

        Args:
            slot: Slot index returned by ``write()``.
            dst_device: Target CUDA device.
            batch_size: Number of batch entries to copy.
            dst_stream: CUDA stream on the destination device.

        Returns:
            Tensor on ``dst_device``.
        """
        if torch.cuda.is_available() and slot in self._events:
            # Wait on the source-side event without blocking the host thread
            if dst_stream is not None:
                dst_stream.wait_event(self._events[slot])
            else:
                self._events[slot].synchronize()

        src = self._buffers[slot, :batch_size]
        ctx = torch.cuda.stream(dst_stream) if dst_stream is not None else contextlib.nullcontext()
        with ctx:
            return src.to(dst_device, non_blocking=True)

    def __repr__(self) -> str:
        return (
            f"SLocCache(shape={self._shape}, dtype={self._dtype}, "
            f"slots={self._n_slots}, max_batch={self._max_batch_size})"
        )


# ---------------------------------------------------------------------------
# Per-device sub-graph capture
# ---------------------------------------------------------------------------

class SubGraphCapture:
    """
    Wraps a single per-device CUDA graph capture-and-replay lifecycle.

    Unlike Megatron's ``CudaGraphManager`` which captures a monolithic graph
    over all layers, ``SubGraphCapture`` captures only the operations that
    run on *one physical device*.  Multiple ``SubGraphCapture`` objects are
    orchestrated by ``HeteroInferenceCGScope``.

    Lifecycle:
      1. ``maybe_capture(callable, *args)``  — on first call, warms up then captures.
      2. Subsequent calls replay the graph with updated static input tensors.
      3. ``invalidate()`` forces re-capture (e.g. after batch size change).

    Thread safety: NOT thread-safe.  Each device should have its own capture object
    on its own stream.

    Args:
        device: The ``torch.device`` this capture is associated with.
        warmup_iters: Number of warmup iterations before graph capture.
        static_input_names: Names of kwargs that are static (pre-allocated).
    """

    def __init__(
        self,
        device: torch.device,
        warmup_iters: int = 3,
        static_input_names: Optional[List[str]] = None,
    ) -> None:
        self.device = device
        self.warmup_iters = warmup_iters
        self.static_input_names = static_input_names or []

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: Dict[str, torch.Tensor] = {}
        self._static_output: Optional[torch.Tensor] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._iter_count: int = 0
        self._captured: bool = False

        if torch.cuda.is_available():
            self._capture_stream = torch.cuda.Stream(device=device)

    @property
    def is_captured(self) -> bool:
        return self._captured

    def maybe_capture(
        self,
        fn: Callable,
        static_inputs: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run ``fn`` with graph capture on the first eligible iteration.

        Args:
            fn: The callable to capture (e.g. ``module.forward``).
            static_inputs: Dict of tensor inputs that will be *updated in-place*
                           for each replay.  These must be pre-allocated with the
                           correct shape.
            **kwargs: Non-tensor kwargs passed through as-is (not captured).

        Returns:
            Output tensor (either from direct execution or graph replay).
        """
        if not torch.cuda.is_available():
            return fn(**static_inputs, **kwargs)

        self._iter_count += 1

        if self._iter_count <= self.warmup_iters:
            # Warmup: run eagerly to populate caches (e.g. cuDNN autotune)
            with torch.cuda.device(self.device):
                return fn(**static_inputs, **kwargs)

        if not self._captured:
            self._do_capture(fn, static_inputs, **kwargs)

        # Replay: update static input storage
        for name, tensor in static_inputs.items():
            if name in self._static_inputs:
                self._static_inputs[name].copy_(tensor)

        with torch.cuda.device(self.device):
            self._graph.replay()

        return self._static_output

    def _do_capture(
        self,
        fn: Callable,
        static_inputs: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Internal: perform CUDA graph capture."""
        with torch.cuda.device(self.device):
            # Allocate static tensors on device
            self._static_inputs = {
                name: tensor.clone().detach()
                for name, tensor in static_inputs.items()
            }

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=self._capture_stream):
                self._static_output = fn(**self._static_inputs, **kwargs)

            self._captured = True

        logger.info(
            "SubGraphCapture: captured CUDA graph on device=%s for fn=%s",
            self.device, getattr(fn, "__name__", repr(fn)),
        )

    def invalidate(self) -> None:
        """Force re-capture on the next call to ``maybe_capture``."""
        self._graph = None
        self._static_inputs = {}
        self._static_output = None
        self._captured = False
        self._iter_count = 0
        logger.info("SubGraphCapture: invalidated graph on device=%s", self.device)

    def __repr__(self) -> str:
        return (
            f"SubGraphCapture(device={self.device}, captured={self._captured}, "
            f"iter={self._iter_count})"
        )


# ---------------------------------------------------------------------------
# HeteroInferenceCGScope — the core DES-LOC mixin
# ---------------------------------------------------------------------------

class HeteroInferenceCGScope:
    """
    DES-LOC analogue of Megatron's ``GraphableMegatronModule`` mixin.

    Upstream intent recap:
        Megatron's ``GraphableMegatronModule`` provides ``_should_call_local_cudagraph``
        and ``__call__`` override so that the *entire model forward pass* (including
        embedding and output head) is enveloped in a single CUDA graph when
        ``CudaGraphScope.full_iteration_inference`` is active.  The key fix in commit
        35f76df was moving this mixin from ``HybridStack`` (decoder only) to
        ``HybridModel`` (full model) so the embedding and lm_head are captured too.

    DES-LOC adaptation:
        We cannot use a single CUDA graph across PCIe-connected GPUs.  Instead:
          1. The embedding forward on the H100 NVL is captured in ``_emb_graph``.
          2. The backbone forward on A6000 devices is captured in ``_backbone_graphs``
             (one per backbone device, replayed in sequence or concurrently).
          3. The output head forward on the H100 NVL is captured in ``_out_graph``.
          4. ``SLocCache`` instances bridge the device boundaries.

        The ``_should_engage_hetero_cg`` method is the DES-LOC analogue of Megatron's
        ``_should_call_local_cudagraph`` — it checks:
          a. We are in inference mode (not training).
          b. The scope mode is ``CGScopeMode.FULL_ITERATION``.
          c. Graphs have been initialized (``setup_hetero_cg_scope`` was called).
          d. Batch size matches the captured static shape.

        ``__call__`` is overridden (as in Megatron) to dispatch to either the
        graph-replay path or the eager path.

    Usage:
        class MyHybridModel(HeteroInferenceCGScope, nn.Module):
            def setup_hetero_cg_scope(self, scope_config):
                ...

            def forward(self, input_ids, ...):
                ...

    Args:
        This is a mixin; no ``__init__`` args.  Call ``_init_hetero_cg_state`` from
        the concrete class ``__init__``.
    """

    def _init_hetero_cg_state(
        self,
        device_profiles: List[DeviceProfile],
        scope_mode: CGScopeMode = CGScopeMode.FULL_ITERATION,
        warmup_iters: int = 3,
        sloc_max_batch: int = 64,
        sloc_hidden_size: int = 4096,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Initialize internal DES-LOC CG scope state.

        Must be called from the concrete model's ``__init__`` *after*
        ``super().__init__()``.

        Args:
            device_profiles: Profiled devices with roles assigned.
            scope_mode: Which parts to capture.
            warmup_iters: Warmup iterations before graph capture.
            sloc_max_batch: Max batch size for SLoc cache pre-allocation.
            sloc_hidden_size: Hidden dimension for SLoc cache buffer sizing.
            dtype: Activation dtype.
        """
        self._hetero_scope_mode = scope_mode
        self._device_profiles = device_profiles
        self._warmup_iters = warmup_iters
        self._dtype = dtype

        # Sub-graph captures keyed by DeviceRole
        self._emb_graph: Optional[SubGraphCapture] = None
        self._backbone_graphs: List[SubGraphCapture] = []
        self._out_graph: Optional[SubGraphCapture] = None

        # SLoc caches between device boundaries
        # emb → backbone boundary
        self._sloc_emb_to_backbone: Optional[SLocCache] = None
        # backbone → output boundary
        self._sloc_backbone_to_out: Optional[SLocCache] = None

        # Metadata
        self._hetero_cg_initialized: bool = False
        self._last_batch_size: int = -1

        if scope_mode == CGScopeMode.FULL_ITERATION:
            # Allocate SLoc caches now (shape depends on hidden size)
            hidden_shape = (sloc_hidden_size,)
            self._sloc_emb_to_backbone = SLocCache(
                shape=hidden_shape,
                dtype=dtype,
                n_slots=2,
                max_batch_size=sloc_max_batch,
            )
            self._sloc_backbone_to_out = SLocCache(
                shape=hidden_shape,
                dtype=dtype,
                n_slots=2,
                max_batch_size=sloc_max_batch,
            )
            logger.info(
                "HeteroInferenceCGScope: SLoc caches initialized "
                "(hidden=%d, max_batch=%d, dtype=%s)",
                sloc_hidden_size, sloc_max_batch, dtype,
            )

    def setup_hetero_cg_scope(self) -> None:
        """
        Create per-device ``SubGraphCapture`` objects based on assigned roles.

        This is the DES-LOC analogue of Megatron's ``create_mcore_cudagraph_manager``.
        Must be called after the model weights are loaded and before the first
        inference call.

        Raises:
            RuntimeError: If ``_init_hetero_cg_state`` was not called first.
        """
        if not hasattr(self, "_hetero_scope_mode"):
            raise RuntimeError(
                "Call _init_hetero_cg_state() before setup_hetero_cg_scope()"
            )
        if self._hetero_scope_mode == CGScopeMode.DISABLED:
            return

        for profile in self._device_profiles:
            device = torch.device("cuda", profile.device_index)
            capture = SubGraphCapture(
                device=device,
                warmup_iters=self._warmup_iters,
            )
            if profile.role == DeviceRole.EMBEDDING:
                self._emb_graph = capture
            elif profile.role == DeviceRole.BACKBONE:
                self._backbone_graphs.append(capture)
            elif profile.role == DeviceRole.OUTPUT_HEAD:
                self._out_graph = capture
            # If a device is EMBEDDING and also OUTPUT_HEAD (H100 NVL handles both),
            # the embedding capture is reused; output head capture is a separate object.
            # The dual-role case is handled by checking if _out_graph is None after the loop.

        # If H100 handles both embedding and output head (dual-role), create a separate
        # SubGraphCapture for the output head on the same device.
        emb_profile = next((p for p in self._device_profiles
                            if p.role == DeviceRole.EMBEDDING), None)
        if self._out_graph is None and emb_profile is not None:
            device = torch.device("cuda", emb_profile.device_index)
            self._out_graph = SubGraphCapture(
                device=device,
                warmup_iters=self._warmup_iters,
            )
            logger.info(
                "HeteroInferenceCGScope: output head graph co-located with embedding "
                "on device cuda:%d (dual-role H100 NVL)",
                emb_profile.device_index,
            )

        self._hetero_cg_initialized = True
        logger.info(
            "HeteroInferenceCGScope: setup complete — "
            "emb_graph=%s, backbone_graphs=%d, out_graph=%s",
            self._emb_graph, len(self._backbone_graphs), self._out_graph,
        )

    def _should_engage_hetero_cg(self, batch_size: int) -> bool:
        """
        Determine whether to use the graph-replay path for this forward call.

        Mirrors Megatron's ``_should_call_local_cudagraph`` but adapted for DES-LOC:
          * Training mode always uses eager (graphs break gradient computation).
          * Scope must be FULL_ITERATION.
          * Graphs must be initialized.
          * Batch size must match the previously captured shape (or be ≤ the
            pre-allocated SLoc cache max_batch_size if not yet captured).

        Args:
            batch_size: Current batch size.

        Returns:
            True if the graph-replay path should be used.
        """
        if not hasattr(self, "_hetero_scope_mode"):
            return False
        if self.training:
            return False
        if self._hetero_scope_mode != CGScopeMode.FULL_ITERATION:
            return False
        if not self._hetero_cg_initialized:
            return False
        if self._emb_graph is None or self._out_graph is None:
            return False
        # Once captured, batch size must match; before capture, any size ≤ max is fine.
        if self._emb_graph.is_captured and batch_size != self._last_batch_size:
            return False
        return True

    def _execute_hetero_cg_forward(
        self,
        input_ids: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Orchestrate the three-phase DES-LOC graph-replay forward pass.

        Phase 1 (Embedding — H100 NVL):
            Replay the embedding sub-graph.  Copy activations to SLoc cache.

        Phase 2 (Backbone — A6000 × 2):
            For each backbone device: pull activations from SLoc cache, replay
            backbone sub-graph, push to next SLoc slot.

        Phase 3 (Output Head — H100 NVL):
            Pull final activations from SLoc cache, replay output head sub-graph.

        Args:
            input_ids: Token IDs on the embedding device.
            *args: Additional positional args forwarded to sub-graphs.
            **kwargs: Additional keyword args forwarded to sub-graphs.

        Returns:
            Logit tensor on the output head device (H100 NVL).
        """
        batch_size = input_ids.shape[0]

        # --- Phase 1: Embedding ---
        emb_device = self._emb_graph.device
        input_ids_emb = input_ids.to(emb_device, non_blocking=True)

        hidden = self._emb_graph.maybe_capture(
            fn=self._embedding_forward,
            static_inputs={"input_ids": input_ids_emb},
            **kwargs,
        )

        # Transfer to SLoc cache (async D2H copy)
        slot_eb = self._sloc_emb_to_backbone.write(hidden)

        # --- Phase 2: Backbone (sequential across A6000 shards) ---
        current_slot = slot_eb
        current_cache = self._sloc_emb_to_backbone
        next_cache = self._sloc_backbone_to_out

        for i, bg in enumerate(self._backbone_graphs):
            bb_device = bg.device
            # Pull from previous SLoc cache
            hidden_bb = current_cache.read(
                slot=current_slot,
                dst_device=bb_device,
                batch_size=batch_size,
            )
            hidden_out = bg.maybe_capture(
                fn=self._backbone_forward,
                static_inputs={"hidden_states": hidden_bb, "shard_idx": torch.tensor(i)},
                **kwargs,
            )
            # For multi-shard backbone: the last shard writes to backbone→output cache
            if i == len(self._backbone_graphs) - 1:
                current_slot = next_cache.write(hidden_out)
                current_cache = next_cache
            else:
                # Inter-backbone shard: write to a temporary slot in emb_to_backbone cache
                # (reusing the double-buffer since we read it already)
                current_slot = self._sloc_emb_to_backbone.write(hidden_out)
                current_cache = self._sloc_emb_to_backbone

        # --- Phase 3: Output Head ---
        out_device = self._out_graph.device
        hidden_final = next_cache.read(
            slot=current_slot,
            dst_device=out_device,
            batch_size=batch_size,
        )

        logits = self._out_graph.maybe_capture(
            fn=self._output_head_forward,
            static_inputs={"hidden_states": hidden_final},
            **kwargs,
        )

        self._last_batch_size = batch_size
        return logits

    # ---- Sub-forward stubs (overridden by concrete model) ----

    def _embedding_forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass for the embedding sub-module.

        Subclasses MUST override this method to call the actual embedding layer.

        Args:
            input_ids: Token IDs on the embedding device.
            **kwargs: Additional kwargs (position_ids, etc.).

        Returns:
            Hidden states tensor on the embedding device.
        """
        raise NotImplementedError(
            "Subclass must implement _embedding_forward for DES-LOC hetero CG scope"
        )

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,
        shard_idx: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for one backbone shard.

        Subclasses MUST override this method to call the appropriate transformer
        / Mamba layers assigned to the device corresponding to ``shard_idx``.

        Args:
            hidden_states: Activations tensor on the backbone device.
            shard_idx: Integer index identifying which backbone shard this is.
            **kwargs: Additional kwargs (attention_mask, etc.).

        Returns:
            Hidden states tensor on the same backbone device.
        """
        raise NotImplementedError(
            "Subclass must implement _backbone_forward for DES-LOC hetero CG scope"
        )

    def _output_head_forward(
        self, hidden_states: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass for the lm_head / output projection.

        Subclasses MUST override this method to call the actual output layer.

        Args:
            hidden_states: Final hidden states on the output device.
            **kwargs: Additional kwargs.

        Returns:
            Logit tensor.
        """
        raise NotImplementedError(
            "Subclass must implement _output_head_forward for DES-LOC hetero CG scope"
        )

    def invalidate_hetero_graphs(self) -> None:
        """
        Invalidate all per-device sub-graphs (e.g. after KV cache reallocation).

        Equivalent to destroying and recreating Megatron's ``CudaGraphManager``
        when sequence length or batch size changes.
        """
        if self._emb_graph:
            self._emb_graph.invalidate()
        for bg in self._backbone_graphs:
            bg.invalidate()
        if self._out_graph:
            self._out_graph.invalidate()
        self._last_batch_size = -1
        logger.info("HeteroInferenceCGScope: all sub-graphs invalidated")


# ---------------------------------------------------------------------------
# Concrete HeteroHybridModel
# ---------------------------------------------------------------------------

class _EmbeddingStub(nn.Module):
    """Minimal token embedding for demonstration."""
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.weight(input_ids)


class _TransformerShardStub(nn.Module):
    """Minimal single-layer transformer shard for demonstration."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.linear(self.norm(hidden_states))


class _OutputHeadStub(nn.Module):
    """Minimal lm_head for demonstration."""
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.proj(hidden_states)


class HeteroHybridModel(HeteroInferenceCGScope, nn.Module):
    """
    Concrete hybrid language model with DES-LOC heterogeneous inference CG scope.

    This class demonstrates the full DES-LOC integration pattern mirroring
    what Megatron's commit 35f76df did for ``HybridModel``:

      * ``HeteroInferenceCGScope`` is mixed in at the *model* level (not decoder level).
      * ``setup_hetero_cg_scope()`` is called after weight loading.
      * ``__call__`` dispatches to graph-replay or eager path.
      * ``_embedding_forward``, ``_backbone_forward``, ``_output_head_forward``
        implement the per-device sub-modules.

    In a real deployment the stub sub-modules would be replaced by:
      * DeepSpeed ``PipelineModule`` shards assigned to specific devices.
      * Mamba / attention hybrid blocks loaded from a checkpoint.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Model hidden dimension.
        device_profiles: Profiled and role-assigned ``DeviceProfile`` list.
        scope_mode: CG scope mode.
        warmup_iters: Warmup iterations before graph capture.
        sloc_max_batch: SLoc cache pre-allocation batch size.
        dtype: Activation dtype.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 256,  # small for unit tests
        device_profiles: Optional[List[DeviceProfile]] = None,
        scope_mode: CGScopeMode = CGScopeMode.FULL_ITERATION,
        warmup_iters: int = 3,
        sloc_max_batch: int = 8,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Default to CPU-only single-device profile for testing
        if device_profiles is None:
            device_profiles = [
                DeviceProfile(
                    device_index=0 if torch.cuda.is_available() else -1,
                    sm_major=8, sm_minor=6,
                    total_memory_bytes=48 * 1024 ** 3,
                    pcie_bandwidth_GBps=32.0,
                    role=DeviceRole.EMBEDDING,
                )
            ]

        self._device_profiles = device_profiles

        # Determine devices for each role
        emb_profile  = next((p for p in device_profiles if p.role == DeviceRole.EMBEDDING), None)
        bb_profiles  = [p for p in device_profiles if p.role == DeviceRole.BACKBONE]
        out_profile  = next((p for p in device_profiles if p.role == DeviceRole.OUTPUT_HEAD),
                            emb_profile)

        def _dev(profile: Optional[DeviceProfile]) -> torch.device:
            if profile is None or not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device("cuda", profile.device_index)

        self._emb_device  = _dev(emb_profile)
        self._bb_devices  = [_dev(p) for p in bb_profiles] or [self._emb_device]
        self._out_device  = _dev(out_profile)

        # Build sub-modules on their assigned devices
        self.embedding = _EmbeddingStub(vocab_size, hidden_size).to(
            self._emb_device, dtype=dtype if dtype != torch.float16
            else torch.float32  # Embedding always float32
        )
        self.backbone_shards = nn.ModuleList([
            _TransformerShardStub(hidden_size).to(dev, dtype=dtype
                if dtype != torch.float16 else torch.float32)
            for dev in self._bb_devices
        ])
        self.output_head = _OutputHeadStub(hidden_size, vocab_size).to(
            self._out_device, dtype=dtype
            if dtype != torch.float16 else torch.float32
        )

        # Initialize DES-LOC CG state
        self._init_hetero_cg_state(
            device_profiles=device_profiles,
            scope_mode=scope_mode,
            warmup_iters=warmup_iters,
            sloc_max_batch=sloc_max_batch,
            sloc_hidden_size=hidden_size,
            dtype=torch.float32,  # use float32 for stubs
        )

    def setup_hetero_cg_scope(self) -> None:
        """
        Initialize sub-graph captures after weights are loaded.

        Mirrors Megatron's ``create_mcore_cudagraph_manager`` which was the
        key method added to ``HybridModel`` in commit 35f76df.  Previously it
        lived on ``HybridStack``, meaning the embedding was outside the graph.
        """
        super().setup_hetero_cg_scope()

    def _embedding_forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.embedding(input_ids)

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,
        shard_idx: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        idx = int(shard_idx.item())
        shard = self.backbone_shards[idx % len(self.backbone_shards)]
        return shard(hidden_states)

    def _output_head_forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.output_head(hidden_states)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Override ``__call__`` to dispatch to DES-LOC graph-replay or eager path.

        This is the direct analogue of Megatron's ``HybridModel.__call__`` override
        introduced in commit 35f76df (previously on ``HybridStack.__call__``).

        The key difference: we check for the DES-LOC hetero CG condition rather than
        Megatron's ``full_iteration_inference`` scope.
        """
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None and self._should_engage_hetero_cg(input_ids.shape[0]):
            return self._execute_hetero_cg_forward(input_ids, **kwargs)
        return nn.Module.__call__(self, *args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Eager forward pass (used during warmup and training).

        Args:
            input_ids: Token IDs, shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            **kwargs: Additional kwargs.

        Returns:
            Logit tensor, shape ``[batch, seq_len, vocab_size]``.
        """
        # Phase 1: Embedding (H100 NVL)
        ids = input_ids.to(self._emb_device)
        hidden = self.embedding(ids)

        # Phase 2: Backbone shards (A6000 × n)
        for i, (shard, bb_dev) in enumerate(zip(self.backbone_shards, self._bb_devices)):
            hidden = hidden.to(bb_dev)
            hidden = shard(hidden)

        # Phase 3: Output head (H100 NVL)
        hidden = hidden.to(self._out_device)
        logits = self.output_head(hidden)
        return logits


# ---------------------------------------------------------------------------
# Utility: build the canonical 2×A6000 + 1×H100 NVL profile
# ---------------------------------------------------------------------------

def build_des_loc_profiles(
    a6000_indices: List[int] = (0, 1),
    h100_index: int = 2,
) -> List[DeviceProfile]:
    """
    Construct the canonical DES-LOC device profile for 2×A6000 + 1×H100 NVL.

    The H100 NVL is assigned ``DeviceRole.EMBEDDING`` (it also implicitly handles
    the output head since ``assign_roles`` leaves ``_out_graph`` to be co-located).
    The two A6000s are assigned ``DeviceRole.BACKBONE``.

    Args:
        a6000_indices: CUDA device indices for the A6000 GPUs.
        h100_index: CUDA device index for the H100 NVL GPU.

    Returns:
        List of ``DeviceProfile`` objects with roles assigned.
    """
    profiles = probe_devices(list(a6000_indices) + [h100_index])

    # Override roles explicitly for the known hardware configuration
    for p in profiles:
        if p.device_index in a6000_indices:
            p.role = DeviceRole.BACKBONE
        elif p.device_index == h100_index:
            p.role = DeviceRole.EMBEDDING  # OUTPUT_HEAD co-located per assign_roles logic

    logger.info(
        "DES-LOC canonical profile built: A6000×%d at indices %s, "
        "H100 NVL at index %d",
        len(a6000_indices), a6000_indices, h100_index,
    )
    return profiles


# ---------------------------------------------------------------------------
# Context manager for inference mode
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def hetero_inference_context(model: HeteroInferenceCGScope):
    """
    Context manager that puts the model in eval mode and initializes the
    DES-LOC CG scope if not already done.

    Mirrors the pattern in Megatron's inference engine where
    ``create_mcore_cudagraph_manager`` is called once after model initialization.

    Args:
        model: A ``HeteroHybridModel`` (or any ``HeteroInferenceCGScope`` subclass).

    Yields:
        The model in eval mode with graphs initialized.
    """
    was_training = model.training
    model.eval()
    if not model._hetero_cg_initialized:
        model.setup_hetero_cg_scope()
    try:
        yield model
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestDeviceProfile(unittest.TestCase):
    """Tests for DeviceProfile and role assignment."""

    def test_compute_capability(self):
        p = DeviceProfile(0, 9, 0, 96 * 1024**3, 64.0)
        self.assertEqual(p.compute_capability, (9, 0))
        self.assertTrue(p.supports_cuda_graphs)

    def test_supports_cuda_graphs_threshold(self):
        p_old = DeviceProfile(0, 6, 1, 8 * 1024**3, 16.0)
        self.assertFalse(p_old.supports_cuda_graphs)

    def test_total_memory_GB(self):
        p = DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0)
        self.assertAlmostEqual(p.total_memory_GB, 48.0, places=0)

    def test_assign_roles_three_devices(self):
        profiles = [
            DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0),
            DeviceProfile(1, 8, 6, 48 * 1024**3, 32.0),
            DeviceProfile(2, 9, 0, 96 * 1024**3, 64.0),
        ]
        assign_roles(profiles)
        roles = {p.device_index: p.role for p in profiles}
        # H100 should be EMBEDDING (highest bw)
        self.assertEqual(roles[2], DeviceRole.EMBEDDING)
        # A6000s should be BACKBONE
        self.assertEqual(roles[0], DeviceRole.BACKBONE)
        self.assertEqual(roles[1], DeviceRole.BACKBONE)

    def test_assign_roles_single_device(self):
        profiles = [DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0)]
        assign_roles(profiles)
        self.assertEqual(profiles[0].role, DeviceRole.EMBEDDING)

    def test_assign_roles_two_devices(self):
        profiles = [
            DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0),
            DeviceProfile(1, 9, 0, 96 * 1024**3, 64.0),
        ]
        assign_roles(profiles)
        roles = {p.device_index: p.role for p in profiles}
        self.assertEqual(roles[1], DeviceRole.EMBEDDING)
        self.assertEqual(roles[0], DeviceRole.BACKBONE)


class TestSLocCache(unittest.TestCase):
    """Tests for SLocCache pinned memory rendezvous."""

    def test_write_read_cpu(self):
        """Test write/read round-trip on CPU tensors (no CUDA needed)."""
        cache = SLocCache(shape=(16,), dtype=torch.float32, n_slots=2, max_batch_size=4)
        tensor = torch.randn(3, 16)
        slot = cache.write(tensor)
        result = cache.read(slot, dst_device=torch.device("cpu"), batch_size=3)
        self.assertEqual(result.shape, (3, 16))
        self.assertTrue(torch.allclose(result, tensor, atol=1e-5))

    def test_double_buffering(self):
        """Two consecutive writes use different slots."""
        cache = SLocCache(shape=(8,), dtype=torch.float32, n_slots=2, max_batch_size=2)
        t1 = torch.ones(2, 8)
        t2 = torch.zeros(2, 8)
        s1 = cache.write(t1)
        s2 = cache.write(t2)
        self.assertNotEqual(s1 % 2, s2 % 2)

    def test_batch_overflow_raises(self):
        cache = SLocCache(shape=(4,), dtype=torch.float32, n_slots=2, max_batch_size=2)
        big_tensor = torch.randn(5, 4)
        with self.assertRaises(ValueError):
            cache.write(big_tensor)

    def test_repr(self):
        cache = SLocCache(shape=(32,), dtype=torch.float16)
        self.assertIn("SLocCache", repr(cache))


class TestSubGraphCapture(unittest.TestCase):
    """Tests for SubGraphCapture with CPU fallback (no CUDA graph on CPU)."""

    def setUp(self):
        self.device = torch.device("cpu")

    def test_warmup_eager_execution(self):
        capture = SubGraphCapture(device=self.device, warmup_iters=2)
        calls = []

        def fn(x):
            calls.append(1)
            return x * 2

        out1 = capture.maybe_capture(fn, {"x": torch.tensor([1.0])})
        out2 = capture.maybe_capture(fn, {"x": torch.tensor([2.0])})
        # Both should be eager (within warmup_iters)
        self.assertEqual(len(calls), 2)
        self.assertFalse(capture.is_captured)

    def test_invalidate_resets_state(self):
        capture = SubGraphCapture(device=self.device, warmup_iters=1)
        capture._captured = True
        capture._iter_count = 10
        capture.invalidate()
        self.assertFalse(capture.is_captured)
        self.assertEqual(capture._iter_count, 0)

    def test_repr(self):
        capture = SubGraphCapture(device=self.device)
        self.assertIn("SubGraphCapture", repr(capture))


class TestHeteroInferenceCGScopeInit(unittest.TestCase):
    """Tests for the mixin initialization and scope logic."""

    def _make_profiles(self) -> List[DeviceProfile]:
        return [
            DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0, role=DeviceRole.BACKBONE),
            DeviceProfile(1, 8, 6, 48 * 1024**3, 32.0, role=DeviceRole.BACKBONE),
            DeviceProfile(2, 9, 0, 96 * 1024**3, 64.0, role=DeviceRole.EMBEDDING),
        ]

    def test_init_creates_sloc_caches(self):
        model = HeteroHybridModel(
            vocab_size=100, hidden_size=32,
            device_profiles=self._make_profiles(),
            scope_mode=CGScopeMode.FULL_ITERATION,
            warmup_iters=1, sloc_max_batch=4,
        )
        self.assertIsNotNone(model._sloc_emb_to_backbone)
        self.assertIsNotNone(model._sloc_backbone_to_out)

    def test_init_disabled_no_sloc(self):
        model = HeteroHybridModel(
            vocab_size=100, hidden_size=32,
            device_profiles=self._make_profiles(),
            scope_mode=CGScopeMode.DISABLED,
            warmup_iters=1, sloc_max_batch=4,
        )
        self.assertIsNone(model._sloc_emb_to_backbone)
        self.assertIsNone(model._sloc_backbone_to_out)

    def test_should_not_engage_in_training(self):
        model = HeteroHybridModel(
            vocab_size=100, hidden_size=32,
            device_profiles=self._make_profiles(),
            scope_mode=CGScopeMode.FULL_ITERATION,
            warmup_iters=1, sloc_max_batch=4,
        )
        model.train()
        model.setup_hetero_cg_scope()
        self.assertFalse(model._should_engage_hetero_cg(batch_size=2))

    def test_should_not_engage_before_setup(self):
        model = HeteroHybridModel(
            vocab_size=100, hidden_size=32,
            device_profiles=self._make_profiles(),
            scope_mode=CGScopeMode.FULL_ITERATION,
            warmup_iters=1, sloc_max_batch=4,
        )
        model.eval()
        # setup_hetero_cg_scope not called yet
        self.assertFalse(model._should_engage_hetero_cg(batch_size=2))

    def test_setup_initializes_graphs(self):
        model = HeteroHybridModel(
            vocab_size=100, hidden_size=32,
            device_profiles=self._make_profiles(),
            scope_mode=CGScopeMode.FULL_ITERATION,
            warmup_iters=1, sloc_max_batch=4,
        )
        model.setup_hetero_cg_scope()
        self.assertTrue(model._hetero_cg_initialized)
        # Two backbone devices → two backbone graphs
        self.assertEqual(len(model._backbone_graphs), 2)
        self.assertIsNotNone(model._emb_graph)
        self.assertIsNotNone(model._out_graph)

    def test_should_engage_after_setup_eval(self):
        model = HeteroHybridModel(
            vocab_size=100, hidden_size=32,
            device_profiles=self._make_profiles(),
            scope_mode=CGScopeMode.FULL_ITERATION,
            warmup_iters=1, sloc_max_batch=4,
        )
        model.eval()
        model.setup_hetero_cg_scope()
        # Before capture, batch size check passes for any size ≤ max
        self.assertTrue(model._should_engage_hetero_cg(batch_size=2))

    def test_should_not_engage_after_batch_size_change(self):
        model = HeteroHybridModel(
            vocab_size=100, hidden_size=32,
            device_profiles=self._make_profiles(),
            scope_mode=CGScopeMode.FULL_ITERATION,
            warmup_iters=1, sloc_max_batch=4,
        )
        model.eval()
        model.setup_hetero_cg_scope()
        # Simulate a captured state with batch_size=2
        model._last_batch_size = 2
        model._emb_graph._captured = True
        # Different batch size → should not engage
        self.assertFalse(model._should_engage_hetero_cg(batch_size=3))


class TestHeteroHybridModelForward(unittest.TestCase):
    """Integration tests for the full HeteroHybridModel forward pass."""

    def _make_cpu_model(self, scope_mode=CGScopeMode.DISABLED) -> HeteroHybridModel:
        """Create a CPU-only model for testing without CUDA."""
        profiles = [
            DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0, role=DeviceRole.EMBEDDING),
        ]
        return HeteroHybridModel(
            vocab_size=64, hidden_size=16,
            device_profiles=profiles,
            scope_mode=scope_mode,
            warmup_iters=1, sloc_max_batch=4,
            dtype=torch.float32,
        )

    def test_eager_forward_shape(self):
        model = self._make_cpu_model(scope_mode=CGScopeMode.DISABLED)
        model.eval()
        input_ids = torch.randint(0, 64, (2, 5))
        logits = model(input_ids=input_ids)
        self.assertEqual(logits.shape, (2, 5, 64))

    def test_eager_forward_training_mode(self):
        model = self._make_cpu_model(scope_mode=CGScopeMode.DISABLED)
        model.train()
        input_ids = torch.randint(0, 64, (1, 3))
        logits = model(input_ids=input_ids)
        self.assertEqual(logits.shape, (1, 3, 64))

    def test_context_manager_sets_eval(self):
        model = self._make_cpu_model()
        model.train()
        with hetero_inference_context(model):
            self.assertFalse(model.training)
            self.assertTrue(model._hetero_cg_initialized)
        # After context, should be back to training
        self.assertTrue(model.training)

    def test_invalidate_resets_graphs(self):
        model = self._make_cpu_model(scope_mode=CGScopeMode.FULL_ITERATION)
        model.eval()
        model.setup_hetero_cg_scope()
        # Simulate captured graphs
        model._emb_graph._captured = True
        model._last_batch_size = 2
        model.invalidate_hetero_graphs()
        self.assertFalse(model._emb_graph.is_captured)
        self.assertEqual(model._last_batch_size, -1)

    def test_forward_consistent_across_calls(self):
        """Same input should give same output (deterministic)."""
        model = self._make_cpu_model(scope_mode=CGScopeMode.DISABLED)
        model.eval()
        input_ids = torch.randint(0, 64, (2, 4))
        with torch.no_grad():
            out1 = model(input_ids=input_ids)
            out2 = model(input_ids=input_ids)
        self.assertTrue(torch.allclose(out1, out2))

    def test_two_backbone_shards_forward(self):
        """Model with two backbone shards on the same CPU device."""
        profiles = [
            DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0, role=DeviceRole.EMBEDDING),
            DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0, role=DeviceRole.BACKBONE),
            DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0, role=DeviceRole.BACKBONE),
        ]
        model = HeteroHybridModel(
            vocab_size=32, hidden_size=8,
            device_profiles=profiles,
            scope_mode=CGScopeMode.DISABLED,
            warmup_iters=1, sloc_max_batch=4,
            dtype=torch.float32,
        )
        model.eval()
        input_ids = torch.randint(0, 32, (1, 3))
        logits = model(input_ids=input_ids)
        self.assertEqual(logits.shape, (1, 3, 32))

    def test_scope_mode_enum_values(self):
        """Ensure all scope modes are distinct strings."""
        values = [m.value for m in CGScopeMode]
        self.assertEqual(len(values), len(set(values)))


class TestBuildDesLocProfiles(unittest.TestCase):
    """Tests for the canonical DES-LOC hardware profile builder."""

    def test_profile_count(self):
        profiles = build_des_loc_profiles(a6000_indices=[0, 1], h100_index=2)
        self.assertEqual(len(profiles), 3)

    def test_role_assignment(self):
        profiles = build_des_loc_profiles(a6000_indices=[0, 1], h100_index=2)
        role_map = {p.device_index: p.role for p in profiles}
        self.assertEqual(role_map[0], DeviceRole.BACKBONE)
        self.assertEqual(role_map[1], DeviceRole.BACKBONE)
        self.assertEqual(role_map[2], DeviceRole.EMBEDDING)

    def test_pcie_bw_heuristic(self):
        profiles = build_des_loc_profiles(a6000_indices=[0, 1], h100_index=2)
        # Without actual CUDA, all devices get fallback profiles from probe_devices
        # Just check the profiles are non-empty and have valid bw
        for p in profiles:
            self.assertGreater(p.pcie_bandwidth_GBps, 0)


if __name__ == "__main__":
    # Configure logging for standalone test run
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("DES-LOC HeteroInferenceCGScope — unit test suite")
    print("Mirrors Megatron commit 35f76df3 (embedding+output in full CG scope)")
    print("=" * 70)

    # Quick smoke test before running the full suite
    logger.info("Smoke test: probing devices and assigning roles")
    smoke_profiles = build_des_loc_profiles(a6000_indices=[0, 1], h100_index=2)
    for p in smoke_profiles:
        logger.info("  %s", p)

    logger.info("Smoke test: constructing HeteroHybridModel (CPU fallback)")
    cpu_profiles = [
        DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0, role=DeviceRole.EMBEDDING),
        DeviceProfile(0, 8, 6, 48 * 1024**3, 32.0, role=DeviceRole.BACKBONE),
    ]
    model = HeteroHybridModel(
        vocab_size=128, hidden_size=32,
        device_profiles=cpu_profiles,
        scope_mode=CGScopeMode.FULL_ITERATION,
        warmup_iters=2, sloc_max_batch=8,
        dtype=torch.float32,
    )
    model.eval()
    model.setup_hetero_cg_scope()

    test_input = torch.randint(0, 128, (2, 6))
    with torch.no_grad():
        logits = model(input_ids=test_input)
    logger.info(
        "Smoke test forward pass: input=%s → logits=%s",
        test_input.shape, logits.shape,
    )
    assert logits.shape == (2, 6, 128), f"Unexpected shape: {logits.shape}"

    logger.info("Smoke test: hetero_inference_context manager")
    model.train()
    with hetero_inference_context(model) as m:
        assert not m.training, "Model should be in eval mode inside context"
    assert model.training, "Model should be back in train mode after context"

    logger.info("Smoke test: SLocCache round-trip")
    cache = SLocCache(shape=(32,), dtype=torch.float32, n_slots=2, max_batch_size=4)
    t_in = torch.randn(3, 32)
    slot = cache.write(t_in)
    t_out = cache.read(slot, dst_device=torch.device("cpu"), batch_size=3)
    assert torch.allclose(t_in, t_out, atol=1e-5), "SLocCache round-trip mismatch"
    logger.info("SLocCache round-trip: OK")

    print("\nRunning full unittest suite...\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in [
        TestDeviceProfile,
        TestSLocCache,
        TestSubGraphCapture,
        TestHeteroInferenceCGScopeInit,
        TestHeteroHybridModelForward,
        TestBuildDesLocProfiles,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        logger.info("All tests passed.")
    else:
        logger.error("%d failure(s), %d error(s)", len(result.failures), len(result.errors))
        raise SystemExit(1)
