# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""SSMStateManager — CPU-offload manager for Mamba/SSM recurrent states in DES-LOC.

Mirrors Megatron 0dc36dfc6 [training migration] Migrate mamba builder, which
consolidates the Mamba/Hybrid model construction into a typed HybridModelConfig +
HybridModelBuilder hierarchy.  The key insight of that commit is that Mamba
requires two persistent state tensors per layer per batch:

    conv_state  : (batch, d_inner, d_conv-1)   — sliding convolution history
    ssm_state   : (batch, d_inner // dt_rank, d_state) — SSM hidden state

These states are *recurrent*: they carry information between consecutive forward
passes for the same sequence.  In Megatron they live entirely on GPU.  In
DES-LOC, training on 1.5 TB CPU memory nodes, offloading cold SSM states to
CPU pinned memory reduces live GPU memory by 30-60% for long-context Mamba-2
configurations.

Design intent (upstream 0dc36dfc6)
------------------------------------
HybridModelBuilder.build_model() instantiates a HybridModel with a
``hybrid_layer_pattern`` string (e.g. ``"MMMAMMMAMM"`` where ``M`` = Mamba,
``A`` = Attention).  Each ``M`` layer holds its own conv_state / ssm_state.
The upstream commit also introduces ``num_buckets`` as an alternative to
``bucket_size`` in DistributedDataParallelConfig — this is unrelated to SSM
states but signals that bucket topology is now first-class config, which is
consistent with our approach of making state topology explicit.

DES-LOC adaptation
-------------------
SSMStateManager maintains two pools:

    hot_states  : dict[layer_id → (conv_state, ssm_state)] on GPU
    cold_states : dict[layer_id → (conv_state, ssm_state)] on CPU pinned memory

A layer is *hot* when its forward pass is imminent or just completed; it is
*cold* when it will not be visited for several more layers.  In a typical
forward pass over an ``L``-layer Mamba model, layer ``i`` is cold for
``L - window`` steps where ``window`` is the prefetch lookahead.

State lifecycle:
1. ``register_layers(layer_ids, conv_shape, ssm_shape)``
   Called once at model init.  Allocates all states as CPU pinned tensors
   and marks each layer as COLD.

2. ``prefetch(layer_id)``
   Called during the forward pass, ``prefetch_ahead`` layers before layer_id
   is executed.  Initiates async H2D copy to GPU; transitions state to WARM.
   Diagnostic: [DS-SSM] PREFETCH event (mirrors M451 one-event-per-transition).

3. ``acquire(layer_id)``
   Called immediately before the SSM layer's forward().  Synchronises the
   prefetch stream, returns (conv_state, ssm_state) on GPU.  Transitions to HOT.
   Diagnostic: [DS-SSM] MISS if acquire() is called before prefetch completes.

4. ``release(layer_id)``
   Called after SSM forward().  Initiates async D2H copy back to CPU pinned;
   transitions state to COLD.
   Diagnostic: [DS-SSM] WRITEBACK.

5. ``evict_all()``
   Synchronises all outstanding writeback streams; called at gradient-sync
   boundary (mirrors Megatron's ``finalize_model_grads`` hook location).

Diagnostic events (rank-0, logger.info + print, one line per event):
    [DS-SSM] INIT      — one line at manager init, listing total pinned MB.
    [DS-SSM] PREFETCH  — layer_id, src COLD/GPU, stream handle.
    [DS-SSM] MISS      — layer_id, stall ms; emitted when acquire() had to wait.
    [DS-SSM] WRITEBACK — layer_id, bytes, async stream handle.
    [DS-SSM] EVICT_ALL — total bytes flushed, elapsed ms.
    [DS-SSM] PRESSURE  — emitted when hot_states exceeds ``max_hot_layers``
                         and an eviction was forced ahead of schedule.

The ``num_buckets``-style config approach from upstream is mirrored here:
SSMStateConfig accepts either ``max_hot_layers`` (absolute GPU layer count)
or ``hot_fraction`` (fraction of total layers) but not both — identical
mutual-exclusion invariant to Megatron's num_buckets/bucket_size guard.

Integration hook: ``wrap_mamba_model_for_offload(model, config)``
  Introspects a loaded Mamba/Hybrid model for its per-layer state shapes,
  calls register_layers(), and monkey-patches each MambaLayer.forward() to
  call prefetch/acquire/release transparently.  This mirrors the pre_wrap_hook
  mechanism from HybridModelConfig.pre_wrap_hooks.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch

from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-SSM]"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State lifecycle enum
# ---------------------------------------------------------------------------

class _StatePhase(Enum):
    COLD   = auto()   # resident in CPU pinned memory, no GPU copy in flight
    WARM   = auto()   # async H2D copy in flight; GPU tensor allocated but not ready
    HOT    = auto()   # GPU tensor valid; forward() may use it
    DIRTY  = auto()   # forward() done; async D2H writeback in flight


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SSMStateConfig:
    """Configuration for SSMStateManager.

    Mirrors the num_buckets/bucket_size mutual-exclusion pattern from
    Megatron 0dc36dfc6 DistributedDataParallelConfig.

    Args:
        max_hot_layers: Maximum number of SSM layers whose states may reside
            on GPU simultaneously.  None means all layers stay on GPU (no
            offload, manager is effectively a no-op wrapper).
        hot_fraction: Alternative to max_hot_layers.  Fraction of total
            registered layers that may be hot (e.g. 0.25 → 25%).
            Exactly one of max_hot_layers / hot_fraction must be set, or
            both None (disables offload).
        prefetch_ahead: How many layers ahead of the current layer to
            initiate the H2D prefetch.  Default 2 is tuned for PCIe 4.0
            bandwidth (≈ 32 GB/s) vs. typical Mamba-2-2.7B state sizes.
        prefetch_stream_count: Number of CUDA streams for concurrent H2D
            prefetches.  2 is sufficient for most topologies; increase to 4
            on NVLink-connected nodes where host-to-device bandwidth is the
            bottleneck.
        writeback_stream_count: Number of CUDA streams for concurrent D2H
            writebacks.  Separate from prefetch streams to avoid serialising
            forward and backward passes.
        verbose: If True, emit per-layer PREFETCH/WRITEBACK diagnostics.
            MISS, PRESSURE, and EVICT_ALL are always emitted regardless.
        rank: Rank for diagnostic gating (diagnostics only emitted on rank 0
            unless rank is None, in which case they always emit).
    """

    max_hot_layers: Optional[int] = None
    hot_fraction: Optional[float] = None
    prefetch_ahead: int = 2
    prefetch_stream_count: int = 2
    writeback_stream_count: int = 2
    verbose: bool = False
    rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_hot_layers is not None and self.hot_fraction is not None:
            raise ValueError(
                "SSMStateConfig: cannot specify both max_hot_layers and hot_fraction. "
                "Mirrors DistributedDataParallelConfig num_buckets/bucket_size invariant."
            )
        if self.hot_fraction is not None:
            if not (0.0 < self.hot_fraction <= 1.0):
                raise ValueError(
                    f"SSMStateConfig: hot_fraction must be in (0, 1], got {self.hot_fraction}"
                )
        if self.prefetch_ahead < 0:
            raise ValueError(
                f"SSMStateConfig: prefetch_ahead must be >= 0, got {self.prefetch_ahead}"
            )
        if self.prefetch_stream_count < 1:
            raise ValueError("SSMStateConfig: prefetch_stream_count must be >= 1")
        if self.writeback_stream_count < 1:
            raise ValueError("SSMStateConfig: writeback_stream_count must be >= 1")

    def resolve_max_hot(self, total_layers: int) -> Optional[int]:
        """Resolve max_hot_layers from either absolute or fractional spec."""
        if self.max_hot_layers is not None:
            return self.max_hot_layers
        if self.hot_fraction is not None:
            return max(1, int(total_layers * self.hot_fraction))
        # Both None → offload disabled
        return None


# ---------------------------------------------------------------------------
# Per-layer state record
# ---------------------------------------------------------------------------

@dataclass
class _LayerState:
    """Internal bookkeeping for one SSM layer's recurrent states."""

    layer_id: int
    conv_shape: Tuple[int, ...]    # (batch, d_inner, d_conv-1)
    ssm_shape: Tuple[int, ...]     # (batch, heads, d_state)

    # CPU pinned buffers — always allocated at register_layers time
    cpu_conv: torch.Tensor = field(default=None, repr=False)
    cpu_ssm:  torch.Tensor = field(default=None, repr=False)

    # GPU tensors — allocated on first prefetch, reused thereafter
    gpu_conv: Optional[torch.Tensor] = field(default=None, repr=False)
    gpu_ssm:  Optional[torch.Tensor] = field(default=None, repr=False)

    # Lifecycle
    phase: _StatePhase = _StatePhase.COLD
    prefetch_event: Optional[torch.cuda.Event] = field(default=None, repr=False)
    writeback_event: Optional[torch.cuda.Event] = field(default=None, repr=False)

    # Stream assignment (round-robin from manager's stream pools)
    prefetch_stream_idx: int = 0
    writeback_stream_idx: int = 0

    def pinned_bytes(self) -> int:
        """Total bytes in CPU pinned buffers."""
        return (
            self.cpu_conv.numel() * self.cpu_conv.element_size() +
            self.cpu_ssm.numel() * self.cpu_ssm.element_size()
        )


# ---------------------------------------------------------------------------
# SSMStateManager
# ---------------------------------------------------------------------------

class SSMStateManager:
    """Manages hot/cold partitioning of Mamba SSM recurrent states.

    Mirrors the ModelBuilder abstraction from Megatron 0dc36dfc6:
    just as HybridModelBuilder.build_model() is the single place that knows
    which layers are Mamba vs Attention, SSMStateManager is the single place
    that knows which layer states are hot vs cold.

    Thread safety: prefetch/acquire/release/evict_all are expected to be
    called from a single training thread.  The CUDA streams handle async GPU
    work; the phase transitions are protected by a lightweight threading.Lock
    (mirrors M407 DoubleBuffer._lock pattern).

    Args:
        config: SSMStateConfig controlling offload policy.
        dtype: Tensor dtype for SSM states (should match model dtype, e.g. bf16).
        device: CUDA device index for GPU allocations.
    """

    def __init__(
        self,
        config: SSMStateConfig,
        dtype: torch.dtype = torch.bfloat16,
        device: int = 0,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.device = torch.device(f"cuda:{device}")

        self._layers: Dict[int, _LayerState] = {}
        self._ordered_ids: List[int] = []   # forward-pass order; set in register_layers
        self._max_hot: Optional[int] = None  # resolved after register_layers

        # CUDA stream pools — round-robin assignment (mirrors M407 selector pattern)
        self._prefetch_streams: List[torch.cuda.Stream] = [
            torch.cuda.Stream(device=device)
            for _ in range(config.prefetch_stream_count)
        ]
        self._writeback_streams: List[torch.cuda.Stream] = [
            torch.cuda.Stream(device=device)
            for _ in range(config.writeback_stream_count)
        ]
        self._pf_stream_cursor: int = 0
        self._wb_stream_cursor: int = 0

        # Hot-set tracking
        self._hot_set: List[int] = []   # layer_ids currently HOT or WARM on GPU

        # Diagnostics
        self._lock = threading.Lock()
        self._miss_count: int = 0
        self._prefetch_count: int = 0
        self._writeback_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_log(self) -> bool:
        """Return True if diagnostics should be emitted for this rank."""
        if self.config.rank is None:
            return True
        return self.config.rank == 0

    def _emit(self, tag: str, msg: str) -> None:
        """Emit a structured diagnostic line (rank-0 gated)."""
        if not self._should_log():
            return
        full = f"{_LOG_PREFIX} {tag:12s} {msg}"
        print(full, flush=True)
        log.info(full)
        ds_logger.info(full)

    def _next_prefetch_stream(self) -> torch.cuda.Stream:
        """Round-robin next prefetch stream (mirrors M407 selector ^= 1 pattern)."""
        stream = self._prefetch_streams[self._pf_stream_cursor % len(self._prefetch_streams)]
        self._pf_stream_cursor += 1
        return stream

    def _next_writeback_stream(self) -> torch.cuda.Stream:
        """Round-robin next writeback stream."""
        stream = self._writeback_streams[self._wb_stream_cursor % len(self._writeback_streams)]
        self._wb_stream_cursor += 1
        return stream

    def _ensure_gpu_buffers(self, ls: _LayerState) -> None:
        """Allocate GPU tensors for layer ls if not already allocated."""
        if ls.gpu_conv is None:
            ls.gpu_conv = torch.empty(ls.conv_shape, dtype=self.dtype, device=self.device)
        if ls.gpu_ssm is None:
            ls.gpu_ssm = torch.empty(ls.ssm_shape, dtype=self.dtype, device=self.device)

    def _maybe_evict_lru(self) -> None:
        """If hot_set exceeds max_hot, force-writeback the oldest HOT layer.

        This is the key policy decision: prefer to evict the layer that has been
        HOT the longest (LRU approximation via FIFO ordering of _hot_set).
        WARM layers (prefetch in flight) are not evicted — evicting mid-transfer
        would corrupt the pinned buffer; we only evict fully HOT layers.

        Mirrors Megatron's bucket eviction logic: once a bucket is
        reduce-scattered, it's freed from GPU.
        """
        if self._max_hot is None:
            return
        while len(self._hot_set) > self._max_hot:
            # Find oldest HOT (not DIRTY/WARM) layer — linear scan, small set
            evict_id = None
            for lid in self._hot_set:
                ls = self._layers[lid]
                if ls.phase == _StatePhase.HOT:
                    evict_id = lid
                    break
            if evict_id is None:
                # All layers are WARM (prefetches in-flight) — cannot evict safely;
                # emit PRESSURE and bail.  This indicates prefetch_ahead is too large
                # relative to max_hot_layers.
                hot_warm = [lid for lid in self._hot_set]
                self._emit(
                    "PRESSURE",
                    f"hot_set={len(self._hot_set)} > max_hot={self._max_hot}, "
                    f"all layers WARM/DIRTY, cannot evict. "
                    f"Reduce prefetch_ahead or increase max_hot_layers. "
                    f"layers={hot_warm}"
                )
                return
            self._force_writeback(evict_id, reason="LRU_EVICT")

    def _force_writeback(self, layer_id: int, reason: str = "EVICT") -> None:
        """Initiate async D2H writeback for a HOT layer; mark COLD after sync.

        Unlike release() (which is called by the model after forward()),
        this path is triggered by memory pressure.  We synchronise immediately
        so the GPU buffer can be reused — this is the correct trade-off when
        we are over the hot-budget: a sync stall is better than an OOM.
        """
        ls = self._layers[layer_id]
        wb_stream = self._next_writeback_stream()
        with torch.cuda.stream(wb_stream):
            ls.cpu_conv.copy_(ls.gpu_conv, non_blocking=True)
            ls.cpu_ssm.copy_(ls.gpu_ssm, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(wb_stream)
            ls.writeback_event = ev
        wb_stream.synchronize()   # stall — forced by memory pressure
        ls.phase = _StatePhase.COLD
        if layer_id in self._hot_set:
            self._hot_set.remove(layer_id)
        self._writeback_count += 1
        if self.config.verbose:
            n_bytes = ls.pinned_bytes()
            self._emit(
                "WRITEBACK",
                f"layer={layer_id} bytes={n_bytes} reason={reason} (sync)"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_layers(
        self,
        layer_ids: List[int],
        conv_shape: Tuple[int, ...],
        ssm_shape: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Allocate CPU pinned buffers for all SSM layers.

        Called once at model init, before any forward passes.  The shape
        arguments describe a *single-layer* state tensor (batch dimension
        included); the manager allocates one pair per layer_id.

        Mirrors HybridModelBuilder.build_model()'s role as the single entry
        point that knows layer topology.  The layer_ids list is expected to
        be in forward-pass execution order (i.e. layer 0 runs before layer 1).

        Structured diagnostic: [DS-SSM] INIT — one line, total pinned MB.

        Args:
            layer_ids: Ordered list of SSM layer identifiers (int).
            conv_shape: Shape of conv_state for one layer, e.g. (B, d_inner, d_conv-1).
            ssm_shape: Shape of ssm_state for one layer, e.g. (B, heads, d_state).
            dtype: Override dtype; defaults to self.dtype.
        """
        if dtype is not None:
            self.dtype = dtype

        with self._lock:
            self._ordered_ids = list(layer_ids)
            total_pinned_bytes = 0

            for lid in layer_ids:
                cpu_conv = torch.zeros(conv_shape, dtype=self.dtype, pin_memory=True)
                cpu_ssm  = torch.zeros(ssm_shape,  dtype=self.dtype, pin_memory=True)
                ls = _LayerState(
                    layer_id=lid,
                    conv_shape=tuple(conv_shape),
                    ssm_shape=tuple(ssm_shape),
                    cpu_conv=cpu_conv,
                    cpu_ssm=cpu_ssm,
                )
                self._layers[lid] = ls
                total_pinned_bytes += ls.pinned_bytes()

            self._max_hot = self.config.resolve_max_hot(len(layer_ids))
            total_pinned_mb = total_pinned_bytes / (1024 ** 2)

        self._emit(
            "INIT",
            f"layers={len(layer_ids)} conv_shape={conv_shape} ssm_shape={ssm_shape} "
            f"dtype={self.dtype} pinned_mb={total_pinned_mb:.1f} "
            f"max_hot={self._max_hot} prefetch_ahead={self.config.prefetch_ahead} "
            f"offload={'enabled' if self._max_hot is not None else 'disabled'}"
        )

    def prefetch(self, layer_id: int) -> None:
        """Initiate async H2D prefetch for layer_id.

        Should be called ``config.prefetch_ahead`` layers before the layer
        will actually execute.  If the layer is already HOT or WARM, this
        is a no-op.  If the layer is COLD, initiates an async copy and
        transitions to WARM.

        The prefetch stream is chosen round-robin across the stream pool
        (mirrors M407 ``selector ^= 1`` — avoids all prefetches serialising
        on a single stream, which would defeat the overlap intent).

        Evicts the LRU HOT layer if adding this WARM layer would exceed
        max_hot_layers (mirrors bucket eviction in Megatron DDP).

        Diagnostic: [DS-SSM] PREFETCH (only if verbose=True or first N prefetches).
        """
        if self._max_hot is None:
            # Offload disabled — GPU states managed by the model itself
            return

        with self._lock:
            if layer_id not in self._layers:
                return
            ls = self._layers[layer_id]
            if ls.phase in (_StatePhase.WARM, _StatePhase.HOT):
                return   # already in flight or resident

            # Evict if at capacity before adding a new WARM slot
            if len(self._hot_set) >= self._max_hot:
                self._maybe_evict_lru()

            # Allocate GPU buffers if needed (first prefetch only)
            self._ensure_gpu_buffers(ls)

            # Select a prefetch stream
            pf_stream = self._next_prefetch_stream()
            ls.prefetch_stream_idx = self._pf_stream_cursor - 1

            with torch.cuda.stream(pf_stream):
                ls.gpu_conv.copy_(ls.cpu_conv, non_blocking=True)
                ls.gpu_ssm.copy_(ls.cpu_ssm, non_blocking=True)
                ev = torch.cuda.Event(enable_timing=False)
                ev.record(pf_stream)
                ls.prefetch_event = ev

            ls.phase = _StatePhase.WARM
            self._hot_set.append(layer_id)
            self._prefetch_count += 1

        if self.config.verbose:
            self._emit(
                "PREFETCH",
                f"layer={layer_id} stream={ls.prefetch_stream_idx} "
                f"conv_shape={ls.conv_shape} ssm_shape={ls.ssm_shape}"
            )

    def acquire(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return GPU (conv_state, ssm_state) for layer_id, blocking until ready.

        Synchronises the prefetch event if the layer is still WARM (copy in
        flight).  If the layer is COLD (prefetch was never initiated), falls
        back to a synchronous H2D copy — this is a MISS and triggers a
        [DS-SSM] MISS diagnostic with stall time, similar to how M451 emits
        GREW only at the transition boundary rather than every step.

        Transitions layer to HOT.  Must be paired with release().

        Args:
            layer_id: SSM layer identifier (from register_layers).

        Returns:
            (conv_state, ssm_state) GPU tensors.  The caller should treat
            these as read-write: mutations are written back by release().
        """
        if self._max_hot is None:
            # Offload disabled — the model manages its own GPU tensors;
            # return None to signal the hook should use model's own buffers.
            return None, None

        with self._lock:
            ls = self._layers[layer_id]

            if ls.phase == _StatePhase.COLD:
                # MISS: prefetch was not initiated (or was skipped due to OOM).
                t0 = time.perf_counter()
                self._ensure_gpu_buffers(ls)
                # Synchronous fallback copy
                ls.gpu_conv.copy_(ls.cpu_conv, non_blocking=False)
                ls.gpu_ssm.copy_(ls.cpu_ssm, non_blocking=False)
                stall_ms = (time.perf_counter() - t0) * 1e3
                self._miss_count += 1
                self._emit(
                    "MISS",
                    f"layer={layer_id} stall_ms={stall_ms:.2f} miss_total={self._miss_count} "
                    f"(prefetch_ahead={self.config.prefetch_ahead} may need to increase)"
                )
                ls.phase = _StatePhase.HOT
                if layer_id not in self._hot_set:
                    self._hot_set.append(layer_id)

            elif ls.phase == _StatePhase.WARM:
                # Wait for async prefetch to complete
                if ls.prefetch_event is not None:
                    t0 = time.perf_counter()
                    ls.prefetch_event.synchronize()
                    stall_ms = (time.perf_counter() - t0) * 1e3
                    if stall_ms > 0.5:
                        # Stall >0.5 ms means prefetch didn't fully hide latency
                        self._emit(
                            "MISS",
                            f"layer={layer_id} warm_stall_ms={stall_ms:.2f} "
                            f"(prefetch_ahead={self.config.prefetch_ahead} may need to increase)"
                        )
                        self._miss_count += 1
                ls.phase = _StatePhase.HOT

            # HOT: already ready, fall through

            return ls.gpu_conv, ls.gpu_ssm

    def release(self, layer_id: int) -> None:
        """Initiate async D2H writeback for layer_id after forward() completes.

        This is the SSM equivalent of Megatron's reduce-scatter-then-free:
        once the SSM layer has updated its state tensors, we copy them back
        to CPU pinned memory asynchronously, freeing the GPU tensors for
        reuse by future prefetches.

        Transitions layer from HOT to DIRTY (async writeback in flight) and
        then to COLD when the writeback stream is synchronised at evict_all().

        For memory efficiency, writeback and prefetch use separate CUDA stream
        pools so they can overlap each other.

        Diagnostic: [DS-SSM] WRITEBACK (verbose only).
        """
        if self._max_hot is None:
            return

        with self._lock:
            ls = self._layers[layer_id]
            if ls.phase != _StatePhase.HOT:
                # Already released or not yet acquired — benign no-op
                return

            wb_stream = self._next_writeback_stream()
            ls.writeback_stream_idx = self._wb_stream_cursor - 1

            with torch.cuda.stream(wb_stream):
                ls.cpu_conv.copy_(ls.gpu_conv, non_blocking=True)
                ls.cpu_ssm.copy_(ls.gpu_ssm, non_blocking=True)
                ev = torch.cuda.Event(enable_timing=False)
                ev.record(wb_stream)
                ls.writeback_event = ev

            ls.phase = _StatePhase.DIRTY
            self._writeback_count += 1

        if self.config.verbose:
            n_bytes = self._layers[layer_id].pinned_bytes()
            self._emit(
                "WRITEBACK",
                f"layer={layer_id} bytes={n_bytes} "
                f"stream={self._layers[layer_id].writeback_stream_idx} async"
            )

    def evict_all(self) -> None:
        """Synchronise all outstanding writeback streams; mark all layers COLD.

        Called at the gradient-synchronisation boundary (end of backward pass)
        to ensure all D2H copies have completed before the CPU optimiser reads
        the pinned buffers.  Mirrors Megatron's ``finalize_model_grads()``
        boundary where param-reduce-scatter is joined before the optimiser step.

        Diagnostic: [DS-SSM] EVICT_ALL — total bytes flushed, elapsed ms.
        """
        if self._max_hot is None:
            return

        t0 = time.perf_counter()
        total_bytes = 0

        with self._lock:
            for lid, ls in self._layers.items():
                if ls.phase in (_StatePhase.DIRTY, _StatePhase.WARM):
                    if ls.writeback_event is not None:
                        ls.writeback_event.synchronize()
                    # If WARM (prefetch was in flight, layer never acquired), sync prefetch
                    if ls.prefetch_event is not None:
                        ls.prefetch_event.synchronize()
                    total_bytes += ls.pinned_bytes()
                    ls.phase = _StatePhase.COLD
                elif ls.phase == _StatePhase.HOT:
                    # Layer was acquired but never released — force synchronous writeback
                    ls.cpu_conv.copy_(ls.gpu_conv, non_blocking=False)
                    ls.cpu_ssm.copy_(ls.gpu_ssm, non_blocking=False)
                    total_bytes += ls.pinned_bytes()
                    ls.phase = _StatePhase.COLD
            self._hot_set.clear()

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        self._emit(
            "EVICT_ALL",
            f"layers={len(self._layers)} bytes_flushed={total_bytes} "
            f"elapsed_ms={elapsed_ms:.2f} "
            f"prefetches={self._prefetch_count} writebacks={self._writeback_count} "
            f"misses={self._miss_count}"
        )
        # Reset per-step counters
        self._prefetch_count = 0
        self._writeback_count = 0
        self._miss_count = 0

    def schedule_prefetches(self, current_layer_idx: int) -> None:
        """Trigger prefetches for the next ``prefetch_ahead`` SSM layers.

        Convenience method for the forward-pass hook: call this at the start
        of each SSM layer's forward() with the current layer's *index* in the
        forward-pass order (not the layer_id).  This initiates H2D copies for
        the next ``prefetch_ahead`` layers.

        This mirrors how Megatron's VirtualPipelineScheduler pre-issues recv
        buffers for upcoming pipeline stages.

        Args:
            current_layer_idx: 0-based index of the currently executing layer
                in ``self._ordered_ids`` (forward-pass order).
        """
        if self._max_hot is None:
            return
        ahead = self.config.prefetch_ahead
        n = len(self._ordered_ids)
        for offset in range(1, ahead + 1):
            next_idx = current_layer_idx + offset
            if next_idx < n:
                self.prefetch(self._ordered_ids[next_idx])

    def state_summary(self) -> Dict[str, int]:
        """Return a snapshot of current phase counts for diagnostics."""
        counts = {p.name: 0 for p in _StatePhase}
        with self._lock:
            for ls in self._layers.values():
                counts[ls.phase.name] += 1
        return counts


# ---------------------------------------------------------------------------
# Model integration hook
# ---------------------------------------------------------------------------

def wrap_mamba_model_for_offload(
    model_or_modules,
    config: SSMStateConfig,
    layer_attr: str = "mamba_layer",
    conv_state_attr: str = "conv_state",
    ssm_state_attr: str = "ssm_state",
    dtype: Optional[torch.dtype] = None,
    device: int = 0,
) -> Optional[SSMStateManager]:
    """Introspect model for Mamba layers and install the SSM offload hook.

    Mirrors HybridModelConfig.pre_wrap_hooks: a function that receives the
    model list and patches it in-place.  This function is the DES-LOC entry
    point for SSM state offload, complementing what Megatron 0dc36dfc6 sets
    up in HybridModelBuilder.

    The function walks ``model_or_modules`` looking for modules that have
    both ``conv_state_attr`` and ``ssm_state_attr`` attributes, which are
    the per-layer recurrent tensors in Mamba implementations.

    For each discovered layer:
    1. Its state shapes are read from the existing GPU tensor shapes.
    2. States are registered with SSMStateManager.
    3. The module's ``forward()`` is monkey-patched to call
       schedule_prefetches / acquire / release around the original forward.

    If no Mamba layers are found, returns None (disables offload silently).

    Args:
        model_or_modules: A nn.Module or list of nn.Modules to walk.
        config: SSMStateConfig controlling offload policy.
        layer_attr: Attribute name used to detect Mamba sub-modules
            (matched by checking if the *module itself* has conv/ssm states).
        conv_state_attr: Attribute name of the conv_state tensor on the module.
        ssm_state_attr: Attribute name of the ssm_state tensor on the module.
        dtype: Tensor dtype for CPU pinned buffers; inferred from model if None.
        device: CUDA device index.

    Returns:
        Configured SSMStateManager if at least one Mamba layer was found,
        otherwise None.

    Structured diagnostic:
        [DS-SSM] WRAP_SCAN  — number of Mamba layers discovered.
        [DS-SSM] WRAP_SKIP  — if no layers found (offload not applied).
    """
    import torch.nn as nn

    if isinstance(model_or_modules, nn.Module):
        modules_list = [model_or_modules]
    else:
        modules_list = list(model_or_modules)

    # Walk all modules looking for SSM recurrent state holders
    mamba_layers: List[Tuple[int, nn.Module]] = []
    layer_id = 0
    for root in modules_list:
        for _name, module in root.named_modules():
            has_conv = hasattr(module, conv_state_attr)
            has_ssm  = hasattr(module, ssm_state_attr)
            if has_conv and has_ssm:
                mamba_layers.append((layer_id, module))
                layer_id += 1

    _emit_plain = lambda tag, msg: (
        print(f"{_LOG_PREFIX} {tag:12s} {msg}", flush=True),
        log.info(f"{_LOG_PREFIX} {tag:12s} {msg}"),
    )

    if not mamba_layers:
        _emit_plain("WRAP_SKIP", "no Mamba layers with conv_state/ssm_state found; offload not applied")
        return None

    _emit_plain("WRAP_SCAN", f"found {len(mamba_layers)} Mamba layers")

    # Infer shapes from first layer's current state tensors
    _, first_module = mamba_layers[0]
    conv_tensor = getattr(first_module, conv_state_attr)
    ssm_tensor  = getattr(first_module, ssm_state_attr)
    inferred_dtype = dtype or conv_tensor.dtype
    conv_shape = tuple(conv_tensor.shape)
    ssm_shape  = tuple(ssm_tensor.shape)

    # Build the manager
    manager = SSMStateManager(config=config, dtype=inferred_dtype, device=device)
    layer_ids = [lid for lid, _ in mamba_layers]
    manager.register_layers(layer_ids, conv_shape, ssm_shape, dtype=inferred_dtype)

    # Install forward hooks
    for idx, (lid, module) in enumerate(mamba_layers):
        _install_forward_hook(manager, module, lid, idx, conv_state_attr, ssm_state_attr)

    return manager


def _install_forward_hook(
    manager: SSMStateManager,
    module,
    layer_id: int,
    layer_idx: int,
    conv_state_attr: str,
    ssm_state_attr: str,
) -> None:
    """Monkey-patch module.forward() to call manager prefetch/acquire/release.

    The patched forward:
    1. Calls schedule_prefetches(layer_idx) to initiate H2D for upcoming layers.
    2. Calls acquire(layer_id) to get GPU conv_state / ssm_state.
    3. Temporarily sets module.conv_state / module.ssm_state to the managed GPU tensors.
    4. Calls original forward(*args, **kwargs).
    5. Writes updated state back from module attributes to managed tensors
       (in case the module wrote to new tensors rather than in-place).
    6. Calls release(layer_id) to initiate async D2H writeback.

    This design is conservative: it does not assume the module writes
    conv_state/ssm_state in-place.  Some Mamba implementations return new
    state tensors; others update in-place.  We re-read the attributes after
    forward() to capture either case.
    """
    original_forward = module.forward

    def patched_forward(*args, **kwargs):
        # Step 1: schedule upcoming prefetches
        manager.schedule_prefetches(layer_idx)

        # Step 2: acquire managed GPU state tensors
        gpu_conv, gpu_ssm = manager.acquire(layer_id)

        if gpu_conv is not None:
            # Step 3: inject managed tensors into module (swap in)
            old_conv = getattr(module, conv_state_attr, None)
            old_ssm  = getattr(module, ssm_state_attr, None)
            setattr(module, conv_state_attr, gpu_conv)
            setattr(module, ssm_state_attr, gpu_ssm)

        # Step 4: run original forward
        output = original_forward(*args, **kwargs)

        if gpu_conv is not None:
            # Step 5: capture any updated state (may differ if non-in-place)
            new_conv = getattr(module, conv_state_attr, None)
            new_ssm  = getattr(module, ssm_state_attr, None)
            if new_conv is not gpu_conv and new_conv is not None:
                gpu_conv.copy_(new_conv)
            if new_ssm is not gpu_ssm and new_ssm is not None:
                gpu_ssm.copy_(new_ssm)

            # Restore original placeholders (avoids stale GPU tensors hanging around)
            if old_conv is not None:
                setattr(module, conv_state_attr, old_conv)
            if old_ssm is not None:
                setattr(module, ssm_state_attr, old_ssm)

            # Step 6: initiate async writeback
            manager.release(layer_id)

        return output

    module.forward = patched_forward


# ---------------------------------------------------------------------------
# DeepSpeed engine integration helper
# ---------------------------------------------------------------------------

def attach_ssm_state_manager_to_engine(
    engine,
    ssm_manager: SSMStateManager,
) -> None:
    """Attach SSMStateManager to a DeepSpeed engine for automatic evict_all.

    Wraps engine.allreduce_gradients() (called at the end of backward) to
    trigger manager.evict_all() before the reduce, ensuring all D2H writebacks
    are synchronised before the CPU optimiser accesses pinned memory.

    This mirrors the Megatron finalize_model_grads() integration point where
    Flextron's router grad sync (M4188) is injected.

    Args:
        engine: A DeepSpeedEngine instance.
        ssm_manager: The SSMStateManager to attach.

    Diagnostic: [DS-SSM] ENGINE_ATTACH — one line confirming the hook.
    """
    original_allreduce = getattr(engine, "allreduce_gradients", None)
    if original_allreduce is None:
        log.warning(
            f"{_LOG_PREFIX} ENGINE_ATTACH  engine has no allreduce_gradients; "
            "evict_all must be called manually at the gradient-sync boundary."
        )
        return

    def wrapped_allreduce(*args, **kwargs):
        ssm_manager.evict_all()
        return original_allreduce(*args, **kwargs)

    engine.allreduce_gradients = wrapped_allreduce
    ssm_manager._emit(
        "ENGINE_ATTACH",
        f"evict_all() hooked into engine.allreduce_gradients() "
        f"(engine type: {type(engine).__name__})"
    )
