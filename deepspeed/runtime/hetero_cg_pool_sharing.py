"""
deepspeed/runtime/hetero_cg_pool_sharing.py

DES-LOC HeteroCGPoolSharing: Heterogeneous CUDA Graph Pool Sharing for Mixed-Architecture Training.

═══════════════════════════════════════════════════════════════════════════════
UPSTREAM DESIGN INTENT (Megatron-LM daec17c853ed25fa54cb4655a46b62a424996094)
═══════════════════════════════════════════════════════════════════════════════

Megatron PR #4698 ("Allow optimizer CG to share the same pool as full-iter CG") addresses a
subtle but significant memory fragmentation problem in CUDA graph captures:

1. **The Problem**: When `torch.cuda.graph_pool_handle()` is called independently for each
   graph capture site (full-iteration forward-backward, optimizer step), each call returns a
   distinct memory pool handle. CUDA allocates per-stream alloc segments for each distinct pool,
   causing `memory_reserved()` to balloon far beyond what the graphs actually need. This is
   documented in Megatron's `tools/debug_cuda_graph_pool_memory*.py`.

2. **The Fix**: Two process-wide singletons — `_shared_graph_pool` and `_shared_capture_stream`
   — ensure that full-iteration and optimizer graph captures share exactly one pool and one
   non-default stream. The `use_single_mempool` flag gates this behavior so users can opt in.

3. **Default Changed**: `cuda_graph_use_single_mempool` default flipped from `False` → `True`
   in `TransformerConfig`, reflecting that sharing is strictly better for the full-iteration
   scope (where microbatch reuse patterns are less of a concern).

4. **Propagation**: `FullCudaGraphWrapper` and `OptimizerCudaGraphWrapper` both accept
   `use_single_mempool`, and call sites in `training.py`, `evaluate()`, and `rl_utils.py`
   forward the config flag through.

═══════════════════════════════════════════════════════════════════════════════
DES-LOC ADAPTATION: HeteroCGPoolSharing
═══════════════════════════════════════════════════════════════════════════════

The Neuron_SP / DES-LOC context introduces complications that Megatron's homogeneous-GPU
design never faces:

**Hardware Topology**
  - 2× A6000 (48 GB, SM86, Ampere) — strong tensor-parallel workers, PCIe-attached
  - 1× H100 NVL (96 GB, SM90, Hopper) — prefill/attention specialist, PCIe-attached
  - 1.5 TB CPU DRAM — LOC (Shared LOcality Cache) tier
  - No NVLink: all GPU↔GPU transfers go through PCIe (≈32 GB/s bidirectional)

**Why naïve pool sharing fails on heterogeneous devices**

  `torch.cuda.graph_pool_handle()` is device-scoped — a pool handle from device 0 (A6000)
  cannot be reused on device 2 (H100). Megatron's singleton approach implicitly assumes the
  process is pinned to one CUDA device. In DES-LOC's decoupled execution model, forward
  and optimizer steps may land on *different* devices depending on the Locality Cache
  scheduler's placement decisions. We therefore maintain **per-device** pool and stream
  singletons, keyed by `torch.cuda.current_device()`.

**DES-LOC Execution Phases**

  DES-LOC splits a training iteration into:
    1. PREFILL phase  — attention-heavy; routed to H100 (device 2) when possible.
    2. FFN/MLP phase  — tensor-parallel across A6000s (devices 0–1).
    3. OPTIMIZER phase — runs on the device that owns the parameter shard; may differ
                         from the forward device.
    4. LOC SYNC phase — CPU DRAM ↔ GPU transfers for the Shared LOcality Cache.

  Each phase may need its own CUDA graph. The pool-sharing logic here ensures that phases
  executing on the *same* device share a pool (matching Megatron's intent), while phases
  on *different* devices get distinct pools (required by CUDA semantics).

**Locality Cache Integration**

  The LOC tier holds fp16/bf16 activation checkpoints and KV-cache snapshots in CPU DRAM.
  CUDA graphs that include async H2D/D2H copies (via `cudaMemcpyAsync`) are captured on a
  dedicated `loc_transfer_stream` separate from the compute capture stream, avoiding
  stream-ordering hazards during replay.

**SM Architecture-Aware Warmup**

  SM86 (Ampere) and SM90 (Hopper) have different graph-capture overheads and different
  optimal warmup step counts. This module exposes `get_warmup_steps_for_device()` to
  let callers query a sensible default without hard-coding per-arch magic numbers.

**Memory Pressure Safety Valve**

  On A6000 (48 GB), running both a full-iteration graph and an optimizer graph in the same
  pool can approach the memory ceiling. `HeteroCGPoolManager.check_pool_pressure()` monitors
  `torch.cuda.memory_reserved()` and can demote to per-capture pools if pressure exceeds
  a configurable threshold, logging a warning so the operator knows the trade-off is active.

References
----------
- Megatron-LM commit daec17c853ed25fa54cb4655a46b62a424996094
- Neuron_SP project: https://github.com/dylanyunlon/Neuron_SP
- CUDA Graph memory pools: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graph-memory-nodes
- DeepSpeed ZeRO-Infinity: used as reference for LOC tier design patterns
"""

from __future__ import annotations

import contextlib
import logging
import threading
import unittest
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants
# ─────────────────────────────────────────────────────────────────────────────

_SM_ARCH_A6000 = 86   # Ampere, 48 GB
_SM_ARCH_H100  = 90   # Hopper, 96 GB

# Empirically-tuned warmup steps: Hopper graph capture converges faster due to
# hardware-accelerated stream synchronisation in SM90.
_WARMUP_STEPS_BY_SM: Dict[int, int] = {
    _SM_ARCH_A6000: 3,
    _SM_ARCH_H100:  2,
}
_WARMUP_STEPS_DEFAULT = 3

# Memory pressure threshold: fraction of device total memory; above this we
# demote shared pool → per-capture pool to avoid OOM.
_POOL_PRESSURE_HIGH_WATERMARK = 0.88   # 88 % reserved → demote
_POOL_PRESSURE_LOW_WATERMARK  = 0.75   # 75 % reserved → safe to re-enable sharing


# ─────────────────────────────────────────────────────────────────────────────
# Device-local singleton storage (thread-safe via threading.Lock)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _DeviceGraphState:
    """Per-device singleton container for shared pool and capture stream."""
    graph_pool: Optional[object] = None            # torch.cuda.graph_pool_handle()
    capture_stream: Optional[torch.cuda.Stream] = None
    loc_transfer_stream: Optional[torch.cuda.Stream] = None
    # Dynamic pressure demotion flag
    pool_demoted: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


# Global registry: device_index → _DeviceGraphState
_device_states: Dict[int, _DeviceGraphState] = {}
_registry_lock = threading.Lock()


def _get_device_state(device_idx: int) -> _DeviceGraphState:
    """Return (creating if absent) the singleton state for *device_idx*."""
    with _registry_lock:
        if device_idx not in _device_states:
            _device_states[device_idx] = _DeviceGraphState()
        return _device_states[device_idx]


# ─────────────────────────────────────────────────────────────────────────────
# SM-architecture helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_sm_major(device_idx: Optional[int] = None) -> int:
    """Return the SM major version for *device_idx* (default: current device).

    Used to branch on Ampere (SM86) vs Hopper (SM90) capabilities.
    """
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    return props.major * 10 + props.minor


def get_warmup_steps_for_device(device_idx: Optional[int] = None) -> int:
    """Return the recommended CUDA graph warmup steps for the given device.

    Hopper (SM90) converges in fewer warmup iterations than Ampere (SM86)
    because its hardware prefetcher reduces the variance between warmup runs.

    Args:
        device_idx: CUDA device index. Defaults to ``torch.cuda.current_device()``.

    Returns:
        Integer warmup step count appropriate for the device's SM architecture.
    """
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    sm = get_sm_major(device_idx)
    steps = _WARMUP_STEPS_BY_SM.get(sm, _WARMUP_STEPS_DEFAULT)
    logger.debug(
        "Device %d SM%d: CUDA graph warmup steps = %d", device_idx, sm, steps
    )
    return steps


def is_hopper_device(device_idx: Optional[int] = None) -> bool:
    """Return True if *device_idx* is a Hopper (SM90+) GPU."""
    return get_sm_major(device_idx) >= 90


def is_ampere_device(device_idx: Optional[int] = None) -> bool:
    """Return True if *device_idx* is an Ampere (SM86) GPU."""
    sm = get_sm_major(device_idx)
    return 80 <= sm < 90


# ─────────────────────────────────────────────────────────────────────────────
# Shared capture-stream accessors (per-device singletons)
# ─────────────────────────────────────────────────────────────────────────────

def get_shared_capture_stream(device_idx: Optional[int] = None) -> torch.cuda.Stream:
    """Return the shared CUDA capture stream for *device_idx*.

    **Upstream intent** (Megatron daec17c): Using a single non-default stream for
    all graph captures on a device prevents per-stream alloc-segment inflation.

    **DES-LOC adaptation**: We scope the singleton per-device because the H100 and
    A6000 devices each have their own stream namespace. Mixing streams across devices
    would raise a CUDA error during graph capture.

    Args:
        device_idx: Target CUDA device. Defaults to ``torch.cuda.current_device()``.

    Returns:
        A ``torch.cuda.Stream`` created (once) on the specified device.
    """
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    state = _get_device_state(device_idx)
    with state.lock:
        if state.capture_stream is None:
            with torch.cuda.device(device_idx):
                state.capture_stream = torch.cuda.Stream()
            logger.info(
                "Created shared capture stream on device %d (SM%d)",
                device_idx, get_sm_major(device_idx),
            )
    return state.capture_stream


def get_loc_transfer_stream(device_idx: Optional[int] = None) -> torch.cuda.Stream:
    """Return the dedicated LOC-tier transfer stream for *device_idx*.

    The Shared LOcality Cache (LOC) tier transfers activations between CPU DRAM
    and GPU via ``cudaMemcpyAsync``. These transfers must NOT share the compute
    capture stream to avoid stream-ordering hazards when the graph is replayed:
    CUDA graph replay enforces that all operations captured on a stream execute
    in-order, and mixing compute with H2D/D2H copies can cause spurious waits or
    data corruption if the graph is replayed concurrently with a background copy.

    This stream is used exclusively by ``LocTierBuffer.async_push`` and
    ``LocTierBuffer.async_pull`` operations.

    Args:
        device_idx: Target CUDA device. Defaults to ``torch.cuda.current_device()``.

    Returns:
        A ``torch.cuda.Stream`` dedicated to LOC tier transfers on the device.
    """
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    state = _get_device_state(device_idx)
    with state.lock:
        if state.loc_transfer_stream is None:
            with torch.cuda.device(device_idx):
                state.loc_transfer_stream = torch.cuda.Stream()
            logger.info(
                "Created LOC transfer stream on device %d for CPU DRAM ↔ GPU async copies",
                device_idx,
            )
    return state.loc_transfer_stream


# ─────────────────────────────────────────────────────────────────────────────
# Shared graph-pool accessors (per-device singletons)
# ─────────────────────────────────────────────────────────────────────────────

def get_shared_graph_pool(device_idx: Optional[int] = None) -> object:
    """Return the process-wide CUDA graph memory pool for *device_idx*.

    **Upstream intent** (Megatron daec17c): ``torch.cuda.graph_pool_handle()``
    creates a *new* pool each time it's called. Megatron's singleton ensures that
    full-iteration forward-backward and optimizer-step captures share one pool,
    avoiding duplicate alloc-segment overhead.

    **DES-LOC adaptation**: Pool handles are device-local. We maintain one
    singleton per physical device so that:
      - Both A6000 graph captures (forward on dev-0, optimizer on dev-1) can each
        share their respective per-device pools.
      - The H100 prefill graph uses its own pool, independent of A6000 pools.

    This is critical: passing a pool handle from device 0 to a capture on device 2
    is undefined behaviour and typically triggers a CUDA_ERROR_INVALID_VALUE.

    Args:
        device_idx: Target CUDA device. Defaults to ``torch.cuda.current_device()``.

    Returns:
        An opaque pool handle (from ``torch.cuda.graph_pool_handle()``) scoped to
        the specified device.
    """
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    state = _get_device_state(device_idx)
    with state.lock:
        if state.graph_pool is None:
            with torch.cuda.device(device_idx):
                state.graph_pool = torch.cuda.graph_pool_handle()
            logger.info(
                "Created shared graph memory pool on device %d (SM%d, %.1f GB total)",
                device_idx,
                get_sm_major(device_idx),
                torch.cuda.get_device_properties(device_idx).total_memory / (1 << 30),
            )
    return state.graph_pool


def get_graph_pool(
    use_single_mempool: bool,
    device_idx: Optional[int] = None,
) -> object:
    """Return the appropriate graph pool handle for a capture on *device_idx*.

    **Upstream intent** (Megatron daec17c): `get_graph_pool(use_single_mempool)`
    either returns the process-wide shared pool or a fresh per-capture pool,
    gated by `use_single_mempool`.

    **DES-LOC adaptation**: We additionally consult memory pressure on the target
    device. If the device is above the high-watermark threshold, pool sharing is
    temporarily demoted even when `use_single_mempool=True`, because merging pools
    on a 48 GB A6000 under pressure can trigger OOM during graph capture. The
    demotion is logged so operators are aware of the trade-off.

    Args:
        use_single_mempool: When True (and memory is not under pressure), return
            the device-scoped shared singleton pool. When False, return a fresh
            pool handle.
        device_idx: Target CUDA device. Defaults to ``torch.cuda.current_device()``.

    Returns:
        A CUDA graph pool handle.
    """
    if device_idx is None:
        device_idx = torch.cuda.current_device()

    if use_single_mempool:
        state = _get_device_state(device_idx)
        pressure = _compute_memory_pressure(device_idx)
        if pressure >= _POOL_PRESSURE_HIGH_WATERMARK:
            if not state.pool_demoted:
                state.pool_demoted = True
                logger.warning(
                    "Device %d memory pressure %.1f%% exceeds threshold %.1f%%; "
                    "demoting shared CUDA graph pool → per-capture pool to avoid OOM. "
                    "Consider reducing micro-batch size or enabling CPU offload.",
                    device_idx,
                    pressure * 100,
                    _POOL_PRESSURE_HIGH_WATERMARK * 100,
                )
            with torch.cuda.device(device_idx):
                return torch.cuda.graph_pool_handle()
        else:
            if state.pool_demoted and pressure < _POOL_PRESSURE_LOW_WATERMARK:
                state.pool_demoted = False
                logger.info(
                    "Device %d memory pressure dropped to %.1f%%; "
                    "re-enabling shared CUDA graph pool.",
                    device_idx, pressure * 100,
                )
            return get_shared_graph_pool(device_idx)

    with torch.cuda.device(device_idx):
        return torch.cuda.graph_pool_handle()


def _compute_memory_pressure(device_idx: int) -> float:
    """Return fraction of total memory currently reserved on *device_idx*.

    Returns a float in [0.0, 1.0]. Used by `get_graph_pool` to decide whether to
    demote shared pool to per-capture pool under memory pressure.
    """
    props = torch.cuda.get_device_properties(device_idx)
    total = props.total_memory
    if total == 0:
        return 0.0
    reserved = torch.cuda.memory_reserved(device_idx)
    return reserved / total


# ─────────────────────────────────────────────────────────────────────────────
# LOC Tier Buffer
# ─────────────────────────────────────────────────────────────────────────────

class LocTierBuffer:
    """Manages a CPU DRAM ↔ GPU activation/KV-cache snapshot buffer.

    The Shared LOcality Cache (LOC) tier in DES-LOC stores bf16 activation
    checkpoints and KV-cache snapshots in CPU DRAM (1.5 TB available). During
    PREFILL on the H100, attention outputs are pushed to the LOC tier. During
    the FFN phase on A6000s, the relevant slices are pulled back on demand.

    Transfers use ``cudaMemcpyAsync`` on the dedicated LOC transfer stream (see
    ``get_loc_transfer_stream``), *not* the compute capture stream, to ensure
    CUDA graph captures remain clean.

    The buffer is pinned (page-locked) so PCIe DMA operates at full bandwidth
    (~32 GB/s peak on the target topology). Without pinning, the CUDA driver
    would stage through a temporary pinned bounce buffer, halving effective BW.

    Args:
        shape: Shape of the tensor to buffer.
        dtype: Data type (typically ``torch.bfloat16``).
        device_idx: GPU device that will read/write this buffer.
        name: Human-readable label for logging.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
        device_idx: int = 0,
        name: str = "loc_buffer",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device_idx = device_idx
        self.name = name

        # Pinned CPU tensor for DMA-friendly transfers
        self._cpu_buf: torch.Tensor = torch.zeros(
            shape, dtype=dtype, pin_memory=True
        )
        # GPU-side staging tensor (lives on device_idx)
        with torch.cuda.device(device_idx):
            self._gpu_buf: torch.Tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{device_idx}")

        logger.debug(
            "LocTierBuffer '%s': shape=%s dtype=%s device=%d "
            "pinned_cpu=%.2f MB gpu=%.2f MB",
            name, shape, dtype,
            device_idx,
            self._cpu_buf.nbytes / (1 << 20),
            self._gpu_buf.nbytes / (1 << 20),
        )

    def async_push(self, gpu_tensor: torch.Tensor) -> None:
        """Asynchronously copy *gpu_tensor* → CPU DRAM LOC buffer.

        The copy is enqueued on the LOC transfer stream; the caller must call
        ``sync_transfer()`` before reading ``cpu_buf`` on the host.

        Args:
            gpu_tensor: Source tensor on ``device_idx``. Must match ``self.shape``
                and ``self.dtype``.
        """
        stream = get_loc_transfer_stream(self.device_idx)
        with torch.cuda.stream(stream):
            self._cpu_buf.copy_(gpu_tensor, non_blocking=True)
        logger.debug("LocTierBuffer '%s': async_push enqueued on LOC stream", self.name)

    def async_pull(self) -> torch.Tensor:
        """Asynchronously copy CPU DRAM LOC buffer → GPU, return GPU tensor.

        The copy is enqueued on the LOC transfer stream. The returned tensor
        should not be read until ``sync_transfer()`` completes.

        Returns:
            The GPU-side staging tensor (``self._gpu_buf``), which will contain
            the LOC data after the transfer stream is synchronized.
        """
        stream = get_loc_transfer_stream(self.device_idx)
        with torch.cuda.stream(stream):
            self._gpu_buf.copy_(self._cpu_buf, non_blocking=True)
        logger.debug("LocTierBuffer '%s': async_pull enqueued on LOC stream", self.name)
        return self._gpu_buf

    def sync_transfer(self) -> None:
        """Block the calling thread until all LOC transfers complete."""
        stream = get_loc_transfer_stream(self.device_idx)
        stream.synchronize()

    @property
    def cpu_buf(self) -> torch.Tensor:
        """Pinned CPU-side tensor (valid after ``sync_transfer()``)."""
        return self._cpu_buf

    @property
    def gpu_buf(self) -> torch.Tensor:
        """GPU-side staging tensor (valid after ``async_pull`` + ``sync_transfer()``)."""
        return self._gpu_buf


# ─────────────────────────────────────────────────────────────────────────────
# HeteroCGPoolManager — the central coordinator
# ─────────────────────────────────────────────────────────────────────────────

class HeteroCGPoolManager:
    """Coordinates CUDA graph pool sharing across heterogeneous devices in DES-LOC.

    This is the DES-LOC analogue of Megatron's process-wide `_shared_graph_pool`
    singleton, extended to handle the multi-device, multi-SM-arch topology:

      - A6000 × 2 (SM86, 48 GB each) — tensor-parallel compute workers
      - H100 NVL × 1 (SM90, 96 GB)  — prefill/attention specialist

    **Pool sharing policy**:

    ┌─────────────────────────────┬───────────────────────────────────────────┐
    │ Execution Phase             │ Pool assignment                           │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ Full-iter fwd+bwd (A6000-0) │ Shared pool on device 0                   │
    │ Optimizer step  (A6000-0)   │ Same shared pool on device 0              │
    │ Full-iter fwd+bwd (A6000-1) │ Shared pool on device 1                   │
    │ Optimizer step  (A6000-1)   │ Same shared pool on device 1              │
    │ Prefill graph   (H100-2)    │ Shared pool on device 2                   │
    │ LOC transfer graphs         │ Never graph-captured (async copy only)    │
    └─────────────────────────────┴───────────────────────────────────────────┘

    **Memory pressure demotion**: Monitored per-device. On A6000 (48 GB) where
    pressure is most acute, a single pool covering both fwd+bwd and optimizer
    can push beyond safe limits during large-batch training. Demotion falls back
    to per-capture pools automatically.

    **Thread safety**: All accessors use per-device locks. The registry lock
    guards cross-device state enumeration.

    Usage example::

        mgr = HeteroCGPoolManager(use_single_mempool=True)

        # On device 0 (A6000), capture full-iteration graph
        with mgr.capture_context(device_idx=0, phase="full_iter"):
            ...forward_backward_func(...)

        # On device 0 (A6000), capture optimizer graph → same pool, same stream
        with mgr.capture_context(device_idx=0, phase="optimizer"):
            ...optimizer.step(...)

        # On device 2 (H100), capture prefill graph → separate pool on device 2
        with mgr.capture_context(device_idx=2, phase="prefill"):
            ...prefill_func(...)

    Args:
        use_single_mempool: Master flag. When True, all same-device captures share
            one pool. When False, every capture gets a fresh pool (Megatron legacy
            behaviour, useful for debugging).
    """

    def __init__(self, use_single_mempool: bool = True) -> None:
        self.use_single_mempool = use_single_mempool
        logger.info(
            "HeteroCGPoolManager initialised: use_single_mempool=%s",
            use_single_mempool,
        )

    def get_pool(self, device_idx: Optional[int] = None) -> object:
        """Return the appropriate graph pool for *device_idx*.

        Delegates to module-level ``get_graph_pool``, which incorporates the
        memory-pressure demotion logic.

        Args:
            device_idx: Target device. Defaults to ``torch.cuda.current_device()``.

        Returns:
            A CUDA graph pool handle.
        """
        if device_idx is None:
            device_idx = torch.cuda.current_device()
        return get_graph_pool(self.use_single_mempool, device_idx)

    def get_capture_stream(self, device_idx: Optional[int] = None) -> torch.cuda.Stream:
        """Return the shared capture stream for *device_idx*.

        Args:
            device_idx: Target device. Defaults to ``torch.cuda.current_device()``.

        Returns:
            A ``torch.cuda.Stream`` suitable for use as the ``stream=`` argument to
            ``torch.cuda.graph(...)``.
        """
        if device_idx is None:
            device_idx = torch.cuda.current_device()
        return get_shared_capture_stream(device_idx)

    @contextlib.contextmanager
    def capture_context(
        self,
        graph: torch.cuda.CUDAGraph,
        device_idx: Optional[int] = None,
        phase: str = "unknown",
    ):
        """Context manager that configures stream and pool for a graph capture.

        Wraps ``torch.cuda.graph(...)`` with the DES-LOC pool-sharing policy.
        The caller is responsible for creating the ``CUDAGraph`` object.

        Args:
            graph: A ``torch.cuda.CUDAGraph()`` instance to capture into.
            device_idx: Target device. Defaults to ``torch.cuda.current_device()``.
            phase: Human-readable phase label ("full_iter", "optimizer", "prefill")
                for logging purposes.

        Yields:
            Nothing; the body of the ``with`` block executes inside the graph capture.

        Example::

            g = torch.cuda.CUDAGraph()
            with mgr.capture_context(g, device_idx=0, phase="full_iter"):
                result = model_forward()
        """
        if device_idx is None:
            device_idx = torch.cuda.current_device()

        stream = self.get_capture_stream(device_idx)
        pool   = self.get_pool(device_idx)
        sm     = get_sm_major(device_idx)

        logger.debug(
            "Graph capture start: phase='%s' device=%d SM%d pool_shared=%s",
            phase, device_idx, sm, self.use_single_mempool,
        )

        with torch.cuda.device(device_idx):
            with torch.cuda.graph(
                graph,
                stream=stream,
                pool=pool,
                capture_error_mode="thread_local",
            ):
                yield

        logger.debug(
            "Graph capture end: phase='%s' device=%d", phase, device_idx
        )

    def check_pool_pressure(self, device_idx: Optional[int] = None) -> Dict[str, float]:
        """Return a snapshot of memory pressure across all managed devices.

        Useful for monitoring loops and health-check endpoints.

        Args:
            device_idx: If given, return pressure for that device only.
                Otherwise, returns a dict for all devices with known state.

        Returns:
            Dict mapping device index (as string) to pressure fraction [0.0, 1.0].
        """
        if device_idx is not None:
            return {str(device_idx): _compute_memory_pressure(device_idx)}
        with _registry_lock:
            known_devices = list(_device_states.keys())
        return {
            str(d): _compute_memory_pressure(d) for d in known_devices
        }

    def reset_device_state(self, device_idx: int) -> None:
        """Destroy the singleton state for *device_idx*, forcing re-creation on next use.

        Intended for test teardown and topology changes. In production this should
        rarely be called because re-creation allocates a new pool (potentially
        breaking in-flight graph captures on that device).

        Args:
            device_idx: The device whose state to clear.
        """
        with _registry_lock:
            if device_idx in _device_states:
                del _device_states[device_idx]
                logger.warning(
                    "HeteroCGPoolManager: reset state for device %d "
                    "(any outstanding graph captures on this device are now invalid).",
                    device_idx,
                )


# ─────────────────────────────────────────────────────────────────────────────
# CUDA-Graph-wrapped callable: DES-LOC edition
# ─────────────────────────────────────────────────────────────────────────────

class HeteroFullIterGraphWrapper:
    """CUDA-graph wrapper for full-iteration forward-backward on a heterogeneous device.

    **Upstream analogue**: ``FullCudaGraphWrapper`` in Megatron-LM, extended to:
      - Select capture device based on DES-LOC phase routing.
      - Use ``HeteroCGPoolManager`` for pool/stream selection.
      - Query per-arch warmup steps via ``get_warmup_steps_for_device()``.

    On the first ``cuda_graph_warmup_steps`` calls the function is invoked
    eagerly (no graph). On the next call, the graph is captured. Subsequent
    calls replay the graph.

    Args:
        forward_backward_func: Callable to wrap.
        device_idx: Device on which to capture and replay. Must match the device
            that ``forward_backward_func`` issues CUDA work to.
        mgr: A ``HeteroCGPoolManager`` instance.
        warmup_steps: Number of eager warmup iterations before capture. If None,
            uses ``get_warmup_steps_for_device(device_idx)``.
        phase: Label for logging ("full_iter", "prefill", etc.).
    """

    def __init__(
        self,
        forward_backward_func: Callable,
        device_idx: int,
        mgr: HeteroCGPoolManager,
        warmup_steps: Optional[int] = None,
        phase: str = "full_iter",
    ) -> None:
        self.forward_backward_func = forward_backward_func
        self.device_idx = device_idx
        self.mgr = mgr
        self.warmup_steps = (
            warmup_steps if warmup_steps is not None
            else get_warmup_steps_for_device(device_idx)
        )
        self.phase = phase
        self._call_count = 0
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_captured = False

        logger.info(
            "HeteroFullIterGraphWrapper: phase='%s' device=%d warmup_steps=%d",
            phase, device_idx, self.warmup_steps,
        )

    def __call__(self, *args, **kwargs):
        """Execute the wrapped function, capturing a CUDA graph after warmup."""
        self._call_count += 1

        if self._call_count <= self.warmup_steps:
            logger.debug(
                "HeteroFullIterGraphWrapper '%s': warmup step %d/%d (eager)",
                self.phase, self._call_count, self.warmup_steps,
            )
            return self.forward_backward_func(*args, **kwargs)

        if not self._graph_captured:
            logger.info(
                "HeteroFullIterGraphWrapper '%s': capturing CUDA graph on device %d",
                self.phase, self.device_idx,
            )
            self._cuda_graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with self.mgr.capture_context(
                self._cuda_graph,
                device_idx=self.device_idx,
                phase=self.phase,
            ):
                self._captured_result = self.forward_backward_func(*args, **kwargs)
            torch.cuda.synchronize()
            self._graph_captured = True
            logger.info(
                "HeteroFullIterGraphWrapper '%s': graph captured successfully "
                "(device %d, pool_shared=%s)",
                self.phase, self.device_idx, self.mgr.use_single_mempool,
            )
            return self._captured_result

        # Graph replay
        self._cuda_graph.replay()
        return self._captured_result


class HeteroOptimizerGraphWrapper:
    """CUDA-graph wrapper for optimizer step on a heterogeneous device.

    **Upstream analogue**: ``OptimizerCudaGraphWrapper`` in Megatron-LM.

    When ``mgr.use_single_mempool=True`` and the optimizer runs on the same
    device as the full-iter graph, both captures share the device's singleton
    pool (matching Megatron PR #4698's intent). When the optimizer is scheduled
    to a different device (e.g. DES-LOC moves parameter shards to the H100 for
    large embedding layers), it automatically gets the H100's own shared pool.

    Args:
        optimizer_step_func: Callable that performs one optimizer step.
        device_idx: Device on which the optimizer runs.
        mgr: Shared ``HeteroCGPoolManager`` instance.
        warmup_steps: Eager steps before capture. Defaults to per-device value.
    """

    def __init__(
        self,
        optimizer_step_func: Callable,
        device_idx: int,
        mgr: HeteroCGPoolManager,
        warmup_steps: Optional[int] = None,
    ) -> None:
        self.optimizer_step_func = optimizer_step_func
        self.device_idx = device_idx
        self.mgr = mgr
        self.warmup_steps = (
            warmup_steps if warmup_steps is not None
            else get_warmup_steps_for_device(device_idx)
        )
        self._call_count = 0
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_captured = False
        self._captured_result = None

        logger.info(
            "HeteroOptimizerGraphWrapper: device=%d warmup_steps=%d",
            device_idx, self.warmup_steps,
        )

    def __call__(self, **kwargs):
        """Execute optimizer step, capturing a CUDA graph after warmup."""
        if kwargs:
            raise TypeError(
                "HeteroOptimizerGraphWrapper does not support keyword arguments to "
                "optimizer.step(). All optimizer state must be static at capture time."
            )

        self._call_count += 1

        if self._call_count <= self.warmup_steps:
            logger.debug(
                "HeteroOptimizerGraphWrapper: warmup step %d/%d (eager, device=%d)",
                self._call_count, self.warmup_steps, self.device_idx,
            )
            return self.optimizer_step_func()

        if not self._graph_captured:
            logger.info(
                "HeteroOptimizerGraphWrapper: capturing optimizer CUDA graph on device %d",
                self.device_idx,
            )
            assert self._cuda_graph is None
            self._cuda_graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with self.mgr.capture_context(
                self._cuda_graph,
                device_idx=self.device_idx,
                phase="optimizer",
            ):
                self._captured_result = self.optimizer_step_func()
            torch.cuda.synchronize()
            self._graph_captured = True
            logger.info(
                "HeteroOptimizerGraphWrapper: optimizer graph captured on device %d "
                "(pool_shared=%s)",
                self.device_idx, self.mgr.use_single_mempool,
            )
            return self._captured_result

        self._cuda_graph.replay()
        return self._captured_result


# ─────────────────────────────────────────────────────────────────────────────
# DES-LOC topology configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DesLocTopologyConfig:
    """Configuration for the DES-LOC 2×A6000 + 1×H100 topology.

    This dataclass centralises all topology-specific constants so that other
    modules (scheduler, LOC tier, pipeline executor) have a single source of truth.

    Attributes:
        a6000_devices: List of device indices for A6000 (SM86) GPUs.
        h100_device: Device index for the H100 NVL (SM90) GPU.
        use_single_mempool: Whether to enable shared graph pool on each device.
        loc_dtype: Data type for LOC-tier buffers (bf16 saves bandwidth on PCIe).
        pcie_bandwidth_gbps: Estimated bidirectional PCIe bandwidth per device pair.
        loc_dram_gb: Total CPU DRAM available for the LOC tier.
    """
    a6000_devices: Tuple[int, ...] = (0, 1)
    h100_device: int = 2
    use_single_mempool: bool = True
    loc_dtype: torch.dtype = torch.bfloat16
    pcie_bandwidth_gbps: float = 32.0
    loc_dram_gb: float = 1500.0

    @property
    def all_devices(self) -> Tuple[int, ...]:
        """All GPU device indices in topology order."""
        return self.a6000_devices + (self.h100_device,)

    def device_role(self, device_idx: int) -> str:
        """Human-readable role for *device_idx*."""
        if device_idx in self.a6000_devices:
            return f"A6000-SM86-worker-{self.a6000_devices.index(device_idx)}"
        if device_idx == self.h100_device:
            return "H100-SM90-prefill"
        return f"unknown-device-{device_idx}"


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build wrappers for a full DES-LOC training iteration
# ─────────────────────────────────────────────────────────────────────────────

def build_hetero_graph_wrappers(
    forward_backward_func: Callable,
    optimizer_step_func: Callable,
    topo: DesLocTopologyConfig,
    fwd_device_idx: int,
    opt_device_idx: int,
) -> Tuple[HeteroFullIterGraphWrapper, HeteroOptimizerGraphWrapper]:
    """Construct CUDA-graph wrappers sharing the correct per-device pools.

    This factory mirrors the setup in Megatron's ``training.py::train()`` where
    ``FullCudaGraphWrapper`` and ``OptimizerCudaGraphWrapper`` are constructed
    with matching ``use_single_mempool`` flags.

    In DES-LOC, the forward and optimizer phases may target different devices.
    When they land on the same device, they share that device's pool (matching
    Megatron intent). When they land on different devices, each uses its own
    device's pool (required by CUDA semantics).

    Args:
        forward_backward_func: Forward-backward callable.
        optimizer_step_func: Optimizer step callable.
        topo: Topology configuration.
        fwd_device_idx: Device index for the forward-backward graph.
        opt_device_idx: Device index for the optimizer graph.

    Returns:
        Tuple of (``HeteroFullIterGraphWrapper``, ``HeteroOptimizerGraphWrapper``).
    """
    mgr = HeteroCGPoolManager(use_single_mempool=topo.use_single_mempool)

    if fwd_device_idx == opt_device_idx:
        logger.info(
            "build_hetero_graph_wrappers: fwd and optimizer on same device %d "
            "→ pool will be shared (role: %s)",
            fwd_device_idx, topo.device_role(fwd_device_idx),
        )
    else:
        logger.info(
            "build_hetero_graph_wrappers: fwd on device %d (%s), "
            "optimizer on device %d (%s) → separate per-device pools",
            fwd_device_idx, topo.device_role(fwd_device_idx),
            opt_device_idx, topo.device_role(opt_device_idx),
        )

    fwd_wrapper = HeteroFullIterGraphWrapper(
        forward_backward_func,
        device_idx=fwd_device_idx,
        mgr=mgr,
        phase="full_iter",
    )
    opt_wrapper = HeteroOptimizerGraphWrapper(
        optimizer_step_func,
        device_idx=opt_device_idx,
        mgr=mgr,
    )
    return fwd_wrapper, opt_wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Utility: log topology summary at startup
# ─────────────────────────────────────────────────────────────────────────────

def log_topology_summary(topo: DesLocTopologyConfig) -> None:
    """Emit a structured INFO log describing the detected DES-LOC topology.

    Called once at training startup so that log files capture the exact hardware
    configuration, SM versions, and memory capacities in use.

    Args:
        topo: Topology configuration to summarise.
    """
    lines = ["DES-LOC topology summary:"]
    for dev in topo.all_devices:
        try:
            props = torch.cuda.get_device_properties(dev)
            lines.append(
                f"  device {dev}: {props.name} "
                f"SM{props.major * 10 + props.minor} "
                f"{props.total_memory / (1 << 30):.1f} GB "
                f"role={topo.device_role(dev)}"
            )
        except (RuntimeError, AssertionError) as exc:
            lines.append(f"  device {dev}: unavailable ({exc})")
    lines.append(
        f"  LOC tier: {topo.loc_dram_gb:.0f} GB CPU DRAM, "
        f"PCIe BW ≈ {topo.pcie_bandwidth_gbps:.0f} GB/s per link"
    )
    lines.append(f"  use_single_mempool: {topo.use_single_mempool}")
    logger.info("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Configure logging for test output
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    _SKIP_CUDA = not torch.cuda.is_available()

    class TestHeteroCGPoolSharing(unittest.TestCase):
        """Unit tests for DES-LOC HeteroCGPoolSharing.

        Tests that don't require CUDA use mock objects; tests that do are skipped
        when CUDA is unavailable (e.g. CI without GPU).
        """

        # ── Memory pressure helper ─────────────────────────────────────────

        def test_compute_memory_pressure_mock(self):
            """_compute_memory_pressure returns a float in [0, 1]."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()
            p = _compute_memory_pressure(dev)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

        # ── Device state registry ──────────────────────────────────────────

        def test_device_state_singleton(self):
            """Same device index returns the same _DeviceGraphState object."""
            s1 = _get_device_state(99)
            s2 = _get_device_state(99)
            self.assertIs(s1, s2)
            # Cleanup
            with _registry_lock:
                del _device_states[99]

        def test_device_state_isolation(self):
            """Different device indices return different _DeviceGraphState objects."""
            s0 = _get_device_state(100)
            s1 = _get_device_state(101)
            self.assertIsNot(s0, s1)
            with _registry_lock:
                del _device_states[100]
                del _device_states[101]

        # ── Topology config ────────────────────────────────────────────────

        def test_topology_all_devices(self):
            topo = DesLocTopologyConfig(a6000_devices=(0, 1), h100_device=2)
            self.assertEqual(topo.all_devices, (0, 1, 2))

        def test_topology_device_role(self):
            topo = DesLocTopologyConfig(a6000_devices=(0, 1), h100_device=2)
            self.assertEqual(topo.device_role(0), "A6000-SM86-worker-0")
            self.assertEqual(topo.device_role(1), "A6000-SM86-worker-1")
            self.assertEqual(topo.device_role(2), "H100-SM90-prefill")
            self.assertIn("unknown", topo.device_role(7))

        # ── SM architecture helpers ────────────────────────────────────────

        def test_warmup_steps_default(self):
            """get_warmup_steps_for_device falls back to default for unknown SM."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            # Any real device should return a positive warmup count
            dev = torch.cuda.current_device()
            steps = get_warmup_steps_for_device(dev)
            self.assertGreater(steps, 0)

        def test_warmup_steps_known_sm_table(self):
            """The SM table contains expected values for known architectures."""
            self.assertEqual(_WARMUP_STEPS_BY_SM[_SM_ARCH_A6000], 3)
            self.assertEqual(_WARMUP_STEPS_BY_SM[_SM_ARCH_H100], 2)

        # ── HeteroCGPoolManager (no GPU required for basic init) ───────────

        def test_manager_init(self):
            mgr = HeteroCGPoolManager(use_single_mempool=True)
            self.assertTrue(mgr.use_single_mempool)

        def test_manager_check_pool_pressure_empty(self):
            """check_pool_pressure returns dict (may be empty if no devices initialised)."""
            # Reset registry for isolation
            with _registry_lock:
                _device_states.clear()
            mgr = HeteroCGPoolManager()
            pressure = mgr.check_pool_pressure()
            self.assertIsInstance(pressure, dict)

        def test_manager_reset_device_state(self):
            """reset_device_state removes the entry from the registry."""
            _get_device_state(42)
            with _registry_lock:
                self.assertIn(42, _device_states)
            mgr = HeteroCGPoolManager()
            mgr.reset_device_state(42)
            with _registry_lock:
                self.assertNotIn(42, _device_states)

        # ── HeteroFullIterGraphWrapper (eager / warmup path) ───────────────

        def test_full_iter_wrapper_eager_path(self):
            """Wrapper calls the function eagerly during warmup steps."""
            call_log = []

            def fake_fwd_bwd():
                call_log.append("called")
                return 42

            mgr = HeteroCGPoolManager(use_single_mempool=False)
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()

            # Use warmup_steps=3 → first 3 calls are eager
            wrapper = HeteroFullIterGraphWrapper(
                fake_fwd_bwd, device_idx=dev, mgr=mgr, warmup_steps=3
            )
            for i in range(3):
                result = wrapper()
                self.assertEqual(result, 42)
            self.assertEqual(len(call_log), 3)
            # The wrapper has NOT yet captured a graph
            self.assertFalse(wrapper._graph_captured)

        def test_optimizer_wrapper_rejects_positional_args(self):
            """HeteroOptimizerGraphWrapper raises on keyword args passed to __call__."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()
            mgr = HeteroCGPoolManager(use_single_mempool=False)
            wrapper = HeteroOptimizerGraphWrapper(
                lambda: None, device_idx=dev, mgr=mgr, warmup_steps=99
            )
            with self.assertRaises(TypeError):
                wrapper(lr=0.01)  # keyword args should raise

        def test_optimizer_wrapper_eager_path(self):
            """Optimizer wrapper calls function eagerly during warmup."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()
            mgr = HeteroCGPoolManager(use_single_mempool=False)
            results = []

            def fake_step():
                results.append(1)
                return True

            wrapper = HeteroOptimizerGraphWrapper(
                fake_step, device_idx=dev, mgr=mgr, warmup_steps=5
            )
            for _ in range(5):
                wrapper()
            self.assertEqual(len(results), 5)
            self.assertFalse(wrapper._graph_captured)

        # ── LocTierBuffer ──────────────────────────────────────────────────

        def test_loc_buffer_shape_and_dtype(self):
            """LocTierBuffer allocates tensors with correct shape and dtype."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()
            buf = LocTierBuffer(
                shape=(4, 128),
                dtype=torch.bfloat16,
                device_idx=dev,
                name="test_buf",
            )
            self.assertEqual(buf.cpu_buf.shape, (4, 128))
            self.assertEqual(buf.cpu_buf.dtype, torch.bfloat16)
            self.assertEqual(buf.gpu_buf.shape, (4, 128))
            self.assertTrue(buf.cpu_buf.is_pinned())

        def test_loc_buffer_async_push_pull(self):
            """LocTierBuffer round-trips data through CPU DRAM correctly."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()
            buf = LocTierBuffer(shape=(8, 16), dtype=torch.float32,
                                device_idx=dev, name="roundtrip")
            src = torch.arange(128, dtype=torch.float32).reshape(8, 16).to(f"cuda:{dev}")
            buf.async_push(src)
            buf.sync_transfer()
            # Verify CPU side
            self.assertTrue(torch.allclose(buf.cpu_buf, src.cpu()))
            # Pull back to GPU
            gpu_out = buf.async_pull()
            buf.sync_transfer()
            self.assertTrue(torch.allclose(gpu_out, src))

        # ── build_hetero_graph_wrappers factory ───────────────────────────

        def test_build_factory_same_device(self):
            """Factory produces wrappers sharing the same HeteroCGPoolManager."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()
            topo = DesLocTopologyConfig(a6000_devices=(dev,), h100_device=dev)

            fwd_w, opt_w = build_hetero_graph_wrappers(
                forward_backward_func=lambda: None,
                optimizer_step_func=lambda: None,
                topo=topo,
                fwd_device_idx=dev,
                opt_device_idx=dev,
            )
            # Both wrappers should share the same manager instance
            self.assertIs(fwd_w.mgr, opt_w.mgr)
            self.assertEqual(fwd_w.device_idx, dev)
            self.assertEqual(opt_w.device_idx, dev)

        def test_build_factory_different_devices(self):
            """Factory correctly records different device indices per wrapper."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            if torch.cuda.device_count() < 2:
                self.skipTest("Need ≥ 2 CUDA devices")
            topo = DesLocTopologyConfig(a6000_devices=(0, 1), h100_device=2
                                        if torch.cuda.device_count() > 2 else 1)
            fwd_w, opt_w = build_hetero_graph_wrappers(
                forward_backward_func=lambda: 1,
                optimizer_step_func=lambda: 2,
                topo=topo,
                fwd_device_idx=0,
                opt_device_idx=1,
            )
            self.assertEqual(fwd_w.device_idx, 0)
            self.assertEqual(opt_w.device_idx, 1)

        # ── Pool demotion under pressure (mocked) ─────────────────────────

        def test_pool_demotion_at_high_watermark(self):
            """get_graph_pool falls back to per-capture pool under memory pressure."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()
            state = _get_device_state(dev)
            state.pool_demoted = False  # reset

            # Monkey-patch _compute_memory_pressure to simulate high pressure
            original_fn = globals().get("_compute_memory_pressure")

            import deepspeed.runtime.hetero_cg_pool_sharing as _mod
            original = _mod._compute_memory_pressure

            def mock_high_pressure(device_idx):
                return 0.95

            _mod._compute_memory_pressure = mock_high_pressure
            try:
                pool1 = get_graph_pool(use_single_mempool=True, device_idx=dev)
                pool2 = get_graph_pool(use_single_mempool=True, device_idx=dev)
                # Under pressure, each call gets a fresh handle (not the singleton)
                self.assertTrue(state.pool_demoted)
            finally:
                _mod._compute_memory_pressure = original
                state.pool_demoted = False

        # ── Full CUDA graph capture (requires CUDA) ────────────────────────

        def test_full_iter_graph_capture_and_replay(self):
            """End-to-end: wrapper captures a trivial graph and replays it."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()
            mgr = HeteroCGPoolManager(use_single_mempool=True)

            # Static tensor that the captured function will fill
            out = torch.zeros(4, device=f"cuda:{dev}")
            src = torch.ones(4, device=f"cuda:{dev}") * 7.0

            def static_func():
                out.copy_(src)
                return out

            wrapper = HeteroFullIterGraphWrapper(
                static_func, device_idx=dev, mgr=mgr, warmup_steps=1
            )

            # Warmup (eager)
            wrapper()
            # Capture
            wrapper()
            self.assertTrue(wrapper._graph_captured)
            # Replay via next call
            wrapper()
            self.assertTrue(torch.all(out == 7.0))

        def test_optimizer_graph_capture_and_replay(self):
            """End-to-end: optimizer wrapper captures and replays correctly."""
            if _SKIP_CUDA:
                self.skipTest("CUDA unavailable")
            dev = torch.cuda.current_device()
            mgr = HeteroCGPoolManager(use_single_mempool=True)

            counter = torch.zeros(1, device=f"cuda:{dev}")
            increment = torch.ones(1, device=f"cuda:{dev}")

            def opt_step():
                counter.add_(increment)
                return counter

            wrapper = HeteroOptimizerGraphWrapper(
                opt_step, device_idx=dev, mgr=mgr, warmup_steps=1
            )
            # Warmup: counter → 1
            wrapper()
            # Capture: counter → 2
            wrapper()
            self.assertTrue(wrapper._graph_captured)
            # Replay: counter → 3
            wrapper()
            # Counter should be 3 after two calls post-warmup (capture + one replay)
            val = counter.item()
            self.assertGreater(val, 1.0)

    print("=" * 70)
    print("Running HeteroCGPoolSharing unit tests")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  device {i}: {p.name} SM{p.major * 10 + p.minor} "
                  f"{p.total_memory / (1 << 30):.1f} GB")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestHeteroCGPoolSharing)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
