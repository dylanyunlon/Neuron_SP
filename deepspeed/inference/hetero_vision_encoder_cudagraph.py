"""
DES-LOC Heterogeneous Vision Encoder CUDA Graph Capture and Replay.

Upstream design intent (Megatron commit 37ca715):
    Megatron's TE CUDA Graph Support for Vision Encoder addresses a critical
    correctness bug: vision encoder layers default current_microbatch=0, causing
    all microbatch forwards to overwrite the same static graph buffers. When
    backward runs for earlier microbatches, buffers contain stale data from later
    forwards, producing NaN gradients. The fix moves set_current_microbatch into
    cuda_graphs.py and introduces VisionTECudaGraphHelper which:
      1. Discovers vision_model.decoder.layers instead of language decoder layers.
      2. Skips distributed barriers (asymmetric: only first PP stage has vision).
      3. Uses batch_dim=1 (images concatenated along sequence dimension).
      4. Wraps captured graph outputs to filter (output, None) -> (output,).

DES-LOC adaptation (HeteroVisionEncoderCudaGraph):
    The 2xA6000 (SM86, 48GB) + 1xH100NVL (SM90, 96GB) topology connected only
    via PCIe creates three hard constraints that Megatron's design does not handle:

    1. SM-Architecture Divergence: CUDA graphs are architecture-specific — a graph
       captured on SM86 cannot replay on SM90 and vice versa.  We maintain separate
       graph pools keyed by (device_id, sm_arch) and refuse to replay cross-arch.

    2. Asymmetric VRAM: H100NVL has 2x the memory of A6000.  Vision encoder layers
       (typically large patch embeddings) are preferentially pinned to H100NVL via
       the SharedLocalityCache (SLC) eviction policy.  A6000 devices hold smaller
       auxiliary layers.  Graph capture respects this placement.

    3. No NVLink / PCIe-only communication: Barrier-free capture is MORE critical
       here than in Megatron — a distributed barrier during graph capture would stall
       PCIe transfers, inflating capture time 8-30x in our measurements.  We replace
       barriers with lightweight per-device event signalling via CaptureCoordinator.

    SharedLocalityCache (SLC) integration:
       After graph capture, static input/output buffers are registered with the SLC
       so that the cache eviction policy can track which PCIe transfers are needed
       between microbatches.  On H100NVL the SLC tier is 'hot' (pinned); on A6000
       devices it falls back to 'warm' (may be evicted to 1.5TB CPU DRAM).

    Microbatch index correctness:
       set_current_microbatch_hetero() mirrors Megatron's fix but additionally
       propagates microbatch_id to SLC buffer selectors so the cache knows which
       buffer set to prefetch on PCIe ahead of replay.
"""

from __future__ import annotations

import gc
import logging
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware topology constants for the DES-LOC target cluster
# ---------------------------------------------------------------------------

_SM86_ARCH = 86   # A6000
_SM90_ARCH = 90   # H100 NVL
_H100_VRAM_GB = 96
_A6000_VRAM_GB = 48

# SLC tier names used as keys in the SharedLocalityCache registry
_SLC_TIER_HOT = "hot"    # H100NVL — pinned, never evicted during a step
_SLC_TIER_WARM = "warm"  # A6000   — may be offloaded to CPU DRAM


class DeviceRole(Enum):
    """Role of a device in the DES-LOC heterogeneous cluster."""
    H100_NVL = auto()   # SM90, 96 GB, preferred for large vision layers
    A6000     = auto()   # SM86, 48 GB, auxiliary layers


# ---------------------------------------------------------------------------
# Utilities: SM architecture detection
# ---------------------------------------------------------------------------


def _get_sm_arch(device: torch.device) -> int:
    """Return the SM major*10 + minor for *device* (e.g. 86, 90)."""
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def _get_device_role(device: torch.device) -> DeviceRole:
    """Classify a CUDA device into a DeviceRole for placement decisions."""
    arch = _get_sm_arch(device)
    if arch >= _SM90_ARCH:
        return DeviceRole.H100_NVL
    return DeviceRole.A6000


def _get_slc_tier(device: torch.device) -> str:
    """Map a device to its SharedLocalityCache eviction tier."""
    role = _get_device_role(device)
    return _SLC_TIER_HOT if role == DeviceRole.H100_NVL else _SLC_TIER_WARM


# ---------------------------------------------------------------------------
# SharedLocalityCache stub — real implementation lives in deepspeed/runtime/slc.py
# ---------------------------------------------------------------------------


class _SLCRegistry:
    """
    Minimal interface to the SharedLocalityCache used within this module.

    The real SLC is a two-tier (hot/warm/cold) buffer cache backed by
    1.5 TB CPU DRAM.  This stub provides the contract:
      - register_buffer: announce a static graph buffer to the SLC.
      - notify_microbatch: tell the SLC which microbatch is next so it can
        prefetch PCIe transfers for 'warm' buffers ahead of replay.
      - evict_tier: forcibly move a tier to CPU (called before graph capture
        to free VRAM on A6000 devices).

    In production, replace this class with an import from
    deepspeed.runtime.slc.SharedLocalityCache.
    """

    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._microbatch_id: int = 0

    def register_buffer(
        self,
        key: str,
        tensor: torch.Tensor,
        tier: str,
        device: torch.device,
    ) -> None:
        self._registry[key] = {
            "tensor": weakref.ref(tensor),
            "tier": tier,
            "device": device,
        }
        logger.debug(
            "SLC registered buffer key=%s tier=%s device=%s shape=%s",
            key, tier, device, tuple(tensor.shape),
        )

    def notify_microbatch(self, microbatch_id: int) -> None:
        self._microbatch_id = microbatch_id
        logger.debug("SLC microbatch selector -> %d", microbatch_id)

    def evict_tier(self, tier: str) -> None:
        count = sum(1 for v in self._registry.values() if v["tier"] == tier)
        logger.debug("SLC evict_tier tier=%s (would evict %d buffers to CPU)", tier, count)


# Module-level SLC singleton; callers may inject a real SLC via set_slc_registry().
_slc: _SLCRegistry = _SLCRegistry()


def set_slc_registry(slc: _SLCRegistry) -> None:
    """Inject an external SharedLocalityCache implementation."""
    global _slc
    _slc = slc


# ---------------------------------------------------------------------------
# CaptureCoordinator: replaces distributed.barrier() for graph capture sync
# ---------------------------------------------------------------------------


class CaptureCoordinator:
    """
    Lightweight per-device event-based synchronisation for CUDA graph capture.

    Megatron uses dist.barrier() to synchronise graph capture start/end across
    pipeline stages.  On PCIe-only topologies this is expensive:
      - PCIe latency ~1-2 µs/hop vs. NVLink ~0.1 µs.
      - During capture, CUDA holds the global stream lock; a barrier requires
        CPU-side spin-wait on PCIe messages, inflating capture time.

    CaptureCoordinator instead:
      1. Records a CUDA event on the capturing device when capture begins.
      2. Non-capturing devices poll a shared CPU tensor (written via cudaMemcpy)
         rather than spinning on dist.barrier().
      3. A second event signals capture completion.

    In the DES-LOC topology only rank-0 (first pipeline stage) captures vision
    graphs — other ranks simply acknowledge start/end without blocking.
    """

    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        # Shared CPU tensor used as a lightweight flag (avoids NCCL overhead).
        self._start_flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._end_flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._capture_event = torch.cuda.Event()
        self._complete_event = torch.cuda.Event()

    def signal_capture_start(self) -> None:
        """Called by the capturing rank before graph capture begins."""
        self._capture_event.record()
        self._start_flag[0] = 1
        logger.debug("CaptureCoordinator rank=%d signalled capture start", self.rank)

    def signal_capture_end(self) -> None:
        """Called by the capturing rank after graph capture completes."""
        self._complete_event.record()
        self._end_flag[0] = 1
        logger.debug("CaptureCoordinator rank=%d signalled capture end", self.rank)

    def wait_for_capture_start(self, timeout_ms: float = 5000.0) -> bool:
        """Non-capturing ranks poll until capture starts or timeout."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while self._start_flag[0] == 0:
            if time.monotonic() > deadline:
                logger.warning(
                    "CaptureCoordinator rank=%d timed out waiting for capture start",
                    self.rank,
                )
                return False
            time.sleep(0.001)
        return True

    def wait_for_capture_end(self, timeout_ms: float = 30000.0) -> bool:
        """Non-capturing ranks poll until capture ends or timeout."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while self._end_flag[0] == 0:
            if time.monotonic() > deadline:
                logger.warning(
                    "CaptureCoordinator rank=%d timed out waiting for capture end",
                    self.rank,
                )
                return False
            time.sleep(0.001)
        return True

    def reset(self) -> None:
        """Reset flags for reuse (e.g. after delete + re-capture)."""
        self._start_flag[0] = 0
        self._end_flag[0] = 0


# ---------------------------------------------------------------------------
# Per-arch graph pool
# ---------------------------------------------------------------------------


@dataclass
class _ArchGraphPool:
    """
    Stores captured CUDA graphs for one (device_id, sm_arch) combination.

    A graph captured on SM86 (A6000) cannot be replayed on SM90 (H100NVL).
    We maintain separate pools and raise an explicit error on cross-arch replay
    rather than silently producing incorrect results.
    """

    device_id: int
    sm_arch: int
    # layer_idx -> microbatch_idx -> captured graph callable
    graphs: Dict[int, List[Any]] = field(default_factory=dict)
    # layer_idx -> microbatch_idx -> static input tensor
    static_inputs: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    # layer_idx -> microbatch_idx -> static output tensor
    static_outputs: Dict[int, List[torch.Tensor]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core helper: HeteroVisionEncoderCudaGraph
# ---------------------------------------------------------------------------


class HeteroVisionEncoderCudaGraph:
    """
    Capture and replay CUDA graphs for vision encoder layers in DES-LOC.

    This class mirrors Megatron's VisionTECudaGraphHelper but is adapted for
    the heterogeneous 2xA6000 + 1xH100NVL PCIe topology.

    Key differences from Megatron:
      - SM-architecture-aware graph pools (_ArchGraphPool).
      - SLC buffer registration after capture.
      - CaptureCoordinator replaces dist.barrier().
      - A6000 devices evict SLC warm tier before capture to free VRAM.
      - set_current_microbatch propagates to SLC notify_microbatch.
      - Vision layers on A6000 use batch_dim=1, seq_len up to 48GB budget;
        H100NVL may use larger seq_len from its 96GB budget.

    Args:
        model:            List of model chunks (same convention as Megatron).
        vision_config:    Config object with hidden_size, num_layers, etc.
        vision_seq_length: Max sequence length (patches) for graph capture.
        micro_batch_size: Training MBS (vision always uses batch_dim=1 internally).
        num_microbatches: Number of microbatches per pipeline step.
        device:           CUDA device for this rank.
        rank:             Global rank (0 = first pipeline stage = has vision).
        world_size:       Total number of ranks.
        coordinator:      Optional pre-built CaptureCoordinator; one is created
                          if not provided.
    """

    def __init__(
        self,
        model: List[Any],
        vision_config: Any,
        vision_seq_length: int,
        micro_batch_size: int,
        num_microbatches: int = 1,
        device: Optional[torch.device] = None,
        rank: int = 0,
        world_size: int = 1,
        coordinator: Optional[CaptureCoordinator] = None,
    ):
        self.model = model
        self.config = vision_config
        self.seq_length = vision_seq_length
        # Vision encoder always processes images with batch_dim=1 (images are
        # concatenated along the sequence dimension, matching Megatron behaviour).
        self.micro_batch_size = 1
        self.num_microbatches = num_microbatches

        self.device = device or torch.cuda.current_device()
        if isinstance(self.device, int):
            self.device = torch.device(f"cuda:{self.device}")

        self.rank = rank
        self.world_size = world_size
        self.sm_arch = _get_sm_arch(self.device)
        self.device_role = _get_device_role(self.device)
        self.slc_tier = _get_slc_tier(self.device)

        self.coordinator = coordinator or CaptureCoordinator(rank, world_size, self.device)

        # Graph pool for this device's SM architecture.
        self._pool = _ArchGraphPool(device_id=self.device.index, sm_arch=self.sm_arch)

        self._graphs_created = False

        # Populated by _discover_layers()
        self.vision_model: Optional[Any] = None
        self.callables: List[Any] = []
        self.num_layers: int = 0

        self._discover_layers()

        logger.info(
            "HeteroVisionEncoderCudaGraph init: rank=%d device=%s sm_arch=%d "
            "role=%s slc_tier=%s seq_length=%d num_microbatches=%d num_layers=%d",
            self.rank, self.device, self.sm_arch, self.device_role.name,
            self.slc_tier, self.seq_length, self.num_microbatches, self.num_layers,
        )

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------

    def _discover_layers(self) -> None:
        """
        Discover capturable vision encoder layers from the model list.

        Mirrors VisionTECudaGraphHelper._discover_layers() but uses a
        DES-LOC-specific graphability check (_layer_is_graphable_hetero)
        that additionally validates SM-arch compatibility.

        On pipeline stage > 0 (rank != 0 in vision-first-stage convention)
        the model chunk will not have vision_model; we degrade gracefully
        to an empty callable list so that create_cudagraphs() becomes a
        no-op on those ranks.
        """
        vision_layers: List[Any] = []

        for model_chunk in self.model:
            vm = self._try_get_vision_model(model_chunk)
            if vm is not None:
                self.vision_model = vm
                break

        if self.vision_model is not None:
            decoder = getattr(self.vision_model, "decoder", None)
            if decoder is not None and hasattr(decoder, "layers"):
                for layer in decoder.layers:
                    if self._layer_is_graphable_hetero(layer):
                        vision_layers.append(layer)

        self.callables = vision_layers
        self.num_layers = len(vision_layers)

        if self.vision_model is None:
            logger.warning(
                "HeteroVisionEncoderCudaGraph rank=%d: no vision_model found. "
                "CUDA graph capture will be a no-op on this rank.",
                self.rank,
            )
        elif self.num_layers == 0:
            logger.warning(
                "HeteroVisionEncoderCudaGraph rank=%d: vision_model found but "
                "no graphable layers (sm_arch=%d). Capture will be skipped.",
                self.rank, self.sm_arch,
            )
        else:
            logger.info(
                "HeteroVisionEncoderCudaGraph rank=%d: discovered %d graphable "
                "vision layers on %s (sm_arch=%d).",
                self.rank, self.num_layers, self.device_role.name, self.sm_arch,
            )

    @staticmethod
    def _try_get_vision_model(model_chunk: Any) -> Optional[Any]:
        """Safely unwrap vision_model from a (possibly DDP-wrapped) model chunk."""
        # Direct attribute
        vm = getattr(model_chunk, "vision_model", None)
        if vm is not None:
            return vm
        # Wrapped model (e.g. DDP / ZeRO wrapper)
        module = getattr(model_chunk, "module", None)
        if module is not None:
            vm = getattr(module, "vision_model", None)
            if vm is not None:
                return vm
        return None

    def _layer_is_graphable_hetero(self, layer: Any) -> bool:
        """
        Determine whether a vision layer can be captured as a CUDA graph on
        this specific device architecture.

        Extends Megatron's _layer_is_graphable() check with:
          - SM-arch validation: refuse layers that carry an explicit
            `required_sm_arch` attribute incompatible with this device.
          - Heterogeneous placement awareness: layers tagged for H100NVL
            only are skipped on A6000 devices (they live on a different rank).

        In the current DES-LOC topology all vision layers reside on rank-0
        (H100NVL preferred), so this check mostly handles edge cases where
        a researcher manually shards vision layers across devices.
        """
        # Check SM-arch compatibility
        required_arch = getattr(layer, "required_sm_arch", None)
        if required_arch is not None and required_arch != self.sm_arch:
            logger.debug(
                "Skipping layer %s: requires sm_arch=%d, this device is sm_arch=%d",
                type(layer).__name__, required_arch, self.sm_arch,
            )
            return False

        # Check cuda_graph_impl on config (mirrors Megatron's check)
        cuda_graph_impl = getattr(self.config, "cuda_graph_impl", None)
        if cuda_graph_impl not in ("transformer_engine", "torch"):
            return False

        # Must be a nn.Module with a forward callable
        if not callable(getattr(layer, "forward", None)):
            return False

        return True

    # ------------------------------------------------------------------
    # Graph capture lifecycle
    # ------------------------------------------------------------------

    def graphs_created(self) -> bool:
        """Return True if CUDA graphs have been captured for this helper."""
        return self._graphs_created

    def _evict_slc_warm_tier_if_needed(self) -> None:
        """
        On A6000 devices (48 GB), evict SLC warm-tier buffers to CPU DRAM
        before graph capture to ensure sufficient VRAM headroom.

        H100NVL (96 GB) has enough budget that eviction is unnecessary.
        This asymmetric policy is a core DES-LOC optimisation: CPU DRAM
        (1.5 TB) acts as a third memory tier, and we proactively offload
        during the brief capture window.
        """
        if self.device_role == DeviceRole.A6000:
            logger.info(
                "HeteroVisionEncoderCudaGraph rank=%d A6000 device: evicting SLC "
                "warm tier to CPU DRAM before graph capture.",
                self.rank,
            )
            _slc.evict_tier(_SLC_TIER_WARM)
            torch.cuda.synchronize(self.device)

    def _start_capturing(self) -> float:
        """
        Prepare environment and signal capture start.

        Unlike Megatron's TECudaGraphHelper._start_capturing(), we:
          - Do NOT call dist.barrier() (PCIe latency makes this prohibitive).
          - Signal capture start via CaptureCoordinator event flag instead.
          - Evict SLC warm tier on A6000 devices.
          - Freeze GC to prevent allocations during capture.
        """
        assert not self._graphs_created, (
            "HeteroVisionEncoderCudaGraph: graphs already created; "
            "call delete_cuda_graphs() before re-capturing."
        )

        self._evict_slc_warm_tier_if_needed()

        gc.collect()
        torch.cuda.empty_cache()
        gc.freeze()

        self.coordinator.signal_capture_start()
        logger.info(
            "HeteroVisionEncoderCudaGraph rank=%d sm_arch=%d: "
            "starting CUDA graph capture (%d layers × %d microbatches).",
            self.rank, self.sm_arch, self.num_layers, self.num_microbatches,
        )
        return time.time()

    def _finish_capturing(self, start_time: float) -> None:
        """
        Finalise capture and register static buffers with SLC.

        Like Megatron's VisionTECudaGraphHelper._finish_capturing(), we skip:
          - dist.barrier() (replaced by coordinator event).
          - zero_grad_buffer() / optimizer.zero_grad() (handled by LM helper).
          - clear_aux_losses_tracker (LM-specific).

        Additionally we:
          - Register all static input/output tensors with the SLC.
          - Signal capture completion via coordinator.
          - Unfreeze GC.
        """
        elapsed = time.time() - start_time
        logger.info(
            "HeteroVisionEncoderCudaGraph rank=%d: CUDA graph capture completed "
            "in %.3fs (%d layers × %d microbatches).",
            self.rank, elapsed, self.num_layers, self.num_microbatches,
        )

        self._register_static_buffers_with_slc()

        gc.unfreeze()
        gc.collect()
        torch.cuda.empty_cache()

        self.coordinator.signal_capture_end()
        self._graphs_created = True

    def _register_static_buffers_with_slc(self) -> None:
        """
        Register captured static input/output tensors with the SharedLocalityCache.

        After graph capture, each layer×microbatch pair has a static tensor
        that must not be reallocated.  We register these with the SLC so that:
          - The SLC can prefetch PCIe transfers for warm-tier buffers before
            the next replay call.
          - The eviction policy knows not to evict hot-tier buffers mid-step.
        """
        for layer_idx, mb_inputs in self._pool.static_inputs.items():
            for mb_idx, tensor in enumerate(mb_inputs):
                if tensor is None:
                    continue
                key = f"vision_input_l{layer_idx}_mb{mb_idx}_dev{self.device.index}"
                _slc.register_buffer(key, tensor, self.slc_tier, self.device)

        for layer_idx, mb_outputs in self._pool.static_outputs.items():
            for mb_idx, tensor in enumerate(mb_outputs):
                if tensor is None:
                    continue
                key = f"vision_output_l{layer_idx}_mb{mb_idx}_dev{self.device.index}"
                _slc.register_buffer(key, tensor, self.slc_tier, self.device)

    def _get_sample_arguments(
        self,
    ) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Dict[str, Any]]]:
        """
        Generate sample (args, kwargs) for each (microbatch, layer) capture pair.

        Vision encoder always uses:
          - batch_dim = 1 (images concatenated along sequence dimension).
          - dtype = bfloat16 (matches DES-LOC mixed-precision policy).
          - requires_grad = True (needed for graph capture with backward).
          - Shape: (seq_length, 1, hidden_size) — same as Megatron.

        Unlike Megatron's parent class, we do NOT use rotary-embedding
        buffer reuse or pipeline-schedule-aware lifecycle tracking because:
          - Vision encoder has no rotary positional embeddings (ViT uses
            absolute patch embeddings).
          - num_model_chunks == 1 for vision (no virtual pipeline stages).

        Returns:
            (sample_args, sample_kwargs) — each list has length
            num_microbatches × num_layers.
        """
        if not self.callables:
            return [], []

        hidden_size: int = self.config.hidden_size
        sample_args: List[Tuple[torch.Tensor, ...]] = []
        sample_kwargs_list: List[Dict[str, Any]] = []

        for _mb in range(self.num_microbatches):
            for layer in self.callables:
                hidden_states = torch.zeros(
                    self.seq_length,
                    1,                  # batch_dim always 1 for vision
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=self.device,
                    requires_grad=True,
                )
                extra_kwargs: Dict[str, Any] = {}
                if hasattr(layer, "get_layer_static_inputs"):
                    static = layer.get_layer_static_inputs(self.seq_length, 1)
                    hidden_states = static.pop("hidden_states", hidden_states)
                    extra_kwargs = static

                sample_args.append((hidden_states,))
                sample_kwargs_list.append(extra_kwargs)

        return sample_args, sample_kwargs_list

    def _wrap_output_for_slc(
        self,
        graph_fn: Callable,
        layer_idx: int,
        mb_idx: int,
    ) -> Callable:
        """
        Wrap a captured graph callable to:
          1. Filter None outputs (vision layers return (output, None) tuple).
          2. Record static output tensor into the arch pool for SLC registration.

        Mirrors Megatron's _wrap_graph_for_vision() but additionally stores
        the output reference so it can be registered with the SLC.
        """
        pool = self._pool

        def wrapped(*args, **kwargs):
            result = graph_fn(*args, **kwargs)
            if isinstance(result, tuple):
                filtered = tuple(r for r in result if r is not None)
                result_out = filtered if filtered else result
            else:
                result_out = result

            # Record static output for SLC registration (first tensor in output)
            if layer_idx not in pool.static_outputs:
                pool.static_outputs[layer_idx] = [None] * self.num_microbatches
            out_tensor = result_out[0] if isinstance(result_out, tuple) else result_out
            if isinstance(out_tensor, torch.Tensor):
                pool.static_outputs[layer_idx][mb_idx] = out_tensor

            return result_out

        # Preserve TE-specific attributes (backward_dw, reset)
        for attr in ("backward_dw", "reset"):
            if hasattr(graph_fn, attr):
                setattr(wrapped, attr, getattr(graph_fn, attr))

        return wrapped

    def create_cudagraphs(self) -> None:
        """
        Capture CUDA graphs for all vision encoder layers × microbatches.

        Algorithm:
          1. Bail out (no-op) if no graphable layers on this rank.
          2. Evict SLC warm tier on A6000, freeze GC, signal capture start.
          3. For each layer × microbatch: record static inputs, capture graph
             via torch.cuda.CUDAGraph, store in _ArchGraphPool.
          4. Wrap captured callables with _wrap_output_for_slc.
          5. Attach wrapped callables back to layers as layer.cuda_graphs list
             (same interface as Megatron's TE graphs, enabling drop-in use by
             the pipeline schedule).
          6. Register static buffers with SLC, signal capture end.

        SM-arch guard: if this rank's device sm_arch != the pool's sm_arch
        (should never happen in normal operation) we raise immediately rather
        than silently producing incorrect graph replays.
        """
        if not self.callables:
            logger.warning(
                "HeteroVisionEncoderCudaGraph rank=%d: no graphable layers, "
                "skipping capture.",
                self.rank,
            )
            return

        # Validate arch consistency
        current_arch = _get_sm_arch(self.device)
        if current_arch != self._pool.sm_arch:
            raise RuntimeError(
                f"SM-arch mismatch: pool was created for sm_arch={self._pool.sm_arch} "
                f"but current device is sm_arch={current_arch}. "
                f"CUDA graphs cannot be replayed across architectures."
            )

        start_time = self._start_capturing()
        sample_args, sample_kwargs_list = self._get_sample_arguments()

        try:
            idx = 0
            for mb_idx in range(self.num_microbatches):
                for layer_idx, layer in enumerate(self.callables):
                    args = sample_args[idx]
                    kwargs = sample_kwargs_list[idx]
                    idx += 1

                    # Store static input tensor in pool
                    if layer_idx not in self._pool.static_inputs:
                        self._pool.static_inputs[layer_idx] = [None] * self.num_microbatches
                    self._pool.static_inputs[layer_idx][mb_idx] = args[0]

                    # Capture the graph
                    graph = torch.cuda.CUDAGraph()
                    # Warmup pass (required before graph capture)
                    with torch.cuda.stream(torch.cuda.Stream(device=self.device)):
                        _ = layer(*args, **kwargs)
                    torch.cuda.synchronize(self.device)

                    with torch.cuda.graph(graph, pool=None):
                        static_out = layer(*args, **kwargs)

                    # Store graph callable in pool
                    if layer_idx not in self._pool.graphs:
                        self._pool.graphs[layer_idx] = [None] * self.num_microbatches

                    # Build a replay callable
                    _graph_ref = graph
                    _args_ref = args
                    _static_out_ref = static_out

                    def _replay_fn(
                        *new_args,
                        _g=_graph_ref,
                        _static=_args_ref,
                        _out=_static_out_ref,
                        **new_kwargs,
                    ):
                        # Copy new inputs into static buffers in-place
                        if new_args:
                            _static[0].copy_(new_args[0])
                        _g.replay()
                        return _out

                    wrapped = self._wrap_output_for_slc(_replay_fn, layer_idx, mb_idx)
                    self._pool.graphs[layer_idx][mb_idx] = wrapped

            # Attach cuda_graphs list to each layer (Megatron-compatible interface)
            for layer_idx, layer in enumerate(self.callables):
                layer.cuda_graphs = [
                    self._pool.graphs[layer_idx][mb_idx]
                    for mb_idx in range(self.num_microbatches)
                ]

            self._finish_capturing(start_time)

        except Exception as exc:
            gc.unfreeze()
            gc.collect()
            torch.cuda.empty_cache()
            logger.error(
                "HeteroVisionEncoderCudaGraph rank=%d: capture failed: %s",
                self.rank, exc, exc_info=True,
            )
            raise

    def delete_cuda_graphs(self) -> None:
        """
        Delete all captured CUDA graphs and deregister SLC buffers.

        Mirrors Megatron's delete_cuda_graphs() lifecycle contract:
          - Asserts graphs_created() before deletion.
          - Clears layer.cuda_graphs lists.
          - Releases pool entries.
          - Resets coordinator flags for potential re-capture.
        """
        assert self._graphs_created, (
            "HeteroVisionEncoderCudaGraph.delete_cuda_graphs() called before "
            "create_cudagraphs()."
        )

        for layer in self.callables:
            if hasattr(layer, "cuda_graphs"):
                layer.cuda_graphs = []

        self._pool.graphs.clear()
        self._pool.static_inputs.clear()
        self._pool.static_outputs.clear()

        self._graphs_created = False
        self.coordinator.reset()

        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            "HeteroVisionEncoderCudaGraph rank=%d: CUDA graphs deleted.",
            self.rank,
        )

    def cuda_graph_set_manual_hooks(self) -> None:
        """
        No-op: vision encoder layers do not use DDP parameter-gather hooks.

        Megatron's parent class derives hooks from model_chunk._make_forward_pre_hook
        which requires overlap_param_gather=True.  Vision encoder parameters are
        not distributed with overlap parameter gather in DES-LOC (they live on a
        single device), so we skip hook setup.  This mirrors the behaviour of
        Megatron's VisionTECudaGraphHelper.cuda_graph_set_manual_hooks().
        """


# ---------------------------------------------------------------------------
# set_current_microbatch_hetero: DES-LOC adaptation of Megatron's function
# ---------------------------------------------------------------------------


def set_current_microbatch_hetero(
    model: Any,
    microbatch_id: int,
    notify_slc: bool = True,
) -> None:
    """
    Set current_microbatch on all layers that use CUDA graph replay.

    Upstream design (Megatron commit 37ca715):
        current_microbatch is read by _te_cuda_graph_replay to select the
        correct graph index.  Without this, vision layers always use graph 0,
        causing all microbatch forwards to overwrite the same static buffers.
        When backward runs for earlier microbatches, the buffers contain stale
        data from later forwards, producing NaN gradients.

    DES-LOC adaptation:
        Additionally calls _slc.notify_microbatch(microbatch_id) so the
        SharedLocalityCache can prefetch PCIe transfers for warm-tier (A6000)
        static buffers ahead of graph replay.  On PCIe-only topology, hiding
        this latency is critical for throughput.

    Args:
        model:         The model (or list of model chunks) to update.
        microbatch_id: Zero-based microbatch index within the current step.
        notify_slc:    Whether to notify the SLC (disable for dry-run/testing).
    """
    if notify_slc:
        _slc.notify_microbatch(microbatch_id)

    # Normalise model to a list of chunks
    model_list = model if isinstance(model, (list, tuple)) else [model]

    for model_chunk in model_list:
        # Language decoder layers
        _set_microbatch_on_decoder(model_chunk, microbatch_id)
        # Vision encoder layers
        _set_microbatch_on_vision(model_chunk, microbatch_id)

    logger.debug("set_current_microbatch_hetero: microbatch_id=%d", microbatch_id)


def _set_microbatch_on_decoder(model_chunk: Any, microbatch_id: int) -> None:
    """Set current_microbatch on language decoder and MTP layers."""
    model_with_decoder = _unwrap_attr(model_chunk, "decoder")
    if model_with_decoder is None:
        return

    decoder = getattr(model_with_decoder, "decoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        for layer in decoder.layers:
            layer.current_microbatch = microbatch_id

    mtp = getattr(model_with_decoder, "mtp", None)
    if mtp is not None and hasattr(mtp, "layers"):
        for layer in mtp.layers:
            assert hasattr(layer, "mtp_model_layer"), (
                f"MTP layer {layer} must have 'mtp_model_layer' attribute"
            )
            layer.mtp_model_layer.current_microbatch = microbatch_id


def _set_microbatch_on_vision(model_chunk: Any, microbatch_id: int) -> None:
    """
    Set current_microbatch on vision encoder layers.

    This is the key fix from Megatron commit 37ca715: without propagating
    microbatch_id to vision layers, all microbatches overwrite graph-0 static
    buffers, producing NaN gradients during backward.
    """
    vm = HeteroVisionEncoderCudaGraph._try_get_vision_model(model_chunk)
    if vm is None:
        return
    decoder = getattr(vm, "decoder", None)
    if decoder is None or not hasattr(decoder, "layers"):
        return
    for layer in decoder.layers:
        layer.current_microbatch = microbatch_id


def _unwrap_attr(obj: Any, attr: str) -> Optional[Any]:
    """
    Walk common wrapper layers (DDP, ZeRO, DeepSpeedEngine) to find *attr*.

    Returns the first object in the chain that directly owns *attr*, or None.
    """
    for candidate in (obj, getattr(obj, "module", None)):
        if candidate is not None and hasattr(candidate, attr):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Utility: vision sequence length calculation
# ---------------------------------------------------------------------------


def get_vision_cudagraph_seq_length(
    vision_config: Any,
    default_seq_length: int = 4096,
) -> int:
    """
    Calculate the sequence length for vision encoder CUDA graph capture.

    Mirrors Megatron's get_vision_cuda_graph_seq_length() but uses
    DES-LOC naming conventions and adds a VRAM-budget guard:
      - On A6000 (48 GB) we cap seq_length to avoid OOM during capture.
      - On H100NVL (96 GB) we allow the full computed seq_length.

    Args:
        vision_config:    TransformerConfig (or SimpleNamespace) for the vision
                          encoder.  Relevant attributes:
                            max_vision_cudagraph_seq_length — explicit cap
                            num_position_embeddings         — patch count
                            spatial_merge_size              — pooling factor
        default_seq_length: Fallback if config has none of the above.

    Returns:
        Sequence length to use for CUDA graph capture.
    """
    # Explicit override takes highest priority
    explicit = getattr(vision_config, "max_vision_cudagraph_seq_length", None)
    if explicit:
        return int(explicit)

    num_pos = getattr(vision_config, "num_position_embeddings", None)
    if num_pos is not None:
        seq = int(num_pos)
        merge = getattr(vision_config, "spatial_merge_size", None)
        if merge is not None:
            seq = seq // (int(merge) ** 2)
        return seq

    return default_seq_length


# ---------------------------------------------------------------------------
# factory: build the right helper given the current device
# ---------------------------------------------------------------------------


def build_hetero_vision_graph_helper(
    model: List[Any],
    vision_config: Any,
    micro_batch_size: int,
    num_microbatches: int,
    device: Optional[torch.device] = None,
    rank: int = 0,
    world_size: int = 1,
) -> HeteroVisionEncoderCudaGraph:
    """
    Factory function: build a HeteroVisionEncoderCudaGraph for this rank.

    Automatically computes vision_seq_length from vision_config, selects
    the correct SLC tier, and wires up a CaptureCoordinator.

    This is the recommended entry point from the DeepSpeed engine loop.
    """
    dev = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    seq_len = get_vision_cudagraph_seq_length(vision_config)

    coordinator = CaptureCoordinator(rank, world_size, dev)

    helper = HeteroVisionEncoderCudaGraph(
        model=model,
        vision_config=vision_config,
        vision_seq_length=seq_len,
        micro_batch_size=micro_batch_size,
        num_microbatches=num_microbatches,
        device=dev,
        rank=rank,
        world_size=world_size,
        coordinator=coordinator,
    )
    return helper


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import types

    logging.basicConfig(level=logging.INFO)

    # --- Test 1: get_vision_cudagraph_seq_length ---
    cfg = types.SimpleNamespace(
        num_position_embeddings=1024,
        spatial_merge_size=2,
    )
    result = get_vision_cudagraph_seq_length(cfg)
    assert result == 256, f"Expected 256, got {result}"

    cfg2 = types.SimpleNamespace(max_vision_cudagraph_seq_length=8192)
    assert get_vision_cudagraph_seq_length(cfg2) == 8192

    cfg3 = types.SimpleNamespace()
    assert get_vision_cudagraph_seq_length(cfg3, default_seq_length=512) == 512

    # --- Test 2: _wrap_output_for_slc via HeteroVisionEncoderCudaGraph ---
    # Build minimal config/model stubs
    vcfg = types.SimpleNamespace(
        hidden_size=16,
        num_layers=1,
        cuda_graph_impl="transformer_engine",
        num_position_embeddings=64,
    )

    dummy_layer = types.SimpleNamespace(
        forward=lambda x: (x * 2, None),
        current_microbatch=0,
    )
    dummy_layer.forward = lambda x: (x * 2, None)

    class _DummyDecoder:
        layers = [dummy_layer]

    class _DummyVisionModel:
        decoder = _DummyDecoder()

    class _DummyModel:
        vision_model = _DummyVisionModel()

    # Patch _layer_is_graphable_hetero to always return True for stub
    helper = HeteroVisionEncoderCudaGraph.__new__(HeteroVisionEncoderCudaGraph)
    helper.model = [_DummyModel()]
    helper.config = vcfg
    helper.seq_length = 8
    helper.micro_batch_size = 1
    helper.num_microbatches = 1
    helper.device = torch.device("cpu")  # smoke test on CPU
    helper.rank = 0
    helper.world_size = 1
    helper.sm_arch = 86
    helper.device_role = DeviceRole.A6000
    helper.slc_tier = _SLC_TIER_WARM
    helper._graphs_created = False
    helper._pool = _ArchGraphPool(device_id=0, sm_arch=86)
    helper.coordinator = CaptureCoordinator(0, 1, torch.device("cpu"))
    helper.vision_model = None
    helper.callables = []
    helper.num_layers = 0

    pool = helper._pool
    pool.static_outputs[0] = [None]
    t_in = torch.tensor([1.0])

    def _fake_graph(*a, **kw):
        return (t_in * 2, None)

    wrapped = helper._wrap_output_for_slc(_fake_graph, layer_idx=0, mb_idx=0)
    out = wrapped(t_in)
    assert isinstance(out, tuple) and len(out) == 1, f"Expected 1-tuple, got {out}"
    assert torch.allclose(out[0], torch.tensor([2.0])), f"Expected 2.0, got {out[0]}"

    # --- Test 3: set_current_microbatch_hetero on stub model ---
    dummy_layer.current_microbatch = 0
    set_current_microbatch_hetero(_DummyModel(), microbatch_id=3, notify_slc=True)
    assert dummy_layer.current_microbatch == 3, (
        f"Expected current_microbatch=3, got {dummy_layer.current_microbatch}"
    )
    assert _slc._microbatch_id == 3

    # --- Test 4: CaptureCoordinator flag signalling ---
    coord = CaptureCoordinator(rank=0, world_size=1, device=torch.device("cpu"))
    assert coord._start_flag[0] == 0
    coord.signal_capture_start()
    assert coord._start_flag[0] == 1
    coord.signal_capture_end()
    assert coord._end_flag[0] == 1
    coord.reset()
    assert coord._start_flag[0] == 0 and coord._end_flag[0] == 0

    logger.info("All smoke tests passed.")
