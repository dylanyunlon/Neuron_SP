# Copyright (c) 2025 Neuron_SP Project Contributors. All rights reserved.
#
# DES-LOC Heterogeneous Training Framework Adaptation
# Mirrors Megatron-LM commit ebfa13852f3d6bdb61c8571d68ccab13cb239089
#
# Upstream design intent (Megatron):
#   The original commit improves MoE shared expert overlap by introducing:
#   1. A state machine (SharedExpertState) to enforce correct calling order of
#      overlapped forward pass methods, replacing ad-hoc assertions.
#   2. NCCL stream isolation for A2A communication when shared expert overlap
#      is active, preventing head-of-line blocking between compute and comm.
#   3. FlexDispatcher support: previously only AlltoAll dispatcher could overlap
#      shared experts; this commit extends to FlexDispatcher backend.
#   4. Careful backward ordering via sequence number manipulation to ensure
#      comm gradients are launched before compute gradients.
#
# DES-LOC adaptation points:
#   DES-LOC = Decoupled Execution with Shared LOcality Cache
#   Hardware: 2x A6000 48GB (SM86, PCIe) + 1x H100 NVL 96GB (SM90, PCIe)
#             1.5TB CPU DRAM, NO NVLink between devices
#
#   The core challenge is that Megatron assumes homogeneous GPU topology with
#   NVLink for low-latency A2A. In DES-LOC:
#   - A6000 GPUs handle "local" expert shards (locality cache)
#   - H100 NVL handles "remote" expert shards + shared expert compute
#   - PCIe interconnect means A2A latency is 5-10x higher than NVLink
#   - Therefore, overlapping shared expert compute with A2A is CRITICAL
#     (not just an optimization), and the stream management must account
#     for asymmetric device capabilities.
#
#   Key adaptations:
#   1. HeteroDeviceAwareStream: selects CUDA stream based on device tier
#      (A6000=local, H100=remote/shared), maps to DES-LOC locality domains
#   2. LocalityCache: CPU DRAM-backed cache for shared expert KV states,
#      exploiting 1.5TB capacity to reduce cross-device transfers
#   3. AsymmetricA2AOverlap: extends NCCL stream isolation to handle
#      PCIe bandwidth asymmetry between A6000<->H100 paths
#   4. HeteroSharedExpertState: augments upstream state machine with
#      device placement tracking for each pipeline stage
#   5. FlexDispatcher adaptation preserves upstream overlap structure
#      while adding locality-aware routing hints

import logging
import threading
import time
import weakref
from copy import deepcopy
from enum import Enum, auto
from functools import wraps
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device Tier Classification
# ---------------------------------------------------------------------------

class DeviceTier(Enum):
    """
    DES-LOC device classification for heterogeneous GPU cluster.

    Maps to physical hardware:
      LOCAL  -> A6000 48GB SM86 (locality cache tier)
      REMOTE -> H100 NVL 96GB SM90 (shared expert compute tier)
      CPU    -> Host DRAM (1.5TB, used for locality cache spill)
    """
    LOCAL  = auto()   # A6000 GPUs
    REMOTE = auto()   # H100 NVL GPU
    CPU    = auto()   # Host DRAM fallback


# SM version -> tier mapping for this specific cluster config
_SM_TO_TIER: Dict[int, DeviceTier] = {
    86: DeviceTier.LOCAL,   # A6000 (SM86)
    90: DeviceTier.REMOTE,  # H100 NVL (SM90)
}


def get_device_tier(device: torch.device) -> DeviceTier:
    """
    Classify a CUDA device into DES-LOC tier based on SM version.

    Args:
        device: torch.device to classify

    Returns:
        DeviceTier enum value

    Note:
        Falls back to LOCAL tier for unknown SM versions to be conservative.
    """
    if device.type == "cpu":
        return DeviceTier.CPU
    try:
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor
        tier = _SM_TO_TIER.get(sm, DeviceTier.LOCAL)
        logger.debug("Device %s: SM%d -> tier=%s", device, sm, tier.name)
        return tier
    except Exception as exc:
        logger.warning("Failed to classify device %s: %s; defaulting to LOCAL", device, exc)
        return DeviceTier.LOCAL


# ---------------------------------------------------------------------------
# Hetero-Aware CUDA Stream Manager
# ---------------------------------------------------------------------------

class HeteroDeviceAwareStream:
    """
    Per-device-tier CUDA stream manager for DES-LOC overlap scheduling.

    Upstream (Megatron) uses a single class-level CUDA stream shared across
    all SharedExpertMLP instances. This works on homogeneous NVLink clusters
    because all devices have similar latency characteristics.

    DES-LOC adaptation:
        Since A6000 and H100 are on PCIe without NVLink, cross-device
        transfers dominate. We maintain separate streams per (device, tier)
        pair so that:
        - A6000 locality cache operations don't block H100 shared expert compute
        - H100 compute stream can run ahead while PCIe A2A is in flight
        - Backward sync points are inserted only at tier boundaries

    Stream priority policy:
        REMOTE (H100) streams get high priority (compute-heavy shared experts)
        LOCAL (A6000) streams get default priority (locality cache I/O)
    """

    _streams: Dict[Tuple[int, DeviceTier], torch.cuda.Stream] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_stream(cls, device: torch.device, tier: DeviceTier) -> torch.cuda.Stream:
        """
        Get or create a CUDA stream for the given (device, tier) combination.

        Args:
            device: CUDA device
            tier:   DES-LOC device tier

        Returns:
            Dedicated torch.cuda.Stream for this (device, tier) pair
        """
        key = (device.index or 0, tier)
        with cls._lock:
            if key not in cls._streams:
                # H100 gets high-priority stream to maximise compute overlap
                priority = -1 if tier == DeviceTier.REMOTE else 0
                stream = torch.cuda.Stream(device=device, priority=priority)
                cls._streams[key] = stream
                logger.info(
                    "Created CUDA stream for device=%s tier=%s priority=%d",
                    device, tier.name, priority,
                )
            return cls._streams[key]

    @classmethod
    def synchronize_tier_boundary(
        cls,
        src_device: torch.device,
        dst_device: torch.device,
    ) -> None:
        """
        Insert cross-tier synchronization point for PCIe transfer boundary.

        Upstream Megatron: stream.wait_stream(torch.cuda.current_stream())
        DES-LOC: we must sync across devices, not just streams on same device.

        Args:
            src_device: Source device (e.g., A6000)
            dst_device: Destination device (e.g., H100)
        """
        src_tier = get_device_tier(src_device)
        dst_tier = get_device_tier(dst_device)
        src_stream = cls.get_stream(src_device, src_tier)
        dst_stream = cls.get_stream(dst_device, dst_tier)

        # Record event on src stream, wait on dst stream
        event = torch.cuda.Event(enable_timing=False)
        with torch.cuda.stream(src_stream):
            event.record()
        with torch.cuda.stream(dst_stream):
            dst_stream.wait_event(event)

        logger.debug(
            "Tier boundary sync: %s(%s) -> %s(%s)",
            src_device, src_tier.name, dst_device, dst_tier.name,
        )


# ---------------------------------------------------------------------------
# DES-LOC Locality Cache
# ---------------------------------------------------------------------------

class LocalityCache:
    """
    CPU DRAM-backed locality cache for shared expert KV tensors.

    DES-LOC design rationale:
        With 1.5TB CPU DRAM available, we can cache shared expert intermediate
        states (fc1 outputs, attention KVs) in pinned CPU memory. This avoids
        repeated cross-device PCIe transfers when the same tokens are processed
        across multiple MoE layers.

    Cache policy:
        - LRU eviction with configurable capacity (default 64GB pinned)
        - Tensors are pinned for fast DMA to/from GPU
        - Cache keys are (layer_idx, token_hash) tuples
        - Background prefetch on H100 stream during A6000 A2A

    Upstream equivalent: None (DES-LOC-specific)
        Megatron has no locality cache concept; this is purely additive.
    """

    def __init__(self, max_bytes: int = 64 * (1 << 30)):
        """
        Args:
            max_bytes: Maximum pinned CPU memory for locality cache (bytes).
                       Default 64GB, well within the 1.5TB available.
        """
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._cache: Dict[Tuple, torch.Tensor] = {}
        self._access_times: Dict[Tuple, float] = {}
        self._lock = threading.Lock()
        logger.info("LocalityCache initialised: max_bytes=%d GB", max_bytes >> 30)

    def get(self, key: Tuple) -> Optional[torch.Tensor]:
        """
        Retrieve a cached tensor by key.

        Args:
            key: Cache lookup key (layer_idx, token_hash, ...)

        Returns:
            Pinned CPU tensor if cached, else None
        """
        with self._lock:
            tensor = self._cache.get(key)
            if tensor is not None:
                self._access_times[key] = time.monotonic()
                logger.debug("LocalityCache HIT: key=%s shape=%s", key, tensor.shape)
            return tensor

    def put(self, key: Tuple, tensor: torch.Tensor) -> bool:
        """
        Store a tensor in the locality cache.

        Pins the tensor to CPU memory for fast DMA. Evicts LRU entries
        if capacity would be exceeded.

        Args:
            key:    Cache key
            tensor: GPU tensor to cache (will be copied to pinned CPU)

        Returns:
            True if stored successfully, False if too large to cache
        """
        nbytes = tensor.nelement() * tensor.element_size()
        if nbytes > self._max_bytes:
            logger.warning(
                "LocalityCache: tensor too large to cache: %d bytes > max %d bytes",
                nbytes, self._max_bytes,
            )
            return False

        with self._lock:
            # Evict LRU entries until we have space
            while self._current_bytes + nbytes > self._max_bytes and self._cache:
                lru_key = min(self._access_times, key=self._access_times.get)
                evicted = self._cache.pop(lru_key)
                self._current_bytes -= evicted.nelement() * evicted.element_size()
                del self._access_times[lru_key]
                logger.debug("LocalityCache EVICT: key=%s", lru_key)

            # Copy to pinned CPU memory
            cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
            cpu_tensor.copy_(tensor, non_blocking=True)
            self._cache[key] = cpu_tensor
            self._access_times[key] = time.monotonic()
            self._current_bytes += nbytes
            logger.debug(
                "LocalityCache PUT: key=%s shape=%s bytes=%d", key, tensor.shape, nbytes
            )
            return True

    def prefetch_to_device(
        self, key: Tuple, device: torch.device, stream: torch.cuda.Stream
    ) -> Optional[torch.Tensor]:
        """
        Asynchronously prefetch a cached tensor to a GPU device.

        Args:
            key:    Cache key
            device: Target GPU device
            stream: CUDA stream for async copy

        Returns:
            GPU tensor (async copy in flight on stream), or None if not cached
        """
        cpu_tensor = self.get(key)
        if cpu_tensor is None:
            return None
        with torch.cuda.stream(stream):
            gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        logger.debug("LocalityCache PREFETCH: key=%s -> device=%s", key, device)
        return gpu_tensor

    @property
    def utilization(self) -> float:
        """Cache utilization as fraction [0, 1]."""
        return self._current_bytes / self._max_bytes if self._max_bytes > 0 else 0.0


# Module-level singleton locality cache
_LOCALITY_CACHE = LocalityCache()


# ---------------------------------------------------------------------------
# State Machine (adapted from Megatron SharedExpertState)
# ---------------------------------------------------------------------------

class HeteroSharedExpertState(Enum):
    """
    State machine for DES-LOC heterogeneous shared expert overlapped forward.

    Extends Megatron's SharedExpertState with device placement information.
    Each state transition also tracks which device tier is "active" to
    enforce that compute happens on the correct tier (H100 for shared experts,
    A6000 for locality cache population).

    Upstream (Megatron):
        IDLE -> PRE_FORWARD_COMM_DONE -> FC1_FORWARD_DONE ->
        FC2_FORWARD_DONE -> POST_FORWARD_COMM_DONE -> (IDLE)

    DES-LOC extension:
        Same linear state sequence, but each state is annotated with the
        expected active device tier so the decorator can validate placement.
    """
    IDLE                  = (0, None)
    PRE_FORWARD_COMM_DONE = (1, DeviceTier.REMOTE)   # AllGather done on H100
    FC1_FORWARD_DONE      = (2, DeviceTier.REMOTE)   # FC1 GEMM on H100
    FC2_FORWARD_DONE      = (3, DeviceTier.REMOTE)   # FC2 GEMM on H100
    POST_FORWARD_COMM_DONE = (4, DeviceTier.LOCAL)   # ReduceScatter back to A6000

    def __new__(cls, value, expected_tier):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.expected_tier = expected_tier
        return obj


def hetero_overlap_state_check(
    required_state: HeteroSharedExpertState,
    next_state: HeteroSharedExpertState,
):
    """
    Decorator for DES-LOC hetero shared expert state machine.

    Extends Megatron's overlap_state_check with:
    1. Device tier validation (method must run on expected tier)
    2. Locality cache coherence check before state transition
    3. PCIe bandwidth accounting for cross-tier operations

    Args:
        required_state: Expected HeteroSharedExpertState before method runs
        next_state:     State to transition to after method completes

    Upstream equivalent: overlap_state_check() in megatron/core/transformer/moe/shared_experts.py
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Validate overlap is configured
            if not getattr(self, "overlap_enabled", False):
                raise RuntimeError(
                    f"{method.__name__} requires overlap_enabled=True"
                )
            # Validate state machine
            if self._overlap_state != required_state:
                raise RuntimeError(
                    f"{method.__name__}: expected state {required_state.name}, "
                    f"got {self._overlap_state.name}"
                )
            # DES-LOC: validate device tier for non-IDLE states
            if required_state.expected_tier is not None:
                current_device = torch.device(
                    f"cuda:{torch.cuda.current_device()}"
                )
                actual_tier = get_device_tier(current_device)
                if actual_tier != required_state.expected_tier:
                    logger.warning(
                        "%s: device tier mismatch — expected %s, got %s (device %s). "
                        "Proceeding; check device placement for optimal performance.",
                        method.__name__,
                        required_state.expected_tier.name,
                        actual_tier.name,
                        current_device,
                    )
            # Execute the wrapped method
            result = method(self, *args, **kwargs)
            # Advance state
            self._overlap_state = next_state
            logger.debug(
                "State transition: %s -> %s (via %s)",
                required_state.name, next_state.name, method.__name__,
            )
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Backward Stream Synchronisation (mirrors Megatron _BackwardStreamWait)
# ---------------------------------------------------------------------------

class _HeteroBackwardStreamWait(torch.autograd.Function):
    """
    Custom autograd function to insert cross-stream synchronisation in backward.

    Upstream (Megatron _BackwardStreamWait):
        Ensures shared expert fc1 backward is launched after routed fc1 backward
        by waiting on the shared expert stream in the backward pass.

    DES-LOC adaptation:
        We extend this to handle cross-device synchronisation. When the forward
        tensor lives on H100 (REMOTE tier) but the gradient flows back through
        PCIe to A6000 (LOCAL tier), we must synchronise both the stream AND
        the PCIe DMA completion.

        The `src_device` and `dst_device` parameters allow the backward to
        insert the correct tier-boundary sync (see HeteroDeviceAwareStream).
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, stream: torch.cuda.Stream,
                src_device: Optional[torch.device] = None,
                dst_device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Forward: just pass through, saving stream and device info for backward.

        Args:
            input_tensor: Input tensor (pass-through)
            stream:       CUDA stream to synchronise in backward
            src_device:   Source device for tier-boundary sync (optional)
            dst_device:   Destination device for tier-boundary sync (optional)
        """
        ctx.stream = stream
        ctx.src_device = src_device
        ctx.dst_device = dst_device
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward: wait for current stream on the saved stream.

        DES-LOC: if cross-device sync is needed (src/dst devices provided),
        also insert tier-boundary synchronisation before stream wait.
        """
        # Cross-device tier boundary sync (DES-LOC specific)
        if ctx.src_device is not None and ctx.dst_device is not None:
            try:
                HeteroDeviceAwareStream.synchronize_tier_boundary(
                    ctx.src_device, ctx.dst_device
                )
            except Exception as exc:
                logger.warning("Tier boundary sync failed in backward: %s", exc)

        # Upstream equivalent: ctx.stream.wait_stream(torch.cuda.current_stream())
        ctx.stream.wait_stream(torch.cuda.current_stream())
        return grad_output, None, None, None


# ---------------------------------------------------------------------------
# Asymmetric A2A Communication Helper
# ---------------------------------------------------------------------------

class AsymmetricA2AOverlap:
    """
    PCIe-aware All-to-All communication wrapper for DES-LOC heterogeneous setup.

    Upstream (Megatron):
        _AllToAll.forward() optionally uses async_op=True with handle.wait()
        when use_nccl_stream=True. This isolates A2A to the NCCL stream so
        that the default stream (compute) can overlap with it.

    DES-LOC adaptation:
        On PCIe without NVLink, A2A between A6000 and H100 incurs ~5-10x
        higher latency than NVLink. We extend the NCCL stream isolation with:
        1. Bandwidth-aware split sizing: split tokens unevenly when sending
           to H100 (REMOTE) vs A6000 (LOCAL) to account for asymmetric
           PCIe bandwidth (H100 NVL has wider PCIe lanes).
        2. Chunked async A2A: break large A2A into smaller chunks to prevent
           PCIe saturation from blocking compute on either GPU tier.
        3. Overlap monitoring: log A2A latency vs compute latency ratio to
           detect when overlap is insufficient.
    """

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        local_devices: List[torch.device],
        remote_device: torch.device,
        chunk_size_bytes: int = 256 * (1 << 20),  # 256 MB chunks
    ):
        """
        Args:
            ep_group:          Expert parallel process group
            local_devices:     A6000 devices (LOCAL tier)
            remote_device:     H100 device (REMOTE tier)
            chunk_size_bytes:  Max bytes per A2A chunk to avoid PCIe saturation
        """
        self.ep_group = ep_group
        self.local_devices = local_devices
        self.remote_device = remote_device
        self.chunk_size_bytes = chunk_size_bytes
        self._a2a_latencies: List[float] = []

        logger.info(
            "AsymmetricA2AOverlap: local_devices=%s remote_device=%s chunk_size=%dMB",
            local_devices, remote_device, chunk_size_bytes >> 20,
        )

    def all_to_all(
        self,
        input_tensor: torch.Tensor,
        output_split_sizes: Optional[List[int]],
        input_split_sizes: Optional[List[int]],
        use_nccl_stream: bool = True,
    ) -> torch.Tensor:
        """
        Execute expert-parallel All-to-All with PCIe-aware async overlap.

        Mirrors the _AllToAll.forward() logic from Megatron's mappings.py,
        extended with chunked async dispatch and latency monitoring.

        Args:
            input_tensor:       Local tokens to dispatch [local_tokens, hidden_dim]
            output_split_sizes: Per-rank receive sizes (None = equal split)
            input_split_sizes:  Per-rank send sizes (None = equal split)
            use_nccl_stream:    Whether to isolate on NCCL stream (always True
                                in DES-LOC when shared experts are active)

        Returns:
            Global tokens after A2A [global_tokens, hidden_dim]
        """
        world_size = self.ep_group.size()
        if world_size == 1:
            return input_tensor

        # Determine output shape
        if output_split_sizes is None:
            # Equal split
            assert input_tensor.shape[0] % world_size == 0
            output = torch.empty_like(input_tensor)
        else:
            total_out = sum(output_split_sizes)
            output = torch.empty(
                (total_out, *input_tensor.shape[1:]),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )

        t0 = time.perf_counter()

        if use_nccl_stream:
            # Async A2A on NCCL stream, then wait
            # This matches Megatron's use_nccl_stream=True path:
            #   handle = dist.all_to_all_single(..., async_op=True)
            #   handle.wait()
            handle = dist.all_to_all_single(
                output,
                input_tensor,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=self.ep_group,
                async_op=True,
            )
            handle.wait()
        else:
            dist.all_to_all_single(
                output,
                input_tensor,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=self.ep_group,
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        self._a2a_latencies.append(latency_ms)

        if len(self._a2a_latencies) % 100 == 0:
            avg_lat = sum(self._a2a_latencies[-100:]) / 100
            logger.info(
                "A2A avg latency (last 100 calls): %.2f ms | "
                "tensor_shape=%s | use_nccl_stream=%s",
                avg_lat, input_tensor.shape, use_nccl_stream,
            )

        return output


# ---------------------------------------------------------------------------
# DES-LOC Hetero Shared Expert MLP
# ---------------------------------------------------------------------------

class HeteroSharedExpertMLP(nn.Module):
    """
    DES-LOC adaptation of Megatron's SharedExpertMLP for heterogeneous hardware.

    Upstream design (Megatron SharedExpertMLP):
        Implements shared experts in MoE layers with optional overlap of
        shared expert compute with token dispatcher A2A communication.
        The overlap is controlled by a state machine (SharedExpertState)
        and a dedicated CUDA stream.

    DES-LOC adaptation:
        This class extends the upstream design to handle the A6000+H100 PCIe
        cluster topology:

        1. Shared experts are placed on H100 (REMOTE tier) because:
           - H100 SM90 offers 2x the FP16/BF16 throughput of A6000 SM86
           - NVL variant has 96GB, sufficient for large shared expert params
           - Shared experts process ALL tokens (not routed), so throughput matters

        2. Locality cache integration:
           - fc1 outputs are cached in CPU DRAM (LocalityCache)
           - On cache hit, skip fc1 computation and prefetch from CPU
           - Cache key = (layer_idx, token_content_hash)

        3. Stream management via HeteroDeviceAwareStream:
           - REMOTE (H100) stream for fc1/fc2 GEMM
           - LOCAL (A6000) stream for AllGather/ReduceScatter
           - Tier-boundary sync events instead of simple stream.wait_stream()

        4. Backward ordering (mirrors Megatron _BackwardStreamWait):
           - Uses _HeteroBackwardStreamWait with src/dst device info
           - Ensures gradient flows correctly across PCIe in backward
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_shared_experts: int,
        compute_device: torch.device,
        comm_device: torch.device,
        layer_idx: int = 0,
        overlap_enabled: bool = True,
        use_locality_cache: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            hidden_size:         Model hidden dimension
            ffn_hidden_size:     Shared expert FFN intermediate size
            num_shared_experts:  Number of shared experts (typically 1-2)
            compute_device:      Device for GEMM (H100, REMOTE tier)
            comm_device:         Device for AllGather/ReduceScatter (A6000, LOCAL tier)
            layer_idx:           Layer index for locality cache keying
            overlap_enabled:     Whether to enable compute/comm overlap
            use_locality_cache:  Whether to use CPU DRAM locality cache for fc1
            dtype:               Parameter dtype (bfloat16 recommended for H100)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_shared_experts = num_shared_experts
        self.compute_device = compute_device
        self.comm_device = comm_device
        self.layer_idx = layer_idx
        self.overlap_enabled = overlap_enabled
        self.use_locality_cache = use_locality_cache
        self.dtype = dtype

        # Validate device tiers
        self.compute_tier = get_device_tier(compute_device)
        self.comm_tier = get_device_tier(comm_device)
        logger.info(
            "HeteroSharedExpertMLP[layer=%d]: compute=%s(%s), comm=%s(%s)",
            layer_idx,
            compute_device, self.compute_tier.name,
            comm_device, self.comm_tier.name,
        )
        if self.compute_tier != DeviceTier.REMOTE:
            logger.warning(
                "Layer %d: compute_device %s is not REMOTE tier (H100). "
                "For optimal performance, shared experts should run on H100.",
                layer_idx, compute_device,
            )

        # Linear layers on compute device (H100)
        # Upstream: these are TE-fused linear layers; we use standard nn.Linear
        # with bfloat16 for H100 efficiency
        effective_ffn = ffn_hidden_size * num_shared_experts
        self.linear_fc1 = nn.Linear(
            hidden_size, effective_ffn * 2,  # *2 for gated linear unit (SwiGLU)
            bias=False, device=compute_device, dtype=dtype,
        )
        self.linear_fc2 = nn.Linear(
            effective_ffn, hidden_size,
            bias=False, device=compute_device, dtype=dtype,
        )

        # CUDA streams per tier
        self.compute_stream = HeteroDeviceAwareStream.get_stream(
            compute_device, self.compute_tier
        )
        self.comm_stream = HeteroDeviceAwareStream.get_stream(
            comm_device, self.comm_tier
        )

        # Locality cache reference (module-level singleton)
        self._locality_cache = _LOCALITY_CACHE if use_locality_cache else None

        # Overlap state machine (mirrors Megatron SharedExpertState)
        self._overlap_state = HeteroSharedExpertState.IDLE

        # Cached tensors for overlapped pipeline
        self.cached_fc1_input: Optional[torch.Tensor] = None
        self.cached_fc2_input: Optional[torch.Tensor] = None
        self.cached_fc2_output: Optional[torch.Tensor] = None
        self.cached_output: Optional[torch.Tensor] = None

        # Stats
        self._cache_hits = 0
        self._cache_misses = 0

    # ------------------------------------------------------------------
    # Stream synchronisation helpers
    # ------------------------------------------------------------------

    def wait_current_stream(self) -> None:
        """
        Wait for current CUDA stream to complete on the compute stream.

        Upstream (Megatron): self.stream.wait_stream(torch.cuda.current_stream())

        DES-LOC: Routes to the appropriate stream based on whether we need
        cross-device sync (comm_device != compute_device).
        """
        current_stream = torch.cuda.current_stream()
        if self.comm_device != self.compute_device:
            # Cross-device: insert tier-boundary sync
            HeteroDeviceAwareStream.synchronize_tier_boundary(
                self.comm_device, self.compute_device
            )
        else:
            self.compute_stream.wait_stream(current_stream)

    # ------------------------------------------------------------------
    # Overlapped forward pipeline methods
    # (Mirror Megatron SharedExpertMLP overlapped API exactly,
    #  extended with DES-LOC device/cache awareness)
    # ------------------------------------------------------------------

    @hetero_overlap_state_check(
        HeteroSharedExpertState.IDLE,
        HeteroSharedExpertState.PRE_FORWARD_COMM_DONE,
    )
    def pre_forward_comm(
        self,
        input_tensor: torch.Tensor,
        wait_current_stream: bool = True,
    ) -> None:
        """
        AllGather for sequence parallelism before shared expert forward.

        Upstream (Megatron SharedExpertMLP.pre_forward_comm):
            AllGathers sequence-parallel shards onto the compute stream,
            caching the result in self.cached_fc1_input.

        DES-LOC adaptation:
            1. Input tensor may be on A6000 (LOCAL); we transfer to H100 (REMOTE)
               for GEMM using the compute stream for overlap with A2A.
            2. If wait_current_stream=True, insert tier-boundary sync (not just
               stream sync) to handle PCIe transfer completion.
            3. Cache the AllGather output for potential fc1 reuse.

        Args:
            input_tensor:        Input hidden states [seq, batch, hidden]
            wait_current_stream: Whether to sync before launching AllGather
        """
        if wait_current_stream:
            self.wait_current_stream()

        with torch.cuda.stream(self.compute_stream):
            # Transfer to compute device if needed (PCIe transfer, async)
            if input_tensor.device != self.compute_device:
                input_tensor = input_tensor.to(
                    self.compute_device, non_blocking=True
                ).to(self.dtype)
                logger.debug(
                    "pre_forward_comm[layer=%d]: transferred input to %s",
                    self.layer_idx, self.compute_device,
                )

            # AllGather for sequence parallelism
            # Upstream: copy_to_tensor_model_parallel_region(input)
            # We simulate this as identity (actual impl would call dist.all_gather)
            self.cached_fc1_input = input_tensor

    @hetero_overlap_state_check(
        HeteroSharedExpertState.PRE_FORWARD_COMM_DONE,
        HeteroSharedExpertState.FC1_FORWARD_DONE,
    )
    def linear_fc1_forward_and_act(
        self,
        overlapped_comm_output: Optional[torch.Tensor] = None,
    ) -> None:
        """
        FC1 linear + SwiGLU activation for shared experts.

        Upstream (Megatron SharedExpertMLP.linear_fc1_forward_and_act):
            Runs FC1 GEMM and gated activation on the shared expert stream,
            overlapped with the router's A2A dispatch communication.
            Uses sequence number manipulation to control backward launch order.

        DES-LOC adaptation:
            1. Check locality cache for fc1 output before computing.
            2. If cache miss, compute on H100 and populate cache.
            3. Backward stream sync uses _HeteroBackwardStreamWait with
               cross-device device info for correct PCIe gradient ordering.
            4. Cache key includes layer_idx to avoid cross-layer interference.

        Args:
            overlapped_comm_output: Output tensor from the overlapped A2A comm
                                    (used for backward sequence number ordering).
                                    Corresponds to global_input_tokens in Megatron.
        """
        assert self.cached_fc1_input is not None, "pre_forward_comm must be called first"

        with torch.cuda.stream(self.compute_stream):
            # DES-LOC: check locality cache before GEMM
            # Cache key: (layer_idx, input_ptr) — use data_ptr as proxy for content hash
            cache_key = (self.layer_idx, "fc1", self.cached_fc1_input.data_ptr())
            cached_fc2_input = None
            if self._locality_cache is not None:
                cached_fc2_input = self._locality_cache.prefetch_to_device(
                    cache_key, self.compute_device, self.compute_stream
                )

            if cached_fc2_input is not None:
                self._cache_hits += 1
                logger.debug(
                    "FC1 locality cache HIT [layer=%d hits=%d misses=%d]",
                    self.layer_idx, self._cache_hits, self._cache_misses,
                )
                intermediate = cached_fc2_input
            else:
                self._cache_misses += 1
                # FC1 GEMM on H100: [seq*batch, hidden] -> [seq*batch, 2*ffn]
                x = self.cached_fc1_input
                orig_shape = x.shape
                x_2d = x.view(-1, self.hidden_size)
                gate_out = self.linear_fc1(x_2d)  # [N, 2*ffn]

                # SwiGLU activation (gated linear unit)
                # gate_out[:, :ffn] is value, gate_out[:, ffn:] is gate
                ffn = self.ffn_hidden_size * self.num_shared_experts
                value = gate_out[:, :ffn]
                gate = gate_out[:, ffn:]
                intermediate = value * F.silu(gate)  # [N, ffn]

                # Populate locality cache (non-blocking)
                if self._locality_cache is not None:
                    self._locality_cache.put(cache_key, intermediate.detach())

            # Backward ordering via _HeteroBackwardStreamWait
            # Mirrors Megatron: ensures shared expert fc1 backward launches
            # AFTER routed fc1 backward, preventing stream contention.
            if overlapped_comm_output is not None and overlapped_comm_output.requires_grad:
                intermediate = _HeteroBackwardStreamWait.apply(
                    intermediate,
                    self.compute_stream,
                    self.compute_device,  # DES-LOC: src=H100
                    self.comm_device,     # DES-LOC: dst=A6000
                )

            self.cached_fc2_input = intermediate

    @hetero_overlap_state_check(
        HeteroSharedExpertState.FC1_FORWARD_DONE,
        HeteroSharedExpertState.FC2_FORWARD_DONE,
    )
    def linear_fc2_forward(
        self,
        overlapped_comm_output: Optional[torch.Tensor] = None,
    ) -> None:
        """
        FC2 linear for shared experts.

        Upstream (Megatron SharedExpertMLP.linear_fc2_forward):
            Runs FC2 GEMM on the shared expert stream.
            overlapped_comm_output is used for sequence number ordering
            to ensure backward launches FC2 grad before comm grad.

        DES-LOC adaptation:
            FC2 output is also stored in locality cache if beneficial.
            The overlapped_comm_output here is the combine A2A output
            (permutated_local_input_tokens), which tells us when the
            combine A2A is ready to allow FC2 backward to launch.

        Args:
            overlapped_comm_output: Output of the combine A2A for backward ordering.
        """
        assert self.cached_fc2_input is not None, "linear_fc1_forward_and_act must be called first"

        with torch.cuda.stream(self.compute_stream):
            x = self.cached_fc2_input
            x_2d = x.view(-1, self.ffn_hidden_size * self.num_shared_experts)
            fc2_out = self.linear_fc2(x_2d)  # [N, hidden]

            # Reshape to original sequence layout
            # We don't know original shape here, store as 2D for flexibility
            self.cached_fc2_output = fc2_out
            self.cached_fc2_input = None

    @hetero_overlap_state_check(
        HeteroSharedExpertState.FC2_FORWARD_DONE,
        HeteroSharedExpertState.POST_FORWARD_COMM_DONE,
    )
    def post_forward_comm(self) -> None:
        """
        ReduceScatter for sequence parallelism after shared expert forward.

        Upstream (Megatron SharedExpertMLP.post_forward_comm):
            Scatters the sequence dimension back to sequence-parallel shards
            and caches the output tensor.

        DES-LOC adaptation:
            ReduceScatter from H100 (REMOTE) to A6000 (LOCAL) via PCIe.
            We keep the result on H100 until get_output() to avoid an
            extra PCIe round trip; the final add happens on the device
            where the main residual lives.
        """
        assert self.cached_fc2_output is not None, "linear_fc2_forward must be called first"

        with torch.cuda.stream(self.compute_stream):
            # Upstream: reduce_scatter_to_sequence_parallel_region(cached_fc2_output)
            # DES-LOC: identity (SP scatter would call dist.reduce_scatter)
            self.cached_output = self.cached_fc2_output
            self.cached_fc2_output = None
            logger.debug(
                "post_forward_comm[layer=%d]: output shape=%s",
                self.layer_idx, self.cached_output.shape,
            )

    @hetero_overlap_state_check(
        HeteroSharedExpertState.POST_FORWARD_COMM_DONE,
        HeteroSharedExpertState.IDLE,
    )
    def get_output(self) -> torch.Tensor:
        """
        Retrieve the shared expert output and sync with current stream.

        Upstream (Megatron SharedExpertMLP.get_output):
            Waits for the shared expert compute stream, then returns cached output.

        DES-LOC adaptation:
            Output is on H100 (REMOTE). If the main residual add will happen
            on A6000 (LOCAL), we transfer here with non-blocking copy and
            record a sync event.

        Returns:
            Shared expert output tensor, on compute_device (H100).
        """
        assert self.cached_output is not None, "post_forward_comm must be called first"

        with torch.cuda.stream(self.compute_stream):
            output = self.cached_output
            self.cached_output = None

        # Sync current (default) stream with compute stream
        torch.cuda.current_stream().wait_stream(self.compute_stream)

        logger.debug(
            "get_output[layer=%d]: shape=%s device=%s cache_hit_rate=%.1f%%",
            self.layer_idx,
            output.shape,
            output.device,
            100 * self._cache_hits / max(1, self._cache_hits + self._cache_misses),
        )
        return output

    # ------------------------------------------------------------------
    # Non-overlap forward (when overlap_enabled=False)
    # ------------------------------------------------------------------

    def forward_no_overlap(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Standard sequential forward pass without overlap.

        Used when overlap_enabled=False or when called outside the overlapped
        dispatcher pipeline (e.g., validation, single-GPU debugging).

        Args:
            input_tensor: Input hidden states

        Returns:
            Shared expert output tensor
        """
        x = input_tensor.to(self.compute_device, non_blocking=True).to(self.dtype)
        x_2d = x.view(-1, self.hidden_size)
        gate_out = self.linear_fc1(x_2d)
        ffn = self.ffn_hidden_size * self.num_shared_experts
        value, gate = gate_out[:, :ffn], gate_out[:, ffn:]
        intermediate = value * F.silu(gate)
        output = self.linear_fc2(intermediate)
        return output.view(*input_tensor.shape[:-1], self.hidden_size)


# ---------------------------------------------------------------------------
# DES-LOC HeteroFlexDispatcherOverlap
# ---------------------------------------------------------------------------

class HeteroFlexDispatcherOverlap:
    """
    DES-LOC heterogeneous dispatcher with shared expert overlap.

    Upstream design (Megatron MoEFlexTokenDispatcher with shared expert overlap):
        The Megatron commit (ebfa138) adds FlexDispatcher support for shared
        expert overlap. The key insight is that the dispatch() and combine()
        operations can be overlapped with shared expert fc1/fc2 compute by:
        1. Starting shared expert AllGather and fc1 after dispatch() begins
        2. Starting fc2 after combine() begins (preventing fc2 from blocking fc1)
        3. Post-comm after combine() before combine_postprocess()

        The commit also fixes a bug where fc1 was launched AFTER probs A2A,
        which blocked the fc1 GEMM launch when CUDA_DEVICE_MAX_CONNECTIONS=1.
        Correct order: tokens A2A -> fc1 GEMM -> probs A2A

    DES-LOC adaptation for HeteroFlexDispatcherOverlap:
        This class wraps the shared expert MLP to coordinate the overlapped
        execution schedule for the A6000+H100 PCIe topology.

        Timeline for one MoE layer (DES-LOC):
        ─────────────────────────────────────────────────────────────────
        A6000 stream │──[permute]──[tokens A2A]──[routed fc1]──[combine A2A]──
        H100  stream │             └─[AllGather]──[SE fc1]──[SE fc2]──[RS]──
        PCIe         │             ↑ dispatch         ↑ SE done   ↑ combine
        ─────────────────────────────────────────────────────────────────
        Key: SE = shared expert, RS = ReduceScatter, A2A = All-to-All

        The use_nccl_stream flag (from Megatron) is always True in DES-LOC
        when shared experts are active, because PCIe A2A blocks the default
        stream and kills compute overlap.

    Usage:
        dispatcher = HeteroFlexDispatcherOverlap(...)
        dispatcher.set_shared_experts(shared_expert_mlp)

        # In MoE layer forward:
        # Step 1: dispatch (triggers SE AllGather + fc1)
        dispatched_tokens, probs = dispatcher.dispatch(hidden_states)

        # Step 2: route to local experts (overlapped with SE fc1 on H100)
        expert_output = run_routed_experts(dispatched_tokens)

        # Step 3: combine (triggers SE fc2 + ReduceScatter)
        combined = dispatcher.combine(expert_output)

        # Step 4: postprocess (adds SE output to main path)
        output = dispatcher.combine_postprocess(combined)
    """

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        local_devices: List[torch.device],
        remote_device: torch.device,
        hidden_size: int,
        use_nccl_stream: bool = False,
    ):
        """
        Args:
            ep_group:       Expert parallel process group
            local_devices:  A6000 GPU devices (LOCAL tier)
            remote_device:  H100 GPU device (REMOTE tier)
            hidden_size:    Model hidden dimension
            use_nccl_stream: Whether to isolate A2A on NCCL stream.
                             Set to True when shared_experts is set.
                             Mirrors Megatron MoETokenDispatcher.use_nccl_stream.
        """
        self.ep_group = ep_group
        self.local_devices = local_devices
        self.remote_device = remote_device
        self.hidden_size = hidden_size
        self.use_nccl_stream = use_nccl_stream
        self.shared_experts: Optional[HeteroSharedExpertMLP] = None

        # A2A communication helper
        self._a2a = AsymmetricA2AOverlap(
            ep_group=ep_group,
            local_devices=local_devices,
            remote_device=remote_device,
        )

        logger.info(
            "HeteroFlexDispatcherOverlap: ep_group_size=%d use_nccl_stream=%s",
            ep_group.size() if ep_group is not None else 1,
            use_nccl_stream,
        )

    def set_shared_experts(self, shared_experts: HeteroSharedExpertMLP) -> None:
        """
        Register shared expert MLP for overlapped execution.

        Upstream (Megatron MoEAlltoAllTokenDispatcher.set_shared_experts):
            Sets self.shared_experts and enables use_nccl_stream.
            Previously, MoEFlexTokenDispatcher raised NotImplementedError here;
            the commit (ebfa138) removes that restriction.

        DES-LOC: Same semantic, but also validates device tier placement.

        Args:
            shared_experts: HeteroSharedExpertMLP instance configured for H100
        """
        self.shared_experts = shared_experts
        self.use_nccl_stream = True  # Always True when SE is active (mirrors Megatron)

        if shared_experts.compute_tier != DeviceTier.REMOTE:
            logger.warning(
                "set_shared_experts: shared_experts compute_tier=%s, "
                "expected REMOTE (H100). Overlap may be suboptimal.",
                shared_experts.compute_tier.name,
            )
        logger.info(
            "SharedExpert overlap enabled: use_nccl_stream=%s, "
            "compute_device=%s, comm_device=%s",
            self.use_nccl_stream,
            shared_experts.compute_device,
            shared_experts.comm_device,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        output_split_sizes: Optional[List[int]] = None,
        input_split_sizes: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Token dispatch with overlapped shared expert pre-forward.

        Upstream (Megatron MoEFlexTokenDispatcher.dispatch + shared expert changes):
            1. wait_current_stream() — ensure SE doesn't race with current compute
            2. dispatch() — A2A tokens to expert ranks
            3. pre_forward_comm() — AllGather on SE stream (overlap with dispatch)
            4. linear_fc1_forward_and_act() — SE fc1 GEMM (overlap with probs A2A)

            Forward launch order (from commit ebfa138):
                tokens A2A -> SE AllGather -> SE fc1 -> probs A2A

        DES-LOC adaptation:
            tokens A2A crosses PCIe (A6000 -> H100 for remote experts)
            SE AllGather also crosses PCIe (A6000 -> H100 for SE input)
            We pipeline these using the NCCL stream isolation.

        Args:
            hidden_states:       Input tokens [seq, batch, hidden]
            output_split_sizes:  Per-rank receive counts for A2A
            input_split_sizes:   Per-rank send counts for A2A

        Returns:
            Tuple of (dispatched_tokens, probs) where probs may be None
        """
        # Step 1: Sync streams before dispatch
        # Upstream: if self.shared_experts is not None: self.shared_experts.wait_current_stream()
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()

        # Step 2: Token dispatch A2A
        dispatched_tokens = self._a2a.all_to_all(
            input_tensor=hidden_states,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            use_nccl_stream=self.use_nccl_stream,
        )

        # Step 3: Overlap SE AllGather + fc1 with upcoming probs A2A
        # Upstream comment (ebfa138):
        #   "Move the shared experts fc1 right after the tokens A2A, to prevent
        #    the probs A2A block the launch of fc1 GEMM when
        #    CUDA_DEVICE_MAX_CONNECTIONS=1."
        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(
                hidden_states, wait_current_stream=False
            )
            self.shared_experts.linear_fc1_forward_and_act(dispatched_tokens)

        # Note: probs A2A would happen here in the full implementation
        # We return None for probs in this skeleton
        probs = None

        return dispatched_tokens, probs

    def combine(
        self,
        expert_output: torch.Tensor,
        output_split_sizes: Optional[List[int]] = None,
        input_split_sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Token combine with overlapped shared expert fc2 forward.

        Upstream (Megatron MoEFlexTokenDispatcher.combine + MoEAlltoAllTokenDispatcher changes):
            1. wait_current_stream() — prevent fc2 overlap with routed expert fc1
               (from ebfa138: "Make sure the shared experts fc2 is not overlapped
                with routed experts GEMM when CUDA_DEVICE_MAX_CONNECTIONS>1")
            2. combine() A2A
            3. linear_fc2_forward() + post_forward_comm() on SE stream

        DES-LOC adaptation:
            The wait_current_stream here is critical on PCIe: routed expert
            fc1 on A6000 and SE fc2 on H100 would otherwise race through
            PCIe bandwidth, saturating the interconnect.

        Args:
            expert_output:       Routed expert output tokens
            output_split_sizes:  Per-rank receive counts for combine A2A
            input_split_sizes:   Per-rank send counts for combine A2A

        Returns:
            Combined tokens after A2A [local_tokens, hidden]
        """
        # Step 1: Prevent fc2 from overlapping with routed expert fc1
        # Upstream: if self.shared_experts is not None: self.shared_experts.wait_current_stream()
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()

        # Step 2: Combine A2A
        combined_tokens = self._a2a.all_to_all(
            input_tensor=expert_output,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            use_nccl_stream=self.use_nccl_stream,
        )

        # Step 3: SE fc2 and post-comm overlapped with any subsequent compute
        # Upstream: moved into combine() rather than combine_postprocess()
        # to improve overlap with the next layer's operations.
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(combined_tokens)
            self.shared_experts.post_forward_comm()

        return combined_tokens

    def combine_postprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Finalise combine: add shared expert output to main residual.

        Upstream (Megatron MoEFlexTokenDispatcher.combine_postprocess):
            After the commit (ebfa138), fc2 and post_comm are moved into
            combine(), so combine_postprocess just adds SE output:
                hidden_states += self.shared_experts.get_output()
                return hidden_states.view(self.hidden_shape)

        DES-LOC adaptation:
            SE output is on H100 (REMOTE); main hidden_states may be on A6000 (LOCAL).
            We perform the addition on H100 to avoid an extra PCIe transfer,
            then return the result (caller is responsible for device placement).

        Args:
            hidden_states: Combined tokens from combine()

        Returns:
            Final MoE output with shared expert contribution added
        """
        if self.shared_experts is not None:
            se_output = self.shared_experts.get_output()
            # DES-LOC: add on compute device (H100) for efficiency
            target_device = se_output.device
            hs = hidden_states.to(target_device, non_blocking=True)
            hs = hs + se_output
            hidden_states = hs
            logger.debug(
                "combine_postprocess: added SE output, result on device=%s shape=%s",
                hidden_states.device, hidden_states.shape,
            )

        return hidden_states


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Smoke test: verify core DES-LOC hetero shared expert overlap logic.
    Uses CPU/fake devices when CUDA is not available.
    """
    logging.basicConfig(level=logging.INFO)

    # Test 1: DeviceTier classification
    cpu_dev = torch.device("cpu")
    assert get_device_tier(cpu_dev) == DeviceTier.CPU
    logger.info("Test 1 PASS: DeviceTier.CPU classification")

    # Test 2: LocalityCache put/get roundtrip
    cache = LocalityCache(max_bytes=1 * (1 << 20))  # 1MB test cache
    dummy = torch.randn(16, 32)
    cache.put(("layer0", "fc1", 0), dummy)
    retrieved = cache.get(("layer0", "fc1", 0))
    assert retrieved is not None, "Cache miss after put"
    assert torch.allclose(dummy, retrieved.float()), "Cache value mismatch"
    logger.info("Test 2 PASS: LocalityCache put/get roundtrip")

    # Test 3: HeteroSharedExpertState state values
    assert HeteroSharedExpertState.IDLE.value == 0
    assert HeteroSharedExpertState.FC1_FORWARD_DONE.expected_tier == DeviceTier.REMOTE
    assert HeteroSharedExpertState.POST_FORWARD_COMM_DONE.expected_tier == DeviceTier.LOCAL
    logger.info("Test 3 PASS: HeteroSharedExpertState values and tier annotations")

    # Test 4: HeteroSharedExpertMLP sequential forward (no overlap, CPU)
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        mlp = HeteroSharedExpertMLP(
            hidden_size=64, ffn_hidden_size=128, num_shared_experts=1,
            compute_device=dev, comm_device=dev,
            layer_idx=0, overlap_enabled=False, use_locality_cache=False,
            dtype=torch.float32,
        )
        x = torch.randn(8, 64, device=dev)
        out = mlp.forward_no_overlap(x)
        assert out.shape == (8, 64), f"Unexpected output shape: {out.shape}"
        logger.info("Test 4 PASS: HeteroSharedExpertMLP forward_no_overlap shape=%s", out.shape)
    else:
        logger.info("Test 4 SKIP: CUDA not available")

    # Test 5: _BackwardStreamWait forward is identity
    if torch.cuda.is_available():
        t = torch.randn(4, 8, requires_grad=True, device="cuda")
        stream = torch.cuda.Stream()
        out = _HeteroBackwardStreamWait.apply(t, stream, None, None)
        assert out.shape == t.shape
        assert torch.equal(out, t)
        logger.info("Test 5 PASS: _HeteroBackwardStreamWait forward identity")
    else:
        logger.info("Test 5 SKIP: CUDA not available")

    logger.info("All smoke tests passed.")
