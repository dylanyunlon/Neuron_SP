"""
deepspeed/inference/hetero_ep_buffer_alloc.py

DES-LOC Heterogeneous Expert Parallel Buffer Allocation
=========================================================

Upstream Design Intent (Megatron commit b45ae738):
---------------------------------------------------
Megatron-LM's fix addresses a subtle but critical bug in Expert Parallelism (EP=1) inference:
when running with a single expert-parallel rank, the original code skipped buffer allocation
entirely, assuming it was unnecessary. However, the mcore_fused_moe Triton kernel unconditionally
reads ``_valid_tokens_tensor`` as a raw pointer during CUDA graph replay — regardless of EP group
size. A null/uninitialized pointer at EP=1 caused silent memory corruption or kernel crashes.

The fix restructures the allocation logic from "EP>1 guards everything" to:
  1. NCCL dispatcher: always allocate (EP=1 included), because it's the explicit user request.
  2. NVLS dispatcher + EP>1: allocate symmetric memory for NVLink-based AllGather.
  3. NVLS dispatcher + EP=1: skip NVLink symmetric memory (requires multi-GPU NVLink topology),
     but still call ``allocate_valid_tokens_tensor()`` to ensure ``_valid_tokens_tensor`` holds
     a stable device pointer before any CUDA graph capture.

DES-LOC Adaptation: HeteroEPBufferAlloc
-----------------------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) operates on a heterogeneous GPU cluster:
  - 2x NVIDIA A6000 48GB (SM86, Ampere) — expert computation workers
  - 1x NVIDIA H100 NVL 96GB (SM90, Hopper) — attention/dense layer anchor + KV cache host
  - PCIe interconnect only (no NVLink between devices)
  - 1.5TB CPU DRAM as spill-over locality cache

The Megatron EP=1 fix maps directly onto DES-LOC's single-rank-per-device-class scenario:
each physical GPU is its own "EP group" from the perspective of expert routing (EP=1 per device
class), yet the MoE Triton kernels still require valid buffer pointers. DES-LOC extends this
with device-class-aware allocation:

  * H100 NVL is the "coordinator" device — it owns the authoritative ``_valid_tokens_tensor``
    and the locality cache metadata tensors.
  * A6000 devices are "worker" devices — they maintain shadow copies of ``_valid_tokens_tensor``
    that are synchronized lazily via PCIe DMA, not NVLink AllGather.
  * Buffer allocation is SM-architecture-aware: H100 (SM90) can use BF16 natively and supports
    larger tile sizes; A6000 (SM86) uses FP16 or FP32 accumulation with smaller tiles.
  * The "shared locality cache" in DES-LOC means that once a token's expert assignment is
    computed on H100, the routing metadata is pinned to CPU DRAM and DMA'd to A6000 workers
    on demand, avoiding redundant GPU→GPU transfers over slow PCIe.

Key DES-LOC invariants maintained here:
  1. ``_valid_tokens_tensor`` must be allocated before any CUDA graph capture on any device.
  2. Buffer shapes must account for per-device expert capacity (H100 handles 2x tokens vs A6000).
  3. Locality cache tensors (CPU-pinned) are allocated alongside GPU tensors for zero-copy access.
  4. EP group size reported to dispatchers reflects logical DES-LOC groups, not physical NCCL groups.

Author: Neuron_SP project (DES-LOC adaptation of Megatron b45ae738)
"""

from __future__ import annotations

import logging
import math
import os
import threading
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Module-level logger — structured, never spammy
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device class taxonomy for DES-LOC heterogeneous cluster
# ---------------------------------------------------------------------------

class DeviceClass(Enum):
    """Classifies GPUs by SM architecture for DES-LOC scheduling decisions."""
    HOPPER = auto()    # SM90 — H100 NVL 96GB, supports BF16 tensor cores, FP8
    AMPERE = auto()    # SM86 — A6000 48GB, supports BF16 but no FP8
    UNKNOWN = auto()   # Fallback for unexpected architectures


class DispatcherType(Enum):
    """MoE token dispatcher backend, mirroring Megatron's inference_moe_token_dispatcher_type."""
    NCCL = auto()       # Standard NCCL AllGather — works on any topology
    NVLS = auto()       # NVLink Symmetric — requires NVLink (unavailable in our cluster)
    DES_LOC_PCIe = auto()  # DES-LOC custom: PCIe DMA + locality cache


# ---------------------------------------------------------------------------
# Per-device buffer capacity configuration
# ---------------------------------------------------------------------------

@dataclass
class DeviceBufferConfig:
    """
    Capacity parameters for buffer allocation, resolved per device class.

    In DES-LOC, the H100 NVL is the coordinator and handles a larger share of
    the token budget (it has 2x VRAM and higher memory bandwidth than A6000).
    A6000 devices each handle a proportional expert shard.

    Attributes
    ----------
    device_class : DeviceClass
        Architecture class determining tile sizes and dtype preferences.
    hidden_size : int
        Model hidden dimension (e.g. 4096 for 7B models).
    moe_hidden_size : int
        Effective hidden size for MoE layers (may differ if latent MoE is used,
        e.g. SuperV3/UltraV3 style architectures with ``moe_latent_size``).
    max_tokens : int
        Maximum sequence tokens this device will process per step.
    topk : int
        MoE router top-k expert selections per token.
    ep_size : int
        Logical expert parallel group size seen by this device's dispatcher.
    dtype : torch.dtype
        Accumulation dtype, resolved per device class (BF16 on H100, FP16 on A6000).
    tp_size : int
        Tensor parallel degree (affects per-rank token count).
    locality_cache_capacity : int
        Number of routing metadata entries to keep in CPU DRAM locality cache.
        Set to 0 to disable (useful for devices with limited PCIe bandwidth budget).
    """
    device_class: DeviceClass
    hidden_size: int
    moe_hidden_size: int
    max_tokens: int
    topk: int
    ep_size: int
    dtype: torch.dtype
    tp_size: int = 1
    locality_cache_capacity: int = 0

    @property
    def per_rank_max_tokens(self) -> int:
        """Tokens handled by this TP rank, rounded up to next power of 2."""
        raw = math.ceil(self.max_tokens / max(self.tp_size, 1))
        return 1 << (raw - 1).bit_length()

    @property
    def expert_buffer_tokens(self) -> int:
        """
        Maximum tokens any single expert may receive, including capacity factor.

        DES-LOC applies a 1.5x capacity factor on H100 (larger buffer, more
        flexible routing) and 1.25x on A6000 (memory-constrained).
        """
        cf = 1.5 if self.device_class == DeviceClass.HOPPER else 1.25
        return math.ceil(self.per_rank_max_tokens * cf)


# ---------------------------------------------------------------------------
# Locality cache: CPU DRAM pinned tensors for zero-copy PCIe transfer
# ---------------------------------------------------------------------------

class LocalityCacheAllocator:
    """
    Manages CPU-pinned tensors that serve as the "Shared LOcality Cache" in DES-LOC.

    When the H100 computes token-to-expert routing decisions, it writes routing
    metadata into pinned CPU tensors. A6000 workers can then DMA-read these tensors
    directly over PCIe without an additional GPU→GPU hop, saving PCIe bandwidth.

    Thread-safe: allocation and invalidation are protected by a reentrant lock.
    """

    def __init__(self, capacity: int, hidden_size: int, topk: int, dtype: torch.dtype):
        """
        Parameters
        ----------
        capacity : int
            Maximum number of token routing records to cache.
        hidden_size : int
            Model hidden dimension (used to size hidden-state scratch buffers).
        topk : int
            MoE top-k, determines routing map width.
        dtype : torch.dtype
            Dtype for hidden-state tensors in the cache.
        """
        self._capacity = capacity
        self._hidden_size = hidden_size
        self._topk = topk
        self._dtype = dtype
        self._lock = threading.RLock()
        self._allocated = False

        # These are lazily allocated to avoid holding pinned memory when unused
        self._routing_map: Optional[torch.Tensor] = None      # [capacity, topk] int32
        self._probs: Optional[torch.Tensor] = None             # [capacity, topk] float32
        self._valid_tokens: Optional[torch.Tensor] = None      # [1] int32
        self._hidden_scratch: Optional[torch.Tensor] = None    # [capacity, hidden_size]

    def allocate(self) -> None:
        """
        Allocate all pinned-memory buffers. Idempotent.

        Must be called before CUDA graph capture. Pinned allocation is slow
        but happens once at model init, never inside a captured region.
        """
        with self._lock:
            if self._allocated:
                return
            logger.info(
                "LocalityCacheAllocator: allocating pinned CPU buffers "
                "(capacity=%d, hidden=%d, topk=%d, dtype=%s)",
                self._capacity, self._hidden_size, self._topk, self._dtype,
            )
            self._routing_map = torch.zeros(
                (self._capacity, self._topk), dtype=torch.int32, pin_memory=True
            )
            self._probs = torch.zeros(
                (self._capacity, self._topk), dtype=torch.float32, pin_memory=True
            )
            self._valid_tokens = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            self._hidden_scratch = torch.zeros(
                (self._capacity, self._hidden_size), dtype=self._dtype, pin_memory=True
            )
            self._allocated = True

    def invalidate(self) -> None:
        """Free pinned buffers and reset state. Called on context teardown."""
        with self._lock:
            self._routing_map = None
            self._probs = None
            self._valid_tokens = None
            self._hidden_scratch = None
            self._allocated = False
            logger.debug("LocalityCacheAllocator: pinned buffers released")

    @property
    def routing_map(self) -> Optional[torch.Tensor]:
        return self._routing_map

    @property
    def probs(self) -> Optional[torch.Tensor]:
        return self._probs

    @property
    def valid_tokens(self) -> Optional[torch.Tensor]:
        return self._valid_tokens

    @property
    def hidden_scratch(self) -> Optional[torch.Tensor]:
        return self._hidden_scratch

    def sync_valid_tokens_to_device(self, device_tensor: torch.Tensor) -> None:
        """
        Non-blocking PCIe DMA from CPU locality cache → GPU device tensor.

        Used by A6000 workers to read the token count that the H100 coordinator
        wrote after routing. Uses the current CUDA stream so it integrates
        naturally with overlapped compute/comm in DES-LOC pipelines.

        Parameters
        ----------
        device_tensor : torch.Tensor
            The GPU-resident ``_valid_tokens_tensor`` on the calling device.
        """
        if self._valid_tokens is None:
            raise RuntimeError(
                "LocalityCacheAllocator.sync_valid_tokens_to_device called before allocate()"
            )
        device_tensor.copy_(self._valid_tokens, non_blocking=True)


# ---------------------------------------------------------------------------
# Per-dispatcher class-level shared state (mirrors Megatron's class variables)
# ---------------------------------------------------------------------------

class _DispatcherSharedState:
    """
    Class-level shared state for DES-LOC dispatcher instances.

    Megatron uses class variables on InferenceAllGatherDispatcherBase to share
    ``_valid_tokens_tensor`` and ``_host_valid_tokens_estimate`` across all
    dispatcher instances (since metadata sync runs only on the first instance).

    DES-LOC extends this pattern: each DeviceClass has its own shared state
    because A6000 and H100 tensors live on different devices and must not alias.
    """

    def __init__(self, device_class: DeviceClass, cuda_device: int):
        self.device_class = device_class
        self.cuda_device = cuda_device
        self._valid_tokens_tensor: Optional[torch.Tensor] = None
        self._host_valid_tokens_estimate: Optional[int] = None
        self._lock = threading.Lock()

    def allocate_valid_tokens_tensor(self) -> None:
        """
        Allocate the per-step valid-tokens scalar on this device.

        Mirrors ``InferenceAllGatherDispatcherBase.allocate_valid_tokens_tensor()``
        from Megatron b45ae738. Must run outside CUDA graph capture so the
        stable address is available during graph replay.

        DES-LOC distinction: called independently per DeviceClass so that
        A6000 workers have their own stable pointers, not shared with H100.
        """
        with self._lock:
            if self._valid_tokens_tensor is not None:
                return
            with torch.cuda.device(self.cuda_device):
                self._valid_tokens_tensor = torch.zeros(
                    1, dtype=torch.int32, device=f"cuda:{self.cuda_device}"
                )
            logger.debug(
                "DispatcherSharedState[%s]: allocated _valid_tokens_tensor on cuda:%d",
                self.device_class.name, self.cuda_device,
            )

    def fill_valid_tokens(self, n: int) -> None:
        """Update the scalar in-place (called from dispatcher token_dispatch)."""
        if self._valid_tokens_tensor is None:
            raise RuntimeError(
                f"fill_valid_tokens called on {self.device_class.name} before allocation"
            )
        self._valid_tokens_tensor.fill_(n)

    @property
    def valid_tokens_tensor(self) -> Optional[torch.Tensor]:
        return self._valid_tokens_tensor


# ---------------------------------------------------------------------------
# Core buffer allocation engine
# ---------------------------------------------------------------------------

class HeteroEPBufferAlloc:
    """
    Heterogeneous Expert Parallel buffer allocator for DES-LOC inference.

    This class adapts Megatron's EP=1 buffer allocation fix (commit b45ae738)
    to the DES-LOC heterogeneous GPU cluster. It manages the lifecycle of:

      1. ``_valid_tokens_tensor`` — per-device GPU scalars required by Triton MoE kernels
         regardless of EP group size (the core Megatron fix).
      2. NCCL AllGather buffers — for explicit NCCL dispatcher configurations.
      3. PCIe-DMA buffers — DES-LOC's replacement for NVLS (which requires NVLink
         unavailable between our A6000 and H100).
      4. Locality cache (CPU-pinned) — shared routing metadata between devices.

    Allocation Strategy Per Device Class
    -------------------------------------
    H100 NVL (SM90, coordinator):
      - Owns authoritative ``_valid_tokens_tensor`` updated after each routing step.
      - Allocates NCCL or PCIe-DMA buffers based on dispatcher type.
      - Populates CPU locality cache so A6000 workers can DMA-fetch metadata.
      - Larger per-expert token capacity (1.5x factor) leveraging 96GB VRAM.

    A6000 (SM86, worker × 2):
      - Each device maintains its own ``_valid_tokens_tensor`` (separate CUDA device).
      - Reads routing metadata from CPU locality cache via PCIe DMA, never from H100 GPU.
      - Smaller per-expert token capacity (1.25x factor) respecting 48GB VRAM.
      - If dispatcher_type is NVLS, raises an informative error: our cluster lacks NVLink.

    EP=1 Handling (the Megatron fix, reinterpreted for DES-LOC)
    -------------------------------------------------------------
    In DES-LOC, each physical device is effectively EP=1 within its own GPU process.
    The Triton MoE kernel still needs a valid pointer for ``_valid_tokens_tensor``.
    ``allocate_for_device()`` always allocates this tensor, then conditionally
    allocates the heavier communication buffers based on dispatcher type and EP size.

    Usage
    -----
    At model initialization (before CUDA graph capture)::

        alloc = HeteroEPBufferAlloc.create_for_cluster(
            model_config=cfg,
            device_assignments={0: DeviceClass.HOPPER, 1: DeviceClass.AMPERE, 2: DeviceClass.AMPERE},
            max_tokens=8192,
            tp_size=1,
        )
        alloc.allocate_all()

    Parameters
    ----------
    device_configs : Dict[int, DeviceBufferConfig]
        Mapping from CUDA device index to its buffer configuration.
    dispatcher_type : DispatcherType
        Which dispatcher backend to use for the MoE AllGather.
    locality_cache_capacity : int
        Size of the CPU locality cache in token-routing records.
    """

    # Registry of live instances for health checks and teardown
    _instances: weakref.WeakSet["HeteroEPBufferAlloc"] = weakref.WeakSet()
    _instances_lock = threading.Lock()

    def __init__(
        self,
        device_configs: Dict[int, DeviceBufferConfig],
        dispatcher_type: DispatcherType,
        locality_cache_capacity: int = 4096,
    ):
        self._device_configs = device_configs
        self._dispatcher_type = dispatcher_type
        self._locality_cache_capacity: int = locality_cache_capacity

        # Per-device shared state (mirrors Megatron class-vars, but per device)
        self._shared_states: Dict[int, _DispatcherSharedState] = {}

        # NCCL AllGather buffers: device_idx → tensor
        self._nccl_hidden_buffers: Dict[int, torch.Tensor] = {}
        self._nccl_probs_buffers: Dict[int, torch.Tensor] = {}

        # PCIe-DMA staging buffers: device_idx → pinned CPU tensor
        self._pcie_staging_buffers: Dict[int, torch.Tensor] = {}

        # Locality cache allocators (one per coordinator device, i.e. H100)
        self._locality_caches: Dict[int, LocalityCacheAllocator] = {}

        self._allocated = False
        self._lock = threading.Lock()

        with HeteroEPBufferAlloc._instances_lock:
            HeteroEPBufferAlloc._instances.add(self)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_for_cluster(
        cls,
        model_config: "HeteroModelConfig",
        device_assignments: Dict[int, DeviceClass],
        max_tokens: int,
        tp_size: int = 1,
        dispatcher_type: Optional[DispatcherType] = None,
        locality_cache_capacity: int = 4096,
    ) -> "HeteroEPBufferAlloc":
        """
        Construct a HeteroEPBufferAlloc from a model config and device map.

        Resolves per-device dtype (BF16 for H100, FP16 for A6000) and selects
        the dispatcher type based on model config if not explicitly provided.

        Parameters
        ----------
        model_config : HeteroModelConfig
            Model configuration with hidden_size, moe_latent_size, moe_router_topk, etc.
        device_assignments : Dict[int, DeviceClass]
            Maps CUDA device index → DeviceClass.
        max_tokens : int
            Maximum tokens processed in one inference step (across all TP ranks).
        tp_size : int
            Tensor parallel degree (determines per-rank token count).
        dispatcher_type : DispatcherType or None
            Override dispatcher selection; if None, inferred from model_config.
        locality_cache_capacity : int
            CPU locality cache size in token-routing records.

        Returns
        -------
        HeteroEPBufferAlloc
            Configured but not yet allocated instance. Call ``allocate_all()`` next.
        """
        # Resolve dispatcher type
        if dispatcher_type is None:
            raw = getattr(model_config, "inference_moe_token_dispatcher_type", "nccl")
            if raw == "nccl":
                dispatcher_type = DispatcherType.NCCL
            elif raw == "nvls":
                # DES-LOC: NVLS requires NVLink, unavailable between A6000 and H100 over PCIe.
                # Transparently downgrade to PCIe-DMA dispatcher and warn the operator.
                logger.warning(
                    "HeteroEPBufferAlloc: model_config requests NVLS dispatcher but cluster "
                    "has no NVLink (PCIe-only topology). Downgrading to DES_LOC_PCIe dispatcher. "
                    "Set inference_moe_token_dispatcher_type='nccl' or 'des_loc_pcie' explicitly "
                    "to suppress this warning."
                )
                dispatcher_type = DispatcherType.DES_LOC_PCIe
            else:
                dispatcher_type = DispatcherType.DES_LOC_PCIe

        moe_hidden = getattr(model_config, "moe_latent_size", None) or model_config.hidden_size

        device_configs: Dict[int, DeviceBufferConfig] = {}
        for dev_idx, dev_class in device_assignments.items():
            # Dtype selection: H100 uses BF16 natively; A6000 supports BF16 but
            # DES-LOC defaults to FP16 for A6000 to maintain numerical parity with
            # older CUDA graphs that may have been captured with FP16 accumulation.
            if dev_class == DeviceClass.HOPPER:
                dtype = torch.bfloat16
                ep_size = 1  # H100 is sole coordinator, EP=1 for its local group
            elif dev_class == DeviceClass.AMPERE:
                dtype = torch.float16
                ep_size = 1  # Each A6000 is its own EP shard in DES-LOC
            else:
                dtype = torch.float32
                ep_size = 1

            device_configs[dev_idx] = DeviceBufferConfig(
                device_class=dev_class,
                hidden_size=model_config.hidden_size,
                moe_hidden_size=moe_hidden,
                max_tokens=max_tokens,
                topk=model_config.moe_router_topk,
                ep_size=ep_size,
                dtype=dtype,
                tp_size=tp_size,
                locality_cache_capacity=locality_cache_capacity
                if dev_class == DeviceClass.HOPPER
                else 0,
            )

        return cls(
            device_configs=device_configs,
            dispatcher_type=dispatcher_type,
            locality_cache_capacity=locality_cache_capacity,
        )

    # ------------------------------------------------------------------
    # Public allocation API
    # ------------------------------------------------------------------

    def allocate_all(self) -> None:
        """
        Allocate all buffers for all registered devices.

        Must be called before CUDA graph capture. Idempotent.

        This is the DES-LOC equivalent of Megatron's buffer allocation block
        inside ``DynamicInferenceContext._allocate_dispatcher_buffers()``,
        restructured to handle the heterogeneous EP=1-per-device topology.

        Allocation order matters for CUDA graph stability:
          1. Shared state (``_valid_tokens_tensor``) — always, for every device.
          2. NCCL or PCIe-DMA communication buffers — conditional on dispatcher type.
          3. Locality cache (CPU-pinned) — only for coordinator (H100) devices.
        """
        with self._lock:
            if self._allocated:
                return

            logger.info(
                "HeteroEPBufferAlloc: starting allocation for %d devices, dispatcher=%s",
                len(self._device_configs), self._dispatcher_type.name,
            )

            for dev_idx, cfg in self._device_configs.items():
                self._allocate_for_device(dev_idx, cfg)

            self._allocated = True
            logger.info("HeteroEPBufferAlloc: allocation complete")

    def _allocate_for_device(self, dev_idx: int, cfg: DeviceBufferConfig) -> None:
        """
        Allocate all buffers for a single CUDA device.

        This is the per-device implementation of the Megatron EP=1 fix:
        ``_valid_tokens_tensor`` is ALWAYS allocated first (step 1), then
        heavier buffers are conditionally allocated (step 2+).

        Parameters
        ----------
        dev_idx : int
            CUDA device index.
        cfg : DeviceBufferConfig
            Buffer sizing configuration for this device.
        """
        # --- Step 1: Always allocate _valid_tokens_tensor ---
        # This is the core of the Megatron EP=1 fix. Even at EP=1, Triton MoE
        # kernels read this pointer. DES-LOC extends it: every physical GPU
        # (H100 or A6000) needs its own stable pointer on its own device.
        state = _DispatcherSharedState(
            device_class=cfg.device_class, cuda_device=dev_idx
        )
        state.allocate_valid_tokens_tensor()
        self._shared_states[dev_idx] = state

        # --- Step 2: Dispatcher-specific communication buffers ---
        if self._dispatcher_type == DispatcherType.NCCL:
            # NCCL path: always allocate, mirroring Megatron's fix where NCCL
            # allocation is no longer guarded by ep_size > 1.
            self._allocate_nccl_buffers(dev_idx, cfg)

        elif self._dispatcher_type == DispatcherType.NVLS:
            # NVLS requires NVLink between GPUs. In DES-LOC with PCIe-only
            # topology, this path should never be reached (factory downgrades
            # to DES_LOC_PCIe), but guard here for safety.
            if cfg.ep_size > 1:
                raise RuntimeError(
                    f"NVLS dispatcher requested on cuda:{dev_idx} but cluster "
                    "has no NVLink. Use DispatcherType.DES_LOC_PCIe instead."
                )
            # EP=1 + NVLS: valid_tokens_tensor is already allocated above.
            # No symmetric memory needed. This exactly mirrors Megatron's new
            # else-branch: ``InferenceAllGatherDispatcherBase.allocate_valid_tokens_tensor()``
            logger.debug(
                "cuda:%d [%s]: EP=1 + NVLS — skipping symmetric memory, "
                "_valid_tokens_tensor already allocated",
                dev_idx, cfg.device_class.name,
            )

        elif self._dispatcher_type == DispatcherType.DES_LOC_PCIe:
            # DES-LOC custom path: PCIe DMA staging buffers replace NVLink AllGather.
            self._allocate_pcie_staging_buffers(dev_idx, cfg)

        # --- Step 3: Locality cache for coordinator devices ---
        if cfg.device_class == DeviceClass.HOPPER and cfg.locality_cache_capacity > 0:
            self._allocate_locality_cache(dev_idx, cfg)

    def _allocate_nccl_buffers(self, dev_idx: int, cfg: DeviceBufferConfig) -> None:
        """
        Allocate NCCL AllGather communication buffers for the given device.

        Mirrors ``NCCLAllGatherDispatcher.allocate_buffers()`` from Megatron,
        adapted for per-device allocation in the DES-LOC multi-device context.

        Buffer shapes:
          hidden_buffer : [per_rank_max_tokens * ep_size, moe_hidden_size]
          probs_buffer  : [per_rank_max_tokens * ep_size, topk]

        For DES-LOC at EP=1 (our typical case), ep_size=1 so these are
        single-rank buffers — but they still must be allocated to satisfy
        the dispatcher's internal pointer checks.
        """
        with torch.cuda.device(dev_idx):
            total_tokens = cfg.per_rank_max_tokens * max(cfg.ep_size, 1)
            hidden_buf = torch.zeros(
                (total_tokens, cfg.moe_hidden_size),
                dtype=cfg.dtype,
                device=f"cuda:{dev_idx}",
            )
            probs_buf = torch.zeros(
                (total_tokens, cfg.topk),
                dtype=torch.float32,
                device=f"cuda:{dev_idx}",
            )
            self._nccl_hidden_buffers[dev_idx] = hidden_buf
            self._nccl_probs_buffers[dev_idx] = probs_buf
        logger.debug(
            "cuda:%d [%s]: NCCL buffers allocated — hidden=%s probs=%s",
            dev_idx, cfg.device_class.name,
            tuple(hidden_buf.shape), tuple(probs_buf.shape),
        )

    def _allocate_pcie_staging_buffers(self, dev_idx: int, cfg: DeviceBufferConfig) -> None:
        """
        Allocate PCIe DMA staging buffers for DES-LOC inter-device routing metadata.

        Unlike NVLS (which uses NVLink symmetric memory for zero-copy GPU→GPU),
        DES-LOC routes metadata through CPU DRAM:
          H100 writes → CPU pinned buffer → PCIe DMA → A6000 reads

        The staging buffer on the GPU side is a small receive-side tensor that
        DMA operations write into. It is intentionally small (token counts and
        routing indices) to minimize PCIe bandwidth consumption per step.

        The actual hidden-state transfer uses a separate path in DES-LOC's
        pipeline (chunked PCIe DMA with double-buffering, managed elsewhere).
        """
        with torch.cuda.device(dev_idx):
            # Routing map staging: [expert_buffer_tokens, topk] int32
            routing_staging = torch.zeros(
                (cfg.expert_buffer_tokens, cfg.topk),
                dtype=torch.int32,
                device=f"cuda:{dev_idx}",
            )
            # Probability staging: [expert_buffer_tokens, topk] float32
            probs_staging = torch.zeros(
                (cfg.expert_buffer_tokens, cfg.topk),
                dtype=torch.float32,
                device=f"cuda:{dev_idx}",
            )
            # CPU-pinned mirror for zero-copy PCIe reads
            routing_pinned = torch.zeros(
                (cfg.expert_buffer_tokens, cfg.topk),
                dtype=torch.int32,
                pin_memory=True,
            )
            probs_pinned = torch.zeros(
                (cfg.expert_buffer_tokens, cfg.topk),
                dtype=torch.float32,
                pin_memory=True,
            )

            self._pcie_staging_buffers[dev_idx] = routing_staging

        logger.debug(
            "cuda:%d [%s]: PCIe staging buffers allocated — routing=%s probs=%s "
            "(pinned mirrors also allocated)",
            dev_idx, cfg.device_class.name,
            tuple(routing_staging.shape), tuple(probs_staging.shape),
        )

    def _allocate_locality_cache(self, dev_idx: int, cfg: DeviceBufferConfig) -> None:
        """
        Allocate the CPU locality cache for the coordinator (H100) device.

        The locality cache stores recent token-routing decisions in CPU DRAM so
        that A6000 workers can fetch them via PCIe DMA without re-computing
        routing or pulling data from the H100 GPU directly.

        Only allocated on H100 (HOPPER) devices to avoid unnecessary pinned
        memory consumption on memory-constrained A6000 workers.
        """
        cache = LocalityCacheAllocator(
            capacity=cfg.locality_cache_capacity,
            hidden_size=cfg.moe_hidden_size,
            topk=cfg.topk,
            dtype=cfg.dtype,
        )
        cache.allocate()
        self._locality_caches[dev_idx] = cache

    # ------------------------------------------------------------------
    # Runtime metadata update (called per inference step)
    # ------------------------------------------------------------------

    def update_valid_tokens(self, dev_idx: int, n_tokens: int) -> None:
        """
        Update the ``_valid_tokens_tensor`` scalar for a device after routing.

        Mirrors the EP=1 branch added in Megatron b45ae738 to both
        ``NCCLAllGatherDispatcher`` and ``NVLSAllGatherVDispatcher``:

            if self.ep_size == 1:
                if self._runs_metadata_sync:
                    InferenceAllGatherDispatcherBase._valid_tokens_tensor.fill_(hidden_states.shape[0])
                return hidden_states, probs

        DES-LOC calls this from the DES-LOC execution controller after each
        token_dispatch step, ensuring the scalar stays current for CUDA graph
        replay even when no AllGather communication occurs (EP=1 case).

        If this device is a coordinator (H100) with a locality cache, also
        updates the CPU-pinned cache so A6000 workers can DMA-fetch it.

        Parameters
        ----------
        dev_idx : int
            CUDA device index to update.
        n_tokens : int
            Number of valid tokens in this step (typically hidden_states.shape[0]).
        """
        state = self._shared_states.get(dev_idx)
        if state is None:
            raise KeyError(f"update_valid_tokens: no shared state for cuda:{dev_idx}")
        state.fill_valid_tokens(n_tokens)

        # Propagate to locality cache if this is a coordinator device
        cache = self._locality_caches.get(dev_idx)
        if cache is not None and cache.valid_tokens is not None:
            cache.valid_tokens.fill_(n_tokens)

    def sync_valid_tokens_to_worker(
        self, coordinator_dev: int, worker_dev: int
    ) -> None:
        """
        DMA the valid_tokens count from the H100 coordinator to an A6000 worker.

        In DES-LOC, routing decisions are made on the H100 and then shared
        with A6000 workers via the locality cache. This method triggers the
        PCIe transfer from the CPU-pinned locality cache to the worker's
        ``_valid_tokens_tensor``.

        Uses non-blocking copy on the current CUDA stream for the worker device,
        allowing overlap with expert computation already in flight on that device.

        Parameters
        ----------
        coordinator_dev : int
            CUDA device index of the H100 coordinator (source).
        worker_dev : int
            CUDA device index of the A6000 worker (destination).
        """
        cache = self._locality_caches.get(coordinator_dev)
        if cache is None:
            raise KeyError(
                f"sync_valid_tokens_to_worker: no locality cache on coordinator cuda:{coordinator_dev}"
            )
        worker_state = self._shared_states.get(worker_dev)
        if worker_state is None or worker_state.valid_tokens_tensor is None:
            raise KeyError(
                f"sync_valid_tokens_to_worker: worker cuda:{worker_dev} has no valid_tokens_tensor"
            )
        cache.sync_valid_tokens_to_device(worker_state.valid_tokens_tensor)

    # ------------------------------------------------------------------
    # Introspection / health
    # ------------------------------------------------------------------

    def get_valid_tokens_tensor(self, dev_idx: int) -> Optional[torch.Tensor]:
        """Return the ``_valid_tokens_tensor`` for ``dev_idx``, or None if unallocated."""
        state = self._shared_states.get(dev_idx)
        return state.valid_tokens_tensor if state else None

    def get_nccl_hidden_buffer(self, dev_idx: int) -> Optional[torch.Tensor]:
        """Return the NCCL hidden-state AllGather buffer for ``dev_idx``."""
        return self._nccl_hidden_buffers.get(dev_idx)

    def get_locality_cache(self, dev_idx: int) -> Optional[LocalityCacheAllocator]:
        """Return the locality cache allocator for coordinator device ``dev_idx``."""
        return self._locality_caches.get(dev_idx)

    def is_allocated(self) -> bool:
        """Return True if ``allocate_all()`` has completed successfully."""
        return self._allocated

    def allocation_summary(self) -> Dict[str, object]:
        """
        Return a dict summarising allocated buffer sizes for diagnostics.

        Useful for logging at model startup to verify buffer footprint.
        """
        summary: Dict[str, object] = {
            "dispatcher_type": self._dispatcher_type.name,
            "n_devices": len(self._device_configs),
            "allocated": self._allocated,
            "devices": {},
        }
        for dev_idx, cfg in self._device_configs.items():
            state = self._shared_states.get(dev_idx)
            vt = state.valid_tokens_tensor if state else None
            dev_summary = {
                "device_class": cfg.device_class.name,
                "dtype": str(cfg.dtype),
                "per_rank_max_tokens": cfg.per_rank_max_tokens,
                "expert_buffer_tokens": cfg.expert_buffer_tokens,
                "valid_tokens_tensor_allocated": vt is not None,
                "nccl_hidden_buffer_shape": (
                    tuple(self._nccl_hidden_buffers[dev_idx].shape)
                    if dev_idx in self._nccl_hidden_buffers
                    else None
                ),
                "pcie_staging_allocated": dev_idx in self._pcie_staging_buffers,
                "locality_cache_allocated": dev_idx in self._locality_caches,
            }
            summary["devices"][f"cuda:{dev_idx}"] = dev_summary
        return summary

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def deallocate(self) -> None:
        """
        Release all allocated buffers and reset state.

        Deallocates GPU tensors (allowing CUDA allocator to reclaim memory)
        and CPU-pinned tensors in the locality caches. After this call,
        ``allocate_all()`` can be called again if needed.
        """
        with self._lock:
            for cache in self._locality_caches.values():
                cache.invalidate()
            self._locality_caches.clear()
            self._nccl_hidden_buffers.clear()
            self._nccl_probs_buffers.clear()
            self._pcie_staging_buffers.clear()
            for state in self._shared_states.values():
                state._valid_tokens_tensor = None
            self._shared_states.clear()
            self._allocated = False
            logger.debug("HeteroEPBufferAlloc: all buffers deallocated")

    def __repr__(self) -> str:
        return (
            f"HeteroEPBufferAlloc("
            f"dispatcher={self._dispatcher_type.name}, "
            f"devices={list(self._device_configs.keys())}, "
            f"allocated={self._allocated})"
        )


# ---------------------------------------------------------------------------
# Model config stub (for standalone use / tests — real code uses DeepSpeed config)
# ---------------------------------------------------------------------------

@dataclass
class HeteroModelConfig:
    """
    Minimal model configuration for HeteroEPBufferAlloc.

    In production Neuron_SP / DES-LOC, this is populated from DeepSpeed's
    model configuration and the DES-LOC heterogeneous cluster descriptor.
    Provided here for standalone use and unit tests.

    Attributes
    ----------
    hidden_size : int
        Model hidden dimension.
    moe_latent_size : int or None
        Latent MoE hidden size (SuperV3/UltraV3 style). If None, ``hidden_size`` is used.
    moe_router_topk : int
        Number of experts selected per token by the MoE router.
    inference_moe_token_dispatcher_type : str
        One of 'nccl', 'nvls', 'des_loc_pcie'. Mirrors Megatron's config field.
    """
    hidden_size: int = 4096
    moe_latent_size: Optional[int] = None
    moe_router_topk: int = 2
    inference_moe_token_dispatcher_type: str = "nccl"


# ---------------------------------------------------------------------------
# DES-LOC cluster descriptor helpers
# ---------------------------------------------------------------------------

def detect_device_class(device_idx: int) -> DeviceClass:
    """
    Detect the DeviceClass of a CUDA device by inspecting its SM architecture.

    Uses ``torch.cuda.get_device_capability()`` which returns (major, minor).
    SM90 → Hopper (H100), SM86 → Ampere (A6000).

    Parameters
    ----------
    device_idx : int
        CUDA device index to inspect.

    Returns
    -------
    DeviceClass
        Detected architecture class.
    """
    if not torch.cuda.is_available():
        return DeviceClass.UNKNOWN
    n_dev = torch.cuda.device_count()
    if device_idx >= n_dev:
        return DeviceClass.UNKNOWN
    major, minor = torch.cuda.get_device_capability(device_idx)
    sm = major * 10 + minor
    if sm >= 90:
        return DeviceClass.HOPPER
    elif sm >= 80:
        return DeviceClass.AMPERE
    return DeviceClass.UNKNOWN


def build_cluster_device_assignments(
    device_indices: Optional[List[int]] = None,
) -> Dict[int, DeviceClass]:
    """
    Build a device → DeviceClass mapping for the DES-LOC cluster.

    If ``device_indices`` is None, probes all visible CUDA devices.

    Parameters
    ----------
    device_indices : list of int or None
        Specific device indices to probe. If None, probes 0..N-1.

    Returns
    -------
    dict
        Mapping from CUDA device index to DeviceClass.
    """
    if not torch.cuda.is_available():
        logger.warning("build_cluster_device_assignments: no CUDA devices available")
        return {}

    if device_indices is None:
        device_indices = list(range(torch.cuda.device_count()))

    assignments: Dict[int, DeviceClass] = {}
    for idx in device_indices:
        cls = detect_device_class(idx)
        assignments[idx] = cls
        logger.debug("cuda:%d → %s", idx, cls.name)

    return assignments


# ---------------------------------------------------------------------------
# Round-up utility (mirrors Megatron's DynamicInferenceContext.round_up_tokens)
# ---------------------------------------------------------------------------

def round_up_tokens(n: int, alignment: int = 64) -> int:
    """
    Round ``n`` up to the next multiple of ``alignment``.

    Used to compute worst-case token counts for buffer sizing, mirroring
    ``DynamicInferenceContext.round_up_tokens()`` in Megatron.

    Parameters
    ----------
    n : int
        Raw token count.
    alignment : int
        Alignment granularity (default 64, matching Megatron's CUDA graph bucket size).

    Returns
    -------
    int
        Rounded-up token count.
    """
    return math.ceil(n / alignment) * alignment


# ---------------------------------------------------------------------------
# Integration point: DES-LOC execution controller hook
# ---------------------------------------------------------------------------

class DESLOCBufferHook:
    """
    Thin adapter between DES-LOC's execution controller and HeteroEPBufferAlloc.

    The DES-LOC execution controller calls this hook at two points:
      1. ``on_model_init()`` — allocates all buffers before CUDA graph capture.
      2. ``on_token_dispatch(dev_idx, n_tokens)`` — updates valid_tokens scalars
         after each routing step, matching Megatron's EP=1 dispatcher fix.

    This separation keeps HeteroEPBufferAlloc stateless with respect to the
    execution timeline, making it easier to unit-test allocation independently.
    """

    def __init__(self, alloc: HeteroEPBufferAlloc, coordinator_dev: int):
        """
        Parameters
        ----------
        alloc : HeteroEPBufferAlloc
            The buffer allocator to drive.
        coordinator_dev : int
            CUDA device index of the H100 coordinator (for valid_tokens sync).
        """
        self._alloc = alloc
        self._coordinator_dev = coordinator_dev

    def on_model_init(self) -> None:
        """Called once at model initialization, before any CUDA graph capture."""
        self._alloc.allocate_all()
        summary = self._alloc.allocation_summary()
        logger.info(
            "DESLOCBufferHook: buffer allocation complete — %s",
            {k: v for k, v in summary.items() if k != "devices"},
        )

    def on_token_dispatch(self, dev_idx: int, n_tokens: int) -> None:
        """
        Called after each MoE routing step to update valid_tokens scalars.

        For the coordinator device, also propagates to the locality cache so
        worker devices can DMA-fetch the updated count.

        Parameters
        ----------
        dev_idx : int
            CUDA device index where routing just completed.
        n_tokens : int
            ``hidden_states.shape[0]`` from the dispatcher's token_dispatch.
        """
        self._alloc.update_valid_tokens(dev_idx, n_tokens)

        # If this is the coordinator, sync to all worker devices
        if dev_idx == self._coordinator_dev:
            configs = self._alloc._device_configs
            for worker_dev, cfg in configs.items():
                if worker_dev != self._coordinator_dev:
                    try:
                        self._alloc.sync_valid_tokens_to_worker(
                            coordinator_dev=self._coordinator_dev,
                            worker_dev=worker_dev,
                        )
                    except KeyError:
                        # Worker may not have locality cache sync configured
                        pass


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    class TestDeviceBufferConfig(unittest.TestCase):
        def setUp(self):
            self.cfg_hopper = DeviceBufferConfig(
                device_class=DeviceClass.HOPPER,
                hidden_size=4096,
                moe_hidden_size=4096,
                max_tokens=8192,
                topk=2,
                ep_size=1,
                dtype=torch.bfloat16,
                tp_size=1,
                locality_cache_capacity=2048,
            )
            self.cfg_ampere = DeviceBufferConfig(
                device_class=DeviceClass.AMPERE,
                hidden_size=4096,
                moe_hidden_size=4096,
                max_tokens=8192,
                topk=2,
                ep_size=1,
                dtype=torch.float16,
                tp_size=2,
                locality_cache_capacity=0,
            )

        def test_per_rank_max_tokens_power_of_two(self):
            # 8192 / 1 = 8192, next pow2 = 8192
            self.assertEqual(self.cfg_hopper.per_rank_max_tokens, 8192)
            # 8192 / 2 = 4096, next pow2 = 4096
            self.assertEqual(self.cfg_ampere.per_rank_max_tokens, 4096)

        def test_expert_buffer_tokens_capacity_factor(self):
            # Hopper: 1.5x
            self.assertEqual(
                self.cfg_hopper.expert_buffer_tokens, math.ceil(8192 * 1.5)
            )
            # Ampere: 1.25x
            self.assertEqual(
                self.cfg_ampere.expert_buffer_tokens, math.ceil(4096 * 1.25)
            )

    class TestRoundUpTokens(unittest.TestCase):
        def test_already_aligned(self):
            self.assertEqual(round_up_tokens(128, 64), 128)

        def test_needs_rounding(self):
            self.assertEqual(round_up_tokens(65, 64), 128)

        def test_zero(self):
            self.assertEqual(round_up_tokens(0, 64), 0)

        def test_one(self):
            self.assertEqual(round_up_tokens(1, 64), 64)

    class TestLocalityCacheAllocator(unittest.TestCase):
        def test_allocate_and_invalidate(self):
            cache = LocalityCacheAllocator(
                capacity=512, hidden_size=64, topk=2, dtype=torch.float16
            )
            self.assertIsNone(cache.routing_map)
            cache.allocate()
            self.assertIsNotNone(cache.routing_map)
            self.assertEqual(cache.routing_map.shape, (512, 2))
            self.assertIsNotNone(cache.valid_tokens)
            self.assertEqual(cache.valid_tokens.shape, (1,))
            cache.invalidate()
            self.assertIsNone(cache.routing_map)

        def test_idempotent_allocate(self):
            cache = LocalityCacheAllocator(
                capacity=64, hidden_size=32, topk=1, dtype=torch.float32
            )
            cache.allocate()
            rt1 = cache.routing_map
            cache.allocate()
            rt2 = cache.routing_map
            # Same object — idempotent
            self.assertIs(rt1, rt2)

        def test_allocate_before_sync_raises(self):
            cache = LocalityCacheAllocator(
                capacity=64, hidden_size=32, topk=1, dtype=torch.float32
            )
            dummy = torch.zeros(1, dtype=torch.int32)
            with self.assertRaises(RuntimeError):
                cache.sync_valid_tokens_to_device(dummy)

    class TestDispatcherSharedState(unittest.TestCase):
        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        def test_allocate_valid_tokens_tensor(self):
            state = _DispatcherSharedState(
                device_class=DeviceClass.AMPERE, cuda_device=0
            )
            self.assertIsNone(state.valid_tokens_tensor)
            state.allocate_valid_tokens_tensor()
            self.assertIsNotNone(state.valid_tokens_tensor)
            self.assertEqual(state.valid_tokens_tensor.shape, (1,))
            self.assertEqual(state.valid_tokens_tensor.device.index, 0)
            self.assertEqual(state.valid_tokens_tensor.dtype, torch.int32)

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        def test_idempotent_allocate(self):
            state = _DispatcherSharedState(
                device_class=DeviceClass.HOPPER, cuda_device=0
            )
            state.allocate_valid_tokens_tensor()
            t1 = state.valid_tokens_tensor
            state.allocate_valid_tokens_tensor()
            t2 = state.valid_tokens_tensor
            self.assertIs(t1, t2)

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        def test_fill_valid_tokens(self):
            state = _DispatcherSharedState(
                device_class=DeviceClass.AMPERE, cuda_device=0
            )
            state.allocate_valid_tokens_tensor()
            state.fill_valid_tokens(42)
            self.assertEqual(state.valid_tokens_tensor.item(), 42)

        def test_fill_before_allocate_raises(self):
            state = _DispatcherSharedState(
                device_class=DeviceClass.AMPERE, cuda_device=0
            )
            with self.assertRaises(RuntimeError):
                state.fill_valid_tokens(10)

    class TestHeteroModelConfig(unittest.TestCase):
        def test_defaults(self):
            cfg = HeteroModelConfig()
            self.assertEqual(cfg.hidden_size, 4096)
            self.assertIsNone(cfg.moe_latent_size)
            self.assertEqual(cfg.moe_router_topk, 2)
            self.assertEqual(cfg.inference_moe_token_dispatcher_type, "nccl")

        def test_moe_hidden_resolution(self):
            # Mirrors Megatron: moe_hidden = moe_latent_size or hidden_size
            cfg = HeteroModelConfig(hidden_size=4096, moe_latent_size=None)
            moe_hidden = cfg.moe_latent_size or cfg.hidden_size
            self.assertEqual(moe_hidden, 4096)

            cfg2 = HeteroModelConfig(hidden_size=4096, moe_latent_size=512)
            moe_hidden2 = cfg2.moe_latent_size or cfg2.hidden_size
            self.assertEqual(moe_hidden2, 512)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    class TestHeteroEPBufferAllocCUDA(unittest.TestCase):
        """
        Tests that require at least one CUDA device.
        Runs against cuda:0 only to work on any single-GPU CI machine.
        """

        def _make_alloc(self, dispatcher_type: DispatcherType) -> HeteroEPBufferAlloc:
            cfg = DeviceBufferConfig(
                device_class=DeviceClass.AMPERE,
                hidden_size=128,
                moe_hidden_size=128,
                max_tokens=256,
                topk=2,
                ep_size=1,
                dtype=torch.float16,
                tp_size=1,
                locality_cache_capacity=0,
            )
            return HeteroEPBufferAlloc(
                device_configs={0: cfg},
                dispatcher_type=dispatcher_type,
                locality_cache_capacity=0,
            )

        def test_nccl_allocates_valid_tokens_tensor(self):
            alloc = self._make_alloc(DispatcherType.NCCL)
            self.assertFalse(alloc.is_allocated())
            alloc.allocate_all()
            self.assertTrue(alloc.is_allocated())
            vt = alloc.get_valid_tokens_tensor(0)
            self.assertIsNotNone(vt)
            self.assertEqual(vt.shape, (1,))
            self.assertEqual(vt.item(), 0)
            alloc.deallocate()

        def test_nccl_allocates_hidden_buffer(self):
            alloc = self._make_alloc(DispatcherType.NCCL)
            alloc.allocate_all()
            hb = alloc.get_nccl_hidden_buffer(0)
            self.assertIsNotNone(hb)
            # per_rank_max_tokens for 256 tokens = 256 (already power of 2), ep_size=1
            self.assertEqual(hb.shape[0], 256)
            self.assertEqual(hb.shape[1], 128)
            alloc.deallocate()

        def test_pcie_allocates_valid_tokens_only_at_ep1(self):
            alloc = self._make_alloc(DispatcherType.DES_LOC_PCIe)
            alloc.allocate_all()
            vt = alloc.get_valid_tokens_tensor(0)
            self.assertIsNotNone(vt)
            # No NCCL hidden buffer for PCIe dispatcher
            hb = alloc.get_nccl_hidden_buffer(0)
            self.assertIsNone(hb)
            alloc.deallocate()

        def test_idempotent_allocation(self):
            alloc = self._make_alloc(DispatcherType.NCCL)
            alloc.allocate_all()
            alloc.allocate_all()  # second call must be a no-op
            self.assertTrue(alloc.is_allocated())
            alloc.deallocate()

        def test_update_valid_tokens(self):
            alloc = self._make_alloc(DispatcherType.NCCL)
            alloc.allocate_all()
            alloc.update_valid_tokens(0, 77)
            vt = alloc.get_valid_tokens_tensor(0)
            self.assertEqual(vt.item(), 77)
            alloc.deallocate()

        def test_update_valid_tokens_before_alloc_raises(self):
            alloc = self._make_alloc(DispatcherType.NCCL)
            with self.assertRaises((KeyError, RuntimeError)):
                alloc.update_valid_tokens(0, 10)

        def test_allocation_summary_keys(self):
            alloc = self._make_alloc(DispatcherType.NCCL)
            alloc.allocate_all()
            summary = alloc.allocation_summary()
            self.assertIn("dispatcher_type", summary)
            self.assertIn("n_devices", summary)
            self.assertIn("allocated", summary)
            self.assertIn("devices", summary)
            self.assertIn("cuda:0", summary["devices"])
            dev_info = summary["devices"]["cuda:0"]
            self.assertTrue(dev_info["valid_tokens_tensor_allocated"])
            alloc.deallocate()

        def test_deallocate_resets_state(self):
            alloc = self._make_alloc(DispatcherType.NCCL)
            alloc.allocate_all()
            alloc.deallocate()
            self.assertFalse(alloc.is_allocated())
            self.assertIsNone(alloc.get_valid_tokens_tensor(0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    class TestHeteroEPBufferAllocWithLocalityCache(unittest.TestCase):
        """Tests for the coordinator device locality cache path."""

        def _make_hopper_alloc(self) -> HeteroEPBufferAlloc:
            cfg = DeviceBufferConfig(
                device_class=DeviceClass.HOPPER,
                hidden_size=64,
                moe_hidden_size=64,
                max_tokens=128,
                topk=2,
                ep_size=1,
                dtype=torch.bfloat16,
                tp_size=1,
                locality_cache_capacity=256,
            )
            return HeteroEPBufferAlloc(
                device_configs={0: cfg},
                dispatcher_type=DispatcherType.NCCL,
                locality_cache_capacity=256,
            )

        def test_locality_cache_allocated_for_hopper(self):
            alloc = self._make_hopper_alloc()
            alloc.allocate_all()
            cache = alloc.get_locality_cache(0)
            self.assertIsNotNone(cache)
            self.assertIsNotNone(cache.routing_map)
            self.assertEqual(cache.routing_map.shape, (256, 2))
            alloc.deallocate()

        def test_update_valid_tokens_propagates_to_cache(self):
            alloc = self._make_hopper_alloc()
            alloc.allocate_all()
            alloc.update_valid_tokens(0, 99)
            cache = alloc.get_locality_cache(0)
            self.assertEqual(cache.valid_tokens.item(), 99)
            alloc.deallocate()

    class TestCreateForCluster(unittest.TestCase):
        """Tests for the factory method (CPU-only, no CUDA allocation)."""

        def test_nvls_downgrade_warning(self):
            cfg = HeteroModelConfig(
                hidden_size=128,
                moe_latent_size=None,
                moe_router_topk=2,
                inference_moe_token_dispatcher_type="nvls",
            )
            with self.assertLogs("deepspeed.inference.hetero_ep_buffer_alloc", level="WARNING") as cm:
                alloc = HeteroEPBufferAlloc.create_for_cluster(
                    model_config=cfg,
                    device_assignments={0: DeviceClass.HOPPER, 1: DeviceClass.AMPERE},
                    max_tokens=512,
                    tp_size=1,
                )
            self.assertEqual(alloc._dispatcher_type, DispatcherType.DES_LOC_PCIe)
            self.assertTrue(any("NVLink" in msg for msg in cm.output))

        def test_nccl_dispatcher_selection(self):
            cfg = HeteroModelConfig(
                hidden_size=128,
                inference_moe_token_dispatcher_type="nccl",
            )
            alloc = HeteroEPBufferAlloc.create_for_cluster(
                model_config=cfg,
                device_assignments={0: DeviceClass.HOPPER},
                max_tokens=256,
            )
            self.assertEqual(alloc._dispatcher_type, DispatcherType.NCCL)

        def test_hopper_gets_bfloat16(self):
            cfg = HeteroModelConfig(hidden_size=128)
            alloc = HeteroEPBufferAlloc.create_for_cluster(
                model_config=cfg,
                device_assignments={0: DeviceClass.HOPPER},
                max_tokens=256,
            )
            self.assertEqual(alloc._device_configs[0].dtype, torch.bfloat16)

        def test_ampere_gets_float16(self):
            cfg = HeteroModelConfig(hidden_size=128)
            alloc = HeteroEPBufferAlloc.create_for_cluster(
                model_config=cfg,
                device_assignments={1: DeviceClass.AMPERE},
                max_tokens=256,
            )
            self.assertEqual(alloc._device_configs[1].dtype, torch.float16)

        def test_moe_latent_size_used_when_set(self):
            cfg = HeteroModelConfig(hidden_size=4096, moe_latent_size=512)
            alloc = HeteroEPBufferAlloc.create_for_cluster(
                model_config=cfg,
                device_assignments={0: DeviceClass.HOPPER},
                max_tokens=256,
            )
            self.assertEqual(alloc._device_configs[0].moe_hidden_size, 512)

        def test_moe_hidden_falls_back_to_hidden_size(self):
            cfg = HeteroModelConfig(hidden_size=4096, moe_latent_size=None)
            alloc = HeteroEPBufferAlloc.create_for_cluster(
                model_config=cfg,
                device_assignments={0: DeviceClass.HOPPER},
                max_tokens=256,
            )
            self.assertEqual(alloc._device_configs[0].moe_hidden_size, 4096)

    class TestDetectDeviceClass(unittest.TestCase):
        def test_no_cuda_returns_unknown(self):
            if torch.cuda.is_available():
                # Can't easily mock capability without CUDA, skip negative path
                self.skipTest("CUDA available, skipping no-CUDA path")
            result = detect_device_class(0)
            self.assertEqual(result, DeviceClass.UNKNOWN)

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        def test_out_of_range_device(self):
            result = detect_device_class(9999)
            self.assertEqual(result, DeviceClass.UNKNOWN)

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        def test_device_zero_is_valid_class(self):
            result = detect_device_class(0)
            self.assertIn(result, [DeviceClass.HOPPER, DeviceClass.AMPERE, DeviceClass.UNKNOWN])

    class TestDESLOCBufferHook(unittest.TestCase):
        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        def test_on_model_init_triggers_allocation(self):
            cfg = HeteroModelConfig(hidden_size=64, moe_router_topk=2)
            alloc = HeteroEPBufferAlloc.create_for_cluster(
                model_config=cfg,
                device_assignments={0: DeviceClass.AMPERE},
                max_tokens=128,
            )
            hook = DESLOCBufferHook(alloc=alloc, coordinator_dev=0)
            self.assertFalse(alloc.is_allocated())
            hook.on_model_init()
            self.assertTrue(alloc.is_allocated())
            alloc.deallocate()

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        def test_on_token_dispatch_updates_scalar(self):
            cfg = HeteroModelConfig(hidden_size=64, moe_router_topk=2)
            alloc = HeteroEPBufferAlloc.create_for_cluster(
                model_config=cfg,
                device_assignments={0: DeviceClass.AMPERE},
                max_tokens=128,
            )
            hook = DESLOCBufferHook(alloc=alloc, coordinator_dev=0)
            hook.on_model_init()
            hook.on_token_dispatch(dev_idx=0, n_tokens=55)
            vt = alloc.get_valid_tokens_tensor(0)
            self.assertEqual(vt.item(), 55)
            alloc.deallocate()

    print("Running HeteroEPBufferAlloc unit tests...", file=sys.stderr)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for tc in [
        TestDeviceBufferConfig,
        TestRoundUpTokens,
        TestLocalityCacheAllocator,
        TestDispatcherSharedState,
        TestHeteroModelConfig,
        TestCreateForCluster,
        TestDetectDeviceClass,
        TestDESLOCBufferHook,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    # CUDA-dependent tests only if available
    if torch.cuda.is_available():
        for tc in [TestHeteroEPBufferAllocCUDA, TestHeteroEPBufferAllocWithLocalityCache]:
            suite.addTests(loader.loadTestsFromTestCase(tc))
    else:
        print("NOTE: CUDA not available — skipping CUDA-dependent tests", file=sys.stderr)

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stderr)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
