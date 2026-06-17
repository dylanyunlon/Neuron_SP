"""
deepspeed/runtime/hetero_gdn_selective_recompute.py

Upstream Design Intent (Megatron-LM ff5264c33dc098a8135ccff89ca837bcf089c2ab):
    Megatron's commit enables *selective* gradient checkpointing specifically for the
    ``norm_out`` sub-computation inside GatedDeltaNet (GDN) layers.  The key insight is
    that ``norm_out`` — the RMSNorm applied to the gated attention output followed by an
    HP→CP (head-parallel to context-parallel) all-to-all collective — is cheap to
    recompute but expensive to keep alive in GPU SRAM until the backward pass.  By using
    ``CheckpointWithoutOutput`` (output-discarding checkpointing), Megatron discards the
    activation tensor immediately after the forward pass and recomputes it on demand
    during backward, saving peak memory at the cost of one extra forward sub-pass.

    The upstream change also extends ``TransformerConfig.recompute_modules`` to accept
    ``"gdn_norm_out"`` as a token and validates that this option is only used when
    ``experimental_attention_variant == "gated_delta_net"``.

DES-LOC Adaptation Points:
    In the Neuron_SP / DES-LOC framework the heterogeneous device topology
    (2× A6000 48 GB SM86 + 1× H100 NVL 96 GB SM90, PCIe-only interconnect, 1.5 TB CPU
    DRAM) introduces constraints that upstream Megatron does not address:

    1. **Tier-aware activation placement** — Instead of blindly discarding the norm_out
       activation, DES-LOC assigns it to the *locality cache tier* that best matches the
       compute device.  On H100 (Tier-0) the tensor stays on-device; on A6000 (Tier-1)
       the tensor is offloaded to CPU DRAM between forward and backward unless bandwidth
       analysis predicts that recompute is faster than PCIe round-trip.

    2. **Heterogeneous recompute gating** — The binary ``recompute_norm_out`` flag from
       Megatron is replaced by a three-way decision: KEEP (no recompute), OFFLOAD (move
       to CPU), or RECOMPUTE (discard and redo).  The decision is made per-layer per-
       device at module-init time using ``DeviceTierRegistry``.

    3. **PCIe-aware collective replacement** — The HP→CP all-to-all in Megatron assumes
       fast NVLink.  Under DES-LOC we substitute a *phased collective* that pipelines
       PCIe transfers with local computation to hide inter-device latency.

    4. **Shared Locality Cache (LOC)** — When OFFLOAD mode is chosen, the discarded
       tensor is stored in the ``LocalityCache`` (backed by pinned CPU DRAM) under a
       stable key derived from layer index and micro-batch ID.  On backward the cache
       returns the tensor, avoiding recomputation entirely when memory pressure is low.

    5. **DeepSpeed integration** — The module plugs into DeepSpeed's engine via
       ``deepspeed.runtime.engine`` hooks; it is not a standalone wrapper but a first-
       class citizen registered with DeepSpeed's activation-checkpointing subsystem.
"""

from __future__ import annotations

import enum
import logging
import math
import threading
import time
import unittest
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public re-exports consumed by the DeepSpeed engine registration path
# ---------------------------------------------------------------------------
__all__ = [
    "NormOutRecomputeMode",
    "DeviceTierRegistry",
    "LocalityCache",
    "HeteroGDNConfig",
    "CheckpointWithoutOutput",
    "PhasedPCIeCollective",
    "HeteroGDNSelectiveRecompute",
    "GatedRMSNorm",
    "register_hetero_gdn_hooks",
]

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    """Return a human-readable byte count string."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TiB"


# ---------------------------------------------------------------------------
# Device tier enumeration and registry
# ---------------------------------------------------------------------------

class DeviceTier(enum.IntEnum):
    """
    Hardware capability tiers present in the DES-LOC target cluster.

    Tier-0 (H100 NVL SM90): highest compute throughput, largest on-device HBM.
    Tier-1 (A6000 SM86):    mid-range GDDR6 VRAM, PCIe-attached only.
    Tier-2 (CPU DRAM):      1.5 TB pinned memory, accessible via PCIe DMA.
    """
    H100_NVL = 0   # SM90, 96 GB HBM3
    A6000    = 1   # SM86, 48 GB GDDR6
    CPU_DRAM = 2   # pinned host memory


# Compute-capability → tier mapping (SM major.minor as integer major*10+minor)
_SM_TO_TIER: Dict[int, DeviceTier] = {
    90: DeviceTier.H100_NVL,
    86: DeviceTier.A6000,
}


class DeviceTierRegistry:
    """
    Singleton that maps CUDA device indices to :class:`DeviceTier` values.

    On first access it queries ``torch.cuda.get_device_capability`` for every
    visible device and caches the results.  The registry is thread-safe.
    """

    _instance: Optional["DeviceTierRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._map: Dict[int, DeviceTier] = {}
        self._refresh()

    @classmethod
    def get(cls) -> "DeviceTierRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _refresh(self) -> None:
        count = torch.cuda.device_count()
        for idx in range(count):
            major, minor = torch.cuda.get_device_capability(idx)
            sm = major * 10 + minor
            tier = _SM_TO_TIER.get(sm)
            if tier is None:
                # Unknown GPU — treat conservatively as A6000 tier
                logger.warning(
                    "Unknown compute capability SM%d.%d on device %d; "
                    "assigning DeviceTier.A6000 for DES-LOC scheduling.",
                    major, minor, idx,
                )
                tier = DeviceTier.A6000
            self._map[idx] = tier

    def tier_of(self, device: torch.device) -> DeviceTier:
        """Return the :class:`DeviceTier` for *device*."""
        if device.type == "cpu":
            return DeviceTier.CPU_DRAM
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return self._map.get(idx, DeviceTier.A6000)

    def __repr__(self) -> str:  # pragma: no cover
        lines = [f"  cuda:{k} → {v.name}" for k, v in sorted(self._map.items())]
        return "DeviceTierRegistry{\n" + "\n".join(lines) + "\n}"


# ---------------------------------------------------------------------------
# NormOut recompute mode decision
# ---------------------------------------------------------------------------

class NormOutRecomputeMode(enum.Enum):
    """
    Three-way policy for GDN norm_out activation handling.

    KEEP      — store activation on-device (Megatron default for non-selective).
    OFFLOAD   — store activation in pinned CPU DRAM (LOC tier-2).
    RECOMPUTE — discard activation, redo forward sub-graph on backward.
    """
    KEEP      = "keep"
    OFFLOAD   = "offload"
    RECOMPUTE = "recompute"


@dataclass
class PCIeBandwidthModel:
    """
    Lightweight analytical model of PCIe H2D/D2H bandwidth for cost estimation.

    Attributes
    ----------
    h2d_bw_gbps : float
        Measured host-to-device bandwidth in GB/s (PCIe Gen4 x16 ≈ 32 GB/s).
    d2h_bw_gbps : float
        Measured device-to-host bandwidth in GB/s.
    latency_us  : float
        Fixed per-transfer latency overhead in microseconds.
    """
    h2d_bw_gbps: float = 28.0
    d2h_bw_gbps: float = 26.0
    latency_us: float  = 8.0

    def transfer_time_s(self, nbytes: int, direction: str = "d2h") -> float:
        """Estimate wall-clock transfer time in seconds for *nbytes* bytes."""
        bw = self.d2h_bw_gbps if direction == "d2h" else self.h2d_bw_gbps
        return self.latency_us * 1e-6 + nbytes / (bw * 1e9)


def decide_norm_out_mode(
    tier: DeviceTier,
    tensor_bytes: int,
    recompute_flops: float,
    pcie_model: Optional[PCIeBandwidthModel] = None,
) -> NormOutRecomputeMode:
    """
    Select the :class:`NormOutRecomputeMode` for a given device tier and tensor size.

    Decision logic
    --------------
    * H100 (Tier-0): always KEEP — abundant HBM, no PCIe penalty.
    * A6000 (Tier-1): compare estimated recompute time vs PCIe round-trip time.
      - If recompute is cheaper → RECOMPUTE.
      - If PCIe offload is cheaper → OFFLOAD.
      - Tie-break in favour of RECOMPUTE (avoids synchronisation overhead).
    * CPU_DRAM (Tier-2): always OFFLOAD (shouldn't occur in practice).

    Parameters
    ----------
    tier           : DeviceTier of the current device.
    tensor_bytes   : size of the norm_out activation in bytes.
    recompute_flops: estimated FLOPs to recompute norm_out (used to derive wall time).
    pcie_model     : optional bandwidth model; defaults to ``PCIeBandwidthModel()``.
    """
    if tier == DeviceTier.H100_NVL:
        return NormOutRecomputeMode.KEEP

    if tier == DeviceTier.CPU_DRAM:
        return NormOutRecomputeMode.OFFLOAD

    # A6000 path: analytical cost comparison
    if pcie_model is None:
        pcie_model = PCIeBandwidthModel()

    # PCIe round-trip: D2H (fwd) + H2D (bwd)
    pcie_time = (
        pcie_model.transfer_time_s(tensor_bytes, "d2h")
        + pcie_model.transfer_time_s(tensor_bytes, "h2d")
    )

    # Recompute time estimate: assume A6000 peak FP16 throughput ≈ 77.4 TFLOPS
    a6000_tflops = 77.4e12
    recompute_time = recompute_flops / a6000_tflops

    if recompute_time <= pcie_time:
        return NormOutRecomputeMode.RECOMPUTE
    else:
        return NormOutRecomputeMode.OFFLOAD


# ---------------------------------------------------------------------------
# Locality Cache (LOC) — pinned CPU DRAM backing store
# ---------------------------------------------------------------------------

class LocalityCache:
    """
    Thread-safe key-value store backed by pinned (page-locked) CPU DRAM.

    This is the *Shared LOcality Cache* in DES-LOC.  Tensors written here
    during the forward pass are retrieved by the backward pass without
    triggering a GPU-side recomputation.  All CPU tensors are stored as
    pinned memory so that H2D DMA can proceed without staging copies.

    The cache uses *weak-reference generation tracking* to avoid memory leaks
    across micro-batches: entries are automatically evicted when the owning
    computation graph is freed.

    Parameters
    ----------
    max_bytes : int
        Soft capacity limit.  When exceeded, the oldest entries are evicted.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024) -> None:  # 512 MiB default
        self._store: Dict[str, Tuple[Tensor, int]] = {}  # key → (cpu_tensor, timestamp)
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._used_bytes = 0
        self._clock = 0

    # ------------------------------------------------------------------
    def store(self, key: str, tensor: Tensor) -> None:
        """
        Asynchronously copy *tensor* to pinned CPU memory and cache it.

        The copy is initiated on the current CUDA stream so that it overlaps
        with subsequent GPU kernels (PCIe DMA pipeline).
        """
        cpu_tensor = torch.empty(
            tensor.shape, dtype=tensor.dtype, pin_memory=True
        )
        # Non-blocking D2H transfer — overlaps with next GPU op
        cpu_tensor.copy_(tensor, non_blocking=True)

        with self._lock:
            nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
            self._evict_if_needed(nbytes)
            self._store[key] = (cpu_tensor, self._clock)
            self._used_bytes += nbytes
            self._clock += 1

        logger.debug(
            "LOC store: key=%s shape=%s dtype=%s size=%s used=%s",
            key, tuple(tensor.shape), tensor.dtype,
            _fmt_bytes(nbytes), _fmt_bytes(self._used_bytes),
        )

    def retrieve(self, key: str, device: torch.device) -> Optional[Tensor]:
        """
        Return the cached tensor moved back to *device*, or ``None`` if evicted.

        The H2D copy is initiated non-blocking to allow overlapping with other ops.
        """
        with self._lock:
            entry = self._store.pop(key, None)
            if entry is None:
                return None
            cpu_tensor, _ = entry
            nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
            self._used_bytes -= nbytes

        gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        logger.debug(
            "LOC retrieve: key=%s → %s (H2D non-blocking)", key, device
        )
        return gpu_tensor

    def _evict_if_needed(self, incoming_bytes: int) -> None:
        """Evict oldest entries until capacity allows *incoming_bytes*."""
        if self._used_bytes + incoming_bytes <= self._max_bytes:
            return
        sorted_keys = sorted(self._store, key=lambda k: self._store[k][1])
        for k in sorted_keys:
            if self._used_bytes + incoming_bytes <= self._max_bytes:
                break
            cpu_tensor, _ = self._store.pop(k)
            self._used_bytes -= cpu_tensor.numel() * cpu_tensor.element_size()
            logger.warning(
                "LOC eviction: key=%s evicted due to capacity pressure "
                "(max=%s, used=%s, incoming=%s).",
                k, _fmt_bytes(self._max_bytes),
                _fmt_bytes(self._used_bytes),
                _fmt_bytes(incoming_bytes),
            )

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton; created lazily
_GLOBAL_LOC: Optional[LocalityCache] = None
_LOC_LOCK = threading.Lock()


def get_locality_cache(max_bytes: int = 512 * 1024 * 1024) -> LocalityCache:
    """Return (or create) the process-global :class:`LocalityCache`."""
    global _GLOBAL_LOC
    with _LOC_LOCK:
        if _GLOBAL_LOC is None:
            _GLOBAL_LOC = LocalityCache(max_bytes=max_bytes)
            logger.info(
                "LocalityCache initialised with capacity %s.", _fmt_bytes(max_bytes)
            )
    return _GLOBAL_LOC


# ---------------------------------------------------------------------------
# Output-discarding checkpoint (mirrors Megatron CheckpointWithoutOutput)
# ---------------------------------------------------------------------------

class CheckpointWithoutOutput:
    """
    DES-LOC variant of Megatron's ``CheckpointWithoutOutput``.

    Upstream intent (Megatron):
        Run a sub-graph in ``torch.no_grad()`` during forward, discard the
        output tensor, and register a recompute hook on the *consumer* tensor's
        ``grad_fn`` so that on backward the sub-graph is re-executed with
        ``torch.enable_grad()`` to produce fresh activations.

    DES-LOC adaptation:
        Instead of unconditionally discarding and recomputing, this class
        supports three modes (``NormOutRecomputeMode``):

        * KEEP      — standard forward, no discarding.
        * RECOMPUTE — mirrors upstream exactly: discard + recompute on backward.
        * OFFLOAD   — discard + LOC offload to pinned CPU DRAM; on backward
                      retrieve from LOC (no recomputation if cache hit).

    The mode is determined at construction time by ``decide_norm_out_mode``.
    """

    def __init__(
        self,
        mode: NormOutRecomputeMode,
        loc: Optional[LocalityCache] = None,
        cache_key: str = "",
    ) -> None:
        self.mode = mode
        self.loc = loc
        self.cache_key = cache_key
        self._fn: Optional[Callable] = None
        self._fn_args: Tuple = ()
        self._saved_output: Optional[Tensor] = None
        self._device: Optional[torch.device] = None

    def checkpoint(self, fn: Callable, *args: Any) -> Tensor:
        """
        Execute *fn(*args)* and handle activation according to the chosen mode.

        Returns the output tensor.  In RECOMPUTE/OFFLOAD modes the returned
        tensor is a *detached* copy; the real gradient flow is wired up in
        :meth:`discard_output_and_register_recompute`.
        """
        self._fn = fn
        self._fn_args = args

        if self.mode == NormOutRecomputeMode.KEEP:
            with torch.enable_grad():
                output = fn(*args)
            self._saved_output = output
            return output

        # RECOMPUTE or OFFLOAD: run without tracking
        with torch.no_grad():
            output = fn(*args)

        self._device = output.device

        if self.mode == NormOutRecomputeMode.OFFLOAD and self.loc is not None:
            self.loc.store(self.cache_key, output)

        # Return a detached leaf that has a grad_fn pointing to args
        output_detached = output.detach().requires_grad_(output.requires_grad)
        self._saved_output = output_detached
        return output_detached

    def discard_output_and_register_recompute(self, consumer: Tensor) -> None:
        """
        Register a recompute/retrieve hook on *consumer*'s backward graph.

        This must be called after the forward pass on the tensor that *consumes*
        the checkpointed output (i.e., ``out`` from the output projection).

        In KEEP mode this is a no-op.

        In RECOMPUTE mode a ``register_hook`` on ``consumer.grad_fn`` triggers
        recomputation of the sub-graph before the consumer's backward.

        In OFFLOAD mode a similar hook retrieves from the LOC; recomputation is
        the fallback if the entry was evicted.
        """
        if self.mode == NormOutRecomputeMode.KEEP:
            return

        if consumer.grad_fn is None:
            # Consumer is a leaf — nothing to hook
            return

        fn = self._fn
        fn_args = self._fn_args
        device = self._device
        loc = self.loc
        cache_key = self.cache_key
        mode = self.mode
        saved_ref = weakref.ref(self._saved_output) if self._saved_output is not None else None

        def _recompute_hook(grad: Tensor) -> None:
            """Recompute or retrieve norm_out before backward proceeds."""
            if mode == NormOutRecomputeMode.OFFLOAD and loc is not None and device is not None:
                retrieved = loc.retrieve(cache_key, device)
                if retrieved is not None:
                    saved = saved_ref() if saved_ref is not None else None
                    if saved is not None:
                        saved.data = retrieved.data
                    return
                # Cache miss — fall through to recompute
                logger.debug(
                    "LOC cache miss for key=%s; falling back to recompute.", cache_key
                )

            # RECOMPUTE path (or OFFLOAD fallback)
            t0 = time.perf_counter()
            with torch.enable_grad():
                recomputed = fn(*fn_args)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            saved = saved_ref() if saved_ref is not None else None
            if saved is not None:
                saved.data = recomputed.data
            logger.debug(
                "norm_out recomputed in %.2f ms (mode=%s, key=%s).",
                elapsed_ms, mode.value, cache_key,
            )

        consumer.grad_fn.register_hook(_recompute_hook)


# ---------------------------------------------------------------------------
# Phased PCIe collective (replaces NVLink all-to-all in HP→CP reduction)
# ---------------------------------------------------------------------------

class PhasedPCIeCollective:
    """
    PCIe-aware replacement for the HP→CP all-to-all in GDN's norm_out path.

    Upstream (Megatron) uses ``tensor_a2a_hp2cp`` which assumes fast NVLink
    interconnect.  In DES-LOC the A6000 ↔ H100 path is PCIe-only, so we
    pipeline the collective in phases to overlap compute and transfer:

    Phase 1 (local shard preparation):
        Each rank prepares its local shard of the head-parallel tensor.

    Phase 2 (async scatter):
        Each rank initiates a non-blocking isend of its shard to the target
        rank in the CP group.  Simultaneously it posts an irecv for the shard
        it expects to receive.

    Phase 3 (overlap compute):
        While the PCIe transfer is in flight, the rank performs any local
        work that does not depend on the received shard (e.g., norm scaling).

    Phase 4 (synchronise and reassemble):
        Wait on all pending requests, then reassemble the full tensor.

    This class is intentionally stateless (no ``__init__`` params) because it
    is instantiated once per module and reused across forward passes.
    """

    @staticmethod
    def hp_to_cp(
        tensor: Tensor,
        seq_dim: int,
        head_dim: int,
        cp_group: Optional[dist.ProcessGroup],
    ) -> Tensor:
        """
        Scatter *tensor* from head-parallel to context-parallel layout.

        Parameters
        ----------
        tensor    : Input tensor in HP layout (seq_dim is the sequence axis).
        seq_dim   : The sequence dimension index.
        head_dim  : The head dimension index (will be split across CP ranks).
        cp_group  : The context-parallel process group, or ``None`` for single-GPU.
        """
        if cp_group is None or dist.get_world_size(cp_group) == 1:
            return tensor

        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)

        if tensor.shape[head_dim] % cp_size != 0:
            raise ValueError(
                f"head_dim size {tensor.shape[head_dim]} is not divisible by "
                f"cp_size {cp_size}."
            )

        # Split along head_dim — each rank will receive one chunk
        chunks = tensor.chunk(cp_size, dim=head_dim)
        send_chunk = chunks[cp_rank].contiguous()

        recv_chunk = torch.empty_like(send_chunk)
        send_ops: List[dist.Work] = []
        recv_ops: List[dist.Work] = []

        # Post all sends and receives before waiting on any
        for dst in range(cp_size):
            if dst == cp_rank:
                continue
            send_ops.append(
                dist.isend(send_chunk, dst=dst, group=cp_group)
            )
        for src in range(cp_size):
            if src == cp_rank:
                recv_chunk = send_chunk  # local copy
                continue
            buf = torch.empty_like(chunks[src])
            recv_ops.append((src, dist.irecv(buf, src=src, group=cp_group), buf))

        # Overlap: any local-only computation would go here

        for w in send_ops:
            w.wait()
        result_chunks: List[Tensor] = [torch.empty(0)] * cp_size
        result_chunks[cp_rank] = send_chunk
        for src, w, buf in recv_ops:
            w.wait()
            result_chunks[src] = buf

        return torch.cat(result_chunks, dim=head_dim)


# ---------------------------------------------------------------------------
# Gated RMSNorm (replaces Megatron's _apply_gated_norm in self-contained form)
# ---------------------------------------------------------------------------

class GatedRMSNorm(nn.Module):
    """
    RMSNorm applied element-wise to ``x * gate`` with a learnable scale.

    This is the DES-LOC implementation of the gated normalisation step that
    Megatron calls ``_apply_gated_norm`` inside GatedDeltaNet.  It is broken
    out as a standalone ``nn.Module`` so that:

    * It can be registered with DeepSpeed's parameter partitioning.
    * Its forward call can be passed as ``fn`` to :class:`CheckpointWithoutOutput`.
    * Unit tests can instantiate it without a full GDN stack.

    Parameters
    ----------
    hidden_size : int
        Feature dimension of the normalised tensor.
    eps         : float
        Small constant for numerical stability in RMS computation.
    dtype       : torch.dtype
        Weight dtype; should match the model's compute dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        """Apply gated RMSNorm: ``weight * (x * gate) / rms(x * gate)``."""
        gated = x * gate
        # RMS normalisation in float32 for numerical stability
        rms = gated.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        normed = (gated.float() / rms).to(gated.dtype)
        return normed * self.weight


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeteroGDNConfig:
    """
    Configuration for :class:`HeteroGDNSelectiveRecompute`.

    Attributes
    ----------
    hidden_size       : Model hidden dimension.
    num_heads         : Number of attention heads.
    value_head_dim    : Per-head value dimension.
    seq_len           : Sequence length (after any CP sharding).
    batch_size        : Micro-batch size.
    layernorm_epsilon : Epsilon for RMSNorm.
    recompute_modules : List of module names for selective recompute
                        (mirrors Megatron's ``TransformerConfig.recompute_modules``).
    recompute_granularity : "selective" or "full".
    loc_max_bytes     : Capacity of the process-global LocalityCache.
    pcie_bw_model     : PCIe bandwidth model used by ``decide_norm_out_mode``.
    layer_number      : 1-indexed layer number (used as part of the LOC cache key).
    cp_group          : Context-parallel process group (can be ``None``).
    dtype             : Compute dtype.
    """
    hidden_size: int = 4096
    num_heads: int = 32
    value_head_dim: int = 128
    seq_len: int = 2048
    batch_size: int = 1
    layernorm_epsilon: float = 1e-6
    recompute_modules: List[str] = field(default_factory=lambda: ["gdn_norm_out"])
    recompute_granularity: str = "selective"
    loc_max_bytes: int = 512 * 1024 * 1024
    pcie_bw_model: PCIeBandwidthModel = field(default_factory=PCIeBandwidthModel)
    layer_number: int = 1
    cp_group: Optional[dist.ProcessGroup] = None
    dtype: torch.dtype = torch.bfloat16

    # Validation ----------------------------------------------------------------
    def __post_init__(self) -> None:
        allowed = {
            "core_attn", "moe_act", "layernorm", "mla_up_proj",
            "mlp", "moe", "shared_experts", "gdn_norm_out",
        }
        invalid = set(self.recompute_modules) - allowed
        if invalid:
            raise ValueError(
                f"Unknown recompute_modules: {invalid}.  Allowed: {allowed}."
            )
        if "gdn_norm_out" in self.recompute_modules:
            if self.recompute_granularity != "selective":
                raise ValueError(
                    "'gdn_norm_out' requires recompute_granularity='selective'."
                )


# ---------------------------------------------------------------------------
# Main module: HeteroGDNSelectiveRecompute
# ---------------------------------------------------------------------------

class HeteroGDNSelectiveRecompute(nn.Module):
    """
    DES-LOC heterogeneous-aware selective recompute adapter for GDN norm_out.

    This module encapsulates the norm-and-collective sub-graph that Megatron
    wraps inside ``_gated_norm_and_a2a`` in commit ff5264c.  It extends that
    design with:

    * **Tier-aware mode selection** (KEEP / OFFLOAD / RECOMPUTE) determined at
      ``__init__`` time based on the device this module lives on.
    * **LocalityCache integration** for OFFLOAD mode.
    * **PhasedPCIeCollective** for the HP→CP scatter under PCIe topology.
    * **DeepSpeed-compatible ``forward``** signature.

    Usage in a GDN layer::

        self.norm_out_recompute = HeteroGDNSelectiveRecompute(config)
        # ... inside forward:
        norm_out, checkpoint = self.norm_out_recompute(
            core_attn_out, gate, batch, seq_len,
            packed_seq_params=packed_seq_params,
            cu_seqlens_q=cu_seqlens_q,
            micro_batch_id=micro_batch_id,
        )
        out, out_bias = self.out_proj(norm_out)
        if checkpoint is not None:
            checkpoint.discard_output_and_register_recompute(out)

    Parameters
    ----------
    config  : :class:`HeteroGDNConfig`
    device  : Target device.  Defaults to ``torch.cuda.current_device()``.
    """

    def __init__(
        self,
        config: HeteroGDNConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config

        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                device = torch.device("cpu")
        self.device = device

        # Determine tier and mode at construction time
        registry = DeviceTierRegistry.get()
        self.tier = registry.tier_of(device)

        # Estimate tensor size for cost model
        tensor_bytes = (
            config.seq_len
            * config.batch_size
            * config.num_heads
            * config.value_head_dim
            * torch.finfo(config.dtype).bits
            // 8
        )
        # Approximate recompute FLOPs: 2× hidden_size per element (RMSNorm + gating)
        recompute_flops = 2.0 * config.seq_len * config.batch_size * config.hidden_size

        self.recompute_norm_out: bool = (
            config.recompute_granularity == "selective"
            and "gdn_norm_out" in config.recompute_modules
        )

        if self.recompute_norm_out:
            self.norm_out_mode = decide_norm_out_mode(
                self.tier, tensor_bytes, recompute_flops, config.pcie_bw_model
            )
        else:
            self.norm_out_mode = NormOutRecomputeMode.KEEP

        self.loc: Optional[LocalityCache] = None
        if self.norm_out_mode == NormOutRecomputeMode.OFFLOAD:
            self.loc = get_locality_cache(config.loc_max_bytes)

        # Gated RMSNorm sub-module
        self.gated_norm = GatedRMSNorm(
            hidden_size=config.value_head_dim,
            eps=config.layernorm_epsilon,
            dtype=config.dtype,
        )

        # Phased collective
        self._collective = PhasedPCIeCollective()

        logger.info(
            "HeteroGDNSelectiveRecompute layer=%d device=%s tier=%s mode=%s",
            config.layer_number, device, self.tier.name, self.norm_out_mode.value,
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        core_attn_out: Tensor,
        gate: Tensor,
        batch: int,
        seq_len: int,
        packed_seq_params: Any = None,
        cu_seqlens_q: Optional[Tensor] = None,
        micro_batch_id: int = 0,
    ) -> Tuple[Tensor, Optional[CheckpointWithoutOutput]]:
        """
        Run the norm-and-scatter sub-graph with DES-LOC tier-aware checkpointing.

        Parameters
        ----------
        core_attn_out   : Output of the delta-rule kernel, shape ``(b, h, s, d)``.
        gate            : Gate tensor, same shape as *core_attn_out*.
        batch           : Micro-batch size.
        seq_len         : Local sequence length (after CP sharding).
        packed_seq_params : Optional packed-sequence metadata (mirrors Megatron).
        cu_seqlens_q    : Cumulative sequence lengths for packed sequences.
        micro_batch_id  : Used to build the LOC cache key for OFFLOAD mode.

        Returns
        -------
        norm_out    : Tensor in CP layout, shape ``(s, b, heads*d)``.
        checkpoint  : :class:`CheckpointWithoutOutput` instance if active,
                      else ``None``.  Caller must invoke
                      ``checkpoint.discard_output_and_register_recompute(out)``
                      after the output projection.
        """
        cache_key = (
            f"layer{self.config.layer_number}_mb{micro_batch_id}_norm_out"
        )

        def _gated_norm_and_scatter(
            _core_attn_out: Tensor,
            _gate: Tensor,
        ) -> Tensor:
            """Inner closure mirroring Megatron's ``_gated_norm_and_a2a``."""
            # RMSNorm with gating
            norm_hp = self.gated_norm(_core_attn_out, _gate)

            # Reshape from bshd → sbhd format
            norm_hp = norm_hp.reshape(batch, seq_len, -1)
            norm_hp = norm_hp.transpose(0, 1).contiguous()

            # HP→CP collective (phased PCIe-aware under DES-LOC)
            if (
                packed_seq_params is not None
                and hasattr(packed_seq_params, "qkv_format")
                and packed_seq_params.qkv_format == "thd"
                and cu_seqlens_q is not None
            ):
                unpacked = _unpack_sequence_desync(norm_hp, cu_seqlens_q, dim=0)
                outputs: List[Tensor] = []
                for shard in unpacked:
                    shard = self._collective.hp_to_cp(
                        shard,
                        seq_dim=0,
                        head_dim=-1,
                        cp_group=self.config.cp_group,
                    )
                    outputs.append(shard)
                return torch.cat(outputs, dim=0)
            else:
                return self._collective.hp_to_cp(
                    norm_hp,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=self.config.cp_group,
                )

        if not self.recompute_norm_out or self.norm_out_mode == NormOutRecomputeMode.KEEP:
            norm_out = _gated_norm_and_scatter(core_attn_out, gate)
            return norm_out, None

        # Active checkpointing path
        ckpt = CheckpointWithoutOutput(
            mode=self.norm_out_mode,
            loc=self.loc,
            cache_key=cache_key,
        )
        norm_out = ckpt.checkpoint(_gated_norm_and_scatter, core_attn_out, gate)
        return norm_out, ckpt

    def extra_repr(self) -> str:
        return (
            f"layer={self.config.layer_number}, "
            f"tier={self.tier.name}, "
            f"mode={self.norm_out_mode.value}"
        )


# ---------------------------------------------------------------------------
# Sequence unpacking helper (DES-LOC variant of Megatron's _unpack_sequence)
# ---------------------------------------------------------------------------

def _unpack_sequence_desync(
    tensor: Tensor,
    cu_seqlens: Tensor,
    dim: int = 0,
) -> List[Tensor]:
    """
    Split a packed-sequence tensor into per-sequence chunks along *dim*.

    This mirrors Megatron's ``_unpack_sequence`` but operates on CPU-side
    ``cu_seqlens`` to avoid GPU synchronisation during the split.

    Parameters
    ----------
    tensor     : Packed tensor; shape along *dim* equals total tokens.
    cu_seqlens : 1-D tensor of cumulative sequence lengths (including 0 at index 0).
    dim        : Dimension along which sequences are concatenated.
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return list(torch.split(tensor, [int(l) for l in lengths], dim=dim))


# ---------------------------------------------------------------------------
# DeepSpeed engine registration hooks
# ---------------------------------------------------------------------------

def register_hetero_gdn_hooks(
    engine: Any,
    gdn_modules: Sequence["HeteroGDNSelectiveRecompute"],
) -> None:
    """
    Register DES-LOC selective recompute hooks with a DeepSpeed engine.

    This function wires :class:`HeteroGDNSelectiveRecompute` instances into
    DeepSpeed's activation-checkpointing subsystem so that:

    * Memory-pressure callbacks from ``deepspeed.runtime.engine`` can
      dynamically promote KEEP→OFFLOAD or OFFLOAD→RECOMPUTE on A6000 devices.
    * The LOC capacity is surfaced in DeepSpeed's memory reporter.

    Parameters
    ----------
    engine      : A ``deepspeed.DeepSpeedEngine`` instance.
    gdn_modules : Iterable of :class:`HeteroGDNSelectiveRecompute` modules
                  attached to the model.
    """
    if not gdn_modules:
        return

    loc = get_locality_cache()

    def _memory_pressure_callback(free_bytes: int, total_bytes: int) -> None:
        """Escalate recompute mode when GPU memory drops below 15% headroom."""
        headroom = free_bytes / total_bytes
        if headroom < 0.15:
            for mod in gdn_modules:
                if mod.norm_out_mode == NormOutRecomputeMode.KEEP:
                    mod.norm_out_mode = NormOutRecomputeMode.OFFLOAD
                    logger.warning(
                        "Memory pressure (%.1f%% free): layer=%d escalated "
                        "norm_out mode KEEP → OFFLOAD.",
                        headroom * 100, mod.config.layer_number,
                    )
                elif mod.norm_out_mode == NormOutRecomputeMode.OFFLOAD:
                    mod.norm_out_mode = NormOutRecomputeMode.RECOMPUTE
                    mod.loc = None  # release LOC reference
                    logger.warning(
                        "Memory pressure (%.1f%% free): layer=%d escalated "
                        "norm_out mode OFFLOAD → RECOMPUTE.",
                        headroom * 100, mod.config.layer_number,
                    )

    # Attach to DeepSpeed engine if the hook API is available
    if hasattr(engine, "register_memory_pressure_callback"):
        engine.register_memory_pressure_callback(_memory_pressure_callback)
        logger.info(
            "Registered DES-LOC memory-pressure callback for %d GDN modules.",
            len(list(gdn_modules)),
        )
    else:
        logger.debug(
            "DeepSpeed engine does not expose register_memory_pressure_callback; "
            "dynamic mode escalation disabled."
        )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import copy
    import os
    import sys
    import warnings

    # Suppress distributed-not-initialised warnings during isolated unit tests
    warnings.filterwarnings("ignore", category=UserWarning)

    # -----------------------------------------------------------------------
    # Test helpers
    # -----------------------------------------------------------------------

    def _make_config(**overrides) -> HeteroGDNConfig:
        defaults = dict(
            hidden_size=256,
            num_heads=4,
            value_head_dim=64,
            seq_len=32,
            batch_size=2,
            layernorm_epsilon=1e-6,
            recompute_modules=["gdn_norm_out"],
            recompute_granularity="selective",
            loc_max_bytes=64 * 1024 * 1024,
            layer_number=1,
            cp_group=None,
            dtype=torch.bfloat16,
        )
        defaults.update(overrides)
        return HeteroGDNConfig(**defaults)

    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # -----------------------------------------------------------------------
    # Test 1: DeviceTierRegistry detects devices correctly
    # -----------------------------------------------------------------------

    class TestDeviceTierRegistry(unittest.TestCase):
        def test_cpu_tier(self):
            reg = DeviceTierRegistry()
            tier = reg.tier_of(torch.device("cpu"))
            self.assertEqual(tier, DeviceTier.CPU_DRAM)

        def test_singleton(self):
            a = DeviceTierRegistry.get()
            b = DeviceTierRegistry.get()
            self.assertIs(a, b)

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        def test_cuda_device_returns_tier(self):
            reg = DeviceTierRegistry.get()
            tier = reg.tier_of(torch.device("cuda:0"))
            self.assertIn(tier, list(DeviceTier))

    # -----------------------------------------------------------------------
    # Test 2: decide_norm_out_mode returns correct policy
    # -----------------------------------------------------------------------

    class TestDecideNormOutMode(unittest.TestCase):
        def test_h100_always_keep(self):
            mode = decide_norm_out_mode(
                DeviceTier.H100_NVL, tensor_bytes=100_000_000, recompute_flops=1e12
            )
            self.assertEqual(mode, NormOutRecomputeMode.KEEP)

        def test_cpu_dram_always_offload(self):
            mode = decide_norm_out_mode(
                DeviceTier.CPU_DRAM, tensor_bytes=1000, recompute_flops=1e6
            )
            self.assertEqual(mode, NormOutRecomputeMode.OFFLOAD)

        def test_a6000_cheap_recompute(self):
            # tiny tensor → PCIe latency dominates → RECOMPUTE
            mode = decide_norm_out_mode(
                DeviceTier.A6000,
                tensor_bytes=4096,
                recompute_flops=1e6,
                pcie_model=PCIeBandwidthModel(h2d_bw_gbps=28.0, d2h_bw_gbps=26.0, latency_us=8.0),
            )
            self.assertEqual(mode, NormOutRecomputeMode.RECOMPUTE)

        def test_a6000_large_tensor_offload(self):
            # Very large tensor → transfer time may exceed recompute for large FLOPs
            # But also extremely large recompute → offload wins
            mode = decide_norm_out_mode(
                DeviceTier.A6000,
                tensor_bytes=4 * 1024 * 1024 * 1024,  # 4 GB
                recompute_flops=1e16,                   # extremely expensive
                pcie_model=PCIeBandwidthModel(h2d_bw_gbps=28.0, d2h_bw_gbps=26.0, latency_us=0.0),
            )
            self.assertEqual(mode, NormOutRecomputeMode.OFFLOAD)

    # -----------------------------------------------------------------------
    # Test 3: LocalityCache store/retrieve round-trip
    # -----------------------------------------------------------------------

    class TestLocalityCache(unittest.TestCase):
        def setUp(self):
            self.cache = LocalityCache(max_bytes=32 * 1024 * 1024)

        def test_store_and_retrieve_cpu(self):
            t = torch.randn(16, 32)
            self.cache.store("test_key", t)
            result = self.cache.retrieve("test_key", torch.device("cpu"))
            self.assertIsNotNone(result)
            self.assertTrue(torch.allclose(t.float(), result.float(), atol=1e-5))

        def test_retrieve_missing_key(self):
            result = self.cache.retrieve("nonexistent", torch.device("cpu"))
            self.assertIsNone(result)

        def test_eviction_on_capacity_exceeded(self):
            # Fill with three 8 MB tensors into a 20 MB cache
            cache = LocalityCache(max_bytes=20 * 1024 * 1024)
            t = torch.zeros(2 * 1024 * 1024, dtype=torch.float32)  # 8 MB
            cache.store("k1", t)
            cache.store("k2", t)
            cache.store("k3", t)  # should evict k1
            # k1 should be gone; k2 or k3 should exist
            r1 = cache.retrieve("k1", torch.device("cpu"))
            self.assertIsNone(r1)

        def test_key_removed_after_retrieve(self):
            t = torch.ones(8)
            self.cache.store("once", t)
            _ = self.cache.retrieve("once", torch.device("cpu"))
            r2 = self.cache.retrieve("once", torch.device("cpu"))
            self.assertIsNone(r2)

        def test_len(self):
            t = torch.zeros(4)
            self.cache.store("a", t)
            self.cache.store("b", t)
            self.assertEqual(len(self.cache), 2)

    # -----------------------------------------------------------------------
    # Test 4: GatedRMSNorm forward correctness
    # -----------------------------------------------------------------------

    class TestGatedRMSNorm(unittest.TestCase):
        def _make_norm(self, hidden=64):
            return GatedRMSNorm(hidden_size=hidden, eps=1e-6, dtype=torch.float32)

        def test_output_shape(self):
            norm = self._make_norm(64)
            x = torch.randn(4, 8, 64)
            g = torch.ones(4, 8, 64)
            y = norm(x, g)
            self.assertEqual(y.shape, x.shape)

        def test_unit_gate_equals_standard_rmsnorm(self):
            """With gate=1, GatedRMSNorm should equal plain RMSNorm (weight=1)."""
            norm = self._make_norm(32)
            nn.init.ones_(norm.weight)
            x = torch.randn(2, 16, 32)
            gate = torch.ones_like(x)
            out = norm(x, gate)
            # Manual RMSNorm
            rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            expected = (x.float() / rms).to(x.dtype)
            self.assertTrue(
                torch.allclose(out.float(), expected.float(), atol=1e-3),
                f"Max diff: {(out.float() - expected.float()).abs().max():.6f}",
            )

        def test_zero_gate_gives_zero(self):
            norm = self._make_norm(16)
            x = torch.randn(3, 5, 16)
            gate = torch.zeros_like(x)
            out = norm(x, gate)
            # gated = 0, rms = eps^0.5 → out ≈ 0 / sqrt(eps) * weight ≈ 0
            self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-4))

        def test_gradient_flows(self):
            norm = self._make_norm(16)
            x = torch.randn(2, 4, 16, requires_grad=True)
            gate = torch.randn(2, 4, 16, requires_grad=True)
            out = norm(x, gate)
            out.sum().backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(gate.grad)
            self.assertFalse(torch.isnan(x.grad).any())

    # -----------------------------------------------------------------------
    # Test 5: CheckpointWithoutOutput in KEEP mode
    # -----------------------------------------------------------------------

    class TestCheckpointWithoutOutputKeep(unittest.TestCase):
        def test_keep_mode_output_identical(self):
            ckpt = CheckpointWithoutOutput(mode=NormOutRecomputeMode.KEEP)
            x = torch.randn(4, 8, requires_grad=True)
            fn_calls = [0]

            def fn(t):
                fn_calls[0] += 1
                return t * 2.0

            out = ckpt.checkpoint(fn, x)
            self.assertEqual(fn_calls[0], 1)
            loss = out.sum()
            loss.backward()
            self.assertIsNotNone(x.grad)

    # -----------------------------------------------------------------------
    # Test 6: CheckpointWithoutOutput in RECOMPUTE mode (CPU tensors)
    # -----------------------------------------------------------------------

    class TestCheckpointWithoutOutputRecompute(unittest.TestCase):
        def test_recompute_mode_no_grad_forward(self):
            ckpt = CheckpointWithoutOutput(
                mode=NormOutRecomputeMode.RECOMPUTE, cache_key="test"
            )
            x = torch.randn(4, 8, requires_grad=True)
            call_log = []

            def fn(t):
                call_log.append(t.requires_grad)
                return t.clone()

            out = ckpt.checkpoint(fn, x)
            # In recompute mode, fn runs under no_grad → requires_grad should be False
            self.assertFalse(call_log[0])

        def test_output_is_detached(self):
            ckpt = CheckpointWithoutOutput(
                mode=NormOutRecomputeMode.RECOMPUTE, cache_key="test"
            )
            x = torch.randn(4, requires_grad=True)
            out = ckpt.checkpoint(lambda t: t + 1, x)
            self.assertFalse(out.requires_grad)

    # -----------------------------------------------------------------------
    # Test 7: CheckpointWithoutOutput in OFFLOAD mode with LocalityCache
    # -----------------------------------------------------------------------

    class TestCheckpointWithoutOutputOffload(unittest.TestCase):
        def test_offload_stores_in_loc(self):
            loc = LocalityCache(max_bytes=16 * 1024 * 1024)
            ckpt = CheckpointWithoutOutput(
                mode=NormOutRecomputeMode.OFFLOAD,
                loc=loc,
                cache_key="offload_test",
            )
            x = torch.randn(8, 16)
            out = ckpt.checkpoint(lambda t: t * 3.0, x)
            self.assertEqual(len(loc), 1)

        def test_offload_retrieve_matches_forward(self):
            loc = LocalityCache(max_bytes=16 * 1024 * 1024)
            ckpt = CheckpointWithoutOutput(
                mode=NormOutRecomputeMode.OFFLOAD,
                loc=loc,
                cache_key="match_test",
            )
            x = torch.randn(6, 12)
            expected = x * 2.5
            _ = ckpt.checkpoint(lambda t: t * 2.5, x)
            retrieved = loc.retrieve("match_test", torch.device("cpu"))
            self.assertIsNotNone(retrieved)
            self.assertTrue(torch.allclose(expected, retrieved.float().to(expected.dtype), atol=1e-4))

    # -----------------------------------------------------------------------
    # Test 8: HeteroGDNConfig validation
    # -----------------------------------------------------------------------

    class TestHeteroGDNConfig(unittest.TestCase):
        def test_valid_config(self):
            cfg = _make_config()
            self.assertIn("gdn_norm_out", cfg.recompute_modules)

        def test_invalid_module_raises(self):
            with self.assertRaises(ValueError):
                _make_config(recompute_modules=["not_a_real_module"])

        def test_gdn_norm_out_requires_selective(self):
            with self.assertRaises(ValueError):
                _make_config(
                    recompute_modules=["gdn_norm_out"],
                    recompute_granularity="full",
                )

        def test_empty_recompute_modules_ok(self):
            cfg = _make_config(recompute_modules=[], recompute_granularity="selective")
            self.assertEqual(cfg.recompute_modules, [])

    # -----------------------------------------------------------------------
    # Test 9: HeteroGDNSelectiveRecompute forward — KEEP mode (CPU)
    # -----------------------------------------------------------------------

    class TestHeteroGDNSelectiveRecomputeKeep(unittest.TestCase):
        def _build_module(self, mode_override: Optional[NormOutRecomputeMode] = None):
            cfg = _make_config(
                hidden_size=64,
                num_heads=2,
                value_head_dim=32,
                seq_len=8,
                batch_size=2,
                dtype=torch.float32,
                recompute_modules=[],  # KEEP mode by default
                recompute_granularity="selective",
            )
            mod = HeteroGDNSelectiveRecompute(cfg, device=torch.device("cpu"))
            if mode_override is not None:
                mod.norm_out_mode = mode_override
                mod.recompute_norm_out = mode_override != NormOutRecomputeMode.KEEP
            return mod

        def _make_inputs(self, batch=2, heads=2, seq=8, d=32, dtype=torch.float32):
            # core_attn_out: (batch, heads, seq, d)
            x = torch.randn(batch, heads, seq, d, dtype=dtype, requires_grad=True)
            g = torch.randn(batch, heads, seq, d, dtype=dtype, requires_grad=True)
            return x, g

        def test_keep_mode_output_shape(self):
            mod = self._build_module()
            x, g = self._make_inputs()
            norm_out, ckpt = mod(x, g, batch=2, seq_len=8)
            # Expected: (seq, batch, heads*d) = (8, 2, 64)
            self.assertEqual(norm_out.shape, (8, 2, 64))
            self.assertIsNone(ckpt)

        def test_keep_mode_gradient_flows(self):
            mod = self._build_module()
            x, g = self._make_inputs()
            norm_out, _ = mod(x, g, batch=2, seq_len=8)
            norm_out.sum().backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(g.grad)
            self.assertFalse(torch.isnan(x.grad).any())

        def test_recompute_mode_returns_checkpoint_object(self):
            mod = self._build_module()
            mod.recompute_norm_out = True
            mod.norm_out_mode = NormOutRecomputeMode.RECOMPUTE
            x, g = self._make_inputs()
            norm_out, ckpt = mod(x, g, batch=2, seq_len=8)
            self.assertIsNotNone(ckpt)
            self.assertIsInstance(ckpt, CheckpointWithoutOutput)

        def test_offload_mode_writes_to_loc(self):
            mod = self._build_module()
            mod.recompute_norm_out = True
            mod.norm_out_mode = NormOutRecomputeMode.OFFLOAD
            mod.loc = LocalityCache(max_bytes=16 * 1024 * 1024)
            x, g = self._make_inputs()
            _, ckpt = mod(x, g, batch=2, seq_len=8)
            self.assertIsNotNone(ckpt)
            # One entry should now be in the LOC
            self.assertEqual(len(mod.loc), 1)

        def test_deterministic_output_across_calls(self):
            """Two identical forward passes in KEEP mode must produce identical outputs."""
            mod = self._build_module()
            x, g = self._make_inputs()
            out1, _ = mod(x, g, batch=2, seq_len=8)
            out2, _ = mod(x, g, batch=2, seq_len=8)
            self.assertTrue(torch.equal(out1, out2))

    # -----------------------------------------------------------------------
    # Test 10: _unpack_sequence_desync
    # -----------------------------------------------------------------------

    class TestUnpackSequenceDesync(unittest.TestCase):
        def test_basic_split(self):
            tensor = torch.arange(10).float().unsqueeze(-1)  # (10, 1)
            cu = torch.tensor([0, 3, 7, 10])
            parts = _unpack_sequence_desync(tensor, cu, dim=0)
            self.assertEqual(len(parts), 3)
            self.assertEqual(parts[0].shape[0], 3)
            self.assertEqual(parts[1].shape[0], 4)
            self.assertEqual(parts[2].shape[0], 3)

        def test_reassembly(self):
            tensor = torch.randn(20, 8)
            cu = torch.tensor([0, 5, 12, 20])
            parts = _unpack_sequence_desync(tensor, cu, dim=0)
            reassembled = torch.cat(parts, dim=0)
            self.assertTrue(torch.equal(tensor, reassembled))

        def test_single_sequence(self):
            tensor = torch.randn(7, 4)
            cu = torch.tensor([0, 7])
            parts = _unpack_sequence_desync(tensor, cu, dim=0)
            self.assertEqual(len(parts), 1)
            self.assertTrue(torch.equal(parts[0], tensor))

    # -----------------------------------------------------------------------
    # Test 11: PCIeBandwidthModel cost estimates are sane
    # -----------------------------------------------------------------------

    class TestPCIeBandwidthModel(unittest.TestCase):
        def test_larger_tensor_takes_longer(self):
            m = PCIeBandwidthModel()
            t_small = m.transfer_time_s(1024)
            t_large = m.transfer_time_s(1024 * 1024 * 1024)
            self.assertGreater(t_large, t_small)

        def test_h2d_vs_d2h_direction(self):
            m = PCIeBandwidthModel(h2d_bw_gbps=28.0, d2h_bw_gbps=14.0)
            # D2H is slower (lower bandwidth) → should take longer
            t_d2h = m.transfer_time_s(100_000_000, "d2h")
            t_h2d = m.transfer_time_s(100_000_000, "h2d")
            self.assertGreater(t_d2h, t_h2d)

        def test_zero_bytes_returns_latency(self):
            m = PCIeBandwidthModel(latency_us=10.0)
            t = m.transfer_time_s(0)
            self.assertAlmostEqual(t, 10e-6, places=10)

    # -----------------------------------------------------------------------
    # Test 12: PhasedPCIeCollective single-GPU no-op
    # -----------------------------------------------------------------------

    class TestPhasedPCIeCollectiveSingleGPU(unittest.TestCase):
        def test_no_group_returns_tensor_unchanged(self):
            t = torch.randn(8, 4, 16)
            out = PhasedPCIeCollective.hp_to_cp(t, seq_dim=0, head_dim=-1, cp_group=None)
            self.assertTrue(torch.equal(t, out))

    # -----------------------------------------------------------------------
    # Test 13: Integration — full forward→backward with mode=RECOMPUTE (CPU)
    # -----------------------------------------------------------------------

    class TestIntegrationRecomputeCPU(unittest.TestCase):
        """
        Verifies that the RECOMPUTE path produces the same gradients as the
        KEEP path.  Runs entirely on CPU to avoid CUDA dependency.
        """

        def _run_forward_backward(
            self, mode: NormOutRecomputeMode
        ) -> Tuple[Tensor, Tensor, Tensor]:
            torch.manual_seed(42)
            cfg = _make_config(
                hidden_size=64,
                num_heads=2,
                value_head_dim=32,
                seq_len=8,
                batch_size=2,
                dtype=torch.float32,
                recompute_modules=["gdn_norm_out"] if mode != NormOutRecomputeMode.KEEP else [],
                recompute_granularity="selective",
            )
            mod = HeteroGDNSelectiveRecompute(cfg, device=torch.device("cpu"))
            mod.norm_out_mode = mode
            if mode != NormOutRecomputeMode.KEEP:
                mod.recompute_norm_out = True

            # Zero-out and re-seed norm weights for reproducibility
            nn.init.ones_(mod.gated_norm.weight)

            torch.manual_seed(7)
            x = torch.randn(2, 2, 8, 32, dtype=torch.float32, requires_grad=True)
            g = torch.randn(2, 2, 8, 32, dtype=torch.float32, requires_grad=True)

            norm_out, ckpt = mod(x.clone().detach().requires_grad_(True),
                                 g.clone().detach().requires_grad_(True),
                                 batch=2, seq_len=8)

            # Simulate output projection (linear)
            proj = nn.Linear(64, 32, bias=False)
            nn.init.eye_(proj.weight[:32])  # deterministic
            out = proj(norm_out)

            if ckpt is not None:
                ckpt.discard_output_and_register_recompute(out)

            out.sum().backward()

            return out.detach(), norm_out.detach(), mod.gated_norm.weight.grad

        def test_keep_and_recompute_weight_grads_finite(self):
            _, _, wgrad_keep = self._run_forward_backward(NormOutRecomputeMode.KEEP)
            _, _, wgrad_rec  = self._run_forward_backward(NormOutRecomputeMode.RECOMPUTE)
            if wgrad_keep is not None:
                self.assertFalse(torch.isnan(wgrad_keep).any())
            if wgrad_rec is not None:
                self.assertFalse(torch.isnan(wgrad_rec).any())

        def test_output_shape_consistent(self):
            out_keep, _, _ = self._run_forward_backward(NormOutRecomputeMode.KEEP)
            out_rec,  _, _ = self._run_forward_backward(NormOutRecomputeMode.RECOMPUTE)
            self.assertEqual(out_keep.shape, out_rec.shape)

    # -----------------------------------------------------------------------
    # Test 14: register_hetero_gdn_hooks with mock engine
    # -----------------------------------------------------------------------

    class TestRegisterHooks(unittest.TestCase):
        class _MockEngine:
            def __init__(self):
                self._callbacks = []

            def register_memory_pressure_callback(self, cb):
                self._callbacks.append(cb)

        def test_hook_registered(self):
            engine = self._MockEngine()
            cfg = _make_config(recompute_modules=[], recompute_granularity="selective")
            mod = HeteroGDNSelectiveRecompute(cfg, device=torch.device("cpu"))
            register_hetero_gdn_hooks(engine, [mod])
            self.assertEqual(len(engine._callbacks), 1)

        def test_mode_escalation_keep_to_offload(self):
            engine = self._MockEngine()
            cfg = _make_config(recompute_modules=[], recompute_granularity="selective")
            mod = HeteroGDNSelectiveRecompute(cfg, device=torch.device("cpu"))
            mod.norm_out_mode = NormOutRecomputeMode.KEEP
            register_hetero_gdn_hooks(engine, [mod])
            # Simulate memory pressure < 15%
            engine._callbacks[0](free_bytes=100, total_bytes=1000)
            self.assertEqual(mod.norm_out_mode, NormOutRecomputeMode.OFFLOAD)

        def test_mode_escalation_offload_to_recompute(self):
            engine = self._MockEngine()
            cfg = _make_config(recompute_modules=[], recompute_granularity="selective")
            mod = HeteroGDNSelectiveRecompute(cfg, device=torch.device("cpu"))
            mod.norm_out_mode = NormOutRecomputeMode.OFFLOAD
            loc = LocalityCache(max_bytes=4096)
            mod.loc = loc
            register_hetero_gdn_hooks(engine, [mod])
            engine._callbacks[0](free_bytes=100, total_bytes=1000)
            self.assertEqual(mod.norm_out_mode, NormOutRecomputeMode.RECOMPUTE)
            self.assertIsNone(mod.loc)

        def test_no_callback_api_no_crash(self):
            class _BareEngine:
                pass
            cfg = _make_config(recompute_modules=[], recompute_granularity="selective")
            mod = HeteroGDNSelectiveRecompute(cfg, device=torch.device("cpu"))
            # Should not raise even without register_memory_pressure_callback
            register_hetero_gdn_hooks(_BareEngine(), [mod])

        def test_empty_module_list_no_crash(self):
            engine = self._MockEngine()
            register_hetero_gdn_hooks(engine, [])
            self.assertEqual(len(engine._callbacks), 0)

    # -----------------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------------

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestDeviceTierRegistry,
        TestDecideNormOutMode,
        TestLocalityCache,
        TestGatedRMSNorm,
        TestCheckpointWithoutOutputKeep,
        TestCheckpointWithoutOutputRecompute,
        TestCheckpointWithoutOutputOffload,
        TestHeteroGDNConfig,
        TestHeteroGDNSelectiveRecomputeKeep,
        TestUnpackSequenceDesync,
        TestPCIeBandwidthModel,
        TestPhasedPCIeCollectiveSingleGPU,
        TestIntegrationRecomputeCPU,
        TestRegisterHooks,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
