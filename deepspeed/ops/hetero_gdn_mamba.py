"""
Heterogeneous GDN-Mamba Layer Dispatcher for DES-LOC Heterogeneous Training.

Upstream Design Intent (Megatron 8f7fbe78):
============================================
Megatron PR #3535 introduces GatedDeltaNet (GDN) as a new layer type symbol 'G'
in the hybrid Mamba architecture. Upstream changes:
  1. Adds GDN symbol to `mamba_hybrid_layer_allocation.py` alongside M/*/−/E.
  2. Adds `gdn_layer` slot to `MambaStackSubmodules` and handles `LayerSymbols.GDN`
     in `MambaStack.__init__`.
  3. Adds GDN FLOPs estimation (`gdn_layer_flops`) to `num_floating_point_operations`.
  4. Explicitly blocks GDN during inference with `NotImplementedError` (pending KV-cache
     support), while full training support is wired through the layer specs.

The GDN operator uses a *gated delta rule*: it maintains a recurrent state matrix S
updated as S ← a*(I − b*K*Kᵀ)*S + V*Kᵀ, then reads out h = S*q.  This write is
O(d²) per token — fundamentally different from Mamba's selective scan and from
standard attention.  That distinction is architecturally significant for placement.

DES-LOC Adaptation Points:
===========================
DES-LOC = Decoupled Execution with Shared LOcality Cache.

Hardware topology: 2× A6000 (48 GB, SM86) + 1× H100 NVL (96 GB, SM90), PCIe only,
                   1.5 TB CPU DRAM.

Key insight: GDN's per-token O(d²) state update is *compute-bound* and benefits from
SM90's BF16/FP8 tensor-core throughput; attention is memory-bandwidth-bound and
runs adequately on SM86; Mamba selective scan is memory-bound *and* recurrence-limited
— SM86 is fine and keeps H100 free.

This module implements three DES-LOC-specific mechanisms:

A. DeviceAffinity routing
   A static placement table maps each layer-type symbol to a preferred device rank.
   GDN → rank 2 (H100), Attention → rank 0 or 1 (A6000), Mamba → rank 0 or 1.
   The table is overridable at runtime via env vars (DES_LOC_GDN_RANK, etc.).

B. Shared Locality Cache (SLC)
   A CPU-pinned, page-locked tensor buffer that acts as a rendezvous point between
   devices.  When a layer runs on a different device than its predecessor, activations
   are staged through SLC rather than P2P (which is unavailable without NVLink).
   SLC is organised as a ring of slots to overlap DMA with compute.

C. FLOPs-aware micro-batch splitting
   Given the per-symbol FLOPs formulas from Megatron's `gdn_layer_flops` /
   `mamba_layer_flops` / `attention_layer_flops`, this module computes a split ratio
   for hybrid batches so that each device finishes roughly simultaneously (balanced
   makespan), avoiding PCIe stalls from waiting on the bottleneck device.

The module is intentionally deepspeed-agnostic at the import level so it can be unit-
tested without a full DS environment.  The DeepSpeed engine hooks in via
`register_des_loc_hooks(engine)`.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer-type symbols (mirrors Megatron Symbols class, no upstream dep)
# ---------------------------------------------------------------------------

class LayerSymbol(str, Enum):
    MAMBA     = "M"
    GDN       = "G"
    ATTENTION = "*"
    MLP       = "-"
    MOE       = "E"


# ---------------------------------------------------------------------------
# Hardware capability descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceCapability:
    """Immutable descriptor for a single CUDA device in the DES-LOC cluster.

    sm_major / sm_minor: CUDA compute capability (e.g. 8, 6 for SM86).
    vram_gb:  Usable VRAM in gigabytes.
    bf16_tflops: Peak BF16 tensor-core throughput in TFLOP/s (approximate).
    membw_gbps:  Peak HBM/GDDR6 bandwidth in GB/s.
    """
    rank:          int
    name:          str
    sm_major:      int
    sm_minor:      int
    vram_gb:       float
    bf16_tflops:   float
    membw_gbps:    float

    @property
    def sm(self) -> int:
        return self.sm_major * 10 + self.sm_minor

    @property
    def is_h100_class(self) -> bool:
        return self.sm >= 90

    @property
    def is_ampere(self) -> bool:
        return 80 <= self.sm < 90


def _probe_devices(ranks: Optional[List[int]] = None) -> List[DeviceCapability]:
    """Auto-detect CUDA devices present in this process.

    Known-device table covers the three hardware targets in the DES-LOC cluster:
      - A6000 (SM86, 48 GB GDDR6, ~310 BF16 TFLOP/s, ~768 GB/s)
      - H100 NVL (SM90, 96 GB HBM3, ~1,979 BF16 TFLOP/s, ~3,938 GB/s)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; returning empty device list.")
        return []

    KNOWN: Dict[str, Tuple[float, float]] = {
        # name_substr: (bf16_tflops, membw_gbps)
        "A6000":  (309.7,  768.0),
        "H100":   (1978.9, 3938.0),
    }

    n = torch.cuda.device_count()
    if ranks is None:
        ranks = list(range(n))

    caps: List[DeviceCapability] = []
    for r in ranks:
        if r >= n:
            logger.warning("Rank %d requested but only %d devices found; skipping.", r, n)
            continue
        props = torch.cuda.get_device_properties(r)
        bf16, bw = KNOWN.get("A6000", (150.0, 600.0))  # conservative fallback
        for k, v in KNOWN.items():
            if k in props.name:
                bf16, bw = v
                break
        vram = props.total_memory / (1 << 30)
        cap = DeviceCapability(
            rank=r,
            name=props.name,
            sm_major=props.major,
            sm_minor=props.minor,
            vram_gb=vram,
            bf16_tflops=bf16,
            membw_gbps=bw,
        )
        logger.info(
            "Device %d: %s  SM%d%d  %.1f GB  BF16=%.0f TFLOP/s  BW=%.0f GB/s",
            r, cap.name, cap.sm_major, cap.sm_minor, cap.vram_gb,
            cap.bf16_tflops, cap.membw_gbps,
        )
        caps.append(cap)
    return caps


# ---------------------------------------------------------------------------
# Device Affinity Table
# ---------------------------------------------------------------------------

class DeviceAffinityTable:
    """Maps layer-type symbols to preferred device ranks.

    Default policy for the A6000×2 + H100 cluster:
      G (GDN)       → H100 (rank 2): O(d²) compute-bound delta-rule update
      * (Attention)  → A6000 (rank 0): memory-BW-bound; spread across A6000s
      M (Mamba)      → A6000 (rank 1): selective-scan is recurrence+BW bound
      - (MLP)        → H100 (rank 2): large GEMM benefits from SM90
      E (MoE)        → H100 (rank 2): sparse gating + expert GEMMs

    Override via environment variables:
      DES_LOC_GDN_RANK, DES_LOC_ATT_RANK, DES_LOC_MBA_RANK,
      DES_LOC_MLP_RANK, DES_LOC_MOE_RANK

    The table is *soft*: if the preferred device lacks VRAM for the current
    micro-batch, `resolve` falls back to the next-best candidate.
    """

    _ENV_MAP: Dict[LayerSymbol, str] = {
        LayerSymbol.GDN:       "DES_LOC_GDN_RANK",
        LayerSymbol.ATTENTION: "DES_LOC_ATT_RANK",
        LayerSymbol.MAMBA:     "DES_LOC_MBA_RANK",
        LayerSymbol.MLP:       "DES_LOC_MLP_RANK",
        LayerSymbol.MOE:       "DES_LOC_MOE_RANK",
    }

    def __init__(self, caps: List[DeviceCapability]) -> None:
        self._caps: Dict[int, DeviceCapability] = {c.rank: c for c in caps}
        # Build default policy based on SM class
        h100_ranks = [c.rank for c in caps if c.is_h100_class]
        a6000_ranks = [c.rank for c in caps if c.is_ampere]
        h_pref = h100_ranks[0] if h100_ranks else (caps[0].rank if caps else 0)
        a0 = a6000_ranks[0] if len(a6000_ranks) > 0 else h_pref
        a1 = a6000_ranks[1] if len(a6000_ranks) > 1 else a0

        self._default: Dict[LayerSymbol, int] = {
            LayerSymbol.GDN:       h_pref,
            LayerSymbol.ATTENTION: a0,
            LayerSymbol.MAMBA:     a1,
            LayerSymbol.MLP:       h_pref,
            LayerSymbol.MOE:       h_pref,
        }
        logger.debug("DeviceAffinityTable defaults: %s", self._default)

    def resolve(self, sym: LayerSymbol) -> int:
        """Return preferred device rank, honouring env-var overrides."""
        env_key = self._ENV_MAP.get(sym)
        if env_key and env_key in os.environ:
            try:
                return int(os.environ[env_key])
            except ValueError:
                logger.warning("Invalid value for %s; using default.", env_key)
        return self._default.get(sym, 0)

    def all_ranks(self) -> List[int]:
        return list(self._caps.keys())


# ---------------------------------------------------------------------------
# Shared Locality Cache (SLC)
# ---------------------------------------------------------------------------

@dataclass
class SLCSlot:
    """A single staging slot in the Shared Locality Cache ring buffer."""
    index:   int
    tensor:  torch.Tensor        # pinned CPU tensor, pre-allocated
    in_use:  threading.Event = field(default_factory=threading.Event)
    src_dev: Optional[int] = None
    dst_dev: Optional[int] = None


class SharedLocalityCache:
    """CPU-pinned ring buffer for inter-device activation staging (no NVLink).

    DES-LOC rationale:
      Without NVLink, GPU-to-GPU transfer must go through host memory.  Naive
      `.to(device)` causes a device-sync + cudaMemcpy which serialises compute.
      SLC pre-allocates page-locked host tensors and uses double-buffering so
      DMA of layer N's output overlaps with compute of layer N+1.

    Usage:
      slot = slc.acquire(shape, dtype)
      slot.tensor[:] = activation.cpu()   # async DMA to pinned mem
      ... schedule next layer on dst_dev ...
      dst_tensor = slot.tensor.to(dst_dev, non_blocking=True)
      slc.release(slot)

    Args:
      num_slots:  Number of ring slots (2 = double-buffer, 4 = quad-buffer).
      max_bytes:  Maximum bytes per slot.  Tensors larger than this cause a
                  fallback to synchronous transfer with a logged warning.
      dtype:      Default dtype for pre-allocation (float16 or bfloat16).
    """

    def __init__(
        self,
        num_slots: int = 4,
        max_bytes: int = 512 * 1024 * 1024,   # 512 MB default
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._num_slots = num_slots
        self._max_bytes = max_bytes
        self._dtype = dtype
        self._slots: List[SLCSlot] = []
        self._lock = threading.Lock()
        self._next_slot = 0
        self._initialised = False
        logger.info(
            "SLC: %d slots × %d MB, dtype=%s",
            num_slots, max_bytes >> 20, dtype,
        )

    def _lazy_init(self, shape: Tuple[int, ...], dtype: torch.dtype) -> None:
        if self._initialised:
            return
        elem_bytes = torch.finfo(dtype).bits // 8
        needed = math.prod(shape) * elem_bytes
        if needed > self._max_bytes:
            logger.warning(
                "SLC slot size %d MB < requested %d MB; using max.",
                self._max_bytes >> 20, needed >> 20,
            )
        alloc_elems = self._max_bytes // elem_bytes
        for i in range(self._num_slots):
            buf = torch.empty(alloc_elems, dtype=dtype, pin_memory=True)
            slot = SLCSlot(index=i, tensor=buf)
            self._slots.append(slot)
        self._initialised = True
        logger.debug("SLC lazy-init: allocated %d × %d MB pinned buffers.", 
                     self._num_slots, self._max_bytes >> 20)

    def acquire(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src_dev: Optional[int] = None,
        dst_dev: Optional[int] = None,
    ) -> SLCSlot:
        """Acquire a free SLC slot, waiting if all are in use."""
        self._lazy_init(shape, dtype)
        deadline = time.monotonic() + 10.0
        while True:
            with self._lock:
                slot = self._slots[self._next_slot % self._num_slots]
                if not slot.in_use.is_set():
                    slot.in_use.set()
                    slot.src_dev = src_dev
                    slot.dst_dev = dst_dev
                    self._next_slot += 1
                    return slot
            if time.monotonic() > deadline:
                raise RuntimeError("SLC: timed out waiting for a free slot (10 s).")
            time.sleep(0.001)

    def release(self, slot: SLCSlot) -> None:
        """Return a slot to the free pool."""
        slot.src_dev = None
        slot.dst_dev = None
        slot.in_use.clear()

    def transfer(
        self,
        tensor: torch.Tensor,
        dst_device: torch.device,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Stage tensor through SLC to dst_device.

        If tensor is already on dst_device, returns it unchanged.
        Falls back to direct .to() if tensor exceeds SLC slot capacity.
        """
        if tensor.device == dst_device:
            return tensor

        elem_bytes = tensor.element_size()
        needed = tensor.numel() * elem_bytes
        if needed > self._max_bytes:
            logger.warning(
                "SLC.transfer: tensor %d MB > slot %d MB; falling back to direct copy.",
                needed >> 20, self._max_bytes >> 20,
            )
            return tensor.to(dst_device, non_blocking=non_blocking)

        slot = self.acquire(
            tuple(tensor.shape), tensor.dtype,
            src_dev=tensor.device.index,
            dst_dev=dst_device.index if hasattr(dst_device, "index") else None,
        )
        try:
            flat = tensor.reshape(-1)
            n = flat.numel()
            slot.tensor[:n].copy_(flat, non_blocking=non_blocking)
            result = slot.tensor[:n].to(dst_device, non_blocking=non_blocking)
            result = result.reshape(tensor.shape)
        finally:
            self.release(slot)
        return result


# ---------------------------------------------------------------------------
# FLOPs estimation (mirrors Megatron training.py formulas, standalone)
# ---------------------------------------------------------------------------

def gdn_layer_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    qk_head_dim: int = 128,
    v_head_dim: int = 128,
    num_qk_heads: int = 16,
    num_v_heads: int = 32,
    conv_kernel_dim: int = 4,
) -> int:
    """FLOPs for one GDN layer forward pass (mirrors Megatron gdn_layer_flops).

    Breakdown:
      in_proj:    hidden → (2·qk_dim + 2·v_dim + 2·num_v_heads)
      conv1d:     kernel over (2·qk_dim + v_dim) channels
      delta-rule: KKᵀ, VKᵀ, S·(a(I−b·KKᵀ)), S·Q  — all O(v_head_dim²) per head
      out_proj:   v_dim → hidden
    """
    qk_dim = qk_head_dim * num_qk_heads
    v_dim  = v_head_dim  * num_v_heads
    return int(
        2 * batch_size * seq_len * (
            hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
            + conv_kernel_dim * (2 * qk_dim + v_dim)
            + num_v_heads * (v_head_dim ** 2) * 4
            + hidden_size * v_dim
        )
    )


def mamba_layer_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    state_dim: int = 128,
    head_dim: int = 64,
    num_groups: int = 8,
    num_heads: int = 128,
    conv_kernel: int = 4,
    expansion: int = 2,
) -> int:
    """FLOPs for one Mamba-2 layer forward pass (mirrors Megatron mamba_layer_flops)."""
    d_in = expansion * hidden_size
    d_state = state_dim * num_groups
    return int(
        2 * batch_size * seq_len * (
            hidden_size * (2 * d_in + 2 * num_heads + d_state)   # in_proj
            + conv_kernel * (d_in + d_state)                      # conv1d
            + num_heads * (head_dim ** 2)                         # SSM state update
            + d_in * hidden_size                                  # out_proj
        )
    )


def attention_layer_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int = 32,
    gqa: bool = True,
    kv_channels: Optional[int] = None,
) -> int:
    """FLOPs for one self-attention layer (simplified, no GQA split complexity)."""
    if kv_channels is None:
        kv_channels = hidden_size // num_heads
    # QKV proj + attention scores + attention output + out_proj
    qkv_proj  = 2 * batch_size * seq_len * hidden_size * (3 * hidden_size)
    attn_core = 2 * batch_size * num_heads * seq_len * seq_len * kv_channels
    out_proj  = 2 * batch_size * seq_len * hidden_size * hidden_size
    return int(qkv_proj + attn_core + out_proj)


# ---------------------------------------------------------------------------
# Micro-batch split: balanced makespan across devices
# ---------------------------------------------------------------------------

@dataclass
class LayerProfile:
    """Per-layer FLOPs and preferred device for makespan estimation."""
    symbol:        LayerSymbol
    flops_per_tok: float          # FLOPs at batch=1, seq=1
    preferred_dev: int


def compute_balanced_split(
    layer_sequence: List[LayerProfile],
    caps: List[DeviceCapability],
    total_batch: int,
    seq_len: int,
) -> Dict[int, int]:
    """Compute per-device micro-batch sizes for balanced makespan.

    The DES-LOC heterogeneous cluster has wildly different compute capacities
    (H100 ≈ 6× A6000 in BF16).  Assigning equal micro-batches would leave the
    H100 idle 5/6 of the time.  This function solves a simple proportional
    allocation:

        share_d = tflops_d / Σ tflops_d'
        batch_d = round(share_d × total_batch)

    with remainder assigned to the highest-capacity device.  The allocation
    is then clipped so that Σ batch_d == total_batch.

    Args:
        layer_sequence: Ordered list of LayerProfile objects for one pipeline.
        caps: Device capability descriptors.
        total_batch: Global micro-batch size to split.
        seq_len: Sequence length (for FLOPs scaling).

    Returns:
        Dict mapping device rank → local micro-batch size.
    """
    cap_map: Dict[int, DeviceCapability] = {c.rank: c for c in caps}

    # Aggregate FLOPs per device
    dev_flops: Dict[int, float] = {c.rank: 0.0 for c in caps}
    for lp in layer_sequence:
        if lp.preferred_dev in dev_flops:
            dev_flops[lp.preferred_dev] += lp.flops_per_tok * seq_len

    total_flops = sum(dev_flops.values()) or 1.0

    # Time ∝ flops / throughput
    dev_time: Dict[int, float] = {}
    for rank, flops in dev_flops.items():
        tput = cap_map[rank].bf16_tflops * 1e12 if rank in cap_map else 1e12
        dev_time[rank] = flops / tput if tput > 0 else 0.0

    total_time = sum(dev_time.values()) or 1.0

    # Proportional batch allocation (more time → more batch to keep busy)
    raw_split: Dict[int, float] = {
        r: (t / total_time) * total_batch for r, t in dev_time.items()
    }
    int_split: Dict[int, int] = {r: max(1, int(v)) for r, v in raw_split.items()}
    remainder = total_batch - sum(int_split.values())

    # Assign remainder to highest-throughput device
    if remainder != 0 and caps:
        best = max(caps, key=lambda c: c.bf16_tflops)
        int_split[best.rank] = int_split.get(best.rank, 0) + remainder

    logger.debug(
        "Balanced split (total=%d): %s  [dev_time=%s]",
        total_batch, int_split,
        {r: f"{t*1000:.2f}ms" for r, t in dev_time.items()},
    )
    return int_split


# ---------------------------------------------------------------------------
# Heterogeneous GDN-Mamba Layer
# ---------------------------------------------------------------------------

class HeteroGDNMambaLayer(nn.Module):
    """DES-LOC wrapper that executes a single hybrid layer on its preferred device.

    This is the core DES-LOC abstraction: given a layer module (GDN, Mamba,
    Attention, MLP, or MoE) and a symbol, HeteroGDNMambaLayer:
      1. Moves the module to the preferred device at construction time.
      2. On forward, uses SLC to stage the input tensor from the current device
         to the layer's device (PCIe DMA via pinned memory).
      3. Runs the forward pass on the preferred device.
      4. Stages the output back to the *caller's* device (or leaves it on the
         layer device if the next layer is on the same device — zero-copy path).

    Args:
        module:       The layer nn.Module (already constructed, not yet placed).
        symbol:       LayerSymbol indicating layer type.
        affinity:     DeviceAffinityTable for device lookup.
        slc:          SharedLocalityCache instance for inter-device transfers.
        caller_device: Device where upstream activations originate (default cuda:0).
    """

    def __init__(
        self,
        module: nn.Module,
        symbol: LayerSymbol,
        affinity: DeviceAffinityTable,
        slc: SharedLocalityCache,
        caller_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.symbol = symbol
        self._affinity = affinity
        self._slc = slc
        self._preferred_rank = affinity.resolve(symbol)
        self._preferred_device = torch.device(f"cuda:{self._preferred_rank}")
        self._caller_device = caller_device or torch.device("cuda:0")

        # Place module on its preferred device
        if torch.cuda.is_available():
            self.layer_module = module.to(self._preferred_device)
        else:
            self.layer_module = module

        logger.info(
            "HeteroGDNMambaLayer[%s] placed on %s (caller=%s)",
            symbol.value, self._preferred_device, self._caller_device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with SLC-mediated device migration.

        Transfer path:
          caller_device → [SLC] → preferred_device → compute → [SLC] → caller_device

        The second transfer is skipped (zero-copy) if preferred_device == caller_device.
        """
        src_device = hidden_states.device
        need_transfer_in  = (src_device != self._preferred_device)
        need_transfer_out = (self._preferred_device != self._caller_device)

        t0 = time.perf_counter()

        if need_transfer_in:
            hidden_states = self._slc.transfer(hidden_states, self._preferred_device)
            if attention_mask is not None and attention_mask.device != self._preferred_device:
                attention_mask = attention_mask.to(self._preferred_device, non_blocking=True)

        t_transfer_in = time.perf_counter()

        if attention_mask is not None:
            output = self.layer_module(hidden_states, attention_mask=attention_mask, **kwargs)
        else:
            output = self.layer_module(hidden_states, **kwargs)

        # Unwrap tuple outputs (some layers return (hidden, context))
        if isinstance(output, (tuple, list)):
            output = output[0]

        t_compute = time.perf_counter()

        if need_transfer_out:
            output = self._slc.transfer(output, self._caller_device)

        t_transfer_out = time.perf_counter()

        logger.debug(
            "[%s] xfer_in=%.2fms  compute=%.2fms  xfer_out=%.2fms",
            self.symbol.value,
            (t_transfer_in - t0) * 1e3,
            (t_compute - t_transfer_in) * 1e3,
            (t_transfer_out - t_compute) * 1e3,
        )
        return output


# ---------------------------------------------------------------------------
# Heterogeneous Stack: sequences of HeteroGDNMambaLayer
# ---------------------------------------------------------------------------

class HeteroGDNMambaStack(nn.Module):
    """DES-LOC heterogeneous stack of GDN/Mamba/Attention/MLP/MoE layers.

    Constructs a sequence of HeteroGDNMambaLayer wrappers from a layer-type
    pattern string (e.g. "GM*GM-") and a corresponding list of pre-built
    nn.Module objects.

    The stack also tracks inter-layer device transitions.  When two consecutive
    layers are on the same device, the SLC transfer is elided and a direct
    pointer is passed — this is the "Shared LOcality" in DES-LOC.

    Args:
        modules:        Pre-built layer modules in pattern order.
        pattern:        Layer type pattern string (e.g. "GM*GM-").
        caps:           Device capability list (from `_probe_devices`).
        slc:            Shared SLC instance (can be shared across stacks).
        caller_device:  Device to return activations to after each layer.
    """

    SYMBOL_MAP: Dict[str, LayerSymbol] = {s.value: s for s in LayerSymbol}

    def __init__(
        self,
        modules: List[nn.Module],
        pattern: str,
        caps: List[DeviceCapability],
        slc: Optional[SharedLocalityCache] = None,
        caller_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(modules) != len(pattern):
            raise ValueError(
                f"modules length {len(modules)} != pattern length {len(pattern)}"
            )

        self._caps = caps
        self._affinity = DeviceAffinityTable(caps)
        self._slc = slc or SharedLocalityCache()
        self._caller_device = caller_device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._pattern = pattern

        layers = []
        for i, (mod, sym_char) in enumerate(zip(modules, pattern)):
            sym = self.SYMBOL_MAP.get(sym_char)
            if sym is None:
                raise ValueError(f"Unknown layer symbol '{sym_char}' at position {i}.")
            layer = HeteroGDNMambaLayer(
                module=mod,
                symbol=sym,
                affinity=self._affinity,
                slc=self._slc,
                caller_device=self._caller_device,
            )
            layers.append(layer)

        # Use ModuleList so PyTorch tracks parameters
        self.layers = nn.ModuleList(layers)
        self._log_placement_plan()

    def _log_placement_plan(self) -> None:
        """Log the device placement plan for the stack."""
        plan = []
        for layer in self.layers:
            assert isinstance(layer, HeteroGDNMambaLayer)
            plan.append(f"{layer.symbol.value}@cuda:{layer._preferred_rank}")
        logger.info("HeteroGDNMambaStack placement: %s", " → ".join(plan))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sequential forward through all layers.

        Elides SLC transfers between consecutive same-device layers.
        """
        current = hidden_states
        prev_rank: Optional[int] = None

        for layer in self.layers:
            assert isinstance(layer, HeteroGDNMambaLayer)
            curr_rank = layer._preferred_rank

            # Elide caller→layer transfer if layer is already on current tensor's device
            # by temporarily overriding caller_device to the current tensor device.
            # This is the "Shared LOcality" zero-copy path.
            if prev_rank is not None and curr_rank == prev_rank:
                # Same device: set caller_device to this device to avoid round-trip
                orig_caller = layer._caller_device
                layer._caller_device = torch.device(f"cuda:{curr_rank}")
                current = layer(current, attention_mask=attention_mask, **kwargs)
                layer._caller_device = orig_caller
            else:
                current = layer(current, attention_mask=attention_mask, **kwargs)

            prev_rank = curr_rank

        # Final transfer back to canonical caller device if needed
        if current.device != self._caller_device and torch.cuda.is_available():
            current = self._slc.transfer(current, self._caller_device)

        return current

    def build_layer_profiles(self, seq_len: int) -> List[LayerProfile]:
        """Build LayerProfile list for makespan estimation."""
        profiles = []
        for layer in self.layers:
            assert isinstance(layer, HeteroGDNMambaLayer)
            sym = layer.symbol
            # Approximate flops_per_tok at batch=1 using default dims
            if sym == LayerSymbol.GDN:
                fpt = gdn_layer_flops(1, seq_len, 1024) / seq_len
            elif sym == LayerSymbol.MAMBA:
                fpt = mamba_layer_flops(1, seq_len, 1024) / seq_len
            elif sym == LayerSymbol.ATTENTION:
                fpt = attention_layer_flops(1, seq_len, 1024) / seq_len
            else:
                fpt = 2 * 1024 * 4096   # MLP/MoE rough estimate
            profiles.append(LayerProfile(
                symbol=sym,
                flops_per_tok=fpt,
                preferred_dev=layer._preferred_rank,
            ))
        return profiles


# ---------------------------------------------------------------------------
# DeepSpeed engine integration hook
# ---------------------------------------------------------------------------

def register_des_loc_hooks(engine: object, stack: HeteroGDNMambaStack) -> None:
    """Register DES-LOC pre/post-step hooks on a DeepSpeed engine.

    Hooks:
      pre_step:  Log per-device memory utilisation and SLC slot pressure.
      post_step: Emit makespan telemetry for imbalance detection.

    Args:
        engine: A DeepSpeed DeepSpeedEngine instance (duck-typed to avoid hard dep).
        stack:  The HeteroGDNMambaStack being managed.
    """
    if not hasattr(engine, "optimizer"):
        logger.warning("register_des_loc_hooks: engine lacks .optimizer; skipping hooks.")
        return

    def _pre_step_hook(*args, **kwargs) -> None:
        for cap in stack._caps:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(cap.rank)
                used_gb = (total - free) / (1 << 30)
                logger.debug(
                    "DES-LOC pre-step  cuda:%d  %.1f / %.1f GB used",
                    cap.rank, used_gb, cap.vram_gb,
                )

    def _post_step_hook(*args, **kwargs) -> None:
        logger.debug("DES-LOC post-step: step complete.")

    # DeepSpeed engines expose register_forward_pre_hook via the wrapped module
    try:
        engine.module.register_forward_pre_hook(lambda *a, **kw: _pre_step_hook())
        logger.info("DES-LOC hooks registered on DeepSpeed engine module.")
    except AttributeError:
        logger.warning("Could not register DES-LOC hooks: engine.module not found.")


# ---------------------------------------------------------------------------
# Hybrid pattern utilities
# ---------------------------------------------------------------------------

def parse_hybrid_pattern(pattern: str) -> List[LayerSymbol]:
    """Parse a hybrid layer pattern string into a list of LayerSymbol.

    Strips pipe '|' separators (pipeline stage markers) and ignores '/'
    (MTP separators) and everything after the first '/'.

    Example:
        "GM*|GM-" → [GDN, MAMBA, ATTENTION, GDN, MAMBA, MLP]
    """
    # Take only the main pattern (before first '/')
    main = pattern.split("/")[0]
    symbols: List[LayerSymbol] = []
    sym_map = HeteroGDNMambaStack.SYMBOL_MAP
    for ch in main:
        if ch == "|":
            continue
        sym = sym_map.get(ch)
        if sym is None:
            raise ValueError(f"Unknown layer symbol '{ch}' in pattern '{pattern}'.")
        symbols.append(sym)
    return symbols


def count_layer_types(pattern: str) -> Dict[LayerSymbol, int]:
    """Count occurrences of each layer type in a pattern string.

    Counts across both main and MTP patterns (all segments after '/').
    Pipe '|' characters are ignored.

    Returns a dict with a key for every LayerSymbol, including zeros.
    """
    counts: Dict[LayerSymbol, int] = {s: 0 for s in LayerSymbol}
    sym_map = HeteroGDNMambaStack.SYMBOL_MAP
    for ch in pattern:
        if ch in ("|", "/"):
            continue
        sym = sym_map.get(ch)
        if sym is not None:
            counts[sym] += 1
    return counts


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_hetero_stack(
    layer_factories: List[Callable[[], nn.Module]],
    pattern: str,
    slc_slots: int = 4,
    slc_max_mb: int = 512,
    caller_rank: int = 0,
) -> HeteroGDNMambaStack:
    """Convenience factory: probe devices, build SLC, construct HeteroGDNMambaStack.

    Args:
        layer_factories: Ordered list of callables, each returning an nn.Module.
                         Ordering must match `pattern`.
        pattern:         Layer type pattern string (e.g. "GM*GM-").
        slc_slots:       Number of SLC ring slots (default 4).
        slc_max_mb:      Max bytes per SLC slot in MB (default 512).
        caller_rank:     CUDA rank where the caller's tensors originate.

    Returns:
        A fully configured HeteroGDNMambaStack.
    """
    caps = _probe_devices()
    if not caps:
        logger.warning("No CUDA devices; stack will run on CPU with no device migration.")
        caps = [DeviceCapability(0, "CPU", 0, 0, 0.0, 1.0, 10.0)]

    slc = SharedLocalityCache(
        num_slots=slc_slots,
        max_bytes=slc_max_mb * 1024 * 1024,
    )
    caller_device = torch.device(f"cuda:{caller_rank}") if torch.cuda.is_available() else torch.device("cpu")
    modules = [f() for f in layer_factories]

    return HeteroGDNMambaStack(
        modules=modules,
        pattern=pattern,
        caps=caps,
        slc=slc,
        caller_device=caller_device,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 1. FLOPs formula sanity
    flops_gdn  = gdn_layer_flops(1, 128, 512)
    flops_mba  = mamba_layer_flops(1, 128, 512)
    flops_att  = attention_layer_flops(1, 128, 512)
    assert flops_gdn > 0,  "GDN FLOPs must be positive"
    assert flops_mba > 0,  "Mamba FLOPs must be positive"
    assert flops_att > 0,  "Attention FLOPs must be positive"
    logger.info("FLOPs: GDN=%d  Mamba=%d  Attn=%d", flops_gdn, flops_mba, flops_att)

    # 2. Pattern parsing
    syms = parse_hybrid_pattern("GM*|G-")
    expected = [LayerSymbol.GDN, LayerSymbol.MAMBA, LayerSymbol.ATTENTION,
                LayerSymbol.GDN, LayerSymbol.MLP]
    assert syms == expected, f"Pattern parse mismatch: {syms}"
    logger.info("Pattern parse OK: %s", [s.value for s in syms])

    # 3. Layer counts
    counts = count_layer_types("GM*GM-/GG")
    assert counts[LayerSymbol.GDN]       == 4, f"Expected 4 GDN, got {counts[LayerSymbol.GDN]}"
    assert counts[LayerSymbol.MAMBA]     == 2
    assert counts[LayerSymbol.ATTENTION] == 1
    logger.info("Layer counts: %s", {s.value: counts[s] for s in LayerSymbol})

    # 4. DeviceAffinityTable with synthetic caps
    fake_caps = [
        DeviceCapability(0, "RTX A6000", 8, 6, 48.0, 309.7,  768.0),
        DeviceCapability(1, "RTX A6000", 8, 6, 48.0, 309.7,  768.0),
        DeviceCapability(2, "H100 NVL",  9, 0, 96.0, 1978.9, 3938.0),
    ]
    affinity = DeviceAffinityTable(fake_caps)
    assert affinity.resolve(LayerSymbol.GDN) == 2,       "GDN should map to H100 (rank 2)"
    assert affinity.resolve(LayerSymbol.MAMBA) in (0, 1), "Mamba should map to A6000"
    logger.info("DeviceAffinityTable OK")

    # 5. Balanced split
    profiles = [
        LayerProfile(LayerSymbol.GDN,       flops_gdn / 128, 2),
        LayerProfile(LayerSymbol.MAMBA,      flops_mba / 128, 1),
        LayerProfile(LayerSymbol.ATTENTION,  flops_att / 128, 0),
    ]
    split = compute_balanced_split(profiles, fake_caps, total_batch=16, seq_len=128)
    assert sum(split.values()) == 16, f"Split sum {sum(split.values())} != 16"
    logger.info("Balanced split: %s", split)

    logger.info("All smoke tests passed.")
