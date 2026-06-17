"""
DES-LOC Heterogeneous GDN-Mamba Integration
============================================

Upstream Design Intent (Megatron commit 8f7fbe78):
    Philip Petrakian's "Introduce GDN to Mamba" adds Gated Delta Net (GDN) as a
    first-class layer type alongside Mamba SSM and standard attention layers in
    hybrid architectures. The key design choices upstream were:

    1. Symbol 'G' introduced into the hybrid pattern DSL (alongside 'M', '*', '-', 'E')
    2. GDN wraps inside a TransformerLayer shell to reuse the existing residual/norm
       infrastructure, with GatedDeltaNet as the self_attention submodule
    3. Layer maps are returned as (mamba, gdn, attention, mlp, moe) tuples — GDN
       counts are tracked separately for FLOPs accounting
    4. FLOPs formula for GDN: in_proj + conv1d + delta-rule recurrence + out_proj,
       parameterized by (qk_head_dim, v_head_dim, num_qk_heads, num_v_heads, conv_kernel_dim)
    5. Inference explicitly raises NotImplementedError for GDN until KV-cache is
       extended to handle the delta-rule recurrent state

DES-LOC Adaptation (HeteroGDNMambaIntegration):
    DES-LOC = Decoupled Execution with Shared LOcality Cache

    Hardware context: 2× A6000 48 GB SM86 + 1× H100 NVL 96 GB SM90, PCIe only,
    1.5 TB CPU DRAM.

    The core insight is that GDN, Mamba-SSM, and standard attention have very
    different compute/memory profiles that map naturally to different devices:

    ┌────────────────┬──────────────────────────────────────────────────────┐
    │ Layer Type     │ DES-LOC Device Assignment                            │
    ├────────────────┼──────────────────────────────────────────────────────┤
    │ GDN (G)        │ H100 NVL (SM90) — delta-rule recurrence is matmul-   │
    │                │ heavy with large v_dim; benefits most from BF16 TF32  │
    │ Mamba SSM (M)  │ A6000 pair (SM86) — selective scan is bandwidth-      │
    │                │ bound; A6000 HBM2e bandwidth is sufficient, avoids    │
    │                │ wasting H100 on scan primitives                        │
    │ Attention (*)  │ H100 NVL — flash-attn2 + large KV fits in 96 GB      │
    │ MLP / MoE      │ Balanced: MLP → A6000, MoE experts → CPU offload     │
    └────────────────┴──────────────────────────────────────────────────────┘

    The "Shared LOcality Cache" is a CPU DRAM tensor cache (up to ~400 GB
    addressable via pin_memory) that holds:
      - GDN recurrent state S (shape: [B, n_v_heads, v_head_dim, v_head_dim])
      - Mamba SSM hidden state h
      - Attention KV cache (prefix cache for inference)

    On each forward step the relevant slice is streamed in via PCIe, used on
    the target device, then written back. For training this degenerates to
    activation checkpointing with heterogeneous remat.

    Adaptation points relative to Megatron diff:
      A1. HeteroLayerSymbols mirrors Megatron Symbols but adds device hints
      A2. HeteroHybridPattern parses 'G'/'M'/'*'/'-'/'E' and emits per-layer
          DeviceAssignment objects consumed by DeepSpeed's engine
      A3. GDNFLOPsEstimator replicates Megatron's gdn_layer_flops() but also
          estimates PCIe transfer overhead for the recurrent state
      A4. LocalityCache manages the shared CPU buffer with async prefetch;
          GDNRecurrentState / MambaHiddenState subclass it
      A5. HeteroGDNMambaStack builds the layer list and assigns each layer to
          its device, replacing MambaStack.build_layers() for DES-LOC runtimes
      A6. Inference guard: like Megatron we raise NotImplementedError for GDN
          inference until the LOC cache supports streaming recurrent state
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A1 — HeteroLayerSymbols (mirrors Megatron Symbols + device hints)
# ---------------------------------------------------------------------------

class HeteroLayerSymbols:
    """
    Mirrors Megatron's ``mamba_hybrid_layer_allocation.Symbols`` with the
    addition of per-symbol device preferences for DES-LOC scheduling.

    Upstream change: Megatron added 'G' to VALID_LAYERS and updated
    VALID_LAYERS set.  Here we extend with DEVICE_PREFERENCE so that the
    DES-LOC runtime can build a DeviceAssignment for each layer without
    additional configuration.
    """

    MAMBA = "M"
    GDN = "G"
    ATTENTION = "*"
    MLP = "-"
    MOE = "E"
    PIPE = "|"
    MTP_SEPARATOR = "/"

    VALID_LAYERS = {MAMBA, GDN, ATTENTION, MLP, MOE}

    # DES-LOC addition: preferred compute device per layer symbol.
    # Values are torch.device strings; resolved at runtime against
    # available devices via DeviceRegistry.
    DEVICE_PREFERENCE: Dict[str, str] = {
        MAMBA: "a6000",       # bandwidth-bound selective scan
        GDN: "h100",          # matmul-heavy delta-rule recurrence
        ATTENTION: "h100",    # large KV, flash-attn2
        MLP: "a6000",         # dense GEMM fills A6000 well
        MOE: "cpu",           # expert offload to CPU DRAM
    }


# ---------------------------------------------------------------------------
# A2 — DeviceAssignment + DeviceRegistry
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DeviceAssignment:
    """
    Carries the resolved torch.device for a single layer and metadata
    needed by DES-LOC's execution scheduler.

    Fields
    ------
    layer_idx : int
        Global (pipeline-absolute) layer index.
    layer_symbol : str
        One of HeteroLayerSymbols.VALID_LAYERS.
    device : torch.device
        Resolved compute device for this layer.
    loc_cache_key : str
        Key into LocalityCache for this layer's recurrent state (if any).
        Empty string for stateless layers (MLP, MoE, pure attention w/o KV).
    pcie_transfer_bytes_estimate : int
        Estimated PCIe bytes per forward step for state I/O. Used by the
        scheduler to pipeline PCIe transfers with compute.
    """
    layer_idx: int
    layer_symbol: str
    device: torch.device
    loc_cache_key: str = ""
    pcie_transfer_bytes_estimate: int = 0


class DeviceRegistry:
    """
    Discovers available GPUs and maps logical names ('h100', 'a6000', 'cpu')
    to concrete torch.device instances.

    DES-LOC adaptation: upstream Megatron assumes homogeneous GPU ranks.
    Here we inspect device capabilities at init time and build a stable
    mapping used throughout the lifetime of the process.
    """

    # SM capability thresholds for classification
    _SM90_MIN = 90   # H100
    _SM86_MIN = 86   # A6000 / RTX 3090

    def __init__(self) -> None:
        self._logical_map: Dict[str, torch.device] = {}
        self._build_map()

    def _build_map(self) -> None:
        h100_devs: List[int] = []
        a6000_devs: List[int] = []

        if not torch.cuda.is_available():
            logger.warning("CUDA not available; all layers will run on CPU.")
            self._logical_map = {
                "h100": torch.device("cpu"),
                "a6000": torch.device("cpu"),
                "cpu": torch.device("cpu"),
            }
            return

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            sm = props.major * 10 + props.minor
            vram_gb = props.total_memory / (1024 ** 3)
            logger.info(
                "GPU %d: %s  SM%d  %.1f GB",
                i, props.name, sm, vram_gb,
            )
            if sm >= self._SM90_MIN:
                h100_devs.append(i)
            elif sm >= self._SM86_MIN:
                a6000_devs.append(i)

        self._logical_map["cpu"] = torch.device("cpu")

        if h100_devs:
            self._logical_map["h100"] = torch.device(f"cuda:{h100_devs[0]}")
            logger.info("H100 mapped to cuda:%d", h100_devs[0])
        else:
            fallback = torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
            self._logical_map["h100"] = fallback
            logger.warning("No SM90 device found; H100 preference → %s", fallback)

        if a6000_devs:
            # Primary A6000 is rank 0 of that tier; secondary used for tensor-parallel
            self._logical_map["a6000"] = torch.device(f"cuda:{a6000_devs[0]}")
            self._logical_map["a6000_secondary"] = (
                torch.device(f"cuda:{a6000_devs[1]}") if len(a6000_devs) > 1
                else self._logical_map["a6000"]
            )
            logger.info(
                "A6000 mapped to cuda:%d (secondary: %s)",
                a6000_devs[0], self._logical_map["a6000_secondary"],
            )
        else:
            fallback = self._logical_map.get("h100", torch.device("cpu"))
            self._logical_map["a6000"] = fallback
            self._logical_map["a6000_secondary"] = fallback
            logger.warning("No SM86 device found; A6000 preference → %s", fallback)

    def resolve(self, logical_name: str) -> torch.device:
        if logical_name not in self._logical_map:
            raise KeyError(
                f"Unknown logical device '{logical_name}'. "
                f"Available: {list(self._logical_map.keys())}"
            )
        return self._logical_map[logical_name]

    @property
    def h100(self) -> torch.device:
        return self._logical_map["h100"]

    @property
    def a6000(self) -> torch.device:
        return self._logical_map["a6000"]

    @property
    def a6000_secondary(self) -> torch.device:
        return self._logical_map["a6000_secondary"]

    @property
    def cpu(self) -> torch.device:
        return self._logical_map["cpu"]


# ---------------------------------------------------------------------------
# A3 — GDNFLOPsEstimator (extends Megatron gdn_layer_flops with PCIe cost)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GDNLayerConfig:
    """
    Hyperparameters for a single GDN layer, matching the parameters added
    in Megatron's training.py gdn_layer_flops() helper.

    Upstream defaults are preserved as field defaults here.
    """
    hidden_size: int = 4096
    qk_head_dim: int = 128
    v_head_dim: int = 128
    num_qk_heads: int = 16
    num_v_heads: int = 32
    conv_kernel_dim: int = 4
    dtype_bytes: int = 2          # BF16
    # DES-LOC addition: PCIe bandwidth in bytes/sec (measured, not assumed)
    pcie_bandwidth_bps: float = 16e9  # PCIe 4.0 x16 practical ~16 GB/s


class GDNFLOPsEstimator:
    """
    Replicates Megatron's ``gdn_layer_flops()`` inner function from
    ``megatron/training/training.py`` (commit 8f7fbe78) and adds a DES-LOC
    extension that accounts for PCIe transfer overhead of the GDN recurrent
    state matrix S ∈ R^{B × n_v_heads × v_head_dim × v_head_dim}.

    Upstream formula (verbatim from diff, line-by-line):
      flops = 2 * B * T * (
          hidden_size * (2*qk_dim + 2*v_dim + 2*num_v_heads)   # in_proj
          + conv_kernel_dim * (2*qk_dim + v_dim)                 # conv1d
          + num_v_heads * v_head_dim**2 * 4                      # delta rule
          + hidden_size * v_dim                                   # out_proj
      )

    DES-LOC addition: state_transfer_seconds() estimates how long PCIe
    transfer of the recurrent state occupies the bus, so the scheduler can
    overlap it with the previous layer's compute.
    """

    def __init__(self, cfg: GDNLayerConfig) -> None:
        self.cfg = cfg
        self._qk_dim = cfg.qk_head_dim * cfg.num_qk_heads
        self._v_dim = cfg.v_head_dim * cfg.num_v_heads

    def compute_flops(self, batch_size: int, seq_len: int) -> int:
        """Return forward-pass FLOPs (not counting backward × 3 multiplier)."""
        c = self.cfg
        qk_dim, v_dim = self._qk_dim, self._v_dim
        inner = (
            c.hidden_size * (2 * qk_dim + 2 * v_dim + 2 * c.num_v_heads)
            + c.conv_kernel_dim * (2 * qk_dim + v_dim)
            + c.num_v_heads * (c.v_head_dim ** 2) * 4
            + c.hidden_size * v_dim
        )
        return 2 * batch_size * seq_len * inner

    def recurrent_state_bytes(self, batch_size: int) -> int:
        """
        DES-LOC: size of S matrix that must travel over PCIe each step.
        S ∈ R^{B × n_v_heads × v_head_dim × v_head_dim}
        """
        c = self.cfg
        numel = batch_size * c.num_v_heads * c.v_head_dim * c.v_head_dim
        return numel * c.dtype_bytes

    def state_transfer_seconds(self, batch_size: int) -> float:
        """Estimated seconds for bidirectional PCIe transfer of S (read + write)."""
        return 2.0 * self.recurrent_state_bytes(batch_size) / self.cfg.pcie_bandwidth_bps

    def arithmetic_intensity(self, batch_size: int, seq_len: int) -> float:
        """FLOPs / byte — useful for roofline analysis of the DES-LOC pipeline."""
        flops = self.compute_flops(batch_size, seq_len)
        # bytes = weights (static, amortised) + state I/O (dynamic)
        state_bytes = self.recurrent_state_bytes(batch_size)
        return flops / max(state_bytes, 1)


# ---------------------------------------------------------------------------
# A4 — LocalityCache: shared CPU DRAM buffer for recurrent states
# ---------------------------------------------------------------------------

class LocalityCache:
    """
    The "Shared LOcality Cache" (LOC) in DES-LOC.

    Manages a pool of pinned CPU tensors that hold recurrent states for all
    stateful layer types (GDN's S matrix, Mamba's h vector).  Layers stream
    their state in/out over PCIe using async CUDA streams, so compute and
    transfer can overlap.

    Design rationale:
      - 1.5 TB CPU DRAM  »  GPU VRAM; we can keep *all* recurrent states
        resident in CPU and only copy the batch-slice needed per step.
      - Pinned memory is mandatory for async D2H/H2D transfers.
      - A separate CUDA stream per (layer, direction) lets the scheduler
        pipeline transfers with compute on the adjacent layer.

    Upstream analogy: Megatron's inference context tracks num_mamba_layers
    and preallocates GPU buffers.  DES-LOC moves these to CPU DRAM.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._store: Dict[str, torch.Tensor] = {}
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._max_entries = max_entries
        logger.info("LocalityCache initialised (max_entries=%d)", max_entries)

    def _stream_for(self, key: str, device: torch.device) -> torch.cuda.Stream:
        stream_key = f"{key}@{device}"
        if stream_key not in self._streams:
            if device.type == "cuda":
                self._streams[stream_key] = torch.cuda.Stream(device=device)
            else:
                self._streams[stream_key] = None  # type: ignore[assignment]
        return self._streams[stream_key]

    def allocate(self, key: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate (or return existing) pinned CPU tensor for *key*.
        Pinned memory is required for async PCIe transfers.
        """
        if key in self._store:
            existing = self._store[key]
            if existing.shape == torch.Size(shape) and existing.dtype == dtype:
                return existing
            logger.debug("LOC: reallocating %s  old=%s  new=%s", key, existing.shape, shape)

        if len(self._store) >= self._max_entries:
            raise RuntimeError(
                f"LocalityCache full ({self._max_entries} entries). "
                "Increase max_entries or reduce the number of stateful layers."
            )

        try:
            tensor = torch.zeros(shape, dtype=dtype, pin_memory=True)
        except RuntimeError:
            logger.warning(
                "pin_memory allocation failed for %s; falling back to pageable memory", key
            )
            tensor = torch.zeros(shape, dtype=dtype)

        self._store[key] = tensor
        logger.debug("LOC: allocated %s  shape=%s  dtype=%s", key, shape, dtype)
        return tensor

    def prefetch_to_device(
        self,
        key: str,
        device: torch.device,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """
        Async H2D copy of cached state.  Returns a device tensor; caller must
        synchronise the associated stream before using it.
        """
        if key not in self._store:
            raise KeyError(f"LOC: key '{key}' not allocated; call allocate() first.")
        cpu_tensor = self._store[key]
        stream = self._stream_for(key, device)
        if stream is not None:
            with torch.cuda.stream(stream):
                gpu_tensor = cpu_tensor.to(device, non_blocking=non_blocking)
        else:
            gpu_tensor = cpu_tensor.to(device)
        return gpu_tensor

    def writeback_from_device(
        self,
        key: str,
        gpu_tensor: torch.Tensor,
        non_blocking: bool = True,
    ) -> None:
        """
        Async D2H copy: write updated state back to CPU cache.
        """
        if key not in self._store:
            raise KeyError(f"LOC: key '{key}' not allocated.")
        cpu_tensor = self._store[key]
        stream = self._stream_for(key, gpu_tensor.device)
        if stream is not None:
            with torch.cuda.stream(stream):
                cpu_tensor.copy_(gpu_tensor, non_blocking=non_blocking)
        else:
            cpu_tensor.copy_(gpu_tensor)

    def synchronize(self, key: str, device: torch.device) -> None:
        """Block until all pending transfers for *key* on *device* are done."""
        stream = self._stream_for(key, device)
        if stream is not None:
            stream.synchronize()

    def evict(self, key: str) -> None:
        """Remove entry from cache (frees pinned memory on next GC cycle)."""
        if key in self._store:
            del self._store[key]
            logger.debug("LOC: evicted %s", key)

    @property
    def allocated_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._store.values())

    def summary(self) -> str:
        gb = self.allocated_bytes / (1024 ** 3)
        return f"LocalityCache: {len(self._store)} entries, {gb:.3f} GB pinned"


class GDNRecurrentState:
    """
    Manages the GDN delta-rule state matrix
    S ∈ R^{B × n_v_heads × v_head_dim × v_head_dim}
    in the LocalityCache.

    DES-LOC adaptation: upstream Megatron raises NotImplementedError for GDN
    inference because recurrent state management was not implemented.  This
    class provides the missing infrastructure for the LOC-based runtime,
    though full autoregressive inference support is still gated behind
    ``DESCLOC_GDN_INFERENCE_ENABLED`` (default: False) matching Megatron's
    conservative stance.
    """

    INFERENCE_ENABLED = False  # mirrors Megatron's NotImplementedError guard

    def __init__(
        self,
        layer_idx: int,
        batch_size: int,
        num_v_heads: int,
        v_head_dim: int,
        dtype: torch.dtype,
        loc_cache: LocalityCache,
    ) -> None:
        self.layer_idx = layer_idx
        self.key = f"gdn_state_layer{layer_idx}"
        self._shape = (batch_size, num_v_heads, v_head_dim, v_head_dim)
        self._dtype = dtype
        self._cache = loc_cache
        loc_cache.allocate(self.key, self._shape, dtype)
        logger.debug(
            "GDNRecurrentState: layer %d  shape=%s  key=%s",
            layer_idx, self._shape, self.key,
        )

    def load(self, device: torch.device) -> torch.Tensor:
        """Stream state from CPU DRAM to *device* (async)."""
        if not self.INFERENCE_ENABLED and not torch.is_grad_enabled():
            # During inference (no grad), raise analogous to Megatron's guard.
            raise NotImplementedError(
                "GDN layers are not supported for inference in DES-LOC. "
                "Set GDNRecurrentState.INFERENCE_ENABLED = True only after "
                "verifying LOC streaming correctness for your workload."
            )
        return self._cache.prefetch_to_device(self.key, device)

    def save(self, updated: torch.Tensor) -> None:
        """Write updated state back to CPU DRAM (async D2H)."""
        self._cache.writeback_from_device(self.key, updated)

    def sync(self, device: torch.device) -> None:
        self._cache.synchronize(self.key, device)

    @property
    def bytes(self) -> int:
        numel = math.prod(self._shape)
        return numel * torch._utils._element_size(self._dtype)


class MambaHiddenState:
    """
    Analogous to GDNRecurrentState but for Mamba's SSM hidden state
    h ∈ R^{B × n_heads × head_dim × state_dim}.

    Kept alongside GDNRecurrentState so the LOC cache API is consistent
    across all stateful layer types in a hybrid model.
    """

    def __init__(
        self,
        layer_idx: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        state_dim: int,
        dtype: torch.dtype,
        loc_cache: LocalityCache,
    ) -> None:
        self.layer_idx = layer_idx
        self.key = f"mamba_state_layer{layer_idx}"
        self._shape = (batch_size, num_heads, head_dim, state_dim)
        self._dtype = dtype
        self._cache = loc_cache
        loc_cache.allocate(self.key, self._shape, dtype)
        logger.debug(
            "MambaHiddenState: layer %d  shape=%s  key=%s",
            layer_idx, self._shape, self.key,
        )

    def load(self, device: torch.device) -> torch.Tensor:
        return self._cache.prefetch_to_device(self.key, device)

    def save(self, updated: torch.Tensor) -> None:
        self._cache.writeback_from_device(self.key, updated)

    def sync(self, device: torch.device) -> None:
        self._cache.synchronize(self.key, device)


# ---------------------------------------------------------------------------
# A2 cont. — HeteroHybridPattern: parse layer pattern → DeviceAssignments
# ---------------------------------------------------------------------------

def parse_hetero_hybrid_pattern(
    pattern: str,
    device_registry: DeviceRegistry,
    gdn_cfg: Optional[GDNLayerConfig] = None,
    batch_size: int = 1,
) -> List[DeviceAssignment]:
    """
    Parse a Megatron-style hybrid layer pattern string and return a list of
    :class:`DeviceAssignment` objects, one per layer (excluding pipe '|' and
    MTP '/' separators).

    Mirrors the logic of Megatron's ``validate_segment_layers()`` and
    ``get_layer_maps_from_layer_type_list()`` (updated in commit 8f7fbe78 to
    handle 'G').

    DES-LOC additions:
      - Each assignment carries the resolved torch.device from DeviceRegistry.
      - Stateful layers (G, M) get a ``loc_cache_key`` and an estimate of
        PCIe transfer bytes per forward step.
      - The MTP sub-pattern (after '/') is parsed but currently mapped to the
        same device preferences as main decoder layers.

    Parameters
    ----------
    pattern :
        E.g. ``"MG*-|MG*-"`` or ``"MGMG/GG/GG"``.
    device_registry :
        Resolved hardware map.
    gdn_cfg :
        GDN hyperparameters for PCIe cost estimation.  Defaults to
        :class:`GDNLayerConfig` with ``hidden_size=4096``.
    batch_size :
        Used only for PCIe cost estimation.

    Returns
    -------
    List[DeviceAssignment]
        In global layer order (pipeline stage boundaries stripped).
    """
    if gdn_cfg is None:
        gdn_cfg = GDNLayerConfig()

    estimator = GDNFLOPsEstimator(gdn_cfg)
    sym = HeteroLayerSymbols

    # Split off MTP sub-pattern
    parts = pattern.split(sym.MTP_SEPARATOR)
    main_pattern = parts[0]
    mtp_reps = parts[1:]  # repeated patterns

    # Strip pipe separators, collect raw layer symbols
    raw_layers: List[str] = []
    for ch in main_pattern:
        if ch == sym.PIPE:
            continue
        if ch not in sym.VALID_LAYERS:
            raise ValueError(
                f"Invalid layer symbol '{ch}' in pattern '{pattern}'. "
                f"Valid: {sym.VALID_LAYERS}"
            )
        raw_layers.append(ch)

    # MTP layers: each segment in mtp_reps is repeated once per occurrence
    mtp_depth = len(mtp_reps)
    for rep_pattern in mtp_reps:
        for ch in rep_pattern:
            if ch == sym.PIPE:
                continue
            if ch not in sym.VALID_LAYERS:
                raise ValueError(
                    f"Invalid MTP symbol '{ch}' in pattern '{pattern}'."
                )
            raw_layers.append(ch)

    logger.info(
        "parse_hetero_hybrid_pattern: %d total layers  (main=%d, mtp_depth=%d)",
        len(raw_layers), len(raw_layers) - sum(len(r) for r in mtp_reps), mtp_depth,
    )

    # Track per-type layer index (mirrors Megatron's layer_maps dict)
    type_counter: Dict[str, int] = {s: 0 for s in sym.VALID_LAYERS}
    assignments: List[DeviceAssignment] = []

    for global_idx, layer_sym in enumerate(raw_layers):
        logical_dev = sym.DEVICE_PREFERENCE[layer_sym]
        device = device_registry.resolve(logical_dev)

        # loc_cache_key for stateful layers
        loc_key = ""
        pcie_bytes = 0
        if layer_sym == sym.GDN:
            loc_key = f"gdn_state_layer{global_idx}"
            pcie_bytes = estimator.recurrent_state_bytes(batch_size) * 2  # R+W
        elif layer_sym == sym.MAMBA:
            # Rough estimate: Mamba state ~ B * n_heads * head_dim * state_dim * 2 bytes
            # Using GDN config fields as proxies (user should pass MambaConfig separately)
            mamba_state_numel = (
                batch_size * gdn_cfg.num_v_heads * gdn_cfg.qk_head_dim * gdn_cfg.v_head_dim
            )
            pcie_bytes = mamba_state_numel * gdn_cfg.dtype_bytes * 2
            loc_key = f"mamba_state_layer{global_idx}"

        assignments.append(DeviceAssignment(
            layer_idx=global_idx,
            layer_symbol=layer_sym,
            device=device,
            loc_cache_key=loc_key,
            pcie_transfer_bytes_estimate=pcie_bytes,
        ))
        type_counter[layer_sym] += 1

    counts_str = "  ".join(f"{s}={type_counter[s]}" for s in sym.VALID_LAYERS)
    logger.info("Layer counts: %s", counts_str)
    return assignments


def get_hetero_layer_counts(pattern: str) -> Dict[str, int]:
    """
    Convenience wrapper matching Megatron's ``get_hybrid_layer_counts()``
    signature (updated in commit 8f7fbe78 to include GDN 'G').

    Returns dict with keys from HeteroLayerSymbols.VALID_LAYERS.
    Does not require a DeviceRegistry — purely symbolic counting.
    """
    sym = HeteroLayerSymbols
    counts: Dict[str, int] = {s: 0 for s in sym.VALID_LAYERS}

    # Split MTP
    parts = pattern.split(sym.MTP_SEPARATOR)
    all_chars: List[str] = []
    for part in parts:
        for ch in part:
            if ch != sym.PIPE and ch in sym.VALID_LAYERS:
                all_chars.append(ch)

    for ch in all_chars:
        counts[ch] += 1

    return counts


# ---------------------------------------------------------------------------
# A5 — HeteroGDNMambaStack: the DES-LOC replacement for MambaStack
# ---------------------------------------------------------------------------

class HeteroGDNLayer(nn.Module):
    """
    Placeholder GDN layer for DES-LOC.

    In production this wraps the real GatedDeltaNet (imported from
    deepspeed.ops.gated_delta_net or from the Megatron module via
    the adapter shim).  The placeholder allows the stack to be instantiated
    and tested without the full SSM dependency tree.

    Upstream: Megatron wraps GatedDeltaNet inside a TransformerLayer shell
    so it reuses the existing residual connection and LayerNorm infra.
    DES-LOC keeps the same shell approach but moves the layer to the device
    specified by its DeviceAssignment.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        device: torch.device,
        assignment: DeviceAssignment,
        loc_cache: LocalityCache,
        gdn_cfg: GDNLayerConfig,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.target_device = device
        self.assignment = assignment
        self.gdn_cfg = gdn_cfg
        self._loc_cache = loc_cache

        hidden_size = getattr(config, "hidden_size", gdn_cfg.hidden_size)

        # Minimal learnable projection placeholders
        # (replace with real GatedDeltaNet submodules in production)
        self.in_proj = nn.Linear(
            hidden_size,
            2 * gdn_cfg.num_qk_heads * gdn_cfg.qk_head_dim
            + 2 * gdn_cfg.num_v_heads * gdn_cfg.v_head_dim
            + 2 * gdn_cfg.num_v_heads,
            bias=False,
            device=device,
        )
        self.out_proj = nn.Linear(
            gdn_cfg.num_v_heads * gdn_cfg.v_head_dim,
            hidden_size,
            bias=False,
            device=device,
        )
        self.norm = nn.LayerNorm(hidden_size, device=device)

        # Register recurrent state in LOC cache
        # (batch_size unknown at init; allocate lazily on first forward)
        self._state: Optional[GDNRecurrentState] = None
        logger.debug(
            "HeteroGDNLayer %d  device=%s  loc_key=%s",
            layer_idx, device, assignment.loc_cache_key,
        )

    def _ensure_state(self, batch_size: int, dtype: torch.dtype) -> GDNRecurrentState:
        if self._state is None or self._state._shape[0] != batch_size:
            self._state = GDNRecurrentState(
                layer_idx=self.layer_idx,
                batch_size=batch_size,
                num_v_heads=self.gdn_cfg.num_v_heads,
                v_head_dim=self.gdn_cfg.v_head_dim,
                dtype=dtype,
                loc_cache=self._loc_cache,
            )
        return self._state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Simplified GDN forward.

        Production implementation would:
          1. Prefetch S from LOC cache (async H2D)
          2. Run GatedDeltaNet recurrence on target_device
          3. Write updated S back to LOC cache (async D2H)
          4. Return output hidden states

        DES-LOC pipeline point: steps 1 and 3 use async CUDA streams so
        the transfer can overlap with the adjacent layer's compute.
        """
        # Move input to this layer's device if needed
        if hidden_states.device != self.target_device:
            hidden_states = hidden_states.to(self.target_device)

        batch_size = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]

        # LOC: prefetch recurrent state
        state_obj = self._ensure_state(batch_size, hidden_states.dtype)

        # During training we gate the state load behind torch.is_grad_enabled()
        # to match Megatron's inference guard semantics
        try:
            _S = state_obj.load(self.target_device)
        except NotImplementedError:
            # GDN inference not yet supported — propagate as in Megatron
            raise

        # --- Simplified delta-rule pass (placeholder) ---
        # In production: full chunked GDN recurrence using _S
        residual = hidden_states
        x = self.norm(hidden_states)
        projected = self.in_proj(x)

        # Derive output via minimal path (not a faithful delta-rule; placeholder only)
        v_dim = self.gdn_cfg.num_v_heads * self.gdn_cfg.v_head_dim
        v_part = projected[..., :v_dim]
        out = self.out_proj(v_part)
        out = out + residual

        # LOC: write back updated state
        updated_S = _S  # placeholder: real code would compute new S
        state_obj.save(updated_S)

        return out


class HeteroMambaLayer(nn.Module):
    """
    Placeholder Mamba SSM layer for DES-LOC.

    Assigned to A6000 devices (bandwidth-bound selective scan).
    Manages MambaHiddenState in LocalityCache.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        device: torch.device,
        assignment: DeviceAssignment,
        loc_cache: LocalityCache,
        gdn_cfg: GDNLayerConfig,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.target_device = device
        self.assignment = assignment
        self._loc_cache = loc_cache
        self.gdn_cfg = gdn_cfg

        hidden_size = getattr(config, "hidden_size", gdn_cfg.hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.norm = nn.LayerNorm(hidden_size, device=device)
        self._state: Optional[MambaHiddenState] = None

    def _ensure_state(self, batch_size: int, dtype: torch.dtype) -> MambaHiddenState:
        if self._state is None or self._state._shape[0] != batch_size:
            self._state = MambaHiddenState(
                layer_idx=self.layer_idx,
                batch_size=batch_size,
                num_heads=self.gdn_cfg.num_qk_heads,
                head_dim=self.gdn_cfg.qk_head_dim,
                state_dim=self.gdn_cfg.v_head_dim,
                dtype=dtype,
                loc_cache=self._loc_cache,
            )
        return self._state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states.device != self.target_device:
            hidden_states = hidden_states.to(self.target_device)

        batch_size = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]
        state_obj = self._ensure_state(batch_size, hidden_states.dtype)
        _h = state_obj.load(self.target_device)

        residual = hidden_states
        out = self.proj(self.norm(hidden_states)) + residual
        state_obj.save(_h)
        return out


class HeteroAttentionLayer(nn.Module):
    """Placeholder attention layer, assigned to H100 in DES-LOC."""

    def __init__(self, config, layer_idx: int, device: torch.device) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.target_device = device
        hidden_size = getattr(config, "hidden_size", 4096)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.norm = nn.LayerNorm(hidden_size, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states.device != self.target_device:
            hidden_states = hidden_states.to(self.target_device)
        return self.proj(self.norm(hidden_states)) + hidden_states


@dataclasses.dataclass
class HeteroStackConfig:
    """Configuration for HeteroGDNMambaStack."""
    hidden_size: int = 256
    layer_pattern: str = "GM*"
    batch_size: int = 1
    gdn_cfg: GDNLayerConfig = dataclasses.field(default_factory=GDNLayerConfig)
    loc_cache_max_entries: int = 512
    dtype: torch.dtype = torch.float32


class HeteroGDNMambaStack(nn.Module):
    """
    DES-LOC heterogeneous replacement for Megatron's ``MambaStack``.

    Megatron's MambaStack (commit 8f7fbe78) added an ``elif layer_type ==
    LayerSymbols.GDN`` branch to ``build_layers()``, routing 'G' symbols to
    a TransformerLayer wrapping GatedDeltaNet.

    DES-LOC extends this to:
      1. Parse the pattern into :class:`DeviceAssignment` objects.
      2. Instantiate each layer on its assigned device.
      3. Bind stateful layers (GDN, Mamba) to entries in :class:`LocalityCache`.
      4. Expose ``loc_cache`` for the DeepSpeed engine to manage PCIe scheduling.

    The forward pass moves hidden states between devices at layer boundaries
    (PCIe transfers).  In a production DES-LOC deployment the scheduler
    overlaps these transfers with adjacent layer compute via CUDA streams.
    """

    def __init__(
        self,
        config: HeteroStackConfig,
        device_registry: DeviceRegistry,
    ) -> None:
        super().__init__()
        self.config = config
        self.registry = device_registry

        self.loc_cache = LocalityCache(max_entries=config.loc_cache_max_entries)

        # Resolve config for GDN layers
        gdn_cfg = config.gdn_cfg
        gdn_cfg.hidden_size = config.hidden_size

        # Parse pattern → assignments
        self.assignments: List[DeviceAssignment] = parse_hetero_hybrid_pattern(
            pattern=config.layer_pattern,
            device_registry=device_registry,
            gdn_cfg=gdn_cfg,
            batch_size=config.batch_size,
        )

        # Build layers
        layers: List[nn.Module] = []
        sym = HeteroLayerSymbols
        for assignment in self.assignments:
            s = assignment.layer_symbol
            dev = assignment.device
            idx = assignment.layer_idx

            if s == sym.GDN:
                layer = HeteroGDNLayer(
                    config=config,
                    layer_idx=idx,
                    device=dev,
                    assignment=assignment,
                    loc_cache=self.loc_cache,
                    gdn_cfg=gdn_cfg,
                )
            elif s == sym.MAMBA:
                layer = HeteroMambaLayer(
                    config=config,
                    layer_idx=idx,
                    device=dev,
                    assignment=assignment,
                    loc_cache=self.loc_cache,
                    gdn_cfg=gdn_cfg,
                )
            elif s == sym.ATTENTION:
                layer = HeteroAttentionLayer(config=config, layer_idx=idx, device=dev)
            elif s in (sym.MLP, sym.MOE):
                # Placeholder: MLP on A6000, MoE on CPU
                hidden_size = config.hidden_size
                layer = nn.Sequential(
                    nn.LayerNorm(hidden_size, device=dev),
                    nn.Linear(hidden_size, hidden_size * 4, bias=False, device=dev),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size, bias=False, device=dev),
                )
                # Monkey-patch forward to handle device transfer
                _dev = dev
                _inner = layer
                class _MLPWrapper(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.inner = _inner
                        self.target_device = _dev
                    def forward(self, h, **kw):
                        if h.device != self.target_device:
                            h = h.to(self.target_device)
                        return self.inner(h) + h
                layer = _MLPWrapper()
            else:
                raise ValueError(f"Unknown layer symbol '{s}'")

            layers.append(layer)
            logger.debug(
                "Built layer %d  sym=%s  device=%s",
                idx, s, dev,
            )

        self.layers = nn.ModuleList(layers)
        logger.info(
            "HeteroGDNMambaStack: %d layers  pattern='%s'\n  %s",
            len(self.layers), config.layer_pattern, self.loc_cache.summary(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sequential forward pass across heterogeneous devices.

        Device transitions (PCIe) occur automatically via .to() at each
        layer boundary.  In production the DES-LOC scheduler replaces these
        with async pre-staged transfers using LOC prefetch streams.
        """
        for i, layer in enumerate(self.layers):
            assignment = self.assignments[i]
            target = assignment.device

            # Move hidden states to the layer's device if needed
            if hidden_states.device != target:
                logger.debug(
                    "Layer %d (%s): PCIe transfer %s → %s  (~%d bytes)",
                    i, assignment.layer_symbol,
                    hidden_states.device, target,
                    hidden_states.numel() * hidden_states.element_size(),
                )
                hidden_states = hidden_states.to(target)

            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        return hidden_states

    def loc_cache_summary(self) -> str:
        return self.loc_cache.summary()

    def total_pcie_bytes_per_step(self) -> int:
        """Sum of estimated PCIe bytes across all stateful layers per forward step."""
        return sum(a.pcie_transfer_bytes_estimate for a in self.assignments)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print("=== DES-LOC HeteroGDNMamba smoke test ===\n")

    # 1. Layer count parsing (mirrors Megatron get_hybrid_layer_counts tests)
    counts = get_hetero_layer_counts("GMGM")
    assert counts["G"] == 2 and counts["M"] == 2, f"count fail: {counts}"

    counts2 = get_hetero_layer_counts("G*GM*")
    assert counts2["G"] == 2 and counts2["*"] == 2 and counts2["M"] == 1, f"count2 fail: {counts2}"

    # 2. FLOPs estimator
    cfg = GDNLayerConfig(hidden_size=256, num_qk_heads=4, num_v_heads=8,
                         qk_head_dim=32, v_head_dim=32)
    est = GDNFLOPsEstimator(cfg)
    flops = est.compute_flops(batch_size=2, seq_len=64)
    assert flops > 0, "FLOPs must be positive"
    state_bytes = est.recurrent_state_bytes(batch_size=2)
    assert state_bytes == 2 * 8 * 32 * 32 * 2, f"state_bytes mismatch: {state_bytes}"

    # 3. DeviceRegistry + pattern parse (CPU fallback for CI)
    registry = DeviceRegistry()
    assignments = parse_hetero_hybrid_pattern(
        pattern="GM*",
        device_registry=registry,
        gdn_cfg=cfg,
        batch_size=2,
    )
    assert len(assignments) == 3, f"expected 3 assignments, got {len(assignments)}"
    assert assignments[0].layer_symbol == "G"
    assert assignments[1].layer_symbol == "M"
    assert assignments[2].layer_symbol == "*"

    # 4. LocalityCache allocation
    loc = LocalityCache(max_entries=16)
    t = loc.allocate("test_key", (2, 8, 32, 32), torch.float32)
    assert t.shape == (2, 8, 32, 32), f"shape mismatch: {t.shape}"

    # 5. HeteroGDNMambaStack forward (CPU devices)
    stack_cfg = HeteroStackConfig(
        hidden_size=256,
        layer_pattern="GM*",
        batch_size=2,
        gdn_cfg=cfg,
    )
    stack = HeteroGDNMambaStack(stack_cfg, registry)
    # Use whichever device the first layer landed on
    first_device = stack.assignments[0].device
    x = torch.randn(16, 2, 256, device=first_device)  # [seq, batch, hidden]
    GDNRecurrentState.INFERENCE_ENABLED = True  # enable for test
    out = stack(x)
    assert out.shape == x.shape or out.shape[2] == 256, f"output shape wrong: {out.shape}"

    print("\nAll smoke tests passed.")
    print(f"Total PCIe bytes/step estimate: {stack.total_pcie_bytes_per_step() / 1e6:.2f} MB")
    print(stack.loc_cache_summary())
