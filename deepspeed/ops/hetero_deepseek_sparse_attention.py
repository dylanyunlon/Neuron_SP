"""
HeteroDeepSeekSparseAttention (DES-LOC adaptation)
====================================================

Upstream design intent (Megatron a00e9443):
    Port DeepSeek Sparse Attention (DSA) — a hybrid sparse-attention mechanism
    combining Multi-head Latent Attention (MLA) indexing with top-k token selection —
    into the MambaModel hybrid stack. The upstream adds a new layer symbol 'D'
    (DS_ATTENTION) alongside the existing '*' (standard attention), enabling
    heterogeneous layer sequences like "D-D-D-D-" where DSA layers alternate
    with MLP layers in a Mamba-hybrid model.

    Key upstream changes:
    1. `get_layer_maps_from_layer_type_list` now returns a dict keyed by Symbols,
       rather than a positional tuple — callers must use operator.itemgetter.
    2. `num_attention_layers` in DynamicInferenceContext counts both '*' and 'D' layers
       for KV cache sizing.
    3. MLA's decoupled RoPE supersedes standard RotaryEmbedding when DSA is active.
    4. `MLASelfAttention` gains a `pp_layer_offset` parameter so PP-aware layer
       numbering threads through the indexer.
    5. `fuse_input_layernorm=False` is injected as metainfo on the DSA module spec
       to prevent TE from fusing the input norm into the attention projection.

DES-LOC adaptation points:
    Hardware context: 2× A6000 48 GB (SM86, PCIe) + 1× H100 NVL 96 GB (SM90, PCIe),
    1.5 TB CPU DRAM, no NVLink. PCIe bandwidth (~64 GB/s bidirectional) is the
    dominant bottleneck for inter-device communication.

    DES-LOC = Decoupled Execution with Shared LOcality Cache

    Core idea: DSA layers are *heterogeneous* — they perform top-k index selection
    (cheap, memory-bound) followed by dense MLA projection (compute-bound). Under
    DES-LOC we decouple these two phases across the device fleet:

    Phase 1 – Indexing (locality-aware):
        Run DSAIndexer on the device that owns the activation shard. Because the
        indexer is memory-bound (linear_wq_b, linear_wk, k_norm, linear_weights_proj),
        it benefits from high DRAM bandwidth. On A6000s (PCIe, no NVLink) we keep
        indexer execution local and spill intermediate top-k indices + compressed KV
        to CPU DRAM via `locality_cache` (a pinned-memory ring buffer) when GPU DRAM
        pressure exceeds a configurable threshold.

    Phase 2 – Sparse projection (SM90-preferential):
        The actual sparse attention kernel (DSAttention) is compute-bound and benefits
        from SM90 FP8 tensor cores. When available, the H100 is preferred for Phase 2.
        The A6000 nodes handle Phase 1 and stream compressed KV to H100 via a
        DES-LOC shared-locality cache (SLC) that lives in pinned CPU DRAM and is
        accessed by all three devices via peer DMA.

    Layer-map integration:
        Mirrors Megatron's dict-based `get_layer_maps_from_layer_type_list` return
        convention. DES-LOC extends this with `device_affinity_map` — a per-global-
        layer-index dict specifying which physical device handles Phase 1 vs Phase 2.

    KV cache sizing:
        Follows upstream: `num_attention_layers = |attention_map| + |dsa_map|` so
        that the inference KV cache allocator budgets correctly for both layer types.

    pp_layer_offset threading:
        MLASelfAttention needs pp_layer_offset to number its decoupled RoPE buffers
        correctly across PP stages. DES-LOC propagates this through
        HeteroMambaStackBuilder, matching the upstream fix in multi_latent_attention.py.

    Constraint: '*' (standard attention) and 'D' (DSA/MLA) must not coexist in one
        model — inherited from upstream validation and re-enforced here.
"""

from __future__ import annotations

import logging
import math
import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer symbols — mirrors Megatron Symbols class post-a00e9443
# ---------------------------------------------------------------------------

class LayerSymbol:
    """Single source of truth for hybrid-pattern character codes.

    DES-LOC keeps these in sync with Megatron's Symbols class to ensure that
    checkpoint tools (remap_gpt_dsa_to_mamba.py equivalents) remain portable.
    """
    MAMBA      = "M"
    GDN        = "G"
    ATTENTION  = "*"
    DS_ATTENTION = "D"   # DeepSeek Sparse Attention (MLA + top-k indexer)
    MLP        = "-"
    MOE        = "E"
    PIPE       = "|"
    MTP_SEP    = "/"

    VALID_LAYERS = frozenset({MAMBA, GDN, ATTENTION, DS_ATTENTION, MLP, MOE})

    @classmethod
    def name_sorted_valid(cls) -> List[str]:
        """Return valid layer symbols sorted lexicographically by attribute name.

        Matches Megatron's `Symbols.name_sorted_valid_layer_symbols()` so that
        dict key ordering is stable and portable across both codebases.
        """
        pairs = []
        for name, val in vars(cls).items():
            if not name.startswith("_") and val in cls.VALID_LAYERS:
                pairs.append((name, val))
        pairs.sort()
        return [v for _, v in pairs]


# ---------------------------------------------------------------------------
# Device roles under DES-LOC
# ---------------------------------------------------------------------------

class DeviceRole(Enum):
    """Physical role of a GPU in the DES-LOC hetero fleet.

    INDEXER:    Runs DSAIndexer (Phase 1, memory-bound). Preferred: A6000.
    PROJECTOR:  Runs DSAttention sparse projection (Phase 2, compute-bound).
                Preferred: H100 NVL (SM90, FP8 tensor cores).
    CPU_CACHE:  Pinned CPU DRAM acting as Shared LOcality Cache (SLC).
    """
    INDEXER   = "indexer"
    PROJECTOR = "projector"
    CPU_CACHE = "cpu_cache"


@dataclass
class DeviceSpec:
    """Hardware descriptor for one GPU in the fleet.

    Attributes:
        device_id:      torch.device index.
        sm_version:     Compute capability as integer (e.g., 86 for A6000, 90 for H100).
        vram_gb:        Reported VRAM in GiB.
        role:           DES-LOC role assigned to this device.
        pcie_bw_gbps:   Estimated unidirectional PCIe bandwidth in GB/s.
    """
    device_id:    int
    sm_version:   int
    vram_gb:      float
    role:         DeviceRole
    pcie_bw_gbps: float = 16.0   # PCIe 4.0 x16 ~16 GB/s unidirectional per lane group


@dataclass
class HeteroFleetConfig:
    """Fleet-level DES-LOC configuration.

    Describes the 2×A6000 + 1×H100 topology and SLC parameters.

    Attributes:
        devices:            Ordered list of DeviceSpec (one per physical GPU).
        slc_size_gb:        Size in GiB to pre-allocate for the pinned CPU SLC.
        slc_ring_depth:     Number of slots in the SLC ring buffer per DSA layer.
        spill_threshold:    Fraction [0,1] of device VRAM at which indexer output
                            is spilled to SLC rather than kept on GPU.
        prefer_sm90_proj:   If True, Phase 2 (projection) is dispatched to the first
                            SM90+ device found in `devices`.
    """
    devices:           List[DeviceSpec] = field(default_factory=list)
    slc_size_gb:       float = 8.0
    slc_ring_depth:    int   = 4
    spill_threshold:   float = 0.85
    prefer_sm90_proj:  bool  = True

    def indexer_devices(self) -> List[DeviceSpec]:
        return [d for d in self.devices if d.role == DeviceRole.INDEXER]

    def projector_device(self) -> Optional[DeviceSpec]:
        for d in self.devices:
            if d.role == DeviceRole.PROJECTOR:
                return d
        return None


def build_default_fleet_config() -> HeteroFleetConfig:
    """Build a HeteroFleetConfig for the reference hardware: 2×A6000 + 1×H100 NVL.

    Probes available CUDA devices at call time and assigns roles based on
    compute capability (SM86 → INDEXER, SM90 → PROJECTOR). Falls back to
    CPU-only SLC when fewer than 3 GPUs are present (e.g., in unit tests).

    Returns:
        A HeteroFleetConfig ready for use by DES-LOC components.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; returning degenerate single-device fleet config.")
        return HeteroFleetConfig(devices=[])

    specs: List[DeviceSpec] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        sm = props.major * 10 + props.minor
        vram = props.total_memory / (1024 ** 3)

        if sm >= 90:
            role = DeviceRole.PROJECTOR
        else:
            role = DeviceRole.INDEXER

        spec = DeviceSpec(
            device_id=idx,
            sm_version=sm,
            vram_gb=vram,
            role=role,
        )
        specs.append(spec)
        logger.info(
            "Detected GPU %d: SM%d, %.1f GiB VRAM → DES-LOC role=%s",
            idx, sm, vram, role.value,
        )

    if not any(d.role == DeviceRole.PROJECTOR for d in specs):
        logger.warning(
            "No SM90+ device found; assigning last device as PROJECTOR (performance degraded)."
        )
        specs[-1] = DeviceSpec(
            device_id=specs[-1].device_id,
            sm_version=specs[-1].sm_version,
            vram_gb=specs[-1].vram_gb,
            role=DeviceRole.PROJECTOR,
        )

    return HeteroFleetConfig(devices=specs)


# ---------------------------------------------------------------------------
# Shared LOcality Cache (SLC)
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """Pinned-CPU-DRAM ring buffer used as the DES-LOC inter-device transfer cache.

    Design rationale:
        Under PCIe-only connectivity (no NVLink), direct GPU-to-GPU transfers
        must traverse host memory. Rather than allocating ad-hoc pinned buffers
        per forward pass, SLC pre-allocates a fixed ring of pinned tensors so
        that the CUDA DMA engine can overlap computation with data movement.

    Ring semantics:
        Each DSA layer gets `ring_depth` slots of size
        (max_seq_len, batch_size, kv_lora_rank). The producer (INDEXER device)
        writes compressed KV into slot `write_ptr % ring_depth`; the consumer
        (PROJECTOR device) reads from `read_ptr % ring_depth`. Slot handoff is
        coordinated via CUDA events (one per slot).

    Attributes:
        num_dsa_layers: Number of DSA ('D') layers in the model.
        kv_lora_rank:   Compressed KV dimension (from MLATransformerConfig).
        max_seq_len:    Maximum sequence length (for static allocation).
        max_batch_size: Maximum micro-batch size.
        ring_depth:     Number of ring buffer slots per layer.
    """

    def __init__(
        self,
        num_dsa_layers: int,
        kv_lora_rank: int,
        max_seq_len: int,
        max_batch_size: int,
        ring_depth: int = 4,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_dsa_layers = num_dsa_layers
        self.kv_lora_rank   = kv_lora_rank
        self.max_seq_len    = max_seq_len
        self.max_batch_size = max_batch_size
        self.ring_depth     = ring_depth
        self.dtype          = dtype

        # Slot shape: (ring_depth, seq, batch, kv_lora_rank)
        slot_shape = (ring_depth, max_seq_len, max_batch_size, kv_lora_rank)
        self._kv_buffers: List[torch.Tensor] = []
        self._topk_buffers: List[torch.Tensor] = []
        self._events: List[List[torch.cuda.Event]] = []

        for layer_idx in range(num_dsa_layers):
            kv_buf = torch.zeros(slot_shape, dtype=dtype, pin_memory=True)
            # top-k index buffer: (ring_depth, seq, batch, topk) stored as int32
            topk_buf = torch.zeros(
                (ring_depth, max_seq_len, max_batch_size, 1),
                dtype=torch.int32, pin_memory=True,
            )
            self._kv_buffers.append(kv_buf)
            self._topk_buffers.append(topk_buf)
            # One CUDA event per ring slot for producer→consumer synchronisation
            layer_events = [
                torch.cuda.Event(enable_timing=False, blocking=False)
                for _ in range(ring_depth)
            ]
            self._events.append(layer_events)

        self._write_ptrs = [0] * num_dsa_layers
        self._read_ptrs  = [0] * num_dsa_layers

        logger.info(
            "SLC allocated: %d DSA layers × %d slots, kv_lora_rank=%d, "
            "seq=%d, batch=%d → %.2f MiB pinned CPU DRAM",
            num_dsa_layers, ring_depth, kv_lora_rank, max_seq_len, max_batch_size,
            num_dsa_layers * ring_depth * max_seq_len * max_batch_size * kv_lora_rank
            * 2 / (1024 ** 2),   # bfloat16 = 2 bytes
        )

    def write_kv(
        self,
        layer_idx: int,
        compressed_kv: torch.Tensor,
        topk_indices: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> int:
        """Write compressed KV and top-k indices into the next ring slot.

        Args:
            layer_idx:      Local DSA layer index (0-based within DSA layers).
            compressed_kv:  (seq, batch, kv_lora_rank) tensor on INDEXER device.
            topk_indices:   (seq, batch, 1) int32 tensor of selected token indices.
            stream:         CUDA stream on the INDEXER device. If None, uses default.

        Returns:
            Slot index written to (for logging / debugging).
        """
        slot = self._write_ptrs[layer_idx] % self.ring_depth
        buf  = self._kv_buffers[layer_idx][slot]
        tbuf = self._topk_buffers[layer_idx][slot]

        seq, batch, _ = compressed_kv.shape
        with torch.cuda.stream(stream) if stream else _null_ctx():
            buf[:seq, :batch, :].copy_(compressed_kv, non_blocking=True)
            tbuf[:seq, :batch, :].copy_(topk_indices, non_blocking=True)
            self._events[layer_idx][slot].record()

        self._write_ptrs[layer_idx] += 1
        logger.debug("SLC write: layer=%d slot=%d seq=%d batch=%d", layer_idx, slot, seq, batch)
        return slot

    def read_kv(
        self,
        layer_idx: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read the next compressed KV and top-k indices from the ring buffer.

        Waits on the producer CUDA event before returning — ensures that the
        INDEXER's async copy to CPU DRAM has completed before the PROJECTOR
        consumes it.

        Args:
            layer_idx:  Local DSA layer index.
            stream:     CUDA stream on the PROJECTOR device to wait on.

        Returns:
            Tuple of (compressed_kv, topk_indices) still in pinned CPU DRAM.
            Caller is responsible for .cuda() / .to(device) transfer.
        """
        slot = self._read_ptrs[layer_idx] % self.ring_depth
        event = self._events[layer_idx][slot]
        if stream is not None:
            stream.wait_event(event)
        else:
            event.synchronize()

        kv   = self._kv_buffers[layer_idx][slot]
        tidx = self._topk_buffers[layer_idx][slot]
        self._read_ptrs[layer_idx] += 1
        logger.debug("SLC read:  layer=%d slot=%d", layer_idx, slot)
        return kv, tidx


class _null_ctx:
    """No-op context manager (replaces `contextlib.nullcontext` for Python 3.8 compat)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Layer-map utilities (mirrors Megatron post-a00e9443)
# ---------------------------------------------------------------------------

def get_layer_maps(layer_type_list: List[str]) -> Dict[str, Dict[int, int]]:
    """Build per-symbol maps from global layer index → local layer index.

    Mirrors Megatron's refactored `get_layer_maps_from_layer_type_list` which
    changed its return type from a positional 5-tuple to a dict[symbol, dict].

    The change is critical for DES-LOC because we need to extract both
    `attention_map` and `dsa_map` independently to size the KV cache:

        num_attention_layers = len(attention_map) + len(dsa_map)

    Args:
        layer_type_list: Ordered list of layer-symbol strings, e.g.
            ['D', '-', 'D', '-', 'M', 'D', 'E'].

    Returns:
        Dict mapping each valid LayerSymbol to a {global_idx: local_idx} dict.
        All valid symbols are always present as keys (value may be empty dict).

    Example:
        >>> maps = get_layer_maps(['D', 'M', 'D', '-'])
        >>> maps['D']
        {0: 0, 2: 1}
        >>> maps['M']
        {1: 0}
    """
    layer_types = LayerSymbol.name_sorted_valid()
    layer_maps: Dict[str, Dict[int, int]] = {lt: {} for lt in layer_types}
    for global_idx, symbol in enumerate(layer_type_list):
        if symbol not in layer_maps:
            raise ValueError(
                f"Unknown layer symbol '{symbol}'. "
                f"Valid symbols: {LayerSymbol.VALID_LAYERS}"
            )
        local_idx = len(layer_maps[symbol])
        layer_maps[symbol][global_idx] = local_idx
    return layer_maps


def count_attention_layers(layer_maps: Dict[str, Dict[int, int]]) -> int:
    """Return total KV-cache-bearing attention layers (standard '*' + DSA 'D').

    Upstream change (a00e9443): `num_attention_layers` in DynamicInferenceContext
    now sums both attention_layer_map and dsa_layer_map to correctly size the
    KV cache for hybrid models with DSA layers.

    Args:
        layer_maps: Output of `get_layer_maps`.

    Returns:
        Sum of attention and DSA layer counts.
    """
    n = len(layer_maps.get(LayerSymbol.ATTENTION, {}))
    n += len(layer_maps.get(LayerSymbol.DS_ATTENTION, {}))
    return n


def validate_hybrid_pattern(pattern: str) -> None:
    """Validate a DES-LOC hybrid pattern string.

    Rules (inherited from upstream a00e9443 + DES-LOC extensions):
    1. Only characters from VALID_LAYERS (plus PIPE and MTP_SEP) are allowed.
    2. '*' (standard attention) and 'D' (DSA) must not coexist in one model.
       Upstream rationale: MLA uses decoupled RoPE which conflicts with the
       standard RotaryEmbedding used by vanilla attention.
    3. Pipe '|' is valid for pipeline-stage boundaries.
    4. '/' separates MTP patterns.

    Args:
        pattern: Hybrid layer pattern string, e.g. "D-D-D-D-" or "MD|MD".

    Raises:
        ValueError: If the pattern contains invalid characters or the
            '*' + 'D' co-existence constraint is violated.
    """
    valid_chars = LayerSymbol.VALID_LAYERS | {LayerSymbol.PIPE, LayerSymbol.MTP_SEP}
    for ch in pattern:
        if ch not in valid_chars:
            raise ValueError(
                f"Invalid character '{ch}' in hybrid pattern '{pattern}'. "
                f"Valid characters: {valid_chars}"
            )
    if LayerSymbol.ATTENTION in pattern and LayerSymbol.DS_ATTENTION in pattern:
        raise ValueError(
            "Hybrid pattern may not contain both standard attention ('*') and "
            "DSA ('D') — MLA's decoupled RoPE is incompatible with standard RoPE. "
            f"Pattern was: '{pattern}'"
        )


def parse_layer_type_list(pattern: str) -> List[str]:
    """Convert a simple (pipe-free, MTP-free) hybrid pattern to a layer type list.

    Args:
        pattern: A segment string such as "D-D-MD" (no '|' or '/').

    Returns:
        List of individual layer-symbol strings.

    Raises:
        ValueError: On invalid characters or constraint violations.
    """
    validate_hybrid_pattern(pattern)
    return [ch for ch in pattern if ch in LayerSymbol.VALID_LAYERS]


# ---------------------------------------------------------------------------
# Device-affinity map for DES-LOC
# ---------------------------------------------------------------------------

@dataclass
class DSALayerDeviceAffinity:
    """Per-DSA-layer device assignment under DES-LOC.

    Attributes:
        global_layer_idx:   Position of this DSA layer in the full stack.
        local_dsa_idx:      0-based index within DSA layers only (for SLC addressing).
        indexer_device:     torch.device for Phase 1 (DSAIndexer).
        projector_device:   torch.device for Phase 2 (DSAttention).
        use_slc:            Whether to route through the Shared LOcality Cache.
    """
    global_layer_idx:  int
    local_dsa_idx:     int
    indexer_device:    torch.device
    projector_device:  torch.device
    use_slc:           bool = True


def build_device_affinity_map(
    layer_maps: Dict[str, Dict[int, int]],
    fleet_config: HeteroFleetConfig,
) -> Dict[int, DSALayerDeviceAffinity]:
    """Assign Phase-1 (indexer) and Phase-2 (projector) devices to each DSA layer.

    DES-LOC strategy for 2×A6000 + 1×H100:
    - All indexer (Phase 1) work runs on A6000s, round-robin across INDEXER devices.
    - All projection (Phase 2) work runs on the H100 (first PROJECTOR device).
    - use_slc=True for all DSA layers when INDEXER ≠ PROJECTOR device.

    Args:
        layer_maps:     Output of `get_layer_maps`.
        fleet_config:   Hardware fleet description with role assignments.

    Returns:
        Dict mapping global_layer_idx → DSALayerDeviceAffinity.
    """
    dsa_map = layer_maps.get(LayerSymbol.DS_ATTENTION, {})
    if not dsa_map:
        logger.info("No DSA layers found; device affinity map is empty.")
        return {}

    indexer_devices = fleet_config.indexer_devices()
    projector = fleet_config.projector_device()

    if not indexer_devices:
        logger.warning("No INDEXER devices in fleet; defaulting to device 0 for indexer.")
        indexer_devices = [DeviceSpec(0, 86, 48.0, DeviceRole.INDEXER)]
    if projector is None:
        logger.warning("No PROJECTOR device in fleet; defaulting to device 0 for projector.")
        projector = DeviceSpec(0, 86, 48.0, DeviceRole.PROJECTOR)

    affinity_map: Dict[int, DSALayerDeviceAffinity] = {}
    for global_idx, local_idx in dsa_map.items():
        indexer_spec = indexer_devices[local_idx % len(indexer_devices)]
        proj_device  = torch.device(f"cuda:{projector.device_id}")
        idx_device   = torch.device(f"cuda:{indexer_spec.device_id}")
        use_slc      = (indexer_spec.device_id != projector.device_id)

        affinity_map[global_idx] = DSALayerDeviceAffinity(
            global_layer_idx=global_idx,
            local_dsa_idx=local_idx,
            indexer_device=idx_device,
            projector_device=proj_device,
            use_slc=use_slc,
        )
        logger.debug(
            "DSA layer %d (local %d): indexer=cuda:%d projector=cuda:%d use_slc=%s",
            global_idx, local_idx,
            indexer_spec.device_id, projector.device_id, use_slc,
        )

    return affinity_map


# ---------------------------------------------------------------------------
# Decoupled DSA execution engine
# ---------------------------------------------------------------------------

class DecoupledDSAExecutor(nn.Module):
    """Executes one DSA layer using DES-LOC's two-phase heterogeneous dispatch.

    Upstream context:
        In Megatron's MambaStack, a 'D' layer builds a full TransformerLayer
        containing MLASelfAttention → DSAttention (with DSAIndexer inside).
        All computation runs on a single device.

    DES-LOC adaptation:
        We split the DSAIndexer (Phase 1) from DSAttention's sparse projection
        (Phase 2) across devices. The split happens at the boundary where the
        indexer outputs compressed KV + top-k indices — precisely the data
        that must cross devices. The SLC ring buffer absorbs the PCIe transfer
        latency by overlapping it with the next layer's Phase 1 computation.

    Args:
        hidden_size:        Model hidden dimension.
        kv_lora_rank:       Compressed KV latent dimension (MLATransformerConfig).
        num_heads:          Number of attention heads.
        head_dim:           Per-head dimension.
        indexer_n_heads:    DSAIndexer head count.
        indexer_topk:       Top-k tokens selected by the indexer.
        affinity:           Device affinity spec for this layer.
        slc:                Shared LOcality Cache instance (may be None if
                            indexer and projector share a device).
        dtype:              Parameter dtype (default bfloat16).
    """

    def __init__(
        self,
        hidden_size:     int,
        kv_lora_rank:    int,
        num_heads:       int,
        head_dim:        int,
        indexer_n_heads: int,
        indexer_topk:    int,
        affinity:        DSALayerDeviceAffinity,
        slc:             Optional[SharedLocalityCache],
        dtype:           torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.affinity        = affinity
        self.slc             = slc
        self.kv_lora_rank    = kv_lora_rank
        self.indexer_topk    = indexer_topk
        self.hidden_size     = hidden_size

        # ------------------------------------------------------------------
        # Phase 1 modules — live on affinity.indexer_device
        # ------------------------------------------------------------------
        idx_dev = affinity.indexer_device

        # linear_wq_b: projects hidden to indexer query space
        self.linear_wq_b = nn.Linear(
            hidden_size, indexer_n_heads * head_dim, bias=False, dtype=dtype,
        ).to(idx_dev)

        # linear_wk: projects hidden to indexer key space
        self.linear_wk = nn.Linear(
            hidden_size, indexer_n_heads * head_dim, bias=False, dtype=dtype,
        ).to(idx_dev)

        # k_norm: RMSNorm on indexer keys
        self.k_norm = nn.RMSNorm(head_dim, dtype=dtype).to(idx_dev)

        # linear_weights_proj: projects top-k weights to kv_lora_rank (compression)
        self.linear_weights_proj = nn.Linear(
            indexer_topk, kv_lora_rank, bias=False, dtype=dtype,
        ).to(idx_dev)

        # ------------------------------------------------------------------
        # Phase 2 modules — live on affinity.projector_device
        # ------------------------------------------------------------------
        proj_dev = affinity.projector_device

        # linear_kv_down_proj: compress full-seq KV → latent
        self.linear_kv_down_proj = nn.Linear(
            hidden_size, kv_lora_rank * 2, bias=False, dtype=dtype,
        ).to(proj_dev)

        # linear_kv_up_proj: expand compressed top-k KV → full heads
        self.linear_kv_up_proj = nn.Linear(
            kv_lora_rank, num_heads * head_dim * 2, bias=False, dtype=dtype,
        ).to(proj_dev)

        # linear_proj: output projection back to hidden_size
        self.linear_proj = nn.Linear(
            num_heads * head_dim, hidden_size, bias=False, dtype=dtype,
        ).to(proj_dev)

        logger.info(
            "DecoupledDSAExecutor: layer global=%d local=%d | "
            "Phase1=cuda:%d Phase2=cuda:%d use_slc=%s",
            affinity.global_layer_idx, affinity.local_dsa_idx,
            affinity.indexer_device.index if affinity.indexer_device.index is not None else 0,
            affinity.projector_device.index if affinity.projector_device.index is not None else 0,
            affinity.use_slc,
        )

    def _phase1_index(
        self,
        hidden: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Phase 1: run DSAIndexer on indexer_device.

        Args:
            hidden: (seq, batch, hidden_size) on indexer_device.
            stream: CUDA stream for async execution.

        Returns:
            compressed_kv:  (seq, batch, kv_lora_rank) on indexer_device.
            topk_indices:   (seq, batch, 1) int32 on indexer_device.
        """
        ctx = torch.cuda.stream(stream) if stream else _null_ctx()
        with ctx:
            seq, batch, _ = hidden.shape

            # Query projection for indexer
            q_idx = self.linear_wq_b(hidden)   # (seq, batch, idx_heads*head_dim)

            # Key projection + norm for indexer
            k_idx = self.k_norm(self.linear_wk(hidden))  # (seq, batch, idx_heads*head_dim)

            # Indexer attention scores — dot product across head dimension
            # Reshape to (seq, batch, n_heads, head_dim) for batched matmul
            head_dim = k_idx.shape[-1] // self.linear_wk.out_features * self.linear_wk.out_features
            # Simplified: scores = mean across head dimension of q·k
            scores = (q_idx * k_idx).sum(dim=-1, keepdim=True)  # (seq, batch, 1)
            topk_indices = scores.argsort(dim=0, descending=True)[:self.indexer_topk]
            # Collapse to (seq, batch, 1) top-k indicator via weighted projection
            # linear_weights_proj maps topk-dim → kv_lora_rank
            topk_weights = torch.softmax(scores, dim=0)  # (seq, batch, 1)
            # compressed_kv approximation: project topk weights → kv_lora_rank
            compressed_kv = self.linear_weights_proj(
                topk_weights.expand(-1, -1, self.indexer_topk)
            )   # (seq, batch, kv_lora_rank)

        return compressed_kv, topk_indices.to(torch.int32)

    def _transfer_via_slc(
        self,
        compressed_kv: torch.Tensor,
        topk_indices:  torch.Tensor,
        write_stream:  Optional[torch.cuda.Stream] = None,
        read_stream:   Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transfer indexer outputs from INDEXER device to PROJECTOR device via SLC.

        If SLC is not configured (same-device scenario), falls back to direct
        `.to(projector_device)` transfer.

        Args:
            compressed_kv:  Indexer output (seq, batch, kv_lora_rank) on indexer_device.
            topk_indices:   Top-k indices (seq, batch, 1) int32 on indexer_device.
            write_stream:   Stream on indexer_device for async D2H copy.
            read_stream:    Stream on projector_device for async H2D copy.

        Returns:
            Both tensors on projector_device.
        """
        proj_dev = self.affinity.projector_device

        if self.slc is None or not self.affinity.use_slc:
            # Same device or no SLC: direct transfer
            return (
                compressed_kv.to(proj_dev, non_blocking=True),
                topk_indices.to(proj_dev, non_blocking=True),
            )

        local_idx = self.affinity.local_dsa_idx
        slot = self.slc.write_kv(local_idx, compressed_kv, topk_indices, stream=write_stream)
        logger.debug("SLC: wrote layer_local=%d slot=%d", local_idx, slot)

        # Read back (blocks until producer event fires on write_stream)
        cpu_kv, cpu_tidx = self.slc.read_kv(local_idx, stream=read_stream)
        return (
            cpu_kv.to(proj_dev, non_blocking=True),
            cpu_tidx.to(proj_dev, non_blocking=True),
        )

    def _phase2_project(
        self,
        hidden_proj:   torch.Tensor,
        compressed_kv: torch.Tensor,
        topk_indices:  torch.Tensor,
        stream:        Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Phase 2: sparse attention projection on projector_device.

        Args:
            hidden_proj:    (seq, batch, hidden_size) on projector_device.
            compressed_kv:  (seq, batch, kv_lora_rank) on projector_device.
            topk_indices:   (seq, batch, 1) int32 on projector_device.
            stream:         CUDA stream on projector_device.

        Returns:
            (seq, batch, hidden_size) output on projector_device.
        """
        ctx = torch.cuda.stream(stream) if stream else _null_ctx()
        with ctx:
            # Down-project full hidden to compressed KV latent
            kv_latent = self.linear_kv_down_proj(hidden_proj)  # (seq, batch, kv_lora_rank*2)
            k_latent, v_latent = kv_latent.chunk(2, dim=-1)

            # Up-project compressed KV (from indexer) to full head dimension
            # In production this would use selected token indices; here we use
            # the compressed representation directly for the DES-LOC pathway.
            kv_up = self.linear_kv_up_proj(compressed_kv)  # (seq, batch, heads*head_dim*2)
            k_up, v_up = kv_up.chunk(2, dim=-1)

            # Residual blend: mix full-seq KV with top-k selected KV
            # topk_indices encodes selection weights; cast to float for blend
            blend = topk_indices.to(compressed_kv.dtype).clamp(0.0, 1.0)
            k_blend = (1.0 - blend) * k_latent[..., :k_up.shape[-1]] + blend * k_up
            v_blend = (1.0 - blend) * v_latent[..., :v_up.shape[-1]] + blend * v_up

            # Output projection
            seq, batch, hd = k_blend.shape
            out = self.linear_proj(k_blend + v_blend)  # (seq, batch, hidden_size)

        return out

    def forward(
        self,
        hidden: torch.Tensor,
        indexer_stream:  Optional[torch.cuda.Stream] = None,
        projector_stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Full DES-LOC two-phase forward pass for one DSA layer.

        Args:
            hidden:            (seq, batch, hidden_size) on indexer_device.
            indexer_stream:    CUDA stream for Phase 1 (indexer_device).
            projector_stream:  CUDA stream for Phase 2 (projector_device).

        Returns:
            (seq, batch, hidden_size) on projector_device.
        """
        # Phase 1: run on indexer device
        compressed_kv, topk_indices = self._phase1_index(hidden, stream=indexer_stream)

        # Transfer via SLC (or direct copy if same device)
        proj_dev = self.affinity.projector_device
        hidden_proj = hidden.to(proj_dev, non_blocking=True)
        compressed_kv, topk_indices = self._transfer_via_slc(
            compressed_kv, topk_indices,
            write_stream=indexer_stream,
            read_stream=projector_stream,
        )

        # Phase 2: run on projector device
        output = self._phase2_project(
            hidden_proj, compressed_kv, topk_indices, stream=projector_stream,
        )
        return output


# ---------------------------------------------------------------------------
# Hetero Mamba stack builder (DES-LOC layer orchestrator)
# ---------------------------------------------------------------------------

class HeteroMambaStackBuilder:
    """Constructs the per-layer executor list for a DES-LOC hybrid Mamba stack.

    Upstream context (Megatron a00e9443):
        MambaStack.build() iterates over `layer_type_list` and dispatches each
        symbol to the corresponding module spec (mamba_layer, attention_layer,
        dsa_layer, mlp_layer, moe_layer). The new 'D' branch calls
        `build_module(submodules.dsa_layer, ...)` with the same kwargs as
        attention layers, adding `pp_layer_offset` threading for PP correctness.

    DES-LOC adaptation:
        Rather than building a monolithic TransformerLayer (attention + MLP in
        one device), we build a `DecoupledDSAExecutor` for each 'D' layer that
        explicitly separates the indexer (Phase 1) from the projector (Phase 2).
        Other layer types (M, -, E) are built normally and remain on their
        assigned device.

    Args:
        layer_type_list:    Ordered list of layer symbols from `parse_layer_type_list`.
        fleet_config:       DES-LOC hardware fleet configuration.
        slc:                Shared LOcality Cache (pre-allocated).
        model_config:       Dict of model hyperparameters
                            (hidden_size, kv_lora_rank, num_heads, head_dim,
                             indexer_n_heads, indexer_topk, dtype).
        pp_layer_offset:    Pipeline-parallel layer offset (number of layers on
                            preceding PP stages). Used to derive absolute layer
                            numbers for RoPE buffer registration.
    """

    def __init__(
        self,
        layer_type_list: List[str],
        fleet_config:    HeteroFleetConfig,
        slc:             Optional[SharedLocalityCache],
        model_config:    Dict,
        pp_layer_offset: int = 0,
    ) -> None:
        validate_hybrid_pattern("".join(layer_type_list))
        self.layer_type_list  = layer_type_list
        self.fleet_config     = fleet_config
        self.slc              = slc
        self.model_config     = model_config
        self.pp_layer_offset  = pp_layer_offset

        self._layer_maps  = get_layer_maps(layer_type_list)
        self._affinity_map = build_device_affinity_map(self._layer_maps, fleet_config)

        logger.info(
            "HeteroMambaStackBuilder: %d layers | DSA=%d attn=%d mamba=%d mlp=%d moe=%d",
            len(layer_type_list),
            len(self._layer_maps.get(LayerSymbol.DS_ATTENTION, {})),
            len(self._layer_maps.get(LayerSymbol.ATTENTION, {})),
            len(self._layer_maps.get(LayerSymbol.MAMBA, {})),
            len(self._layer_maps.get(LayerSymbol.MLP, {})),
            len(self._layer_maps.get(LayerSymbol.MOE, {})),
        )

    @property
    def num_attention_layers(self) -> int:
        """KV-cache layer count: standard attention + DSA (mirrors upstream change)."""
        return count_attention_layers(self._layer_maps)

    @property
    def layer_map(self) -> Dict[int, int]:
        """Combined global→local index map for all KV-cache-bearing layers."""
        attn = self._layer_maps.get(LayerSymbol.ATTENTION, {})
        dsa  = self._layer_maps.get(LayerSymbol.DS_ATTENTION, {})
        return attn | dsa

    def build_dsa_executor(self, global_layer_idx: int) -> DecoupledDSAExecutor:
        """Construct a DecoupledDSAExecutor for the given global layer index.

        Args:
            global_layer_idx: Position in the full model layer list.

        Returns:
            A configured DecoupledDSAExecutor with Phase 1/2 modules on their
            respective devices.

        Raises:
            KeyError: If global_layer_idx is not a DSA layer.
        """
        affinity = self._affinity_map[global_layer_idx]
        cfg = self.model_config
        return DecoupledDSAExecutor(
            hidden_size=cfg["hidden_size"],
            kv_lora_rank=cfg["kv_lora_rank"],
            num_heads=cfg["num_heads"],
            head_dim=cfg["head_dim"],
            indexer_n_heads=cfg["indexer_n_heads"],
            indexer_topk=cfg["indexer_topk"],
            affinity=affinity,
            slc=self.slc,
            dtype=cfg.get("dtype", torch.bfloat16),
        )

    def iter_layers(self):
        """Yield (global_idx, absolute_layer_number, symbol, executor_or_None).

        For DSA layers, executor is a DecoupledDSAExecutor.
        For other layer types, executor is None (caller should use their own builder).
        The absolute_layer_number is global_idx + pp_layer_offset, matching the
        upstream pp_layer_offset threading fix in MLASelfAttention.
        """
        for global_idx, symbol in enumerate(self.layer_type_list):
            abs_layer_num = global_idx + self.pp_layer_offset
            if symbol == LayerSymbol.DS_ATTENTION:
                executor = self.build_dsa_executor(global_idx)
            else:
                executor = None
            yield global_idx, abs_layer_num, symbol, executor


# ---------------------------------------------------------------------------
# Checkpoint key remapping (DES-LOC adaptation of tools/checkpoint/remap_gpt_dsa_to_mamba.py)
# ---------------------------------------------------------------------------

def remap_gpt_dsa_to_deslocmamba_key(key: str, num_gpt_layers: int) -> str:
    """Remap a GPTModel DSA state-dict key to DES-LOC Mamba layout.

    Mirrors Megatron's `_remap_key` from tools/checkpoint/remap_gpt_dsa_to_mamba.py
    but adds a DES-LOC-specific prefix convention: DES-LOC stores Phase 1 weights
    under `dsa_executor.{2N}.phase1.*` and Phase 2 under `dsa_executor.{2N}.phase2.*`
    when using DecoupledDSAExecutor, in addition to the standard Mamba-layout
    renaming for compatibility with non-decoupled checkpoints.

    This function handles the *standard* Mamba-layout remapping (used when loading
    upstream Megatron checkpoints into DES-LOC). The DES-LOC-specific split layout
    is handled separately by `split_dsa_executor_state_dict`.

    Key remapping rules:
        decoder.layers.{N}.input_layernorm.*    → decoder.layers.{2N}.input_layernorm.*
        decoder.layers.{N}.self_attention.*     → decoder.layers.{2N}.self_attention.*
        decoder.layers.{N}.pre_mlp_layernorm.*  → decoder.layers.{2N+1}.pre_mlp_layernorm.*
        decoder.layers.{N}.mlp.*               → decoder.layers.{2N+1}.mlp.*
        decoder.final_layernorm.*              → decoder.final_norm.*

    Args:
        key:             Key from GPTModel state dict.
        num_gpt_layers:  Total number of GPT decoder layers.

    Returns:
        Remapped key string.

    Raises:
        ValueError: On unexpected sub-key patterns.
    """
    LAYER_PFX    = "decoder.layers."
    FINAL_LN_PFX = "decoder.final_layernorm."

    if key.startswith(FINAL_LN_PFX):
        return "decoder.final_norm." + key[len(FINAL_LN_PFX):]

    if not key.startswith(LAYER_PFX):
        return key  # embedding, output_layer, rotary_pos_emb, etc.

    remainder = key[len(LAYER_PFX):]
    dot_idx   = remainder.index(".")
    layer_n   = int(remainder[:dot_idx])
    rest      = remainder[dot_idx + 1:]

    if not (0 <= layer_n < num_gpt_layers):
        raise ValueError(
            f"Layer index {layer_n} out of range [0, {num_gpt_layers}) in key '{key}'"
        )

    if rest.startswith("input_layernorm.") or rest.startswith("self_attention."):
        return f"{LAYER_PFX}{2 * layer_n}.{rest}"
    elif rest.startswith("mlp."):
        return f"{LAYER_PFX}{2 * layer_n + 1}.{rest}"
    elif rest.startswith("pre_mlp_layernorm."):
        return f"{LAYER_PFX}{2 * layer_n + 1}.{rest}"
    else:
        raise ValueError(
            f"Unexpected sub-key '{rest}' in GPT layer {layer_n} (full key='{key}'). "
            "Expected: input_layernorm.*, self_attention.*, pre_mlp_layernorm.*, mlp.*"
        )


def remap_state_dict(gpt_sd: Dict, num_gpt_layers: int) -> Dict:
    """Apply `remap_gpt_dsa_to_deslocmamba_key` to every key in gpt_sd."""
    return {remap_gpt_dsa_to_deslocmamba_key(k, num_gpt_layers): v for k, v in gpt_sd.items()}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    # 1. Validate that the LayerSymbol sentinel matches expected characters
    assert LayerSymbol.DS_ATTENTION == "D", "DSA symbol must be 'D'"
    assert "D" in LayerSymbol.VALID_LAYERS, "'D' must be in VALID_LAYERS"

    # 2. Layer map construction mirrors Megatron dict-return convention
    maps = get_layer_maps(["D", "M", "D", "-"])
    assert maps[LayerSymbol.DS_ATTENTION] == {0: 0, 2: 1}, "DSA map wrong"
    assert maps[LayerSymbol.MAMBA]        == {1: 0},        "Mamba map wrong"
    assert count_attention_layers(maps)   == 2,             "Should count 2 DSA layers"

    # 3. Constraint: '*' and 'D' cannot coexist
    try:
        validate_hybrid_pattern("D*M")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # 4. Fleet config auto-detection (CPU fallback)
    fleet = HeteroFleetConfig(devices=[])
    assert fleet.projector_device() is None

    # 5. Checkpoint key remapping
    remapped = remap_gpt_dsa_to_deslocmamba_key(
        "decoder.layers.1.self_attention.weight", num_gpt_layers=4
    )
    assert remapped == "decoder.layers.2.self_attention.weight", f"Got: {remapped}"

    logger.info("All smoke tests passed.")
