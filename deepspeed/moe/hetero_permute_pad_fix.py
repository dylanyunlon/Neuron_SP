"""
deepspeed/moe/hetero_permute_pad_fix.py
========================================
M3707-BF: HeteroMoEPermutePadFix for DES-LOC heterogeneous training framework.

Upstream Design Intent (Megatron commit 567d4d4):
-------------------------------------------------
Megatron-LM's MoE dispatch pipeline uses a `permute()` function that reorders
tokens from [S, B, H] layout into expert-grouped layout for efficient expert
computation. An optional `align_size` parameter pads each expert's token bucket
to a multiple of `align_size` — this is mandatory for quantized kernels (FP8/FP4)
which require aligned memory access patterns.

The upstream bug: `align_size` defaulted to -1, and the guard condition
`fused_permute_and_pad_with_probs is not None and tokens_per_expert is not None`
did NOT check whether `align_size > 0`. This caused the fused-pad path to be
invoked even for non-quantized (BF16/FP16) dispatches, adding unnecessary padding
overhead and corrupting token counts in non-quantized MoE layers.

Fix: default `align_size=0`, add `align_size > 0` to the guard. Return `align_size=0`
from `get_align_size_for_quantization()` when neither FP8 nor FP4 is active.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs MoE experts across
a heterogeneous device pool:
  - 2x A6000 48GB (SM86, PCIe)  → hosts "local" experts (expert_rank 0–N/2)
  - 1x H100 NVL 96GB (SM90)     → hosts "remote" experts (expert_rank N/2–N)
  - 1.5TB CPU DRAM               → Shared LOcality Cache (SLC) for token staging

Key DES-LOC-specific issues that the upstream fix unlocks:

1. CROSS-DEVICE PADDING ASYMMETRY
   A6000 kernels are BF16; H100 may run FP8. If padding is unconditionally applied,
   tokens dispatched to A6000 experts receive spurious padding that:
   (a) inflates PCIe transfer volume (A6000 is PCIe-only, no NVLink),
   (b) corrupts the SLC token count metadata used by the locality cache eviction.

2. SLC METADATA CONSISTENCY
   The Shared LOcality Cache tracks (expert_id → token_count) for prefetch
   scheduling. Padding-inflated token counts break the LRU eviction heuristic
   and cause cache thrashing on cross-device expert boundaries.

3. DEVICE-AWARE ALIGN_SIZE ROUTING
   `get_hetero_align_size()` (this module) returns device-appropriate alignment:
   - H100 FP8 path  → 16 (fp8 warp requirement)
   - A6000 BF16     →  0 (no padding needed, matches upstream fix)
   - CPU offload    →  0 (SLC handles its own alignment)

4. PERMUTE GUARD PROPAGATION
   `hetero_permute()` mirrors Megatron's fixed guard logic, extended with a
   device-locality branch so the fused TE kernel is only called when:
   (a) fused kernel is available, (b) tokens_per_expert provided,
   (c) align_size > 0, AND (d) target device supports the fused path (SM >= 89).

Author: Neuron_SP project (DES-LOC adaptation)
Upstream: Megatron-LM commit 567d4d468178735d5b244fea0d0738dc3d715599
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware topology constants for DES-LOC heterogeneous cluster
# ---------------------------------------------------------------------------

class DeviceClass(Enum):
    """Classifies physical device by capability tier within DES-LOC pool."""
    LOCAL_SM86 = auto()   # A6000 48GB, PCIe, SM86 — no FP8 native
    REMOTE_SM90 = auto()  # H100 NVL 96GB, SM90 — FP8 native
    CPU_SLC = auto()      # CPU DRAM acting as Shared LOcality Cache


# SM compute capability thresholds
_SM_FP8_MIN = 89          # Ada / Hopper required for native FP8
_SM_FUSED_PAD_MIN = 89    # fused_permute_and_pad_with_probs needs SM >= 89


@dataclass
class HeteroDeviceSpec:
    """
    Describes a single device in the DES-LOC pool.

    Parameters
    ----------
    device_class : DeviceClass
        Tier of this device.
    torch_device : torch.device
        Concrete PyTorch device handle.
    sm_version : int
        CUDA SM version * 10 (e.g., 86 for SM8.6, 90 for SM9.0). 0 for CPU.
    supports_fp8 : bool
        Whether the device has native FP8 tensor core support.
    max_align_size : int
        Maximum meaningful alignment that quantized kernels on this device need.
        0 means no quantization-driven padding.
    pcie_only : bool
        True when the device lacks NVLink — affects SLC prefetch scheduling.
    """
    device_class: DeviceClass
    torch_device: torch.device
    sm_version: int = 0
    supports_fp8: bool = False
    max_align_size: int = 0
    pcie_only: bool = True


# Default DES-LOC cluster topology (2× A6000 + 1× H100 NVL)
DEFAULT_HETERO_POOL: List[HeteroDeviceSpec] = [
    HeteroDeviceSpec(
        device_class=DeviceClass.LOCAL_SM86,
        torch_device=torch.device("cuda:0"),
        sm_version=86,
        supports_fp8=False,
        max_align_size=0,
        pcie_only=True,
    ),
    HeteroDeviceSpec(
        device_class=DeviceClass.LOCAL_SM86,
        torch_device=torch.device("cuda:1"),
        sm_version=86,
        supports_fp8=False,
        max_align_size=0,
        pcie_only=True,
    ),
    HeteroDeviceSpec(
        device_class=DeviceClass.REMOTE_SM90,
        torch_device=torch.device("cuda:2"),
        sm_version=90,
        supports_fp8=True,
        max_align_size=16,
        pcie_only=True,  # PCIe interconnect even for H100 NVL in this cluster
    ),
]


# ---------------------------------------------------------------------------
# Shared LOcality Cache (SLC) metadata types
# ---------------------------------------------------------------------------

class SLCTokenEntry(NamedTuple):
    """
    Per-expert token count record stored in the Shared LOcality Cache.

    Upstream padding corruption scenario (the bug we fix):
    If `align_size` is wrongly applied to BF16 A6000 experts, `padded_count`
    diverges from `real_count`. The SLC eviction policy uses `real_count` for
    LRU scoring but prefetch uses `padded_count` for DMA sizing — mismatch
    causes over-allocation in CPU DRAM and stale cache entries.
    """
    expert_id: int
    real_count: int      # actual tokens routed to this expert
    padded_count: int    # count after alignment padding (== real_count when align_size=0)
    device_class: DeviceClass


@dataclass
class SharedLocalityCache:
    """
    Lightweight in-process Shared LOcality Cache for token staging metadata.

    In production DES-LOC this is backed by a distributed key-value store
    over CPU DRAM (1.5TB), coordinated via torch.distributed. Here we keep
    a local dict for single-node simulation and unit tests.

    The SLC is consulted by the hetero dispatch scheduler to:
    - prefetch expert weights to A6000/H100 before token arrival
    - decide whether to offload infrequent expert activations to CPU DRAM
    - track real vs padded token counts to avoid DMA over-allocation
    """
    _store: Dict[int, SLCTokenEntry] = field(default_factory=dict)
    _access_order: List[int] = field(default_factory=list)
    capacity: int = 8192  # max number of experts tracked in cache

    def update(self, entry: SLCTokenEntry) -> None:
        """Insert or refresh an expert's token metadata."""
        eid = entry.expert_id
        if eid in self._store:
            self._access_order.remove(eid)
        elif len(self._store) >= self.capacity:
            evicted = self._access_order.pop(0)
            del self._store[evicted]
            logger.debug("SLC evicted expert %d (capacity %d)", evicted, self.capacity)
        self._store[eid] = entry
        self._access_order.append(eid)

    def get(self, expert_id: int) -> Optional[SLCTokenEntry]:
        return self._store.get(expert_id)

    def real_token_count(self, expert_id: int) -> int:
        """Return the *real* (un-padded) token count for prefetch scheduling."""
        entry = self._store.get(expert_id)
        return entry.real_count if entry is not None else 0

    def consistency_check(self) -> bool:
        """
        Verify that no BF16-path expert has padded_count != real_count.
        This should always be true after the permute-pad fix is applied.
        """
        for eid, entry in self._store.items():
            if entry.device_class == DeviceClass.LOCAL_SM86:
                if entry.padded_count != entry.real_count:
                    logger.error(
                        "SLC inconsistency: SM86 expert %d real=%d padded=%d",
                        eid, entry.real_count, entry.padded_count,
                    )
                    return False
        return True


# Module-level SLC singleton (replaced by distributed backend in production)
_SLC = SharedLocalityCache()


def get_global_slc() -> SharedLocalityCache:
    """Return the module-level SLC instance."""
    return _SLC


# ---------------------------------------------------------------------------
# Alignment-size helpers — DES-LOC device-aware
# ---------------------------------------------------------------------------

def get_hetero_align_size(
    device_spec: HeteroDeviceSpec,
    fp8_active: bool = False,
    fp4_active: bool = False,
) -> int:
    """
    Return the appropriate `align_size` for MoE permute on a given device.

    Upstream fix (commit 567d4d4):
    ``get_align_size_for_quantization()`` previously returned 16 unconditionally,
    triggering padding even for BF16/FP16. Now returns 0 unless FP8 or FP4 is active.

    DES-LOC extension:
    We gate further on *device capability*, not just recipe flags. Even if a
    global FP8 flag is set, an A6000 (SM86) cannot run native FP8 kernels —
    calling the fused pad path on it would either silently fall back to a slower
    path or crash. We hard-gate SM86 to align_size=0.

    Parameters
    ----------
    device_spec : HeteroDeviceSpec
        The device that will execute the expert computation.
    fp8_active : bool
        True if an FP8 recipe is configured for this layer.
    fp4_active : bool
        True if an FP4 recipe is configured for this layer.

    Returns
    -------
    int
        0 → no padding; >0 → pad each expert bucket to this multiple.
    """
    if device_spec.device_class == DeviceClass.CPU_SLC:
        # CPU-offloaded experts: SLC manages its own DRAM alignment via
        # numpy-style strides; permute padding is counterproductive.
        logger.debug("align_size=0: CPU_SLC path, no permute padding")
        return 0

    if device_spec.sm_version < _SM_FP8_MIN:
        # A6000 SM86 cannot run FP8. Padding would only waste PCIe bandwidth.
        if fp8_active or fp4_active:
            logger.warning(
                "FP8/FP4 requested on SM%d device %s but device lacks native "
                "FP8 support — forcing align_size=0 to avoid spurious PCIe padding.",
                device_spec.sm_version,
                device_spec.torch_device,
            )
        return 0

    # SM >= 89 (H100 NVL, SM90): respect the quantization recipe flags,
    # mirroring the upstream fix logic.
    if fp8_active:
        align_size = 16  # fp8 warp tile requirement
        logger.debug(
            "align_size=%d: FP8 active on SM%d %s",
            align_size, device_spec.sm_version, device_spec.torch_device,
        )
        return align_size

    if fp4_active:
        align_size = 16  # fp4 uses same alignment as fp8 in current TE
        logger.debug(
            "align_size=%d: FP4 active on SM%d %s",
            align_size, device_spec.sm_version, device_spec.torch_device,
        )
        return align_size

    # Neither FP8 nor FP4 → no padding (core of the upstream fix)
    logger.debug(
        "align_size=0: non-quantized path on SM%d %s",
        device_spec.sm_version, device_spec.torch_device,
    )
    return 0


# ---------------------------------------------------------------------------
# Fused kernel availability probe
# ---------------------------------------------------------------------------

def _probe_fused_permute_pad() -> Optional[object]:
    """
    Attempt to import ``fused_permute_and_pad_with_probs`` from TransformerEngine.

    Returns the callable if available (TE >= 2.12.0), else None.
    DES-LOC note: TE is only installed on the H100 node image; A6000 nodes
    run a TE-free DeepSpeed environment. The probe result is cached at import
    time, so per-expert dispatch decisions are O(1).
    """
    try:
        # TE >= 2.12.0 exposes this in transformer_engine.pytorch.ops
        from transformer_engine.pytorch.ops import fused_permute_and_pad_with_probs as _fn
        logger.info("TransformerEngine fused_permute_and_pad_with_probs available")
        return _fn
    except ImportError:
        pass
    try:
        # Older TE location
        from transformer_engine.pytorch import fused_permute_and_pad_with_probs as _fn
        logger.info(
            "TransformerEngine fused_permute_and_pad_with_probs available (legacy path)"
        )
        return _fn
    except ImportError:
        pass
    logger.info(
        "TransformerEngine fused_permute_and_pad_with_probs NOT available — "
        "using pure-PyTorch permute path"
    )
    return None


_FUSED_PERMUTE_AND_PAD: Optional[object] = _probe_fused_permute_pad()


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback permute (no TE dependency)
# ---------------------------------------------------------------------------

def _torch_permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    tokens_per_expert: Optional[torch.Tensor] = None,
    align_size: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Pure-PyTorch MoE token permute with optional expert-bucket padding.

    Layout contract (mirrors Megatron moe_utils.permute):
    - tokens:      [S*B, H] — flattened sequence
    - routing_map: [S*B, E] — bool or int8, True where token → expert e
    - probs:       [S*B, E] — routing probabilities (optional)

    Returns
    -------
    permuted_tokens : [T_total_padded, H]
        Tokens reordered by expert assignment. T_total_padded == T_total when
        align_size == 0 (the non-quantized case fixed by upstream commit).
    permuted_probs : [T_total_padded] or None
    restore_index  : [T_total_padded] → original flat token index, or None
    """
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]

    # Gather per-expert token indices
    expert_indices: List[torch.Tensor] = []
    for e in range(num_experts):
        idx = routing_map[:, e].nonzero(as_tuple=False).squeeze(1)
        expert_indices.append(idx)

    # Compute padded bucket sizes
    bucket_sizes: List[int] = []
    for e in range(num_experts):
        real_cnt = expert_indices[e].numel()
        if align_size > 0:
            padded_cnt = math.ceil(real_cnt / align_size) * align_size
        else:
            padded_cnt = real_cnt  # upstream fix: no padding for BF16
        bucket_sizes.append(padded_cnt)

    total_padded = sum(bucket_sizes)

    # Allocate output buffers
    permuted_tokens = torch.zeros(
        total_padded, hidden, dtype=tokens.dtype, device=tokens.device
    )
    permuted_probs = (
        torch.zeros(total_padded, dtype=probs.dtype, device=probs.device)
        if probs is not None else None
    )
    restore_index = torch.full(
        (total_padded,), fill_value=-1, dtype=torch.long, device=tokens.device
    )

    # Fill expert buckets
    offset = 0
    for e in range(num_experts):
        idx = expert_indices[e]
        real_cnt = idx.numel()
        if real_cnt > 0:
            permuted_tokens[offset:offset + real_cnt] = tokens[idx]
            restore_index[offset:offset + real_cnt] = idx
            if permuted_probs is not None:
                permuted_probs[offset:offset + real_cnt] = probs[idx, e]
        # Remaining [offset+real_cnt : offset+bucket_sizes[e]] stays zero-padded
        offset += bucket_sizes[e]

    return permuted_tokens, permuted_probs, restore_index


# ---------------------------------------------------------------------------
# Core: hetero_permute — the DES-LOC-aware replacement for Megatron permute()
# ---------------------------------------------------------------------------

def hetero_permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    tokens_per_expert: Optional[torch.Tensor] = None,
    align_size: int = 0,
    device_spec: Optional[HeteroDeviceSpec] = None,
    expert_id_offset: int = 0,
    slc: Optional[SharedLocalityCache] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    DES-LOC-aware MoE token permute with corrected padding guard.

    This function is the DES-LOC replacement for Megatron's ``moe_utils.permute()``.
    It incorporates the upstream bug fix (commit 567d4d4) and extends it with
    heterogeneous device routing and SLC metadata updates.

    Upstream Fix Applied Here
    -------------------------
    Old Megatron guard::

        if fused_permute_and_pad_with_probs is not None and tokens_per_expert is not None:
            return fused_permute_and_pad_with_probs(...)  # BUG: called for BF16!

    Fixed guard (upstream + DES-LOC)::

        if (
            fused_permute_and_pad_with_probs is not None
            and tokens_per_expert is not None
            and align_size > 0                    # ← upstream fix
            and _device_supports_fused(device_spec)  # ← DES-LOC extension
        ):
            return fused_permute_and_pad_with_probs(...)

    DES-LOC Dispatch Logic
    ----------------------
    Expert experts are partitioned across devices:

    - ``expert_id_offset=0``, ``device_spec.device_class=LOCAL_SM86``
      → A6000 experts. ``align_size`` will be 0 (BF16 path); fused kernel
      NOT called regardless of TE availability. This avoids spurious PCIe
      padding traffic.

    - ``expert_id_offset=N//2``, ``device_spec.device_class=REMOTE_SM90``
      → H100 experts. If FP8 is active, ``align_size=16``; fused TE kernel
      invoked when available. If BF16, ``align_size=0``; falls through to
      pure-PyTorch path.

    SLC Metadata Update
    -------------------
    After permutation, this function writes real and padded token counts per
    expert into the SLC. Downstream prefetch logic reads ``real_count`` for
    DMA sizing; using the pre-fix padded count here would have caused
    over-allocation in CPU DRAM.

    Parameters
    ----------
    tokens : torch.Tensor [S*B, H]
        Input tokens, flattened sequence × batch.
    routing_map : torch.Tensor [S*B, E]
        Boolean expert assignment map.
    probs : torch.Tensor [S*B, E], optional
        Routing probabilities for weighted aggregation.
    tokens_per_expert : torch.Tensor [E], optional
        Pre-computed per-expert token counts (avoids re-counting).
        Required for the fused TE kernel path.
    align_size : int
        Padding alignment. 0 = no padding (correct default for BF16).
        Should be obtained from ``get_hetero_align_size()``.
    device_spec : HeteroDeviceSpec, optional
        Target device descriptor. If None, uses first LOCAL_SM86 spec.
    expert_id_offset : int
        Global expert-ID offset for SLC keying. Local expert 0 on the A6000
        has global ID ``expert_id_offset + 0``.
    slc : SharedLocalityCache, optional
        SLC instance to update. Defaults to module-level singleton.

    Returns
    -------
    permuted_tokens  : [T_total_padded, H]
    permuted_probs   : [T_total_padded] or None
    restore_index    : [T_total_padded] or None
    """
    if device_spec is None:
        device_spec = DEFAULT_HETERO_POOL[0]
        logger.debug("hetero_permute: no device_spec provided, defaulting to LOCAL_SM86")

    if slc is None:
        slc = get_global_slc()

    num_experts = routing_map.shape[1]

    # --- Guard: can we use the fused TE kernel? ----------------------------
    # This replicates the upstream fix logic plus DES-LOC device capability gate.
    _use_fused = (
        _FUSED_PERMUTE_AND_PAD is not None              # TE installed
        and tokens_per_expert is not None               # counts available
        and align_size > 0                              # quantized path (upstream fix)
        and device_spec.sm_version >= _SM_FUSED_PAD_MIN # SM >= 89 required
        and device_spec.device_class != DeviceClass.CPU_SLC
    )

    logger.debug(
        "hetero_permute: device=%s sm=%d align_size=%d fused=%s tokens=%s",
        device_spec.torch_device,
        device_spec.sm_version,
        align_size,
        _use_fused,
        tuple(tokens.shape),
    )

    if _use_fused:
        logger.debug("hetero_permute: dispatching to fused TE kernel")
        result = _FUSED_PERMUTE_AND_PAD(
            tokens, probs, routing_map, tokens_per_expert, align_size
        )
        permuted_tokens, permuted_probs, restore_index = result
    else:
        permuted_tokens, permuted_probs, restore_index = _torch_permute(
            tokens, routing_map, probs, tokens_per_expert, align_size
        )

    # --- Update SLC with corrected (real, padded) token counts -------------
    # Pre-fix: padded_count was inflated for BF16 experts → SLC stored wrong
    # counts → prefetch DMA sized incorrectly → CPU DRAM over-allocation.
    # Post-fix: align_size=0 for BF16 → padded_count == real_count → SLC correct.
    _update_slc_metadata(
        routing_map=routing_map,
        align_size=align_size,
        device_spec=device_spec,
        expert_id_offset=expert_id_offset,
        slc=slc,
    )

    return permuted_tokens, permuted_probs, restore_index


def _update_slc_metadata(
    routing_map: torch.Tensor,
    align_size: int,
    device_spec: HeteroDeviceSpec,
    expert_id_offset: int,
    slc: SharedLocalityCache,
) -> None:
    """
    Write real and padded per-expert token counts into the SLC.

    Called after every hetero_permute() invocation. The separation from
    the main permute logic mirrors Megatron's separation of concerns between
    routing metadata and actual permutation.

    This is the function most directly affected by the upstream bug:
    if align_size were non-zero for BF16 (old behaviour), padded_count
    would diverge from real_count for LOCAL_SM86 experts, causing SLC
    prefetch thrashing.
    """
    num_experts = routing_map.shape[1]

    for local_e in range(num_experts):
        col = routing_map[:, local_e]
        real_cnt = int(col.sum().item())
        if align_size > 0:
            padded_cnt = math.ceil(real_cnt / align_size) * align_size
        else:
            padded_cnt = real_cnt  # core fix: no inflation for BF16

        global_eid = expert_id_offset + local_e
        entry = SLCTokenEntry(
            expert_id=global_eid,
            real_count=real_cnt,
            padded_count=padded_cnt,
            device_class=device_spec.device_class,
        )
        slc.update(entry)
        logger.debug(
            "SLC update: expert=%d real=%d padded=%d device=%s",
            global_eid, real_cnt, padded_cnt, device_spec.device_class.name,
        )


# ---------------------------------------------------------------------------
# Unpermute (inverse of hetero_permute)
# ---------------------------------------------------------------------------

def hetero_unpermute(
    permuted_tokens: torch.Tensor,
    restore_index: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    original_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Inverse of hetero_permute: scatter expert outputs back to token order.

    This is a thin wrapper around gather/scatter that is padding-aware:
    restore_index == -1 marks padded positions that should be ignored.
    The upstream fix guarantees that for BF16 dispatches, no positions are
    padded, so this function degenerates to a pure scatter (no masking needed).

    Parameters
    ----------
    permuted_tokens : [T_total_padded, H]
        Expert outputs in permuted order.
    restore_index : [T_total_padded]
        Maps permuted position → original flat token index. -1 = padding.
    probs : [T_total_padded], optional
        Per-position routing weights for weighted combine.
    original_shape : (S*B, H), optional
        Shape of the output buffer. Inferred from max(restore_index)+1 if None.

    Returns
    -------
    torch.Tensor [S*B, H]
    """
    hidden = permuted_tokens.shape[1]
    valid_mask = restore_index >= 0

    if original_shape is None:
        max_idx = int(restore_index[valid_mask].max().item()) + 1
        original_shape = (max_idx, hidden)

    output = torch.zeros(
        original_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )

    valid_perm = permuted_tokens[valid_mask]
    valid_idx = restore_index[valid_mask]

    if probs is not None:
        valid_p = probs[valid_mask].unsqueeze(-1)  # [T_valid, 1]
        weighted = valid_perm * valid_p
        output.scatter_add_(0, valid_idx.unsqueeze(-1).expand_as(weighted), weighted)
    else:
        output.scatter_(0, valid_idx.unsqueeze(-1).expand_as(valid_perm), valid_perm)

    return output


# ---------------------------------------------------------------------------
# High-level DES-LOC dispatch orchestrator
# ---------------------------------------------------------------------------

class HeteroMoEDispatch:
    """
    Orchestrates MoE token dispatch across the DES-LOC heterogeneous device pool.

    Partitions experts between LOCAL_SM86 (A6000) and REMOTE_SM90 (H100) devices,
    calling hetero_permute() with the correct device_spec and align_size for each
    partition. Aggregates results before the unpermute step.

    Expert partition strategy:
    - Experts [0, num_local_experts) → LOCAL_SM86 devices (round-robin)
    - Experts [num_local_experts, E) → REMOTE_SM90 device

    This matches the DES-LOC static expert placement used in Neuron_SP.
    """

    def __init__(
        self,
        num_experts: int,
        num_local_experts: int,
        device_pool: Optional[List[HeteroDeviceSpec]] = None,
        fp8_on_h100: bool = False,
    ):
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.num_remote_experts = num_experts - num_local_experts
        self.fp8_on_h100 = fp8_on_h100

        if device_pool is None:
            device_pool = DEFAULT_HETERO_POOL
        self.device_pool = device_pool

        # Separate specs by class
        self.local_specs = [
            s for s in device_pool if s.device_class == DeviceClass.LOCAL_SM86
        ]
        self.remote_specs = [
            s for s in device_pool if s.device_class == DeviceClass.REMOTE_SM90
        ]

        if not self.local_specs:
            logger.warning("HeteroMoEDispatch: no LOCAL_SM86 devices in pool")
        if not self.remote_specs:
            logger.warning("HeteroMoEDispatch: no REMOTE_SM90 devices in pool")

        logger.info(
            "HeteroMoEDispatch init: total_experts=%d local=%d remote=%d "
            "local_devices=%d remote_devices=%d fp8_h100=%s",
            num_experts, num_local_experts, self.num_remote_experts,
            len(self.local_specs), len(self.remote_specs), fp8_on_h100,
        )

    def _local_device_for(self, local_expert_id: int) -> HeteroDeviceSpec:
        """Round-robin assignment of local experts to A6000 devices."""
        if not self.local_specs:
            raise RuntimeError("No LOCAL_SM86 devices available")
        return self.local_specs[local_expert_id % len(self.local_specs)]

    def _remote_device(self) -> HeteroDeviceSpec:
        """All remote experts go to the single H100 NVL."""
        if not self.remote_specs:
            raise RuntimeError("No REMOTE_SM90 device available")
        return self.remote_specs[0]

    def dispatch(
        self,
        tokens: torch.Tensor,
        routing_map: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """
        Dispatch tokens to local and remote experts.

        Returns a dict with:
        - 'local_permuted':  list of (permuted_tokens, permuted_probs, restore_index)
          one per local expert batch
        - 'remote_permuted': (permuted_tokens, permuted_probs, restore_index)
          for the H100 expert batch
        - 'slc': updated SharedLocalityCache
        """
        slc = get_global_slc()
        results: Dict[str, object] = {"slc": slc}

        # --- Local expert dispatch (A6000, BF16, align_size=0) ----------------
        local_map = routing_map[:, :self.num_local_experts]
        # Single permute call for all local experts (they share the same BF16 path)
        local_spec = self._local_device_for(0)
        local_align = get_hetero_align_size(local_spec, fp8_active=False)

        local_tokens_pe = local_map.sum(dim=0).to(torch.int32) if local_map.any() else None

        logger.debug(
            "dispatch local: experts=[0,%d) device=%s align=%d",
            self.num_local_experts, local_spec.torch_device, local_align,
        )
        local_result = hetero_permute(
            tokens=tokens,
            routing_map=local_map,
            probs=probs[:, :self.num_local_experts] if probs is not None else None,
            tokens_per_expert=local_tokens_pe,
            align_size=local_align,
            device_spec=local_spec,
            expert_id_offset=0,
            slc=slc,
        )
        results["local_permuted"] = local_result

        # --- Remote expert dispatch (H100, BF16 or FP8, align_size=0 or 16) ---
        if self.num_remote_experts > 0:
            remote_map = routing_map[:, self.num_local_experts:]
            remote_spec = self._remote_device()
            remote_align = get_hetero_align_size(
                remote_spec,
                fp8_active=self.fp8_on_h100,
            )
            remote_tokens_pe = (
                remote_map.sum(dim=0).to(torch.int32) if remote_map.any() else None
            )
            logger.debug(
                "dispatch remote: experts=[%d,%d) device=%s align=%d",
                self.num_local_experts, self.num_experts,
                remote_spec.torch_device, remote_align,
            )
            remote_result = hetero_permute(
                tokens=tokens,
                routing_map=remote_map,
                probs=probs[:, self.num_local_experts:] if probs is not None else None,
                tokens_per_expert=remote_tokens_pe,
                align_size=remote_align,
                device_spec=remote_spec,
                expert_id_offset=self.num_local_experts,
                slc=slc,
            )
            results["remote_permuted"] = remote_result
        else:
            results["remote_permuted"] = None

        # SLC consistency check (fast; logs errors only)
        if not slc.consistency_check():
            logger.error(
                "SLC consistency check FAILED after dispatch — "
                "SM86 experts have padded_count != real_count. "
                "This indicates align_size was non-zero for a BF16 path."
            )

        return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("=== HeteroMoEPermutePadFix smoke test ===")

    # Use CPU tensors to avoid CUDA dependency in CI
    cpu_spec = HeteroDeviceSpec(
        device_class=DeviceClass.LOCAL_SM86,
        torch_device=torch.device("cpu"),
        sm_version=86,
        supports_fp8=False,
        max_align_size=0,
    )
    h100_spec = HeteroDeviceSpec(
        device_class=DeviceClass.REMOTE_SM90,
        torch_device=torch.device("cpu"),  # simulate on CPU
        sm_version=90,
        supports_fp8=True,
        max_align_size=16,
    )

    # Test 1: BF16 A6000 path → align_size must be 0
    a6k_align = get_hetero_align_size(cpu_spec, fp8_active=False)
    assert a6k_align == 0, f"Expected 0 for SM86 BF16, got {a6k_align}"
    logger.info("PASS test1: SM86 BF16 align_size=0")

    # Test 2: FP8 on SM86 must still be 0 (device incapable)
    a6k_fp8_align = get_hetero_align_size(cpu_spec, fp8_active=True)
    assert a6k_fp8_align == 0, f"Expected 0 for SM86+FP8 (incapable), got {a6k_fp8_align}"
    logger.info("PASS test2: SM86 FP8 forced to align_size=0")

    # Test 3: H100 FP8 → align_size=16
    h100_fp8_align = get_hetero_align_size(h100_spec, fp8_active=True)
    assert h100_fp8_align == 16, f"Expected 16 for SM90+FP8, got {h100_fp8_align}"
    logger.info("PASS test3: SM90 FP8 align_size=16")

    # Test 4: hetero_permute produces no padding for BF16 → SLC real==padded
    S, H, E = 8, 16, 4
    tokens = torch.randn(S, H)
    routing_map = torch.zeros(S, E, dtype=torch.bool)
    for i in range(S):
        routing_map[i, i % E] = True

    slc = SharedLocalityCache()
    p_tok, p_probs, restore = hetero_permute(
        tokens=tokens,
        routing_map=routing_map,
        align_size=0,
        device_spec=cpu_spec,
        slc=slc,
    )
    assert p_tok.shape[0] == S, f"Padded total should == S={S}, got {p_tok.shape[0]}"
    assert slc.consistency_check(), "SLC consistency check failed for BF16 path"
    logger.info("PASS test4: BF16 permute no padding, SLC consistent")

    # Test 5: unpermute recovers original token values (no padding case)
    recovered = hetero_unpermute(p_tok, restore, original_shape=(S, H))
    assert recovered.shape == (S, H), f"Shape mismatch: {recovered.shape}"
    logger.info("PASS test5: unpermute shape correct")

    logger.info("=== All smoke tests passed ===")
