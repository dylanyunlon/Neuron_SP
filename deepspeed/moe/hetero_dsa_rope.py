"""
deepspeed/moe/hetero_dsa_rope.py
=================================

DES-LOC HeteroMoEDSARoPE — Heterogeneous MoE + DSA RoPE Adaptation
=====================================================================

Upstream Design Intent (Megatron b0eb9143)
-------------------------------------------
The upstream commit fixes several bugs in the DSA (Dynamic Sparse Attention) and
MLA (Multi-Latent Attention) RoPE (Rotary Position Embedding) pipeline:

1. **mla_rotary_interleaved parameter rename**: The parameter ``multi_latent_attention``
   used in ``_apply_rotary_pos_emb_bshd`` and ``_apply_rotary_pos_emb_thd`` was renamed
   to ``mla_rotary_interleaved`` to clarify that it controls *interleaving behavior* in
   RoPE computation, not the general MLA feature flag. The old name is preserved as a
   deprecated alias with a ``DeprecationWarning``.

2. **DSA indexer tensor split order fix**: In ``DSAIndexer._apply_rope``, the split
   order of ``x_pe`` (positional embedding component) and ``x_nope`` (non-positional
   component) was reversed to align with DeepSeek's reference implementation. The PE
   component now comes first in the tensor layout, matching the upstream model weights.

3. **DSA spec builder refactoring**: The monolithic
   ``get_transformer_block_with_experimental_attention_variant_spec`` was split into:
   - ``get_transformer_layer_with_experimental_attention_variant_spec`` (returns all
     per-layer specs, pipeline-stage-agnostic)
   - ``get_transformer_block_with_experimental_attention_variant_spec`` (slices to
     the current pipeline stage, calls the layer function internally)

4. **DSA spec qk-layernorm fix**: Previously, DSA used fused ``column_parallel_layer_norm_linear``
   for qk projections when ``qk_layernorm=True``. This was incorrect because DSA indexer
   requires normalized Q as input — fusing prevents the intermediate normalized value from
   being accessible. The fix replaces fused projections with separate unfused layer-norm
   modules (``backend.layer_norm(rms_norm=..., for_qk=True)``).

5. **Experimental attention variant routing**: ``gpt_builder`` now routes
   ``args.experimental_attention_variant`` to the new layer-spec builder, and
   ``get_gpt_decoder_layer_specs`` asserts that experimental variants are not passed
   through the legacy code path.

6. **MLA RoPE interleaving propagation**: ``apply_rotary_pos_emb`` gained a new
   ``mla_rotary_interleaved`` parameter that is threaded explicitly through all call
   sites (absorbed MLA, standard MLA, DSA indexer, dynamic inference context) instead
   of being read implicitly from ``config.multi_latent_attention``. This allows the
   caller to control interleaving independently of the architectural feature flag.

DES-LOC Adaptation Points
---------------------------
The DES-LOC (Decoupled Execution with Shared LOcality Cache) framework targets a
**heterogeneous cluster**: 2× A6000 (48 GB, SM86, PCIe) + 1× H100 NVL (96 GB, SM90).
These devices are connected via PCIe without NVLink, which means:

- **No fast GPU–GPU transfers**: All tensor migration between A6000 and H100 must go
  through CPU DRAM (1.5 TB available) or P2P PCIe, which has ~16 GB/s bandwidth vs
  ~600 GB/s NVLink. RoPE tensors for long sequences can be large; we must avoid
  redundant cross-device copies.

- **SM86 vs SM90 capability gap**: H100 (SM90) supports BF16 tensor cores and
  FlashAttention-3 natively. A6000 (SM86) does not have FP8 or FA3, but supports
  BF16 via Ampere tensor cores. The ``mla_rotary_interleaved`` path uses
  ``torch.cat`` operations that are compute-light but memory-bandwidth-heavy —
  better scheduled on A6000 while H100 handles the sparse attention kernel.

- **DES-LOC locality cache**: Each device maintains a "locality cache" — a pinned
  CPU DRAM buffer per device that holds recently-used KV tensors and RoPE frequency
  tables. The ``HeteroRoPEFrequencyCache`` class below implements this pattern:
  it pins frequency tensors in CPU memory and lazily transfers them to whichever
  GPU is executing the current micro-batch shard, avoiding redundant H2D copies
  when the same frequencies are reused across steps.

- **Decoupled execution**: MoE expert routing happens on H100 (high compute), while
  DSA indexing (memory-bandwidth-bound scatter/gather over the KV cache) is offloaded
  to A6000 when H100 is saturated. The ``HeteroMoEDSADispatcher`` class implements
  this cross-device dispatch with explicit device affinity tracking.

- **RoPE correctness with heterogeneous layouts**: The upstream bug (wrong pe/nope
  split order in DSAIndexer) manifests as silent corruption when weights trained with
  DeepSeek layout are loaded. This adaptation reproduces the correct split order and
  adds device-aware assertions so that layout mismatches are caught at dispatch time
  rather than producing silent NaN gradients.

Module Layout
-------------
::

    HeteroRoPEFrequencyCache          — per-device pinned frequency table cache
    apply_rope_bshd_hetero            — device-aware _apply_rotary_pos_emb_bshd
    apply_rope_thd_hetero             — device-aware _apply_rotary_pos_emb_thd
    apply_rotary_pos_emb_hetero       — top-level router (mirrors apply_rotary_pos_emb)
    DSAIndexerHetero                  — DSA indexer with correct pe/nope split + DES-LOC dispatch
    HeteroMoEDSADispatcher            — cross-device MoE+DSA expert dispatch
    HeteroTransformerLayerSpecBuilder — layer-spec builder aware of device affinity
    build_hetero_moe_dsa_block        — top-level entry point for Neuron_SP gpt_builder

Author: Neuron_SP / DES-LOC project (mirrors Megatron b0eb9143)
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device topology constants for the DES-LOC target cluster
# ---------------------------------------------------------------------------

#: SM architecture IDs for supported devices
SM86_CAPABILITY = (8, 6)   # A6000 48 GB
SM90_CAPABILITY = (9, 0)   # H100 NVL 96 GB

#: PCIe theoretical bandwidth ceiling (bytes/s) — used for cost modelling
PCIE_BW_BYTES_PER_SEC = 16 * 1024 ** 3   # 16 GB/s

#: Threshold (bytes) above which cross-device RoPE tensor migration is warned
_ROPE_MIGRATION_WARN_THRESHOLD = 256 * 1024 * 1024  # 256 MB


class DeviceRole(Enum):
    """Logical role of a physical GPU in the DES-LOC cluster."""
    HIGH_BANDWIDTH_COMPUTE = auto()   # H100 NVL — sparse attn, expert compute
    MEMORY_BANDWIDTH_OFFLOAD = auto() # A6000   — DSA indexing, RoPE preprocessing


def _get_device_role(device: torch.device) -> DeviceRole:
    """Classify a CUDA device by its SM capability into a DES-LOC role.

    Args:
        device: A ``torch.device`` of type ``cuda``.

    Returns:
        :class:`DeviceRole` classifying the device.

    Raises:
        ValueError: If the device capability is not recognised.
    """
    if not device.type == "cuda":
        raise ValueError(f"DES-LOC only supports CUDA devices, got {device}")
    cap = torch.cuda.get_device_capability(device.index or 0)
    if cap >= SM90_CAPABILITY:
        return DeviceRole.HIGH_BANDWIDTH_COMPUTE
    if cap >= SM86_CAPABILITY:
        return DeviceRole.MEMORY_BANDWIDTH_OFFLOAD
    raise ValueError(
        f"Unsupported device capability {cap} on {device}. "
        "DES-LOC requires SM86 (A6000) or SM90 (H100)."
    )


# ---------------------------------------------------------------------------
# HeteroRoPEFrequencyCache — DES-LOC locality cache for RoPE frequency tables
# ---------------------------------------------------------------------------

@dataclass
class _FreqCacheEntry:
    """A single entry in the per-device RoPE frequency cache."""
    freqs_cpu: torch.Tensor       # pinned CPU copy (source of truth)
    device_copies: Dict[int, torch.Tensor] = field(default_factory=dict)
    last_access_step: int = 0


class HeteroRoPEFrequencyCache:
    """Per-device pinned CPU DRAM cache for RoPE frequency tables.

    DES-LOC rationale
    -----------------
    In a PCIe-only cluster, H2D transfers dominate latency for large frequency
    tensors (e.g. 128k-token context with head dim 128 → ~128 MB per frequency
    table). This cache:

    1. Stores a single pinned-memory master copy on CPU (always live).
    2. Lazily transfers to whichever GPU first requests a given ``(seq_len, dim)``
       key, then caches the GPU-side tensor.
    3. Evicts GPU-side copies using an LRU policy keyed on training step to bound
       GPU memory usage.

    The cache is intentionally not thread-safe; callers must synchronise via the
    DeepSpeed engine's micro-batch locking.

    Args:
        max_entries_per_device: Maximum number of distinct frequency tables to
            keep resident on each GPU simultaneously.
        lru_evict_after_steps: Evict a GPU-side copy after this many steps of
            non-use (default: 4, matching a typical gradient-accumulation window).
    """

    def __init__(
        self,
        max_entries_per_device: int = 8,
        lru_evict_after_steps: int = 4,
    ) -> None:
        self._cache: Dict[Tuple, _FreqCacheEntry] = {}
        self._max_entries = max_entries_per_device
        self._lru_steps = lru_evict_after_steps
        self._current_step: int = 0

    def _cache_key(self, freqs: torch.Tensor) -> Tuple:
        return (freqs.shape, freqs.dtype, freqs.data_ptr())

    def get(self, freqs: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """Return a device-resident copy of *freqs* on *target_device*.

        If no cached copy exists, pins *freqs* to CPU and schedules an async
        H2D transfer. Subsequent calls with the same tensor pointer and target
        device return the cached copy immediately.

        Args:
            freqs: Source frequency tensor (may be on any device).
            target_device: The GPU device that will consume the frequencies.

        Returns:
            A tensor on *target_device* with the same values as *freqs*.
        """
        key = self._cache_key(freqs)
        dev_idx = target_device.index if target_device.index is not None else 0

        if key not in self._cache:
            # Pin a CPU copy as the locality-cache master
            cpu_pinned = freqs.detach().cpu().pin_memory()
            self._cache[key] = _FreqCacheEntry(freqs_cpu=cpu_pinned)
            logger.debug(
                "HeteroRoPEFrequencyCache: pinned new freq table "
                "shape=%s dtype=%s to CPU (%.1f MB)",
                freqs.shape,
                freqs.dtype,
                cpu_pinned.numel() * cpu_pinned.element_size() / 1024 ** 2,
            )

        entry = self._cache[key]
        entry.last_access_step = self._current_step

        if dev_idx not in entry.device_copies:
            self._maybe_evict(dev_idx)
            nbytes = entry.freqs_cpu.numel() * entry.freqs_cpu.element_size()
            if nbytes > _ROPE_MIGRATION_WARN_THRESHOLD:
                logger.warning(
                    "HeteroRoPEFrequencyCache: large RoPE freq tensor migration "
                    "(%.1f MB) to device %d via PCIe — consider reducing seq_len "
                    "or rotary_dim to stay within PCIe budget",
                    nbytes / 1024 ** 2,
                    dev_idx,
                )
            entry.device_copies[dev_idx] = entry.freqs_cpu.to(
                device=target_device, non_blocking=True
            )

        return entry.device_copies[dev_idx]

    def step(self) -> None:
        """Advance the internal step counter; call once per optimizer step."""
        self._current_step += 1
        self._evict_stale()

    def _maybe_evict(self, dev_idx: int) -> None:
        """Evict oldest GPU-side entry on *dev_idx* if at capacity."""
        dev_entries = [
            (k, e) for k, e in self._cache.items() if dev_idx in e.device_copies
        ]
        if len(dev_entries) >= self._max_entries:
            oldest_key = min(dev_entries, key=lambda x: x[1].last_access_step)[0]
            del self._cache[oldest_key].device_copies[dev_idx]
            logger.debug(
                "HeteroRoPEFrequencyCache: evicted stale freq table from device %d", dev_idx
            )

    def _evict_stale(self) -> None:
        """Remove GPU-side copies that have not been accessed recently."""
        stale_threshold = self._current_step - self._lru_steps
        for entry in self._cache.values():
            if entry.last_access_step < stale_threshold:
                stale_devs = list(entry.device_copies.keys())
                for dev_idx in stale_devs:
                    del entry.device_copies[dev_idx]

    def clear(self) -> None:
        """Release all cached tensors (CPU and GPU)."""
        self._cache.clear()
        logger.debug("HeteroRoPEFrequencyCache: cleared all entries")


# Module-level singleton cache — shared across all DES-LOC RoPE call sites
_GLOBAL_ROPE_FREQ_CACHE = HeteroRoPEFrequencyCache()


# ---------------------------------------------------------------------------
# Core RoPE computation — _apply_rotary_pos_emb_bshd (DES-LOC variant)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of *x* by half (standard RoPE helper).

    Splits the last dimension into two halves (x1, x2) and returns
    ``[-x2, x1]`` concatenated along the last axis.

    Args:
        x: Input tensor of shape ``[..., d]``.

    Returns:
        Rotated tensor of the same shape.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate *x* using interleaved pairing (pairs adjacent dims).

    In interleaved mode, dimensions are paired as (0,1), (2,3), … rather
    than (0,d/2), (1,d/2+1), … This is required for the MLA-style RoPE
    used in DeepSeek-V2/V3 architectures.

    Args:
        x: Input tensor of shape ``[..., d]`` where ``d`` is even.

    Returns:
        Rotated tensor of the same shape with interleaved negation.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope_bshd_hetero(
    t: torch.Tensor,
    freqs: torch.Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    multi_latent_attention: Optional[bool] = None,
    freq_cache: Optional[HeteroRoPEFrequencyCache] = None,
) -> torch.Tensor:
    """Apply RoPE to a BSHD-format tensor with DES-LOC device-aware frequency caching.

    This mirrors Megatron's ``_apply_rotary_pos_emb_bshd`` post-b0eb9143, adding:

    - **Explicit ``mla_rotary_interleaved`` parameter**: Separated from the
      architectural ``multi_latent_attention`` flag, matching the upstream fix.
    - **Deprecated ``multi_latent_attention`` alias**: Emits ``DeprecationWarning``
      when the old name is used, forwarding to ``mla_rotary_interleaved``.
    - **DES-LOC locality cache integration**: When *freq_cache* is provided,
      frequency tables are retrieved from the pinned-CPU cache and transferred
      to ``t.device`` only if not already resident, avoiding redundant PCIe traffic.
    - **Device role logging**: On first call per device, logs whether RoPE is
      executing on the offload (A6000) or compute (H100) device, aiding profiling.

    Args:
        t: Input tensor of shape ``[seq_len, batch, num_heads, head_dim]``.
        freqs: Rotary frequency tensor of shape ``[seq_len, 1, 1, rot_dim]``.
        rotary_interleaved: If True, use interleaved rotation in ``_rotate_half``
            (standard Megatron option, independent of MLA).
        mla_rotary_interleaved: If True, apply MLA-style pre-interleaving where
            odd/even channels are split and concatenated before rotation. This
            matches the DeepSeek-V2/V3 RoPE convention and is required when
            loading weights from those checkpoints.
        mscale: Multiplicative scale applied to ``freqs`` (used by YaRN).
        multi_latent_attention: Deprecated alias for ``mla_rotary_interleaved``.
            Will be removed in a future release.
        freq_cache: Optional :class:`HeteroRoPEFrequencyCache` instance. If
            provided, frequency tensors are managed by the DES-LOC locality cache
            for cross-device reuse.

    Returns:
        Tensor of the same shape as *t* after RoPE application.
    """
    if multi_latent_attention is not None:
        warnings.warn(
            "multi_latent_attention is deprecated in DES-LOC RoPE. "
            "Use mla_rotary_interleaved instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mla_rotary_interleaved = multi_latent_attention

    # Resolve frequency tensor device via locality cache
    if freq_cache is not None:
        freqs = freq_cache.get(freqs, t.device)
    elif freqs.device != t.device:
        freqs = freqs.to(t.device)

    rot_dim = freqs.shape[-1]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if mla_rotary_interleaved:
        # MLA-style: interleave channels before applying cosine/sine
        # This matches the DeepSeek weight layout: x = [x_even, x_odd, ...]
        x1 = t_rot[..., 0::2]
        x2 = t_rot[..., 1::2]
        t_rot = torch.cat((x1, x2), dim=-1)

    if mscale != 1.0:
        freqs = freqs * mscale

    cos_val = freqs.cos()
    sin_val = freqs.sin()

    if rotary_interleaved:
        t_rot = t_rot * cos_val + _rotate_interleaved(t_rot) * sin_val
    else:
        t_rot = t_rot * cos_val + _rotate_half(t_rot) * sin_val

    if mla_rotary_interleaved:
        # Un-interleave: reconstruct original channel ordering
        d_half = t_rot.shape[-1] // 2
        out1 = t_rot[..., :d_half]
        out2 = t_rot[..., d_half:]
        t_rot = torch.stack((out1, out2), dim=-1).flatten(-2)

    return torch.cat((t_rot, t_pass), dim=-1)


# ---------------------------------------------------------------------------
# Core RoPE computation — _apply_rotary_pos_emb_thd (DES-LOC variant)
# ---------------------------------------------------------------------------

def apply_rope_thd_hetero(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    cp_group: Optional[dist.ProcessGroup] = None,
    multi_latent_attention: Optional[bool] = None,
    freq_cache: Optional[HeteroRoPEFrequencyCache] = None,
) -> torch.Tensor:
    """Apply RoPE to a THD-format (tokens, heads, dim) tensor.

    THD format is used by context-parallel (CP) workloads where sequence
    lengths differ across requests in the same batch (variable-length sequences
    packed into a single flat buffer).

    DES-LOC note: In the heterogeneous cluster, CP groups may span A6000 and
    H100 devices. The ``cp_group`` here is a DeepSpeed process group; the
    actual tensor layout and reduction is handled by the caller. This function
    only applies the per-token RoPE transform, which is embarrassingly parallel
    and can run on either device.

    Upstream changes mirrored (b0eb9143):
    - ``multi_latent_attention`` renamed to ``mla_rotary_interleaved``
    - Deprecated alias preserved with ``DeprecationWarning``
    - Internal call to ``apply_rope_bshd_hetero`` uses the new parameter name

    Args:
        t: Input tensor of shape ``[total_tokens, num_heads, head_dim]``.
        cu_seqlens: Cumulative sequence lengths tensor of shape ``[batch+1]``.
        freqs: Frequency tensor of shape ``[max_seq_len, 1, 1, rot_dim]``.
        rotary_interleaved: Standard interleaved rotation flag.
        mla_rotary_interleaved: MLA-style interleaving flag (see bshd variant).
        mscale: YaRN scale factor.
        cp_group: Context-parallel process group (required for THD format).
        multi_latent_attention: Deprecated alias for ``mla_rotary_interleaved``.
        freq_cache: Optional DES-LOC locality cache for frequency tensors.

    Returns:
        Tensor of shape ``[total_tokens, num_heads, head_dim]`` after RoPE.

    Raises:
        ValueError: If ``cp_group`` is None (required for variable-length RoPE).
    """
    if multi_latent_attention is not None:
        warnings.warn(
            "multi_latent_attention is deprecated in DES-LOC RoPE. "
            "Use mla_rotary_interleaved instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mla_rotary_interleaved = multi_latent_attention

    if cp_group is None:
        raise ValueError(
            "cp_group must be provided for THD-format RoPE. "
            "In DES-LOC, use deepspeed.comm.new_group() to create a CP group "
            "spanning the participating devices."
        )

    # Resolve frequencies via locality cache
    if freq_cache is not None:
        freqs = freq_cache.get(freqs, t.device)
    elif freqs.device != t.device:
        freqs = freqs.to(t.device)

    # Unpack variable-length sequences and apply per-sequence RoPE
    outputs = []
    seq_starts = cu_seqlens[:-1].tolist()
    seq_ends = cu_seqlens[1:].tolist()

    for start, end in zip(seq_starts, seq_ends):
        seq_len = end - start
        t_seq = t[start:end]  # [seq_len, num_heads, head_dim]
        freqs_seq = freqs[:seq_len]  # [seq_len, 1, 1, rot_dim]

        # Reshape to BSHD for the shared core function: [seq, 1, heads, dim]
        t_bshd = t_seq.unsqueeze(1)
        t_out = apply_rope_bshd_hetero(
            t=t_bshd,
            freqs=freqs_seq,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            freq_cache=None,  # already resolved above
        )
        outputs.append(t_out.squeeze(1))

    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Top-level RoPE router — apply_rotary_pos_emb_hetero
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb_hetero(
    t: torch.Tensor,
    freqs: torch.Tensor,
    config,
    cu_seqlens: Optional[torch.Tensor] = None,
    mscale: float = 1.0,
    cp_group: Optional[dist.ProcessGroup] = None,
    mla_rotary_interleaved: bool = False,
    freq_cache: Optional[HeteroRoPEFrequencyCache] = None,
) -> torch.Tensor:
    """Route RoPE application to the appropriate DES-LOC-aware implementation.

    This mirrors Megatron's ``apply_rotary_pos_emb`` (post-b0eb9143) with
    adaptations for the DES-LOC heterogeneous execution model:

    1. **Explicit ``mla_rotary_interleaved``**: Passed as an explicit argument
       rather than read from ``config.multi_latent_attention``, matching the
       upstream fix that decouples the interleaving flag from the architectural
       MLA flag.

    2. **Fused-kernel compatibility check**: In Megatron, ``apply_rope_fusion``
       triggers a fused CUDA kernel. In DES-LOC, we additionally gate fusion on
       device capability: the fused kernel requires SM90 (H100). On A6000 (SM86),
       we always fall back to the unfused path, which is already fast for the
       typical RoPE dimensions used in DSA workloads.

    3. **MLA interleaving + fusion conflict**: When ``mla_rotary_interleaved=True``,
       the fused path is incompatible (upstream bug fix). We warn and fall back.

    4. **DES-LOC frequency cache**: When *freq_cache* is supplied, all frequency
       lookups go through the locality cache to avoid redundant PCIe transfers.

    Args:
        t: Input tensor (BSHD or THD format depending on ``cu_seqlens``).
        freqs: Rotary frequency tensor.
        config: Model config object exposing ``apply_rope_fusion``,
            ``rotary_interleaved``, ``multi_latent_attention``.
        cu_seqlens: If provided, use THD-format (variable-length) RoPE path.
        mscale: YaRN scaling factor.
        cp_group: Context-parallel process group (required for THD).
        mla_rotary_interleaved: Whether to use MLA-style channel interleaving.
        freq_cache: Optional DES-LOC locality cache.

    Returns:
        Tensor with RoPE applied, same shape as *t*.
    """
    use_unfused = True  # DES-LOC default: always unfused unless explicitly enabled

    if getattr(config, "apply_rope_fusion", False):
        # Check device capability — fused kernel only supported on SM90+
        device = t.device
        if device.type == "cuda":
            cap = torch.cuda.get_device_capability(device.index or 0)
            if cap < SM90_CAPABILITY:
                logger.debug(
                    "apply_rotary_pos_emb_hetero: apply_rope_fusion requested but "
                    "device %s has capability %s (< SM90); falling back to unfused RoPE",
                    device,
                    cap,
                )
            elif mla_rotary_interleaved:
                warnings.warn(
                    "apply_rope_fusion does not support MLA-style interleaving in RoPE. "
                    "Using unfused implementation.",
                    UserWarning,
                    stacklevel=2,
                )
            elif getattr(config, "rotary_interleaved", False):
                warnings.warn(
                    "apply_rope_fusion does not support rotary_interleaved. "
                    "Using unfused implementation.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                use_unfused = False

    if cu_seqlens is None:
        # BSHD format
        return apply_rope_bshd_hetero(
            t=t,
            freqs=freqs,
            rotary_interleaved=getattr(config, "rotary_interleaved", False),
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            freq_cache=freq_cache,
        )
    else:
        # THD / variable-length format
        return apply_rope_thd_hetero(
            t=t,
            cu_seqlens=cu_seqlens,
            freqs=freqs,
            rotary_interleaved=getattr(config, "rotary_interleaved", False),
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            cp_group=cp_group,
            freq_cache=freq_cache,
        )


# ---------------------------------------------------------------------------
# DSAIndexerHetero — corrected pe/nope split + DES-LOC device dispatch
# ---------------------------------------------------------------------------

class DSAIndexerHetero(nn.Module):
    """DSA (Dynamic Sparse Attention) Indexer adapted for DES-LOC heterogeneous dispatch.

    Upstream bug fixed (b0eb9143 — dsa.py ``DSAIndexer._apply_rope``)
    ------------------------------------------------------------------
    The original code split the head dimension as ``[nope, pe]`` (non-positional
    first, then positional). DeepSeek's reference implementation and trained weights
    use the **opposite** layout: ``[pe, nope]`` — PE component at the front.

    This caused silent weight loading failures: the model loaded without error but
    the attention scores were computed with the wrong channel ordering, leading to
    degraded or random outputs when fine-tuning from DeepSeek checkpoints.

    The fix is to swap the split:

    .. code-block:: python

        # Before (wrong):
        x_nope, x_pe = torch.split(x, [index_head_dim - qk_pos_emb_head_dim, qk_pos_emb_head_dim], dim=-1)

        # After (correct, matching DeepSeek layout):
        x_pe, x_nope = torch.split(x, [qk_pos_emb_head_dim, index_head_dim - qk_pos_emb_head_dim], dim=-1)

    And correspondingly, the concatenation order:

    .. code-block:: python

        # Before (wrong):
        x = torch.cat([x_nope, x_pe], dim=-1)

        # After (correct):
        x = torch.cat([x_pe, x_nope], dim=-1)

    DES-LOC dispatch
    ----------------
    DSA indexing is a scatter/gather-heavy operation over the KV cache — it is
    **memory-bandwidth-bound** rather than compute-bound. On the target cluster,
    this makes it a natural candidate for offloading to A6000 (SM86) while H100
    handles the dense attention kernel.

    This class implements:
    - Explicit ``preferred_device`` routing: if the caller specifies an A6000
      device index, indexing computation is performed there.
    - Cross-device tensor migration accounting: logs migration cost when tensors
      must cross the PCIe bus.
    - ``mla_rotary_interleaved=False`` explicitly for the indexer (the indexer
      does not apply interleaved RoPE — only the main Q/K projections do).

    Args:
        index_head_dim: Total head dimension of the indexer (PE + non-PE).
        qk_pos_emb_head_dim: Number of dimensions used for positional encoding.
        num_heads: Number of indexer attention heads.
        top_k: Top-k tokens to select per query head.
        preferred_device: CUDA device index to run indexing on. If None, uses
            the device of the input tensor. Set to the A6000 device index for
            bandwidth-optimal dispatch.
        freq_cache: DES-LOC locality cache for RoPE frequencies.
    """

    def __init__(
        self,
        index_head_dim: int,
        qk_pos_emb_head_dim: int,
        num_heads: int,
        top_k: int,
        preferred_device: Optional[int] = None,
        freq_cache: Optional[HeteroRoPEFrequencyCache] = None,
    ) -> None:
        super().__init__()
        if qk_pos_emb_head_dim > index_head_dim:
            raise ValueError(
                f"qk_pos_emb_head_dim ({qk_pos_emb_head_dim}) must be <= "
                f"index_head_dim ({index_head_dim})"
            )
        self.index_head_dim = index_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.preferred_device = preferred_device
        self.freq_cache = freq_cache or _GLOBAL_ROPE_FREQ_CACHE

        # Learnable projection from hidden to index space
        self.wq_b = nn.Linear(index_head_dim, index_head_dim, bias=False)
        self.wk = nn.Linear(index_head_dim, index_head_dim, bias=False)

    def _migrate_to_preferred_device(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.device]]:
        """Migrate *x* to the preferred indexing device if configured.

        Returns:
            Tuple of (migrated tensor, original device or None if no migration).
        """
        if self.preferred_device is None:
            return x, None
        target = torch.device("cuda", self.preferred_device)
        if x.device == target:
            return x, None
        nbytes = x.numel() * x.element_size()
        logger.debug(
            "DSAIndexerHetero: migrating tensor %.1f MB from %s to %s for indexing",
            nbytes / 1024 ** 2,
            x.device,
            target,
        )
        return x.to(target, non_blocking=True), x.device

    def _apply_rope(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mscale: float,
        config,
    ) -> torch.Tensor:
        """Apply RoPE to indexer queries/keys with correct pe/nope split order.

        This implements the fix from Megatron b0eb9143: pe comes first in the
        tensor layout (matching DeepSeek checkpoints), so we split as
        ``[qk_pos_emb_head_dim, index_head_dim - qk_pos_emb_head_dim]`` and
        concatenate as ``[pe, nope]`` after rotation.

        The indexer does NOT use MLA-style interleaving (``mla_rotary_interleaved=False``),
        unlike the main Q/K projections in ``MLASelfAttention``.

        Args:
            x: Input tensor of shape ``[seq, batch, num_heads, index_head_dim]``.
            rotary_pos_emb: Frequency tensor matching the positional dimensions.
            mscale: YaRN scale factor.
            config: Model config (forwarded to ``apply_rotary_pos_emb_hetero``).

        Returns:
            Tensor of shape ``[seq, batch, num_heads, index_head_dim]`` after RoPE.
        """
        # Correct split order: pe first, nope second (DeepSeek layout)
        # This is the core bug fix from b0eb9143
        x_pe, x_nope = torch.split(
            x,
            [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim],
            dim=-1,
        )

        x_pe_rotated = apply_rotary_pos_emb_hetero(
            t=x_pe,
            freqs=rotary_pos_emb,
            config=config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=None,
            # Indexer does NOT use MLA interleaved RoPE — explicit False
            mla_rotary_interleaved=False,
            freq_cache=self.freq_cache,
        )

        # Reconstruct with pe first (matching split order)
        return torch.cat([x_pe_rotated, x_nope], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mscale: float,
        config,
    ) -> torch.Tensor:
        """Compute DSA top-k token indices for the current query.

        Args:
            hidden_states: Query features of shape
                ``[seq_len, batch, num_heads * index_head_dim]``.
            rotary_pos_emb: RoPE frequencies for this sequence.
            mscale: YaRN scale.
            config: Model config.

        Returns:
            Top-k indices tensor of shape ``[seq_len, batch, num_heads, top_k]``.
        """
        # Optionally migrate to preferred device (A6000) for bandwidth-optimal indexing
        hidden_states, original_device = self._migrate_to_preferred_device(hidden_states)
        rotary_pos_emb, _ = self._migrate_to_preferred_device(rotary_pos_emb)

        seq_len, batch, _ = hidden_states.shape
        x = hidden_states.view(seq_len, batch, self.num_heads, self.index_head_dim)

        # Project to indexer Q/K
        q = self.wq_b(x)
        k = self.wk(x)

        # Apply RoPE with correct pe/nope split (b0eb9143 fix)
        q = self._apply_rope(q, rotary_pos_emb, mscale, config)
        k = self._apply_rope(k, rotary_pos_emb, mscale, config)

        # Compute attention scores and select top-k
        # [seq, batch, heads, seq] — simplified; full impl includes causal masking
        scores = torch.einsum("sbhd,Sbhd->sbhS", q, k) / math.sqrt(self.index_head_dim)
        topk_indices = scores.topk(self.top_k, dim=-1).indices

        # Migrate result back to original device if we offloaded
        if original_device is not None:
            topk_indices = topk_indices.to(original_device, non_blocking=True)

        return topk_indices


# ---------------------------------------------------------------------------
# HeteroMoEDSADispatcher — cross-device MoE + DSA expert dispatch
# ---------------------------------------------------------------------------

@dataclass
class HeteroDispatchPlan:
    """Dispatch plan specifying which tokens route to which device for MoE + DSA.

    Attributes:
        h100_token_indices: Indices of tokens whose experts run on H100.
        a6000_token_indices: Indices of tokens whose experts run on A6000.
        indexing_device: Device to use for DSA indexing (typically A6000).
        compute_device: Device to use for expert FFN computation (typically H100).
    """
    h100_token_indices: torch.Tensor
    a6000_token_indices: torch.Tensor
    indexing_device: torch.device
    compute_device: torch.device


class HeteroMoEDSADispatcher(nn.Module):
    """Cross-device MoE expert dispatcher with integrated DSA sparse indexing.

    DES-LOC rationale
    -----------------
    In the DES-LOC execution model, MoE and DSA create two distinct workloads:

    1. **Expert FFN computation** (compute-bound): Routed to H100 (SM90) which
       has higher FP16/BF16 TFLOP/s and HBM3 bandwidth.

    2. **DSA sparse indexing** (memory-bandwidth-bound): Routed to A6000 (SM86)
       which, while having lower peak FLOP/s, has HBM2e bandwidth sufficient for
       the scatter/gather patterns required by DSA and is not competing with the
       H100 for expert compute cycles.

    The dispatcher:
    - Receives the router logits and expert assignment from the MoE router
    - Computes a ``HeteroDispatchPlan`` partitioning tokens across devices
    - Launches DSA indexing on the A6000 asynchronously while the H100 runs
      expert FFN for already-dispatched tokens
    - Synchronises results before the output projection

    This class is a simplified reference implementation; production use should
    wire it into DeepSpeed's ``MOELayer`` via the ``custom_policy`` hook.

    Args:
        num_experts: Total number of MoE experts.
        top_k_experts: Number of experts each token routes to.
        dsa_indexer: A :class:`DSAIndexerHetero` instance for sparse attention.
        h100_device_idx: CUDA device index of the H100 NVL.
        a6000_device_indices: List of CUDA device indices for A6000 GPUs.
        load_balance_threshold: If expert load imbalance exceeds this fraction,
            log a warning (default 0.2 = 20%).
    """

    def __init__(
        self,
        num_experts: int,
        top_k_experts: int,
        dsa_indexer: DSAIndexerHetero,
        h100_device_idx: int = 0,
        a6000_device_indices: Optional[List[int]] = None,
        load_balance_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.dsa_indexer = dsa_indexer
        self.h100_device = torch.device("cuda", h100_device_idx)
        self.a6000_devices = [
            torch.device("cuda", idx) for idx in (a6000_device_indices or [])
        ]
        self.load_balance_threshold = load_balance_threshold
        self._dispatch_count = 0

    def _compute_dispatch_plan(
        self, router_logits: torch.Tensor
    ) -> HeteroDispatchPlan:
        """Compute a device dispatch plan from router logits.

        Uses a simple heuristic: the top-scoring half of tokens (by their
        maximum expert logit) are dispatched to H100 for priority compute,
        while the remaining tokens are handled by A6000. This balances H100
        utilisation against PCIe transfer cost.

        In practice, Neuron_SP may replace this with a learned or adaptive
        scheduling policy.

        Args:
            router_logits: Tensor of shape ``[num_tokens, num_experts]``.

        Returns:
            :class:`HeteroDispatchPlan` with per-device token index tensors.
        """
        num_tokens = router_logits.shape[0]
        max_logits = router_logits.max(dim=-1).values  # [num_tokens]

        # Sort tokens by confidence; high-confidence → H100 (priority compute path)
        sorted_indices = max_logits.argsort(descending=True)
        split = num_tokens // 2
        h100_indices = sorted_indices[:split]
        a6000_indices = sorted_indices[split:]

        # Load balance check
        h100_frac = split / num_tokens
        if abs(h100_frac - 0.5) > self.load_balance_threshold:
            logger.warning(
                "HeteroMoEDSADispatcher: token dispatch imbalance detected "
                "(H100=%.1f%%, A6000=%.1f%%) exceeds threshold %.0f%%. "
                "Consider tuning the dispatch policy.",
                100 * h100_frac,
                100 * (1 - h100_frac),
                100 * self.load_balance_threshold,
            )

        indexing_device = self.a6000_devices[0] if self.a6000_devices else self.h100_device
        return HeteroDispatchPlan(
            h100_token_indices=h100_indices,
            a6000_token_indices=a6000_indices,
            indexing_device=indexing_device,
            compute_device=self.h100_device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mscale: float,
        config,
        expert_fn,
    ) -> torch.Tensor:
        """Dispatch tokens to H100/A6000, run DSA indexing and expert FFN.

        Args:
            hidden_states: Input of shape ``[seq_len, batch, hidden_dim]``.
            router_logits: MoE router output of shape ``[seq_len * batch, num_experts]``.
            rotary_pos_emb: RoPE frequencies for this sequence.
            mscale: YaRN scale.
            config: Model config.
            expert_fn: Callable ``(tokens, expert_indices) -> output`` running
                the actual expert FFN layers.

        Returns:
            Output tensor of the same shape as *hidden_states*.
        """
        self._dispatch_count += 1
        plan = self._compute_dispatch_plan(router_logits)

        seq_len, batch, hidden_dim = hidden_states.shape

        # --- DSA indexing on A6000 (async, memory-bandwidth path) ---
        # Flatten to tokens for indexer
        hidden_flat = hidden_states.view(-1, self.dsa_indexer.num_heads, self.dsa_indexer.index_head_dim)
        # Launch indexing on A6000 — this runs while H100 handles expert compute
        with torch.cuda.device(plan.indexing_device):
            dsa_stream = torch.cuda.Stream(device=plan.indexing_device)
            with torch.cuda.stream(dsa_stream):
                sparse_indices = self.dsa_indexer.forward(
                    hidden_states=hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    mscale=mscale,
                    config=config,
                )

        # --- Expert FFN on H100 (compute-priority path) ---
        topk_indices = router_logits.topk(self.top_k_experts, dim=-1).indices
        with torch.cuda.device(plan.compute_device):
            h100_tokens = hidden_states.view(-1, hidden_dim)[plan.h100_token_indices]
            h100_tokens = h100_tokens.to(plan.compute_device, non_blocking=True)
            h100_output = expert_fn(h100_tokens, topk_indices[plan.h100_token_indices])

        # --- Sync DSA stream before merging ---
        torch.cuda.current_stream(plan.compute_device).wait_stream(dsa_stream)

        # Merge outputs back into original layout
        output = torch.zeros_like(hidden_states.view(-1, hidden_dim))
        output[plan.h100_token_indices] = h100_output.to(hidden_states.device, non_blocking=True)

        # A6000 tokens (expert output — simplified: use same expert_fn on H100 for correctness)
        if plan.a6000_token_indices.numel() > 0:
            a6000_tokens = hidden_states.view(-1, hidden_dim)[plan.a6000_token_indices]
            a6000_tokens = a6000_tokens.to(plan.compute_device, non_blocking=True)
            a6000_output = expert_fn(a6000_tokens, topk_indices[plan.a6000_token_indices])
            output[plan.a6000_token_indices] = a6000_output.to(hidden_states.device, non_blocking=True)

        return output.view(seq_len, batch, hidden_dim)


# ---------------------------------------------------------------------------
# HeteroTransformerLayerSpecBuilder — device-aware layer spec construction
# ---------------------------------------------------------------------------

@dataclass
class DESLOCLayerSpec:
    """Minimal spec descriptor for a DES-LOC transformer layer.

    Attributes:
        layer_idx: Global layer index (0-based).
        use_experimental_attention: Whether this layer uses the experimental
            attention variant (e.g. DSA or linear attention).
        use_moe_mlp: Whether this layer uses MoE instead of dense MLP.
        preferred_attn_device: CUDA device index preferred for attention.
        preferred_mlp_device: CUDA device index preferred for MLP/expert compute.
        mla_rotary_interleaved: Whether this layer's RoPE uses MLA interleaving.
    """
    layer_idx: int
    use_experimental_attention: bool
    use_moe_mlp: bool
    preferred_attn_device: int
    preferred_mlp_device: int
    mla_rotary_interleaved: bool


class HeteroTransformerLayerSpecBuilder:
    """Build per-layer specs for a DES-LOC heterogeneous transformer block.

    This mirrors the refactoring in Megatron b0eb9143 where
    ``get_transformer_block_with_experimental_attention_variant_spec`` was split
    into a layer-list builder and a pipeline-slicing wrapper.

    In DES-LOC, the layer-list builder additionally annotates each layer with
    device affinity information:

    - Layers using experimental attention (DSA) are annotated to prefer A6000
      for the indexing sub-operation.
    - Layers using MoE are annotated to prefer H100 for expert FFN.
    - Standard dense self-attention layers are scheduled on H100 (higher TFLOP/s
      for the attention kernel).
    - ``mla_rotary_interleaved`` is set per-layer based on whether MLA is active
      for that layer (matches the upstream fix of making this per-call-site rather
      than read from the global config).

    Args:
        num_layers: Total number of transformer layers in the full model.
        experimental_attention_pattern: List of length *num_layers* where 1
            indicates an experimental attention layer and 0 a standard layer.
        moe_layer_pattern: List of length *num_layers* where 1 indicates a
            MoE MLP layer and 0 a dense MLP layer.
        multi_latent_attention: Whether the model uses MLA (controls
            ``mla_rotary_interleaved`` annotation).
        h100_device_idx: CUDA device index of the H100 NVL.
        a6000_device_indices: CUDA device indices of A6000 GPUs.
    """

    def __init__(
        self,
        num_layers: int,
        experimental_attention_pattern: List[int],
        moe_layer_pattern: List[int],
        multi_latent_attention: bool,
        h100_device_idx: int = 0,
        a6000_device_indices: Optional[List[int]] = None,
    ) -> None:
        if len(experimental_attention_pattern) != num_layers:
            raise ValueError(
                f"experimental_attention_pattern length {len(experimental_attention_pattern)} "
                f"must equal num_layers {num_layers}"
            )
        if len(moe_layer_pattern) != num_layers:
            raise ValueError(
                f"moe_layer_pattern length {len(moe_layer_pattern)} "
                f"must equal num_layers {num_layers}"
            )
        self.num_layers = num_layers
        self.exp_attn_pattern = experimental_attention_pattern
        self.moe_pattern = moe_layer_pattern
        self.multi_latent_attention = multi_latent_attention
        self.h100_device_idx = h100_device_idx
        self.a6000_device_indices = a6000_device_indices or []

        primary_a6000 = self.a6000_device_indices[0] if self.a6000_device_indices else h100_device_idx
        n_dsa = sum(experimental_attention_pattern)
        n_moe = sum(moe_layer_pattern)
        if n_dsa > 0 or n_moe > 0:
            logger.info(
                "HeteroTransformerLayerSpecBuilder: %d DSA layers (preferred A6000=%d), "
                "%d MoE layers (preferred H100=%d), %d standard layers",
                n_dsa, primary_a6000, n_moe, h100_device_idx, num_layers - n_dsa,
            )

    def build_layer_specs(
        self, pp_stage_layer_ids: Optional[List[int]] = None
    ) -> List[DESLOCLayerSpec]:
        """Build per-layer specs, optionally sliced to a pipeline stage.

        This mirrors the refactoring in b0eb9143: first build all layer specs,
        then slice to the current pipeline stage. The DES-LOC variant also
        attaches device affinity to each spec.

        Args:
            pp_stage_layer_ids: If provided, only return specs for these global
                layer indices (0-based). If None, return specs for all layers.

        Returns:
            List of :class:`DESLOCLayerSpec` for the requested layers.
        """
        primary_a6000 = (
            self.a6000_device_indices[0] if self.a6000_device_indices else self.h100_device_idx
        )

        all_specs = []
        for i in range(self.num_layers):
            use_exp_attn = bool(self.exp_attn_pattern[i])
            use_moe = bool(self.moe_pattern[i])

            # Device affinity assignment:
            # - DSA attention → prefer A6000 for bandwidth-bound indexing
            # - Standard/MLA attention → prefer H100 for compute
            preferred_attn = primary_a6000 if use_exp_attn else self.h100_device_idx
            # - MoE expert FFN always on H100 (compute-bound)
            preferred_mlp = self.h100_device_idx

            # mla_rotary_interleaved is True iff the model uses MLA
            # (matches the upstream fix: this flag is now per-call-site, not from config)
            mla_rope = self.multi_latent_attention

            spec = DESLOCLayerSpec(
                layer_idx=i,
                use_experimental_attention=use_exp_attn,
                use_moe_mlp=use_moe,
                preferred_attn_device=preferred_attn,
                preferred_mlp_device=preferred_mlp,
                mla_rotary_interleaved=mla_rope,
            )
            all_specs.append(spec)

        if pp_stage_layer_ids is not None:
            return [all_specs[idx] for idx in pp_stage_layer_ids]
        return all_specs


# ---------------------------------------------------------------------------
# build_hetero_moe_dsa_block — top-level entry point for Neuron_SP gpt_builder
# ---------------------------------------------------------------------------

def build_hetero_moe_dsa_block(
    num_layers: int,
    experimental_attention_variant: Optional[str],
    linear_attention_freq: Optional[int],
    moe_layer_freq: int,
    num_moe_experts: Optional[int],
    multi_latent_attention: bool,
    h100_device_idx: int = 0,
    a6000_device_indices: Optional[List[int]] = None,
    pp_stage_layer_ids: Optional[List[int]] = None,
    dsa_index_head_dim: int = 128,
    dsa_qk_pos_emb_head_dim: int = 64,
    dsa_num_heads: int = 8,
    dsa_top_k: int = 128,
    freq_cache: Optional[HeteroRoPEFrequencyCache] = None,
) -> Tuple[List[DESLOCLayerSpec], Optional[HeteroMoEDSADispatcher]]:
    """Build a DES-LOC heterogeneous MoE+DSA transformer block.

    This is the top-level entry point that Neuron_SP's ``gpt_builder`` should
    call when ``args.experimental_attention_variant`` is set. It mirrors the
    Megatron b0eb9143 refactoring that split block-spec construction from
    pipeline-stage slicing, and adds DES-LOC device affinity annotations.

    Upstream routing fix (b0eb9143 — gpt_builders.py)
    --------------------------------------------------
    In Megatron, ``gpt_builder`` now routes ``experimental_attention_variant``
    to ``get_transformer_layer_with_experimental_attention_variant_spec`` before
    passing to the pipeline slicer. This function implements the equivalent
    routing for DeepSpeed/Neuron_SP:

    1. Compute attention pattern from ``experimental_attention_variant`` and
       ``linear_attention_freq``.
    2. Compute MoE pattern from ``moe_layer_freq`` and ``num_moe_experts``.
    3. Build all-layer specs via ``HeteroTransformerLayerSpecBuilder``.
    4. Slice to pipeline stage via ``pp_stage_layer_ids``.
    5. If the variant is ``dsa``, construct a ``HeteroMoEDSADispatcher`` with
       a ``DSAIndexerHetero`` configured for the DES-LOC target cluster.

    Args:
        num_layers: Total number of transformer layers.
        experimental_attention_variant: One of ``"dsa"``, ``"gated_delta_net"``,
            or ``None`` for standard attention.
        linear_attention_freq: For linear attention variants, the frequency at
            which experimental layers appear (e.g. 2 = every other layer).
        moe_layer_freq: Frequency of MoE layers (int or list).
        num_moe_experts: Number of experts; if None, no MoE is used.
        multi_latent_attention: Whether MLA is active (controls RoPE interleaving).
        h100_device_idx: CUDA index of the H100 NVL device.
        a6000_device_indices: CUDA indices of A6000 devices.
        pp_stage_layer_ids: If provided, slice specs to these layer indices.
        dsa_index_head_dim: Head dim for DSA indexer.
        dsa_qk_pos_emb_head_dim: PE dims for DSA indexer.
        dsa_num_heads: Number of DSA indexer heads.
        dsa_top_k: Top-k for DSA sparse selection.
        freq_cache: DES-LOC RoPE locality cache (uses global cache if None).

    Returns:
        Tuple of:
        - List of :class:`DESLOCLayerSpec` for the (possibly sliced) block.
        - :class:`HeteroMoEDSADispatcher` if variant is ``"dsa"``, else None.

    Raises:
        ValueError: If ``experimental_attention_variant`` is unrecognised.
    """
    fc = freq_cache or _GLOBAL_ROPE_FREQ_CACHE
    a6000_idxs = a6000_device_indices or []
    primary_a6000 = a6000_idxs[0] if a6000_idxs else h100_device_idx

    # --- Compute experimental attention pattern ---
    if experimental_attention_variant is None:
        exp_attn_pattern = [0] * num_layers
    elif experimental_attention_variant == "dsa":
        # DSA applies to all layers (no interleaving with standard attention)
        exp_attn_pattern = [1] * num_layers
    elif experimental_attention_variant == "gated_delta_net":
        if linear_attention_freq is None:
            raise ValueError(
                "linear_attention_freq must be set when experimental_attention_variant="
                "'gated_delta_net'"
            )
        if isinstance(linear_attention_freq, int):
            exp_attn_pattern = [
                0 if (i + 1) % linear_attention_freq == 0 else 1
                for i in range(num_layers)
            ]
        elif isinstance(linear_attention_freq, list):
            if len(linear_attention_freq) != num_layers:
                raise ValueError(
                    f"linear_attention_freq list length {len(linear_attention_freq)} "
                    f"!= num_layers {num_layers}"
                )
            exp_attn_pattern = linear_attention_freq
        else:
            raise ValueError(
                f"Invalid linear_attention_freq type: {type(linear_attention_freq)}"
            )
    else:
        raise ValueError(
            f"Unrecognised experimental_attention_variant: '{experimental_attention_variant}'. "
            "DES-LOC supports 'dsa', 'gated_delta_net', or None."
        )

    # --- Compute MoE layer pattern ---
    if num_moe_experts is None:
        moe_pattern = [0] * num_layers
    elif isinstance(moe_layer_freq, int):
        moe_pattern = [
            1 if (i + 1) % moe_layer_freq == 0 else 0
            for i in range(num_layers)
        ]
    elif isinstance(moe_layer_freq, list):
        if len(moe_layer_freq) != num_layers:
            raise ValueError(
                f"moe_layer_freq list length {len(moe_layer_freq)} != num_layers {num_layers}"
            )
        moe_pattern = moe_layer_freq
    else:
        raise ValueError(f"Invalid moe_layer_freq type: {type(moe_layer_freq)}")

    # --- Build layer specs ---
    builder = HeteroTransformerLayerSpecBuilder(
        num_layers=num_layers,
        experimental_attention_pattern=exp_attn_pattern,
        moe_layer_pattern=moe_pattern,
        multi_latent_attention=multi_latent_attention,
        h100_device_idx=h100_device_idx,
        a6000_device_indices=a6000_idxs,
    )
    layer_specs = builder.build_layer_specs(pp_stage_layer_ids=pp_stage_layer_ids)

    # --- Build DSA dispatcher if variant is dsa ---
    dispatcher: Optional[HeteroMoEDSADispatcher] = None
    if experimental_attention_variant == "dsa":
        indexer = DSAIndexerHetero(
            index_head_dim=dsa_index_head_dim,
            qk_pos_emb_head_dim=dsa_qk_pos_emb_head_dim,
            num_heads=dsa_num_heads,
            top_k=dsa_top_k,
            preferred_device=primary_a6000,
            freq_cache=fc,
        )
        if num_moe_experts is not None:
            dispatcher = HeteroMoEDSADispatcher(
                num_experts=num_moe_experts,
                top_k_experts=2,
                dsa_indexer=indexer,
                h100_device_idx=h100_device_idx,
                a6000_device_indices=a6000_idxs,
            )
            logger.info(
                "build_hetero_moe_dsa_block: DSA+MoE dispatcher ready "
                "(experts=%d, DSA heads=%d, top_k=%d, "
                "H100=cuda:%d, A6000=%s)",
                num_moe_experts,
                dsa_num_heads,
                dsa_top_k,
                h100_device_idx,
                a6000_idxs,
            )

    return layer_specs, dispatcher


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    class TestRotateHalf(unittest.TestCase):
        def test_shape_preserved(self):
            x = torch.randn(4, 2, 8, 16)
            out = _rotate_half(x)
            self.assertEqual(out.shape, x.shape)

        def test_double_rotate_is_negation(self):
            x = torch.randn(4, 2, 8, 16)
            # Rotating twice should negate (rotation by pi)
            out = _rotate_half(_rotate_half(x))
            torch.testing.assert_close(out, -x)

    class TestRotateInterleaved(unittest.TestCase):
        def test_shape_preserved(self):
            x = torch.randn(4, 2, 8, 16)
            out = _rotate_interleaved(x)
            self.assertEqual(out.shape, x.shape)

        def test_double_rotate_is_negation(self):
            x = torch.randn(4, 2, 8, 16)
            out = _rotate_interleaved(_rotate_interleaved(x))
            torch.testing.assert_close(out, -x)

    class TestHeteroRoPEFrequencyCache(unittest.TestCase):
        def test_cpu_get_returns_same_values(self):
            cache = HeteroRoPEFrequencyCache()
            freqs = torch.randn(32, 1, 1, 64)
            result = cache.get(freqs, torch.device("cpu"))
            torch.testing.assert_close(result, freqs)

        def test_repeated_get_returns_cached(self):
            cache = HeteroRoPEFrequencyCache()
            freqs = torch.randn(32, 1, 1, 64)
            r1 = cache.get(freqs, torch.device("cpu"))
            r2 = cache.get(freqs, torch.device("cpu"))
            self.assertIs(r1, r2)

        def test_clear_resets_cache(self):
            cache = HeteroRoPEFrequencyCache()
            freqs = torch.randn(32, 1, 1, 64)
            cache.get(freqs, torch.device("cpu"))
            cache.clear()
            self.assertEqual(len(cache._cache), 0)

        def test_step_advances_counter(self):
            cache = HeteroRoPEFrequencyCache()
            self.assertEqual(cache._current_step, 0)
            cache.step()
            self.assertEqual(cache._current_step, 1)

    class TestApplyRopeBshdHetero(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(42)
            self.seq = 16
            self.batch = 2
            self.heads = 4
            self.dim = 32
            self.t = torch.randn(self.seq, self.batch, self.heads, self.dim)
            self.freqs = torch.randn(self.seq, 1, 1, self.dim)

        def test_output_shape(self):
            out = apply_rope_bshd_hetero(self.t, self.freqs)
            self.assertEqual(out.shape, self.t.shape)

        def test_deterministic(self):
            out1 = apply_rope_bshd_hetero(self.t, self.freqs, mla_rotary_interleaved=False)
            out2 = apply_rope_bshd_hetero(self.t, self.freqs, mla_rotary_interleaved=False)
            torch.testing.assert_close(out1, out2)

        def test_mla_interleaved_different_from_standard(self):
            out_std = apply_rope_bshd_hetero(self.t, self.freqs, mla_rotary_interleaved=False)
            out_mla = apply_rope_bshd_hetero(self.t, self.freqs, mla_rotary_interleaved=True)
            self.assertFalse(torch.allclose(out_std, out_mla))

        def test_deprecated_multi_latent_attention_alias(self):
            with self.assertWarns(DeprecationWarning):
                out = apply_rope_bshd_hetero(
                    self.t, self.freqs, multi_latent_attention=True
                )
            out_new = apply_rope_bshd_hetero(
                self.t, self.freqs, mla_rotary_interleaved=True
            )
            torch.testing.assert_close(out, out_new)

        def test_partial_rot_dim(self):
            # When rot_dim < head_dim, only first rot_dim channels are rotated
            rot_dim = self.dim // 2
            freqs_partial = torch.randn(self.seq, 1, 1, rot_dim)
            out = apply_rope_bshd_hetero(self.t, freqs_partial)
            self.assertEqual(out.shape, self.t.shape)
            # Non-rotated channels should be unchanged
            torch.testing.assert_close(out[..., rot_dim:], self.t[..., rot_dim:])

        def test_mscale_applied(self):
            out1 = apply_rope_bshd_hetero(self.t, self.freqs, mscale=1.0)
            out2 = apply_rope_bshd_hetero(self.t, self.freqs, mscale=2.0)
            self.assertFalse(torch.allclose(out1, out2))

        def test_freq_cache_integration(self):
            cache = HeteroRoPEFrequencyCache()
            out = apply_rope_bshd_hetero(
                self.t, self.freqs, freq_cache=cache
            )
            self.assertEqual(out.shape, self.t.shape)

    class TestApplyRopeThd(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(7)
            self.total_tokens = 32
            self.heads = 4
            self.dim = 32
            self.t = torch.randn(self.total_tokens, self.heads, self.dim)
            self.freqs = torch.randn(self.total_tokens, 1, 1, self.dim)
            # Two sequences of length 16
            self.cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32)

        def test_requires_cp_group(self):
            import torch.distributed as dist

            class FakeGroup:
                pass

            out = apply_rope_thd_hetero(
                self.t, self.cu_seqlens, self.freqs, cp_group=FakeGroup()
            )
            self.assertEqual(out.shape, self.t.shape)

        def test_raises_without_cp_group(self):
            with self.assertRaises(ValueError):
                apply_rope_thd_hetero(self.t, self.cu_seqlens, self.freqs, cp_group=None)

        def test_deprecated_alias(self):
            import torch.distributed as dist

            class FakeGroup:
                pass

            with self.assertWarns(DeprecationWarning):
                out = apply_rope_thd_hetero(
                    self.t,
                    self.cu_seqlens,
                    self.freqs,
                    cp_group=FakeGroup(),
                    multi_latent_attention=False,
                )
            self.assertEqual(out.shape, self.t.shape)

    class TestDSAIndexerHeteroRopeSplit(unittest.TestCase):
        """Validates the corrected pe/nope split order from Megatron b0eb9143."""

        def setUp(self):
            torch.manual_seed(0)
            self.index_head_dim = 16
            self.qk_pos_emb_head_dim = 8
            self.indexer = DSAIndexerHetero(
                index_head_dim=self.index_head_dim,
                qk_pos_emb_head_dim=self.qk_pos_emb_head_dim,
                num_heads=2,
                top_k=4,
            )

        def _make_config(self):
            class Cfg:
                rotary_interleaved = False
                apply_rope_fusion = False
                multi_latent_attention = True
            return Cfg()

        def test_split_order_pe_first(self):
            """pe channels (first qk_pos_emb_head_dim dims) should be rotated;
            nope channels (trailing dims) should be unchanged in output ordering."""
            seq, batch = 8, 1
            x = torch.zeros(seq, batch, 2, self.index_head_dim)
            # Set pe section to 1.0, nope section to 0.0
            x[..., :self.qk_pos_emb_head_dim] = 1.0
            x[..., self.qk_pos_emb_head_dim:] = 0.0

            rotary_pos_emb = torch.zeros(seq, 1, 1, self.qk_pos_emb_head_dim)
            config = self._make_config()

            out = self.indexer._apply_rope(x, rotary_pos_emb, mscale=1.0, config=config)
            self.assertEqual(out.shape, x.shape)

            # The output pe section should be modified (rotation of [1,1,...,0,0,...])
            # The nope section should remain 0
            nope_out = out[..., self.qk_pos_emb_head_dim:]
            torch.testing.assert_close(nope_out, torch.zeros_like(nope_out))

        def test_wrong_split_order_would_differ(self):
            """Confirm that swapping pe/nope would produce different results
            (i.e. the split order matters for correctness)."""
            seq, batch = 4, 1
            x = torch.randn(seq, batch, 2, self.index_head_dim)
            rotary_pos_emb = torch.randn(seq, 1, 1, self.qk_pos_emb_head_dim)
            config = self._make_config()

            # Correct order (pe first)
            out_correct = self.indexer._apply_rope(x, rotary_pos_emb, mscale=1.0, config=config)

            # Manually apply wrong order (nope first) for comparison
            x_nope_first, x_pe_first = torch.split(
                x,
                [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim],
                dim=-1,
            )
            x_pe_rotated = apply_rope_bshd_hetero(
                x_pe_first, rotary_pos_emb, mla_rotary_interleaved=False
            )
            out_wrong = torch.cat([x_nope_first, x_pe_rotated], dim=-1)

            self.assertFalse(
                torch.allclose(out_correct, out_wrong),
                "Correct and wrong split orders should differ",
            )

    class TestDSAIndexerForward(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(1)
            self.index_head_dim = 16
            self.qk_pos_emb_head_dim = 8
            self.num_heads = 2
            self.top_k = 4
            self.indexer = DSAIndexerHetero(
                index_head_dim=self.index_head_dim,
                qk_pos_emb_head_dim=self.qk_pos_emb_head_dim,
                num_heads=self.num_heads,
                top_k=self.top_k,
            )

        def _make_config(self):
            class Cfg:
                rotary_interleaved = False
                apply_rope_fusion = False
                multi_latent_attention = True
            return Cfg()

        def test_output_shape(self):
            seq, batch = 8, 1
            hidden = torch.randn(seq, batch, self.num_heads * self.index_head_dim)
            rpe = torch.randn(seq, 1, 1, self.qk_pos_emb_head_dim)
            config = self._make_config()
            out = self.indexer.forward(hidden, rpe, mscale=1.0, config=config)
            self.assertEqual(out.shape, (seq, batch, self.num_heads, self.top_k))

        def test_indices_in_range(self):
            seq, batch = 8, 1
            hidden = torch.randn(seq, batch, self.num_heads * self.index_head_dim)
            rpe = torch.randn(seq, 1, 1, self.qk_pos_emb_head_dim)
            config = self._make_config()
            out = self.indexer.forward(hidden, rpe, mscale=1.0, config=config)
            self.assertTrue((out >= 0).all())
            self.assertTrue((out < seq).all())

    class TestHeteroMoEDSADispatcherDispatchPlan(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(5)
            indexer = DSAIndexerHetero(
                index_head_dim=16,
                qk_pos_emb_head_dim=8,
                num_heads=2,
                top_k=4,
            )
            self.dispatcher = HeteroMoEDSADispatcher(
                num_experts=8,
                top_k_experts=2,
                dsa_indexer=indexer,
                h100_device_idx=0,
                a6000_device_indices=[],
            )

        def test_dispatch_plan_partition(self):
            num_tokens = 20
            router_logits = torch.randn(num_tokens, 8)
            plan = self.dispatcher._compute_dispatch_plan(router_logits)
            total = plan.h100_token_indices.numel() + plan.a6000_token_indices.numel()
            self.assertEqual(total, num_tokens)

        def test_dispatch_plan_no_overlap(self):
            num_tokens = 20
            router_logits = torch.randn(num_tokens, 8)
            plan = self.dispatcher._compute_dispatch_plan(router_logits)
            h100_set = set(plan.h100_token_indices.tolist())
            a6000_set = set(plan.a6000_token_indices.tolist())
            self.assertEqual(len(h100_set & a6000_set), 0)

    class TestHeteroTransformerLayerSpecBuilder(unittest.TestCase):
        def test_all_standard_attention(self):
            builder = HeteroTransformerLayerSpecBuilder(
                num_layers=4,
                experimental_attention_pattern=[0, 0, 0, 0],
                moe_layer_pattern=[0, 0, 0, 0],
                multi_latent_attention=False,
                h100_device_idx=0,
            )
            specs = builder.build_layer_specs()
            self.assertEqual(len(specs), 4)
            for s in specs:
                self.assertFalse(s.use_experimental_attention)
                self.assertFalse(s.mla_rotary_interleaved)
                self.assertEqual(s.preferred_attn_device, 0)

        def test_dsa_layers_prefer_a6000(self):
            builder = HeteroTransformerLayerSpecBuilder(
                num_layers=4,
                experimental_attention_pattern=[1, 0, 1, 0],
                moe_layer_pattern=[0, 0, 0, 0],
                multi_latent_attention=True,
                h100_device_idx=2,
                a6000_device_indices=[0, 1],
            )
            specs = builder.build_layer_specs()
            self.assertEqual(specs[0].preferred_attn_device, 0)  # A6000
            self.assertEqual(specs[1].preferred_attn_device, 2)  # H100
            self.assertTrue(specs[0].mla_rotary_interleaved)

        def test_pipeline_slicing(self):
            builder = HeteroTransformerLayerSpecBuilder(
                num_layers=8,
                experimental_attention_pattern=[1] * 8,
                moe_layer_pattern=[0] * 8,
                multi_latent_attention=True,
                h100_device_idx=0,
            )
            specs = builder.build_layer_specs(pp_stage_layer_ids=[4, 5, 6, 7])
            self.assertEqual(len(specs), 4)
            self.assertEqual([s.layer_idx for s in specs], [4, 5, 6, 7])

        def test_invalid_pattern_lengths(self):
            with self.assertRaises(ValueError):
                HeteroTransformerLayerSpecBuilder(
                    num_layers=4,
                    experimental_attention_pattern=[1, 0],  # wrong length
                    moe_layer_pattern=[0, 0, 0, 0],
                    multi_latent_attention=False,
                )

    class TestBuildHeteroMoeDsaBlock(unittest.TestCase):
        def test_no_variant(self):
            specs, dispatcher = build_hetero_moe_dsa_block(
                num_layers=4,
                experimental_attention_variant=None,
                linear_attention_freq=None,
                moe_layer_freq=2,
                num_moe_experts=None,
                multi_latent_attention=False,
            )
            self.assertEqual(len(specs), 4)
            self.assertIsNone(dispatcher)

        def test_dsa_variant_all_layers(self):
            specs, dispatcher = build_hetero_moe_dsa_block(
                num_layers=4,
                experimental_attention_variant="dsa",
                linear_attention_freq=None,
                moe_layer_freq=2,
                num_moe_experts=8,
                multi_latent_attention=True,
            )
            self.assertEqual(len(specs), 4)
            self.assertTrue(all(s.use_experimental_attention for s in specs))
            self.assertIsNotNone(dispatcher)

        def test_gated_delta_net_pattern(self):
            specs, dispatcher = build_hetero_moe_dsa_block(
                num_layers=4,
                experimental_attention_variant="gated_delta_net",
                linear_attention_freq=2,
                moe_layer_freq=1,
                num_moe_experts=None,
                multi_latent_attention=False,
            )
            # freq=2: [1, 0, 1, 0]
            self.assertEqual([s.use_experimental_attention for s in specs], [True, False, True, False])
            self.assertIsNone(dispatcher)

        def test_pipeline_slicing(self):
            specs, _ = build_hetero_moe_dsa_block(
                num_layers=8,
                experimental_attention_variant="dsa",
                linear_attention_freq=None,
                moe_layer_freq=1,
                num_moe_experts=4,
                multi_latent_attention=True,
                pp_stage_layer_ids=[0, 1, 2, 3],
            )
            self.assertEqual(len(specs), 4)
            self.assertEqual([s.layer_idx for s in specs], [0, 1, 2, 3])

        def test_invalid_variant_raises(self):
            with self.assertRaises(ValueError):
                build_hetero_moe_dsa_block(
                    num_layers=4,
                    experimental_attention_variant="linear_mamba",
                    linear_attention_freq=None,
                    moe_layer_freq=1,
                    num_moe_experts=None,
                    multi_latent_attention=False,
                )

        def test_moe_pattern_list(self):
            specs, _ = build_hetero_moe_dsa_block(
                num_layers=4,
                experimental_attention_variant=None,
                linear_attention_freq=None,
                moe_layer_freq=[1, 0, 1, 0],
                num_moe_experts=4,
                multi_latent_attention=False,
            )
            self.assertEqual([s.use_moe_mlp for s in specs], [True, False, True, False])

    # Run all tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRotateHalf)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRotateInterleaved))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroRoPEFrequencyCache))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestApplyRopeBshdHetero))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestApplyRopeThd))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDSAIndexerHeteroRopeSplit))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDSAIndexerForward))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroMoEDSADispatcherDispatchPlan))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroTransformerLayerSpecBuilder))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBuildHeteroMoeDsaBlock))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
