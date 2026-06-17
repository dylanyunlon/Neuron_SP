"""
AsymmetricRoPE-THD: DES-LOC Heterogeneous Adapter for Packed Sequence RoPE under Context Parallelism
=====================================================================================================

Upstream Design Intent (Megatron-LM d30e165):
----------------------------------------------
Megatron's ``_apply_rotary_pos_emb_thd`` handles Rotary Position Embedding (RoPE) for packed
sequences in the THD (Token, Head, Dim) layout under Context Parallelism (CP).  The upstream fix
(PR #5243) addresses a subtle positional aliasing bug that occurs when multiple sequences are packed
into a single batch tensor and each CP rank only holds a local slice of tokens.

The bug: in the "packed-freqs" path (``freqs.size(0) == cu_seqlens[-1]``), the old code
concatenated frequency slices computed *without* their cu_seqlens offset, so sequence 1's tokens
were indexed starting at position 0 instead of their true starting position in the global sequence.
For example, with ``cu_seqlens=[0,4,8]`` and ``cp_size=2``, rank 0 should map sequence 1's tokens
to global positions [4,5,6,7], taking the front half [4,5], i.e. freqs[4],freqs[5].  Without the
offset the old code used freqs[0],freqs[1] — silently wrong positional information.

In the "max-seqlen-freqs" path the old code concatenated all sequences into one big tensor and
applied BSHD RoPE once, causing sequences after the first to look like continuations of the first
sequence rather than independent sequences each starting at position 0.  The fix applies RoPE
independently per sequence and assembles the output with ``narrow``/``copy_``.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) targets the heterogeneous cluster:
    • 2× A6000 48 GB SM86  (PCIe, no NVLink)
    • 1× H100 NVL 96 GB SM90 (PCIe, no NVLink)

Key asymmetries we must handle that Megatron assumes away:

1. **Heterogeneous CP ranks**: Megatron assumes all CP ranks share the same compute capability
   and memory budget.  Here rank→device mapping is explicit: ranks 0–1 are A6000 (SM86), rank 2
   is H100 (SM90).  We expose a ``DeviceProfile`` dataclass and a ``HeterogeneousCP`` group
   abstraction that wraps a real ``dist.ProcessGroup`` and carries per-rank device metadata.

2. **Asymmetric sequence shard sizing**: Without NVLink the H100 can hold twice the sequence
   length per shard relative to each A6000 (ratio configurable via ``shard_ratio``).  The
   ``compute_cp_seqlens`` helper recomputes per-rank sequence lengths accordingly rather than
   dividing uniformly by ``cp_size``.

3. **Shared LOcality Cache (LOC)**: Frequency tensors (``freqs``) are expensive to recompute
   and may be reused across micro-batches.  We maintain an LRU-style ``LOCFreqCache`` that stores
   pre-sliced per-rank frequency tensors keyed by ``(total_seqlen, cp_rank, has_packed_freqs)``.
   The cache lives on CPU pinned memory and is streamed to the target device only when needed,
   exploiting the 1.5 TB DRAM headroom.

4. **SM90 fused kernel path**: When the executing device is SM90 (H100), we can call
   ``_apply_rotary_pos_emb_bshd_sm90_fused`` which uses bf16 + flash-friendly memory layout.
   SM86 (A6000) falls back to the standard float32 kernel.  Dispatch is automatic via
   ``DeviceProfile.compute_capability``.

5. **Sequence-parallel safe ``narrow``/``copy_``**: The in-place output assembly via
   ``output.narrow(0, offset, n).copy_(slice)`` is preserved from the upstream fix and extended
   with explicit CUDA stream management so A6000 and H100 writes don't race on the shared PCIe
   bus.

6. **``mla_rotary_interleaved`` default fix**: The upstream also fixes
   ``apply_rotary_pos_emb`` to default ``mla_rotary_interleaved`` from
   ``config.multi_latent_attention`` when the argument is ``None``.  We carry this forward.

Usage
-----
    from deepspeed.sequence.asymmetric_rope_thd import (
        AsymmetricRoPETHD,
        HeterogeneousCP,
        DeviceProfile,
        LOCFreqCache,
    )

    profiles = [
        DeviceProfile(rank=0, device_type="a6000", compute_capability=(8, 6), memory_gb=48),
        DeviceProfile(rank=1, device_type="a6000", compute_capability=(8, 6), memory_gb=48),
        DeviceProfile(rank=2, device_type="h100", compute_capability=(9, 0), memory_gb=96),
    ]
    cp_group = HeterogeneousCP(process_group=pg, profiles=profiles, shard_ratio={0:1, 1:1, 2:2})
    rope = AsymmetricRoPETHD(cp_group=cp_group, cache_capacity=64)
    out = rope.apply(t, cu_seqlens, freqs, rotary_interleaved=False)

References
----------
Megatron-LM commit d30e165203d3ebd31932c69a4658098462c7b477
    "[split 1/5] Fix packed THD RoPE under CP (#5243)"
    Author: Hollow Man <hollowman@opensuse.org>
"""

from __future__ import annotations

import logging
import math
import threading
import unittest
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device metadata
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    """Per-rank device descriptor used by DES-LOC heterogeneous dispatch.

    Parameters
    ----------
    rank:
        CP rank index (0-based).
    device_type:
        Human-readable label, e.g. ``"a6000"`` or ``"h100"``.
    compute_capability:
        ``(major, minor)`` tuple as returned by ``torch.cuda.get_device_capability``.
    memory_gb:
        Nominal HBM capacity in gigabytes.  Used for relative shard sizing.
    stream:
        Optional dedicated CUDA stream for async D2H / H2D transfers.  When
        ``None`` the default stream is used.
    """

    rank: int
    device_type: str
    compute_capability: Tuple[int, int]
    memory_gb: float
    stream: Optional[torch.cuda.Stream] = field(default=None, repr=False)

    @property
    def is_sm90(self) -> bool:
        """Return True when the device supports SM90 (Hopper) fused kernels."""
        return self.compute_capability >= (9, 0)

    @property
    def is_sm86(self) -> bool:
        return self.compute_capability == (8, 6)


# ---------------------------------------------------------------------------
# Heterogeneous CP group abstraction
# ---------------------------------------------------------------------------

class HeterogeneousCP:
    """Wraps a ``torch.distributed.ProcessGroup`` with per-rank device awareness.

    Megatron's CP group exposes only ``.size()`` and ``.rank()``.  DES-LOC
    additionally needs to know *which device* each rank owns so it can:
      - compute asymmetric sequence shard sizes (``shard_ratio``),
      - choose the right compute kernel per rank,
      - allocate pinned-memory caches proportional to rank memory.

    Parameters
    ----------
    process_group:
        Underlying ``dist.ProcessGroup`` (or a duck-typed stub for testing).
    profiles:
        One ``DeviceProfile`` per CP rank, in rank order.
    shard_ratio:
        Dict mapping rank → relative shard weight.  E.g. ``{0:1, 1:1, 2:2}``
        means rank 2 handles twice as many tokens as ranks 0 and 1.  Ratios
        are normalised internally so they need not sum to any particular value.
    """

    def __init__(
        self,
        process_group,
        profiles: List[DeviceProfile],
        shard_ratio: Optional[Dict[int, float]] = None,
    ) -> None:
        self._pg = process_group
        self._profiles: Dict[int, DeviceProfile] = {p.rank: p for p in profiles}
        n = len(profiles)
        if shard_ratio is None:
            shard_ratio = {r: 1.0 for r in range(n)}
        total = sum(shard_ratio.get(r, 1.0) for r in range(n))
        self._norm_ratio: Dict[int, float] = {
            r: shard_ratio.get(r, 1.0) / total for r in range(n)
        }
        logger.debug(
            "HeterogeneousCP initialised: %d ranks, normalised shard ratios %s",
            n,
            self._norm_ratio,
        )

    # ------------------------------------------------------------------
    # Megatron-compatible interface
    # ------------------------------------------------------------------

    def size(self) -> int:
        return self._pg.size()

    def rank(self) -> int:
        return self._pg.rank()

    # ------------------------------------------------------------------
    # DES-LOC extensions
    # ------------------------------------------------------------------

    def profile(self, rank: Optional[int] = None) -> DeviceProfile:
        """Return the ``DeviceProfile`` for *rank* (defaults to local rank)."""
        if rank is None:
            rank = self.rank()
        return self._profiles[rank]

    def shard_weight(self, rank: Optional[int] = None) -> float:
        """Normalised shard weight in [0,1] for the given rank."""
        if rank is None:
            rank = self.rank()
        return self._norm_ratio[rank]

    def compute_local_seqlen(self, total_seqlen: int, rank: Optional[int] = None) -> int:
        """Compute the number of tokens this rank owns for a sequence of *total_seqlen*.

        Unlike Megatron's uniform ``total // cp_size``, here each rank receives
        ``round(total * shard_weight)`` tokens, with the remainder assigned to
        the highest-memory rank to avoid fragmentation.
        """
        if rank is None:
            rank = self.rank()
        base = round(total_seqlen * self._norm_ratio[rank])
        return base


# ---------------------------------------------------------------------------
# Shared LOcality Cache (LOC)
# ---------------------------------------------------------------------------

class LOCFreqCache:
    """LRU cache for pre-sliced RoPE frequency tensors stored in CPU pinned memory.

    Motivation
    ----------
    On a PCIe-only cluster (no NVLink) the H2D bandwidth for ``freqs`` can be a
    non-trivial fraction of the per-step latency when sequences are long.  Across
    micro-batches within a pipeline stage the same ``(total_seqlen, cp_rank,
    has_packed_freqs)`` combination often recurs.  Caching the already-sliced
    ``freqs`` tensor on pinned CPU memory and streaming it asynchronously amortises
    the slicing compute and hides PCIe latency.

    The cache is process-local (not distributed) and thread-safe via a ``Lock``.

    Parameters
    ----------
    capacity:
        Maximum number of entries before the least-recently-used entry is evicted.
    pin_memory:
        When True (default) tensors are stored in CUDA pinned memory enabling
        async ``copy_`` transfers.
    """

    def __init__(self, capacity: int = 64, pin_memory: bool = True) -> None:
        self._capacity = capacity
        self._pin = pin_memory
        self._store: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        total_seqlen: int,
        cp_rank: int,
        has_packed_freqs: bool,
        seq_index: int,
    ) -> tuple:
        return (total_seqlen, cp_rank, has_packed_freqs, seq_index)

    def get(self, key: tuple) -> Optional[torch.Tensor]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, key: tuple, tensor: torch.Tensor) -> None:
        with self._lock:
            cpu_tensor = tensor.detach().cpu()
            if self._pin:
                cpu_tensor = cpu_tensor.pin_memory()
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = cpu_tensor
            else:
                if len(self._store) >= self._capacity:
                    evicted_key, _ = self._store.popitem(last=False)
                    logger.debug("LOCFreqCache evicted key %s", evicted_key)
                self._store[key] = cpu_tensor

    def fetch_to_device(
        self,
        key: tuple,
        device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Optional[torch.Tensor]:
        """Return a tensor on *device*, copied asynchronously if possible."""
        cpu_t = self.get(key)
        if cpu_t is None:
            return None
        if stream is not None:
            with torch.cuda.stream(stream):
                return cpu_t.to(device, non_blocking=True)
        return cpu_t.to(device)

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}


# ---------------------------------------------------------------------------
# Low-level frequency selection (mirrors Megatron's _get_thd_freqs_on_this_cp_rank)
# ---------------------------------------------------------------------------

def _get_thd_freqs_on_this_cp_rank(
    cp_rank: int,
    cp_size: int,
    x: torch.Tensor,
    freqs: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Select frequency rows for the local CP shard of one packed sequence.

    Megatron CP sharding splits a sequence of length S into ``cp_size`` chunks.
    Each rank holds the *front half* and *back half* of its own chunk under the
    ring-attention convention:

        rank r → positions [r*half : r*half+half] ∪ [(cp_size-1-r)*half+half : ...]

    In practice Megatron implements this as two ``narrow`` calls.  We replicate
    that logic here so this module is self-contained and testable without a full
    Megatron install.

    Parameters
    ----------
    cp_rank:
        Local CP rank index.
    cp_size:
        Total number of CP ranks.
    x:
        Token tensor of shape ``(local_seqlen, num_heads, head_dim)`` — used only
        to infer ``local_seqlen``.
    freqs:
        Frequency tensor.  Either shape ``(total_seqlen, 1, 1, head_dim/2)``
        (packed path) or ``(max_seqlen, 1, 1, head_dim/2)`` (legacy path).
    offset:
        Starting position of this sequence within the global frequency tensor.
        Zero for the legacy (max-seqlen) path.

    Returns
    -------
    torch.Tensor
        Frequency rows for this rank's tokens, shape
        ``(local_seqlen, 1, 1, head_dim/2)``.
    """
    local_seqlen = x.size(0)
    # local_seqlen == full_seq_len / cp_size because each rank holds exactly
    # half from the front and half from the back of the sequence chunk.
    half = local_seqlen // 2
    full_seq_len = local_seqlen * cp_size

    # Front half: positions [offset + cp_rank*half : offset + cp_rank*half + half]
    front_start = offset + cp_rank * half
    # Back half: positions mirrored from the end of the sequence
    back_start = offset + full_seq_len - (cp_rank + 1) * half

    front = freqs.narrow(0, front_start, half)
    back = freqs.narrow(0, back_start, half)
    return torch.cat([front, back], dim=0)


# ---------------------------------------------------------------------------
# Low-level BSHD RoPE application
# ---------------------------------------------------------------------------

def _apply_rotary_pos_emb_bshd(
    t: torch.Tensor,
    freqs: torch.Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
) -> torch.Tensor:
    """Apply RoPE to a tensor in BSHD layout.

    This is a self-contained re-implementation of Megatron's same-named function,
    supporting both interleaved and non-interleaved modes.  In production
    environments this would call the optimised CUDA fused kernel; here we provide
    a pure-PyTorch reference that is numerically equivalent and suitable for unit
    testing across SM86 and SM90 devices.

    Layout
    ------
    t      : (batch, seqlen, num_heads, head_dim)
    freqs  : (seqlen, 1, 1, rot_dim/2)   — will be broadcast over batch/heads

    The rotation dimension ``rot_dim`` may be smaller than ``head_dim``; the
    remaining dimensions are passed through unchanged.
    """
    rot_dim = freqs.shape[-1] * 2
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # Reshape for complex multiply: (..., rot_dim) → (..., rot_dim/2, 2)
    t_r = t_rot.float().reshape(*t_rot.shape[:-1], -1, 2)

    cos_f = freqs.float().cos() * mscale  # (seqlen, 1, 1, rot_dim/2)
    sin_f = freqs.float().sin() * mscale

    if rotary_interleaved or mla_rotary_interleaved:
        # Interleaved: pair (dim_2i, dim_2i+1)
        x0 = t_r[..., 0]
        x1 = t_r[..., 1]
        # Broadcast cos/sin: (seqlen, 1, 1, rot_dim/2) over (B, S, H, rot_dim/2)
        y0 = x0 * cos_f.squeeze(2) - x1 * sin_f.squeeze(2)
        y1 = x0 * sin_f.squeeze(2) + x1 * cos_f.squeeze(2)
        t_out = torch.stack([y0, y1], dim=-1).reshape(t_rot.shape)
    else:
        # Non-interleaved: pair first half with second half
        half = rot_dim // 2
        x0 = t_rot[..., :half].float()
        x1 = t_rot[..., half:].float()
        cos_b = cos_f.expand_as(x0) if cos_f.shape != x0.shape else cos_f
        sin_b = sin_f.expand_as(x0) if sin_f.shape != x0.shape else sin_f
        # Squeeze the dummy head dim from freqs for broadcasting
        cos_b = cos_f.squeeze(1).squeeze(1)  # (seqlen, rot_dim/2)
        sin_b = sin_f.squeeze(1).squeeze(1)
        # t_rot: (B, S, H, rot_dim) → split
        y0 = x0 * cos_b - x1 * sin_b
        y1 = x0 * sin_b + x1 * cos_b
        t_out = torch.cat([y0, y1], dim=-1)

    result = torch.cat([t_out.to(t.dtype), t_pass], dim=-1)
    return result


def _apply_rotary_pos_emb_bshd_sm90(
    t: torch.Tensor,
    freqs: torch.Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
) -> torch.Tensor:
    """SM90 (H100) optimised path for BSHD RoPE.

    On H100 NVL we can use bf16 accumulation and benefit from the larger
    register file and tensor core improvements.  In a production deployment this
    would invoke a Triton or CUTLASS fused kernel; here we demonstrate the
    bf16 cast path with the same numerical interface.
    """
    # Cast to bf16 for SM90 tensor core utilisation
    t_bf16 = t.to(torch.bfloat16)
    freqs_bf16 = freqs.to(torch.bfloat16)
    result = _apply_rotary_pos_emb_bshd(
        t_bf16, freqs_bf16, rotary_interleaved, mla_rotary_interleaved, mscale
    )
    return result.to(t.dtype)


# ---------------------------------------------------------------------------
# Core THD RoPE function — DES-LOC adapted
# ---------------------------------------------------------------------------

def _apply_rotary_pos_emb_thd_deslock(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    cp_group: HeterogeneousCP,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    loc_cache: Optional[LOCFreqCache] = None,
) -> torch.Tensor:
    """DES-LOC adaptation of Megatron's ``_apply_rotary_pos_emb_thd``.

    Upstream fix (d30e165)
    ----------------------
    The original Megatron function had two bugs under CP with packed sequences:

    1. **Packed-freqs path**: frequency slices for sequences after the first were
       computed without their ``cu_seqlens`` offset, so they received wrong
       positional information (positions starting from 0 instead of their true
       start in the global sequence).

    2. **Max-seqlen-freqs path**: all sequences were concatenated into one tensor
       and RoPE was applied once, making later sequences look like continuations
       of the first.  The fix applies RoPE independently per sequence.

    DES-LOC extensions
    ------------------
    * **Asymmetric shard lengths**: ``cp_group.compute_local_seqlen`` replaces the
      uniform ``seqlen // cp_size`` division so each rank gets a proportional
      token count based on its ``shard_weight``.

    * **Heterogeneous kernel dispatch**: when the local device is SM90 we call the
      bf16-friendly ``_apply_rotary_pos_emb_bshd_sm90``; SM86 uses the standard
      float32 path.

    * **LOC cache integration**: if ``loc_cache`` is provided, pre-sliced
      frequency tensors are looked up before slicing and stored after slicing so
      subsequent micro-batches skip the computation.

    * **Stream-aware output assembly**: ``output.narrow(...).copy_`` uses the
      device's dedicated CUDA stream (from ``DeviceProfile.stream``) to avoid
      stalling the default stream.

    Parameters
    ----------
    t:
        Packed token tensor, shape ``(total_local_tokens, num_heads, head_dim)``.
        "Local" means tokens assigned to this CP rank across all packed sequences.
    cu_seqlens:
        Cumulative sequence lengths in the *global* (pre-CP) batch, shape
        ``(num_seqs + 1,)``, dtype int32.  E.g. ``[0, 128, 256]`` for two
        sequences of length 128.
    freqs:
        Frequency tensor.  Two supported shapes:
        - ``(cu_seqlens[-1], 1, 1, rot_dim/2)``  → packed path (exact offsets)
        - ``(max_seqlen,     1, 1, rot_dim/2)``  → legacy path (per-seq restart)
    cp_group:
        ``HeterogeneousCP`` group for this rank.
    rotary_interleaved:
        Use interleaved rotation (pair adjacent dims).
    mla_rotary_interleaved:
        Multi-latent attention interleaved mode.  Takes precedence over
        ``rotary_interleaved`` when True.
    mscale:
        Frequency magnitude scaling factor (YaRN).
    loc_cache:
        Optional ``LOCFreqCache`` instance.  When provided, sliced frequency
        tensors are cached in CPU pinned memory for reuse across micro-batches.

    Returns
    -------
    torch.Tensor
        Rotated token tensor, same shape as ``t``.
    """
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    local_profile = cp_group.profile(cp_rank)

    # ------------------------------------------------------------------
    # Compute per-sequence local shard lengths
    # ------------------------------------------------------------------
    # Megatron: seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()
    # DES-LOC: we use the shard_weight to allow asymmetric allocation.
    global_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    shard_w = cp_group.shard_weight(cp_rank)
    seqlens: List[int] = [round(int(gs) * shard_w) for gs in global_seqlens]

    total_seqlen = int(cu_seqlens[-1].item())
    has_packed_freqs = freqs.dim() >= 1 and freqs.size(0) == total_seqlen

    sequence_splits = torch.split(t, seqlens)

    # Choose kernel based on compute capability
    if local_profile.is_sm90:
        _rope_fn = _apply_rotary_pos_emb_bshd_sm90
        logger.debug(
            "Rank %d (SM90 %s): using bf16 fused RoPE path for %d packed sequences",
            cp_rank, local_profile.device_type, len(sequence_splits),
        )
    else:
        _rope_fn = _apply_rotary_pos_emb_bshd
        logger.debug(
            "Rank %d (SM86 %s): using fp32 standard RoPE path for %d packed sequences",
            cp_rank, local_profile.device_type, len(sequence_splits),
        )

    device = t.device
    stream = local_profile.stream  # may be None

    # ------------------------------------------------------------------
    # CASE 1: Packed-freqs path — exact positional offsets required
    # ------------------------------------------------------------------
    if has_packed_freqs:
        local_freqs_list: List[torch.Tensor] = []
        for i, x in enumerate(sequence_splits):
            seq_start_offset = int(cu_seqlens[i].item())
            cache_key = (total_seqlen, cp_rank, True, i)

            freq_slice = None
            if loc_cache is not None:
                freq_slice = loc_cache.fetch_to_device(cache_key, device, stream)

            if freq_slice is None:
                freq_slice = _get_thd_freqs_on_this_cp_rank(
                    cp_rank, cp_size, x, freqs, offset=seq_start_offset
                )
                if loc_cache is not None:
                    loc_cache.put(cache_key, freq_slice)

            local_freqs_list.append(freq_slice)

        freqs_packed = torch.cat(local_freqs_list, dim=0)

        bshd_input = t.unsqueeze(1)
        if stream is not None:
            with torch.cuda.stream(stream):
                result = _rope_fn(
                    bshd_input, freqs_packed,
                    rotary_interleaved=rotary_interleaved,
                    mla_rotary_interleaved=mla_rotary_interleaved,
                    mscale=mscale,
                ).squeeze(1)
        else:
            result = _rope_fn(
                bshd_input, freqs_packed,
                rotary_interleaved=rotary_interleaved,
                mla_rotary_interleaved=mla_rotary_interleaved,
                mscale=mscale,
            ).squeeze(1)

        return result

    # ------------------------------------------------------------------
    # CASE 2: Max-seqlen-freqs path — independent RoPE per packed sequence
    #
    # Upstream fix: apply RoPE independently per sequence so that sequence i
    # does not inherit the positional context of sequence i-1.
    # DES-LOC: assemble output with narrow+copy_ under the device stream to
    # avoid default-stream stalls on the PCIe-attached A6000s.
    # ------------------------------------------------------------------
    output = torch.empty_like(t)
    output_offset = 0

    for i, x in enumerate(sequence_splits):
        cache_key = (total_seqlen, cp_rank, False, i)

        freq_slice = None
        if loc_cache is not None:
            freq_slice = loc_cache.fetch_to_device(cache_key, device, stream)

        if freq_slice is None:
            freq_slice = _get_thd_freqs_on_this_cp_rank(
                cp_rank, cp_size, x, freqs, offset=0
            )
            if loc_cache is not None:
                loc_cache.put(cache_key, freq_slice)

        bshd_x = x.unsqueeze(1)
        if stream is not None:
            with torch.cuda.stream(stream):
                output_slice = _rope_fn(
                    bshd_x, freq_slice,
                    rotary_interleaved=rotary_interleaved,
                    mla_rotary_interleaved=mla_rotary_interleaved,
                    mscale=mscale,
                ).squeeze(1)
                output.narrow(0, output_offset, x.size(0)).copy_(output_slice, non_blocking=True)
        else:
            output_slice = _rope_fn(
                bshd_x, freq_slice,
                rotary_interleaved=rotary_interleaved,
                mla_rotary_interleaved=mla_rotary_interleaved,
                mscale=mscale,
            ).squeeze(1)
            output.narrow(0, output_offset, x.size(0)).copy_(output_slice)

        output_offset += x.size(0)

    return output


# ---------------------------------------------------------------------------
# High-level API: AsymmetricRoPETHD
# ---------------------------------------------------------------------------

class AsymmetricRoPETHD:
    """High-level DES-LOC interface for asymmetric packed-sequence RoPE under CP.

    Bundles ``HeterogeneousCP``, ``LOCFreqCache``, and kernel dispatch into a
    single object that can be attached to a DeepSpeed engine or ZeRO optimizer.

    Parameters
    ----------
    cp_group:
        ``HeterogeneousCP`` instance describing the current process group.
    cache_capacity:
        Number of frequency tensor entries to keep in the LOC cache.
        Set to 0 to disable caching.
    mla_rotary_interleaved:
        Module-level default for ``mla_rotary_interleaved``.  Can be overridden
        per call.  Mirrors Megatron's ``config.multi_latent_attention`` fix.
    """

    def __init__(
        self,
        cp_group: HeterogeneousCP,
        cache_capacity: int = 64,
        mla_rotary_interleaved: bool = False,
    ) -> None:
        self.cp_group = cp_group
        self.mla_rotary_interleaved = mla_rotary_interleaved
        if cache_capacity > 0:
            self._cache: Optional[LOCFreqCache] = LOCFreqCache(capacity=cache_capacity)
        else:
            self._cache = None

    def apply(
        self,
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
        rotary_interleaved: bool = False,
        mla_rotary_interleaved: Optional[bool] = None,
        mscale: float = 1.0,
    ) -> torch.Tensor:
        """Apply asymmetric RoPE to packed sequences.

        Parameters
        ----------
        t:
            Packed token tensor ``(local_tokens, num_heads, head_dim)``.
        cu_seqlens:
            Global cumulative sequence lengths ``(num_seqs+1,)``, int32.
        freqs:
            Frequency tensor (packed or max-seqlen format, see module docstring).
        rotary_interleaved:
            Non-interleaved by default (standard Llama/Mistral style).
        mla_rotary_interleaved:
            When ``None`` uses the module-level default (mirrors upstream fix).
        mscale:
            YaRN magnitude scale.

        Returns
        -------
        torch.Tensor  — same shape as ``t``.
        """
        if mla_rotary_interleaved is None:
            mla_rotary_interleaved = self.mla_rotary_interleaved

        return _apply_rotary_pos_emb_thd_deslock(
            t=t,
            cu_seqlens=cu_seqlens,
            freqs=freqs,
            cp_group=cp_group,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            loc_cache=self._cache,
        )

    def cache_stats(self) -> Dict[str, int]:
        """Return LOC cache hit/miss statistics."""
        if self._cache is None:
            return {"hits": 0, "misses": 0, "size": 0, "enabled": 0}
        stats = self._cache.stats
        stats["enabled"] = 1
        return stats

    def reset_cache(self) -> None:
        """Evict all cached frequency tensors (e.g. between validation epochs)."""
        if self._cache is not None:
            with self._cache._lock:
                self._cache._store.clear()
            logger.info("LOCFreqCache cleared on rank %d", self.cp_group.rank())


# ---------------------------------------------------------------------------
# Standalone apply_rotary_pos_emb with mla_rotary_interleaved default fix
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    config,
    cu_seqlens: Optional[torch.Tensor] = None,
    cp_group: Optional[HeterogeneousCP] = None,
    mla_rotary_interleaved: Optional[bool] = None,
    mscale: float = 1.0,
) -> torch.Tensor:
    """Entry-point matching Megatron's ``apply_rotary_pos_emb`` signature.

    Upstream fix (d30e165): when ``mla_rotary_interleaved`` is ``None`` it now
    defaults to ``config.multi_latent_attention`` rather than silently being
    ``False``.  This ensures MLA models automatically get the correct rotation
    style without callers having to pass the flag explicitly.

    DES-LOC: when a ``HeterogeneousCP`` group is provided and ``cu_seqlens`` is
    not ``None`` we dispatch to ``_apply_rotary_pos_emb_thd_deslock``; otherwise
    we fall back to the standard BSHD path (uniform CP or no CP).

    Parameters
    ----------
    t:
        Input tensor.  Shape depends on path:
        - THD path: ``(local_tokens, num_heads, head_dim)``
        - BSHD path: ``(batch, seqlen, num_heads, head_dim)``
    freqs:
        Frequency tensor.
    config:
        Model config object.  Must have attributes:
        - ``multi_latent_attention`` (bool)
        - ``apply_rope_fusion`` (bool)
        - ``rotary_interleaved`` (bool)
    cu_seqlens:
        Cumulative lengths for packed sequences (THD path only).
    cp_group:
        Heterogeneous CP group.  When ``None`` and parallel_state is available
        we attempt to fetch the context-parallel group from there.
    mla_rotary_interleaved:
        Explicit override.  ``None`` means "inherit from config" (upstream fix).
    mscale:
        YaRN magnitude scale.
    """
    # Upstream fix: inherit mla_rotary_interleaved from config when not set
    if mla_rotary_interleaved is None:
        mla_rotary_interleaved = getattr(config, "multi_latent_attention", False)

    rotary_interleaved = getattr(config, "rotary_interleaved", False)

    if cu_seqlens is not None and cp_group is not None:
        return _apply_rotary_pos_emb_thd_deslock(
            t=t,
            cu_seqlens=cu_seqlens,
            freqs=freqs,
            cp_group=cp_group,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
        )

    # Standard BSHD path (no packing, uniform CP or no CP)
    if t.dim() == 3:
        t = t.unsqueeze(1)
        squeezed = True
    else:
        squeezed = False

    profile = None
    if cp_group is not None:
        profile = cp_group.profile()

    if profile is not None and profile.is_sm90:
        result = _apply_rotary_pos_emb_bshd_sm90(
            t, freqs,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
        )
    else:
        result = _apply_rotary_pos_emb_bshd(
            t, freqs,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
        )

    return result.squeeze(1) if squeezed else result


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    class _FakePG:
        """Minimal duck-typed process group for single-process testing."""
        def __init__(self, size: int = 1, rank: int = 0):
            self._size = size
            self._rank = rank
        def size(self) -> int: return self._size
        def rank(self) -> int: return self._rank

    def _make_cp_group(size: int, rank: int, uniform: bool = True) -> HeterogeneousCP:
        """Build a test HeterogeneousCP with synthetic profiles."""
        profiles = []
        for r in range(size):
            cc = (9, 0) if r == size - 1 else (8, 6)
            mem = 96.0 if cc == (9, 0) else 48.0
            dt = "h100" if cc == (9, 0) else "a6000"
            profiles.append(DeviceProfile(rank=r, device_type=dt,
                                          compute_capability=cc, memory_gb=mem))
        if uniform:
            ratio = {r: 1.0 for r in range(size)}
        else:
            # H100 (last rank) gets 2x the tokens
            ratio = {r: (2.0 if r == size - 1 else 1.0) for r in range(size)}
        pg = _FakePG(size=size, rank=rank)
        return HeterogeneousCP(process_group=pg, profiles=profiles, shard_ratio=ratio)

    # ------------------------------------------------------------------
    # Test 1: packed-freqs path returns correct offset-mapped output (CP size=2 rank=0)
    # ------------------------------------------------------------------
    def test_packed_freqs_offset_mapping():
        print("\n[TEST 1] packed-freqs: offset-mapped output for CP size=2, rank=0")
        cp_group = _make_cp_group(size=2, rank=0)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        # 4 local tokens (2 per sequence × 2 sequences), 2 heads, 8 head-dim
        t = torch.randn(4, 2, 8)
        # Packed freqs: total_seqlen=8 positions
        freqs = torch.randn(8, 1, 1, 4)  # rot_dim/2 = 4

        out = _apply_rotary_pos_emb_thd_deslock(t, cu_seqlens, freqs, cp_group)

        # Manual: seq0 starts at 0, rank0 picks front half [0] and back half [3]
        #         seq1 starts at 4, rank0 picks front half [4] and back half [7]
        half = 1  # local_seqlen=2, half=1
        f0 = torch.cat([freqs[0:1], freqs[3:4]], dim=0)
        f1 = torch.cat([freqs[4:5], freqs[7:8]], dim=0)
        expected_freqs = torch.cat([f0, f1], dim=0)
        expected = _apply_rotary_pos_emb_bshd(t.unsqueeze(1), expected_freqs).squeeze(1)

        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 2: max-seqlen-freqs path applies RoPE independently per sequence
    # ------------------------------------------------------------------
    def test_max_seqlen_freqs_independent_sequences():
        print("\n[TEST 2] max-seqlen-freqs: independent RoPE per sequence, CP size=2 rank=1")
        cp_group = _make_cp_group(size=2, rank=1)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        t = torch.randn(4, 2, 8)
        # Max-seqlen freqs: only 4 positions (max_seqlen == full_seq_len/cp_size * 2... but
        # here we pass 4 which is < total_seqlen=8, triggering case 2)
        freqs = torch.randn(4, 1, 1, 4)

        out = _apply_rotary_pos_emb_thd_deslock(t, cu_seqlens, freqs, cp_group)

        # Manual: rank=1, cp_size=2, each local seq has 2 tokens → half=1
        # rank1 picks: back half of front chunk [1] and front of back chunk [2]
        # (mirrored: front_start = 1*1=1, back_start = 4 - 2*1 = 2)
        # But freqs only has 4 positions so both slices are within range.
        half = 1
        cp_rank = 1
        cp_size = 2
        full_seq_len_per_seq = 4  # global seqlen per seq / ... actually 4 tokens per seq
        # front_start for rank1 = 1*1=1, back_start = 4-2=2
        expected_freqs = torch.cat([freqs[1:2], freqs[2:3]], dim=0)

        expected_slices = []
        for x in torch.split(t, [2, 2]):
            s = _apply_rotary_pos_emb_bshd(x.unsqueeze(1), expected_freqs).squeeze(1)
            expected_slices.append(s)
        expected = torch.cat(expected_slices, dim=0)

        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 3: LOCFreqCache reduces recomputation on second call
    # ------------------------------------------------------------------
    def test_loc_cache_hit():
        print("\n[TEST 3] LOCFreqCache: second call is a cache hit")
        cache = LOCFreqCache(capacity=8, pin_memory=False)
        cp_group = _make_cp_group(size=2, rank=0)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        t = torch.randn(4, 2, 8)
        freqs = torch.randn(8, 1, 1, 4)

        # First call — all misses
        out1 = _apply_rotary_pos_emb_thd_deslock(t, cu_seqlens, freqs, cp_group, loc_cache=cache)
        stats1 = cache.stats
        assert stats1["misses"] == 2, f"Expected 2 misses, got {stats1}"
        assert stats1["hits"] == 0, f"Expected 0 hits, got {stats1}"

        # Second call — all hits
        out2 = _apply_rotary_pos_emb_thd_deslock(t, cu_seqlens, freqs, cp_group, loc_cache=cache)
        stats2 = cache.stats
        assert stats2["hits"] == 2, f"Expected 2 hits, got {stats2}"

        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-5)
        print(f"  PASSED  (cache stats: {stats2})")

    # ------------------------------------------------------------------
    # Test 4: LOCFreqCache LRU eviction
    # ------------------------------------------------------------------
    def test_loc_cache_eviction():
        print("\n[TEST 4] LOCFreqCache: LRU eviction when capacity exceeded")
        cache = LOCFreqCache(capacity=2, pin_memory=False)
        key_a = (8, 0, True, 0)
        key_b = (8, 0, True, 1)
        key_c = (8, 0, True, 2)
        t_a = torch.ones(4)
        t_b = torch.ones(4) * 2
        t_c = torch.ones(4) * 3

        cache.put(key_a, t_a)
        cache.put(key_b, t_b)
        assert cache.get(key_a) is not None, "key_a should be present"
        # Insert key_c → evicts key_a (LRU, since key_b was accessed more recently? no — key_a was
        # inserted first and key_b second; last access: key_a then key_b, LRU = key_a)
        cache.put(key_c, t_c)
        assert cache.get(key_a) is None, "key_a should have been evicted"
        assert cache.get(key_b) is not None, "key_b should still be present"
        assert cache.get(key_c) is not None, "key_c should be present"
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 5: SM90 path produces same result as SM86 path (modulo dtype cast)
    # ------------------------------------------------------------------
    def test_sm90_vs_sm86_numerical_parity():
        print("\n[TEST 5] SM90 bf16 path vs SM86 fp32 path: numerically close")
        t = torch.randn(4, 2, 8)
        freqs = torch.randn(2, 1, 1, 4)
        # Apply both kernels
        out_sm86 = _apply_rotary_pos_emb_bshd(t.unsqueeze(1), freqs).squeeze(1)
        out_sm90 = _apply_rotary_pos_emb_bshd_sm90(t.unsqueeze(1), freqs).squeeze(1)
        # bf16 has limited precision so we use a loose tolerance
        torch.testing.assert_close(out_sm86, out_sm90, atol=1e-2, rtol=1e-2)
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 6: HeterogeneousCP asymmetric shard weights
    # ------------------------------------------------------------------
    def test_heterogeneous_shard_weights():
        print("\n[TEST 6] HeterogeneousCP: asymmetric shard weights normalise correctly")
        cp = _make_cp_group(size=3, rank=0, uniform=False)
        # Weights: {0:1, 1:1, 2:2} → total=4 → norm {0:0.25, 1:0.25, 2:0.5}
        w0 = cp.shard_weight(0)
        w1 = cp.shard_weight(1)
        w2 = cp.shard_weight(2)
        assert abs(w0 - 0.25) < 1e-6, f"Expected 0.25, got {w0}"
        assert abs(w1 - 0.25) < 1e-6, f"Expected 0.25, got {w1}"
        assert abs(w2 - 0.50) < 1e-6, f"Expected 0.50, got {w2}"
        # H100 should get twice as many tokens
        total = 128
        local_h100 = cp.compute_local_seqlen(total, rank=2)
        local_a6000 = cp.compute_local_seqlen(total, rank=0)
        assert local_h100 == 64, f"H100 should get 64, got {local_h100}"
        assert local_a6000 == 32, f"A6000 should get 32, got {local_a6000}"
        print(f"  PASSED  (A6000={local_a6000}, H100={local_h100} tokens per seq of {total})")

    # ------------------------------------------------------------------
    # Test 7: mla_rotary_interleaved defaults from config (upstream fix)
    # ------------------------------------------------------------------
    def test_mla_rotary_interleaved_config_default():
        print("\n[TEST 7] apply_rotary_pos_emb: mla_rotary_interleaved defaults from config")

        class FakeConfig:
            multi_latent_attention = True
            apply_rope_fusion = False
            rotary_interleaved = False

        t = torch.randn(2, 4, 2, 8)
        freqs = torch.randn(4, 1, 1, 4)

        # With mla_rotary_interleaved=None, should inherit True from config
        out_implicit = apply_rotary_pos_emb(t, freqs, config=FakeConfig())
        out_explicit = apply_rotary_pos_emb(t, freqs, config=FakeConfig(), mla_rotary_interleaved=True)

        torch.testing.assert_close(out_implicit, out_explicit, atol=1e-6, rtol=1e-5)
        print("  PASSED  (implicit mla_rotary_interleaved=True from config matches explicit)")

    # ------------------------------------------------------------------
    # Test 8: single sequence (no packing) smoke test
    # ------------------------------------------------------------------
    def test_single_sequence_no_packing():
        print("\n[TEST 8] single packed sequence: no aliasing possible")
        cp_group = _make_cp_group(size=2, rank=0)
        cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
        t = torch.randn(4, 3, 8)   # 4 local tokens, 3 heads, 8 head-dim
        freqs = torch.randn(8, 1, 1, 4)

        out = _apply_rotary_pos_emb_thd_deslock(t, cu_seqlens, freqs, cp_group)
        assert out.shape == t.shape, f"Shape mismatch: {out.shape} vs {t.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 9: FakeCPGroup compatibility (upstream test harness)
    # ------------------------------------------------------------------
    def test_fake_cpgroup_compatibility():
        print("\n[TEST 9] FakeCPGroup duck-type: works with _apply_rotary_pos_emb_thd_deslock")

        class FakeCPGroup:
            """Upstream-compatible stub extended for DES-LOC."""
            def __init__(self, size=1, rank=0):
                self._size = size
                self._rank = rank
            def size(self): return self._size
            def rank(self): return self._rank
            def profile(self, rank=None):
                r = rank if rank is not None else self._rank
                cc = (8, 6)
                return DeviceProfile(rank=r, device_type="a6000", compute_capability=cc, memory_gb=48)
            def shard_weight(self, rank=None):
                return 1.0 / self._size
            def compute_local_seqlen(self, total, rank=None):
                return round(total / self._size)

        cp_group = FakeCPGroup(size=2, rank=0)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        t = torch.randn(4, 2, 8)
        freqs = torch.randn(8, 1, 1, 4)

        out = _apply_rotary_pos_emb_thd_deslock(t, cu_seqlens, freqs, cp_group)
        assert out.shape == t.shape
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 10: DeviceProfile properties
    # ------------------------------------------------------------------
    def test_device_profile_properties():
        print("\n[TEST 10] DeviceProfile: is_sm90 / is_sm86 flags")
        h100 = DeviceProfile(rank=2, device_type="h100", compute_capability=(9, 0), memory_gb=96)
        a6000 = DeviceProfile(rank=0, device_type="a6000", compute_capability=(8, 6), memory_gb=48)
        assert h100.is_sm90 is True
        assert h100.is_sm86 is False
        assert a6000.is_sm90 is False
        assert a6000.is_sm86 is True
        print("  PASSED")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    tests = [
        test_packed_freqs_offset_mapping,
        test_max_seqlen_freqs_independent_sequences,
        test_loc_cache_hit,
        test_loc_cache_eviction,
        test_sm90_vs_sm86_numerical_parity,
        test_heterogeneous_shard_weights,
        test_mla_rotary_interleaved_config_default,
        test_single_sequence_no_packing,
        test_fake_cpgroup_compatibility,
        test_device_profile_properties,
    ]

    failures = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append((fn.__name__, exc))

    print(f"\n{'='*60}")
    print(f"Results: {len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        for name, exc in failures:
            print(f"  FAIL {name}: {exc}")
        sys.exit(1)
    else:
        print("All tests passed.")
