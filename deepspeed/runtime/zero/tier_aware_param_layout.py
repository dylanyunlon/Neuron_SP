# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""TierAwareParamLayout — GPU-tier-proportional flat buffer layout for DES-LOC ZeRO.

Mirrors Megatron 55b8111ad — "DDP refactoring: Extract parameter layout computation
into optimizer classmethod", reinterpreted as TierAwareParamLayout for DES-LOC
heterogeneous GPU clusters where flat buffer partitioning must be weighted by
each rank's available VRAM.

Upstream design intent (55b8111ad)
------------------------------------
Megatron's commit extracts parameter layout computation (bucket boundaries,
per-param buffer offsets, alignment padding) out of _ParamAndGradBuffer.__init__
and into a standalone classmethod ``DistributedOptimizer.compute_full_param_layout``.

The key data structures introduced:

    BufferKey(param_dtype, grad_dtype, is_expert_parallel)
        — identifies a distinct flat buffer.

    PerBufferParamLayout(param_index_map, bucket_indices, per_bucket_numel_unpadded)
        — the pre-computed layout for one buffer: where each param lives, where
          bucket boundaries fall, and how much of each bucket is real data vs. padding.

    FullParamLayout(layouts: Dict[BufferKey, PerBufferParamLayout])
        — the collection of all per-buffer layouts for an entire model.

The separation means that DDP receives a pre-baked layout object rather than
computing offsets itself, enabling external code (the optimizer) to influence
bucket boundaries before buffers are allocated.  The padding helpers ``pad_param_start``
and ``pad_bucket_end`` enforce DP-divisibility and 128-byte alignment constraints.

DES-LOC adaptation
-------------------
In a homogeneous cluster every DP rank holds an equal slice of the flat buffer.
In a heterogeneous DES-LOC cluster (e.g. A6000 48 GB + H100 80 GB), naïve
equal slicing wastes H100 VRAM and over-fills A6000 VRAM, causing OOM on the
weaker tier.

TierAwareParamLayout replaces the homogeneous equal-slice assumption with a
VRAM-proportional partition scheme:

    weight_r  =  vram_r / Σ vram_j   for rank r in the DP group

    slice_r   =  round(weight_r × total_flat_numel)

                 (adjusted so slices sum exactly to total_flat_numel)

The result is that H100 (80 GB) ranks hold ~62.5 % of parameters and A6000
(48 GB) ranks hold ~37.5 %, matching their relative VRAM capacity.  The split
is computed before flat buffer allocation so every rank allocates only its own
slice, and the DP collective (reduce-scatter / all-gather) is size-matched.

Key classes / functions
-----------------------
``GPUVRAMInfo``
    Lightweight dataclass carrying (rank, vram_gb) for one DP-group member.
    Probed locally via ``probe_local_vram()`` and all-gathered by
    ``gather_dp_vram_infos()``.

``TierAwareParamLayout``
    Extends the Megatron PerBufferParamLayout concept with tier-proportional
    bucket boundaries.  ``compute()`` is the classmethod entry-point, mirroring
    ``DistributedOptimizer.compute_full_param_layout``.

``TierAwareFullLayout``
    Wraps a Dict[BufferKey, TierAwareParamLayout], one entry per dtype group,
    analogous to Megatron's FullParamLayout.  ``compute_for_model`` is the
    top-level factory function called before ZeRO optimizer construction.

``apply_tier_aware_layout``
    Stamps ``_tier_layout`` onto each parameter so ZeRO stage-3's
    partitioned-param coordinator can query this rank's (start, end) slice
    without repeating the layout computation.

Alignment rules (mirrors upstream pad_param_start / pad_bucket_end)
----------------------------------------------------------------------
Within a buffer:
  - Each param start is rounded up to a multiple of 64 elements
    (= 128-byte alignment for ≥ 16-bit dtypes).
  - Each bucket end is rounded up to lcm(dp_world_size, 128) elements
    so that all-reduce / reduce-scatter slices are equally sized.
  - When ``pad_for_high_nccl_busbw=True`` the bucket-end divisor is extended to
    lcm(dp_world_size, 128, 2^16), matching Megatron's high-busbw padding.

These rules are implemented in ``_pad_param_start`` and ``_pad_bucket_end``
(local reimplementations; no megatron.core imports used anywhere in this file).

Decision boundary diagnostics
-------------------------------
All layout decisions are logged at INFO level with the [DS-TAPL] prefix:

    [DS-TAPL] PROBE   — local VRAM probe result per rank.
    [DS-TAPL] GATHER  — all-gathered VRAM table across DP group.
    [DS-TAPL] WEIGHT  — per-rank weight and assigned element count.
    [DS-TAPL] LAYOUT  — per-buffer layout summary (total numel, #buckets,
                        padding overhead).
    [DS-TAPL] SLICE   — per-rank slice boundaries (start_elem, end_elem).
    [DS-TAPL] STAMP   — summary after apply_tier_aware_layout() stamps params.
    [DS-TAPL] WARN    — unexpected conditions (e.g. uniform VRAM, homogeneous
                        cluster detected — falls back gracefully).

Integration
-----------
Typical usage before ZeRO optimizer construction::

    from deepspeed.runtime.zero.tier_aware_param_layout import (
        compute_for_model,
        apply_tier_aware_layout,
        gather_dp_vram_infos,
    )

    # Gather VRAM across DP group once.
    vram_infos = gather_dp_vram_infos(dp_process_group)

    # Compute tier-aware layout for all dtype groups.
    full_layout = compute_for_model(
        params=list(model.parameters()),
        dp_process_group=dp_process_group,
        vram_infos=vram_infos,
        bucket_size=40_000_000,
        grad_reduce_in_fp32=True,
    )

    # Stamp _tier_layout on each param so ZeRO stage-3 can query slices.
    apply_tier_aware_layout(model.parameters(), full_layout)

    # Pass full_layout into ZeRO engine / optimizer init as needed.

No megatron.core imports are used anywhere in this file.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

_LOG_PREFIX = "[DS-TAPL]"
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alignment helpers — local reimplementations of Megatron pad_param_start /
# pad_bucket_end so we carry zero megatron.core dependency.
# ---------------------------------------------------------------------------

_PARAM_START_ALIGNMENT = 64  # 64 elems × ≥2 bytes = 128-byte alignment


def _pad_param_start(index: int) -> int:
    """Round *index* up to the nearest multiple of ``_PARAM_START_ALIGNMENT``.

    Mirrors Megatron ``pad_param_start``: ensures each parameter in a flat
    buffer begins at a 128-byte aligned address (assuming ≥ 16-bit dtype).
    """
    align = _PARAM_START_ALIGNMENT
    return int(math.ceil(index / align) * align)


def _pad_bucket_end(
    index: int,
    dp_world_size: int,
    pad_for_high_nccl_busbw: bool = False,
) -> int:
    """Round *index* up so that the bucket size is divisible by the DP world
    size (and 128-byte alignment).

    Mirrors Megatron ``pad_bucket_end``:
      - Base divisor:  lcm(dp_world_size, 128)
      - High-busbw:    lcm(dp_world_size, 128, 2^16)
    """
    if pad_for_high_nccl_busbw:
        divisor = math.lcm(dp_world_size, 128, 2**16)
    else:
        divisor = math.lcm(dp_world_size, 128)
    return int(math.ceil(index / divisor) * divisor)


# ---------------------------------------------------------------------------
# VRAM probing and all-gather
# ---------------------------------------------------------------------------

@dataclass
class GPUVRAMInfo:
    """VRAM capacity (in GB) for one DP-group member.

    Intentionally lightweight — only the fields needed for proportional
    partitioning.  Extended GPU metadata (bandwidth, compute capability, …)
    lives in ``hetero_mesh.GPUTierInfo`` and is not imported here.
    """
    rank: int
    vram_gb: float
    device_name: str = ""


def probe_local_vram(rank: Optional[int] = None) -> GPUVRAMInfo:
    """Return a ``GPUVRAMInfo`` for the current CUDA device.

    Falls back to ``vram_gb=0`` when CUDA is unavailable (CI / CPU-only
    environments), which causes downstream code to fall back to uniform
    partitioning.

    Args:
        rank: Optional rank override; defaults to ``dist.get_rank()`` when
              ``dist`` is initialised, else 0.
    """
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0

    if not torch.cuda.is_available():
        log.debug("%s PROBE rank=%d: CUDA unavailable, vram_gb=0", _LOG_PREFIX, rank)
        return GPUVRAMInfo(rank=rank, vram_gb=0.0, device_name="cpu")

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    vram_gb = props.total_memory / (1024 ** 3)
    info = GPUVRAMInfo(rank=rank, vram_gb=vram_gb, device_name=props.name)
    log.info(
        "%s PROBE rank=%d device=%s vram_gb=%.1f",
        _LOG_PREFIX, rank, props.name, vram_gb,
    )
    return info


def gather_dp_vram_infos(
    dp_process_group: Optional[dist.ProcessGroup] = None,
) -> List[GPUVRAMInfo]:
    """All-gather VRAM information across *dp_process_group*.

    Each rank probes its own VRAM and broadcasts it.  The returned list is
    ordered by DP rank and has exactly ``dp_world_size`` entries.

    Args:
        dp_process_group: The data-parallel process group.  Uses the default
                          group when ``None``.

    Returns:
        List[GPUVRAMInfo] indexed by DP rank.
    """
    if not dist.is_initialized():
        # Single-process / test mode — return a single-entry list.
        local = probe_local_vram(rank=0)
        return [local]

    world_size = dist.get_world_size(group=dp_process_group)
    my_rank = dist.get_rank(group=dp_process_group)
    local = probe_local_vram(rank=my_rank)

    # Pack (vram_gb, rank) as a float tensor for all-gather.
    # Encode device_name length as a third field (unused after gather; names
    # are not communicated — just vram_gb and rank).
    local_tensor = torch.tensor(
        [local.vram_gb, float(my_rank)],
        dtype=torch.float64,
        device="cpu",
    )
    gathered = [torch.zeros(2, dtype=torch.float64, device="cpu")
                for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor, group=dp_process_group)

    infos: List[GPUVRAMInfo] = []
    for t in gathered:
        vram_gb = float(t[0].item())
        rank = int(t[1].item())
        infos.append(GPUVRAMInfo(rank=rank, vram_gb=vram_gb))

    log.info(
        "%s GATHER dp_world_size=%d vram_table=%s",
        _LOG_PREFIX,
        world_size,
        [(info.rank, f"{info.vram_gb:.1f}GB") for info in infos],
    )
    return infos


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_vram_weights(vram_infos: List[GPUVRAMInfo]) -> List[float]:
    """Compute the fractional VRAM weight for each DP rank.

    When all VRAM values are zero (CPU / test environment) or identical
    (homogeneous cluster), returns uniform weights and emits a WARN log.

    Args:
        vram_infos: Ordered by DP rank.

    Returns:
        List of floats summing to 1.0, one per DP rank.
    """
    total = sum(info.vram_gb for info in vram_infos)
    if total == 0.0:
        log.warning(
            "%s WARN all VRAM probes returned 0 — using uniform weights", _LOG_PREFIX
        )
        n = len(vram_infos)
        return [1.0 / n] * n

    weights = [info.vram_gb / total for info in vram_infos]

    # Detect homogeneous clusters (all weights within 1 % of uniform).
    n = len(weights)
    uniform = 1.0 / n
    if all(abs(w - uniform) < 0.01 for w in weights):
        log.warning(
            "%s WARN homogeneous cluster detected (all VRAM ≈ %.1f GB) — "
            "tier-aware layout is equivalent to uniform partitioning",
            _LOG_PREFIX, vram_infos[0].vram_gb,
        )

    for i, (info, w) in enumerate(zip(vram_infos, weights)):
        log.info(
            "%s WEIGHT rank=%d vram_gb=%.1f weight=%.4f",
            _LOG_PREFIX, info.rank, info.vram_gb, w,
        )
    return weights


def compute_tier_slices(
    total_numel: int,
    vram_infos: List[GPUVRAMInfo],
) -> List[Tuple[int, int]]:
    """Compute (start, end) element slices for each DP rank proportional to VRAM.

    The slices are contiguous and non-overlapping, covering ``[0, total_numel)``.
    The last rank absorbs any rounding remainder so that ``end[-1] == total_numel``.

    Args:
        total_numel: Total number of elements in the flat buffer.
        vram_infos: Ordered by DP rank.

    Returns:
        List of ``(start, end)`` tuples, one per DP rank.
    """
    weights = compute_vram_weights(vram_infos)
    n = len(weights)
    sizes = [round(w * total_numel) for w in weights]

    # Adjust last rank to absorb rounding error.
    sizes[-1] = total_numel - sum(sizes[:-1])

    slices: List[Tuple[int, int]] = []
    cursor = 0
    for rank, size in enumerate(sizes):
        start = cursor
        end = cursor + size
        slices.append((start, end))
        log.info(
            "%s SLICE rank=%d start=%d end=%d numel=%d",
            _LOG_PREFIX, rank, start, end, size,
        )
        cursor = end

    assert cursor == total_numel, (
        f"Slice computation error: cursor={cursor} != total_numel={total_numel}"
    )
    return slices


# ---------------------------------------------------------------------------
# BufferKey — mirrors Megatron's BufferKey namedtuple
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BufferKey:
    """Identifies a distinct flat parameter/gradient buffer.

    Mirrors Megatron ``BufferKey`` from ``megatron.core.optimizer.param_layout``:

    - ``param_dtype``:      Storage dtype (``torch.uint8`` for FP8/NVFP4,
                            else ``param.dtype``).
    - ``grad_dtype``:       Gradient-reduction dtype (``torch.float`` when
                            ``grad_reduce_in_fp32``, else ``param.dtype``).
    - ``is_expert_parallel``: Whether this buffer holds expert-parallel params
                            that require a different DP group.
    """
    param_dtype: torch.dtype
    grad_dtype: torch.dtype
    is_expert_parallel: bool


# ---------------------------------------------------------------------------
# TierAwareParamLayout — per-buffer layout with tier-proportional slices
# ---------------------------------------------------------------------------

@dataclass
class TierAwareParamLayout:
    """Pre-computed parameter layout for one flat buffer, tier-aware.

    Analogous to Megatron ``PerBufferParamLayout`` but extended with
    ``tier_slices``: per-rank (start, end) element slices that are
    proportional to each rank's available VRAM rather than uniform.

    Fields
    ------
    param_index_map : Dict[torch.nn.Parameter, Tuple[int, int, int]]
        Mapping from param → (start_elem, end_elem, bucket_id).
        Offsets include alignment padding.
    bucket_indices : List[Tuple[int, int]]
        (start, end) element range for each bucket (end is padded).
    per_bucket_numel_unpadded : List[int]
        Unpadded element count for each bucket (useful for actual data copying).
    total_numel : int
        Total padded size of the flat buffer (= ``bucket_indices[-1][1]``).
    tier_slices : List[Tuple[int, int]]
        Per-DP-rank (start_elem, end_elem) partitions, VRAM-proportional.
    param_indices : List[int]
        Per-parameter index among same-dtype params (for checkpoint compat).
    """
    param_index_map: Dict  # param → (start, end, bucket_id)
    bucket_indices: List[Tuple[int, int]]
    per_bucket_numel_unpadded: List[int]
    total_numel: int
    tier_slices: List[Tuple[int, int]]
    param_indices: List[int]

    def slice_for_rank(self, dp_rank: int) -> Tuple[int, int]:
        """Return the (start, end) element slice assigned to *dp_rank*."""
        return self.tier_slices[dp_rank]

    @classmethod
    def compute(
        cls,
        params: List[torch.nn.Parameter],
        vram_infos: List[GPUVRAMInfo],
        dp_world_size: int,
        bucket_size: Optional[int] = None,
        grad_reduce_in_fp32: bool = True,
        use_distributed_optimizer: bool = True,
        pad_for_high_nccl_busbw: bool = False,
    ) -> "TierAwareParamLayout":
        """Compute a tier-aware layout for *params* (all belonging to one buffer).

        Mirrors ``DistributedOptimizer.compute_full_param_layout`` for a single
        buffer.  Parameters are iterated in reverse order (backprop order) and
        grouped into buckets of approximately ``bucket_size`` elements.

        Alignment padding is applied when ``use_distributed_optimizer=True``:
          - param starts aligned to ``_PARAM_START_ALIGNMENT`` elements.
          - bucket ends aligned so the bucket size is divisible by
            ``lcm(dp_world_size, 128)`` (or extended when
            ``pad_for_high_nccl_busbw``).

        VRAM-proportional slices are derived from the total padded numel after
        all bucket boundaries are fixed.

        Args:
            params: All parameters for this buffer, in model iteration order.
            vram_infos: Per-DP-rank VRAM info (ordered by DP rank).
            dp_world_size: DP group size (used for alignment divisor).
            bucket_size: Approximate number of elements per bucket.
            grad_reduce_in_fp32: Whether gradients are reduced in FP32.
            use_distributed_optimizer: Whether to apply alignment padding.
            pad_for_high_nccl_busbw: Whether to use 2^16 bucket alignment.

        Returns:
            TierAwareParamLayout with filled param_index_map, bucket_indices,
            per_bucket_numel_unpadded, total_numel, tier_slices, param_indices.
        """
        param_index_map: Dict = {}
        bucket_indices: List[Tuple[int, int]] = []
        per_bucket_numel_unpadded: List[int] = []
        param_indices_list: List[int] = []

        param_start = 0
        bucket_start = 0
        bucket_params: List = []
        bucket_id = 0
        dtype_offset_counter: Dict[torch.dtype, int] = {}

        def _align_param_start(idx: int) -> int:
            if use_distributed_optimizer:
                return _pad_param_start(idx)
            return idx

        def _close_bucket(bucket_end_raw: int) -> int:
            """Finalise current bucket and return the (padded) next start."""
            nonlocal bucket_start, bucket_id
            per_bucket_numel_unpadded.append(bucket_end_raw - bucket_start)
            if use_distributed_optimizer:
                padded_end = _pad_bucket_end(
                    bucket_end_raw, dp_world_size, pad_for_high_nccl_busbw
                )
            else:
                padded_end = bucket_end_raw
            bucket_indices.append((bucket_start, padded_end))
            bucket_start = padded_end
            bucket_id += 1
            return padded_end

        # Walk parameters in *reverse* order (approximate backprop order so
        # the first bucket to finish its all-reduce is the last layer).
        for param in reversed(params):
            param_start = _align_param_start(param_start)

            # Shared-embedding params each need their own bucket (so the
            # embedding all-reduce and DP reduce-scatter do not overlap).
            needs_own_bucket = (
                getattr(param, "shared_embedding", False)
                and use_distributed_optimizer
            )

            if needs_own_bucket and bucket_params:
                new_start = _close_bucket(param_start)
                param_start = _align_param_start(new_start)

            numel = param.data.nelement()
            param_end = param_start + numel
            param_index_map[param] = (param_start, param_end, bucket_id)
            bucket_params.append(param)

            # Track per-param dtype offset for checkpoint compatibility.
            logical_dtype = param.dtype
            offset = dtype_offset_counter.get(logical_dtype, 0)
            param_indices_list.append(offset)
            dtype_offset_counter[logical_dtype] = offset + 1

            bucket_full = (
                bucket_size is not None
                and (param_end - bucket_start) >= bucket_size
            )
            if bucket_full or needs_own_bucket:
                param_start = _close_bucket(param_end)
                bucket_params = []
            else:
                param_start = param_end

        # Close final bucket if any params remain.
        if bucket_params:
            _close_bucket(param_end)  # type: ignore[possibly-undefined]

        total_numel = bucket_indices[-1][1] if bucket_indices else 0

        # Compute VRAM-proportional per-rank slices from the padded total.
        tier_slices = compute_tier_slices(total_numel, vram_infos)

        layout = cls(
            param_index_map=param_index_map,
            bucket_indices=bucket_indices,
            per_bucket_numel_unpadded=per_bucket_numel_unpadded,
            total_numel=total_numel,
            tier_slices=tier_slices,
            param_indices=param_indices_list,
        )

        log.info(
            "%s LAYOUT num_params=%d total_numel=%d num_buckets=%d "
            "numel_unpadded=%d padding_overhead=%.2f%%",
            _LOG_PREFIX,
            len(params),
            total_numel,
            len(bucket_indices),
            sum(per_bucket_numel_unpadded),
            100.0 * (total_numel - sum(per_bucket_numel_unpadded)) / max(total_numel, 1),
        )
        return layout


# ---------------------------------------------------------------------------
# Per-dtype grouping helper (mirrors Megatron group_params_for_buffers)
# ---------------------------------------------------------------------------

def _is_fp8_or_nvfp4(param: torch.nn.Parameter) -> bool:
    """Return True if *param* uses FP8 or NVFP4 storage.

    We probe via dtype heuristics to avoid importing TE / NVFP4 wrappers
    which may not be installed.  The upstream check uses ``is_float8tensor``
    and ``is_nvfp4tensor``; we replicate the effect here by checking uint8
    (both FP8 and NVFP4 store their data as uint8 in Megatron's convention).
    """
    # Attempt lightweight duck-typed checks for TE Float8Tensor / NVFP4Tensor.
    try:
        from transformer_engine.pytorch.float8_tensor import Float8Tensor  # type: ignore
        if isinstance(param, Float8Tensor):
            return True
    except ImportError:
        pass
    # NVFP4: no stable public isinstance check; rely on attribute.
    if getattr(param, "_is_nvfp4", False):
        return True
    return False


def group_params_by_buffer_key(
    params: Iterable[torch.nn.Parameter],
    grad_reduce_in_fp32: bool,
) -> Dict[BufferKey, List[torch.nn.Parameter]]:
    """Group *params* into per-buffer lists keyed by ``BufferKey``.

    Mirrors Megatron ``group_params_for_buffers`` (added in the same upstream
    commit) without any megatron.core dependency.

    Args:
        params: Iterable of parameters (must have ``requires_grad=True``).
        grad_reduce_in_fp32: Whether to use FP32 for gradient reduction.

    Returns:
        Dict mapping ``BufferKey`` → list of params belonging to that buffer.
    """
    groups: Dict[BufferKey, List[torch.nn.Parameter]] = {}
    for param in params:
        if not param.requires_grad:
            continue

        param_dtype = param.dtype
        if _is_fp8_or_nvfp4(param):
            param_dtype = torch.uint8
        grad_dtype = torch.float if grad_reduce_in_fp32 else param.dtype
        is_ep = not getattr(param, "allreduce", True)

        key = BufferKey(param_dtype, grad_dtype, is_ep)
        groups.setdefault(key, []).append(param)

    return groups


# ---------------------------------------------------------------------------
# TierAwareFullLayout — wraps per-buffer layouts for the whole model
# ---------------------------------------------------------------------------

@dataclass
class TierAwareFullLayout:
    """Collection of tier-aware layouts for all flat buffers in a model.

    Analogous to Megatron ``FullParamLayout(layouts: Dict[BufferKey, …])``.

    Attributes
    ----------
    layouts : Dict[BufferKey, TierAwareParamLayout]
        One layout per distinct (param_dtype, grad_dtype, is_expert_parallel)
        combination found in the model.
    vram_infos : List[GPUVRAMInfo]
        The VRAM info used to derive all tier_slices in the layouts.
    """
    layouts: Dict[BufferKey, TierAwareParamLayout]
    vram_infos: List[GPUVRAMInfo]

    def layout_for(self, key: BufferKey) -> Optional[TierAwareParamLayout]:
        """Return the layout for *key*, or ``None`` if no params matched."""
        return self.layouts.get(key)


# ---------------------------------------------------------------------------
# Top-level factory function
# ---------------------------------------------------------------------------

def compute_for_model(
    params: Iterable[torch.nn.Parameter],
    dp_process_group: Optional[dist.ProcessGroup] = None,
    vram_infos: Optional[List[GPUVRAMInfo]] = None,
    bucket_size: int = 40_000_000,
    grad_reduce_in_fp32: bool = True,
    use_distributed_optimizer: bool = True,
    pad_for_high_nccl_busbw: bool = False,
) -> TierAwareFullLayout:
    """Compute tier-aware flat buffer layouts for all dtype groups in *params*.

    This is the single entry-point that external code (ZeRO engine /
    optimizer constructor) should call.  It mirrors the role of
    ``DistributedOptimizer.compute_full_param_layout`` in Megatron's upstream
    commit but produces VRAM-proportional slices instead of uniform ones.

    Args:
        params: All trainable parameters of the model.
        dp_process_group: The data-parallel process group.  Defaults to
            the global default group.
        vram_infos: Pre-gathered VRAM info.  When ``None``, all-gathered
            automatically via ``gather_dp_vram_infos``.
        bucket_size: Approximate flat-buffer elements per gradient bucket.
        grad_reduce_in_fp32: Whether gradients are reduced in FP32.
        use_distributed_optimizer: Whether to apply alignment padding.
        pad_for_high_nccl_busbw: Extend bucket padding to 2^16 alignment.

    Returns:
        TierAwareFullLayout containing one TierAwareParamLayout per buffer key.
    """
    param_list = [p for p in params if p.requires_grad]

    if vram_infos is None:
        vram_infos = gather_dp_vram_infos(dp_process_group)

    dp_world_size = len(vram_infos)

    buffer_groups = group_params_by_buffer_key(param_list, grad_reduce_in_fp32)

    layouts: Dict[BufferKey, TierAwareParamLayout] = {}
    for key, group_params in buffer_groups.items():
        layout = TierAwareParamLayout.compute(
            params=group_params,
            vram_infos=vram_infos,
            dp_world_size=dp_world_size,
            bucket_size=bucket_size,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=use_distributed_optimizer,
            pad_for_high_nccl_busbw=pad_for_high_nccl_busbw,
        )
        layouts[key] = layout

    full_layout = TierAwareFullLayout(layouts=layouts, vram_infos=vram_infos)
    log.info(
        "%s LAYOUT complete num_buffer_keys=%d",
        _LOG_PREFIX, len(layouts),
    )
    return full_layout


# ---------------------------------------------------------------------------
# apply_tier_aware_layout — stamp params with their slice info
# ---------------------------------------------------------------------------

def apply_tier_aware_layout(
    params: Iterable[torch.nn.Parameter],
    full_layout: TierAwareFullLayout,
    dp_rank: Optional[int] = None,
    grad_reduce_in_fp32: bool = True,
) -> int:
    """Stamp ``_tier_layout_slice`` onto each parameter.

    After this call, ZeRO stage-3's partitioned-param coordinator can read
    ``param._tier_layout_slice`` → ``(start_elem, end_elem)`` to know the
    element range this rank is responsible for without re-computing the layout.

    Also stamps ``_tier_layout_bucket_id`` so downstream bucketing logic can
    verify consistency with the pre-computed bucket boundaries.

    Args:
        params: Model parameters to stamp.
        full_layout: The TierAwareFullLayout produced by ``compute_for_model``.
        dp_rank: This rank's position within the DP group.  Defaults to
            ``dist.get_rank()`` when dist is initialised, else 0.
        grad_reduce_in_fp32: Must match the value used in ``compute_for_model``.

    Returns:
        Number of parameters stamped (for logging / testing).
    """
    if dp_rank is None:
        dp_rank = dist.get_rank() if dist.is_initialized() else 0

    stamped = 0
    buffer_groups = group_params_by_buffer_key(
        [p for p in params if p.requires_grad], grad_reduce_in_fp32
    )

    for key, group_params in buffer_groups.items():
        layout = full_layout.layout_for(key)
        if layout is None:
            log.warning(
                "%s STAMP: no layout for BufferKey %s — skipping %d params",
                _LOG_PREFIX, key, len(group_params),
            )
            continue

        rank_slice = layout.slice_for_rank(dp_rank)

        for param in group_params:
            info = layout.param_index_map.get(param)
            if info is None:
                continue
            start_elem, end_elem, bucket_id = info
            param._tier_layout_slice = (start_elem, end_elem)  # type: ignore[attr-defined]
            param._tier_layout_bucket_id = bucket_id  # type: ignore[attr-defined]
            param._tier_rank_slice = rank_slice  # type: ignore[attr-defined]
            stamped += 1

    log.info(
        "%s STAMP rank=%d stamped=%d params with tier_layout_slice",
        _LOG_PREFIX, dp_rank, stamped,
    )
    return stamped


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "GPUVRAMInfo",
    "BufferKey",
    "TierAwareParamLayout",
    "TierAwareFullLayout",
    # VRAM probing
    "probe_local_vram",
    "gather_dp_vram_infos",
    "compute_vram_weights",
    "compute_tier_slices",
    # Layout computation
    "compute_for_model",
    "group_params_by_buffer_key",
    # Layout application
    "apply_tier_aware_layout",
    # Alignment helpers (exported for unit tests)
    "_pad_param_start",
    "_pad_bucket_end",
]
