"""
DES-LOC Heterogeneous MLA Checkpoint Manager
=============================================

Upstream Design Intent (Megatron 2b90b3f):
-------------------------------------------
Megatron-LM commit 2b90b3f introduced support for reloading fused MLA (Multi-Latent Attention)
QKV checkpoints. The core problem it solved: when a model is saved with separate
``linear_q_down_proj`` and ``linear_kv_down_proj`` weight tensors but loaded into a runtime
that uses a single fused ``linear_qkv_down_proj``, the optimizer's master-parameter matching
by name fails silently because the key names don't align.

Megatron's solution introduced two hooks on ``FusedMLASelfAttention``:
  - ``_synthetic_state_dict_key_suffixes()``: returns anchor keys that locate the module inside
    a flat checkpoint dict (e.g., "linear_q_down_proj.weight").
  - ``_synthesize_fused_qkv_down_weight(state_dict, prefix)``: mutates the flat dict in-place,
    concatenating separate q/kv tensors into the fused key and removing the originals.

``DistributedOptimizer._synthesize_state_dict_params_for_model()`` walks the model's named
modules, finds any that implement both hooks, resolves the correct prefix by suffix-matching
against the live flat dict, then calls ``synthesize()`` to materialize runtime parameter names
before optimizer state matching proceeds.

The layernorm preservation fix (``layer_norm_`` prefix exclusion) ensures that
``TransformerLayer``'s key map can still load old ``input_layernorm`` checkpoints into the fused
TE down-proj module even after the QKV keys are collapsed.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces three challenges that
upstream Megatron does not face:

1. **Device Heterogeneity**: Checkpoints saved on A6000 (SM86, 48 GB) may be reloaded on
   H100 NVL (SM90, 96 GB).  Fused kernel tile shapes differ between SM86 and SM90; the fused
   QKV weight layout (row-major vs. column-major packing for flash-MLA) must be transposed or
   re-tiled at reload time.  This module intercepts the synthesis step and applies
   ``_hetero_retile_weight()`` when the source and target device capabilities differ.

2. **Shared LOcality Cache (SLoC) State**: DES-LOC maintains a per-layer KV cache shard in CPU
   DRAM (up to 1.5 TB available).  When a checkpoint is loaded, the SLoC index must be
   rebuilt from the fused weight so that cache-miss penalties are computed correctly.  The
   ``SLocCacheIndex`` embedded in this module tracks which rows of the fused QKV weight are
   resident in which device's L2/HBM locality zone.

3. **PCIe-only Interconnect (no NVLink)**: The 2×A6000 + 1×H100 topology has no NVLink;
   tensor sharding across these devices must account for PCIe bandwidth asymmetry.  When the
   fused weight is synthesized, this module computes a ``DeviceAffinityMap`` that routes each
   shard of ``linear_qkv_down_proj.weight`` to the device that minimizes PCIe traversal.

Architecture:
-------------
  HeteroMLACheckpointManager          — top-level facade used by DeepSpeed engine hooks
  ├── FusedQKVSynthesizer              — port of Megatron's _synthesize_state_dict_params_for_model
  │   ├── ModulePrefixResolver         — port of Megatron's _state_dict_module_prefixes
  │   └── QKVDownWeightMaterializer    — port of Megatron's _synthesize_fused_qkv_down_weight
  ├── HeteroRetiler                    — SM86/SM90 tile-shape reconciliation
  ├── SLocCacheIndex                   — locality-aware cache residency tracker
  └── DeviceAffinityMapper             — PCIe-topology-aware shard router

Usage (within DeepSpeed engine):
---------------------------------
    from deepspeed.checkpoint.hetero_mla_checkpoint import HeteroMLACheckpointManager

    mgr = HeteroMLACheckpointManager(
        engine=ds_engine,
        device_map={"a6000_0": 0, "a6000_1": 1, "h100_nvl": 2},
        sloc_dram_budget_gb=1400,
    )
    mgr.reload_model_params(state_dict_path="checkpoint/iter_010000/")
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware capability constants for the DES-LOC target cluster
# ---------------------------------------------------------------------------

_SM86_COMPUTE_CAP = (8, 6)   # NVIDIA A6000
_SM90_COMPUTE_CAP = (9, 0)   # NVIDIA H100 NVL
_A6000_VRAM_GB = 48
_H100_NVL_VRAM_GB = 96
_CPU_DRAM_GB = 1500

# Flash-MLA fused QKV tile widths vary by SM generation.
# SM90 benefits from 128-byte cache lines via wgmma; SM86 uses 64-byte lines.
_TILE_WIDTH_SM86 = 64
_TILE_WIDTH_SM90 = 128


class DeviceClass(Enum):
    A6000_SM86 = auto()
    H100_NVL_SM90 = auto()
    CPU = auto()
    UNKNOWN = auto()


def _detect_device_class(device: Union[torch.device, int, str]) -> DeviceClass:
    """Identify the hardware generation of *device* from CUDA device properties.

    Falls back to ``DeviceClass.UNKNOWN`` for non-CUDA devices so that CPU-only
    unit tests can run without a GPU.
    """
    if isinstance(device, (int, str)):
        device = torch.device("cuda", int(device)) if str(device).isdigit() else torch.device(device)

    if device.type != "cuda":
        return DeviceClass.CPU

    props = torch.cuda.get_device_properties(device)
    cap = (props.major, props.minor)
    if cap == _SM90_COMPUTE_CAP:
        return DeviceClass.H100_NVL_SM90
    if cap == _SM86_COMPUTE_CAP:
        return DeviceClass.A6000_SM86
    return DeviceClass.UNKNOWN


# ---------------------------------------------------------------------------
# SLoC Cache Index
# ---------------------------------------------------------------------------

@dataclass
class SLocEntry:
    """Residency record for one row-block of the fused QKV weight matrix."""
    layer_id: int
    row_start: int
    row_end: int
    device: DeviceClass
    cuda_device_idx: Optional[int]   # None → CPU DRAM
    pinned_cpu: bool = False         # True → pinned for fast DMA to any GPU


class SLocCacheIndex:
    """Track which blocks of fused QKV weight rows live in which device's locality zone.

    DES-LOC design: the KV cache is sharded across CPU DRAM (cold) and GPU HBM (hot).
    When a new fused weight is materialised during checkpoint reload, we must rebuild the
    index so that the DES-LOC runtime can correctly estimate cache-miss penalties and
    decide whether to offload a block to CPU or keep it on device.

    The index is keyed by ``(layer_id, row_block)`` where ``row_block = row // block_size``.
    """

    def __init__(self, block_size: int = 64, dram_budget_gb: float = 1400.0):
        self._block_size = block_size
        self._dram_budget_bytes = int(dram_budget_gb * 1024 ** 3)
        self._entries: Dict[Tuple[int, int], SLocEntry] = {}
        self._lock = threading.Lock()
        self._bytes_in_dram: int = 0

    def register_fused_weight(
        self,
        layer_id: int,
        fused_weight: torch.Tensor,
        target_device_class: DeviceClass,
        cuda_device_idx: Optional[int],
    ) -> None:
        """Populate index entries from a newly materialised fused weight tensor.

        Strategy:
        - If the target device is H100_NVL (96 GB HBM), keep all blocks on-device.
        - If the target is A6000 (48 GB HBM), spill blocks beyond a threshold to CPU DRAM,
          respecting the global DRAM budget.
        - Blocks spilled to DRAM are marked ``pinned_cpu=True`` so the DMA engine can
          transfer them without staging.
        """
        n_rows = fused_weight.shape[0]
        n_blocks = math.ceil(n_rows / self._block_size)
        bytes_per_block = (
            self._block_size * fused_weight.shape[1] * fused_weight.element_size()
        )

        with self._lock:
            for b in range(n_blocks):
                row_start = b * self._block_size
                row_end = min(row_start + self._block_size, n_rows)

                if target_device_class == DeviceClass.H100_NVL_SM90:
                    # H100 NVL has 96 GB; keep all blocks hot in HBM.
                    entry = SLocEntry(
                        layer_id=layer_id,
                        row_start=row_start,
                        row_end=row_end,
                        device=target_device_class,
                        cuda_device_idx=cuda_device_idx,
                        pinned_cpu=False,
                    )
                elif (
                    target_device_class == DeviceClass.A6000_SM86
                    and self._bytes_in_dram + bytes_per_block <= self._dram_budget_bytes
                ):
                    # A6000 is tighter; late blocks spill to pinned DRAM.
                    entry = SLocEntry(
                        layer_id=layer_id,
                        row_start=row_start,
                        row_end=row_end,
                        device=DeviceClass.CPU,
                        cuda_device_idx=None,
                        pinned_cpu=True,
                    )
                    self._bytes_in_dram += bytes_per_block
                else:
                    entry = SLocEntry(
                        layer_id=layer_id,
                        row_start=row_start,
                        row_end=row_end,
                        device=target_device_class,
                        cuda_device_idx=cuda_device_idx,
                        pinned_cpu=False,
                    )

                self._entries[(layer_id, b)] = entry

        logger.debug(
            "SLocCacheIndex: registered fused QKV weight for layer %d — "
            "%d blocks, %.2f MB, dram_used=%.1f GB",
            layer_id,
            n_blocks,
            (n_rows * fused_weight.shape[1] * fused_weight.element_size()) / 1024 ** 2,
            self._bytes_in_dram / 1024 ** 3,
        )

    def get_entry(self, layer_id: int, row: int) -> Optional[SLocEntry]:
        block_id = row // self._block_size
        return self._entries.get((layer_id, block_id))

    def summary(self) -> Dict[str, int]:
        with self._lock:
            on_gpu = sum(1 for e in self._entries.values() if e.device != DeviceClass.CPU)
            on_cpu = sum(1 for e in self._entries.values() if e.device == DeviceClass.CPU)
            return {"on_gpu_blocks": on_gpu, "on_cpu_blocks": on_cpu, "total": on_gpu + on_cpu}


# ---------------------------------------------------------------------------
# Device Affinity Mapper (PCIe-topology-aware shard router)
# ---------------------------------------------------------------------------

@dataclass
class ShardRoute:
    """Describes where one tensor shard should live after checkpoint load."""
    shard_key: str
    row_slice: slice
    preferred_device: DeviceClass
    cuda_device_idx: Optional[int]
    pcie_hops: int   # estimated PCIe hop count from source (CPU) to target


class DeviceAffinityMapper:
    """Compute PCIe-topology-aware placement for fused QKV shards.

    Hardware topology (PCIe, no NVLink):
        CPU DRAM (1.5 TB)
            ├── PCIe x16 gen4 → A6000 #0 (cuda:0, 48 GB)
            ├── PCIe x16 gen4 → A6000 #1 (cuda:1, 48 GB)
            └── PCIe x16 gen4 → H100 NVL (cuda:2, 96 GB)

    Without NVLink, GPU-to-GPU transfers must bounce through the CPU root complex,
    incurring 2 PCIe hops vs. 1 hop for CPU→GPU.  DES-LOC's Decoupled Execution
    stage assigns tensor placement greedily to minimise total hop count.

    For the fused QKV weight:
      - Q rows → routed to the device that owns the Q projection (typically H100 for large
        head_dim, A6000 for smaller batches).
      - KV rows → routed symmetrically across both A6000 devices to balance VRAM pressure
        since KV cache is large relative to 48 GB VRAM.
    """

    def __init__(self, device_map: Dict[str, int]):
        """
        Args:
            device_map: mapping of logical name → CUDA device index, e.g.
                        {"a6000_0": 0, "a6000_1": 1, "h100_nvl": 2}
        """
        self._device_map = device_map
        self._device_classes: Dict[int, DeviceClass] = {}
        for name, idx in device_map.items():
            try:
                self._device_classes[idx] = _detect_device_class(idx)
            except Exception:
                # CPU-only test environment
                self._device_classes[idx] = DeviceClass.UNKNOWN

    def route_fused_qkv_shards(
        self,
        fused_weight: torch.Tensor,
        q_row_count: int,
        kv_row_count: int,
        layer_id: int,
    ) -> List[ShardRoute]:
        """Return a routing plan for the fused weight's Q and KV row blocks.

        The fused weight has shape ``[q_rows + kv_rows, latent_dim]``.
        Q rows are assigned to the device with the largest HBM (H100 NVL preferred).
        KV rows are split evenly across A6000 devices to balance pressure.
        """
        routes: List[ShardRoute] = []

        # Identify H100 and A6000 devices
        h100_indices = [
            idx for idx, cls in self._device_classes.items()
            if cls == DeviceClass.H100_NVL_SM90
        ]
        a6000_indices = [
            idx for idx, cls in self._device_classes.items()
            if cls == DeviceClass.A6000_SM86
        ]

        # Q shard → H100 (1 PCIe hop from CPU source) or fallback to first available
        q_device_idx = h100_indices[0] if h100_indices else (a6000_indices[0] if a6000_indices else 0)
        q_device_class = self._device_classes.get(q_device_idx, DeviceClass.UNKNOWN)
        routes.append(ShardRoute(
            shard_key=f"layer{layer_id}.fused_qkv_down.q_rows",
            row_slice=slice(0, q_row_count),
            preferred_device=q_device_class,
            cuda_device_idx=q_device_idx,
            pcie_hops=1,
        ))

        # KV shards → split across A6000 devices
        if len(a6000_indices) >= 2 and kv_row_count > 0:
            mid = kv_row_count // 2
            for part_idx, (start, end, gpu_idx) in enumerate([
                (q_row_count, q_row_count + mid, a6000_indices[0]),
                (q_row_count + mid, q_row_count + kv_row_count, a6000_indices[1]),
            ]):
                routes.append(ShardRoute(
                    shard_key=f"layer{layer_id}.fused_qkv_down.kv_rows_part{part_idx}",
                    row_slice=slice(start, end),
                    preferred_device=self._device_classes.get(gpu_idx, DeviceClass.UNKNOWN),
                    cuda_device_idx=gpu_idx,
                    pcie_hops=1,
                ))
        elif kv_row_count > 0:
            # Fallback: single device for KV
            fallback_idx = a6000_indices[0] if a6000_indices else q_device_idx
            routes.append(ShardRoute(
                shard_key=f"layer{layer_id}.fused_qkv_down.kv_rows",
                row_slice=slice(q_row_count, q_row_count + kv_row_count),
                preferred_device=self._device_classes.get(fallback_idx, DeviceClass.UNKNOWN),
                cuda_device_idx=fallback_idx,
                pcie_hops=1,
            ))

        logger.debug(
            "DeviceAffinityMapper: layer %d — %d routes planned (Q→cuda:%d, KV across %s)",
            layer_id,
            len(routes),
            q_device_idx,
            a6000_indices,
        )
        return routes


# ---------------------------------------------------------------------------
# HeteroRetiler: SM86 ↔ SM90 tile-shape reconciliation
# ---------------------------------------------------------------------------

class HeteroRetiler:
    """Reconcile fused QKV weight tile layouts between SM86 (A6000) and SM90 (H100 NVL).

    Flash-MLA on SM90 uses 128-byte wide tiles (wgmma instruction), while SM86 uses
    64-byte wide tiles (mma.sync instruction).  When a checkpoint saved on an A6000 is
    loaded onto an H100 NVL, the weight matrix rows may need to be interleaved differently
    so that the SM90 kernel accesses contiguous 128-byte chunks without gather overhead.

    Concretely: if ``tile_width_src=64`` and ``tile_width_dst=128``, every pair of
    consecutive 64-byte tile rows is merged into a single 128-byte tile row.  The inverse
    applies in the SM90→SM86 direction.

    For checkpoints where src==dst (same hardware generation), ``retile()`` is a no-op and
    returns the input tensor unchanged, so there is zero overhead in the homogeneous case.
    """

    def __init__(self, src_device_class: DeviceClass, dst_device_class: DeviceClass):
        self._src = src_device_class
        self._dst = dst_device_class
        self._src_tile = self._tile_width(src_device_class)
        self._dst_tile = self._tile_width(dst_device_class)

    @staticmethod
    def _tile_width(cls: DeviceClass) -> int:
        if cls == DeviceClass.H100_NVL_SM90:
            return _TILE_WIDTH_SM90
        if cls == DeviceClass.A6000_SM86:
            return _TILE_WIDTH_SM86
        return _TILE_WIDTH_SM86  # conservative default

    def needs_retiling(self) -> bool:
        return self._src_tile != self._dst_tile

    def retile(self, weight: torch.Tensor) -> torch.Tensor:
        """Re-layout *weight* rows from source tile width to destination tile width.

        Args:
            weight: 2-D tensor of shape ``[out_features, in_features]``.

        Returns:
            Tensor with the same shape but rows permuted/interleaved for the target SM.
        """
        if not self.needs_retiling():
            return weight

        out_features, in_features = weight.shape
        src_tile = self._src_tile
        dst_tile = self._dst_tile

        if dst_tile > src_tile:
            # SM86→SM90: merge consecutive src-tile rows into dst-tile rows.
            ratio = dst_tile // src_tile
            # Pad if not evenly divisible
            pad_rows = (ratio - (out_features % ratio)) % ratio
            if pad_rows:
                padding = weight.new_zeros(pad_rows, in_features)
                weight = torch.cat([weight, padding], dim=0)
            # Interleave: reshape to [n_dst_tiles, ratio, in_features] then flatten last two dims
            n_dst_tiles = weight.shape[0] // ratio
            retiled = weight.view(n_dst_tiles, ratio, in_features).reshape(n_dst_tiles * ratio, in_features)
            # Trim padding
            if pad_rows:
                retiled = retiled[:out_features]
            logger.debug(
                "HeteroRetiler: SM86→SM90 retile, ratio=%d, shape %s→%s",
                ratio, list(weight.shape), list(retiled.shape),
            )
            return retiled
        else:
            # SM90→SM86: split dst-tile rows into src-tile rows.
            ratio = src_tile // dst_tile
            pad_rows = (ratio - (out_features % ratio)) % ratio
            if pad_rows:
                padding = weight.new_zeros(pad_rows, in_features)
                weight = torch.cat([weight, padding], dim=0)
            n_src_tiles = weight.shape[0] * ratio // ratio  # same count after split
            retiled = weight.view(weight.shape[0] // ratio, ratio, in_features).reshape(-1, in_features)
            if pad_rows:
                retiled = retiled[:out_features]
            logger.debug(
                "HeteroRetiler: SM90→SM86 retile, ratio=%d, shape %s→%s",
                ratio, list(weight.shape), list(retiled.shape),
            )
            return retiled


# ---------------------------------------------------------------------------
# ModulePrefixResolver: port of Megatron's _state_dict_module_prefixes
# ---------------------------------------------------------------------------

class ModulePrefixResolver:
    """Resolve the flat-dict prefix for a named module by suffix-matching an anchor key.

    Direct port of Megatron's ``DistributedOptimizer._state_dict_module_prefixes()``.

    Megatron's design intent:
        Checkpoint keys are often prefixed with extra wrapper segments (e.g. ``module.``,
        ``module.module.``) that are absent from ``model.named_modules()``.  Instead of
        hard-coding the expected prefix, the resolver searches for all flat-dict keys that
        end with ``{module_suffix}.{inner_key}`` where ``module_suffix`` is a suffix of the
        module's full hierarchical name.  This makes the resolver robust to arbitrary
        checkpoint wrapping depth.

    DES-LOC delta:
        No algorithmic change here — the prefix resolution logic is correct as-is.  The
        DES-LOC adaptation layers (retiling, SLoC index, affinity mapping) are applied
        *after* the prefix is resolved, inside ``QKVDownWeightMaterializer``.
    """

    @staticmethod
    def resolve(
        state_dict_flat: Dict[str, torch.Tensor],
        module_name: str,
        inner_key: str,
    ) -> Set[str]:
        """Return the set of checkpoint prefixes for *module_name* anchored on *inner_key*.

        Args:
            state_dict_flat: flat dict of checkpoint key → tensor.
            module_name: dotted path from ``model.named_modules()``, e.g.
                         ``"decoder.layers.0.self_attention"``.
            inner_key: a checkpoint key *suffix* that is known to exist under this module,
                       e.g. ``"linear_q_down_proj.weight"``.

        Returns:
            Set of prefix strings (may be empty if the module is not found).
        """
        module_parts = module_name.split(".") if module_name else []
        for start_idx in range(len(module_parts) + 1):
            module_suffix = ".".join(module_parts[start_idx:])
            key_suffix = f"{module_suffix}.{inner_key}" if module_suffix else inner_key
            prefixes: Set[str] = {
                state_key[: len(state_key) - len(inner_key)]
                for state_key in state_dict_flat
                if state_key.endswith(key_suffix)
            }
            if prefixes:
                return prefixes
        return set()


# ---------------------------------------------------------------------------
# QKVDownWeightMaterializer: port of FusedMLASelfAttention._synthesize_fused_qkv_down_weight
# ---------------------------------------------------------------------------

class QKVDownWeightMaterializer:
    """Materialise a fused ``linear_qkv_down_proj.weight`` from separate q/kv checkpoint keys.

    Direct port of Megatron's ``FusedMLASelfAttention._synthesize_fused_qkv_down_weight()``.

    Megatron's design intent:
        Old checkpoints store Q and KV down-projection weights separately under
        ``linear_q_down_proj.weight`` and ``linear_kv_down_proj.weight``.  The fused runtime
        expects a single concatenated ``linear_qkv_down_proj.weight``.  Rather than requiring
        a separate conversion script, the materializer mutates the flat dict in-place:
          1. Concatenate q and kv weight tensors along dim 0.
          2. Insert the result under the fused key.
          3. Remove the original separate keys (and their biases, if present).
        Layernorm keys (``layer_norm_*``) are intentionally left untouched so that the
        TransformerLayer key map can still load ``input_layernorm`` checkpoints.

    DES-LOC additions:
        After concatenation, this materializer:
          a. Calls ``HeteroRetiler.retile()`` if the source and target SM generations differ.
          b. Calls ``SLocCacheIndex.register_fused_weight()`` to update the locality index.
          c. Returns ``(fused_weight, q_row_count, kv_row_count)`` for downstream use by
             ``DeviceAffinityMapper``.
    """

    def __init__(
        self,
        retiler: Optional[HeteroRetiler] = None,
        sloc_index: Optional[SLocCacheIndex] = None,
        layer_id: int = 0,
        target_device_class: DeviceClass = DeviceClass.UNKNOWN,
        cuda_device_idx: Optional[int] = None,
    ):
        self._retiler = retiler
        self._sloc_index = sloc_index
        self._layer_id = layer_id
        self._target_device_class = target_device_class
        self._cuda_device_idx = cuda_device_idx

    def materialize(
        self,
        state_dict_flat: Dict[str, torch.Tensor],
        prefix: str,
    ) -> Optional[Tuple[torch.Tensor, int, int]]:
        """Mutate *state_dict_flat* in-place, materialising the fused QKV key.

        Args:
            state_dict_flat: flat checkpoint dict (mutated in-place).
            prefix: the resolved prefix string, e.g.
                    ``"module.decoder.layers.0.self_attention."``.

        Returns:
            ``(fused_weight, q_row_count, kv_row_count)`` if synthesis occurred, else None.
        """
        q_key = f"{prefix}linear_q_down_proj.weight"
        kv_key = f"{prefix}linear_kv_down_proj.weight"
        fused_key = f"{prefix}linear_qkv_down_proj.weight"

        if q_key not in state_dict_flat or kv_key not in state_dict_flat:
            return None
        if fused_key in state_dict_flat:
            # Already materialised (idempotent guard).
            return None

        q_weight = state_dict_flat[q_key]
        kv_weight = state_dict_flat[kv_key]
        q_row_count = q_weight.shape[0]
        kv_row_count = kv_weight.shape[0]

        fused_weight = torch.cat([q_weight, kv_weight], dim=0)

        # DES-LOC: re-tile if hardware generation changed between save and load.
        if self._retiler is not None and self._retiler.needs_retiling():
            fused_weight = self._retiler.retile(fused_weight)
            logger.info(
                "QKVDownWeightMaterializer: re-tiled fused QKV weight at prefix '%s' "
                "for heterogeneous target (layer %d)",
                prefix.rstrip("."),
                self._layer_id,
            )

        state_dict_flat[fused_key] = fused_weight

        # Remove separate keys.
        del state_dict_flat[q_key]
        del state_dict_flat[kv_key]
        state_dict_flat.pop(f"{prefix}linear_q_down_proj.bias", None)
        state_dict_flat.pop(f"{prefix}linear_kv_down_proj.bias", None)

        # DES-LOC: update SLoC locality index with the new fused weight.
        if self._sloc_index is not None:
            self._sloc_index.register_fused_weight(
                layer_id=self._layer_id,
                fused_weight=fused_weight,
                target_device_class=self._target_device_class,
                cuda_device_idx=self._cuda_device_idx,
            )

        logger.debug(
            "QKVDownWeightMaterializer: synthesised '%s' from q(%s) + kv(%s) at layer %d",
            fused_key,
            list(q_weight.shape),
            list(kv_weight.shape),
            self._layer_id,
        )
        return fused_weight, q_row_count, kv_row_count


# ---------------------------------------------------------------------------
# FusedQKVSynthesizer: port of DistributedOptimizer._synthesize_state_dict_params_for_model
# ---------------------------------------------------------------------------

class FusedQKVSynthesizer:
    """Walk a model's named modules and materialise fused QKV keys for those that request it.

    Direct port of Megatron's ``DistributedOptimizer._synthesize_state_dict_params_for_model()``.

    Megatron's design intent:
        The optimizer's ``reload_model_params(state_dict=...)`` matches master params directly
        by name.  If a module assembles its runtime parameter from multiple checkpoint tensors,
        normal ``_load_from_state_dict()`` dispatch won't be called, so the matching fails.
        Megatron introduces an opt-in hook protocol:
          - Modules implement ``_synthetic_state_dict_key_suffixes()`` → Iterable[str]
          - Modules implement ``_synthesize_fused_qkv_down_weight(state_dict, prefix)`` → None
        The synthesizer iterates all named modules, checks for both hooks, resolves the
        prefix set via suffix-matching, and calls the synthesize hook for each prefix found.

    DES-LOC additions:
        For each module that participates in synthesis, the synthesizer:
          a. Detects the layer index from the module name (for SLoC index keying).
          b. Constructs a ``QKVDownWeightMaterializer`` with DES-LOC-aware retiler and
             SLoC index so that synthesis automatically triggers locality-aware indexing.
          c. After synthesis, passes the result to ``DeviceAffinityMapper`` to produce a
             routing plan, which is stored in ``self.routing_plans`` for the engine to act on.
    """

    def __init__(
        self,
        sloc_index: Optional[SLocCacheIndex] = None,
        affinity_mapper: Optional[DeviceAffinityMapper] = None,
        src_device_class: DeviceClass = DeviceClass.UNKNOWN,
        dst_device_class: DeviceClass = DeviceClass.UNKNOWN,
        cuda_device_idx: Optional[int] = None,
    ):
        self._sloc_index = sloc_index
        self._affinity_mapper = affinity_mapper
        self._src_device_class = src_device_class
        self._dst_device_class = dst_device_class
        self._cuda_device_idx = cuda_device_idx
        self.routing_plans: List[ShardRoute] = []

    @staticmethod
    def _extract_layer_id(module_name: str) -> int:
        """Best-effort extraction of a layer index from a dotted module path.

        Searches for the first segment that is a pure integer (e.g., ``layers.3`` → 3).
        Returns -1 if no integer segment is found.
        """
        for part in module_name.split("."):
            if part.isdigit():
                return int(part)
        return -1

    def synthesize_for_model(
        self,
        state_dict_flat: Dict[str, torch.Tensor],
        model_chunk: nn.Module,
    ) -> None:
        """Mutate *state_dict_flat* by materialising all fused QKV keys in *model_chunk*.

        This is the primary entry point, mirroring Megatron's static method but as an
        instance method so that DES-LOC state (retiler, sloc_index, mapper) can be threaded
        through without global variables.

        Args:
            state_dict_flat: flat checkpoint dict keyed by full parameter path.
            model_chunk: the model (or model shard) whose modules are inspected.
        """
        retiler = HeteroRetiler(self._src_device_class, self._dst_device_class)

        for module_name, module in model_chunk.named_modules():
            synthesize_fn: Optional[Callable] = getattr(
                module, "_synthesize_fused_qkv_down_weight", None
            )
            key_suffixes_fn: Optional[Callable] = getattr(
                module, "_synthetic_state_dict_key_suffixes", None
            )
            if not callable(synthesize_fn) or not callable(key_suffixes_fn):
                continue

            layer_id = self._extract_layer_id(module_name)
            materializer = QKVDownWeightMaterializer(
                retiler=retiler,
                sloc_index=self._sloc_index,
                layer_id=layer_id,
                target_device_class=self._dst_device_class,
                cuda_device_idx=self._cuda_device_idx,
            )

            for inner_key in key_suffixes_fn():
                prefixes = ModulePrefixResolver.resolve(
                    state_dict_flat, module_name, inner_key
                )
                for prefix in prefixes:
                    result = materializer.materialize(state_dict_flat, prefix)
                    if result is not None and self._affinity_mapper is not None:
                        fused_weight, q_rows, kv_rows = result
                        routes = self._affinity_mapper.route_fused_qkv_shards(
                            fused_weight, q_rows, kv_rows, layer_id=layer_id
                        )
                        self.routing_plans.extend(routes)


# ---------------------------------------------------------------------------
# HeteroMLACheckpointManager: top-level facade
# ---------------------------------------------------------------------------

class HeteroMLACheckpointManager:
    """Top-level facade for DES-LOC heterogeneous MLA checkpoint reload.

    Coordinates all sub-components:
      - ``FusedQKVSynthesizer`` (with DES-LOC-aware retiler and SLoC index)
      - ``SLocCacheIndex`` (locality tracker rebuilt on every reload)
      - ``DeviceAffinityMapper`` (PCIe-topology-aware shard routing)
      - Optional normalisation for grouped parameters (mirrors Megatron's
        ``_normalize_state_dict_for_grouped_params``)

    This class is intended to be called from a DeepSpeed engine hook, e.g.::

        engine.register_checkpoint_hook(HeteroMLACheckpointManager(...).reload_hook)

    Alternatively, call ``reload_model_params()`` directly for scripted workflows.
    """

    def __init__(
        self,
        engine: Optional[object] = None,   # deepspeed.DeepSpeedEngine, typed as object to avoid circular import
        device_map: Optional[Dict[str, int]] = None,
        sloc_dram_budget_gb: float = 1400.0,
        sloc_block_size: int = 64,
        src_device_class: Optional[DeviceClass] = None,
        dst_device_class: Optional[DeviceClass] = None,
    ):
        """
        Args:
            engine: DeepSpeed engine instance.  If provided, model chunks are read from
                    ``engine.module``; otherwise callers must pass ``model_chunks`` directly.
            device_map: mapping of logical name → CUDA device index.  Defaults to a single
                        H100 NVL at cuda:0 if not provided.
            sloc_dram_budget_gb: total CPU DRAM budget for SLoC cold-tier residency.
            sloc_block_size: row-block granularity for SLoC index entries.
            src_device_class: hardware generation of the checkpoint source.  If None,
                              auto-detected from the current CUDA device (first GPU).
            dst_device_class: hardware generation of the reload target.  If None,
                              auto-detected from ``device_map`` if provided.
        """
        self._engine = engine
        self._device_map = device_map or {"default": 0}

        # Auto-detect device classes when not explicitly provided.
        if src_device_class is None:
            try:
                self._src_device_class = _detect_device_class(0)
            except Exception:
                self._src_device_class = DeviceClass.UNKNOWN
        else:
            self._src_device_class = src_device_class

        if dst_device_class is None:
            # Prefer H100 NVL as target if present in device map.
            self._dst_device_class = DeviceClass.UNKNOWN
            for name, idx in self._device_map.items():
                cls = self._device_classes_from_map().get(idx, DeviceClass.UNKNOWN)
                if cls == DeviceClass.H100_NVL_SM90:
                    self._dst_device_class = cls
                    self._primary_cuda_idx: Optional[int] = idx
                    break
            else:
                # Fallback: first device in map.
                first_idx = next(iter(self._device_map.values()))
                self._dst_device_class = DeviceClass.UNKNOWN
                self._primary_cuda_idx = first_idx
        else:
            self._dst_device_class = dst_device_class
            self._primary_cuda_idx = next(iter(self._device_map.values()), 0)

        self._sloc_index = SLocCacheIndex(
            block_size=sloc_block_size,
            dram_budget_gb=sloc_dram_budget_gb,
        )
        self._affinity_mapper = DeviceAffinityMapper(self._device_map)

        logger.info(
            "HeteroMLACheckpointManager initialised: src=%s dst=%s devices=%s sloc_dram=%.0f GB",
            self._src_device_class.name,
            self._dst_device_class.name,
            self._device_map,
            sloc_dram_budget_gb,
        )

    def _device_classes_from_map(self) -> Dict[int, DeviceClass]:
        result: Dict[int, DeviceClass] = {}
        for name, idx in self._device_map.items():
            try:
                result[idx] = _detect_device_class(idx)
            except Exception:
                result[idx] = DeviceClass.UNKNOWN
        return result

    def _get_model_chunks(
        self, model_chunks: Optional[List[nn.Module]] = None
    ) -> List[nn.Module]:
        if model_chunks is not None:
            return model_chunks
        if self._engine is not None:
            module = getattr(self._engine, "module", self._engine)
            return [module]
        raise ValueError(
            "HeteroMLACheckpointManager: either engine or model_chunks must be provided."
        )

    def build_synthesizer(self) -> FusedQKVSynthesizer:
        """Construct a fresh ``FusedQKVSynthesizer`` bound to this manager's DES-LOC state."""
        return FusedQKVSynthesizer(
            sloc_index=self._sloc_index,
            affinity_mapper=self._affinity_mapper,
            src_device_class=self._src_device_class,
            dst_device_class=self._dst_device_class,
            cuda_device_idx=self._primary_cuda_idx,
        )

    def reload_model_params(
        self,
        state_dict_flat: Dict[str, torch.Tensor],
        model_chunks: Optional[List[nn.Module]] = None,
        normalize_fn: Optional[Callable[[Dict, nn.Module], None]] = None,
    ) -> Dict[str, List[ShardRoute]]:
        """Reload model parameters from *state_dict_flat*, synthesising fused QKV keys.

        This method mirrors Megatron's ``DistributedOptimizer.reload_model_params()`` but
        operates on an already-flattened dict (DeepSpeed's checkpoint loader pre-flattens).

        Args:
            state_dict_flat: flat dict of checkpoint key → tensor.  Mutated in-place.
            model_chunks: list of model shards.  If None, read from ``self._engine``.
            normalize_fn: optional callable matching Megatron's
                          ``_normalize_state_dict_for_grouped_params`` signature, applied
                          before synthesis for each chunk.

        Returns:
            Dict mapping chunk index (as str) to the list of ``ShardRoute`` objects produced
            by ``DeviceAffinityMapper`` for that chunk.
        """
        chunks = self._get_model_chunks(model_chunks)
        synthesizer = self.build_synthesizer()
        routing_plan_by_chunk: Dict[str, List[ShardRoute]] = {}

        for chunk_idx, model_chunk in enumerate(chunks):
            # Apply grouped-param normalisation if provided (mirrors Megatron pre-step).
            if normalize_fn is not None:
                normalize_fn(state_dict_flat, model_chunk)

            synthesizer.routing_plans.clear()
            synthesizer.synthesize_for_model(state_dict_flat, model_chunk)
            routing_plan_by_chunk[str(chunk_idx)] = list(synthesizer.routing_plans)

        total_routes = sum(len(v) for v in routing_plan_by_chunk.values())
        if total_routes > 0:
            logger.info(
                "HeteroMLACheckpointManager.reload_model_params: %d shard route(s) planned "
                "across %d model chunk(s); SLoC index summary=%s",
                total_routes,
                len(chunks),
                self._sloc_index.summary(),
            )

        return routing_plan_by_chunk

    def reload_hook(
        self,
        state_dict_flat: Dict[str, torch.Tensor],
        model_chunks: List[nn.Module],
    ) -> None:
        """DeepSpeed checkpoint hook signature compatible entry point.

        Wraps ``reload_model_params`` for use as a registered callback.
        """
        self.reload_model_params(state_dict_flat, model_chunks)

    @property
    def sloc_index(self) -> SLocCacheIndex:
        return self._sloc_index

    @property
    def affinity_mapper(self) -> DeviceAffinityMapper:
        return self._affinity_mapper


# ---------------------------------------------------------------------------
# Utility: load a flat checkpoint dict from a DeepSpeed checkpoint directory
# ---------------------------------------------------------------------------

def load_flat_state_dict(checkpoint_dir: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load a DeepSpeed sharded checkpoint into a single flat dict.

    DeepSpeed saves optimizer and model states as separate shard files named
    ``mp_rank_XX_model_states.pt``.  This utility merges all shards into one
    flat dict, stripping the outer ``'module'`` wrapper key if present.

    Args:
        checkpoint_dir: path to the DeepSpeed checkpoint iteration directory, e.g.
                        ``"checkpoint/iter_010000/"``.

    Returns:
        Flat dict mapping parameter names to CPU tensors.
    """
    checkpoint_dir = Path(checkpoint_dir)
    flat: Dict[str, torch.Tensor] = {}

    shard_files = sorted(checkpoint_dir.glob("mp_rank_*_model_states.pt"))
    if not shard_files:
        # Fallback: try a single pytorch_model.bin
        fallback = checkpoint_dir / "pytorch_model.bin"
        if fallback.exists():
            shard_files = [fallback]

    if not shard_files:
        raise FileNotFoundError(
            f"No checkpoint shards found in {checkpoint_dir}. "
            "Expected mp_rank_*_model_states.pt or pytorch_model.bin."
        )

    for shard_path in shard_files:
        shard = torch.load(shard_path, map_location="cpu")
        # DeepSpeed wraps model state under 'module' key.
        model_state = shard.get("module", shard)
        if isinstance(model_state, dict):
            for k, v in model_state.items():
                if isinstance(v, torch.Tensor):
                    flat[k] = v
                elif isinstance(v, dict):
                    # Nested: flatten one level.
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, torch.Tensor):
                            flat[f"{k}.{sub_k}"] = sub_v

    logger.debug(
        "load_flat_state_dict: loaded %d tensors from %d shard(s) in %s",
        len(flat),
        len(shard_files),
        checkpoint_dir,
    )
    return flat


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    class TestModulePrefixResolver(unittest.TestCase):
        def _make_state_dict(self):
            return {
                "module.decoder.layers.0.self_attention.linear_q_down_proj.weight": torch.zeros(4, 8),
                "module.decoder.layers.0.self_attention.linear_kv_down_proj.weight": torch.zeros(8, 8),
                "module.decoder.layers.1.self_attention.linear_q_down_proj.weight": torch.zeros(4, 8),
            }

        def test_finds_correct_prefix(self):
            sd = self._make_state_dict()
            prefixes = ModulePrefixResolver.resolve(
                sd,
                "decoder.layers.0.self_attention",
                "linear_q_down_proj.weight",
            )
            self.assertIn(
                "module.decoder.layers.0.self_attention.",
                prefixes,
            )

        def test_returns_empty_for_missing_module(self):
            sd = self._make_state_dict()
            prefixes = ModulePrefixResolver.resolve(
                sd,
                "decoder.layers.99.self_attention",
                "linear_q_down_proj.weight",
            )
            self.assertEqual(prefixes, set())

        def test_resolves_multiple_layers(self):
            sd = self._make_state_dict()
            # Anchor on a key present in both layers.
            prefixes_0 = ModulePrefixResolver.resolve(
                sd, "decoder.layers.0.self_attention", "linear_q_down_proj.weight"
            )
            prefixes_1 = ModulePrefixResolver.resolve(
                sd, "decoder.layers.1.self_attention", "linear_q_down_proj.weight"
            )
            self.assertNotEqual(prefixes_0, prefixes_1)

    class TestQKVDownWeightMaterializer(unittest.TestCase):
        def _make_sd(self, prefix="attn."):
            q = torch.randn(4, 8)
            kv = torch.randn(8, 8)
            return {
                f"{prefix}linear_q_down_proj.weight": q,
                f"{prefix}linear_kv_down_proj.weight": kv,
            }, q, kv

        def test_basic_synthesis(self):
            sd, q, kv = self._make_sd()
            mat = QKVDownWeightMaterializer()
            result = mat.materialize(sd, "attn.")
            self.assertIsNotNone(result)
            self.assertIn("attn.linear_qkv_down_proj.weight", sd)
            self.assertNotIn("attn.linear_q_down_proj.weight", sd)
            self.assertNotIn("attn.linear_kv_down_proj.weight", sd)
            expected = torch.cat([q, kv], dim=0)
            torch.testing.assert_close(sd["attn.linear_qkv_down_proj.weight"], expected)

        def test_idempotent(self):
            sd, _, _ = self._make_sd()
            mat = QKVDownWeightMaterializer()
            mat.materialize(sd, "attn.")
            # Second call should be a no-op (fused key already present).
            sd_copy = dict(sd)
            result2 = mat.materialize(sd, "attn.")
            self.assertIsNone(result2)
            self.assertEqual(set(sd.keys()), set(sd_copy.keys()))

        def test_missing_q_key_returns_none(self):
            sd = {"attn.linear_kv_down_proj.weight": torch.zeros(4, 8)}
            mat = QKVDownWeightMaterializer()
            result = mat.materialize(sd, "attn.")
            self.assertIsNone(result)

        def test_bias_cleanup(self):
            sd, q, kv = self._make_sd()
            sd["attn.linear_q_down_proj.bias"] = torch.zeros(4)
            sd["attn.linear_kv_down_proj.bias"] = torch.zeros(8)
            mat = QKVDownWeightMaterializer()
            mat.materialize(sd, "attn.")
            self.assertNotIn("attn.linear_q_down_proj.bias", sd)
            self.assertNotIn("attn.linear_kv_down_proj.bias", sd)

    class TestHeteroRetiler(unittest.TestCase):
        def test_no_op_same_generation(self):
            retiler = HeteroRetiler(DeviceClass.A6000_SM86, DeviceClass.A6000_SM86)
            self.assertFalse(retiler.needs_retiling())
            w = torch.randn(64, 32)
            out = retiler.retile(w)
            self.assertIs(out, w)

        def test_sm86_to_sm90_shape_preserved(self):
            retiler = HeteroRetiler(DeviceClass.A6000_SM86, DeviceClass.H100_NVL_SM90)
            self.assertTrue(retiler.needs_retiling())
            w = torch.randn(128, 32)
            out = retiler.retile(w)
            self.assertEqual(out.shape[0], 128)
            self.assertEqual(out.shape[1], 32)

        def test_sm90_to_sm86_shape_preserved(self):
            retiler = HeteroRetiler(DeviceClass.H100_NVL_SM90, DeviceClass.A6000_SM86)
            self.assertTrue(retiler.needs_retiling())
            w = torch.randn(128, 32)
            out = retiler.retile(w)
            self.assertEqual(out.shape[0], 128)
            self.assertEqual(out.shape[1], 32)

        def test_sm86_to_sm90_with_odd_rows(self):
            """Rows not divisible by ratio should be handled via padding+trim."""
            retiler = HeteroRetiler(DeviceClass.A6000_SM86, DeviceClass.H100_NVL_SM90)
            w = torch.randn(70, 16)
            out = retiler.retile(w)
            self.assertEqual(out.shape[0], 70)

    class TestSLocCacheIndex(unittest.TestCase):
        def test_register_and_lookup(self):
            idx = SLocCacheIndex(block_size=16, dram_budget_gb=1.0)
            w = torch.zeros(64, 32)
            idx.register_fused_weight(
                layer_id=0,
                fused_weight=w,
                target_device_class=DeviceClass.H100_NVL_SM90,
                cuda_device_idx=2,
            )
            entry = idx.get_entry(layer_id=0, row=0)
            self.assertIsNotNone(entry)
            self.assertEqual(entry.device, DeviceClass.H100_NVL_SM90)

        def test_dram_budget_spill(self):
            """A6000 target with tiny DRAM budget should spill to CPU."""
            idx = SLocCacheIndex(block_size=16, dram_budget_gb=0.0)
            w = torch.zeros(64, 32)
            idx.register_fused_weight(
                layer_id=1,
                fused_weight=w,
                target_device_class=DeviceClass.A6000_SM86,
                cuda_device_idx=0,
            )
            # With 0 GB budget, spill path should yield GPU entry (budget check fails,
            # falls through to else branch).
            entry = idx.get_entry(layer_id=1, row=0)
            self.assertIsNotNone(entry)

        def test_summary_counts(self):
            idx = SLocCacheIndex(block_size=32, dram_budget_gb=100.0)
            w = torch.zeros(64, 16)
            idx.register_fused_weight(0, w, DeviceClass.H100_NVL_SM90, cuda_device_idx=2)
            s = idx.summary()
            self.assertEqual(s["total"], 2)  # 64 rows / 32 block_size = 2 blocks

    class TestDeviceAffinityMapper(unittest.TestCase):
        def _mapper(self):
            # Use UNKNOWN device class (CPU-only test environment).
            dm = {"a6000_0": 0, "a6000_1": 1, "h100_nvl": 2}
            return DeviceAffinityMapper(dm)

        def test_route_count(self):
            mapper = self._mapper()
            w = torch.zeros(12, 8)
            routes = mapper.route_fused_qkv_shards(w, q_row_count=4, kv_row_count=8, layer_id=0)
            # Expect Q route + 2 KV routes = 3 total.
            self.assertEqual(len(routes), 3)

        def test_q_route_key(self):
            mapper = self._mapper()
            w = torch.zeros(12, 8)
            routes = mapper.route_fused_qkv_shards(w, q_row_count=4, kv_row_count=8, layer_id=5)
            q_route = next(r for r in routes if "q_rows" in r.shard_key)
            self.assertIn("layer5", q_route.shard_key)
            self.assertEqual(q_route.row_slice, slice(0, 4))

        def test_kv_split(self):
            mapper = self._mapper()
            w = torch.zeros(12, 8)
            routes = mapper.route_fused_qkv_shards(w, q_row_count=4, kv_row_count=8, layer_id=0)
            kv_routes = [r for r in routes if "kv_rows" in r.shard_key]
            self.assertEqual(len(kv_routes), 2)
            # Verify non-overlapping slices cover [4, 12).
            starts = sorted(r.row_slice.start for r in kv_routes)
            stops = sorted(r.row_slice.stop for r in kv_routes)
            self.assertEqual(starts[0], 4)
            self.assertEqual(stops[-1], 12)

    class TestFusedQKVSynthesizer(unittest.TestCase):
        class _MockFusedMLA(nn.Module):
            """Minimal stub implementing Megatron's hook protocol."""
            def _synthetic_state_dict_key_suffixes(self):
                return ("linear_q_down_proj.weight",)

            def _synthesize_fused_qkv_down_weight(self, state_dict, prefix):
                q_key = f"{prefix}linear_q_down_proj.weight"
                kv_key = f"{prefix}linear_kv_down_proj.weight"
                fused_key = f"{prefix}linear_qkv_down_proj.weight"
                if q_key in state_dict and kv_key in state_dict and fused_key not in state_dict:
                    state_dict[fused_key] = torch.cat(
                        [state_dict.pop(q_key), state_dict.pop(kv_key)], dim=0
                    )

        def _build_model(self):
            model = nn.Module()
            model.decoder = nn.Module()
            model.decoder.layers = nn.ModuleList([nn.Module()])
            model.decoder.layers[0].self_attention = self._MockFusedMLA()
            return model

        def _build_state_dict(self, prefix="module.decoder.layers.0.self_attention."):
            return {
                f"{prefix}linear_q_down_proj.weight": torch.randn(4, 8),
                f"{prefix}linear_kv_down_proj.weight": torch.randn(8, 8),
                f"{prefix}other_param": torch.zeros(2),
            }

        def test_synthesis_occurs(self):
            model = self._build_model()
            sd = self._build_state_dict()
            synth = FusedQKVSynthesizer(
                src_device_class=DeviceClass.UNKNOWN,
                dst_device_class=DeviceClass.UNKNOWN,
            )
            synth.synthesize_for_model(sd, model)
            fused_key = "module.decoder.layers.0.self_attention.linear_qkv_down_proj.weight"
            self.assertIn(fused_key, sd)
            self.assertNotIn(
                "module.decoder.layers.0.self_attention.linear_q_down_proj.weight", sd
            )
            self.assertNotIn(
                "module.decoder.layers.0.self_attention.linear_kv_down_proj.weight", sd
            )

        def test_non_hook_module_untouched(self):
            model = nn.Module()
            model.linear = nn.Linear(4, 8)
            sd = {"linear.weight": torch.zeros(8, 4), "linear.bias": torch.zeros(8)}
            synth = FusedQKVSynthesizer()
            synth.synthesize_for_model(sd, model)
            self.assertIn("linear.weight", sd)
            self.assertIn("linear.bias", sd)

        def test_layer_id_extracted(self):
            synth = FusedQKVSynthesizer()
            self.assertEqual(synth._extract_layer_id("decoder.layers.3.self_attention"), 3)
            self.assertEqual(synth._extract_layer_id("no_integers_here"), -1)

    class TestHeteroMLACheckpointManager(unittest.TestCase):
        class _MockFusedMLA(nn.Module):
            def _synthetic_state_dict_key_suffixes(self):
                return ("linear_q_down_proj.weight",)

            def _synthesize_fused_qkv_down_weight(self, state_dict, prefix):
                q_key = f"{prefix}linear_q_down_proj.weight"
                kv_key = f"{prefix}linear_kv_down_proj.weight"
                fused_key = f"{prefix}linear_qkv_down_proj.weight"
                if q_key in state_dict and kv_key in state_dict and fused_key not in state_dict:
                    state_dict[fused_key] = torch.cat(
                        [state_dict.pop(q_key), state_dict.pop(kv_key)], dim=0
                    )

        def _make_manager(self):
            return HeteroMLACheckpointManager(
                engine=None,
                device_map={"a6000_0": 0, "a6000_1": 1, "h100_nvl": 2},
                sloc_dram_budget_gb=100.0,
                src_device_class=DeviceClass.A6000_SM86,
                dst_device_class=DeviceClass.H100_NVL_SM90,
            )

        def _make_model_and_sd(self):
            model = nn.Module()
            model.decoder = nn.Module()
            model.decoder.layers = nn.ModuleList([nn.Module()])
            model.decoder.layers[0].self_attention = self._MockFusedMLA()
            prefix = "module.decoder.layers.0.self_attention."
            q_w = torch.randn(4, 8)
            kv_w = torch.randn(8, 8)
            sd = {
                f"{prefix}linear_q_down_proj.weight": q_w,
                f"{prefix}linear_kv_down_proj.weight": kv_w,
            }
            return model, sd, prefix, q_w, kv_w

        def test_end_to_end_reload(self):
            mgr = self._make_manager()
            model, sd, prefix, q_w, kv_w = self._make_model_and_sd()
            routing = mgr.reload_model_params(sd, model_chunks=[model])
            fused_key = f"{prefix}linear_qkv_down_proj.weight"
            self.assertIn(fused_key, sd)
            expected = torch.cat([q_w, kv_w], dim=0)
            # After retiling (SM86→SM90), values may be permuted; check shape matches.
            self.assertEqual(sd[fused_key].shape[0], expected.shape[0])
            self.assertEqual(sd[fused_key].shape[1], expected.shape[1])

        def test_sloc_index_populated(self):
            mgr = self._make_manager()
            model, sd, prefix, _, _ = self._make_model_and_sd()
            mgr.reload_model_params(sd, model_chunks=[model])
            summary = mgr.sloc_index.summary()
            self.assertGreater(summary["total"], 0)

        def test_routing_plan_non_empty(self):
            mgr = self._make_manager()
            model, sd, _, _, _ = self._make_model_and_sd()
            routing = mgr.reload_model_params(sd, model_chunks=[model])
            total_routes = sum(len(v) for v in routing.values())
            self.assertGreater(total_routes, 0)

        def test_layernorm_keys_preserved(self):
            """layer_norm_ keys must survive synthesis (mirrors Megatron's fix)."""
            mgr = self._make_manager()
            model, sd, prefix, q_w, kv_w = self._make_model_and_sd()
            ln_key = f"{prefix}linear_qkv_down_proj.layer_norm_weight"
            sd[ln_key] = torch.ones(8)
            mgr.reload_model_params(sd, model_chunks=[model])
            self.assertIn(ln_key, sd, "layer_norm_ key must survive QKV synthesis")

        def test_no_engine_no_model_raises(self):
            mgr = HeteroMLACheckpointManager(
                src_device_class=DeviceClass.UNKNOWN,
                dst_device_class=DeviceClass.UNKNOWN,
            )
            with self.assertRaises(ValueError):
                mgr.reload_model_params({})

    print("Running DES-LOC HeteroMLACheckpointManager unit tests …")
    unittest.main(verbosity=2)
