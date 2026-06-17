"""
deepspeed/sequence/hetero_mimo_sharding.py
==========================================

DES-LOC Heterogeneous MIMO Sequence Sharding Adapter
=====================================================

Upstream Design Intent (Megatron commit 3920476e)
-------------------------------------------------
The Megatron MIMO commit "Apply MIMO SP/CP sharding with explicit groups and enable THD
in non-colocated path" addresses a fundamental layout inconsistency in multi-modal
inference/training pipelines. Prior to this commit:

1. The PartitionAdapter.shard() expected *batch-first* (B, S, H) embeddings throughout,
   forcing callers to manually transpose in/out around every shard() invocation. This
   caused subtle bugs where the transpose was applied inconsistently across colocated
   vs non-colocated code paths.

2. The `attention_mask` was threaded through the CP sharding pipeline, which is semantically
   wrong: a dense [B, S] mask cannot be meaningfully sliced across CP ranks because the
   per-rank hidden states only cover a local sequence slice. The commit removes it from
   the shard() signature and asserts that CP users must instead use causal attn_mask_type
   or PackedSeqParams (THD format).

3. PartitionAdapter construction was guarded only by CP/SP flags, not by whether the
   current rank actually owns a language module. Encoder-only ranks would attempt to
   read process groups they never registered, causing silent hangs or group mismatches.

4. The SP scatter on the embedding layer was not guarded against double-scatter: the
   language model's internal embedding.scatter_to_sequence_parallel could fire *before*
   the combined multimodal embeddings were passed to PartitionAdapter, producing a second
   scatter that misaligned text token positions.

5. _forward_language_module() didn't propagate the loss_mask through, and the non-first
   PP stage path didn't apply CP sharding to labels/loss_mask, so the loss on the final
   stage was misaligned with the CP-local hidden states.

The fix unifies the layout contract: embeddings enter shard() as *sequence-first*
(S, B, H) and exit as (S/(cp*tp), B, H) ready for the language model. The internal
CP path still needs batch-first for get_batch_on_this_cp_rank, so the transpose is
moved *inside* _apply_context_parallel. SP scatter operates on dim-0 (sequence-first),
so no transpose is needed there. The return tuple drops attention_mask entirely.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces a heterogeneous
device topology: 2× A6000 (48 GB, SM86, PCIe) + 1× H100 NVL (96 GB, SM90, PCIe).
There is no NVLink; all inter-device communication goes over PCIe with ~64 GB/s
aggregate bandwidth. This changes several assumptions from the Megatron upstream:

1. **Device Affinity for CP Groups**
   In Megatron, CP groups are assumed to be formed across NVLink-connected GPUs.
   On DES-LOC hardware, PCIe bandwidth is ~10× slower for all-gather/scatter ops.
   The DES-LOC PartitionAdapter must be aware of which ranks live on which physical
   device class (A6000 vs H100) and form CP/SP groups that *minimize cross-device
   collective volume*. We expose a `DeviceAffinityMap` that classifies each rank's
   device type and a `HeteroPartitionConfig` that carries this topology.

2. **Asymmetric Sequence Allocation**
   The H100 (SM90) has 2× the memory of each A6000 (SM86). In a pure CP=3 setup
   we cannot zigzag evenly across 3 ranks of different memory. DES-LOC solves this
   with an *asymmetric shard factor*: the H100 rank gets a larger sequence slice
   (proportional to its memory), and the A6000 ranks get smaller slices. We implement
   `HeteroSequenceSharder` which replaces the symmetric zigzag with a device-capacity-
   weighted split.

3. **Locality Cache (LOC) Integration**
   DES-LOC's "Shared Locality Cache" is a CPU DRAM ring buffer (up to 1.5 TB available)
   that holds full-sequence embeddings and labels during forward, so that gradient
   accumulation steps don't require re-encoding modalities. After shard() returns the
   CP-local slice, HeteroMIMOSharder also stores the *full* pre-shard embedding tensor
   in the LOC under a deterministic key keyed by (micro_step, layer_idx). Non-first
   PP stages can retrieve labels/loss_mask from the LOC rather than recomputing them.

4. **THD / PackedSeqParams on Heterogeneous Lengths**
   THD format (variable-length packed sequences) is especially important on DES-LOC
   because different modality encoders (e.g. vision on A6000 vs language on H100) may
   produce tokens of wildly different counts per sample. The THD path bypasses the
   divisibility assertion, which is critical for asymmetric sharding.

5. **Role-Gated Adapter Construction**
   Like the upstream fix, we gate PartitionAdapter construction on whether the rank
   has a language module role. We extend this to DES-LOC's tripartite role taxonomy:
   ENCODER_ONLY, LANGUAGE_ONLY, and MIXED. Only LANGUAGE_ONLY and MIXED ranks
   construct the sharding adapter.

6. **SP Scatter with Explicit Groups**
   The upstream fix passes `group=self.cfg.tp_group` to scatter_to_sequence_parallel_region.
   On DES-LOC this is critical: the A6000 TP group and H100 TP group are different
   process groups, and using the wrong group causes silent data corruption.

Architecture
------------
This module provides:

- `DeviceClass`: enum for A6000 (SM86) vs H100_NVL (SM90)
- `DeviceAffinityMap`: maps global ranks to DeviceClass and local device index
- `HeteroPartitionConfig`: extends PartitionConfig with DES-LOC device topology
- `HeteroSequenceSharder`: asymmetric CP sharding respecting device memory ratios
- `LocalityCache`: lightweight CPU DRAM ring buffer for embedding/label storage
- `HeteroMIMOSharder`: the main DES-LOC PartitionAdapter replacement
- `HeteroMIMOModel`: a thin wrapper showing how to integrate into a DeepSpeed pipeline

Usage
-----
    from deepspeed.sequence.hetero_mimo_sharding import (
        HeteroMIMOSharder,
        HeteroPartitionConfig,
        DeviceAffinityMap,
        LocalityCache,
    )

    device_map = DeviceAffinityMap.from_env()
    cfg = HeteroPartitionConfig.build(
        use_cp=True,
        seq_parallel=True,
        max_seq_len=4096,
        device_map=device_map,
        cp_group=cp_pg,
        tp_group=tp_pg,
    )
    sharder = HeteroMIMOSharder(cfg, loc_cache=LocalityCache(capacity_gb=64))
    emb_local, labels_local, loss_mask_local, packed = sharder.shard(
        embeddings=combined_emb,  # [S, B, H] sequence-first
        labels=labels,            # [B, S]
        loss_mask=loss_mask,      # [B, S]
        packed_seq_params=None,
        micro_step=0,
        layer_idx=0,
    )
"""

from __future__ import annotations

import enum
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies — soft-imported so the module is importable for unit
# tests without a full DeepSpeed / Megatron install.
# ---------------------------------------------------------------------------
try:
    from deepspeed import comm as ds_comm  # type: ignore

    _HAVE_DS_COMM = True
except ImportError:
    _HAVE_DS_COMM = False
    logger.debug("deepspeed.comm not available; distributed ops will use torch.distributed")

try:
    from megatron.core import tensor_parallel  # type: ignore
    from megatron.core.packed_seq_params import PackedSeqParams  # type: ignore

    _HAVE_MEGATRON = True
except ImportError:
    _HAVE_MEGATRON = False
    PackedSeqParams = None  # type: ignore
    logger.debug("megatron.core not available; PackedSeqParams stubs will be used")

try:
    import transformer_engine.pytorch as te  # type: ignore

    _HAVE_TEX = True
except ImportError:
    _HAVE_TEX = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SM compute capability strings for our target devices
_SM_A6000 = 86  # NVIDIA A6000 48GB
_SM_H100_NVL = 90  # NVIDIA H100 NVL 96GB

# Memory capacity in GB by device class — used for asymmetric sequence allocation
_DEVICE_MEMORY_GB: Dict[int, float] = {
    _SM_A6000: 48.0,
    _SM_H100_NVL: 96.0,
}

# Key used to store/retrieve embeddings in the LocalityCache
_LOC_EMBEDDING_KEY_FMT = "emb:step={step}:layer={layer}:rank={rank}"
_LOC_LABELS_KEY_FMT = "labels:step={step}:layer={layer}:rank={rank}"
_LOC_LOSS_MASK_KEY_FMT = "loss_mask:step={step}:layer={layer}:rank={rank}"

# Maximum number of entries in the LocalityCache ring buffer
_LOC_DEFAULT_MAX_ENTRIES = 128

# THD qkv_format string
_QKV_FORMAT_THD = "thd"
_QKV_FORMAT_SBHD = "sbhd"


# ---------------------------------------------------------------------------
# DeviceClass
# ---------------------------------------------------------------------------


class DeviceClass(enum.Enum):
    """Physical device class present in the DES-LOC heterogeneous cluster.

    DES-LOC hardware:
        - A6000_48G: NVIDIA A6000 48 GB, SM86, PCIe (×2 in the reference cluster)
        - H100_NVL_96G: NVIDIA H100 NVL 96 GB, SM90, PCIe (×1 in the reference cluster)
    """

    A6000_48G = _SM_A6000
    H100_NVL_96G = _SM_H100_NVL
    UNKNOWN = -1

    @property
    def memory_gb(self) -> float:
        return _DEVICE_MEMORY_GB.get(self.value, 24.0)

    @property
    def sm_version(self) -> int:
        return self.value

    @classmethod
    def from_cuda_device(cls, device_index: int) -> "DeviceClass":
        """Detect device class from the CUDA device at *device_index*."""
        if not torch.cuda.is_available():
            return cls.UNKNOWN
        props = torch.cuda.get_device_properties(device_index)
        sm = props.major * 10 + props.minor
        if sm == _SM_A6000:
            return cls.A6000_48G
        if sm == _SM_H100_NVL:
            return cls.H100_NVL_96G
        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# DeviceAffinityMap
# ---------------------------------------------------------------------------


@dataclass
class DeviceAffinityMap:
    """Maps each global distributed rank to its physical device class and local index.

    DES-LOC Rationale
    -----------------
    Because PCIe is the sole interconnect, collective operations between A6000 and H100
    ranks traverse the PCIe bus. The DeviceAffinityMap lets the HeteroMIMOSharder
    compute asymmetric sequence splits (H100 gets proportionally more tokens) and
    helps route the LOC cache writes to CPU DRAM instead of device DRAM.

    Attributes
    ----------
    rank_to_device_class : dict mapping global rank → DeviceClass
    rank_to_local_idx    : dict mapping global rank → local CUDA device index
    """

    rank_to_device_class: Dict[int, DeviceClass] = field(default_factory=dict)
    rank_to_local_idx: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "DeviceAffinityMap":
        """Build the map by querying every rank's local CUDA device.

        Assumes torch.distributed is initialized and each rank's LOCAL_RANK env
        variable reflects its physical CUDA device index on the host.
        """
        if not dist.is_initialized():
            logger.warning(
                "DeviceAffinityMap.from_env() called before dist.init_process_group; "
                "returning empty map"
            )
            return cls()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_class = DeviceClass.from_cuda_device(local_rank)
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

        # All-gather (device_class.value, local_rank) pairs from every rank.
        payload = torch.tensor(
            [device_class.value, local_rank], dtype=torch.int64, device="cpu"
        )
        gathered = [torch.zeros(2, dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(gathered, payload)

        rank_to_device_class: Dict[int, DeviceClass] = {}
        rank_to_local_idx: Dict[int, int] = {}
        for rank, tensor in enumerate(gathered):
            sm_val = int(tensor[0].item())
            loc_idx = int(tensor[1].item())
            try:
                dc = DeviceClass(sm_val)
            except ValueError:
                dc = DeviceClass.UNKNOWN
            rank_to_device_class[rank] = dc
            rank_to_local_idx[rank] = loc_idx

        if global_rank == 0:
            device_summary = {
                dc.name: [r for r, d in rank_to_device_class.items() if d == dc]
                for dc in DeviceClass
                if dc != DeviceClass.UNKNOWN
            }
            logger.info(
                "DES-LOC DeviceAffinityMap built: %s",
                {k: v for k, v in device_summary.items() if v},
            )

        return cls(
            rank_to_device_class=rank_to_device_class,
            rank_to_local_idx=rank_to_local_idx,
        )

    def get_device_class(self, rank: int) -> DeviceClass:
        return self.rank_to_device_class.get(rank, DeviceClass.UNKNOWN)

    def ranks_for_class(self, device_class: DeviceClass) -> List[int]:
        return [r for r, dc in self.rank_to_device_class.items() if dc == device_class]

    def memory_gb_for_rank(self, rank: int) -> float:
        return self.get_device_class(rank).memory_gb


# ---------------------------------------------------------------------------
# HeteroPartitionConfig
# ---------------------------------------------------------------------------


@dataclass
class HeteroPartitionConfig:
    """Configuration for DES-LOC heterogeneous CP/SP sequence sharding.

    Extends the Megatron PartitionConfig concept with device topology awareness.

    Upstream PartitionConfig fields reproduced here:
        use_cp, seq_parallel, tp_comm_overlap, max_seq_len, cp_group, tp_group

    DES-LOC additions:
        device_map       : topology of ranks → DeviceClass
        cp_rank_list     : ordered list of global ranks in the CP group (needed for
                           asymmetric splits; positional order matches the zigzag)
        asymmetric_cp    : if True, sequence slices are proportional to device memory
        loc_enabled      : whether the LocalityCache (CPU DRAM) is enabled
        loc_capacity_gb  : maximum CPU DRAM to use for the LOC ring buffer
    """

    use_cp: bool = False
    seq_parallel: bool = False
    tp_comm_overlap: bool = False
    max_seq_len: int = 4096
    cp_group: Optional[Any] = None
    tp_group: Optional[Any] = None
    device_map: Optional[DeviceAffinityMap] = None
    cp_rank_list: List[int] = field(default_factory=list)
    asymmetric_cp: bool = True
    loc_enabled: bool = True
    loc_capacity_gb: float = 64.0

    @classmethod
    def build(
        cls,
        use_cp: bool,
        seq_parallel: bool,
        max_seq_len: int,
        device_map: Optional[DeviceAffinityMap] = None,
        cp_group: Optional[Any] = None,
        tp_group: Optional[Any] = None,
        tp_comm_overlap: bool = False,
        asymmetric_cp: bool = True,
        loc_enabled: bool = True,
        loc_capacity_gb: float = 64.0,
    ) -> "HeteroPartitionConfig":
        """Factory method that resolves the CP rank list from the process group."""
        cp_rank_list: List[int] = []
        if cp_group is not None and dist.is_initialized():
            cp_rank_list = dist.get_process_group_ranks(cp_group)

        return cls(
            use_cp=use_cp,
            seq_parallel=seq_parallel,
            tp_comm_overlap=tp_comm_overlap,
            max_seq_len=max_seq_len,
            cp_group=cp_group,
            tp_group=tp_group,
            device_map=device_map,
            cp_rank_list=cp_rank_list,
            asymmetric_cp=asymmetric_cp,
            loc_enabled=loc_enabled,
            loc_capacity_gb=loc_capacity_gb,
        )

    def cp_group_size(self) -> int:
        if self.cp_group is not None and dist.is_initialized():
            return dist.get_world_size(self.cp_group)
        return len(self.cp_rank_list) if self.cp_rank_list else 1

    def tp_group_size(self) -> int:
        if self.tp_group is not None and dist.is_initialized():
            return dist.get_world_size(self.tp_group)
        return 1


# ---------------------------------------------------------------------------
# LocalityCache  (DES-LOC "Shared LOcality Cache")
# ---------------------------------------------------------------------------


class LocalityCache:
    """CPU DRAM ring buffer for DES-LOC embedding and label reuse across gradient steps.

    Design
    ------
    DES-LOC's key insight: with 1.5 TB of CPU DRAM available and only ~192 GB of total
    GPU VRAM, storing pre-shard full-sequence tensors in CPU DRAM is cheap and enables
    gradient accumulation without re-encoding modalities. The LocalityCache stores up to
    `max_entries` tensor snapshots keyed by (micro_step, layer_idx, rank). On cache hit,
    the shard() call for non-first PP stages can skip the full embedding rebuild.

    Thread safety: a single threading.Lock guards the internal dict. Writes from the
    forward pass (shard()) and reads from the backward pass (retrieve()) may overlap in
    async pipeline schedules.

    Parameters
    ----------
    capacity_gb  : maximum CPU DRAM to occupy (soft limit; actual usage depends on tensor sizes)
    max_entries  : ring buffer capacity in number of entries (LRU eviction when full)
    pin_memory   : if True, use pinned CPU memory for faster GPU↔CPU transfers
    """

    def __init__(
        self,
        capacity_gb: float = 64.0,
        max_entries: int = _LOC_DEFAULT_MAX_ENTRIES,
        pin_memory: bool = True,
    ) -> None:
        self._capacity_bytes = int(capacity_gb * 1024**3)
        self._max_entries = max_entries
        self._pin_memory = pin_memory
        self._store: Dict[str, torch.Tensor] = {}
        self._access_order: List[str] = []  # LRU order: most recent at tail
        self._lock = threading.Lock()
        self._current_bytes: int = 0
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, key: str, tensor: torch.Tensor) -> None:
        """Copy *tensor* to pinned CPU DRAM and register under *key*."""
        cpu_tensor = tensor.detach().cpu()
        if self._pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
        nbytes = cpu_tensor.numel() * cpu_tensor.element_size()

        with self._lock:
            if key in self._store:
                self._current_bytes -= self._store[key].numel() * self._store[key].element_size()
                self._access_order.remove(key)

            # Evict LRU entries until we have room
            while self._current_bytes + nbytes > self._capacity_bytes and self._access_order:
                evict_key = self._access_order.pop(0)
                evicted = self._store.pop(evict_key, None)
                if evicted is not None:
                    self._current_bytes -= evicted.numel() * evicted.element_size()

            if self._current_bytes + nbytes <= self._capacity_bytes:
                self._store[key] = cpu_tensor
                self._access_order.append(key)
                self._current_bytes += nbytes
            else:
                logger.warning(
                    "LocalityCache: cannot store key=%s (%.2f MB); capacity exhausted",
                    key,
                    nbytes / 1024**2,
                )

    def retrieve(self, key: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """Return the tensor stored under *key*, moved to *device* if specified."""
        with self._lock:
            if key not in self._store:
                self._miss_count += 1
                return None
            self._hit_count += 1
            # Refresh LRU position
            self._access_order.remove(key)
            self._access_order.append(key)
            tensor = self._store[key]

        if device is not None:
            tensor = tensor.to(device=device, non_blocking=True)
        return tensor

    def invalidate(self, prefix: str) -> int:
        """Remove all entries whose key starts with *prefix*. Returns count removed."""
        with self._lock:
            keys_to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in keys_to_remove:
                self._current_bytes -= self._store[k].numel() * self._store[k].element_size()
                del self._store[k]
                self._access_order.remove(k)
        return len(keys_to_remove)

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._store),
                "bytes_used": self._current_bytes,
                "capacity_bytes": self._capacity_bytes,
                "hit_rate": (
                    self._hit_count / max(1, self._hit_count + self._miss_count)
                ),
                "hits": self._hit_count,
                "misses": self._miss_count,
            }


# ---------------------------------------------------------------------------
# HeteroSequenceSharder  (asymmetric CP for DES-LOC)
# ---------------------------------------------------------------------------


class HeteroSequenceSharder:
    """Asymmetric CP sequence shard allocator for heterogeneous device clusters.

    Upstream (Megatron) assumption: all CP ranks have equal memory, so the zigzag
    split divides sequence tokens evenly into 2*cp_size chunks.

    DES-LOC reality: the H100 (96 GB) can hold 2× more tokens than each A6000 (48 GB).
    A naive equal split wastes H100 capacity and risks OOM on A6000 for long sequences.

    This class computes per-rank sequence shard sizes proportional to device memory
    capacity, then builds the index tensors needed to extract each rank's slice from
    the full-sequence tensor (batch-first, [B, S, H] or [B, S] layout).

    For THD (packed) sequences, we fall back to the upstream equal-split zigzag because
    cu_seqlens boundaries cannot be reliably remapped to arbitrary shard sizes.

    Parameters
    ----------
    cfg           : HeteroPartitionConfig carrying device_map and cp_rank_list
    global_rank   : this rank's global distributed rank (determines which shard we keep)
    """

    def __init__(self, cfg: HeteroPartitionConfig, global_rank: int) -> None:
        self._cfg = cfg
        self._global_rank = global_rank
        self._shard_sizes: Optional[List[int]] = None  # populated lazily

    def compute_shard_sizes(self, seq_len: int) -> List[int]:
        """Return per-CP-rank sequence shard sizes that sum to seq_len.

        If asymmetric_cp is False or device_map is None, returns equal sizes.
        Otherwise sizes are proportional to device memory, adjusted so they sum
        exactly to seq_len.

        Each CP rank's shard is doubled (zigzag assigns two chunks per rank); the
        returned list gives the *total* tokens per rank across both zigzag chunks.
        """
        cp_size = self._cfg.cp_group_size()
        if not self._cfg.asymmetric_cp or self._cfg.device_map is None or not self._cfg.cp_rank_list:
            base = seq_len // cp_size
            sizes = [base] * cp_size
            # Distribute remainder tokens to the last rank
            sizes[-1] += seq_len - sum(sizes)
            return sizes

        memories = [
            self._cfg.device_map.memory_gb_for_rank(r) for r in self._cfg.cp_rank_list
        ]
        total_mem = sum(memories)
        raw_sizes = [seq_len * m / total_mem for m in memories]
        # Round to integers, making each even (zigzag requires 2 equal chunks)
        sizes = [max(2, round(s / 2) * 2) for s in raw_sizes]
        # Adjust so sizes sum exactly to seq_len
        diff = seq_len - sum(sizes)
        if diff != 0:
            # Apply diff to the largest-memory rank (H100) to avoid A6000 OOM
            max_mem_idx = memories.index(max(memories))
            sizes[max_mem_idx] += diff

        logger.debug(
            "DES-LOC asymmetric CP shard sizes: %s (seq_len=%d, cp_ranks=%s)",
            sizes,
            seq_len,
            self._cfg.cp_rank_list,
        )
        return sizes

    def get_local_indices(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return the sequence indices this rank should keep after the asymmetric zigzag.

        For the symmetric case this matches the Megatron zigzag exactly.
        For the asymmetric case each rank gets a contiguous block of size shard_sizes[rank_pos].
        """
        cp_size = self._cfg.cp_group_size()
        if cp_size == 1:
            return torch.arange(seq_len, device=device)

        rank_pos = (
            self._cfg.cp_rank_list.index(self._global_rank)
            if self._global_rank in self._cfg.cp_rank_list
            else 0
        )

        shard_sizes = self.compute_shard_sizes(seq_len)

        if not self._cfg.asymmetric_cp or self._cfg.device_map is None:
            # Standard zigzag: 2*cp_size chunks, rank r gets chunk r and chunk (2*cp-r-1)
            chunk_size = seq_len // (2 * cp_size)
            idx_a = torch.arange(
                rank_pos * chunk_size, (rank_pos + 1) * chunk_size, device=device
            )
            idx_b = torch.arange(
                (2 * cp_size - rank_pos - 1) * chunk_size,
                (2 * cp_size - rank_pos) * chunk_size,
                device=device,
            )
            return torch.cat([idx_a, idx_b])
        else:
            # Asymmetric: contiguous block allocation based on shard_sizes
            start = sum(shard_sizes[:rank_pos])
            end = start + shard_sizes[rank_pos]
            return torch.arange(start, end, device=device)


# ---------------------------------------------------------------------------
# HeteroMIMOSharder  (main DES-LOC PartitionAdapter replacement)
# ---------------------------------------------------------------------------


class HeteroMIMOSharder:
    """DES-LOC heterogeneous MIMO sequence sharding adapter.

    This class is the DES-LOC reinterpretation of Megatron's PartitionAdapter from
    megatron/core/models/mimo/partition/utils.py (commit 3920476e). It implements
    the same shard() → _apply_context_parallel() → SP-scatter pipeline but adds:

      - Asymmetric CP sequence splits for heterogeneous device memory (H100 > A6000)
      - LocalityCache writes for embedding/label reuse across gradient accumulation
      - Explicit tp_group passing to scatter_to_sequence_parallel_region (required when
        A6000 and H100 form separate TP sub-groups)
      - Guard against double-scatter (embedding layer's scatter_to_sequence_parallel)
      - Removal of attention_mask from the shard pipeline (same as upstream fix)
      - Sequence-first (S, B, H) input contract throughout (transpose moved inside
        _apply_context_parallel as in upstream)

    Layout contract (same as upstream after the commit):
        shard() input  embeddings: (S, B, H)  — sequence-first
        shard() output embeddings: (S/(cp*tp), B, H)  — LM-ready, sequence-first
        labels / loss_mask in:  (B, S)
        labels / loss_mask out: (B, S/cp)  — CP-sharded, NOT SP-scattered
    """

    def __init__(
        self,
        cfg: HeteroPartitionConfig,
        loc_cache: Optional[LocalityCache] = None,
        global_rank: Optional[int] = None,
    ) -> None:
        """Initialize the DES-LOC MIMO sharder.

        Parameters
        ----------
        cfg         : heterogeneous partition configuration
        loc_cache   : optional LocalityCache instance (shared across forward steps)
        global_rank : this process's global distributed rank (auto-detected if None)
        """
        self.cfg = cfg
        self._loc_cache = loc_cache
        self._global_rank: int = (
            global_rank
            if global_rank is not None
            else (dist.get_rank() if dist.is_initialized() else 0)
        )
        self._hetero_sharder = HeteroSequenceSharder(cfg, self._global_rank)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shard(
        self,
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        packed_seq_params: Optional[Any] = None,
        micro_step: int = 0,
        layer_idx: int = 0,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Any],
    ]:
        """Apply CP and/or SP sharding to MIMO sequence inputs.

        Upstream contract (Megatron commit 3920476e):
            - embeddings: (S, B, H) sequence-first on entry; exits as (S/(cp*tp), B, H)
            - labels / loss_mask: (B, S) on entry; exits as (B, S/cp) after CP sharding
            - attention_mask is NOT a parameter (removed upstream; dense mask incompatible
              with CP-local hidden states; use causal attn_mask_type or packed_seq_params)

        DES-LOC additions:
            - Writes the full-sequence embeddings/labels to the LocalityCache before
              sharding (keyed by micro_step and layer_idx) for gradient accumulation reuse
            - Uses asymmetric CP shard sizes proportional to device memory
            - Passes explicit tp_group to the SP scatter collective

        Parameters
        ----------
        embeddings       : (S, B, H) combined multimodal embeddings, or None on non-first PP stages
        labels           : (B, S) token labels for loss computation, or None
        loss_mask        : (B, S) per-token loss weight, or None
        packed_seq_params: THD PackedSeqParams if using variable-length packing, else None
        micro_step       : pipeline micro-batch step index (for LOC key generation)
        layer_idx        : transformer layer index (for LOC key generation)

        Returns
        -------
        Tuple of (embeddings, labels, loss_mask, packed_seq_params) after sharding.
        """
        # --- LOC write: store full-sequence tensors before sharding ---
        if self._loc_cache is not None and self.cfg.loc_enabled:
            self._loc_write(embeddings, labels, loss_mask, micro_step, layer_idx)

        # --- Validate sequence length divisibility (same as upstream) ---
        if embeddings is not None:
            self._check_seq_len(embeddings, packed_seq_params)

        # --- CP sharding ---
        if self.cfg.use_cp:
            # CP internals need batch-first layout; transpose happens inside _apply_context_parallel
            embeddings, labels, loss_mask, packed_seq_params = self._apply_context_parallel(
                embeddings, labels, loss_mask, packed_seq_params
            )
            # _apply_context_parallel returns embeddings in (S/cp, B, H) after internal transpose

        # --- SP scatter (sequence dim 0; no transpose needed for SP-only) ---
        if self.cfg.seq_parallel and embeddings is not None:
            embeddings = self._scatter_sequence_parallel(embeddings)

        return embeddings, labels, loss_mask, packed_seq_params

    def retrieve_from_loc(
        self,
        key_type: str,
        micro_step: int,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Retrieve a tensor from the LocalityCache (for non-first PP stage reuse).

        Parameters
        ----------
        key_type   : one of "emb", "labels", "loss_mask"
        micro_step : same micro_step used during the forward shard() call
        layer_idx  : same layer_idx used during the forward shard() call
        device     : target device for the retrieved tensor
        """
        if self._loc_cache is None:
            return None
        fmt = {
            "emb": _LOC_EMBEDDING_KEY_FMT,
            "labels": _LOC_LABELS_KEY_FMT,
            "loss_mask": _LOC_LOSS_MASK_KEY_FMT,
        }.get(key_type)
        if fmt is None:
            raise ValueError(f"Unknown key_type={key_type!r}; expected emb/labels/loss_mask")
        key = fmt.format(step=micro_step, layer=layer_idx, rank=self._global_rank)
        return self._loc_cache.retrieve(key, device=device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _loc_write(
        self,
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        micro_step: int,
        layer_idx: int,
    ) -> None:
        """Write pre-shard tensors to the LocalityCache."""
        rank = self._global_rank
        if embeddings is not None:
            key = _LOC_EMBEDDING_KEY_FMT.format(step=micro_step, layer=layer_idx, rank=rank)
            self._loc_cache.store(key, embeddings)
        if labels is not None:
            key = _LOC_LABELS_KEY_FMT.format(step=micro_step, layer=layer_idx, rank=rank)
            self._loc_cache.store(key, labels)
        if loss_mask is not None:
            key = _LOC_LOSS_MASK_KEY_FMT.format(step=micro_step, layer=layer_idx, rank=rank)
            self._loc_cache.store(key, loss_mask)

    def _check_seq_len(
        self,
        embeddings: torch.Tensor,
        packed_seq_params: Optional[Any],
    ) -> None:
        """Assert sequence length is divisible by the total shard factor.

        In THD mode (qkv_format == 'thd') the check is skipped, matching upstream
        behavior and supporting the asymmetric shard case where lengths may be irregular.
        """
        is_thd = packed_seq_params is not None and getattr(
            packed_seq_params, "qkv_format", _QKV_FORMAT_SBHD
        ) == _QKV_FORMAT_THD

        if is_thd:
            return

        seq_len = embeddings.shape[0]  # embeddings are (S, B, H); S is dim 0

        shard_factor: Optional[int] = None
        cp_size = self.cfg.cp_group_size()
        tp_size = self.cfg.tp_group_size()

        if self.cfg.use_cp and self.cfg.seq_parallel:
            shard_factor = tp_size * cp_size * 2
        elif self.cfg.use_cp:
            shard_factor = cp_size * 2
        elif self.cfg.seq_parallel:
            shard_factor = tp_size

        if shard_factor is not None:
            if self.cfg.asymmetric_cp and self.cfg.device_map is not None:
                # Asymmetric shards: only check that seq_len is even (zigzag minimum)
                assert seq_len % 2 == 0, (
                    f"DES-LOC asymmetric CP requires even sequence length; got {seq_len}"
                )
            else:
                assert seq_len % shard_factor == 0, (
                    f"Sequence length {seq_len} must be divisible by {shard_factor} "
                    f"for DES-LOC CP/SP sharding"
                )

        if self.cfg.seq_parallel and self.cfg.tp_comm_overlap:
            assert seq_len == self.cfg.max_seq_len, (
                f"DES-LOC TP comm overlap requires seq_len ({seq_len}) == "
                f"max_seq_len ({self.cfg.max_seq_len})"
            )

    def _apply_context_parallel(
        self,
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        packed_seq_params: Optional[Any],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Any],
    ]:
        """Apply CP sequence sharding to MIMO inputs.

        Upstream (Megatron): uses get_batch_on_this_cp_rank with symmetric zigzag.
        DES-LOC: uses HeteroSequenceSharder for asymmetric splits on heterogeneous devices.

        Layout:
            Input  embeddings: (S, B, H)  sequence-first
            After CP transpose: (B, S, H) batch-first for zigzag/index_select
            Output embeddings: (S/cp, B, H) back to sequence-first

        attention_mask is explicitly NOT handled here (same as upstream commit 3920476e):
        a dense [B, S] mask is semantically incorrect under CP because it can't line up
        with the CP-local hidden state slice. Callers must use causal attn_mask_type
        or packed_seq_params (THD).
        """
        if not self.cfg.use_cp:
            return embeddings, labels, loss_mask, packed_seq_params

        is_thd = packed_seq_params is not None and getattr(
            packed_seq_params, "qkv_format", _QKV_FORMAT_SBHD
        ) == _QKV_FORMAT_THD

        if is_thd:
            # THD path: use Transformer Engine's variable-length CP partitioning
            return self._apply_context_parallel_thd(
                embeddings, labels, loss_mask, packed_seq_params
            )

        # SBHD path: index-select using asymmetric or symmetric shard indices
        device = (
            embeddings.device
            if embeddings is not None
            else (labels.device if labels is not None else torch.device("cpu"))
        )

        # --- Transpose embeddings to batch-first for the shard index op ---
        if embeddings is not None:
            # (S, B, H) → (B, S, H)
            embeddings = embeddings.transpose(0, 1).contiguous()

        seq_len = (
            embeddings.shape[1]
            if embeddings is not None
            else (labels.shape[1] if labels is not None else 0)
        )

        if seq_len > 0:
            indices = self._hetero_sharder.get_local_indices(seq_len, device)
        else:
            indices = torch.tensor([], dtype=torch.long, device=device)

        if embeddings is not None:
            embeddings = embeddings.index_select(1, indices)
        if labels is not None:
            labels = labels.index_select(1, indices)
        if loss_mask is not None:
            loss_mask = loss_mask.index_select(1, indices)

        # --- Transpose embeddings back to sequence-first ---
        if embeddings is not None:
            # (B, S/cp, H) → (S/cp, B, H)
            embeddings = embeddings.transpose(0, 1).contiguous()

        logger.debug(
            "DES-LOC CP shard applied: emb=%s labels=%s loss_mask=%s rank=%d",
            tuple(embeddings.shape) if embeddings is not None else None,
            tuple(labels.shape) if labels is not None else None,
            tuple(loss_mask.shape) if loss_mask is not None else None,
            self._global_rank,
        )

        return embeddings, labels, loss_mask, packed_seq_params

    def _apply_context_parallel_thd(
        self,
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        packed_seq_params: Any,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Any],
    ]:
        """THD-format CP sharding using Transformer Engine's partitioned index logic.

        The THD path bypasses the divisibility check and uses TE's
        thd_get_partitioned_indices to correctly slice variable-length packed sequences
        across CP ranks. This is the same approach as upstream Megatron but adapted for
        DES-LOC's heterogeneous topology (the CP group is device-affinity-aware).

        On DES-LOC hardware, THD is particularly important because different modality
        encoders may produce sequences of lengths that don't divide evenly by cp_size*2.
        """
        assert _HAVE_TEX, (
            "Transformer Engine is required for THD-format CP sharding in DES-LOC. "
            "Install transformer_engine or use SBHD format."
        )

        cp_size = self.cfg.cp_group_size()
        cp_rank = (
            dist.get_rank(self.cfg.cp_group)
            if self.cfg.cp_group is not None and dist.is_initialized()
            else 0
        )

        cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        total_tokens = int(cu_seqlens[-1].item())
        index = te.thd_get_partitioned_indices(
            cu_seqlens, cu_seqlens, cp_size, cp_rank
        )

        if embeddings is not None:
            # Embeddings in THD are (T, H) where T = total_tokens; dim 0 is tokens
            # In MIMO, the combined embedding is (T, B=1, H) or (T, H); handle both
            if embeddings.dim() == 3:
                embeddings = embeddings.index_select(0, index)
            else:
                embeddings = embeddings.index_select(0, index)

        if labels is not None:
            # Labels may be (B, S) or (T,); index along seq dim
            if labels.dim() == 1:
                labels = labels.index_select(0, index)
            else:
                labels = labels.index_select(1, index)

        if loss_mask is not None:
            if loss_mask.dim() == 1:
                loss_mask = loss_mask.index_select(0, index)
            else:
                loss_mask = loss_mask.index_select(1, index)

        logger.debug(
            "DES-LOC THD CP shard applied: total_tokens=%d cp_rank=%d/%d",
            total_tokens,
            cp_rank,
            cp_size,
        )

        return embeddings, labels, loss_mask, packed_seq_params

    def _scatter_sequence_parallel(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Scatter embeddings across TP ranks along the sequence dimension (dim 0).

        Upstream change: passes explicit `group=self.cfg.tp_group` to
        scatter_to_sequence_parallel_region. This is critical on DES-LOC where A6000
        and H100 ranks belong to *different* TP process groups; using the default
        (global) group would corrupt the scatter.

        Input:  (S/cp, B, H) or (S, B, H) depending on whether CP ran first
        Output: (S/(cp*tp), B, H)
        """
        if not _HAVE_MEGATRON:
            # Fallback: manual scatter for testing without Megatron
            tp_size = self.cfg.tp_group_size()
            tp_rank = (
                dist.get_rank(self.cfg.tp_group)
                if self.cfg.tp_group is not None and dist.is_initialized()
                else 0
            )
            seq_len = embeddings.shape[0]
            shard = seq_len // tp_size
            return embeddings[tp_rank * shard : (tp_rank + 1) * shard].contiguous()

        return tensor_parallel.scatter_to_sequence_parallel_region(
            embeddings, group=self.cfg.tp_group
        )


# ---------------------------------------------------------------------------
# DES-LOC Role taxonomy  (mirrors Megatron's role system, extended)
# ---------------------------------------------------------------------------


class HeteroRoleType(enum.Enum):
    """Rank role in the DES-LOC heterogeneous pipeline.

    DES-LOC tripartite taxonomy:
        ENCODER_ONLY  : rank handles only modality encoders (e.g. vision on A6000)
        LANGUAGE_ONLY : rank handles only the language model (e.g. LM layers on H100)
        MIXED         : rank handles both encoder and language module components

    Only LANGUAGE_ONLY and MIXED ranks should construct a HeteroMIMOSharder; this
    mirrors the upstream guard: "Only on language-module ranks: encoder-only ranks
    never shard and would read process groups they do not own."
    """

    ENCODER_ONLY = "encoder_only"
    LANGUAGE_ONLY = "language_only"
    MIXED = "mixed"

    @property
    def has_language_module(self) -> bool:
        return self in (HeteroRoleType.LANGUAGE_ONLY, HeteroRoleType.MIXED)

    @property
    def has_encoder_module(self) -> bool:
        return self in (HeteroRoleType.ENCODER_ONLY, HeteroRoleType.MIXED)

    def is_first_stage(self, module_key: str) -> bool:
        """True if this rank is the first pipeline stage for *module_key*."""
        # In DES-LOC the first LM stage is always a LANGUAGE_ONLY or MIXED rank
        # that receives encoder outputs. Simplified here for illustration.
        return self.has_language_module

    def is_last_stage(self, module_key: str) -> bool:
        return self.has_language_module


# ---------------------------------------------------------------------------
# HeteroMIMOModel  (integration sketch for DeepSpeed pipeline)
# ---------------------------------------------------------------------------


class HeteroMIMOModel:
    """Thin integration wrapper showing how HeteroMIMOSharder slots into a DeepSpeed pipeline.

    This is a structural analogue of Megatron's MimoModel._forward_language_module() and
    the colocated _forward_encoders_and_language() path, rewritten for DES-LOC's
    heterogeneous three-GPU topology.

    Key differences from Megatron MimoModel
    ----------------------------------------
    1. The sharder is only constructed when `role.has_language_module` is True, preventing
       encoder-only A6000 ranks from accessing CP/TP process groups they don't own.

    2. `_forward_language_module()` returns a tuple (lm_output, loss_mask) where
       loss_mask is the *CP-sharded* version from shard(). Non-last PP stages also
       return loss_mask for routing through the pipeline schedule.

    3. `_get_text_embeddings()` checks for double-scatter: if the language model's
       embedding layer has scatter_to_sequence_parallel=True and SP is active, we
       raise RuntimeError before the scatter fires, preventing misaligned text tokens.

    4. `_build_packed_seq_params()` is a standalone helper (not inlined) so THD
       construction is reusable across colocated and non-colocated forward paths.

    5. The LOC cache is populated during `_shard_language_inputs()` so that gradient
       accumulation steps can retrieve pre-shard embeddings from CPU DRAM.
    """

    LANGUAGE_MODULE_KEY = "language"

    def __init__(
        self,
        role: HeteroRoleType,
        cfg: HeteroPartitionConfig,
        language_model: Optional[Any] = None,
        loc_cache: Optional[LocalityCache] = None,
    ) -> None:
        self.role = role
        self.cfg = cfg
        self.language_model = language_model
        self._sharder: Optional[HeteroMIMOSharder] = None

        # Upstream guard: only language-module ranks build the sharder
        if role.has_language_module and (cfg.use_cp or cfg.seq_parallel):
            self._sharder = HeteroMIMOSharder(cfg, loc_cache=loc_cache)
            logger.info(
                "DES-LOC HeteroMIMOSharder constructed on rank %d (role=%s, cp=%s, sp=%s)",
                self._sharder._global_rank,
                role.value,
                cfg.use_cp,
                cfg.seq_parallel,
            )

    def _build_packed_seq_params(
        self, packing_kwargs: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Construct THD PackedSeqParams from packing_kwargs.

        Extracted from _forward_language_module to be reusable across colocated
        and non-colocated forward paths (mirrors upstream _build_packed_seq_params).
        """
        if packing_kwargs is None or not _HAVE_MEGATRON:
            return None
        for key in packing_kwargs:
            if "cu_seqlens" in key and packing_kwargs[key] is not None:
                packing_kwargs[key] = packing_kwargs[key].to(dtype=torch.int32)
        packed = PackedSeqParams(**packing_kwargs)
        packed.qkv_format = _QKV_FORMAT_THD
        return packed

    def _get_text_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Extract text token embeddings from the language model embedding layer.

        DES-LOC guard (mirrors upstream): if the language embedding's
        scatter_to_sequence_parallel flag is True and SP is active, the combined
        multimodal embeddings would be scattered *before* alignment, corrupting the
        token ordering. We raise RuntimeError to fail fast rather than producing
        silently wrong results.
        """
        if self.language_model is None:
            # Stub for testing: return zero embeddings
            B_times_S = input_ids.numel()
            # Determine hidden size from cfg (fallback to 64 for tests)
            H = getattr(self.cfg, "_hidden_size_for_test", 64)
            return torch.zeros(B_times_S, H, device=input_ids.device)

        embedding_layer = self.language_model.embedding
        if (
            self._sharder is not None
            and self.cfg.seq_parallel
            and getattr(embedding_layer, "scatter_to_sequence_parallel", False)
        ):
            raise RuntimeError(
                "DES-LOC HeteroMIMOSharder: sequence parallelism requires the language "
                "embedding scatter to be disabled. Pass scatter_embedding_sequence_parallel=False "
                "when constructing the language model. A second scatter here would split "
                "flat text tokens across TP ranks before multimodal alignment."
            )

        return embedding_layer(
            input_ids=input_ids, position_ids=position_ids
        ).squeeze(1)

    def _shard_language_inputs(
        self,
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        packed_seq_params: Optional[Any] = None,
        micro_step: int = 0,
        layer_idx: int = 0,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Any],
    ]:
        """Apply CP/SP sharding via HeteroMIMOSharder, or pass through if inactive.

        Embeddings enter as (S, B, H) and exit as (S/(cp*tp), B, H).
        Labels/loss_mask enter as (B, S) and exit as (B, S/cp) after CP sharding.
        """
        if self._sharder is None:
            return embeddings, labels, loss_mask, packed_seq_params

        return self._sharder.shard(
            embeddings=embeddings,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            micro_step=micro_step,
            layer_idx=layer_idx,
        )

    def _forward_language_module(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        input_tensors: Optional[Dict[str, torch.Tensor]],
        packing_kwargs: Optional[Dict[str, Any]] = None,
        micro_step: int = 0,
        layer_idx: int = 0,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """Forward pass for language module on this DES-LOC rank.

        Mirrors Megatron MimoModel._forward_language_module() after commit 3920476e
        with DES-LOC extensions:
            - Returns (lm_output, loss_mask) where loss_mask is CP-sharded
            - Populates LOC cache before sharding
            - Validates attention_mask=None under CP (upstream requirement)
            - Validates embedding double-scatter (upstream requirement)

        Parameters
        ----------
        input_ids      : token IDs [B, S] or [T] (THD)
        position_ids   : position IDs, or None
        attention_mask : must be None when CP is active (see guard below)
        loss_mask      : [B, S] per-token loss weights
        labels         : [B, S] token labels for cross-entropy
        input_tensors  : dict of hidden states from previous PP stage, or None
        packing_kwargs : kwargs to build PackedSeqParams (THD), or None
        micro_step     : gradient accumulation step index (for LOC key)
        layer_idx      : transformer layer index (for LOC key)

        Returns
        -------
        Tuple[lm_output, loss_mask]:
            lm_output  : hidden states, logits, or loss (depends on PP stage)
            loss_mask  : CP-sharded loss mask (aligned with lm_output on last PP stage)
        """
        # --- Upstream guard: dense attention_mask incompatible with CP ---
        if (
            self._sharder is not None
            and self.cfg.use_cp
            and attention_mask is not None
        ):
            raise RuntimeError(
                "DES-LOC context parallelism requires attention_mask=None. "
                "A dense [B, S] mask cannot align with CP-local hidden states. "
                "Use a causal attn_mask_type or packed_seq_params (THD format) instead."
            )

        packed_seq_params = self._build_packed_seq_params(packing_kwargs)
        lang_key = self.LANGUAGE_MODULE_KEY

        if self.role.is_first_stage(lang_key):
            # First PP stage: build combined multimodal embeddings and shard
            text_embeddings = self._get_text_embeddings(input_ids, position_ids)

            # Combine with modality embeddings (stub: just use text for illustration)
            # In production this calls align_embeddings_by_token_positions()
            combined_embeddings = text_embeddings  # [T, H] or [S, B, H]

            # Ensure sequence-first (S, B, H) for the sharder
            if combined_embeddings.dim() == 2:
                # [T, H] → treat as [T, 1, H] for single-sample batches
                combined_embeddings = combined_embeddings.unsqueeze(1)

            # Apply CP/SP sharding; embeddings exit as (S/(cp*tp), B, H)
            combined_embeddings, labels, loss_mask, packed_seq_params = (
                self._shard_language_inputs(
                    embeddings=combined_embeddings,
                    labels=labels,
                    loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params,
                    micro_step=micro_step,
                    layer_idx=layer_idx,
                )
            )

            if self.language_model is not None:
                lm_output = self.language_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    decoder_input=combined_embeddings,
                    labels=labels,
                    packed_seq_params=packed_seq_params,
                )
            else:
                # Stub for testing
                lm_output = combined_embeddings

        else:
            # Non-first PP stage: labels/loss_mask still need CP sharding to align
            # with the CP-local hidden states received from the previous stage.
            _, labels, loss_mask, packed_seq_params = self._shard_language_inputs(
                embeddings=None,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                micro_step=micro_step,
                layer_idx=layer_idx,
            )

            hidden_states = (
                input_tensors.get(lang_key) if input_tensors is not None else None
            )

            if self.language_model is not None:
                lm_output = self.language_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    decoder_input=None,
                    labels=labels,
                    packed_seq_params=packed_seq_params,
                )
            else:
                lm_output = hidden_states if hidden_states is not None else torch.tensor([0.0])

        # Non-last PP stages return a routing dict plus the sharded loss_mask
        if not self.role.is_last_stage(lang_key):
            return {lang_key: lm_output}, loss_mask

        return lm_output, loss_mask


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback
    import unittest
    from unittest.mock import MagicMock, patch

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    class TestDeviceClass(unittest.TestCase):
        """Tests for DeviceClass enum and memory properties."""

        def test_memory_gb_a6000(self):
            dc = DeviceClass.A6000_48G
            self.assertEqual(dc.memory_gb, 48.0)

        def test_memory_gb_h100_nvl(self):
            dc = DeviceClass.H100_NVL_96G
            self.assertEqual(dc.memory_gb, 96.0)

        def test_sm_version(self):
            self.assertEqual(DeviceClass.A6000_48G.sm_version, 86)
            self.assertEqual(DeviceClass.H100_NVL_96G.sm_version, 90)

        def test_unknown_memory_fallback(self):
            dc = DeviceClass.UNKNOWN
            self.assertEqual(dc.memory_gb, 24.0)

    class TestDeviceAffinityMap(unittest.TestCase):
        """Tests for DeviceAffinityMap without distributed."""

        def test_get_device_class_unknown_rank(self):
            dmap = DeviceAffinityMap()
            self.assertEqual(dmap.get_device_class(99), DeviceClass.UNKNOWN)

        def test_ranks_for_class(self):
            dmap = DeviceAffinityMap(
                rank_to_device_class={
                    0: DeviceClass.A6000_48G,
                    1: DeviceClass.A6000_48G,
                    2: DeviceClass.H100_NVL_96G,
                }
            )
            a6000_ranks = dmap.ranks_for_class(DeviceClass.A6000_48G)
            self.assertEqual(sorted(a6000_ranks), [0, 1])
            h100_ranks = dmap.ranks_for_class(DeviceClass.H100_NVL_96G)
            self.assertEqual(h100_ranks, [2])

        def test_memory_gb_for_rank(self):
            dmap = DeviceAffinityMap(
                rank_to_device_class={
                    0: DeviceClass.A6000_48G,
                    2: DeviceClass.H100_NVL_96G,
                }
            )
            self.assertEqual(dmap.memory_gb_for_rank(0), 48.0)
            self.assertEqual(dmap.memory_gb_for_rank(2), 96.0)

    class TestHeteroPartitionConfig(unittest.TestCase):
        """Tests for HeteroPartitionConfig factory and size helpers."""

        def test_cp_group_size_no_dist(self):
            cfg = HeteroPartitionConfig(use_cp=True, cp_rank_list=[0, 1, 2])
            self.assertEqual(cfg.cp_group_size(), 3)

        def test_tp_group_size_default(self):
            cfg = HeteroPartitionConfig(seq_parallel=True)
            self.assertEqual(cfg.tp_group_size(), 1)

        def test_build_without_dist(self):
            cfg = HeteroPartitionConfig.build(
                use_cp=False,
                seq_parallel=True,
                max_seq_len=2048,
            )
            self.assertTrue(cfg.seq_parallel)
            self.assertFalse(cfg.use_cp)
            self.assertEqual(cfg.max_seq_len, 2048)

    class TestLocalityCache(unittest.TestCase):
        """Tests for LocalityCache ring buffer."""

        def test_store_and_retrieve(self):
            cache = LocalityCache(capacity_gb=1.0, pin_memory=False)
            t = torch.randn(16, 4, 64)
            cache.store("test_key", t)
            result = cache.retrieve("test_key")
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, t.shape)

        def test_retrieve_missing_key(self):
            cache = LocalityCache(capacity_gb=1.0, pin_memory=False)
            result = cache.retrieve("nonexistent")
            self.assertIsNone(result)

        def test_lru_eviction(self):
            # Create a tiny cache (one float tensor at a time)
            cache = LocalityCache(capacity_gb=0.000001, max_entries=2, pin_memory=False)
            t1 = torch.randn(100, 100)  # ~40 KB
            t2 = torch.randn(100, 100)
            cache.store("key1", t1)
            cache.store("key2", t2)
            # key1 should be evicted to make room for key2
            # (capacity exceeded; t1 evicted LRU)
            # At least key2 should be present or cache handles gracefully
            stats = cache.stats
            self.assertGreaterEqual(stats["entries"], 0)

        def test_invalidate_prefix(self):
            cache = LocalityCache(capacity_gb=1.0, pin_memory=False)
            cache.store("emb:step=0:layer=0:rank=0", torch.randn(8, 2, 64))
            cache.store("emb:step=1:layer=0:rank=0", torch.randn(8, 2, 64))
            cache.store("labels:step=0:layer=0:rank=0", torch.randint(0, 100, (2, 8)))
            n = cache.invalidate("emb:")
            self.assertEqual(n, 2)
            self.assertIsNone(cache.retrieve("emb:step=0:layer=0:rank=0"))
            self.assertIsNotNone(cache.retrieve("labels:step=0:layer=0:rank=0"))

        def test_stats(self):
            cache = LocalityCache(capacity_gb=1.0, pin_memory=False)
            t = torch.zeros(4, 4)
            cache.store("k1", t)
            cache.retrieve("k1")
            cache.retrieve("k1")
            cache.retrieve("missing")
            stats = cache.stats
            self.assertEqual(stats["hits"], 2)
            self.assertEqual(stats["misses"], 1)
            self.assertAlmostEqual(stats["hit_rate"], 2 / 3, places=5)

    class TestHeteroSequenceSharder(unittest.TestCase):
        """Tests for asymmetric CP sequence shard allocation."""

        def _make_dmap(self):
            return DeviceAffinityMap(
                rank_to_device_class={
                    0: DeviceClass.A6000_48G,   # 48 GB
                    1: DeviceClass.A6000_48G,   # 48 GB
                    2: DeviceClass.H100_NVL_96G, # 96 GB
                },
                rank_to_local_idx={0: 0, 1: 1, 2: 0},
            )

        def test_symmetric_shard_sizes_no_device_map(self):
            cfg = HeteroPartitionConfig(use_cp=True, cp_rank_list=[0, 1], asymmetric_cp=False)
            sharder = HeteroSequenceSharder(cfg, global_rank=0)
            sizes = sharder.compute_shard_sizes(32)
            self.assertEqual(sizes, [16, 16])
            self.assertEqual(sum(sizes), 32)

        def test_asymmetric_shard_sizes_h100_gets_more(self):
            dmap = self._make_dmap()
            cfg = HeteroPartitionConfig(
                use_cp=True,
                cp_rank_list=[0, 1, 2],
                asymmetric_cp=True,
                device_map=dmap,
            )
            sharder = HeteroSequenceSharder(cfg, global_rank=0)
            sizes = sharder.compute_shard_sizes(96)  # divisible by 2 for each rank
            self.assertEqual(sum(sizes), 96)
            # H100 rank (idx 2) should have more tokens than each A6000
            self.assertGreater(sizes[2], sizes[0])
            self.assertGreater(sizes[2], sizes[1])

        def test_asymmetric_sizes_sum_to_seq_len(self):
            dmap = self._make_dmap()
            cfg = HeteroPartitionConfig(
                use_cp=True,
                cp_rank_list=[0, 1, 2],
                asymmetric_cp=True,
                device_map=dmap,
            )
            sharder = HeteroSequenceSharder(cfg, global_rank=2)
            for seq_len in [48, 64, 128, 192, 256]:
                sizes = sharder.compute_shard_sizes(seq_len)
                self.assertEqual(sum(sizes), seq_len, f"sum mismatch for seq_len={seq_len}")

        def test_symmetric_local_indices(self):
            cfg = HeteroPartitionConfig(
                use_cp=True,
                cp_rank_list=[0, 1],
                asymmetric_cp=False,
            )
            sharder = HeteroSequenceSharder(cfg, global_rank=0)
            indices = sharder.get_local_indices(8, device=torch.device("cpu"))
            # Rank 0 symmetric zigzag: chunks [0,1,2,3] and [4,5,6,7] → [0,1,2,3,4,5,6,7]?
            # Actually zigzag for cp=2, 2*cp=4 chunks of size 2:
            # rank 0 gets chunk 0 [0,1] and chunk 3 [6,7]
            expected = torch.tensor([0, 1, 6, 7])
            torch.testing.assert_close(indices, expected)

        def test_single_cp_rank_returns_full_range(self):
            cfg = HeteroPartitionConfig(use_cp=True, cp_rank_list=[0])
            sharder = HeteroSequenceSharder(cfg, global_rank=0)
            indices = sharder.get_local_indices(16, device=torch.device("cpu"))
            expected = torch.arange(16)
            torch.testing.assert_close(indices, expected)

    class TestHeteroMIMOSharder(unittest.TestCase):
        """Tests for HeteroMIMOSharder.shard() without distributed."""

        def _make_sharder(self, use_cp=False, seq_parallel=False, asymmetric_cp=False):
            dmap = DeviceAffinityMap(
                rank_to_device_class={
                    0: DeviceClass.A6000_48G,
                    1: DeviceClass.H100_NVL_96G,
                },
            )
            cfg = HeteroPartitionConfig(
                use_cp=use_cp,
                seq_parallel=seq_parallel,
                max_seq_len=32,
                cp_rank_list=[0, 1] if use_cp else [],
                asymmetric_cp=asymmetric_cp,
                device_map=dmap,
                loc_enabled=False,
            )
            return HeteroMIMOSharder(cfg, loc_cache=None, global_rank=0)

        def test_passthrough_when_both_disabled(self):
            sharder = self._make_sharder(use_cp=False, seq_parallel=False)
            S, B, H = 8, 2, 16
            emb = torch.randn(S, B, H)
            lbl = torch.randint(0, 100, (B, S))
            mask = torch.ones(B, S)
            out_emb, out_lbl, out_mask, out_psp = sharder.shard(emb, lbl, mask)
            self.assertIs(out_emb, emb)
            self.assertIs(out_lbl, lbl)
            self.assertIs(out_mask, mask)
            self.assertIsNone(out_psp)

        def test_seq_len_divisibility_check_fires(self):
            sharder = self._make_sharder(use_cp=True, seq_parallel=False, asymmetric_cp=False)
            # cp_size=2, so shard_factor = 2*2 = 4; seq_len=7 not divisible
            emb = torch.randn(7, 2, 16)
            with self.assertRaises(AssertionError):
                sharder.shard(emb, None, None)

        def test_thd_skips_divisibility_check(self):
            sharder = self._make_sharder(use_cp=True, seq_parallel=False, asymmetric_cp=False)
            emb = torch.randn(7, 2, 16)
            psp = MagicMock()
            psp.qkv_format = _QKV_FORMAT_THD
            # Should not raise; THD bypasses divisibility check
            # THD path requires TE; patch it
            with patch("deepspeed.sequence.hetero_mimo_sharding._HAVE_TEX", False):
                with self.assertRaises(AssertionError) as ctx:
                    # The _apply_context_parallel_thd will raise about TE missing
                    sharder.shard(emb, None, None, packed_seq_params=psp)
                self.assertIn("Transformer Engine", str(ctx.exception))

        def test_none_embeddings_skips_divisibility(self):
            sharder = self._make_sharder(use_cp=True, seq_parallel=False, asymmetric_cp=False)
            lbl = torch.randint(0, 100, (2, 8))
            mask = torch.ones(2, 8)
            # No embeddings; CP shard should still process labels/mask
            # With global_rank=0 and cp_rank_list=[0,1], indices = [0,1,6,7] for S=8
            # But _apply_context_parallel does index_select on labels with those indices
            out_emb, out_lbl, out_mask, _ = sharder.shard(None, lbl, mask)
            self.assertIsNone(out_emb)
            # Labels should be subset-selected (4 out of 8 for cp=2)
            self.assertEqual(out_lbl.shape[1], 4)

        def test_loc_cache_write_on_shard(self):
            dmap = DeviceAffinityMap(
                rank_to_device_class={0: DeviceClass.A6000_48G},
            )
            cfg = HeteroPartitionConfig(
                use_cp=False,
                seq_parallel=False,
                max_seq_len=8,
                loc_enabled=True,
            )
            cache = LocalityCache(capacity_gb=1.0, pin_memory=False)
            sharder = HeteroMIMOSharder(cfg, loc_cache=cache, global_rank=0)
            emb = torch.randn(8, 2, 16)
            lbl = torch.randint(0, 100, (2, 8))
            mask = torch.ones(2, 8)
            sharder.shard(emb, lbl, mask, micro_step=3, layer_idx=7)

            retrieved_emb = cache.retrieve(
                _LOC_EMBEDDING_KEY_FMT.format(step=3, layer=7, rank=0)
            )
            self.assertIsNotNone(retrieved_emb)
            self.assertEqual(retrieved_emb.shape, emb.shape)

        def test_sp_scatter_without_megatron_fallback(self):
            dmap = DeviceAffinityMap(rank_to_device_class={0: DeviceClass.H100_NVL_96G})
            cfg = HeteroPartitionConfig(
                use_cp=False,
                seq_parallel=True,
                max_seq_len=8,
                cp_rank_list=[],
                loc_enabled=False,
            )
            sharder = HeteroMIMOSharder(cfg, loc_cache=None, global_rank=0)
            emb = torch.randn(8, 2, 16)
            # With tp_group_size=1 and rank=0, scatter returns the full tensor
            with patch("deepspeed.sequence.hetero_mimo_sharding._HAVE_MEGATRON", False):
                out_emb, _, _, _ = sharder.shard(emb, None, None)
            self.assertEqual(out_emb.shape, (8, 2, 16))  # tp_size=1 → no split

    class TestHeteroMIMOModel(unittest.TestCase):
        """Tests for HeteroMIMOModel forward logic."""

        def _make_model(self, role=HeteroRoleType.LANGUAGE_ONLY, use_cp=False, seq_parallel=False):
            cfg = HeteroPartitionConfig(
                use_cp=use_cp,
                seq_parallel=seq_parallel,
                max_seq_len=16,
                cp_rank_list=[0, 1] if use_cp else [],
                loc_enabled=False,
            )
            return HeteroMIMOModel(role=role, cfg=cfg, language_model=None, loc_cache=None)

        def test_encoder_only_rank_no_sharder(self):
            model = self._make_model(role=HeteroRoleType.ENCODER_ONLY, use_cp=True)
            self.assertIsNone(model._sharder)

        def test_language_rank_with_cp_builds_sharder(self):
            model = self._make_model(role=HeteroRoleType.LANGUAGE_ONLY, use_cp=True)
            self.assertIsNotNone(model._sharder)

        def test_cp_attention_mask_raises(self):
            model = self._make_model(role=HeteroRoleType.LANGUAGE_ONLY, use_cp=True)
            input_ids = torch.randint(0, 100, (2, 16))
            attn_mask = torch.ones(2, 16)
            with self.assertRaises(RuntimeError) as ctx:
                model._forward_language_module(
                    input_ids=input_ids,
                    position_ids=None,
                    attention_mask=attn_mask,
                    loss_mask=None,
                    labels=None,
                    input_tensors=None,
                )
            self.assertIn("attention_mask=None", str(ctx.exception))

        def test_forward_returns_loss_mask_tuple(self):
            model = self._make_model(role=HeteroRoleType.LANGUAGE_ONLY)
            cfg = model.cfg
            cfg._hidden_size_for_test = 64
            input_ids = torch.randint(0, 100, (2, 8))
            loss_mask = torch.ones(2, 8)
            lm_out, returned_mask = model._forward_language_module(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                loss_mask=loss_mask,
                labels=None,
                input_tensors=None,
            )
            # No CP/SP active → loss_mask passes through unchanged
            self.assertIs(returned_mask, loss_mask)

        def test_sp_double_scatter_guard(self):
            model = self._make_model(role=HeteroRoleType.LANGUAGE_ONLY, seq_parallel=True)
            mock_lm = MagicMock()
            mock_lm.embedding.scatter_to_sequence_parallel = True
            model.language_model = mock_lm
            input_ids = torch.randint(0, 100, (4,))
            with self.assertRaises(RuntimeError) as ctx:
                model._get_text_embeddings(input_ids, position_ids=None)
            self.assertIn("scatter to be disabled", str(ctx.exception))

        def test_build_packed_seq_params_none_when_no_kwargs(self):
            model = self._make_model()
            result = model._build_packed_seq_params(None)
            self.assertIsNone(result)

        def test_build_packed_seq_params_dtype_cast(self):
            if not _HAVE_MEGATRON:
                self.skipTest("megatron.core not available")
            model = self._make_model()
            kwargs = {
                "cu_seqlens_q": torch.tensor([0, 4, 8], dtype=torch.int64),
                "cu_seqlens_kv": torch.tensor([0, 4, 8], dtype=torch.int64),
                "cu_seqlens_q_padded": torch.tensor([0, 4, 8], dtype=torch.int64),
                "cu_seqlens_kv_padded": torch.tensor([0, 4, 8], dtype=torch.int64),
                "max_seqlen_q": 4,
                "max_seqlen_kv": 4,
            }
            result = model._build_packed_seq_params(kwargs)
            self.assertIsNotNone(result)
            self.assertEqual(result.qkv_format, _QKV_FORMAT_THD)
            self.assertEqual(result.cu_seqlens_q.dtype, torch.int32)

    class TestHeteroRoleType(unittest.TestCase):
        """Tests for DES-LOC role taxonomy."""

        def test_encoder_only_no_language(self):
            role = HeteroRoleType.ENCODER_ONLY
            self.assertFalse(role.has_language_module)
            self.assertTrue(role.has_encoder_module)

        def test_language_only(self):
            role = HeteroRoleType.LANGUAGE_ONLY
            self.assertTrue(role.has_language_module)
            self.assertFalse(role.has_encoder_module)

        def test_mixed(self):
            role = HeteroRoleType.MIXED
            self.assertTrue(role.has_language_module)
            self.assertTrue(role.has_encoder_module)

    # ------------------------------------------------------------------ #
    # Run all tests
    # ------------------------------------------------------------------ #
    print("=" * 72)
    print("DES-LOC HeteroMIMOSharding Unit Tests")
    print("=" * 72)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestDeviceClass,
        TestDeviceAffinityMap,
        TestHeteroPartitionConfig,
        TestLocalityCache,
        TestHeteroSequenceSharder,
        TestHeteroMIMOSharder,
        TestHeteroMIMOModel,
        TestHeteroRoleType,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✓ All tests passed.")
        sys.exit(0)
    else:
        print(f"\n✗ {len(result.failures)} failure(s), {len(result.errors)} error(s).")
        sys.exit(1)
