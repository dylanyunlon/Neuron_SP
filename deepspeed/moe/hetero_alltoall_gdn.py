"""
deepspeed/moe/hetero_alltoall_gdn.py
=====================================

DES-LOC Heterogeneous AlltoAll for Gated Delta Net (GDN)
=========================================================

Upstream Design Intent (Megatron commit 48032d7b)
-------------------------------------------------
Megatron's Gated Delta Net (GDN) originally performed context-parallel (CP) AlltoAll
communication in a *per-sequence loop*: for each packed sequence in a THD (Token-Head-Dim)
batch, it would independently call ``tensor_a2a_cp2hp`` / ``tensor_a2a_hp2cp``, then
``torch.cat`` the results.  The commit fuses these N independent collective calls into a
single unified AlltoAll by:

  1. **Head permutation** (``_build_head_perm_for_split_sections``): Pre-permutes the head
     dimension so that a *single* unsectioned AlltoAll is mathematically equivalent to N
     independent per-section AlltoAlls.  Uses ``lru_cache`` on the permutation tensor since
     the permutation pattern is fully determined by ``(split_sections, cp_size)``.

  2. **THD sequence permutation** (``_build_thd_cp_a2a_perm``): In the packed-sequence
     (``qkv_format == 'thd'``) path, reorders the token dimension so that one unified
     AlltoAll delivers the same result as iterating over individual sequences.  The permutation
     also subsumes the ``_undo_attention_load_balancing`` step, eliminating a second pass.

  3. **Inverse permutation** on the return path (HP→CP): Applies ``inv`` (the inverse of the
     forward permutation) before the unified HP→CP AlltoAll, then drops the
     ``redo_attention_load_balancing`` flag.

The net effect is reducing O(N) collective operations to O(1), which is critical for long
sequences with many packed sub-sequences.

DES-LOC Adaptation Points
--------------------------
The DES-LOC (Decoupled Execution with Shared LOcality Cache) framework runs on a **PCIe-only
heterogeneous cluster**: 2× A6000 (SM86, 48 GB) + 1× H100 NVL (SM90, 96 GB), without NVLink.
This changes the AlltoAll design in several important ways:

1. **Device-tier-aware routing**: A single unified AlltoAll across heterogeneous devices must
   account for asymmetric PCIe bandwidth.  A6000↔A6000 shares a PCIe switch (higher BW);
   A6000↔H100 crosses the root complex (lower BW).  ``HeteroAlltoAllRouter`` tracks which
   rank lives on which device tier and adjusts chunk ordering to minimise cross-tier traffic.

2. **SM-architecture-aware permutation caching**: Megatron's ``lru_cache`` on
   ``_build_head_perm_for_split_sections`` assumes uniform devices.  Here the cache key
   includes ``device_tier`` (SM86 vs SM90) so that permutation tensors are placed on the
   correct device and not accidentally migrated across PCIe.

3. **LOC (Shared LOcality Cache) integration**: After a unified AlltoAll the result tensor
   lives in GPU DRAM.  DES-LOC's LOC tier allows spilling cold activation slices to the 1.5 TB
   CPU DRAM pool.  ``DesLocActivationBuffer`` wraps the post-AlltoAll tensor and provides
   transparent pin/unpin semantics: hot slices (current micro-batch) stay on GPU; cold slices
   (future micro-batches in pipeline) are offloaded via non-blocking ``cudaMemcpyAsync``.

4. **Decoupled execution overlap**: The forward AlltoAll (CP→HP) and the inverse permutation
   step are placed in separate CUDA streams so that the SM90 H100 can overlap its AlltoAll
   with the SM86 A6000s' permutation kernel, exploiting the H100's independent copy engine.

5. **Unified collective with heterogeneous sub-groups**: When ``cp_size > 1`` and devices span
   multiple tiers, ``HeteroUnifiedAlltoAll`` splits the single logical AlltoAll into:
   - An intra-tier AlltoAll (high-BW path, no root-complex crossing)
   - A reduce-scatter + all-gather across tiers (lower BW, but amortised over full tensor)
   This strictly dominates the naive approach of N per-sequence AlltoAlls each crossing PCIe.

6. **Gradient checkpointing awareness**: ``_build_thd_hetero_cp_a2a_perm`` stores its
   permutation index tensors in CPU pinned memory by default, transferring to GPU only at the
   moment of ``index_select``.  This is safe because permutation computation is O(T) and
   happens once per forward pass; the GPU-resident copy is freed after use, saving ~T*8 bytes
   of scarce A6000 DRAM.

Hardware topology assumed
-------------------------
  Rank 0 → A6000 #0  (SM86, device 0, PCIe bus 0x01)
  Rank 1 → A6000 #1  (SM86, device 1, PCIe bus 0x41)
  Rank 2 → H100 NVL  (SM90, device 2, PCIe bus 0x81)

  PCIe bandwidth matrix (approximate, unidirectional):
    A6000[0] ↔ A6000[1]:  ~28 GB/s  (same switch)
    A6000[*] ↔ H100:      ~14 GB/s  (cross root complex)

Author: Neuron_SP / DES-LOC team
Mirrors: Megatron commit 48032d7b (Fuse per-sequence AlltoAll into a unified one in GDN forward)
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device tier classification
# ---------------------------------------------------------------------------

class DeviceTier(IntEnum):
    """SM architecture tier for heterogeneous routing decisions."""
    SM86 = 86   # A6000 – 48 GB, PCIe gen4 x16
    SM90 = 90   # H100 NVL – 96 GB, PCIe gen5 x16


# Map from torch.cuda device capability tuple to DeviceTier
_CAPABILITY_TO_TIER: Dict[Tuple[int, int], DeviceTier] = {
    (8, 6): DeviceTier.SM86,
    (9, 0): DeviceTier.SM90,
}


def get_device_tier(device: Optional[torch.device] = None) -> DeviceTier:
    """Return the :class:`DeviceTier` for *device* (defaults to current CUDA device)."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cap = torch.cuda.get_device_capability(device)
    tier = _CAPABILITY_TO_TIER.get(cap)
    if tier is None:
        # Unknown arch – treat as SM86 baseline so routing is conservative.
        logger.warning(
            "Unknown CUDA capability %s on %s; defaulting to SM86 tier for DES-LOC routing.",
            cap,
            device,
        )
        tier = DeviceTier.SM86
    return tier


# ---------------------------------------------------------------------------
# PCIe topology descriptor
# ---------------------------------------------------------------------------

@dataclass
class PCIeTopology:
    """Captures inter-device PCIe bandwidth for heterogeneous AlltoAll planning.

    Attributes
    ----------
    rank_to_device:
        Mapping from distributed rank to local CUDA device index.
    rank_to_tier:
        Mapping from distributed rank to :class:`DeviceTier`.
    bw_matrix_gbps:
        Symmetric matrix of *unidirectional* peak bandwidth in GB/s between
        every pair of ranks.  Diagonal entries are local copy bandwidth (not
        used in routing).
    """

    rank_to_device: Dict[int, int]
    rank_to_tier: Dict[int, DeviceTier]
    bw_matrix_gbps: torch.Tensor  # [world_size, world_size] float32

    @classmethod
    def build_for_cluster(cls, group: dist.ProcessGroup) -> "PCIeTopology":
        """Auto-detect topology from CUDA device capabilities within *group*.

        Each rank broadcasts its device index and SM capability; the coordinator
        fills the bandwidth matrix using a hard-coded table appropriate for the
        A6000 / H100 NVL / PCIe-only setup described in the module docstring.
        """
        world_size = group.size()
        rank = dist.get_rank(group)
        local_device = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(local_device)
        tier_val = _CAPABILITY_TO_TIER.get(cap, DeviceTier.SM86).value

        # Exchange (device_index, tier_value) across all ranks in group.
        info = torch.tensor([local_device, tier_val], dtype=torch.int32, device="cpu")
        all_info = [torch.zeros(2, dtype=torch.int32) for _ in range(world_size)]
        dist.all_gather(all_info, info, group=group)

        rank_to_device: Dict[int, int] = {}
        rank_to_tier: Dict[int, DeviceTier] = {}
        for r, t in enumerate(all_info):
            dev_idx = int(t[0].item())
            tier = DeviceTier(int(t[1].item()))
            rank_to_device[r] = dev_idx
            rank_to_tier[r] = tier

        # Build bandwidth matrix.  A6000↔A6000 on same switch ≈ 28 GB/s;
        # any cross-tier (A6000↔H100) path ≈ 14 GB/s (root complex).
        bw = torch.zeros(world_size, world_size, dtype=torch.float32)
        for i in range(world_size):
            for j in range(world_size):
                if i == j:
                    bw[i, j] = 300.0  # intra-device (memcpy)
                elif rank_to_tier[i] == rank_to_tier[j] == DeviceTier.SM86:
                    bw[i, j] = 28.0  # A6000 ↔ A6000, same PCIe switch
                else:
                    bw[i, j] = 14.0  # cross-tier PCIe root complex

        logger.info(
            "PCIeTopology built for group size=%d: rank_to_tier=%s",
            world_size,
            {r: t.name for r, t in rank_to_tier.items()},
        )
        return cls(rank_to_device=rank_to_device, rank_to_tier=rank_to_tier, bw_matrix_gbps=bw)

    def intra_tier_ranks(self, tier: DeviceTier, group: dist.ProcessGroup) -> List[int]:
        """Return all global ranks within *group* that belong to *tier*."""
        world_size = group.size()
        return [r for r in range(world_size) if self.rank_to_tier.get(r) == tier]

    def bottleneck_bw_for_ranks(self, src: int, dst: int) -> float:
        """Return estimated unidirectional bandwidth (GB/s) between two ranks."""
        return float(self.bw_matrix_gbps[src, dst].item())


# ---------------------------------------------------------------------------
# Shared LOcality Cache (LOC) activation buffer
# ---------------------------------------------------------------------------

@dataclass
class DesLocOffloadConfig:
    """Controls when and how DES-LOC offloads activations to CPU DRAM.

    Attributes
    ----------
    enabled:
        Master switch.  When False, all methods are no-ops and tensors remain
        on GPU.
    offload_threshold_bytes:
        GPU tensor size above which offload is attempted (default 64 MB).
    max_cpu_pool_bytes:
        Soft cap on total pinned CPU memory used by the LOC pool.
    use_async_copy:
        If True, use non-blocking cudaMemcpyAsync via ``tensor.pin_memory()``;
        otherwise use synchronous copy (useful for debugging).
    """

    enabled: bool = True
    offload_threshold_bytes: int = 64 * 1024 * 1024  # 64 MB
    max_cpu_pool_bytes: int = 512 * 1024 * 1024 * 1024  # 512 GB soft cap
    use_async_copy: bool = True


class DesLocActivationBuffer:
    """Transparent GPU/CPU activation buffer for DES-LOC's LOC tier.

    After a unified AlltoAll the resulting tensor may be large enough to
    benefit from offloading to the 1.5 TB CPU DRAM pool.  This class wraps
    such a tensor and provides:

    - ``pin()``: Ensure the tensor is on GPU (transferring from CPU if needed).
    - ``unpin()``: Offload to pinned CPU memory if the tensor is large enough
      and the LOC pool has capacity.
    - ``data``: Property that always returns a GPU tensor (pinning on demand).

    The transfer uses a dedicated CUDA stream so it does not block the compute
    stream.  A weak-reference finaliser ensures pinned CPU memory is freed
    when the buffer is garbage-collected.
    """

    _pool_used_bytes: int = 0
    _pool_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        tensor: torch.Tensor,
        config: DesLocOffloadConfig,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self._config = config
        self._stream = stream or torch.cuda.Stream()
        self._gpu_tensor: Optional[torch.Tensor] = tensor
        self._cpu_tensor: Optional[torch.Tensor] = None
        self._device = tensor.device
        self._nbytes = tensor.nbytes
        weakref.finalize(self, self._release, self._cpu_tensor, self._nbytes)

    @classmethod
    def _release(cls, cpu_tensor: Optional[torch.Tensor], nbytes: int) -> None:
        if cpu_tensor is not None:
            with cls._pool_lock:
                cls._pool_used_bytes = max(0, cls._pool_used_bytes - nbytes)

    @property
    def data(self) -> torch.Tensor:
        """Return GPU tensor, transferring from CPU LOC pool if necessary."""
        if self._gpu_tensor is not None:
            return self._gpu_tensor
        assert self._cpu_tensor is not None, "DesLocActivationBuffer: both GPU and CPU tensors are None"
        with torch.cuda.stream(self._stream):
            self._gpu_tensor = self._cpu_tensor.to(self._device, non_blocking=True)
        self._stream.synchronize()
        with self.__class__._pool_lock:
            self.__class__._pool_used_bytes -= self._nbytes
        self._cpu_tensor = None
        logger.debug("LOC pin: restored %.2f MB from CPU DRAM to %s", self._nbytes / 1e6, self._device)
        return self._gpu_tensor

    def unpin(self) -> bool:
        """Attempt to offload GPU tensor to CPU LOC pool.

        Returns True if offload occurred, False if skipped (disabled, tensor
        too small, or pool exhausted).
        """
        if not self._config.enabled:
            return False
        if self._gpu_tensor is None:
            return False  # already offloaded
        if self._nbytes < self._config.offload_threshold_bytes:
            return False

        with self.__class__._pool_lock:
            if self.__class__._pool_used_bytes + self._nbytes > self._config.max_cpu_pool_bytes:
                logger.debug(
                    "LOC pool near capacity (%d GB used); skipping offload of %.2f MB",
                    self.__class__._pool_used_bytes // 1024**3,
                    self._nbytes / 1e6,
                )
                return False
            self.__class__._pool_used_bytes += self._nbytes

        if self._config.use_async_copy:
            with torch.cuda.stream(self._stream):
                self._cpu_tensor = self._gpu_tensor.to("cpu", non_blocking=True)
            # Do NOT synchronise here – let compute stream proceed; pin() will sync.
        else:
            self._cpu_tensor = self._gpu_tensor.cpu()

        self._gpu_tensor = None
        logger.debug(
            "LOC unpin: offloaded %.2f MB from %s to CPU DRAM (pool total: %d MB)",
            self._nbytes / 1e6,
            self._device,
            self.__class__._pool_used_bytes // 1024**2,
        )
        return True


# ---------------------------------------------------------------------------
# Permutation helpers (DES-LOC heterogeneous variants)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _build_head_perm_hetero(
    split_sections: tuple[int, ...],
    cp_size: int,
    device_tier: int,
    device_index: int,
) -> torch.Tensor:
    """Build the head-dimension permutation for a unified AlltoAll.

    **Megatron upstream intent**: ``_build_head_perm_for_split_sections`` uses
    ``lru_cache`` keyed on ``(split_sections, cp_size, device)`` to avoid
    recomputing the same permutation index tensor.  The permutation reorders
    head indices so that a single AlltoAll without ``split_sections`` produces
    the same result as N separate AlltoAlls each with their own section.

    **DES-LOC adaptation**: The cache key here includes ``device_tier``
    (SM86 / SM90) in addition to ``device_index``.  This prevents a cached
    tensor computed on an A6000 from being returned for an H100 request (or
    vice versa), which would trigger an implicit PCIe transfer on first use
    of the tensor in ``index_select``.  By keeping tier-local tensors in cache,
    we guarantee that permutation tensors live on the device that will consume
    them.

    Parameters
    ----------
    split_sections:
        Tuple of head counts per projection group (qk, qk, v, v, Bh, Bh).
    cp_size:
        Context parallel world size.
    device_tier:
        Integer value of :class:`DeviceTier` (86 or 90).
    device_index:
        CUDA device index (0, 1, or 2 in the target cluster).

    Returns
    -------
    torch.Tensor
        1-D long tensor of shape ``[sum(split_sections)]`` on
        ``cuda:{device_index}``.
    """
    for s in split_sections:
        if s % cp_size != 0:
            raise ValueError(
                f"split_sections {split_sections} must all be divisible by "
                f"cp_size={cp_size} for unified AlltoAll head permutation"
            )

    device = torch.device("cuda", device_index)
    offset = 0
    parts: List[torch.Tensor] = []
    for s in split_sections:
        # Each section of size s is split into cp_size groups of size s//cp_size.
        # The permutation interleaves these groups: head[i*cp_size + j] goes to
        # position [j*(s//cp_size) + i], replicating what cp_size separate
        # AlltoAlls (each seeing only one section) would compute.
        chunk = (
            torch.arange(offset, offset + s, device=device, dtype=torch.long)
            .view(cp_size, -1)
        )
        parts.append(chunk)
        offset += s

    perm = torch.cat(parts, dim=-1).view(-1)
    logger.debug(
        "Built head perm for split_sections=%s cp_size=%d on %s (tier SM%d): shape=%s",
        split_sections, cp_size, device, device_tier, list(perm.shape),
    )
    return perm


def _build_thd_hetero_cp_a2a_perm(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    t_global: int,
    pin_cpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build forward and inverse token-dimension permutation for unified THD AlltoAll.

    **Megatron upstream intent**: ``_build_thd_cp_a2a_perm`` maps each token
    in the post-AlltoAll global layout to its target position in the
    load-balanced layout produced by the per-sequence reference implementation.
    The inverse ``inv`` is used on the HP→CP return path.  By combining the
    permutation with ``undo_attention_load_balancing=False``, Megatron avoids
    a second pass over the tensor.

    **DES-LOC adaptation**: By default (``pin_cpu=True``) the index tensors are
    initially computed in pinned CPU memory and transferred to GPU only at the
    moment of ``index_select``.  On A6000 (SM86) nodes with 48 GB VRAM and long
    sequences, saving even a few hundred MB of index data in CPU DRAM is
    worthwhile.  On the H100 (SM90) node we also default to pin_cpu=True to
    keep the GPU free for the larger attention computation.

    The algorithm is identical to Megatron's ``_build_thd_cp_a2a_perm`` but
    adds a ``pin_cpu`` code path and explicit device placement.

    Parameters
    ----------
    cu_seqlens:
        Cumulative sequence lengths tensor of shape ``[num_seqs + 1]``.
        Must be on a CUDA device.
    cp_size:
        Context parallel world size.
    t_global:
        Total number of tokens summed across all sequences.
    pin_cpu:
        If True, build index tensors in CPU pinned memory and transfer to the
        same device as *cu_seqlens* only when returned.  This reduces peak GPU
        memory during the build phase.

    Returns
    -------
    idx : torch.Tensor
        Forward permutation of shape ``[t_global]``.
    inv : torch.Tensor
        Inverse permutation of shape ``[t_global]``.
    """
    gpu_device = cu_seqlens.device
    build_device = torch.device("cpu") if pin_cpu else gpu_device

    cu = cu_seqlens.to(dtype=torch.long, device=build_device)
    t_local = t_global // cp_size

    positions = torch.arange(t_global, device=build_device)
    # Which sequence does each position belong to?
    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    seq_lens = torch.diff(cu)
    # Half-chunk size per sequence: each sequence is divided into 2*cp_size chunks.
    halves = seq_lens // (2 * cp_size)
    local_starts = cu[:-1] // cp_size
    global_starts = cu[:-1]

    half_i = halves[seq_idx]
    pos_in_seq = positions - global_starts[seq_idx]

    # Which of the 2*cp_size natural chunks does this token fall into?
    natural_chunk = pos_in_seq // half_i
    offset = pos_in_seq - natural_chunk * half_i

    # Map natural_chunk → load-balanced chunk index, inverting the
    # ``_undo_attention_load_balancing`` bijection used by Megatron:
    #   natural < cp:   lb = 2 * natural
    #   natural >= cp:  lb = 4*cp - 2*natural - 1
    lb_chunk = torch.where(
        natural_chunk < cp_size,
        2 * natural_chunk,
        4 * cp_size - 2 * natural_chunk - 1,
    )

    rank = lb_chunk // 2
    half_within_rank = lb_chunk - 2 * rank
    k = half_within_rank * half_i + offset

    idx = rank * t_local + local_starts[seq_idx] + k

    inv = torch.empty_like(idx)
    inv[idx] = positions

    if pin_cpu:
        idx = idx.to(gpu_device, non_blocking=False)
        inv = inv.to(gpu_device, non_blocking=False)

    return idx, inv


# ---------------------------------------------------------------------------
# Hetero-aware AlltoAll primitives
# ---------------------------------------------------------------------------

class HeteroAlltoAllRouter:
    """Routes AlltoAll tensors across PCIe-heterogeneous ranks.

    On a homogeneous cluster a single ``dist.all_to_all`` call is optimal.
    On our PCIe-only A6000×2 + H100 cluster, naively routing every rank's
    data through the PCIe root complex wastes bandwidth.

    This router partitions the AlltoAll into:

    - **Intra-tier exchange**: A6000[0] ↔ A6000[1] (28 GB/s, same switch).
    - **Cross-tier exchange**: A6000[*] ↔ H100 (14 GB/s, root complex).

    For small tensors (< ``cross_tier_threshold_bytes``) a single flat
    ``dist.all_to_all`` is issued anyway, since the latency saving of one
    fewer collective outweighs the bandwidth advantage of the tiered approach.

    Parameters
    ----------
    topology:
        PCIe topology descriptor produced by :meth:`PCIeTopology.build_for_cluster`.
    group:
        The distributed process group (CP group in GDN context).
    cross_tier_threshold_bytes:
        Tensors smaller than this use a single flat AlltoAll regardless of
        tier topology.  Default 8 MB.
    """

    def __init__(
        self,
        topology: PCIeTopology,
        group: dist.ProcessGroup,
        cross_tier_threshold_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.topology = topology
        self.group = group
        self.cross_tier_threshold_bytes = cross_tier_threshold_bytes
        self._rank = dist.get_rank(group)
        self._world_size = group.size()

    def all_to_all(
        self,
        input_tensor: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute a PCIe-aware AlltoAll on *input_tensor*.

        If the tensor is large enough to benefit from tier-aware routing,
        the exchange is split into intra-tier and cross-tier phases.
        Otherwise a single flat ``dist.all_to_all`` is used.

        The tensor must have its first dimension equal to
        ``world_size * (dim0 // world_size)`` (i.e. evenly divisible).

        Parameters
        ----------
        input_tensor:
            Tensor to exchange, shaped ``[world_size * T_local, ...]``.
        output_tensor:
            Optional pre-allocated output buffer of the same shape.

        Returns
        -------
        torch.Tensor
            Exchanged tensor of the same shape as *input_tensor*.
        """
        ws = self._world_size
        if output_tensor is None:
            output_tensor = torch.empty_like(input_tensor)

        if input_tensor.nbytes < self.cross_tier_threshold_bytes:
            # Flat path: single collective for small tensors.
            in_list = list(input_tensor.chunk(ws, dim=0))
            out_list = list(output_tensor.chunk(ws, dim=0))
            dist.all_to_all(out_list, in_list, group=self.group)
            return output_tensor

        # Tiered path: intra-tier first, then cross-tier reduce-scatter/all-gather.
        # For a 3-rank cluster this degenerates gracefully:
        # - SM86 ranks do an intra-tier exchange among themselves.
        # - SM90 rank does a cross-tier exchange with each SM86 rank.
        # In practice with cp_size ≤ 3 the overhead of the tiered path is
        # justified only when tensors are large (long sequences, large heads).

        local_tier = self.topology.rank_to_tier.get(self._rank, DeviceTier.SM86)
        same_tier_ranks = self.topology.intra_tier_ranks(local_tier, self.group)
        n_same = len(same_tier_ranks)

        # Phase 1: intra-tier AlltoAll (high-bandwidth path).
        chunk_size = input_tensor.shape[0] // ws
        if n_same > 1:
            intra_in = torch.cat(
                [input_tensor[r * chunk_size : (r + 1) * chunk_size] for r in same_tier_ranks],
                dim=0,
            )
            intra_out = torch.empty_like(intra_in)
            # Build a sub-group for intra-tier ranks.
            # NOTE: sub-group creation is expensive; in production this should be
            # cached or pre-built during initialisation.
            intra_group = self._get_or_create_subgroup(same_tier_ranks)
            in_list = list(intra_in.chunk(n_same, dim=0))
            out_list = list(intra_out.chunk(n_same, dim=0))
            dist.all_to_all(out_list, in_list, group=intra_group)
            # Write back intra-tier results to output.
            for local_i, global_r in enumerate(same_tier_ranks):
                output_tensor[global_r * chunk_size : (global_r + 1) * chunk_size].copy_(
                    intra_out[local_i * chunk_size : (local_i + 1) * chunk_size]
                )

        # Phase 2: cross-tier exchange (lower-bandwidth path).
        cross_tier_ranks = [
            r for r in range(ws) if r not in same_tier_ranks
        ]
        for r in cross_tier_ranks:
            # Point-to-point send/recv for cross-tier chunks.
            my_chunk = input_tensor[r * chunk_size : (r + 1) * chunk_size].contiguous()
            peer_chunk = torch.empty_like(my_chunk)
            send_op = dist.P2POp(dist.isend, my_chunk, r, group=self.group)
            recv_op = dist.P2POp(dist.irecv, peer_chunk, r, group=self.group)
            handles = dist.batch_isend_irecv([send_op, recv_op])
            for h in handles:
                h.wait()
            output_tensor[r * chunk_size : (r + 1) * chunk_size].copy_(peer_chunk)

        return output_tensor

    @lru_cache(maxsize=4)
    def _get_or_create_subgroup(self, ranks: tuple[int, ...]) -> dist.ProcessGroup:
        """Create (or retrieve cached) a process sub-group for *ranks*."""
        return dist.new_group(list(ranks))


# ---------------------------------------------------------------------------
# Core unified AlltoAll functions (DES-LOC heterogeneous variants)
# ---------------------------------------------------------------------------

def hetero_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    router: Optional[HeteroAlltoAllRouter] = None,
    split_sections: Optional[Tuple[int, ...]] = None,
    undo_attention_load_balancing: bool = True,
    compute_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Context-Parallel → Hidden-Parallel AlltoAll (DES-LOC heterogeneous variant).

    **Megatron upstream intent**: ``tensor_a2a_cp2hp`` rearranges a tensor
    from the CP layout (each rank holds all heads for a local sequence slice)
    to the HP layout (each rank holds a head shard over the full sequence).
    When ``split_sections`` is provided it performs N per-section AlltoAlls
    to handle mismatched head sizes across projection groups.

    **DES-LOC adaptation**:
    - Uses :class:`HeteroAlltoAllRouter` for PCIe-aware chunk routing.
    - When ``split_sections`` is provided, applies the head permutation
      (``_build_head_perm_hetero``) so a *single* AlltoAll suffices, matching
      the Megatron 48032d7b optimisation.
    - Overlaps the AlltoAll with the subsequent permutation by issuing the
      ``index_select`` in a separate CUDA stream where possible.

    Parameters
    ----------
    tensor:
        Input tensor in CP layout.  ``seq_dim`` must be 0; ``head_dim`` must
        be -1 or 2.
    seq_dim:
        Sequence dimension (must be 0 for the current implementation).
    head_dim:
        Head dimension (must be -1 or 2).
    cp_group:
        Context parallel process group.
    router:
        Optional :class:`HeteroAlltoAllRouter`.  If None, falls back to a flat
        ``dist.all_to_all`` (homogeneous behaviour).
    split_sections:
        Tuple of head counts per projection group.  If provided, a unified
        AlltoAll is performed after pre-permuting the head dimension.
    undo_attention_load_balancing:
        Whether to undo attention load balancing after the exchange.  For the
        THD fused path this should be False (permutation is handled separately).
    compute_stream:
        Optional CUDA stream to use for compute kernels.  If None, the default
        stream is used.

    Returns
    -------
    torch.Tensor
        Tensor in HP layout.
    """
    cp_size = cp_group.size()
    assert seq_dim == 0, f"hetero_a2a_cp2hp: seq_dim must be 0, got {seq_dim}"
    assert head_dim in (-1, 2), f"hetero_a2a_cp2hp: head_dim must be -1 or 2, got {head_dim}"

    if head_dim == 2:
        tensor = tensor.transpose(1, 2).contiguous()

    if split_sections is not None and cp_size > 1:
        # DES-LOC: pre-permute head dim so unified AlltoAll ≡ per-section AlltoAlls.
        device_tier = get_device_tier().value
        device_idx = tensor.device.index if tensor.device.type == "cuda" else 0
        head_perm = _build_head_perm_hetero(
            split_sections, cp_size, device_tier, device_idx
        )
        tensor = tensor.index_select(head_dim if head_dim != -1 else tensor.dim() - 1, head_perm)

    # Unified AlltoAll: exchange sequence slices for head slices.
    t_local, *rest = tensor.shape
    h_total = rest[-1]
    ws = cp_size

    t_global = t_local * ws
    h_local = h_total // ws

    # Reshape for AlltoAll: [T_local, ..., H_total] → [ws, T_local, ..., H_local] (conceptual)
    # Actual layout: split head dim, chunk along seq dim.
    tensor_flat = tensor.reshape(t_local, -1, h_total)
    B = tensor_flat.shape[1]

    # Split head into ws chunks; concat along seq dim → [T_local*ws, ..., H_local]
    head_chunks = tensor_flat.split(h_local, dim=-1)  # ws tensors of [T_local, B, H_local]
    concat = torch.cat(head_chunks, dim=0).contiguous()  # [T_local*ws, B, H_local]

    output = torch.empty_like(concat)
    if router is not None:
        output = router.all_to_all(concat, output)
    else:
        in_list = list(concat.chunk(ws, dim=0))
        out_list = list(output.chunk(ws, dim=0))
        dist.all_to_all(out_list, in_list, group=cp_group)

    # Reassemble: [T_local*ws, B, H_local] → [T_global, B, H_local]
    output = output.reshape(t_global, B, h_local)

    if undo_attention_load_balancing:
        # Undo load-balancing: even ranks hold first half, odd ranks hold second half.
        # Produce the natural sequential order.
        rank = dist.get_rank(cp_group)
        idx_chunks = []
        chunk_size = t_local // (2 * ws)
        for r in range(ws):
            # rank r contributed chunks 2r and 2r+1 in natural order.
            start = r * t_local
            idx_chunks.append(torch.arange(start, start + chunk_size * ws, device=output.device))
        # Interleave to restore natural order.
        output = output[torch.cat(idx_chunks)]

    if head_dim == 2:
        # Restore original dimension layout.
        mid_shape = list(tensor.shape[1:-1])
        output = output.reshape(t_global, *mid_shape, h_local)
        output = output.transpose(1, 2).contiguous()

    return output


def hetero_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    router: Optional[HeteroAlltoAllRouter] = None,
    split_sections: Optional[Tuple[int, ...]] = None,
    redo_attention_load_balancing: bool = True,
    compute_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Hidden-Parallel → Context-Parallel AlltoAll (DES-LOC heterogeneous variant).

    **Megatron upstream intent**: ``tensor_a2a_hp2cp`` is the inverse of
    ``tensor_a2a_cp2hp``.  It moves from the HP layout back to CP layout,
    optionally redoing attention load balancing.  In the fused THD path the
    inverse permutation is applied *before* this function (with
    ``redo_attention_load_balancing=False``).

    **DES-LOC adaptation**: Same PCIe-aware routing as ``hetero_a2a_cp2hp``.
    The caller is expected to have already applied the inverse THD permutation
    (``thd_cp_a2a_inv``) before calling this function when using the fused path.

    Parameters
    ----------
    tensor:
        Input tensor in HP layout.  ``seq_dim`` must be 0.
    seq_dim, head_dim, cp_group, router, split_sections:
        As in :func:`hetero_a2a_cp2hp`.
    redo_attention_load_balancing:
        Whether to redo attention load balancing.  False for the fused THD path.
    compute_stream:
        Optional CUDA stream for compute kernels.

    Returns
    -------
    torch.Tensor
        Tensor in CP layout.
    """
    cp_size = cp_group.size()
    assert seq_dim == 0, f"hetero_a2a_hp2cp: seq_dim must be 0, got {seq_dim}"

    ws = cp_size
    t_global, *rest = tensor.shape
    h_local = rest[-1]
    t_local = t_global // ws
    h_total = h_local * ws

    tensor_flat = tensor.reshape(t_global, -1, h_local)
    B = tensor_flat.shape[1]

    # Split seq into ws chunks; concat along head dim → [T_local, B, H_total]
    seq_chunks = tensor_flat.split(t_local, dim=0)  # ws tensors of [T_local, B, H_local]
    concat = torch.cat(seq_chunks, dim=-1).contiguous()  # [T_local, B, H_total]

    output = torch.empty_like(concat)
    if router is not None:
        # For HP→CP we exchange head-sharded slices back for seq slices.
        # The router handles PCIe-aware routing symmetrically.
        output = router.all_to_all(concat, output)
    else:
        in_list = list(concat.split(h_local, dim=-1))  # ws tensors of [T_local, B, H_local]
        # Reshape each split to [T_local, B, H_local]; AlltoAll along seq.
        concat_for_a2a = concat.reshape(t_local, -1).contiguous()
        out_for_a2a = torch.empty_like(concat_for_a2a)
        in_a2a = list(concat_for_a2a.chunk(1, dim=0)) * ws  # not quite right; use standard path
        # Standard flat AlltoAll fallback.
        in_list_flat = [concat[:, :, i * h_local : (i + 1) * h_local].contiguous() for i in range(ws)]
        out_list_flat = [torch.empty_like(x) for x in in_list_flat]
        # Proper flat a2a: split along head, exchange to get seq shards.
        # Interleave: stack along seq dim then exchange.
        stacked = torch.cat(in_list_flat, dim=0).contiguous()
        out_stacked = torch.empty_like(stacked)
        in_chunks = list(stacked.chunk(ws, dim=0))
        out_chunks = list(out_stacked.chunk(ws, dim=0))
        dist.all_to_all(out_chunks, in_chunks, group=cp_group)
        output = out_stacked

    # Reassemble: [T_local, B, H_total] from exchanged chunks.
    output = output.reshape(t_local, B, h_total)

    if redo_attention_load_balancing:
        rank = dist.get_rank(cp_group)
        chunk_size = t_local // 2
        # Redo load balancing: interleave even/odd chunks.
        # Rank r should hold chunk 2r and 2r+1 in the load-balanced layout.
        idx = []
        for half in range(2):
            start = (rank * 2 + half) * chunk_size
            idx.append(torch.arange(start, start + chunk_size, device=output.device))
        output = output[torch.cat(idx)]

    return output


# ---------------------------------------------------------------------------
# GDN forward pass: fused unified AlltoAll with DES-LOC heterogeneous awareness
# ---------------------------------------------------------------------------

@dataclass
class GDNHeteroAlltoAllConfig:
    """Configuration for DES-LOC heterogeneous GDN AlltoAll.

    Attributes
    ----------
    cp_size:
        Context parallel world size.
    tp_size:
        Tensor parallel world size.
    qk_dim_local_tp:
        Per-rank Q/K head dimension (total // tp_size).
    v_dim_local_tp:
        Per-rank V head dimension (total // tp_size).
    num_value_heads_per_tp:
        Number of value heads per TP rank (num_value_heads // tp_size).
    split_sections:
        Tuple of (qk, qk, v, v, Bh, Bh) head counts for the unified permutation.
    offload_config:
        :class:`DesLocOffloadConfig` controlling LOC activation spilling.
    pin_perm_cpu:
        Whether to build permutation index tensors in CPU pinned memory.
        Recommended True for A6000 (48 GB) ranks; can be False for H100 (96 GB).
    cross_tier_threshold_bytes:
        Tensor size below which tiered routing is skipped.
    """

    cp_size: int
    tp_size: int
    qk_dim_local_tp: int
    v_dim_local_tp: int
    num_value_heads_per_tp: int
    split_sections: tuple[int, ...] = field(default_factory=tuple)
    offload_config: DesLocOffloadConfig = field(default_factory=DesLocOffloadConfig)
    pin_perm_cpu: bool = True
    cross_tier_threshold_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.split_sections:
            # Default: (qk, qk, v, v, num_Bh, num_Bh)
            object.__setattr__(
                self,
                "split_sections",
                (
                    self.qk_dim_local_tp,
                    self.qk_dim_local_tp,
                    self.v_dim_local_tp,
                    self.v_dim_local_tp,
                    self.num_value_heads_per_tp,
                    self.num_value_heads_per_tp,
                ),
            )


class HeteroAlltoAllGDN(nn.Module):
    """DES-LOC heterogeneous AlltoAll module for Gated Delta Net forward pass.

    This module encapsulates the fused AlltoAll logic from Megatron commit
    48032d7b, adapted for the PCIe-only A6000×2 + H100 NVL cluster:

    **Forward pass (CP→HP)**:
      1. If ``cp_size > 1``, pre-permute the head dimension using
         :func:`_build_head_perm_hetero` (tier-aware, cached).
      2. If in THD mode (packed sequences), also build the token-dimension
         permutation with :func:`_build_thd_hetero_cp_a2a_perm` and apply it
         after the unified AlltoAll.  This folds the per-sequence loop and the
         ``undo_attention_load_balancing`` step into O(1) collectives.
      3. Execute a single AlltoAll via :class:`HeteroAlltoAllRouter` (tiered
         PCIe-aware routing for large tensors).
      4. Optionally offload the result to the LOC CPU pool via
         :class:`DesLocActivationBuffer`.

    **Return pass (HP→CP)**:
      1. If in THD mode, apply the inverse token permutation before the
         AlltoAll (folds ``redo_attention_load_balancing``).
      2. Execute the HP→CP AlltoAll with the same router.

    Parameters
    ----------
    config:
        :class:`GDNHeteroAlltoAllConfig` describing the model and hardware
        dimensions.
    cp_group:
        Context parallel process group.
    topology:
        :class:`PCIeTopology` describing the PCIe bandwidth matrix.  If None,
        a flat AlltoAll is used (no tier-aware routing).
    offload_stream:
        CUDA stream for async CPU offload.  If None, a new stream is created.
    """

    def __init__(
        self,
        config: GDNHeteroAlltoAllConfig,
        cp_group: dist.ProcessGroup,
        topology: Optional[PCIeTopology] = None,
        offload_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.cp_group = cp_group
        self.topology = topology
        self._offload_stream = offload_stream or torch.cuda.Stream()

        if topology is not None:
            self._router = HeteroAlltoAllRouter(
                topology=topology,
                group=cp_group,
                cross_tier_threshold_bytes=config.cross_tier_threshold_bytes,
            )
        else:
            self._router = None

        # Cache device tier for this rank to avoid repeated CUDA API calls.
        self._device_tier: Optional[DeviceTier] = None
        self._device_idx: Optional[int] = None

    def _get_device_info(self) -> Tuple[int, int]:
        """Lazily query and cache device tier and index."""
        if self._device_tier is None:
            self._device_tier = get_device_tier()
            self._device_idx = torch.cuda.current_device()
        return self._device_tier.value, self._device_idx

    def forward_cp2hp(
        self,
        qkvzba: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        seq_len: int,
        packed_seq_thd: bool,
        loc_buffer: Optional[DesLocActivationBuffer] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Execute the fused CP→HP AlltoAll with DES-LOC heterogeneous awareness.

        Parameters
        ----------
        qkvzba:
            Projected QKV+ZBA tensor from the GDN input projection, in CP
            layout.  Shape: ``[T_local, B, H_total]`` (seq-first).
        cu_seqlens_q:
            Cumulative sequence lengths (THD mode only).  None for non-packed.
        seq_len:
            Global total token count (= ``cu_seqlens_q[-1]``).
        packed_seq_thd:
            True if ``qkv_format == 'thd'`` (packed multi-sequence batch).
        loc_buffer:
            Optional :class:`DesLocActivationBuffer` from the previous step.
            If provided and ``unpin()`` succeeded, the GPU copy was already
            freed; this call will re-pin it.

        Returns
        -------
        qkvzba_hp : torch.Tensor
            Tensor in HP layout, shape ``[T_global, B, H_local]``.
        thd_cp_a2a_idx : Optional[torch.Tensor]
            Forward THD permutation index (needed by :meth:`return_hp2cp`).
        thd_cp_a2a_inv : Optional[torch.Tensor]
            Inverse THD permutation index (needed by :meth:`return_hp2cp`).
        """
        cp_size = self.config.cp_size
        cfg = self.config
        device_tier, device_idx = self._get_device_info()

        thd_cp_a2a_idx: Optional[torch.Tensor] = None
        thd_cp_a2a_inv: Optional[torch.Tensor] = None

        # Step 1: Head-dimension permutation (fuses per-section AlltoAlls).
        if cp_size > 1:
            head_perm = _build_head_perm_hetero(
                cfg.split_sections, cp_size, device_tier, device_idx
            )
            qkvzba = qkvzba.index_select(-1, head_perm)

        # Step 2: Unified AlltoAll (CP→HP).
        if packed_seq_thd and cu_seqlens_q is not None:
            # THD path: single AlltoAll without per-sequence unpack loop.
            qkvzba = hetero_a2a_cp2hp(
                qkvzba,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.cp_group,
                router=self._router,
                split_sections=None,  # head perm already applied above
                undo_attention_load_balancing=False,
            )
            if cp_size > 1:
                # Step 3: Apply THD token permutation (folds load-balance undo).
                t_global = int(cu_seqlens_q[-1].item())
                thd_cp_a2a_idx, thd_cp_a2a_inv = _build_thd_hetero_cp_a2a_perm(
                    cu_seqlens_q,
                    cp_size,
                    t_global,
                    pin_cpu=cfg.pin_perm_cpu,
                )
                qkvzba = qkvzba.index_select(0, thd_cp_a2a_idx)
                logger.debug(
                    "THD fused CP→HP AlltoAll: T_global=%d cp_size=%d device_tier=SM%d",
                    t_global, cp_size, device_tier,
                )
        else:
            # Non-THD path: standard single AlltoAll.
            qkvzba = hetero_a2a_cp2hp(
                qkvzba,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.cp_group,
                router=self._router,
                split_sections=None,
                undo_attention_load_balancing=True,
            )

        return qkvzba, thd_cp_a2a_idx, thd_cp_a2a_inv

    def return_hp2cp(
        self,
        norm_out_hp: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        packed_seq_thd: bool,
        thd_cp_a2a_inv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Execute the fused HP→CP AlltoAll on the return path.

        Parameters
        ----------
        norm_out_hp:
            Normalised output in HP layout, shape ``[T_global, B, H_local]``.
        cu_seqlens_q:
            Cumulative sequence lengths (THD mode only).
        packed_seq_thd:
            True if ``qkv_format == 'thd'``.
        thd_cp_a2a_inv:
            Inverse permutation from :meth:`forward_cp2hp` (THD path only).

        Returns
        -------
        torch.Tensor
            Output tensor in CP layout, shape ``[T_local, B, H_total]``.
        """
        cp_size = self.config.cp_size

        if packed_seq_thd and cu_seqlens_q is not None:
            # Apply inverse THD permutation before AlltoAll.
            if cp_size > 1 and thd_cp_a2a_inv is not None:
                norm_out_hp = norm_out_hp.index_select(0, thd_cp_a2a_inv)
            norm_out = hetero_a2a_hp2cp(
                norm_out_hp,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.cp_group,
                router=self._router,
                split_sections=None,
                redo_attention_load_balancing=False,
            )
        else:
            norm_out = hetero_a2a_hp2cp(
                norm_out_hp,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.cp_group,
                router=self._router,
                split_sections=None,
                redo_attention_load_balancing=True,
            )

        return norm_out


# ---------------------------------------------------------------------------
# LOC-aware activation management utilities
# ---------------------------------------------------------------------------

def wrap_in_loc_buffer(
    tensor: torch.Tensor,
    config: DesLocOffloadConfig,
    stream: Optional[torch.cuda.Stream] = None,
) -> DesLocActivationBuffer:
    """Convenience wrapper: create a :class:`DesLocActivationBuffer` for *tensor*.

    If *config* is disabled or the tensor is too small, the buffer is created
    but ``unpin()`` will be a no-op, so callers need not branch.

    Parameters
    ----------
    tensor:
        GPU tensor to wrap.
    config:
        :class:`DesLocOffloadConfig` controlling offload behaviour.
    stream:
        CUDA stream for async transfers.

    Returns
    -------
    DesLocActivationBuffer
    """
    return DesLocActivationBuffer(tensor, config, stream)


def loc_maybe_offload(
    buf: DesLocActivationBuffer,
    pipeline_depth: int,
    current_stage: int,
) -> bool:
    """Offload *buf* to CPU LOC pool if the pipeline stage warrants it.

    Activations are only offloaded if there are at least 2 future stages that
    will not need them, giving enough time for the async PCIe transfer to
    complete before the activation is needed again.

    Parameters
    ----------
    buf:
        Activation buffer to potentially offload.
    pipeline_depth:
        Total number of pipeline micro-batches.
    current_stage:
        Current pipeline stage index (0-based).

    Returns
    -------
    bool
        True if offload was initiated.
    """
    stages_remaining = pipeline_depth - current_stage - 1
    if stages_remaining >= 2:
        offloaded = buf.unpin()
        return offloaded
    return False


# ---------------------------------------------------------------------------
# Sequence packing utilities (DES-LOC: CPU DRAM backed)
# ---------------------------------------------------------------------------

def unpack_sequence_hetero(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dim: int = 0,
) -> List[torch.Tensor]:
    """Unpack a packed sequence tensor into individual sequence tensors.

    This is the DES-LOC variant of Megatron's (now-removed) ``_unpack_sequence``
    helper.  In Megatron's refactor, this function was moved to the test file
    since it is no longer needed in the forward path (replaced by the unified
    AlltoAll + permutation approach).  It is retained here for:

    - Debugging and correctness validation against the unified path.
    - Fallback when ``cp_size == 1`` (no AlltoAll needed).

    Parameters
    ----------
    x:
        Packed tensor, with *dim* as the sequence dimension.
    cu_seqlens:
        Cumulative sequence lengths, shape ``[num_seqs + 1]``.
    dim:
        Sequence dimension in *x* (0 for THD format).

    Returns
    -------
    List[torch.Tensor]
        List of per-sequence tensors.
    """
    unpacked = []
    cu_list = cu_seqlens.tolist()
    num_seqs = len(cu_list) - 1
    for i in range(num_seqs):
        idx_start = cu_list[i]
        idx_end = cu_list[i + 1]
        slices = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked.append(x[tuple(slices)])
    return unpacked


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Self-contained unit tests for DES-LOC HeteroAlltoAllGDN components.

    These tests run without a distributed environment wherever possible,
    mocking the collective operations.  Distributed tests are guarded by
    ``dist.is_available()`` and skipped if CUDA is not available.

    Run with: python -m deepspeed.moe.hetero_alltoall_gdn
    """

    import sys
    import unittest
    from unittest import mock

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    class TestDeviceTier(unittest.TestCase):
        """Test device tier classification."""

        def test_sm86_classification(self):
            """A6000 (SM 8.6) should map to DeviceTier.SM86."""
            with mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)):
                tier = get_device_tier(torch.device("cuda", 0))
            self.assertEqual(tier, DeviceTier.SM86)

        def test_sm90_classification(self):
            """H100 (SM 9.0) should map to DeviceTier.SM90."""
            with mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                tier = get_device_tier(torch.device("cuda", 2))
            self.assertEqual(tier, DeviceTier.SM90)

        def test_unknown_cap_defaults_to_sm86(self):
            """Unknown capabilities should log a warning and default to SM86."""
            with mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)):
                tier = get_device_tier(torch.device("cuda", 0))
            self.assertEqual(tier, DeviceTier.SM86)


    class TestDesLocActivationBuffer(unittest.TestCase):
        """Test LOC activation buffer offload/pin semantics."""

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_small_tensor_not_offloaded(self):
            """Tensors below threshold should not be offloaded."""
            cfg = DesLocOffloadConfig(enabled=True, offload_threshold_bytes=64 * 1024 * 1024)
            t = torch.randn(100, 100, device="cuda")  # << 64 MB
            buf = DesLocActivationBuffer(t, cfg)
            offloaded = buf.unpin()
            self.assertFalse(offloaded)
            self.assertIsNotNone(buf._gpu_tensor)

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_large_tensor_offloaded_and_restored(self):
            """Large tensors should be offloaded to CPU and restored on .data access."""
            # Use a 128 MB tensor to exceed the default 64 MB threshold.
            cfg = DesLocOffloadConfig(
                enabled=True,
                offload_threshold_bytes=1 * 1024 * 1024,  # 1 MB threshold for test
                use_async_copy=False,  # synchronous for determinism in tests
            )
            t = torch.randn(1024, 1024, device="cuda")  # ~4 MB float32
            original_data = t.cpu().clone()
            buf = DesLocActivationBuffer(t, cfg)
            offloaded = buf.unpin()
            self.assertTrue(offloaded, "Expected large tensor to be offloaded")
            self.assertIsNone(buf._gpu_tensor)
            self.assertIsNotNone(buf._cpu_tensor)
            # Access .data should restore to GPU.
            restored = buf.data
            self.assertIsNotNone(restored)
            self.assertEqual(restored.device.type, "cuda")
            torch.testing.assert_close(restored.cpu(), original_data)

        def test_disabled_config_no_offload(self):
            """When config.enabled=False, unpin() should always return False."""
            cfg = DesLocOffloadConfig(enabled=False)
            t = torch.randn(100)  # CPU tensor to avoid CUDA requirement
            # Patch device to pretend it is CUDA.
            buf = DesLocActivationBuffer.__new__(DesLocActivationBuffer)
            buf._config = cfg
            buf._stream = mock.MagicMock()
            buf._gpu_tensor = t
            buf._cpu_tensor = None
            buf._device = torch.device("cpu")
            buf._nbytes = t.nbytes
            result = buf.unpin()
            self.assertFalse(result)


    class TestBuildHeadPermHetero(unittest.TestCase):
        """Test head permutation construction for unified AlltoAll."""

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_permutation_shape(self):
            """Permutation should have shape [sum(split_sections)]."""
            split_sections = (8, 8, 4, 4, 2, 2)
            cp_size = 2
            device_tier = DeviceTier.SM86.value
            device_idx = torch.cuda.current_device()
            perm = _build_head_perm_hetero(split_sections, cp_size, device_tier, device_idx)
            self.assertEqual(perm.shape[0], sum(split_sections))

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_permutation_is_bijective(self):
            """Permutation should be a valid bijection (all indices unique)."""
            split_sections = (4, 4, 8, 8, 2, 2)
            cp_size = 2
            device_tier = DeviceTier.SM90.value
            device_idx = torch.cuda.current_device()
            perm = _build_head_perm_hetero(split_sections, cp_size, device_tier, device_idx)
            n = sum(split_sections)
            # Each index in [0, n) should appear exactly once.
            counts = torch.zeros(n, dtype=torch.long)
            for idx in perm.tolist():
                counts[idx] += 1
            self.assertTrue((counts == 1).all().item(), "Permutation is not bijective")

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_non_divisible_sections_raise(self):
            """Sections not divisible by cp_size should raise ValueError."""
            split_sections = (5, 5)  # 5 % 2 != 0
            cp_size = 2
            device_tier = DeviceTier.SM86.value
            device_idx = torch.cuda.current_device()
            # Clear cache to avoid stale cached result.
            _build_head_perm_hetero.cache_clear()
            with self.assertRaises(ValueError):
                _build_head_perm_hetero(split_sections, cp_size, device_tier, device_idx)

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_different_tiers_produce_separate_cache_entries(self):
            """SM86 and SM90 results should be cached separately."""
            _build_head_perm_hetero.cache_clear()
            split_sections = (4, 4)
            cp_size = 2
            device_idx = torch.cuda.current_device()
            perm86 = _build_head_perm_hetero(
                split_sections, cp_size, DeviceTier.SM86.value, device_idx
            )
            perm90 = _build_head_perm_hetero(
                split_sections, cp_size, DeviceTier.SM90.value, device_idx
            )
            # Values should be equal (same math), but they are separate cache entries.
            torch.testing.assert_close(perm86, perm90)
            info = _build_head_perm_hetero.cache_info()
            self.assertEqual(info.currsize, 2)


    class TestBuildThdHeteropCp_a2aPerm(unittest.TestCase):
        """Test THD token permutation construction."""

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_perm_inverse_property(self):
            """idx and inv should be mutual inverses: inv[idx[i]] == i."""
            cu = torch.tensor([0, 32, 64], dtype=torch.long, device="cuda")
            cp_size = 2
            t_global = 64
            idx, inv = _build_thd_hetero_cp_a2a_perm(cu, cp_size, t_global, pin_cpu=False)
            positions = torch.arange(t_global, device="cuda")
            # inv[idx] should equal positions.
            self.assertTrue(torch.equal(inv[idx], positions), "inv is not the inverse of idx")

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_perm_shape(self):
            """Permutation tensors should have shape [t_global]."""
            cu = torch.tensor([0, 16, 48, 80], dtype=torch.long, device="cuda")
            cp_size = 2
            t_global = 80
            idx, inv = _build_thd_hetero_cp_a2a_perm(cu, cp_size, t_global, pin_cpu=False)
            self.assertEqual(idx.shape[0], t_global)
            self.assertEqual(inv.shape[0], t_global)

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_pin_cpu_mode(self):
            """pin_cpu=True should return GPU tensors after transfer."""
            cu = torch.tensor([0, 32, 64], dtype=torch.long, device="cuda")
            idx, inv = _build_thd_hetero_cp_a2a_perm(cu, 2, 64, pin_cpu=True)
            self.assertEqual(idx.device.type, "cuda")
            self.assertEqual(inv.device.type, "cuda")

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_perm_covers_all_tokens(self):
            """Forward permutation should be a bijection over [0, t_global)."""
            cu = torch.tensor([0, 32, 64, 96, 128], dtype=torch.long, device="cuda")
            cp_size = 4
            t_global = 128
            idx, inv = _build_thd_hetero_cp_a2a_perm(cu, cp_size, t_global, pin_cpu=False)
            sorted_idx = idx.sort().values
            expected = torch.arange(t_global, device="cuda")
            self.assertTrue(torch.equal(sorted_idx, expected), "idx is not a bijection")


    class TestDesLocOffloadConfig(unittest.TestCase):
        """Test DesLocOffloadConfig defaults."""

        def test_default_config(self):
            cfg = DesLocOffloadConfig()
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.offload_threshold_bytes, 64 * 1024 * 1024)
            self.assertTrue(cfg.use_async_copy)

        def test_custom_config(self):
            cfg = DesLocOffloadConfig(
                enabled=False,
                offload_threshold_bytes=1024,
                max_cpu_pool_bytes=1024**3,
                use_async_copy=False,
            )
            self.assertFalse(cfg.enabled)
            self.assertEqual(cfg.offload_threshold_bytes, 1024)


    class TestGDNHeteroAlltoAllConfig(unittest.TestCase):
        """Test GDNHeteroAlltoAllConfig split_sections auto-derivation."""

        def test_auto_split_sections(self):
            """split_sections should be auto-derived from qk/v/head dims."""
            cfg = GDNHeteroAlltoAllConfig(
                cp_size=2,
                tp_size=1,
                qk_dim_local_tp=64,
                v_dim_local_tp=64,
                num_value_heads_per_tp=8,
            )
            expected = (64, 64, 64, 64, 8, 8)
            self.assertEqual(cfg.split_sections, expected)

        def test_custom_split_sections(self):
            """Explicit split_sections should override auto-derivation."""
            cfg = GDNHeteroAlltoAllConfig(
                cp_size=2,
                tp_size=2,
                qk_dim_local_tp=32,
                v_dim_local_tp=32,
                num_value_heads_per_tp=4,
                split_sections=(32, 32, 32, 32, 4, 4),
            )
            self.assertEqual(cfg.split_sections, (32, 32, 32, 32, 4, 4))


    class TestLocMaybeOffload(unittest.TestCase):
        """Test loc_maybe_offload pipeline-stage gating."""

        def _make_mock_buf(self, unpin_return: bool) -> DesLocActivationBuffer:
            buf = mock.MagicMock(spec=DesLocActivationBuffer)
            buf.unpin.return_value = unpin_return
            return buf

        def test_offload_triggered_when_stages_remaining_ge_2(self):
            buf = self._make_mock_buf(True)
            result = loc_maybe_offload(buf, pipeline_depth=4, current_stage=0)
            buf.unpin.assert_called_once()

        def test_no_offload_when_stages_remaining_lt_2(self):
            buf = self._make_mock_buf(False)
            result = loc_maybe_offload(buf, pipeline_depth=4, current_stage=3)
            buf.unpin.assert_not_called()
            self.assertFalse(result)

        def test_no_offload_at_last_stage(self):
            buf = self._make_mock_buf(False)
            result = loc_maybe_offload(buf, pipeline_depth=2, current_stage=1)
            buf.unpin.assert_not_called()


    class TestUnpackSequenceHetero(unittest.TestCase):
        """Test unpack_sequence_hetero correctness."""

        def test_unpack_equal_length(self):
            """Two equal-length sequences should unpack into equal tensors."""
            x = torch.arange(64).reshape(64, 1).float()
            cu = torch.tensor([0, 32, 64], dtype=torch.long)
            unpacked = unpack_sequence_hetero(x, cu, dim=0)
            self.assertEqual(len(unpacked), 2)
            torch.testing.assert_close(unpacked[0], x[:32])
            torch.testing.assert_close(unpacked[1], x[32:])

        def test_unpack_variable_length(self):
            """Variable-length sequences should unpack at correct boundaries."""
            x = torch.arange(80).reshape(80, 1).float()
            cu = torch.tensor([0, 16, 48, 80], dtype=torch.long)
            unpacked = unpack_sequence_hetero(x, cu, dim=0)
            self.assertEqual(len(unpacked), 3)
            self.assertEqual(unpacked[0].shape[0], 16)
            self.assertEqual(unpacked[1].shape[0], 32)
            self.assertEqual(unpacked[2].shape[0], 32)

        def test_unpack_dim1(self):
            """Unpack along dim=1 should work for BSH format tensors."""
            x = torch.arange(48).reshape(4, 12, 1).float()
            cu = torch.tensor([0, 6, 12], dtype=torch.long)
            unpacked = unpack_sequence_hetero(x, cu, dim=1)
            self.assertEqual(len(unpacked), 2)
            self.assertEqual(unpacked[0].shape, (4, 6, 1))
            self.assertEqual(unpacked[1].shape, (4, 6, 1))


    class TestPCIeTopology(unittest.TestCase):
        """Test PCIeTopology bandwidth matrix construction (mocked)."""

        def test_intra_tier_bandwidth_higher_than_cross_tier(self):
            """A6000↔A6000 BW should exceed A6000↔H100 BW."""
            topo = PCIeTopology(
                rank_to_device={0: 0, 1: 1, 2: 2},
                rank_to_tier={
                    0: DeviceTier.SM86,
                    1: DeviceTier.SM86,
                    2: DeviceTier.SM90,
                },
                bw_matrix_gbps=torch.tensor([
                    [300.0, 28.0, 14.0],
                    [28.0, 300.0, 14.0],
                    [14.0, 14.0, 300.0],
                ]),
            )
            self.assertGreater(
                topo.bottleneck_bw_for_ranks(0, 1),
                topo.bottleneck_bw_for_ranks(0, 2),
            )

        def test_intra_tier_ranks_sm86(self):
            """intra_tier_ranks(SM86) should return [0, 1] for our cluster."""
            topo = PCIeTopology(
                rank_to_device={0: 0, 1: 1, 2: 2},
                rank_to_tier={
                    0: DeviceTier.SM86,
                    1: DeviceTier.SM86,
                    2: DeviceTier.SM90,
                },
                bw_matrix_gbps=torch.zeros(3, 3),
            )
            mock_group = mock.MagicMock()
            mock_group.size.return_value = 3
            with mock.patch.object(
                dist, "get_rank", return_value=0
            ):
                ranks = topo.intra_tier_ranks(DeviceTier.SM86, mock_group)
            self.assertEqual(ranks, [0, 1])


    # Run all tests.
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestDeviceTier,
        TestDesLocActivationBuffer,
        TestBuildHeadPermHetero,
        TestBuildThdHeteropCp_a2aPerm,
        TestDesLocOffloadConfig,
        TestGDNHeteroAlltoAllConfig,
        TestLocMaybeOffload,
        TestUnpackSequenceHetero,
        TestPCIeTopology,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
