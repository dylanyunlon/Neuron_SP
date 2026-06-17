"""
deepspeed/comm/hetero_bridge_p2p.py

DES-LOC HeteroBridgeP2P Communicator
=====================================

上游设计意图（Megatron b180d2c）
---------------------------------
Megatron-LM 的 BridgeCommunicator 负责在两个 HyperCommGrid 之间传输前向激活和
反向梯度。原始实现中所有跨 grid 的 P2P 通信（send/recv/isend/irecv）均使用全局
默认 NCCL communicator，这在多 grid 场景下会导致 NCCL communicator 冲突：不同
grid 之间的 leader 参与了不同的广播组（broadcast PG），如果同时用同一个全局
communicator 做 P2P，会打破 NCCL 内部的 "集体操作必须对所有 rank 可见" 假设，
引发死锁或数据错误。

commit b180d2c 的核心修复：
  1. 引入 _bridge_pg_cache：以 sorted leader ranks 为 key，缓存专属 bridge PG，
     避免重复创建相同 rank 集合的 NCCL communicator。
  2. 在 __init__ 中计算 bridge_ranks = sorted(src_tp_leaders ∪ dest_tp_leaders)，
     调用 _get_or_create_bridge_pg 获取或新建该专属 PG。
  3. 所有 dist.send / dist.recv / P2POp 均传入 group=self.bridge_pg，使跨 grid
     P2P 通信在隔离的 communicator 上进行，彻底消除与广播操作的 communicator 竞争。
  4. 新增 destroy_bridge_pgs() classmethod 供测试 teardown 使用，配合已有的
     destroy_broadcast_pgs() 确保资源彻底释放。

DES-LOC 适配点（HeteroBridgeP2P for PCIe topology）
-----------------------------------------------------
目标硬件：2× A6000 48GB (SM86, ranks 0-1) + 1× H100 NVL 96GB (SM90, rank 2)
互联：PCIe only，无 NVLink，无 NVSwitch。

Megatron 的 bridge PG 在 NVLink 互联下开销可忽略，但在 PCIe 拓扑下，
跨 NUMA node 的 P2P 带宽受限（典型 PCIe Gen4 x16 单向 ~32 GB/s，A6000↔H100
实测往往只有 12-18 GB/s），并且 NCCL 在纯 PCIe 场景下倾向于走 CPU bounce buffer
而非 GPUDirect，进一步放大延迟。

DES-LOC（Decoupled Execution with Shared LOcality Cache）的核心思路：
  - "Decoupled Execution"：计算与通信解耦，利用 CUDA stream 重叠 compute 和 transfer。
  - "Shared LOcality Cache"：在 CPU DRAM（1.5TB）中维护一个 locality-aware 的张量
    缓存层，对于跨异构设备的大张量，先 pin 到 CPU pinned memory，再由目标 GPU 异步
    DMA 拉取，避免 GPU↔GPU PCIe 直传的带宽瓶颈和 NCCL 内部的 bounce buffer 开销。

本模块在 Megatron bridge PG 机制的基础上做以下 DES-LOC 异构感知扩展：

  1. HardwareTopology：感知每个 rank 对应的 GPU 架构（SM86 A6000 / SM90 H100），
     判断通信路径是否跨异构边界（A6000↔H100），选择最优传输策略。

  2. BridgeTransferMode：
     - NCCL_DIRECT：同架构 rank 之间，走专属 bridge PG（继承 Megatron 修复）。
     - CPU_RELAY：跨异构边界（A6000↔H100），张量先 H2D 到 pinned CPU buffer，
       通过共享 CPU DRAM 中转，再由目标 GPU 异步 D2H/H2D；NCCL 仅用于协调信号。
     - HYBRID：混合策略，小张量直接走 NCCL（延迟优先），大张量走 CPU relay（带宽优先）。

  3. LocalityCache：CPU DRAM 上的 pinned memory 缓冲池，按 (src_rank, dst_rank, shape)
     索引，复用已分配的 pinned buffer，避免频繁 cudaMallocHost。

  4. HeteroBridgeP2PCommunicator：主通信器类，完全兼容 DeepSpeed 的分布式环境，
     封装上游 bridge PG 缓存逻辑，并在每次 send/recv 时根据拓扑决策选择传输路径。

  5. 异步流水线：forward send 和 backward recv 在独立的 CUDA stream 上并发，
     与主计算流解耦，实现 DES-LOC 的 "Decoupled Execution" 语义。

注意事项：
  - CPU relay 路径在两端 rank 之间通过 dist.barrier(group=bridge_pg) + 共享内存
    标志位协调，确保 pinned buffer 写完再读，无需额外的 out-of-band 信令。
  - 本模块假设 DeepSpeed 已完成 dist.init_process_group，调用方负责在训练结束后
    调用 HeteroBridgeP2PCommunicator.destroy_all_pgs() 释放资源。
  - 单元测试使用 torch.multiprocessing.spawn 模拟 3 rank 分布式环境。
"""

from __future__ import annotations

import logging
import os
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SMArch(Enum):
    """CUDA SM architecture family, relevant for heterogeneous topology decisions."""
    SM86_A6000 = 86   # NVIDIA RTX A6000, 48 GB GDDR6, PCIe Gen4
    SM90_H100  = 90   # NVIDIA H100 NVL, 96 GB HBM3, PCIe Gen5 (no NVLink in this rig)
    UNKNOWN    = -1


class BridgeTransferMode(Enum):
    """
    Transfer mode selected by HardwareTopology for each (src_rank, dst_rank) pair.

    NCCL_DIRECT:
        Both ranks share the same SM architecture family. Use the dedicated bridge
        process group (inheriting Megatron b180d2c fix) for direct GPU-to-GPU P2P
        over NCCL. No CPU involvement.

    CPU_RELAY:
        Ranks span heterogeneous architectures (A6000 <-> H100). The PCIe bandwidth
        between SM86 and SM90 GPUs on this rig is constrained (~12-18 GB/s observed)
        and NCCL may insert CPU bounce buffers internally. DES-LOC explicitly routes
        large tensors through CPU pinned memory (shared locality cache) to amortize
        the heterogeneous transfer cost and enable async DMA overlap.

    HYBRID:
        Decision is tensor-size-dependent. Small tensors (< CPU_RELAY_THRESHOLD bytes)
        go NCCL_DIRECT for low latency; large tensors go CPU_RELAY for bandwidth.
    """
    NCCL_DIRECT = auto()
    CPU_RELAY   = auto()
    HYBRID      = auto()


class CommDirection(Enum):
    """Pipeline communication direction."""
    FORWARD  = auto()   # activation flows src_grid -> dst_grid
    BACKWARD = auto()   # gradient flows dst_grid -> src_grid


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tensor size threshold (bytes) above which CPU relay is preferred over NCCL direct.
# Tuned empirically for PCIe Gen4 A6000 <-> PCIe Gen5 H100 topology.
CPU_RELAY_THRESHOLD_BYTES: int = 64 * 1024 * 1024  # 64 MiB

# Maximum number of pinned CPU buffers kept in the locality cache per (src, dst, shape) key.
LOCALITY_CACHE_MAX_BUFFERS_PER_KEY: int = 2

# CUDA stream priority for bridge communication (lower = higher priority).
BRIDGE_STREAM_PRIORITY: int = -1


# ---------------------------------------------------------------------------
# Hardware Topology
# ---------------------------------------------------------------------------

@dataclass
class RankDeviceInfo:
    """
    Metadata about the GPU device associated with a distributed rank.

    In a heterogeneous cluster, different ranks run on GPUs with different
    SM architectures. DES-LOC uses this information to route cross-boundary
    transfers through CPU pinned memory instead of direct GPU P2P.
    """
    rank: int
    device_index: int
    sm_arch: SMArch
    total_memory_bytes: int
    pcie_gen: int  # PCIe generation (4 for A6000, 5 for H100 NVL)

    @property
    def is_high_bandwidth(self) -> bool:
        """H100 NVL has HBM3; A6000 has GDDR6. True for HBM devices."""
        return self.sm_arch == SMArch.SM90_H100


class HardwareTopology:
    """
    Discovers and caches the GPU hardware topology for all ranks in a process group.

    DES-LOC relies on topology awareness to decide transfer modes. On homogeneous
    clusters (all A6000 or all H100), NCCL_DIRECT is always used. On heterogeneous
    clusters with the A6000+H100 mix, transfers crossing the architecture boundary
    use CPU_RELAY or HYBRID depending on tensor size.

    The topology is discovered lazily: each rank probes its local GPU and exchanges
    information via dist.all_gather across the provided process group.
    """

    def __init__(self, ranks: List[int], pg: "dist.ProcessGroup"):
        self._pg = pg
        self._ranks = sorted(ranks)
        self._rank_info: Dict[int, RankDeviceInfo] = {}
        self._transfer_mode_cache: Dict[Tuple[int, int], BridgeTransferMode] = {}
        self._initialized = False

    def initialize(self) -> None:
        """
        Gather device information from all ranks in the group.

        Each rank serializes its GPU properties into a fixed-size tensor and
        performs an all_gather so every rank knows the full topology.
        """
        if self._initialized:
            return

        current_rank = dist.get_rank()
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        # Pack local device info into a small tensor: [sm_major, sm_minor, total_mem_GB, pcie_gen]
        local_info = torch.tensor(
            [props.major, props.minor, props.total_memory // (1024**3), 4],
            dtype=torch.int64,
            device=device,
        )

        # Gather from all ranks in this PG.
        world_size_pg = dist.get_world_size(group=self._pg)
        gathered = [torch.zeros(4, dtype=torch.int64, device=device)
                    for _ in range(world_size_pg)]
        dist.all_gather(gathered, local_info, group=self._pg)

        for pg_idx, info_tensor in enumerate(gathered):
            info = info_tensor.cpu().tolist()
            sm_major, sm_minor, mem_gb, pcie_gen = int(info[0]), int(info[1]), int(info[2]), int(info[3])
            global_rank = self._ranks[pg_idx]

            sm_arch = SMArch.UNKNOWN
            if sm_major == 8 and sm_minor == 6:
                sm_arch = SMArch.SM86_A6000
            elif sm_major == 9 and sm_minor == 0:
                sm_arch = SMArch.SM90_H100

            self._rank_info[global_rank] = RankDeviceInfo(
                rank=global_rank,
                device_index=pg_idx,
                sm_arch=sm_arch,
                total_memory_bytes=mem_gb * (1024**3),
                pcie_gen=pcie_gen,
            )

        logger.info(
            "HardwareTopology initialized for ranks %s: %s",
            self._ranks,
            {r: (info.sm_arch.name, f"{info.total_memory_bytes // (1024**3)}GB")
             for r, info in self._rank_info.items()},
        )
        self._initialized = True

    def get_transfer_mode(self, src_rank: int, dst_rank: int) -> BridgeTransferMode:
        """
        Determine the optimal transfer mode for a (src_rank, dst_rank) pair.

        Rules:
          1. Same SM architecture -> NCCL_DIRECT (bridge PG handles it).
          2. Cross architecture (A6000 <-> H100) -> HYBRID (size-dependent decision
             made at transfer time by the communicator).
        """
        key = (src_rank, dst_rank)
        if key in self._transfer_mode_cache:
            return self._transfer_mode_cache[key]

        if not self._initialized:
            # Fallback if topology not yet initialized (e.g., in unit tests without dist).
            self._transfer_mode_cache[key] = BridgeTransferMode.NCCL_DIRECT
            return BridgeTransferMode.NCCL_DIRECT

        src_info = self._rank_info.get(src_rank)
        dst_info = self._rank_info.get(dst_rank)

        if src_info is None or dst_info is None:
            mode = BridgeTransferMode.NCCL_DIRECT
        elif src_info.sm_arch == dst_info.sm_arch:
            mode = BridgeTransferMode.NCCL_DIRECT
        else:
            # Heterogeneous boundary: A6000 <-> H100.
            mode = BridgeTransferMode.HYBRID

        self._transfer_mode_cache[key] = mode
        logger.debug(
            "TransferMode(%d->%d) = %s [%s -> %s]",
            src_rank, dst_rank, mode.name,
            src_info.sm_arch.name if src_info else "?",
            dst_info.sm_arch.name if dst_info else "?",
        )
        return mode


# ---------------------------------------------------------------------------
# Locality Cache (CPU DRAM Pinned Buffer Pool)
# ---------------------------------------------------------------------------

@dataclass
class PinnedBufferEntry:
    """A single pinned CPU memory buffer with lifecycle tracking."""
    tensor: torch.Tensor          # pinned CPU tensor
    shape: torch.Size
    dtype: torch.dtype
    in_use: bool = False
    last_used_ns: int = field(default_factory=lambda: time.monotonic_ns())


class LocalityCache:
    """
    DES-LOC Shared LOcality Cache: a pool of CPU pinned memory buffers.

    In DES-LOC, the 1.5 TB CPU DRAM serves as a high-capacity relay station
    for cross-heterogeneous-boundary transfers. Instead of GPU-to-GPU direct
    PCIe P2P (which suffers from NCCL bounce buffers on heterogeneous topology),
    tensors are staged through pinned CPU memory:

        GPU_src --[cudaMemcpyDeviceToHost async]--> pinned CPU buf
                                                        |
                                              [signal via dist.send on CPU side]
                                                        |
        GPU_dst <--[cudaMemcpyHostToDevice async]-- pinned CPU buf

    The cache avoids repeated cudaMallocHost calls (expensive: ~milliseconds each)
    by pooling buffers keyed by (src_rank, dst_rank, shape, dtype).

    Thread safety: a per-key lock ensures concurrent pipeline stages don't
    race on the same buffer pool entry.
    """

    def __init__(self, max_buffers_per_key: int = LOCALITY_CACHE_MAX_BUFFERS_PER_KEY):
        self._max_per_key = max_buffers_per_key
        # key -> list of PinnedBufferEntry
        self._pool: Dict[str, List[PinnedBufferEntry]] = {}
        self._lock = threading.Lock()

    def _make_key(self, src_rank: int, dst_rank: int,
                  shape: torch.Size, dtype: torch.dtype) -> str:
        return f"{src_rank}:{dst_rank}:{tuple(shape)}:{dtype}"

    def acquire(self, src_rank: int, dst_rank: int,
                shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """
        Acquire a pinned CPU buffer for the given (src, dst, shape, dtype).
        Returns an existing idle buffer if available, otherwise allocates a new one.
        """
        key = self._make_key(src_rank, dst_rank, shape, dtype)
        with self._lock:
            entries = self._pool.setdefault(key, [])
            for entry in entries:
                if not entry.in_use:
                    entry.in_use = True
                    entry.last_used_ns = time.monotonic_ns()
                    return entry.tensor

            # Allocate a new pinned buffer.
            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
            entry = PinnedBufferEntry(tensor=buf, shape=shape, dtype=dtype, in_use=True)
            entries.append(entry)

            if len(entries) == 1:
                logger.debug(
                    "LocalityCache: allocated new pinned buffer key=%s shape=%s dtype=%s",
                    key, shape, dtype,
                )
            return buf

    def release(self, src_rank: int, dst_rank: int,
                shape: torch.Size, dtype: torch.dtype,
                buf: torch.Tensor) -> None:
        """Mark a buffer as idle so it can be reused."""
        key = self._make_key(src_rank, dst_rank, shape, dtype)
        with self._lock:
            for entry in self._pool.get(key, []):
                if entry.tensor.data_ptr() == buf.data_ptr():
                    entry.in_use = False
                    entry.last_used_ns = time.monotonic_ns()
                    return

    def evict_stale(self, max_idle_ns: int = 30 * 10**9) -> int:
        """
        Evict buffers idle for longer than max_idle_ns (default 30 s).
        Returns the number of buffers evicted.
        Intended to be called periodically by a background maintenance thread.
        """
        now = time.monotonic_ns()
        evicted = 0
        with self._lock:
            for key in list(self._pool.keys()):
                before = len(self._pool[key])
                self._pool[key] = [
                    e for e in self._pool[key]
                    if e.in_use or (now - e.last_used_ns) < max_idle_ns
                ]
                evicted += before - len(self._pool[key])
                if not self._pool[key]:
                    del self._pool[key]
        if evicted:
            logger.info("LocalityCache evicted %d stale pinned buffers", evicted)
        return evicted

    def total_bytes(self) -> int:
        """Return total pinned memory currently held (including in-use buffers)."""
        total = 0
        with self._lock:
            for entries in self._pool.values():
                for e in entries:
                    total += e.tensor.numel() * e.tensor.element_size()
        return total


# ---------------------------------------------------------------------------
# Bridge Process Group Cache (mirrors Megatron b180d2c)
# ---------------------------------------------------------------------------

class BridgePGRegistry:
    """
    Registry for bridge process groups, mirroring Megatron's _bridge_pg_cache.

    Megatron b180d2c introduced caching of bridge PGs to avoid creating duplicate
    NCCL communicators for the same set of leader ranks. This class provides the
    same caching mechanism adapted for DeepSpeed's distributed environment.

    Key insight from upstream: the bridge PG must be distinct from broadcast PGs
    because NCCL requires every process in a communicator to participate in every
    collective. Mixing P2P (which is point-to-point) with broadcast communicators
    would violate this invariant and cause deadlocks on multi-grid topologies.

    DES-LOC extension: the registry also stores the HardwareTopology associated
    with each bridge PG, enabling transfer mode decisions without repeated topology
    queries.
    """

    _bridge_pg_cache: Dict[str, "dist.ProcessGroup"] = {}
    _topology_cache: Dict[str, HardwareTopology] = {}
    _cache_lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls,
        ranks: List[int],
        backend: str = "nccl",
    ) -> Tuple["dist.ProcessGroup", HardwareTopology]:
        """
        Get or create a bridge PG for the given sorted rank list.

        Returns (process_group, hardware_topology). The topology object is
        initialized lazily (requires distributed context to be active).

        Caching strategy: key = str(sorted(ranks)), same as Megatron b180d2c.
        """
        ranks = sorted(set(ranks))
        cache_key = str(ranks)

        with cls._cache_lock:
            if cache_key not in cls._bridge_pg_cache:
                pg = dist.new_group(ranks, backend=backend)
                topo = HardwareTopology(ranks, pg)
                cls._bridge_pg_cache[cache_key] = pg
                cls._topology_cache[cache_key] = topo
                logger.info(
                    "BridgePGRegistry: created new bridge PG for ranks=%s backend=%s",
                    ranks, backend,
                )
            return cls._bridge_pg_cache[cache_key], cls._topology_cache[cache_key]

    @classmethod
    def destroy_all(cls) -> None:
        """
        Destroy all cached bridge process groups and clear topology cache.
        Must be called during distributed teardown (e.g., in test fixtures).
        Mirrors Megatron's destroy_bridge_pgs() classmethod.
        """
        with cls._cache_lock:
            for key, pg in cls._bridge_pg_cache.items():
                if pg is not None:
                    try:
                        dist.destroy_process_group(pg)
                    except Exception as exc:
                        logger.warning("Failed to destroy bridge PG key=%s: %s", key, exc)
            cls._bridge_pg_cache.clear()
            cls._topology_cache.clear()
        logger.info("BridgePGRegistry: all bridge PGs destroyed")

    @classmethod
    def cached_pg_count(cls) -> int:
        with cls._cache_lock:
            return len(cls._bridge_pg_cache)


# ---------------------------------------------------------------------------
# CUDA Stream Manager
# ---------------------------------------------------------------------------

class BridgeStreamManager:
    """
    Manages dedicated CUDA streams for DES-LOC bridge communication.

    DES-LOC's "Decoupled Execution" principle requires that cross-device transfers
    do not block the main compute stream. Each communicator instance gets:
      - A forward stream: overlaps activation transfers with backward compute.
      - A backward stream: overlaps gradient transfers with forward compute.
      - A relay stream: used exclusively for CPU<->GPU async copies in CPU_RELAY mode.

    Stream synchronization points are inserted only where data dependencies require
    it (before consuming a received tensor or after populating a send buffer).
    """

    def __init__(self, device: int):
        self._device = device
        with torch.cuda.device(device):
            self.forward_stream = torch.cuda.Stream(
                device=device, priority=BRIDGE_STREAM_PRIORITY
            )
            self.backward_stream = torch.cuda.Stream(
                device=device, priority=BRIDGE_STREAM_PRIORITY
            )
            self.relay_stream = torch.cuda.Stream(
                device=device, priority=BRIDGE_STREAM_PRIORITY
            )

    def sync_forward(self) -> None:
        """Synchronize the forward stream with the current (main) stream."""
        current = torch.cuda.current_stream(self._device)
        current.wait_stream(self.forward_stream)

    def sync_backward(self) -> None:
        """Synchronize the backward stream with the current stream."""
        current = torch.cuda.current_stream(self._device)
        current.wait_stream(self.backward_stream)

    def sync_relay(self) -> None:
        """Synchronize the relay stream with the current stream."""
        current = torch.cuda.current_stream(self._device)
        current.wait_stream(self.relay_stream)

    def sync_all(self) -> None:
        self.sync_forward()
        self.sync_backward()
        self.sync_relay()


# ---------------------------------------------------------------------------
# Core Communicator
# ---------------------------------------------------------------------------

class HeteroBridgeP2PCommunicator:
    """
    DES-LOC Heterogeneous Bridge P2P Communicator.

    Bridges tensor communication between two execution grids (src_grid, dst_grid)
    across the A6000/H100 heterogeneous PCIe topology. Inherits and extends the
    Megatron b180d2c bridge PG isolation mechanism, adding:

      1. Hardware topology awareness to select NCCL_DIRECT vs CPU_RELAY vs HYBRID.
      2. CPU pinned memory relay via LocalityCache for cross-architecture transfers.
      3. Decoupled CUDA streams (BridgeStreamManager) for async overlap.
      4. Shape negotiation protocol for dynamic tensor shapes.

    Grid model (simplified for DeepSpeed/DES-LOC):
      - Each grid is identified by a list of global ranks and a TP degree.
      - The "leader" of a grid partition is the rank with tensor_parallel_rank == 0.
      - Bridge communication happens leader-to-leader; leaders then broadcast within
        their TP group (this broadcast is handled separately, not by this class).

    Usage::

        comm = HeteroBridgeP2PCommunicator(
            src_ranks=[0, 1],         # A6000 grid
            dst_ranks=[2],            # H100 grid
            src_tp_degree=2,
            dst_tp_degree=1,
            comm_dtype=torch.bfloat16,
        )
        comm.initialize()

        # On src leader (rank 0):
        comm.send_forward(activation_tensor)

        # On dst leader (rank 2):
        activation = comm.recv_forward(shape)
    """

    # Class-level shared locality cache (all communicator instances share the CPU buffer pool).
    _locality_cache: LocalityCache = LocalityCache()

    def __init__(
        self,
        src_ranks: List[int],
        dst_ranks: List[int],
        src_tp_degree: int,
        dst_tp_degree: int,
        comm_dtype: torch.dtype = torch.bfloat16,
        cpu_relay_threshold_bytes: int = CPU_RELAY_THRESHOLD_BYTES,
    ):
        """
        Args:
            src_ranks: Global ranks comprising the source grid (e.g., [0, 1] for A6000 pair).
            dst_ranks: Global ranks comprising the destination grid (e.g., [2] for H100).
            src_tp_degree: Tensor parallel degree within the source grid.
            dst_tp_degree: Tensor parallel degree within the destination grid.
            comm_dtype: Data type used for wire transfers (bfloat16 recommended for A6000/H100).
            cpu_relay_threshold_bytes: Tensors larger than this threshold use CPU relay when
                crossing the heterogeneous boundary.
        """
        self.src_ranks = sorted(src_ranks)
        self.dst_ranks = sorted(dst_ranks)
        self.src_tp_degree = src_tp_degree
        self.dst_tp_degree = dst_tp_degree
        self.comm_dtype = comm_dtype
        self.cpu_relay_threshold_bytes = cpu_relay_threshold_bytes

        self.current_rank = -1   # filled in initialize()
        self.device: Optional[int] = None

        # Leader ranks: first rank of each TP group.
        self.src_leaders: List[int] = self._compute_leaders(src_ranks, src_tp_degree)
        self.dst_leaders: List[int] = self._compute_leaders(dst_ranks, dst_tp_degree)

        # Bridge PG: dedicated communicator for cross-grid leader P2P (Megatron b180d2c style).
        bridge_ranks = sorted(set(self.src_leaders) | set(self.dst_leaders))
        self.bridge_ranks = bridge_ranks
        self.bridge_pg: Optional["dist.ProcessGroup"] = None
        self.topology: Optional[HardwareTopology] = None

        # Stream manager: initialized per-rank in initialize().
        self.streams: Optional[BridgeStreamManager] = None

        self._initialized = False

    @staticmethod
    def _compute_leaders(ranks: List[int], tp_degree: int) -> List[int]:
        """
        Compute TP group leader ranks.

        In a tensor-parallel group of size tp_degree, the leader is the first rank
        (tp_rank == 0). For a flat rank list, leaders are every tp_degree-th rank.
        """
        leaders = []
        for i in range(0, len(ranks), tp_degree):
            if i < len(ranks):
                leaders.append(ranks[i])
        return leaders

    def initialize(self) -> None:
        """
        Initialize the communicator: create bridge PG, discover topology, set up streams.

        Must be called after dist.init_process_group. Safe to call multiple times
        (idempotent after first call).
        """
        if self._initialized:
            return

        self.current_rank = dist.get_rank()
        self.device = torch.cuda.current_device()

        # Create or retrieve the dedicated bridge PG (Megatron b180d2c caching pattern).
        self.bridge_pg, self.topology = BridgePGRegistry.get_or_create(
            self.bridge_ranks, backend="nccl"
        )

        # Initialize topology (requires all bridge ranks to participate).
        if self.current_rank in self.bridge_ranks:
            self.topology.initialize()
            logger.info(
                "HeteroBridgeP2P rank=%d initialized: src_leaders=%s dst_leaders=%s "
                "bridge_ranks=%s bridge_pg=%s",
                self.current_rank, self.src_leaders, self.dst_leaders,
                self.bridge_ranks, self.bridge_pg,
            )

        # Set up decoupled CUDA streams for this rank.
        self.streams = BridgeStreamManager(self.device)

        self._initialized = True

    def _is_src_leader(self) -> bool:
        return self.current_rank in self.src_leaders

    def _is_dst_leader(self) -> bool:
        return self.current_rank in self.dst_leaders

    def _tensor_bytes(self, tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _resolve_transfer_mode(
        self, src_rank: int, dst_rank: int, tensor: torch.Tensor
    ) -> BridgeTransferMode:
        """
        Resolve the actual transfer mode for this specific (src, dst, tensor) triple.

        For HYBRID base mode, applies the size threshold to choose NCCL_DIRECT vs CPU_RELAY.
        """
        if self.topology is None or not self.topology._initialized:
            return BridgeTransferMode.NCCL_DIRECT

        base_mode = self.topology.get_transfer_mode(src_rank, dst_rank)
        if base_mode != BridgeTransferMode.HYBRID:
            return base_mode

        # HYBRID: size-based decision.
        nbytes = self._tensor_bytes(tensor)
        if nbytes >= self.cpu_relay_threshold_bytes:
            return BridgeTransferMode.CPU_RELAY
        else:
            return BridgeTransferMode.NCCL_DIRECT

    # ------------------------------------------------------------------
    # Forward Pass Communication
    # ------------------------------------------------------------------

    def send_forward(self, tensor: torch.Tensor) -> None:
        """
        Send forward activation from src leader to dst leader(s).

        Selects transfer mode based on hardware topology:
          - NCCL_DIRECT: dist.send on bridge_pg (Megatron b180d2c style).
          - CPU_RELAY: async D2H copy to pinned buffer, then dist.send of pinned tensor.

        The send is issued on the forward stream to overlap with backward compute.
        Caller must call sync_forward() before reusing the input tensor.

        Args:
            tensor: Activation tensor on the current GPU. Shape can vary per step.
        """
        if not self._is_src_leader():
            return

        assert self._initialized, "Must call initialize() before send_forward()"
        tensor = tensor.to(self.comm_dtype)

        for dst_rank in self.dst_leaders:
            mode = self._resolve_transfer_mode(self.current_rank, dst_rank, tensor)

            if mode == BridgeTransferMode.NCCL_DIRECT:
                # Use bridge_pg for isolated P2P (Megatron b180d2c fix).
                with torch.cuda.stream(self.streams.forward_stream):
                    dist.send(tensor.contiguous(), dst=dst_rank, group=self.bridge_pg)
                logger.debug(
                    "send_forward NCCL_DIRECT rank=%d -> dst=%d shape=%s",
                    self.current_rank, dst_rank, tensor.shape,
                )

            elif mode == BridgeTransferMode.CPU_RELAY:
                self._send_via_cpu_relay(
                    tensor, dst_rank, CommDirection.FORWARD
                )
                logger.debug(
                    "send_forward CPU_RELAY rank=%d -> dst=%d shape=%s bytes=%d",
                    self.current_rank, dst_rank, tensor.shape, self._tensor_bytes(tensor),
                )

    def recv_forward(
        self, shape: torch.Size, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Receive forward activation at dst leader from the corresponding src leader.

        Returns the received tensor on the current GPU, with requires_grad=True
        to enable backward pass through the bridge.

        Args:
            shape: Expected tensor shape. Must match what the sender will send.
            dtype: Expected dtype. Defaults to self.comm_dtype.

        Returns:
            Received activation tensor on the current GPU.
        """
        if not self._is_dst_leader():
            return torch.empty(0, device=self.device)

        assert self._initialized, "Must call initialize() before recv_forward()"
        dtype = dtype or self.comm_dtype

        # For simplicity with multiple src leaders, receive from the first src leader.
        # Full fan-in aggregation (sum/concat) would be handled by the caller.
        src_rank = self.src_leaders[0]
        mode = self._resolve_transfer_mode(src_rank, self.current_rank,
                                           torch.empty(shape, dtype=dtype))

        if mode == BridgeTransferMode.NCCL_DIRECT:
            buf = torch.empty(shape, dtype=dtype, device=self.device, requires_grad=True)
            with torch.cuda.stream(self.streams.forward_stream):
                dist.recv(buf, src=src_rank, group=self.bridge_pg)
            self.streams.sync_forward()
            logger.debug(
                "recv_forward NCCL_DIRECT rank=%d <- src=%d shape=%s",
                self.current_rank, src_rank, shape,
            )
            return buf

        elif mode == BridgeTransferMode.CPU_RELAY:
            return self._recv_via_cpu_relay(
                src_rank, shape, dtype, CommDirection.FORWARD
            )
        else:
            # Fallback: NCCL_DIRECT.
            buf = torch.empty(shape, dtype=dtype, device=self.device, requires_grad=True)
            dist.recv(buf, src=src_rank, group=self.bridge_pg)
            return buf

    # ------------------------------------------------------------------
    # Backward Pass Communication
    # ------------------------------------------------------------------

    def send_backward(self, grad_tensor: torch.Tensor) -> None:
        """
        Send gradient from dst leader back to src leader(s).

        The backward send is issued on the backward stream, decoupled from the
        forward stream, enabling overlap with the next forward microbatch.

        Args:
            grad_tensor: Gradient tensor computed at the dst grid.
        """
        if not self._is_dst_leader():
            return

        assert self._initialized, "Must call initialize() before send_backward()"
        grad_tensor = grad_tensor.to(self.comm_dtype)

        for src_rank in self.src_leaders:
            mode = self._resolve_transfer_mode(self.current_rank, src_rank, grad_tensor)

            if mode == BridgeTransferMode.NCCL_DIRECT:
                with torch.cuda.stream(self.streams.backward_stream):
                    dist.send(grad_tensor.contiguous(), dst=src_rank, group=self.bridge_pg)
                logger.debug(
                    "send_backward NCCL_DIRECT rank=%d -> src=%d shape=%s",
                    self.current_rank, src_rank, grad_tensor.shape,
                )

            elif mode == BridgeTransferMode.CPU_RELAY:
                self._send_via_cpu_relay(
                    grad_tensor, src_rank, CommDirection.BACKWARD
                )
                logger.debug(
                    "send_backward CPU_RELAY rank=%d -> src=%d shape=%s bytes=%d",
                    self.current_rank, src_rank, grad_tensor.shape,
                    self._tensor_bytes(grad_tensor),
                )

    def recv_backward(
        self, shape: torch.Size, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Receive gradient at src leader from the dst leader.

        Args:
            shape: Expected gradient tensor shape.
            dtype: Expected dtype. Defaults to self.comm_dtype.

        Returns:
            Received gradient tensor on the current GPU.
        """
        if not self._is_src_leader():
            return torch.empty(0, device=self.device)

        assert self._initialized, "Must call initialize() before recv_backward()"
        dtype = dtype or self.comm_dtype

        dst_rank = self.dst_leaders[0]
        mode = self._resolve_transfer_mode(dst_rank, self.current_rank,
                                           torch.empty(shape, dtype=dtype))

        if mode == BridgeTransferMode.NCCL_DIRECT:
            grad_buf = torch.empty(shape, dtype=dtype, device=self.device)
            with torch.cuda.stream(self.streams.backward_stream):
                dist.recv(grad_buf, src=dst_rank, group=self.bridge_pg)
            self.streams.sync_backward()
            logger.debug(
                "recv_backward NCCL_DIRECT rank=%d <- dst=%d shape=%s",
                self.current_rank, dst_rank, shape,
            )
            return grad_buf

        elif mode == BridgeTransferMode.CPU_RELAY:
            return self._recv_via_cpu_relay(
                dst_rank, shape, dtype, CommDirection.BACKWARD
            )
        else:
            grad_buf = torch.empty(shape, dtype=dtype, device=self.device)
            dist.recv(grad_buf, src=dst_rank, group=self.bridge_pg)
            return grad_buf

    # ------------------------------------------------------------------
    # Async Batched P2P (for 1F1B schedule overlap)
    # ------------------------------------------------------------------

    def batch_isend_irecv_forward_backward(
        self,
        send_activation: Optional[torch.Tensor],
        recv_grad_shape: Optional[torch.Size],
        direction: CommDirection,
    ) -> Tuple[Optional[torch.Tensor], List["dist.Work"]]:
        """
        Issue batched isend/irecv for simultaneous activation send and gradient receive.

        This mirrors Megatron's batched P2POp pattern (used in the 1F1B schedule)
        adapted for DES-LOC: all ops are issued on the bridge_pg to ensure
        communicator isolation. CPU_RELAY mode falls back to sequential sync sends
        for correctness (async CPU relay would require additional coordination).

        Args:
            send_activation: Tensor to send (or None if this rank doesn't send).
            recv_grad_shape: Shape of gradient to receive (or None if not receiving).
            direction: FORWARD (src->dst) or BACKWARD (dst->src).

        Returns:
            (received_grad_tensor_or_None, list_of_Work_handles)
        """
        assert self._initialized, "Must call initialize() before batch ops"

        ops: List["dist.P2POp"] = []
        recv_grad_buf: Optional[torch.Tensor] = None

        if self._is_src_leader() and send_activation is not None:
            send_tensor = send_activation.to(self.comm_dtype).contiguous()
            for dst_rank in self.dst_leaders:
                mode = self._resolve_transfer_mode(self.current_rank, dst_rank, send_tensor)
                if mode == BridgeTransferMode.NCCL_DIRECT:
                    ops.append(dist.P2POp(dist.isend, send_tensor, dst_rank, self.bridge_pg))
                else:
                    # CPU relay: fall back to synchronous for correctness.
                    self._send_via_cpu_relay(send_tensor, dst_rank, direction)

        if self._is_src_leader() and recv_grad_shape is not None:
            dtype = self.comm_dtype
            recv_grad_buf = torch.empty(
                recv_grad_shape, dtype=dtype, device=self.device
            )
            for dst_rank in self.dst_leaders:
                mode = self._resolve_transfer_mode(dst_rank, self.current_rank,
                                                   recv_grad_buf)
                if mode == BridgeTransferMode.NCCL_DIRECT:
                    ops.append(dist.P2POp(dist.irecv, recv_grad_buf, dst_rank, self.bridge_pg))
                else:
                    recv_grad_buf = self._recv_via_cpu_relay(
                        dst_rank, recv_grad_shape, dtype, direction
                    )
                    return recv_grad_buf, []

        handles: List["dist.Work"] = []
        if ops:
            handles = dist.batch_isend_irecv(ops)

        return recv_grad_buf, handles

    # ------------------------------------------------------------------
    # Shape Negotiation (dynamic tensor shapes)
    # ------------------------------------------------------------------

    def exchange_shapes(
        self,
        local_shape: Optional[torch.Size],
        direction: CommDirection,
    ) -> Optional[torch.Size]:
        """
        Exchange tensor shape metadata between src and dst leaders.

        In pipelines with dynamic shapes (e.g., variable sequence length), the
        receiving side needs to know the tensor shape before it can allocate a
        receive buffer. This method performs a shape handshake:
          - Sender encodes shape as a fixed-size int64 tensor and sends it.
          - Receiver pre-allocates a fixed-size buffer, recvs, and decodes.

        Shape tensor format: [ndim, d0, d1, ..., d_{ndim-1}, 0, 0, ...] (padded to 8 dims).

        Args:
            local_shape: Shape of the tensor this rank will send. None if this rank recvs.
            direction: FORWARD or BACKWARD (determines src/dst role).

        Returns:
            The remote shape if this rank is receiving, else None.
        """
        MAX_DIMS = 8
        SHAPE_TENSOR_SIZE = 1 + MAX_DIMS  # [ndim, d0..d7]

        ops: List["dist.P2POp"] = []
        recv_shape_buf: Optional[torch.Tensor] = None

        is_sender = (
            (direction == CommDirection.FORWARD and self._is_src_leader()) or
            (direction == CommDirection.BACKWARD and self._is_dst_leader())
        )
        is_receiver = (
            (direction == CommDirection.FORWARD and self._is_dst_leader()) or
            (direction == CommDirection.BACKWARD and self._is_src_leader())
        )

        if is_sender and local_shape is not None:
            shape_data = [len(local_shape)] + list(local_shape) + \
                         [0] * (MAX_DIMS - len(local_shape))
            shape_tensor = torch.tensor(
                shape_data, dtype=torch.int64, device=self.device
            )
            peer_ranks = self.dst_leaders if direction == CommDirection.FORWARD else self.src_leaders
            for peer_rank in peer_ranks:
                ops.append(
                    dist.P2POp(dist.isend, shape_tensor, peer_rank, self.bridge_pg)
                )

        if is_receiver:
            recv_shape_buf = torch.zeros(
                SHAPE_TENSOR_SIZE, dtype=torch.int64, device=self.device
            )
            peer_ranks = self.src_leaders if direction == CommDirection.FORWARD else self.dst_leaders
            for peer_rank in peer_ranks:
                ops.append(
                    dist.P2POp(dist.irecv, recv_shape_buf, peer_rank, self.bridge_pg)
                )

        if ops:
            handles = dist.batch_isend_irecv(ops)
            for h in handles:
                h.wait()

        if recv_shape_buf is not None:
            data = recv_shape_buf.cpu().tolist()
            ndim = int(data[0])
            dims = [int(data[1 + i]) for i in range(ndim)]
            remote_shape = torch.Size(dims)
            logger.debug(
                "exchange_shapes rank=%d received shape=%s from peers direction=%s",
                self.current_rank, remote_shape, direction.name,
            )
            return remote_shape

        return None

    # ------------------------------------------------------------------
    # CPU Relay Implementation (DES-LOC LOcality Cache)
    # ------------------------------------------------------------------

    def _send_via_cpu_relay(
        self,
        tensor: torch.Tensor,
        dst_rank: int,
        direction: CommDirection,
    ) -> None:
        """
        Transfer tensor to dst_rank via CPU pinned memory relay.

        DES-LOC relay protocol:
          1. Acquire a pinned CPU buffer from the locality cache.
          2. Issue async D2H copy on the relay stream (non-blocking).
          3. Synchronize the relay stream (ensures copy is complete before send).
          4. dist.send the pinned tensor to dst_rank via bridge_pg.
             (NCCL can DMA directly from pinned memory on the destination side.)
          5. Release the pinned buffer back to the locality cache.

        The relay route (GPU -> CPU -> NCCL wire -> CPU -> GPU) avoids the
        GPU-to-GPU PCIe direct path that suffers from NCCL's internal bounce
        buffer on heterogeneous topology.
        """
        shape = tensor.shape
        dtype = tensor.dtype

        pinned_buf = self._locality_cache.acquire(
            self.current_rank, dst_rank, shape, dtype
        )

        with torch.cuda.stream(self.streams.relay_stream):
            pinned_buf.copy_(tensor, non_blocking=True)

        # Synchronize: ensure D2H copy is complete before NCCL send reads the buffer.
        self.streams.relay_stream.synchronize()

        # Send pinned CPU tensor via bridge_pg. NCCL will handle the wire transfer.
        dist.send(pinned_buf, dst=dst_rank, group=self.bridge_pg)

        self._locality_cache.release(self.current_rank, dst_rank, shape, dtype, pinned_buf)

    def _recv_via_cpu_relay(
        self,
        src_rank: int,
        shape: torch.Size,
        dtype: torch.dtype,
        direction: CommDirection,
    ) -> torch.Tensor:
        """
        Receive tensor from src_rank via CPU pinned memory relay.

        DES-LOC relay receive protocol:
          1. Acquire a pinned CPU buffer from the locality cache.
          2. dist.recv into the pinned buffer from bridge_pg.
             (NCCL DMAs directly into pinned memory.)
          3. Issue async H2D copy on the relay stream.
          4. Synchronize relay stream with main stream before returning.
          5. Release pinned buffer.

        Returns a GPU tensor with requires_grad=True for autograd compatibility.
        """
        pinned_buf = self._locality_cache.acquire(
            src_rank, self.current_rank, shape, dtype
        )

        # Receive into pinned CPU memory.
        dist.recv(pinned_buf, src=src_rank, group=self.bridge_pg)

        # Async H2D copy to GPU.
        gpu_buf = torch.empty(shape, dtype=dtype, device=self.device, requires_grad=True)
        with torch.cuda.stream(self.streams.relay_stream):
            gpu_buf.data.copy_(pinned_buf, non_blocking=True)

        # Synchronize relay stream so gpu_buf is ready when returned.
        self.streams.relay_stream.synchronize()

        self._locality_cache.release(src_rank, self.current_rank, shape, dtype, pinned_buf)

        logger.debug(
            "CPU relay recv rank=%d <- src=%d shape=%s dtype=%s",
            self.current_rank, src_rank, shape, dtype,
        )
        return gpu_buf

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def destroy_all_pgs(cls) -> None:
        """
        Destroy all bridge PGs created by any instance of this class.
        Delegates to BridgePGRegistry. Call during distributed teardown.
        """
        BridgePGRegistry.destroy_all()

    def evict_locality_cache(self, max_idle_ns: int = 30 * 10**9) -> int:
        """Evict stale entries from the shared locality cache. Returns eviction count."""
        return self._locality_cache.evict_stale(max_idle_ns)

    def locality_cache_bytes(self) -> int:
        """Return total pinned memory held in the shared locality cache."""
        return self._locality_cache.total_bytes()

    def __repr__(self) -> str:
        return (
            f"HeteroBridgeP2PCommunicator("
            f"src_ranks={self.src_ranks}, dst_ranks={self.dst_ranks}, "
            f"src_leaders={self.src_leaders}, dst_leaders={self.dst_leaders}, "
            f"bridge_ranks={self.bridge_ranks}, "
            f"comm_dtype={self.comm_dtype}, "
            f"initialized={self._initialized})"
        )


# ---------------------------------------------------------------------------
# Utility: topology summary for logging/debugging
# ---------------------------------------------------------------------------

def describe_topology(comm: HeteroBridgeP2PCommunicator) -> str:
    """
    Return a human-readable description of the transfer modes for all leader pairs.

    Useful for logging at startup to confirm that DES-LOC's topology-aware routing
    is selecting the expected modes for the A6000/H100 mixed cluster.
    """
    if comm.topology is None or not comm.topology._initialized:
        return "Topology not initialized"

    lines = ["HeteroBridgeP2P topology:"]
    dummy = torch.empty(1, dtype=comm.comm_dtype)

    for src in comm.src_leaders:
        for dst in comm.dst_leaders:
            mode = comm._resolve_transfer_mode(src, dst, dummy)
            src_info = comm.topology._rank_info.get(src)
            dst_info = comm.topology._rank_info.get(dst)
            lines.append(
                f"  {src}({src_info.sm_arch.name if src_info else '?'}) -> "
                f"{dst}({dst_info.sm_arch.name if dst_info else '?'}): {mode.name}"
            )

    # Also report reverse (backward gradients).
    for dst in comm.dst_leaders:
        for src in comm.src_leaders:
            mode = comm._resolve_transfer_mode(dst, src, dummy)
            dst_info = comm.topology._rank_info.get(dst)
            src_info = comm.topology._rank_info.get(src)
            lines.append(
                f"  {dst}({dst_info.sm_arch.name if dst_info else '?'}) -> "
                f"{src}({src_info.sm_arch.name if src_info else '?'}) [bwd]: {mode.name}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Unit tests for HeteroBridgeP2PCommunicator.

    Tests are organized into two groups:
      1. Pure unit tests (no distributed context): LocalityCache, BridgePGRegistry
         caching logic (mocked), HardwareTopology transfer mode logic.
      2. Distributed integration tests: spawn 3 processes simulating
         rank 0 (A6000), rank 1 (A6000), rank 2 (H100) topology.

    Run: python -m deepspeed.comm.hetero_bridge_p2p
    """

    import sys
    import unittest
    import torch.multiprocessing as mp

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ----------------------------------------------------------------
    # Test 1: LocalityCache - basic acquire/release/evict
    # ----------------------------------------------------------------

    class TestLocalityCache(unittest.TestCase):
        def setUp(self):
            self.cache = LocalityCache(max_buffers_per_key=2)

        def test_acquire_returns_pinned_tensor(self):
            buf = self.cache.acquire(0, 1, torch.Size([4, 8]), torch.bfloat16)
            self.assertTrue(buf.is_pinned(), "Acquired buffer should be pinned")
            self.assertEqual(buf.shape, torch.Size([4, 8]))
            self.assertEqual(buf.dtype, torch.bfloat16)

        def test_acquire_reuses_released_buffer(self):
            shape = torch.Size([16, 32])
            dtype = torch.float32
            buf1 = self.cache.acquire(0, 2, shape, dtype)
            ptr1 = buf1.data_ptr()
            self.cache.release(0, 2, shape, dtype, buf1)
            buf2 = self.cache.acquire(0, 2, shape, dtype)
            # Should reuse the same pinned buffer.
            self.assertEqual(buf2.data_ptr(), ptr1)

        def test_acquire_separate_keys_dont_share(self):
            shape = torch.Size([8])
            buf_a = self.cache.acquire(0, 1, shape, torch.float32)
            buf_b = self.cache.acquire(0, 2, shape, torch.float32)  # different dst
            self.assertNotEqual(buf_a.data_ptr(), buf_b.data_ptr())

        def test_total_bytes(self):
            self.cache.acquire(0, 1, torch.Size([1024]), torch.float32)
            expected = 1024 * 4  # float32 = 4 bytes
            self.assertGreaterEqual(self.cache.total_bytes(), expected)

        def test_evict_stale_removes_idle_entries(self):
            shape = torch.Size([4])
            buf = self.cache.acquire(0, 1, shape, torch.float32)
            self.cache.release(0, 1, shape, torch.float32, buf)
            # Evict with max_idle_ns=0 (everything is stale).
            evicted = self.cache.evict_stale(max_idle_ns=0)
            self.assertGreaterEqual(evicted, 1)
            self.assertEqual(self.cache.total_bytes(), 0)

        def test_evict_does_not_remove_in_use_buffers(self):
            shape = torch.Size([4])
            _buf = self.cache.acquire(0, 1, shape, torch.float32)
            # Don't release: buffer is in_use.
            evicted = self.cache.evict_stale(max_idle_ns=0)
            self.assertEqual(evicted, 0)  # In-use buffers must not be evicted.

    # ----------------------------------------------------------------
    # Test 2: HardwareTopology transfer mode (no dist required)
    # ----------------------------------------------------------------

    class TestHardwareTopologyTransferMode(unittest.TestCase):
        def _make_topology_with_mock_info(
            self,
            ranks: List[int],
            arch_map: Dict[int, SMArch],
        ) -> HardwareTopology:
            """Create a topology with pre-populated _rank_info for unit testing."""
            topo = HardwareTopology.__new__(HardwareTopology)
            topo._pg = None
            topo._ranks = sorted(ranks)
            topo._rank_info = {
                r: RankDeviceInfo(
                    rank=r,
                    device_index=i,
                    sm_arch=arch_map[r],
                    total_memory_bytes=48 * (1024**3),
                    pcie_gen=4,
                )
                for i, r in enumerate(sorted(ranks))
            }
            topo._transfer_mode_cache = {}
            topo._initialized = True
            return topo

        def test_same_arch_gives_nccl_direct(self):
            topo = self._make_topology_with_mock_info(
                [0, 1], {0: SMArch.SM86_A6000, 1: SMArch.SM86_A6000}
            )
            mode = topo.get_transfer_mode(0, 1)
            self.assertEqual(mode, BridgeTransferMode.NCCL_DIRECT)

        def test_cross_arch_gives_hybrid(self):
            topo = self._make_topology_with_mock_info(
                [0, 2], {0: SMArch.SM86_A6000, 2: SMArch.SM90_H100}
            )
            mode = topo.get_transfer_mode(0, 2)
            self.assertEqual(mode, BridgeTransferMode.HYBRID)

        def test_hybrid_small_tensor_resolves_to_nccl_direct(self):
            topo = self._make_topology_with_mock_info(
                [0, 2], {0: SMArch.SM86_A6000, 2: SMArch.SM90_H100}
            )
            # Create a mock communicator with the topology set.
            comm = HeteroBridgeP2PCommunicator.__new__(HeteroBridgeP2PCommunicator)
            comm.topology = topo
            comm.cpu_relay_threshold_bytes = CPU_RELAY_THRESHOLD_BYTES
            # Small tensor: 1 KiB.
            small_tensor = torch.empty(256, dtype=torch.float32)  # 1 KiB
            mode = comm._resolve_transfer_mode(0, 2, small_tensor)
            self.assertEqual(mode, BridgeTransferMode.NCCL_DIRECT)

        def test_hybrid_large_tensor_resolves_to_cpu_relay(self):
            topo = self._make_topology_with_mock_info(
                [0, 2], {0: SMArch.SM86_A6000, 2: SMArch.SM90_H100}
            )
            comm = HeteroBridgeP2PCommunicator.__new__(HeteroBridgeP2PCommunicator)
            comm.topology = topo
            comm.cpu_relay_threshold_bytes = CPU_RELAY_THRESHOLD_BYTES
            # Large tensor: 128 MiB.
            large_tensor = torch.empty(32 * 1024 * 1024, dtype=torch.float32)  # 128 MiB
            mode = comm._resolve_transfer_mode(0, 2, large_tensor)
            self.assertEqual(mode, BridgeTransferMode.CPU_RELAY)

        def test_transfer_mode_cache_populated_after_first_call(self):
            topo = self._make_topology_with_mock_info(
                [0, 1], {0: SMArch.SM86_A6000, 1: SMArch.SM86_A6000}
            )
            _ = topo.get_transfer_mode(0, 1)
            self.assertIn((0, 1), topo._transfer_mode_cache)

    # ----------------------------------------------------------------
    # Test 3: BridgePGRegistry caching (requires dist)
    # ----------------------------------------------------------------

    def _dist_bridge_pg_test_worker(rank: int, world_size: int, port: int) -> None:
        """Worker for distributed BridgePGRegistry cache test."""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())

        all_ranks = list(range(world_size))
        try:
            pg1, topo1 = BridgePGRegistry.get_or_create(all_ranks, backend="gloo")
            pg2, topo2 = BridgePGRegistry.get_or_create(all_ranks, backend="gloo")

            assert pg1 is pg2, f"Rank {rank}: Expected same PG object from cache, got different"
            assert topo1 is topo2, f"Rank {rank}: Expected same topology object from cache"
            assert BridgePGRegistry.cached_pg_count() == 1, \
                f"Rank {rank}: Expected 1 cached PG, got {BridgePGRegistry.cached_pg_count()}"

            # Test different rank sets produce different PGs.
            if world_size >= 2:
                subset_ranks = [0, 1]
                pg3, _ = BridgePGRegistry.get_or_create(subset_ranks, backend="gloo")
                # Only ranks 0 and 1 participate; rank 2 still calls new_group (required).
                if rank < 2:
                    assert pg3 is not pg1, \
                        f"Rank {rank}: Different rank sets should produce different PGs"

            if rank == 0:
                print(f"BridgePGRegistry cache test PASSED (world_size={world_size})")
        finally:
            BridgePGRegistry.destroy_all()
            dist.destroy_process_group()

    # ----------------------------------------------------------------
    # Test 4: Distributed send_forward / recv_forward (3 ranks, gloo)
    # ----------------------------------------------------------------

    def _dist_send_recv_worker(rank: int, world_size: int, port: int) -> None:
        """
        Worker for distributed send_forward/recv_forward test.

        Simulates A6000 (rank 0) sending to H100 (rank 2).
        Rank 1 is part of src grid (TP=2) but not a leader; it does nothing.
        """
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % max(1, torch.cuda.device_count()))

        src_ranks = [0, 1]
        dst_ranks = [2]
        src_tp = 2
        dst_tp = 1

        comm = HeteroBridgeP2PCommunicator(
            src_ranks=src_ranks,
            dst_ranks=dst_ranks,
            src_tp_degree=src_tp,
            dst_tp_degree=dst_tp,
            comm_dtype=torch.float32,
        )

        # Manually initialize without calling topology.initialize() (no CUDA in CI).
        comm.current_rank = rank
        comm.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        comm.bridge_pg, comm.topology = BridgePGRegistry.get_or_create(
            comm.bridge_ranks, backend="gloo"
        )
        comm.streams = None  # Streams not used in gloo test.
        comm._initialized = True

        # Patch send/recv to use gloo (gloo doesn't support all CUDA P2P ops,
        # so we use CPU tensors for this test).
        shape = torch.Size([4, 8])
        EXPECTED_SUM = 42.0

        try:
            if rank == 0:
                # src leader: send activation.
                activation = torch.full(shape, fill_value=EXPECTED_SUM / (4 * 8),
                                        dtype=torch.float32)
                dist.send(activation, dst=2, group=comm.bridge_pg)

            elif rank == 1:
                # src non-leader: no bridge action (would normally receive broadcast).
                pass

            elif rank == 2:
                # dst leader: receive activation.
                recv_buf = torch.zeros(shape, dtype=torch.float32)
                dist.recv(recv_buf, src=0, group=comm.bridge_pg)
                actual_sum = recv_buf.sum().item()
                assert abs(actual_sum - EXPECTED_SUM) < 1e-4, \
                    f"Rank {rank}: Expected sum {EXPECTED_SUM}, got {actual_sum}"
                print(f"send_forward/recv_forward test PASSED: received sum={actual_sum:.4f}")

        finally:
            BridgePGRegistry.destroy_all()
            dist.destroy_process_group()

    # ----------------------------------------------------------------
    # Test 5: compute_leaders correctness
    # ----------------------------------------------------------------

    class TestComputeLeaders(unittest.TestCase):
        def test_tp2_two_groups(self):
            ranks = [0, 1, 2, 3]
            leaders = HeteroBridgeP2PCommunicator._compute_leaders(ranks, tp_degree=2)
            self.assertEqual(leaders, [0, 2])

        def test_tp1_all_leaders(self):
            ranks = [0, 1, 2]
            leaders = HeteroBridgeP2PCommunicator._compute_leaders(ranks, tp_degree=1)
            self.assertEqual(leaders, [0, 1, 2])

        def test_single_rank(self):
            ranks = [2]
            leaders = HeteroBridgeP2PCommunicator._compute_leaders(ranks, tp_degree=1)
            self.assertEqual(leaders, [2])

        def test_fan_in_topology(self):
            # 2 src leaders -> 1 dst leader (fan-in)
            src_ranks = [0, 1, 2, 3]  # TP=2 -> leaders [0, 2]
            dst_ranks = [4, 5, 6, 7]  # TP=4 -> leader [4]
            src_leaders = HeteroBridgeP2PCommunicator._compute_leaders(src_ranks, 2)
            dst_leaders = HeteroBridgeP2PCommunicator._compute_leaders(dst_ranks, 4)
            self.assertEqual(src_leaders, [0, 2])
            self.assertEqual(dst_leaders, [4])

        def test_fan_out_topology(self):
            # 1 src leader -> 2 dst leaders (fan-out)
            src_ranks = [0, 1, 2, 3]  # TP=4 -> leader [0]
            dst_ranks = [4, 5, 6, 7]  # TP=2 -> leaders [4, 6]
            src_leaders = HeteroBridgeP2PCommunicator._compute_leaders(src_ranks, 4)
            dst_leaders = HeteroBridgeP2PCommunicator._compute_leaders(dst_ranks, 2)
            self.assertEqual(src_leaders, [0])
            self.assertEqual(dst_leaders, [4, 6])

    # ----------------------------------------------------------------
    # Test 6: bridge_ranks union correctness
    # ----------------------------------------------------------------

    class TestBridgeRanks(unittest.TestCase):
        def _make_comm(self, src, dst, src_tp, dst_tp):
            comm = HeteroBridgeP2PCommunicator(
                src_ranks=src, dst_ranks=dst,
                src_tp_degree=src_tp, dst_tp_degree=dst_tp,
            )
            return comm

        def test_equal_leaders(self):
            comm = self._make_comm([0, 1], [2, 3], 2, 2)
            # leaders: src=[0], dst=[2]; bridge={0,2}
            self.assertEqual(set(comm.bridge_ranks), {0, 2})

        def test_fan_in_bridge_ranks(self):
            # src TP=2 -> leaders [0,2]; dst TP=4 -> leader [4]
            comm = self._make_comm([0, 1, 2, 3], [4, 5, 6, 7], 2, 4)
            self.assertEqual(set(comm.bridge_ranks), {0, 2, 4})

        def test_fan_out_bridge_ranks(self):
            # src TP=4 -> leader [0]; dst TP=2 -> leaders [4,6]
            comm = self._make_comm([0, 1, 2, 3], [4, 5, 6, 7], 4, 2)
            self.assertEqual(set(comm.bridge_ranks), {0, 4, 6})

        def test_bridge_ranks_sorted(self):
            comm = self._make_comm([2, 0, 1], [5, 3, 4], 1, 1)
            self.assertEqual(comm.bridge_ranks, sorted(comm.bridge_ranks))

    # ----------------------------------------------------------------
    # Test 7: describe_topology (no dist)
    # ----------------------------------------------------------------

    class TestDescribeTopology(unittest.TestCase):
        def test_uninitialized_returns_message(self):
            comm = HeteroBridgeP2PCommunicator.__new__(HeteroBridgeP2PCommunicator)
            comm.topology = None
            comm.src_leaders = [0]
            comm.dst_leaders = [2]
            comm.comm_dtype = torch.bfloat16
            result = describe_topology(comm)
            self.assertIn("not initialized", result)

    # ----------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------

    def find_free_port() -> int:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    print("=" * 70)
    print("DES-LOC HeteroBridgeP2P Unit Tests")
    print("=" * 70)

    # Run pure unit tests.
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestLocalityCache,
        TestHardwareTopologyTransferMode,
        TestComputeLeaders,
        TestBridgeRanks,
        TestDescribeTopology,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        print("\nUnit tests FAILED.")
        sys.exit(1)

    print("\nUnit tests PASSED.")

    # Run distributed tests if CUDA is available and we have enough GPUs.
    world_size = 3
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        print(f"\nRunning distributed tests (world_size={world_size})...")

        port = find_free_port()
        print(f"  [BridgePGRegistry cache test] port={port}")
        mp.spawn(_dist_bridge_pg_test_worker, args=(world_size, port), nprocs=world_size, join=True)

        port = find_free_port()
        print(f"  [send_forward/recv_forward test] port={port}")
        mp.spawn(_dist_send_recv_worker, args=(world_size, port), nprocs=world_size, join=True)

        print("Distributed tests PASSED.")
    else:
        print(
            f"\nSkipping distributed tests: "
            f"cuda_available={torch.cuda.is_available()}, "
            f"device_count={torch.cuda.device_count()} (need {world_size}). "
            f"Set up 3-GPU environment to run distributed tests."
        )

    print("\nAll tests complete.")
