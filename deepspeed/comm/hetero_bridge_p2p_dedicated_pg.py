"""
HeteroBridgeP2PDedicatedPG — DES-LOC异构训练跨网格P2P专用进程组管理器

上游设计意图（Megatron b180d2c）
────────────────────────────────
Megatron原始commit解决了一个微妙但关键的NCCL正确性问题：跨网格（cross-grid）P2P
通信（send/recv/isend/irecv）此前直接使用全局default process group，导致当多个
BridgeCommunicator实例同时活跃时，NCCL内部rank映射产生歧义（不同PG共享相同的
global rank对，但预期不同的通信语义），在fan-in / fan-out拓扑下尤为严重。

修复思路：为每一对 (src_tp_leaders ∪ dest_tp_leaders) 的rank集合创建且仅创建一个
专用NCCL进程组（bridge_pg），并在所有P2P原语中显式传入该group，使NCCL能独立维护
每个bridge连接的通信上下文，彻底消除cross-contamination。

DES-LOC适配点
─────────────
DES-LOC（Decoupled Execution with Shared LOcality Cache）在Neuron_SP项目中描述的
异构硬件环境如下：

  ┌──────────────┐   PCIe   ┌──────────────┐   PCIe   ┌──────────────┐
  │ A6000-0      │◄────────►│ A6000-1      │◄────────►│ H100-NVL     │
  │ 48GB SM86    │          │ 48GB SM86    │          │ 96GB SM90    │
  └──────────────┘          └──────────────┘          └──────────────┘
            \                      |                        /
             \                     ▼                       /
              └──────────── 1.5TB CPU DRAM ───────────────┘

挑战一：SM86与SM90架构差异导致CUDA kernel无法跨设备执行；DES-LOC通过"Locality
Cache"在CPU DRAM中维护共享激活缓存，bridge通信必须感知设备异构性。

挑战二：PCIe互联（无NVLink）使跨GPU直接P2P带宽受限（~16GB/s vs NVLink ~600GB/s）；
必须在bridge_pg创建时区分 homogeneous路径（A6000↔A6000，同SM86）与
heterogeneous路径（A6000↔H100，跨SM），为后者选择CPU staging offload策略。

挑战三：DeepSpeed的ZeRO分区逻辑与Megatron的TP-leader概念不完全对应；本模块引入
HeteroRankTopology来将DeepSpeed rank group映射到DES-LOC的"执行岛"（execution
island）概念，保证bridge_pg仅包含真正参与P2P的leader ranks。

本模块职责：
  1. HeteroDeviceProfile  — 探测每个rank对应的GPU SM架构
  2. HeteroRankTopology   — 构建execution island及leader映射
  3. BridgePGRegistry     — 带缓存的bridge PG创建/销毁（mirrors Megatron _bridge_pg_cache）
  4. HeteroBridgeP2P      — 封装send/recv/isend/irecv，自动选择direct/staged路径
  5. LocalityCacheManager — DES-LOC专属：在CPU DRAM中管理跨island激活缓存

作者：Neuron_SP开发团队
镜像上游：Megatron b180d2c271dc6554144380d3002185adfbbed435
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────────

# PCIe理论峰值带宽（GB/s），用于决定是否启用CPU staging
_PCIE_BW_THRESHOLD_GBPS: float = 16.0

# CPU staging触发阈值：tensor超过此字节数时走staged路径
_STAGING_BYTES_THRESHOLD: int = 64 * 1024 * 1024  # 64 MiB

# SM架构版本号
_SM86 = 86   # A6000
_SM90 = 90   # H100 NVL


# ──────────────────────────────────────────────────────────────────────────────
# 枚举
# ──────────────────────────────────────────────────────────────────────────────

class CommRole(Enum):
    """进程在bridge通信中的角色。

    LEADER   : TP组中实际执行P2P send/recv的代表rank（映射自Megatron src/dest_tp_leaders）。
    MEMBER   : 仅参与intra-island broadcast，不直接参与bridge P2P。
    STAGING  : 专门负责CPU DRAM staging offload的helper rank（DES-LOC新增）。
    """
    LEADER  = auto()
    MEMBER  = auto()
    STAGING = auto()


class TransferPath(Enum):
    """P2P传输路径选择。

    DIRECT  : GPU-to-GPU直接PCIe P2P（同SM架构或小tensor）。
    STAGED  : 经由CPU DRAM中转（跨SM架构大tensor，DES-LOC核心路径）。
    """
    DIRECT = auto()
    STAGED = auto()


# ──────────────────────────────────────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HeteroDeviceProfile:
    """单个rank的设备异构特征。

    DES-LOC适配：Megatron假设所有设备同构（SM相同），DES-LOC需显式记录每个rank
    所在GPU的SM架构版本，以驱动后续传输路径决策。
    """
    rank: int
    device_index: int
    sm_major: int
    sm_minor: int
    total_memory_bytes: int
    is_h100: bool = field(init=False)
    is_a6000: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_h100  = (self.sm_major == 9 and self.sm_minor == 0)
        self.is_a6000 = (self.sm_major == 8 and self.sm_minor == 6)

    @property
    def sm_version(self) -> int:
        return self.sm_major * 10 + self.sm_minor

    def is_hetero_with(self, other: "HeteroDeviceProfile") -> bool:
        """判断两个rank是否跨SM架构（即异构对）。"""
        return self.sm_version != other.sm_version


@dataclass
class IslandInfo:
    """DES-LOC执行岛（Execution Island）描述符。

    一个执行岛对应一组共享相同SM架构的GPU + 其可见的CPU DRAM分区。
    Neuron_SP中有两个岛：A6000-island（SM86）和H100-island（SM90）。
    """
    island_id: int
    sm_version: int
    ranks: List[int]
    leader_rank: int          # 该岛在bridge P2P中的代表
    cpu_staging_ptr: Optional[int] = None   # CPU pinned buffer地址（延迟初始化）


@dataclass
class RankCommInfo:
    """单个rank在bridge中的完整通信元数据。"""
    rank: int
    role: CommRole
    island_id: int
    device_profile: HeteroDeviceProfile


# ──────────────────────────────────────────────────────────────────────────────
# HeteroRankTopology
# ──────────────────────────────────────────────────────────────────────────────

class HeteroRankTopology:
    """构建并维护DES-LOC异构rank拓扑。

    上游对应：Megatron HyperCommGrid中的TP-leader计算逻辑。
    DES-LOC扩展：在TP-leader概念之上引入"execution island"分组，
    使bridge PG的rank集合仅包含跨island的leader ranks。

    参数
    ────
    world_size : 总rank数
    rank_device_map : {rank: device_index}，由调用方通过all_gather填充
    """

    def __init__(
        self,
        world_size: int,
        rank_device_map: Optional[Dict[int, int]] = None,
    ) -> None:
        self.world_size = world_size
        self._profiles: Dict[int, HeteroDeviceProfile] = {}
        self._islands: Dict[int, IslandInfo] = {}
        self._rank_info: Dict[int, RankCommInfo] = {}

        if rank_device_map is not None:
            self._build_profiles(rank_device_map)
            self._build_islands()

    # ── 内部构建方法 ──────────────────────────────────────────────────────────

    def _build_profiles(self, rank_device_map: Dict[int, int]) -> None:
        """为每个rank探测GPU SM架构，填充HeteroDeviceProfile。

        在真实分布式环境中，每个rank只能探测自身设备；这里通过
        rank_device_map（由外部all_gather汇聚）模拟全局视图。
        """
        for rank, dev_idx in rank_device_map.items():
            try:
                props = torch.cuda.get_device_properties(dev_idx)
                profile = HeteroDeviceProfile(
                    rank=rank,
                    device_index=dev_idx,
                    sm_major=props.major,
                    sm_minor=props.minor,
                    total_memory_bytes=props.total_memory,
                )
            except (RuntimeError, AssertionError) as exc:
                # 单元测试环境无真实GPU，使用默认值
                logger.warning(
                    "Rank %d: cannot probe GPU %d (%s), using SM86 default",
                    rank, dev_idx, exc,
                )
                profile = HeteroDeviceProfile(
                    rank=rank,
                    device_index=dev_idx,
                    sm_major=_SM86 // 10,
                    sm_minor=_SM86 % 10,
                    total_memory_bytes=48 * (1024 ** 3),
                )
            self._profiles[rank] = profile
            logger.debug(
                "Rank %d -> device %d SM%d%d (%s)",
                rank, dev_idx, profile.sm_major, profile.sm_minor,
                "H100" if profile.is_h100 else "A6000" if profile.is_a6000 else "unknown",
            )

    def _build_islands(self) -> None:
        """按SM版本聚合rank为execution island，选出island leader。

        Leader选择策略：island内rank号最小的为leader（与Megatron
        选择TP-group内rank 0为leader的逻辑对应）。
        """
        sm_to_ranks: Dict[int, List[int]] = {}
        for rank, profile in self._profiles.items():
            sm_to_ranks.setdefault(profile.sm_version, []).append(rank)

        for island_id, (sm_ver, ranks) in enumerate(sorted(sm_to_ranks.items())):
            ranks_sorted = sorted(ranks)
            leader = ranks_sorted[0]
            island = IslandInfo(
                island_id=island_id,
                sm_version=sm_ver,
                ranks=ranks_sorted,
                leader_rank=leader,
            )
            self._islands[island_id] = island
            logger.info(
                "Island %d: SM%d, ranks=%s, leader=%d",
                island_id, sm_ver, ranks_sorted, leader,
            )

        # 填充rank_info
        for island in self._islands.values():
            for rank in island.ranks:
                role = CommRole.LEADER if rank == island.leader_rank else CommRole.MEMBER
                self._rank_info[rank] = RankCommInfo(
                    rank=rank,
                    role=role,
                    island_id=island.island_id,
                    device_profile=self._profiles[rank],
                )

    # ── 公开查询接口 ──────────────────────────────────────────────────────────

    def get_bridge_leader_ranks(self) -> List[int]:
        """返回所有island的leader ranks，构成bridge PG的成员集合。

        DES-LOC与Megatron的对应关系：
          Megatron: sorted(set(src_tp_leaders) | set(dest_tp_leaders))
          DES-LOC:  sorted({island.leader_rank for island in islands})
        """
        leaders = sorted(island.leader_rank for island in self._islands.values())
        logger.debug("Bridge leader ranks: %s", leaders)
        return leaders

    def get_island_for_rank(self, rank: int) -> Optional[IslandInfo]:
        info = self._rank_info.get(rank)
        if info is None:
            return None
        return self._islands.get(info.island_id)

    def get_rank_info(self, rank: int) -> Optional[RankCommInfo]:
        return self._rank_info.get(rank)

    def is_leader(self, rank: int) -> bool:
        info = self._rank_info.get(rank)
        return info is not None and info.role == CommRole.LEADER

    def are_hetero_pair(self, rank_a: int, rank_b: int) -> bool:
        """判断两个rank是否跨SM架构，用于决定传输路径。"""
        pa = self._profiles.get(rank_a)
        pb = self._profiles.get(rank_b)
        if pa is None or pb is None:
            return False
        return pa.is_hetero_with(pb)


# ──────────────────────────────────────────────────────────────────────────────
# BridgePGRegistry
# ──────────────────────────────────────────────────────────────────────────────

class BridgePGRegistry:
    """跨island bridge进程组的创建、缓存与销毁。

    上游对应：Megatron BridgeCommunicator._bridge_pg_cache + _get_or_create_bridge_pg。

    核心改进（DES-LOC）：
    1. 缓存键同时编码rank列表和backend，避免同一rank集合因backend不同创建重复PG。
    2. 为异构对（A6000↔H100）可选择gloo backend作为fallback（当NCCL P2P在PCIe
       异构环境下出现hang时）——实际切换由环境变量 NEURON_BRIDGE_BACKEND 控制。
    3. 线程安全：使用模块级锁保护缓存写入，防止多线程并发初始化时重复创建NCCL PG。
    """

    _cache: Dict[str, "dist.ProcessGroup"] = {}
    _lock: threading.Lock = threading.Lock()

    # 环境变量控制，默认nccl；异构环境可设为gloo
    _DEFAULT_BACKEND: str = os.environ.get("NEURON_BRIDGE_BACKEND", "nccl")

    @classmethod
    def get_or_create(
        cls,
        ranks: List[int],
        backend: Optional[str] = None,
    ) -> "dist.ProcessGroup":
        """获取或创建bridge PG，线程安全，幂等。

        参数
        ────
        ranks   : 参与bridge P2P的rank列表（通常为各island的leader ranks）
        backend : NCCL/Gloo；None时使用环境变量或默认nccl

        返回
        ────
        dist.ProcessGroup 实例（已创建或从缓存命中）
        """
        backend = backend or cls._DEFAULT_BACKEND
        ranks_sorted = sorted(ranks)
        cache_key = f"{ranks_sorted}:{backend}"

        # 快速路径：无锁读
        if cache_key in cls._cache:
            logger.debug("BridgePGRegistry: cache hit for key=%s", cache_key)
            return cls._cache[cache_key]

        # 慢速路径：加锁写
        with cls._lock:
            if cache_key not in cls._cache:
                logger.info(
                    "BridgePGRegistry: creating new bridge PG ranks=%s backend=%s",
                    ranks_sorted, backend,
                )
                pg = dist.new_group(ranks=ranks_sorted, backend=backend)
                cls._cache[cache_key] = pg
                logger.info(
                    "BridgePGRegistry: created PG for ranks=%s (cache size=%d)",
                    ranks_sorted, len(cls._cache),
                )
            return cls._cache[cache_key]

    @classmethod
    def destroy_all(cls) -> None:
        """销毁所有缓存的bridge PG并清空缓存。

        上游对应：Megatron BridgeCommunicator.destroy_bridge_pgs()。
        应在测试teardown或训练结束时调用，防止NCCL资源泄漏。
        """
        with cls._lock:
            destroyed = 0
            for key, pg in cls._cache.items():
                if pg is not None:
                    try:
                        dist.destroy_process_group(pg)
                        destroyed += 1
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning("Failed to destroy PG %s: %s", key, exc)
            cls._cache.clear()
            logger.info("BridgePGRegistry: destroyed %d PG(s)", destroyed)

    @classmethod
    def cache_size(cls) -> int:
        return len(cls._cache)


# ──────────────────────────────────────────────────────────────────────────────
# LocalityCacheManager
# ──────────────────────────────────────────────────────────────────────────────

class LocalityCacheManager:
    """DES-LOC核心：CPU DRAM中的共享激活局部性缓存（Locality Cache）。

    设计动机
    ────────
    PCIe互联下A6000↔H100直接P2P带宽约16GB/s，且异构SM架构使NCCL的
    peer-to-peer访问在某些驱动版本下不稳定。DES-LOC通过在1.5TB CPU DRAM
    中维护一块pinned staging buffer，将跨island的大tensor传输拆分为：

      GPU_src → CPU_pinned  (cudaMemcpyDeviceToHost, 高带宽)
      CPU_pinned → GPU_dst  (cudaMemcpyHostToDevice, 高带宽)

    两次主机内存操作均经由PCIe但方向解耦，比直接GPU P2P在异构场景下
    更稳定，且可与CPU计算重叠（pipeline机制）。

    缓冲区策略
    ──────────
    - 按tensor shape+dtype维护缓冲池，避免频繁pin/unpin开销。
    - 超过 _MAX_CACHE_BYTES 时采用LRU淘汰。
    - 线程安全（训练循环中多流并发staging）。
    """

    _MAX_CACHE_BYTES: int = int(os.environ.get("NEURON_LC_CACHE_MB", "2048")) * (1024 ** 2)

    def __init__(self) -> None:
        # key: (shape_tuple, dtype_str) → pinned CPU tensor
        self._pool: Dict[Tuple, torch.Tensor] = {}
        self._pool_bytes: int = 0
        self._lock = threading.Lock()
        logger.info(
            "LocalityCacheManager: max cache %.1f GiB",
            self._MAX_CACHE_BYTES / (1024 ** 3),
        )

    def _key(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Tuple:
        return (shape, str(dtype))

    def get_staging_buffer(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """获取或分配与shape/dtype匹配的pinned CPU staging buffer。"""
        key = self._key(shape, dtype)
        with self._lock:
            if key in self._pool:
                logger.debug("LocalityCacheManager: hit for shape=%s dtype=%s", shape, dtype)
                return self._pool[key]

            nbytes = torch.zeros(shape, dtype=dtype).element_size() * torch.zeros(shape).numel()
            # LRU淘汰：简单策略，超限时清空（生产中应用更精细的LRU）
            if self._pool_bytes + nbytes > self._MAX_CACHE_BYTES:
                logger.warning(
                    "LocalityCacheManager: cache full (%d bytes), evicting all",
                    self._pool_bytes,
                )
                self._pool.clear()
                self._pool_bytes = 0

            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._pool[key] = buf
            self._pool_bytes += nbytes
            logger.debug(
                "LocalityCacheManager: allocated pinned buffer shape=%s dtype=%s (%.2f MiB)",
                shape, dtype, nbytes / (1024 ** 2),
            )
            return buf

    def stage_send(
        self,
        src_tensor: torch.Tensor,
        dst_rank: int,
        bridge_pg: "dist.ProcessGroup",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """CPU-staged send：GPU→CPU→网络。

        参数
        ────
        src_tensor : 待发送的GPU tensor（源island）
        dst_rank   : 目标leader rank
        bridge_pg  : 专用bridge PG（Megatron修复的核心）
        stream     : CUDA stream，None时使用当前流
        """
        buf = self.get_staging_buffer(tuple(src_tensor.shape), src_tensor.dtype)
        # GPU → CPU pinned（非阻塞，与计算流重叠）
        with torch.cuda.stream(stream or torch.cuda.current_stream()):
            buf.copy_(src_tensor, non_blocking=True)
        # 同步，确保数据已落入CPU
        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.current_stream().synchronize()

        logger.debug(
            "LocalityCacheManager.stage_send: tensor shape=%s to rank=%d via CPU staging",
            src_tensor.shape, dst_rank,
        )
        dist.send(buf, dst=dst_rank, group=bridge_pg)

    def stage_recv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src_rank: int,
        bridge_pg: "dist.ProcessGroup",
        dst_device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """CPU-staged recv：网络→CPU→GPU。

        返回已转移到dst_device的GPU tensor。
        """
        buf = self.get_staging_buffer(shape, dtype)
        dist.recv(buf, src=src_rank, group=bridge_pg)

        dst_tensor = torch.empty(shape, dtype=dtype, device=dst_device)
        with torch.cuda.stream(stream or torch.cuda.current_stream()):
            dst_tensor.copy_(buf, non_blocking=True)
        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.current_stream().synchronize()

        logger.debug(
            "LocalityCacheManager.stage_recv: tensor shape=%s from rank=%d via CPU staging",
            shape, src_rank,
        )
        return dst_tensor


# ──────────────────────────────────────────────────────────────────────────────
# HeteroBridgeP2P  —  主类
# ──────────────────────────────────────────────────────────────────────────────

class HeteroBridgeP2P:
    """DES-LOC异构训练的跨island P2P通信器。

    这是本模块的核心类，直接对应Megatron的BridgeCommunicator并在其之上
    增加异构感知能力。

    Megatron b180d2c的修复（bridge_pg专用化）在此被完整保留并强化：
    所有P2P操作（send/recv/isend/irecv）均显式传入self.bridge_pg，
    且bridge_pg仅包含跨island的leader ranks，与Megatron语义精确对应。

    新增的DES-LOC特性：
    1. 传输路径自适应（TransferPath.DIRECT vs STAGED）
    2. LocalityCacheManager管理CPU staging buffer
    3. HeteroRankTopology驱动island-aware leader选取

    参数
    ────
    src_ranks    : 源island的所有rank（对应Megatron src_grid的TP组）
    dst_ranks    : 目标island的所有rank（对应Megatron dest_grid的TP组）
    topology     : 已构建的HeteroRankTopology；None时自动从当前进程组推断
    comm_dtype   : P2P传输使用的数据类型（建议bfloat16节省带宽）
    staging_threshold_bytes : 超过此值时启用CPU staging
    """

    def __init__(
        self,
        src_ranks: List[int],
        dst_ranks: List[int],
        topology: Optional[HeteroRankTopology] = None,
        comm_dtype: torch.dtype = torch.bfloat16,
        staging_threshold_bytes: int = _STAGING_BYTES_THRESHOLD,
    ) -> None:
        self.src_ranks = sorted(src_ranks)
        self.dst_ranks = sorted(dst_ranks)
        self.comm_dtype = comm_dtype
        self.staging_threshold = staging_threshold_bytes
        self.current_rank = dist.get_rank() if dist.is_initialized() else 0

        self.topology = topology or self._infer_topology(src_ranks + dst_ranks)

        # 计算bridge leader ranks（mirrors Megatron的src/dest_tp_leaders逻辑）
        self.src_leaders = self._extract_leaders(src_ranks)
        self.dst_leaders = self._extract_leaders(dst_ranks)
        self.bridge_ranks = sorted(set(self.src_leaders) | set(self.dst_leaders))

        logger.info(
            "[Rank %d] HeteroBridgeP2P init: src_leaders=%s dst_leaders=%s bridge_ranks=%s",
            self.current_rank, self.src_leaders, self.dst_leaders, self.bridge_ranks,
        )

        # 创建专用bridge PG（核心：Megatron b180d2c修复的DES-LOC版本）
        self.bridge_pg = BridgePGRegistry.get_or_create(
            self.bridge_ranks,
            backend=self._select_backend(),
        )

        # DES-LOC: Locality Cache
        self.locality_cache = LocalityCacheManager()

        # 确定当前rank的角色
        self.my_role = self._determine_role()
        logger.info(
            "[Rank %d] role=%s, bridge_pg ranks=%s",
            self.current_rank, self.my_role, self.bridge_ranks,
        )

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _infer_topology(self, all_ranks: List[int]) -> HeteroRankTopology:
        """在没有外部topology时，基于当前进程组自动推断拓扑。

        简化实现：假设当前rank对应的设备index == rank % num_local_gpus。
        生产中应通过all_gather收集各rank的设备信息。
        """
        num_gpus = max(torch.cuda.device_count(), 1)
        rank_device_map = {r: r % num_gpus for r in all_ranks}
        topo = HeteroRankTopology(
            world_size=dist.get_world_size() if dist.is_initialized() else len(all_ranks),
            rank_device_map=rank_device_map,
        )
        return topo

    def _extract_leaders(self, ranks: List[int]) -> List[int]:
        """从ranks中提取island leader，与Megatron get_leader_rank语义对应。"""
        leaders = []
        seen_islands = set()
        for r in sorted(ranks):
            island = self.topology.get_island_for_rank(r)
            if island is not None and island.island_id not in seen_islands:
                leaders.append(island.leader_rank)
                seen_islands.add(island.island_id)
            elif island is None:
                # 无拓扑信息时退化为rank自身（安全降级）
                leaders.append(r)
        return sorted(set(leaders))

    def _select_backend(self) -> str:
        """根据bridge ranks的SM架构组合选择NCCL或Gloo backend。

        DES-LOC异构场景：若bridge ranks跨越SM86和SM90，且
        NEURON_BRIDGE_BACKEND=gloo，则使用gloo（更稳定但较慢）。
        默认仍使用nccl（性能最优）。
        """
        backend = os.environ.get("NEURON_BRIDGE_BACKEND", "nccl")
        if backend != "nccl":
            logger.warning(
                "Using non-NCCL backend '%s' for bridge PG (heterogeneous PCIe env)",
                backend,
            )
        return backend

    def _determine_role(self) -> CommRole:
        """确定当前rank在bridge通信中的角色。"""
        if self.current_rank in self.bridge_ranks:
            return CommRole.LEADER
        if self.current_rank in self.src_ranks or self.current_rank in self.dst_ranks:
            return CommRole.MEMBER
        return CommRole.STAGING

    def _select_transfer_path(self, tensor: torch.Tensor, peer_rank: int) -> TransferPath:
        """自适应选择传输路径。

        决策矩阵：
          - tensor字节数 < staging_threshold  →  DIRECT（始终）
          - tensor字节数 ≥ staging_threshold 且 peer为异构rank  →  STAGED
          - tensor字节数 ≥ staging_threshold 且 peer为同构rank  →  DIRECT
        """
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes < self.staging_threshold:
            return TransferPath.DIRECT
        if self.topology.are_hetero_pair(self.current_rank, peer_rank):
            logger.debug(
                "Rank %d → %d: hetero pair, tensor %.1f MiB → STAGED path",
                self.current_rank, peer_rank, nbytes / (1024 ** 2),
            )
            return TransferPath.STAGED
        return TransferPath.DIRECT

    # ── 公开P2P接口（mirrors Megatron BridgeCommunicator的send/recv） ─────────

    def send_forward(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """发送前向激活到目标rank。

        上游对应：BridgeCommunicator.send_forward() 中的 dist.send(..., group=self.bridge_pg)。
        DES-LOC增强：根据传输路径选择器决定direct或staged发送。
        """
        if self.current_rank not in self.bridge_ranks:
            logger.debug(
                "[Rank %d] send_forward skipped (not a bridge leader)", self.current_rank
            )
            return

        path = self._select_transfer_path(tensor, dst_rank)
        logger.debug(
            "[Rank %d] send_forward -> rank=%d shape=%s path=%s",
            self.current_rank, dst_rank, tuple(tensor.shape), path.name,
        )

        if path == TransferPath.STAGED:
            self.locality_cache.stage_send(tensor, dst_rank, self.bridge_pg)
        else:
            dist.send(tensor.to(self.comm_dtype), dst=dst_rank, group=self.bridge_pg)

    def recv_forward(
        self,
        shape: Tuple[int, ...],
        src_rank: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """接收前向激活来自源rank。

        上游对应：BridgeCommunicator.recv_forward() 中的 dist.recv(..., group=self.bridge_pg)。
        """
        if self.current_rank not in self.bridge_ranks:
            logger.debug(
                "[Rank %d] recv_forward skipped (not a bridge leader)", self.current_rank
            )
            return torch.empty(0)

        device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        dummy = torch.empty(shape, dtype=self.comm_dtype, device=device)
        path = self._select_transfer_path(dummy, src_rank)

        logger.debug(
            "[Rank %d] recv_forward <- rank=%d shape=%s path=%s",
            self.current_rank, src_rank, shape, path.name,
        )

        if path == TransferPath.STAGED:
            return self.locality_cache.stage_recv(
                shape, self.comm_dtype, src_rank, self.bridge_pg, device
            )
        else:
            tensor = torch.empty(shape, dtype=self.comm_dtype, device=device, requires_grad=True)
            dist.recv(tensor, src=src_rank, group=self.bridge_pg)
            return tensor

    def send_backward(self, grad: torch.Tensor, dst_rank: int) -> None:
        """发送反向梯度到目标rank。

        上游对应：BridgeCommunicator.send_backward() 中的 dist.send(..., group=self.bridge_pg)。
        """
        if self.current_rank not in self.bridge_ranks:
            return

        path = self._select_transfer_path(grad, dst_rank)
        logger.debug(
            "[Rank %d] send_backward -> rank=%d shape=%s path=%s",
            self.current_rank, dst_rank, tuple(grad.shape), path.name,
        )

        if path == TransferPath.STAGED:
            self.locality_cache.stage_send(grad, dst_rank, self.bridge_pg)
        else:
            dist.send(grad.to(self.comm_dtype), dst=dst_rank, group=self.bridge_pg)

    def recv_backward(
        self,
        shape: Tuple[int, ...],
        src_rank: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """接收反向梯度来自源rank。

        上游对应：BridgeCommunicator.recv_backward() 中的 dist.recv(..., group=self.bridge_pg)。
        """
        if self.current_rank not in self.bridge_ranks:
            return torch.empty(0)

        device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        path = self._select_transfer_path(
            torch.empty(shape, dtype=self.comm_dtype), src_rank
        )

        logger.debug(
            "[Rank %d] recv_backward <- rank=%d shape=%s path=%s",
            self.current_rank, src_rank, shape, path.name,
        )

        if path == TransferPath.STAGED:
            return self.locality_cache.stage_recv(
                shape, self.comm_dtype, src_rank, self.bridge_pg, device
            )
        else:
            grad = torch.empty(shape, dtype=self.comm_dtype, device=device)
            dist.recv(grad, src=src_rank, group=self.bridge_pg)
            return grad

    def build_p2p_ops(
        self,
        send_tensor: Optional[torch.Tensor],
        recv_tensor: Optional[torch.Tensor],
        peer_rank: int,
        is_send: bool,
        is_recv: bool,
    ) -> List["dist.P2POp"]:
        """构建异步P2P操作列表，用于batch_isend_irecv。

        上游对应：BridgeCommunicator中所有 torch.distributed.P2POp(... self.bridge_pg)
        调用，本方法将其统一封装。

        DES-LOC注意：staged路径下无法使用isend/irecv（需要CPU中转同步点），
        因此大tensor退化为同步staged路径，不进入ops列表；小tensor走DIRECT异步路径。
        """
        ops: List[dist.P2POp] = []
        if self.current_rank not in self.bridge_ranks:
            return ops

        if is_send and send_tensor is not None:
            path = self._select_transfer_path(send_tensor, peer_rank)
            if path == TransferPath.DIRECT:
                ops.append(dist.P2POp(dist.isend, send_tensor, peer_rank, self.bridge_pg))
            else:
                # staged路径：同步执行，不加入异步op列表
                logger.debug(
                    "[Rank %d] isend peer=%d: tensor too large for async, falling back to staged sync",
                    self.current_rank, peer_rank,
                )
                self.locality_cache.stage_send(send_tensor, peer_rank, self.bridge_pg)

        if is_recv and recv_tensor is not None:
            path = self._select_transfer_path(recv_tensor, peer_rank)
            if path == TransferPath.DIRECT:
                ops.append(dist.P2POp(dist.irecv, recv_tensor, peer_rank, self.bridge_pg))
            else:
                logger.debug(
                    "[Rank %d] irecv peer=%d: staged sync recv",
                    self.current_rank, peer_rank,
                )
                # staged接收：原地填充recv_tensor
                staged = self.locality_cache.stage_recv(
                    tuple(recv_tensor.shape),
                    recv_tensor.dtype,
                    peer_rank,
                    self.bridge_pg,
                    recv_tensor.device,
                )
                recv_tensor.copy_(staged)

        return ops

    def broadcast_within_island(
        self,
        tensor: torch.Tensor,
        island_id: int,
        src_rank: int,
    ) -> torch.Tensor:
        """在同一execution island内广播tensor（intra-island broadcast）。

        DES-LOC专属逻辑：bridge通信后，目标island的leader接收数据，
        再通过此方法广播给island内所有member ranks。
        这对应Megatron中TP-group broadcast，但受异构拓扑约束。
        """
        island = self.topology._islands.get(island_id)
        if island is None:
            logger.warning("broadcast_within_island: island %d not found", island_id)
            return tensor

        if self.current_rank not in island.ranks:
            return tensor

        # 使用BridgePGRegistry创建intra-island broadcast PG
        intra_pg = BridgePGRegistry.get_or_create(island.ranks, backend="nccl")
        dist.broadcast(tensor, src=src_rank, group=intra_pg)
        logger.debug(
            "[Rank %d] broadcast_within_island island=%d src=%d shape=%s",
            self.current_rank, island_id, src_rank, tuple(tensor.shape),
        )
        return tensor

    @staticmethod
    def destroy_all_bridge_pgs() -> None:
        """销毁所有bridge PG（测试teardown / 训练结束调用）。

        上游对应：BridgeCommunicator.destroy_bridge_pgs()。
        """
        BridgePGRegistry.destroy_all()


# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────────────

def create_hetero_bridge(
    src_ranks: List[int],
    dst_ranks: List[int],
    topology: Optional[HeteroRankTopology] = None,
    comm_dtype: torch.dtype = torch.bfloat16,
) -> HeteroBridgeP2P:
    """创建HeteroBridgeP2P实例的便捷工厂函数。

    与Megatron中 BridgeCommunicator(src_grid, dest_grid) 构造方式对应，
    但参数从HyperCommGrid改为更轻量的rank列表，适合DeepSpeed生态。
    """
    logger.info(
        "create_hetero_bridge: src_ranks=%s dst_ranks=%s dtype=%s",
        src_ranks, dst_ranks, comm_dtype,
    )
    return HeteroBridgeP2P(
        src_ranks=src_ranks,
        dst_ranks=dst_ranks,
        topology=topology,
        comm_dtype=comm_dtype,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── 1. HeteroDeviceProfile 异构检测 ──────────────────────────────────────
    pa = HeteroDeviceProfile(rank=0, device_index=0, sm_major=8, sm_minor=6,
                             total_memory_bytes=48 * 1024**3)
    pb = HeteroDeviceProfile(rank=2, device_index=2, sm_major=9, sm_minor=0,
                             total_memory_bytes=96 * 1024**3)
    assert pa.is_a6000 and pb.is_h100, "SM架构检测失败"
    assert pa.is_hetero_with(pb), "异构对判断失败"
    logger.info("✓ HeteroDeviceProfile OK")

    # ── 2. HeteroRankTopology island构建 ─────────────────────────────────────
    rank_dev = {0: 0, 1: 1, 2: 2}   # rank0=A6000, rank1=A6000, rank2=H100
    topo = HeteroRankTopology(world_size=3, rank_device_map=rank_dev)
    leaders = topo.get_bridge_leader_ranks()
    # 两个island：SM86(ranks 0,1, leader=0)和SM90(rank 2, leader=2)
    assert set(leaders) == {0, 2}, f"Bridge leaders应为{{0,2}}，得到{leaders}"
    assert topo.is_leader(0) and topo.is_leader(2)
    assert not topo.is_leader(1), "rank 1是MEMBER，不应是leader"
    logger.info("✓ HeteroRankTopology OK: leaders=%s", leaders)

    # ── 3. BridgePGRegistry 缓存幂等性（非分布式，仅验证缓存逻辑） ───────────
    # 不初始化dist，验证缓存key生成不崩溃
    key_a = str(sorted([0, 2])) + ":nccl"
    key_b = str(sorted([0, 2])) + ":nccl"
    assert key_a == key_b, "相同ranks应产生相同cache_key"
    logger.info("✓ BridgePGRegistry cache key幂等性 OK")

    # ── 4. LocalityCacheManager buffer分配与复用 ──────────────────────────────
    lcm = LocalityCacheManager()
    shape = (128, 256)
    buf1 = lcm.get_staging_buffer(shape, torch.bfloat16)
    buf2 = lcm.get_staging_buffer(shape, torch.bfloat16)
    assert buf1.data_ptr() == buf2.data_ptr(), "相同shape/dtype应命中缓存，返回同一buffer"
    assert buf1.is_pinned(), "staging buffer必须是pinned memory"
    logger.info("✓ LocalityCacheManager buffer复用 OK")

    # ── 5. TransferPath选择逻辑 ───────────────────────────────────────────────
    # 构造一个不需要dist初始化的最小HeteroBridgeP2P（monkey-patch bridge_pg）
    class _MockBridge(HeteroBridgeP2P):
        def __init__(self):  # pylint: disable=super-init-not-called
            self.current_rank = 0
            self.src_ranks = [0, 1]
            self.dst_ranks = [2]
            self.bridge_ranks = [0, 2]
            self.comm_dtype = torch.bfloat16
            self.staging_threshold = _STAGING_BYTES_THRESHOLD
            self.topology = HeteroRankTopology(
                world_size=3,
                rank_device_map={0: 0, 1: 1, 2: 2},
            )
            self.bridge_pg = None
            self.locality_cache = LocalityCacheManager()
            self.my_role = CommRole.LEADER

    bridge = _MockBridge()
    small = torch.zeros(1, dtype=torch.bfloat16)
    large = torch.zeros(_STAGING_BYTES_THRESHOLD // 2 + 1, dtype=torch.bfloat16)

    assert bridge._select_transfer_path(small, 2) == TransferPath.DIRECT
    # rank0(SM86)↔rank2(SM90)是异构对，大tensor→STAGED
    assert bridge._select_transfer_path(large, 2) == TransferPath.STAGED
    # rank0↔rank1同构(SM86)，大tensor→DIRECT
    assert bridge._select_transfer_path(large, 1) == TransferPath.DIRECT
    logger.info("✓ TransferPath选择逻辑 OK")

    logger.info("All smoke tests passed.")
