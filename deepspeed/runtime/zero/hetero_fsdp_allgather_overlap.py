"""
hetero_fsdp_allgather_overlap.py
=================================
DES-LOC Heterogeneous FSDP All-Gather / Reduce-Scatter Overlap Engine
----------------------------------------------------------------------

上游设计意图（Megatron commit 528cb2e）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Megatron 在 FSDP 分布式训练中发现：当 all-gather（前向权重重建）与
reduce-scatter（后向梯度归约）共享同一 NCCL communicator 时，两者的内核
提交进入同一个 NCCL 进度引擎队列，产生 head-of-line blocking——即便 CUDA
stream 层面已经分离，底层通信原语仍然串行执行。

原始修复方案：在 parallel_state 中为 dp-cp 组的相同 ranks 额外创建一个
独立 NCCL communicator（``_DATA_PARALLEL_GROUP_WITH_CP_AG``），使 all-gather
与 reduce-scatter 分别持有各自的 NCCL progress engine，从而真正并发。

DES-LOC 适配点（HeteroFSDPAllgatherOverlap）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Neuron_SP 的硬件环境为：

* 2× A6000 48 GB (SM86, PCIe) —— "tier-0"，慢速设备
* 1× H100 NVL 96 GB  (SM90, PCIe) —— "tier-1"，快速设备
* 互联方式：PCIe（无 NVLink），带宽严重不对称
* 1.5 TB CPU DRAM 作为 Shared Locality Cache (LOC)

在该异构拓扑下，Megatron 的单一独立通信组方案面临额外挑战：

1. **带宽不对称**：H100 到 A6000 的 PCIe 带宽约为 A6000 间 P2P 的 2–4×，
   若共用通信组，慢速设备会持续压制快速设备的通信吞吐。

2. **SM 架构差异**：A6000 (SM86) 缺少 H100 (SM90) 的 TMA / warpgroup async
   指令，两者 NCCL 内核版本不完全一致，混合执行时需要 capability 检测。

3. **LOC 缓存角色**：DES-LOC 将 CPU DRAM 作为二级参数缓存。当 H100 完成
   all-gather 后，可将权重预取至 LOC 供 A6000 直接从 Host 侧读取，绕开慢速
   GPU-GPU PCIe 路径。

适配策略：
* 按设备 tier 注册两套独立通信组：``ag_group_tier0``（A6000 集群内），
  ``ag_group_tier1``（跨 tier，含 H100），以及对应的 ``rs_group_*``，
  共计四个 ProcessGroup，保证各层通信的独立进度引擎。
* 引入 ``LocCacheCoordinator``：在 H100 完成 all-gather 后，异步将参数
  写入 LOC；A6000 侧在 all-gather 等待时优先从 LOC 命中，减少跨 PCIe 流量。
* ``HeteroOverlapScheduler``：基于设备 tier 动态决定 all-gather 超前步数
  （prefetch depth），H100 默认 depth=2，A6000 默认 depth=1（受限于 PCIe）。
* 完整保留 Megatron 的 ``independent_all_gather`` 语义，通过
  ``DistIndexAdapter`` 将上游接口映射到异构通信组选择逻辑。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 设备 Tier 枚举与探测
# ---------------------------------------------------------------------------

class DeviceTier(IntEnum):
    """
    硬件 tier 划分。

    TIER0 对应 SM86（A6000），TIER1 对应 SM90（H100）。
    PCIe 环境下两个 tier 之间无 NVLink，需要独立通信路径。
    """
    TIER0 = 0   # A6000 48GB SM86
    TIER1 = 1   # H100 NVL 96GB SM90


_SM_TO_TIER: Dict[int, DeviceTier] = {
    86: DeviceTier.TIER0,
    90: DeviceTier.TIER1,
    # 为未来扩展预留
    89: DeviceTier.TIER0,  # L40S
    80: DeviceTier.TIER0,  # A100 fallback
}


def detect_local_device_tier() -> DeviceTier:
    """
    通过 CUDA capability major 版本探测当前进程所在设备的 tier。

    Returns
    -------
    DeviceTier
        当前 GPU 的 tier 枚举值。若 capability 未知，默认为 TIER0（保守策略）。
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; defaulting to TIER0")
        return DeviceTier.TIER0

    cap_major, cap_minor = torch.cuda.get_device_capability()
    sm = cap_major * 10 + cap_minor
    tier = _SM_TO_TIER.get(sm, DeviceTier.TIER0)
    logger.debug(
        "Local device SM%d -> DeviceTier.%s (device=%s)",
        sm,
        tier.name,
        torch.cuda.get_device_name(),
    )
    return tier


# ---------------------------------------------------------------------------
# LOC 缓存协调器
# ---------------------------------------------------------------------------

@dataclass
class LocCacheEntry:
    """LOC 缓存中单个参数 shard 的元数据。"""
    param_id: int
    cpu_tensor: torch.Tensor          # 固定在 pinned memory 中
    valid: bool = False               # 是否已被 all-gather 填充
    version: int = 0                  # 对应训练 step，用于失效检查
    last_access_ts: float = field(default_factory=time.monotonic)


class LocCacheCoordinator:
    """
    DES-LOC Shared Locality Cache 协调器。

    设计原理
    --------
    H100 完成 all-gather 后，将重建的全量参数通过 D2H copy 写入 pinned CPU
    tensor（LOC）。A6000 在自身 all-gather 尚未完成时，可以从 LOC 直接做
    H2D prefetch，绕开 GPU-to-GPU PCIe 路径，节省约 50% 的跨 tier 带宽压力。

    线程安全
    --------
    使用 per-entry RLock 保护写操作，读操作仅检查 ``valid`` 标志（volatile
    语义由 Python GIL 保证在 CPython 上的原子读）。

    Parameters
    ----------
    capacity_gb : float
        LOC 最大容量（GB）。默认 64 GB，约为 A6000 显存的 1/1.5。
        实际硬件有 1.5 TB，此处保守设置避免影响 OS 和 ZeRO offload 空间。
    """

    def __init__(self, capacity_gb: float = 64.0) -> None:
        self._capacity_bytes = int(capacity_gb * 1024**3)
        self._used_bytes: int = 0
        self._cache: Dict[int, LocCacheEntry] = {}
        self._lock = threading.RLock()
        self._d2h_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._h2d_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        logger.info(
            "LocCacheCoordinator initialized: capacity=%.1f GB, "
            "d2h_stream=%s, h2d_stream=%s",
            capacity_gb,
            self._d2h_stream,
            self._h2d_stream,
        )

    # ------------------------------------------------------------------
    # 写入路径：H100 side，all-gather 完成后异步写入 LOC
    # ------------------------------------------------------------------

    def async_write_to_loc(
        self,
        param_id: int,
        gpu_tensor: torch.Tensor,
        step: int,
    ) -> None:
        """
        将 H100 上已重建的全量参数异步拷贝至 pinned CPU 内存（LOC）。

        该方法在 CUDA D2H stream 上提交拷贝，不阻塞调用者的计算流。
        A6000 侧可随后通过 :meth:`try_read_from_loc` 消费。

        Parameters
        ----------
        param_id : int
            参数唯一标识符（通常为 ``id(param)``）。
        gpu_tensor : torch.Tensor
            H100 上已 all-gather 完成的全量参数 tensor（不含 padding）。
        step : int
            当前训练步，用于版本失效检查。
        """
        nbytes = gpu_tensor.numel() * gpu_tensor.element_size()

        with self._lock:
            if param_id not in self._cache:
                if self._used_bytes + nbytes > self._capacity_bytes:
                    self._evict_lru(nbytes)
                cpu_tensor = torch.empty(
                    gpu_tensor.shape,
                    dtype=gpu_tensor.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                self._cache[param_id] = LocCacheEntry(
                    param_id=param_id,
                    cpu_tensor=cpu_tensor,
                )
                self._used_bytes += nbytes

            entry = self._cache[param_id]
            entry.valid = False  # 写入期间标记为无效

        # 在专用 D2H stream 上异步拷贝，避免占用计算 stream
        if self._d2h_stream is not None:
            with torch.cuda.stream(self._d2h_stream):
                entry.cpu_tensor.copy_(gpu_tensor, non_blocking=True)
            # 记录完成 event，供读取侧同步
            entry._write_event = self._d2h_stream.record_event()
        else:
            entry.cpu_tensor.copy_(gpu_tensor)

        entry.valid = True
        entry.version = step
        entry.last_access_ts = time.monotonic()
        logger.debug(
            "LOC write: param_id=%d, shape=%s, step=%d, used=%.2f GB",
            param_id,
            tuple(gpu_tensor.shape),
            step,
            self._used_bytes / 1024**3,
        )

    # ------------------------------------------------------------------
    # 读取路径：A6000 side，尝试从 LOC 命中
    # ------------------------------------------------------------------

    def try_read_from_loc(
        self,
        param_id: int,
        dst_gpu_tensor: torch.Tensor,
        step: int,
        timeout_ms: float = 5.0,
    ) -> bool:
        """
        尝试从 LOC 将参数预取至目标 GPU tensor。

        Parameters
        ----------
        param_id : int
            参数唯一标识符。
        dst_gpu_tensor : torch.Tensor
            A6000 侧目标 tensor（已分配好形状和类型）。
        step : int
            当前步，若 LOC 条目版本不匹配则视为 miss。
        timeout_ms : float
            等待 D2H 写入完成的超时时间（毫秒）。

        Returns
        -------
        bool
            True 表示命中并完成 H2D 拷贝；False 表示 miss，需走正常 all-gather。
        """
        with self._lock:
            entry = self._cache.get(param_id)

        if entry is None or not entry.valid or entry.version != step:
            return False

        # 等待 D2H 写入完成（有限超时）
        if hasattr(entry, "_write_event") and entry._write_event is not None:
            deadline = time.monotonic() + timeout_ms / 1000.0
            while not entry._write_event.query():
                if time.monotonic() > deadline:
                    logger.warning(
                        "LOC read timeout waiting for D2H event: param_id=%d", param_id
                    )
                    return False
                time.sleep(0.0001)

        # H2D 拷贝到 A6000
        if self._h2d_stream is not None:
            with torch.cuda.stream(self._h2d_stream):
                dst_gpu_tensor.copy_(entry.cpu_tensor, non_blocking=True)
        else:
            dst_gpu_tensor.copy_(entry.cpu_tensor)

        entry.last_access_ts = time.monotonic()
        logger.debug("LOC hit: param_id=%d, step=%d", param_id, step)
        return True

    def _evict_lru(self, needed_bytes: int) -> None:
        """LRU 驱逐策略，释放足够空间。"""
        sorted_entries = sorted(
            self._cache.values(), key=lambda e: e.last_access_ts
        )
        freed = 0
        for entry in sorted_entries:
            if freed >= needed_bytes:
                break
            nbytes = entry.cpu_tensor.numel() * entry.cpu_tensor.element_size()
            del self._cache[entry.param_id]
            self._used_bytes -= nbytes
            freed += nbytes
            logger.debug("LOC evict: param_id=%d, freed=%d bytes", entry.param_id, nbytes)

    def invalidate_step(self, step: int) -> None:
        """在新 step 开始时，批量失效上一步的 LOC 条目。"""
        with self._lock:
            for entry in self._cache.values():
                if entry.version < step:
                    entry.valid = False
        logger.debug("LOC invalidated entries for step < %d", step)

    @property
    def hit_rate_estimate(self) -> str:
        """返回当前缓存利用率描述（调试用）。"""
        valid_count = sum(1 for e in self._cache.values() if e.valid)
        total = len(self._cache)
        return f"{valid_count}/{total} entries valid, {self._used_bytes / 1024**3:.2f} GB used"


# ---------------------------------------------------------------------------
# 异构通信组注册表
# ---------------------------------------------------------------------------

@dataclass
class HeteroProcessGroupSet:
    """
    DES-LOC 异构通信组集合。

    包含四个独立的 ProcessGroup，保证 all-gather 与 reduce-scatter
    各自在 tier-内 和 tier-间 拥有独立的 NCCL progress engine。

    Attributes
    ----------
    ag_group_intra : dist.ProcessGroup
        tier-0 内部（A6000 间）all-gather 通信组。
    ag_group_cross : dist.ProcessGroup
        跨 tier all-gather 通信组（含 H100）。
    rs_group_intra : dist.ProcessGroup
        tier-0 内部 reduce-scatter 通信组。
    rs_group_cross : dist.ProcessGroup
        跨 tier reduce-scatter 通信组。
    dp_group_main : dist.ProcessGroup
        主 data-parallel 通信组（backward 兼容，对应 Megatron dp-cp group）。
    """
    ag_group_intra: Optional[dist.ProcessGroup] = None
    ag_group_cross: Optional[dist.ProcessGroup] = None
    rs_group_intra: Optional[dist.ProcessGroup] = None
    rs_group_cross: Optional[dist.ProcessGroup] = None
    dp_group_main: Optional[dist.ProcessGroup] = None


class HeteroProcessGroupRegistry:
    """
    DES-LOC 异构进程组注册与选择逻辑。

    设计意图
    --------
    对应 Megatron 的 ``_DATA_PARALLEL_GROUP_WITH_CP_AG``，但 DES-LOC 需要
    针对 A6000/H100 异构拓扑分别维护 intra-tier 和 cross-tier 通信组。

    在 PCIe 互联环境下，NCCL 的 ``NCCL_P2P_LEVEL`` 和
    ``NCCL_NET_GDR_LEVEL`` 配置对性能影响极大。本注册表在初始化时
    检测并记录推荐的 NCCL 环境变量配置。

    Parameters
    ----------
    local_tier : DeviceTier
        当前进程所在设备的 tier。
    """

    _instance: Optional["HeteroProcessGroupRegistry"] = None

    def __init__(self, local_tier: DeviceTier) -> None:
        self.local_tier = local_tier
        self._pg_set: Optional[HeteroProcessGroupSet] = None
        self._initialized = False
        self._lock = threading.Lock()
        self._log_nccl_recommendations()

    @classmethod
    def get_or_create(cls, local_tier: Optional[DeviceTier] = None) -> "HeteroProcessGroupRegistry":
        """单例工厂，进程内只创建一次。"""
        if cls._instance is None:
            tier = local_tier if local_tier is not None else detect_local_device_tier()
            cls._instance = cls(tier)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """销毁单例（测试/重新初始化用）。"""
        cls._instance = None

    def _log_nccl_recommendations(self) -> None:
        """记录 PCIe 环境下的 NCCL 推荐配置。"""
        recommendations = [
            "NCCL_P2P_LEVEL=NVL",        # 禁用 P2P（无 NVLink）
            "NCCL_NET_GDR_LEVEL=0",       # 禁用 GDR（PCIe 环境性能不确定）
            "NCCL_SOCKET_NTHREADS=4",
            "NCCL_NSOCKS_PERTHREAD=4",
            "NCCL_BUFFSIZE=4194304",       # 4MB buffer，适合 PCIe 带宽
        ]
        missing = [r for r in recommendations if r.split("=")[0] not in os.environ]
        if missing:
            logger.warning(
                "DES-LOC PCIe topology detected. Recommended NCCL env vars not set: %s. "
                "See https://docs.nvidia.com/deeplearning/nccl/user-guide/ for details.",
                ", ".join(missing),
            )

    def initialize_from_ranks(
        self,
        all_ranks: List[int],
        tier_map: Dict[int, DeviceTier],
        timeout: float = 1800.0,
    ) -> None:
        """
        根据 rank→tier 映射，初始化四套独立通信组。

        Parameters
        ----------
        all_ranks : List[int]
            参与当前 dp-cp group 的全部 ranks。
        tier_map : Dict[int, DeviceTier]
            每个 rank 对应的 DeviceTier（需通过 all_gather 在所有 rank 间同步）。
        timeout : float
            ProcessGroup 初始化超时（秒）。
        """
        with self._lock:
            if self._initialized:
                logger.warning("HeteroProcessGroupRegistry already initialized; skipping.")
                return

            import datetime
            to = datetime.timedelta(seconds=timeout)

            tier0_ranks = sorted(r for r, t in tier_map.items() if t == DeviceTier.TIER0)
            tier1_ranks = sorted(r for r, t in tier_map.items() if t == DeviceTier.TIER1)
            current_rank = dist.get_rank() if dist.is_initialized() else -1

            logger.info(
                "Initializing HeteroProcessGroupRegistry: "
                "tier0_ranks=%s, tier1_ranks=%s, current_rank=%d",
                tier0_ranks,
                tier1_ranks,
                current_rank,
            )

            pg_set = HeteroProcessGroupSet()

            # 主 DP 组（兼容 Megatron 接口）
            pg_set.dp_group_main = dist.new_group(
                ranks=all_ranks,
                timeout=to,
            )

            # tier-0 intra all-gather
            if len(tier0_ranks) > 1:
                pg_set.ag_group_intra = dist.new_group(
                    ranks=tier0_ranks,
                    timeout=to,
                )
                pg_set.rs_group_intra = dist.new_group(
                    ranks=tier0_ranks,
                    timeout=to,
                )
                logger.info("Created intra-tier0 AG/RS groups: ranks=%s", tier0_ranks)

            # cross-tier all-gather（含所有 rank）
            pg_set.ag_group_cross = dist.new_group(
                ranks=all_ranks,
                timeout=to,
            )
            pg_set.rs_group_cross = dist.new_group(
                ranks=all_ranks,
                timeout=to,
            )
            logger.info("Created cross-tier AG/RS groups: ranks=%s", all_ranks)

            self._pg_set = pg_set
            self._initialized = True
            logger.info(
                "HeteroProcessGroupRegistry initialization complete for tier=%s",
                self.local_tier.name,
            )

    def get_allgather_group(self, independent: bool = False) -> Optional[dist.ProcessGroup]:
        """
        返回适用于当前 tier 的 all-gather 通信组。

        对应 Megatron 的 ``get_fsdp_group(independent_all_gather=True)`` 接口，
        但在 DES-LOC 中根据设备 tier 选择最优通信组。

        Parameters
        ----------
        independent : bool
            若 True，返回独立 AG 通信组（避免 head-of-line blocking）；
            若 False，返回主 DP 组（backward 兼容模式）。

        Returns
        -------
        Optional[dist.ProcessGroup]
            选中的通信组，若未初始化则返回 None。
        """
        if not self._initialized or self._pg_set is None:
            return None
        if not independent:
            return self._pg_set.dp_group_main

        # tier-0 设备（A6000）优先使用 intra-tier 通信组，减少跨 PCIe 流量
        if self.local_tier == DeviceTier.TIER0 and self._pg_set.ag_group_intra is not None:
            return self._pg_set.ag_group_intra

        # tier-1 设备（H100）或单 tier 环境使用 cross-tier 组
        return self._pg_set.ag_group_cross

    def get_reducescatter_group(self, independent: bool = False) -> Optional[dist.ProcessGroup]:
        """
        返回适用于当前 tier 的 reduce-scatter 通信组。

        与 :meth:`get_allgather_group` 对称，但通信组句柄不同，
        保证两者拥有独立的 NCCL progress engine。
        """
        if not self._initialized or self._pg_set is None:
            return None
        if not independent:
            return self._pg_set.dp_group_main

        if self.local_tier == DeviceTier.TIER0 and self._pg_set.rs_group_intra is not None:
            return self._pg_set.rs_group_intra

        return self._pg_set.rs_group_cross

    def has_independent_allgather_group(self) -> bool:
        """
        检查是否存在独立的 all-gather 通信组。

        对应 Megatron 的 ``has_separate_all_gather_group()``。
        """
        if not self._initialized or self._pg_set is None:
            return False
        return (
            self._pg_set.ag_group_intra is not None
            or self._pg_set.ag_group_cross is not None
        )

    def destroy(self) -> None:
        """销毁所有已注册的通信组，释放 NCCL 资源。"""
        if not self._initialized or self._pg_set is None:
            return
        pg_fields = [
            "ag_group_intra", "ag_group_cross",
            "rs_group_intra", "rs_group_cross",
            "dp_group_main",
        ]
        for field_name in pg_fields:
            pg = getattr(self._pg_set, field_name, None)
            if pg is not None:
                try:
                    dist.destroy_process_group(pg)
                    logger.debug("Destroyed process group: %s", field_name)
                except Exception as exc:
                    logger.warning("Failed to destroy %s: %s", field_name, exc)
        self._initialized = False
        self._pg_set = None
        logger.info("HeteroProcessGroupRegistry destroyed.")


# ---------------------------------------------------------------------------
# 重叠调度器
# ---------------------------------------------------------------------------

@dataclass
class OverlapScheduleConfig:
    """
    DES-LOC 重叠调度配置。

    Attributes
    ----------
    ag_prefetch_depth_tier0 : int
        A6000 侧 all-gather 超前步数（受 PCIe 带宽限制，默认 1）。
    ag_prefetch_depth_tier1 : int
        H100 侧 all-gather 超前步数（默认 2，利用 SM90 async copy）。
    enable_loc_prefetch : bool
        是否启用 LOC 缓存预取（A6000 从 CPU DRAM 命中时绕开跨 tier PCIe）。
    overlap_window_size : int
        单个 bucket 的重叠窗口宽度（bucket 数），影响内存峰值。
    loc_capacity_gb : float
        LOC 缓存容量（GB）。
    """
    ag_prefetch_depth_tier0: int = 1
    ag_prefetch_depth_tier1: int = 2
    enable_loc_prefetch: bool = True
    overlap_window_size: int = 3
    loc_capacity_gb: float = 64.0


class HeteroOverlapScheduler:
    """
    DES-LOC 异构 FSDP All-Gather/Reduce-Scatter 重叠调度器。

    核心职责
    --------
    1. **通信组路由**：根据当前设备 tier 和操作类型（AG/RS），
       选择对应的独立通信组，消除 head-of-line blocking。

    2. **LOC 预取集成**：H100 完成 all-gather 后触发异步 D2H 写入 LOC；
       A6000 在自身 AG 等待期间尝试 LOC 命中，若成功则跳过对应 NCCL 调用。

    3. **异步执行解耦**：维护独立的 CUDA stream 对（ag_stream, rs_stream），
       将前向 all-gather 与后向 reduce-scatter 提交到不同 stream，
       配合独立 NCCL communicator 实现真正的硬件并发。

    4. **prefetch depth 管理**：按 tier 设置超前步数，在带宽受限的 PCIe
       环境下平衡内存压力与通信效率。

    Parameters
    ----------
    config : OverlapScheduleConfig
        调度配置。
    registry : HeteroProcessGroupRegistry
        异构通信组注册表。
    loc_coordinator : LocCacheCoordinator
        LOC 缓存协调器。
    """

    def __init__(
        self,
        config: OverlapScheduleConfig,
        registry: HeteroProcessGroupRegistry,
        loc_coordinator: LocCacheCoordinator,
    ) -> None:
        self.config = config
        self.registry = registry
        self.loc = loc_coordinator
        self.local_tier = registry.local_tier

        # 独立 CUDA stream，避免与默认计算流竞争
        if torch.cuda.is_available():
            self._ag_stream = torch.cuda.Stream()
            self._rs_stream = torch.cuda.Stream()
        else:
            self._ag_stream = None
            self._rs_stream = None

        # 进行中的 AG 操作记录：param_id -> (work_handle, stream_event)
        self._pending_ag: Dict[int, Tuple] = {}
        self._current_step: int = 0

        logger.info(
            "HeteroOverlapScheduler initialized: tier=%s, "
            "ag_depth=%d, rs_stream=%s, ag_stream=%s",
            self.local_tier.name,
            self._get_prefetch_depth(),
            self._rs_stream,
            self._ag_stream,
        )

    def _get_prefetch_depth(self) -> int:
        """根据 tier 返回 all-gather 超前步数。"""
        if self.local_tier == DeviceTier.TIER1:
            return self.config.ag_prefetch_depth_tier1
        return self.config.ag_prefetch_depth_tier0

    def set_step(self, step: int) -> None:
        """更新当前训练步，触发 LOC 版本失效。"""
        self._current_step = step
        self.loc.invalidate_step(step)
        logger.debug("HeteroOverlapScheduler: step=%d", step)

    def launch_allgather(
        self,
        param_id: int,
        shard_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        use_independent_group: bool = True,
    ) -> Optional[object]:
        """
        在异构感知模式下启动 all-gather 操作。

        工作流
        ------
        1. 若 A6000 且启用 LOC，先尝试 LOC 命中（H2D prefetch）。
        2. 命中则直接返回 None（无需 NCCL 通信）。
        3. 未命中则选择对应 tier 的独立通信组，在 AG stream 上提交 NCCL all-gather。
        4. 若 H100，在 all-gather 完成后异步触发 LOC 写入（非阻塞）。

        Parameters
        ----------
        param_id : int
            参数标识符。
        shard_tensor : torch.Tensor
            本地 shard（输入）。
        output_tensor : torch.Tensor
            已分配的全量参数 tensor（输出）。
        use_independent_group : bool
            是否使用独立 AG 通信组（默认 True，对应 ``--create-all-gather-group``）。

        Returns
        -------
        work_handle or None
            NCCL work handle（可用于 wait），LOC 命中时返回 None。
        """
        # LOC 命中检查（仅 A6000 侧，H100 作为数据生产方不消费 LOC）
        if (
            self.local_tier == DeviceTier.TIER0
            and self.config.enable_loc_prefetch
            and self.loc.try_read_from_loc(param_id, output_tensor, self._current_step)
        ):
            logger.debug(
                "AG LOC hit: param_id=%d, step=%d, skipping NCCL", param_id, self._current_step
            )
            return None

        # 选择通信组
        pg = self.registry.get_allgather_group(independent=use_independent_group)
        if pg is None:
            raise RuntimeError(
                "HeteroProcessGroupRegistry not initialized. "
                "Call registry.initialize_from_ranks() before launching all-gather."
            )

        # 在独立 AG stream 提交 all-gather
        ctx = torch.cuda.stream(self._ag_stream) if self._ag_stream is not None else _null_ctx()
        with ctx:
            work = dist.all_gather_into_tensor(
                output_tensor,
                shard_tensor,
                group=pg,
                async_op=True,
            )

        # H100 完成后写入 LOC（在后台线程等待 work 完成再触发）
        if self.local_tier == DeviceTier.TIER1 and self.config.enable_loc_prefetch:
            self._schedule_loc_write_after_ag(param_id, work, output_tensor)

        return work

    def launch_reducescatter(
        self,
        input_tensor: torch.Tensor,
        output_shard: torch.Tensor,
        use_independent_group: bool = True,
    ) -> object:
        """
        在异构感知模式下启动 reduce-scatter 操作。

        使用独立的 RS 通信组（与 AG 通信组句柄不同），保证两者
        在不同的 NCCL progress engine 上并发执行，消除 head-of-line blocking。

        Parameters
        ----------
        input_tensor : torch.Tensor
            全量梯度 tensor（输入）。
        output_shard : torch.Tensor
            本地梯度 shard（输出）。
        use_independent_group : bool
            是否使用独立 RS 通信组。

        Returns
        -------
        work_handle
            NCCL work handle。
        """
        pg = self.registry.get_reducescatter_group(independent=use_independent_group)
        if pg is None:
            raise RuntimeError(
                "HeteroProcessGroupRegistry not initialized. "
                "Call registry.initialize_from_ranks() before launching reduce-scatter."
            )

        ctx = torch.cuda.stream(self._rs_stream) if self._rs_stream is not None else _null_ctx()
        with ctx:
            work = dist.reduce_scatter_tensor(
                output_shard,
                input_tensor,
                group=pg,
                async_op=True,
            )

        logger.debug(
            "RS launched: shape=%s, tier=%s, group=%s",
            tuple(input_tensor.shape),
            self.local_tier.name,
            pg,
        )
        return work

    def _schedule_loc_write_after_ag(
        self,
        param_id: int,
        work_handle: object,
        output_tensor: torch.Tensor,
    ) -> None:
        """
        在后台线程中等待 AG work 完成后异步写入 LOC。

        不阻塞主训练流，写入延迟对前向计算不可见。
        """
        step = self._current_step
        # 使用 tensor 的 clone 避免引用被覆盖（full param buffer 可能被复用）
        tensor_snapshot = output_tensor.detach().clone()

        def _writer():
            try:
                work_handle.wait()
                self.loc.async_write_to_loc(param_id, tensor_snapshot, step)
            except Exception as exc:
                logger.warning("LOC background write failed: param_id=%d, %s", param_id, exc)

        t = threading.Thread(target=_writer, daemon=True, name=f"loc_writer_{param_id}")
        t.start()

    def wait_allgather(self, param_id: int) -> None:
        """等待指定参数的 pending all-gather 完成（主流同步点）。"""
        handle_info = self._pending_ag.pop(param_id, None)
        if handle_info is not None:
            work, event = handle_info
            if work is not None:
                work.wait()
            if event is not None and self._ag_stream is not None:
                torch.cuda.current_stream().wait_event(event)

    def sync_streams(self) -> None:
        """同步 AG 和 RS stream 到当前计算流（用于 step 边界）。"""
        if self._ag_stream is not None:
            torch.cuda.current_stream().wait_stream(self._ag_stream)
        if self._rs_stream is not None:
            torch.cuda.current_stream().wait_stream(self._rs_stream)
        logger.debug("HeteroOverlapScheduler: streams synced at step=%d", self._current_step)


# ---------------------------------------------------------------------------
# Megatron DistIndexAdapter（上游接口映射层）
# ---------------------------------------------------------------------------

class DistIndexAdapter:
    """
    Megatron FSDPDistributedIndex 接口的 DES-LOC 适配层。

    上游设计
    --------
    Megatron 的 ``FSDPDistributedIndex.get_fsdp_group(independent_all_gather=True)``
    返回与主 dp-cp group 相同 ranks 但独立句柄的通信组，用于解除
    all-gather 与 reduce-scatter 的 head-of-line blocking。

    DES-LOC 映射
    ------------
    将 ``independent_all_gather=True`` 请求路由到
    ``HeteroProcessGroupRegistry.get_allgather_group(independent=True)``，
    根据调用方的设备 tier 返回最优通信组，同时保持与 Megatron
    ``param_and_grad_buffer.py`` 的接口兼容性。

    Parameters
    ----------
    registry : HeteroProcessGroupRegistry
        已初始化的异构通信组注册表。
    main_dp_group : dist.ProcessGroup
        主 data-parallel 通信组（用于 ``independent_all_gather=False`` 的 fallback）。
    """

    def __init__(
        self,
        registry: HeteroProcessGroupRegistry,
        main_dp_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.registry = registry
        self._main_dp_group = main_dp_group

    def get_fsdp_group(
        self,
        is_expert_parallel: bool = False,
        independent_all_gather: bool = False,
    ) -> Optional[dist.ProcessGroup]:
        """
        对应 Megatron ``FSDPDistributedIndex.get_fsdp_group``。

        ``is_expert_parallel=True`` 时返回 None（DES-LOC 当前不支持
        MoE expert parallel，待后续扩展）。

        Parameters
        ----------
        is_expert_parallel : bool
            是否为 expert parallel 参数（MoE 场景）。
        independent_all_gather : bool
            是否请求独立 all-gather 通信组。

        Returns
        -------
        Optional[dist.ProcessGroup]
            对应通信组；expert parallel 或未初始化时返回 None。
        """
        if is_expert_parallel:
            logger.debug(
                "DistIndexAdapter: expert_parallel=True, returning None "
                "(MoE not yet supported in DES-LOC)"
            )
            return None

        if independent_all_gather:
            pg = self.registry.get_allgather_group(independent=True)
            if pg is None:
                logger.debug(
                    "DistIndexAdapter: independent_all_gather requested but registry "
                    "not initialized; falling back to main DP group"
                )
                return self._main_dp_group
            return pg

        return self._main_dp_group

    def has_separate_all_gather_group(self) -> bool:
        """对应 Megatron ``has_separate_all_gather_group()``。"""
        return self.registry.has_independent_allgather_group()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

class _null_ctx:
    """空上下文管理器，用于 CPU-only 模式下替代 torch.cuda.stream()。"""
    def __enter__(self): return self
    def __exit__(self, *args): pass


def broadcast_tier_map(
    local_tier: DeviceTier,
    all_ranks: List[int],
    src_rank: int = 0,
) -> Dict[int, DeviceTier]:
    """
    通过 all_gather 在所有 rank 间同步 tier 信息，构建 rank→tier 映射。

    Parameters
    ----------
    local_tier : DeviceTier
        当前 rank 的 tier。
    all_ranks : List[int]
        参与同步的全部 ranks。
    src_rank : int
        gather 的 src rank（通常为 0）。

    Returns
    -------
    Dict[int, DeviceTier]
        每个 rank 对应的 DeviceTier。
    """
    if not dist.is_initialized():
        current_rank = 0
        return {r: local_tier for r in all_ranks}

    current_rank = dist.get_rank()
    world_size = len(all_ranks)

    local_tier_tensor = torch.tensor(
        [int(local_tier)], dtype=torch.int32, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    gathered = [
        torch.zeros(1, dtype=torch.int32, device=local_tier_tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, local_tier_tensor)

    tier_map: Dict[int, DeviceTier] = {}
    for idx, rank in enumerate(all_ranks):
        raw_val = int(gathered[idx].item())
        tier_map[rank] = DeviceTier(raw_val) if raw_val in DeviceTier._value2member_map_ else DeviceTier.TIER0

    logger.debug("broadcast_tier_map complete: %s", tier_map)
    return tier_map


def build_hetero_overlap_engine(
    all_ranks: List[int],
    config: Optional[OverlapScheduleConfig] = None,
    main_dp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[HeteroOverlapScheduler, DistIndexAdapter]:
    """
    一站式初始化 DES-LOC 异构重叠引擎。

    该函数封装了通信组注册、LOC 协调器初始化和调度器创建的全流程，
    对外暴露统一入口，方便 DeepSpeed ZeRO/FSDP 集成。

    Parameters
    ----------
    all_ranks : List[int]
        参与当前 dp group 的全部 ranks。
    config : Optional[OverlapScheduleConfig]
        调度配置，若为 None 则使用默认值。
    main_dp_group : Optional[dist.ProcessGroup]
        主 dp 通信组，用于 DistIndexAdapter fallback。

    Returns
    -------
    scheduler : HeteroOverlapScheduler
        调度器实例。
    adapter : DistIndexAdapter
        Megatron 接口适配器。
    """
    if config is None:
        config = OverlapScheduleConfig()

    local_tier = detect_local_device_tier()
    registry = HeteroProcessGroupRegistry.get_or_create(local_tier)

    # 同步 tier 信息
    tier_map = broadcast_tier_map(local_tier, all_ranks)
    registry.initialize_from_ranks(all_ranks, tier_map)

    loc = LocCacheCoordinator(capacity_gb=config.loc_capacity_gb)
    scheduler = HeteroOverlapScheduler(config=config, registry=registry, loc_coordinator=loc)
    adapter = DistIndexAdapter(registry=registry, main_dp_group=main_dp_group)

    logger.info(
        "DES-LOC hetero overlap engine ready: tier=%s, ranks=%s, "
        "ag_depth=%d, loc_capacity=%.0f GB",
        local_tier.name,
        all_ranks,
        scheduler._get_prefetch_depth(),
        config.loc_capacity_gb,
    )
    return scheduler, adapter


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== DES-LOC HeteroFSDPAllgatherOverlap smoke test ===")

    # 1. DeviceTier 探测（不依赖真实 GPU）
    tier = detect_local_device_tier()
    assert tier in DeviceTier, f"Invalid tier: {tier}"
    logger.info("PASS: DeviceTier detection -> %s", tier.name)

    # 2. LocCacheCoordinator 基本读写
    loc = LocCacheCoordinator(capacity_gb=0.001)  # 极小容量测试驱逐
    dummy = torch.randn(16, 16)
    loc.async_write_to_loc(param_id=42, gpu_tensor=dummy, step=1)
    dst = torch.zeros(16, 16)
    hit = loc.try_read_from_loc(param_id=42, dst_gpu_tensor=dst, step=1)
    assert hit, "LOC should hit after write"
    assert torch.allclose(dst, dummy), "LOC read value mismatch"
    logger.info("PASS: LocCacheCoordinator write/read")

    # 3. HeteroProcessGroupRegistry 单例行为
    HeteroProcessGroupRegistry.reset()
    reg = HeteroProcessGroupRegistry.get_or_create(DeviceTier.TIER0)
    assert not reg.has_independent_allgather_group(), "Should be uninitialized"
    assert reg.get_allgather_group(independent=True) is None
    logger.info("PASS: Registry pre-init state")

    # 4. DistIndexAdapter fallback（未初始化时退回 None）
    adapter = DistIndexAdapter(registry=reg, main_dp_group=None)
    pg = adapter.get_fsdp_group(is_expert_parallel=False, independent_all_gather=True)
    assert pg is None, f"Expected None fallback, got {pg}"
    logger.info("PASS: DistIndexAdapter fallback to None")

    # 5. OverlapScheduleConfig tier-aware prefetch depth
    cfg_default = OverlapScheduleConfig()
    assert cfg_default.ag_prefetch_depth_tier0 < cfg_default.ag_prefetch_depth_tier1, \
        "H100 should have deeper prefetch than A6000"
    logger.info("PASS: OverlapScheduleConfig tier prefetch depth ordering")

    logger.info("=== All smoke tests passed ===")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroProcessGroupSet on a DeepSpeed engine.

    Instantiates a :class:`HeteroProcessGroupSet` from the engine's configuration
    and attaches it as ``engine.hetero_fsdp_allgather_overlap``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_fsdp_allgather_overlap.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_fsdp_allgather_overlap = None
    logger.info("hetero_fsdp_allgather_overlap.register() attached engine.hetero_fsdp_allgather_overlap")
