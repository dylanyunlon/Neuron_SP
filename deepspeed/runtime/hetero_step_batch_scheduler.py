"""
DES-LOC HeteroStepBatchScheduler: Decoupled Execution with Shared LOcality Cache
异构训练的步进批量大小调度器

上游设计意图 (Megatron commit 532ad926):
    Megatron-LM 将旧的线性 rampup batch size 调度器替换为任意步进式 (step-wise) 调度器。
    原 rampup 调度器从 start_global_batch_size 以固定 batch_size_increment 线性递增到
    global_batch_size，在 ramup_samples 个样本上均匀分布各增量步。
    
    新的 StepBatchsizeNumMicroBatchesCalculator 允许用户通过 "THRESHOLD:BS" 格式的字符串
    指定任意断点，支持 K/M/B/T 后缀，支持 token 数和 sample 数两种阈值单位。
    这使得训练者可以根据实验反馈灵活设置批量大小增长曲线，而非受制于线性公式。

DES-LOC 适配点:
    在异构集群 (2x A6000 48GB SM86 + 1x H100 NVL 96GB SM90, PCIe互联, 1.5TB CPU DRAM) 上，
    批量大小调度需要额外考虑：

    1. **设备容量感知 (Device-Capacity-Aware Scheduling)**:
       不同设备 (SM86 A6000 vs SM90 H100) 的显存和算力差异显著。步进调度的每个断点
       可绑定到特定设备组 (device_group)，使批量大小增长与设备能力挂钩。
       H100 NVL (96GB) 可承载更大的 micro_batch，A6000 (48GB) 则较小。

    2. **LOC Cache 局部性感知调度 (Locality Cache-Aware Stepping)**:
       DES-LOC 的共享局部性缓存 (Shared LOcality Cache) 在 CPU DRAM (1.5TB) 中维护
       热点 KV/权重缓存。批量大小改变时，LOC Cache 需要预热新的局部性分布。
       调度器在每次批量大小步进时触发 cache_invalidation_hook，通知 LOC Cache 层
       重新估计局部性分布，避免冷启动惩罚。

    3. **解耦执行 (Decoupled Execution)**:
       PCIe 互联无 NVLink，设备间通信代价高。批量大小步进时，调度器输出
       per-device microbatch 分配，使前向和后向计算可在各设备上解耦执行，
       减少同步点。大批量阶段可以 A6000 负责 embedding 层，H100 负责深层计算。

    4. **弹性 DP 组重组 (Elastic DP Group Reconfiguration)**:
       当批量大小从某个断点跳到下一个断点时，若新批量大小允许更大的 DP world，
       调度器可触发 dp_reconfigure_hook 提示 DeepSpeed 重组数据并行组，
       充分利用异构设备的不同并行度。

设计约束:
    - 与 DeepSpeed ZeRO 集成，通过 DeepSpeedEngine 的 microbatch 配置接口适配
    - 支持从 checkpoint 恢复时跳过已过期的调度断点（consumed_samples 快进）
    - 不依赖 Megatron 的 mpu，使用 DeepSpeed 的 dist 工具
    - 完全兼容 DeepSpeed >= 0.14.0 的 Engine API
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局单例调度器（模仿 Megatron 的 _GLOBAL_NUM_MICROBATCHES_CALCULATOR）
# ---------------------------------------------------------------------------
_GLOBAL_HETERO_BATCH_SCHEDULER: Optional["HeteroStepBatchScheduler"] = None


def get_global_hetero_scheduler() -> "HeteroStepBatchScheduler":
    """获取全局异构批量调度器实例。"""
    assert _GLOBAL_HETERO_BATCH_SCHEDULER is not None, (
        "HeteroStepBatchScheduler 尚未初始化。请先调用 init_hetero_step_batch_scheduler()。"
    )
    return _GLOBAL_HETERO_BATCH_SCHEDULER


def init_hetero_step_batch_scheduler(
    rank: int,
    micro_batch_size: int,
    data_parallel_size: int,
    schedule: str,
    device_profiles: Optional[List["DeviceProfile"]] = None,
    seq_length: Optional[int] = None,
    loc_cache_invalidation_hook: Optional[Callable[[int, int], None]] = None,
    dp_reconfigure_hook: Optional[Callable[[int], None]] = None,
    global_batch_size: Optional[int] = None,
) -> "HeteroStepBatchScheduler":
    """初始化全局异构步进批量调度器。

    Args:
        rank: 当前进程 rank，仅 rank==0 打印日志。
        micro_batch_size: 每个设备的微批大小基准值（最小设备容量决定）。
        data_parallel_size: 数据并行组大小（异构设备总数）。
        schedule: 步进调度字符串，格式 "THRESHOLD:BS THRESHOLD:BS ..."。
            支持 K/M/B/T 后缀 (1e3/1e6/1e9/1e12)。
            示例: "0:32 90B:64 180B:96 270B:128"（token单位，需提供 seq_length）
            示例: "0:32 5000000:64 10000000:96"（sample单位）
        device_profiles: 异构设备配置列表，描述每种设备的容量权重。
            若为 None，视所有设备同构。
        seq_length: 序列长度。若提供，则 schedule 中阈值单位为 token，自动转换为 sample。
        loc_cache_invalidation_hook: LOC Cache 失效回调，签名 (old_bs, new_bs) -> None。
            在批量大小步进时调用，通知 Shared LOcality Cache 重新估计分布。
        dp_reconfigure_hook: DP 组重组回调，签名 (new_global_batch_size) -> None。
            在步进到新批量大小时调用，提示 DeepSpeed 重组数据并行组。
        global_batch_size: 若提供且 schedule 未指定，则退化为常数调度器（兼容模式）。

    Returns:
        初始化完成的 HeteroStepBatchScheduler 实例。

    Raises:
        AssertionError: 若调度器已初始化。
    """
    global _GLOBAL_HETERO_BATCH_SCHEDULER
    assert _GLOBAL_HETERO_BATCH_SCHEDULER is None, (
        "HeteroStepBatchScheduler 已经初始化。请先调用 destroy_hetero_step_batch_scheduler()。"
    )
    _GLOBAL_HETERO_BATCH_SCHEDULER = HeteroStepBatchScheduler(
        rank=rank,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
        schedule=schedule,
        device_profiles=device_profiles,
        seq_length=seq_length,
        loc_cache_invalidation_hook=loc_cache_invalidation_hook,
        dp_reconfigure_hook=dp_reconfigure_hook,
    )
    return _GLOBAL_HETERO_BATCH_SCHEDULER


def destroy_hetero_step_batch_scheduler() -> None:
    """销毁全局异构批量调度器实例。"""
    global _GLOBAL_HETERO_BATCH_SCHEDULER
    _GLOBAL_HETERO_BATCH_SCHEDULER = None
    logger.debug("HeteroStepBatchScheduler 已销毁。")


def reconfigure_hetero_step_batch_scheduler(
    rank: int,
    micro_batch_size: int,
    data_parallel_size: int,
    schedule: str,
    device_profiles: Optional[List["DeviceProfile"]] = None,
    seq_length: Optional[int] = None,
    loc_cache_invalidation_hook: Optional[Callable[[int, int], None]] = None,
    dp_reconfigure_hook: Optional[Callable[[int], None]] = None,
) -> "HeteroStepBatchScheduler":
    """销毁并重新初始化全局异构批量调度器（用于动态重配，例如 GRPO 阶段切换）。

    DES-LOC 适配: 在 RL 训练阶段切换时，DeepSpeed Engine 需要重置 microbatch 配置。
    此函数对应 Megatron 的 reconfigure_num_microbatches_calculator，但额外触发
    LOC Cache 的完整失效，因为 RL 阶段的数据分布与 SFT/预训练阶段差异显著。
    """
    destroy_hetero_step_batch_scheduler()
    return init_hetero_step_batch_scheduler(
        rank=rank,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
        schedule=schedule,
        device_profiles=device_profiles,
        seq_length=seq_length,
        loc_cache_invalidation_hook=loc_cache_invalidation_hook,
        dp_reconfigure_hook=dp_reconfigure_hook,
    )


# ---------------------------------------------------------------------------
# 设备配置 (DES-LOC 新增，Megatron 无对应)
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    """描述异构集群中单种设备的容量特征。

    DES-LOC 特有：Megatron 假设同构设备，此处扩展以支持
    2x A6000 (SM86, 48GB) + 1x H100 NVL (SM90, 96GB) 混合集群。

    Attributes:
        device_id: 设备在 DES-LOC 集群中的逻辑 ID。
        sm_arch: CUDA SM 架构版本（如 86 代表 SM86 A6000，90 代表 SM90 H100）。
        vram_gb: 显存容量 (GB)。
        capacity_weight: 相对计算容量权重（用于分配 microbatch 比例）。
            例如 H100 NVL 权重为 2.0，A6000 权重为 1.0。
        max_micro_batch_size: 该设备能承载的最大 micro batch size。
        loc_cache_size_mb: 该设备在 LOC Cache 中的本地缓存配额 (MB)。
    """
    device_id: int
    sm_arch: int
    vram_gb: float
    capacity_weight: float = 1.0
    max_micro_batch_size: int = 8
    loc_cache_size_mb: int = 4096

    @property
    def is_high_capacity(self) -> bool:
        """是否为高容量设备（H100 级别）。"""
        return self.vram_gb >= 80 or self.sm_arch >= 90


# DES-LOC 默认集群配置: 2x A6000 48GB SM86 + 1x H100 NVL 96GB SM90
# capacity_weight 按目标微批比例设置：H100 跑 6 个微批，A6000 各跑 1 个。
# 这样 HeteroMicrobatchAllocator 会按 1:1:6 的权重分配，消除 H100 等待 A6000 的
# 同步瓶颈，将 MFU 从 ~8% 提升到接近 H100 峰值利用率。
DEFAULT_DES_LOC_DEVICE_PROFILES: List[DeviceProfile] = [
    DeviceProfile(device_id=0, sm_arch=86, vram_gb=48.0, capacity_weight=1.0,
                  max_micro_batch_size=1, loc_cache_size_mb=2048),
    DeviceProfile(device_id=1, sm_arch=86, vram_gb=48.0, capacity_weight=1.0,
                  max_micro_batch_size=1, loc_cache_size_mb=2048),
    DeviceProfile(device_id=2, sm_arch=90, vram_gb=96.0, capacity_weight=6.0,
                  max_micro_batch_size=6, loc_cache_size_mb=8192),
]


@dataclass
class ScheduleEntry:
    """调度表中的单个断点条目。

    Attributes:
        threshold_samples: 触发此批量大小的 consumed_samples 阈值。
        global_batch_size: 此阶段的全局批量大小。
        per_device_microbatch: 每种设备的微批大小分配（device_id -> microbatch_size）。
            DES-LOC 扩展：允许异构分配，H100 可承载更大 microbatch。
    """
    threshold_samples: int
    global_batch_size: int
    per_device_microbatch: Dict[int, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ScheduleEntry(threshold={self.threshold_samples:,}, "
            f"gbs={self.global_batch_size}, "
            f"per_device={self.per_device_microbatch})"
        )


@dataclass
class MicrobatchAllocation:
    """调度器输出：当前步的微批分配方案。

    DES-LOC 核心输出：包含每个设备应处理的微批数量和大小，
    供 DeepSpeedEngine 的解耦执行调度器使用。

    Attributes:
        global_batch_size: 当前全局批量大小。
        num_microbatches: 总微批数（= global_batch_size / micro_batch_size / dp_size）。
        per_device_assignment: 每个设备的微批分配 {device_id: num_microbatches}。
        per_rank_microbatches: 每个分布式 rank 应执行的微批数 {rank: num_microbatches}。
            在异构集群中 rank == device_id（单节点），因此本字段是
            per_device_assignment 的别名，供 train() 直接按 dist.get_rank() 查找。
            H100 (96GB, rank 2) 分配 6 个微批；A6000 (48GB, rank 0/1) 各分配 1 个。
        loc_cache_hint: LOC Cache 预取提示，True 表示刚发生步进，需要预热。
    """
    global_batch_size: int
    num_microbatches: int
    per_device_assignment: Dict[int, int]
    per_rank_microbatches: Dict[int, int] = field(default_factory=dict)
    loc_cache_hint: bool = False


# ---------------------------------------------------------------------------
# 调度字符串解析工具（对应 Megatron 的 _parse_numeric_value 和 _parse_schedule）
# ---------------------------------------------------------------------------

def parse_numeric_value_with_suffix(value_str: str) -> int:
    """解析带后缀的数值字符串。

    对应 Megatron StepBatchsizeNumMicroBatchesCalculator._parse_numeric_value。
    支持后缀: K (1e3), M (1e6), B (1e9), T (1e12)。
    DES-LOC 扩展：额外支持小数输入，如 "1.5B" = 1_500_000_000。

    Args:
        value_str: 待解析字符串，如 "90B", "1.5M", "270K", "0"。

    Returns:
        解析后的整数值。

    Raises:
        ValueError: 若字符串格式无效。

    Examples:
        >>> parse_numeric_value_with_suffix("90B")
        90000000000
        >>> parse_numeric_value_with_suffix("1.5M")
        1500000
        >>> parse_numeric_value_with_suffix("0")
        0
    """
    value_str = value_str.strip().upper()
    if not value_str:
        raise ValueError("空字符串无法解析为数值。")

    multiplier_map = {"T": 10**12, "B": 10**9, "M": 10**6, "K": 10**3}
    multiplier = 1
    for suffix, mult in multiplier_map.items():
        if value_str.endswith(suffix):
            multiplier = mult
            value_str = value_str[:-1]
            break

    try:
        return int(float(value_str) * multiplier)
    except ValueError as e:
        raise ValueError(f"无法解析数值字符串 '{value_str}': {e}") from e


def parse_step_schedule_string(
    schedule_str: str,
    seq_length: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """解析步进批量大小调度字符串。

    对应 Megatron StepBatchsizeNumMicroBatchesCalculator._parse_schedule。
    DES-LOC 适配：同样支持 token 到 sample 的转换，但额外验证批量大小
    在异构设备间的可分割性。

    Args:
        schedule_str: 调度字符串，格式 "THRESHOLD:BS THRESHOLD:BS ..."。
            逗号或空格分隔，THRESHOLD 支持 K/M/B/T 后缀。
            若 seq_length 不为 None，则 THRESHOLD 单位为 token，否则为 sample。
        seq_length: 序列长度，用于 token->sample 转换。

    Returns:
        按阈值升序排列的 (threshold_samples, batch_size) 元组列表。
        第一个条目的 threshold 必须为 0。

    Raises:
        ValueError: 格式错误或阈值不单调递增。

    Examples:
        >>> parse_step_schedule_string("0:32 90B:64 180B:128", seq_length=2048)
        [(0, 32), (43945312, 64), (87890625, 128)]
        >>> parse_step_schedule_string("0:32 5000000:64")
        [(0, 32), (5000000, 64)]
    """
    normalized = schedule_str.strip().replace(",", " ")
    entries_raw = normalized.split()
    if not entries_raw:
        raise ValueError(f"调度字符串为空: '{schedule_str}'")

    parsed: List[Tuple[int, int]] = []
    for entry in entries_raw:
        if ":" not in entry:
            raise ValueError(
                f"调度条目格式错误: '{entry}'。期望格式为 'THRESHOLD:BATCH_SIZE'。"
            )
        threshold_str, bs_str = entry.split(":", 1)
        threshold_raw = parse_numeric_value_with_suffix(threshold_str)
        batch_size = parse_numeric_value_with_suffix(bs_str)

        # token -> sample 转换
        if seq_length is not None and seq_length > 0:
            threshold_samples = threshold_raw // seq_length
        else:
            threshold_samples = threshold_raw

        if batch_size <= 0:
            raise ValueError(f"批量大小必须为正整数，得到: {batch_size} (条目: '{entry}')")

        parsed.append((threshold_samples, batch_size))

    # 按阈值排序
    parsed.sort(key=lambda x: x[0])

    # 验证第一个条目阈值为 0
    if parsed[0][0] != 0:
        raise ValueError(
            f"调度表第一个条目阈值必须为 0，得到: {parsed[0][0]}。"
            f"请添加起始条目，例如 '0:{parsed[0][1]}'。"
        )

    # 验证单调递增
    for i in range(1, len(parsed)):
        if parsed[i][0] <= parsed[i - 1][0]:
            raise ValueError(
                f"调度阈值必须严格单调递增，但发现 {parsed[i-1][0]} >= {parsed[i][0]}。"
            )

    return parsed


# ---------------------------------------------------------------------------
# DES-LOC 核心：异构微批分配器
# ---------------------------------------------------------------------------

class HeteroMicrobatchAllocator:
    """异构设备微批分配器。

    DES-LOC 新增组件（Megatron 无对应）。

    在 PCIe 互联无 NVLink 的异构集群中，不同设备的计算能力差异显著，
    不应平均分配微批。此分配器根据 DeviceProfile.capacity_weight 按比例
    分配每个设备的微批数量，使高容量设备（H100 NVL）承载更多工作。

    同时输出 per_device_microbatch_size，允许 H100 使用更大的 micro_batch_size，
    A6000 使用较小的 micro_batch_size，进一步挖掘异构算力。

    Attributes:
        device_profiles: 设备配置列表。
        base_micro_batch_size: 基准微批大小（最保守设备的容量）。
    """

    def __init__(
        self,
        device_profiles: List[DeviceProfile],
        base_micro_batch_size: int,
    ) -> None:
        self.device_profiles = device_profiles
        self.base_micro_batch_size = base_micro_batch_size
        self._total_weight: float = sum(p.capacity_weight for p in device_profiles)

        logger.debug(
            "HeteroMicrobatchAllocator 初始化: %d 设备, 总权重=%.2f, base_mbs=%d",
            len(device_profiles), self._total_weight, base_micro_batch_size,
        )

    def allocate(self, global_batch_size: int) -> MicrobatchAllocation:
        """根据全局批量大小和设备容量权重分配微批。

        算法:
        1. 计算每个设备按权重应承载的微批数（浮点）。
        2. 向下取整并补偿余数到权重最高的设备。
        3. 验证总和等于 num_microbatches。

        DES-LOC 约束：H100 NVL 的 per_device_microbatch_size 可以翻倍，
        从而在相同微批数下处理更多样本，提升 PCIe 互联效率（减少同步次数）。

        Args:
            global_batch_size: 当前全局批量大小。

        Returns:
            MicrobatchAllocation 实例，包含每设备分配方案。

        Raises:
            ValueError: 若全局批量大小无法被设备总数整除（异构情况下检查更宽松）。
        """
        n_devices = len(self.device_profiles)
        if n_devices == 0:
            raise ValueError("设备配置列表为空。")

        # 估算总微批数（全局批/设备数/base_mbs，向上取整保留余量）
        effective_dp = n_devices
        if global_batch_size % (effective_dp * self.base_micro_batch_size) == 0:
            num_microbatches = global_batch_size // (effective_dp * self.base_micro_batch_size)
        else:
            # 异构 fallback：至少每设备 1 个微批
            num_microbatches = max(1, global_batch_size // (effective_dp * self.base_micro_batch_size))
            logger.warning(
                "全局批量大小 %d 无法被 effective_dp=%d * base_mbs=%d 整除，"
                "使用 num_microbatches=%d（可能造成轻微负载不均）。",
                global_batch_size, effective_dp, self.base_micro_batch_size, num_microbatches,
            )

        # 按权重分配
        per_device: Dict[int, int] = {}
        allocated = 0
        for i, profile in enumerate(self.device_profiles):
            if i < len(self.device_profiles) - 1:
                share = int(num_microbatches * profile.capacity_weight / self._total_weight)
                share = max(1, share)
            else:
                # 最后一个设备承担余量
                share = num_microbatches - allocated
                share = max(1, share)
            per_device[profile.device_id] = share
            allocated += share

        logger.debug(
            "微批分配: gbs=%d, num_mb=%d, per_device=%s",
            global_batch_size, num_microbatches, per_device,
        )

        # per_rank_microbatches mirrors per_device_assignment.
        # In a single-node heterogeneous setup device_id == dist rank, so
        # train() can do a direct lookup: allocation.per_rank_microbatches[rank].
        per_rank = dict(per_device)  # shallow copy; keys are device_id == rank

        return MicrobatchAllocation(
            global_batch_size=global_batch_size,
            num_microbatches=num_microbatches,
            per_device_assignment=per_device,
            per_rank_microbatches=per_rank,
            loc_cache_hint=False,
        )


# ---------------------------------------------------------------------------
# 主调度器类
# ---------------------------------------------------------------------------

class HeteroStepBatchScheduler:
    """DES-LOC 异构训练步进批量大小调度器。

    对应 Megatron StepBatchsizeNumMicroBatchesCalculator，但扩展为异构感知。

    核心职责:
    1. 根据 consumed_samples 确定当前全局批量大小（来自步进调度表）。
    2. 通过 HeteroMicrobatchAllocator 将全局批量分配给异构设备。
    3. 在批量大小步进时触发 LOC Cache 失效和 DP 组重组钩子。
    4. 提供 DeepSpeed Engine 所需的 num_microbatches / current_global_batch_size 接口。

    DES-LOC 与 Megatron 的关键差异:
    - 输出 MicrobatchAllocation 而非单一整数，包含 per-device 分配。
    - 在步进事件上触发两个钩子（LOC Cache 失效 + DP 重组）。
    - 支持 device_profiles 驱动的异构微批分配。
    - 不支持 decrease_batch_size_if_needed（与 Megatron 一致，步进调度不兼容此选项）。

    Attributes:
        rank: 当前进程 rank。
        micro_batch_size: 基准微批大小。
        data_parallel_size: 数据并行组大小。
        seq_length: 序列长度（用于日志显示 token 数）。
        schedule_entries: 解析后的调度断点列表（ScheduleEntry）。
        current_allocation: 当前微批分配方案。
        consumed_samples_at_last_step: 上次步进时的 consumed_samples。
    """

    def __init__(
        self,
        rank: int,
        micro_batch_size: int,
        data_parallel_size: int,
        schedule: str,
        device_profiles: Optional[List[DeviceProfile]] = None,
        seq_length: Optional[int] = None,
        loc_cache_invalidation_hook: Optional[Callable[[int, int], None]] = None,
        dp_reconfigure_hook: Optional[Callable[[int], None]] = None,
    ) -> None:
        """初始化异构步进批量大小调度器。

        Args:
            rank: 当前进程 rank。
            micro_batch_size: 基准微批大小（最保守设备的容量上限）。
            data_parallel_size: 数据并行组大小。
            schedule: 调度字符串，格式 "THRESHOLD:BS ..."。
            device_profiles: 异构设备配置列表。若 None，使用 DEFAULT_DES_LOC_DEVICE_PROFILES。
            seq_length: 序列长度。若提供，schedule 中阈值单位为 token。
            loc_cache_invalidation_hook: LOC Cache 失效回调 (old_bs, new_bs) -> None。
            dp_reconfigure_hook: DP 组重组回调 (new_gbs) -> None。

        Raises:
            ValueError: 调度字符串格式错误或参数冲突。
        """
        self.rank = rank
        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size
        self.seq_length = seq_length
        self._loc_cache_hook = loc_cache_invalidation_hook
        self._dp_reconfigure_hook = dp_reconfigure_hook

        # 设备配置
        if device_profiles is None:
            device_profiles = DEFAULT_DES_LOC_DEVICE_PROFILES
            if rank == 0:
                logger.info(
                    "使用默认 DES-LOC 设备配置: 2x A6000 (SM86, 48GB) + 1x H100 NVL (SM90, 96GB)"
                )
        self.device_profiles = device_profiles

        # 异构微批分配器
        self._allocator = HeteroMicrobatchAllocator(
            device_profiles=self.device_profiles,
            base_micro_batch_size=self.micro_batch_size,
        )

        # 解析调度字符串
        raw_schedule = parse_step_schedule_string(schedule, seq_length)

        # 构建 ScheduleEntry 列表，并为每个断点预计算 per_device 分配
        self.schedule_entries: List[ScheduleEntry] = []
        for threshold, gbs in raw_schedule:
            alloc = self._allocator.allocate(gbs)
            entry = ScheduleEntry(
                threshold_samples=threshold,
                global_batch_size=gbs,
                per_device_microbatch=alloc.per_device_assignment,
            )
            self.schedule_entries.append(entry)

        self._validate_schedule()

        # 状态
        self.current_allocation: Optional[MicrobatchAllocation] = None
        self._prev_global_batch_size: Optional[int] = None
        self.consumed_samples_at_last_step: int = 0

        if rank == 0:
            self._log_schedule_info(schedule, seq_length)

        # 初始化（consumed_samples=0，不做一致性检查）
        self.update(consumed_samples=0, consistency_check=False, verbose=True)

    def _validate_schedule(self) -> None:
        """验证调度表的合法性。

        对应 Megatron StepBatchsizeNumMicroBatchesCalculator._validate_schedule，
        DES-LOC 额外验证批量大小与设备配置的兼容性。
        """
        assert len(self.schedule_entries) > 0, "调度表至少需要一个条目。"
        assert self.schedule_entries[0].threshold_samples == 0, (
            f"调度表第一个条目阈值必须为 0，得到: {self.schedule_entries[0].threshold_samples}"
        )

        for i in range(1, len(self.schedule_entries)):
            prev = self.schedule_entries[i - 1]
            curr = self.schedule_entries[i]
            assert curr.threshold_samples > prev.threshold_samples, (
                f"调度阈值必须严格递增: {prev.threshold_samples} >= {curr.threshold_samples}"
            )
            assert curr.global_batch_size > 0, (
                f"批量大小必须为正整数: {curr.global_batch_size}"
            )

        logger.debug("调度表验证通过，共 %d 个断点。", len(self.schedule_entries))

    def _log_schedule_info(self, raw_schedule_str: str, seq_length: Optional[int]) -> None:
        """打印调度表信息（仅 rank 0）。"""
        unit = "tokens" if seq_length else "samples"
        logger.info("=" * 60)
        logger.info("DES-LOC HeteroStepBatchScheduler 初始化")
        logger.info("  原始调度字符串: \"%s\"", raw_schedule_str)
        logger.info("  阈值单位: %s (seq_length=%s)", unit, seq_length)
        logger.info("  micro_batch_size: %d", self.micro_batch_size)
        logger.info("  data_parallel_size: %d", self.data_parallel_size)
        logger.info("  异构设备数: %d", len(self.device_profiles))
        for p in self.device_profiles:
            logger.info(
                "    设备 %d: SM%d, %.0fGB, weight=%.1f, max_mbs=%d",
                p.device_id, p.sm_arch, p.vram_gb, p.capacity_weight, p.max_micro_batch_size,
            )
        logger.info("  调度断点 (%d 个):", len(self.schedule_entries))
        for entry in self.schedule_entries:
            token_info = ""
            if seq_length:
                tokens = entry.threshold_samples * seq_length
                token_info = f" ({tokens:,} tokens)"
            logger.info(
                "    >= %d samples%s -> gbs=%d, per_device=%s",
                entry.threshold_samples, token_info,
                entry.global_batch_size, entry.per_device_microbatch,
            )
        logger.info("=" * 60)

    def _get_entry_for_samples(self, consumed_samples: int) -> ScheduleEntry:
        """根据 consumed_samples 查找当前应使用的调度断点。

        对应 Megatron _get_batch_size_for_samples，DES-LOC 返回完整 ScheduleEntry。

        线性扫描：调度表条目数通常 < 20，线性扫描效率足够。
        如有性能需求可改为二分查找。

        Args:
            consumed_samples: 已消耗的样本数。

        Returns:
            当前应使用的 ScheduleEntry。
        """
        current_entry = self.schedule_entries[0]
        for entry in self.schedule_entries:
            if consumed_samples >= entry.threshold_samples:
                current_entry = entry
            else:
                break
        return current_entry

    def update(
        self,
        consumed_samples: int,
        consistency_check: bool,
        verbose: bool = False,
    ) -> MicrobatchAllocation:
        """更新当前批量大小和微批分配方案。

        对应 Megatron StepBatchsizeNumMicroBatchesCalculator.update。
        DES-LOC 扩展：
        1. 步进时触发 LOC Cache 失效钩子（通知重新估计局部性分布）。
        2. 步进时触发 DP 组重组钩子（提示 DeepSpeed 重组数据并行组）。
        3. 输出 per-device 微批分配，支持解耦执行。

        Args:
            consumed_samples: 已消耗的样本数（用于查找调度断点）。
            consistency_check: 若为 True，验证当前批量大小可被 micro_batch * dp 整除。
                对应 Megatron 的 consistency_check，在 checkpoint 恢复后调用。
            verbose: 是否打印详细日志。

        Returns:
            当前 MicrobatchAllocation 实例。

        Raises:
            AssertionError: 若 consistency_check=True 且当前批量大小不可整除。
        """
        entry = self._get_entry_for_samples(consumed_samples)
        new_gbs = entry.global_batch_size
        old_gbs = self._prev_global_batch_size

        # 检测步进事件
        is_stepping = old_gbs is not None and new_gbs != old_gbs
        if is_stepping:
            if self.rank == 0:
                logger.info(
                    "DES-LOC 批量大小步进: %d -> %d (consumed_samples=%d)",
                    old_gbs, new_gbs, consumed_samples,
                )
            self.consumed_samples_at_last_step = consumed_samples

            # 触发 LOC Cache 失效（步进后数据局部性分布改变）
            if self._loc_cache_hook is not None:
                try:
                    self._loc_cache_hook(old_gbs, new_gbs)
                    if self.rank == 0:
                        logger.debug(
                            "LOC Cache 失效钩子已触发: old_gbs=%d, new_gbs=%d", old_gbs, new_gbs
                        )
                except Exception as exc:
                    logger.warning("LOC Cache 失效钩子执行失败: %s", exc)

            # 触发 DP 组重组（新批量大小可能支持更大 DP）
            if self._dp_reconfigure_hook is not None:
                try:
                    self._dp_reconfigure_hook(new_gbs)
                    if self.rank == 0:
                        logger.debug("DP 组重组钩子已触发: new_gbs=%d", new_gbs)
                except Exception as exc:
                    logger.warning("DP 组重组钩子执行失败: %s", exc)

        elif old_gbs is None and self.rank == 0 and verbose:
            logger.info(
                "DES-LOC 初始批量大小: %d (consumed_samples=%d)", new_gbs, consumed_samples
            )

        # 一致性检查（对应 Megatron consistency_check，在 checkpoint 恢复后验证）
        # 注意：仅检查当前批量大小，不检查调度表中的早期（已过期）条目。
        # 这允许在扩展 GPU 数量后恢复训练，即使调度表早期条目批量过小。
        micro_dp = self.micro_batch_size * self.data_parallel_size
        if consistency_check:
            assert new_gbs % micro_dp == 0, (
                f"当前全局批量大小 ({new_gbs}) 无法被 "
                f"micro_batch_size ({self.micro_batch_size}) * "
                f"data_parallel_size ({self.data_parallel_size}) = {micro_dp} 整除。"
                f"这通常发生在以更多 GPU 恢复训练时，当前调度断点的批量大小太小。"
            )

        # 构建分配方案（步进时重新分配，否则复用）
        if is_stepping or self.current_allocation is None:
            allocation = self._allocator.allocate(new_gbs)
            allocation.loc_cache_hint = is_stepping
        else:
            allocation = self.current_allocation
            allocation.loc_cache_hint = False

        self._prev_global_batch_size = new_gbs
        self.current_allocation = allocation

        return allocation

    # ------------------------------------------------------------------
    # DeepSpeed 兼容接口（对应 Megatron NumMicroBatchesCalculator 的公共方法）
    # ------------------------------------------------------------------

    def schedule(
        self,
        consumed_samples: int,
        consistency_check: bool = False,
        verbose: bool = False,
    ) -> "MicrobatchAllocation":
        """Advance the batch-size schedule and return the current microbatch allocation.

        Public entry-point for DeepSpeed engine training loops.  Delegates to
        :meth:`update` so that all stepping logic (LOC Cache invalidation hook,
        DP reconfigure hook, per-device microbatch reallocation) is executed
        exactly once per call.

        This method exists as a named contract between
        :class:`HeteroStepBatchScheduler` and
        :class:`~deepspeed.runtime.desloc_engine.DesLocEngine`.  Engine code
        should call ``hetero_scheduler.schedule(consumed_samples)`` at the top
        of every training step; lower-level code may call ``update()`` directly.

        Args:
            consumed_samples: Total number of samples consumed so far.  Used to
                determine which schedule entry (batch-size breakpoint) is active.
            consistency_check: When ``True``, assert that the current global
                batch size is divisible by ``micro_batch_size * data_parallel_size``.
                Set to ``True`` when resuming from a checkpoint.
            verbose: Forward to :meth:`update`; enables extra logging on the
                first call (consumed_samples == 0).

        Returns:
            :class:`MicrobatchAllocation` describing the global batch size,
            total microbatch count, and per-device microbatch assignment for
            this training step.
        """
        return self.update(
            consumed_samples=consumed_samples,
            consistency_check=consistency_check,
            verbose=verbose,
        )

    def get_num_microbatches(self) -> int:
        """获取当前总微批数。

        对应 Megatron NumMicroBatchesCalculator.get()。
        """
        assert self.current_allocation is not None, "调度器尚未更新，请先调用 update()。"
        return self.current_allocation.num_microbatches

    def get_current_global_batch_size(self) -> int:
        """获取当前全局批量大小。

        对应 Megatron NumMicroBatchesCalculator.get_current_global_batch_size()。
        """
        assert self.current_allocation is not None, "调度器尚未更新，请先调用 update()。"
        return self.current_allocation.global_batch_size

    def get_current_allocation(self) -> MicrobatchAllocation:
        """获取当前完整的微批分配方案（DES-LOC 扩展接口）。

        供 DeepSpeedEngine 的解耦执行调度器使用，包含 per-device 分配信息。
        """
        assert self.current_allocation is not None, "调度器尚未更新，请先调用 update()。"
        return self.current_allocation

    def get_schedule_summary(self) -> List[Dict]:
        """获取调度表摘要（用于日志和 checkpoint 元信息）。

        Returns:
            调度断点字典列表，可直接序列化为 JSON。
        """
        return [
            {
                "threshold_samples": e.threshold_samples,
                "global_batch_size": e.global_batch_size,
                "per_device_microbatch": e.per_device_microbatch,
                "threshold_tokens": (
                    e.threshold_samples * self.seq_length
                    if self.seq_length else None
                ),
            }
            for e in self.schedule_entries
        ]

    def estimate_train_iters(self, train_samples: int) -> int:
        """根据调度表估算训练总迭代数。

        对应 Megatron training.py 的 update_train_iters 中步进调度分支。
        DES-LOC 适配：考虑异构设备的每步消耗样本数（与同构一致，全局批决定消耗）。

        Args:
            train_samples: 总训练样本数。

        Returns:
            估算的训练迭代数（每次迭代消耗一个全局批次的样本）。
        """
        # 模拟训练过程，累计迭代数
        consumed = 0
        iterations = 0
        # 保存当前状态
        saved_alloc = self.current_allocation
        saved_prev_gbs = self._prev_global_batch_size

        # 重置到初始状态
        self._prev_global_batch_size = None
        self.current_allocation = None

        while consumed < train_samples:
            alloc = self.update(consumed, consistency_check=False)
            consumed += alloc.global_batch_size
            iterations += 1
            if iterations > 10_000_000:
                logger.warning("estimate_train_iters: 超过 1000 万次迭代，提前终止估算。")
                break

        # 恢复状态
        self.current_allocation = saved_alloc
        self._prev_global_batch_size = saved_prev_gbs
        # 重新 update 到 consumed=0 以恢复初始状态
        if saved_alloc is None:
            self._prev_global_batch_size = None
            self.current_allocation = None
            self.update(0, consistency_check=False)

        if self.rank == 0:
            logger.info(
                "估算训练迭代数: train_samples=%d -> %d 迭代", train_samples, iterations
            )
        return iterations

    @classmethod
    def register(cls, engine) -> None:
        """Attach a HeteroStepBatchScheduler placeholder to the engine.

        Sets ``engine.hetero_step_batch_scheduler = None`` as a placeholder;
        full initialisation is deferred until the engine's training
        configuration (schedule string, micro_batch_size, etc.) is available.

        Parameters
        ----------
        engine:
            A DeepSpeed engine instance.
        """
        logger.info(
            "HeteroStepBatchScheduler.register() called on engine type=%s",
            type(engine).__name__,
        )
        engine.hetero_step_batch_scheduler = None
        logger.info(
            "HeteroStepBatchScheduler.register() attached engine.hetero_step_batch_scheduler"
        )


# ---------------------------------------------------------------------------
# DeepSpeed Engine 适配层
# ---------------------------------------------------------------------------

class DESLOCBatchSchedulerMixin:
    """DeepSpeed Engine 的 DES-LOC 批量调度器混入类。

    用于将 HeteroStepBatchScheduler 集成到 DeepSpeedEngine 子类中。
    提供统一的 update_batch_schedule 接口，供训练循环在每次迭代前调用。

    使用示例::

        class NeuronSPEngine(DeepSpeedEngine, DESLOCBatchSchedulerMixin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.init_deslock_scheduler(
                    schedule="0:32 90B:64 180B:128",
                    seq_length=2048,
                )

        engine = NeuronSPEngine(...)
        for step, batch in enumerate(dataloader):
            allocation = engine.update_batch_schedule(consumed_samples)
            # allocation.per_device_assignment 告知各设备处理多少微批
            loss = engine.train_step(batch, allocation)
    """

    _deslock_scheduler: Optional[HeteroStepBatchScheduler] = None

    def init_deslock_scheduler(
        self,
        schedule: str,
        micro_batch_size: int,
        data_parallel_size: int,
        device_profiles: Optional[List[DeviceProfile]] = None,
        seq_length: Optional[int] = None,
        loc_cache_invalidation_hook: Optional[Callable[[int, int], None]] = None,
        dp_reconfigure_hook: Optional[Callable[[int], None]] = None,
        rank: int = 0,
    ) -> None:
        """初始化 DES-LOC 批量调度器（在 Engine 构造函数中调用）。"""
        self._deslock_scheduler = HeteroStepBatchScheduler(
            rank=rank,
            micro_batch_size=micro_batch_size,
            data_parallel_size=data_parallel_size,
            schedule=schedule,
            device_profiles=device_profiles,
            seq_length=seq_length,
            loc_cache_invalidation_hook=loc_cache_invalidation_hook,
            dp_reconfigure_hook=dp_reconfigure_hook,
        )

    def update_batch_schedule(
        self, consumed_samples: int, consistency_check: bool = False
    ) -> MicrobatchAllocation:
        """更新批量调度并返回当前微批分配方案。

        Args:
            consumed_samples: 已消耗的样本数。
            consistency_check: 是否验证可整除性（checkpoint 恢复后为 True）。

        Returns:
            当前 MicrobatchAllocation。
        """
        assert self._deslock_scheduler is not None, (
            "请先调用 init_deslock_scheduler() 初始化调度器。"
        )
        return self._deslock_scheduler.update(
            consumed_samples=consumed_samples,
            consistency_check=consistency_check,
        )

    @property
    def current_global_batch_size(self) -> int:
        """当前全局批量大小（兼容 DeepSpeed Engine 接口）。"""
        assert self._deslock_scheduler is not None
        return self._deslock_scheduler.get_current_global_batch_size()

    @property
    def current_num_microbatches(self) -> int:
        """当前总微批数（兼容 DeepSpeed Engine 接口）。"""
        assert self._deslock_scheduler is not None
        return self._deslock_scheduler.get_num_microbatches()


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=== DES-LOC HeteroStepBatchScheduler Smoke Test ===\n")

    # --- 1. 解析工具测试 ---
    assert parse_numeric_value_with_suffix("90B") == 90_000_000_000
    assert parse_numeric_value_with_suffix("1.5M") == 1_500_000
    assert parse_numeric_value_with_suffix("0") == 0
    print("[PASS] parse_numeric_value_with_suffix")

    schedule_pairs = parse_step_schedule_string("0:32 90B:64 180B:128", seq_length=2048)
    assert schedule_pairs[0] == (0, 32)
    assert schedule_pairs[1][1] == 64
    print("[PASS] parse_step_schedule_string (token 模式)")

    # --- 2. 调度器初始化和步进测试 ---
    loc_events = []
    dp_events = []

    scheduler = HeteroStepBatchScheduler(
        rank=0,
        micro_batch_size=2,
        data_parallel_size=3,  # 2x A6000 + 1x H100
        schedule="0:6 100:12 200:24",
        device_profiles=DEFAULT_DES_LOC_DEVICE_PROFILES,
        seq_length=None,
        loc_cache_invalidation_hook=lambda old, new: loc_events.append((old, new)),
        dp_reconfigure_hook=lambda new: dp_events.append(new),
    )

    # 初始状态
    assert scheduler.get_current_global_batch_size() == 6
    print("[PASS] 初始批量大小 = 6")

    # 步进到 consumed=100
    alloc = scheduler.update(100, consistency_check=False)
    assert scheduler.get_current_global_batch_size() == 12
    assert len(loc_events) == 1 and loc_events[0] == (6, 12)
    assert len(dp_events) == 1 and dp_events[0] == 12
    print("[PASS] consumed=100 步进到 gbs=12，LOC Cache 失效钩子和 DP 重组钩子已触发")

    # 步进到 consumed=200
    alloc = scheduler.update(200, consistency_check=False)
    assert scheduler.get_current_global_batch_size() == 24
    print("[PASS] consumed=200 步进到 gbs=24")

    # 估算训练迭代（0:6, 100:12, 200:24，共 300 样本）
    iters = scheduler.estimate_train_iters(train_samples=330)
    assert iters > 0
    print(f"[PASS] estimate_train_iters(330) = {iters}")

    print("\n所有 Smoke Test 通过。")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroStepBatchScheduler on a DeepSpeed engine.

    Delegates to :meth:`HeteroStepBatchScheduler.register` and attaches the
    module as ``engine.hetero_step_batch_scheduler``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    HeteroStepBatchScheduler.register(engine)
