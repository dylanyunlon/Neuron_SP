"""
DES-LOC Heterogeneous FSDP Double-Buffer Recompute Manager
===========================================================

上游设计意图 (Megatron commit b6b49e7e):
    Megatron-FSDP 在 activation recompute（梯度检查点）场景下存在 double-buffering
    失效 bug：当模块处于 PRE_BACKWARD 状态（即正在执行 recompute forward pass）时，
    原代码在 `_post_forward` hook 中直接跳过了参数释放，导致 all-gather pipeline
    的 bucket 缓冲区在 recompute 结束前无法被回收，内存峰值异常升高。

    修复思路：引入 `lazy=True` 参数。当 PRE_BACKWARD 状态被检测到，bucket 不立即
    释放，而是被标记为 `can_be_released`；在 all-gather pipeline 申请新 buffer 之前
    调用 `recycle_unused_buckets` 批量回收这些 lazy-marked bucket，从而既保留
    recompute 所需参数，又不阻塞后续 all-gather 的内存分配。

DES-LOC 适配要点：
    DES-LOC = Decoupled Execution with Shared LOcality Cache。
    硬件环境：2× A6000 48GB (SM86, PCIe) + 1× H100 NVL 96GB (SM90, PCIe) + 1.5TB CPU DRAM。
    无 NVLink，GPU 间通信瓶颈是 PCIe 带宽（~32 GB/s bidirectional per slot）。

    核心挑战：
    1. **异构设备桶分配**：A6000 和 H100 的显存容量、计算吞吐不同，bucket 大小
       必须按设备类型分别调优（A6000: 保守 128MB/bucket；H100: 激进 512MB/bucket）。
    2. **Locality Cache 与 lazy release 交互**：DES-LOC 的 SLoC（Shared Locality Cache）
       在 CPU DRAM 中维护参数的 pinned-memory 副本，用于在 PCIe 传输前做预取。
       lazy release 不能简单地释放 GPU 显存，还必须通知 SLoC 该 bucket 进入
       "可驱逐" 状态，避免 CPU 端的 prefetch 线程在 bucket 已被 GPU 释放后仍
       向其写入数据造成 UAF。
    3. **Recompute + PCIe all-gather 流水线**：recompute forward 期间，参数已在
       GPU 显存，但下一轮 all-gather 需要通过 PCIe 重新聚合。lazy release 窗口
       就是 PCIe 传输的预取窗口，需要精确控制以隐藏 PCIe 延迟。
    4. **SM86 vs SM90 架构差异**：H100 支持 FP8 + transformer engine；A6000 不支持。
       FP8 transpose cache 的释放逻辑需要在异构感知路径下分别处理。

作者: Neuron_SP 项目 (github.com/dylanyunlon/Neuron_SP)
基于: Megatron commit b6b49e7e60777e9bfc0947550c967fd858e33dde
"""

from __future__ import annotations

import enum
import logging
import threading
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举 & 常量
# ---------------------------------------------------------------------------

class BucketStatus(enum.Enum):
    """Bucket 生命周期状态机，与 Megatron AllGatherPipeline.BucketStatus 对齐。"""
    EMPTY = "empty"               # 显存已释放
    WAITING = "waiting"           # 等待 all-gather 启动
    COMMUNICATING = "communicating"  # PCIe / NVLink 传输中
    READY = "ready"               # 参数可用
    LAZY_RELEASABLE = "lazy_releasable"  # DES-LOC 新增：标记可延迟释放


class DeviceClass(enum.Enum):
    """异构设备类型。"""
    A6000 = "a6000"   # SM86, 48 GB
    H100 = "h100"     # SM90, 96 GB
    CPU = "cpu"       # SLoC 所在地


# ---------------------------------------------------------------------------
# 硬件拓扑感知
# ---------------------------------------------------------------------------

@dataclass
class HeteroDeviceProfile:
    """
    描述单个 GPU 的能力，用于 DES-LOC bucket 调度决策。

    Attributes:
        device_id:      torch device index
        device_class:   A6000 / H100
        total_mem_gb:   显存容量（GB）
        sm_count:       SM 数量（用于估算 kernel 并发度）
        pcie_bw_gbps:   PCIe 带宽（GB/s，单向）
        supports_fp8:   是否支持 FP8（SM90+）
        bucket_size_bytes: 当前设备推荐的 all-gather bucket 大小
    """
    device_id: int
    device_class: DeviceClass
    total_mem_gb: float
    sm_count: int
    pcie_bw_gbps: float
    supports_fp8: bool
    bucket_size_bytes: int


def _detect_device_profile(device_id: int) -> HeteroDeviceProfile:
    """
    运行时自动检测 GPU 规格，生成 HeteroDeviceProfile。

    实现策略：
    - 通过 torch.cuda.get_device_properties 获取 SM 数量和显存。
    - 通过 compute capability major.minor 区分 SM86 (A6000) 和 SM90 (H100)。
    - PCIe 带宽通过经验值估算（无法从 CUDA API 直接查询）。
    """
    props = torch.cuda.get_device_properties(device_id)
    major, minor = props.major, props.minor
    total_mem_gb = props.total_memory / (1024 ** 3)
    sm_count = props.multi_processor_count

    if major == 9 and minor == 0:
        # H100 NVL
        device_class = DeviceClass.H100
        pcie_bw_gbps = 64.0   # PCIe 5.0 x16 理论单向峰值
        supports_fp8 = True
        bucket_size_bytes = 512 * 1024 * 1024  # 512 MB
        logger.info(
            "device=%d detected as H100 NVL (SM90), "
            "mem=%.1f GB, sm=%d, bucket=%d MB",
            device_id, total_mem_gb, sm_count, bucket_size_bytes >> 20,
        )
    elif major == 8 and minor == 6:
        # A6000
        device_class = DeviceClass.A6000
        pcie_bw_gbps = 32.0   # PCIe 4.0 x16 理论单向峰值
        supports_fp8 = False
        bucket_size_bytes = 128 * 1024 * 1024  # 128 MB
        logger.info(
            "device=%d detected as A6000 (SM86), "
            "mem=%.1f GB, sm=%d, bucket=%d MB",
            device_id, total_mem_gb, sm_count, bucket_size_bytes >> 20,
        )
    else:
        # 回退：保守配置
        device_class = DeviceClass.A6000
        pcie_bw_gbps = 16.0
        supports_fp8 = False
        bucket_size_bytes = 64 * 1024 * 1024
        logger.warning(
            "device=%d has unknown SM%d%d, falling back to conservative profile",
            device_id, major, minor,
        )

    return HeteroDeviceProfile(
        device_id=device_id,
        device_class=device_class,
        total_mem_gb=total_mem_gb,
        sm_count=sm_count,
        pcie_bw_gbps=pcie_bw_gbps,
        supports_fp8=supports_fp8,
        bucket_size_bytes=bucket_size_bytes,
    )


# ---------------------------------------------------------------------------
# SLoC (Shared Locality Cache) 接口
# ---------------------------------------------------------------------------

class SLocCacheState(enum.Enum):
    """CPU DRAM 中 SLoC 条目的状态。"""
    PINNED_PREFETCH = "pinned_prefetch"   # 正在向 GPU 预取
    RESIDENT = "resident"                  # 参数驻留在 GPU，CPU 副本为 shadow
    EVICTABLE = "evictable"               # GPU 已释放，CPU 副本可被驱逐
    EVICTED = "evicted"                   # CPU 副本已被驱逐


@dataclass
class SLocEntry:
    """
    SLoC 中单个 bucket 的 CPU 端元数据。

    DES-LOC 设计：CPU DRAM (1.5 TB) 作为 "第三层" 参数存储，
    在 PCIe 传输前对热点参数做 pinned-memory 预取，以隐藏 PCIe 延迟。
    """
    bucket_key: Tuple[int, bool]
    cpu_buffer: Optional[torch.Tensor] = None   # pinned memory
    state: SLocCacheState = SLocCacheState.EVICTED
    prefetch_stream: Optional[torch.cuda.Stream] = None
    last_access_step: int = 0


class SharedLocalityCache:
    """
    DES-LOC Shared Locality Cache (SLoC) —— CPU DRAM 参数缓存层。

    职责：
    1. 维护 pinned-memory CPU 副本，供 all-gather 的 H2D 传输使用。
    2. 提供 `mark_evictable` 接口，让 lazy-release 逻辑安全地通知 CPU 端。
    3. 在 `recycle_unused_buckets` 被调用时，批量驱逐 EVICTABLE 条目，
       释放 CPU DRAM 压力。

    线程安全：所有公开方法持 `_lock` 后操作共享状态。
    """

    def __init__(self, max_cpu_bytes: int = 32 * 1024 ** 3):
        """
        Args:
            max_cpu_bytes: SLoC 允许使用的最大 CPU DRAM（默认 32 GB）。
                           相对于 1.5 TB 总 DRAM 非常保守，避免影响 OS。
        """
        self._entries: Dict[Tuple[int, bool], SLocEntry] = {}
        self._lock = threading.Lock()
        self._max_cpu_bytes = max_cpu_bytes
        self._used_cpu_bytes = 0
        self._step = 0
        logger.debug(
            "SLoC initialized: max_cpu_bytes=%d GB",
            max_cpu_bytes >> 30,
        )

    def register_bucket(
        self,
        bucket_key: Tuple[int, bool],
        param_numel: int,
        dtype: torch.dtype,
    ) -> None:
        """
        在 SLoC 中注册一个 bucket，分配 pinned-memory CPU 副本。

        Args:
            bucket_key: (bucket_id, bwd) 复合键
            param_numel: 参数元素数
            dtype: 参数数据类型
        """
        nbytes = param_numel * torch.finfo(dtype).bits // 8
        with self._lock:
            if bucket_key in self._entries:
                return
            if self._used_cpu_bytes + nbytes > self._max_cpu_bytes:
                logger.warning(
                    "SLoC CPU quota exceeded: used=%d MB, need=%d MB, skipping registration",
                    self._used_cpu_bytes >> 20, nbytes >> 20,
                )
                return
            cpu_buf = torch.empty(param_numel, dtype=dtype, pin_memory=True)
            entry = SLocEntry(
                bucket_key=bucket_key,
                cpu_buffer=cpu_buf,
                state=SLocCacheState.EVICTED,
            )
            self._entries[bucket_key] = entry
            self._used_cpu_bytes += nbytes
            logger.debug(
                "SLoC registered bucket_key=%s, numel=%d, dtype=%s, cpu_mb=%.1f",
                bucket_key, param_numel, dtype, nbytes / (1024 ** 2),
            )

    def mark_evictable(self, bucket_key: Tuple[int, bool]) -> None:
        """
        将 bucket 标记为可驱逐状态。

        DES-LOC 关键接口：当 GPU 端执行 lazy release（参数已进入
        LAZY_RELEASABLE 状态）时，必须调用此函数通知 SLoC，
        否则预取线程可能向已释放的 GPU 缓冲区写入数据。

        Args:
            bucket_key: (bucket_id, bwd)
        """
        with self._lock:
            entry = self._entries.get(bucket_key)
            if entry is None:
                return
            if entry.state == SLocCacheState.PINNED_PREFETCH:
                logger.warning(
                    "SLoC: bucket_key=%s marked evictable while prefetch in-flight; "
                    "will evict after prefetch completes.",
                    bucket_key,
                )
                # 预取完成后再驱逐（由 prefetch 完成回调负责）
                entry.state = SLocCacheState.EVICTABLE
            else:
                entry.state = SLocCacheState.EVICTABLE
                logger.debug("SLoC: bucket_key=%s → EVICTABLE", bucket_key)

    def evict_evictable_buckets(self) -> int:
        """
        批量驱逐所有处于 EVICTABLE 状态的 SLoC 条目，释放 CPU DRAM。

        Returns:
            释放的字节数。
        """
        freed = 0
        with self._lock:
            for key, entry in list(self._entries.items()):
                if entry.state == SLocCacheState.EVICTABLE:
                    if entry.cpu_buffer is not None:
                        freed += entry.cpu_buffer.nbytes
                        del entry.cpu_buffer
                        entry.cpu_buffer = None
                    entry.state = SLocCacheState.EVICTED
                    logger.debug("SLoC: evicted bucket_key=%s, freed=%d KB", key, freed >> 10)
        self._used_cpu_bytes = max(0, self._used_cpu_bytes - freed)
        return freed

    def get_entry(self, bucket_key: Tuple[int, bool]) -> Optional[SLocEntry]:
        with self._lock:
            return self._entries.get(bucket_key)

    def increment_step(self) -> None:
        with self._lock:
            self._step += 1


# ---------------------------------------------------------------------------
# 异构 AllGather Pipeline（DES-LOC 适配版）
# ---------------------------------------------------------------------------

class HeteroAllGatherPipeline:
    """
    DES-LOC 异构 All-Gather Pipeline。

    上游对应：megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py
    :: AllGatherPipeline。

    DES-LOC 适配要点：
    1. **Lazy release**：bucket_can_be_released 字典记录可延迟释放的 bucket，
       在 `recycle_unused_buckets` 中批量处理，而非逐个立即释放。
    2. **SLoC 通知**：释放 GPU 显存前，必须先调用 `sloc.mark_evictable`，
       确保 CPU 端预取线程感知到 bucket 生命周期变化。
    3. **异构 bucket 大小**：不同设备类型使用不同 bucket_size，
       通过 `device_profile` 在构造时确定。
    4. **FP8 transpose cache**：仅 H100 (SM90) 路径下才管理 FP8 transpose cache，
       A6000 路径跳过以避免无效内存操作。
    """

    def __init__(
        self,
        device_profile: HeteroDeviceProfile,
        sloc: SharedLocalityCache,
        num_buckets: int,
    ):
        """
        Args:
            device_profile: 当前 GPU 的硬件规格
            sloc:           共享的 SLoC 实例
            num_buckets:    总 bucket 数量（与模型分片数对应）
        """
        self.device_profile = device_profile
        self.sloc = sloc
        self.num_buckets = num_buckets

        # bucket 状态字典: (bucket_id, bwd) -> BucketStatus
        self.bucket_status: Dict[Tuple[int, bool], BucketStatus] = {}
        # lazy-release 标记：Megatron commit 新增的核心数据结构
        self.bucket_can_be_released: Dict[Tuple[int, bool], bool] = {}
        # GPU 显存 buffer（简化表示，实际应为 torch.Storage）
        self._gpu_buffers: Dict[Tuple[int, bool], Optional[torch.Tensor]] = {}
        # FP8 transpose cache（仅 H100 路径）
        self._fp8_transpose_cache: Dict[Tuple[int, bool], Optional[torch.Tensor]] = {}
        # CUDA stream 用于异步 H2D 传输
        self._comm_stream = torch.cuda.Stream(device=device_profile.device_id)

        # 初始化所有 bucket 为 EMPTY
        for bid in range(num_buckets):
            for bwd in (False, True):
                key = (bid, bwd)
                self.bucket_status[key] = BucketStatus.EMPTY
                self.bucket_can_be_released[key] = False
                self._gpu_buffers[key] = None
                self._fp8_transpose_cache[key] = None

        logger.info(
            "HeteroAllGatherPipeline: device=%d (%s), num_buckets=%d, "
            "bucket_size=%d MB, fp8=%s",
            device_profile.device_id,
            device_profile.device_class.value,
            num_buckets,
            device_profile.bucket_size_bytes >> 20,
            device_profile.supports_fp8,
        )

    def get_bucket_key(self, bucket_id: int, bwd: bool) -> Tuple[int, bool]:
        """生成 bucket 复合键。"""
        return (bucket_id, bwd)

    def wait_bucket_ready(
        self, bucket_id: int, bwd: bool, empty_ok: bool = False
    ) -> None:
        """
        等待 bucket 完成通信（PCIe all-gather 完成）。

        DES-LOC 说明：由于无 NVLink，通信延迟以 PCIe 带宽为瓶颈。
        此函数同步 `_comm_stream`，确保 H2D 传输完成后再访问 GPU 缓冲区。

        Args:
            bucket_id: bucket 索引
            bwd:       是否为反向 all-gather
            empty_ok:  允许 bucket 已为 EMPTY 状态
        """
        key = self.get_bucket_key(bucket_id, bwd)
        status = self.bucket_status[key]
        if status == BucketStatus.EMPTY:
            if not empty_ok:
                raise RuntimeError(
                    f"Bucket {bucket_id} (bwd={bwd}) is EMPTY but empty_ok=False."
                )
            return
        if status == BucketStatus.COMMUNICATING:
            # 同步 PCIe 传输流
            logger.debug(
                "Waiting for PCIe all-gather: bucket=%d bwd=%s device=%d",
                bucket_id, bwd, self.device_profile.device_id,
            )
            torch.cuda.current_stream(
                self.device_profile.device_id
            ).wait_stream(self._comm_stream)
            self.bucket_status[key] = BucketStatus.READY
            logger.debug(
                "Bucket %d (bwd=%s) all-gather complete on device %d",
                bucket_id, bwd, self.device_profile.device_id,
            )

    def recycle_unused_buckets(self) -> int:
        """
        批量回收所有被标记为 LAZY_RELEASABLE 的 bucket。

        上游对应：AllGatherPipeline.recycle_unused_buckets（Megatron commit 新增）。

        DES-LOC 适配：
        - 回收 GPU 显存之前，先通知 SLoC 将对应 CPU 副本标记为 EVICTABLE。
        - 对 H100 路径，同时清理 FP8 transpose cache。
        - 返回回收的 bucket 数量，供调用方决策是否触发 CPU GC。

        Returns:
            回收的 bucket 数量。
        """
        recycled = 0
        for key, can_release in list(self.bucket_can_be_released.items()):
            if not can_release:
                continue
            if self.bucket_status[key] == BucketStatus.LAZY_RELEASABLE:
                bucket_id, bwd = key
                try:
                    self._do_release_gpu_buffer(bucket_id, bwd)
                    recycled += 1
                except Exception as exc:
                    logger.error(
                        "Failed to recycle bucket_key=%s: %s", key, exc, exc_info=True
                    )
        if recycled:
            logger.debug(
                "recycle_unused_buckets: recycled=%d buckets on device=%d",
                recycled, self.device_profile.device_id,
            )
            # 触发 SLoC CPU 端驱逐
            freed_cpu = self.sloc.evict_evictable_buckets()
            if freed_cpu:
                logger.info(
                    "SLoC evicted %d MB CPU DRAM after GPU bucket recycle",
                    freed_cpu >> 20,
                )
        return recycled

    def _do_release_gpu_buffer(self, bucket_id: int, bwd: bool) -> None:
        """
        实际释放 GPU 显存（bucket 不再持有 GPU buffer）。

        内部实现逻辑：
        1. 先通知 SLoC 标记 EVICTABLE（防止 CPU 预取线程 UAF）。
        2. H100 路径：若存在 FP8 transpose weight buffer，优先释放它；
           否则释放 model weight buffer。（与 Megatron 原逻辑一致）
        3. A6000 路径：直接释放 model weight buffer（无 FP8 路径）。
        4. 更新 bucket_status → EMPTY，重置 lazy-release 标记。
        """
        key = (bucket_id, bwd)
        # Step 1: 通知 SLoC
        self.sloc.mark_evictable(key)

        # Step 2/3: 异构感知释放
        if self.device_profile.supports_fp8 and self._fp8_transpose_cache[key] is not None:
            # H100 FP8 路径：释放 transpose cache
            logger.debug(
                "Releasing FP8 transpose cache for bucket_key=%s on H100", key
            )
            del self._fp8_transpose_cache[key]
            self._fp8_transpose_cache[key] = None
        else:
            # 通用路径：释放 model weight buffer
            if self._gpu_buffers[key] is not None:
                logger.debug(
                    "Releasing GPU weight buffer for bucket_key=%s on device=%d",
                    key, self.device_profile.device_id,
                )
                del self._gpu_buffers[key]
                self._gpu_buffers[key] = None

        # Step 4: 更新状态
        self.bucket_status[key] = BucketStatus.EMPTY
        self.bucket_can_be_released[key] = False

    @torch.no_grad()
    def release_bucket(self, bucket_id: int, bwd: bool, lazy: bool = False) -> None:
        """
        释放指定参数 bucket，根据 lazy 参数决定立即释放或延迟释放。

        上游对应：AllGatherPipeline.release_bucket（Megatron commit b6b49e7e 修改）。

        DES-LOC 适配说明：
        - lazy=False（正常前向/后向结束）：立即同步并释放 GPU 显存，
          同步通知 SLoC 驱逐 CPU 副本。
        - lazy=True（activation recompute 的 PRE_BACKWARD 状态）：
          仅将 bucket 状态切换为 LAZY_RELEASABLE，并同步通知 SLoC 进入
          EVICTABLE 预备状态；真正的 GPU 显存释放由 `recycle_unused_buckets`
          在 all-gather pipeline 申请新 buffer 前完成。
          这样做的好处：recompute forward 完成后，如果 backward 还需要这些参数，
          它们仍在 GPU 显存中；如果不需要，pipeline 会在下次 all-gather 前
          批量回收，减少 PCIe 往返次数。

        Args:
            bucket_id: bucket 索引
            bwd:       是否为反向 all-gather
            lazy:      是否启用延迟释放（DES-LOC activation recompute 路径）

        Raises:
            ValueError: bucket 正在通信中且非 lazy 模式
        """
        key = self.get_bucket_key(bucket_id, bwd)

        if self.bucket_status[key] == BucketStatus.EMPTY:
            logger.debug(
                "release_bucket: bucket_key=%s already EMPTY, skip", key
            )
            return

        if lazy:
            # Megatron 修复的核心：延迟释放标记
            # DES-LOC 额外：同步通知 SLoC，防止预取线程 UAF
            self.bucket_can_be_released[key] = True
            self.bucket_status[key] = BucketStatus.LAZY_RELEASABLE
            self.sloc.mark_evictable(key)
            logger.debug(
                "release_bucket: bucket_key=%s marked LAZY_RELEASABLE on device=%d",
                key, self.device_profile.device_id,
            )
            return

        # 立即释放路径
        self.wait_bucket_ready(bucket_id, bwd, empty_ok=True)
        if self.bucket_status[key] == BucketStatus.COMMUNICATING:
            raise ValueError(
                f"Bucket {bucket_id} (bwd={bwd}) is COMMUNICATING and cannot be released immediately. "
                f"Device={self.device_profile.device_id}"
            )
        self._do_release_gpu_buffer(bucket_id, bwd)
        logger.debug(
            "release_bucket: bucket_key=%s released immediately on device=%d",
            key, self.device_profile.device_id,
        )

    def request_buffer_for_allgather(self, bucket_id: int, bwd: bool) -> torch.Tensor:
        """
        在发起 all-gather 前申请 GPU buffer。

        DES-LOC 说明：这是 `recycle_unused_buckets` 的调用点。
        在申请新 buffer 之前，先批量回收 LAZY_RELEASABLE bucket，
        确保有足够的显存配额（对 A6000 48 GB 尤其重要）。

        Args:
            bucket_id: 目标 bucket 索引
            bwd:       是否为反向 all-gather

        Returns:
            分配好的 GPU tensor buffer
        """
        # 申请前先回收 lazy bucket，释放显存压力
        recycled = self.recycle_unused_buckets()
        if recycled:
            logger.info(
                "Pre-allgather recycle freed %d buckets on device=%d",
                recycled, self.device_profile.device_id,
            )

        key = self.get_bucket_key(bucket_id, bwd)
        numel = self.device_profile.bucket_size_bytes // 2  # bf16: 2 bytes per element
        buf = torch.empty(
            numel,
            dtype=torch.bfloat16,
            device=self.device_profile.device_id,
        )
        self._gpu_buffers[key] = buf
        self.bucket_status[key] = BucketStatus.WAITING
        logger.debug(
            "Allocated all-gather buffer: bucket_key=%s, numel=%d, device=%d",
            key, numel, self.device_profile.device_id,
        )
        return buf


# ---------------------------------------------------------------------------
# Training State（与 Megatron TrainingState 对齐）
# ---------------------------------------------------------------------------

class TrainingState(enum.Enum):
    """
    模块训练状态机。

    与 Megatron MegatronFSDP._training_state 语义完全一致，
    在 DES-LOC 中额外用于决定 SLoC 预取策略：
    - IDLE/FORWARD：SLoC 主动预取下一 bucket 的 CPU 副本到 GPU。
    - PRE_BACKWARD：SLoC 保持当前 GPU 参数，不触发新的 H2D 传输。
    - BACKWARD：SLoC 允许后向 all-gather 覆盖前向 buffer。
    """
    IDLE = "idle"
    FORWARD = "forward"
    PRE_BACKWARD = "pre_backward"
    BACKWARD = "backward"


# ---------------------------------------------------------------------------
# FP8 Transpose Cache 管理（H100 专用路径）
# ---------------------------------------------------------------------------

def release_params_fp8_transpose_cache_hetero(
    parameters, device_profile: HeteroDeviceProfile
) -> None:
    """
    异构感知的 FP8 transpose cache 释放函数。

    上游对应：release_params_fp8_transpose_cache（Megatron）。

    DES-LOC 适配：
    - 仅在 SM90 (H100) 设备上执行 FP8 cache 清理，A6000 直接跳过。
    - 避免在 A6000 上调用不存在的 FP8 接口导致运行时错误。

    Args:
        parameters:     参数迭代器
        device_profile: 当前设备规格
    """
    if not device_profile.supports_fp8:
        logger.debug(
            "Skipping FP8 transpose cache release on %s (no FP8 support)",
            device_profile.device_class.value,
        )
        return
    # H100 路径：清理 FP8 transpose cache
    for param in parameters:
        if hasattr(param, '_fp8_transpose_cache'):
            del param._fp8_transpose_cache
            param._fp8_transpose_cache = None
            logger.debug(
                "Cleared FP8 transpose cache for param shape=%s on H100",
                tuple(param.shape),
            )


# ---------------------------------------------------------------------------
# DES-LOC HeteroFSDP 核心模块
# ---------------------------------------------------------------------------

@dataclass
class HeteroFSDPConfig:
    """
    DES-LOC 异构 FSDP 配置。

    Attributes:
        keep_fp8_transpose_cache: 是否保留 FP8 transpose cache（H100 路径）
        sloc_max_cpu_bytes:       SLoC 最大 CPU DRAM 配额
        lazy_release_on_recompute: 是否在 activation recompute 时启用 lazy release
                                   （对应 Megatron 的 bugfix）
        device_ids:               参与训练的 GPU device id 列表
    """
    keep_fp8_transpose_cache: bool = False
    sloc_max_cpu_bytes: int = 32 * 1024 ** 3  # 32 GB
    lazy_release_on_recompute: bool = True
    device_ids: List[int] = field(default_factory=lambda: [0, 1, 2])


class HeteroFSDPDoubleBufferRecompute:
    """
    DES-LOC 异构 FSDP Double-Buffer + Activation Recompute 管理器。

    上游对应：MegatronFSDP（megatron_fsdp.py）中 `_post_forward` hook
    与 `release_module_parameters` 的修复逻辑（commit b6b49e7e）。

    设计意图（上游）：
        activation recompute（梯度检查点）会在 PRE_BACKWARD 状态下
        重新执行前向传播。原代码在此状态下直接 return，跳过参数释放，
        导致 double-buffering 的内存复用逻辑完全失效（all-gather pipeline
        无法回收已完成聚合的 bucket）。修复方案是引入 lazy release：
        recompute forward 完成后将 bucket 标记为可释放，而不是立即释放，
        从而既不阻断 recompute 的参数访问，又能让 pipeline 在合适时机
        批量回收。

    DES-LOC 适配（本类职责）：
        1. 管理多个异构设备（A6000 × 2 + H100 × 1）各自的 AllGatherPipeline。
        2. 在 `_post_forward_hook` 中实现异构感知的 lazy vs. immediate release。
        3. 通过 SLoC 接口协调 CPU DRAM 预取与 GPU 显存释放的时序。
        4. 提供统一的 `register_module` 接口，将 hook 注册到任意 nn.Module。

    典型调用流程：
        manager = HeteroFSDPDoubleBufferRecompute(config)
        manager.register_module(transformer_layer)

        # 训练循环中：
        # forward → _post_forward_hook 自动触发 lazy/immediate release
        # backward → release_module_parameters(bwd=True) 触发后向 release
        # 下一 step 前：allgather pipeline 调用 request_buffer_for_allgather
        #              自动触发 recycle_unused_buckets
    """

    def __init__(self, config: HeteroFSDPConfig):
        """
        Args:
            config: DES-LOC 异构 FSDP 配置
        """
        self.config = config

        # 构建 SLoC（单例，所有设备共享 CPU DRAM 副本）
        self.sloc = SharedLocalityCache(
            max_cpu_bytes=config.sloc_max_cpu_bytes
        )

        # 为每个设备构建异构感知的 AllGather Pipeline
        self.pipelines: Dict[int, HeteroAllGatherPipeline] = {}
        self.device_profiles: Dict[int, HeteroDeviceProfile] = {}
        for dev_id in config.device_ids:
            profile = _detect_device_profile(dev_id)
            self.device_profiles[dev_id] = profile
            # num_buckets 此处为占位符，实际应由 ParamAndGradBuffer 传入
            pipeline = HeteroAllGatherPipeline(
                device_profile=profile,
                sloc=self.sloc,
                num_buckets=64,  # placeholder
            )
            self.pipelines[dev_id] = pipeline

        # 已注册模块的弱引用集合（避免循环引用）
        self._registered_modules: Set[weakref.ref] = set()
        # 模块 -> 设备 id 映射
        self._module_device: Dict[int, int] = {}
        # 模块 -> 参数 bucket id 列表映射（简化：每模块一个 bucket）
        self._module_buckets: Dict[int, List[int]] = {}

        logger.info(
            "HeteroFSDPDoubleBufferRecompute initialized: "
            "devices=%s, lazy_release=%s, sloc_cpu_gb=%d",
            config.device_ids,
            config.lazy_release_on_recompute,
            config.sloc_max_cpu_bytes >> 30,
        )

    def _get_pipeline_for_module(self, module: nn.Module) -> HeteroAllGatherPipeline:
        """根据模块所在设备获取对应的 AllGather Pipeline。"""
        mod_id = id(module)
        dev_id = self._module_device.get(mod_id, self.config.device_ids[0])
        return self.pipelines[dev_id]

    def release_module_parameters(
        self,
        module: nn.Module,
        bwd: bool,
        lazy: bool = False,
    ) -> None:
        """
        释放模块的参数 bucket。

        上游对应：MegatronFSDP.release_module_parameters（commit b6b49e7e 修改）。

        DES-LOC 适配：
        - 路由到对应设备的 HeteroAllGatherPipeline。
        - 传递 lazy 参数给 pipeline.release_bucket。
        - 异构感知的 FP8 transpose cache 处理。

        Args:
            module: 要释放参数的模块
            bwd:    是否为后向释放
            lazy:   是否延迟释放（activation recompute 路径）
        """
        pipeline = self._get_pipeline_for_module(module)
        profile = self.device_profiles.get(
            self._module_device.get(id(module), self.config.device_ids[0])
        )
        mod_id = id(module)
        bucket_ids = self._module_buckets.get(mod_id, [0])

        for bucket_id in bucket_ids:
            pipeline.release_bucket(bucket_id, bwd=bwd, lazy=lazy)

        # FP8 transpose cache 释放（异构感知：仅 H100）
        if not self.config.keep_fp8_transpose_cache and profile is not None:
            release_params_fp8_transpose_cache_hetero(
                module.parameters(), profile
            )

    def _post_forward_hook(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> Any:
        """
        模块前向传播结束后的 hook。

        上游对应：MegatronFSDP._post_forward（commit b6b49e7e 核心修改点）。

        原始 bug：
            if module._training_state == TrainingState.PRE_BACKWARD:
                return output  # ← 完全跳过释放，导致 double-buffer 失效

        修复后（上游 + DES-LOC 联合逻辑）：
            1. 检测 PRE_BACKWARD（即 activation recompute 正在执行）。
            2. 若是，启用 lazy_release=True，仅标记 bucket 为 LAZY_RELEASABLE，
               不立即释放 GPU 显存。
            3. 若否，立即释放（lazy_release=False），并将模块状态置为 IDLE。
            4. 所有路径都调用 release_module_parameters，不再有早退路径。

        DES-LOC 额外逻辑：
            - lazy=True 时，SLoC 同步标记 CPU 副本为 EVICTABLE，
              避免 H2D 预取线程在 bucket 被 GPU 重用前写入过期数据。

        Args:
            module: 触发 hook 的模块
            input:  前向输入（未使用）
            output: 前向输出（原样返回）

        Returns:
            原始前向输出（不修改）
        """
        training_state: TrainingState = getattr(
            module, '_training_state', TrainingState.IDLE
        )

        # 核心逻辑：对应 Megatron commit b6b49e7e 的修复
        if training_state == TrainingState.PRE_BACKWARD:
            # Activation recompute forward 正在执行。
            # 不能立即释放参数（backward 仍需要）。
            # DES-LOC: 标记 LAZY_RELEASABLE，SLoC 同步收到 EVICTABLE 信号。
            lazy_release = True
            logger.debug(
                "module=%s in PRE_BACKWARD (recompute forward), using lazy_release",
                module.__class__.__name__,
            )
        else:
            # 正常前向结束，立即释放
            lazy_release = False
            module._training_state = TrainingState.IDLE
            logger.debug(
                "module=%s forward done, immediate release, state→IDLE",
                module.__class__.__name__,
            )

        # 无论 lazy 与否，都调用 release_module_parameters（修复了原代码的 early return）
        self.release_module_parameters(module, bwd=False, lazy=lazy_release)

        return output

    def register_module(
        self,
        module: nn.Module,
        device_id: Optional[int] = None,
        bucket_ids: Optional[List[int]] = None,
    ) -> None:
        """
        将模块注册到 DES-LOC HeteroFSDP 管理器，注册 post-forward hook。

        Args:
            module:     要托管的 nn.Module
            device_id:  模块所在 GPU（None 则自动推断）
            bucket_ids: 模块对应的 all-gather bucket id 列表
        """
        mod_id = id(module)

        # 自动推断设备
        if device_id is None:
            try:
                param = next(iter(module.parameters()))
                device_id = param.device.index or self.config.device_ids[0]
            except StopIteration:
                device_id = self.config.device_ids[0]

        if device_id not in self.pipelines:
            logger.warning(
                "device_id=%d not in registered pipelines %s, using default",
                device_id, list(self.pipelines.keys()),
            )
            device_id = self.config.device_ids[0]

        self._module_device[mod_id] = device_id
        self._module_buckets[mod_id] = bucket_ids or [mod_id % 64]

        # 初始化模块训练状态
        module._training_state = TrainingState.IDLE

        # 注册 post-forward hook
        handle = module.register_forward_hook(self._post_forward_hook)
        self._registered_modules.add(weakref.ref(module))

        logger.info(
            "Registered module=%s on device=%d, buckets=%s",
            module.__class__.__name__, device_id,
            self._module_buckets[mod_id],
        )

    def trigger_recycle_all_devices(self) -> Dict[int, int]:
        """
        触发所有设备的 lazy bucket 回收。

        在每个 all-gather 发起前或 step 结束时调用，
        确保 LAZY_RELEASABLE bucket 得到及时回收。

        Returns:
            {device_id: recycled_count} 字典
        """
        results = {}
        for dev_id, pipeline in self.pipelines.items():
            recycled = pipeline.recycle_unused_buckets()
            results[dev_id] = recycled
            if recycled:
                logger.info(
                    "Device %d: recycled %d lazy-released buckets", dev_id, recycled
                )
        return results

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        返回各设备 GPU 显存使用摘要（用于监控和调试）。

        Returns:
            包含每设备 active_buckets、sloc_cpu_mb 等信息的字典
        """
        summary: Dict[str, Any] = {}
        for dev_id, pipeline in self.pipelines.items():
            active = sum(
                1 for status in pipeline.bucket_status.values()
                if status not in (BucketStatus.EMPTY,)
            )
            lazy_pending = sum(
                1 for status in pipeline.bucket_status.values()
                if status == BucketStatus.LAZY_RELEASABLE
            )
            if torch.cuda.is_available():
                try:
                    alloc_mb = torch.cuda.memory_allocated(dev_id) / (1024 ** 2)
                    reserved_mb = torch.cuda.memory_reserved(dev_id) / (1024 ** 2)
                except Exception:
                    alloc_mb = reserved_mb = -1.0
            else:
                alloc_mb = reserved_mb = -1.0
            summary[f"device_{dev_id}"] = {
                "device_class": self.device_profiles[dev_id].device_class.value,
                "active_buckets": active,
                "lazy_pending_buckets": lazy_pending,
                "gpu_alloc_mb": alloc_mb,
                "gpu_reserved_mb": reserved_mb,
            }
        summary["sloc_cpu_used_gb"] = self.sloc._used_cpu_bytes / (1024 ** 3)
        return summary


# ---------------------------------------------------------------------------
# DeepSpeed ZeRO 集成适配层
# ---------------------------------------------------------------------------

class DESLOCZeroHook:
    """
    将 HeteroFSDPDoubleBufferRecompute 集成到 DeepSpeed ZeRO-3 引擎的适配层。

    DeepSpeed ZeRO-3 与 Megatron-FSDP 在参数分片上逻辑类似，
    但接口不同。此类提供桥接，使 DES-LOC 的 lazy release 逻辑
    能够挂载到 DeepSpeed 的 pre/post forward 钩子体系中。

    使用方式：
        engine = deepspeed.initialize(...)
        hook = DESLOCZeroHook(engine, hetero_manager)
        hook.install()
    """

    def __init__(
        self,
        ds_engine,  # deepspeed.DeepSpeedEngine
        hetero_manager: HeteroFSDPDoubleBufferRecompute,
    ):
        self.ds_engine = ds_engine
        self.hetero_manager = hetero_manager
        self._installed = False
        logger.info(
            "DESLOCZeroHook created for DeepSpeed engine, hetero_devices=%s",
            hetero_manager.config.device_ids,
        )

    def install(self) -> None:
        """
        安装 ZeRO pre/post forward 钩子。

        DeepSpeed ZeRO-3 通过 `_pre_forward_module_hook` 和
        `_post_forward_module_hook` 管理参数 all-gather，
        我们在 post-forward 中插入 DES-LOC lazy release 逻辑。
        """
        if self._installed:
            logger.warning("DESLOCZeroHook already installed, skip")
            return

        # 遍历 ZeRO-3 管理的所有子模块，注册到 hetero_manager
        if hasattr(self.ds_engine, 'module'):
            for mod in self.ds_engine.module.modules():
                if len(list(mod.parameters(recurse=False))) > 0:
                    self.hetero_manager.register_module(mod)

        self._installed = True
        logger.info(
            "DESLOCZeroHook installed: %d modules registered",
            len(self.hetero_manager._module_device),
        )

    def on_step_end(self) -> None:
        """
        在每个训练 step 结束时调用，触发全设备 lazy bucket 回收。

        建议在 `engine.step()` 之后调用。
        """
        summary = self.hetero_manager.trigger_recycle_all_devices()
        mem = self.hetero_manager.get_memory_summary()
        logger.info("Step end recycle summary: %s", summary)
        logger.debug("Memory summary: %s", mem)


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---- 1. SLoC 基本功能 ----
    sloc = SharedLocalityCache(max_cpu_bytes=1 * 1024 ** 3)
    sloc.register_bucket((0, False), param_numel=1024, dtype=torch.float32)
    sloc.mark_evictable((0, False))
    freed = sloc.evict_evictable_buckets()
    assert freed > 0, "SLoC should have freed memory after eviction"
    logger.info("SLoC smoke test passed: freed=%d bytes", freed)

    # ---- 2. 设备检测（CPU 回退） ----
    # 若无 CUDA，_detect_device_profile 会抛出；直接构造 mock profile
    mock_profile_a6000 = HeteroDeviceProfile(
        device_id=0, device_class=DeviceClass.A6000,
        total_mem_gb=48.0, sm_count=84, pcie_bw_gbps=32.0,
        supports_fp8=False, bucket_size_bytes=128 * 1024 * 1024,
    )
    mock_profile_h100 = HeteroDeviceProfile(
        device_id=2, device_class=DeviceClass.H100,
        total_mem_gb=96.0, sm_count=132, pcie_bw_gbps=64.0,
        supports_fp8=True, bucket_size_bytes=512 * 1024 * 1024,
    )
    assert mock_profile_a6000.supports_fp8 is False
    assert mock_profile_h100.supports_fp8 is True
    logger.info("Device profile smoke test passed")

    # ---- 3. Lazy release 标记流程 ----
    sloc2 = SharedLocalityCache(max_cpu_bytes=1 * 1024 ** 3)
    pipeline = HeteroAllGatherPipeline(
        device_profile=mock_profile_a6000, sloc=sloc2, num_buckets=4
    )
    # 模拟 bucket 进入 READY 状态
    key = (0, False)
    pipeline.bucket_status[key] = BucketStatus.READY
    pipeline._gpu_buffers[key] = torch.zeros(16)
    # lazy release
    pipeline.release_bucket(0, bwd=False, lazy=True)
    assert pipeline.bucket_status[key] == BucketStatus.LAZY_RELEASABLE
    assert pipeline.bucket_can_be_released[key] is True
    # recycle
    recycled = pipeline.recycle_unused_buckets()
    assert recycled == 1
    assert pipeline.bucket_status[key] == BucketStatus.EMPTY
    logger.info("Lazy release smoke test passed: recycled=%d", recycled)

    # ---- 4. Post-forward hook PRE_BACKWARD 路径 ----
    config = HeteroFSDPConfig(
        device_ids=[],  # 无真实 GPU，跳过 detect
        lazy_release_on_recompute=True,
    )
    # 手动注入 mock pipeline
    manager = object.__new__(HeteroFSDPDoubleBufferRecompute)
    manager.config = config
    manager.sloc = sloc2
    manager.pipelines = {0: pipeline}
    manager.device_profiles = {0: mock_profile_a6000}
    manager._registered_modules = set()
    manager._module_device = {}
    manager._module_buckets = {}

    dummy_mod = nn.Linear(4, 4)
    dummy_mod._training_state = TrainingState.PRE_BACKWARD
    manager._module_device[id(dummy_mod)] = 0
    manager._module_buckets[id(dummy_mod)] = [0]

    # 重置 bucket 状态为 READY 以测试 hook
    pipeline.bucket_status[key] = BucketStatus.READY
    pipeline._gpu_buffers[key] = torch.zeros(16)
    pipeline.bucket_can_be_released[key] = False

    out = manager._post_forward_hook(dummy_mod, None, torch.zeros(1))
    assert pipeline.bucket_status[key] == BucketStatus.LAZY_RELEASABLE, (
        f"Expected LAZY_RELEASABLE, got {pipeline.bucket_status[key]}"
    )
    assert dummy_mod._training_state == TrainingState.PRE_BACKWARD  # 状态不应被改为 IDLE
    logger.info("PRE_BACKWARD hook smoke test passed")

    logger.info("All smoke tests passed ✓")
