"""
DES-LOC Heterogeneous H2D Stream Synchronization
=================================================

上游设计意图 (Megatron ae67076):
    Megatron 的 HybridDeviceOptimizer 在 overlap_cpu_optimizer_d2h_h2d 模式下，
    将 optimizer state 分成两部分：一部分留在 GPU（快路径），另一部分 offload 到 CPU
    （慢路径）。CPU 做完 optimizer step 后，需要把更新后的参数异步拷贝回 GPU (H2D)，
    然后让 default stream 等待这条 H2D 拷贝完成，再继续 forward。

    原始 bug (issue #3140)：
        代码错误地调用了 self._d2h_stream.record_event().wait(...)
        ——即 Device-to-Host stream 记录的 event——但此时真正需要等待的是
        H2D (Host-to-Device) stream 完成数据写回 GPU DRAM，应使用
        self._h2d_stream.record_event().wait(torch.cuda.current_stream())。

    修复逻辑：
        param_copy_back_gpu_hook 里，CPU optimizer step 结束、数据已异步
        copy_ 到 gpu_param.data 之后，必须在 h2d_stream 上记录 event，
        并让 default compute stream wait 该 event，否则 forward 可能读到
        尚未写完的旧参数值，导致数值错误（不一定 NaN，但精度偏移）。

DES-LOC 适配点:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) 在此基础上进一步
    扩展到 2× A6000 (SM86, 48 GB) + 1× H100 NVL (SM90, 96 GB) 的异构拓扑：

    1. 多设备流管理 (HeteroStreamRegistry):
       每个 CUDA device 维护独立的 h2d_stream / d2h_stream / compute_stream，
       避免跨设备 stream 污染。A6000 与 H100 的 SM 架构不同，stream 调度策略
       不能共用一套默认参数。

    2. Locality Cache 感知的 H2D 路由 (LocalityCacheRouter):
       DES-LOC 在 CPU DRAM (1.5 TB) 上维护一个 Shared Locality Cache，
       存放近期被多个 device 复用的 optimizer state 副本。
       H2D 拷贝前先查询 cache：若目标 device 上已有 up-to-date 副本则跳过
       拷贝，只做 stream event 同步；若 cache miss 则走完整 pinned-memory 路径。

    3. 异构同步栅栏 (HeteroSyncBarrier):
       PCIe 互联无 NVLink，A6000→H100 的 peer copy 必须经过 CPU，因此引入
       两阶段栅栏：(a) device-local h2d_stream event；(b) cross-device CPU
       futex，保证两块 A6000 的参数写回都完成后，H100 才开始依赖这些参数的
       计算（流水线并行场景）。

    4. SM86/SM90 异构感知 (ArchAwareStreamPriority):
       H100 NVL (SM90) 支持更细粒度的 stream 优先级 (0..-5)，A6000 (SM86)
       只支持 0..-2。本模块在创建 stream 时查询 device capability 并自动
       选择合法优先级，避免 CUDA_ERROR_INVALID_VALUE。

使用方式::

    from deepspeed.runtime.hetero_h2d_stream_sync import (
        HeteroStreamRegistry,
        LocalityCacheRouter,
        HeteroSyncBarrier,
        HeteroH2DStreamSync,
    )

    # 在 DeepSpeed engine 初始化阶段
    registry = HeteroStreamRegistry(device_ids=[0, 1, 2])
    cache    = LocalityCacheRouter(capacity_bytes=4 * 1024**3)  # 4 GB
    barrier  = HeteroSyncBarrier(device_ids=[0, 1, 2])
    h2d_sync = HeteroH2DStreamSync(registry, cache, barrier)

    # 在 optimizer step 后的 hook 中
    h2d_sync.copy_back_and_sync(param, cpu_tensor, device_id=0)
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.cuda

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_PINNED_ALLOC_ALIGN = 2 * 1024 * 1024  # 2 MiB — matches Linux THP size
_CACHE_EVICTION_LRU = "lru"
_SM86_MAX_STREAM_PRIORITY = -2   # A6000
_SM90_MAX_STREAM_PRIORITY = -5   # H100 NVL
_DEFAULT_STREAM_PRIORITY  = 0


# ---------------------------------------------------------------------------
# 1. HeteroStreamRegistry — 每个 device 独立的 stream 三元组
# ---------------------------------------------------------------------------

@dataclass
class _DeviceStreams:
    """Per-device stream 三元组，持有 compute / h2d / d2h 三条 CUDA stream。

    DES-LOC 适配:
        A6000 (SM86) 和 H100 (SM90) 对 stream 优先级范围的支持不同。
        通过查询 ``torch.cuda.Stream`` 允许的优先级范围动态设置，
        避免在 SM86 上使用 SM90 独有的低优先级值。
    """
    device_id:      int
    compute_stream: torch.cuda.Stream
    h2d_stream:     torch.cuda.Stream
    d2h_stream:     torch.cuda.Stream
    sm_major:       int
    sm_minor:       int


class HeteroStreamRegistry:
    """为每个 CUDA device 创建并管理独立的 stream 三元组。

    上游关联:
        Megatron HybridDeviceOptimizer 对每个进程维护单一的 _h2d_stream /
        _d2h_stream。在多 GPU 单进程（如 DeepSpeed ZeRO-3 + tensor parallel）
        场景下，不同 device 共享同一对 stream 会导致不必要的串行化。

    DES-LOC 适配:
        - 为每个 device_id 单独创建 stream，按 SM 架构设置合法优先级。
        - H100 (SM90) 的 h2d_stream 使用更高优先级（更负的值），
          优先服务大模型层的参数写回，减少 H100 等待时间。
        - 暴露 get_h2d_stream / get_d2h_stream / get_compute_stream 接口，
          供 HeteroH2DStreamSync 调用。
    """

    def __init__(self, device_ids: List[int]) -> None:
        self._registry: Dict[int, _DeviceStreams] = {}
        for did in device_ids:
            self._init_device(did)
        logger.info(
            "[HeteroStreamRegistry] initialized streams for devices: %s", device_ids
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_sm_version(self, device_id: int) -> Tuple[int, int]:
        """返回 (sm_major, sm_minor)，例如 A6000 → (8, 6)，H100 → (9, 0)。"""
        prop = torch.cuda.get_device_properties(device_id)
        return prop.major, prop.minor

    def _choose_priority(self, sm_major: int, is_h2d: bool) -> int:
        """根据 SM 架构为 h2d stream 选择合法的最高优先级。

        SM90 (H100) 支持更深的优先级队列，h2d stream 可以抢占低优先级 kernel，
        缩短参数写回的等待窗口。SM86 (A6000) 只支持两级，设为 -2 即可。
        compute stream 始终用默认优先级 0，避免影响正常计算调度。
        """
        if not is_h2d:
            return _DEFAULT_STREAM_PRIORITY
        if sm_major >= 9:   # SM90 → H100 NVL
            return _SM90_MAX_STREAM_PRIORITY
        elif sm_major == 8: # SM86 → A6000
            return _SM86_MAX_STREAM_PRIORITY
        else:
            return _DEFAULT_STREAM_PRIORITY

    def _init_device(self, device_id: int) -> None:
        sm_major, sm_minor = self._get_sm_version(device_id)
        h2d_priority     = self._choose_priority(sm_major, is_h2d=True)
        d2h_priority     = self._choose_priority(sm_major, is_h2d=False)
        compute_priority = _DEFAULT_STREAM_PRIORITY

        with torch.cuda.device(device_id):
            # 确认优先级在设备允许范围内
            lo, hi = torch.cuda.Stream.priority_range()
            h2d_priority = max(h2d_priority, lo)   # lo 是最高优先级（最负）

            compute_stream = torch.cuda.Stream(
                device=device_id, priority=compute_priority
            )
            h2d_stream = torch.cuda.Stream(
                device=device_id, priority=h2d_priority
            )
            d2h_stream = torch.cuda.Stream(
                device=device_id, priority=d2h_priority
            )

        self._registry[device_id] = _DeviceStreams(
            device_id=device_id,
            compute_stream=compute_stream,
            h2d_stream=h2d_stream,
            d2h_stream=d2h_stream,
            sm_major=sm_major,
            sm_minor=sm_minor,
        )
        logger.debug(
            "[HeteroStreamRegistry] device=%d SM%d%d h2d_priority=%d",
            device_id, sm_major, sm_minor, h2d_priority,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_h2d_stream(self, device_id: int) -> torch.cuda.Stream:
        return self._registry[device_id].h2d_stream

    def get_d2h_stream(self, device_id: int) -> torch.cuda.Stream:
        return self._registry[device_id].d2h_stream

    def get_compute_stream(self, device_id: int) -> torch.cuda.Stream:
        return self._registry[device_id].compute_stream

    def sm_version(self, device_id: int) -> Tuple[int, int]:
        ds = self._registry[device_id]
        return ds.sm_major, ds.sm_minor

    def all_device_ids(self) -> List[int]:
        return list(self._registry.keys())


# ---------------------------------------------------------------------------
# 2. LocalityCacheRouter — Shared Locality Cache 感知的 H2D 路由
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    """Locality Cache 中的一条记录。

    Fields:
        cpu_ptr:    参数在 CPU DRAM 上的数据指针（用于 staleness 检测）
        version:    optimizer step 计数，用于判断 GPU 副本是否过期
        device_id:  缓存命中的目标 device
        pinned_buf: 与 cpu_ptr 关联的 pinned-memory buffer（可选，加速 H2D）
        last_access:最后访问时间戳（用于 LRU eviction）
    """
    cpu_ptr:     int
    version:     int
    device_id:   int
    pinned_buf:  Optional[torch.Tensor]
    last_access: float = field(default_factory=time.monotonic)


class LocalityCacheRouter:
    """DES-LOC Shared Locality Cache 的路由层。

    设计动机:
        在 1.5 TB CPU DRAM 环境下，不同的 pipeline stage 可能反复读取同一组
        参数的 CPU 副本（例如共享 embedding 层），每次都触发完整 H2D 拷贝
        既浪费 PCIe 带宽，又拉长 critical path。

        LocalityCacheRouter 维护一个 (param_id → _CacheEntry) 的哈希表，
        记录每个参数最近一次成功写回 GPU 的版本号。若当前 version 未变，
        则直接跳过 copy_，仅做 stream event 同步（因为 GPU 上的数据已是
        最新的）。

    上游关联:
        Megatron cpu_offloading/hybrid_optimizer.py 中的 param_copy_back_gpu_hook
        对每个 param 无条件执行 gpu_param.data.copy_(param.data, non_blocking=True)。
        LocalityCacheRouter 在此之上增加了版本门控，cache hit 时绕过 copy_。

    容量管理:
        按字节计算缓存使用量，超过 capacity_bytes 时 LRU evict。
        pinned buffer 在 evict 时显式释放（通过 del + gc），避免 CPU DRAM 泄漏。
    """

    def __init__(
        self,
        capacity_bytes: int = 4 * 1024 ** 3,
        eviction_policy: str = _CACHE_EVICTION_LRU,
    ) -> None:
        self._capacity    = capacity_bytes
        self._used_bytes  = 0
        self._policy      = eviction_policy
        self._cache: Dict[int, _CacheEntry] = {}   # param_id → entry
        self._lock        = threading.Lock()
        logger.info(
            "[LocalityCacheRouter] capacity=%.2f GB policy=%s",
            capacity_bytes / 1024 ** 3,
            eviction_policy,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _entry_bytes(self, entry: _CacheEntry) -> int:
        if entry.pinned_buf is not None:
            return entry.pinned_buf.nbytes
        return 0

    def _evict_lru_until(self, needed_bytes: int) -> None:
        """驱逐最久未访问的条目，直到释放出 needed_bytes。"""
        while self._used_bytes + needed_bytes > self._capacity and self._cache:
            # 找到 last_access 最小的 key
            lru_key = min(self._cache, key=lambda k: self._cache[k].last_access)
            victim  = self._cache.pop(lru_key)
            freed   = self._entry_bytes(victim)
            self._used_bytes -= freed
            # 显式释放 pinned buffer
            if victim.pinned_buf is not None:
                del victim.pinned_buf
            logger.debug(
                "[LocalityCacheRouter] evicted param_id=%d freed=%d bytes",
                lru_key, freed,
            )

    def _make_pinned_buffer(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """为 cpu_tensor 创建对齐的 pinned-memory 拷贝，加速后续 H2D 传输。

        注意：pinned memory 是稀缺资源，只对 >= 1 MB 的参数创建，
        小参数直接走 pageable 路径（延迟稍高但节省 OS 页表资源）。
        """
        if cpu_tensor.nbytes < 1 * 1024 * 1024:
            return None
        pinned = torch.empty_like(cpu_tensor, pin_memory=True)
        pinned.copy_(cpu_tensor)
        return pinned

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def query(
        self,
        param_id: int,
        current_version: int,
        device_id: int,
    ) -> bool:
        """查询 param_id 在 device_id 上的 GPU 副本是否已是 current_version。

        Returns:
            True  → cache hit，GPU 副本已是最新，调用方可跳过 copy_
            False → cache miss 或版本过期，需要执行 H2D 拷贝
        """
        with self._lock:
            entry = self._cache.get(param_id)
            if entry is None:
                return False
            if entry.version < current_version:
                return False
            if entry.device_id != device_id:
                # 参数迁移到了不同 device（TP/PP rebalance），强制刷新
                return False
            entry.last_access = time.monotonic()
            return True

    def update(
        self,
        param_id: int,
        cpu_tensor: torch.Tensor,
        current_version: int,
        device_id: int,
    ) -> Optional[torch.Tensor]:
        """H2D 拷贝完成后，更新 cache entry 并返回 pinned buffer（若已创建）。

        Returns:
            pinned buffer（若存在），供调用方直接用于下次 H2D 而无需再次
            pin memory；若无则返回 None。
        """
        with self._lock:
            existing = self._cache.get(param_id)
            if existing is not None:
                existing.version    = current_version
                existing.device_id  = device_id
                existing.last_access = time.monotonic()
                # 更新 pinned buffer（如果 tensor 地址变了）
                if existing.cpu_ptr != cpu_tensor.data_ptr():
                    old_bytes = self._entry_bytes(existing)
                    existing.pinned_buf = self._make_pinned_buffer(cpu_tensor)
                    existing.cpu_ptr    = cpu_tensor.data_ptr()
                    new_bytes = self._entry_bytes(existing)
                    self._used_bytes += new_bytes - old_bytes
                return existing.pinned_buf
            else:
                pinned    = self._make_pinned_buffer(cpu_tensor)
                new_bytes = pinned.nbytes if pinned is not None else 0
                self._evict_lru_until(new_bytes)
                entry = _CacheEntry(
                    cpu_ptr    = cpu_tensor.data_ptr(),
                    version    = current_version,
                    device_id  = device_id,
                    pinned_buf = pinned,
                )
                self._cache[param_id] = entry
                self._used_bytes += new_bytes
                return pinned

    def invalidate(self, param_id: int) -> None:
        """强制使某个参数的 cache entry 失效（例如梯度清零后）。"""
        with self._lock:
            entry = self._cache.pop(param_id, None)
            if entry is not None:
                freed = self._entry_bytes(entry)
                self._used_bytes -= freed
                if entry.pinned_buf is not None:
                    del entry.pinned_buf

    @property
    def stats(self) -> Dict:
        with self._lock:
            return {
                "num_entries": len(self._cache),
                "used_bytes":  self._used_bytes,
                "capacity_bytes": self._capacity,
                "utilization": self._used_bytes / max(self._capacity, 1),
            }


# ---------------------------------------------------------------------------
# 3. HeteroSyncBarrier — 跨设备两阶段同步栅栏
# ---------------------------------------------------------------------------

class HeteroSyncBarrier:
    """跨 device 的两阶段同步栅栏，专为 PCIe-only（无 NVLink）拓扑设计。

    问题背景:
        在流水线并行（Pipeline Parallel）场景下，stage N 的参数更新完成后，
        stage N+1 才能开始 forward。若 stage N 分布在两块 A6000 上，
        stage N+1 在 H100 上，则 H100 必须等两块 A6000 都完成 H2D 写回。

        无 NVLink 时，CUDA peer access 需要经过 CPU，不能直接用
        cudaStreamWaitEvent 跨设备同步。

    解决方案 — 两阶段栅栏:
        Phase 1 (device-local):  每个 device 在自己的 h2d_stream 上
                                 record_event，表示本 device 的 H2D 已完成。
        Phase 2 (cross-device):  CPU 侧用 threading.Barrier 等待所有参与
                                 device 的 phase-1 event 都 query 成功，
                                 然后释放等待中的 device（通常是 H100）
                                 继续执行。

    注意事项:
        - barrier 是 per-step 的，每个 optimizer step 重置一次。
        - 只有被标记为 "需要跨设备同步" 的参数才走 phase-2；
          device-local 参数只走 phase-1（与 Megatron 原始行为一致）。
        - H100 侧调用 wait_all_devices_ready() 会阻塞当前 Python 线程
          直到所有 A6000 完成；这是 CPU-side blocking，不影响 GPU kernel
          调度（H100 的 compute stream 在 CPU 释放前已暂停提交新 kernel）。
    """

    def __init__(self, device_ids: List[int]) -> None:
        self._device_ids = list(device_ids)
        self._events: Dict[int, Optional[torch.cuda.Event]] = {
            did: None for did in device_ids
        }
        self._lock      = threading.Lock()
        self._ready_set: set = set()
        self._cond      = threading.Condition(self._lock)
        logger.info(
            "[HeteroSyncBarrier] initialized for devices: %s", device_ids
        )

    def reset(self) -> None:
        """每个 optimizer step 开始时重置栅栏状态。"""
        with self._lock:
            self._events   = {did: None for did in self._device_ids}
            self._ready_set.clear()

    def record_device_ready(
        self, device_id: int, h2d_stream: torch.cuda.Stream
    ) -> None:
        """Phase 1：在 h2d_stream 上 record event，标记本 device 已写回完成。

        DES-LOC 适配:
            对应 Megatron 修复后的：
                self._h2d_stream.record_event().wait(torch.cuda.current_stream())
            此处我们先 record，不立即 wait，把 wait 延迟到 phase-2，
            使多个 device 的 H2D 能并行执行（重叠 PCIe 传输）。
        """
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        with torch.cuda.stream(h2d_stream):
            event.record(h2d_stream)

        with self._lock:
            self._events[device_id] = event
            self._cond.notify_all()

        logger.debug(
            "[HeteroSyncBarrier] device=%d recorded H2D ready event", device_id
        )

    def wait_device_local(
        self,
        device_id: int,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        """仅等待本 device 的 H2D event（对应 Megatron 修复的单 device 场景）。

        等效于修复后的：
            self._h2d_stream.record_event().wait(torch.cuda.current_stream())
        """
        with self._lock:
            event = self._events.get(device_id)

        if event is None:
            logger.warning(
                "[HeteroSyncBarrier] wait_device_local called before "
                "record_device_ready on device=%d", device_id
            )
            return

        # 让 compute_stream 等待 h2d event，GPU 侧非阻塞
        compute_stream.wait_event(event)
        logger.debug(
            "[HeteroSyncBarrier] device=%d compute_stream waiting on H2D event",
            device_id,
        )

    def _poll_event_done(self, event: torch.cuda.Event, timeout: float = 30.0) -> bool:
        """CPU 侧轮询 CUDA event 是否完成，超时返回 False。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if event.query():
                return True
            time.sleep(1e-4)   # 100 µs 轮询间隔，避免 busy-wait 烧 CPU
        return False

    def wait_all_devices_ready(
        self,
        participating_device_ids: List[int],
        timeout: float = 30.0,
    ) -> None:
        """Phase 2：阻塞 CPU 线程，直到所有 participating devices 的 H2D 完成。

        典型调用方：H100 侧的 pipeline stage，在提交依赖 A6000 参数的
        kernel 之前调用。
        """
        deadline = time.monotonic() + timeout
        for did in participating_device_ids:
            with self._lock:
                # 等待对应 device record 其 event
                while self._events.get(did) is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"[HeteroSyncBarrier] timeout waiting for "
                            f"device={did} to record H2D event"
                        )
                    self._cond.wait(timeout=min(remaining, 0.1))
                event = self._events[did]

            # CPU-side 轮询 event
            remaining = deadline - time.monotonic()
            if not self._poll_event_done(event, timeout=remaining):
                raise TimeoutError(
                    f"[HeteroSyncBarrier] H2D event on device={did} "
                    f"did not complete within {timeout}s"
                )
            logger.debug(
                "[HeteroSyncBarrier] device=%d H2D confirmed done (CPU poll)",
                did,
            )

        logger.info(
            "[HeteroSyncBarrier] all devices %s H2D ready",
            participating_device_ids,
        )


# ---------------------------------------------------------------------------
# 4. HeteroH2DStreamSync — 核心协调器
# ---------------------------------------------------------------------------

class HeteroH2DStreamSync:
    """DES-LOC 异构 H2D 流同步的核心协调器。

    将 Megatron fix (ae67076) 的单设备 h2d_stream 修复扩展为
    多设备、cache 感知、架构适配的完整实现。

    工作流程（对应 param_copy_back_gpu_hook）::

        CPU optimizer step 完成
              │
              ▼
        [LocalityCacheRouter.query]  ──── hit ──▶  跳过 copy_，直接 phase-1
              │ miss
              ▼
        pinned_buf = cache.update(...)    # 更新/创建 pinned buffer
              │
              ▼
        [h2d_stream 上] gpu_param.data.copy_(src, non_blocking=True)
              │
              ▼
        [HeteroSyncBarrier.record_device_ready]   # phase-1 record event
              │
        ┌─────┴───────────────────────────────┐
        │ device-local (同 device 的 compute) │  → wait_device_local
        │ cross-device (H100 等待 A6000)      │  → wait_all_devices_ready
        └─────────────────────────────────────┘

    参数:
        registry:       HeteroStreamRegistry 实例
        cache:          LocalityCacheRouter 实例
        barrier:        HeteroSyncBarrier 实例
        step_counter:   外部传入的可变列表 [int]，记录当前 optimizer step，
                        用于 cache version 判断；若为 None 则内部自增。
    """

    def __init__(
        self,
        registry:    HeteroStreamRegistry,
        cache:       LocalityCacheRouter,
        barrier:     HeteroSyncBarrier,
        step_counter: Optional[List[int]] = None,
    ) -> None:
        self._registry     = registry
        self._cache        = cache
        self._barrier      = barrier
        self._step_counter = step_counter if step_counter is not None else [0]
        self._h2d_skip_count  = 0
        self._h2d_exec_count  = 0
        logger.info("[HeteroH2DStreamSync] initialized")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        """每个 optimizer step 开始时调用，重置栅栏并递增版本计数。"""
        self._step_counter[0] += 1
        self._barrier.reset()
        logger.debug(
            "[HeteroH2DStreamSync] begin_step version=%d",
            self._step_counter[0],
        )

    def copy_back_and_sync(
        self,
        gpu_param:    torch.Tensor,
        cpu_tensor:   torch.Tensor,
        device_id:    int,
        param_id:     Optional[int] = None,
        cross_device: bool = False,
        participating_devices: Optional[List[int]] = None,
    ) -> None:
        """将 CPU optimizer 更新后的参数写回 GPU，并正确同步 H2D stream。

        这是 Megatron ae67076 修复的核心逻辑在 DES-LOC 中的等价实现：

        Megatron 原始（修复后）::

            for param in _param_generator(optimizer):
                gpu_param = self.cpu_copys_map_gpu_param[param]
                gpu_param.data.copy_(param.data, non_blocking=True)
            # 修复前错误地用了 _d2h_stream，修复后正确使用 _h2d_stream
            self._h2d_stream.record_event().wait(torch.cuda.current_stream())

        DES-LOC 扩展::

            - 多设备独立 h2d_stream（通过 registry 获取）
            - cache 感知路由（cache hit 时跳过 copy_）
            - 异构同步栅栏（跨设备时使用 wait_all_devices_ready）

        参数:
            gpu_param:    目标 GPU tensor（in-place 更新）
            cpu_tensor:   CPU 上 optimizer 更新后的参数值
            device_id:    目标 GPU 的 device id
            param_id:     参数的唯一 ID，用于 cache 查询；若 None 则
                          用 id(gpu_param) 代替（弱引用安全）
            cross_device: 若 True，写回完成后通知 barrier（用于 PP 同步）
            participating_devices: cross_device=True 时，调用方等待的
                          device id 列表；通常是上游 stage 的所有 device
        """
        if param_id is None:
            param_id = id(gpu_param)

        version    = self._step_counter[0]
        h2d_stream = self._registry.get_h2d_stream(device_id)
        compute_st = self._registry.get_compute_stream(device_id)

        # ── Phase 0: cache 查询 ────────────────────────────────────────
        cache_hit = self._cache.query(param_id, version, device_id)

        if cache_hit:
            # GPU 副本已是最新，只需确保 compute stream 等待 h2d stream
            # （上一步的 copy_ 可能仍在 h2d_stream 上 in-flight）
            self._barrier.record_device_ready(device_id, h2d_stream)
            self._barrier.wait_device_local(device_id, compute_st)
            self._h2d_skip_count += 1
            logger.debug(
                "[HeteroH2DStreamSync] cache HIT param_id=%d device=%d "
                "version=%d skip copy_",
                param_id, device_id, version,
            )
            return

        # ── Phase 1: 执行 H2D copy_ ───────────────────────────────────
        # 获取（或创建）pinned buffer 作为 H2D 的 src，避免 pageable copy
        pinned_buf = self._cache.update(param_id, cpu_tensor, version, device_id)
        src = pinned_buf if pinned_buf is not None else cpu_tensor

        with torch.cuda.stream(h2d_stream):
            # non_blocking=True：H2D DMA 在 h2d_stream 上异步执行
            gpu_param.data.copy_(src, non_blocking=True)

        self._h2d_exec_count += 1
        logger.debug(
            "[HeteroH2DStreamSync] H2D copy_ param_id=%d device=%d "
            "nbytes=%d version=%d pinned=%s",
            param_id, device_id, cpu_tensor.nbytes, version,
            pinned_buf is not None,
        )

        # ── Phase 2: record H2D event 并同步 ─────────────────────────
        # 对应 Megatron 修复：使用 _h2d_stream（而非 _d2h_stream）record event
        self._barrier.record_device_ready(device_id, h2d_stream)

        if cross_device and participating_devices:
            # 跨设备场景（H100 等待 A6000 们完成 H2D）：CPU 侧阻塞等待
            self._barrier.wait_all_devices_ready(participating_devices)
            logger.info(
                "[HeteroH2DStreamSync] cross-device sync done, "
                "devices=%s all H2D complete",
                participating_devices,
            )
        else:
            # device-local 场景：让 compute stream GPU 侧 wait H2D event
            # 这是 Megatron ae67076 修复的直接等价
            self._barrier.wait_device_local(device_id, compute_st)

    def make_param_copy_back_hook(
        self,
        cpu_to_gpu_map: Dict[int, torch.Tensor],
        device_id:      int,
        cross_device:   bool = False,
        participating_devices: Optional[List[int]] = None,
    ) -> callable:
        """创建 DeepSpeed optimizer hook，批量处理一个 optimizer group 的写回。

        对应 Megatron HybridDeviceOptimizer._make_param_copy_back_gpu_hook。
        返回的 hook 在 optimizer.step() 之后被调用。

        参数:
            cpu_to_gpu_map: {cpu_param_data_ptr → gpu_param tensor} 映射
            device_id:      本 hook 负责的 GPU device id
            cross_device:   是否需要跨设备栅栏（PP 边界处设为 True）
            participating_devices: 跨设备等待的 device 列表

        示例::

            hook = h2d_sync.make_param_copy_back_hook(
                cpu_to_gpu_map={id(p): gpu_p for p, gpu_p in pairs},
                device_id=0,
            )
            optimizer.register_step_post_hook(hook)
        """
        # 使用 weakref 避免 hook 持有 optimizer 强引用导致内存泄漏
        sync_ref = weakref.ref(self)

        def _hook(optimizer, args, kwargs):
            sync = sync_ref()
            if sync is None:
                return
            for cpu_ptr, gpu_param in cpu_to_gpu_map.items():
                # 找到对应的 CPU tensor（通过 ptr 反查）
                cpu_tensor = _find_cpu_tensor_by_ptr(optimizer, cpu_ptr)
                if cpu_tensor is None:
                    logger.warning(
                        "[HeteroH2DStreamSync] hook: cannot find cpu tensor "
                        "for ptr=%d, skipping", cpu_ptr
                    )
                    continue
                sync.copy_back_and_sync(
                    gpu_param=gpu_param,
                    cpu_tensor=cpu_tensor,
                    device_id=device_id,
                    param_id=cpu_ptr,
                    cross_device=cross_device,
                    participating_devices=participating_devices,
                )

        return _hook

    @property
    def stats(self) -> Dict:
        total = self._h2d_skip_count + self._h2d_exec_count
        return {
            "h2d_executed":  self._h2d_exec_count,
            "h2d_skipped":   self._h2d_skip_count,
            "cache_hit_rate": self._h2d_skip_count / max(total, 1),
            "step":          self._step_counter[0],
            "cache":         self._cache.stats,
        }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _find_cpu_tensor_by_ptr(
    optimizer: torch.optim.Optimizer, target_ptr: int
) -> Optional[torch.Tensor]:
    """在 optimizer param_groups 中按 data_ptr 查找 CPU tensor。

    DES-LOC 说明：
        Megatron 用 cpu_copys_map_gpu_param dict 做正向映射（cpu→gpu）；
        这里做反向查找（ptr→tensor），供 hook 使用。
        实际部署时应由调用方维护双向 map，此函数作为 fallback。
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not p.is_cuda and p.data_ptr() == target_ptr:
                return p
    return None


def build_des_loc_h2d_sync(
    device_ids: List[int],
    cache_capacity_gb: float = 4.0,
) -> HeteroH2DStreamSync:
    """工厂函数：一行代码创建完整的 DES-LOC H2D 同步栈。

    参数:
        device_ids:        参与训练的 CUDA device id 列表，例如 [0, 1, 2]
        cache_capacity_gb: Locality Cache 容量（GB），默认 4 GB

    返回:
        配置好的 HeteroH2DStreamSync 实例

    示例::

        h2d_sync = build_des_loc_h2d_sync([0, 1, 2], cache_capacity_gb=8.0)
        h2d_sync.begin_step()
        h2d_sync.copy_back_and_sync(gpu_p, cpu_p, device_id=0)
    """
    registry = HeteroStreamRegistry(device_ids)
    cache    = LocalityCacheRouter(
        capacity_bytes=int(cache_capacity_gb * 1024 ** 3)
    )
    barrier  = HeteroSyncBarrier(device_ids)
    return HeteroH2DStreamSync(registry, cache, barrier)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not torch.cuda.is_available():
        print("No CUDA available, skipping smoke test.")
        sys.exit(0)

    n_dev = torch.cuda.device_count()
    device_ids = list(range(min(n_dev, 3)))
    print(f"Smoke test on devices: {device_ids}")

    # ── 构建 DES-LOC H2D 同步栈 ──────────────────────────────────────
    h2d_sync = build_des_loc_h2d_sync(device_ids, cache_capacity_gb=0.5)

    # ── Test 1: HeteroStreamRegistry SM 版本查询 ─────────────────────
    for did in device_ids:
        sm = h2d_sync._registry.sm_version(did)
        assert len(sm) == 2 and sm[0] >= 7, f"Unexpected SM version {sm} on device {did}"
        print(f"  device={did} SM{sm[0]}{sm[1]} ✓")

    # ── Test 2: 单设备 copy_back_and_sync，验证 GPU 数据正确性 ───────
    did = device_ids[0]
    size = (256, 256)
    cpu_p = torch.randn(*size)                         # CPU optimizer 输出
    gpu_p = torch.zeros(*size, device=f"cuda:{did}")   # 待写回的 GPU param

    h2d_sync.begin_step()
    h2d_sync.copy_back_and_sync(gpu_p, cpu_p, device_id=did, param_id=42)
    torch.cuda.synchronize(did)

    assert torch.allclose(gpu_p.cpu(), cpu_p, atol=1e-5), \
        "H2D copy correctness failed"
    print("  copy_back_and_sync correctness ✓")

    # ── Test 3: cache hit 路径（第二次 begin_step 版本相同 → miss；相同步骤 → hit）
    # 同一 step 内第二次写回同一 param 应走 hit 路径
    h2d_sync._step_counter[0] -= 1          # 回退 version，模拟同一 step
    h2d_sync._barrier.reset()
    skip_before = h2d_sync._h2d_skip_count
    h2d_sync.copy_back_and_sync(gpu_p, cpu_p, device_id=did, param_id=42)
    assert h2d_sync._h2d_skip_count == skip_before + 1, \
        "Cache hit path not triggered"
    print("  cache hit path ✓")

    # ── Test 4: LocalityCacheRouter stats ────────────────────────────
    stats = h2d_sync.stats
    assert "cache_hit_rate" in stats
    assert stats["h2d_executed"] >= 1
    print(f"  stats={stats} ✓")

    # ── Test 5: HeteroSyncBarrier device-local wait（无死锁） ────────
    barrier   = h2d_sync._barrier
    h2d_st    = h2d_sync._registry.get_h2d_stream(did)
    compute_st = h2d_sync._registry.get_compute_stream(did)
    barrier.reset()
    barrier.record_device_ready(did, h2d_st)
    barrier.wait_device_local(did, compute_st)
    torch.cuda.synchronize(did)
    print("  HeteroSyncBarrier device-local wait ✓")

    print("\nAll smoke tests passed.")
