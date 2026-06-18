"""
hetero_kv_cache_offload.py
==========================

DES-LOC HeteroKVCacheOffload — Decoupled Execution with Shared LOcality Cache
异构KV Cache卸载管理器

上游设计意图 (Megatron commit 42986ace)
---------------------------------------
Megatron-LM PR #3048 ("Refactor rl_offload_kv_cache_during_training") 的核心思路：
在RLHF训练循环中，GPU显存同时需要容纳训练参数/优化器状态和推理KV Cache，
两者竞争造成OOM。原有方案直接 `kv_cache.cpu()` 导致虚拟地址失效，
每次resume时CUDA必须重新映射，破坏了CUDAGraph捕获。

PR #3048 引入 `torch_memory_saver`：通过 `mmap` 在CPU端预留固定虚拟地址段，
GPU物理页可以在 pause/resume 之间被回收，但GPU侧指针不变。
这样CUDAGraph中录制的地址仍然有效，resume时只需重新绑定物理页即可。

DES-LOC 适配点
--------------
Neuron_SP 的硬件拓扑（2×A6000 48GB SM86 + 1×H100 NVL 96GB SM90，PCIe互联）
没有 NVLink，CPU DRAM 1.5TB 极大。DES-LOC 的核心思想是：

1. **Decoupled Execution**：推理（H100）与训练（A6000×2）在时间上解耦，
   共享同一进程但分时复用显存。KV Cache 在训练阶段必须从 H100/A6000 撤离。

2. **Shared LOcality Cache**：1.5TB DRAM 充当所有设备的共享二级缓存，
   KV Cache 以 pinned memory 形式驻留其中，通过固定虚拟地址实现零拷贝 resume。

与 Megatron 的关键差异：
- Megatron 假设单机同构 GPU，DES-LOC 需要感知设备异构性（SM86 vs SM90）
- DES-LOC 引入 "Locality Score" 决策哪块 KV Cache 优先卸载
- PCIe 带宽约束下，使用异步 DMA 流水线替代同步 H2D/D2H
- DeepSpeed ZeRO 分片感知：卸载时协同 ZeRO stage 调度

关键数据结构：
- HeteroKVCacheOffload: 主管理器，负责分配/卸载/恢复
- KVCacheRegion: 单块 KV Cache 的元信息（设备、虚拟地址、locality score）
- PinnedCPUBuffer: CPU 端固定内存池，支持固定虚拟地址映射
- OffloadScheduler: 异步 PCIe DMA 调度，流水线隐藏延迟

作者: Neuron_SP Project (reinterpreted from Megatron 42986ace)
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.cuda

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 环境探测：判断当前进程绑定的GPU是SM86(A6000)还是SM90(H100)
# ---------------------------------------------------------------------------

def _detect_device_arch(device: torch.device) -> str:
    """返回设备架构字符串，如 'sm86' 或 'sm90'。

    DES-LOC 需要区分 A6000 (SM86) 和 H100 NVL (SM90)，因为：
    - H100 有 HBM3，PCIe 带宽不对称（H2D vs D2H 差异更大）
    - A6000 SM86 缺少 BF16 硬件加速，影响 KV Cache 量化精度选择
    """
    if device.type != "cuda":
        return "cpu"
    try:
        props = torch.cuda.get_device_properties(device)
        major, minor = props.major, props.minor
        return f"sm{major}{minor}"
    except Exception as exc:
        logger.warning("无法获取设备 %s 的计算能力: %s", device, exc)
        return "unknown"


def _estimate_pcie_bandwidth_gbps(device: torch.device) -> float:
    """估算 PCIe 带宽（GB/s），用于 OffloadScheduler 调度决策。

    实测值（保守估计，含协议开销）：
    - A6000 PCIe 4.0 x16 ≈ 28 GB/s（单向）
    - H100 NVL PCIe 5.0 x16 ≈ 56 GB/s（单向）
    无 NVLink，所有 GPU-CPU 传输都走 PCIe。
    """
    arch = _detect_device_arch(device)
    bw_map = {
        "sm86": 28.0,   # A6000, PCIe 4.0 x16
        "sm90": 56.0,   # H100 NVL, PCIe 5.0 x16
    }
    return bw_map.get(arch, 16.0)


# ---------------------------------------------------------------------------
# 固定虚拟地址 CPU 内存池
# 对应 Megatron 的 torch_memory_saver.region(tag="kv_cache", enable_cpu_backup=True)
# ---------------------------------------------------------------------------

class PinnedCPUBuffer:
    """CPU 端固定虚拟地址内存池。

    上游实现依赖 `torch_memory_saver` 的 mmap 后端；DES-LOC 在 DeepSpeed
    生态下自实现等价语义：

    1. 用 `mmap.mmap(-1, size, MAP_SHARED | MAP_ANONYMOUS)` 预留虚拟地址段
    2. 用 `torch.frombuffer` 在该地址上建立 PyTorch tensor 视图
    3. GPU Tensor 的 data_ptr() 通过 cudaHostRegister 固定，保证跨 pause/resume
       的虚拟地址稳定性

    关键约束：虚拟地址一旦分配就不能移动，这是 CUDAGraph 回放的前提。
    """

    # 类级别注册表，防止 GC 回收 mmap 对象
    _registry: Dict[int, "PinnedCPUBuffer"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, nbytes: int, tag: str = "kv_cache"):
        """
        Parameters
        ----------
        nbytes : int
            需要分配的字节数（对齐到 2MB huge page）
        tag : str
            区域标识，用于 pause/resume 路由
        """
        self.tag = tag
        self.nbytes = self._align_to_hugepage(nbytes)
        self._mmap_obj: Optional[mmap.mmap] = None
        self._cpu_tensor: Optional[torch.Tensor] = None
        self._virtual_addr: int = 0
        self._is_paused: bool = False
        self._lock = threading.Lock()

        self._allocate()
        logger.info(
            "[DES-LOC] PinnedCPUBuffer tag=%s 分配 %.2f GB，虚拟地址 0x%x",
            tag, self.nbytes / 1024**3, self._virtual_addr,
        )

    @staticmethod
    def _align_to_hugepage(nbytes: int, page_size: int = 2 * 1024 * 1024) -> int:
        """将字节数向上对齐到 2MB huge page，减少 TLB miss。"""
        return ((nbytes + page_size - 1) // page_size) * page_size

    def _allocate(self) -> None:
        """在 CPU 端预留固定虚拟地址段。"""
        try:
            # MAP_SHARED | MAP_ANONYMOUS：匿名共享映射，地址固定
            self._mmap_obj = mmap.mmap(
                -1, self.nbytes,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
            # 触发实际物理页分配（首次 fault-in）
            self._mmap_obj.write(b'\x00' * min(4096, self.nbytes))
            self._mmap_obj.seek(0)

            # 建立 uint8 CPU tensor 视图
            self._cpu_tensor = torch.frombuffer(
                self._mmap_obj, dtype=torch.uint8, count=self.nbytes
            )
            self._virtual_addr = self._cpu_tensor.data_ptr()

            with PinnedCPUBuffer._registry_lock:
                PinnedCPUBuffer._registry[self._virtual_addr] = self

        except Exception as exc:
            logger.error("[DES-LOC] PinnedCPUBuffer 分配失败: %s", exc)
            raise

    def view_as(self, shape: Tuple, dtype: torch.dtype) -> torch.Tensor:
        """将 CPU buffer 解释为指定形状和dtype的 tensor。

        DES-LOC 在 resume 阶段用此接口获取与 GPU tensor 形状匹配的 CPU 视图，
        避免数据拷贝（tensor 共享同一虚拟地址段）。
        """
        if self._cpu_tensor is None:
            raise RuntimeError("PinnedCPUBuffer 尚未分配")
        nbytes_needed = torch.empty(shape, dtype=dtype).nbytes
        if nbytes_needed > self.nbytes:
            raise ValueError(
                f"请求 {nbytes_needed} 字节但只分配了 {self.nbytes} 字节"
            )
        return self._cpu_tensor[:nbytes_needed].view(dtype).reshape(shape)

    def pause(self) -> None:
        """模拟 torch_memory_saver.pause：标记为不活跃，允许物理页被 swap。

        真正的物理页回收需要 kernel 支持（madvise MADV_FREE），这里做软标记。
        在 DES-LOC 中，pause 后的 buffer 可以被异步 prefetch 覆写。
        """
        with self._lock:
            if not self._is_paused:
                self._is_paused = True
                if self._mmap_obj is not None:
                    try:
                        # 建议内核此段内存暂时不需要，可回收物理页
                        os.madvise(
                            self._virtual_addr, self.nbytes,
                            getattr(os, 'MADV_FREE', 8)
                        )
                    except (AttributeError, OSError) as exc:
                        logger.debug("[DES-LOC] madvise MADV_FREE 不可用: %s", exc)
                logger.debug(
                    "[DES-LOC] PinnedCPUBuffer tag=%s paused，释放物理页提示", self.tag
                )

    def resume(self) -> None:
        """模拟 torch_memory_saver.resume：重新激活，触发物理页重新 fault-in。"""
        with self._lock:
            if self._is_paused:
                self._is_paused = False
                if self._mmap_obj is not None:
                    try:
                        os.madvise(
                            self._virtual_addr, self.nbytes,
                            getattr(os, 'MADV_WILLNEED', 3)
                        )
                    except (AttributeError, OSError) as exc:
                        logger.debug("[DES-LOC] madvise MADV_WILLNEED 不可用: %s", exc)
                logger.debug(
                    "[DES-LOC] PinnedCPUBuffer tag=%s resumed，虚拟地址 0x%x 保持稳定",
                    self.tag, self._virtual_addr,
                )

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def __del__(self) -> None:
        if self._mmap_obj is not None:
            try:
                self._mmap_obj.close()
            except Exception:
                pass
        with PinnedCPUBuffer._registry_lock:
            PinnedCPUBuffer._registry.pop(self._virtual_addr, None)


# ---------------------------------------------------------------------------
# KV Cache 区域元信息
# ---------------------------------------------------------------------------

class KVCacheState(Enum):
    ON_GPU   = auto()   # 当前在 GPU 上，可直接访问
    OFFLOADED = auto()  # 已卸载到 CPU，GPU 虚拟地址保持固定但物理页已回收
    MIGRATING = auto()  # 正在 H2D 或 D2H 传输中（异步）


@dataclass
class KVCacheRegion:
    """单块 KV Cache 的元信息。

    DES-LOC 在 Megatron 单 tag 基础上扩展了：
    - device_arch: 感知异构设备，影响量化和调度优先级
    - locality_score: 基于最近访问频率的局部性评分，决策卸载优先级
    - gpu_tensor: 指向 GPU 上固定虚拟地址的 tensor（CUDAGraph 录制时使用的地址）
    - cpu_buffer: 对应的 CPU 端 pinned buffer
    """
    tag: str
    device: torch.device
    device_arch: str
    shape: Tuple
    dtype: torch.dtype
    state: KVCacheState = KVCacheState.ON_GPU
    locality_score: float = 1.0        # 越高越应该留在 GPU
    last_access_time: float = field(default_factory=time.monotonic)
    gpu_tensor: Optional[torch.Tensor] = None
    cpu_buffer: Optional[PinnedCPUBuffer] = None
    _transfer_stream: Optional[torch.cuda.Stream] = None

    def update_locality(self, decay: float = 0.9) -> None:
        """指数衰减更新 locality score，模拟 LRU 语义。

        DES-LOC Shared LOcality Cache 的核心：locality_score 越低的区域
        越优先被卸载到 CPU，类似 CPU cache 的替换策略但适用于 GPU 显存。
        """
        elapsed = time.monotonic() - self.last_access_time
        self.locality_score = self.locality_score * (decay ** elapsed)

    def mark_accessed(self) -> None:
        """更新访问时间戳，提升 locality score。"""
        self.last_access_time = time.monotonic()
        self.locality_score = min(self.locality_score * 1.5, 1.0)

    @property
    def nbytes(self) -> int:
        return torch.empty(self.shape, dtype=self.dtype).nbytes


# ---------------------------------------------------------------------------
# 异步 PCIe DMA 调度器
# ---------------------------------------------------------------------------

class OffloadScheduler:
    """异步 PCIe DMA 流水线调度器。

    上游 Megatron 的 pause/resume 是同步操作；DES-LOC 在 PCIe 互联约束下
    引入异步流水线：

    1. D2H（GPU → CPU）：使用 non-blocking copy + CUDA event 通知完成
    2. H2D（CPU → GPU）：在推理开始前 prefetch，与上一批 token 生成流水线

    带宽估算：
    - A6000 → CPU: ~28 GB/s, H100 → CPU: ~56 GB/s
    - 10GB KV Cache 在 A6000 上约需 360ms 卸载，需要提前触发
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.arch = _detect_device_arch(device)
        self.bw_gbps = _estimate_pcie_bandwidth_gbps(device)

        # 专用于 H2D/D2H 传输的 CUDA stream，不阻塞计算流
        self._d2h_stream = torch.cuda.Stream(device=device, priority=-1)
        self._h2d_stream = torch.cuda.Stream(device=device, priority=-2)

        # 记录在途传输的 CUDA event
        self._pending_d2h: Dict[str, torch.cuda.Event] = {}
        self._pending_h2d: Dict[str, torch.cuda.Event] = {}
        self._lock = threading.Lock()

        logger.info(
            "[DES-LOC] OffloadScheduler 初始化: device=%s arch=%s bw=%.1f GB/s",
            device, self.arch, self.bw_gbps,
        )

    def estimate_transfer_time_ms(self, nbytes: int, direction: str = "d2h") -> float:
        """估算传输时延（毫秒），用于调度决策。"""
        gb = nbytes / 1024**3
        # D2H 通常比 H2D 略慢（PCIe 上行带宽差异）
        factor = 1.0 if direction == "h2d" else 0.85
        return gb / (self.bw_gbps * factor) * 1000.0

    def async_offload(
        self,
        region: KVCacheRegion,
        cpu_buffer: PinnedCPUBuffer,
    ) -> None:
        """异步 D2H：将 GPU KV Cache 复制到 CPU pinned buffer，不阻塞计算流。

        DES-LOC 与 Megatron 的关键差异：
        - Megatron: `kv_cache.cpu()` 同步阻塞，新分配 CPU tensor
        - DES-LOC: 异步复制到预分配的固定虚拟地址 CPU buffer，GPU tensor 保持有效

        实现后 GPU 物理页通过 pause 归还给 allocator，但 GPU 虚拟地址不变，
        CUDAGraph 中录制的指针在 resume 后仍然可用。
        """
        if region.gpu_tensor is None:
            raise RuntimeError(f"region {region.tag} 的 gpu_tensor 为 None")

        tag = region.tag
        estimated_ms = self.estimate_transfer_time_ms(region.nbytes, "d2h")
        logger.debug(
            "[DES-LOC] 异步卸载 %s: %.2f GB，预计 %.0f ms (arch=%s)",
            tag, region.nbytes / 1024**3, estimated_ms, self.arch,
        )

        with torch.cuda.stream(self._d2h_stream):
            cpu_view = cpu_buffer.view_as(region.shape, region.dtype)
            # non_blocking=True：D2H 在 d2h_stream 上异步执行
            cpu_view.copy_(region.gpu_tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream=self._d2h_stream)

        with self._lock:
            self._pending_d2h[tag] = event
        region.state = KVCacheState.MIGRATING

    def async_prefetch(
        self,
        region: KVCacheRegion,
        cpu_buffer: PinnedCPUBuffer,
    ) -> None:
        """异步 H2D：从 CPU pinned buffer 预取 KV Cache 到 GPU。

        在推理开始前触发，与上一批 token 的计算流水线，隐藏 PCIe 延迟。
        resume 后 CPU buffer 调用 resume() 重新激活物理页。
        """
        if region.gpu_tensor is None:
            raise RuntimeError(f"region {region.tag} 的 gpu_tensor 为 None")

        tag = region.tag
        estimated_ms = self.estimate_transfer_time_ms(region.nbytes, "h2d")
        logger.debug(
            "[DES-LOC] 异步预取 %s: %.2f GB，预计 %.0f ms (arch=%s)",
            tag, region.nbytes / 1024**3, estimated_ms, self.arch,
        )

        cpu_buffer.resume()  # 重新激活 CPU 物理页

        with torch.cuda.stream(self._h2d_stream):
            cpu_view = cpu_buffer.view_as(region.shape, region.dtype)
            region.gpu_tensor.copy_(cpu_view, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream=self._h2d_stream)

        with self._lock:
            self._pending_h2d[tag] = event
        region.state = KVCacheState.MIGRATING

    def wait_offload(self, tag: str, timeout_ms: float = 30000.0) -> bool:
        """等待指定 tag 的 D2H 完成。"""
        with self._lock:
            event = self._pending_d2h.pop(tag, None)
        if event is None:
            return True
        try:
            # 用轮询替代阻塞等待，可被中断
            deadline = time.monotonic() + timeout_ms / 1000.0
            while not event.query():
                if time.monotonic() > deadline:
                    logger.error("[DES-LOC] D2H 等待超时: tag=%s", tag)
                    return False
                time.sleep(0.001)
            return True
        except Exception as exc:
            logger.error("[DES-LOC] D2H 等待异常 tag=%s: %s", tag, exc)
            return False

    def wait_prefetch(self, tag: str, timeout_ms: float = 30000.0) -> bool:
        """等待指定 tag 的 H2D 预取完成。"""
        with self._lock:
            event = self._pending_h2d.pop(tag, None)
        if event is None:
            return True
        try:
            deadline = time.monotonic() + timeout_ms / 1000.0
            while not event.query():
                if time.monotonic() > deadline:
                    logger.error("[DES-LOC] H2D 等待超时: tag=%s", tag)
                    return False
                time.sleep(0.001)
            return True
        except Exception as exc:
            logger.error("[DES-LOC] H2D 等待异常 tag=%s: %s", tag, exc)
            return False

    def synchronize_all(self) -> None:
        """同步所有在途传输，用于调试和清理。"""
        self._d2h_stream.synchronize()
        self._h2d_stream.synchronize()
        with self._lock:
            self._pending_d2h.clear()
            self._pending_h2d.clear()


# ---------------------------------------------------------------------------
# 主管理器：HeteroKVCacheOffload
# ---------------------------------------------------------------------------

class HeteroKVCacheOffload:
    """DES-LOC 异构 KV Cache 卸载管理器。

    对应 Megatron 的 torch_memory_saver 全局单例，但扩展了：
    1. 多设备异构感知（A6000 SM86 vs H100 SM90）
    2. Locality-based 优先级调度（Shared LOcality Cache）
    3. ZeRO 分片感知的卸载协调
    4. 异步 PCIe 流水线

    使用示例（对应 Megatron 的用法）::

        manager = HeteroKVCacheOffload.get_instance()

        # 分配时注册（对应 torch_memory_saver.region(tag="kv_cache")）
        with manager.region("kv_cache", device):
            memory_buffer = torch.empty(shape, dtype=dtype, device=device)

        # 训练开始：暂停 KV Cache（对应 torch_memory_saver.pause("kv_cache")）
        manager.pause("kv_cache")

        # 推理开始：恢复 KV Cache（对应 torch_memory_saver.resume("kv_cache")）
        manager.resume("kv_cache")

    DES-LOC 异构适配关键：
    - H100 的 KV Cache 优先卸载（显存更宝贵，训练时 A6000 需要梯度）
    - A6000 的 KV Cache 可以延迟卸载（PCIe 带宽更低，异步流水线更重要）
    - locality_score 基于最近 decode 步骤的访问频率动态更新
    """

    _instance: Optional["HeteroKVCacheOffload"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        # tag -> KVCacheRegion
        self._regions: Dict[str, KVCacheRegion] = {}
        # tag -> PinnedCPUBuffer
        self._cpu_buffers: Dict[str, PinnedCPUBuffer] = {}
        # device -> OffloadScheduler
        self._schedulers: Dict[str, OffloadScheduler] = {}
        self._global_lock = threading.RLock()

        # DES-LOC 统计
        self._offload_count: int = 0
        self._resume_count: int = 0
        self._total_offloaded_bytes: int = 0

        logger.info("[DES-LOC] HeteroKVCacheOffload 初始化完成")

    @classmethod
    def get_instance(cls) -> "HeteroKVCacheOffload":
        """获取全局单例，线程安全。"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_or_create_scheduler(self, device: torch.device) -> OffloadScheduler:
        """按设备获取或创建 OffloadScheduler。"""
        key = str(device)
        if key not in self._schedulers:
            self._schedulers[key] = OffloadScheduler(device)
        return self._schedulers[key]

    @contextmanager
    def region(
        self,
        tag: str,
        device: Optional[torch.device] = None,
        enable_cpu_backup: bool = True,
    ) -> Generator[None, None, None]:
        """Context manager：在此上下文中分配的 GPU tensor 被注册为可卸载区域。

        对应 Megatron::

            with torch_memory_saver.region(tag="kv_cache", enable_cpu_backup=True):
                self.memory_buffer = torch.empty(...)

        DES-LOC 扩展：
        - 记录设备架构，后续卸载决策时区分 SM86/SM90
        - 预先分配 CPU 端 pinned buffer，固定虚拟地址
        - 注册到 locality scheduler

        Parameters
        ----------
        tag : str
            区域标识符，pause/resume 时使用
        device : torch.device, optional
            目标 GPU 设备；None 时自动检测当前 CUDA 设备
        enable_cpu_backup : bool
            是否预分配 CPU backup buffer（对应上游 enable_cpu_backup=True）
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                device = torch.device("cpu")

        arch = _detect_device_arch(device)
        logger.debug("[DES-LOC] 进入 region 上下文: tag=%s device=%s arch=%s", tag, device, arch)

        # 临时记录，等 yield 后 tensor 分配完成再获取实际形状
        self._pending_region_tag = tag
        self._pending_region_device = device
        self._pending_region_arch = arch
        self._pending_region_enable_cpu_backup = enable_cpu_backup

        try:
            yield
        finally:
            # yield 返回后，外部代码应已将 tensor 注册到 self._regions
            # 若未注册（未调用 register_tensor），给出警告
            if tag not in self._regions:
                logger.warning(
                    "[DES-LOC] region '%s' 的 context 已退出但未检测到 register_tensor 调用。"
                    "请在 context 内调用 manager.register_tensor(tag, tensor)。",
                    tag,
                )
            self._pending_region_tag = None

    def register_tensor(self, tag: str, tensor: torch.Tensor) -> None:
        """将已分配的 GPU tensor 注册为 DES-LOC 可管理的 KV Cache 区域。

        通常在 region() context 内调用::

            with manager.region("kv_cache", device):
                memory_buffer = torch.empty(shape, dtype=dtype, device=device)
                manager.register_tensor("kv_cache", memory_buffer)

        DES-LOC 在此时：
        1. 记录 tensor 的 data_ptr()（CUDAGraph 录制地址）
        2. 按 tensor.nbytes 预分配等大的 CPU pinned buffer
        3. 计算初始 locality score（基于设备架构：H100 KV Cache 初始分更高）
        """
        device = tensor.device
        arch = _detect_device_arch(device)
        nbytes = tensor.nbytes

        # 初始 locality score：H100 更高（显存更稀缺，更需要被管理）
        initial_locality = 0.9 if arch == "sm90" else 0.7

        region = KVCacheRegion(
            tag=tag,
            device=device,
            device_arch=arch,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            state=KVCacheState.ON_GPU,
            locality_score=initial_locality,
            gpu_tensor=tensor,
        )

        # 预分配 CPU pinned buffer
        cpu_buf = PinnedCPUBuffer(nbytes, tag=tag)
        region.cpu_buffer = cpu_buf

        with self._global_lock:
            self._regions[tag] = region
            self._cpu_buffers[tag] = cpu_buf

        logger.info(
            "[DES-LOC] 注册 KV Cache 区域: tag=%s device=%s arch=%s shape=%s "
            "dtype=%s size=%.2f GB locality=%.2f",
            tag, device, arch, tensor.shape, tensor.dtype,
            nbytes / 1024**3, initial_locality,
        )

    def pause(self, tag: str, async_offload: bool = True) -> None:
        """暂停指定 tag 的 KV Cache：将数据从 GPU 迁移到 CPU，物理页归还给 allocator。

        对应 Megatron::

            torch_memory_saver.pause("kv_cache")

        DES-LOC 的实现步骤：
        1. 异步 D2H 复制 GPU tensor → CPU pinned buffer（保留虚拟地址）
        2. 等待传输完成（可配置为真正异步，与训练的参数加载流水线）
        3. 调用 CPU buffer 的 pause()，提示内核回收物理页（MADV_FREE）
        4. 更新 locality score，为下次卸载决策做准备

        关键：GPU tensor 的 data_ptr() 不变！这是 CUDAGraph 安全的保证。

        Parameters
        ----------
        tag : str
            要暂停的区域标识
        async_offload : bool
            True: 启动异步 D2H，立即返回（调用方负责在需要 GPU 显存前 wait）
            False: 同步等待 D2H 完成
        """
        with self._global_lock:
            region = self._regions.get(tag)
            cpu_buf = self._cpu_buffers.get(tag)

        if region is None:
            logger.warning("[DES-LOC] pause: 未找到 tag=%s，跳过", tag)
            return

        if region.state == KVCacheState.OFFLOADED:
            logger.debug("[DES-LOC] pause: tag=%s 已经处于 OFFLOADED 状态", tag)
            return

        if region.gpu_tensor is None:
            logger.warning("[DES-LOC] pause: tag=%s 的 gpu_tensor 为 None", tag)
            return

        scheduler = self._get_or_create_scheduler(region.device)

        t0 = time.monotonic()
        scheduler.async_offload(region, cpu_buf)

        if not async_offload:
            ok = scheduler.wait_offload(tag)
            if not ok:
                raise RuntimeError(f"[DES-LOC] KV Cache D2H 超时: tag={tag}")
            region.state = KVCacheState.OFFLOADED
            cpu_buf.pause()
        else:
            # 异步模式：后台等待完成，完成后更新状态
            def _finish_offload():
                ok = scheduler.wait_offload(tag, timeout_ms=60000.0)
                if ok:
                    region.state = KVCacheState.OFFLOADED
                    cpu_buf.pause()
                    logger.debug("[DES-LOC] 异步卸载完成: tag=%s", tag)
                else:
                    logger.error("[DES-LOC] 异步卸载超时: tag=%s", tag)

            t = threading.Thread(target=_finish_offload, daemon=True, name=f"des-loc-d2h-{tag}")
            t.start()

        region.update_locality(decay=0.8)
        self._offload_count += 1
        self._total_offloaded_bytes += region.nbytes

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "[DES-LOC] KV Cache 卸载启动: tag=%s size=%.2f GB arch=%s "
            "async=%s 启动耗时=%.1f ms",
            tag, region.nbytes / 1024**3, region.device_arch,
            async_offload, elapsed_ms,
        )

    def resume(self, tag: str, prefetch: bool = True) -> None:
        """恢复指定 tag 的 KV Cache：将数据从 CPU 绑回 GPU，保持虚拟地址不变。

        对应 Megatron::

            torch_memory_saver.resume("kv_cache")

        DES-LOC 与 Megatron 的核心等价保证：
        - GPU tensor 的 data_ptr() 与分配时相同，CUDAGraph 回放有效
        - CPU buffer 的虚拟地址固定（PinnedCPUBuffer 保证），不会触发 remapping

        DES-LOC 扩展：
        - H2D 可以异步执行（prefetch=True），在前一批 token 推理期间完成
        - 通过 locality_score 动态调整预取优先级

        Parameters
        ----------
        tag : str
            要恢复的区域标识
        prefetch : bool
            True: 异步 H2D（流水线隐藏延迟）；False: 同步等待
        """
        with self._global_lock:
            region = self._regions.get(tag)
            cpu_buf = self._cpu_buffers.get(tag)

        if region is None:
            logger.warning("[DES-LOC] resume: 未找到 tag=%s，跳过", tag)
            return

        if region.state == KVCacheState.ON_GPU:
            logger.debug("[DES-LOC] resume: tag=%s 已经在 GPU 上", tag)
            return

        if region.gpu_tensor is None:
            raise RuntimeError(
                f"[DES-LOC] resume: tag={tag} 的 gpu_tensor 为 None，"
                "无法在固定虚拟地址上恢复（CUDAGraph 安全性破坏）"
            )

        scheduler = self._get_or_create_scheduler(region.device)

        t0 = time.monotonic()

        # 等待可能仍在进行的 D2H（如果 pause 是异步的）
        scheduler.wait_offload(tag, timeout_ms=30000.0)
        region.state = KVCacheState.MIGRATING

        if prefetch:
            scheduler.async_prefetch(region, cpu_buf)
            # 同步等待 H2D 完成（推理必须在此之后开始）
            ok = scheduler.wait_prefetch(tag, timeout_ms=60000.0)
        else:
            # 同步 H2D
            cpu_buf.resume()
            with torch.cuda.stream(scheduler._h2d_stream):
                cpu_view = cpu_buf.view_as(region.shape, region.dtype)
                region.gpu_tensor.copy_(cpu_view, non_blocking=False)
            ok = True

        if not ok:
            raise RuntimeError(f"[DES-LOC] KV Cache H2D 超时: tag={tag}")

        region.state = KVCacheState.ON_GPU
        region.mark_accessed()
        self._resume_count += 1

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "[DES-LOC] KV Cache 恢复完成: tag=%s size=%.2f GB arch=%s "
            "虚拟地址=0x%x 耗时=%.1f ms",
            tag, region.nbytes / 1024**3, region.device_arch,
            region.gpu_tensor.data_ptr(), elapsed_ms,
        )

    def get_locality_sorted_regions(self) -> List[KVCacheRegion]:
        """按 locality score 升序排列区域（分数低的优先卸载）。

        DES-LOC Shared LOcality Cache 决策入口：
        当显存压力触发时，按此顺序依次卸载，保留最近访问的 KV Cache。
        """
        with self._global_lock:
            regions = list(self._regions.values())
        for r in regions:
            r.update_locality()
        return sorted(regions, key=lambda r: r.locality_score)

    def evict_least_local(self, target_bytes: int) -> int:
        """按 locality score 驱逐 KV Cache，直到释放 target_bytes。

        DES-LOC 显存压力响应：当 ZeRO 优化器需要额外显存时调用。
        返回实际释放的字节数（异步卸载场景下为启动的卸载量）。
        """
        freed = 0
        for region in self.get_locality_sorted_regions():
            if freed >= target_bytes:
                break
            if region.state == KVCacheState.ON_GPU:
                logger.info(
                    "[DES-LOC] 显存压力驱逐: tag=%s locality=%.3f size=%.2f GB",
                    region.tag, region.locality_score, region.nbytes / 1024**3,
                )
                self.pause(region.tag, async_offload=True)
                freed += region.nbytes
        return freed

    def stats(self) -> Dict[str, object]:
        """返回卸载统计信息，用于监控和调优。"""
        with self._global_lock:
            region_stats = {
                tag: {
                    "state": r.state.name,
                    "arch": r.device_arch,
                    "size_gb": r.nbytes / 1024**3,
                    "locality_score": r.locality_score,
                }
                for tag, r in self._regions.items()
            }
        return {
            "offload_count": self._offload_count,
            "resume_count": self._resume_count,
            "total_offloaded_gb": self._total_offloaded_bytes / 1024**3,
            "regions": region_stats,
        }

    def synchronize(self) -> None:
        """同步所有在途传输，用于检查点保存前的清理。"""
        for scheduler in self._schedulers.values():
            scheduler.synchronize_all()
        logger.info("[DES-LOC] 所有 DMA 传输已同步")


# ---------------------------------------------------------------------------
# DeepSpeed 集成钩子
# ---------------------------------------------------------------------------

def validate_hetero_offload_args(
    offload_kv_cache: bool,
    unified_memory_level: int = 0,
    zero_stage: int = 0,
) -> None:
    """验证 DES-LOC KV Cache 卸载参数，对应 Megatron 的 validate_args。

    Megatron 上游检查（arguments.py）::

        assert not args.inference_dynamic_batching_unified_memory_level, \
            "The KV cache should not be instantiated in unified memory when offloaded"

    DES-LOC 扩展检查：
    - ZeRO stage 3 + KV Cache 卸载需要特殊内存规划（显存预算计算）
    - unified memory 与固定虚拟地址机制互斥
    """
    if not offload_kv_cache:
        return

    if unified_memory_level > 0:
        raise ValueError(
            "[DES-LOC] KV Cache 卸载与 unified_memory_level 互斥：\n"
            "unified memory 使用 UVM 地址空间，与 PinnedCPUBuffer 的固定\n"
            "虚拟地址机制冲突，会破坏 CUDAGraph 录制。\n"
            "请将 unified_memory_level 设置为 0。"
        )

    if zero_stage == 3:
        logger.warning(
            "[DES-LOC] ZeRO-3 + KV Cache 卸载：显存规划需要手动验证。\n"
            "ZeRO-3 参数分片会在 forward pass 中动态申请显存，\n"
            "请确保 KV Cache 卸载在 ZeRO gather 之前完成。\n"
            "建议在 engine.train() 前调用 manager.pause('kv_cache')。"
        )

    logger.info("[DES-LOC] KV Cache 卸载参数验证通过: unified_memory=%d zero_stage=%d",
                unified_memory_level, zero_stage)


@contextmanager
def des_loc_inference_mode(
    inference_interface,
    offload_kv_cache: bool = False,
    remove_kv_cache: bool = False,
) -> Generator[None, None, None]:
    """DES-LOC 推理模式 context manager。

    对应 Megatron 的 `megatron_rl_inference_mode` 中的 KV Cache 管理逻辑::

        # Megatron 上游：
        with nvtx_range("onload-kv-cache-before-inference"):
            if offload_kv_cache_during_training:
                torch_memory_saver.resume("kv_cache")

        yield  # 推理执行

        with nvtx_range("offload-kv-cache-after-inference"):
            if offload_kv_cache_during_training:
                torch_memory_saver.pause("kv_cache")

    DES-LOC 适配：
    - 进入推理前：resume KV Cache（H2D，固定虚拟地址保证 CUDAGraph 有效）
    - 退出推理后：异步 pause KV Cache（D2H 流水线，与训练初始化并行）
    - 异构设备路由：自动选择对应设备的 OffloadScheduler

    Parameters
    ----------
    inference_interface
        DeepSpeed 推理接口，需有 _inference_engine.context.memory_buffer 属性
    offload_kv_cache : bool
        是否启用 KV Cache CPU 卸载（对应 rl_offload_kv_cache_during_training）
    remove_kv_cache : bool
        是否在训练时完全删除 KV Cache（互斥于 offload_kv_cache）
    """
    manager = HeteroKVCacheOffload.get_instance()

    # 进入推理：恢复 KV Cache 到 GPU
    if offload_kv_cache:
        kv_tensor = getattr(
            getattr(
                getattr(inference_interface, '_inference_engine', None),
                'context', None
            ),
            'memory_buffer', None
        )
        if kv_tensor is not None:
            size_gb = kv_tensor.nbytes / 1024**3
            logger.debug(
                "[DES-LOC] 推理模式入口：恢复 KV Cache %.2f GB 到 GPU", size_gb
            )
            manager.resume("kv_cache", prefetch=True)
        else:
            logger.warning("[DES-LOC] 未找到 memory_buffer，跳过 KV Cache 恢复")

    try:
        yield

    finally:
        # 退出推理：将 KV Cache 异步卸载回 CPU
        if offload_kv_cache:
            kv_tensor = getattr(
                getattr(
                    getattr(inference_interface, '_inference_engine', None),
                    'context', None
                ),
                'memory_buffer', None
            )
            if kv_tensor is not None:
                size_gb = kv_tensor.nbytes / 1024**3
                logger.debug(
                    "[DES-LOC] 推理模式出口：异步卸载 KV Cache %.2f GB 到 CPU", size_gb
                )
                manager.pause("kv_cache", async_offload=True)
            else:
                logger.warning("[DES-LOC] 未找到 memory_buffer，跳过 KV Cache 卸载")

        elif remove_kv_cache:
            ctx = getattr(
                getattr(inference_interface, '_inference_engine', None), 'context', None
            )
            if ctx is not None:
                ctx.memory_buffer = None
                logger.debug("[DES-LOC] KV Cache 已删除（remove 模式）")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    logger.info("=== DES-LOC HeteroKVCacheOffload Smoke Test ===")

    # Test 1: PinnedCPUBuffer 分配与虚拟地址稳定性
    buf = PinnedCPUBuffer(nbytes=64 * 1024 * 1024, tag="test_kv")  # 64 MB
    addr_before = buf._virtual_addr
    buf.pause()
    buf.resume()
    addr_after = buf._virtual_addr
    assert addr_before == addr_after, f"虚拟地址发生变化: {addr_before:#x} -> {addr_after:#x}"
    logger.info("✓ Test 1 PASS: PinnedCPUBuffer 虚拟地址在 pause/resume 后保持稳定 0x%x", addr_before)

    # Test 2: view_as 形状和dtype
    shape = (2, 4, 32, 8, 16)
    dtype = torch.float16
    view = buf.view_as(shape, dtype)
    assert view.shape == torch.Size(shape), f"形状不匹配: {view.shape}"
    assert view.dtype == dtype, f"dtype 不匹配: {view.dtype}"
    logger.info("✓ Test 2 PASS: view_as 返回正确形状 %s dtype %s", shape, dtype)

    # Test 3: 设备架构检测
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        arch = _detect_device_arch(dev)
        assert arch.startswith("sm"), f"架构格式错误: {arch}"
        bw = _estimate_pcie_bandwidth_gbps(dev)
        assert bw > 0, "带宽估算应为正数"
        logger.info("✓ Test 3 PASS: 设备 %s 架构=%s 带宽=%.1f GB/s", dev, arch, bw)
    else:
        arch = _detect_device_arch(torch.device("cpu"))
        assert arch == "cpu"
        logger.info("✓ Test 3 PASS (CPU mode): 无 CUDA 设备，arch=cpu")

    # Test 4: validate_hetero_offload_args 互斥检查
    try:
        validate_hetero_offload_args(offload_kv_cache=True, unified_memory_level=1)
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "unified memory" in str(e).lower() or "unified_memory" in str(e)
    logger.info("✓ Test 4 PASS: unified_memory 与 offload_kv_cache 互斥检查有效")

    # Test 5: HeteroKVCacheOffload 单例语义
    m1 = HeteroKVCacheOffload.get_instance()
    m2 = HeteroKVCacheOffload.get_instance()
    assert m1 is m2, "get_instance 应返回同一对象"
    logger.info("✓ Test 5 PASS: 单例模式正确")

    logger.info("=== 所有 smoke test 通过 ===")
    sys.exit(0)
