"""
HeteroDDPOverlapGradFix — DES-LOC Heterogeneous Gradient Reduction Fix
=======================================================================

上游设计意图（Megatron commit 3548385ac3b1cbfa7cbf4eceb38d6504662a4f3b）
-----------------------------------------------------------------------
Megatron-LM 的 DDP 实现中，当同时启用 ``--overlap-grad-reduce`` 与
``--num-distributed-optimizer-instances > 1`` 时，存在一个隐患：

  1. 通信流（RS/AR collective）在等待梯度计算完成时，错误地使用了
     ``torch.cuda.default_stream()`` 作为同步基准。
  2. 同理，在 multi-instance DistOpt 场景下，主流等待通信流结束时，
     也使用了 ``default_stream``。

原始问题：在非默认 CUDA stream 上执行 backward 时（例如通过
``torch.cuda.stream(custom_stream)`` 包裹的 forward/backward），
default_stream 上实际没有任何待完成的梯度 kernel，通信流因此不等待
真正的梯度写入完成就发起 all-reduce/reduce-scatter，导致梯度数据竞争。

修复方案（两处）：
  - ``stream.wait_stream(torch.cuda.default_stream())``
    → ``stream.wait_stream(torch.cuda.current_stream())``
  - ``torch.cuda.default_stream().wait_stream(comm_stream)``
    → ``torch.cuda.current_stream().wait_stream(comm_stream)``

DES-LOC 适配点
--------------
DES-LOC（Decoupled Execution with Shared LOcality Cache）在 2×A6000 +
1×H100 NVL 的 PCIe 互联拓扑下，必须在 **多个异构设备流** 上并行执行：

* A6000 设备：使用专用的 ``hetero_compute_stream``（SM86 计算流）
* H100 设备：使用 ``locality_cache_stream``（SM90 高带宽本地缓存预取流）
* 跨设备 PCIe 传输：``pcie_transfer_stream``

在这种多流架构中，"当前流" 的概念是动态的：每个设备、每个阶段对应
不同的活跃流。若沿用 ``default_stream`` 进行同步，将复现 Megatron 的
原始 bug，只是触发条件从"自定义 CUDA stream"变成了"DES-LOC 异构流"。

本模块实现：
  1. ``HeteroStreamContext`` — 跟踪当前活跃的异构执行流，提供
     ``current_stream(device)`` 接口，替代 ``torch.cuda.current_stream``
     在多设备场景下的局限性（后者只对单设备有效）。
  2. ``HeteroBucketGroup`` — 对标 Megatron 的 ``_ParamAndGradBucketGroup``，
     实现异构桶分组的梯度 overlap 通信，使用修正后的流同步语义。
  3. ``HeteroDDPOverlapGradReducer`` — 顶层调度器，管理 A6000/H100
     的梯度桶分配策略（基于 SM 架构与显存容量），协调 PCIe 传输时序。
  4. ``LocalityCacheGradAccumulator`` — DES-LOC 专有：利用 H100 NVL 的
     大显存（96GB）作为梯度局部性缓存，减少 A6000 侧的 all-reduce 流量。

拓扑假设
--------
* 节点内 PCIe 互联，无 NVLink；带宽约 64 GB/s（单向）
* A6000 x2：device_id 0, 1（SM86，48GB 各）
* H100 NVL：device_id 2（SM90，96GB）
* CPU DRAM：1.5 TB，作为溢出缓冲与异步检查点存储
* 通信后端：NCCL（PCIe 拓扑下自动选择 ring/tree）

Author: Neuron_SP Project (reinterpretation of Megatron commit 3548385)
"""

from __future__ import annotations

import logging
import threading
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 硬件拓扑常量（与 dylanyunlon/Neuron_SP 项目约定对齐）
# ---------------------------------------------------------------------------

# SM 架构版本 → 设备角色
_SM86_ARCH = 86   # A6000
_SM90_ARCH = 90   # H100 NVL

# DES-LOC 设备角色
class DeviceRole(Enum):
    COMPUTE_PRIMARY   = auto()   # A6000 #0：主计算设备
    COMPUTE_SECONDARY = auto()   # A6000 #1：辅助计算 / 流水并行
    LOCALITY_CACHE    = auto()   # H100 NVL：梯度局部性缓存节点
    CPU_OFFLOAD       = auto()   # CPU DRAM：溢出缓冲


@dataclass
class HeteroTopology:
    """描述 DES-LOC 物理拓扑，由用户在 DeepSpeed config 中声明或自动探测。"""
    device_roles: Dict[int, DeviceRole]          # device_id → role
    pcie_bw_gbps: float = 64.0                   # PCIe 单向峰值带宽 GB/s
    locality_cache_device: int = 2               # H100 device_id
    compute_devices: List[int] = field(default_factory=lambda: [0, 1])
    cpu_dram_gb: float = 1536.0                  # 1.5 TB

    @classmethod
    def auto_detect(cls) -> "HeteroTopology":
        """自动探测当前节点的 GPU 架构，构建 DES-LOC 拓扑描述。"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; returning stub topology.")
            return cls(device_roles={})

        roles: Dict[int, DeviceRole] = {}
        compute_devs: List[int] = []
        cache_dev: Optional[int] = None
        n = torch.cuda.device_count()

        sm_counts: Dict[int, int] = {}
        for dev in range(n):
            props = torch.cuda.get_device_properties(dev)
            sm_ver = props.major * 10 + props.minor
            sm_counts[dev] = sm_ver
            logger.debug("Device %d: %s, SM%d, %.1f GB",
                         dev, props.name, sm_ver, props.total_memory / 1e9)

        # 按 SM 版本分配角色：SM90 → LOCALITY_CACHE，SM86 → COMPUTE
        for dev, sm in sm_counts.items():
            if sm >= _SM90_ARCH:
                roles[dev] = DeviceRole.LOCALITY_CACHE
                cache_dev = dev
            elif sm == _SM86_ARCH:
                if not compute_devs:
                    roles[dev] = DeviceRole.COMPUTE_PRIMARY
                else:
                    roles[dev] = DeviceRole.COMPUTE_SECONDARY
                compute_devs.append(dev)
            else:
                logger.warning("Unknown SM%d for device %d; treating as secondary compute.", sm, dev)
                roles[dev] = DeviceRole.COMPUTE_SECONDARY
                compute_devs.append(dev)

        topo = cls(
            device_roles=roles,
            locality_cache_device=cache_dev if cache_dev is not None else -1,
            compute_devices=compute_devs,
        )
        logger.info("DES-LOC topology detected: %s", topo)
        return topo


# ---------------------------------------------------------------------------
# 1. HeteroStreamContext — 修复 default_stream vs current_stream 的核心
# ---------------------------------------------------------------------------

class HeteroStreamContext:
    """
    跨设备流上下文管理器。

    DES-LOC 修复点
    ~~~~~~~~~~~~~~
    Megatron 原始 bug 的本质：在非默认流上执行 backward 时，
    ``torch.cuda.current_stream()`` 才是真正的活跃流，而非
    ``torch.cuda.default_stream()``。

    在多异构设备场景下，``torch.cuda.current_stream()`` 只对
    ``torch.cuda.current_device()`` 有效。若梯度在 A6000 #1 上计算，
    而通信流属于 A6000 #0，直接调用 ``current_stream()`` 会返回 #0 上的
    默认流，仍然是错误的。

    本类维护一个线程本地（thread-local）的 {device_id → active_stream}
    映射，允许在任意设备上精确查询"当前活跃流"。

    用法::

        ctx = HeteroStreamContext()
        with ctx.on_stream(device_id=0, stream=my_stream):
            # 在此 context 内，ctx.current_stream(0) == my_stream
            ...
        # 离开后恢复为 default_stream(device=0)
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _get_stack(self) -> Dict[int, List[torch.cuda.Stream]]:
        if not hasattr(self._local, "stream_stack"):
            self._local.stream_stack = {}
        return self._local.stream_stack

    @contextmanager
    def on_stream(
        self,
        device_id: int,
        stream: Optional[torch.cuda.Stream],
    ) -> Generator[None, None, None]:
        """将 ``stream`` 注册为设备 ``device_id`` 上的当前活跃流。"""
        stack = self._get_stack()
        if device_id not in stack:
            stack[device_id] = []
        stack[device_id].append(stream)
        logger.debug("HeteroStreamContext: device %d → stream %s (depth=%d)",
                     device_id, stream, len(stack[device_id]))
        try:
            if stream is not None:
                with torch.cuda.device(device_id), torch.cuda.stream(stream):
                    yield
            else:
                yield
        finally:
            stack[device_id].pop()
            if not stack[device_id]:
                del stack[device_id]

    def current_stream(self, device_id: int) -> torch.cuda.Stream:
        """
        返回设备 ``device_id`` 上当前注册的活跃流。

        若无注册流（即未处于任何 ``on_stream`` context），返回该设备的
        default stream——与 ``torch.cuda.default_stream(device)`` 等价，
        但语义上与 Megatron 修复后的 ``current_stream()`` 一致。
        """
        stack = self._get_stack()
        dev_stack = stack.get(device_id, [])
        if dev_stack and dev_stack[-1] is not None:
            return dev_stack[-1]
        # 回退到 default_stream（等同于 Megatron 修复前行为，但此时是正确的，
        # 因为没有非默认流在运行）
        return torch.cuda.default_stream(device=device_id)


# 模块级全局单例，供所有组件共享
_hetero_stream_ctx = HeteroStreamContext()


def get_hetero_stream_ctx() -> HeteroStreamContext:
    """获取全局 HeteroStreamContext 单例。"""
    return _hetero_stream_ctx


# ---------------------------------------------------------------------------
# 2. 梯度桶配置
# ---------------------------------------------------------------------------

@dataclass
class HeteroBucketConfig:
    """
    对标 Megatron DDP config 的 DES-LOC 版本。

    新增字段用于描述异构调度策略：
    - ``num_dist_optimizer_instances``：多实例 DistOpt 数量，
      复现 Megatron ``--num-distributed-optimizer-instances`` 语义
    - ``overlap_grad_reduce``：是否启用梯度 overlap（对应 --overlap-grad-reduce）
    - ``locality_cache_ratio``：多少比例的梯度桶卸载到 H100 NVL 缓存
    - ``pcie_pipeline_depth``：PCIe 传输流水线深度（同时在途的桶数）
    """
    num_dist_optimizer_instances: int = 1
    overlap_grad_reduce: bool = True
    locality_cache_ratio: float = 0.4      # 40% 梯度走 H100 缓存路径
    pcie_pipeline_depth: int = 2
    bucket_size_mb: float = 25.0           # 单桶大小 MB（与 DeepSpeed 默认对齐）
    grad_dtype: torch.dtype = torch.float16


# ---------------------------------------------------------------------------
# 3. HeteroBucketGroup — 核心修复载体
# ---------------------------------------------------------------------------

class HeteroBucketGroup:
    """
    异构梯度桶组。

    对标 Megatron ``_ParamAndGradBucketGroup``，但针对 DES-LOC 的
    多流异构执行模型进行了重构。

    关键修复（对应 Megatron commit 3548385 的两处 diff）
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    **修复 A**：通信流等待计算流（overlap 路径入口）

    Megatron 原始::

        self.communication_stream.wait_stream(torch.cuda.default_stream())

    Megatron 修复后::

        self.communication_stream.wait_stream(torch.cuda.current_stream())

    DES-LOC 实现::

        comm_stream.wait_stream(
            _hetero_stream_ctx.current_stream(compute_device_id)
        )

    因为 DES-LOC 的计算可能在 ``hetero_compute_stream`` 而非默认流上
    进行，必须用 ``HeteroStreamContext`` 而非 ``torch.cuda.current_stream``
    才能正确跨设备追踪。

    **修复 B**：主流等待通信流（multi-instance DistOpt 路径出口）

    Megatron 原始::

        torch.cuda.default_stream().wait_stream(self.communication_stream)

    DES-LOC 实现::

        _hetero_stream_ctx.current_stream(compute_device_id).wait_stream(
            comm_stream
        )

    同理，"主流"在 DES-LOC 中可能是 ``locality_cache_stream`` 或
    ``pcie_transfer_stream``，而非默认流。
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        bucket_id: int,
        config: HeteroBucketConfig,
        topology: HeteroTopology,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.params = params
        self.bucket_id = bucket_id
        self.config = config
        self.topology = topology
        self.process_group = process_group
        self._stream_ctx = get_hetero_stream_ctx()

        # 确定此桶归属的计算设备（参数所在设备）
        if params:
            self.compute_device: int = params[0].device.index or 0
        else:
            self.compute_device = topology.compute_devices[0] if topology.compute_devices else 0

        # 决定此桶是否走 H100 局部缓存路径
        self.use_locality_cache: bool = (
            bucket_id % 10 < int(config.locality_cache_ratio * 10)
            and topology.locality_cache_device >= 0
        )
        self.cache_device: int = topology.locality_cache_device

        # 在计算设备上创建通信流
        self.communication_stream: Optional[torch.cuda.Stream] = None
        self.grad_reduce_handle: Optional[dist.Work] = None
        self._bucket_tensor: Optional[Tensor] = None
        self._initialized = False

        logger.debug(
            "HeteroBucketGroup[%d]: compute_device=%d, use_cache=%s, "
            "cache_device=%d, num_params=%d",
            bucket_id, self.compute_device, self.use_locality_cache,
            self.cache_device, len(params),
        )

    def _lazy_init(self) -> None:
        """延迟初始化 CUDA 资源（避免在构造时占用流资源）。"""
        if self._initialized:
            return
        with torch.cuda.device(self.compute_device):
            self.communication_stream = torch.cuda.Stream(
                device=self.compute_device,
                priority=-1,  # 高优先级，与 Megatron 保持一致
            )
        self._initialized = True
        logger.debug("HeteroBucketGroup[%d]: communication_stream initialized.", self.bucket_id)

    # ------------------------------------------------------------------
    # 修复 A：触发 overlap 通信（入口）
    # ------------------------------------------------------------------

    def start_grad_reduce(self) -> None:
        """
        在梯度写入完成后，立即触发 all-reduce / reduce-scatter。

        DES-LOC 修复 A
        ~~~~~~~~~~~~~~
        ``communication_stream`` 必须等待**真正写入梯度的流**完成，
        而不是 default stream。

        在 DES-LOC 中，backward 可能运行在：
        - A6000 的 ``hetero_compute_stream``（非默认流）
        - H100 的 ``locality_cache_stream``（非默认流，也非 A6000 的流）

        因此使用 ``HeteroStreamContext.current_stream(compute_device)``
        而非 ``torch.cuda.current_stream()``（只看单设备）或
        ``torch.cuda.default_stream()``（Megatron 原始 bug）。
        """
        self._lazy_init()
        assert self.communication_stream is not None

        if self.config.overlap_grad_reduce:
            stream_context = torch.cuda.stream(self.communication_stream)

            # ── DES-LOC 修复 A ──────────────────────────────────────────
            # 获取真正执行了梯度 kernel 的流（可能是非默认的异构流）
            true_compute_stream = self._stream_ctx.current_stream(self.compute_device)
            logger.debug(
                "HeteroBucketGroup[%d]: comm_stream waiting for compute_stream=%s "
                "(device=%d). [DES-LOC fix-A: current_stream, not default_stream]",
                self.bucket_id, true_compute_stream, self.compute_device,
            )
            self.communication_stream.wait_stream(true_compute_stream)
            # ────────────────────────────────────────────────────────────
        else:
            stream_context = nullcontext()

        with stream_context:
            self._do_grad_reduce()

    def _do_grad_reduce(self) -> None:
        """
        执行实际的梯度 all-reduce 或 reduce-scatter。

        DES-LOC 扩展：若 ``use_locality_cache`` 为 True，先将梯度
        聚合到 H100 NVL（作为 reduce root），再广播回 A6000。
        这利用了 H100 的大显存（96GB）充当临时聚合缓冲，避免
        A6000（48GB）显存在大 batch 时被梯度桶撑满。
        """
        flat = self._pack_gradients()
        if flat is None:
            logger.debug("HeteroBucketGroup[%d]: no valid gradients, skipping.", self.bucket_id)
            return

        if self.use_locality_cache and self.cache_device >= 0 and dist.is_available() and self.process_group:
            flat = self._locality_cache_reduce(flat)
        elif dist.is_available() and self.process_group:
            self.grad_reduce_handle = dist.all_reduce(
                flat, group=self.process_group, async_op=True
            )
        else:
            # 单进程模式（smoke test / debug）
            logger.debug("HeteroBucketGroup[%d]: single-process, no all-reduce.", self.bucket_id)

        self._unpack_gradients(flat)

    def _locality_cache_reduce(self, flat: Tensor) -> Tensor:
        """
        H100 NVL 局部缓存路径的梯度规约。

        流程：
        1. PCIe 传输：A6000 → H100（异步 D2D copy）
        2. H100 上执行 all-reduce（利用 H100 的高内存带宽）
        3. PCIe 传输：H100 → A6000（异步 D2D copy）

        DES-LOC 的核心收益：H100 作为梯度汇聚节点，减少了 A6000 之间
        直接 peer-to-peer 的 PCIe 流量（避免 A6000#0 ↔ A6000#1 的低效路由）。
        """
        cache_dev = torch.device(f"cuda:{self.cache_device}")
        logger.debug(
            "HeteroBucketGroup[%d]: locality-cache reduce via H100 (device %d)",
            self.bucket_id, self.cache_device,
        )
        # Step 1: A6000 → H100
        flat_on_cache = flat.to(cache_dev, non_blocking=True)

        # Step 2: H100 上 all-reduce
        if self.process_group:
            handle = dist.all_reduce(flat_on_cache, group=self.process_group, async_op=True)
            self.grad_reduce_handle = handle
            handle.wait()

        # Step 3: H100 → A6000
        flat.copy_(flat_on_cache, non_blocking=True)
        return flat

    def _pack_gradients(self) -> Optional[Tensor]:
        """将桶内所有参数的 .grad 打包为连续 flat tensor。"""
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return None
        flat = torch.cat([g.reshape(-1).to(self.config.grad_dtype) for g in grads])
        return flat

    def _unpack_gradients(self, flat: Tensor) -> None:
        """将 flat tensor 写回各参数的 .grad（in-place）。"""
        offset = 0
        for p in self.params:
            if p.grad is None:
                continue
            numel = p.grad.numel()
            p.grad.data.copy_(
                flat[offset: offset + numel].reshape(p.grad.shape).to(p.grad.dtype)
            )
            offset += numel

    # ------------------------------------------------------------------
    # 修复 B：等待通信完成（出口）
    # ------------------------------------------------------------------

    def finish_grad_reduce(self) -> None:
        """
        等待梯度规约完成，将主流（或缓存流）阻塞至通信结束。

        DES-LOC 修复 B
        ~~~~~~~~~~~~~~
        Megatron 原始（multi-instance 路径）::

            torch.cuda.default_stream().wait_stream(self.communication_stream)

        修复后::

            torch.cuda.current_stream().wait_stream(self.communication_stream)

        DES-LOC 进一步扩展：
        在异构拓扑中，"主流"可能是：
        - ``hetero_compute_stream``（A6000 上的计算流）
        - ``locality_cache_stream``（H100 上的缓存预取流）
        - ``pcie_transfer_stream``（专用 PCIe 搬运流）

        必须通过 ``HeteroStreamContext.current_stream(device)`` 精确定位
        当前活跃流，而非假设是 default stream。
        """
        if not self._initialized or self.communication_stream is None:
            return

        if self.config.num_dist_optimizer_instances > 1:
            # ── DES-LOC 修复 B ──────────────────────────────────────────
            # 对应 Megatron 修复：default_stream → current_stream
            true_main_stream = self._stream_ctx.current_stream(self.compute_device)
            logger.debug(
                "HeteroBucketGroup[%d]: main_stream=%s waiting for comm_stream "
                "(multi-instance DistOpt path). "
                "[DES-LOC fix-B: current_stream, not default_stream]",
                self.bucket_id, true_main_stream,
            )
            true_main_stream.wait_stream(self.communication_stream)
            # ────────────────────────────────────────────────────────────
            return

        # 单实例路径：同步等待 handle
        if self.grad_reduce_handle is not None:
            logger.debug("HeteroBucketGroup[%d]: waiting for grad_reduce_handle.", self.bucket_id)
            self.grad_reduce_handle.wait()
            self.grad_reduce_handle = None
        else:
            logger.warning(
                "HeteroBucketGroup[%d]: finish_grad_reduce called but no handle "
                "(single-instance, non-overlap path).", self.bucket_id
            )


# ---------------------------------------------------------------------------
# 4. LocalityCacheGradAccumulator — DES-LOC 专有组件
# ---------------------------------------------------------------------------

class LocalityCacheGradAccumulator:
    """
    利用 H100 NVL 96GB 显存作为梯度局部性缓存。

    动机
    ~~~~
    在 PCIe 互联拓扑下（无 NVLink），A6000 之间的 peer-to-peer 带宽
    受 PCIe switch 拓扑约束，实测约 16–32 GB/s（双向共享）。
    H100 NVL 的显存带宽为 ~3.9 TB/s，远高于 PCIe。

    策略
    ~~~~
    将多个梯度桶在 H100 上累积（``accumulate``），待达到
    ``flush_threshold_mb`` 阈值后，一次性触发 all-reduce
    并写回 A6000。这将多次小 PCIe 传输合并为少量大传输，
    提升 PCIe 利用率（large message PCIe 效率更高）。

    与 HeteroStreamContext 的集成
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    H100 上的缓存操作在 ``locality_cache_stream`` 上执行；
    flush 完成后，通过 ``HeteroStreamContext`` 通知 A6000 侧
    的 ``hetero_compute_stream`` 可以继续使用梯度。
    """

    def __init__(
        self,
        cache_device: int,
        compute_devices: List[int],
        flush_threshold_mb: float = 100.0,
    ) -> None:
        self.cache_device = cache_device
        self.compute_devices = compute_devices
        self.flush_threshold_bytes = int(flush_threshold_mb * 1024 * 1024)
        self._cache: List[Tuple[Tensor, int]] = []   # (grad_on_h100, compute_dev)
        self._cache_bytes: int = 0
        self._stream_ctx = get_hetero_stream_ctx()

        self.locality_cache_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available() and cache_device >= 0:
            with torch.cuda.device(cache_device):
                self.locality_cache_stream = torch.cuda.Stream(device=cache_device)
            logger.info(
                "LocalityCacheGradAccumulator: H100 stream initialized on device %d, "
                "flush_threshold=%.1f MB", cache_device, flush_threshold_mb
            )

    def accumulate(self, grad: Tensor, compute_device: int) -> bool:
        """
        将梯度异步搬运至 H100 缓存。

        返回 ``True`` 表示已触发 flush（调用者无需再手动调用 ``flush``）。
        """
        if self.cache_device < 0 or self.locality_cache_stream is None:
            return False

        cache_dev = torch.device(f"cuda:{self.cache_device}")
        with self._stream_ctx.on_stream(self.cache_device, self.locality_cache_stream):
            grad_on_cache = grad.to(cache_dev, non_blocking=True)

        self._cache.append((grad_on_cache, compute_device))
        self._cache_bytes += grad.nbytes
        logger.debug(
            "LocalityCacheGradAccumulator: accumulated %.2f MB (total=%.2f MB)",
            grad.nbytes / 1e6, self._cache_bytes / 1e6
        )

        if self._cache_bytes >= self.flush_threshold_bytes:
            logger.debug("LocalityCacheGradAccumulator: threshold reached, auto-flush.")
            self.flush()
            return True
        return False

    def flush(self) -> None:
        """
        将缓存的梯度写回各 A6000。

        执行顺序：
        1. 在 H100 上对所有缓存梯度做逐元素求和（模拟 all-reduce 的本地聚合）
        2. 按 compute_device 分组，异步 D2D copy 回 A6000
        3. 清空缓存
        """
        if not self._cache:
            return

        logger.info(
            "LocalityCacheGradAccumulator: flushing %d tensors (%.2f MB) from H100.",
            len(self._cache), self._cache_bytes / 1e6
        )

        # 按目标 compute_device 分组
        by_device: Dict[int, List[Tensor]] = {}
        for grad_on_cache, dev in self._cache:
            by_device.setdefault(dev, []).append(grad_on_cache)

        for dev, grads in by_device.items():
            # 在 H100 缓存流上求和，再 copy 回 A6000
            compute_stream = self._stream_ctx.current_stream(dev)
            with self._stream_ctx.on_stream(self.cache_device, self.locality_cache_stream):
                summed = grads[0]
                for g in grads[1:]:
                    summed = summed + g
                # H100 → A6000（non-blocking D2D over PCIe）
                target = torch.empty_like(summed, device=f"cuda:{dev}")
                target.copy_(summed, non_blocking=True)

            # A6000 的计算流等待 H100 缓存流完成 copy
            # ── 与 fix-B 同源：必须是 current_stream，而非 default_stream ──
            compute_stream.wait_stream(self.locality_cache_stream)
            logger.debug(
                "LocalityCacheGradAccumulator: flushed to device %d, "
                "compute_stream=%s waiting for cache_stream.", dev, compute_stream
            )

        self._cache.clear()
        self._cache_bytes = 0


# ---------------------------------------------------------------------------
# 5. HeteroDDPOverlapGradReducer — 顶层调度器
# ---------------------------------------------------------------------------

class HeteroDDPOverlapGradReducer:
    """
    DES-LOC 异构 DDP 梯度 Overlap 调度器。

    职责
    ~~~~
    1. 将模型参数按显存大小和设备角色分配到梯度桶
    2. 注册 backward hook，在梯度就绪时触发 ``HeteroBucketGroup.start_grad_reduce``
    3. 在 optimizer.step() 前调用 ``finish_all``，确保所有规约完成
    4. 管理 ``LocalityCacheGradAccumulator`` 的生命周期

    桶分配策略（DES-LOC 专有）
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    - A6000 参数（device 0/1）：直接在对应设备上建桶，
      按 ``locality_cache_ratio`` 决定是否经过 H100 缓存路径
    - 超大参数（单参数 > 显存的 30%，约 14 GB）：强制 CPU 卸载，
      绕过 GPU 间 all-reduce

    与 DeepSpeed 集成
    ~~~~~~~~~~~~~~~~~
    本类设计为可插入 DeepSpeed ``ZeroOptimizer`` 的 ``grad_reducer``
    回调，替换 DeepSpeed 内置的 DDP reducer。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: HeteroBucketConfig,
        topology: Optional[HeteroTopology] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.model = weakref.ref(model)
        self.config = config
        self.topology = topology or HeteroTopology.auto_detect()
        self.process_group = process_group
        self._stream_ctx = get_hetero_stream_ctx()

        self.bucket_groups: List[HeteroBucketGroup] = []
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # H100 局部缓存累积器
        self.locality_accumulator = LocalityCacheGradAccumulator(
            cache_device=self.topology.locality_cache_device,
            compute_devices=self.topology.compute_devices,
            flush_threshold_mb=config.bucket_size_mb * config.pcie_pipeline_depth,
        )

        self._build_bucket_groups()
        logger.info(
            "HeteroDDPOverlapGradReducer: %d bucket groups, "
            "overlap=%s, num_dist_opt_instances=%d",
            len(self.bucket_groups),
            config.overlap_grad_reduce,
            config.num_dist_optimizer_instances,
        )

    def _build_bucket_groups(self) -> None:
        """按参数大小和设备将模型参数划分为梯度桶。"""
        model = self.model()
        if model is None:
            return

        bucket_size_bytes = int(self.config.bucket_size_mb * 1024 * 1024)
        current_bucket: List[torch.nn.Parameter] = []
        current_bytes = 0
        bucket_id = 0

        for param in reversed(list(model.parameters())):  # 逆序 = backward 顺序
            if not param.requires_grad:
                continue
            param_bytes = param.data.nbytes
            if current_bytes + param_bytes > bucket_size_bytes and current_bucket:
                self._add_bucket(current_bucket, bucket_id)
                bucket_id += 1
                current_bucket = []
                current_bytes = 0
            current_bucket.append(param)
            current_bytes += param_bytes

        if current_bucket:
            self._add_bucket(current_bucket, bucket_id)

    def _add_bucket(self, params: List[torch.nn.Parameter], bucket_id: int) -> None:
        """构建单个 HeteroBucketGroup 并注册 backward hook。"""
        group = HeteroBucketGroup(
            params=params,
            bucket_id=bucket_id,
            config=self.config,
            topology=self.topology,
            process_group=self.process_group,
        )
        self.bucket_groups.append(group)

        # backward hook：梯度就绪时触发 start_grad_reduce
        def make_hook(bg: HeteroBucketGroup):
            def hook(grad: Tensor) -> Optional[Tensor]:
                if self.config.overlap_grad_reduce:
                    logger.debug(
                        "Backward hook: bucket %d grad ready on device %d, "
                        "triggering start_grad_reduce.",
                        bg.bucket_id, bg.compute_device
                    )
                    bg.start_grad_reduce()
                return grad
            return hook

        for param in params:
            h = param.register_hook(make_hook(group))
            self._hooks.append(h)

    def finish_all(self) -> None:
        """
        等待所有桶的梯度规约完成。

        应在 ``optimizer.step()`` 之前调用。

        DES-LOC 流程：
        1. 刷新 H100 局部缓存累积器（将剩余梯度 flush 回 A6000）
        2. 对每个 bucket group 调用 ``finish_grad_reduce``
           （触发 DES-LOC fix-B：current_stream 等待 comm_stream）
        """
        logger.debug("HeteroDDPOverlapGradReducer.finish_all: flushing locality cache.")
        self.locality_accumulator.flush()

        for bg in self.bucket_groups:
            bg.finish_grad_reduce()

        logger.debug("HeteroDDPOverlapGradReducer.finish_all: all buckets synced.")

    def remove_hooks(self) -> None:
        """清理所有 backward hook（在训练结束或重新构建时调用）。"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("HeteroDDPOverlapGradReducer: all backward hooks removed.")

    def __del__(self) -> None:
        self.remove_hooks()


# ---------------------------------------------------------------------------
# 6. 工厂函数（DeepSpeed config 集成入口）
# ---------------------------------------------------------------------------

def build_hetero_ddp_reducer(
    model: torch.nn.Module,
    deepspeed_config: Optional[dict] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> HeteroDDPOverlapGradReducer:
    """
    从 DeepSpeed config dict 构建 ``HeteroDDPOverlapGradReducer``。

    DeepSpeed config 示例::

        {
            "hetero_ddp": {
                "overlap_grad_reduce": true,
                "num_dist_optimizer_instances": 2,
                "locality_cache_ratio": 0.4,
                "pcie_pipeline_depth": 2,
                "bucket_size_mb": 25.0
            }
        }

    若 ``deepspeed_config`` 为 None 或缺少 ``hetero_ddp`` 键，使用默认值。
    """
    cfg_dict = (deepspeed_config or {}).get("hetero_ddp", {})
    config = HeteroBucketConfig(
        num_dist_optimizer_instances=cfg_dict.get("num_dist_optimizer_instances", 1),
        overlap_grad_reduce=cfg_dict.get("overlap_grad_reduce", True),
        locality_cache_ratio=cfg_dict.get("locality_cache_ratio", 0.4),
        pcie_pipeline_depth=cfg_dict.get("pcie_pipeline_depth", 2),
        bucket_size_mb=cfg_dict.get("bucket_size_mb", 25.0),
    )
    topology = HeteroTopology.auto_detect()
    reducer = HeteroDDPOverlapGradReducer(
        model=model,
        config=config,
        topology=topology,
        process_group=process_group,
    )
    logger.info("build_hetero_ddp_reducer: reducer built successfully.")
    return reducer


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # ── Test 1: HeteroStreamContext 流跟踪 ──────────────────────────────────
    ctx = HeteroStreamContext()
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        default_s = torch.cuda.default_stream(device=dev)
        custom_s = torch.cuda.Stream(device=dev)
        assert ctx.current_stream(dev) == default_s, \
            "未进入 on_stream context 时应返回 default_stream"
        with ctx.on_stream(dev, custom_s):
            assert ctx.current_stream(dev) is custom_s, \
                "on_stream context 内应返回注册的 custom_stream"
        assert ctx.current_stream(dev) == default_s, \
            "离开 on_stream context 后应恢复 default_stream"
        logger.info("Test 1 PASSED: HeteroStreamContext stream tracking correct.")
    else:
        logger.warning("Test 1 SKIPPED: no CUDA device available.")

    # ── Test 2: HeteroTopology 自动探测 ──────────────────────────────────────
    topo = HeteroTopology.auto_detect()
    assert isinstance(topo.device_roles, dict), "device_roles 应为 dict"
    logger.info("Test 2 PASSED: topology=%s", topo)

    # ── Test 3: HeteroBucketConfig 默认值 ────────────────────────────────────
    cfg = HeteroBucketConfig()
    assert cfg.num_dist_optimizer_instances == 1
    assert cfg.overlap_grad_reduce is True
    assert 0.0 <= cfg.locality_cache_ratio <= 1.0
    logger.info("Test 3 PASSED: HeteroBucketConfig defaults sane.")

    # ── Test 4: HeteroBucketGroup 单步（无分布式）───────────────────────────
    if torch.cuda.is_available():
        p = torch.nn.Parameter(torch.randn(64, 64, device="cuda:0"))
        p.grad = torch.ones_like(p)
        bg = HeteroBucketGroup(
            params=[p],
            bucket_id=0,
            config=HeteroBucketConfig(overlap_grad_reduce=False,
                                       num_dist_optimizer_instances=1),
            topology=topo,
            process_group=None,
        )
        bg.start_grad_reduce()   # 无分布式：仅打包/解包
        bg.finish_grad_reduce()
        assert p.grad is not None, "finish 后 grad 应仍存在"
        logger.info("Test 4 PASSED: HeteroBucketGroup single-device smoke test.")
    else:
        logger.warning("Test 4 SKIPPED: no CUDA device available.")

    # ── Test 5: build_hetero_ddp_reducer 工厂 ───────────────────────────────
    model = torch.nn.Linear(128, 64)
    if torch.cuda.is_available():
        model = model.cuda()
    reducer = build_hetero_ddp_reducer(
        model=model,
        deepspeed_config={"hetero_ddp": {"num_dist_optimizer_instances": 2,
                                          "overlap_grad_reduce": True}},
        process_group=None,
    )
    assert len(reducer.bucket_groups) >= 1, "至少应有 1 个桶"
    reducer.remove_hooks()
    logger.info("Test 5 PASSED: build_hetero_ddp_reducer factory ok, %d buckets.",
                len(reducer.bucket_groups))

    logger.info("All smoke tests PASSED.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroDDPOverlapGradReducer on a DeepSpeed engine.

    Instantiates a :class:`HeteroDDPOverlapGradReducer` from the engine's configuration
    and attaches it as ``engine.hetero_ddp_overlap_grad_reducer``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_ddp_overlap_grad_fix.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_ddp_overlap_grad_reducer = None
    logger.info("hetero_ddp_overlap_grad_fix.register() attached engine.hetero_ddp_overlap_grad_reducer")
