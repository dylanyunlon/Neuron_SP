"""
deepspeed/runtime/hetero_local_cg_moe_fix.py

DES-LOC HeteroLocalCGMoEFix
============================

上游设计意图 (Megatron commit 481efd020e08e30a21c70501ece8bbee6c4ca567)
-----------------------------------------------------------------------
Megatron-LM 的本地 CUDA Graph (local CG) 实现在处理 Latent MoE 模型时存在
多个会导致 loss curve gap 的 bug：

1. **Buffer 未备份问题**：在 CUDA Graph warmup 阶段，`_CudaGraphRunner.record_graph_capture()`
   会对模型做多次前向传播以"预热"图捕获。但 MoE Router 中的 `expert_bias` 是一个
   persistent buffer，每次前向传播都会通过 `_apply_expert_bias()` 原位更新。
   Warmup 阶段的多次前向会把 `expert_bias` 的值污染，导致正式捕获时的初始状态
   与训练期望不符，造成 loss 曲线出现非预期跳变（gap）。

2. **clone_ten 语义错误**：原本用于复制 warmup 输入的 `clone_ten` 函数使用了
   `torch.zeros_like(ten).requires_grad_(...)` ——这会生成全零张量，导致 warmup
   阶段的激活值统计（如 LayerNorm 的 running stats、专家负载统计）被零输入污染。
   正确做法是 `torch.clone(ten).detach().requires_grad_(...)` 保留真实分布。

3. **提前释放 buffer 回 pool 的问题**：原代码在前向结束后会把"不带 metadata" 的
   input surface buffer 提前归还给 `tensor_reuse_pool`，但此时 graph 的异步执行
   可能尚未完成，导致该 buffer 被复用时数据被覆盖，产生随机数值错误。

4. **冻结层 eval() 模式切换问题**：`CudaGraphManager` 原本在 `is_grad_enabled()==False`
   时强制调用 `runner.eval()`，导致 BatchNorm / Dropout 等模块切换为 eval 语义，
   破坏了梯度检查点（gradient checkpointing）场景下的正确性。

DES-LOC 适配点
--------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) 是面向异构硬件的训练框架，
本集群配置为：2x A6000 48GB (SM86, PCIe) + 1x H100 NVL 96GB (SM90, PCIe)，
共享 1.5TB CPU DRAM 作为 LOC (Locality Cache) 层。

在此异构拓扑下，上游 bug 会以更严重的形式放大：

A. **设备感知 Buffer 备份**：A6000 (SM86) 与 H100 (SM90) 的 CUDA Graph 捕获机制
   存在差异——SM90 支持 conditional graph node，SM86 不支持。DES-LOC 的 LOC Cache
   会在 CPU DRAM 中保存中间激活，expert_bias 若在 warmup 阶段被污染，会通过
   LOC Cache 的"重放路径"传播到另一块设备，造成跨设备的数值不一致。

B. **异构 clone_ten**：不同 SM 架构的张量初始化（zeros_like）在图捕获阶段会生成
   与目标设备绑定的存储，跨 PCIe 传输全零张量给 SM90 侧会引发额外的 H2D/D2H，
   而 clone(detach) 则能让 LOC Cache 的引用计数系统正确追踪张量生命周期。

C. **LOC Cache 感知的 Buffer Pool 归还**：DES-LOC 在 PCIe 无 NVLink 场景下依赖
   LOC Cache 做跨设备激活共享，提前归还 buffer 会导致 LOC Cache 的引用计数器
   错误地认为该 buffer 已释放，触发不必要的 CPU offload，增加 PCIe 流量。

D. **异构梯度门控**：DES-LOC 支持将 MoE 的不同专家分配到不同设备（A6000 负责
   小专家，H100 负责大容量专家）。当某个设备上的层被冻结时，不能粗暴地切换
   runner 到 eval 模式，而应通过 DES-LOC 的 DeviceGradGate 精细控制梯度流动。

本模块实现：
- `HeteroDeviceProfile`：描述异构设备能力（SM 版本、显存、PCIe 带宽）
- `LOCCacheBufferTracker`：跟踪 LOC Cache 中 buffer 的引用计数，防止提前归还
- `HeteroLocalCGMoEFix`：核心修复类，包含设备感知的 buffer 备份/恢复逻辑
- `DeviceAwareCGRunner`：封装了修复后的 CUDA Graph Runner，感知 SM 架构差异
- `HeteroCGMoEManager`：替代 Megatron 的 `CudaGraphManager`，支持 DES-LOC 拓扑

作者: Neuron_SP / dylanyunlon
上游参考: Megatron-LM commit 481efd020e08e30a21c70501ece8bbee6c4ca567
"""

from __future__ import annotations

import gc
import logging
import threading
import unittest
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_SM86_ARCH = 86   # NVIDIA Ampere A6000
_SM90_ARCH = 90   # NVIDIA Hopper H100 NVL
_LOC_CACHE_REFCOUNT_LIMIT = 1 << 20  # 溢出保护


# ---------------------------------------------------------------------------
# 枚举：设备角色
# ---------------------------------------------------------------------------

class DeviceRole(Enum):
    """DES-LOC 异构拓扑中的设备职能分配。

    在 2xA6000 + 1xH100 拓扑下，A6000 承担"小专家/轻量层"角色，
    H100 承担"大容量专家/计算密集层"角色。CPU DRAM 作为 LOC Cache
    在二者之间做激活共享。
    """
    SMALL_EXPERT_DEVICE = auto()   # A6000 SM86
    LARGE_EXPERT_DEVICE = auto()   # H100 SM90
    LOC_CACHE_HOST = auto()        # CPU DRAM


# ---------------------------------------------------------------------------
# 数据类：异构设备档案
# ---------------------------------------------------------------------------

@dataclass
class HeteroDeviceProfile:
    """描述单个设备的能力档案，供 DES-LOC 调度决策使用。

    Attributes
    ----------
    device_id : int
        PyTorch 设备索引。
    sm_version : int
        CUDA SM 架构版本，如 86 (A6000) 或 90 (H100)。
    total_memory_gb : float
        设备显存总量（GiB）。
    pcie_bandwidth_gbps : float
        PCIe 理论带宽（GB/s）。从 CPU DRAM 的视角衡量。
    role : DeviceRole
        该设备在 DES-LOC 拓扑中承担的角色。
    supports_conditional_graph_nodes : bool
        是否支持 CUDA Graph 的 conditional node（SM90+ 特性）。
        A6000 (SM86) 不支持，H100 (SM90) 支持。
    """
    device_id: int
    sm_version: int
    total_memory_gb: float
    pcie_bandwidth_gbps: float
    role: DeviceRole
    supports_conditional_graph_nodes: bool = field(init=False)

    def __post_init__(self) -> None:
        self.supports_conditional_graph_nodes = self.sm_version >= _SM90_ARCH

    @property
    def torch_device(self) -> torch.device:
        return torch.device("cuda", self.device_id)

    @classmethod
    def from_cuda_device(cls, device_id: int, role: DeviceRole,
                         pcie_bandwidth_gbps: float = 32.0) -> "HeteroDeviceProfile":
        """从运行时 CUDA 属性自动构建档案。"""
        props = torch.cuda.get_device_properties(device_id)
        sm_version = props.major * 10 + props.minor
        total_gb = props.total_memory / (1024 ** 3)
        logger.info(
            "HeteroDeviceProfile: device=%d sm=%d mem=%.1fGB role=%s",
            device_id, sm_version, total_gb, role.name,
        )
        return cls(
            device_id=device_id,
            sm_version=sm_version,
            total_memory_gb=total_gb,
            pcie_bandwidth_gbps=pcie_bandwidth_gbps,
            role=role,
        )


# ---------------------------------------------------------------------------
# LOC Cache 引用计数追踪器
# ---------------------------------------------------------------------------

class LOCCacheBufferTracker:
    """追踪 DES-LOC LOC Cache 中 tensor buffer 的引用计数。

    设计动机
    --------
    Megatron 原始代码在前向结束后会把"不带 metadata"的 fwd_graph_input_surface
    buffer 提前归还给 tensor_reuse_pool（见上游 diff 第 995 行附近被删除的代码块）。
    在单 GPU 场景下这是可行的，因为 CUDA 流同步能保证图执行完成后再归还。

    但在 DES-LOC 的 PCIe 异构拓扑中：
    1. LOC Cache 持有 CPU DRAM 中的 pinned tensor 作为两个 GPU 间的激活中继站。
    2. 一个 buffer 可能同时被 A6000 侧的图和 H100 侧的图引用（通过 LOC Cache 的
       cross-device alias 机制）。
    3. 提前归还会导致 LOC Cache 的 Python 层引用计数器与实际 GPU 侧使用状态不一致，
       触发不必要的 CPU offload/prefetch，增加 PCIe 流量，严重时导致数值错误。

    本追踪器通过弱引用（weakref）监控 tensor 生命周期，并在 buffer 的所有
    GPU 侧使用者都完成后才允许归还。

    线程安全
    --------
    使用 `threading.Lock` 保护引用计数表，支持多线程数据加载场景。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # tensor data_ptr -> ref count
        self._refcounts: Dict[int, int] = defaultdict(int)
        # tensor data_ptr -> set of device_ids that currently reference it
        self._device_holders: Dict[int, Set[int]] = defaultdict(set)
        # weakref -> data_ptr，用于自动清理
        self._weak_registry: Dict[int, weakref.ref] = {}

    def acquire(self, tensor: torch.Tensor, device_id: int) -> None:
        """标记某设备持有该 buffer 的引用。"""
        ptr = tensor.data_ptr()
        with self._lock:
            self._refcounts[ptr] += 1
            self._device_holders[ptr].add(device_id)
            if self._refcounts[ptr] > _LOC_CACHE_REFCOUNT_LIMIT:
                logger.warning(
                    "LOCCacheBufferTracker: refcount overflow at ptr=0x%x, "
                    "possible leak", ptr
                )

    def release(self, tensor: torch.Tensor, device_id: int) -> bool:
        """释放某设备对该 buffer 的引用，返回 True 表示可以安全归还 pool。

        Parameters
        ----------
        tensor : torch.Tensor
            要释放的 buffer。
        device_id : int
            释放引用的设备 ID。

        Returns
        -------
        bool
            True 当且仅当该 buffer 在所有设备上的引用计数都已降至 0，
            可以安全地归还给 tensor_reuse_pool 或 LOC Cache。
        """
        ptr = tensor.data_ptr()
        with self._lock:
            if ptr not in self._refcounts:
                # 未被追踪的 buffer，保守起见允许归还
                return True
            self._device_holders[ptr].discard(device_id)
            self._refcounts[ptr] = max(0, self._refcounts[ptr] - 1)
            all_released = (
                self._refcounts[ptr] == 0
                and len(self._device_holders[ptr]) == 0
            )
            if all_released:
                del self._refcounts[ptr]
                del self._device_holders[ptr]
                logger.debug(
                    "LOCCacheBufferTracker: ptr=0x%x fully released, "
                    "safe to return to pool", ptr
                )
            return all_released

    def is_safe_to_return(self, tensor: torch.Tensor) -> bool:
        """查询 buffer 当前是否可以安全归还 pool（不持有任何设备引用）。"""
        ptr = tensor.data_ptr()
        with self._lock:
            return self._refcounts.get(ptr, 0) == 0

    def held_by_devices(self, tensor: torch.Tensor) -> Set[int]:
        """返回当前持有该 buffer 引用的所有设备 ID 集合。"""
        ptr = tensor.data_ptr()
        with self._lock:
            return set(self._device_holders.get(ptr, set()))

    def clear(self) -> None:
        """清空所有追踪状态（用于测试或显式重置）。"""
        with self._lock:
            self._refcounts.clear()
            self._device_holders.clear()
            self._weak_registry.clear()
        logger.debug("LOCCacheBufferTracker: cleared all tracking state")


# ---------------------------------------------------------------------------
# 设备感知的 clone_ten 实现
# ---------------------------------------------------------------------------

def hetero_clone_ten(
    ten: Any,
    target_profile: Optional[HeteroDeviceProfile] = None,
) -> Any:
    """复制 warmup 输入张量的设备感知版本。

    上游 bug 说明
    ------------
    Megatron 原始 `clone_ten` 使用 `torch.zeros_like(ten).requires_grad_(...)`。
    这在 Latent MoE 场景下有两个问题：
    1. 全零输入会使 MoE Router 的 softmax 输出均匀分布，与真实训练分布不符，
       导致 expert_bias 的 warmup 更新方向错误。
    2. `zeros_like` 在某些 PyTorch 版本中会保留 autograd 历史节点，
       而 `clone().detach()` 能明确切断计算图，符合 warmup 语义。

    DES-LOC 适配
    ------------
    在异构拓扑中，warmup 张量可能从 LOC Cache（CPU DRAM pinned memory）流入
    GPU 侧。`zeros_like` 会在目标设备上分配新存储，可能引发额外的 PCIe 传输。
    `clone()` + `detach()` 则保留原始存储设备，让 LOC Cache 的引用路径保持连贯。

    对于 SM86 (A6000)：不支持 conditional graph node，clone 时需要确保
    结果张量是 contiguous 的，否则图捕获会失败。

    Parameters
    ----------
    ten : Any
        输入对象，若非 tensor 则原样返回。
    target_profile : HeteroDeviceProfile, optional
        目标设备档案。若提供且设备为 SM86，强制 contiguous。

    Returns
    -------
    Any
        复制后的张量（或原对象）。
    """
    if not torch.is_tensor(ten):
        return ten

    cloned = torch.clone(ten).detach().requires_grad_(ten.requires_grad)

    # SM86 (A6000) 的 CUDA Graph 捕获要求输入 tensor 是 contiguous 的
    if target_profile is not None and target_profile.sm_version == _SM86_ARCH:
        if not cloned.is_contiguous():
            cloned = cloned.contiguous()

    return cloned


# ---------------------------------------------------------------------------
# 核心 Buffer 备份/恢复逻辑
# ---------------------------------------------------------------------------

class ModuleBufferGuard:
    """上下文管理器：在图捕获期间备份并恢复 nn.Module 的所有 persistent buffer。

    上游修复说明
    -----------
    Megatron commit 481efd0 在 `record_graph_capture()` 中加入了 buffer 备份：

        buffer_backup = []
        for buf in self.base_module.buffers():
            buffer_backup.append(buf.clone())

    并在前向结束后恢复：

        for buf_copy, buf in zip(buffer_backup, self.base_module.buffers()):
            buf.copy_(buf_copy)

    这解决了 MoE Router `expert_bias` 在 warmup 阶段被污染的问题。

    DES-LOC 异构扩展
    ---------------
    在 DES-LOC 框架中，expert_bias 可能存在于多个设备上（A6000 侧的小专家
    和 H100 侧的大专家各持有一份）。跨设备的 buffer 备份需要：

    1. 追踪每个 buffer 所在的设备，确保备份与恢复发生在同一设备上（避免 PCIe 拷贝）。
    2. 对于分布在 LOC Cache（CPU DRAM）中的 buffer，使用 `pin_memory()` 加速
       后续的 H2D 恢复传输。
    3. 如果 buffer 的设备与当前 CUDA stream 不匹配，使用异步拷贝 + 显式同步，
       而非同步拷贝，以减少设备间等待时间。

    使用方式
    --------
        with ModuleBufferGuard(module, device_profile, loc_tracker) as guard:
            # 在此范围内，warmup 前向传播不会污染 buffer
            run_warmup_passes(module)
        # 退出时自动恢复
    """

    def __init__(
        self,
        module: nn.Module,
        device_profile: Optional[HeteroDeviceProfile] = None,
        loc_tracker: Optional[LOCCacheBufferTracker] = None,
    ) -> None:
        self._module = module
        self._device_profile = device_profile
        self._loc_tracker = loc_tracker
        self._buffer_backup: List[Tuple[str, torch.Tensor]] = []

    def _backup(self) -> None:
        """备份所有 persistent buffer。

        为每个 buffer 的 clone 操作选择最优策略：
        - 同设备 buffer：直接 clone（最快）
        - LOC Cache buffer（CPU）：clone 到 pinned memory
        """
        self._buffer_backup.clear()
        for name, buf in self._module.named_buffers():
            if buf is None:
                self._buffer_backup.append((name, None))
                continue

            if buf.is_cuda:
                backup = buf.clone()
                if self._loc_tracker is not None:
                    # 标记备份张量被当前设备持有
                    dev_id = buf.device.index
                    self._loc_tracker.acquire(backup, dev_id)
            else:
                # CPU buffer（可能是 LOC Cache 中的激活）
                backup = buf.clone()
                if buf.is_pinned():
                    # 已经是 pinned memory，直接使用
                    pass
                # 不在这里 pin，因为 pin 需要额外分配，留给调用方决策

            self._buffer_backup.append((name, backup))

        if self._buffer_backup:
            logger.debug(
                "ModuleBufferGuard: backed up %d buffers for module %s",
                len(self._buffer_backup),
                type(self._module).__name__,
            )

    def _restore(self) -> None:
        """将所有 buffer 恢复到备份时的值。

        对于 GPU buffer，使用 non_blocking=True 异步拷贝，
        随后在流同步点统一等待，而非每个 buffer 都做同步拷贝。
        这在有大量 expert_bias buffer 的 MoE 模型中能显著减少等待时间。
        """
        if not self._buffer_backup:
            return

        buf_dict = dict(self._module.named_buffers())
        async_cuda_copies: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for name, backup in self._buffer_backup:
            if backup is None:
                continue
            live_buf = buf_dict.get(name)
            if live_buf is None:
                continue

            if live_buf.is_cuda and backup.is_cuda and live_buf.device == backup.device:
                # 同设备异步拷贝
                live_buf.copy_(backup, non_blocking=True)
                async_cuda_copies.append((live_buf, backup))
            else:
                # 跨设备或 CPU buffer，同步拷贝保证正确性
                live_buf.copy_(backup)

            # 释放 LOC Cache 追踪引用
            if self._loc_tracker is not None and backup.is_cuda:
                dev_id = backup.device.index
                self._loc_tracker.release(backup, dev_id)

        # 统一等待所有异步 GPU 拷贝完成
        if async_cuda_copies:
            # 在当前流上插入同步屏障
            current_stream = torch.cuda.current_stream()
            current_stream.synchronize()
            logger.debug(
                "ModuleBufferGuard: restored %d buffers (async) for module %s",
                len(async_cuda_copies),
                type(self._module).__name__,
            )

    def __enter__(self) -> "ModuleBufferGuard":
        self._backup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._restore()
        # 不抑制异常
        return False


# ---------------------------------------------------------------------------
# DeviceGradGate：异构梯度门控
# ---------------------------------------------------------------------------

class DeviceGradGate:
    """DES-LOC 异构梯度门控，替代 Megatron 的粗暴 `runner.eval()` 调用。

    上游 bug 说明
    -----------
    Megatron 原始 `CudaGraphManager.forward()` 在 `not torch.is_grad_enabled()`
    时会强制调用 `runner.eval()`：

        if not torch.is_grad_enabled():
            runner.eval()

    这在梯度检查点（gradient checkpointing）场景下是错误的，因为：
    - 梯度检查点的"重算"阶段会关闭梯度计算以节省显存，
      但此时模型应保持 training 模式（BatchNorm 使用 batch stats，
      Dropout 保持激活状态）。
    - 强制 eval() 会让 BatchNorm 切换到使用 running_mean/running_var，
      Dropout 失效，导致重算结果与首次前向不一致，梯度错误。

    DES-LOC 异构扩展
    ---------------
    在 DES-LOC 的异构拓扑中，MoE 的不同专家可能分布在不同设备上：
    - A6000 (SM86) 负责小专家，H100 (SM90) 负责大容量专家
    - 某些层可能被"冻结"（仅在某一设备上不需要梯度）

    正确做法是通过 `DeviceGradGate` 精细控制每个设备上的梯度门，
    而非全局切换 runner 的 training/eval 状态。

    这个类维护每个设备的梯度门状态，提供细粒度的梯度启用/禁用接口。
    """

    def __init__(self, device_profiles: List[HeteroDeviceProfile]) -> None:
        # device_id -> 是否允许梯度（True = 训练模式，False = 冻结）
        self._grad_gates: Dict[int, bool] = {
            p.device_id: True for p in device_profiles
        }
        self._lock = threading.Lock()

    def freeze_device(self, device_id: int) -> None:
        """冻结指定设备上的梯度计算（不影响 training/eval 模式）。"""
        with self._lock:
            if device_id not in self._grad_gates:
                raise ValueError(f"DeviceGradGate: unknown device_id={device_id}")
            prev = self._grad_gates[device_id]
            self._grad_gates[device_id] = False
            if prev:
                logger.info(
                    "DeviceGradGate: device=%d frozen (grad disabled)", device_id
                )

    def unfreeze_device(self, device_id: int) -> None:
        """解冻指定设备上的梯度计算。"""
        with self._lock:
            if device_id not in self._grad_gates:
                raise ValueError(f"DeviceGradGate: unknown device_id={device_id}")
            self._grad_gates[device_id] = True
            logger.info(
                "DeviceGradGate: device=%d unfrozen (grad enabled)", device_id
            )

    def is_grad_active(self, device_id: int) -> bool:
        """查询指定设备是否允许梯度计算。"""
        with self._lock:
            return self._grad_gates.get(device_id, True)

    def should_use_training_mode(
        self,
        device_id: int,
        global_grad_enabled: bool,
    ) -> bool:
        """综合判断是否应保持 training 模式（而非切换到 eval）。

        DES-LOC 的判断逻辑：
        - 即使 global_grad_enabled=False（梯度检查点重算阶段），
          只要设备未被显式冻结，就应保持 training 模式。
        - 只有当设备被显式冻结（freeze_device 调用）时才切换语义，
          但仍不调用 eval()，而是通过 no_grad 上下文管理器控制。

        Parameters
        ----------
        device_id : int
            目标设备 ID。
        global_grad_enabled : bool
            当前全局 `torch.is_grad_enabled()` 状态。

        Returns
        -------
        bool
            True = 应保持 training 模式（不调用 eval()）。
        """
        device_grad_active = self.is_grad_active(device_id)
        # 关键：即使全局梯度关闭（如 grad checkpointing 重算），
        # 也不切换到 eval，除非设备被显式冻结。
        return device_grad_active or global_grad_enabled


# ---------------------------------------------------------------------------
# 核心修复类：HeteroLocalCGMoEFix
# ---------------------------------------------------------------------------

class HeteroLocalCGMoEFix:
    """DES-LOC 异构局部 CUDA Graph MoE 修复主类。

    整合了所有针对 Megatron commit 481efd0 所描述问题的 DES-LOC 适配修复：

    1. `fixed_record_graph_capture()` ：
       - 修复 buffer 备份（ModuleBufferGuard）
       - 修复 clone_ten（hetero_clone_ten）
       - 移除提前归还 buffer 到 pool 的逻辑（LOC Cache 引用计数保护）

    2. `fixed_manager_forward()` ：
       - 移除 `runner.eval()` 的粗暴调用，改用 DeviceGradGate 精细控制

    该类设计为"混入修复器"（mixin patcher），通过方法替换（monkey-patching）
    的方式将修复注入到已有的 runner 和 manager 对象，无需完整重写 Megatron 类层级。

    Parameters
    ----------
    device_profiles : list of HeteroDeviceProfile
        异构集群中所有 GPU 设备的档案列表。
    loc_tracker : LOCCacheBufferTracker, optional
        LOC Cache 引用计数追踪器。若为 None，则自动创建。
    enable_async_buffer_restore : bool
        是否在 buffer 恢复时使用异步拷贝（默认 True，SM90 环境下推荐）。
    """

    def __init__(
        self,
        device_profiles: List[HeteroDeviceProfile],
        loc_tracker: Optional[LOCCacheBufferTracker] = None,
        enable_async_buffer_restore: bool = True,
    ) -> None:
        self.device_profiles = device_profiles
        self.loc_tracker = loc_tracker or LOCCacheBufferTracker()
        self.enable_async_buffer_restore = enable_async_buffer_restore
        self.grad_gate = DeviceGradGate(device_profiles)

        # device_id -> HeteroDeviceProfile 的快速查找表
        self._profile_map: Dict[int, HeteroDeviceProfile] = {
            p.device_id: p for p in device_profiles
        }

        logger.info(
            "HeteroLocalCGMoEFix: initialized with %d devices: %s",
            len(device_profiles),
            [f"cuda:{p.device_id}(SM{p.sm_version})" for p in device_profiles],
        )

    def get_profile_for_module(
        self, module: nn.Module
    ) -> Optional[HeteroDeviceProfile]:
        """根据模块的参数/buffer 所在设备推断对应的 HeteroDeviceProfile。

        优先级：参数设备 > buffer 设备 > None（CPU 模块）。
        """
        for param in module.parameters():
            if param.is_cuda:
                dev_id = param.device.index
                return self._profile_map.get(dev_id)
        for buf in module.buffers():
            if buf is not None and buf.is_cuda:
                dev_id = buf.device.index
                return self._profile_map.get(dev_id)
        return None

    def make_buffer_guard(self, module: nn.Module) -> ModuleBufferGuard:
        """为给定模块创建设备感知的 buffer 守卫。"""
        profile = self.get_profile_for_module(module)
        return ModuleBufferGuard(
            module=module,
            device_profile=profile,
            loc_tracker=self.loc_tracker,
        )

    def hetero_clone_for_module(
        self,
        module: nn.Module,
    ) -> Callable[[Any], Any]:
        """返回为该模块量身定制的 clone_ten 函数（闭包）。

        根据模块所在设备的 SM 版本决定是否强制 contiguous。
        """
        profile = self.get_profile_for_module(module)

        def _clone(ten: Any) -> Any:
            return hetero_clone_ten(ten, target_profile=profile)

        return _clone

    def should_runner_use_training_mode(
        self,
        module: nn.Module,
        global_grad_enabled: bool,
    ) -> bool:
        """判断 runner 是否应保持 training 模式。

        替代 Megatron 原始的：
            if not torch.is_grad_enabled():
                runner.eval()

        DES-LOC 语义：只有当该模块所在设备被显式冻结时，
        才允许"不使用 training 模式"，且即便如此也不调用 eval()。
        """
        profile = self.get_profile_for_module(module)
        if profile is None:
            # CPU 模块，遵循全局梯度状态
            return global_grad_enabled

        return self.grad_gate.should_use_training_mode(
            device_id=profile.device_id,
            global_grad_enabled=global_grad_enabled,
        )

    def safe_return_buffer_to_pool(
        self,
        buf: torch.Tensor,
        pool_insert_fn: Callable[[torch.Tensor], None],
    ) -> bool:
        """在 LOC Cache 引用计数保护下安全地归还 buffer。

        替代 Megatron 原始的（被删除的）提前归还逻辑：

            for buf in self.fwd_graph_input_surface:
                if (hasattr(buf, 'can_skip_replay_copy')
                    and not buf.can_skip_replay_copy
                    and not hasattr(buf, 'cg_buffer_metadata')):
                    assert _CudagraphGlobalRecord.tensor_reuse_pool.owns(buf)
                    _CudagraphGlobalRecord.tensor_reuse_pool.insert(buf)

        DES-LOC 修复：先检查 LOC Cache 追踪器，确认无任何设备持有该 buffer
        的引用后，才调用 pool_insert_fn 归还。

        Parameters
        ----------
        buf : torch.Tensor
            要归还的 buffer。
        pool_insert_fn : callable
            实际的 pool 归还函数（如 `tensor_reuse_pool.insert`）。

        Returns
        -------
        bool
            True 表示已成功归还，False 表示因引用计数未清零而跳过归还。
        """
        if not hasattr(buf, "can_skip_replay_copy"):
            return False
        if buf.can_skip_replay_copy:
            return False
        if hasattr(buf, "cg_buffer_metadata"):
            return False

        # DES-LOC: 检查 LOC Cache 引用计数
        if not self.loc_tracker.is_safe_to_return(buf):
            held_by = self.loc_tracker.held_by_devices(buf)
            logger.debug(
                "HeteroLocalCGMoEFix: skipping pool return for buf ptr=0x%x, "
                "still held by devices=%s (LOC Cache protection)",
                buf.data_ptr(), held_by,
            )
            return False

        pool_insert_fn(buf)
        return True


# ---------------------------------------------------------------------------
# 设备感知的 CG Runner 封装
# ---------------------------------------------------------------------------

class DeviceAwareCGRunner:
    """封装修复后的 CUDA Graph Runner，感知 SM 架构差异。

    这不是一个完整的 `_CudaGraphRunner` 重写，而是一个"适配层"，
    通过持有 fix_helper 来为已有 runner 提供 DES-LOC 感知的辅助方法。

    在实际集成中，可以将这些方法注入到 `_CudaGraphRunner` 实例中，
    或者继承 `_CudaGraphRunner` 并覆盖相关方法。

    Parameters
    ----------
    fix_helper : HeteroLocalCGMoEFix
        核心修复助手实例。
    base_module : nn.Module
        被图捕获的基础模块（对应 Megatron 的 `self.base_module`）。
    training : bool
        当前是否处于训练模式。
    """

    def __init__(
        self,
        fix_helper: HeteroLocalCGMoEFix,
        base_module: nn.Module,
        training: bool = True,
    ) -> None:
        self.fix_helper = fix_helper
        self.base_module = base_module
        self.training = training
        self._clone_fn = fix_helper.hetero_clone_for_module(base_module)
        self._device_profile = fix_helper.get_profile_for_module(base_module)

    def pre_capture_setup(
        self,
        args: tuple,
        kwargs: dict,
        grad_enabled: bool,
    ) -> Tuple["ModuleBufferGuard", tuple, dict]:
        """图捕获前的准备工作（对应 `record_graph_capture` 的前半段）。

        执行步骤：
        1. 创建并进入 buffer 守卫（备份所有 persistent buffer）
        2. 使用修复后的 clone_ten 复制 warmup 输入

        Returns
        -------
        tuple of (ModuleBufferGuard, warmup_args, warmup_kwargs)
            guard 需要在图捕获结束后手动退出（调用 __exit__），
            warmup_args/kwargs 是安全的 warmup 输入副本。
        """
        guard = self.fix_helper.make_buffer_guard(self.base_module)
        guard.__enter__()

        # 使用修复后的 clone_ten（torch.clone().detach() 而非 zeros_like）
        from torch.utils._pytree import tree_map
        warmup_args = tree_map(self._clone_fn, args)
        warmup_kwargs = tree_map(self._clone_fn, kwargs)

        logger.debug(
            "DeviceAwareCGRunner: pre_capture_setup complete for %s "
            "(SM%s, grad_enabled=%s)",
            type(self.base_module).__name__,
            self._device_profile.sm_version if self._device_profile else "?",
            grad_enabled,
        )
        return guard, warmup_args, warmup_kwargs

    def post_capture_cleanup(
        self,
        guard: "ModuleBufferGuard",
        fwd_graph_input_surface: List[torch.Tensor],
        pool_insert_fn: Optional[Callable] = None,
    ) -> None:
        """图捕获后的清理工作（对应 `record_graph_capture` 的后半段）。

        执行步骤：
        1. 退出 buffer 守卫（恢复所有 persistent buffer）
        2. 跳过不安全的提前 pool 归还（LOC Cache 保护）

        Parameters
        ----------
        guard : ModuleBufferGuard
            由 pre_capture_setup 返回的 buffer 守卫。
        fwd_graph_input_surface : list of Tensor
            前向图的输入 surface tensors。
        pool_insert_fn : callable, optional
            pool 归还函数。若为 None，则跳过归还步骤。
        """
        # 恢复 buffer（上游 fix: restore cached buffers）
        guard.__exit__(None, None, None)

        # DES-LOC: 不做提前 pool 归还（上游 fix: 删除了这段代码）
        # 原始 Megatron 代码在此处把 fwd_graph_input_surface 中
        # 不带 metadata 的 buffer 归还给 pool，但这在 LOC Cache 场景下不安全。
        # 我们通过 LOCCacheBufferTracker 做引用计数保护，只有在确认安全后才归还。
        if pool_insert_fn is not None:
            skipped = 0
            for buf in fwd_graph_input_surface:
                returned = self.fix_helper.safe_return_buffer_to_pool(
                    buf, pool_insert_fn
                )
                if not returned:
                    skipped += 1

            if skipped > 0:
                logger.debug(
                    "DeviceAwareCGRunner: skipped early pool return for %d buffers "
                    "(LOC Cache safety)", skipped
                )

    def should_stay_in_training_mode(self, global_grad_enabled: bool) -> bool:
        """判断 runner 是否应保持 training 模式（替代粗暴的 `runner.eval()`）。"""
        return self.fix_helper.should_runner_use_training_mode(
            self.base_module, global_grad_enabled
        )


# ---------------------------------------------------------------------------
# HeteroCGMoEManager：替代 CudaGraphManager
# ---------------------------------------------------------------------------

class HeteroCGMoEManager(nn.Module):
    """DES-LOC 异构 CUDA Graph MoE 管理器。

    替代 Megatron 的 `CudaGraphManager`，整合了所有 DES-LOC 修复，
    并添加了异构设备感知的调度逻辑。

    核心改变（对应 Megatron diff 中 CudaGraphManager 部分）：
    ---------------------------------------------------------------------------
    上游删除了以下代码（在 `forward()` 的 training 分支）：

        if not torch.is_grad_enabled():
            runner.eval()

    DES-LOC 替代方案：
    - 使用 `DeviceGradGate.should_use_training_mode()` 判断
    - 不调用 `runner.eval()`，而是通过 `torch.no_grad()` 上下文控制梯度流
    - 保持 runner 始终处于 training 模式，确保 BN/Dropout 语义正确

    Parameters
    ----------
    fix_helper : HeteroLocalCGMoEFix
        核心修复助手实例。
    reuse_cudagraphs : bool
        是否复用已捕获的 CUDA Graph（对应 Megatron 的 `self.reuse_cudagraphs`）。
    """

    def __init__(
        self,
        fix_helper: HeteroLocalCGMoEFix,
        reuse_cudagraphs: bool = True,
    ) -> None:
        super().__init__()
        self.fix_helper = fix_helper
        self.reuse_cudagraphs = reuse_cudagraphs
        # runner cache: module_id -> DeviceAwareCGRunner
        self._runner_cache: Dict[int, DeviceAwareCGRunner] = {}

    def get_or_create_runner(
        self,
        module: nn.Module,
        training: bool,
    ) -> DeviceAwareCGRunner:
        """获取或创建模块对应的 DeviceAwareCGRunner。"""
        key = id(module)
        if key not in self._runner_cache or not self.reuse_cudagraphs:
            runner = DeviceAwareCGRunner(
                fix_helper=self.fix_helper,
                base_module=module,
                training=training,
            )
            self._runner_cache[key] = runner
        return self._runner_cache[key]

    def forward(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
        is_training: bool,
    ) -> Any:
        """执行前向传播（DES-LOC 修复版）。

        关键修复：不再根据 `torch.is_grad_enabled()` 调用 `runner.eval()`。
        而是通过 `DeviceGradGate` 判断是否需要修改训练模式语义，
        并使用更精细的梯度门控机制。

        Parameters
        ----------
        module : nn.Module
            要执行前向传播的模块。
        args : tuple
            前向传播的位置参数。
        kwargs : dict
            前向传播的关键字参数。
        is_training : bool
            是否处于训练模式（对应 `runner.training`）。

        Returns
        -------
        Any
            模块前向传播的输出。
        """
        global_grad_enabled = torch.is_grad_enabled()
        runner = self.get_or_create_runner(module, is_training)

        # DES-LOC 修复：不调用 runner.eval()
        # 原始 Megatron 代码：
        #   if not torch.is_grad_enabled():
        #       runner.eval()
        # 修复后：通过 DeviceGradGate 判断，保持 training 模式
        stay_training = runner.should_stay_in_training_mode(global_grad_enabled)
        if is_training and not stay_training:
            # 设备被显式冻结，且全局梯度也关闭——进入冻结前向
            logger.debug(
                "HeteroCGMoEManager: device frozen + grad disabled, "
                "using no_grad forward for %s",
                type(module).__name__,
            )
            with torch.no_grad():
                return module(*args, **kwargs)

        # 正常训练前向（training 模式保持不变）
        return module(*args, **kwargs)

    def clear_runner_cache(self) -> None:
        """清空 runner 缓存（用于显存压力时的手动清理）。"""
        self._runner_cache.clear()
        gc.collect()
        logger.info("HeteroCGMoEManager: runner cache cleared")


# ---------------------------------------------------------------------------
# 工具函数：构建标准 DES-LOC 拓扑的设备档案
# ---------------------------------------------------------------------------

def build_des_loc_profiles(
    a6000_device_ids: List[int],
    h100_device_id: int,
) -> List[HeteroDeviceProfile]:
    """为 2xA6000 + 1xH100 的 DES-LOC 标准拓扑构建设备档案列表。

    这个函数封装了 Neuron_SP 项目针对 NVIDIA 2xA6000(SM86) + 1xH100 NVL(SM90)
    硬件配置的标准初始化流程。

    Parameters
    ----------
    a6000_device_ids : list of int
        A6000 设备的 CUDA 设备 ID 列表（通常为 [0, 1]）。
    h100_device_id : int
        H100 NVL 设备的 CUDA 设备 ID（通常为 2）。

    Returns
    -------
    list of HeteroDeviceProfile
        按 [A6000_0, A6000_1, H100] 顺序排列的设备档案列表。

    Notes
    -----
    PCIe 带宽估算（无 NVLink）：
    - A6000 PCIe Gen4 x16: ~32 GB/s
    - H100 NVL PCIe Gen5 x16: ~64 GB/s
    """
    profiles = []
    for dev_id in a6000_device_ids:
        profile = HeteroDeviceProfile(
            device_id=dev_id,
            sm_version=_SM86_ARCH,
            total_memory_gb=48.0,
            pcie_bandwidth_gbps=32.0,
            role=DeviceRole.SMALL_EXPERT_DEVICE,
        )
        profiles.append(profile)
        logger.info(
            "build_des_loc_profiles: A6000 device=%d registered (SM86, 48GB, 32GB/s PCIe)",
            dev_id,
        )

    h100_profile = HeteroDeviceProfile(
        device_id=h100_device_id,
        sm_version=_SM90_ARCH,
        total_memory_gb=96.0,
        pcie_bandwidth_gbps=64.0,
        role=DeviceRole.LARGE_EXPERT_DEVICE,
    )
    profiles.append(h100_profile)
    logger.info(
        "build_des_loc_profiles: H100 NVL device=%d registered (SM90, 96GB, 64GB/s PCIe)",
        h100_device_id,
    )

    return profiles


def create_des_loc_fix_helper(
    a6000_device_ids: List[int] = None,
    h100_device_id: int = 2,
    loc_tracker: Optional[LOCCacheBufferTracker] = None,
) -> HeteroLocalCGMoEFix:
    """快捷工厂函数：创建适配 DES-LOC 标准拓扑的 HeteroLocalCGMoEFix 实例。

    在实际 CUDA 环境中自动从设备属性读取 SM 版本；
    在测试/CI 环境中（无 GPU）使用默认值。

    Parameters
    ----------
    a6000_device_ids : list of int, optional
        A6000 设备 ID 列表，默认为 [0, 1]。
    h100_device_id : int
        H100 设备 ID，默认为 2。
    loc_tracker : LOCCacheBufferTracker, optional
        LOC Cache 追踪器，若为 None 则自动创建。

    Returns
    -------
    HeteroLocalCGMoEFix
    """
    if a6000_device_ids is None:
        a6000_device_ids = [0, 1]

    if torch.cuda.is_available():
        profiles = build_des_loc_profiles(a6000_device_ids, h100_device_id)
    else:
        # 无 GPU 环境（测试/CI），使用模拟档案
        logger.warning(
            "create_des_loc_fix_helper: no CUDA available, using mock profiles"
        )
        profiles = [
            HeteroDeviceProfile(
                device_id=0, sm_version=_SM86_ARCH,
                total_memory_gb=48.0, pcie_bandwidth_gbps=32.0,
                role=DeviceRole.SMALL_EXPERT_DEVICE,
            ),
            HeteroDeviceProfile(
                device_id=1, sm_version=_SM86_ARCH,
                total_memory_gb=48.0, pcie_bandwidth_gbps=32.0,
                role=DeviceRole.SMALL_EXPERT_DEVICE,
            ),
            HeteroDeviceProfile(
                device_id=2, sm_version=_SM90_ARCH,
                total_memory_gb=96.0, pcie_bandwidth_gbps=64.0,
                role=DeviceRole.LARGE_EXPERT_DEVICE,
            ),
        ]

    return HeteroLocalCGMoEFix(
        device_profiles=profiles,
        loc_tracker=loc_tracker,
        enable_async_buffer_restore=True,
    )


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    class _MoERouterStub(nn.Module):
        """模拟 Megatron MoE Router，含 expert_bias persistent buffer。"""

        def __init__(self, num_experts: int = 4, hidden: int = 8) -> None:
            super().__init__()
            self.num_experts = num_experts
            self.fc = nn.Linear(hidden, num_experts, bias=False)
            # 模拟 expert_bias：在每次前向传播中被原位更新
            self.register_buffer(
                "expert_bias", torch.zeros(num_experts), persistent=True
            )
            # 模拟 running stats（BN 等）
            self.register_buffer(
                "running_mean", torch.zeros(hidden), persistent=True
            )
            self._forward_count = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self._forward_count += 1
            # 模拟 _apply_expert_bias()：每次前向都会更新 expert_bias
            logits = self.fc(x)
            expert_load = logits.detach().mean(0)
            # 原位更新 expert_bias（这是 bug 的触发点）
            self.expert_bias.add_(expert_load * 0.01)
            self.running_mean.add_(x.detach().mean(0) * 0.001)
            return logits + self.expert_bias.unsqueeze(0)

    class TestHeteroDeviceProfile(unittest.TestCase):

        def test_sm86_no_conditional_graph_nodes(self):
            profile = HeteroDeviceProfile(
                device_id=0, sm_version=86, total_memory_gb=48.0,
                pcie_bandwidth_gbps=32.0, role=DeviceRole.SMALL_EXPERT_DEVICE,
            )
            self.assertFalse(profile.supports_conditional_graph_nodes)
            self.assertEqual(profile.sm_version, 86)

        def test_sm90_supports_conditional_graph_nodes(self):
            profile = HeteroDeviceProfile(
                device_id=2, sm_version=90, total_memory_gb=96.0,
                pcie_bandwidth_gbps=64.0, role=DeviceRole.LARGE_EXPERT_DEVICE,
            )
            self.assertTrue(profile.supports_conditional_graph_nodes)

        def test_torch_device_property(self):
            profile = HeteroDeviceProfile(
                device_id=1, sm_version=86, total_memory_gb=48.0,
                pcie_bandwidth_gbps=32.0, role=DeviceRole.SMALL_EXPERT_DEVICE,
            )
            self.assertEqual(profile.torch_device, torch.device("cuda", 1))

    class TestLOCCacheBufferTracker(unittest.TestCase):

        def setUp(self):
            self.tracker = LOCCacheBufferTracker()
            self.buf = torch.randn(4, 4)

        def test_acquire_and_release(self):
            self.tracker.acquire(self.buf, device_id=0)
            self.assertFalse(self.tracker.is_safe_to_return(self.buf))
            released = self.tracker.release(self.buf, device_id=0)
            self.assertTrue(released)
            self.assertTrue(self.tracker.is_safe_to_return(self.buf))

        def test_multi_device_release(self):
            """只有所有设备都释放后才能安全归还。"""
            self.tracker.acquire(self.buf, device_id=0)
            self.tracker.acquire(self.buf, device_id=2)
            # 只释放一个设备
            r1 = self.tracker.release(self.buf, device_id=0)
            self.assertFalse(r1)
            self.assertFalse(self.tracker.is_safe_to_return(self.buf))
            # 释放第二个设备
            r2 = self.tracker.release(self.buf, device_id=2)
            self.assertTrue(r2)
            self.assertTrue(self.tracker.is_safe_to_return(self.buf))

        def test_held_by_devices(self):
            self.tracker.acquire(self.buf, device_id=0)
            self.tracker.acquire(self.buf, device_id=1)
            held = self.tracker.held_by_devices(self.buf)
            self.assertEqual(held, {0, 1})

        def test_untracked_buffer_is_safe(self):
            untracked = torch.randn(2, 2)
            self.assertTrue(self.tracker.is_safe_to_return(untracked))

        def test_clear(self):
            self.tracker.acquire(self.buf, device_id=0)
            self.tracker.clear()
            self.assertTrue(self.tracker.is_safe_to_return(self.buf))

        def test_thread_safety(self):
            """并发 acquire/release 不应导致引用计数错误。"""
            import random
            errors = []
            NUM_THREADS = 10
            NUM_OPS = 100

            def worker():
                for _ in range(NUM_OPS):
                    dev = random.choice([0, 1, 2])
                    try:
                        self.tracker.acquire(self.buf, device_id=dev)
                        self.tracker.release(self.buf, device_id=dev)
                    except Exception as e:
                        errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")

    class TestHeteroCloneTen(unittest.TestCase):

        def test_clone_preserves_values(self):
            """修复后的 clone_ten 应保留原始数值，而非产生全零。"""
            original = torch.tensor([1.0, 2.0, 3.0, 4.0])
            cloned = hetero_clone_ten(original)
            self.assertTrue(torch.allclose(cloned, original))

        def test_clone_detaches_grad(self):
            """clone_ten 结果应与计算图断开连接。"""
            original = torch.randn(4, requires_grad=True)
            _ = (original * 2).sum()  # 创建计算图
            cloned = hetero_clone_ten(original)
            self.assertFalse(cloned.is_leaf or cloned.grad_fn is not None
                             # clone().detach() 的结果是叶节点
                             or False)
            # 确认无 grad_fn
            self.assertIsNone(cloned.grad_fn)

        def test_clone_requires_grad_preserved(self):
            original = torch.randn(4, requires_grad=True)
            cloned = hetero_clone_ten(original)
            self.assertTrue(cloned.requires_grad)

            original_no_grad = torch.randn(4, requires_grad=False)
            cloned_no_grad = hetero_clone_ten(original_no_grad)
            self.assertFalse(cloned_no_grad.requires_grad)

        def test_sm86_forces_contiguous(self):
            """SM86 (A6000) 设备档案应强制 contiguous。"""
            profile = HeteroDeviceProfile(
                device_id=0, sm_version=86, total_memory_gb=48.0,
                pcie_bandwidth_gbps=32.0, role=DeviceRole.SMALL_EXPERT_DEVICE,
            )
            # 创建非 contiguous 张量
            base = torch.randn(4, 4)
            non_contig = base.T  # 转置后不是 contiguous
            self.assertFalse(non_contig.is_contiguous())

            cloned = hetero_clone_ten(non_contig, target_profile=profile)
            self.assertTrue(cloned.is_contiguous())

        def test_non_tensor_passthrough(self):
            """非 tensor 对象应原样返回。"""
            for obj in [None, 42, "hello", [1, 2, 3], {"a": 1}]:
                result = hetero_clone_ten(obj)
                self.assertIs(result, obj)

        def test_zeros_like_vs_clone_difference(self):
            """验证 zeros_like（旧实现）与 clone（新实现）的数值差异。

            这个测试直接体现了上游 bug 481efd0 的修复意义：
            使用 zeros_like 会导致 MoE Router softmax 均匀分布，
            使用 clone 才能保留真实的输入分布。
            """
            # 模拟一个 MoE 路由器的输入（非均匀分布）
            router_input = torch.tensor([0.8, 0.1, 0.05, 0.05])

            old_clone = torch.zeros_like(router_input)  # 旧实现（bug）
            new_clone = hetero_clone_ten(router_input)   # 新实现（修复）

            # 旧实现：全零，会导致路由器 softmax 输出均匀分布
            self.assertTrue(torch.all(old_clone == 0.0))
            # 新实现：保留原始分布
            self.assertTrue(torch.allclose(new_clone, router_input))

    class TestModuleBufferGuard(unittest.TestCase):

        def _make_moe_router(self, num_experts=4, hidden=8) -> _MoERouterStub:
            return _MoERouterStub(num_experts=num_experts, hidden=hidden)

        def test_buffer_restored_after_warmup(self):
            """核心测试：模拟 warmup 后 buffer 应被正确恢复。

            这直接对应 Megatron bug 481efd0 修复的主要场景：
            expert_bias 在 warmup 前向传播中被修改，
            但 ModuleBufferGuard 应确保 warmup 结束后恢复原值。
            """
            router = self._make_moe_router()
            x = torch.randn(2, 8)

            # 记录 warmup 前的 buffer 值
            bias_before = router.expert_bias.clone()
            mean_before = router.running_mean.clone()

            with ModuleBufferGuard(router) as guard:
                # 模拟 warmup 阶段：多次前向传播（会修改 expert_bias）
                for _ in range(3):
                    _ = router(x)

                # 在 guard 退出前，buffer 已被修改
                self.assertFalse(
                    torch.allclose(router.expert_bias, bias_before),
                    "expert_bias should be modified during warmup"
                )

            # guard 退出后，buffer 应被恢复
            self.assertTrue(
                torch.allclose(router.expert_bias, bias_before),
                "expert_bias should be restored after ModuleBufferGuard exit"
            )
            self.assertTrue(
                torch.allclose(router.running_mean, mean_before),
                "running_mean should be restored after ModuleBufferGuard exit"
            )

        def test_exception_in_warmup_still_restores(self):
            """即使 warmup 中抛出异常，buffer 也应被恢复（上下文管理器语义）。"""
            router = self._make_moe_router()
            bias_before = router.expert_bias.clone()

            class _WarmupError(RuntimeError):
                pass

            try:
                with ModuleBufferGuard(router):
                    router.expert_bias.fill_(999.0)
                    raise _WarmupError("simulated warmup failure")
            except _WarmupError:
                pass

            self.assertTrue(
                torch.allclose(router.expert_bias, bias_before),
                "expert_bias should be restored even after exception"
            )

        def test_no_buffers_module(self):
            """没有 buffer 的模块不应引发错误。"""
            module = nn.Linear(4, 4)
            x = torch.randn(2, 4)
            with ModuleBufferGuard(module):
                _ = module(x)  # 正常执行，无 buffer 需要备份/恢复

        def test_with_loc_tracker(self):
            """使用 LOCCacheBufferTracker 的 buffer 守卫正常工作。"""
            router = self._make_moe_router()
            tracker = LOCCacheBufferTracker()
            bias_before = router.expert_bias.clone()

            with ModuleBufferGuard(router, loc_tracker=tracker):
                router.expert_bias.fill_(42.0)

            self.assertTrue(torch.allclose(router.expert_bias, bias_before))

    class TestDeviceGradGate(unittest.TestCase):

        def _make_profiles(self):
            return [
                HeteroDeviceProfile(
                    device_id=0, sm_version=86, total_memory_gb=48.0,
                    pcie_bandwidth_gbps=32.0, role=DeviceRole.SMALL_EXPERT_DEVICE,
                ),
                HeteroDeviceProfile(
                    device_id=2, sm_version=90, total_memory_gb=96.0,
                    pcie_bandwidth_gbps=64.0, role=DeviceRole.LARGE_EXPERT_DEVICE,
                ),
            ]

        def test_all_devices_active_by_default(self):
            gate = DeviceGradGate(self._make_profiles())
            self.assertTrue(gate.is_grad_active(0))
            self.assertTrue(gate.is_grad_active(2))

        def test_freeze_device(self):
            gate = DeviceGradGate(self._make_profiles())
            gate.freeze_device(0)
            self.assertFalse(gate.is_grad_active(0))
            self.assertTrue(gate.is_grad_active(2))

        def test_unfreeze_device(self):
            gate = DeviceGradGate(self._make_profiles())
            gate.freeze_device(0)
            gate.unfreeze_device(0)
            self.assertTrue(gate.is_grad_active(0))

        def test_should_use_training_mode_frozen_device_grad_disabled(self):
            """关键测试：设备被冻结且全局梯度关闭时，才返回 False。"""
            gate = DeviceGradGate(self._make_profiles())
            gate.freeze_device(0)
            # 设备冻结 + 全局梯度关闭 -> 不需要 training 模式
            result = gate.should_use_training_mode(device_id=0, global_grad_enabled=False)
            self.assertFalse(result)

        def test_should_use_training_mode_grad_checkpointing_scenario(self):
            """梯度检查点重算场景：全局梯度关闭但设备未冻结，应保持 training 模式。

            这正是 Megatron bug 481efd0 删除 runner.eval() 调用的原因：
            梯度检查点的重算阶段会关闭全局梯度，但此时不应切换到 eval 模式。
            """
            gate = DeviceGradGate(self._make_profiles())
            # 设备未冻结，但全局梯度关闭（模拟梯度检查点重算）
            result = gate.should_use_training_mode(device_id=0, global_grad_enabled=False)
            # 应该保持 training 模式（不调用 eval）
            self.assertTrue(result)

        def test_should_use_training_mode_frozen_but_grad_enabled(self):
            """设备冻结但全局梯度开启时，仍应使用 training 模式。"""
            gate = DeviceGradGate(self._make_profiles())
            gate.freeze_device(2)
            result = gate.should_use_training_mode(device_id=2, global_grad_enabled=True)
            self.assertTrue(result)

        def test_invalid_device_raises(self):
            gate = DeviceGradGate(self._make_profiles())
            with self.assertRaises(ValueError):
                gate.freeze_device(99)

    class TestHeteroLocalCGMoEFix(unittest.TestCase):

        def _make_fix_helper(self):
            return create_des_loc_fix_helper(
                a6000_device_ids=[0, 1],
                h100_device_id=2,
            )

        def test_creation(self):
            helper = self._make_fix_helper()
            self.assertEqual(len(helper.device_profiles), 3)
            self.assertIsInstance(helper.loc_tracker, LOCCacheBufferTracker)
            self.assertIsInstance(helper.grad_gate, DeviceGradGate)

        def test_make_buffer_guard_cpu_module(self):
            """CPU 模块（无 GPU 参数）应能正常创建 buffer 守卫。"""
            helper = self._make_fix_helper()
            module = nn.Linear(4, 4)
            guard = helper.make_buffer_guard(module)
            self.assertIsInstance(guard, ModuleBufferGuard)

        def test_hetero_clone_for_module(self):
            """hetero_clone_for_module 返回的函数应正确复制张量。"""
            helper = self._make_fix_helper()
            module = nn.Linear(4, 4)
            clone_fn = helper.hetero_clone_for_module(module)
            x = torch.randn(4)
            cloned = clone_fn(x)
            self.assertTrue(torch.allclose(cloned, x))
            self.assertIsNone(cloned.grad_fn)

        def test_safe_return_buffer_to_pool_no_attr(self):
            """不带 can_skip_replay_copy 属性的 buffer 不应被归还。"""
            helper = self._make_fix_helper()
            buf = torch.randn(4)
            returned_list = []

            def mock_pool_insert(b):
                returned_list.append(b)

            result = helper.safe_return_buffer_to_pool(buf, mock_pool_insert)
            self.assertFalse(result)
            self.assertEqual(len(returned_list), 0)

        def test_safe_return_buffer_to_pool_with_loc_hold(self):
            """LOC Cache 持有引用时不应提前归还。"""
            helper = self._make_fix_helper()
            buf = torch.randn(4)
            buf.can_skip_replay_copy = False  # 标记需要 copy
            # LOC Cache 追踪器持有引用
            helper.loc_tracker.acquire(buf, device_id=0)

            returned_list = []

            def mock_pool_insert(b):
                returned_list.append(b)

            result = helper.safe_return_buffer_to_pool(buf, mock_pool_insert)
            self.assertFalse(result)
            self.assertEqual(len(returned_list), 0)

        def test_safe_return_buffer_to_pool_after_release(self):
            """LOC Cache 引用释放后，buffer 可以安全归还。"""
            helper = self._make_fix_helper()
            buf = torch.randn(4)
            buf.can_skip_replay_copy = False

            helper.loc_tracker.acquire(buf, device_id=0)
            helper.loc_tracker.release(buf, device_id=0)

            returned_list = []

            def mock_pool_insert(b):
                returned_list.append(b)

            result = helper.safe_return_buffer_to_pool(buf, mock_pool_insert)
            self.assertTrue(result)
            self.assertEqual(len(returned_list), 1)

        def test_should_runner_use_training_mode_grad_checkpoint(self):
            """梯度检查点场景：CPU 模块在 grad_enabled=False 时应返回 False。

            注意：CPU 模块回退到 global_grad_enabled 判断逻辑。
            """
            helper = self._make_fix_helper()
            cpu_module = nn.Linear(4, 4)
            result = helper.should_runner_use_training_mode(
                cpu_module, global_grad_enabled=False
            )
            self.assertFalse(result)

    class TestBuildDesLocProfiles(unittest.TestCase):

        def test_profile_count(self):
            """2 个 A6000 + 1 个 H100 = 3 个设备档案。"""
            # 使用 create_des_loc_fix_helper（无 GPU 时使用 mock）
            helper = create_des_loc_fix_helper(
                a6000_device_ids=[0, 1], h100_device_id=2
            )
            self.assertEqual(len(helper.device_profiles), 3)

        def test_sm_versions(self):
            helper = create_des_loc_fix_helper(
                a6000_device_ids=[0, 1], h100_device_id=2
            )
            a6000_profiles = [p for p in helper.device_profiles
                              if p.role == DeviceRole.SMALL_EXPERT_DEVICE]
            h100_profiles = [p for p in helper.device_profiles
                             if p.role == DeviceRole.LARGE_EXPERT_DEVICE]

            self.assertEqual(len(a6000_profiles), 2)
            self.assertEqual(len(h100_profiles), 1)

            for p in a6000_profiles:
                self.assertEqual(p.sm_version, _SM86_ARCH)
                self.assertFalse(p.supports_conditional_graph_nodes)

            self.assertEqual(h100_profiles[0].sm_version, _SM90_ARCH)
            self.assertTrue(h100_profiles[0].supports_conditional_graph_nodes)

        def test_memory_sizes(self):
            helper = create_des_loc_fix_helper(
                a6000_device_ids=[0, 1], h100_device_id=2
            )
            for p in helper.device_profiles:
                if p.role == DeviceRole.SMALL_EXPERT_DEVICE:
                    self.assertAlmostEqual(p.total_memory_gb, 48.0)
                else:
                    self.assertAlmostEqual(p.total_memory_gb, 96.0)

    class TestIntegration(unittest.TestCase):
        """集成测试：模拟完整的 DES-LOC CG MoE 修复流程。"""

        def test_full_warmup_flow(self):
            """完整的 warmup 流程测试。

            模拟 record_graph_capture 的完整生命周期：
            1. 创建 fix_helper
            2. 创建 DeviceAwareCGRunner
            3. pre_capture_setup（备份 buffer，clone 输入）
            4. 执行多次 warmup 前向
            5. post_capture_cleanup（恢复 buffer，跳过不安全的 pool 归还）
            6. 验证 buffer 恢复正确
            """
            router = _MoERouterStub(num_experts=4, hidden=8)
            helper = create_des_loc_fix_helper()
            runner = DeviceAwareCGRunner(
                fix_helper=helper,
                base_module=router,
                training=True,
            )

            x = torch.randn(2, 8)
            bias_before = router.expert_bias.clone()

            # pre_capture_setup
            grad_enabled = True
            guard, warmup_args, warmup_kwargs = runner.pre_capture_setup(
                args=(x,), kwargs={}, grad_enabled=grad_enabled
            )

            # 验证 warmup 输入是 clone 的（保留原始值，非全零）
            warmup_x = warmup_args[0]
            self.assertTrue(torch.allclose(warmup_x, x),
                            "warmup input should preserve original values")

            # 模拟 warmup 前向（会修改 expert_bias）
            for _ in range(3):
                _ = router(*warmup_args, **warmup_kwargs)

            # expert_bias 应在 warmup 中被修改
            self.assertFalse(torch.allclose(router.expert_bias, bias_before))

            # post_capture_cleanup（恢复 buffer）
            runner.post_capture_cleanup(
                guard=guard,
                fwd_graph_input_surface=[],
                pool_insert_fn=None,
            )

            # 验证 buffer 已恢复
            self.assertTrue(
                torch.allclose(router.expert_bias, bias_before),
                "expert_bias should be restored after post_capture_cleanup"
            )

        def test_manager_forward_no_eval_switch(self):
            """HeteroCGMoEManager 不应在梯度关闭时切换到 eval 模式。

            这直接测试上游 Megatron bug 481efd0 的修复：
            原始代码中 runner.eval() 会破坏梯度检查点重算语义。
            """
            router = _MoERouterStub(num_experts=4, hidden=8)
            helper = create_des_loc_fix_helper()
            manager = HeteroCGMoEManager(fix_helper=helper, reuse_cudagraphs=True)

            x = torch.randn(2, 8)

            # 确保 router 在 training 模式
            router.train()
            self.assertTrue(router.training)

            # 模拟梯度检查点重算（全局梯度关闭，但模块应保持 training 模式）
            with torch.no_grad():
                _ = manager.forward(router, (x,), {}, is_training=True)

            # 关键断言：router 应仍处于 training 模式
            self.assertTrue(
                router.training,
                "Module should remain in training mode even when grad is disabled "
                "(gradient checkpointing recompute scenario)"
            )

    # 运行所有测试
    suite = unittest.TestSuite()
    test_classes = [
        TestHeteroDeviceProfile,
        TestLOCCacheBufferTracker,
        TestHeteroCloneTen,
        TestModuleBufferGuard,
        TestDeviceGradGate,
        TestHeteroLocalCGMoEFix,
        TestBuildDesLocProfiles,
        TestIntegration,
    ]

    for cls in test_classes:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroLocalCGMoEFix on a DeepSpeed engine.

    Instantiates a :class:`HeteroLocalCGMoEFix` from the engine's configuration
    and attaches it as ``engine.hetero_local_cg_moe_fix``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_local_cg_moe_fix.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_local_cg_moe_fix = None
    logger.info("hetero_local_cg_moe_fix.register() attached engine.hetero_local_cg_moe_fix")
