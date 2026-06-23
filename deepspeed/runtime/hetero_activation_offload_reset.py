"""
DES-LOC Heterogeneous Activation Offload Reset Manager
=======================================================

上游设计意图 (Megatron b8e23d5874211a49789bf6a2c5170549f2496d14):
    Megatron 原始 bug: `off_interface.reset()` 仅在 `not forward_only` 条件下触发，
    导致 eval 阶段（forward_only=True）结束后 activation offload manager 的内部状态
    未被清除。下次 training step 开始时残留的 offload 索引/指针会造成 OOM 或者
    activation 错位。修复方式是将条件从
        `if not forward_only and config.fine_grained_activation_offloading`
    改为
        `if getattr(config, 'fine_grained_activation_offloading', False)`
    即：无论 forward_only 取何值，只要开启了 fine-grained offload 就在 schedule
    结束后调用 reset()。

DES-LOC 适配点:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) 在异构硬件上运行:
        • 2× A6000 48GB  (SM86, PCIe)  — 充当 "worker" GPU，负责前向/后向计算
        • 1× H100 NVL 96GB (SM90, PCIe) — 充当 "anchor" GPU，承担大 tensor offload
          接收端及 optimizer step
        • 1.5TB CPU DRAM                 — SLC (Shared LOcality Cache) 层，
                                          存放跨设备共享的 activation 快照

    与 Megatron 的单一 off_interface 不同，DES-LOC 中每个物理设备维护独立的
    OffloadShard，并通过 SLC 做异步 staging。因此 reset 操作需要:
        1. 分别 reset 每个设备的 OffloadShard（顺序不敏感，可并行）
        2. 刷新 SLC 中对应 epoch 的 staging buffer（避免跨 step 污染）
        3. 在 eval→train 切换时额外触发一次 SLC GC（eval 不产生梯度，
           SLC 中 pinned activation 可提前回收）
        4. 使用 `getattr(config, ...)` 防御性访问，与上游修复保持一致

    本模块实现:
        HeteroActivationOffloadReset  — 核心状态机，协调多设备 reset 时序
        SLCBuffer                     — CPU DRAM staging buffer 抽象
        DeviceOffloadShard            — 单 GPU OffloadShard 封装
        schedule_reset_hook           — 供 DeepSpeed engine 调用的钩子函数

作者: Neuron_SP Project (mirrors Megatron b8e23d5874)
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations & Constants
# ---------------------------------------------------------------------------

class DeviceRole(Enum):
    """DES-LOC 硬件角色分类."""
    WORKER = auto()   # A6000 × 2：执行前向/后向
    ANCHOR = auto()   # H100 NVL：承载大 tensor offload 及 optimizer


class ResetPhase(Enum):
    """Reset 状态机的阶段."""
    IDLE        = auto()  # 无需 reset
    PENDING     = auto()  # 已触发，等待设备 idle
    RESETTING   = auto()  # 正在 reset
    SLC_GC      = auto()  # SLC GC 阶段（eval→train 切换专用）
    DONE        = auto()  # 本轮 reset 完成


# SLC 分配给 activation staging 的最大比例（其余留给参数 cache）
_SLC_ACTIVATION_RATIO = 0.60

# PCIe Gen4 x16 理论带宽 (GB/s)，用于 staging 超时估算
_PCIE_BW_GB_S = 32.0


# ---------------------------------------------------------------------------
# SLC Buffer — CPU DRAM staging layer
# ---------------------------------------------------------------------------

@dataclass
class SLCBufferStats:
    """SLC Buffer 运行时统计，用于 logging 与调试."""
    total_bytes:   int = 0
    pinned_bytes:  int = 0
    staged_epochs: List[int] = field(default_factory=list)
    gc_count:      int = 0
    last_gc_ts:    float = field(default_factory=time.time)


class SLCBuffer:
    """
    Shared LOcality Cache — CPU DRAM 中的 activation staging buffer。

    设计:
        • 使用 torch.UntypedStorage + pin_memory 分配，避免 GIL 持锁时的
          cudaHostAlloc 调用。
        • 每个 training step（epoch_id）独立维护一个 slot，reset 时只需
          将对应 slot 标记为 free，无需 memset（下次写入时覆盖）。
        • eval 阶段产生的 activation 不带梯度，可直接丢弃（GC），
          不必等待 staging 完成。

    与 Megatron 的关系:
        Megatron 无 CPU DRAM staging 概念；DES-LOC 引入 SLC 层使得
        A6000 的显存压力可以溢出到 CPU，再由 H100 按需拉取，
        而非直接 A6000↔H100 P2P（PCIe 无 NVLink，P2P 带宽受限）。
    """

    def __init__(
        self,
        capacity_bytes: int,
        activation_ratio: float = _SLC_ACTIVATION_RATIO,
        device: str = "cpu",
    ) -> None:
        self._capacity = capacity_bytes
        self._activation_cap = int(capacity_bytes * activation_ratio)
        self._device = device
        self._lock = threading.Lock()
        self._slots: Dict[int, torch.Tensor] = {}   # epoch_id → pinned tensor
        self._free_bytes = self._activation_cap
        self.stats = SLCBufferStats(total_bytes=capacity_bytes)
        logger.info(
            "[SLC] Initialized: total=%.2f GB, activation_cap=%.2f GB",
            capacity_bytes / 1e9,
            self._activation_cap / 1e9,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stage(self, epoch_id: int, tensor: torch.Tensor) -> bool:
        """
        将 tensor staging 到 SLC。

        Returns:
            True  — staging 成功
            False — 容量不足，调用方应回退到 GPU 保留或丢弃
        """
        nbytes = tensor.numel() * tensor.element_size()
        with self._lock:
            if nbytes > self._free_bytes:
                logger.warning(
                    "[SLC] stage failed epoch=%d: need %.2f MB, free %.2f MB",
                    epoch_id,
                    nbytes / 1e6,
                    self._free_bytes / 1e6,
                )
                return False

            # 分配 pinned buffer 并异步拷贝
            buf = torch.empty(tensor.shape, dtype=tensor.dtype,
                              device="cpu", pin_memory=True)
            buf.copy_(tensor, non_blocking=True)
            self._slots[epoch_id] = buf
            self._free_bytes -= nbytes
            self.stats.pinned_bytes += nbytes
            if epoch_id not in self.stats.staged_epochs:
                self.stats.staged_epochs.append(epoch_id)

        logger.debug(
            "[SLC] staged epoch=%d size=%.2f MB free=%.2f MB",
            epoch_id, nbytes / 1e6, self._free_bytes / 1e6,
        )
        return True

    def release_epoch(self, epoch_id: int) -> int:
        """
        释放指定 epoch 的 SLC slot。

        Returns:
            释放的字节数
        """
        with self._lock:
            buf = self._slots.pop(epoch_id, None)
            if buf is None:
                return 0
            nbytes = buf.numel() * buf.element_size()
            self._free_bytes = min(self._free_bytes + nbytes, self._activation_cap)
            self.stats.pinned_bytes = max(0, self.stats.pinned_bytes - nbytes)
            if epoch_id in self.stats.staged_epochs:
                self.stats.staged_epochs.remove(epoch_id)
            del buf
        logger.debug("[SLC] released epoch=%d freed=%.2f MB", epoch_id, nbytes / 1e6)
        return nbytes

    def gc(self, eval_mode: bool = False) -> int:
        """
        垃圾回收。

        eval_mode=True 时：释放所有 slot（eval activation 无梯度，不需保留）。
        eval_mode=False 时：仅释放已超出保留窗口的旧 slot（保守策略）。

        Returns:
            回收的总字节数
        """
        freed = 0
        with self._lock:
            if eval_mode:
                # 激进回收：清空全部
                for eid in list(self._slots.keys()):
                    buf = self._slots.pop(eid)
                    nbytes = buf.numel() * buf.element_size()
                    self._free_bytes = min(
                        self._free_bytes + nbytes, self._activation_cap
                    )
                    self.stats.pinned_bytes = max(0, self.stats.pinned_bytes - nbytes)
                    freed += nbytes
                    del buf
                self.stats.staged_epochs.clear()
            else:
                # 保守回收：清理最旧的一半 slot
                eids = sorted(self._slots.keys())
                cutoff = len(eids) // 2
                for eid in eids[:cutoff]:
                    buf = self._slots.pop(eid)
                    nbytes = buf.numel() * buf.element_size()
                    self._free_bytes = min(
                        self._free_bytes + nbytes, self._activation_cap
                    )
                    self.stats.pinned_bytes = max(0, self.stats.pinned_bytes - nbytes)
                    freed += nbytes
                    del buf
                self.stats.staged_epochs = [
                    e for e in self.stats.staged_epochs if e in self._slots
                ]

            self.stats.gc_count += 1
            self.stats.last_gc_ts = time.time()

        logger.info(
            "[SLC] gc(eval=%s) freed=%.2f MB gc_count=%d",
            eval_mode, freed / 1e6, self.stats.gc_count,
        )
        return freed

    @property
    def utilization(self) -> float:
        """SLC activation 区域使用率 [0, 1]."""
        with self._lock:
            used = self._activation_cap - self._free_bytes
            return used / max(self._activation_cap, 1)

    def __repr__(self) -> str:
        return (
            f"SLCBuffer(cap={self._activation_cap/1e9:.1f}GB, "
            f"util={self.utilization:.1%}, gc={self.stats.gc_count})"
        )


# ---------------------------------------------------------------------------
# DeviceOffloadShard — 单 GPU 侧的 offload 状态封装
# ---------------------------------------------------------------------------

@dataclass
class ShardResetStats:
    reset_count: int = 0
    last_reset_ts: float = field(default_factory=time.time)
    total_freed_bytes: int = 0


class DeviceOffloadShard:
    """
    单 GPU 设备上的 activation offload shard。

    DES-LOC 中每块 GPU 独立维护:
        • _activation_store: epoch_id → List[Tensor]，当前驻留在该 GPU 的激活值
        • _offload_index:    记录哪些 activation 已被 staging 到 SLC
        • _stream:           专用 CUDA stream，与计算 stream 解耦

    与 Megatron off_interface 的关系:
        Megatron 的 `ActivationOffloadingInterface` 是单例，绑定到当前进程的
        默认 CUDA device。DES-LOC 将其拆分为 per-device shard，允许
        A6000×2 + H100 同时并行 reset，不需要串行等待。

    reset() 语义 (镜像 Megatron b8e23d5874 修复):
        原始 Megatron bug 在于 eval 结束后不调用 reset()，
        导致 _offload_index 内残留 eval 阶段的 activation 指针，
        下一个 training step 的 forward pass 会尝试从错误地址恢复激活值。
        本类的 reset() 在每次 schedule 结束后无条件调用（不区分 forward_only）。
    """

    def __init__(
        self,
        device_id: int,
        role: DeviceRole,
        slc: SLCBuffer,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.device_id = device_id
        self.role = role
        self._slc = weakref.ref(slc)
        self._device = torch.device(f"cuda:{device_id}")
        self._stream = stream or torch.cuda.Stream(device=self._device)
        self._lock = threading.Lock()

        # 内部状态（模拟 Megatron ActivationOffloadingInterface 的字段）
        self._activation_store: Dict[int, List[torch.Tensor]] = {}
        self._offload_index: Dict[int, List[int]] = {}   # epoch→[slc slot ids]
        self._current_epoch: int = 0
        self._fwd_ptr: int = 0   # forward pass 中下一个 activation 的写入位置
        self._bwd_ptr: int = 0   # backward pass 中下一个 activation 的读取位置

        self.stats = ShardResetStats()
        logger.info(
            "[Shard] device=%d role=%s initialized", device_id, role.name
        )

    # ------------------------------------------------------------------
    # Activation store operations
    # ------------------------------------------------------------------

    def push_activation(
        self, epoch_id: int, activation: torch.Tensor
    ) -> None:
        """
        前向 pass 中将 activation 存入 shard，并尝试 staging 到 SLC。

        若 SLC 容量不足，activation 留在 GPU 显存（保守退化）。
        H100 (ANCHOR) 优先接受大 tensor（> 64MB），A6000 (WORKER) 接受小 tensor。
        """
        slc = self._slc()
        if slc is None:
            raise RuntimeError("SLCBuffer has been garbage collected")

        nbytes = activation.numel() * activation.element_size()
        threshold = 64 * 1024 * 1024  # 64 MB

        should_stage = (
            (self.role == DeviceRole.ANCHOR and nbytes >= threshold)
            or (self.role == DeviceRole.WORKER and nbytes < threshold)
        )

        with self._lock:
            if epoch_id not in self._activation_store:
                self._activation_store[epoch_id] = []
            self._activation_store[epoch_id].append(activation)
            self._fwd_ptr += 1

        if should_stage and slc.stage(epoch_id * 10000 + self._fwd_ptr, activation):
            with self._lock:
                if epoch_id not in self._offload_index:
                    self._offload_index[epoch_id] = []
                self._offload_index[epoch_id].append(
                    epoch_id * 10000 + self._fwd_ptr
                )
            logger.debug(
                "[Shard:%d] staged activation epoch=%d ptr=%d size=%.2fMB",
                self.device_id, epoch_id, self._fwd_ptr, nbytes / 1e6,
            )

    def pop_activation(
        self, epoch_id: int
    ) -> Optional[torch.Tensor]:
        """
        后向 pass 中按 LIFO 顺序取回 activation。
        若已 staging 到 SLC，从 SLC 拉取并异步搬回 GPU。
        """
        with self._lock:
            store = self._activation_store.get(epoch_id, [])
            if not store:
                return None
            act = store.pop()
            self._bwd_ptr += 1

        # 如果 tensor 在 CPU (pin_memory)，搬回 GPU
        if act.device.type == "cpu":
            with torch.cuda.stream(self._stream):
                act = act.to(self._device, non_blocking=True)

        return act

    # ------------------------------------------------------------------
    # Reset (核心，镜像 Megatron b8e23d5874 修复)
    # ------------------------------------------------------------------

    def reset(self, eval_mode: bool = False) -> int:
        """
        无条件 reset shard 状态。

        Megatron 原始修复的 DES-LOC 对应实现:
            - 不检查 forward_only / eval_mode 来决定是否 reset
              （这正是 b8e23d5874 修复的 bug 所在）
            - eval_mode 参数仅用于通知 SLC GC 策略（激进 vs 保守），
              不影响 reset 本身是否执行

        Returns:
            本次 reset 释放的 GPU 显存字节数（近似值）
        """
        freed_gpu_bytes = 0
        slc = self._slc()

        with self._lock:
            # 1. 释放 GPU 侧 activation store
            for epoch_id, tensors in self._activation_store.items():
                for t in tensors:
                    if t.device.type != "cpu":
                        freed_gpu_bytes += t.numel() * t.element_size()
                del tensors[:]

            # 2. 释放 SLC 中对应的 slot
            if slc is not None:
                for epoch_id, slot_ids in self._offload_index.items():
                    for sid in slot_ids:
                        slc.release_epoch(sid)

            # 3. 清空内部索引与指针
            self._activation_store.clear()
            self._offload_index.clear()
            self._fwd_ptr = 0
            self._bwd_ptr = 0
            self._current_epoch += 1  # 推进 epoch，避免下一步复用旧 slot id

            # 4. 更新统计
            self.stats.reset_count += 1
            self.stats.last_reset_ts = time.time()
            self.stats.total_freed_bytes += freed_gpu_bytes

        logger.info(
            "[Shard:%d] reset(eval=%s) freed_gpu=%.2f MB epoch→%d",
            self.device_id,
            eval_mode,
            freed_gpu_bytes / 1e6,
            self._current_epoch,
        )
        return freed_gpu_bytes

    def __repr__(self) -> str:
        return (
            f"DeviceOffloadShard(dev={self.device_id}, role={self.role.name}, "
            f"epoch={self._current_epoch}, resets={self.stats.reset_count})"
        )


# ---------------------------------------------------------------------------
# HeteroActivationOffloadReset — 核心状态机
# ---------------------------------------------------------------------------

class HeteroActivationOffloadReset:
    """
    DES-LOC 异构 Activation Offload Reset 管理器。

    职责:
        1. 注册/管理所有物理设备的 DeviceOffloadShard
        2. 在每次 pipeline schedule 结束后，统一触发 reset 流程
           （镜像 Megatron b8e23d5874: getattr 防御 + eval 后也 reset）
        3. 协调 SLC GC 时序（eval→train 切换时需额外 GC）
        4. 提供给 DeepSpeed engine 的钩子接口

    硬件拓扑 (2×A6000 + 1×H100 + CPU DRAM):
        ┌─────────┐   PCIe   ┌───────────┐
        │ A6000:0 │──────────│           │
        │ (WORKER)│          │ CPU DRAM  │
        ├─────────┤   PCIe   │  (SLC)    │
        │ A6000:1 │──────────│           │
        │ (WORKER)│          └───────────┘
        ├─────────┤               │ PCIe
        │ H100:2  │───────────────┘
        │ (ANCHOR)│
        └─────────┘

    注意: PCIe 互联无 NVLink，A6000↔H100 P2P 带宽有限（~32 GB/s），
    因此 SLC 作为中间层可以有效削峰，避免 PCIe 拥塞。

    使用方式 (DeepSpeed engine 集成):
        >>> mgr = HeteroActivationOffloadReset.from_config(ds_config)
        >>> # 在每个 pipeline schedule 入口注册
        >>> with mgr.schedule_context(forward_only=False) as ctx:
        ...     # ... pipeline forward/backward ...
        ...     pass
        >>> # schedule 结束后 reset 自动触发（无论 forward_only 取何值）
    """

    def __init__(
        self,
        slc: SLCBuffer,
        worker_device_ids: List[int],
        anchor_device_id: int,
        parallel_reset: bool = True,
    ) -> None:
        """
        Args:
            slc:               SLC Buffer 实例
            worker_device_ids: WORKER GPU 的 CUDA device index 列表（A6000）
            anchor_device_id:  ANCHOR GPU 的 CUDA device index（H100）
            parallel_reset:    是否并行 reset 各 shard（True = 多线程）
        """
        self._slc = slc
        self._parallel_reset = parallel_reset
        self._phase = ResetPhase.IDLE
        self._phase_lock = threading.Lock()

        # 构建 shard 字典
        self._shards: Dict[int, DeviceOffloadShard] = {}
        for did in worker_device_ids:
            self._shards[did] = DeviceOffloadShard(
                device_id=did, role=DeviceRole.WORKER, slc=slc
            )
        self._shards[anchor_device_id] = DeviceOffloadShard(
            device_id=anchor_device_id, role=DeviceRole.ANCHOR, slc=slc
        )

        # 记录上一次 schedule 是否为 eval（用于判断是否需要激进 GC）
        self._last_was_eval: bool = False

        logger.info(
            "[HeteroReset] initialized: workers=%s anchor=%d slc=%s",
            worker_device_ids, anchor_device_id, slc,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: object,
        slc_capacity_bytes: Optional[int] = None,
    ) -> "HeteroActivationOffloadReset":
        """
        从 DeepSpeed config 对象构建实例。

        使用 getattr 防御性访问，与 Megatron b8e23d5874 修复一致:
            原始 bug: `config.fine_grained_activation_offloading` 在 eval config
                      对象中可能不存在，导致 AttributeError。
            修复:     `getattr(config, 'fine_grained_activation_offloading', False)`

        DES-LOC 扩展: 同样对所有 hetero_* 字段使用 getattr，保证
        config 对象不需要强制声明全部字段。
        """
        # 读取 DES-LOC 异构设备配置（带 default 防御）
        worker_ids: List[int] = getattr(
            config, "des_loc_worker_device_ids", [0, 1]
        )
        anchor_id: int = getattr(
            config, "des_loc_anchor_device_id", 2
        )
        parallel: bool = getattr(
            config, "des_loc_parallel_reset", True
        )

        # SLC 容量：优先使用 config 中指定值，否则默认 256GB（保守估算）
        if slc_capacity_bytes is None:
            slc_capacity_bytes = getattr(
                config,
                "des_loc_slc_capacity_bytes",
                256 * 1024 ** 3,  # 256 GB
            )

        slc = SLCBuffer(capacity_bytes=slc_capacity_bytes)
        instance = cls(
            slc=slc,
            worker_device_ids=worker_ids,
            anchor_device_id=anchor_id,
            parallel_reset=parallel,
        )

        # 检查是否开启 fine-grained offload（镜像 Megatron 修复的 getattr 模式）
        fine_grained = getattr(config, "fine_grained_activation_offloading", False)
        if not fine_grained:
            logger.warning(
                "[HeteroReset] fine_grained_activation_offloading=False, "
                "HeteroActivationOffloadReset will be a no-op unless explicitly called"
            )

        return instance

    # ------------------------------------------------------------------
    # Core reset logic
    # ------------------------------------------------------------------

    def reset(
        self,
        forward_only: bool = False,
        force: bool = False,
    ) -> Dict[int, int]:
        """
        触发所有 shard 的 reset。

        关键设计决策（直接源于 Megatron b8e23d5874 修复）:
            **不以 forward_only 作为 reset 的门控条件。**

            Megatron 原始 bug:
                `if not forward_only and config.fine_grained_activation_offloading:`
                → eval (forward_only=True) 后不 reset → 状态污染 → 下一 step OOM

            修复后逻辑:
                `if getattr(config, 'fine_grained_activation_offloading', False):`
                → 无论 forward_only，只要 offload 开启就 reset

            DES-LOC 实现:
                reset() 直接调用即可，forward_only 仅传递给 SLC GC
                以决定回收激进程度，不影响 reset 是否执行。

        Args:
            forward_only: 是否为 eval-only schedule（影响 SLC GC 策略）
            force:        跳过 phase 检查，强制 reset（用于异常恢复）

        Returns:
            Dict[device_id → freed_gpu_bytes]
        """
        with self._phase_lock:
            if self._phase == ResetPhase.RESETTING and not force:
                logger.warning("[HeteroReset] reset() called while already RESETTING, skipping")
                return {}
            self._phase = ResetPhase.RESETTING

        eval_mode = forward_only  # eval schedule 对应激进 SLC GC

        logger.info(
            "[HeteroReset] reset START forward_only=%s eval_mode=%s shards=%d",
            forward_only, eval_mode, len(self._shards),
        )
        t0 = time.perf_counter()

        freed: Dict[int, int] = {}

        if self._parallel_reset:
            freed = self._parallel_reset_shards(eval_mode)
        else:
            freed = self._serial_reset_shards(eval_mode)

        # SLC GC：eval→train 切换时需额外运行激进 GC
        gc_freed = 0
        if eval_mode or (self._last_was_eval and not eval_mode):
            # eval 结束或 eval→train 切换：激进清理 SLC
            gc_freed = self._slc.gc(eval_mode=True)
            with self._phase_lock:
                self._phase = ResetPhase.SLC_GC
        else:
            gc_freed = self._slc.gc(eval_mode=False)

        self._last_was_eval = eval_mode

        elapsed = (time.perf_counter() - t0) * 1e3
        total_freed_gpu = sum(freed.values())

        logger.info(
            "[HeteroReset] reset DONE in %.2f ms | freed_gpu=%.2f MB freed_slc=%.2f MB | slc_util=%.1f%%",
            elapsed,
            total_freed_gpu / 1e6,
            gc_freed / 1e6,
            self._slc.utilization * 100,
        )

        with self._phase_lock:
            self._phase = ResetPhase.DONE

        return freed

    def _parallel_reset_shards(self, eval_mode: bool) -> Dict[int, int]:
        """多线程并行 reset 所有 shard（无 NVLink 时各 GPU 独立，可安全并行）."""
        results: Dict[int, int] = {}
        threads: List[threading.Thread] = []
        result_lock = threading.Lock()

        def _reset_one(did: int, shard: DeviceOffloadShard) -> None:
            freed = shard.reset(eval_mode=eval_mode)
            with result_lock:
                results[did] = freed

        for did, shard in self._shards.items():
            t = threading.Thread(
                target=_reset_one,
                args=(did, shard),
                name=f"shard-reset-{did}",
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30.0)
            if t.is_alive():
                logger.error("[HeteroReset] shard reset thread %s timed out!", t.name)

        return results

    def _serial_reset_shards(self, eval_mode: bool) -> Dict[int, int]:
        """串行 reset（调试 / 单线程环境备用路径）."""
        return {
            did: shard.reset(eval_mode=eval_mode)
            for did, shard in self._shards.items()
        }

    # ------------------------------------------------------------------
    # Schedule context manager
    # ------------------------------------------------------------------

    class _ScheduleContext:
        """
        上下文管理器，供 DeepSpeed pipeline schedule 入口包裹。

        __exit__ 中无条件调用 reset()，无论是否发生异常，
        与 Megatron b8e23d5874 修复后的语义完全一致。
        """

        def __init__(
            self,
            manager: "HeteroActivationOffloadReset",
            forward_only: bool,
        ) -> None:
            self._mgr = manager
            self._forward_only = forward_only

        def __enter__(self) -> "_ScheduleContext":
            with self._mgr._phase_lock:
                self._mgr._phase = ResetPhase.PENDING
            logger.debug(
                "[ScheduleCtx] enter forward_only=%s", self._forward_only
            )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            if exc_type is not None:
                logger.warning(
                    "[ScheduleCtx] exception during schedule: %s, forcing reset",
                    exc_type.__name__,
                )
            # 无论 forward_only，无论是否有异常，都执行 reset
            self._mgr.reset(
                forward_only=self._forward_only,
                force=(exc_type is not None),
            )
            return False  # 不吞掉异常

    def schedule_context(self, forward_only: bool = False) -> "_ScheduleContext":
        """
        返回 schedule 上下文管理器。

        用法:
            >>> with hetero_mgr.schedule_context(forward_only=is_eval):
            ...     run_pipeline_schedule(...)
            # reset() 在 with 块退出时自动调用
        """
        return self._ScheduleContext(self, forward_only)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_shard(self, device_id: int) -> DeviceOffloadShard:
        if device_id not in self._shards:
            raise KeyError(f"device_id={device_id} not registered in HeteroReset")
        return self._shards[device_id]

    @property
    def phase(self) -> ResetPhase:
        with self._phase_lock:
            return self._phase

    def summary(self) -> str:
        lines = [
            f"HeteroActivationOffloadReset phase={self._phase.name}",
            f"  SLC: {self._slc}",
        ]
        for did, shard in self._shards.items():
            lines.append(f"  {shard}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HeteroActivationOffloadReset("
            f"shards={list(self._shards.keys())}, "
            f"phase={self._phase.name}, "
            f"slc_util={self._slc.utilization:.1%})"
        )


# ---------------------------------------------------------------------------
# DeepSpeed engine 钩子函数
# ---------------------------------------------------------------------------

def schedule_reset_hook(
    manager: HeteroActivationOffloadReset,
    config: object,
    forward_only: bool,
) -> None:
    """
    供 DeepSpeed engine 在每次 pipeline schedule 结束后调用的钩子。

    直接镜像 Megatron b8e23d5874 修复后的调用点:

        Megatron (修复后):
            if getattr(config, 'fine_grained_activation_offloading', False):
                off_interface.reset()

        DES-LOC (本函数):
            if getattr(config, 'fine_grained_activation_offloading', False):
                manager.reset(forward_only=forward_only)

    区别在于 DES-LOC 额外传入 forward_only，以便 SLC GC 选择合适策略，
    但 reset() 本身不以 forward_only 作为执行门控（这正是 Megatron bug 所在）。

    Args:
        manager:      HeteroActivationOffloadReset 实例
        config:       DeepSpeed / Megatron config 对象（任意 duck-typed）
        forward_only: 当前 schedule 是否为 eval-only（影响 SLC GC 策略）
    """
    # 防御性 getattr，与 Megatron b8e23d5874 修复保持语义一致
    fine_grained = getattr(config, "fine_grained_activation_offloading", False)

    if not fine_grained:
        logger.debug("[hook] fine_grained_activation_offloading=False, skip reset")
        return

    logger.debug(
        "[hook] triggering reset: fine_grained=True forward_only=%s", forward_only
    )
    manager.reset(forward_only=forward_only)


def build_hetero_reset_manager(
    config: object,
    slc_capacity_bytes: Optional[int] = None,
) -> Optional[HeteroActivationOffloadReset]:
    """
    工厂函数：根据 config 决定是否构建 HeteroActivationOffloadReset。

    若 fine_grained_activation_offloading=False，返回 None（与 Megatron 行为对齐，
    不做无谓初始化）。

    Args:
        config:              config 对象
        slc_capacity_bytes:  手动指定 SLC 容量，None 时从 config 读取

    Returns:
        HeteroActivationOffloadReset 实例，或 None
    """
    fine_grained = getattr(config, "fine_grained_activation_offloading", False)
    if not fine_grained:
        logger.info(
            "[build] fine_grained_activation_offloading=False → "
            "HeteroActivationOffloadReset not built"
        )
        return None

    mgr = HeteroActivationOffloadReset.from_config(
        config, slc_capacity_bytes=slc_capacity_bytes
    )
    logger.info("[build] HeteroActivationOffloadReset built: %s", mgr)
    return mgr


# ---------------------------------------------------------------------------
# DeepSpeed engine registration
# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroActivationOffloadReset on a DeepSpeed engine.

    Builds a :class:`HeteroActivationOffloadReset` from the engine's
    configuration and attaches it as ``engine.hetero_activation_offload_reset``.
    A schedule-exit hook is registered so that :func:`schedule_reset_hook` is
    called automatically after every pipeline schedule completes.

    The ``getattr`` defensive access pattern mirrors the Megatron b8e23d5874
    fix: if ``fine_grained_activation_offloading`` is absent from the config
    object, the manager is still built but logs a warning.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance (e.g. ``DeepSpeedEngine`` or
        ``PipelineEngine``).  The engine's ``config`` (or ``ds_config``)
        attribute is used to read DES-LOC heterogeneous device settings.
    """
    logger.info(
        "hetero_activation_offload_reset.register() called on engine type=%s",
        type(engine).__name__,
    )

    # Resolve config object from engine
    config = getattr(engine, "config", None) or getattr(engine, "ds_config", None)
    if config is None:
        logger.warning(
            "[register] engine has no config/ds_config attribute; "
            "using default HeteroActivationOffloadReset configuration"
        )

        class _DefaultCfg:
            fine_grained_activation_offloading = True
        config = _DefaultCfg()

    # Build the manager via the factory (returns None if offloading is disabled)
    mgr = build_hetero_reset_manager(config)

    if mgr is None:
        # Offloading disabled — attach a sentinel so callers can check
        engine.hetero_activation_offload_reset = None
        logger.info(
            "[register] fine_grained_activation_offloading=False; "
            "no HeteroActivationOffloadReset attached to engine."
        )
        return

    engine.hetero_activation_offload_reset = mgr

    # Register a post-schedule hook if the engine supports it
    if hasattr(engine, "register_hook_on_step_end"):
        def _post_schedule_hook():
            forward_only = getattr(engine, "forward_only", False)
            schedule_reset_hook(mgr, config, forward_only=forward_only)
        engine.register_hook_on_step_end(_post_schedule_hook)
        logger.info(
            "HeteroActivationOffloadReset registered via register_hook_on_step_end."
        )
    else:
        logger.info(
            "Engine does not expose register_hook_on_step_end; "
            "manager stored at engine.hetero_activation_offload_reset only. "
            "Call schedule_reset_hook() manually after each schedule."
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ---- 构造最小 mock config ----
    class _Cfg:
        fine_grained_activation_offloading = True
        des_loc_worker_device_ids = [0]   # 只使用 device 0 做 smoke test
        des_loc_anchor_device_id = 0      # 同一设备简化测试
        des_loc_parallel_reset = False    # 单线程，避免 CI 并发问题
        des_loc_slc_capacity_bytes = 512 * 1024 * 1024  # 512 MB

    cfg = _Cfg()
    mgr = HeteroActivationOffloadReset.from_config(cfg)

    # assert 1: 初始 phase 应为 IDLE
    assert mgr.phase == ResetPhase.IDLE, f"expected IDLE, got {mgr.phase}"

    # assert 2: SLC staging / release 正确维护 utilization
    slc = mgr._slc
    dummy = torch.zeros(1024 * 1024, dtype=torch.float32)  # 4 MB
    ok = slc.stage(epoch_id=42, tensor=dummy)
    assert ok, "SLC stage should succeed for 4MB tensor in 512MB buffer"
    assert slc.utilization > 0.0, "utilization should be non-zero after staging"

    # assert 3: eval 后 reset 触发 (forward_only=True 不应阻止 reset)
    freed = mgr.reset(forward_only=True)  # 模拟 eval schedule 结束
    assert mgr.phase == ResetPhase.DONE, f"expected DONE, got {mgr.phase}"

    # assert 4: schedule_context 正常退出后 phase=DONE
    mgr._phase = ResetPhase.IDLE
    with mgr.schedule_context(forward_only=False):
        pass  # 模拟空 training schedule
    assert mgr.phase == ResetPhase.DONE

    # assert 5: getattr 防御 — config 缺字段时 hook 不崩溃
    class _MinimalCfg:
        pass  # 故意不声明 fine_grained_activation_offloading

    schedule_reset_hook(mgr, _MinimalCfg(), forward_only=False)  # 应静默跳过

    print("All smoke tests passed.")
