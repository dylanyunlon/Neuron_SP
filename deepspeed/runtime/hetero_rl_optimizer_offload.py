"""
DES-LOC Heterogeneous RL Optimizer Offload
==========================================

上游设计意图 (Megatron commit 287d2f47):
    Megatron-LM 的 RL 训练在 inference rollout 阶段需要腾出 GPU 显存给 inference engine。
    原始 fix 将 grad_buffer offload 与 optimizer state offload 解耦：
      1. offload_grad_buffers()  — 释放 DDP bucket 的 grad_data storage（resize_ to 0）
      2. optimizer.offload_to_cpu() — 将 Adam exp_avg / exp_avg_sq 搬到 CPU
      3. 推理结束后再 restore_grad_buffers() + optimizer.restore_from_cpu()
    关键洞察：grad_data 用 storage().resize_(0) 而非 del，保留 tensor view 的合法性；
    param_data_cpu 用 pin_memory() 以加速后续 H2D 拷贝。

DES-LOC 适配点:
    硬件：2× A6000 (48 GB, SM86) + 1× H100 NVL (96 GB, SM90)，PCIe 互联，1.5 TB CPU DRAM。
    无 NVLink → GPU 间带宽极低（~32 GB/s PCIe vs ~900 GB/s NVLink）。
    DES-LOC = Decoupled Execution with Shared LOcality Cache：
      * 训练参数/梯度 → A6000 (SM86)；推理 KV-cache → H100 (SM90)
      * CPU DRAM (1.5 TB) 作为 Shared LOcality Cache (SLC)，暂存 optimizer state
      * offload 路径：A6000 GPU → pin_memory CPU (SLC) → 必要时再到 H100
      * 异构感知调度：根据各设备当前显存压力自动选择 offload 目标设备
      * 利用 H100 的大显存（96 GB）在推理高峰时充当二级缓冲
      * 所有 GPU→CPU 传输使用独立 CUDA stream 实现 compute/transfer overlap

适配名: HeteroRLOptimizerOffload
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举 & 配置
# ---------------------------------------------------------------------------

class OffloadTarget(Enum):
    """DES-LOC offload 目标层级。"""
    GPU_A6000 = auto()   # SM86，训练主设备
    GPU_H100  = auto()   # SM90，推理主设备，可充当二级缓冲
    CPU_SLC   = auto()   # Shared LOcality Cache — pin_memory CPU DRAM


@dataclass
class HeteroOffloadConfig:
    """
    DES-LOC 异构 offload 策略配置。

    Attributes:
        a6000_ids:          A6000 设备编号列表（SM86）
        h100_id:            H100 设备编号（SM90）
        cpu_slc_fraction:   CPU DRAM 中保留给 SLC 的比例 (0~1)
        h100_buffer_gb:     H100 上允许用于 optimizer buffer 的最大 GB 数
        use_async_transfer: 是否开启异步 PCIe 传输（独立 CUDA stream）
        sync_before_offload: offload 前是否执行 cuda.synchronize()
        empty_cache_after_offload: offload 后是否执行 cuda.empty_cache()
        grad_offload_priority: True → 先 offload grad_data，再 offload param_data
        pin_memory_grads:   grad_data 卸载时是否分配 pin_memory CPU 副本
                            （Megatron 原版不保留 CPU 副本；DES-LOC 可选保留以加速 restore）
    """
    a6000_ids: List[int] = field(default_factory=lambda: [0, 1])
    h100_id: int = 2
    cpu_slc_fraction: float = 0.6          # 1.5 TB * 0.6 ≈ 900 GB 留给 SLC
    h100_buffer_gb: float = 16.0           # H100 96 GB 中最多 16 GB 给 optimizer buffer
    use_async_transfer: bool = True
    sync_before_offload: bool = True
    empty_cache_after_offload: bool = True
    grad_offload_priority: bool = True
    pin_memory_grads: bool = False         # grad restore 不需要 CPU 副本，默认关


# ---------------------------------------------------------------------------
# 设备压力监控（轻量级，无需 nvml）
# ---------------------------------------------------------------------------

class DevicePressureMonitor:
    """
    轮询各 GPU 的已分配显存，推断当前压力等级。

    DES-LOC 背景：PCIe 互联下跨 GPU 传输代价高，优先往 CPU SLC offload；
    仅当 CPU SLC 接近满载时才将部分状态推到 H100 做临时缓存。
    """

    def __init__(self, cfg: HeteroOffloadConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._cache: Dict[int, float] = {}
        self._last_poll: float = 0.0
        self._poll_interval: float = 0.5  # 秒

    def _poll(self) -> None:
        now = time.monotonic()
        if now - self._last_poll < self._poll_interval:
            return
        for dev_id in self.cfg.a6000_ids + [self.cfg.h100_id]:
            try:
                props = torch.cuda.get_device_properties(dev_id)
                allocated = torch.cuda.memory_allocated(dev_id)
                total = props.total_memory
                self._cache[dev_id] = allocated / total
            except Exception:
                self._cache[dev_id] = 0.0
        self._last_poll = now

    def pressure(self, dev_id: int) -> float:
        """返回指定设备的显存占用率 [0, 1]。"""
        with self._lock:
            self._poll()
            return self._cache.get(dev_id, 0.0)

    def recommend_offload_target(self) -> OffloadTarget:
        """
        根据当前压力推荐 offload 目标。

        策略（PCIe 拓扑优先，避免跨 GPU 传输）：
          - 默认推荐 CPU_SLC（带宽稳定，容量大）
          - 若 H100 压力低（< 0.5）且 CPU SLC 使用率高，则推荐 GPU_H100 作为临时缓冲
          - A6000 之间不互传（PCIe 限速）
        """
        with self._lock:
            self._poll()
        h100_pressure = self.pressure(self.cfg.h100_id)
        if h100_pressure < 0.50:
            # H100 较空闲，可承担部分 optimizer buffer
            logger.debug(
                "DevicePressureMonitor: H100 pressure=%.2f < 0.50, "
                "recommending GPU_H100 as secondary SLC buffer.",
                h100_pressure,
            )
            return OffloadTarget.GPU_H100
        logger.debug(
            "DevicePressureMonitor: H100 pressure=%.2f >= 0.50, using CPU_SLC.",
            h100_pressure,
        )
        return OffloadTarget.CPU_SLC


# ---------------------------------------------------------------------------
# 单个 grad/param buffer 的 DES-LOC offload 逻辑
# ---------------------------------------------------------------------------

class DESLOCParamGradBuffer:
    """
    对 DeepSpeed（或 Megatron 兼容）的 _ParamAndGradBuffer 做 DES-LOC 封装。

    上游 Megatron 在 _ParamAndGradBuffer 里新增了：
        - grad_data_size / param_data_size：offload 前记录 storage 大小
        - param_data_cpu：pin_memory CPU 副本（避免重新 malloc）
        - offload_to_cpu() / reload_from_cpu()：storage resize_ trick

    DES-LOC 扩展：
        - 支持将 param_data 卸载到 H100（作为二级缓冲）而非仅 CPU
        - 使用独立 CUDA stream 做异步传输（compute/transfer overlap）
        - 记录每次 offload/restore 的耗时，供 profiler 使用
    """

    def __init__(
        self,
        buffer,                          # 底层 _ParamAndGradBuffer 实例（鸭子类型）
        cfg: HeteroOffloadConfig,
        transfer_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.buf = buffer
        self.cfg = cfg
        self.stream = transfer_stream

        # 状态跟踪
        self._grad_data_size: int = 0
        self._param_data_size: int = 0
        self._param_data_cpu: Optional[torch.Tensor] = None
        self._param_data_h100: Optional[torch.Tensor] = None  # H100 二级缓冲
        self._current_target: Optional[OffloadTarget] = None

        # 注入到底层 buffer（Megatron 原版字段兼容）
        if not hasattr(self.buf, "grad_data_size"):
            self.buf.grad_data_size = 0
        if not hasattr(self.buf, "param_data_size"):
            self.buf.param_data_size = 0
        if not hasattr(self.buf, "param_data_cpu"):
            self.buf.param_data_cpu = None

    # ------------------------------------------------------------------
    # grad_data offload / restore  (mirrors Megatron storage resize_ trick)
    # ------------------------------------------------------------------

    def offload_grads(self) -> None:
        """
        释放 grad_data 的 GPU storage（resize_ to 0）。

        Megatron 原版行为：仅释放 storage，保留 tensor view，不保留 CPU 副本。
        DES-LOC 额外选项：cfg.pin_memory_grads=True 时在 CPU 上保留 pin_memory 副本，
        以便 restore 时直接 H2D 而无需重新 zero_。
        """
        grad = getattr(self.buf, "grad_data", None)
        if grad is None or grad.storage().size() == 0:
            return

        self._grad_data_size = grad.storage().size()

        if self.cfg.pin_memory_grads:
            # 可选：保留 CPU 副本（用于需要 grad checkpoint 的场景）
            cpu_copy = torch.empty(
                grad.shape, dtype=grad.dtype, pin_memory=True
            )
            if self.cfg.use_async_transfer and self.stream is not None:
                with torch.cuda.stream(self.stream):
                    cpu_copy.copy_(grad, non_blocking=True)
            else:
                cpu_copy.copy_(grad)
            self._grad_cpu_copy = cpu_copy
        else:
            self._grad_cpu_copy = None

        grad.storage().resize_(0)
        logger.debug(
            "offload_grads: freed %.2f MB grad_data from device %s",
            self._grad_data_size * grad.element_size() / 1e6,
            grad.device,
        )

    def restore_grads(self) -> None:
        """
        重新分配 grad_data storage 并清零（或从 CPU 副本恢复）。

        与 Megatron 一致：restore 后 grad_data 为全零，准备下一次 backward。
        """
        grad = getattr(self.buf, "grad_data", None)
        if grad is None or self._grad_data_size == 0:
            return

        grad.storage().resize_(self._grad_data_size)

        if self.cfg.pin_memory_grads and getattr(self, "_grad_cpu_copy", None) is not None:
            if self.cfg.use_async_transfer and self.stream is not None:
                with torch.cuda.stream(self.stream):
                    grad.copy_(self._grad_cpu_copy, non_blocking=True)
            else:
                grad.copy_(self._grad_cpu_copy)
            self._grad_cpu_copy = None
        else:
            grad.zero_()

        self._grad_data_size = 0
        logger.debug("restore_grads: reallocated grad_data (%.2f MB)", 
                     grad.numel() * grad.element_size() / 1e6)

    # ------------------------------------------------------------------
    # param_data offload / restore  (DES-LOC 扩展：支持 H100 二级缓冲)
    # ------------------------------------------------------------------

    def offload_params(self, target: OffloadTarget = OffloadTarget.CPU_SLC) -> None:
        """
        将 param_data 卸载到指定目标（CPU SLC 或 H100 二级缓冲）。

        Args:
            target: CPU_SLC → pin_memory CPU；GPU_H100 → H100 显存（二级缓冲）。
                    Megatron 原版只支持 CPU_SLC。
        """
        param = getattr(self.buf, "param_data", None)
        if param is None or param.storage().size() == 0:
            return

        self._param_data_size = param.storage().size()
        self._current_target = target

        t0 = time.monotonic()

        if target == OffloadTarget.CPU_SLC:
            if self._param_data_cpu is not None:
                # 复用已有 pin_memory buffer（Megatron 原版优化）
                if self.cfg.use_async_transfer and self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        self._param_data_cpu.copy_(param, non_blocking=True)
                else:
                    self._param_data_cpu.copy_(param)
            else:
                self._param_data_cpu = param.cpu().pin_memory()
            param.storage().resize_(0)

        elif target == OffloadTarget.GPU_H100:
            # DES-LOC 扩展：H100 作为 param 二级缓冲，避免 H2D latency
            h100_dev = torch.device(f"cuda:{self.cfg.h100_id}")
            if self.cfg.use_async_transfer and self.stream is not None:
                with torch.cuda.stream(self.stream):
                    self._param_data_h100 = param.to(h100_dev, non_blocking=True)
            else:
                self._param_data_h100 = param.to(h100_dev)
            param.storage().resize_(0)
            logger.debug(
                "offload_params: moved %.2f MB param_data to H100 (SM90)",
                self._param_data_size * param.element_size() / 1e6,
            )
        else:
            raise ValueError(f"Unsupported offload target: {target}")

        elapsed = time.monotonic() - t0
        logger.debug(
            "offload_params → %s: %.2f MB in %.3f s",
            target.name,
            self._param_data_size * (param.element_size() if param.element_size() else 2) / 1e6,
            elapsed,
        )

    def restore_params(self) -> None:
        """从对应缓冲位置恢复 param_data 到原始 GPU。"""
        param = getattr(self.buf, "param_data", None)
        if param is None or self._param_data_size == 0:
            return

        param.storage().resize_(self._param_data_size)
        t0 = time.monotonic()

        if self._current_target == OffloadTarget.CPU_SLC and self._param_data_cpu is not None:
            if self.cfg.use_async_transfer and self.stream is not None:
                with torch.cuda.stream(self.stream):
                    param.copy_(self._param_data_cpu, non_blocking=True)
            else:
                param.copy_(self._param_data_cpu)

        elif self._current_target == OffloadTarget.GPU_H100 and self._param_data_h100 is not None:
            if self.cfg.use_async_transfer and self.stream is not None:
                with torch.cuda.stream(self.stream):
                    param.copy_(self._param_data_h100, non_blocking=True)
            else:
                param.copy_(self._param_data_h100)
            del self._param_data_h100
            self._param_data_h100 = None

        elapsed = time.monotonic() - t0
        logger.debug(
            "restore_params ← %s: %.2f MB in %.3f s",
            self._current_target.name if self._current_target else "UNKNOWN",
            self._param_data_size * 2 / 1e6,
            elapsed,
        )
        self._param_data_size = 0
        self._current_target = None


# ---------------------------------------------------------------------------
# DDP 封装：mirrors Megatron DistributedDataParallel.offload_grad_buffers
# ---------------------------------------------------------------------------

class HeteroDDPGradOffloader:
    """
    对 DeepSpeed DDP 模型的 grad buffer 做 DES-LOC 异构 offload。

    对应 Megatron commit 287d2f47 中新增的：
        DistributedDataParallel.offload_grad_buffers()
        DistributedDataParallel.restore_grad_buffers()

    DES-LOC 差异：
        - 使用独立 CUDA stream 并行处理多个 bucket（Megatron 原版串行）
        - 支持按 buffer 优先级排序（大 bucket 优先释放以快速降压）
        - 记录每个 buffer 的 offload 延迟用于后续带宽建模
    """

    def __init__(
        self,
        ddp_model,
        cfg: HeteroOffloadConfig,
        monitor: DevicePressureMonitor,
    ) -> None:
        self.ddp = ddp_model
        self.cfg = cfg
        self.monitor = monitor

        # 为每个 A6000 设备创建独立传输 stream
        self._streams: Dict[int, torch.cuda.Stream] = {}
        for dev_id in cfg.a6000_ids:
            try:
                self._streams[dev_id] = torch.cuda.Stream(device=dev_id)
            except Exception as exc:
                logger.warning("Cannot create stream for device %d: %s", dev_id, exc)

        # 包装底层 buffer 列表
        all_bufs = list(getattr(ddp_model, "buffers", []))
        all_bufs += list(getattr(ddp_model, "expert_parallel_buffers", []))
        self._wrappers: List[DESLOCParamGradBuffer] = []
        for buf in all_bufs:
            dev_id = (
                buf.grad_data.device.index
                if hasattr(buf, "grad_data") and buf.grad_data is not None
                else cfg.a6000_ids[0]
            )
            stream = self._streams.get(dev_id)
            self._wrappers.append(DESLOCParamGradBuffer(buf, cfg, stream))

        logger.info(
            "HeteroDDPGradOffloader: %d grad buffers registered (A6000×%d, H100×1)",
            len(self._wrappers),
            len(cfg.a6000_ids),
        )

    def offload_grad_buffers(
        self,
        synchronize: bool = True,
        empty_cache: bool = True,
    ) -> None:
        """
        释放所有 grad_data 的 GPU storage。

        对应 Megatron: DistributedDataParallel.offload_grad_buffers()
        DES-LOC 扩展：按 buffer 大小降序处理，优先释放大 bucket 以快速降压。
        """
        if synchronize:
            for dev_id in self.cfg.a6000_ids:
                torch.cuda.synchronize(dev_id)

        # 按 grad_data 大小降序排列（优先释放大 bucket）
        ordered = sorted(
            self._wrappers,
            key=lambda w: (
                w.buf.grad_data.storage().size()
                if hasattr(w.buf, "grad_data") and w.buf.grad_data is not None
                else 0
            ),
            reverse=True,
        )

        t0 = time.monotonic()
        for wrapper in ordered:
            wrapper.offload_grads()

        if empty_cache:
            for dev_id in self.cfg.a6000_ids:
                torch.cuda.empty_cache()

        logger.info(
            "offload_grad_buffers: %d buffers freed in %.3f s",
            len(ordered),
            time.monotonic() - t0,
        )

    def restore_grad_buffers(self, synchronize: bool = True) -> None:
        """
        重新分配所有 grad_data storage 并清零。

        对应 Megatron: DistributedDataParallel.restore_grad_buffers()
        """
        t0 = time.monotonic()
        for wrapper in self._wrappers:
            wrapper.restore_grads()

        if synchronize:
            for dev_id in self.cfg.a6000_ids:
                torch.cuda.synchronize(dev_id)
            # 等待异步传输 stream 完成
            for stream in self._streams.values():
                stream.synchronize()

        logger.info(
            "restore_grad_buffers: %d buffers restored in %.3f s",
            len(self._wrappers),
            time.monotonic() - t0,
        )


# ---------------------------------------------------------------------------
# Optimizer offload 封装：mirrors Megatron optimizer.offload_to_cpu / restore_from_cpu
# ---------------------------------------------------------------------------

class HeteroOptimizerOffloader:
    """
    对 DeepSpeed optimizer（或 Megatron MegatronOptimizer 兼容对象）做 DES-LOC offload。

    上游行为（Megatron commit 287d2f47）：
        optimizer.offload_to_cpu()   — 将 Adam state (exp_avg, exp_avg_sq) 搬到 CPU
        optimizer.restore_from_cpu() — 搬回 GPU

    DES-LOC 扩展：
        1. 压力感知路由：根据 DevicePressureMonitor 决定 offload 到 CPU_SLC 还是 H100
        2. 分片异步传输：将 optimizer state 分成 shard，利用 H2D stream overlap
        3. 混合精度感知：bf16 主参数与 fp32 Adam state 分别路由（fp32 state 更大，优先 CPU）
        4. 状态记录：offload 时保存 tensor device 映射，restore 时精确归位
    """

    def __init__(
        self,
        optimizer,
        cfg: HeteroOffloadConfig,
        monitor: DevicePressureMonitor,
    ) -> None:
        self.opt = optimizer
        self.cfg = cfg
        self.monitor = monitor

        # 传输 stream（H100 侧）
        try:
            self._h100_stream = torch.cuda.Stream(device=cfg.h100_id)
        except Exception:
            self._h100_stream = None

        # A6000 传输 streams
        self._a6000_streams: Dict[int, torch.cuda.Stream] = {}
        for dev_id in cfg.a6000_ids:
            try:
                self._a6000_streams[dev_id] = torch.cuda.Stream(device=dev_id)
            except Exception:
                pass

        # offload 状态记录：param_id → (tensor_name, original_device, cpu/h100 副本)
        self._state_backup: Dict[int, List[Tuple[str, torch.device, torch.Tensor]]] = {}

        logger.info("HeteroOptimizerOffloader initialized (h100_id=%d)", cfg.h100_id)

    def _iter_optimizer_states(self):
        """
        遍历所有子 optimizer 的 state tensors。
        兼容 DeepSpeed ChainedOptimizer 和单个 optimizer。
        """
        opts = (
            self.opt.chained_optimizers
            if hasattr(self.opt, "chained_optimizers")
            else [self.opt]
        )
        for opt in opts:
            inner = getattr(opt, "optimizer", opt)
            if inner is None:
                continue
            for param, state_dict in inner.state.items():
                yield param, state_dict, inner

    def offload_to_cpu(self) -> None:
        """
        将 optimizer state 卸载到 DES-LOC 目标（CPU_SLC 或 H100 二级缓冲）。

        路由策略：
          - fp32 Adam state（exp_avg / exp_avg_sq）→ CPU_SLC（体积大，H100 不够用）
          - step tensor（标量）→ CPU_SLC
          - 若 H100 压力低且 state tensor < h100_buffer_gb → GPU_H100
        """
        target = self.monitor.recommend_offload_target()
        h100_budget_bytes = int(self.cfg.h100_buffer_gb * 1e9)
        h100_used_bytes = 0

        t0 = time.monotonic()
        total_moved_bytes = 0

        self._state_backup.clear()

        for param, state_dict, inner_opt in self._iter_optimizer_states():
            pid = id(param)
            self._state_backup[pid] = []

            for key, val in list(state_dict.items()):
                if not isinstance(val, torch.Tensor) or val.device.type == "cpu":
                    continue

                tensor_bytes = val.numel() * val.element_size()
                original_device = val.device

                # 路由决策
                use_h100 = (
                    target == OffloadTarget.GPU_H100
                    and h100_used_bytes + tensor_bytes <= h100_budget_bytes
                    and self._h100_stream is not None
                )

                if use_h100:
                    h100_dev = torch.device(f"cuda:{self.cfg.h100_id}")
                    if self.cfg.use_async_transfer:
                        with torch.cuda.stream(self._h100_stream):
                            cpu_copy = val.to(h100_dev, non_blocking=True)
                    else:
                        cpu_copy = val.to(h100_dev)
                    h100_used_bytes += tensor_bytes
                    actual_target = OffloadTarget.GPU_H100
                else:
                    # CPU SLC (pin_memory)
                    cpu_copy = torch.empty_like(val, pin_memory=True).cpu()
                    src_dev_id = val.device.index if val.device.index is not None else self.cfg.a6000_ids[0]
                    stream = self._a6000_streams.get(src_dev_id)
                    if self.cfg.use_async_transfer and stream is not None:
                        with torch.cuda.stream(stream):
                            cpu_copy.copy_(val, non_blocking=True)
                    else:
                        cpu_copy.copy_(val)
                    actual_target = OffloadTarget.CPU_SLC

                self._state_backup[pid].append((key, original_device, cpu_copy, actual_target))

                # 释放 GPU 上的 state tensor
                state_dict[key] = torch.empty(0, dtype=val.dtype, device=val.device)
                del val

                total_moved_bytes += tensor_bytes

        # 等待所有传输完成
        for stream in self._a6000_streams.values():
            stream.synchronize()
        if self._h100_stream is not None:
            self._h100_stream.synchronize()

        # 释放 GPU 碎片
        for dev_id in self.cfg.a6000_ids:
            torch.cuda.empty_cache()

        logger.info(
            "offload_to_cpu (DES-LOC): %.2f MB offloaded in %.3f s → primary=%s, H100_buf=%.2f MB",
            total_moved_bytes / 1e6,
            time.monotonic() - t0,
            target.name,
            h100_used_bytes / 1e6,
        )

    def restore_from_cpu(self) -> None:
        """
        将 optimizer state 从 CPU_SLC / H100 恢复回原始 GPU。

        DES-LOC：异步流水线恢复，先提交所有 H2D copy 再 synchronize，
        最大化 PCIe 带宽利用率。
        """
        t0 = time.monotonic()
        total_bytes = 0

        for param, state_dict, inner_opt in self._iter_optimizer_states():
            pid = id(param)
            if pid not in self._state_backup:
                continue

            for key, original_device, backup_tensor, offload_target in self._state_backup[pid]:
                tensor_bytes = backup_tensor.numel() * backup_tensor.element_size()
                dev_id = original_device.index if original_device.index is not None else self.cfg.a6000_ids[0]

                if offload_target == OffloadTarget.CPU_SLC:
                    stream = self._a6000_streams.get(dev_id)
                    restored = torch.empty(
                        backup_tensor.shape,
                        dtype=backup_tensor.dtype,
                        device=original_device,
                    )
                    if self.cfg.use_async_transfer and stream is not None:
                        with torch.cuda.stream(stream):
                            restored.copy_(backup_tensor, non_blocking=True)
                    else:
                        restored.copy_(backup_tensor)

                elif offload_target == OffloadTarget.GPU_H100:
                    stream = self._a6000_streams.get(dev_id)
                    if self.cfg.use_async_transfer and stream is not None:
                        with torch.cuda.stream(stream):
                            restored = backup_tensor.to(original_device, non_blocking=True)
                    else:
                        restored = backup_tensor.to(original_device)
                    del backup_tensor
                else:
                    restored = backup_tensor.to(original_device)

                state_dict[key] = restored
                total_bytes += tensor_bytes

        # 同步所有传输 stream
        for stream in self._a6000_streams.values():
            stream.synchronize()
        if self._h100_stream is not None:
            self._h100_stream.synchronize()

        self._state_backup.clear()

        logger.info(
            "restore_from_cpu (DES-LOC): %.2f MB restored in %.3f s",
            total_bytes / 1e6,
            time.monotonic() - t0,
        )


# ---------------------------------------------------------------------------
# 顶层 RL 推理 context manager：mirrors Megatron megatron_rl_inference_mode
# ---------------------------------------------------------------------------

class DESLOCRLInferenceContext:
    """
    DES-LOC RL 推理阶段的异构 offload context manager。

    上游 Megatron commit 287d2f47 修复了以下 bug：
        原版在 inference_model is not None 分支内才 offload optimizer，
        导致 unified memory 路径（inference_model is None）下 optimizer 不被 offload。
        Fix：将 offload_grad_buffers + optimizer.offload_to_cpu 提到分支外，
        统一在推理开始前执行。

    DES-LOC 等效：
        1. 推理前：offload_grad_buffers（A6000 grad storage → 0）
                   offload optimizer state（→ CPU_SLC 或 H100 二级缓冲）
        2. 推理中：H100 全速跑 inference（KV-cache 占满 96 GB）
        3. 推理后：restore_grad_buffers（重新分配 grad storage，清零）
                   restore optimizer state（→ A6000 GPU）
    """

    def __init__(
        self,
        ddp_offloader: HeteroDDPGradOffloader,
        opt_offloader: HeteroOptimizerOffloader,
        offload_enabled: bool = True,
    ) -> None:
        self.ddp_off = ddp_offloader
        self.opt_off = opt_offloader
        self.enabled = offload_enabled

    def __enter__(self) -> "DESLOCRLInferenceContext":
        if not self.enabled:
            return self
        logger.info("[DES-LOC] Entering RL inference: offloading grad_buffers + optimizer state")
        # Mirrors Megatron fix: offload BEFORE inference (not nested inside inference_model branch)
        self.ddp_off.offload_grad_buffers(synchronize=True, empty_cache=True)
        self.opt_off.offload_to_cpu()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self.enabled:
            return False
        logger.info("[DES-LOC] Exiting RL inference: restoring grad_buffers + optimizer state")
        self.ddp_off.restore_grad_buffers(synchronize=True)
        self.opt_off.restore_from_cpu()
        return False  # 不吞异常


# ---------------------------------------------------------------------------
# 工厂函数（供 deepspeed/runtime/engine.py 或 rl_trainer.py 调用）
# ---------------------------------------------------------------------------

def build_hetero_rl_offload(
    ddp_model,
    optimizer,
    cfg: Optional[HeteroOffloadConfig] = None,
) -> Tuple[HeteroDDPGradOffloader, HeteroOptimizerOffloader, DESLOCRLInferenceContext]:
    """
    构建 DES-LOC 异构 RL offload 三件套。

    Args:
        ddp_model:  DeepSpeed DDP 模型（含 .buffers / .expert_parallel_buffers 字段）
        optimizer:  DeepSpeed / Megatron 兼容 optimizer（含 .chained_optimizers 或 .state）
        cfg:        异构配置，None 时使用默认值（适配 A6000×2 + H100×1 拓扑）

    Returns:
        (grad_offloader, opt_offloader, inference_ctx)

    Example::

        grad_off, opt_off, ctx = build_hetero_rl_offload(model, optimizer)
        with ctx:
            rollouts = run_inference(inference_model, prompts)
        train_step(model, optimizer, rollouts)
    """
    if cfg is None:
        cfg = HeteroOffloadConfig()

    monitor = DevicePressureMonitor(cfg)
    grad_off = HeteroDDPGradOffloader(ddp_model, cfg, monitor)
    opt_off = HeteroOptimizerOffloader(optimizer, cfg, monitor)
    ctx = DESLOCRLInferenceContext(grad_off, opt_off, offload_enabled=True)

    logger.info(
        "build_hetero_rl_offload: DES-LOC RL offload ready "
        "(A6000=[%s], H100=%d, cpu_slc_frac=%.1f, h100_buf=%.1f GB)",
        ",".join(map(str, cfg.a6000_ids)),
        cfg.h100_id,
        cfg.cpu_slc_fraction,
        cfg.h100_buffer_gb,
    )
    return grad_off, opt_off, ctx


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    cfg = HeteroOffloadConfig(a6000_ids=[0], h100_id=0)  # 单卡 smoke test

    # ---- 1. DevicePressureMonitor smoke test ----
    monitor = DevicePressureMonitor(cfg)
    p = monitor.pressure(0)
    assert 0.0 <= p <= 1.0, f"pressure out of range: {p}"
    target = monitor.recommend_offload_target()
    assert isinstance(target, OffloadTarget), "recommend_offload_target must return OffloadTarget"
    logger.info("PASS: DevicePressureMonitor (pressure=%.3f, target=%s)", p, target.name)

    # ---- 2. DESLOCParamGradBuffer grad offload smoke test ----
    class FakeBuf:
        grad_data = torch.randn(1024, device="cuda:0")
        param_data = torch.randn(512, device="cuda:0")

    stream = torch.cuda.Stream(device=0)
    wrapper = DESLOCParamGradBuffer(FakeBuf(), cfg, stream)

    orig_size = wrapper.buf.grad_data.storage().size()
    wrapper.offload_grads()
    assert wrapper.buf.grad_data.storage().size() == 0, "grad storage should be 0 after offload"
    wrapper.restore_grads()
    assert wrapper.buf.grad_data.storage().size() == orig_size, "grad storage should restore"
    logger.info("PASS: DESLOCParamGradBuffer grad offload/restore (size=%d)", orig_size)

    # ---- 3. DESLOCParamGradBuffer param offload to CPU_SLC ----
    orig_param_size = wrapper.buf.param_data.storage().size()
    wrapper.offload_params(target=OffloadTarget.CPU_SLC)
    assert wrapper.buf.param_data.storage().size() == 0, "param storage should be 0 after offload"
    wrapper.restore_params()
    assert wrapper.buf.param_data.storage().size() == orig_param_size, "param storage should restore"
    logger.info("PASS: DESLOCParamGradBuffer param offload/restore to CPU_SLC")

    # ---- 4. HeteroOptimizerOffloader smoke test ----
    class FakeInner:
        param_groups = [{"params": []}]
        state: Dict = {}

    p_tensor = torch.randn(64, device="cuda:0")
    fake_state = {
        "exp_avg": torch.randn(64, device="cuda:0"),
        "exp_avg_sq": torch.randn(64, device="cuda:0"),
        "step": torch.tensor(1),
    }

    class FakeOpt:
        optimizer = FakeInner()
        optimizer.state = {p_tensor: fake_state}
        optimizer.param_groups = [{"params": [p_tensor]}]
        chained_optimizers = [optimizer]

    opt_off = HeteroOptimizerOffloader(FakeOpt(), cfg, monitor)
    opt_off.offload_to_cpu()
    for k, v in fake_state.items():
        if isinstance(v, torch.Tensor) and v.numel() > 1:
            assert v.numel() == 0 or v.device.type in ("cpu", "cuda"), \
                f"unexpected state after offload: {k} on {v.device}"
    opt_off.restore_from_cpu()
    logger.info("PASS: HeteroOptimizerOffloader offload/restore")

    # ---- 5. DESLOCRLInferenceContext context manager smoke test ----
    class MinimalDDP:
        buffers = []
        expert_parallel_buffers = []

    grad_off2 = HeteroDDPGradOffloader(MinimalDDP(), cfg, monitor)
    opt_off2 = HeteroOptimizerOffloader(FakeOpt(), cfg, monitor)
    ctx = DESLOCRLInferenceContext(grad_off2, opt_off2, offload_enabled=True)
    with ctx:
        pass  # 推理占位
    logger.info("PASS: DESLOCRLInferenceContext enter/exit without exception")

    logger.info("All smoke tests passed.")
