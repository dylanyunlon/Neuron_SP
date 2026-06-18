"""
Heterogeneous Aggressive Checkpoint Manager for DES-LOC
========================================================

上游设计意图 (Megatron commit 3bb539e):
    Megatron-LM 将 GPT-15B 和 DeepSeekV3-proxy 的 save-interval 从 5000 步压缩到
    1000/250 步，retain-interval 统一收敛到 5000 步。这是一种"激进检查点"策略：
    以更高的 I/O 频率换取更细粒度的容错恢复粒度。对于大规模长时训练，单次故障
    恢复的代价远超频繁保存的额外开销，因此激进策略是合理的工程取舍。

DES-LOC 适配点 (Decoupled Execution with Shared LOcality Cache):
    1. **异构设备感知间隔调度**：
       - A6000 x2 (SM86, 48GB PCIe) 是显存受限设备，激进保存时需要先将激活状态
         offload 到 CPU DRAM，再序列化到磁盘，否则会在 checkpoint 窗口内 OOM。
       - H100 NVL (SM90, 96GB PCIe) 显存充裕，可以在 GPU 上直接序列化，不需要
         中间 CPU buffer。
       - 本模块根据设备 SM 版本自动选择 save_interval / retain_interval 策略，
         复刻 Megatron 的"激进"语义，但分层适配到异构拓扑。

    2. **SLC (Shared LOcality Cache) 层快照**：
       - DES-LOC 的 SLC 维护跨设备的 KV-style 共享缓存，checkpoint 时必须同步
         SLC 快照，否则恢复后 SLC miss-rate 会骤增导致性能退化。
       - 本模块在每次 trigger_save() 时额外序列化 SLC meta-index。

    3. **去耦执行上下文 (Decoupled Execution) 的 epoch fence**：
       - DES-LOC 允许 A6000 和 H100 以不同微批次节奏执行，checkpoint 需要在
         "逻辑全局步"对齐，而非各设备本地步数。本模块维护 global_step 计数器
         并通过 dist barrier 强制对齐。

    4. **PCIe 限速下的异步写入**：
       - 无 NVLink 环境下设备间带宽约 16-32 GB/s，连续同步写会阻塞训练 loop。
         本模块使用后台线程池异步落盘，训练主线程只等待"写入确认信号"。

硬件假设:
    - 2x NVIDIA A6000 48GB (SM86), device_id 0,1
    - 1x NVIDIA H100 NVL 96GB (SM90), device_id 2
    - PCIe 互联，无 NVLink，1.5 TB CPU DRAM

作者: Neuron_SP / DES-LOC Team
对应上游: Megatron-LM commit 3bb539e723f077067522960410e277ad08a5be19
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware topology constants for the DES-LOC reference cluster
# ---------------------------------------------------------------------------

_SM86_DEVICES = frozenset({0, 1})   # A6000 x2
_SM90_DEVICES = frozenset({2})       # H100 NVL

# Megatron upstream aggressive schedule (mapped per device class)
# GPT-scale (≥10B): save_interval=1000, retain_interval=5000
# MoE/DeepSeek-scale: save_interval=250,  retain_interval=5000
_SCHEDULE_TABLE: Dict[str, Dict[str, int]] = {
    "gpt_scale": {
        "sm86_save_interval":    1000,
        "sm90_save_interval":    1000,
        "retain_interval":       5000,
    },
    "moe_scale": {
        "sm86_save_interval":    500,   # 保守一倍：A6000 PCIe offload 代价更高
        "sm90_save_interval":    250,   # H100 可以跟上游完全对齐
        "retain_interval":       5000,
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    """
    对单张 GPU 的 DES-LOC 运行时画像。

    Attributes
    ----------
    device_id:      CUDA device index
    sm_version:     compute capability 主版本 * 10 + 次版本，e.g. SM86→86
    vram_gb:        显存容量（GB）
    is_pcie_only:   是否为纯 PCIe 互联（无 NVLink）
    save_interval:  本设备触发 checkpoint 的本地步间隔
    """
    device_id:    int
    sm_version:   int
    vram_gb:      float
    is_pcie_only: bool = True
    save_interval: int = 1000


@dataclass
class SLCMeta:
    """
    Shared LOcality Cache 快照元信息。

    DES-LOC 的 SLC 是一个跨设备的分布式 KV 缓存，存储 attention KV 对的
    热点副本以减少跨设备重算。checkpoint 时需要保存 SLC 的 index 结构，
    以便恢复后能快速重建热点映射，避免冷启动性能坑。

    Attributes
    ----------
    global_step:    触发快照时的全局逻辑步
    index:          {layer_idx: {token_range: device_id}} 映射
    version:        SLC 版本号，用于校验一致性
    timestamp:      UNIX 时间戳
    """
    global_step: int
    index:       Dict[int, Dict[Tuple[int, int], int]] = field(default_factory=dict)
    version:     int = 0
    timestamp:   float = field(default_factory=time.time)


@dataclass
class CheckpointEvent:
    """
    异步写入队列中的单个事件。

    Attributes
    ----------
    global_step:    逻辑全局步
    save_path:      目标目录
    state_dict:     模型 + 优化器状态字典（已在 CPU 上）
    slc_meta:       SLC 快照
    is_retain:      是否为"retain"级别（长期保留）
    """
    global_step: int
    save_path:   Path
    state_dict:  Dict[str, Any]
    slc_meta:    SLCMeta
    is_retain:   bool = False


# ---------------------------------------------------------------------------
# Core: HeteroAggressiveCheckpointManager
# ---------------------------------------------------------------------------

class HeteroAggressiveCheckpointManager:
    """
    DES-LOC 异构激进检查点管理器。

    该类是 Megatron-LM commit 3bb539e "More aggressive checkpointing" 在
    DES-LOC 异构训练框架中的完整诠释。上游 commit 仅修改 YAML 配置数字，
    本实现将这一策略意图转化为运行时自适应逻辑：

      - 根据设备 SM 版本自动选取 save_interval
      - 在 PCIe 拓扑下将 GPU tensor 异步 offload 到 CPU DRAM 再落盘
      - 保存 SLC 快照以避免恢复后的缓存冷启动
      - 通过 dist.barrier 在"全局逻辑步"边界对齐异构设备

    Parameters
    ----------
    save_dir:           checkpoint 根目录
    model_scale:        "gpt_scale" 或 "moe_scale"，决定间隔策略
    device_profiles:    集群中所有 GPU 的 DeviceProfile 列表
    retain_interval:    保留级 checkpoint 的步间隔（默认跟 Megatron 上游 5000）
    async_workers:      后台 I/O 线程数
    cpu_offload:        SM86 设备是否强制 CPU offload（默认 True）
    rank:               当前进程的 dist rank
    world_size:         总进程数
    """

    def __init__(
        self,
        save_dir:         str,
        model_scale:      str = "gpt_scale",
        device_profiles:  Optional[List[DeviceProfile]] = None,
        retain_interval:  int = 5000,
        async_workers:    int = 2,
        cpu_offload:      bool = True,
        rank:             int = 0,
        world_size:       int = 1,
    ) -> None:
        self.save_dir        = Path(save_dir)
        self.model_scale     = model_scale
        self.retain_interval = retain_interval
        self.cpu_offload     = cpu_offload
        self.rank            = rank
        self.world_size      = world_size

        # 默认设备拓扑：2x A6000 + 1x H100
        self.device_profiles: List[DeviceProfile] = device_profiles or [
            DeviceProfile(device_id=0, sm_version=86, vram_gb=48.0),
            DeviceProfile(device_id=1, sm_version=86, vram_gb=48.0),
            DeviceProfile(device_id=2, sm_version=90, vram_gb=96.0),
        ]

        self._schedule       = _SCHEDULE_TABLE[model_scale]
        self._global_step    = 0
        self._slc_version    = 0

        # 为每张卡分配 save_interval
        self._per_device_interval: Dict[int, int] = {}
        for dp in self.device_profiles:
            self._per_device_interval[dp.device_id] = self._resolve_interval(dp)
            dp.save_interval = self._per_device_interval[dp.device_id]

        # 全局逻辑步间隔 = 所有设备间隔的最大公约数（保守策略）
        self._global_interval = self._compute_global_interval()

        # 异步 I/O
        self._executor  = ThreadPoolExecutor(max_workers=async_workers,
                                             thread_name_prefix="des_loc_ckpt")
        self._io_queue: queue.Queue[CheckpointEvent] = queue.Queue()
        self._pending_futures: List[Future] = []

        # SLC index（由外部 SLC 模块注入，此处存 meta 副本）
        self._slc_index: Dict[int, Dict[Tuple[int, int], int]] = {}

        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[DES-LOC Ckpt] Initialized HeteroAggressiveCheckpointManager | "
            "scale=%s global_interval=%d retain_interval=%d async_workers=%d",
            model_scale, self._global_interval, retain_interval, async_workers,
        )
        for dp in self.device_profiles:
            logger.info(
                "[DES-LOC Ckpt] Device %d (SM%d, %.0fGB) → save_interval=%d",
                dp.device_id, dp.sm_version, dp.vram_gb, dp.save_interval,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        训练循环每个全局步调用一次。

        在 DES-LOC 的去耦执行模型中，各设备可能以不同频率完成微批次，
        但 global_step 由 pipeline 协调器统一推进，本方法由协调器调用。
        """
        self._global_step += 1
        if self._should_save(self._global_step):
            logger.debug("[DES-LOC Ckpt] step=%d triggers save check", self._global_step)

    def trigger_save(
        self,
        state_dict: Dict[str, Any],
        slc_index:  Optional[Dict[int, Dict[Tuple[int, int], int]]] = None,
        force:      bool = False,
    ) -> Optional[Future]:
        """
        尝试触发 checkpoint。由训练主循环在每步末尾调用。

        Parameters
        ----------
        state_dict: 包含 'model' 和 'optimizer' 的状态字典（仍在 GPU 上）
        slc_index:  当前 SLC 的 index 快照（可选，由 SLC 模块传入）
        force:      忽略间隔检查，强制保存（用于训练结束或异常终止）

        Returns
        -------
        Future or None: 如果触发了异步写入，返回对应 Future；否则返回 None
        """
        if not force and not self._should_save(self._global_step):
            return None

        # dist barrier：确保所有 rank 在同一全局步触发 checkpoint
        if self.world_size > 1 and dist.is_initialized():
            logger.debug(
                "[DES-LOC Ckpt] rank=%d waiting at barrier for step=%d",
                self.rank, self._global_step,
            )
            dist.barrier()

        # 只有 rank 0 实际写磁盘（其余 rank 参与 barrier 即可）
        if self.rank != 0:
            return None

        # 更新 SLC meta
        if slc_index is not None:
            self._slc_index = slc_index
        slc_meta = self._snapshot_slc()

        # CPU offload：将 GPU tensor 移到 CPU DRAM
        cpu_state = self._offload_to_cpu(state_dict)

        is_retain  = self._is_retain_step(self._global_step)
        event_path = self._build_save_path(self._global_step, is_retain)

        event = CheckpointEvent(
            global_step=self._global_step,
            save_path=event_path,
            state_dict=cpu_state,
            slc_meta=slc_meta,
            is_retain=is_retain,
        )

        fut = self._executor.submit(self._async_write, event)
        self._pending_futures.append(fut)

        log_level = logging.INFO if is_retain else logging.DEBUG
        logger.log(
            log_level,
            "[DES-LOC Ckpt] Scheduled %s checkpoint → %s (step=%d)",
            "RETAIN" if is_retain else "transient",
            event_path,
            self._global_step,
        )

        # 清理已完成的 future，避免内存泄漏
        self._gc_futures()

        return fut

    def load_latest(
        self,
        map_location: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        从 save_dir 中加载最新的 checkpoint。

        DES-LOC 恢复策略：
          1. 优先从 retain 目录加载（更稳定）
          2. 如果 retain 目录为空，回退到最新 transient checkpoint
          3. 同时恢复 SLC meta-index 以重建缓存热点映射

        Returns
        -------
        dict with keys: 'model', 'optimizer', 'global_step', 'slc_meta'
        or None if no checkpoint found.
        """
        retain_dir   = self.save_dir / "retain"
        transient_dir = self.save_dir / "transient"

        candidate = self._find_latest_ckpt(retain_dir)
        if candidate is None:
            logger.warning(
                "[DES-LOC Ckpt] No retain checkpoint found, "
                "falling back to transient directory"
            )
            candidate = self._find_latest_ckpt(transient_dir)

        if candidate is None:
            logger.warning("[DES-LOC Ckpt] No checkpoint found in %s", self.save_dir)
            return None

        logger.info("[DES-LOC Ckpt] Loading checkpoint from %s", candidate)
        payload = torch.load(candidate / "checkpoint.pt", map_location=map_location or "cpu")

        slc_meta_path = candidate / "slc_meta.pt"
        if slc_meta_path.exists():
            slc_payload = torch.load(slc_meta_path, map_location="cpu")
            self._slc_index = slc_payload.get("index", {})
            logger.info(
                "[DES-LOC Ckpt] Restored SLC meta-index with %d layer entries (step=%d)",
                len(self._slc_index),
                slc_payload.get("global_step", -1),
            )
        else:
            logger.warning("[DES-LOC Ckpt] No SLC meta found at %s — cold SLC start", candidate)

        self._global_step = payload.get("global_step", 0)
        return payload

    def prune_old_checkpoints(self) -> None:
        """
        清理超出 retain_interval 窗口的旧 checkpoint。

        保留策略（镜像 Megatron 上游语义）：
          - transient 目录：只保留最近 retain_interval / global_interval 个
          - retain 目录：永久保留（由外部策略决定删除）

        DES-LOC 额外保证：在 PCIe 环境下，不在训练关键路径上执行删除操作。
        """
        transient_dir = self.save_dir / "transient"
        if not transient_dir.exists():
            return

        steps = sorted(
            int(p.name) for p in transient_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        )

        keep_count = max(1, self.retain_interval // self._global_interval)
        to_remove  = steps[:-keep_count] if len(steps) > keep_count else []

        for step in to_remove:
            target = transient_dir / str(step)
            logger.debug("[DES-LOC Ckpt] Pruning transient checkpoint: %s", target)
            import shutil
            shutil.rmtree(target, ignore_errors=True)

        if to_remove:
            logger.info(
                "[DES-LOC Ckpt] Pruned %d transient checkpoints, kept %d",
                len(to_remove), min(keep_count, len(steps)),
            )

    def update_slc_index(
        self,
        layer_idx: int,
        token_range: Tuple[int, int],
        device_id: int,
    ) -> None:
        """
        由 SLC 模块调用，更新当前热点缓存的 index 映射。

        Parameters
        ----------
        layer_idx:   Transformer 层索引
        token_range: (start, end) token 位置范围
        device_id:   持有该 KV 副本的设备 ID
        """
        if layer_idx not in self._slc_index:
            self._slc_index[layer_idx] = {}
        self._slc_index[layer_idx][token_range] = device_id

    def wait_all(self, timeout: float = 300.0) -> None:
        """
        等待所有后台 checkpoint 写入完成。

        Parameters
        ----------
        timeout: 最长等待秒数，超时后记录警告但不抛异常
        """
        deadline = time.monotonic() + timeout
        for fut in self._pending_futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("[DES-LOC Ckpt] wait_all timeout exceeded")
                break
            try:
                fut.result(timeout=remaining)
            except Exception as exc:  # noqa: BLE001
                logger.error("[DES-LOC Ckpt] Async checkpoint write failed: %s", exc)
        self._pending_futures.clear()
        logger.info("[DES-LOC Ckpt] All pending checkpoint writes drained")

    def shutdown(self) -> None:
        """
        优雅关闭：等待所有后台 I/O 线程完成后再退出。
        训练结束时必须调用，避免进程退出时未完成的写入被截断。
        """
        logger.info("[DES-LOC Ckpt] Shutting down checkpoint manager")
        self.wait_all()
        self._executor.shutdown(wait=True)
        logger.info("[DES-LOC Ckpt] Shutdown complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_interval(self, dp: DeviceProfile) -> int:
        """
        根据设备 SM 版本和模型规模查表，返回 save_interval。

        DES-LOC 策略：
          - SM90 (H100): 跟 Megatron 上游完全对齐
          - SM86 (A6000): MoE 规模时保守一倍（PCIe offload 代价）；
                          GPT 规模时与上游对齐（offload 开销可接受）
        """
        sched = self._schedule
        if dp.sm_version >= 90:
            return sched["sm90_save_interval"]
        else:
            return sched["sm86_save_interval"]

    def _compute_global_interval(self) -> int:
        """
        计算全局逻辑步间隔。

        使用所有设备 save_interval 中的最小值作为全局触发步，
        确保任何一台设备都不会在其期望间隔内被跳过。
        同时保证 global_interval 能整除 retain_interval。
        """
        intervals = list(self._per_device_interval.values())
        min_interval = min(intervals)
        # 确保 retain_interval 是 global_interval 的整数倍
        while self.retain_interval % min_interval != 0:
            min_interval -= 1
            if min_interval <= 0:
                min_interval = 1
                break
        return max(1, min_interval)

    def _should_save(self, step: int) -> bool:
        """判断当前步是否需要触发 checkpoint。"""
        if step <= 0:
            return False
        return step % self._global_interval == 0

    def _is_retain_step(self, step: int) -> bool:
        """判断当前步是否需要作为 retain 级别保存（长期保留）。"""
        return step % self.retain_interval == 0

    def _build_save_path(self, step: int, is_retain: bool) -> Path:
        """构建 checkpoint 目标路径。"""
        tier = "retain" if is_retain else "transient"
        p    = self.save_dir / tier / str(step)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _snapshot_slc(self) -> SLCMeta:
        """
        对当前 SLC index 做快照，返回 SLCMeta 对象。

        注意：此处做浅拷贝即可，因为 _slc_index 的 leaf value（device_id）
        是 int，不可变；token_range tuple 也不可变。
        """
        self._slc_version += 1
        snapshot = {
            layer: dict(ranges)
            for layer, ranges in self._slc_index.items()
        }
        return SLCMeta(
            global_step=self._global_step,
            index=snapshot,
            version=self._slc_version,
            timestamp=time.time(),
        )

    def _offload_to_cpu(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 state_dict 中所有 GPU tensor 移到 CPU DRAM。

        DES-LOC 的 CPU DRAM 是 1.5 TB，足以容纳多个完整的模型状态快照。
        对 SM86 设备（A6000, PCIe），offload 是避免 OOM 的必要步骤；
        对 SM90 设备（H100 NVL），如果显存充足，可以跳过 offload 直接序列化，
        但本实现统一走 CPU 路径以简化逻辑（H100 快，差异不显著）。

        Parameters
        ----------
        state_dict: 可能包含嵌套 dict 的状态字典

        Returns
        -------
        所有 tensor 已在 CPU 上的新字典（不修改原始 state_dict）
        """
        if not self.cpu_offload:
            return state_dict

        def _to_cpu(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            elif isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                converted = [_to_cpu(v) for v in obj]
                return type(obj)(converted)
            else:
                return obj

        t0     = time.monotonic()
        result = _to_cpu(state_dict)
        elapsed = time.monotonic() - t0

        logger.debug(
            "[DES-LOC Ckpt] CPU offload completed in %.3fs (step=%d)",
            elapsed, self._global_step,
        )
        return result

    def _async_write(self, event: CheckpointEvent) -> None:
        """
        后台线程执行实际的磁盘写入。

        写入两个文件：
          1. checkpoint.pt  — model + optimizer state + global_step
          2. slc_meta.pt    — SLC index 快照 + 版本信息

        DES-LOC 选择分开存储 SLC meta 的原因：
          - SLC meta 恢复后可以独立于模型权重使用（缓存预热）
          - 单独文件方便 SLC 模块直接读取，不需要加载完整 checkpoint

        PCIe 环境下写入带宽有限，torch.save 使用 pickle 协议 4（默认），
        对大 tensor 有足够的流式写入效率。
        """
        t0 = time.monotonic()
        try:
            ckpt_path = event.save_path / "checkpoint.pt"
            payload   = {
                "model":       event.state_dict.get("model", {}),
                "optimizer":   event.state_dict.get("optimizer", {}),
                "global_step": event.global_step,
                "is_retain":   event.is_retain,
            }
            torch.save(payload, ckpt_path, _use_new_zipfile_serialization=True)

            slc_path    = event.save_path / "slc_meta.pt"
            slc_payload = {
                "global_step": event.slc_meta.global_step,
                "index":       event.slc_meta.index,
                "version":     event.slc_meta.version,
                "timestamp":   event.slc_meta.timestamp,
            }
            torch.save(slc_payload, slc_path)

            elapsed = time.monotonic() - t0
            logger.info(
                "[DES-LOC Ckpt] Written %s checkpoint → %s (%.2fs, step=%d)",
                "RETAIN" if event.is_retain else "transient",
                event.save_path,
                elapsed,
                event.global_step,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[DES-LOC Ckpt] FAILED to write checkpoint at step=%d: %s",
                event.global_step, exc,
            )
            raise

    def _find_latest_ckpt(self, directory: Path) -> Optional[Path]:
        """返回 directory 下步数最大的 checkpoint 子目录，不存在返回 None。"""
        if not directory.exists():
            return None
        candidates = [
            p for p in directory.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: int(p.name))

    def _gc_futures(self) -> None:
        """清理已完成的 Future 对象，避免列表无限增长。"""
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        """当前全局逻辑步（只读）。"""
        return self._global_step

    @property
    def global_interval(self) -> int:
        """全局 checkpoint 触发间隔（只读）。"""
        return self._global_interval

    @property
    def slc_index(self) -> Dict[int, Dict[Tuple[int, int], int]]:
        """当前 SLC index 的只读视图。"""
        return dict(self._slc_index)

    def __repr__(self) -> str:
        return (
            f"HeteroAggressiveCheckpointManager("
            f"scale={self.model_scale!r}, "
            f"global_interval={self._global_interval}, "
            f"retain_interval={self.retain_interval}, "
            f"devices={[dp.device_id for dp in self.device_profiles]!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_des_loc_checkpoint_manager(
    save_dir:    str,
    model_scale: str = "gpt_scale",
    rank:        int = 0,
    world_size:  int = 1,
    **kwargs,
) -> HeteroAggressiveCheckpointManager:
    """
    便捷工厂函数：使用 DES-LOC 参考集群默认拓扑构建 checkpoint manager。

    对应 Megatron 上游 3bb539e 中对不同模型规模使用不同 save_interval 的策略：
      - gpt_scale  → save_interval=1000 (对应 gpt3_15b_8t)
      - moe_scale  → save_interval=250  (对应 deepseekv3_proxy)

    Parameters
    ----------
    save_dir:    checkpoint 保存目录
    model_scale: "gpt_scale" 或 "moe_scale"
    rank:        当前 dist rank
    world_size:  总 rank 数
    **kwargs:    透传给 HeteroAggressiveCheckpointManager

    Returns
    -------
    HeteroAggressiveCheckpointManager 实例
    """
    profiles = [
        DeviceProfile(device_id=0, sm_version=86, vram_gb=48.0, is_pcie_only=True),
        DeviceProfile(device_id=1, sm_version=86, vram_gb=48.0, is_pcie_only=True),
        DeviceProfile(device_id=2, sm_version=90, vram_gb=96.0, is_pcie_only=True),
    ]
    return HeteroAggressiveCheckpointManager(
        save_dir=save_dir,
        model_scale=model_scale,
        device_profiles=profiles,
        rank=rank,
        world_size=world_size,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Test 1: gpt_scale global_interval 与 Megatron 上游对齐 ──────────
        mgr = build_des_loc_checkpoint_manager(tmpdir, model_scale="gpt_scale")
        assert mgr.global_interval == 1000, (
            f"gpt_scale global_interval should be 1000, got {mgr.global_interval}"
        )
        logger.info("Test 1 passed: gpt_scale global_interval=1000")

        # ── Test 2: moe_scale H100 与 Megatron 上游 250 步对齐 ──────────────
        mgr_moe = build_des_loc_checkpoint_manager(tmpdir + "_moe", model_scale="moe_scale")
        h100_profile = next(
            dp for dp in mgr_moe.device_profiles if dp.sm_version == 90
        )
        assert h100_profile.save_interval == 250, (
            f"H100 moe_scale interval should be 250, got {h100_profile.save_interval}"
        )
        logger.info("Test 2 passed: H100 moe_scale save_interval=250")

        # ── Test 3: A6000 moe_scale 保守策略 > H100 间隔 ────────────────────
        a6000_profile = next(
            dp for dp in mgr_moe.device_profiles if dp.sm_version == 86
        )
        assert a6000_profile.save_interval > h100_profile.save_interval, (
            "A6000 should have larger interval than H100 for moe_scale (PCIe conservatism)"
        )
        logger.info(
            "Test 3 passed: A6000(%d) > H100(%d) for moe_scale",
            a6000_profile.save_interval, h100_profile.save_interval,
        )

        # ── Test 4: trigger_save 写入磁盘并可 load_latest ────────────────────
        dummy_state = {
            "model":     {"weight": torch.randn(4, 4)},
            "optimizer": {"step": torch.tensor(42)},
        }
        mgr_moe._global_step = 250  # 手动推进到触发点
        mgr_moe.update_slc_index(layer_idx=0, token_range=(0, 128), device_id=2)
        fut = mgr_moe.trigger_save(dummy_state, force=True)
        if fut is not None:
            fut.result(timeout=30)

        loaded = mgr_moe.load_latest(map_location="cpu")
        assert loaded is not None, "load_latest should return a payload"
        assert loaded["global_step"] == 250
        logger.info("Test 4 passed: save → load round-trip at step=250")

        # ── Test 5: prune_old_checkpoints 不删除唯一 checkpoint ──────────────
        mgr_moe.prune_old_checkpoints()
        remaining = mgr_moe.load_latest(map_location="cpu")
        assert remaining is not None, "Sole checkpoint should not be pruned"
        logger.info("Test 5 passed: sole checkpoint survives pruning")

        mgr_moe.shutdown()
        mgr.shutdown()

    logger.info("All smoke tests passed ✓")
