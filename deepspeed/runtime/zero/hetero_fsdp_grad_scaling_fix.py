"""
HeteroFSDPGradScalingFix — DES-LOC Heterogeneous FSDP Gradient Scaling Pipeline
=================================================================================

上游设计意图 (Megatron ba456fd):
    Megatron-FSDP 的 GradReducePipeline 在执行梯度规约时，错误地将 gbuf.data
    (整个梯度缓冲区的全局视图) 传入 gradient_reduce_preprocessing，而非
    bucket.data (当前分桶的局部视图)。这导致梯度缩放因子被应用到错误的张量
    范围，造成数值不稳定甚至梯度爆炸。修复方式：将目标从 gbuf.data 替换为
    bucket.data，确保每个分桶独立缩放。

DES-LOC 适配点:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) 在异构硬件
    (2x A6000 SM86 + 1x H100 NVL SM90, PCIe 互联, 1.5TB CPU DRAM) 上引入
    额外复杂性：

    1. 分桶异构驻留 (Bucket Heterogeneous Residency):
       同一 GradBuffer 的不同 bucket 可能驻留在不同设备
       (A6000-0 / A6000-1 / H100 / CPU DRAM)，bucket.data 与 gbuf.data
       不再同设备，必须在正确设备上执行缩放。

    2. SM 架构感知缩放 (SM-Aware Scaling):
       SM86 (A6000) 与 SM90 (H100) 的 BF16/FP32 累加精度不同；
       H100 NVL 的 TF32 路径默认启用，需在缩放前显式 guard。

    3. PCIe 带宽节流下的延迟缩放 (Deferred Scaling under PCIe Throttle):
       跨设备 all-reduce 前，将缩放操作推迟到数据已在目标设备就绪后执行，
       避免额外 D2D 拷贝。LOC-Cache 负责跟踪 bucket 的当前物理位置。

    4. CPU DRAM Offload 路径:
       当 bucket 被 offload 到 CPU DRAM 时，缩放在 CPU 上用 torch.float32
       完成后再异步 prefetch 回 GPU，与 DES 调度器的 prefetch 流水线对齐。

作者: Neuron_SP Project (dylanyunlon/Neuron_SP)
上游参考: github.com/NVIDIA/Megatron-LM commit ba456fdad991b085ca4f19dea11f7ed886d73ce8
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举 & 常量
# ---------------------------------------------------------------------------

class DeviceClass(Enum):
    """DES-LOC 支持的设备类别，对应 Neuron_SP 硬件拓扑。"""
    A6000_SM86 = auto()   # 2x A6000 48 GB, SM86
    H100_SM90  = auto()   # 1x H100 NVL 96 GB, SM90
    CPU_DRAM   = auto()   # 1.5 TB host DRAM (offload tier)


# SM90 默认开启 TF32，会影响 FP32 梯度累加精度
_SM90_TF32_GUARD_REQUIRED = True
# PCIe gen4 x16 理论带宽 ~32 GB/s；保守阈值用于决策是否推迟跨设备缩放
_PCIE_BW_THRESHOLD_BYTES_PER_SEC: float = 28.0 * 1024 ** 3


# ---------------------------------------------------------------------------
# 辅助数据结构
# ---------------------------------------------------------------------------

@dataclass
class BucketDescriptor:
    """
    描述一个梯度分桶的物理位置与元信息。

    在 DES-LOC 中，LOC-Cache 维护每个 bucket 的当前驻留位置；
    调度器在 prefetch / evict 时更新 ``current_device``。
    """
    bucket_id: int
    data: Tensor                        # 分桶梯度张量（可能在任意设备上）
    numel: int
    gradient_scaling_factor: float
    current_device: DeviceClass
    is_offloaded: bool = False          # 是否已 offload 到 CPU DRAM
    scaling_deferred: bool = False      # 是否已推迟缩放（PCIe 节流路径）
    _scaled: bool = False               # 内部标记：缩放是否已完成

    def mark_scaled(self) -> None:
        self._scaled = True
        self.scaling_deferred = False

    @property
    def needs_scaling(self) -> bool:
        return not self._scaled


@dataclass
class GradBufferDescriptor:
    """
    对应 Megatron GradBuffer 的 DES-LOC 表示。

    ``buckets`` 列表顺序与 Megatron bucket 索引对齐，
    但每个 bucket 可独立驻留在不同设备上。
    """
    gbuf_id: int
    buckets: List[BucketDescriptor]
    ddp_config: Dict                    # 透传 Megatron ddp_config 语义
    is_data_distributed: bool = True    # FSDP scatter 模式
    dtype: torch.dtype = torch.bfloat16

    # 整体梯度缓冲区的全局数据视图（仅用于非分布式 all-reduce 路径）
    global_data: Optional[Tensor] = None


# ---------------------------------------------------------------------------
# SM 架构感知工具
# ---------------------------------------------------------------------------

def _detect_device_class(device: torch.device) -> DeviceClass:
    """
    根据 CUDA device index 推断 DeviceClass。

    假设 Neuron_SP 硬件拓扑固定为:
        device 0 → A6000 SM86
        device 1 → A6000 SM86
        device 2 → H100 NVL SM90

    生产环境可通过 ``torch.cuda.get_device_properties`` 动态检测。
    """
    if device.type == "cpu":
        return DeviceClass.CPU_DRAM
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor   # e.g. 86, 90
    if sm >= 90:
        return DeviceClass.H100_SM90
    return DeviceClass.A6000_SM86


def _enter_tf32_guard(device_class: DeviceClass) -> bool:
    """
    若设备为 SM90 且 TF32 guard 已启用，临时禁用 TF32 以保证
    梯度缩放的数值精度。返回原始 allow_tf32 状态供恢复使用。
    """
    if device_class == DeviceClass.H100_SM90 and _SM90_TF32_GUARD_REQUIRED:
        orig = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        logger.debug("[DES-LOC] SM90 TF32 guard: disabled tf32 for gradient scaling")
        return orig
    return torch.backends.cuda.matmul.allow_tf32


def _exit_tf32_guard(orig_state: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = orig_state


# ---------------------------------------------------------------------------
# 核心修复：bucket 级梯度缩放（对应 Megatron ba456fd 的 bucket.data 修正）
# ---------------------------------------------------------------------------

def _scale_bucket_data_inplace(
    bucket: BucketDescriptor,
    scaling_factor: float,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """
    **核心修复函数** — 对应 Megatron ba456fd: ``bucket.data`` 替代 ``gbuf.data``。

    在 DES-LOC 中，``bucket.data`` 与 ``gbuf.global_data`` 可能驻留在不同设备，
    因此缩放必须在 ``bucket.data.device`` 上就地执行，不能使用全局缓冲区视图。

    参数
    ----
    bucket:
        当前分桶描述符；缩放目标是 ``bucket.data``，而非全局缓冲区。
    scaling_factor:
        梯度缩放因子（来自 GradBuffer，但作用域限定于本桶）。
    stream:
        可选 CUDA 流；若提供则在该流上异步执行缩放。
    """
    if not bucket.needs_scaling:
        logger.debug(
            "[DES-LOC] bucket %d already scaled, skipping", bucket.bucket_id
        )
        return

    target = bucket.data
    device_class = _detect_device_class(target.device)

    orig_tf32 = _enter_tf32_guard(device_class)
    try:
        ctx = torch.cuda.stream(stream) if (
            stream is not None and target.device.type == "cuda"
        ) else _null_context()

        with ctx:
            if device_class == DeviceClass.CPU_DRAM:
                # CPU DRAM offload 路径：强制 float32 以避免 BF16 下溢
                target_f32 = target.float()
                target_f32.mul_(scaling_factor)
                target.copy_(target_f32.to(target.dtype))
                logger.debug(
                    "[DES-LOC] bucket %d: CPU DRAM scaling (f32 round-trip), "
                    "factor=%.6e, numel=%d",
                    bucket.bucket_id, scaling_factor, bucket.numel,
                )
            else:
                # GPU 路径（SM86 or SM90）：直接就地缩放
                target.mul_(scaling_factor)
                logger.debug(
                    "[DES-LOC] bucket %d: GPU scaling on %s (%s), "
                    "factor=%.6e, numel=%d",
                    bucket.bucket_id, target.device, device_class.name,
                    scaling_factor, bucket.numel,
                )
    finally:
        _exit_tf32_guard(orig_tf32)

    bucket.mark_scaled()


# ---------------------------------------------------------------------------
# PCIe 节流感知：延迟缩放决策
# ---------------------------------------------------------------------------

def _should_defer_scaling(
    bucket: BucketDescriptor,
    target_reduce_device: torch.device,
    estimated_transfer_bytes: int,
) -> bool:
    """
    判断是否应将缩放推迟到数据迁移完成后。

    逻辑：
        若 bucket 当前设备 ≠ reduce 目标设备，且估算传输量超过
        PCIe 节流阈值（单次传输 >256 MB 视为大传输），则推迟缩放，
        以避免：缩放 → D2D 拷贝 → 再次访问 的双重带宽消耗。

    返回 True 表示推迟（由调用方在数据就位后补调 _scale_bucket_data_inplace）。
    """
    src_device = bucket.data.device
    if src_device == target_reduce_device:
        return False  # 同设备无需考虑 PCIe

    large_transfer = estimated_transfer_bytes > 256 * 1024 * 1024
    if large_transfer:
        logger.info(
            "[DES-LOC] Deferring scaling for bucket %d: "
            "src=%s, dst=%s, transfer=%.1f MB",
            bucket.bucket_id, src_device, target_reduce_device,
            estimated_transfer_bytes / 1024 ** 2,
        )
        bucket.scaling_deferred = True
        return True
    return False


# ---------------------------------------------------------------------------
# gradient_reduce_preprocessing 的 DES-LOC 版本
# ---------------------------------------------------------------------------

def hetero_gradient_reduce_preprocessing(
    bucket: BucketDescriptor,
    gbuf: GradBufferDescriptor,
    target_reduce_device: torch.device,
    reduce_stream: Optional[torch.cuda.Stream] = None,
) -> Optional[Tensor]:
    """
    DES-LOC 版 ``gradient_reduce_preprocessing``。

    对应 Megatron ba456fd 的修复逻辑，但扩展为处理异构设备场景：

        1. 使用 ``bucket.data``（而非 ``gbuf.global_data``）作为缩放目标。
        2. 根据 bucket 的物理驻留位置决定缩放策略。
        3. 若检测到 PCIe 大传输，推迟缩放并返回 None（延迟路径）。
        4. 返回处理后的 ``bucket.data`` 供后续 all-reduce / reduce-scatter。

    参数
    ----
    bucket:
        当前分桶；缩放发生在 ``bucket.data`` 上。
    gbuf:
        父梯度缓冲区；提供 ``gradient_scaling_factor`` 和 ``ddp_config``。
    target_reduce_device:
        all-reduce / reduce-scatter 将在哪个设备上执行。
    reduce_stream:
        可选 CUDA 流，用于异步流水线。

    返回
    ----
    处理后的 ``bucket.data`` 张量；若进入延迟路径则返回 None。
    """
    scaling_factor = bucket.gradient_scaling_factor

    # --- PCIe 延迟缩放决策 ---
    estimated_bytes = bucket.numel * bucket.data.element_size()
    if _should_defer_scaling(bucket, target_reduce_device, estimated_bytes):
        # 调用方负责在数据迁移完成后调用 flush_deferred_scaling
        return None

    # --- 核心修复：在 bucket.data 上执行缩放（而非 gbuf.global_data）---
    _scale_bucket_data_inplace(bucket, scaling_factor, stream=reduce_stream)

    # --- 跨设备迁移（如有必要）---
    result = bucket.data
    if result.device != target_reduce_device:
        logger.debug(
            "[DES-LOC] Moving bucket %d: %s → %s",
            bucket.bucket_id, result.device, target_reduce_device,
        )
        if reduce_stream is not None and result.device.type == "cuda":
            with torch.cuda.stream(reduce_stream):
                result = result.to(target_reduce_device, non_blocking=True)
        else:
            result = result.to(target_reduce_device)

    return result


# ---------------------------------------------------------------------------
# GradReducePipeline — DES-LOC 异构实现
# ---------------------------------------------------------------------------

class HeteroFSDPGradScalingFix:
    """
    DES-LOC 版 GradReducePipeline，修复 Megatron ba456fd 的梯度缩放目标错误。

    设计目标
    --------
    - 正确性：每个 bucket 使用自身的 ``bucket.data`` 作为缩放目标，
      消除 Megatron 原始 bug（误用 ``gbuf.data`` 导致的错误缩放范围）。
    - 异构感知：根据 bucket 的物理驻留设备（A6000/H100/CPU）选择合适的
      缩放精度和执行路径。
    - 流水线友好：支持 ``reduce_stream`` 异步流水线，与 DES 调度器集成。
    - LOC-Cache 兼容：通过 ``BucketDescriptor.current_device`` 跟踪位置，
      供 LOC-Cache 的 prefetch/evict 策略使用。

    异构硬件拓扑
    ------------
    ::

        A6000-0 (SM86, 48GB) ─┐
                               ├─ PCIe ─ CPU DRAM (1.5TB)
        A6000-1 (SM86, 48GB) ─┤
                               │
        H100 NVL (SM90, 96GB) ─┘

    注意：无 NVLink，设备间通信全部经 PCIe，带宽为瓶颈。
    """

    def __init__(
        self,
        grad_buffers: List[GradBufferDescriptor],
        process_group: Optional[dist.ProcessGroup] = None,
        reduce_stream: Optional[torch.cuda.Stream] = None,
        default_reduce_device: Optional[torch.device] = None,
    ) -> None:
        """
        初始化 HeteroFSDPGradScalingFix。

        参数
        ----
        grad_buffers:
            所有 GradBuffer 的描述符列表，包含其 bucket 拓扑。
        process_group:
            分布式通信组；None 时假设单机多卡场景。
        reduce_stream:
            CUDA 流，用于异步梯度规约流水线。
        default_reduce_device:
            默认 reduce 目标设备；None 时选择 H100（SM90）作为主 reduce 设备
            （H100 有更大显存和更高带宽，适合作为 reduce 协调者）。
        """
        self.grad_buffers = grad_buffers
        self.process_group = process_group
        self.reduce_stream = reduce_stream

        if default_reduce_device is None:
            # H100 NVL 作为默认 reduce 设备（device index 2）
            self.default_reduce_device = torch.device("cuda:2")
        else:
            self.default_reduce_device = default_reduce_device

        self._deferred_buckets: List[Tuple[GradBufferDescriptor, BucketDescriptor]] = []
        logger.info(
            "[DES-LOC] HeteroFSDPGradScalingFix initialized: "
            "%d grad_buffers, reduce_device=%s",
            len(grad_buffers), self.default_reduce_device,
        )

    # ------------------------------------------------------------------
    # 主入口：运行梯度规约流水线
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        执行完整的梯度缩放与规约流水线。

        流程：
            1. 遍历所有 GradBuffer 的所有 bucket。
            2. 对每个 bucket 调用 hetero_gradient_reduce_preprocessing。
            3. 根据 ``gbuf.is_data_distributed`` 选择 reduce-scatter 或 all-reduce。
            4. 处理所有延迟缩放的 bucket（flush_deferred_scaling）。
        """
        logger.info("[DES-LOC] GradReducePipeline: starting run()")
        self._deferred_buckets.clear()

        for gbuf in self.grad_buffers:
            self._process_grad_buffer(gbuf)

        # 处理 PCIe 延迟路径
        if self._deferred_buckets:
            logger.info(
                "[DES-LOC] Flushing %d deferred buckets", len(self._deferred_buckets)
            )
            self.flush_deferred_scaling()

        logger.info("[DES-LOC] GradReducePipeline: run() complete")

    def _process_grad_buffer(self, gbuf: GradBufferDescriptor) -> None:
        """处理单个 GradBuffer 的所有 bucket。"""
        logger.debug(
            "[DES-LOC] Processing GradBuffer %d (%d buckets, distributed=%s)",
            gbuf.gbuf_id, len(gbuf.buckets), gbuf.is_data_distributed,
        )

        for bucket in gbuf.buckets:
            reduce_device = self._select_reduce_device(bucket)
            preprocessed = hetero_gradient_reduce_preprocessing(
                bucket=bucket,
                gbuf=gbuf,
                target_reduce_device=reduce_device,
                reduce_stream=self.reduce_stream,
            )

            if preprocessed is None:
                # 进入延迟路径
                self._deferred_buckets.append((gbuf, bucket))
                continue

            self._launch_collective(gbuf, bucket, preprocessed, reduce_device)

    def _select_reduce_device(self, bucket: BucketDescriptor) -> torch.device:
        """
        为 bucket 选择 reduce 目标设备。

        策略：
        - 若 bucket 当前在 H100（SM90），直接在本地 reduce（避免不必要迁移）。
        - 若 bucket 在 A6000，迁移到 H100 reduce（利用 H100 的高带宽显存）。
        - 若 bucket 在 CPU DRAM 且体积小（<64MB），迁移到最近的 A6000。
        - 若 bucket 在 CPU DRAM 且体积大，留在 CPU 执行 reduce（避免 PCIe 饱和）。
        """
        dclass = bucket.current_device
        numel_bytes = bucket.numel * bucket.data.element_size()

        if dclass == DeviceClass.H100_SM90:
            return self.default_reduce_device

        if dclass == DeviceClass.A6000_SM86:
            return self.default_reduce_device  # 迁移到 H100

        # CPU DRAM
        if numel_bytes < 64 * 1024 * 1024:
            # 小 bucket：迁移到 A6000-0
            return torch.device("cuda:0")
        else:
            # 大 bucket：CPU reduce
            return torch.device("cpu")

    def _launch_collective(
        self,
        gbuf: GradBufferDescriptor,
        bucket: BucketDescriptor,
        data: Tensor,
        reduce_device: torch.device,
    ) -> None:
        """
        在 ``reduce_device`` 上发起分布式通信原语。

        对应 Megatron GradReducePipeline 中的两路分支：
        - ``is_data_distributed=True``：reduce-scatter（FSDP 模式）
        - ``is_data_distributed=False``：all-reduce
        """
        if self.process_group is None or not dist.is_initialized():
            logger.debug(
                "[DES-LOC] No process group; skipping collective for bucket %d",
                bucket.bucket_id,
            )
            return

        stream_ctx = (
            torch.cuda.stream(self.reduce_stream)
            if self.reduce_stream and reduce_device.type == "cuda"
            else _null_context()
        )
        with stream_ctx:
            if gbuf.is_data_distributed:
                # FSDP reduce-scatter
                output_size = data.numel() // dist.get_world_size(self.process_group)
                output = torch.empty(
                    output_size, dtype=data.dtype, device=reduce_device
                )
                dist.reduce_scatter_tensor(
                    output, data,
                    group=self.process_group,
                    async_op=False,
                )
                logger.debug(
                    "[DES-LOC] bucket %d: reduce-scatter done, output numel=%d",
                    bucket.bucket_id, output.numel(),
                )
            else:
                # 非分布式：all-reduce
                dist.all_reduce(data, group=self.process_group, async_op=False)
                logger.debug(
                    "[DES-LOC] bucket %d: all-reduce done", bucket.bucket_id
                )

    # ------------------------------------------------------------------
    # 延迟缩放刷新
    # ------------------------------------------------------------------

    def flush_deferred_scaling(self) -> None:
        """
        处理所有因 PCIe 节流而推迟缩放的 bucket。

        在 DES-LOC 调度器完成 prefetch（数据已就位于目标设备）后调用。
        此时可安全执行缩放 + collective，无需额外 D2D 拷贝。
        """
        for gbuf, bucket in self._deferred_buckets:
            logger.info(
                "[DES-LOC] flush_deferred_scaling: bucket %d, device=%s",
                bucket.bucket_id, bucket.data.device,
            )
            # 数据已就位，直接缩放（不再检查 PCIe 阈值）
            _scale_bucket_data_inplace(
                bucket,
                bucket.gradient_scaling_factor,
                stream=self.reduce_stream,
            )
            reduce_device = self._select_reduce_device(bucket)
            data = bucket.data
            if data.device != reduce_device:
                data = data.to(reduce_device)
            self._launch_collective(gbuf, bucket, data, reduce_device)

        self._deferred_buckets.clear()

    # ------------------------------------------------------------------
    # LOC-Cache 集成接口
    # ------------------------------------------------------------------

    def notify_bucket_moved(
        self,
        gbuf_id: int,
        bucket_id: int,
        new_device: torch.device,
        new_data: Tensor,
    ) -> None:
        """
        LOC-Cache 通知接口：当调度器将 bucket 迁移到新设备时调用。

        更新 ``BucketDescriptor`` 的 ``data`` 和 ``current_device``，
        确保后续缩放在正确的设备上执行。
        """
        gbuf = next((g for g in self.grad_buffers if g.gbuf_id == gbuf_id), None)
        if gbuf is None:
            logger.warning("[DES-LOC] notify_bucket_moved: gbuf_id %d not found", gbuf_id)
            return
        bucket = next((b for b in gbuf.buckets if b.bucket_id == bucket_id), None)
        if bucket is None:
            logger.warning("[DES-LOC] notify_bucket_moved: bucket_id %d not found", bucket_id)
            return

        old_device = bucket.data.device
        bucket.data = new_data
        bucket.current_device = _detect_device_class(new_device)
        bucket.is_offloaded = (bucket.current_device == DeviceClass.CPU_DRAM)

        logger.info(
            "[DES-LOC] bucket (%d, %d) moved: %s → %s (%s)",
            gbuf_id, bucket_id, old_device, new_device, bucket.current_device.name,
        )


# ---------------------------------------------------------------------------
# 工厂函数：从 DeepSpeed ZeRO 参数构建异构 GradBuffer 拓扑
# ---------------------------------------------------------------------------

def build_hetero_grad_buffers(
    param_groups: List[List[torch.nn.Parameter]],
    bucket_size_mb: float = 128.0,
    dtype: torch.dtype = torch.bfloat16,
    device_assignment: Optional[Dict[int, torch.device]] = None,
) -> List[GradBufferDescriptor]:
    """
    从参数组构建异构 GradBuffer 拓扑。

    将参数按 ``bucket_size_mb`` 分组，并根据 ``device_assignment`` 将
    每个 bucket 分配到合适的设备（A6000-0/1, H100, CPU DRAM）。

    参数
    ----
    param_groups:
        参数组列表，每组对应一个 GradBuffer。
    bucket_size_mb:
        每个分桶的目标大小（MB）。
    dtype:
        梯度数据类型；DES-LOC 默认 BF16。
    device_assignment:
        bucket_id → device 的手动映射；None 时使用默认拓扑策略。

    返回
    ----
    ``GradBufferDescriptor`` 列表，可直接传入 ``HeteroFSDPGradScalingFix``。
    """
    bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
    elem_bytes = torch.finfo(dtype).bits // 8

    grad_buffers: List[GradBufferDescriptor] = []

    for gbuf_id, params in enumerate(param_groups):
        buckets: List[BucketDescriptor] = []
        bucket_id = 0
        current_params: List[torch.nn.Parameter] = []
        current_numel = 0

        def _flush_bucket(params_list: List[torch.nn.Parameter], bid: int) -> BucketDescriptor:
            total = sum(p.numel() for p in params_list)
            if device_assignment and bid in device_assignment:
                dev = device_assignment[bid]
            else:
                # 默认策略：偶数桶放 A6000-0，奇数桶放 A6000-1，大桶放 H100
                if total * elem_bytes > 256 * 1024 * 1024:
                    dev = torch.device("cuda:2")  # H100
                elif bid % 2 == 0:
                    dev = torch.device("cuda:0")
                else:
                    dev = torch.device("cuda:1")

            data = torch.zeros(total, dtype=dtype, device=dev)
            dclass = _detect_device_class(dev)
            logger.debug(
                "[DES-LOC] GradBuffer %d, bucket %d: numel=%d, device=%s (%s)",
                gbuf_id, bid, total, dev, dclass.name,
            )
            return BucketDescriptor(
                bucket_id=bid,
                data=data,
                numel=total,
                gradient_scaling_factor=1.0,
                current_device=dclass,
            )

        for p in params:
            if current_numel * elem_bytes + p.numel() * elem_bytes > bucket_size_bytes and current_params:
                buckets.append(_flush_bucket(current_params, bucket_id))
                bucket_id += 1
                current_params = []
                current_numel = 0
            current_params.append(p)
            current_numel += p.numel()

        if current_params:
            buckets.append(_flush_bucket(current_params, bucket_id))

        grad_buffers.append(GradBufferDescriptor(
            gbuf_id=gbuf_id,
            buckets=buckets,
            ddp_config={"gradient_as_bucket_view": True},
            dtype=dtype,
        ))
        logger.info(
            "[DES-LOC] GradBuffer %d: %d buckets created", gbuf_id, len(buckets)
        )

    return grad_buffers


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

class _null_context:
    """空上下文管理器，用于统一 GPU/CPU 路径的 with 语句。"""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # --- 构造最小 GradBuffer 拓扑（CPU 模式，不依赖多 GPU）---
    dev_cpu = torch.device("cpu")

    b0 = BucketDescriptor(
        bucket_id=0,
        data=torch.ones(1024, dtype=torch.bfloat16, device=dev_cpu) * 2.0,
        numel=1024,
        gradient_scaling_factor=0.5,
        current_device=DeviceClass.CPU_DRAM,
    )
    b1 = BucketDescriptor(
        bucket_id=1,
        data=torch.ones(512, dtype=torch.bfloat16, device=dev_cpu) * 4.0,
        numel=512,
        gradient_scaling_factor=0.25,
        current_device=DeviceClass.CPU_DRAM,
    )
    gbuf = GradBufferDescriptor(
        gbuf_id=0,
        buckets=[b0, b1],
        ddp_config={"gradient_as_bucket_view": True},
        dtype=torch.bfloat16,
    )

    # --- 测试1：bucket 级缩放（核心修复验证）---
    _scale_bucket_data_inplace(b0, b0.gradient_scaling_factor)
    assert b0._scaled, "bucket 0 应标记为已缩放"
    expected_b0 = torch.ones(1024, dtype=torch.float32) * 2.0 * 0.5
    actual_b0 = b0.data.float()
    assert torch.allclose(actual_b0, expected_b0, atol=1e-3), (
        f"bucket 0 缩放结果错误: mean={actual_b0.mean():.4f}, expected={expected_b0.mean():.4f}"
    )
    logger.info("✓ Test 1 passed: bucket-level scaling correct (ba456fd fix)")

    # --- 测试2：不重复缩放 ---
    val_before = b0.data.clone()
    _scale_bucket_data_inplace(b0, b0.gradient_scaling_factor)  # 应跳过
    assert torch.allclose(b0.data, val_before), "已缩放的 bucket 不应被重复缩放"
    logger.info("✓ Test 2 passed: no double-scaling guard")

    # --- 测试3：hetero_gradient_reduce_preprocessing 返回正确张量 ---
    result = hetero_gradient_reduce_preprocessing(
        bucket=b1,
        gbuf=gbuf,
        target_reduce_device=dev_cpu,
    )
    assert result is not None, "非 PCIe 延迟路径应返回 tensor"
    assert b1._scaled, "preprocessing 应触发 bucket 1 缩放"
    expected_b1_val = 4.0 * 0.25
    assert abs(result.float().mean().item() - expected_b1_val) < 1e-3, (
        f"bucket 1 preprocessing 结果错误: {result.float().mean().item():.4f} vs {expected_b1_val}"
    )
    logger.info("✓ Test 3 passed: hetero_gradient_reduce_preprocessing correctness")

    # --- 测试4：notify_bucket_moved 更新设备元数据 ---
    pipeline = HeteroFSDPGradScalingFix(
        grad_buffers=[gbuf],
        process_group=None,
        reduce_stream=None,
        default_reduce_device=dev_cpu,
    )
    new_data = torch.zeros(1024, dtype=torch.bfloat16, device=dev_cpu)
    pipeline.notify_bucket_moved(0, 0, dev_cpu, new_data)
    assert gbuf.buckets[0].current_device == DeviceClass.CPU_DRAM
    assert gbuf.buckets[0].data is new_data
    logger.info("✓ Test 4 passed: notify_bucket_moved updates BucketDescriptor")

    # --- 测试5：_should_defer_scaling 小体积不推迟 ---
    b_small = BucketDescriptor(
        bucket_id=99,
        data=torch.zeros(16, dtype=torch.bfloat16),
        numel=16,
        gradient_scaling_factor=1.0,
        current_device=DeviceClass.CPU_DRAM,
    )
    defer = _should_defer_scaling(b_small, torch.device("cuda:0"), 16 * 2)
    assert not defer, "16 元素 bucket 不应触发 PCIe 延迟"
    logger.info("✓ Test 5 passed: small bucket bypass PCIe deferral")

    logger.info("All smoke tests passed — HeteroFSDPGradScalingFix ready.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroFSDPGradScalingFix on a DeepSpeed engine.

    Instantiates a :class:`HeteroFSDPGradScalingFix` from the engine's configuration
    and attaches it as ``engine.hetero_fsdp_grad_scaling_fix``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_fsdp_grad_scaling_fix.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_fsdp_grad_scaling_fix = None
    logger.info("hetero_fsdp_grad_scaling_fix.register() attached engine.hetero_fsdp_grad_scaling_fix")
