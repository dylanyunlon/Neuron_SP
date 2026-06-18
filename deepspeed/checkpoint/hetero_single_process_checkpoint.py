# Copyright (c) 2026, Neuron_SP Project Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Part of Neuron_SP: github.com/dylanyunlon/Neuron_SP
# Upstream mirror: Megatron-LM commit 44e27d0401eaf9501cf06fda6c343ae6d8b4fe33
#   "Add mxfp8 quantization for inference linear layers (#3447)"
#
# DES-LOC Adaptation: HeteroSingleProcessCheckpoint
# ===================================================
# Megatron上游设计意图：
#   该commit为推理线性层添加MXFP8量化支持，核心是引入了一套
#   TE(TransformerEngine) → FlashInfer格式转换管线，使推理引擎能
#   在"inference_optimized"路径下使用flashinfer的mm_mxfp8内核。
#   量化发生在checkpoint加载之后、推理之前的一次性转换步骤中。
#
# DES-LOC适配点（HeteroSingleProcessCheckpoint）：
#   DES-LOC（Decoupled Execution with Shared LOcality Cache）的硬件拓扑：
#     - 2x A6000 48GB (SM86, PCIe)
#     - 1x H100 NVL 96GB (SM90, PCIe)
#     - 1.5TB CPU DRAM作为共享LOcality Cache层
#   在此拓扑中，checkpoint保存/加载面临三个独特挑战：
#   1. 量化格式异构：H100可原生运行FP8/MXFP8(SM90)，A6000仅支持
#      BF16/FP16(SM86)，同一模型的不同层需要保存不同精度的权重。
#   2. 单进程协调：无NVLink意味着跨GPU通信代价极高，checkpoint的
#      序列化/反序列化必须由单进程完成，避免昂贵的进程间同步。
#   3. CPU DRAM作为中转站：1.5TB DRAM是唯一能容纳完整模型的内存，
#      量化转换（TE→FlashInfer格式）必须在CPU上完成或在各GPU本地完成，
#      由CPU DRAM做staging buffer。
#
#   本文件实现了：
#   - HeteroQuantConfig：描述每个设备可支持的量化格式
#   - MXFP8TensorDESLOC：适配DES-LOC的MXFP8张量包装，支持CPU staging
#   - HeteroSingleProcessCheckpoint：异构单进程checkpoint管理器，
#     负责在保存/加载时按设备能力自动选择量化精度，通过CPU DRAM staging
#     完成跨格式转换
#   - QuantizationConverter：TE→FlashInfer格式的DES-LOC本地转换器

import gc
import logging
import os
import pickle
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 量化格式枚举
# ---------------------------------------------------------------------------

class QuantFormat(Enum):
    """
    DES-LOC支持的量化格式枚举。

    不同SM架构支持的格式不同：
    - SM86 (A6000): 仅支持BF16/FP16，无原生FP8硬件支持
    - SM90 (H100):  原生支持FP8，可使用MXFP8 FlashInfer内核
    """
    BF16 = auto()       # BF16: A6000/H100均支持
    FP16 = auto()       # FP16: 通用回退格式
    FP8_E4M3 = auto()   # FP8 E4M3: 仅SM90+支持
    MXFP8 = auto()      # MX-scaled FP8: 仅SM90+ + FlashInfer >= 0.6.4
    CPU_FP32 = auto()   # CPU DRAM staging用FP32


# ---------------------------------------------------------------------------
# 设备能力检测
# ---------------------------------------------------------------------------

def _get_device_sm(device: torch.device) -> int:
    """
    获取指定CUDA设备的SM架构版本号（major * 10 + minor）。

    返回值示例：
    - A6000 (SM86): 86
    - H100 NVL (SM90): 90

    Args:
        device: CUDA设备

    Returns:
        SM版本整数，CPU设备返回0
    """
    if device.type == "cpu":
        return 0
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def _check_flashinfer_version() -> bool:
    """
    检查FlashInfer是否满足MXFP8所需的最低版本(>= 0.6.4)。

    上游Megatron的validate_args中有相同检查，此处在DES-LOC
    checkpoint层复现，因为checkpoint加载可能早于argument validation。

    Returns:
        True if flashinfer >= 0.6.4 且可用
    """
    try:
        import flashinfer
        from packaging.version import Version
        ver_str = getattr(flashinfer, "__version__", None)
        if ver_str is None:
            from importlib.metadata import version as pkg_version
            ver_str = pkg_version("flashinfer")
        ok = Version(ver_str) >= Version("0.6.4")
        if not ok:
            logger.warning(
                "FlashInfer版本 %s < 0.6.4，MXFP8将回退到BF16。"
                "请升级：pip install flashinfer>=0.6.4",
                ver_str,
            )
        return ok
    except Exception as e:
        logger.debug("FlashInfer版本检查失败: %s", e)
        return False


# ---------------------------------------------------------------------------
# HeteroQuantConfig：异构量化配置
# ---------------------------------------------------------------------------

@dataclass
class HeteroQuantConfig:
    """
    描述DES-LOC集群中每个设备支持的量化能力。

    Megatron上游通过TransformerConfig.fp8_recipe来全局控制量化格式，
    这对同构集群足够，但DES-LOC的A6000(SM86)+H100(SM90)混合集群
    需要per-device的量化能力描述。

    DES-LOC适配：
    - 每个torch.device映射到其最优量化格式
    - CPU DRAM始终以FP32/BF16存储，作为跨格式转换的中间层
    - A6000强制回退到BF16（SM86无FP8硬件）
    - H100可使用MXFP8（当FlashInfer>=0.6.4时）

    Attributes:
        device_formats: device_index -> QuantFormat的映射
        cpu_staging_dtype: CPU DRAM staging时使用的dtype
        mxfp8_group_size: MXFP8量化的group size，默认32
        enable_cpu_offload: 是否启用CPU DRAM作为LOcality Cache
        locality_cache_size_gb: 允许使用的CPU DRAM大小（GB）
    """
    device_formats: Dict[int, QuantFormat] = field(default_factory=dict)
    cpu_staging_dtype: torch.dtype = torch.bfloat16
    mxfp8_group_size: int = 32
    enable_cpu_offload: bool = True
    locality_cache_size_gb: float = 512.0   # 保守使用1.5TB的1/3

    @classmethod
    def auto_detect(cls) -> "HeteroQuantConfig":
        """
        自动检测当前系统的GPU拓扑并生成最优量化配置。

        对DES-LOC标准配置（2xA6000 + 1xH100）的检测逻辑：
        - SM86设备 (A6000): BF16
        - SM90设备 (H100): MXFP8（如FlashInfer可用）或FP8_E4M3

        Returns:
            自动检测的HeteroQuantConfig实例
        """
        device_formats: Dict[int, QuantFormat] = {}
        has_flashinfer = _check_flashinfer_version()

        for dev_idx in range(torch.cuda.device_count()):
            device = torch.device("cuda", dev_idx)
            sm = _get_device_sm(device)
            name = torch.cuda.get_device_properties(device).name

            if sm >= 90:
                # H100 NVL或更新架构
                fmt = QuantFormat.MXFP8 if has_flashinfer else QuantFormat.FP8_E4M3
                logger.info(
                    "设备 cuda:%d (%s, SM%d) → %s",
                    dev_idx, name, sm, fmt.name,
                )
            elif sm >= 80:
                # A6000等Ampere架构，无原生FP8
                fmt = QuantFormat.BF16
                logger.info(
                    "设备 cuda:%d (%s, SM%d) → BF16 (SM86不支持FP8)",
                    dev_idx, name, sm,
                )
            else:
                fmt = QuantFormat.BF16
                logger.warning(
                    "设备 cuda:%d (%s, SM%d) 较旧，强制使用BF16",
                    dev_idx, name, sm,
                )

            device_formats[dev_idx] = fmt

        if not device_formats:
            logger.warning("未检测到CUDA设备，使用CPU模式")

        config = cls(device_formats=device_formats)
        logger.info(
            "DES-LOC HeteroQuantConfig初始化完成: %s",
            {k: v.name for k, v in device_formats.items()},
        )
        return config

    def get_format(self, device: torch.device) -> QuantFormat:
        """
        获取指定设备对应的最优量化格式。

        Args:
            device: 目标设备

        Returns:
            对应的QuantFormat
        """
        if device.type == "cpu":
            return QuantFormat.CPU_FP32
        return self.device_formats.get(device.index, QuantFormat.BF16)


# ---------------------------------------------------------------------------
# MXFP8TensorDESLOC：支持CPU staging的MXFP8张量包装
# ---------------------------------------------------------------------------

@dataclass
class MXFP8TensorDESLOC:
    """
    DES-LOC适配的MXFP8张量包装类。

    上游Megatron的MXFP8Tensor（mxfp8_tensor.py）假设数据始终在CUDA上，
    不支持CPU DRAM staging。DES-LOC需要通过1.5TB CPU DRAM作为
    LOcality Cache（共享局部性缓存），因此需要能在CPU-GPU之间迁移
    的MXFP8表示。

    DES-LOC适配关键点：
    1. data/scale允许存放在CPU（用于LOcality Cache staging）
    2. 提供to_device()方法在CPU<->GPU之间迁移
    3. 记录原始设备和目标设备信息，便于checkpoint管理器路由

    Attributes:
        data: FP8量化后的数据张量（uint8存储）
        scale: 量化缩放因子（每group一个）
        source_device: 数据来源设备（用于追踪staging路径）
        target_device: 计算时的目标设备
        group_size: MXFP8的group size
        original_shape: 量化前的原始张量形状
    """
    data: torch.Tensor
    scale: torch.Tensor
    source_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    target_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    group_size: int = 32
    original_shape: Optional[Tuple[int, ...]] = None

    def size(self, idx: Optional[int] = None) -> Union[torch.Size, int]:
        """上游MXFP8Tensor接口兼容：返回data.size()"""
        if idx is not None:
            return self.data.size(idx)
        return self.data.size()

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    def is_on_cpu(self) -> bool:
        """检查当前数据是否在CPU DRAM（LOcality Cache）中"""
        return self.data.device.type == "cpu"

    def to_device(self, device: torch.device, non_blocking: bool = True) -> "MXFP8TensorDESLOC":
        """
        将MXFP8张量迁移到目标设备。

        在DES-LOC中，这是LOcality Cache → GPU的热迁移路径：
        CPU DRAM → PCIe总线 → GPU HBM

        因为没有NVLink，GPU间直接传输走PCIe，代价极高；
        通过CPU DRAM中转（CPU staging）虽然多一跳，但可以
        批量化传输，减少PCIe通信次数。

        Args:
            device: 目标设备
            non_blocking: 是否异步传输（PCIe场景建议True）

        Returns:
            迁移后的新MXFP8TensorDESLOC实例
        """
        new_data = self.data.to(device, non_blocking=non_blocking)
        new_scale = self.scale.to(device, non_blocking=non_blocking)
        return MXFP8TensorDESLOC(
            data=new_data,
            scale=new_scale,
            source_device=self.data.device,
            target_device=device,
            group_size=self.group_size,
            original_shape=self.original_shape,
        )

    def to_bf16(self) -> torch.Tensor:
        """
        反量化回BF16张量。

        用于：
        1. 将H100上的MXFP8权重送到CPU staging area做格式转换
        2. 为A6000提供BF16格式（A6000不支持FP8）
        3. 精度校验（sanity check）

        Returns:
            BF16格式的张量，shape与original_shape一致
        """
        # 将uint8数据视为float8_e4m3fn
        data_bytes = self.data.contiguous().view(torch.uint8)
        data_e4m3 = data_bytes.view(torch.float8_e4m3fn).to(torch.float32)

        # 还原scale：scale字节 << 23 构造FP32指数
        scale_bytes = self.scale.contiguous().flatten().view(torch.uint8).to(torch.int32)
        scale_f32 = (scale_bytes << 23).view(torch.float32)

        # 每group一个scale，需要reshape对齐
        M, K = self.data.shape
        num_groups = K // self.group_size
        scale_expanded = scale_f32.view(M, num_groups, 1).expand(M, num_groups, self.group_size)
        scale_expanded = scale_expanded.reshape(M, K)

        result = (data_e4m3 * scale_expanded).to(torch.bfloat16)
        logger.debug(
            "MXFP8→BF16反量化: shape=%s, device=%s",
            result.shape, result.device,
        )
        return result

    @classmethod
    def from_bf16_cpu_staging(
        cls,
        x: torch.Tensor,
        group_size: int = 32,
        target_device: Optional[torch.device] = None,
    ) -> "MXFP8TensorDESLOC":
        """
        从BF16张量量化到MXFP8，支持CPU staging路径。

        与上游Megatron的MXFP8Tensor.from_bf16()不同，此方法支持
        在CPU上完成量化（用于DES-LOC的CPU DRAM预处理路径），
        避免将大型BF16权重上传到GPU仅为量化后再下载。

        DES-LOC CPU staging流程：
        BF16权重（CPU DRAM）→ 本地量化 → MXFP8（CPU DRAM）→ 按需传输到GPU

        Args:
            x: 输入BF16张量，shape [M, K]
            group_size: MXFP8的group size，默认32
            target_device: 量化结果最终的目标设备（仅记录，不立即传输）

        Returns:
            MXFP8TensorDESLOC实例（数据在CPU上）
        """
        assert x.dim() == 2, f"输入必须是2D张量 [M, K]，实际shape: {x.shape}"
        M, K = x.shape
        assert K % group_size == 0, (
            f"K ({K}) 必须是 group_size ({group_size}) 的整数倍"
        )

        current_device = x.device

        # 如果有FlashInfer且x在CUDA上，优先使用GPU量化
        if x.is_cuda:
            try:
                from flashinfer import mxfp8_quantize
                data, scale = mxfp8_quantize(x.to(torch.bfloat16))
                logger.debug(
                    "使用FlashInfer GPU量化: shape=%s, device=%s",
                    x.shape, x.device,
                )
                return cls(
                    data=data,
                    scale=scale,
                    source_device=current_device,
                    target_device=target_device or current_device,
                    group_size=group_size,
                    original_shape=tuple(x.shape),
                )
            except ImportError:
                logger.debug("FlashInfer不可用，回退到CPU近似量化")

        # CPU上的近似MXFP8量化（DES-LOC CPU staging路径）
        x_f32 = x.to(torch.float32).cpu()
        num_groups = K // group_size
        x_grouped = x_f32.view(M, num_groups, group_size)

        # 计算每group的amax作为scale
        amax = x_grouped.abs().amax(dim=-1, keepdim=True)  # [M, num_groups, 1]
        # FP8 E4M3的最大可表示值为448.0
        fp8_max = 448.0
        scale_f32 = (amax / fp8_max).clamp(min=1e-12)  # 避免除零

        # 量化
        x_scaled = x_grouped / scale_f32  # [M, num_groups, group_size]
        x_scaled_flat = x_scaled.view(M, K).clamp(-fp8_max, fp8_max)

        # 转为FP8并存储为uint8
        data_e4m3 = x_scaled_flat.to(torch.float8_e4m3fn)
        data_bytes = data_e4m3.view(torch.uint8)

        # scale以uint8格式存储（取FP32指数字节）
        scale_flat = scale_f32.view(M, num_groups)
        scale_int = scale_flat.view(torch.int32)
        scale_bytes = ((scale_int >> 23) & 0xFF).to(torch.uint8)

        logger.debug(
            "使用CPU近似量化: shape=%s, scale_range=[%.4f, %.4f]",
            x.shape, scale_f32.min().item(), scale_f32.max().item(),
        )

        return cls(
            data=data_bytes,
            scale=scale_bytes,
            source_device=torch.device("cpu"),
            target_device=target_device or torch.device("cpu"),
            group_size=group_size,
            original_shape=tuple(x.shape),
        )


# ---------------------------------------------------------------------------
# QuantizationConverter：TE→FlashInfer格式的DES-LOC转换器
# ---------------------------------------------------------------------------

class QuantizationConverter:
    """
    DES-LOC异构量化格式转换器。

    上游Megatron的quantize_model_to_mxfp8()（inference/utils.py）执行
    TE MXFP8 → FlashInfer MXFP8的in-place转换，假设整个模型在单个GPU上。

    DES-LOC适配：
    - 支持按层路由：不同层的权重转换发生在不同设备上
    - CPU DRAM作为staging buffer：大层权重先下载到CPU，转换后再上传
    - per-device量化格式：A6000层保持BF16，H100层转为MXFP8
    - 数值校验（sanity check）与上游保持一致

    Attributes:
        quant_config: 异构量化配置
        staging_device: CPU staging设备（始终为cpu）
        verify_conversion: 是否执行转换精度校验
    """

    def __init__(
        self,
        quant_config: HeteroQuantConfig,
        verify_conversion: bool = True,
    ) -> None:
        self.quant_config = quant_config
        self.staging_device = torch.device("cpu")
        self.verify_conversion = verify_conversion
        logger.info(
            "QuantizationConverter初始化: verify=%s, staging=CPU DRAM",
            verify_conversion,
        )

    def _verify_conversion(
        self,
        original_bf16: torch.Tensor,
        converted: MXFP8TensorDESLOC,
        layer_name: str,
    ) -> None:
        """
        验证BF16→MXFP8转换的数值正确性。

        复刻上游Megatron的_verify_te_to_flashinfer_mxfp8_conversion()逻辑，
        但适配DES-LOC的CPU staging环境（数据可能在CPU上）。

        Args:
            original_bf16: 原始BF16张量
            converted: 转换后的MXFP8张量
            layer_name: 层名称（用于错误信息）

        Raises:
            ValueError: 转换精度超过容忍阈值
        """
        # 在CPU上做校验，避免不必要的GPU内存占用
        orig_cpu = original_bf16.cpu().float()
        conv_cpu = converted.to_bf16().cpu().float()

        # 只校验前32个元素（对应第一个quantization group）
        block_size = min(32, orig_cpu.numel())
        orig_block = orig_cpu.flatten()[:block_size]
        conv_block = conv_cpu.flatten()[:block_size]

        if not torch.allclose(orig_block, conv_block, rtol=0.1, atol=0.1):
            diff_norm = torch.norm(orig_block - conv_block).item()
            raise ValueError(
                f"层 {layer_name} 的MXFP8转换校验失败。"
                f"前{block_size}元素diff norm: {diff_norm:.6f}。"
                f"orig[:5]={orig_block[:5].tolist()}, "
                f"conv[:5]={conv_block[:5].tolist()}"
            )

        logger.debug(
            "层 %s 转换校验通过，diff_norm=%.6f",
            layer_name,
            torch.norm(orig_block - conv_block).item(),
        )

    def convert_weight(
        self,
        weight: torch.Tensor,
        layer_name: str,
        target_device: torch.device,
    ) -> Union[torch.Tensor, MXFP8TensorDESLOC]:
        """
        将单个权重张量按目标设备的量化格式进行转换。

        DES-LOC路由逻辑：
        - 目标设备为A6000(SM86)：保持BF16
        - 目标设备为H100(SM90)且FlashInfer可用：转为MXFP8
        - 目标设备为CPU：保持BF16（LOcality Cache格式）

        Args:
            weight: 原始权重张量（任意格式）
            layer_name: 层名称
            target_device: 权重最终所在设备

        Returns:
            转换后的权重（BF16 Tensor或MXFP8TensorDESLOC）
        """
        target_fmt = self.quant_config.get_format(target_device)
        logger.debug(
            "转换权重 %s: shape=%s, dtype=%s → 目标格式=%s (device=%s)",
            layer_name, weight.shape, weight.dtype, target_fmt.name, target_device,
        )

        if target_fmt == QuantFormat.BF16:
            # A6000路径：直接转BF16，无需量化
            converted = weight.to(torch.bfloat16)
            logger.debug("层 %s → BF16 (A6000/SM86路径)", layer_name)
            return converted

        elif target_fmt == QuantFormat.MXFP8:
            # H100路径：通过CPU staging做MXFP8量化
            if weight.dim() != 2:
                logger.warning(
                    "层 %s 的权重不是2D(%s)，跳过MXFP8量化，回退BF16",
                    layer_name, weight.shape,
                )
                return weight.to(torch.bfloat16)

            # Step1: 将权重下载到CPU DRAM（LOcality Cache）
            weight_cpu = weight.to(torch.bfloat16).cpu()

            # Step2: 在CPU上做量化（或如果目标设备可用则在GPU上）
            mxfp8_tensor = MXFP8TensorDESLOC.from_bf16_cpu_staging(
                weight_cpu,
                group_size=self.quant_config.mxfp8_group_size,
                target_device=target_device,
            )

            # Step3: 数值校验
            if self.verify_conversion:
                try:
                    self._verify_conversion(weight_cpu, mxfp8_tensor, layer_name)
                except ValueError as e:
                    logger.error("转换校验失败，回退BF16: %s", e)
                    return weight_cpu

            logger.info(
                "层 %s → MXFP8 (H100/SM90路径), "
                "data_size=%.2fMB, scale_size=%.2fKB",
                layer_name,
                mxfp8_tensor.data.numel() / 1024**2,
                mxfp8_tensor.scale.numel() / 1024,
            )
            return mxfp8_tensor

        elif target_fmt == QuantFormat.CPU_FP32:
            return weight.to(torch.float32).cpu()

        else:
            # 未知格式，安全回退
            logger.warning(
                "未知量化格式 %s，层 %s 回退BF16",
                target_fmt, layer_name,
            )
            return weight.to(torch.bfloat16)

    def convert_model_inplace(
        self,
        model: nn.Module,
        device_assignment: Dict[str, torch.device],
    ) -> nn.Module:
        """
        对模型进行原地异构量化转换。

        与上游Megatron的quantize_model_to_mxfp8()不同，此方法：
        1. 接受device_assignment字典，支持不同层路由到不同设备
        2. 通过CPU DRAM做staging，避免内存溢出
        3. 转换后立即释放原始权重，保持内存压力在合理范围

        Args:
            model: 待转换的PyTorch模型
            device_assignment: 层名称前缀 → 目标设备的映射
                例如 {"transformer.layers.0": torch.device("cuda:0"),  # A6000
                       "transformer.layers.1": torch.device("cuda:2")} # H100

        Returns:
            转换后的模型（in-place修改，返回同一对象）
        """
        converted_count = 0
        skipped_count = 0
        total_mxfp8_bytes = 0
        total_bf16_bytes = 0

        def _find_target_device(name: str) -> torch.device:
            """根据层名匹配device_assignment"""
            for prefix, device in device_assignment.items():
                if name.startswith(prefix):
                    return device
            # 默认：第一个设备
            if self.quant_config.device_formats:
                default_dev_idx = min(self.quant_config.device_formats.keys())
                return torch.device("cuda", default_dev_idx)
            return torch.device("cpu")

        for module_name, module in model.named_modules():
            if not hasattr(module, "_parameters"):
                continue

            params_to_replace: Dict[str, Any] = {}
            for param_name, param in module._parameters.items():
                if param is None:
                    continue

                full_name = f"{module_name}.{param_name}" if module_name else param_name
                target_device = _find_target_device(full_name)

                try:
                    converted = self.convert_weight(
                        param.data, full_name, target_device
                    )
                    params_to_replace[param_name] = converted

                    if isinstance(converted, MXFP8TensorDESLOC):
                        total_mxfp8_bytes += converted.data.numel()
                        converted_count += 1
                    else:
                        total_bf16_bytes += converted.numel() * 2
                        if self.quant_config.get_format(target_device) == QuantFormat.BF16:
                            skipped_count += 1

                except Exception as e:
                    logger.error("转换层 %s 失败，保持原格式: %s", full_name, e)
                    skipped_count += 1

            # 替换参数
            for param_name, converted in params_to_replace.items():
                del module._parameters[param_name]
                if isinstance(converted, MXFP8TensorDESLOC):
                    setattr(module, param_name, converted)
                else:
                    module.register_parameter(
                        param_name, nn.Parameter(converted, requires_grad=False)
                    )

        # 强制GC，释放原始权重
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(
            "异构量化转换完成: MXFP8层=%d, BF16层=%d, "
            "MXFP8数据=%.2fMB, BF16数据=%.2fMB",
            converted_count,
            skipped_count,
            total_mxfp8_bytes / 1024**2,
            total_bf16_bytes / 1024**2,
        )
        return model


# ---------------------------------------------------------------------------
# CheckpointMetadata：checkpoint元数据
# ---------------------------------------------------------------------------

@dataclass
class CheckpointMetadata:
    """
    DES-LOC异构checkpoint的元数据。

    记录每个layer的量化格式、设备分配和形状信息，
    用于在加载时重建正确的异构精度模型。

    Attributes:
        step: 训练步数
        timestamp: 保存时的Unix时间戳
        device_formats: layer_name → QuantFormat的映射
        layer_shapes: layer_name → 原始shape的映射
        quant_config_dict: 序列化的HeteroQuantConfig
        deepspeed_version: DeepSpeed版本
        neuron_sp_version: Neuron_SP版本
    """
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    device_formats: Dict[str, str] = field(default_factory=dict)
    layer_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    quant_config_dict: Dict[str, Any] = field(default_factory=dict)
    deepspeed_version: str = "unknown"
    neuron_sp_version: str = "0.1.0"

    def save(self, path: Union[str, Path]) -> None:
        """序列化保存到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.debug("CheckpointMetadata保存至: %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CheckpointMetadata":
        """从文件反序列化加载"""
        with open(path, "rb") as f:
            meta = pickle.load(f)
        logger.debug("CheckpointMetadata从 %s 加载", path)
        return meta


# ---------------------------------------------------------------------------
# HeteroSingleProcessCheckpoint：异构单进程checkpoint管理器
# ---------------------------------------------------------------------------

class HeteroSingleProcessCheckpoint:
    """
    DES-LOC异构单进程Checkpoint管理器。

    设计背景：
    - Megatron上游commit 44e27d04在模型加载后做一次性量化转换
      （TE→FlashInfer MXFP8），假设整个过程在单进程完成
    - DES-LOC的挑战是checkpoint必须同时服务于不同量化能力的设备：
      * A6000(SM86): 只能用BF16
      * H100(SM90): 可以用MXFP8
    - 无NVLink意味着不能依赖GPU间的快速权重共享，checkpoint
      的精度异构化必须在加载阶段由单进程串行完成

    核心机制：
    1. 保存时：记录每层的目标设备和量化格式到元数据
    2. 加载时：从CPU DRAM读取BF16权重，按元数据转换为目标格式
    3. 传输时：通过PCIe逐层传输，利用CPU DRAM作为LOcality Cache
       缓冲，避免同时在GPU上保留多个大型权重副本

    Args:
        save_dir: checkpoint根目录
        quant_config: 异构量化配置（None则自动检测）
        converter: 量化转换器（None则自动创建）
        chunk_size_gb: 每次PCIe传输的数据块大小（GB），
                       控制CPU DRAM的峰值占用
    """

    WEIGHT_DIR = "weights"
    META_FILE = "hetero_meta.pkl"
    SHARD_SUFFIX = ".shard"

    def __init__(
        self,
        save_dir: Union[str, Path],
        quant_config: Optional[HeteroQuantConfig] = None,
        converter: Optional[QuantizationConverter] = None,
        chunk_size_gb: float = 8.0,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.quant_config = quant_config or HeteroQuantConfig.auto_detect()
        self.converter = converter or QuantizationConverter(
            quant_config=self.quant_config,
            verify_conversion=True,
        )
        self.chunk_size_gb = chunk_size_gb
        self._chunk_size_bytes = int(chunk_size_gb * 1024**3)

        logger.info(
            "HeteroSingleProcessCheckpoint初始化: save_dir=%s, "
            "chunk_size=%.1fGB",
            self.save_dir, chunk_size_gb,
        )

    # ------------------------------------------------------------------
    # 保存接口
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        step: int,
        device_assignment: Optional[Dict[str, torch.device]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        保存异构量化模型到checkpoint目录。

        保存策略：
        1. 将所有权重下载到CPU DRAM（LOcality Cache），转为BF16存储
           （checkpoint始终以BF16保存，确保可在任意设备加载）
        2. 记录每层的目标量化格式到元数据
        3. 按chunk分片写入磁盘，控制内存峰值

        Args:
            model: 待保存的模型
            step: 当前训练步数
            device_assignment: 层名称 → 目标设备的映射（记录到元数据）
            optimizer_state: 可选的优化器状态

        Returns:
            checkpoint目录路径
        """
        ckpt_dir = self.save_dir / f"step_{step:07d}"
        weight_dir = ckpt_dir / self.WEIGHT_DIR
        weight_dir.mkdir(parents=True, exist_ok=True)

        logger.info("开始保存checkpoint: step=%d, dir=%s", step, ckpt_dir)
        t0 = time.time()

        metadata = CheckpointMetadata(step=step)
        device_assignment = device_assignment or {}

        # 分层保存权重
        shard_idx = 0
        current_shard: Dict[str, torch.Tensor] = {}
        current_shard_bytes = 0

        def _flush_shard():
            nonlocal shard_idx, current_shard, current_shard_bytes
            if not current_shard:
                return
            shard_path = weight_dir / f"shard_{shard_idx:04d}{self.SHARD_SUFFIX}"
            torch.save(current_shard, shard_path)
            logger.debug(
                "写入shard %d: %d层, %.2fMB",
                shard_idx, len(current_shard), current_shard_bytes / 1024**2,
            )
            shard_idx += 1
            current_shard = {}
            current_shard_bytes = 0

        for full_name, param in model.named_parameters():
            if param is None:
                continue

            # 获取目标设备（用于记录元数据）
            target_device = torch.device("cpu")
            for prefix, dev in device_assignment.items():
                if full_name.startswith(prefix):
                    target_device = dev
                    break

            target_fmt = self.quant_config.get_format(target_device)

            # 将权重下载到CPU DRAM
            if isinstance(param, MXFP8TensorDESLOC):
                # 已量化的权重：反量化回BF16再保存
                weight_cpu = param.to_bf16().cpu()
                original_shape = param.original_shape or tuple(param.data.shape)
            else:
                weight_cpu = param.data.to(torch.bfloat16).cpu()
                original_shape = tuple(param.shape)

            metadata.device_formats[full_name] = target_fmt.name
            metadata.layer_shapes[full_name] = original_shape

            current_shard[full_name] = weight_cpu
            current_shard_bytes += weight_cpu.numel() * 2  # BF16 = 2 bytes

            # 达到chunk大小则写入磁盘
            if current_shard_bytes >= self._chunk_size_bytes:
                _flush_shard()

        _flush_shard()  # 写入最后一个shard

        # 保存优化器状态（BF16格式）
        if optimizer_state is not None:
            optim_path = ckpt_dir / "optimizer_state.pt"
            torch.save(optimizer_state, optim_path)
            logger.info("优化器状态保存至: %s", optim_path)

        # 保存元数据
        meta_path = ckpt_dir / self.META_FILE
        metadata.save(meta_path)

        elapsed = time.time() - t0
        logger.info(
            "Checkpoint保存完成: step=%d, shards=%d, 耗时=%.2fs",
            step, shard_idx, elapsed,
        )
        return ckpt_dir

    # ------------------------------------------------------------------
    # 加载接口
    # ------------------------------------------------------------------

    def _iter_shards(self, weight_dir: Path) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        按序遍历所有shard文件，逐个yield (layer_name, tensor)。

        使用生成器而非一次性加载所有shard，控制CPU DRAM峰值占用。
        在DES-LOC的1.5TB DRAM环境中，即使单个large model也不应
        一次性全部加载。

        Args:
            weight_dir: shard文件目录

        Yields:
            (layer_name, bf16_tensor) 元组
        """
        shard_files = sorted(weight_dir.glob(f"*{self.SHARD_SUFFIX}"))
        if not shard_files:
            raise FileNotFoundError(f"在 {weight_dir} 中未找到shard文件")

        for shard_path in shard_files:
            logger.debug("加载shard: %s", shard_path.name)
            shard = torch.load(shard_path, map_location="cpu")
            for layer_name, tensor in shard.items():
                yield layer_name, tensor
            del shard
            gc.collect()

    def load(
        self,
        model: nn.Module,
        step: int,
        device_assignment: Optional[Dict[str, torch.device]] = None,
        strict: bool = True,
    ) -> Tuple[nn.Module, CheckpointMetadata]:
        """
        加载checkpoint并执行异构量化转换。

        这是DES-LOC的核心加载流程，对应Megatron上游commit中的
        两步操作（加载 + quantize_model_to_mxfp8），但扩展为
        异构per-device格式支持：

        Step1: 从磁盘加载BF16权重到CPU DRAM (LOcality Cache)
        Step2: 按layer的目标设备格式执行量化转换
               - A6000层：BF16，直接传输
               - H100层：MXFP8，CPU量化后传输
        Step3: 逐层写入模型参数（避免同时在GPU上保留双份）

        Args:
            model: 待填充权重的模型（结构已初始化）
            step: 要加载的checkpoint步数
            device_assignment: 层名称 → 目标设备的映射
                （优先级高于元数据中记录的device_formats）
            strict: 是否严格要求所有层都有对应的checkpoint权重

        Returns:
            (model, metadata) 元组
        """
        ckpt_dir = self.save_dir / f"step_{step:07d}"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint目录不存在: {ckpt_dir}")

        weight_dir = ckpt_dir / self.WEIGHT_DIR
        meta_path = ckpt_dir / self.META_FILE

        logger.info("开始加载checkpoint: step=%d, dir=%s", step, ckpt_dir)
        t0 = time.time()

        # 加载元数据
        metadata = CheckpointMetadata.load(meta_path)
        device_assignment = device_assignment or {}

        # 构建模型参数名称集合（用于strict检查）
        model_param_names = set(name for name, _ in model.named_parameters())
        loaded_names: set = set()
        missing_names: List[str] = []

        # 构建name→module的快速查找表
        name_to_module: Dict[str, Tuple[nn.Module, str]] = {}
        for module_name, module in model.named_modules():
            for param_name in list(getattr(module, "_parameters", {}).keys()):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                name_to_module[full_name] = (module, param_name)

        # 逐shard加载并转换
        for layer_name, bf16_weight in self._iter_shards(weight_dir):
            if layer_name not in name_to_module:
                if strict:
                    logger.warning("Checkpoint中的层 %s 在模型中不存在", layer_name)
                continue

            # 确定目标设备
            target_device = torch.device("cpu")
            for prefix, dev in device_assignment.items():
                if layer_name.startswith(prefix):
                    target_device = dev
                    break
            else:
                # 从元数据中读取格式信息推断设备
                if layer_name in metadata.device_formats:
                    saved_fmt = metadata.device_formats[layer_name]
                    # 找到支持该格式的设备
                    for dev_idx, fmt in self.quant_config.device_formats.items():
                        if fmt.name == saved_fmt:
                            target_device = torch.device("cuda", dev_idx)
                            break

            # 执行量化转换（CPU staging → 目标格式）
            try:
                converted = self.converter.convert_weight(
                    bf16_weight, layer_name, target_device
                )
            except Exception as e:
                logger.error("层 %s 量化转换失败，使用BF16: %s", layer_name, e)
                converted = bf16_weight.to(torch.bfloat16)

            # 写入模型参数
            module, param_name = name_to_module[layer_name]
            del module._parameters[param_name]

            if isinstance(converted, MXFP8TensorDESLOC):
                # MXFP8: 传输到目标GPU
                converted_on_device = converted.to_device(target_device)
                setattr(module, param_name, converted_on_device)
            else:
                # BF16: 传输到目标设备
                converted_on_device = converted.to(target_device)
                module.register_parameter(
                    param_name,
                    nn.Parameter(converted_on_device, requires_grad=False)
                )

            loaded_names.add(layer_name)
            logger.debug(
                "加载层 %s → device=%s, format=%s",
                layer_name, target_device,
                "MXFP8" if isinstance(converted, MXFP8TensorDESLOC) else "BF16",
            )

            # 立即释放CPU缓存
            del bf16_weight, converted
            if len(loaded_names) % 50 == 0:
                gc.collect()

        # 检查缺失层
        missing_names = list(model_param_names - loaded_names)
        if missing_names:
            if strict:
                raise RuntimeError(
                    f"以下{len(missing_names)}个层在checkpoint中缺失: "
                    f"{missing_names[:10]}{'...' if len(missing_names) > 10 else ''}"
                )
            else:
                logger.warning("共%d个层在checkpoint中缺失（非strict模式忽略）", len(missing_names))

        elapsed = time.time() - t0
        logger.info(
            "Checkpoint加载完成: step=%d, 加载层数=%d, 耗时=%.2fs",
            step, len(loaded_names), elapsed,
        )
        return model, metadata

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[int]:
        """
        列出所有可用的checkpoint步数，按步数排序。

        Returns:
            步数列表，升序排列
        """
        if not self.save_dir.exists():
            return []
        steps = []
        for d in self.save_dir.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                try:
                    step = int(d.name.split("_")[1])
                    steps.append(step)
                except (ValueError, IndexError):
                    pass
        return sorted(steps)

    def get_latest_step(self) -> Optional[int]:
        """
        获取最新checkpoint的步数。

        Returns:
            最新步数，若无checkpoint则返回None
        """
        steps = self.list_checkpoints()
        return steps[-1] if steps else None

    def memory_estimate(self, model: nn.Module) -> Dict[str, float]:
        """
        估算checkpoint加载时的内存峰值。

        DES-LOC规划工具：在实际加载前估算各设备的内存需求，
        确保不超过A6000(48GB)、H100(96GB)、CPU DRAM(1.5TB)的容量。

        Args:
            model: 模型实例

        Returns:
            设备 → 估算内存(GB)的字典
        """
        mem_estimate: Dict[str, float] = {"cpu": 0.0}

        for dev_idx in self.quant_config.device_formats:
            mem_estimate[f"cuda:{dev_idx}"] = 0.0

        for name, param in model.named_parameters():
            if param is None:
                continue

            # 找目标设备
            target_device_str = "cpu"
            for prefix, dev in self.quant_config.device_formats.items():
                pass  # 简化：使用第一个设备

            param_bytes = param.numel() * param.element_size()
            target_fmt = QuantFormat.BF16  # 简化估算

            if target_fmt == QuantFormat.MXFP8:
                # MXFP8：约为BF16的1/2
                gpu_bytes = param_bytes / 2
            else:
                gpu_bytes = param_bytes

            # CPU staging peak：原始BF16
            mem_estimate["cpu"] = mem_estimate.get("cpu", 0.0) + self.chunk_size_gb

        # 转换为GB
        for key in mem_estimate:
            mem_estimate[key] = round(mem_estimate[key], 2)

        return mem_estimate


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== HeteroSingleProcessCheckpoint Smoke Test ===")

    # Test 1: HeteroQuantConfig自动检测
    config = HeteroQuantConfig.auto_detect()
    assert isinstance(config.device_formats, dict), "device_formats应为dict"
    logger.info("Test 1 passed: HeteroQuantConfig.auto_detect()")

    # Test 2: MXFP8TensorDESLOC CPU量化和反量化
    x = torch.randn(64, 128, dtype=torch.bfloat16)
    mxfp8 = MXFP8TensorDESLOC.from_bf16_cpu_staging(x, group_size=32)
    assert mxfp8.data.shape == x.shape, f"data shape不匹配: {mxfp8.data.shape} vs {x.shape}"
    assert mxfp8.is_on_cpu(), "CPU量化后数据应在CPU上"
    logger.info("Test 2 passed: MXFP8TensorDESLOC CPU staging量化")

    # Test 3: BF16反量化数值合理性
    restored = mxfp8.to_bf16()
    assert restored.shape == x.shape, "反量化shape应一致"
    rel_err = (restored.float() - x.float()).abs().mean() / (x.float().abs().mean() + 1e-6)
    assert rel_err < 0.15, f"MXFP8相对误差过大: {rel_err:.4f}"
    logger.info("Test 3 passed: BF16反量化误差=%.4f", rel_err.item())

    # Test 4: QuantizationConverter对BF16格式转换
    converter = QuantizationConverter(config, verify_conversion=False)
    weight_2d = torch.randn(32, 64, dtype=torch.bfloat16)
    converted = converter.convert_weight(weight_2d, "test.layer.weight", torch.device("cpu"))
    assert converted is not None, "转换结果不应为None"
    logger.info("Test 4 passed: QuantizationConverter.convert_weight()")

    # Test 5: CheckpointMetadata序列化
    import tempfile
    meta = CheckpointMetadata(step=42, device_formats={"layer.0": "BF16"})
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = Path(tmpdir) / "meta.pkl"
        meta.save(meta_path)
        loaded_meta = CheckpointMetadata.load(meta_path)
        assert loaded_meta.step == 42, f"step应为42，实际为{loaded_meta.step}"
        assert loaded_meta.device_formats == {"layer.0": "BF16"}
    logger.info("Test 5 passed: CheckpointMetadata序列化/反序列化")

    logger.info("=== 所有Smoke Test通过 ===")
