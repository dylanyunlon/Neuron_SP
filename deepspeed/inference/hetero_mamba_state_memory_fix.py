"""
DES-LOC异构训练框架 — 混合Mamba/Attention模型推理上下文内存管理

上游设计意图 (Megatron dcc6d62):
    Megatron原始commit修复了当 max_requests 设置时Mamba状态内存过度分配的问题。
    核心问题是：旧代码总是按照 mamba_memory_ratio 比例切分内存，即使用户已经
    明确指定了最大并发请求数，导致 Mamba 状态缓冲区分配远超实际需要。
    修复引入了 max_requests 感知的精确内存计算路径：
      1. 按 max_requests 精确计算 Mamba 状态实际所需内存
      2. 剩余内存全部归还给 KV cache 块
      3. 新增 BlockOverflowError，防止单请求序列超过总块数
      4. 增强初始化日志，提供完整内存分配摘要

DES-LOC 适配点:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) 将计算与内存
    管理解耦，支持在异构硬件 (2x A6000 48GB SM86 + 1x H100 NVL 96GB SM90)
    上的联合推理。适配工作包括：

    1. HeteroMemoryRouter: 将 KV cache 块和 Mamba 状态路由到不同设备
       - H100 NVL 96GB: 主 KV cache 热块 (高带宽优先)
       - A6000 48GB x2: Mamba SSM/Conv 状态 + KV cache 冷块
       - CPU DRAM 1.5TB: 换页缓冲区 (paused buffer 对应物)

    2. SharedLocalityCache: 跨设备共享的局部性感知缓存索引
       - PCIe 互联下的块迁移策略（无 NVLink，需显式管理带宽）
       - 前缀缓存在异构设备间的一致性维护

    3. 精确 Mamba 内存计算被扩展为设备感知版本
       - 每个设备独立计算可容纳的 mamba_max_requests
       - 跨设备汇总得到全局 max_requests 上界

    4. BlockOverflowError 扩展为 HeteroBlockOverflowError
       - 区分"本地设备块不足"和"全局块不足"两种情形

硬件拓扑:
    H100 NVL (96GB, SM90) <--PCIe--> A6000_0 (48GB, SM86)
                          <--PCIe--> A6000_1 (48GB, SM86)
    所有设备 <--PCIe/QPI--> CPU DRAM (1.5TB)
"""

from __future__ import annotations

import logging
import math
from dataclasses import InitVar, dataclass, field
from enum import Enum, auto
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量与枚举
# ---------------------------------------------------------------------------

# 硬件拓扑常量（匹配目标机器）
_H100_NVL_DEVICE_IDX: int = 0      # H100 NVL 96GB，SM90
_A6000_0_DEVICE_IDX: int = 1       # A6000 48GB，SM86
_A6000_1_DEVICE_IDX: int = 2       # A6000 48GB，SM86
_CPU_DEVICE: str = "cpu"

# PCIe 带宽估算 (GB/s)，用于迁移代价计算
_PCIE_BW_GBPS: float = 16.0        # PCIe Gen4 x16 单向峰值
_PCIE_LATENCY_US: float = 5.0      # 基础延迟 μs


class DeviceRole(Enum):
    """DES-LOC 中每个设备承担的角色。"""
    KV_HOT    = auto()   # 热 KV 块（高频访问，放 H100）
    KV_COLD   = auto()   # 冷 KV 块（低频访问，放 A6000）
    MAMBA_SSM = auto()   # Mamba SSM 状态（放 A6000，节约 H100 带宽）
    CPU_SWAP  = auto()   # CPU DRAM 换页缓冲区


class MemoryBudgetResult(NamedTuple):
    """内存分配结果，对应 Megatron 修复中的分配计算输出。"""
    kv_active_bytes: int            # 活跃 KV cache 字节数
    kv_paused_bytes: int            # 暂停 KV cache 字节数（CPU swap）
    mamba_bytes: int                # Mamba 状态总字节数
    mamba_max_requests: int         # 实际可服务的 Mamba 最大并发请求数
    kv_active_blocks: int           # 活跃 KV 块数
    kv_paused_blocks: int           # 暂停 KV 块数
    device_assignments: Dict[str, DeviceRole]  # 每个分配区域的设备归属


# ---------------------------------------------------------------------------
# 异常类
# ---------------------------------------------------------------------------

class HeteroBlockOverflowError(Exception):
    """
    单个请求所需 KV 块数超过可用总块数。

    对应 Megatron dcc6d62 引入的 BlockOverflowError，扩展为异构场景：
    区分"本地设备（H100/A6000）块不足"和"全局块（含 CPU swap）不足"。
    """

    def __init__(
        self,
        request_id: int,
        required_blocks: int,
        available_local: int,
        available_global: int,
        can_swap_to_cpu: bool = False,
    ):
        self.request_id = request_id
        self.required_blocks = required_blocks
        self.available_local = available_local
        self.available_global = available_global
        self.can_swap_to_cpu = can_swap_to_cpu
        local_or_global = "local" if not can_swap_to_cpu else "global (including CPU swap)"
        super().__init__(
            f"Request {request_id}: needs {required_blocks} blocks, "
            f"but only {available_local} local GPU blocks available "
            f"({available_global} global incl. CPU). "
            f"Overflow is {'recoverable via CPU swap' if can_swap_to_cpu else 'fatal'}."
        )


class MambaMemoryInsufficientError(Exception):
    """Mamba 状态内存不足以容纳请求的并发数。"""

    def __init__(self, max_requests: int, needed_bytes: int, total_bytes: int):
        super().__init__(
            f"Cannot allocate Mamba states for {max_requests} requests: "
            f"need {_fmt_bytes(needed_bytes)}, total buffer is {_fmt_bytes(total_bytes)}."
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _fmt_bytes(n_bytes: int) -> str:
    """
    将字节数转换为人类可读字符串。

    对应 Megatron get_mem_size_str，修复了原始实现中 n_bytes==0 时
    陷入无意义循环的 bug（dcc6d62 同批修复）。
    """
    if n_bytes == 0:
        return "0 bytes"
    for exp, suffix in ((4, "TB"), (3, "GB"), (2, "MB"), (1, "KB"), (0, "bytes")):
        nquery = int(1024 ** exp)
        if round(n_bytes / nquery) >= 1:
            return f"{n_bytes / nquery:.2f} {suffix}"
    return f"{n_bytes} bytes"


def _compute_block_size_bytes(
    num_heads: int,
    head_dim: int,
    block_size_tokens: int,
    dtype: torch.dtype,
    kv_factor: int = 2,   # K + V
) -> int:
    """
    计算单个 KV cache 块的字节大小。

    参数:
        num_heads: 注意力头数
        head_dim: 每个头的维度 (kv_channels)
        block_size_tokens: 每块包含的 token 数
        dtype: KV cache 数据类型
        kv_factor: K/V 对数，默认 2
    """
    elem_bytes = torch.empty(0, dtype=dtype).element_size()
    return kv_factor * block_size_tokens * num_heads * head_dim * elem_bytes


def _pcie_transfer_time_us(size_bytes: int) -> float:
    """估算 PCIe 传输给定字节数所需时间（μs）。"""
    bw_bytes_per_us = (_PCIE_BW_GBPS * 1024**3) / 1e6
    return _PCIE_LATENCY_US + size_bytes / bw_bytes_per_us


# ---------------------------------------------------------------------------
# Mamba 状态配置
# ---------------------------------------------------------------------------

@dataclass
class MambaStateConfig:
    """
    Mamba 层状态的形状与类型配置。

    对应 Megatron MambaInferenceStateConfig，适配 DES-LOC 异构内存路由需求，
    新增 device_preference 字段以指导状态放置策略。
    """
    conv_state_shape: Tuple[int, ...]    # (d_inner, d_conv-1) 或类似形状
    ssm_state_shape: Tuple[int, ...]     # (num_heads, head_dim, d_state) 或类似
    conv_dtype: torch.dtype = torch.float16
    ssm_dtype: torch.dtype = torch.float16
    num_mamba_layers: int = 1
    device_preference: DeviceRole = DeviceRole.MAMBA_SSM  # 默认放 A6000

    @property
    def conv_bytes_per_request(self) -> int:
        """单请求 conv 状态字节数（所有 Mamba 层）。"""
        elem = torch.empty(0, dtype=self.conv_dtype).element_size()
        return math.prod(self.conv_state_shape) * elem * self.num_mamba_layers

    @property
    def ssm_bytes_per_request(self) -> int:
        """单请求 SSM 状态字节数（所有 Mamba 层）。"""
        elem = torch.empty(0, dtype=self.ssm_dtype).element_size()
        return math.prod(self.ssm_state_shape) * elem * self.num_mamba_layers

    @property
    def total_bytes_per_request(self) -> int:
        """单请求 Mamba 状态总字节数。"""
        return self.conv_bytes_per_request + self.ssm_bytes_per_request


# ---------------------------------------------------------------------------
# 异构设备内存视图
# ---------------------------------------------------------------------------

@dataclass
class HeteroDeviceMemoryView:
    """
    DES-LOC 异构设备内存视图。

    在 PCIe 互联、无 NVLink 的拓扑下，为每个设备维护独立的内存预算，
    并通过 SharedLocalityCache 索引实现跨设备块的逻辑统一。

    设备布局（固定拓扑）:
        device 0 (H100 NVL 96GB):  hot KV blocks
        device 1 (A6000 48GB):     Mamba states + cold KV blocks
        device 2 (A6000 48GB):     Mamba states + cold KV blocks
        CPU:                       paused/swap KV blocks
    """
    # 总容量（字节）
    h100_total_bytes: int = int(96 * 1024**3)
    a6000_0_total_bytes: int = int(48 * 1024**3)
    a6000_1_total_bytes: int = int(48 * 1024**3)
    cpu_total_bytes: int = int(1536 * 1024**3)   # 1.5TB

    # 保留比例（用于系统/模型权重）
    gpu_reserve_ratio: float = 0.15   # 预留 15% 给模型权重等
    cpu_reserve_ratio: float = 0.05   # CPU 预留 5%

    @property
    def h100_kv_budget_bytes(self) -> int:
        """H100 可用于 KV cache 的字节预算。"""
        return int(self.h100_total_bytes * (1.0 - self.gpu_reserve_ratio))

    @property
    def a6000_kv_budget_bytes(self) -> int:
        """两块 A6000 合计可用于 Mamba + 冷 KV 的字节预算。"""
        a0 = int(self.a6000_0_total_bytes * (1.0 - self.gpu_reserve_ratio))
        a1 = int(self.a6000_1_total_bytes * (1.0 - self.gpu_reserve_ratio))
        return a0 + a1

    @property
    def cpu_swap_budget_bytes(self) -> int:
        """CPU DRAM 可用于换页的字节预算。"""
        return int(self.cpu_total_bytes * (1.0 - self.cpu_reserve_ratio))

    @property
    def total_gpu_budget_bytes(self) -> int:
        return self.h100_kv_budget_bytes + self.a6000_kv_budget_bytes


# ---------------------------------------------------------------------------
# 核心：异构 Mamba 内存分配器
# ---------------------------------------------------------------------------

class HeteroMambaStateMemoryAllocator:
    """
    DES-LOC 异构 Mamba 状态内存分配器。

    核心算法（对应并扩展 Megatron dcc6d62 的 max_requests 感知分配路径）:

    原始 Megatron 逻辑（单设备）:
        if mamba_memory_ratio is set:
            # 旧路径：固定比例切分（over-provision 问题根源）
            mamba_bytes = total * ratio
        elif max_requests is set:
            # 新路径 (dcc6d62 修复)：精确计算
            mamba_bytes = max_requests * bytes_per_request
            ratio = mamba_bytes / total
            kv_bytes = total * (1 - ratio)

    DES-LOC 扩展（多设备）:
        1. 将 Mamba 状态优先放置在 A6000（释放 H100 带宽给 KV 热块）
        2. 若 A6000 不足，溢出到 H100（但记录带宽压力警告）
        3. KV 热块放 H100，冷块放 A6000 剩余空间
        4. CPU DRAM 作为 paused buffer（对应 Megatron 的 paused_buffer）
        5. 精确计算每设备可服务的 mamba_max_requests，取交集

    SharedLocalityCache 集成:
        - 前缀缓存索引在 H100 上维护（访问延迟最低）
        - 实际 Mamba 状态块可在 A6000 或 H100 上，通过索引间接访问
        - 块迁移由 PCIe 带宽模型指导（_pcie_transfer_time_us）
    """

    def __init__(
        self,
        mamba_config: MambaStateConfig,
        hetero_view: HeteroDeviceMemoryView,
        block_size_tokens: int,
        num_heads: int,
        head_dim: int,
        kv_dtype: torch.dtype = torch.float16,
        num_speculative_tokens: int = 0,
        enable_prefix_caching: bool = False,
        prefix_cache_budget_bytes: int = 0,
        verbose: bool = False,
    ):
        self.mamba_config = mamba_config
        self.hetero_view = hetero_view
        self.block_size_tokens = block_size_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_dtype = kv_dtype
        self.num_speculative_tokens = num_speculative_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_cache_budget_bytes = prefix_cache_budget_bytes
        self.verbose = verbose

        self._block_size_bytes = _compute_block_size_bytes(
            num_heads, head_dim, block_size_tokens, kv_dtype
        )
        logger.debug(
            "HeteroMambaStateMemoryAllocator init: block_size=%s, "
            "mamba_per_req=%s, H100_budget=%s, A6000_budget=%s, CPU_budget=%s",
            _fmt_bytes(self._block_size_bytes),
            _fmt_bytes(mamba_config.total_bytes_per_request),
            _fmt_bytes(hetero_view.h100_kv_budget_bytes),
            _fmt_bytes(hetero_view.a6000_kv_budget_bytes),
            _fmt_bytes(hetero_view.cpu_swap_budget_bytes),
        )

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def compute_budget(
        self,
        max_requests: Optional[int] = None,
        mamba_memory_ratio: Optional[float] = None,
    ) -> MemoryBudgetResult:
        """
        计算异构内存分配预算。

        这是 dcc6d62 修复逻辑在 DES-LOC 框架下的主入口。
        优先使用 max_requests 精确路径（避免过度分配），
        回退到 ratio 路径（向后兼容）。

        参数:
            max_requests: 最大并发请求数（精确路径，优先）
            mamba_memory_ratio: Mamba 内存占比（比例路径，回退）

        返回:
            MemoryBudgetResult: 各分区的字节数和块数

        异常:
            MambaMemoryInsufficientError: 总 GPU 内存不足以容纳 max_requests
            ValueError: 参数非法
        """
        if max_requests is not None and mamba_memory_ratio is not None:
            logger.warning(
                "Both max_requests=%d and mamba_memory_ratio=%.3f are set. "
                "max_requests takes precedence (DES-LOC precise path).",
                max_requests, mamba_memory_ratio,
            )

        if max_requests is not None:
            result = self._compute_budget_precise(max_requests)
        elif mamba_memory_ratio is not None:
            result = self._compute_budget_ratio(mamba_memory_ratio)
        else:
            # 纯 attention 模型或无 Mamba：全量分配给 KV
            result = self._compute_budget_kv_only()

        if self.verbose:
            self._log_budget_summary(result)

        return result

    def validate_request_blocks(
        self,
        request_id: int,
        prompt_len: int,
        max_gen_tokens: int,
        budget: MemoryBudgetResult,
    ) -> None:
        """
        验证单个请求所需 KV 块数不超过可用总量。

        对应 Megatron dcc6d62 在 DynamicInferenceEngine._add_request 中引入的
        BlockOverflowError 检查，扩展为异构场景（区分本地/全局溢出）。

        参数:
            request_id: 请求 ID
            prompt_len: prompt token 数
            max_gen_tokens: 最大生成 token 数
            budget: 已计算的内存预算

        异常:
            HeteroBlockOverflowError: 块数不足
        """
        max_tokens = prompt_len + max_gen_tokens
        required_blocks = math.ceil(max_tokens / self.block_size_tokens)

        # 本地 GPU 可用块（热 + 冷，减去 dummy 块）
        local_blocks = budget.kv_active_blocks - 1  # -1 for dummy block
        # 全局可用块（含 CPU swap）
        global_blocks = local_blocks + budget.kv_paused_blocks

        if required_blocks > global_blocks:
            raise HeteroBlockOverflowError(
                request_id=request_id,
                required_blocks=required_blocks,
                available_local=local_blocks,
                available_global=global_blocks,
                can_swap_to_cpu=False,
            )
        elif required_blocks > local_blocks:
            # 本地不足但全局够，可以通过 CPU swap 恢复
            logger.info(
                "Request %d: needs %d blocks, only %d local GPU blocks available. "
                "Will use CPU swap (%d paused blocks). PCIe transfer est: %s μs",
                request_id, required_blocks, local_blocks, budget.kv_paused_blocks,
                f"{_pcie_transfer_time_us(required_blocks * self._block_size_bytes):.1f}",
            )
            raise HeteroBlockOverflowError(
                request_id=request_id,
                required_blocks=required_blocks,
                available_local=local_blocks,
                available_global=global_blocks,
                can_swap_to_cpu=True,
            )

    # ------------------------------------------------------------------
    # 内部：精确路径（dcc6d62 核心修复的 DES-LOC 扩展）
    # ------------------------------------------------------------------

    def _compute_budget_precise(self, max_requests: int) -> MemoryBudgetResult:
        """
        max_requests 感知的精确内存分配（DES-LOC 版本）。

        算法步骤:
          1. 计算 max_requests 所需的 Mamba 状态总内存
          2. 优先从 A6000 扣除（保护 H100 带宽）
          3. 若 A6000 不足，从 H100 补充
          4. KV 热块 = H100 剩余空间
          5. KV 冷块 = A6000 剩余空间
          6. CPU swap = CPU DRAM 预算

        对应 Megatron 原始修复（单设备版）:
            mamba_memory_needed = max_requests * mamba_states_memory_per_request
            mamba_memory_ratio = mamba_memory_needed / total_memory
            buffer_size_bytes = int(buffer_size_bytes * (1.0 - mamba_memory_ratio))
        """
        mamba_per_req = self.mamba_config.total_bytes_per_request
        # 投机解码时需要额外的 Mamba 状态副本
        if self.num_speculative_tokens > 0:
            spec_multiplier = self.num_speculative_tokens + 1
            mamba_per_req_with_spec = mamba_per_req * spec_multiplier
        else:
            mamba_per_req_with_spec = mamba_per_req

        mamba_total_needed = max_requests * mamba_per_req_with_spec

        # 前缀缓存 Mamba 槽
        if self.enable_prefix_caching and self.prefix_cache_budget_bytes > 0:
            prefix_mamba_slots = self.prefix_cache_budget_bytes // mamba_per_req
            mamba_total_needed += prefix_mamba_slots * mamba_per_req
            logger.debug(
                "Prefix caching: reserving %d Mamba slots (%s) on A6000.",
                prefix_mamba_slots, _fmt_bytes(prefix_mamba_slots * mamba_per_req),
            )
        else:
            prefix_mamba_slots = 0

        total_gpu_budget = self.hetero_view.total_gpu_budget_bytes
        if mamba_total_needed >= total_gpu_budget:
            raise MambaMemoryInsufficientError(max_requests, mamba_total_needed, total_gpu_budget)

        # --- 从 A6000 优先扣除 Mamba 内存 ---
        a6000_budget = self.hetero_view.a6000_kv_budget_bytes
        h100_budget = self.hetero_view.h100_kv_budget_bytes

        device_assignments: Dict[str, DeviceRole] = {}

        if mamba_total_needed <= a6000_budget:
            # A6000 足够容纳全部 Mamba 状态
            a6000_remaining = a6000_budget - mamba_total_needed
            h100_remaining = h100_budget
            device_assignments["mamba"] = DeviceRole.MAMBA_SSM
            logger.debug(
                "Mamba states fit entirely on A6000: %s used, %s remaining on A6000.",
                _fmt_bytes(mamba_total_needed), _fmt_bytes(a6000_remaining),
            )
        else:
            # A6000 不足，溢出到 H100
            overflow = mamba_total_needed - a6000_budget
            a6000_remaining = 0
            h100_remaining = max(0, h100_budget - overflow)
            device_assignments["mamba"] = DeviceRole.KV_HOT   # 实际混放
            logger.warning(
                "Mamba states overflow A6000 by %s, spilling %s onto H100. "
                "This may reduce KV cache hot block capacity.",
                _fmt_bytes(overflow), _fmt_bytes(overflow),
            )

        # --- KV cache 分配 ---
        # H100: 热块
        kv_active_bytes_h100 = h100_remaining
        device_assignments["kv_hot"] = DeviceRole.KV_HOT
        # A6000 剩余: 冷块
        kv_cold_bytes_a6000 = a6000_remaining
        device_assignments["kv_cold"] = DeviceRole.KV_COLD
        # CPU: swap 块
        kv_paused_bytes = self.hetero_view.cpu_swap_budget_bytes
        device_assignments["kv_paused"] = DeviceRole.CPU_SWAP

        kv_active_bytes = kv_active_bytes_h100 + kv_cold_bytes_a6000
        kv_active_blocks = max(2, kv_active_bytes // self._block_size_bytes)
        kv_paused_blocks = kv_paused_bytes // self._block_size_bytes

        logger.info(
            "DES-LOC precise allocation: mamba=%s (for %d reqs), "
            "kv_hot=%s (%d blocks on H100), kv_cold=%s, kv_swap=%s (%d CPU blocks)",
            _fmt_bytes(mamba_total_needed), max_requests,
            _fmt_bytes(kv_active_bytes_h100),
            kv_active_bytes_h100 // self._block_size_bytes,
            _fmt_bytes(kv_cold_bytes_a6000),
            _fmt_bytes(kv_paused_bytes), kv_paused_blocks,
        )

        return MemoryBudgetResult(
            kv_active_bytes=kv_active_bytes,
            kv_paused_bytes=kv_paused_bytes,
            mamba_bytes=mamba_total_needed,
            mamba_max_requests=max_requests,
            kv_active_blocks=kv_active_blocks,
            kv_paused_blocks=kv_paused_blocks,
            device_assignments=device_assignments,
        )

    def _compute_budget_ratio(self, mamba_memory_ratio: float) -> MemoryBudgetResult:
        """
        固定比例切分路径（向后兼容，对应 Megatron 旧有逻辑）。

        警告：此路径可能导致 over-provision，dcc6d62 的目的之一就是减少对此路径的依赖。
        在 DES-LOC 中保留用于无法预知 max_requests 的场景。
        """
        if not (0.0 <= mamba_memory_ratio <= 1.0):
            raise ValueError(
                f"mamba_memory_ratio must be in [0, 1], got {mamba_memory_ratio}"
            )

        total_gpu = self.hetero_view.total_gpu_budget_bytes
        mamba_bytes = int(total_gpu * mamba_memory_ratio)
        kv_bytes = total_gpu - mamba_bytes

        # 按 H100/A6000 比例分配 KV
        h100_ratio = self.hetero_view.h100_kv_budget_bytes / total_gpu
        kv_h100 = int(kv_bytes * h100_ratio)
        kv_a6000 = kv_bytes - kv_h100

        kv_active_bytes = kv_h100 + kv_a6000
        kv_paused_bytes = self.hetero_view.cpu_swap_budget_bytes
        kv_active_blocks = max(2, kv_active_bytes // self._block_size_bytes)
        kv_paused_blocks = kv_paused_bytes // self._block_size_bytes

        mamba_per_req = self.mamba_config.total_bytes_per_request
        mamba_max_requests = mamba_bytes // mamba_per_req if mamba_per_req > 0 else 0

        logger.warning(
            "DES-LOC ratio-based allocation (may over-provision): "
            "ratio=%.3f, mamba=%s, kv_active=%s, derived max_requests=%d",
            mamba_memory_ratio, _fmt_bytes(mamba_bytes),
            _fmt_bytes(kv_active_bytes), mamba_max_requests,
        )

        return MemoryBudgetResult(
            kv_active_bytes=kv_active_bytes,
            kv_paused_bytes=kv_paused_bytes,
            mamba_bytes=mamba_bytes,
            mamba_max_requests=mamba_max_requests,
            kv_active_blocks=kv_active_blocks,
            kv_paused_blocks=kv_paused_blocks,
            device_assignments={
                "mamba": DeviceRole.MAMBA_SSM,
                "kv_hot": DeviceRole.KV_HOT,
                "kv_cold": DeviceRole.KV_COLD,
                "kv_paused": DeviceRole.CPU_SWAP,
            },
        )

    def _compute_budget_kv_only(self) -> MemoryBudgetResult:
        """纯 Attention 模型（无 Mamba）：全量分配给 KV cache。"""
        h100_budget = self.hetero_view.h100_kv_budget_bytes
        a6000_budget = self.hetero_view.a6000_kv_budget_bytes
        cpu_budget = self.hetero_view.cpu_swap_budget_bytes

        kv_active_bytes = h100_budget + a6000_budget
        kv_active_blocks = max(2, kv_active_bytes // self._block_size_bytes)
        kv_paused_blocks = cpu_budget // self._block_size_bytes

        logger.info(
            "DES-LOC KV-only allocation: kv_active=%s (%d blocks), kv_swap=%s (%d blocks)",
            _fmt_bytes(kv_active_bytes), kv_active_blocks,
            _fmt_bytes(cpu_budget), kv_paused_blocks,
        )

        return MemoryBudgetResult(
            kv_active_bytes=kv_active_bytes,
            kv_paused_bytes=cpu_budget,
            mamba_bytes=0,
            mamba_max_requests=0,
            kv_active_blocks=kv_active_blocks,
            kv_paused_blocks=kv_paused_blocks,
            device_assignments={
                "kv_hot": DeviceRole.KV_HOT,
                "kv_cold": DeviceRole.KV_COLD,
                "kv_paused": DeviceRole.CPU_SWAP,
            },
        )

    # ------------------------------------------------------------------
    # 日志摘要（对应 Megatron dcc6d62 增强的 log_lines 输出）
    # ------------------------------------------------------------------

    def _log_budget_summary(self, result: MemoryBudgetResult) -> None:
        """
        输出完整的内存分配摘要日志。

        对应 Megatron dcc6d62 中大幅扩展的 log_lines 列表，
        DES-LOC 版本额外包含设备归属和 PCIe 传输代价估算。
        """
        mamba_cfg = self.mamba_config
        spec_mult = self.num_speculative_tokens + 1 if self.num_speculative_tokens > 0 else 1

        lines = [
            "=" * 60,
            "DES-LOC HeteroMambaStateMemoryAllocator: allocation summary",
            "=" * 60,
            f"  Hardware topology:",
            f"    H100 NVL (SM90):         {_fmt_bytes(self.hetero_view.h100_total_bytes)} total",
            f"    A6000 x2 (SM86):         {_fmt_bytes(self.hetero_view.a6000_0_total_bytes + self.hetero_view.a6000_1_total_bytes)} total",
            f"    CPU DRAM:                {_fmt_bytes(self.hetero_view.cpu_total_bytes)} total",
            f"  KV cache:",
            f"    block_size_bytes:        {_fmt_bytes(self._block_size_bytes)}",
            f"    active_blocks (GPU):     {result.kv_active_blocks} ({_fmt_bytes(result.kv_active_bytes)})",
            f"    paused_blocks (CPU):     {result.kv_paused_blocks} ({_fmt_bytes(result.kv_paused_bytes)})",
        ]

        if result.mamba_bytes > 0:
            lines += [
                f"  Mamba states ({mamba_cfg.device_preference.name}):",
                f"    num_mamba_layers:        {mamba_cfg.num_mamba_layers}",
                f"    conv_state_shape:        {mamba_cfg.conv_state_shape}",
                f"    ssm_state_shape:         {mamba_cfg.ssm_state_shape}",
                f"    per_request:             {_fmt_bytes(mamba_cfg.total_bytes_per_request)}",
                f"    max_requests:            {result.mamba_max_requests}",
                f"    total:                   {_fmt_bytes(result.mamba_bytes)}",
            ]
            if self.num_speculative_tokens > 0:
                per_req_spec = mamba_cfg.total_bytes_per_request * spec_mult
                lines += [
                    f"  Speculative decoding (n={self.num_speculative_tokens}):",
                    f"    per_request (x{spec_mult}):       {_fmt_bytes(per_req_spec)}",
                ]
            if self.enable_prefix_caching and self.prefix_cache_budget_bytes > 0:
                slots = self.prefix_cache_budget_bytes // mamba_cfg.total_bytes_per_request
                lines += [
                    f"  Mamba prefix cache:",
                    f"    budget:                  {_fmt_bytes(self.prefix_cache_budget_bytes)}",
                    f"    slots:                   {slots}",
                ]

        lines += [
            f"  Device assignments:        {result.device_assignments}",
            f"  PCIe swap cost estimate:   {_pcie_transfer_time_us(self._block_size_bytes):.1f} μs/block",
            "=" * 60,
        ]

        logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# DES-LOC 推理配置（对应 Megatron InferenceConfig）
# ---------------------------------------------------------------------------

@dataclass
class DESLOCInferenceConfig:
    """
    DES-LOC 异构推理配置。

    对应 Megatron InferenceConfig，新增:
      1. verbose: InitVar（对应 dcc6d62 引入的 verbose 参数）
      2. hetero_view: 异构设备内存视图
      3. mamba_config: Mamba 状态配置（替代 mamba_inference_state_config）
      4. 精确 / 比例两种内存分配路径的控制开关

    InitVar 设计说明（对应 Megatron 原始 PR）:
        verbose 被声明为 InitVar 而非普通 field，原因是：
        它仅在 __post_init__ 中使用，不应作为配置的持久化字段。
        但为了支持日志调试，需要在初始化时传入。
    """
    # --- 基础参数 ---
    max_sequence_length: int = 2048
    block_size_tokens: int = 256
    num_heads: int = 32
    head_dim: int = 128
    kv_dtype: torch.dtype = torch.float16

    # --- Mamba 参数 ---
    mamba_config: Optional[MambaStateConfig] = None
    max_requests: Optional[int] = None
    mamba_memory_ratio: Optional[float] = None
    num_speculative_tokens: int = 0

    # --- 前缀缓存 ---
    enable_prefix_caching: bool = False
    prefix_cache_budget_gb: float = 0.0

    # --- 异构硬件视图 ---
    hetero_view: HeteroDeviceMemoryView = field(
        default_factory=HeteroDeviceMemoryView
    )

    # --- 精度控制 ---
    prefix_caching_routing_alpha: float = 0.5  # 对应 Megatron 同名字段

    # InitVar: 不存储，仅 __post_init__ 使用（对应 dcc6d62 引入的模式）
    verbose: InitVar[bool] = False

    def __post_init__(self, verbose: bool):
        # 对应 dcc6d62: self._verbose = verbose
        self._verbose = verbose

        # 验证
        if not (0.0 <= self.prefix_caching_routing_alpha <= 1.0):
            raise ValueError(
                f"prefix_caching_routing_alpha must be in [0, 1], "
                f"got {self.prefix_caching_routing_alpha}"
            )
        if self.mamba_memory_ratio is not None:
            if not (0.0 <= self.mamba_memory_ratio <= 1.0):
                raise ValueError(
                    f"mamba_memory_ratio must be in [0, 1], got {self.mamba_memory_ratio}"
                )
        if self.max_requests is not None and self.max_requests <= 0:
            raise ValueError(f"max_requests must be positive, got {self.max_requests}")

        logger.debug(
            "DESLOCInferenceConfig initialized: max_requests=%s, "
            "mamba_memory_ratio=%s, verbose=%s",
            self.max_requests, self.mamba_memory_ratio, self._verbose,
        )

    def build_allocator(self) -> HeteroMambaStateMemoryAllocator:
        """
        构建 HeteroMambaStateMemoryAllocator 实例。

        这是 DES-LOC 中连接配置与实际内存分配的工厂方法，
        对应 Megatron DynamicInferenceContext.__init__ 中的内联分配逻辑。
        """
        if self.mamba_config is None:
            # 构造一个空的 MambaStateConfig（纯 Attention 模型）
            mamba_cfg = MambaStateConfig(
                conv_state_shape=(0,),
                ssm_state_shape=(0,),
                num_mamba_layers=0,
            )
        else:
            mamba_cfg = self.mamba_config

        prefix_bytes = int(self.prefix_cache_budget_gb * 1024**3)

        return HeteroMambaStateMemoryAllocator(
            mamba_config=mamba_cfg,
            hetero_view=self.hetero_view,
            block_size_tokens=self.block_size_tokens,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            kv_dtype=self.kv_dtype,
            num_speculative_tokens=self.num_speculative_tokens,
            enable_prefix_caching=self.enable_prefix_caching,
            prefix_cache_budget_bytes=prefix_bytes,
            verbose=self._verbose,
        )

    def compute_memory_budget(self) -> MemoryBudgetResult:
        """配置的一站式入口：构建分配器并计算预算。"""
        allocator = self.build_allocator()
        return allocator.compute_budget(
            max_requests=self.max_requests,
            mamba_memory_ratio=self.mamba_memory_ratio,
        )


# ---------------------------------------------------------------------------
# SharedLocalityCache — DES-LOC 核心组件（KV 块局部性索引）
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """
    跨设备共享的局部性感知 KV 块索引。

    DES-LOC 的"Shared LOcality Cache"名称来源于此组件：
    它维护一个全局的块 ID 到物理设备位置的映射，
    并根据访问热度动态调整块的物理位置（H100 ↔ A6000 ↔ CPU）。

    在 PCIe 互联、无 NVLink 的拓扑下，迁移代价由 _pcie_transfer_time_us 估算。
    前缀缓存命中的 Mamba 状态直接从 A6000 读取，避免 PCIe 跨跳。

    核心数据结构:
        _hot_blocks: Set[int]   — 在 H100 上的活跃块 ID 集合
        _cold_blocks: Set[int]  — 在 A6000 上的冷块 ID 集合
        _cpu_blocks: Set[int]   — 在 CPU DRAM 上的换页块 ID 集合
        _access_count: Dict[int, int] — 块访问计数（用于 LRU 驱逐）
    """

    def __init__(self, budget: MemoryBudgetResult, block_size_bytes: int):
        self.budget = budget
        self.block_size_bytes = block_size_bytes

        self._hot_blocks: set = set()
        self._cold_blocks: set = set()
        self._cpu_blocks: set = set()
        self._access_count: Dict[int, int] = {}

        self._hot_capacity = budget.kv_active_blocks
        self._cpu_capacity = budget.kv_paused_blocks

        logger.debug(
            "SharedLocalityCache: hot_cap=%d blocks, cpu_cap=%d blocks",
            self._hot_capacity, self._cpu_capacity,
        )

    def register_block(self, block_id: int, initial_device: DeviceRole = DeviceRole.KV_HOT):
        """注册新块到缓存索引。"""
        self._access_count[block_id] = 0
        if initial_device == DeviceRole.KV_HOT:
            self._hot_blocks.add(block_id)
        elif initial_device == DeviceRole.KV_COLD:
            self._cold_blocks.add(block_id)
        else:
            self._cpu_blocks.add(block_id)

    def access_block(self, block_id: int) -> Tuple[DeviceRole, float]:
        """
        访问一个块，返回其当前位置和估算的访问延迟（μs）。

        若块在 A6000 或 CPU 上，触发后台迁移提示（实际迁移由调度器执行）。
        """
        self._access_count[block_id] = self._access_count.get(block_id, 0) + 1

        if block_id in self._hot_blocks:
            return DeviceRole.KV_HOT, 0.1   # H100 本地，~0.1 μs
        elif block_id in self._cold_blocks:
            latency = _pcie_transfer_time_us(self.block_size_bytes)
            logger.debug(
                "Block %d on A6000 (cold): PCIe fetch cost ~%.1f μs",
                block_id, latency,
            )
            return DeviceRole.KV_COLD, latency
        elif block_id in self._cpu_blocks:
            latency = _pcie_transfer_time_us(self.block_size_bytes) * 2  # CPU→GPU 更慢
            logger.debug(
                "Block %d on CPU (swap): PCIe fetch cost ~%.1f μs",
                block_id, latency,
            )
            return DeviceRole.CPU_SWAP, latency
        else:
            logger.error("Block %d not found in SharedLocalityCache!", block_id)
            return DeviceRole.KV_HOT, 0.0

    def promote_block(self, block_id: int) -> bool:
        """将块从 cold/CPU 提升到 hot（H100）。返回是否成功。"""
        if block_id in self._hot_blocks:
            return True  # 已经在 hot
        if len(self._hot_blocks) >= self._hot_capacity:
            # 需要驱逐一个 hot 块
            if not self._evict_lru_hot():
                logger.warning("Cannot promote block %d: hot tier full and no LRU candidate.", block_id)
                return False
        if block_id in self._cold_blocks:
            self._cold_blocks.discard(block_id)
        elif block_id in self._cpu_blocks:
            self._cpu_blocks.discard(block_id)
        self._hot_blocks.add(block_id)
        logger.debug("Block %d promoted to hot (H100).", block_id)
        return True

    def _evict_lru_hot(self) -> bool:
        """从 hot 层驱逐访问最少的块到 cold 层。"""
        if not self._hot_blocks:
            return False
        lru_id = min(self._hot_blocks, key=lambda b: self._access_count.get(b, 0))
        self._hot_blocks.discard(lru_id)
        self._cold_blocks.add(lru_id)
        logger.debug("Evicted block %d from hot to cold (A6000).", lru_id)
        return True

    @property
    def stats(self) -> Dict[str, int]:
        """返回当前缓存分布统计。"""
        return {
            "hot_blocks": len(self._hot_blocks),
            "cold_blocks": len(self._cold_blocks),
            "cpu_blocks": len(self._cpu_blocks),
            "total_tracked": len(self._access_count),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- 构建 Mamba 配置 ---
    mamba_cfg = MambaStateConfig(
        conv_state_shape=(544, 4),
        ssm_state_shape=(8, 64, 16),
        conv_dtype=torch.float16,
        ssm_dtype=torch.float16,
        num_mamba_layers=2,
    )

    # 1. 精确路径 smoke test (max_requests=4)
    cfg_precise = DESLOCInferenceConfig(
        max_sequence_length=512,
        block_size_tokens=256,
        num_heads=8,
        head_dim=64,
        mamba_config=mamba_cfg,
        max_requests=4,
        verbose=True,
    )
    budget = cfg_precise.compute_memory_budget()
    assert budget.mamba_max_requests == 4, f"Expected 4, got {budget.mamba_max_requests}"
    assert budget.kv_active_blocks >= 2, "Must have at least 2 KV blocks"
    assert budget.mamba_bytes > 0, "Mamba bytes should be non-zero"
    print(f"[OK] precise path: mamba={_fmt_bytes(budget.mamba_bytes)}, "
          f"kv_blocks={budget.kv_active_blocks}")

    # 2. 更多请求 → 更少 KV 块
    cfg_many = DESLOCInferenceConfig(
        max_sequence_length=512,
        block_size_tokens=256,
        num_heads=8,
        head_dim=64,
        mamba_config=mamba_cfg,
        max_requests=64,
    )
    budget_many = cfg_many.compute_memory_budget()
    assert budget.kv_active_blocks >= budget_many.kv_active_blocks, \
        "Fewer requests should yield more KV blocks"
    print(f"[OK] max_requests=4 → {budget.kv_active_blocks} blocks "
          f"> max_requests=64 → {budget_many.kv_active_blocks} blocks")

    # 3. BlockOverflow 检查
    allocator = cfg_precise.build_allocator()
    try:
        allocator.validate_request_blocks(
            request_id=99,
            prompt_len=200000,   # 极大 prompt
            max_gen_tokens=1000,
            budget=budget,
        )
        assert False, "Should have raised HeteroBlockOverflowError"
    except HeteroBlockOverflowError as e:
        assert e.request_id == 99
        print(f"[OK] HeteroBlockOverflowError caught: {e}")

    # 4. _fmt_bytes 边界
    assert _fmt_bytes(0) == "0 bytes"
    assert "GB" in _fmt_bytes(2 * 1024**3)
    print("[OK] _fmt_bytes: zero and GB cases pass")

    # 5. SharedLocalityCache 基本操作
    cache = SharedLocalityCache(budget, _compute_block_size_bytes(8, 64, 256, torch.float16))
    cache.register_block(42, DeviceRole.KV_COLD)
    role, latency = cache.access_block(42)
    assert role == DeviceRole.KV_COLD
    assert latency > 0
    print(f"[OK] SharedLocalityCache: cold block latency={latency:.1f} μs")

    print("\n=== All smoke tests passed ===")
