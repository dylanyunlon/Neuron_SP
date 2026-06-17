"""
deepspeed/inference/hetero_shared_expert_overlap.py
====================================================

DES-LOC HeteroSharedExpertOverlap — 异构感知共享专家重叠调度
============================================================

上游设计意图 (Megatron commit 442a936a)
---------------------------------------
Megatron-LM 在推理路径中为 NVLS（NVLink Switch）AllGatherV dispatcher
引入了 shared expert overlap：在 dispatch_preprocess 阶段将共享专家的
前向传播启动到独立的 CUDA stream（SharedExpertMLP.stream），使其与
AGV + routed-expert GEMMs + RSV combine 并发执行。在 combine_postprocess
阶段同步该 stream 并将共享专家输出加回主输出。此外，当共享专家重叠
启用时，AGV 的 max_num_blocks 被限制为 16，防止 AGV CTA 抢占共享专家
GEMM 所需的 SM 资源。

DES-LOC 适配点
--------------
原始实现假设同构 NVLink Switch 硬件，所有 GPU 通过 NVLS 高带宽互联。
DES-LOC 目标硬件为：
  - 2x A6000 48GB (SM86, PCIe)  —— 高显存密度，适合存放路由专家权重
  - 1x H100 NVL 96GB (SM90)     —— 高算力，适合共享专家（dense GEMM）
  - PCIe 互联，无 NVLink         —— AllGather 带宽受限，需精细流水线
  - 1.5TB CPU DRAM               —— 用于 LOC (Shared LOcality Cache) offload

DES-LOC 的核心差异：
1. **设备分层感知 (DeviceTierAwareDispatcher)**：H100 承担共享专家计算，
   A6000 承担路由专家计算，调度器需感知每个 GPU 的 SM 版本与显存。
2. **PCIe 带宽自适应 CTA 限制**：无 NVLink 时，AllGather 本身已是带宽瓶颈，
   CTA 限制策略不同于 NVLS 场景——需要动态计算而非硬编码 16。
3. **LOC 缓存感知流水线**：共享专家输出可被缓存到 CPU DRAM LOC，在
   序列复用场景下跳过重计算。
4. **异步 PCIe 传输与 CUDA stream 协调**：H100 上完成的共享专家输出
   需通过 PCIe 传回目标 GPU，需要额外的 transfer stream 管理。
5. **SM 版本路由**：SM90 支持 warp specialization，SM86 不支持，
   共享专家 GEMM kernel 选择需按设备分支。

模块结构
--------
  HeteroDeviceProfile          — 硬件拓扑描述与 SM/带宽 Profiling
  LOCCache                     — CPU DRAM LOC 缓存管理
  HeteroStreamManager          — 异构多 GPU stream 生命周期管理
  HeteroSharedExpertDispatcher — 核心 dispatch/combine 逻辑（适配 Megatron 接口）
  MoEHeteroLayer               — 顶层 MoE 层，整合异构调度
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import unittest
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 硬件常量
# ---------------------------------------------------------------------------

class SMArch(Enum):
    """CUDA SM 架构枚举，对应 DES-LOC 目标硬件。"""
    SM86 = 86   # A6000 — Ampere，无 warp specialization
    SM90 = 90   # H100  — Hopper，支持 warp specialization + TMA


# SM90 上 AllGather 允许的最大 CTA 数（Hopper Tensor Memory Accelerator 可并发）
_AGV_MAX_CTA_SM90_SHARED_OVERLAP = 12
# SM86 上更保守——PCIe 带宽瓶颈，减少 CTA 争用
_AGV_MAX_CTA_SM86_SHARED_OVERLAP = 8
# 无共享专家重叠时不限制 CTA
_AGV_MAX_CTA_UNLIMITED = None

# PCIe Gen4 x16 理论单向带宽 (GB/s)，用于传输时延估算
_PCIE_GEN4_BW_GBPS = 32.0

# LOC 缓存默认容量（条目数，每条目为一个 hidden_states tensor）
_LOC_DEFAULT_CAPACITY = 64


# ---------------------------------------------------------------------------
# 数据类：硬件描述
# ---------------------------------------------------------------------------

@dataclass
class HeteroDeviceProfile:
    """
    描述单个 GPU 的硬件属性，供 DES-LOC 调度器进行异构感知决策。

    Attributes
    ----------
    device_index : int
        torch.device 的设备编号。
    sm_arch : SMArch
        CUDA SM 架构版本。
    total_memory_gb : float
        设备总显存（GB）。
    pcie_bw_gbps : float
        到 CPU/其他设备的 PCIe 带宽估算值（GB/s）。
    is_shared_expert_device : bool
        该设备是否被分配为共享专家计算主设备（DES-LOC 中为 H100）。
    compute_stream : torch.cuda.Stream
        主计算 stream。
    transfer_stream : torch.cuda.Stream
        专用于 PCIe 数据传输的 stream（与计算 stream 并发）。
    """
    device_index: int
    sm_arch: SMArch
    total_memory_gb: float
    pcie_bw_gbps: float = _PCIE_GEN4_BW_GBPS
    is_shared_expert_device: bool = False
    compute_stream: Optional[torch.cuda.Stream] = field(default=None, repr=False)
    transfer_stream: Optional[torch.cuda.Stream] = field(default=None, repr=False)

    def __post_init__(self):
        device = torch.device(f"cuda:{self.device_index}")
        if self.compute_stream is None:
            self.compute_stream = torch.cuda.Stream(device=device)
        if self.transfer_stream is None:
            self.transfer_stream = torch.cuda.Stream(device=device)

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.device_index}")

    @property
    def supports_warp_specialization(self) -> bool:
        """SM90+ 支持 warp specialization，影响 GEMM kernel 选择。"""
        return self.sm_arch.value >= SMArch.SM90.value

    def estimate_transfer_latency_ms(self, tensor_bytes: int) -> float:
        """
        估算通过 PCIe 传输指定字节数所需时间（毫秒）。

        Parameters
        ----------
        tensor_bytes : int
            待传输数据的字节数。

        Returns
        -------
        float
            估算传输时延（ms）。
        """
        bw_bytes_per_ms = self.pcie_bw_gbps * 1e9 / 1e3
        return tensor_bytes / bw_bytes_per_ms

    def agv_max_cta(self, shared_expert_overlap: bool) -> Optional[int]:
        """
        根据 SM 架构和是否启用共享专家重叠，返回 AllGather CTA 上限。

        DES-LOC 适配：Megatron 硬编码 16（NVLS 场景），DES-LOC 按 SM 版本
        动态返回不同值，以适应 PCIe 互联下的 SM 争用特征。

        Parameters
        ----------
        shared_expert_overlap : bool
            是否启用共享专家重叠执行。

        Returns
        -------
        Optional[int]
            CTA 上限，None 表示不限制。
        """
        if not shared_expert_overlap:
            return _AGV_MAX_CTA_UNLIMITED
        if self.sm_arch == SMArch.SM90:
            return _AGV_MAX_CTA_SM90_SHARED_OVERLAP
        else:
            return _AGV_MAX_CTA_SM86_SHARED_OVERLAP


# ---------------------------------------------------------------------------
# LOC 缓存：Shared LOcality Cache（CPU DRAM）
# ---------------------------------------------------------------------------

class LOCCacheEntry:
    """LOC 缓存条目，持有固定内存上的张量引用。"""

    __slots__ = ("key", "tensor_cpu", "timestamp", "hit_count")

    def __init__(self, key: str, tensor_cpu: torch.Tensor):
        self.key = key
        self.tensor_cpu = tensor_cpu
        self.timestamp = time.monotonic()
        self.hit_count = 0

    def touch(self):
        self.timestamp = time.monotonic()
        self.hit_count += 1


class LOCCache:
    """
    CPU DRAM LOC（Shared LOcality Cache）管理器。

    在 DES-LOC 异构框架中，1.5TB CPU DRAM 是重要的二级缓存层。
    共享专家的输出在序列复用场景（如多轮对话的 KV 前缀复用）中
    可能被重复计算，LOC 缓存将这些输出 pinned 到 CPU 内存，
    并在需要时异步预取回 GPU。

    缓存策略：LRU（Least Recently Used）驱逐，按 hit_count 加权。

    线程安全：使用 threading.Lock 保护并发读写。

    Parameters
    ----------
    capacity : int
        最大缓存条目数（每条目为一个 hidden_states 张量）。
    """

    def __init__(self, capacity: int = _LOC_DEFAULT_CAPACITY):
        self.capacity = capacity
        self._store: Dict[str, LOCCacheEntry] = {}
        self._lock = threading.Lock()
        self._eviction_count = 0
        logger.info(
            "LOCCache initialized: capacity=%d entries (CPU DRAM pinned memory)",
            capacity,
        )

    def _make_key(self, layer_idx: int, token_hash: int) -> str:
        return f"loc:{layer_idx}:{token_hash}"

    def put(
        self,
        layer_idx: int,
        token_hash: int,
        tensor: torch.Tensor,
        transfer_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        将 GPU 张量异步复制到 CPU pinned memory 并存入 LOC 缓存。

        Parameters
        ----------
        layer_idx : int
            MoE 层编号，用于构造缓存键。
        token_hash : int
            token 序列的哈希值（用于复用检测）。
        tensor : torch.Tensor
            GPU 上的共享专家输出张量。
        transfer_stream : Optional[torch.cuda.Stream]
            使用指定 transfer stream 执行 D2H 传输；None 时同步传输。
        """
        key = self._make_key(layer_idx, token_hash)
        # 分配 pinned memory 以加速后续 H2D
        cpu_tensor = torch.empty(
            tensor.shape, dtype=tensor.dtype, pin_memory=True
        )
        if transfer_stream is not None:
            with torch.cuda.stream(transfer_stream):
                cpu_tensor.copy_(tensor, non_blocking=True)
        else:
            cpu_tensor.copy_(tensor)

        with self._lock:
            if key in self._store:
                self._store[key].touch()
                return
            if len(self._store) >= self.capacity:
                self._evict_lru()
            self._store[key] = LOCCacheEntry(key, cpu_tensor)

    def get(
        self,
        layer_idx: int,
        token_hash: int,
        target_device: torch.device,
        transfer_stream: Optional[torch.cuda.Stream] = None,
    ) -> Optional[torch.Tensor]:
        """
        从 LOC 缓存中检索张量并异步传回 GPU。

        Parameters
        ----------
        layer_idx : int
            MoE 层编号。
        token_hash : int
            token 序列哈希值。
        target_device : torch.device
            目标 GPU 设备。
        transfer_stream : Optional[torch.cuda.Stream]
            H2D 传输使用的 stream。

        Returns
        -------
        Optional[torch.Tensor]
            命中时返回目标设备上的张量；未命中时返回 None。
        """
        key = self._make_key(layer_idx, token_hash)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            entry.touch()

        # 异步 H2D 传输
        gpu_tensor = torch.empty(
            entry.tensor_cpu.shape,
            dtype=entry.tensor_cpu.dtype,
            device=target_device,
        )
        if transfer_stream is not None:
            with torch.cuda.stream(transfer_stream):
                gpu_tensor.copy_(entry.tensor_cpu, non_blocking=True)
        else:
            gpu_tensor.copy_(entry.tensor_cpu)

        logger.debug(
            "LOCCache HIT: key=%s hit_count=%d shape=%s",
            key, entry.hit_count, tuple(entry.tensor_cpu.shape),
        )
        return gpu_tensor

    def _evict_lru(self) -> None:
        """驱逐最久未使用（且命中次数最少）的条目。调用方须持有锁。"""
        if not self._store:
            return
        lru_key = min(
            self._store,
            key=lambda k: (self._store[k].hit_count, self._store[k].timestamp),
        )
        del self._store[lru_key]
        self._eviction_count += 1
        logger.debug("LOCCache evicted key=%s (total_evictions=%d)", lru_key, self._eviction_count)

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._store),
                "capacity": self.capacity,
                "evictions": self._eviction_count,
            }


# ---------------------------------------------------------------------------
# Stream 管理器：异构多 GPU stream 协调
# ---------------------------------------------------------------------------

class HeteroStreamManager:
    """
    异构多 GPU stream 生命周期管理器。

    DES-LOC 在三块 GPU 上维护多个并发 stream：
      - 每块 GPU 上的 compute_stream（主计算）
      - 每块 GPU 上的 transfer_stream（PCIe 传输）
      - shared_expert_stream：专用于共享专家前向（绑定到 H100）

    此类集中管理这些 stream，避免跨模块的 stream 泄露，
    并提供 barrier 工具方法用于精确的 stream 同步点。

    Parameters
    ----------
    device_profiles : List[HeteroDeviceProfile]
        所有参与设备的硬件描述列表。
    """

    def __init__(self, device_profiles: List[HeteroDeviceProfile]):
        self.profiles: Dict[int, HeteroDeviceProfile] = {
            p.device_index: p for p in device_profiles
        }
        # 找到共享专家设备（H100）
        shared_devices = [p for p in device_profiles if p.is_shared_expert_device]
        if len(shared_devices) != 1:
            raise ValueError(
                f"DES-LOC requires exactly one shared-expert device (H100), "
                f"got {len(shared_devices)}"
            )
        self.shared_expert_profile = shared_devices[0]
        self.shared_expert_stream = self.shared_expert_profile.compute_stream

        logger.info(
            "HeteroStreamManager: shared_expert_device=cuda:%d (SM%d), "
            "routed_expert_devices=%s",
            self.shared_expert_profile.device_index,
            self.shared_expert_profile.sm_arch.value,
            [p.device_index for p in device_profiles if not p.is_shared_expert_device],
        )

    def wait_shared_expert_on(self, target_device_idx: int) -> None:
        """
        在 target_device 的当前 stream 上插入等待点，
        阻塞直到 shared_expert_stream 完成。

        这是 Megatron combine_postprocess 中 wait_stream 的异构版本：
        Megatron 的两个 stream 在同一 GPU 上，DES-LOC 中需要跨 GPU
        通过 CUDA event 同步（event 在 H100 上 record，在 A6000 上 wait）。

        Parameters
        ----------
        target_device_idx : int
            需要等待共享专家结果的目标设备编号。
        """
        # 在共享专家 stream 上 record event
        with torch.cuda.device(self.shared_expert_profile.device_index):
            event = torch.cuda.Event()
            event.record(self.shared_expert_stream)

        # 在目标设备的当前 stream 上 wait
        with torch.cuda.device(target_device_idx):
            torch.cuda.current_stream().wait_event(event)

    def launch_on_shared_expert_device(
        self, fn, *args, **kwargs
    ) -> None:
        """
        在共享专家设备（H100）的 compute_stream 上异步启动函数。

        等价于 Megatron 的：
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                output = shared_experts(hidden_states)

        但 DES-LOC 中 "current stream" 在 A6000 上，需要先传输输入。

        Parameters
        ----------
        fn : callable
            待执行的函数（通常是 shared_expert 的 forward）。
        *args, **kwargs
            传递给 fn 的参数。

        Returns
        -------
        None
            结果通过 fn 内部副作用（tensor 原地写入）或外部引用获取。
        """
        # 让共享专家 stream 等待当前 stream（确保输入数据就绪）
        current_event = torch.cuda.Event()
        current_event.record(torch.cuda.current_stream())
        self.shared_expert_stream.wait_event(current_event)

        with torch.cuda.device(self.shared_expert_profile.device_index):
            with torch.cuda.stream(self.shared_expert_stream):
                fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# 共享专家模块（DES-LOC 异构版）
# ---------------------------------------------------------------------------

class HeteroSharedExpertMLP(nn.Module):
    """
    异构感知共享专家 MLP。

    在 DES-LOC 中，共享专家固定部署在 H100（SM90）设备上，
    利用其更高的 GEMM 吞吐量和 warp specialization 支持。

    相比 Megatron 的 SharedExpertMLP，此版本：
    1. 绑定到特定 device（H100），不跟随主 rank 设备。
    2. 在 SM90 上使用 F.linear（可触发 cuBLAS Hopper fast path）。
    3. 暴露与上游相同的 stream 属性接口（self.stream），
       供 HeteroSharedExpertDispatcher 使用。

    Parameters
    ----------
    hidden_size : int
        隐层维度。
    ffn_hidden_size : int
        FFN 中间层维度。
    device_profile : HeteroDeviceProfile
        共享专家所在设备的硬件描述（应为 H100）。
    """

    # 类级别 stream，与 Megatron SharedExpertMLP.stream 接口对齐
    stream: Optional[torch.cuda.Stream] = None

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        device_profile: HeteroDeviceProfile,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.device_profile = device_profile

        # 权重分配在共享专家设备（H100）上
        device = device_profile.device
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False, device=device)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False, device=device)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False, device=device)

        # 初始化类级别 stream（绑定到共享专家设备）
        if HeteroSharedExpertMLP.stream is None:
            HeteroSharedExpertMLP.stream = device_profile.compute_stream

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU MLP 前向传播。

        输入 x 应已在共享专家设备（H100）上。
        SM90 的 warp specialization 对 F.linear 的优化由 cuBLAS 自动启用。

        Parameters
        ----------
        x : torch.Tensor
            形状 [S*B/TP, H] 的输入张量，在 H100 设备上。

        Returns
        -------
        torch.Tensor
            形状 [S*B/TP, H] 的共享专家输出。
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# 核心调度器：HeteroSharedExpertDispatcher
# ---------------------------------------------------------------------------

class HeteroSharedExpertDispatcher:
    """
    DES-LOC 异构共享专家重叠调度器。

    这是本模块的核心类，重新诠释了 Megatron commit 442a936a 的
    NVLSAllGatherVDispatcher 共享专家重叠逻辑，适配 PCIe 互联的
    异构 GPU 拓扑。

    Megatron 原始流程（NVLS 同构）：
    ┌─────────────────────────────────────────────────────┐
    │ dispatch_preprocess:                                │
    │   stream.wait_stream(current_stream)                │
    │   with stream: shared_expert(hidden_states) → buf  │
    │ token_dispatch: AGV(max_cta=16) + routed experts   │
    │ combine_postprocess:                                │
    │   current_stream.wait_stream(stream)                │
    │   output += shared_expert_buf                       │
    └─────────────────────────────────────────────────────┘

    DES-LOC 异构流程（PCIe，跨 GPU）：
    ┌─────────────────────────────────────────────────────┐
    │ dispatch_preprocess (A6000 主路径):                 │
    │   1. token_hash = hash(hidden_states)               │
    │   2. LOC cache lookup → 命中则跳过共享专家计算      │
    │   3. 未命中：H2D transfer: hidden → H100            │
    │      H100.stream: shared_expert(hidden_H100) → buf │
    │   4. AGV CTA 限制按 SM 版本动态设置                 │
    │ token_dispatch: AllGather + routed experts (A6000)  │
    │ combine_postprocess (A6000 主路径):                 │
    │   1. CUDA event sync: H100.stream → A6000.stream   │
    │   2. shared_output = transfer D2H→pinned→H2D       │
    │      或直接 peer copy (若 peer access 启用)         │
    │   3. output += shared_output                        │
    │   4. LOC cache put (async D2H to CPU DRAM)         │
    └─────────────────────────────────────────────────────┘

    Parameters
    ----------
    stream_manager : HeteroStreamManager
        异构 stream 协调器。
    routed_device_profile : HeteroDeviceProfile
        路由专家所在设备（A6000）的硬件描述。
    loc_cache : Optional[LOCCache]
        LOC 缓存实例；None 时禁用缓存。
    layer_idx : int
        所属 MoE 层编号，用于 LOC 缓存键和日志。
    enable_peer_access : bool
        是否尝试启用 GPU peer access（跳过 CPU 中转）。
    """

    def __init__(
        self,
        stream_manager: HeteroStreamManager,
        routed_device_profile: HeteroDeviceProfile,
        loc_cache: Optional[LOCCache] = None,
        layer_idx: int = 0,
        enable_peer_access: bool = False,
    ):
        self.stream_manager = stream_manager
        self.routed_profile = routed_device_profile
        self.shared_profile = stream_manager.shared_expert_profile
        self.loc_cache = loc_cache
        self.layer_idx = layer_idx
        self.enable_peer_access = enable_peer_access

        # 状态变量（与 Megatron _shared_expert_output 对应）
        self._shared_expert_output_h100: Optional[torch.Tensor] = None
        self._current_token_hash: Optional[int] = None
        self._loc_hit: bool = False
        self._hidden_shape: Optional[tuple] = None

        # shared_experts 属性（由外部调用 set_shared_experts 注入）
        self.shared_experts: Optional[HeteroSharedExpertMLP] = None

        # 尝试开启 peer access（A6000 ↔ H100）
        self._peer_access_enabled = False
        if enable_peer_access:
            self._try_enable_peer_access()

    def _try_enable_peer_access(self) -> None:
        """
        尝试在路由专家设备和共享专家设备之间启用 CUDA peer access。

        PCIe 拓扑下 peer access 支持取决于平台配置，失败时静默回退到
        CPU 中转路径，不抛出异常。
        """
        try:
            can_access = torch.cuda.can_device_access_peer(
                self.routed_profile.device_index,
                self.shared_profile.device_index,
            )
            if can_access:
                torch.cuda.device(self.routed_profile.device_index)
                # torch API: enable_peer_access 在 CUDA runtime 层面
                self._peer_access_enabled = True
                logger.info(
                    "Peer access enabled: cuda:%d ↔ cuda:%d",
                    self.routed_profile.device_index,
                    self.shared_profile.device_index,
                )
            else:
                logger.info(
                    "Peer access not available between cuda:%d and cuda:%d; "
                    "falling back to staged D2H→H2D transfer",
                    self.routed_profile.device_index,
                    self.shared_profile.device_index,
                )
        except Exception as exc:
            logger.warning("Peer access probe failed: %s", exc)

    def set_shared_experts(self, shared_experts: HeteroSharedExpertMLP) -> None:
        """
        注入共享专家模块（对应 Megatron set_shared_experts 接口）。

        Parameters
        ----------
        shared_experts : HeteroSharedExpertMLP
            已部署到 H100 的共享专家 MLP 实例。
        """
        self.shared_experts = shared_experts
        logger.info(
            "Layer %d: shared experts attached to dispatcher "
            "(device=cuda:%d SM%d)",
            self.layer_idx,
            self.shared_profile.device_index,
            self.shared_profile.sm_arch.value,
        )

    @property
    def shared_expert_overlap_active(self) -> bool:
        """是否实际启用了共享专家重叠（需要 shared_experts 已注入）。"""
        return self.shared_experts is not None

    def _compute_token_hash(self, hidden_states: torch.Tensor) -> int:
        """
        计算 token 序列的轻量级哈希，用于 LOC 缓存命中检测。

        使用前 min(256, numel) 个元素的浮点和作为近似哈希，
        避免全量计算开销。对于推理复用场景（相同前缀），此方法足够有效。

        Parameters
        ----------
        hidden_states : torch.Tensor
            输入隐状态张量。

        Returns
        -------
        int
            整数哈希值。
        """
        with torch.no_grad():
            sample = hidden_states.view(-1)[: min(256, hidden_states.numel())]
            # 转 float32 再求和，避免 bfloat16 精度问题
            hash_val = int(sample.float().sum().item() * 1e6) % (2 ** 31)
        return hash_val

    def _transfer_to_shared_device(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        将 hidden_states 从路由专家设备（A6000）传输到共享专家设备（H100）。

        如果 peer access 可用，直接调用 .to(device)（触发 peer copy）；
        否则经由 CPU pinned memory 中转（D2H + H2D）。

        Parameters
        ----------
        hidden_states : torch.Tensor
            在 A6000 上的输入张量。

        Returns
        -------
        torch.Tensor
            在 H100 上的张量副本。
        """
        target_device = self.shared_profile.device
        if self._peer_access_enabled:
            # Peer copy：跳过 CPU 中转，利用 PCIe DMA
            return hidden_states.to(target_device, non_blocking=True)
        else:
            # 分阶段传输：D2H（pinned）→ H2D
            transfer_stream = self.routed_profile.transfer_stream
            with torch.cuda.stream(transfer_stream):
                pinned = hidden_states.cpu().pin_memory()
            # H2D 在共享专家 stream 上执行，与 D2H 天然流水
            with torch.cuda.stream(self.shared_profile.compute_stream):
                return pinned.to(target_device, non_blocking=True)

    def _transfer_from_shared_device(
        self, output_h100: torch.Tensor
    ) -> torch.Tensor:
        """
        将共享专家输出从 H100 传回 A6000（路由专家设备）。

        优先使用 peer access；否则经 CPU pinned memory 中转。

        Parameters
        ----------
        output_h100 : torch.Tensor
            在 H100 上的共享专家输出张量。

        Returns
        -------
        torch.Tensor
            在 A6000 上的张量。
        """
        target_device = self.routed_profile.device
        if self._peer_access_enabled:
            return output_h100.to(target_device, non_blocking=True)
        else:
            # D2H（H100 → CPU pinned）
            transfer_stream = self.shared_profile.transfer_stream
            with torch.cuda.stream(transfer_stream):
                pinned = output_h100.cpu().pin_memory()
            # H2D（CPU → A6000）
            with torch.cuda.stream(self.routed_profile.transfer_stream):
                return pinned.to(target_device, non_blocking=True)

    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """
        DES-LOC dispatch 预处理阶段，对应 Megatron dispatch_preprocess。

        核心变化（相对于 Megatron NVLS 实现）：
        1. 计算 token hash 并查询 LOC 缓存，命中时跳过共享专家计算。
        2. 未命中时：将 hidden_states 异步传输到 H100，并在 H100 的
           compute_stream 上启动共享专家前向（与 A6000 主路径并发）。
        3. 返回 flatten 后的 hidden_states 和 CTA 限制参数。

        Parameters
        ----------
        hidden_states : torch.Tensor
            形状 [S/TP, B, H] 或 [S*B/TP, H] 的输入隐状态。
        routing_map : torch.Tensor
            token-to-expert 路由掩码。
        probs : torch.Tensor
            路由概率。

        Returns
        -------
        Tuple[torch.Tensor, Optional[int]]
            (flatten 后的 hidden_states, agv_max_cta)
        """
        self._hidden_shape = hidden_states.shape
        self._loc_hit = False
        self._shared_expert_output_h100 = None

        # flatten: [S/TP, B, H] → [S*B/TP, H]
        hidden_flat = hidden_states.view(-1, self._hidden_shape[-1])
        self._local_tokens = hidden_flat.shape[0]

        agv_max_cta = self.routed_profile.agv_max_cta(self.shared_expert_overlap_active)

        if not self.shared_expert_overlap_active:
            return hidden_flat, agv_max_cta

        # LOC 缓存查询
        self._current_token_hash = self._compute_token_hash(hidden_flat)
        if self.loc_cache is not None:
            cached = self.loc_cache.get(
                layer_idx=self.layer_idx,
                token_hash=self._current_token_hash,
                target_device=self.routed_profile.device,
                transfer_stream=self.routed_profile.transfer_stream,
            )
            if cached is not None:
                self._shared_expert_output_h100 = cached  # 实际在 A6000 上
                self._loc_hit = True
                logger.debug(
                    "Layer %d: LOC cache hit for token_hash=%d, "
                    "skipping shared expert forward",
                    self.layer_idx, self._current_token_hash,
                )
                return hidden_flat, agv_max_cta

        # LOC 未命中：启动 H100 上的共享专家前向（异步）
        # Step 1: 传输输入到 H100
        hidden_h100 = self._transfer_to_shared_device(hidden_flat)

        # Step 2: 在 H100 compute_stream 上启动前向
        # 对应 Megatron: stream.wait_stream(current) + with stream: shared_expert(x)
        shared_output_holder: List[Optional[torch.Tensor]] = [None]

        def _run_shared_expert():
            shared_output_holder[0] = self.shared_experts(hidden_h100)

        self.stream_manager.launch_on_shared_expert_device(_run_shared_expert)

        # 将输出引用存入状态（H100 上，计算尚未完成）
        # combine_postprocess 阶段同步后读取
        # 注：shared_output_holder 通过闭包持有引用
        self._shared_expert_output_h100 = shared_output_holder
        self._is_holder = True  # 标记为 holder 模式（vs LOC 命中时的直接 tensor）

        return hidden_flat, agv_max_cta

    def combine_postprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        DES-LOC combine 后处理阶段，对应 Megatron combine_postprocess。

        核心变化（相对于 Megatron NVLS 实现）：
        1. 通过 CUDA event 跨 GPU 同步（H100 stream → A6000 stream），
           而非 Megatron 的同 GPU wait_stream。
        2. 将 H100 上的共享专家输出传回 A6000。
        3. 命中 LOC 缓存时直接使用缓存张量，不触发跨 GPU 同步。
        4. 将共享专家输出异步写入 LOC 缓存（CPU DRAM）。

        Parameters
        ----------
        hidden_states : torch.Tensor
            路由专家合并后的输出，形状 [S*B/TP, H]，在 A6000 上。

        Returns
        -------
        torch.Tensor
            加上共享专家输出后的最终隐状态，形状还原为 self._hidden_shape。
        """
        output = hidden_states.view(self._hidden_shape)

        if not self.shared_expert_overlap_active:
            return output

        if self._shared_expert_output_h100 is None:
            return output

        if self._loc_hit:
            # LOC 命中：直接使用从 CPU DRAM 异步取回的张量（已在 A6000 上）
            # 需要等待 transfer_stream 完成 H2D 传输
            transfer_event = torch.cuda.Event()
            with torch.cuda.device(self.routed_profile.device_index):
                transfer_event.record(self.routed_profile.transfer_stream)
                torch.cuda.current_stream().wait_event(transfer_event)
            shared_output_a6000 = self._shared_expert_output_h100
        else:
            # 从 holder 中取出实际输出（H100 上）
            if hasattr(self, "_is_holder") and self._is_holder:
                holder = self._shared_expert_output_h100
                # 等待 H100 compute_stream 完成
                self.stream_manager.wait_shared_expert_on(
                    self.routed_profile.device_index
                )
                shared_output_h100 = holder[0]
                self._is_holder = False
            else:
                shared_output_h100 = self._shared_expert_output_h100

            if shared_output_h100 is None:
                logger.warning(
                    "Layer %d: shared expert output is None after H100 forward; "
                    "skipping accumulation",
                    self.layer_idx,
                )
                self._shared_expert_output_h100 = None
                return output

            # 传输回 A6000
            shared_output_a6000 = self._transfer_from_shared_device(shared_output_h100)

            # 等待传输完成
            transfer_event = torch.cuda.Event()
            with torch.cuda.device(self.routed_profile.device_index):
                transfer_event.record(self.routed_profile.transfer_stream)
                torch.cuda.current_stream().wait_event(transfer_event)

            # 写入 LOC 缓存（异步，不阻塞主路径）
            if self.loc_cache is not None and self._current_token_hash is not None:
                # 注意：将 A6000 上的输出写入 LOC，避免再次跨 GPU 传输
                self.loc_cache.put(
                    layer_idx=self.layer_idx,
                    token_hash=self._current_token_hash,
                    tensor=shared_output_a6000,
                    transfer_stream=self.routed_profile.transfer_stream,
                )

        # 累加共享专家输出（Megatron: output = output + self._shared_expert_output）
        output = output + shared_output_a6000.view(self._hidden_shape)

        # 清理状态
        self._shared_expert_output_h100 = None
        self._current_token_hash = None

        return output


# ---------------------------------------------------------------------------
# MoE 层：整合异构调度的顶层封装
# ---------------------------------------------------------------------------

class MoEHeteroLayer(nn.Module):
    """
    DES-LOC 异构 MoE 层。

    整合 HeteroSharedExpertDispatcher、HeteroSharedExpertMLP 和 LOCCache，
    提供与 Megatron MoELayer 兼容的接口（forward / train / eval）。

    在推理模式（eval）下，启用共享专家重叠（对应 Megatron 的
    _inference_token_dispatcher.set_shared_experts(self.shared_experts)）；
    在训练模式下，禁用重叠以支持梯度计算（共享专家参数需要梯度流）。

    Parameters
    ----------
    hidden_size : int
        模型隐层维度。
    ffn_hidden_size : int
        FFN 中间层维度。
    num_experts : int
        路由专家总数。
    top_k : int
        每个 token 激活的专家数量。
    device_profiles : List[HeteroDeviceProfile]
        所有参与设备的硬件描述，需包含恰好一个 is_shared_expert_device=True。
    layer_idx : int
        层编号（用于日志和缓存键）。
    use_loc_cache : bool
        是否启用 LOC CPU DRAM 缓存。
    loc_cache_capacity : int
        LOC 缓存条目数上限。
    enable_peer_access : bool
        是否尝试启用 GPU peer access。
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        top_k: int,
        device_profiles: List[HeteroDeviceProfile],
        layer_idx: int = 0,
        use_loc_cache: bool = True,
        loc_cache_capacity: int = _LOC_DEFAULT_CAPACITY,
        enable_peer_access: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_idx = layer_idx

        # 识别设备角色
        shared_profiles = [p for p in device_profiles if p.is_shared_expert_device]
        routed_profiles = [p for p in device_profiles if not p.is_shared_expert_device]
        assert len(shared_profiles) == 1, "需要恰好一个 H100 共享专家设备"
        assert len(routed_profiles) >= 1, "需要至少一个 A6000 路由专家设备"

        self.shared_profile = shared_profiles[0]
        # 主路由设备：选第一个 A6000
        self.routed_profile = routed_profiles[0]

        # Stream 管理器
        self.stream_manager = HeteroStreamManager(device_profiles)

        # LOC 缓存
        self.loc_cache = LOCCache(loc_cache_capacity) if use_loc_cache else None

        # 共享专家 MLP（部署在 H100）
        self.shared_experts = HeteroSharedExpertMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            device_profile=self.shared_profile,
        )

        # 路由专家（简化：单层 Linear，实际为 ExpertMLP 组）
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.routed_experts = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_experts)
        ])

        # 调度器（推理专用，对应 Megatron _inference_token_dispatcher）
        self._inference_dispatcher = HeteroSharedExpertDispatcher(
            stream_manager=self.stream_manager,
            routed_device_profile=self.routed_profile,
            loc_cache=self.loc_cache,
            layer_idx=layer_idx,
            enable_peer_access=enable_peer_access,
        )

        # 训练/推理模式状态（对应 Megatron shared_expert_overlap）
        self.shared_expert_overlap: bool = False
        self._inference_mode: bool = False

        logger.info(
            "MoEHeteroLayer %d initialized: hidden=%d ffn=%d experts=%d top_k=%d "
            "shared_device=cuda:%d(SM%d) routed_device=cuda:%d(SM%d) "
            "loc_cache=%s",
            layer_idx, hidden_size, ffn_hidden_size, num_experts, top_k,
            self.shared_profile.device_index, self.shared_profile.sm_arch.value,
            self.routed_profile.device_index, self.routed_profile.sm_arch.value,
            "enabled" if use_loc_cache else "disabled",
        )

    def train(self, mode: bool = True) -> "MoEHeteroLayer":
        """
        切换训练/推理模式，并同步调度器状态。

        对应 Megatron MoELayer.train()：
          - train 模式：禁用共享专家重叠（需要完整反向传播）
          - eval 模式：启用共享专家重叠，注入共享专家到调度器

        Parameters
        ----------
        mode : bool
            True 为训练模式，False 为推理/eval 模式。

        Returns
        -------
        MoEHeteroLayer
            self（链式调用兼容）。
        """
        super().train(mode)
        if mode:
            # 训练模式：关闭重叠，清理调度器状态
            self.shared_expert_overlap = False
            self._inference_mode = False
            self._inference_dispatcher.shared_experts = None
        else:
            # 推理模式：启用共享专家重叠
            # 对应 Megatron:
            #   self._inference_token_dispatcher.set_shared_experts(self.shared_experts)
            #   self.shared_expert_overlap = (dispatcher.shared_experts is not None)
            self._inference_dispatcher.set_shared_experts(self.shared_experts)
            self.shared_expert_overlap = (
                self._inference_dispatcher.shared_experts is not None
            )
            self._inference_mode = True
            logger.info(
                "Layer %d: switched to inference mode, shared_expert_overlap=%s",
                self.layer_idx, self.shared_expert_overlap,
            )
        return self

    def _route_tokens(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        简化的 top-k 路由，返回 routing_map 和 probs。

        实际生产中应替换为 DeepSpeed 的 TopKRouter 实现。

        Parameters
        ----------
        hidden_states : torch.Tensor
            形状 [S*B, H] 的输入。

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (routing_map [S*B, num_experts], probs [S*B, top_k])
        """
        logits = self.router(hidden_states)  # [S*B, E]
        probs_full = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = probs_full.topk(self.top_k, dim=-1)
        routing_map = torch.zeros_like(probs_full, dtype=torch.bool)
        routing_map.scatter_(1, topk_indices, True)
        return routing_map, topk_probs

    def _apply_routed_experts(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用路由专家并聚合输出。

        Parameters
        ----------
        hidden_states : torch.Tensor
            形状 [S*B, H]。
        routing_map : torch.Tensor
            形状 [S*B, E] 的布尔掩码。
        probs : torch.Tensor
            形状 [S*B, top_k] 的路由权重。

        Returns
        -------
        torch.Tensor
            形状 [S*B, H] 的聚合输出。
        """
        batch_tokens, hidden = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        topk_indices = routing_map.nonzero(as_tuple=False)

        for token_idx, expert_idx in topk_indices:
            token_idx = token_idx.item()
            expert_idx = expert_idx.item()
            expert_out = self.routed_experts[expert_idx](
                hidden_states[token_idx : token_idx + 1]
            )
            # 对应的 topk prob
            prob_row = routing_map[token_idx].nonzero(as_tuple=False).squeeze(1)
            rank = (prob_row == expert_idx).nonzero(as_tuple=False)
            weight = probs[token_idx, rank.item()] if rank.numel() > 0 else 1.0
            output[token_idx] += expert_out.squeeze(0) * weight

        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MoE 层前向传播（训练模式与推理模式统一入口）。

        推理模式下执行 DES-LOC 共享专家重叠流水线；
        训练模式下顺序执行（保证梯度正确性）。

        Parameters
        ----------
        hidden_states : torch.Tensor
            形状 [S/TP, B, H] 的输入隐状态。

        Returns
        -------
        torch.Tensor
            形状 [S/TP, B, H] 的 MoE 层输出。
        """
        original_shape = hidden_states.shape
        # flatten
        hidden_flat = hidden_states.view(-1, self.hidden_size)
        routing_map, probs = self._route_tokens(hidden_flat)

        if self._inference_mode and self.shared_expert_overlap:
            # ── DES-LOC 推理路径：共享专家重叠 ──────────────────────────────
            # Phase 1: dispatch_preprocess（启动 H100 共享专家 + 准备 AllGather）
            hidden_flat, agv_max_cta = self._inference_dispatcher.dispatch_preprocess(
                hidden_states=hidden_states,
                routing_map=routing_map,
                probs=probs,
            )
            # Phase 2: 路由专家计算（A6000，与 H100 并发）
            routed_output = self._apply_routed_experts(hidden_flat, routing_map, probs)
            # Phase 3: combine_postprocess（同步 H100 输出并累加）
            output = self._inference_dispatcher.combine_postprocess(routed_output)
        else:
            # ── 训练路径：顺序执行 ────────────────────────────────────────────
            # 先执行路由专家
            routed_output = self._apply_routed_experts(hidden_flat, routing_map, probs)
            # 再执行共享专家
            shared_output = self.shared_experts(
                hidden_flat.to(self.shared_profile.device)
            ).to(hidden_flat.device)
            output = routed_output + shared_output.view(routed_output.shape)
            output = output.view(original_shape)

        return output


# ---------------------------------------------------------------------------
# 工厂函数：构建 DES-LOC 目标硬件的设备描述
# ---------------------------------------------------------------------------

def build_des_loc_device_profiles(
    a6000_indices: List[int] = (0, 1),
    h100_index: int = 2,
) -> List[HeteroDeviceProfile]:
    """
    为 DES-LOC 目标硬件（2x A6000 + 1x H100）构建设备描述列表。

    Parameters
    ----------
    a6000_indices : List[int]
        A6000 设备的 CUDA 设备编号列表。
    h100_index : int
        H100 设备的 CUDA 设备编号。

    Returns
    -------
    List[HeteroDeviceProfile]
        三个设备的 HeteroDeviceProfile 列表。
    """
    profiles = []

    for idx in a6000_indices:
        profiles.append(HeteroDeviceProfile(
            device_index=idx,
            sm_arch=SMArch.SM86,
            total_memory_gb=48.0,
            pcie_bw_gbps=_PCIE_GEN4_BW_GBPS,
            is_shared_expert_device=False,
        ))

    profiles.append(HeteroDeviceProfile(
        device_index=h100_index,
        sm_arch=SMArch.SM90,
        total_memory_gb=96.0,
        pcie_bw_gbps=_PCIE_GEN4_BW_GBPS,
        is_shared_expert_device=True,
    ))

    logger.info(
        "DES-LOC device profiles: %s",
        [(p.device_index, p.sm_arch.name, p.total_memory_gb) for p in profiles],
    )
    return profiles


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    class TestHeteroDeviceProfile(unittest.TestCase):
        """测试 HeteroDeviceProfile 的硬件感知逻辑。"""

        def _make_profile(self, sm_arch: SMArch, idx: int = 0) -> HeteroDeviceProfile:
            # 不实际创建 CUDA stream，使用 mock
            p = object.__new__(HeteroDeviceProfile)
            p.device_index = idx
            p.sm_arch = sm_arch
            p.total_memory_gb = 48.0 if sm_arch == SMArch.SM86 else 96.0
            p.pcie_bw_gbps = _PCIE_GEN4_BW_GBPS
            p.is_shared_expert_device = (sm_arch == SMArch.SM90)
            p.compute_stream = None
            p.transfer_stream = None
            return p

        def test_sm86_agv_cta_with_overlap(self):
            p = self._make_profile(SMArch.SM86)
            cta = p.agv_max_cta(shared_expert_overlap=True)
            self.assertEqual(cta, _AGV_MAX_CTA_SM86_SHARED_OVERLAP)
            self.assertLess(cta, _AGV_MAX_CTA_SM90_SHARED_OVERLAP,
                            "SM86 应比 SM90 分配更少 CTA（PCIe 更受限）")

        def test_sm90_agv_cta_with_overlap(self):
            p = self._make_profile(SMArch.SM90)
            cta = p.agv_max_cta(shared_expert_overlap=True)
            self.assertEqual(cta, _AGV_MAX_CTA_SM90_SHARED_OVERLAP)

        def test_no_overlap_cta_unlimited(self):
            for sm in [SMArch.SM86, SMArch.SM90]:
                p = self._make_profile(sm)
                cta = p.agv_max_cta(shared_expert_overlap=False)
                self.assertIsNone(cta)

        def test_sm90_supports_warp_specialization(self):
            p = self._make_profile(SMArch.SM90)
            self.assertTrue(p.supports_warp_specialization)

        def test_sm86_no_warp_specialization(self):
            p = self._make_profile(SMArch.SM86)
            self.assertFalse(p.supports_warp_specialization)

        def test_transfer_latency_estimate(self):
            p = self._make_profile(SMArch.SM86)
            # 1GB 数据，PCIe Gen4 理论 ~31.25ms
            latency = p.estimate_transfer_latency_ms(1 * 1024 ** 3)
            expected = 1024 / _PCIE_GEN4_BW_GBPS  # ms
            self.assertAlmostEqual(latency, expected, places=1)

    class TestLOCCache(unittest.TestCase):
        """测试 LOC 缓存的基本读写和驱逐逻辑（CPU only，无 CUDA 依赖）。"""

        def setUp(self):
            self.cache = LOCCache(capacity=4)

        def _make_tensor(self, val: float = 1.0) -> torch.Tensor:
            return torch.full((8, 16), val, dtype=torch.float32)

        def test_put_and_get_hit(self):
            t = self._make_tensor(1.0)
            self.cache.put(0, 12345, t, transfer_stream=None)
            result = self.cache.get(0, 12345, torch.device("cpu"), transfer_stream=None)
            self.assertIsNotNone(result)
            self.assertTrue(torch.allclose(result, t))

        def test_get_miss(self):
            result = self.cache.get(0, 99999, torch.device("cpu"), transfer_stream=None)
            self.assertIsNone(result)

        def test_eviction_on_capacity(self):
            for i in range(5):
                t = self._make_tensor(float(i))
                self.cache.put(0, i, t, transfer_stream=None)
            stats = self.cache.stats
            self.assertLessEqual(stats["entries"], self.cache.capacity)
            self.assertGreater(stats["evictions"], 0)

        def test_hit_count_increments(self):
            t = self._make_tensor(2.0)
            self.cache.put(0, 777, t, transfer_stream=None)
            for _ in range(3):
                self.cache.get(0, 777, torch.device("cpu"), transfer_stream=None)
            with self.cache._lock:
                key = self.cache._make_key(0, 777)
                entry = self.cache._store[key]
            self.assertEqual(entry.hit_count, 3)

        def test_lru_evicts_least_used(self):
            """容量为 4，插入 4 条后再插入 1 条，应驱逐 hit_count 最低的。"""
            # 插入 4 条
            for i in range(4):
                t = self._make_tensor(float(i))
                self.cache.put(0, i, t, transfer_stream=None)
            # 多次访问 key=1,2,3
            for _ in range(5):
                self.cache.get(0, 1, torch.device("cpu"))
                self.cache.get(0, 2, torch.device("cpu"))
                self.cache.get(0, 3, torch.device("cpu"))
            # 插入第 5 条，应驱逐 key=0（hit_count=0）
            self.cache.put(0, 4, self._make_tensor(4.0))
            with self.cache._lock:
                key_0 = self.cache._make_key(0, 0)
                self.assertNotIn(key_0, self.cache._store,
                                 "key=0 应被 LRU 驱逐（hit_count=0）")

        def test_stats_structure(self):
            stats = self.cache.stats
            self.assertIn("entries", stats)
            self.assertIn("capacity", stats)
            self.assertIn("evictions", stats)

    class TestTokenHashStability(unittest.TestCase):
        """测试 token hash 计算的稳定性（相同输入产生相同 hash）。"""

        def _make_dispatcher_stub(self) -> HeteroSharedExpertDispatcher:
            """构造最小化 dispatcher stub，不依赖 CUDA。"""
            d = object.__new__(HeteroSharedExpertDispatcher)
            d.layer_idx = 0
            d.loc_cache = None
            d.shared_experts = None
            d._is_holder = False
            d._peer_access_enabled = False
            d._shared_expert_output_h100 = None
            d._current_token_hash = None
            d._loc_hit = False
            d._hidden_shape = None
            return d

        def test_same_input_same_hash(self):
            d = self._make_dispatcher_stub()
            t = torch.randn(32, 64)
            h1 = d._compute_token_hash(t)
            h2 = d._compute_token_hash(t)
            self.assertEqual(h1, h2)

        def test_different_input_different_hash(self):
            d = self._make_dispatcher_stub()
            t1 = torch.randn(32, 64)
            t2 = torch.randn(32, 64)
            h1 = d._compute_token_hash(t1)
            h2 = d._compute_token_hash(t2)
            # 理论上不同，但极小概率相同；用足够大的张量降低碰撞率
            self.assertNotEqual(h1, h2)

        def test_hash_in_valid_range(self):
            d = self._make_dispatcher_stub()
            t = torch.randn(128, 256)
            h = d._compute_token_hash(t)
            self.assertGreaterEqual(h, 0)
            self.assertLess(h, 2 ** 31)

    class TestSMArchCTAPolicy(unittest.TestCase):
        """测试 SM 架构与 CTA 策略的一致性约束。"""

        def test_sm86_cta_less_than_sm90(self):
            """SM86 的 CTA 限制应严格小于 SM90（反映 PCIe 更强的 SM 争用压力）。"""
            self.assertLess(
                _AGV_MAX_CTA_SM86_SHARED_OVERLAP,
                _AGV_MAX_CTA_SM90_SHARED_OVERLAP,
            )

        def test_cta_limits_positive(self):
            self.assertGreater(_AGV_MAX_CTA_SM86_SHARED_OVERLAP, 0)
            self.assertGreater(_AGV_MAX_CTA_SM90_SHARED_OVERLAP, 0)

        def test_unlimited_is_none(self):
            self.assertIsNone(_AGV_MAX_CTA_UNLIMITED)

    class TestSharedExpertMLP(unittest.TestCase):
        """测试 HeteroSharedExpertMLP 的前向传播（CPU fallback）。"""

        def _make_cpu_profile(self) -> HeteroDeviceProfile:
            p = object.__new__(HeteroDeviceProfile)
            p.device_index = 0
            p.sm_arch = SMArch.SM90
            p.total_memory_gb = 96.0
            p.pcie_bw_gbps = _PCIE_GEN4_BW_GBPS
            p.is_shared_expert_device = True
            p.compute_stream = None
            p.transfer_stream = None
            return p

        def _build_mlp(self) -> HeteroSharedExpertMLP:
            """构建 CPU 上的 MLP（绕过 CUDA stream 初始化）。"""
            mlp = object.__new__(HeteroSharedExpertMLP)
            mlp.hidden_size = 64
            mlp.ffn_hidden_size = 128
            # 使用 CPU 线性层
            mlp.gate_proj = nn.Linear(64, 128, bias=False)
            mlp.up_proj = nn.Linear(64, 128, bias=False)
            mlp.down_proj = nn.Linear(128, 64, bias=False)
            return mlp

        def test_forward_shape(self):
            mlp = self._build_mlp()
            x = torch.randn(16, 64)
            with torch.no_grad():
                out = mlp(x)
            self.assertEqual(out.shape, (16, 64))

        def test_forward_deterministic(self):
            mlp = self._build_mlp()
            x = torch.randn(8, 64)
            with torch.no_grad():
                o1 = mlp(x)
                o2 = mlp(x)
            self.assertTrue(torch.allclose(o1, o2))

        def test_swiglu_nonlinearity(self):
            """SwiGLU 输出应与纯 Linear 不同（验证激活函数被应用）。"""
            mlp = self._build_mlp()
            x = torch.randn(4, 64)
            with torch.no_grad():
                out = mlp(x)
            # 纯线性输出（无激活）
            plain = mlp.down_proj(mlp.gate_proj(x) * mlp.up_proj(x))
            # SwiGLU 使用 silu(gate) * up，不应等于 gate * up
            self.assertFalse(torch.allclose(out, plain))

    class TestLOCCacheEdgeCases(unittest.TestCase):
        """边界情况测试。"""

        def test_capacity_one(self):
            cache = LOCCache(capacity=1)
            t1 = torch.ones(4, 8)
            t2 = torch.ones(4, 8) * 2
            cache.put(0, 1, t1)
            cache.put(0, 2, t2)
            stats = cache.stats
            self.assertEqual(stats["entries"], 1)
            self.assertEqual(stats["evictions"], 1)

        def test_duplicate_put_no_duplicate_entry(self):
            cache = LOCCache(capacity=8)
            t = torch.randn(4, 16)
            cache.put(0, 42, t)
            cache.put(0, 42, t)  # 重复写入
            with cache._lock:
                self.assertEqual(len(cache._store), 1)

        def test_concurrent_put_get(self):
            """多线程并发读写不应引发异常。"""
            cache = LOCCache(capacity=16)
            errors = []

            def writer(key_offset):
                try:
                    for i in range(10):
                        t = torch.randn(8, 16)
                        cache.put(0, key_offset + i, t)
                except Exception as e:
                    errors.append(e)

            def reader():
                try:
                    for i in range(30):
                        cache.get(0, i % 5, torch.device("cpu"))
                except Exception as e:
                    errors.append(e)

            threads = (
                [threading.Thread(target=writer, args=(i * 20,)) for i in range(3)]
                + [threading.Thread(target=reader) for _ in range(2)]
            )
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"并发异常: {errors}")

    # ── 运行所有测试 ────────────────────────────────────────────────────────────

    # 配置日志输出，测试时显示 DEBUG 级别
    logging.basicConfig(
        level=logging.WARNING,  # 测试时抑制 INFO，只显示警告及以上
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in [
        TestHeteroDeviceProfile,
        TestLOCCache,
        TestTokenHashStability,
        TestSMArchCTAPolicy,
        TestSharedExpertMLP,
        TestLOCCacheEdgeCases,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出 LOC 缓存统计示例
    demo_cache = LOCCache(capacity=8)
    for i in range(6):
        demo_cache.put(0, i, torch.randn(16, 64))
    for _ in range(3):
        demo_cache.get(0, 2, torch.device("cpu"))
    print(f"\nLOC Cache demo stats: {demo_cache.stats}")

    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
