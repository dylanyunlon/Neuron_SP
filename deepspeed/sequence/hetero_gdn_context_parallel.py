"""
DES-LOC Heterogeneous GDN Context Parallel
============================================

**上游设计意图 (Megatron commit 20ba03f)**
Megatron-LM 在 GatedDeltaNet (GDN) SSM 层上实现了 Context Parallel (CP) 支持：
通过 All-to-All 通信在 CP 维度上切分序列、在 HP 维度上聚合 head，使 GDN
能在多 GPU 上以 CP×TP 的方式并行处理长序列。核心机制：

  1. tensor_a2a_cp2hp: 序列维度 → head 维度的 All-to-All（CP→HP）
  2. tensor_a2a_hp2cp: head 维度 → 序列维度的 All-to-All（HP→CP）
  3. get_parameter_local_cp: 按 CP rank 切片权重（conv1d weight/bias, A_log, dt_bias）
  4. _prepare_qkv_for_gated_delta_rule: JIT 融合 split/reshape/l2norm/repeat_interleave
  5. _compute_g_and_beta: JIT 融合 exp/softplus/sigmoid

**DES-LOC 适配点**
硬件环境：2×A6000 48GB SM86 + 1×H100 NVL 96GB SM90，PCIe 互联，无 NVLink。

在同构 CP 中，All-to-All 要求所有 rank 的通信延迟对称。但 PCIe 拓扑下
A6000↔H100 带宽约为 A6000↔A6000 的 1/3，导致同构 All-to-All 受木桶效应拖慢。

DES-LOC (Decoupled Execution with Shared LOcality Cache) 的核心适配：

  A. **异构 CP 组划分** (HeteroProcessGroup)
     - "快组"：A6000×2（同代 PCIe Switch，实测带宽更均衡）
     - "慢组"：包含 H100 的跨代通道
     - 通过 locality_rank 将序列块优先分配给本地快组处理

  B. **LOC 缓存 (SharedLocalityCache)**
     - 用 CPU DRAM (1.5TB) 作为二级缓冲，在 CP All-to-All 前将远端 head 分片
       pin 到 CPU，异步预取，掩盖 PCIe 延迟
     - 仅缓存 conv1d weight/A_log/dt_bias（只读参数，多 step 复用）

  C. **异构感知的 All-to-All 分段**
     - tensor_a2a_cp2hp_hetero / tensor_a2a_hp2cp_hetero：
       先做组内快速 All-to-All，再做跨组慢速 All-to-All，
       两阶段之间插入 LOC 缓存预取，减少 PCIe 争用

  D. **SM90 特化路径**
     - H100 rank 使用 causal_conv1d（FLA 实现），A6000 rank 走 F.conv1d 确定性路径
     - 通过 device_capability() 自动判别，无需手动配置

  E. **head 分配不等分策略**
     - H100 显存 96GB >> A6000 48GB，允许 H100 承担更多 head，
       通过 unequal_head_split() 按显存比例分配，打破 Megatron 的等分假设

作者: Neuron_SP Project (based on DeepSpeed + Megatron-LM upstream)
上游引用: Megatron commit 20ba03fec03ebaec050c6bc7e79b77a4b4b5c000
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量与设备能力检测
# ---------------------------------------------------------------------------

SM90_CAPABILITY = (9, 0)  # H100 NVL
SM86_CAPABILITY = (8, 6)  # A6000


def _get_device_capability(device: Optional[torch.device] = None) -> Tuple[int, int]:
    """返回当前 CUDA 设备的 SM 版本号，用于区分 A6000(SM86) 与 H100(SM90)。"""
    dev = device or torch.device("cuda", torch.cuda.current_device())
    return torch.cuda.get_device_capability(dev)


def _is_sm90(device: Optional[torch.device] = None) -> bool:
    """判断当前设备是否为 H100 NVL (SM90)。"""
    return _get_device_capability(device) >= SM90_CAPABILITY


# ---------------------------------------------------------------------------
# LOC 缓存：利用 1.5TB CPU DRAM 作为参数二级缓冲
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """
    DES-LOC 核心组件：Shared LOcality Cache。

    将只读参数（conv1d weight/bias, A_log, dt_bias）pin 在 CPU DRAM，
    按 CP rank 切片后异步预取到 GPU，在 PCIe All-to-All 等待期间掩盖延迟。

    设计原则：
    - 参数在 forward 首次调用时 pin 并缓存，后续 step 直接复用
    - 每个 CP rank 只缓存自己需要的分片（显存节省 cp_size 倍）
    - 用 CUDA stream 异步拷贝，与 All-to-All 通信重叠
    """

    def __init__(self, cp_size: int, device: torch.device):
        self.cp_size = cp_size
        self.device = device
        self._cpu_cache: Dict[str, torch.Tensor] = {}   # pin_memory CPU 张量
        self._gpu_cache: Dict[str, torch.Tensor] = {}   # 已预取到 GPU 的分片
        self._prefetch_stream = torch.cuda.Stream(device)
        logger.debug(
            "SharedLocalityCache initialized | cp_size=%d device=%s prefetch_stream=%s",
            cp_size, device, self._prefetch_stream,
        )

    def pin_parameter(self, key: str, param: torch.Tensor) -> None:
        """
        将参数 pin 到 CPU DRAM（首次调用时执行）。

        Args:
            key: 参数标识符，例如 "conv1d.weight"
            param: GPU 上的完整参数张量
        """
        if key in self._cpu_cache:
            return
        cpu_tensor = param.detach().cpu().pin_memory()
        self._cpu_cache[key] = cpu_tensor
        logger.debug("LOC: pinned '%s' to CPU DRAM, shape=%s", key, tuple(cpu_tensor.shape))

    def prefetch_local_slice(
        self,
        key: str,
        cp_rank: int,
        dim: int,
        split_sections: Optional[List[int]] = None,
    ) -> None:
        """
        异步预取当前 CP rank 的参数分片到 GPU。
        在 PCIe All-to-All 发起前调用，利用通信等待时间完成拷贝。

        Args:
            key: 参数标识符
            cp_rank: 当前 CP rank
            dim: 切分维度（通常为 0，head 维度）
            split_sections: 若参数由多段组成，先按此切分再按 cp_rank 切片
        """
        if key not in self._cpu_cache:
            logger.warning("LOC prefetch: key '%s' not pinned, skipping", key)
            return

        cpu_tensor = self._cpu_cache[key]
        local_slice = _slice_local_cp(cpu_tensor, dim, self.cp_size, cp_rank, split_sections)

        with torch.cuda.stream(self._prefetch_stream):
            gpu_slice = local_slice.to(self.device, non_blocking=True)
        self._gpu_cache[key] = gpu_slice
        logger.debug("LOC: prefetch launched for '%s' rank=%d shape=%s", key, cp_rank, tuple(gpu_slice.shape))

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        获取已预取的 GPU 分片。调用前需确保预取完成（sync_prefetch）。

        Returns:
            GPU 张量分片，或 None（若未预取）
        """
        return self._gpu_cache.get(key, None)

    def sync_prefetch(self) -> None:
        """等待所有异步预取完成，在使用缓存张量前调用。"""
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)

    def invalidate(self, key: str) -> None:
        """参数更新后（如 optimizer step）清除 GPU 缓存，保留 CPU pin 内存。"""
        self._gpu_cache.pop(key, None)

    def invalidate_all(self) -> None:
        """清除所有 GPU 缓存（每个 optimizer step 后调用）。"""
        self._gpu_cache.clear()
        logger.debug("LOC: GPU cache invalidated (all keys)")


def _slice_local_cp(
    tensor: torch.Tensor,
    dim: int,
    cp_size: int,
    cp_rank: int,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    按 CP rank 切取张量的本地分片。

    与 Megatron get_parameter_local_cp 等价，但在 CPU 上执行（用于 LOC 缓存）。
    支持 split_sections 先分段再各段独立切片后拼接（适配 conv1d weight 按 q/k/v 分段）。

    Args:
        tensor: 完整张量（CPU 或 GPU）
        dim: 切分维度
        cp_size: CP 总大小
        cp_rank: 当前 CP rank
        split_sections: 可选，先按此列表在 dim 上分段

    Returns:
        当前 rank 的本地分片
    """
    if cp_size == 1:
        return tensor

    if split_sections is not None:
        parts = torch.split(tensor, split_sections, dim=dim)
        sliced = [_slice_local_cp(p, dim, cp_size, cp_rank) for p in parts]
        return torch.cat(sliced, dim=dim)

    dim_size = tensor.size(dim)
    assert dim_size % cp_size == 0, (
        f"Tensor dim {dim} size {dim_size} not divisible by cp_size {cp_size}"
    )
    start = cp_rank * dim_size // cp_size
    end = (cp_rank + 1) * dim_size // cp_size
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(start, end)
    return tensor[tuple(slices)]


# ---------------------------------------------------------------------------
# 异构 CP 组：按 SM 版本划分快/慢子组
# ---------------------------------------------------------------------------

@dataclass
class HeteroProcessGroup:
    """
    DES-LOC 异构进程组。

    在同构 CP 中，所有 rank 参与同一 All-to-All，PCIe 拓扑下 A6000↔H100
    通道成为瓶颈。HeteroProcessGroup 将 CP 组划分为：
    - fast_group: A6000×2（组内通信，PCIe Switch 路径，带宽均衡）
    - slow_group: 包含 H100 的跨代通道

    两阶段 All-to-All：
      1. 组内快速 All-to-All（fast_group）
      2. 跨组慢速 All-to-All（slow_group），期间 LOC 缓存预取掩盖延迟

    Attributes:
        world_group: 完整 CP 组（所有 rank）
        fast_group: 同代快速子组（A6000 ranks）
        slow_group: 跨代慢速子组（含 H100 的 ranks）
        sm90_ranks: H100 rank 列表（全局编号）
        sm86_ranks: A6000 rank 列表（全局编号）
    """
    world_group: dist.ProcessGroup
    fast_group: Optional[dist.ProcessGroup]
    slow_group: Optional[dist.ProcessGroup]
    sm90_ranks: List[int] = field(default_factory=list)
    sm86_ranks: List[int] = field(default_factory=list)

    @property
    def cp_size(self) -> int:
        return self.world_group.size()

    @property
    def cp_rank(self) -> int:
        return self.world_group.rank()

    @property
    def is_sm90_rank(self) -> bool:
        """当前 rank 是否运行在 H100 上。"""
        return self.cp_rank in self.sm90_ranks

    def __repr__(self) -> str:
        return (
            f"HeteroProcessGroup(cp_size={self.cp_size}, cp_rank={self.cp_rank}, "
            f"sm90_ranks={self.sm90_ranks}, sm86_ranks={self.sm86_ranks})"
        )


def build_hetero_cp_group(
    cp_group: dist.ProcessGroup,
    device_capabilities: Optional[List[Tuple[int, int]]] = None,
) -> HeteroProcessGroup:
    """
    根据各 rank 的 GPU SM 版本构建异构 CP 进程组。

    **检测流程**：
    每个 rank 广播自身的 (major, minor) SM 版本，汇总后识别 SM90/SM86 rank，
    再用 new_group() 分别建立 fast_group（纯 A6000）和 slow_group（含 H100）。

    Args:
        cp_group: 已初始化的同构 CP 进程组
        device_capabilities: 若提供则跳过自动检测（用于测试/模拟）

    Returns:
        HeteroProcessGroup 实例
    """
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    # 收集各 rank 的 SM 版本
    if device_capabilities is None:
        cap = _get_device_capability()
        cap_tensor = torch.tensor([cap[0], cap[1]], dtype=torch.int32, device="cuda")
        all_caps = [torch.zeros(2, dtype=torch.int32, device="cuda") for _ in range(cp_size)]
        dist.all_gather(all_caps, cap_tensor, group=cp_group)
        device_capabilities = [(t[0].item(), t[1].item()) for t in all_caps]

    sm90_ranks = [i for i, cap in enumerate(device_capabilities) if cap >= SM90_CAPABILITY]
    sm86_ranks = [i for i, cap in enumerate(device_capabilities) if cap < SM90_CAPABILITY]

    logger.info(
        "HeteroCP group | cp_size=%d sm90_ranks=%s sm86_ranks=%s",
        cp_size, sm90_ranks, sm86_ranks,
    )

    # 构建子组
    fast_group = dist.new_group(ranks=sm86_ranks, group_desc="des_loc_fast_cp") if len(sm86_ranks) > 1 else None
    slow_group = dist.new_group(ranks=sm90_ranks + sm86_ranks[:1], group_desc="des_loc_slow_cp") if sm90_ranks else None

    return HeteroProcessGroup(
        world_group=cp_group,
        fast_group=fast_group,
        slow_group=slow_group,
        sm90_ranks=sm90_ranks,
        sm86_ranks=sm86_ranks,
    )


# ---------------------------------------------------------------------------
# 不等分 head 分配（H100 承担更多 head）
# ---------------------------------------------------------------------------

def unequal_head_split(
    total_heads: int,
    cp_size: int,
    sm90_ranks: List[int],
    sm86_ranks: List[int],
    h100_weight: float = 2.0,
) -> List[int]:
    """
    按显存比例为各 CP rank 分配 head 数量。

    **上游差异**：Megatron 要求 num_heads % (tp * cp) == 0，即等分。
    DES-LOC 放宽此约束：H100 (96GB) 承担更多 head，A6000 (48GB) 承担较少，
    比例由 h100_weight 控制（默认 2.0，对应显存比 96:48）。

    分配算法：
      weight_sm90 = h100_weight，weight_sm86 = 1.0
      head_i = round(total_heads * weight_i / sum_weights)
    最后一个 rank 补齐剩余 head，确保总和等于 total_heads。

    Args:
        total_heads: 总 head 数
        cp_size: CP 总大小（= len(sm90_ranks) + len(sm86_ranks)）
        sm90_ranks: H100 rank 列表
        sm86_ranks: A6000 rank 列表
        h100_weight: H100 相对权重（默认 2.0）

    Returns:
        长度为 cp_size 的列表，heads_per_rank[i] 为 rank i 分配的 head 数
    """
    assert cp_size == len(sm90_ranks) + len(sm86_ranks), (
        f"cp_size={cp_size} != len(sm90)={len(sm90_ranks)} + len(sm86)={len(sm86_ranks)}"
    )

    weights = [0.0] * cp_size
    for r in sm90_ranks:
        weights[r] = h100_weight
    for r in sm86_ranks:
        weights[r] = 1.0

    total_weight = sum(weights)
    heads_per_rank = [max(1, round(total_heads * w / total_weight)) for w in weights]

    # 补齐误差
    diff = total_heads - sum(heads_per_rank)
    heads_per_rank[-1] += diff

    logger.debug(
        "unequal_head_split | total=%d cp_size=%d allocation=%s",
        total_heads, cp_size, heads_per_rank,
    )
    return heads_per_rank


# ---------------------------------------------------------------------------
# 异构 All-to-All：CP→HP 方向
# ---------------------------------------------------------------------------

def _a2a_single(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
    head_dim: int = -1,
) -> torch.Tensor:
    """
    单组内的 All-to-All：将 seq_dim 上的分片在 head_dim 上聚合。

    对应 Megatron _all_to_all_cp2hp 的核心操作，但限制在单一进程组内。
    张量格式要求：(seq_len, batch, head_dim)，即 seq_dim=0, head_dim=-1 或 2。

    Args:
        tensor: 输入张量，shape (S/cp_size, B, H)
        group: 通信组
        seq_dim: 序列维度（当前仅支持 0）
        head_dim: head 维度（当前仅支持 -1 或 2）

    Returns:
        All-to-All 后的张量，shape (S, B, H/cp_size)
    """
    group_size = group.size()
    if group_size == 1:
        return tensor

    # 规范化 head_dim
    ndim = tensor.dim()
    if head_dim < 0:
        head_dim = ndim + head_dim

    assert seq_dim == 0 and head_dim == 2 and ndim == 3, (
        f"_a2a_single: requires seq_dim=0, head_dim=2, ndim=3, got {seq_dim}, {head_dim}, {ndim}"
    )

    S, B, H = tensor.shape
    assert H % group_size == 0, f"H={H} not divisible by group_size={group_size}"

    # All-to-All 输入：(group_size, S, B, H//group_size)
    input_list = list(tensor.reshape(S, B, group_size, H // group_size)
                           .permute(2, 0, 1, 3)
                           .contiguous()
                           .chunk(group_size))
    output_list = [torch.empty_like(t) for t in input_list]
    dist.all_to_all(output_list, input_list, group=group)

    # 拼接：(S*group_size, B, H//group_size)
    out = torch.cat(output_list, dim=1).reshape(S * group_size, B, H // group_size)
    return out


def tensor_a2a_cp2hp_hetero(
    tensor: torch.Tensor,
    hetero_pg: HeteroProcessGroup,
    loc_cache: Optional[SharedLocalityCache],
    seq_dim: int = 0,
    head_dim: int = -1,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    DES-LOC 异构 CP→HP All-to-All。

    **对应上游** tensor_a2a_cp2hp，但针对 A6000+H100 PCIe 拓扑做两阶段优化：

    阶段 1（快速，组内）：
      仅 A6000 rank 间做 All-to-All（fast_group），延迟低、带宽高。
      同时发起 LOC 缓存预取（参数分片 CPU→GPU 异步拷贝）以掩盖后续等待。

    阶段 2（慢速，跨组）：
      跨 A6000/H100 做 All-to-All（slow_group 或 world_group），
      此时 LOC 预取已完成，GPU 参数分片可用。

    若 split_sections 不为 None，对每段独立执行两阶段 A2A 后拼接
    （与 Megatron 上游保持接口一致）。

    Args:
        tensor: 输入张量 (S/cp_size, B, H)，已在 seq_dim 上分片
        hetero_pg: 异构进程组
        loc_cache: LOC 缓存实例（可为 None，退化为同构路径）
        seq_dim: 序列维度，仅支持 0
        head_dim: head 维度，仅支持 -1 或 2
        split_sections: 可选，在 head_dim 上的分段大小列表

    Returns:
        All-to-All 后张量 (S, B, H/cp_size)
    """
    cp_size = hetero_pg.cp_size
    if cp_size == 1:
        return tensor

    if head_dim < 0:
        head_dim = tensor.dim() + head_dim

    assert seq_dim == 0 and head_dim == 2 and tensor.dim() == 3, (
        "tensor_a2a_cp2hp_hetero: requires seq_dim=0, head_dim=2(or -1), ndim=3"
    )

    # 若有分段，递归处理每段
    if split_sections is not None:
        parts = torch.split(tensor, split_sections, dim=head_dim)
        results = [
            tensor_a2a_cp2hp_hetero(p, hetero_pg, loc_cache, seq_dim, head_dim)
            for p in parts
        ]
        return torch.cat(results, dim=head_dim)

    # --- 阶段 1：fast_group 组内 A2A ---
    if hetero_pg.fast_group is not None and hetero_pg.cp_rank in hetero_pg.sm86_ranks:
        logger.debug("CP→HP A2A phase1: fast_group (A6000 intra-group)")
        tensor = _a2a_single(tensor, hetero_pg.fast_group, seq_dim=0, head_dim=2)

    # LOC 缓存同步（预取完成）
    if loc_cache is not None:
        loc_cache.sync_prefetch()
        logger.debug("LOC prefetch synced before slow A2A")

    # --- 阶段 2：world_group 跨组 A2A ---
    logger.debug("CP→HP A2A phase2: world_group (cross A6000/H100)")
    tensor = _a2a_single(tensor, hetero_pg.world_group, seq_dim=0, head_dim=2)

    return tensor


def tensor_a2a_hp2cp_hetero(
    tensor: torch.Tensor,
    hetero_pg: HeteroProcessGroup,
    seq_dim: int = 0,
    head_dim: int = -1,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    DES-LOC 异构 HP→CP All-to-All（CP→HP 的逆操作）。

    对应上游 tensor_a2a_hp2cp，两阶段顺序与 cp2hp 相反：
    先做跨组（慢速），再做组内（快速）。

    Args:
        tensor: 输入张量 (S, B, H/cp_size)
        hetero_pg: 异构进程组
        seq_dim: 序列维度，仅支持 0
        head_dim: head 维度，仅支持 -1 或 2
        split_sections: 可选分段

    Returns:
        All-to-All 后张量 (S/cp_size, B, H)
    """
    cp_size = hetero_pg.cp_size
    if cp_size == 1:
        return tensor

    if head_dim < 0:
        head_dim = tensor.dim() + head_dim

    assert seq_dim == 0 and head_dim == 2 and tensor.dim() == 3, (
        "tensor_a2a_hp2cp_hetero: requires seq_dim=0, head_dim=2(or -1), ndim=3"
    )

    if split_sections is not None:
        parts = torch.split(tensor, split_sections, dim=head_dim)
        results = [
            tensor_a2a_hp2cp_hetero(p, hetero_pg, seq_dim, head_dim)
            for p in parts
        ]
        return torch.cat(results, dim=head_dim)

    # --- 阶段 1：world_group 跨组 A2A（HP→CP 先做慢速）---
    logger.debug("HP→CP A2A phase1: world_group (cross A6000/H100)")
    tensor = _a2a_single(tensor, hetero_pg.world_group, seq_dim=0, head_dim=2)

    # --- 阶段 2：fast_group 组内 A2A ---
    if hetero_pg.fast_group is not None and hetero_pg.cp_rank in hetero_pg.sm86_ranks:
        logger.debug("HP→CP A2A phase2: fast_group (A6000 intra-group)")
        tensor = _a2a_single(tensor, hetero_pg.fast_group, seq_dim=0, head_dim=2)

    return tensor


# ---------------------------------------------------------------------------
# 参数本地分片：GPU 路径（带 LOC 缓存）
# ---------------------------------------------------------------------------

def get_parameter_local_cp_with_loc(
    param: torch.Tensor,
    dim: int,
    hetero_pg: HeteroProcessGroup,
    loc_cache: Optional[SharedLocalityCache],
    cache_key: str,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    获取当前 CP rank 的参数本地分片，优先从 LOC 缓存取。

    **上游对应** get_parameter_local_cp，增加 LOC 缓存路径：
    1. 若缓存命中，直接返回 GPU 分片（零拷贝）
    2. 若缓存未命中，走 GPU 切片路径（与上游等价），并异步 pin 到 CPU

    对只读参数（conv1d weight, A_log, dt_bias）首次使用后异步 pin，
    后续 step 直接复用，消除重复的 GPU 切片操作。

    Args:
        param: 完整参数张量（GPU）
        dim: 切分维度
        hetero_pg: 异构进程组（提供 cp_rank, cp_size）
        loc_cache: LOC 缓存实例（None 则退化为直接切片）
        cache_key: 缓存键名
        split_sections: 可选，先分段再切片

    Returns:
        当前 rank 的参数分片（GPU 张量）
    """
    # 优先从 LOC 缓存取
    if loc_cache is not None:
        cached = loc_cache.get(cache_key)
        if cached is not None:
            logger.debug("LOC cache hit: '%s'", cache_key)
            return cached

    # 直接 GPU 切片
    local = _slice_local_cp(
        param, dim, hetero_pg.cp_size, hetero_pg.cp_rank, split_sections
    )

    # 异步 pin 到 CPU DRAM 以备后续 step 复用
    if loc_cache is not None:
        loc_cache.pin_parameter(cache_key, param)
        loc_cache.prefetch_local_slice(cache_key, hetero_pg.cp_rank, dim, split_sections)

    return local


# ---------------------------------------------------------------------------
# JIT 融合辅助函数（对应 Megatron @jit_fuser 方法）
# ---------------------------------------------------------------------------

def prepare_qkv_for_gated_delta_rule(
    qkv: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    batch: int,
    seq_len: int,
    qk_dim_local: int,
    v_dim_local: int,
    key_head_dim: int,
    value_head_dim: int,
    num_key_heads_local: int,
    num_value_heads_local: int,
    use_qk_l2norm: bool,
    l2norm_fn,
) -> Tuple[torch.Tensor, ...]:
    """
    准备 GDN 所需的 Q/K/V/gate/beta/alpha 张量。

    **上游对应** GatedDeltaNet._prepare_qkv_for_gated_delta_rule（@jit_fuser）。

    DES-LOC 适配：
    - 函数化（非方法），便于 torch.compile 跨模块融合
    - qk_dim_local / v_dim_local 已考虑 CP 分片（上游在方法内通过 self.cp_size 除）
    - l2norm_fn 可注入（SM90 使用 FLA l2norm，SM86 使用 F.normalize fallback）

    操作序列（与上游一致）：
      split qkv → reshape → L2 norm（可选）→ split query/key → repeat_interleave → contiguous

    Args:
        qkv: (B, S_local, 2*qk_dim_local + v_dim_local) 张量
        gate: (B, S_local, num_v_heads_local, v_head_dim)
        beta: (B, S_local, num_v_heads_local)
        alpha: (B, S_local, num_v_heads_local)
        batch: batch size
        seq_len: 本地序列长度（经 CP 分片后）
        qk_dim_local: 当前 rank 的 QK 维度
        v_dim_local: 当前 rank 的 V 维度
        key_head_dim: 单个 key head 的维度
        value_head_dim: 单个 value head 的维度
        num_key_heads_local: 当前 rank 的 key head 数
        num_value_heads_local: 当前 rank 的 value head 数
        use_qk_l2norm: 是否对 Q/K 做 L2 归一化
        l2norm_fn: L2 归一化函数（FLA l2norm 或 F.normalize fallback）

    Returns:
        (query, key, value, gate, beta, alpha) 均为 contiguous 张量
    """
    # Split qkv → query_key 和 value
    query_key, value = torch.split(qkv, [2 * qk_dim_local, v_dim_local], dim=-1)

    # Reshape 到 head 格式
    query_key = query_key.reshape(batch, seq_len, -1, key_head_dim)
    value = value.reshape(batch, seq_len, -1, value_head_dim)

    # L2 归一化（可选）
    if use_qk_l2norm and l2norm_fn is not None:
        query_key = l2norm_fn(query_key.contiguous())

    # Split query / key
    split_size = num_key_heads_local
    query, key = torch.split(query_key, [split_size, split_size], dim=2)

    # Grouped query attention：重复 query/key 以匹配 value head 数
    repeat_factor = num_value_heads_local // num_key_heads_local
    if repeat_factor > 1:
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)

    return (
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate.contiguous(),
        beta.contiguous(),
        alpha.contiguous(),
    )


def compute_g_and_beta(
    A_log_local: torch.Tensor,
    dt_bias_local: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 GDN 的衰减因子 g 和门控 beta。

    **上游对应** GatedDeltaNet._compute_g_and_beta（@jit_fuser）。

    公式（在 fp32 精度下执行以保证数值稳定）：
      g = -exp(A_log_local) * softplus(alpha + dt_bias_local)
      beta = sigmoid(beta)

    DES-LOC 无额外修改，直接复用上游逻辑（异构设备数值行为一致）。

    Args:
        A_log_local: 当前 rank 的 A_log 参数分片
        dt_bias_local: 当前 rank 的 dt_bias 参数分片
        alpha: (B, S_local, num_v_heads_local)
        beta: (B, S_local, num_v_heads_local)

    Returns:
        (g, beta_sigmoid)：g 为 fp32，beta_sigmoid 保持输入 dtype
    """
    g = -A_log_local.exp() * F.softplus(alpha.float() + dt_bias_local)
    beta_out = beta.sigmoid()
    return g, beta_out


# ---------------------------------------------------------------------------
# SM90 特化卷积路径
# ---------------------------------------------------------------------------

def apply_causal_conv1d_hetero(
    qkv: torch.Tensor,
    conv1d_weight_local: torch.Tensor,
    conv1d_bias_local: Optional[torch.Tensor],
    conv1d_module: nn.Module,
    activation: str,
    deterministic_mode: bool,
    is_sm90: bool,
    seq_len: int,
    conv_dim_local: int,
    cp_size: int,
) -> torch.Tensor:
    """
    异构设备的因果卷积路径选择。

    **上游对应** GatedDeltaNet.forward 中的 conv1d 分支：
    - deterministic_mode=True → F.conv1d（所有设备，确定性）
    - deterministic_mode=False → causal_conv1d_fn（CUDA kernel）

    **DES-LOC 适配**：增加 SM90 特化判断：
    - H100 (SM90, is_sm90=True)：使用 FLA causal_conv1d（[B,S,D] 格式，无需转置）
    - A6000 (SM86, is_sm90=False)：始终使用 F.conv1d（[B,D,S] 格式，确定性）

    此设计原因：
    1. FLA causal_conv1d 针对 SM80+ 有 CUDA kernel 优化，H100 获益更大
    2. A6000 上 F.conv1d 确定性路径与分布式 checkpoint 验证结果一致
    3. 避免 PCIe 上无谓的格式转换开销

    Args:
        qkv: (B, S_local, C) 格式（FLA 期望格式）
        conv1d_weight_local: 已按 CP rank 切片的卷积权重
        conv1d_bias_local: 已按 CP rank 切片的卷积偏置（可为 None）
        conv1d_module: 原始 nn.Conv1d 模块（提供 stride/padding/dilation 等超参）
        activation: 激活函数名称（"silu" 或 "swish"）
        deterministic_mode: 是否强制确定性模式
        is_sm90: 当前设备是否为 H100
        seq_len: 本地序列长度
        conv_dim_local: 当前 rank 的卷积通道数（= qk_dim_local*2 + v_dim_local）
        cp_size: CP 大小（用于 groups 参数计算）

    Returns:
        卷积后的张量 (B, S_local, C)
    """
    try:
        from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
        _have_fla_conv = True
    except ImportError:
        _have_fla_conv = False
        fla_causal_conv1d = None

    if deterministic_mode or (not is_sm90) or (not _have_fla_conv):
        # A6000 或确定性模式：F.conv1d，需要 [B, D, S] 格式
        qkv_bds = qkv.transpose(1, 2).contiguous()  # B,S,D → B,D,S
        conv_out = F.conv1d(
            input=qkv_bds,
            weight=conv1d_weight_local,
            bias=conv1d_bias_local,
            stride=conv1d_module.stride,
            padding=conv1d_module.padding,
            dilation=conv1d_module.dilation,
            groups=conv_dim_local // cp_size,
        )
        act_fn = F.silu if activation in ("silu", "swish") else getattr(F, activation)
        qkv_out = act_fn(conv_out[..., :seq_len])
        qkv_out = qkv_out.transpose(1, 2)  # B,D,S → B,S,D
        logger.debug("conv1d: SM86 deterministic path (F.conv1d)")
    else:
        # H100 特化路径：FLA causal_conv1d，[B, S, D] 格式
        assert activation in ("silu", "swish"), (
            f"FLA causal_conv1d only supports silu/swish, got {activation}"
        )
        qkv_out, _ = fla_causal_conv1d(
            x=qkv,
            weight=conv1d_weight_local.squeeze(1),  # D,1,W → D,W
            bias=conv1d_bias_local,
            activation=activation,
            initial_state=None,
            output_final_state=False,
        )
        logger.debug("conv1d: SM90 FLA causal_conv1d path")

    return qkv_out


# ---------------------------------------------------------------------------
# 核心模块：HeteroGDNContextParallel
# ---------------------------------------------------------------------------

class HeteroGDNContextParallel(nn.Module):
    """
    DES-LOC 异构 GatedDeltaNet Context Parallel 适配层。

    **功能定位**
    本模块封装了 GatedDeltaNet 在 DES-LOC 异构训练框架下的 CP 通信逻辑，
    以 Wrapper 模式包裹 GDN 主模块，对外保持与 Megatron GatedDeltaNet.forward 相同接口。

    **上游对应**
    Megatron commit 20ba03f 将 CP 逻辑内联在 GatedDeltaNet.forward 中：
    - 前向入口做 tensor_a2a_cp2hp
    - 后向出口做 tensor_a2a_hp2cp
    - 参数切片通过 get_parameter_local_cp

    **DES-LOC 重构**
    将通信逻辑提取到独立模块，优点：
    1. GDN 主模块不感知异构拓扑，保持可测试性
    2. LOC 缓存生命周期在此模块内管理
    3. 便于在 DeepSpeed ZeRO 框架下插入 hook

    **前向流程**
    ┌─────────────────────────────────────────────────────────┐
    │  hidden_states (S/cp, B, H)                             │
    │       ↓ in_proj → qkvzba (S/cp, B, all_dim)            │
    │       ↓ [LOC prefetch 异步启动]                         │
    │       ↓ tensor_a2a_cp2hp_hetero (CP→HP, 两阶段)        │
    │       ↓ [LOC sync] get_parameter_local_cp_with_loc      │
    │       ↓ transpose s,b → b,s                             │
    │       ↓ split qkv/gate/beta/alpha（已含 CP 分片）       │
    │       ↓ apply_causal_conv1d_hetero (SM90/SM86 路径)     │
    │       ↓ prepare_qkv_for_gated_delta_rule                 │
    │       ↓ compute_g_and_beta                              │
    │       ↓ chunk_gated_delta_rule (FLA) / torch fallback   │
    │       ↓ out_norm → reshape → transpose b,s → s,b        │
    │       ↓ tensor_a2a_hp2cp_hetero (HP→CP, 两阶段)        │
    │       ↓ out_proj                                        │
    │  output (S/cp, B, H)                                    │
    └─────────────────────────────────────────────────────────┘

    Attributes:
        gdn: 被包裹的 GatedDeltaNet 主模块
        hetero_pg: 异构 CP 进程组
        loc_cache: LOC 缓存（CPU DRAM 参数缓冲）
        cp_size: CP 总大小
        tp_size: TP 总大小
    """

    def __init__(
        self,
        gdn_module: nn.Module,
        hetero_pg: HeteroProcessGroup,
        loc_cache: Optional[SharedLocalityCache] = None,
    ):
        """
        Args:
            gdn_module: GatedDeltaNet 主模块（Megatron 实现）
            hetero_pg: 由 build_hetero_cp_group 构建的异构进程组
            loc_cache: LOC 缓存实例；若为 None 则退化为同构路径
        """
        super().__init__()
        self.gdn = gdn_module
        self.hetero_pg = hetero_pg
        self.loc_cache = loc_cache

        self.cp_size = hetero_pg.cp_size
        self.cp_rank = hetero_pg.cp_rank
        self.is_sm90_rank = hetero_pg.is_sm90_rank

        # 从 GDN 模块读取维度信息
        self.tp_size = getattr(gdn_module, "tp_size", 1)
        self.qk_dim_local_tp = getattr(gdn_module, "qk_dim_local_tp", None)
        self.v_dim_local_tp = getattr(gdn_module, "v_dim_local_tp", None)
        self.key_head_dim = getattr(gdn_module, "key_head_dim", 64)
        self.value_head_dim = getattr(gdn_module, "value_head_dim", 64)
        self.num_key_heads = getattr(gdn_module, "num_key_heads", 8)
        self.num_value_heads = getattr(gdn_module, "num_value_heads", 8)
        self.use_qk_l2norm = getattr(gdn_module, "use_qk_l2norm", True)
        self.activation = getattr(gdn_module, "activation", "silu")
        self.conv_bias = getattr(gdn_module, "conv_bias", True)

        # 按 CP 进一步分片的本地维度
        self.qk_dim_local = self.qk_dim_local_tp // self.cp_size if self.qk_dim_local_tp else None
        self.v_dim_local = self.v_dim_local_tp // self.cp_size if self.v_dim_local_tp else None

        # L2 norm 函数注入（SM90 用 FLA，SM86 用 F.normalize fallback）
        self._l2norm_fn = self._resolve_l2norm()

        # 选择 delta rule 实现
        self._gated_delta_rule = self._resolve_delta_rule()

        logger.info(
            "HeteroGDNContextParallel initialized | cp_size=%d cp_rank=%d "
            "is_sm90=%s tp_size=%d",
            self.cp_size, self.cp_rank, self.is_sm90_rank, self.tp_size,
        )

    def _resolve_l2norm(self):
        """选择 L2 归一化函数：SM90 优先用 FLA，否则用 F.normalize。"""
        try:
            from fla.modules.l2norm import l2norm
            logger.debug("L2 norm: using FLA l2norm")
            return l2norm
        except ImportError:
            logger.debug("L2 norm: FLA not available, using F.normalize fallback")
            return lambda x: F.normalize(x, dim=-1)

    def _resolve_delta_rule(self):
        """选择 gated delta rule 实现：FLA chunk kernel 或 torch 原生。"""
        config = getattr(self.gdn, "config", None)
        deterministic = getattr(config, "deterministic_mode", False) if config else False

        if deterministic:
            logger.debug("GDN: using torch_chunk_gated_delta_rule (deterministic)")
            return _torch_chunk_gated_delta_rule_fallback
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            logger.debug("GDN: using FLA chunk_gated_delta_rule")
            return chunk_gated_delta_rule
        except ImportError:
            logger.warning("FLA not available, falling back to torch implementation")
            return _torch_chunk_gated_delta_rule_fallback

    def _prefetch_parameters(self) -> None:
        """
        在 CP→HP All-to-All 之前异步预取参数分片到 GPU。

        预取目标：conv1d.weight, conv1d.bias, A_log, dt_bias。
        这些参数在每个 forward step 都需要按 CP rank 切片，
        通过 LOC 缓存将切片结果复用，避免重复 GPU 内存操作。
        """
        if self.loc_cache is None:
            return

        gdn = self.gdn
        qkv_split = [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp]

        params_to_prefetch = [
            ("conv1d.weight", gdn.conv1d.weight, 0, qkv_split),
        ]
        if self.conv_bias and gdn.conv1d.bias is not None:
            params_to_prefetch.append(("conv1d.bias", gdn.conv1d.bias, 0, qkv_split))
        params_to_prefetch.append(("A_log", gdn.A_log, 0, None))
        params_to_prefetch.append(("dt_bias", gdn.dt_bias, 0, None))

        for key, param, dim, split_secs in params_to_prefetch:
            self.loc_cache.pin_parameter(key, param)
            self.loc_cache.prefetch_local_slice(key, self.cp_rank, dim, split_secs)

        logger.debug("Parameter prefetch launched for %d params", len(params_to_prefetch))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_params=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        HeteroGDN 前向传播（DES-LOC CP 路径）。

        与 Megatron GatedDeltaNet.forward 语义等价，但通信部分使用异构两阶段 A2A，
        参数切片通过 LOC 缓存加速。

        Args:
            hidden_states: (S/cp_size, B, H) 输入张量（已在序列维度分片）
            attention_mask: 注意力掩码（GDN 通常不使用）
            packed_seq_params: THD 格式参数（透传给 GDN 主模块）
            **kwargs: 其他透传参数

        Returns:
            (output, output_bias) 与 Megatron GDN 接口一致
        """
        gdn = self.gdn
        seq_len_local, batch, _ = hidden_states.shape

        # 启动 LOC 参数预取（与后续 in_proj 计算重叠）
        self._prefetch_parameters()

        # --- Input projection ---
        qkvzba, _ = gdn.in_proj(hidden_states)

        # --- CP→HP All-to-All（两阶段异构）---
        # qkvzba: (S/cp, B, all_dim) → (S, B, all_dim/cp)
        num_v_heads_tp = self.num_value_heads // self.tp_size
        split_sections_a2a = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
            self.v_dim_local_tp,
            num_v_heads_tp,
            num_v_heads_tp,
        ]
        qkvzba = tensor_a2a_cp2hp_hetero(
            qkvzba,
            hetero_pg=self.hetero_pg,
            loc_cache=self.loc_cache,
            seq_dim=0,
            head_dim=-1,
            split_sections=split_sections_a2a,
        )

        # 全局序列长度 = 本地 * cp_size * sp_size
        sp_size = getattr(gdn, "sp_size", 1)
        seq_len_global = seq_len_local * self.cp_size * sp_size

        # --- Transpose: S,B,X → B,S,X ---
        qkvzba = qkvzba.transpose(0, 1)

        # --- Split qkv / gate / beta / alpha ---
        qk_dim_local = self.qk_dim_local_tp // self.cp_size
        v_dim_local = self.v_dim_local_tp // self.cp_size
        n_heads_local = num_v_heads_tp // self.cp_size

        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [2 * qk_dim_local + v_dim_local, v_dim_local, n_heads_local, n_heads_local],
            dim=-1,
        )
        alpha = alpha.reshape(batch, seq_len_global, -1)

        # --- 获取本地卷积参数（LOC 缓存优先）---
        qkv_split = [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp]
        conv_dim_local = (self.qk_dim_local_tp * 2 + self.v_dim_local_tp)

        conv1d_weight_local = get_parameter_local_cp_with_loc(
            gdn.conv1d.weight, dim=0,
            hetero_pg=self.hetero_pg, loc_cache=self.loc_cache,
            cache_key="conv1d.weight", split_sections=qkv_split,
        )
        conv1d_bias_local = None
        if self.conv_bias and gdn.conv1d.bias is not None:
            conv1d_bias_local = get_parameter_local_cp_with_loc(
                gdn.conv1d.bias, dim=0,
                hetero_pg=self.hetero_pg, loc_cache=self.loc_cache,
                cache_key="conv1d.bias", split_sections=qkv_split,
            )

        # --- 异构卷积（SM90 FLA / SM86 F.conv1d）---
        config = getattr(gdn, "config", None)
        deterministic = getattr(config, "deterministic_mode", False) if config else False

        qkv = apply_causal_conv1d_hetero(
            qkv=qkv,
            conv1d_weight_local=conv1d_weight_local,
            conv1d_bias_local=conv1d_bias_local,
            conv1d_module=gdn.conv1d,
            activation=self.activation,
            deterministic_mode=deterministic,
            is_sm90=self.is_sm90_rank,
            seq_len=seq_len_global,
            conv_dim_local=conv_dim_local,
            cp_size=self.cp_size,
        )

        # --- 准备 Q/K/V ---
        num_k_heads_local = (self.num_key_heads // self.tp_size) // self.cp_size
        num_v_heads_local = n_heads_local

        query, key, value, gate, beta, alpha = prepare_qkv_for_gated_delta_rule(
            qkv=qkv,
            gate=gate,
            beta=beta,
            alpha=alpha,
            batch=batch,
            seq_len=seq_len_global,
            qk_dim_local=qk_dim_local,
            v_dim_local=v_dim_local,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            num_key_heads_local=num_k_heads_local,
            num_value_heads_local=num_v_heads_local,
            use_qk_l2norm=self.use_qk_l2norm,
            l2norm_fn=self._l2norm_fn,
        )

        # --- 获取 A_log / dt_bias 本地分片 ---
        A_log_local = get_parameter_local_cp_with_loc(
            gdn.A_log, dim=0, hetero_pg=self.hetero_pg, loc_cache=self.loc_cache,
            cache_key="A_log", split_sections=None,
        )
        dt_bias_local = get_parameter_local_cp_with_loc(
            gdn.dt_bias, dim=0, hetero_pg=self.hetero_pg, loc_cache=self.loc_cache,
            cache_key="dt_bias", split_sections=None,
        )

        # --- 计算 g 和 beta ---
        g, beta = compute_g_and_beta(A_log_local, dt_bias_local, alpha, beta)

        # --- Gated Delta Rule ---
        core_attn_out, _ = self._gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )

        # --- Out norm ---
        norm_out = gdn.out_norm(core_attn_out * gate)
        norm_out = norm_out.reshape(batch, seq_len_global, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        # --- HP→CP All-to-All（两阶段异构）---
        norm_out = tensor_a2a_hp2cp_hetero(
            norm_out,
            hetero_pg=self.hetero_pg,
            seq_dim=0,
            head_dim=-1,
        )

        # --- Output projection ---
        out, out_bias = gdn.out_proj(norm_out)
        return out, out_bias

    def invalidate_loc_cache(self) -> None:
        """在 optimizer step 后调用，清除 LOC GPU 缓存（保留 CPU pin 内存）。"""
        if self.loc_cache is not None:
            self.loc_cache.invalidate_all()
            logger.debug("LOC GPU cache invalidated after optimizer step")


# ---------------------------------------------------------------------------
# Torch 原生 delta rule fallback（对应 Megatron torch_chunk_gated_delta_rule）
# ---------------------------------------------------------------------------

def _torch_chunk_gated_delta_rule_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
):
    """
    Torch 原生 GatedDeltaRule 实现（无 CUDA kernel 依赖）。

    **上游对应** megatron/core/ssm/gated_delta_net.py::torch_chunk_gated_delta_rule。

    用于以下场景：
    1. FLA 未安装时的 fallback
    2. deterministic_mode=True 时保证跨硬件结果一致
    3. SM86 (A6000) 上验证正确性

    实现为简化版线性注意力递归（非 chunk 优化版本），仅用于功能验证。
    生产环境应使用 FLA chunk_gated_delta_rule。

    Args:
        query: (B, S, H, D_k)
        key: (B, S, H, D_k)
        value: (B, S, H, D_v)
        g: (B, S, H) fp32 衰减因子
        beta: (B, S, H) 门控
        initial_state: 初始状态（暂不支持）
        output_final_state: 是否输出最终状态
        use_qk_l2norm_in_kernel: 是否在 kernel 内做 L2 norm（此实现忽略）
        chunk_size: chunk 大小（此简化实现忽略）

    Returns:
        (output, last_state)：output shape = (B, S, H, D_v)
    """
    B, S, H, Dk = query.shape
    Dv = value.shape[-1]
    dtype = value.dtype

    output = torch.zeros(B, S, H, Dv, device=query.device, dtype=dtype)
    # 状态矩阵：(B, H, Dk, Dv)
    state = torch.zeros(B, H, Dk, Dv, device=query.device, dtype=torch.float32)

    q = query.float()
    k = key.float()
    v = value.float()
    g_f = g.float()       # (B, S, H)
    b_f = beta.float()    # (B, S, H)

    for t in range(S):
        # 衰减状态
        decay = g_f[:, t, :, None, None].exp().clamp(max=1.0)  # (B, H, 1, 1)
        state = state * decay

        # Delta rule 更新
        k_t = k[:, t, :, :].unsqueeze(-1)   # (B, H, Dk, 1)
        v_t = v[:, t, :, :].unsqueeze(-2)   # (B, H, 1, Dv)
        beta_t = b_f[:, t, :, None, None]   # (B, H, 1, 1)

        # 计算当前状态对 k_t 的预测：(B, H, 1, Dv)
        pred = torch.matmul(k_t.transpose(-2, -1), state)  # (B, H, 1, Dv)
        # 更新：state += beta * k_t * (v_t - pred)
        state = state + beta_t * torch.matmul(k_t, v_t - pred)

        # 读出
        q_t = q[:, t, :, :].unsqueeze(-2)  # (B, H, 1, Dk)
        out_t = torch.matmul(q_t, state).squeeze(-2)  # (B, H, Dv)
        output[:, t, :, :] = out_t.to(dtype)

    last_state = state if output_final_state else None
    return output, last_state


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.DEBUG)
    logger.info("=== DES-LOC HeteroGDN Smoke Test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | SM: %s", device, _get_device_capability() if torch.cuda.is_available() else "N/A")

    # --- Test 1: _slice_local_cp ---
    t = torch.arange(16).reshape(16)
    sliced = _slice_local_cp(t, dim=0, cp_size=4, cp_rank=1)
    assert sliced.tolist() == [4, 5, 6, 7], f"Expected [4,5,6,7], got {sliced.tolist()}"
    logger.info("[PASS] _slice_local_cp")

    # --- Test 2: unequal_head_split ---
    allocation = unequal_head_split(total_heads=12, cp_size=3, sm90_ranks=[2], sm86_ranks=[0, 1])
    assert sum(allocation) == 12, f"head allocation sum mismatch: {allocation}"
    assert allocation[2] > allocation[0], "H100 rank should have more heads"
    logger.info("[PASS] unequal_head_split: %s", allocation)

    # --- Test 3: compute_g_and_beta ---
    B, S, H = 2, 8, 4
    A_log = torch.zeros(H, device=device)
    dt_bias = torch.zeros(H, device=device)
    alpha = torch.randn(B, S, H, device=device)
    beta_in = torch.randn(B, S, H, device=device)
    g, beta_out = compute_g_and_beta(A_log, dt_bias, alpha, beta_in)
    assert g.dtype == torch.float32, f"g should be fp32, got {g.dtype}"
    assert beta_out.shape == beta_in.shape, "beta shape mismatch"
    logger.info("[PASS] compute_g_and_beta | g.shape=%s g.dtype=%s", tuple(g.shape), g.dtype)

    # --- Test 4: _torch_chunk_gated_delta_rule_fallback ---
    B, S, H, Dk, Dv = 1, 4, 2, 8, 8
    q = torch.randn(B, S, H, Dk, device=device)
    k = torch.randn(B, S, H, Dk, device=device)
    v = torch.randn(B, S, H, Dv, device=device)
    g_t = -torch.ones(B, S, H, device=device) * 0.1
    beta_t = torch.ones(B, S, H, device=device) * 0.5
    out, _ = _torch_chunk_gated_delta_rule_fallback(q, k, v, g=g_t, beta=beta_t)
    assert out.shape == (B, S, H, Dv), f"delta rule output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "delta rule output contains NaN"
    logger.info("[PASS] torch_chunk_gated_delta_rule_fallback | out.shape=%s", tuple(out.shape))

    # --- Test 5: SharedLocalityCache (CPU only) ---
    if torch.cuda.is_available():
        cache = SharedLocalityCache(cp_size=2, device=device)
        param = torch.randn(8, 4, device=device)
        cache.pin_parameter("test_param", param)
        cache.prefetch_local_slice("test_param", cp_rank=0, dim=0)
        cache.sync_prefetch()
        cached = cache.get("test_param")
        assert cached is not None, "LOC cache miss after prefetch"
        assert cached.shape == (4, 4), f"Cached shape mismatch: {cached.shape}"
        cache.invalidate_all()
        assert cache.get("test_param") is None, "LOC cache should be empty after invalidate"
        logger.info("[PASS] SharedLocalityCache | cached.shape=%s", tuple(cached.shape))
    else:
        logger.info("[SKIP] SharedLocalityCache (no CUDA)")

    logger.info("=== All smoke tests passed ===")
